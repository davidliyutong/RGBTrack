"""Detection algorithm module"""

import os
import sys
import logging
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import cv2
import torch
import numpy as np
import trimesh
from trimesh import bounds

from PIL import Image
from .config import DetectionConfig, CalibrationConfig
import nvdiffrast.torch as dr
from estimater import ScorePredictor, PoseRefinePredictor, FoundationPose
from tools import binary_search_depth, render_cad_depth, draw_posed_3d_box, draw_xyz_axis, set_seed, render_cad_mask
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Detection result data structure"""
    timestamp: float
    frame_id: int
    pose: np.ndarray  # Object pose in camera frame (4x4)
    processing_time_ms: float
    frame_shape: tuple
    camera_pose: Optional[np.ndarray] = None  # Camera pose in world frame (4x4)
    camera_intrinsics: Optional[np.ndarray] = None  # Camera intrinsics matrix (3x3)
    preview_jpeg: Optional[bytes] = None  # JPEG-encoded RGB preview for viewer
    mask_png: Optional[bytes] = None  # PNG-encoded mask for viewer/debug

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        result = {
            'timestamp': self.timestamp,
            'frame_id': self.frame_id,
            'pose': self.pose.tolist(),
            'processing_time_ms': self.processing_time_ms,
            'frame_shape': self.frame_shape
        }

        # Include camera pose if available
        if self.camera_pose is not None:
            result['camera_pose'] = self.camera_pose.tolist()

        if self.camera_intrinsics is not None:
            result['camera_intrinsics'] = self.camera_intrinsics.tolist()

        if self.preview_jpeg is not None:
            result['preview_jpeg'] = self.preview_jpeg

        if self.mask_png is not None:
            result['mask_png'] = self.mask_png

        return result


def build_sam2_model(sam2_repo: str, sam2_checkpoint: str, device: str, cfg_rel: str):
    if not os.path.isdir(sam2_repo):
        raise RuntimeError(f"SAM2 repo not found: {sam2_repo}")
    sys.path.insert(0, sam2_repo)
    from sam2.build_sam import build_sam2  # type: ignore
    cwd = os.getcwd()
    try:
        sam2_checkpoint = os.path.abspath(sam2_checkpoint)
        os.chdir(sam2_repo)
        model = build_sam2(config_file=cfg_rel, ckpt_path=sam2_checkpoint, device=device)
    finally:
        os.chdir(cwd)
    return model


def generate_masks_with_sam2(image_rgb: np.ndarray, model) -> List[dict]:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
    mask_generator = SAM2AutomaticMaskGenerator(model)
    masks = mask_generator.generate(image_rgb)
    return masks


def load_clip(device: str):
    try:
        import clip  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: 'clip'. Install via: pip install git+https://github.com/openai/CLIP.git"
        ) from e
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, clip


def rank_masks_by_prompt(image_rgb: np.ndarray, masks: List[dict], prompt: str, device: str) -> tuple[np.ndarray, float]:
    """
    Rank SAM2-generated masks using CLIP text-image similarity.
    
    Args:
        image_rgb: RGB image (H, W, 3) uint8
        masks: List of SAM2 mask dictionaries
        prompt: Text prompt describing target object
        device: Device string ('cuda' or 'cpu')
    
    Returns:
        best_mask: Best mask as uint8 array (H, W)
        best_score: CLIP similarity score
    """
    if len(masks) == 0:
        raise RuntimeError("No masks produced by SAM2.")
    
    model, preprocess, clip = load_clip(device)
    
    text = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    best_score = -1.0
    best_mask = None
    
    for m in masks:
        seg = m.get("segmentation", None)
        if seg is None:
            continue
        seg = seg.astype(bool)
        if seg.sum() == 0:
            continue
        
        masked = np.zeros_like(image_rgb)
        masked[seg] = image_rgb[seg]
        
        ys, xs = np.where(seg)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        crop = masked[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        
        pil_img = Image.fromarray(crop)
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(input_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            score = float((image_features @ text_features.T).squeeze().item())
        
        if score > best_score:
            best_score = score
            best_mask = seg
    
    if best_mask is None:
        raise RuntimeError("Failed to select a mask with CLIP scoring.")
    
    return best_mask.astype(np.uint8) * 255, best_score


class DetectionAlgorithm:
    """
    Detection and tracking algorithm wrapper.
    Uses SAM2 + CLIP for first-frame detection, FoundationPose for tracking.
    """
    
    def __init__(
        self,
        detection_config: DetectionConfig,
        calibration_config: CalibrationConfig
    ):
        """
        Initialize detection algorithm.
        
        Args:
            detection_config: Detection configuration
            calibration_config: Calibration configuration (for camera intrinsics)
        """
        from .config import CalibrationConfig
        self.config = detection_config
        self.calibration_config = calibration_config
        self._initialized = False
        
        # Models
        self.sam2_model = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip = None
        self.mesh = None
        self.estimator = None
        self.glctx = None
        
        # State
        self._current_pose: Optional[np.ndarray] = None
        self._current_mask: Optional[np.ndarray] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def initialize(self) -> bool:
        """
        Load all models (SAM2, CLIP, ScoreNet, RefineNet, FoundationPose).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Initializing detection algorithm...")
            
            # Load mesh
            logger.info(f"Loading mesh from {self.config.mesh_path}")
            self.mesh = trimesh.load(self.config.mesh_path)
            
            # Load SAM2 model
            logger.info("Loading SAM2 model...")
            code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            sam2_repo = os.path.join(code_dir, "segment-anything-2-real-time")
            
            # Auto-select config based on checkpoint name
            cfg_rel = self.config.sam2_cfg
            ckpt_name = os.path.basename(self.config.sam2_checkpoint).lower()
            if "sam2.1" in ckpt_name:
                if "tiny" in ckpt_name:
                    cfg_rel = "configs/sam2.1/sam2.1_hiera_t.yaml"
                elif "small" in ckpt_name:
                    cfg_rel = "configs/sam2.1/sam2.1_hiera_s.yaml"
                elif "base_plus" in ckpt_name or "b_plus" in ckpt_name:
                    cfg_rel = "configs/sam2.1/sam2.1_hiera_b+.yaml"
                elif "large" in ckpt_name:
                    cfg_rel = "configs/sam2.1/sam2.1_hiera_l.yaml"
            
            self.sam2_model = build_sam2_model(
                sam2_repo=sam2_repo,
                sam2_checkpoint=self.config.sam2_checkpoint,
                device=self._device,
                cfg_rel=cfg_rel
            )
            logger.info("✓ SAM2 model loaded")
            
            # Load CLIP
            logger.info("Loading CLIP model...")
            self.clip_model, self.clip_preprocess, self.clip = load_clip(self._device)
            logger.info("✓ CLIP model loaded")
            
            # Initialize FoundationPose
            logger.info("Initializing FoundationPose...")
            self.glctx = dr.RasterizeCudaContext()
            
            self.estimator = FoundationPose(
                model_pts=self.mesh.vertices,
                model_normals=self.mesh.vertex_normals,
                mesh=self.mesh,
                scorer=ScorePredictor(),
                refiner=PoseRefinePredictor(),
                glctx=self.glctx,
                debug=0,
                debug_dir="./debug"
            )
            logger.info("✓ FoundationPose initialized")
            
            self._initialized = True
            logger.info("✓ Detection algorithm initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize detection algorithm: {e}", exc_info=True)
            return False
    
    def detect_first_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run SAM2 + CLIP on frame to extract object mask.
        Uses binary_search_depth to find initial pose (matching test_demo_without_depth.py).
        
        Args:
            frame: RGB image (H, W, 3) uint8
        
        Returns:
            Binary mask (H, W) uint8, or None if detection failed
        """
        if not self._initialized:
            logger.error("Detection algorithm not initialized")
            return None
        
        try:
            logger.info("Running first-frame detection...")
            t0 = time.time()
            
            # Generate masks with SAM2 (matching generate_image_marker.py)
            masks = generate_masks_with_sam2(frame, self.sam2_model)
            
            if len(masks) == 0:
                logger.error("SAM2 produced no masks")
                return None
            
            # Rank masks by CLIP similarity to prompt (matching generate_image_marker.py)
            best_mask, score = rank_masks_by_prompt(
                frame, masks, self.config.prompt, self._device
            )
            
            logger.info(f"Best CLIP score: {score:.4f}")
            
            # Convert mask to boolean
            mask_bool = (best_mask > 0).astype(bool)
            
            # Use binary_search_depth to find initial pose (matching test_demo_without_depth.py line 98)
            logger.info("Running binary_search_depth for initial pose...")
            pose = binary_search_depth(
                est=self.estimator,
                mesh=self.mesh,
                rgb=frame,
                mask=mask_bool,
                K=np.array(self.calibration_config.K, dtype=np.float64),
                debug=False
            )
            
            self._current_pose = pose
            self._current_mask = best_mask
            
            elapsed = (time.time() - t0) * 1000
            logger.info(f"✓ Detection completed in {elapsed:.1f}ms")
            
            return best_mask
            
        except Exception as e:
            logger.error(f"First-frame detection failed: {e}", exc_info=True)
            return None
    
    def track(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Track object in new frame using FoundationPose.track_one().
        Uses render_cad_depth to generate depth for tracking (matching test_demo_without_depth.py).
        
        Args:
            frame: RGB image (H, W, 3) uint8
        
        Returns:
            4x4 pose matrix (object-to-camera), or None if tracking failed
        """
        if not self._initialized:
            logger.error("Detection algorithm not initialized")
            return None
        
        if self._current_pose is None:
            logger.error("No initial pose. Run detect_first_frame() first.")
            return None
        
        try:
            # Render CAD depth from current pose (matching test_demo_without_depth.py line 113)
            K = np.array(self.calibration_config.K, dtype=np.float64)
            depth = render_cad_depth(
                pose=self._current_pose,
                mesh_model=self.mesh,
                K=K,
                w=frame.shape[1],
                h=frame.shape[0]
            )
            
            # Track with rendered depth (matching test_demo_without_depth.py line 115)
            pose = self.estimator.track_one(
                rgb=frame,
                depth=depth,
                K=K,
                iteration=self.config.track_refine_iter
            )
            
            self._current_pose = pose
            # Update mask from estimator's last mask
            if hasattr(self.estimator, 'mask_last') and self.estimator.mask_last is not None:
                self._current_mask = (self.estimator.mask_last * 255).astype(np.uint8)
            
            return pose
            
        except Exception as e:
            logger.error(f"Tracking failed: {e}", exc_info=True)
            return None
    
    def reset(self):
        """Clear tracking state, reset to initial state"""
        logger.info("Resetting detection algorithm...")
        self._current_pose = None
        self._current_mask = None
        if self.estimator is not None:
            self.estimator.pose_last = None
            self.estimator.track_good = False
        logger.info("✓ Detection algorithm reset")
    
    @property
    def current_pose(self) -> Optional[np.ndarray]:
        """Get current 4x4 pose matrix"""
        return self._current_pose
    
    @property
    def current_mask(self) -> Optional[np.ndarray]:
        """Get current object mask"""
        return self._current_mask
    
    @property
    def is_initialized(self) -> bool:
        """Check if detector is initialized"""
        return self._initialized