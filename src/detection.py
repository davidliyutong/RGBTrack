"""Detection algorithm module"""

import os
import sys
import gc
import logging
import threading
import time
from collections import deque
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
    timestamp: int
    frame_id: int
    pose: np.ndarray  # Object pose in camera frame (4x4)
    linvel: np.ndarray  # Linear velocity (3,)
    angvel: np.ndarray  # Angular velocity (3,)
    processing_time_ms: float
    frame_shape: tuple

    # NEW: FPS metrics
    camera_fps: float = 0.0
    inference_fps: float = 0.0
    # Camera pose in world frame (4x4)
    camera_pose: Optional[np.ndarray] = None
    # Camera intrinsics matrix (3x3)
    camera_intrinsics: Optional[np.ndarray] = None
    preview_jpeg: Optional[bytes] = None  # JPEG-encoded RGB preview for viewer
    mask_png: Optional[bytes] = None  # PNG-encoded mask for viewer/debug

    valid: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        result = {
            'timestamp': self.timestamp,
            'frame_id': self.frame_id,
            'pose': self.pose.tolist(),
            'linvel': self.linvel.tolist() if self.linvel is not None else None,
            'angvel': self.angvel.tolist() if self.angvel is not None else None,
            'processing_time_ms': self.processing_time_ms,
            'frame_shape': self.frame_shape,
            # NEW: FPS metrics
            'camera_fps': self.camera_fps,
            'inference_fps': self.inference_fps,
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
        model = build_sam2(config_file=cfg_rel,
                           ckpt_path=sam2_checkpoint, device=device)
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
        self.estimator: Optional[FoundationPose] = None
        self.glctx = None
        self._sam2_model_loaded = False

        # State
        self._current_pose: Optional[np.ndarray] = None
        self._current_linvel: np.ndarray = np.array([0.0, 0.0, 0.0])
        self._current_angvel: np.ndarray = np.array([0.0, 0.0, 0.0])
        self._current_mask: Optional[np.ndarray] = None
        self._prev_pose: Optional[np.ndarray] = None
        self._prev_pose_time: Optional[float] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Inference resolution limits (separate for detect vs track)
        self._detect_max_shorter_side: int = getattr(
            detection_config, 'detect_max_shorter_side', 480
        ) or 0
        self._track_max_shorter_side: int = getattr(
            detection_config, 'track_max_shorter_side', 0
        ) or 0

        # NEW: FPS tracking
        self._inference_timestamps: deque = deque(maxlen=5)
        self._fps_lock = threading.Lock()
        self._last_inference_time_ms: float = 0.0

    @property
    def fps(self) -> float:
        """Get current inference FPS (averaged over last 5 frames)"""
        with self._fps_lock:
            if len(self._inference_timestamps) < 2:
                return 0.0
            dt = self._inference_timestamps[-1] - self._inference_timestamps[0]
            if dt > 0:
                return (len(self._inference_timestamps) - 1) / dt
            return 0.0

    @property
    def last_inference_time_ms(self) -> float:
        """Get last inference processing time in ms"""
        return self._last_inference_time_ms

    def _record_inference(self, start_time: float):
        """Record inference completion for FPS calculation"""
        with self._fps_lock:
            self._inference_timestamps.append(time.time())
            self._last_inference_time_ms = (time.time() - start_time) * 1000

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

            # Initialize FoundationPose
            logger.info("Initializing FoundationPose...")
            self.glctx = dr.RasterizeCudaContext()

            self.estimator = FoundationPose(
                model_pts=self.mesh.vertices,  # type: ignore
                model_normals=self.mesh.vertex_normals,  # type: ignore
                mesh=self.mesh,
                scorer=ScorePredictor(),
                refiner=PoseRefinePredictor(),
                glctx=self.glctx,
                debug=0,
                debug_dir="./debug"
            )
            logger.info("✓ FoundationPose initialized")

            # Keep CLIP resident for fast restart/re-detection.
            logger.info("Loading CLIP model...")
            self.clip_model, self.clip_preprocess, self.clip = load_clip(self._device)
            logger.info("✓ CLIP model loaded")

            # Keep SAM2 resident for fast restart/re-detection.
            self._load_first_frame_models()
            self._log_cuda_memory("after FoundationPose init")

            self._initialized = True
            logger.info("✓ Detection algorithm initialized successfully")
            return True

        except Exception as e:
            logger.error(
                f"Failed to initialize detection algorithm: {e}", exc_info=True)
            return False

    def _load_first_frame_models(self) -> None:
        if self._sam2_model_loaded:
            return

        logger.info("Loading first-frame model (SAM2)...")
        code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        sam2_repo = os.path.join(code_dir, "segment-anything-2-real-time")

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
            cfg_rel=cfg_rel,
        )
        self._sam2_model_loaded = True
        self._log_cuda_memory("after loading SAM2")

    def _unload_first_frame_models(self) -> None:
        if not self._sam2_model_loaded:
            return

        logger.info("Unloading first-frame model (SAM2) to reduce CUDA usage")
        self.sam2_model = None
        self._sam2_model_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._log_cuda_memory("after unloading SAM2")

    # ---- resolution helpers ------------------------------------------------
    def _resize_for_inference(
        self, frame: np.ndarray, K: np.ndarray,
        mask: Optional[np.ndarray] = None,
        max_shorter_side: int = 0,
    ) -> tuple:
        """Downscale *frame* (and optionally *mask*) so the shorter side
        is <= *max_shorter_side*.  Returns ``(frame, K, mask, scale)``.
        *scale* is the factor applied (1.0 = no resize).
        K is adjusted in-place-safe (a copy is returned)."""
        if max_shorter_side <= 0:
            return frame, K.copy(), mask, 1.0
        h, w = frame.shape[:2]
        shorter = min(h, w)
        if shorter <= max_shorter_side:
            return frame, K.copy(), mask, 1.0
        scale = max_shorter_side / shorter
        new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
        frame_s = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        K_s = K.copy()
        K_s[0, :] *= scale
        K_s[1, :] *= scale
        mask_s = None
        if mask is not None:
            mask_s = cv2.resize(
                mask.astype(np.uint8), (new_w, new_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(mask.dtype)
        logger.info(
            "Resized frame %dx%d -> %dx%d (scale %.3f) for inference",
            w, h, new_w, new_h, scale,
        )
        return frame_s, K_s, mask_s, scale

    @staticmethod
    def _upscale_mask(mask: np.ndarray, orig_hw: tuple, scale: float) -> np.ndarray:
        """Upscale a binary mask back to original resolution."""
        if scale >= 1.0:
            return mask
        return cv2.resize(
            mask, (orig_hw[1], orig_hw[0]),
            interpolation=cv2.INTER_NEAREST
        )

    def _log_cuda_memory(self, stage: str) -> None:
        if not torch.cuda.is_available():
            return
        try:
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            logger.info(
                "CUDA memory (%s): allocated=%.1fMB reserved=%.1fMB",
                stage,
                allocated,
                reserved,
            )
        except Exception:
            pass

    def rank_masks_by_prompt(self, image_rgb: np.ndarray, masks: List[dict], prompt: str, device: str) -> tuple[np.ndarray, float]:
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

        text = self.clip.tokenize([prompt]).to(device)  # type: ignore
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)  # type: ignore
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
            input_tensor = self.clip_preprocess(
                pil_img).unsqueeze(0).to(device)  # type: ignore

            with torch.no_grad():
                image_features = self.clip_model.encode_image(  # type: ignore
                    input_tensor)  # type: ignore
                image_features /= image_features.norm(dim=-1, keepdim=True)
                score = float(
                    (image_features @ text_features.T).squeeze().item())

            if score > best_score:
                best_score = score
                best_mask = seg

        if best_mask is None:
            raise RuntimeError("Failed to select a mask with CLIP scoring.")

        return best_mask.astype(np.uint8) * 255, best_score

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
            self._load_first_frame_models()

            orig_hw = frame.shape[:2]
            K_orig = np.array(self.calibration_config.K, dtype=np.float64)

            # Generate masks with SAM2 (matching generate_image_marker.py)
            masks = generate_masks_with_sam2(frame, self.sam2_model)

            if len(masks) == 0:
                logger.error("SAM2 produced no masks")
                return None

            # Rank masks by CLIP similarity to prompt (matching generate_image_marker.py)
            best_mask, score = self.rank_masks_by_prompt(
                frame, masks, self.config.prompt, self._device
            )

            logger.info(f"Best CLIP score: {score:.4f}")

            # Convert mask to boolean
            mask_bool = (best_mask > 0).astype(bool)

            # Downscale for FoundationPose to avoid GPU OOM on high-res frames
            frame_s, K_s, mask_s, scale = self._resize_for_inference(
                frame, K_orig, mask_bool,
                max_shorter_side=self._detect_max_shorter_side,
            )

            # Use binary_search_depth to find initial pose (matching test_demo_without_depth.py line 98)
            logger.info("Running binary_search_depth for initial pose...")
            pose = binary_search_depth(
                est=self.estimator,
                mesh=self.mesh,
                rgb=frame_s,
                mask=mask_s,
                K=K_s,
                debug=False
            )

            self._current_pose = pose
            self._current_mask = best_mask  # keep at original resolution
            self._current_linvel = np.zeros(3, dtype=np.float64)
            self._current_angvel = np.zeros(3, dtype=np.float64)
            self._prev_pose = pose.copy()
            self._prev_pose_time = time.time()

            # Record inference completion
            self._record_inference(t0)

            elapsed = (time.time() - t0) * 1000
            logger.info(f"✓ Detection completed in {elapsed:.1f}ms")

            return best_mask

        except Exception as e:
            logger.error(f"First-frame detection failed: {e}", exc_info=True)
            return None

    def track(self, frame: np.ndarray) -> tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Track object in new frame using FoundationPose.track_one().
        Uses render_cad_depth to generate depth for tracking (matching test_demo_without_depth.py).

        Args:
            frame: RGB image (H, W, 3) uint8

        Returns:
            4x4 pose matrix (object-to-camera), linear velocity (3,), angular velocity (3,), or None if tracking failed
        """
        if not self._initialized:
            logger.error("Detection algorithm not initialized")
            return None, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])

        if self._current_pose is None:
            logger.error("No initial pose. Run detect_first_frame() first.")
            return None, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])

        try:
            t0 = time.time()

            K_orig = np.array(self.calibration_config.K, dtype=np.float64)

            # Optionally downscale for tracking (default 0 = full resolution)
            frame_t, K_t, _, _ = self._resize_for_inference(
                frame, K_orig,
                max_shorter_side=self._track_max_shorter_side,
            )

            # Render CAD depth from current pose (matching test_demo_without_depth.py line 113)
            depth = render_cad_depth(
                pose=self._current_pose,
                mesh_model=self.mesh,
                K=K_t,
                w=frame_t.shape[1],
                h=frame_t.shape[0]
            )

            # Track with rendered depth (matching test_demo_without_depth.py line 115)
            pose = self.estimator.track_one(  # type: ignore
                rgb=frame_t,
                depth=depth,
                K=K_t,
                iteration=self.config.track_refine_iter
            )

            now = time.time()
            self._update_velocities(pose, now)
            self._current_pose = pose
            # Update mask from estimator's last mask
            if hasattr(self.estimator, 'mask_last') and self.estimator.mask_last is not None:  # type: ignore
                self._current_mask = (self.estimator.mask_last * 255).astype(np.uint8)  # type: ignore

            # Record inference completion
            self._record_inference(t0)

            return pose, self._current_linvel if self._current_linvel is not None else np.array([0.0, 0.0, 0.0]), self._current_angvel if self._current_angvel is not None else np.array([0.0, 0.0, 0.0])

        except Exception as e:
            logger.error(f"Tracking failed: {e}", exc_info=True)
            return None, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])

    def _update_velocities(self, new_pose: np.ndarray, now: float) -> None:
        """Compute linear and angular velocity from consecutive poses."""
        if self._prev_pose is None or self._prev_pose_time is None:
            self._current_linvel = np.zeros(3, dtype=np.float64)
            self._current_angvel = np.zeros(3, dtype=np.float64)
            self._prev_pose = new_pose.copy()
            self._prev_pose_time = now
            return

        dt = now - self._prev_pose_time
        if dt < 1e-9:
            return

        # Linear velocity: d(translation) / dt
        self._current_linvel = (new_pose[:3, 3] - self._prev_pose[:3, 3]) / dt

        # Angular velocity from relative rotation
        R_prev = self._prev_pose[:3, :3]
        R_curr = new_pose[:3, :3]
        R_rel = R_curr @ R_prev.T
        rotvec = Rotation.from_matrix(R_rel).as_rotvec()
        self._current_angvel = rotvec / dt

        self._prev_pose = new_pose.copy()
        self._prev_pose_time = now

    def reset(self):
        """Clear tracking state, reset to initial state"""
        logger.info("Resetting detection algorithm...")
        self._current_pose = None
        self._current_mask = None
        self._current_linvel = np.array([0.0, 0.0, 0.0])
        self._current_angvel = np.array([0.0, 0.0, 0.0])
        self._prev_pose = None
        self._prev_pose_time = None
        if self.estimator is not None:
            self.estimator.pose_last = None
            self.estimator.track_good = False
        # Reset FPS tracking
        with self._fps_lock:
            self._inference_timestamps.clear()
            self._last_inference_time_ms = 0.0
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
    def current_linvel(self) -> np.ndarray:
        """Get current linear velocity (3,) in m/s"""
        return self._current_linvel

    @property
    def current_angvel(self) -> np.ndarray:
        """Get current angular velocity (3,) in rad/s (rotation vector form)"""
        return self._current_angvel

    @property
    def is_initialized(self) -> bool:
        """Check if detector is initialized"""
        return self._initialized
