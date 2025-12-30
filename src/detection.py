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
from .config import DetectionConfig
import nvdiffrast.torch as dr
from estimater import ScorePredictor, PoseRefinePredictor, FoundationPose
from tools import binary_search_depth, render_cad_depth, draw_posed_3d_box, draw_xyz_axis

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Detection result data structure"""
    timestamp: float
    frame_id: int
    pose: np.ndarray
    processing_time_ms: float
    frame_shape: tuple

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'frame_id': self.frame_id,
            'pose': self.pose.tolist(),
            'processing_time_ms': self.processing_time_ms,
            'frame_shape': self.frame_shape
        }


def build_sam2_model(sam2_repo: str, sam2_checkpoint: str, device: str, cfg_rel: str):
    if not os.path.isdir(sam2_repo):
        raise RuntimeError(f"SAM2 repo not found: {sam2_repo}")
    sys.path.insert(0, sam2_repo)
    from sam2.build_sam import build_sam2  # type: ignore
    cwd = os.getcwd()
    try:
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


class DetectionAlgorithm:
    """
    Detection algorithm wrapper.
    Replace this with your actual detection algorithm.
    """

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.frame_count = 0
        logger.info(f"Initializing detection algorithm")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        cfg_rel = self.config.sam2_cfg
        if self.config.sam2_checkpoint:
            ckpt_name = os.path.basename(self.config.sam2_checkpoint).lower()
            if "sam2.1" in ckpt_name:
                if "tiny" in ckpt_name:
                    cfg_rel = "configs/sam2.1/sam2.1_hiera_t.yaml"
                elif "small" in ckpt_name:
                    cfg_rel = "configs/sam2.1/sam2.1_hiera_s.yaml"
                elif "base_plus" in ckpt_name or "b_plus" in ckpt_name or "b+.pt" in ckpt_name or "_b+." in ckpt_name:
                    cfg_rel = "configs/sam2.1/sam2.1_hiera_b+.yaml"
                elif "large" in ckpt_name or "hiera_l" in ckpt_name:
                    cfg_rel = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.sam2_model = build_sam2_model("./segment-anything-2-real-time", self.config.sam2_checkpoint, self.device, cfg_rel)  # FIXME: hardcoded repo path

        # Prepare CLIP
        self.clip_model, self.clip_preprocess, self.clip = load_clip(self.device)
        with torch.no_grad():
            clip_text = self.clip.tokenize([self.config.prompt]).to(self.device)
            self.text_features = self.clip_model.encode_text(clip_text)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        self.mask_u8: np.ndarray | None = None
        self.mask_index = 0
        self.score = 0.0
        self.last_pose: Optional[np.ndarray] = None

        # Prepare Mesh
        self.mesh = trimesh.load(self.config.mesh_path)
        self.to_origin, self.extents = bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-self.extents / 2, self.extents / 2], axis=0).reshape(2, 3)
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            debug_dir='/tmp/estimation_debug',
            debug=False,
            glctx=self.glctx,
        )
        self.K = np.array(self.config.K)

        # mode selection
        self.mode = self.config.mode  # 0: depth search, 1: last depth
        self.track_refine_iter = self.config.track_refine_iter

    def _rank_masks_by_prompt(self, image_rgb: np.ndarray, masks: List[dict], prompt: str, device: str) -> tuple[np.ndarray, float]:
        if len(masks) == 0:
            raise RuntimeError("No masks produced by SAM2.")

        best_score = -1.0
        best_mask = None

        with torch.no_grad():
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
                input_tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(device)  # pyright: ignore[reportAttributeAccessIssue, reportGeneralTypeIssues]
                image_features = self.clip_model.encode_image(input_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                score = float((image_features @ self.text_features.T).squeeze().item())

                if score > best_score:
                    best_score = score
                    best_mask = seg
            if best_mask is None:
                raise RuntimeError("Failed to select a mask with CLIP scoring.")

        return best_mask.astype(np.uint8) * 255, best_score

    def _initialize_mask_and_score(self, frame: np.ndarray):
        # frame is rgb
        masks = generate_masks_with_sam2(frame, self.sam2_model)
        try:
            self.mask_u8, self.score = self._rank_masks_by_prompt(
                frame,
                masks,
                self.config.prompt,
                self.device
            )
            self.mask_index = 0
            return True
        except RuntimeError as e:
            logger.error(f"Error during mask ranking: {e}")
            self.mask_u8 = None
            self.mask_index = 0
            self.score = 0.0
        finally:
            return False

    def detect(self, frame: np.ndarray) -> DetectionResult | None:
        """
        Run detection on a frame.

        Args:
            frame: Input image (RGB format)

        Returns:
            DetectionResult object
        """
        start_time = time.time()
        timestamp = time.time()

        if self.mask_u8 is None:
            if not self._initialize_mask_and_score(frame):
                return None

        if self.mask_index == 0:
            pose = binary_search_depth(self.est, self.mesh, frame, self.mask_u8, self.K, debug=True)
            self.last_pose = pose
        else:
            if self.mode == 0:
                last_depth = np.zeros_like(self.mask_u8)
            elif self.mode == 1:
                last_depth = render_cad_depth(self.last_pose, self.mesh, self.K)
            pose = self.est.track_one(
                rgb=frame, depth=last_depth, K=self.K, iteration=self.track_refine_iter
            )
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        result = DetectionResult(
            timestamp=timestamp,
            frame_id=self.frame_count,
            pose=pose,
            processing_time_ms=processing_time,
            frame_shape=frame.shape
        )
        self.mask_index += 1
        self.frame_count += 1
        return result

    def draw_detections(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """
        Draw detection results on frame.

        Args:
            frame: Input frame
            result: Detection result

        Returns:
            Frame with drawings
        """
        output = frame.copy()
        center_pose = result.pose @ np.linalg.inv(self.to_origin)
        vis = draw_posed_3d_box(
            self.K, img=output, ob_in_cam=center_pose, bbox=self.bbox
        )
        vis = draw_xyz_axis(
            output,
            ob_in_cam=center_pose,
            scale=0.1,
            K=self.K, # pyright: ignore[reportArgumentType]
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )

        # Draw statistics
        stats_text = f"Time: {result.processing_time_ms:.1f}ms"
        vis = cv2.putText(
            output,
            stats_text,
            (10, output.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        return vis
