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
from .config import DetectionConfig, SystemConfig
import nvdiffrast.torch as dr
from estimater import ScorePredictor, PoseRefinePredictor, FoundationPose
from tools import binary_search_depth, render_cad_depth, draw_posed_3d_box, draw_xyz_axis, set_seed
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