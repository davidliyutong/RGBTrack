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


class PoseKalmanFilter:
    """
    Kalman Filter for 6D pose (SE(3)) filtering.

    State: [x, y, z, qw, qx, qy, qz, vx, vy, vz] (10D)
    - Position: (x, y, z)
    - Orientation: quaternion (qw, qx, qy, qz) with qw as real part
    - Velocity: (vx, vy, vz)

    Observation: [x, y, z, qw, qx, qy, qz] (7D)
    """

    def __init__(self, dt: float = 0.033, process_noise: float = 0.01, measurement_noise: float = 0.05):
        """
        Initialize Kalman Filter for 6D pose.

        Args:
            dt: Time step between frames (default 0.033 for 30fps)
            process_noise: Process noise magnitude
            measurement_noise: Measurement noise magnitude
        """
        self.dt = dt
        self.initialized = False

        # State dimension: 10 (xyz + quaternion + velocity)
        self.state_dim = 10
        # Observation dimension: 7 (xyz + quaternion)
        self.obs_dim = 7

        # State vector: [x, y, z, qw, qx, qy, qz, vx, vy, vz]
        self.x = np.zeros(self.state_dim)

        # State covariance matrix
        self.P = np.eye(self.state_dim) * 1.0

        # State transition matrix (constant velocity model)
        self.F = np.eye(self.state_dim)
        self.F[0, 7] = dt  # x = x + vx * dt
        self.F[1, 8] = dt  # y = y + vy * dt
        self.F[2, 9] = dt  # z = z + vz * dt
        # Quaternion and velocity remain unchanged in transition

        # Observation matrix (we observe position and quaternion, not velocity)
        self.H = np.zeros((self.obs_dim, self.state_dim))
        self.H[0:7, 0:7] = np.eye(7)  # Observe xyz and quaternion

        # Process noise covariance
        self.Q = np.eye(self.state_dim) * process_noise
        self.Q[0:3, 0:3] *= 0.1  # Lower noise for position
        self.Q[3:7, 3:7] *= 0.05  # Lower noise for quaternion
        self.Q[7:10, 7:10] *= 1.0  # Higher noise for velocity

        # Measurement noise covariance
        self.R = np.eye(self.obs_dim) * measurement_noise
        self.R[0:3, 0:3] *= 1.0  # Position measurement noise
        self.R[3:7, 3:7] *= 0.5  # Quaternion measurement noise

    def _normalize_quaternion(self):
        """Ensure quaternion part of state is normalized."""
        quat = self.x[3:7]
        norm = np.linalg.norm(quat)
        if norm > 1e-6:
            self.x[3:7] = quat / norm

    def _pose_to_observation(self, pose: np.ndarray) -> np.ndarray:
        """
        Convert 4x4 pose matrix to observation vector.

        Args:
            pose: 4x4 transformation matrix

        Returns:
            7D observation [x, y, z, qw, qx, qy, qz]
        """
        # Extract translation
        translation = pose[:3, 3]

        # Extract rotation and convert to quaternion
        rotation_matrix = pose[:3, :3]
        rotation = Rotation.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()  # Returns [x, y, z, w] # type: ignore

        # Convert to [w, x, y, z] format
        quaternion_wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])

        # Combine into observation
        observation = np.concatenate([translation, quaternion_wxyz])
        return observation

    def _observation_to_pose(self, obs: np.ndarray) -> np.ndarray:
        """
        Convert observation vector to 4x4 pose matrix.

        Args:
            obs: 7D observation [x, y, z, qw, qx, qy, qz]

        Returns:
            4x4 transformation matrix
        """
        pose = np.eye(4)

        # Set translation
        pose[:3, 3] = obs[:3]

        # Convert quaternion to rotation matrix
        # obs format: [x, y, z, qw, qx, qy, qz]
        quat_xyzw = np.array([obs[4], obs[5], obs[6], obs[3]])  # [qx, qy, qz, qw]
        rotation = Rotation.from_quat(quat_xyzw)
        pose[:3, :3] = rotation.as_matrix()

        return pose

    def initialize(self, pose: np.ndarray):
        """
        Initialize the filter with the first pose measurement.

        Args:
            pose: 4x4 transformation matrix
        """
        obs = self._pose_to_observation(pose)

        # Initialize state
        self.x[:7] = obs
        self.x[7:] = 0.0  # Zero velocity

        # Reset covariance
        self.P = np.eye(self.state_dim) * 1.0
        self.P[7:10, 7:10] = 10.0  # High uncertainty in velocity initially

        self.initialized = True
        logger.info("Kalman filter initialized")

    def predict(self):
        """Prediction step of Kalman filter."""
        if not self.initialized:
            return

        # State prediction: x = F * x
        self.x = self.F @ self.x

        # Normalize quaternion after prediction
        self._normalize_quaternion()

        # Covariance prediction: P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, pose: np.ndarray) -> np.ndarray:
        """
        Update step of Kalman filter with new pose measurement.

        Args:
            pose: 4x4 transformation matrix (measurement)

        Returns:
            Filtered 4x4 transformation matrix
        """
        if not self.initialized:
            self.initialize(pose)
            return pose

        # Convert pose to observation
        z = self._pose_to_observation(pose)

        # Handle quaternion sign ambiguity (q and -q represent same rotation)
        state_quat = self.x[3:7]
        meas_quat = z[3:7]
        if np.dot(state_quat, meas_quat) < 0:
            z[3:7] = -z[3:7]

        # Innovation: y = z - H * x
        y = z - self.H @ self.x

        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: K = P * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update: x = x + K * y
        self.x = self.x + K @ y

        # Normalize quaternion after update
        self._normalize_quaternion()

        # Covariance update: P = (I - K * H) * P
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ self.P

        # Convert filtered state back to pose
        filtered_obs = self.x[:7]
        filtered_pose = self._observation_to_pose(filtered_obs)

        return filtered_pose

    def filter_pose(self, pose: np.ndarray) -> np.ndarray:
        """
        Apply Kalman filtering to a pose measurement (predict + update).

        Args:
            pose: 4x4 transformation matrix (measurement)

        Returns:
            Filtered 4x4 transformation matrix
        """
        self.predict()
        return self.update(pose)

    def reset(self):
        """Reset the filter."""
        self.initialized = False
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 1.0


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


class DetectionAlgorithm:
    """
    Detection algorithm wrapper.
    Replace this with your actual detection algorithm.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.detection_config = config.detection
        self.frame_count = 0
        set_seed(0)
        logger.info(f"Initializing detection algorithm")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        cfg_rel = self.detection_config.sam2_cfg
        if self.detection_config.sam2_checkpoint:
            ckpt_name = os.path.basename(self.detection_config.sam2_checkpoint).lower()
            if "sam2.1" in ckpt_name:
                if "tiny" in ckpt_name:
                    cfg_rel = "configs/sam2.1/sam2.1_hiera_t.yaml"
                elif "small" in ckpt_name:
                    cfg_rel = "configs/sam2.1/sam2.1_hiera_s.yaml"
                elif "base_plus" in ckpt_name or "b_plus" in ckpt_name or "b+.pt" in ckpt_name or "_b+." in ckpt_name:
                    cfg_rel = "configs/sam2.1/sam2.1_hiera_b+.yaml"
                elif "large" in ckpt_name or "hiera_l" in ckpt_name:
                    cfg_rel = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.sam2_model = build_sam2_model("./segment-anything-2-real-time", self.detection_config.sam2_checkpoint, self.device, cfg_rel)

        # Prepare CLIP
        self.clip_model, self.clip_preprocess, self.clip = load_clip(self.device)
        with torch.no_grad():
            clip_text = self.clip.tokenize([self.detection_config.prompt]).to(self.device)
            self.text_features = self.clip_model.encode_text(clip_text)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        self.mask_u8: np.ndarray | None = None
        self.mask_index = 0
        self.score = 0.0
        self.last_pose: Optional[np.ndarray] = None

        # Prepare Mesh
        self.mesh = trimesh.load(self.detection_config.mesh_path)
        self.to_origin, self.extents = bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-self.extents / 2, self.extents / 2], axis=0).reshape(2, 3)
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(
            model_pts=self.mesh.vertices, # type: ignore
            model_normals=self.mesh.vertex_normals, # type: ignore
            mesh=self.mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            debug_dir='/tmp/estimation_debug',
            debug=False,
            glctx=self.glctx,
        )
        self.K = np.array(self.config.calibration.K)

        # Camera extrinsics (camera pose in world frame)
        self.camera_pose = np.array(self.config.calibration.extrinsic_matrix)
        logger.info(f"Camera pose:\n{self.camera_pose}")

        # mode selection
        self.mode = self.detection_config.mode  # 0: depth search, 1: last depth
        self.track_refine_iter = self.detection_config.track_refine_iter

        # Initialize Kalman filter for pose smoothing
        fps = getattr(self.detection_config, 'fps', 30)  # Default to 30fps if not specified
        dt = 1.0 / fps
        process_noise = getattr(self.detection_config, 'kalman_process_noise', 0.01)
        measurement_noise = getattr(self.detection_config, 'kalman_measurement_noise', 0.05)
        self.use_kalman = self.detection_config.use_kalman_filter

        self.kalman_filter = PoseKalmanFilter(
            dt=dt,
            process_noise=process_noise,
            measurement_noise=measurement_noise
        )
        logger.info(f"Kalman filter configured: enabled={self.use_kalman}, dt={dt:.4f}, "
                    f"process_noise={process_noise}, measurement_noise={measurement_noise}")

    def _encode_preview_jpeg(self, frame_rgb: np.ndarray, max_width: int = 640, jpeg_quality: int = 70) -> Optional[bytes]:
        """Encode a lightweight RGB preview with mask overlay for viewer streaming."""
        try:
            preview = frame_rgb.copy()

            if self.mask_u8 is not None and self.mask_u8.shape == preview.shape[:2]:
                mask_bool = self.mask_u8 > 0
                if np.any(mask_bool):
                    overlay = preview.copy()
                    overlay[mask_bool] = np.array([255, 0, 0], dtype=np.uint8)
                    preview = cv2.addWeighted(preview, 0.65, overlay, 0.35, 0)

            h, w = preview.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                new_size = (max_width, max(1, int(h * scale)))
                preview = cv2.resize(preview, new_size, interpolation=cv2.INTER_AREA)

            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(
                ".jpg",
                preview_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            if not ok:
                return None
            return encoded.tobytes()
        except Exception as e:
            logger.debug(f"Failed to encode preview jpeg: {e}")
            return None

    def _encode_mask_png(self) -> Optional[bytes]:
        """Encode current mask to PNG bytes."""
        if self.mask_u8 is None:
            return None
        try:
            ok, encoded = cv2.imencode(".png", self.mask_u8)
            if not ok:
                return None
            return encoded.tobytes()
        except Exception as e:
            logger.debug(f"Failed to encode mask png: {e}")
            return None

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
                self.detection_config.prompt,
                self.device
            )
            self.mask_index = 0
            return True
        except RuntimeError as e:
            logger.error(f"Error during mask ranking: {e}")
            self.mask_u8 = None
            self.mask_index = 0
            self.score = 0.0
            return False

    def detect(self, frame: np.ndarray) -> Optional[DetectionResult]:
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

        H, W = frame.shape[:2]

        if self.mask_index == 0:
            mask_bool = self.mask_u8.astype(bool) # type: ignore
            pose = binary_search_depth(self.est, self.mesh, frame, mask_bool, self.K, w=W, h=H, debug=True)
            self.last_mask = mask_bool
            self.last_pose = pose
            # Reset Kalman filter on new initialization
            if self.use_kalman:
                self.kalman_filter.reset()
        else:
            if self.mode == 0:
                last_depth = np.zeros_like(self.last_mask)
            else:  # mode == 1 or fallback
                last_depth = render_cad_depth(self.last_pose, self.mesh, self.K, w=W, h=H)
            pose = self.est.track_one(
                rgb=frame, depth=last_depth, K=self.K, iteration=self.track_refine_iter
            )

        # Store raw pose for tracking (before Kalman filtering)
        self.last_pose = pose

        # Apply Kalman filtering to smooth the pose for output
        if self.use_kalman:
            pose = self.kalman_filter.filter_pose(pose)
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        preview_jpeg = self._encode_preview_jpeg(frame)
        mask_png = self._encode_mask_png()
        result = DetectionResult(
            timestamp=timestamp,
            frame_id=self.frame_count,
            pose=pose,
            processing_time_ms=processing_time,
            frame_shape=frame.shape,
            camera_pose=self.camera_pose,  # Include camera extrinsics
            camera_intrinsics=self.K,
            preview_jpeg=preview_jpeg,
            mask_png=mask_png,
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
            K=self.K,  # pyright: ignore[reportArgumentType]
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
