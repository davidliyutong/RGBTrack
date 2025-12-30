"""Camera calibration using AprilTag boards"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json
import aprilgrid
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class AprilTagBoardConfig:
    """Configuration for AprilTag board"""

    # Available AprilTag families
    APRILTAG_FAMILIES = {
        "t16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "t25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "t36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }

    def __init__(
        self,
        family: str = "t36h11",
        tags_x: int = 4,
        tags_y: int = 3,
        tag_size: float = 0.04,  # meters
        tag_spacing: float = 0.01,  # meters (spacing between tags)
        first_marker_id: int = 0,
    ):
        """
        Initialize AprilTag board configuration.

        Args:
            family: AprilTag family name (e.g., "t36h11")
            tags_x: Number of tags in X direction
            tags_y: Number of tags in Y direction
            tag_size: Size of each tag in meters
            tag_spacing: Spacing between tags in meters
            first_marker_id: ID of the first marker
        """
        self.family = family
        self.tags_x = tags_x
        self.tags_y = tags_y
        self.tag_size = tag_size
        self.tag_spacing = tag_spacing
        self.first_marker_id = first_marker_id

        if family not in self.APRILTAG_FAMILIES:
            raise ValueError(f"Unknown AprilTag family: {family}")

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.APRILTAG_FAMILIES[family])

        # Create GridBoard for AprilTag
        self.board = cv2.aruco.GridBoard(  # type: ignore
            (tags_x, tags_y),
            tag_size,
            tag_spacing,
            self.aruco_dict,
            ids=np.arange(first_marker_id, first_marker_id + tags_x * tags_y).reshape(-1, 1)
        )
        self.detector_params = cv2.aruco.DetectorParameters()
        self.object_points_3d = self._generate_object_points()

    def _generate_object_points(self) -> Dict[int, np.ndarray]:
        """
        Generate 3D object points for each tag in the grid.

        Returns:
            Dictionary mapping tag ID to 4 corner points in 3D
        """
        object_points = {}
        center_distance = self.tag_size + self.tag_spacing
        half_size = self.tag_size / 2.0

        for row in range(self.tags_y):
            for col in range(self.tags_x):
                tag_id = self.first_marker_id + row * self.tags_x + col

                # Center of the tag in the grid
                center_x = col * center_distance
                center_y = row * center_distance

                # 4 corners of the tag (counter-clockwise from bottom-left)
                corners = np.array([
                    [center_x - half_size, center_y - half_size, 0],  # bottom-left
                    [center_x + half_size, center_y - half_size, 0],  # bottom-right
                    [center_x + half_size, center_y + half_size, 0],  # top-right
                    [center_x - half_size, center_y + half_size, 0],  # top-left
                ], dtype=np.float32)

                object_points[tag_id] = corners

        return object_points

    @classmethod
    def get_available_families(cls) -> List[str]:
        """Get list of available AprilTag family names"""
        return list(cls.APRILTAG_FAMILIES.keys())


class CameraCalibrator:
    """Camera calibration using AprilTag boards"""

    def __init__(self, board_config: AprilTagBoardConfig):
        """
        Initialize calibrator.

        Args:
            board_config: AprilTag board configuration
        """
        self.board_config = board_config
        self.calibration_images: List[np.ndarray] = []
        self.all_object_points: List[np.ndarray] = []
        self.all_marker_corners: List[np.ndarray] = []
        self.all_marker_ids: List[np.ndarray] = []
        self.image_size: Optional[Tuple[int, int]] = None

    def add_image(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Add a calibration image and detect board markers/corners.

        Args:
            image: Input image (BGR or RGB)

        Returns:
            Tuple of (success, annotated_image)
            - success: True if board was detected
            - annotated_image: Image with detected corners/markers drawn
        """
        return self._add_apriltag_image(image)

    def _add_apriltag_image(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Add AprilTag calibration image"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Store image size
        if self.image_size is None:
            self.image_size = (gray.shape[1], gray.shape[0])

        # Detect AprilTag markers
        detector = aprilgrid.Detector(
            tag_family_name=self.board_config.family,
            refine_edges=True,
            decode_sharpening=0.25
        )
        detections = detector.detect(gray)
        object_points = []
        marker_corners = []
        marker_ids = []
        for detection in detections:
            if detection.tag_id in self.board_config.object_points_3d:
                object_points.append(self.board_config.object_points_3d[detection.tag_id])
                marker_corners.append(np.array(detection.corners).reshape(4, 2))
                marker_ids.append(detection.tag_id)

        # If at least one marker detected
        if len(marker_ids) > 0:
            self.calibration_images.append(image.copy())
            # Flatten object points and image points for this image
            obj_pts = np.concatenate(object_points, axis=0).astype(np.float32)
            img_pts = np.concatenate(marker_corners, axis=0).astype(np.float32)
            self.all_object_points.append(obj_pts)
            self.all_marker_corners.append(img_pts)
            self.all_marker_ids.append(np.array(marker_ids))

            # Draw detected markers
            annotated = image.copy()
            # Convert marker_ids to numpy array for cv2.aruco.drawDetectedMarkers
            marker_ids_array = np.array(marker_ids).reshape(-1, 1)
            # Reshape marker_corners for drawDetectedMarkers (needs shape (1, 4, 2) for each marker)
            marker_corners_for_draw = [c.reshape(1, 4, 2) for c in marker_corners]
            cv2.aruco.drawDetectedMarkers(annotated, marker_corners_for_draw, marker_ids_array)

            logger.info(f"AprilTag board detected with {len(marker_ids)} markers")
            return True, annotated

        logger.warning("Failed to detect AprilTag board")
        return False, image

    def clear_images(self):
        """Clear all collected calibration images"""
        self.calibration_images.clear()
        self.all_object_points.clear()
        self.all_marker_corners.clear()
        self.all_marker_ids.clear()
        self.image_size = None
        logger.info("Cleared all calibration images")

    def get_image_count(self) -> int:
        """Get number of successfully captured calibration images"""
        return len(self.calibration_images)

    def calibrate(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], str]:
        """
        Run camera calibration.

        Returns:
            Tuple of (success, K, dist_coef, message)
            - success: True if calibration succeeded
            - K: 3x3 camera intrinsic matrix
            - dist_coef: 1x5 distortion coefficients [k1, k2, p1, p2, k3]
            - message: Status message
        """
        return self._calibrate_apriltag()

    def _calibrate_apriltag(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], str]:
        """Run AprilTag calibration"""
        if len(self.all_marker_corners) < 3:
            msg = f"Need at least 3 calibration images, got {len(self.all_marker_corners)}"
            logger.error(msg)
            return False, None, None, msg

        if self.image_size is None:
            msg = "Image size not set"
            logger.error(msg)
            return False, None, None, msg

        try:
            logger.info(f"Running AprilTag calibration with {len(self.all_marker_corners)} images...")
            # Perform calibration
            flags = 0
            flags |= cv2.CALIB_RATIONAL_MODEL  # Use rational distortion model
            
            rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self.all_object_points,
                self.all_marker_corners,
                self.image_size,
                None,
                None,
                flags=flags,
            )
            dist_coeffs = dist_coeffs.flatten()[:5]  # Keep only k1, k2, p1, p2, k3

            if camera_matrix is None or dist_coeffs is None:
                msg = "Calibration failed - no camera matrix returned"
                logger.error(msg)
                return False, None, None, msg

            # RMS error is already computed by calibrateCamera
            msg = f"AprilTag calibration successful! RMS reprojection error: {rms_error:.4f} pixels"
            logger.info(msg)
            logger.info(f"Number of images used: {len(self.all_marker_corners)}")
            logger.info(f"Camera matrix:\n{camera_matrix}")
            logger.info(f"Distortion coefficients: {dist_coeffs.ravel()}")

            return True, camera_matrix, dist_coeffs, msg

        except Exception as e:
            msg = f"Calibration error: {str(e)}"
            logger.error(msg, exc_info=True)
            return False, None, None, msg

    def save_images(self, output_dir: Path) -> int:
        """
        Save all calibration images to disk.

        Args:
            output_dir: Directory to save images

        Returns:
            Number of images saved
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(self.calibration_images):
            filename = output_dir / f"calib_{i:03d}.png"
            cv2.imwrite(str(filename), img)

        logger.info(f"Saved {len(self.calibration_images)} images to {output_dir}")
        return len(self.calibration_images)

    def load_images(self, image_dir: Path) -> Tuple[int, int]:
        """
        Load calibration images from directory.

        Args:
            image_dir: Directory containing calibration images

        Returns:
            Tuple of (total_loaded, successfully_detected)
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            logger.error(f"Directory not found: {image_dir}")
            return 0, 0

        # Find all image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            image_files.extend(sorted(image_dir.glob(ext)))

        total_loaded = 0
        successfully_detected = 0

        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    total_loaded += 1
                    success, _ = self.add_image(img)
                    if success:
                        successfully_detected += 1
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")

        logger.info(f"Loaded {total_loaded} images, detected board in {successfully_detected}")
        return total_loaded, successfully_detected


def generate_apriltag_board(
    board_config: AprilTagBoardConfig,
    output_path: Path,
    dpi: int = 300
) -> bool:
    """
    Generate and save an AprilTag board image for printing.

    Args:
        board_config: AprilTag board configuration
        output_path: Output file path
        dpi: DPI for the output image

    Returns:
        True if successful
    """
    try:
        # Calculate board size in mm
        board_size_mm = (
            board_config.tags_x * (board_config.tag_size + board_config.tag_spacing) * 1000,
            board_config.tags_y * (board_config.tag_size + board_config.tag_spacing) * 1000
        )

        # Calculate pixels needed
        pixels_per_mm = dpi / 25.4
        img_size = (
            int(board_size_mm[0] * pixels_per_mm),
            int(board_size_mm[1] * pixels_per_mm)
        )

        # Generate board image
        board_img = board_config.board.generateImage(img_size)

        # Save image
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), board_img)

        logger.info(f"Generated AprilTag board: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to generate board: {e}")
        return False
