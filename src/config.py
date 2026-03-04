"""Configuration management for the RGBTrack system"""

import yaml
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field


class CameraConfig(BaseModel):
    """Camera configuration parameters"""
    device_sn: str = ""  # Camera serial number

    # Video file support (for testing with recorded videos)
    file: str = Field(
        default="", description="Video file path (format: file:///path/to/file.ext), disables hardware controls when set")

    # Resolution
    width: int = 1280
    height: int = 720
    fps: int = 30

    exposure_time_ms: int = Field(
        default=30, description="Exposure time in milliseconds")
    exposure_min: int = 1
    exposure_max: int = 100

    gain: float = 1.0
    gain_min: float = 0.0
    gain_max: float = 10.0

    # Gamma
    gamma: float = 1.0
    gamma_min: float = 0.1
    gamma_max: float = 5.0

    # White balance
    wb_mode: Literal["auto", "manual"] = "auto"

    # RGB balance (for manual white balance)
    red_balance: float = 1.0
    green_balance: float = 1.0
    blue_balance: float = 1.0
    balance_min: float = 0
    balance_max: float = 400

    # Mode
    mode: Literal["high_speed", "normal"] = "normal"

    # Undistortion
    undistort: bool = False


class CalibrationConfig(BaseModel):
    """Camera calibration configuration"""
    # Camera intrinsic matrix (3x3)
    K: list[list[float]] = Field(
        default=[
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0]
        ],
        description="Camera intrinsic matrix (3x3)"
    )
    extrinsic_matrix: list[list[float]] = Field(
        default=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        description="Camera extrinsic matrix (4x4 transformation from camera to world frame)"
    )

    # Distortion coefficients (1x5: k1, k2, p1, p2, k3)
    dist_coef: list[float] = Field(
        default=[0.0, 0.0, 0.0, 0.0, 0.0],
        description="Distortion coefficients [k1, k2, p1, p2, k3]"
    )

    # AprilTag board configuration
    apriltag_family: str = Field(
        default="t36h11", description="AprilTag family")
    apriltag_tags_x: int = Field(
        default=6, description="Number of tags in X direction")
    apriltag_tags_y: int = Field(
        default=6, description="Number of tags in Y direction")
    apriltag_tag_size: float = Field(
        default=0.05, description="Tag size in meters")
    apriltag_tag_spacing: float = Field(
        default=0.01, description="Tag spacing in meters")
    apriltag_first_marker: int = Field(
        default=0, description="First marker ID")

    # Calibration image storage
    calibration_images_dir: str = Field(
        default="calibration_images", description="Directory to store calibration images")


class DetectionConfig(BaseModel):
    """Detection algorithm configuration"""
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    mesh_path: str = "models/detector.obj"
    prompt: str = ""
    sam2_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_checkpoint: str = "checkpoints/sam2.1_hiera_tiny.pt"
    track_refine_iter: int = 5

    # Max shorter-side resolution for FoundationPose first-frame detection
    # (register / binary_search_depth processes ~252 pose hypotheses).
    # Set to 0 to disable downscaling.
    detect_max_shorter_side: int = Field(
        default=480,
        description="Downscale frames so shorter side <= this during detection")

    # Max shorter-side resolution for FoundationPose per-frame tracking
    # (track_one processes only 1 pose, so higher res is usually fine).
    # Set to 0 to disable downscaling (use full camera resolution).
    track_max_shorter_side: int = Field(
        default=0,
        description="Downscale frames so shorter side <= this during tracking (0 = full res)")

    # Kalman filter settings for pose smoothing
    use_kalman_filter: bool = Field(
        default=False, description="Enable Kalman filter for pose smoothing")
    kalman_process_noise: float = Field(
        default=0.01, description="Kalman filter process noise")
    kalman_measurement_noise: float = Field(
        default=0.05, description="Kalman filter measurement noise")

    mesh_scale: float = Field(
        default=1.0, description="Scale factor for the loaded mesh model used in detection (for testing / debugging purposes)"
    )


class ZMQConfig(BaseModel):
    """ZeroMQ server configuration"""
    # Transport type: 'tcp' or 'ipc' (Unix socket)
    transport: Literal["tcp", "ipc"] = "ipc"

    # TCP settings (used when transport='tcp')
    host: str = "localhost"
    port: int = 5555
    control_port: int = 5556

    # Unix socket settings (used when transport='ipc')
    socket_path: str = "/tmp/rgbtrack.sock"
    control_socket_path: str = "/tmp/rgbtrack_control.sock"

    # 0 means publish as fast as new results arrive
    # >0 means fixed interval publish (repeat latest payload when needed)
    publish_interval_ms: int = 0

    @property
    def address(self) -> str:
        """Get the ZMQ address for detection results based on transport type"""
        if self.transport == "ipc":
            return f"ipc://{self.socket_path}"
        else:
            return f"tcp://{self.host}:{self.port}"

    @property
    def results_address(self) -> str:
        """Get the result PUB address."""
        return self.address

    @property
    def control_address(self) -> str:
        """Get the control REP address."""
        if self.transport == "ipc":
            return f"ipc://{self.control_socket_path}"
        return f"tcp://{self.host}:{self.control_port}"

    @property
    def publish_interval_sec(self) -> float:
        """Get publish interval in seconds."""
        return max(0, self.publish_interval_ms) / 1000.0


class ViserConfig(BaseModel):
    """Viser 3D visualization configuration"""
    host: str = "0.0.0.0"
    port: int = 8080
    show_camera_cone: bool = True
    show_trajectory: bool = True
    trajectory_length: int = 100
    camera_cone_scale: float = 0.1


class CalibrationUIConfig(BaseModel):
    """Calibration UI configuration"""
    host: str = ""
    port: int = 7860


class SystemConfig(BaseModel):
    """Overall system configuration with YAML persistence support"""
    camera: CameraConfig = Field(default_factory=CameraConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    zmq: ZMQConfig = Field(default_factory=ZMQConfig)
    viser: ViserConfig = Field(default_factory=ViserConfig)
    calibration_ui: CalibrationUIConfig = Field(
        default_factory=CalibrationUIConfig
    )

    @classmethod
    def from_yaml(cls, path: Path) -> "SystemConfig":
        """Load configuration from YAML file"""
        if not path.exists():
            # Return default config if file doesn't exist
            return cls()

        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            return cls(**data) if data else cls()
        except Exception as e:
            print(f"Warning: Failed to load config from {path}: {e}")
            return cls()

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file"""
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            # Add a header comment
            f.write("# RGBTrack System Configuration\n")
            f.write("# This file is automatically managed by the application\n\n")
            yaml.dump(self.model_dump(), f, default_flow_style=False,
                      sort_keys=False, indent=2)
