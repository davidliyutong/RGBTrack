"""Configuration management for the RGBTrack system"""

import yaml
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field


class CameraConfig(BaseModel):
    """Camera configuration parameters"""
    device_sn: str = ""  # Camera serial number
    exposure_time_ms: int = Field(default=30, description="Exposure time in milliseconds")
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

    # Resolution
    width: int = 1280
    height: int = 720

    # Distortion parameters (k1, k2, p1, p2, k3)
    distortion_k1: float = 0.0
    distortion_k2: float = 0.0
    distortion_p1: float = 0.0
    distortion_p2: float = 0.0
    distortion_k3: float = 0.0
    distortion_min: float = -1.0
    distortion_max: float = 1.0

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
    
    # Distortion coefficients (1x5: k1, k2, p1, p2, k3)
    dist_coef: list[float] = Field(
        default=[0.0, 0.0, 0.0, 0.0, 0.0],
        description="Distortion coefficients [k1, k2, p1, p2, k3]"
    )
    
    # AprilTag board configuration
    apriltag_family: str = Field(default="t36h11", description="AprilTag family")
    apriltag_tags_x: int = Field(default=6, description="Number of tags in X direction")
    apriltag_tags_y: int = Field(default=6, description="Number of tags in Y direction")
    apriltag_tag_size: float = Field(default=0.05, description="Tag size in meters")
    apriltag_tag_spacing: float = Field(default=0.01, description="Tag spacing in meters")
    apriltag_first_marker: int = Field(default=0, description="First marker ID")
    
    # Calibration image storage
    calibration_images_dir: str = Field(default="calibration_images", description="Directory to store calibration images")


class DetectionConfig(BaseModel):
    """Detection algorithm configuration"""
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    mesh_path: str = "models/detector.obj"
    prompt: str = ""
    sam2_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_checkpoint: str = "checkpoints/sam2.1_hiera_tiny.pt"
    mode: int = 0
    track_refine_iter: int = 5
    K: list[list[float]] = Field(
        default=[[600.0, 0.0, 320.0],
                 [0.0, 600.0, 240.0],
                 [0.0, 0.0, 1.0]],
        description="Camera intrinsic matrix"
    )


class ZMQConfig(BaseModel):
    """ZeroMQ server configuration"""
    # Transport type: 'tcp' or 'ipc' (Unix socket)
    transport: Literal["tcp", "ipc"] = "ipc"
    
    # TCP settings (used when transport='tcp')
    host: str = "localhost"
    port: int = 5555
    
    # Unix socket settings (used when transport='ipc')
    socket_path: str = "/tmp/rgbtrack.sock"

    @property
    def address(self) -> str:
        """Get the ZMQ address based on transport type"""
        if self.transport == "ipc":
            return f"ipc://{self.socket_path}"
        else:
            return f"tcp://{self.host}:{self.port}"


class SystemConfig(BaseModel):
    """Overall system configuration with YAML persistence support"""
    camera: CameraConfig = Field(default_factory=CameraConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    zmq: ZMQConfig = Field(default_factory=ZMQConfig)

    # UI settings
    ui_host: str = "0.0.0.0"
    ui_port: int = 7860

    # Thread settings
    max_fps: int = 30
    frame_buffer_size: int = 10

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SystemConfig":
        """Load configuration from YAML file"""
        path = Path(path)
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

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            # Add a header comment
            f.write("# RGBTrack System Configuration\n")
            f.write("# This file is automatically managed by the application\n\n")
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False, indent=2)
