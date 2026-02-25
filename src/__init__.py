"""RGBTrack Multi-threaded Framework"""

__version__ = "0.1.0"

# Export main classes for convenient imports
from .config import SystemConfig, CameraConfig, DetectionConfig, ZMQConfig, CalibrationConfig, ViserConfig
from .camera import CameraBase, DummyCamera, create_camera
from .calibration_webui import CalibrationWebUI
from .calibration import (
    CameraCalibrator,
    AprilTagBoardConfig,
    generate_apriltag_board
)
from .detection import DetectionAlgorithm, DetectionResult

__all__ = [
    # Configuration
    'SystemConfig',
    'CameraConfig', 
    'DetectionConfig',
    'ZMQConfig',
    'CalibrationConfig',
    'ViserConfig',
    # Camera
    'CameraBase',
    'DummyCamera',
    'create_camera',
    # Detection
    'DetectionAlgorithm',
    'DetectionResult',
    # UI
    'CalibrationWebUI',
    # Calibration
    'CameraCalibrator',
    'AprilTagBoardConfig',
    'generate_apriltag_board',
]
