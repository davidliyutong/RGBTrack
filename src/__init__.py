"""RGBTrack Multi-threaded Framework"""

__version__ = "0.1.0"

# Export main classes for convenient imports
from .config import SystemConfig, CameraConfig, DetectionConfig, ZMQConfig
from .camera import CameraBase, DummyCamera, create_camera
from .detection import DetectionAlgorithm, DetectionResult
from .zmq_publisher import ZMQPublisher
from .zmq_subscriber import ZMQSubscriber, EnhancedSubscriber, create_subscriber_from_config
from .webui import WebUI

__all__ = [
    # Configuration
    'SystemConfig',
    'CameraConfig', 
    'DetectionConfig',
    'ZMQConfig',
    # Camera
    'CameraBase',
    'DummyCamera',
    'create_camera',
    # Detection
    'DetectionAlgorithm',
    'DetectionResult',
    # ZMQ
    'ZMQPublisher',
    'ZMQSubscriber',
    'EnhancedSubscriber',
    'create_subscriber_from_config',
    # UI
    'WebUI',
]
