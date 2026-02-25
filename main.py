"""RGBTrack Inference Service - Main Entry Point"""

import logging
import signal
import sys
import time
import threading
from pathlib import Path
from enum import Enum, auto
from typing import Optional

import numpy as np

from src.config import SystemConfig
from src.camera import CameraBase, create_camera
from src.detection import DetectionAlgorithm, DetectionResult
from src.zmq_publisher import ZMQPublisher
from src.viser_ui import ViserWebUI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rgbtrack_inference.log')
    ]
)

logger = logging.getLogger(__name__)


class TrackingState(Enum):
    """Tracking state machine states"""
    IDLE = auto()       # Waiting for user to start detection
    DETECTING = auto()  # Running first-frame detection
    TRACKING = auto()   # Actively tracking
    PAUSED = auto()     # User paused tracking


class RGBTrackInferenceService:
    """
    Main RGBTrack inference service.
    
    Architecture:
    - Main thread: Runs viser WebUI (blocking)
    - Detection thread: Background frame processing loop
    - Callbacks: UI buttons trigger state changes via thread-safe methods
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the inference service.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.state = TrackingState.IDLE
        self._state_lock = threading.Lock()
        
        # Components
        self.camera: Optional[CameraBase] = None
        self.detection: Optional[DetectionAlgorithm] = None
        self.zmq_pub: Optional[ZMQPublisher] = None
        self.viser_ui: Optional[ViserWebUI] = None
        
        # State variables
        self._running = False
        self._detection_thread: Optional[threading.Thread] = None
        self._current_pose: Optional[np.ndarray] = None
        self._current_mask: Optional[np.ndarray] = None
        self._frame_count = 0
        self._start_time = 0.0
        self._fps = 0.0
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Initializing RGBTrack inference service...")
            
            # 1. Initialize camera
            logger.info("Initializing camera...")
            self.camera = create_camera(
                self.config.camera,
                use_dummy=False  # Use real camera for inference
            )
            if not self.camera.open():
                logger.error("Failed to open camera")
                return False
            logger.info("✓ Camera initialized")
            
            # 2. Initialize detection algorithm
            logger.info("Initializing detection algorithm...")
            self.detection = DetectionAlgorithm(
                detection_config=self.config.detection,
                calibration_config=self.config.calibration
            )
            if not self.detection.initialize():
                logger.error("Failed to initialize detection algorithm")
                return False
            logger.info("✓ Detection algorithm initialized")
            
            # 3. Initialize ZMQ publisher
            logger.info("Initializing ZMQ publisher...")
            self.zmq_pub = ZMQPublisher(self.config.zmq)
            self.zmq_pub.start()
            logger.info(f"✓ ZMQ publisher started on {self.config.zmq.address}")
            
            # 4. Initialize viser UI
            logger.info("Initializing viser UI...")
            self.viser_ui = ViserWebUI(
                config=self.config,
                detection=self.detection,
                on_start_detection=self._handle_start_detection,
                on_pause=self._handle_pause,
                on_resume=self._handle_resume,
                on_reset=self._handle_reset
            )
            logger.info("✓ Viser UI initialized")
            
            logger.info("=" * 60)
            logger.info("✓ All components initialized successfully")
            logger.info("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            return False
    
    def start(self):
        """Start the inference service"""
        if not self.initialize():
            logger.error("Failed to initialize service")
            return
        
        self._running = True
        self._start_time = time.time()
        
        # Start detection thread
        self._detection_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True,
            name="detection_thread"
        )
        self._detection_thread.start()
        
        logger.info("✓ Detection thread started")
        logger.info("=" * 60)
        logger.info("✓ Starting viser UI (blocking)...")
        logger.info(f"  Viser UI: http://localhost:{self.config.viser.port}")
        logger.info(f"  Gradio UI (calibration): http://localhost:{self.config.ui_port}")
        logger.info("=" * 60)
        
        # Launch viser UI (blocks until closed)
        try:
            self.viser_ui.launch()
        except Exception as e:
            logger.error(f"Viser UI error: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """Stop the service and cleanup resources"""
        if not self._running:
            return
        
        logger.info("Stopping RGBTrack inference service...")
        self._running = False
        
        # Stop detection thread
        if self._detection_thread is not None:
            self._detection_thread.join(timeout=2.0)
        
        # Cleanup components
        if self.camera is not None:
            logger.info("Closing camera...")
            self.camera.close()
        
        if self.zmq_pub is not None:
            logger.info("Stopping ZMQ publisher...")
            self.zmq_pub.stop()
        
        logger.info("=" * 60)
        logger.info("✓ Service stopped")
        logger.info("=" * 60)
    
    def _detection_loop(self):
        """
        Main detection/tracking loop (runs in background thread).
        
        State transitions:
        IDLE → DETECTING: User triggers start_detection
        DETECTING → TRACKING: Detection successful
        TRACKING → PAUSED: User pauses
        PAUSED → TRACKING: User resumes
        TRACKING/PAUSED → IDLE: User resets
        """
        frame_id = 0
        
        while self._running:
            with self._state_lock:
                state = self.state
            
            if state == TrackingState.IDLE:
                # Wait in idle state
                time.sleep(0.1)
                continue
            
            elif state == TrackingState.DETECTING:
                # Capture single frame for detection
                frame = self.camera.capture_frame()
                if frame is None:
                    logger.error("Failed to capture frame for detection")
                    with self._state_lock:
                        self.state = TrackingState.IDLE
                    continue
                
                try:
                    logger.info("Running first-frame detection...")
                    t0 = time.time()
                    
                    mask = self.detection.detect_first_frame(frame)
                    
                    if mask is not None:
                        pose = self.detection.current_pose
                        self._current_pose = pose
                        self._current_mask = mask
                        
                        # Publish result
                        result = DetectionResult(
                            timestamp=time.time(),
                            frame_id=frame_id,
                            pose=pose,
                            processing_time_ms=(time.time() - t0) * 1000,
                            frame_shape=frame.shape,
                            camera_intrinsics=np.array(self.config.calibration.K),
                            mask_png=None  # Can add mask encoding if needed
                        )
                        self.zmq_pub.publish(result)
                        
                        logger.info(f"✓ Detection successful, transitioning to TRACKING")
                        with self._state_lock:
                            self.state = TrackingState.TRACKING
                    else:
                        logger.error("Detection failed, no mask extracted")
                        with self._state_lock:
                            self.state = TrackingState.IDLE
                    
                except Exception as e:
                    logger.error(f"Detection error: {e}", exc_info=True)
                    with self._state_lock:
                        self.state = TrackingState.IDLE
            
            elif state == TrackingState.TRACKING:
                # Continuous tracking loop
                t0 = time.time()
                
                frame = self.camera.capture_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                try:
                    # Track object in frame
                    pose = self.detection.track(frame)
                    
                    if pose is not None:
                        self._current_pose = pose
                        self._current_mask = self.detection.current_mask
                        
                        # Update frame count and FPS
                        self._frame_count += 1
                        elapsed = time.time() - self._start_time
                        if elapsed > 0:
                            self._fps = self._frame_count / elapsed
                        
                        # Publish result
                        result = DetectionResult(
                            timestamp=time.time(),
                            frame_id=frame_id,
                            pose=pose,
                            processing_time_ms=(time.time() - t0) * 1000,
                            frame_shape=frame.shape,
                            camera_intrinsics=np.array(self.config.calibration.K),
                            mask_png=None
                        )
                        self.zmq_pub.publish(result)
                        
                        # Update viser UI
                        if self.viser_ui is not None:
                            self.viser_ui.update_visualization(
                                pose=pose,
                                mask=self._current_mask,
                                frame_shape=frame.shape,
                                fps=self._fps,
                                status="TRACKING"
                            )
                    
                    frame_id += 1
                    
                except Exception as e:
                    logger.error(f"Tracking error: {e}", exc_info=True)
            
            elif state == TrackingState.PAUSED:
                # Paused state - just update UI with last known pose
                if self.viser_ui is not None and self._current_pose is not None:
                    current_frame = self.camera.capture_frame()
                    frame_shape = current_frame.shape if current_frame is not None else (480, 640, 3)
                    
                    self.viser_ui.update_visualization(
                        pose=self._current_pose,
                        mask=self._current_mask,
                        frame_shape=frame_shape,
                        fps=self._fps,
                        status="PAUSED"
                    )
                time.sleep(0.1)
    
    # User control callbacks (called from UI thread)
    def _handle_start_detection(self) -> bool:
        """User clicked 'Start Detection' button"""
        with self._state_lock:
            if self.state == TrackingState.IDLE:
                self.state = TrackingState.DETECTING
                logger.info("→ State: DETECTING")
                return True
            else:
                logger.warning(f"Cannot start detection in state {self.state}")
                return False
    
    def _handle_pause(self):
        """User clicked 'Pause' button"""
        with self._state_lock:
            if self.state == TrackingState.TRACKING:
                self.state = TrackingState.PAUSED
                logger.info("→ State: PAUSED")
    
    def _handle_resume(self):
        """User clicked 'Resume' button"""
        with self._state_lock:
            if self.state == TrackingState.PAUSED:
                self.state = TrackingState.TRACKING
                logger.info("→ State: TRACKING")
    
    def _handle_reset(self):
        """User clicked 'Reset' button"""
        with self._state_lock:
            logger.info("→ State: IDLE (reset)")
            self.state = TrackingState.IDLE
            self._current_pose = None
            self._current_mask = None
            if self.detection is not None:
                self.detection.reset()
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {sig}, initiating shutdown...")
        self.stop()
        sys.exit(0)


def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("RGBTrack Inference Service")
    logger.info("=" * 60)
    
    # Load configuration
    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Please run src/app.py first to generate configuration")
        sys.exit(1)
    
    config = SystemConfig.from_yaml(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    
    # Create and start service
    service = RGBTrackInferenceService(config)
    
    try:
        service.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
