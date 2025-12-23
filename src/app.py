"""Main application with multi-threaded framework"""

import logging
import signal
import sys
import threading
import time
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

import numpy as np
import cv2

from .camera import CameraBase, MindVisionCamera, create_camera
from .config import SystemConfig
from .detection import DetectionAlgorithm, DetectionResult
from .webui import WebUI
from .zmq_publisher import ZMQPublisher

# Default configuration file path
DEFAULT_CONFIG_FILE = Path("config.yaml")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rgbtrack.log')
    ]
)

logger = logging.getLogger(__name__)


class RGBTrackApplication:
    """
    Main application with multi-threaded framework.

    Architecture:
    - Main thread: Coordinates all components
    - UI thread: Runs Gradio web interface
    - Detection thread: Processes frames and runs detection algorithm
    - ZMQ thread: Publishes detection results (runs within ZMQPublisher)
    """

    def __init__(self, config: Optional[SystemConfig] = None, config_file: Optional[Path] = None, use_dummy_camera: bool = True):
        """
        Initialize the application.

        Args:
            config: System configuration (uses defaults if None)
            config_file: Path to configuration file (loads from file if provided)
            use_dummy_camera: Whether to use dummy camera for testing
        """
        self.config_file = config_file or DEFAULT_CONFIG_FILE

        # Load config from file if it exists, otherwise use provided or default
        if config is None and self.config_file.exists():
            logger.info(f"Loading configuration from {self.config_file}")
            self.config = SystemConfig.from_yaml(self.config_file)
        else:
            self.config = config or SystemConfig()
            # Save default config if file doesn't exist
            if not self.config_file.exists():
                logger.info(f"Creating default configuration at {self.config_file}")
                self.config.to_yaml(self.config_file)

        self.use_dummy_camera = use_dummy_camera

        # Components
        self.camera: Optional[CameraBase] = None
        self.detector: Optional[DetectionAlgorithm] = None
        self.zmq_publisher: Optional[ZMQPublisher] = None
        self.webui: WebUI | None = None 

        # Threading
        self._running = False
        self._live_mode = False
        self._detection_thread: Optional[threading.Thread] = None
        self._frame_queue: Queue[np.ndarray] = Queue(maxsize=self.config.frame_buffer_size)

        # Statistics
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._fps = 0.0

        # Shutdown handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {sig}, shutting down...")
        self.stop()
        sys.exit(0)

    def initialize(self) -> bool:
        """Initialize all components"""
        try:
            logger.info("Initializing RGBTrack application...")

            # 1. Initialize camera
            logger.info("Initializing camera...")
            self.camera = create_camera(self.config.camera, use_dummy=self.use_dummy_camera)
            # if not self.camera.open():
            #     logger.error("Failed to open camera")
            #     return False

            # 2. Initialize detection algorithm
            logger.info("Initializing detection algorithm...")
            self.detector = DetectionAlgorithm(self.config.detection)

            # 3. Initialize ZMQ publisher
            logger.info("Initializing ZMQ publisher...")
            self.zmq_publisher = ZMQPublisher(self.config.zmq)
            self.zmq_publisher.start()

            # 4. Initialize Web UI
            logger.info("Initializing Web UI...")
            self.webui = WebUI(
                camera_config=self.config.camera,
                camera_cls=type(self.camera),
                on_preview=self._handle_preview,
                on_live_toggle=self._handle_live_toggle,
                on_config_update=self._handle_config_update,
                on_camera_reset=self._handle_camera_reset,
                on_save_config=self._handle_save_config,
                on_wb_calibrate=self._handle_wb_calibrate,
                config_path=self.config_file,
                host=self.config.ui_host,
                port=self.config.ui_port
            )

            logger.info("✓ All components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            return False

    def start(self):
        """Start the application"""
        if not self.initialize():
            logger.error("Failed to initialize application")
            return

        self._running = True
        logger.info("Starting RGBTrack application...")

        try:
            # Start detection thread
            self._detection_thread = threading.Thread(
                target=self._detection_loop,
                daemon=True,
                name="DetectionThread"
            )
            self._detection_thread.start()
            logger.info("✓ Detection thread started")

            # Start Web UI (this will block in the main thread)
            logger.info("✓ Starting Web UI...")
            self.webui.launch(share=False) # pyright: ignore[reportOptionalMemberAccess]

            # Keep main thread alive
            logger.info("✓ Application running. Press Ctrl+C to stop.")
            while self._running:
                time.sleep(1)
                self._update_statistics()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Application error: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self):
        """Stop the application"""
        if not self._running:
            return

        logger.info("Stopping RGBTrack application...")
        self._running = False
        self._live_mode = False

        # Stop detection thread
        if self._detection_thread is not None:
            logger.info("Stopping detection thread...")
            self._detection_thread.join(timeout=2.0)

        # Stop ZMQ publisher
        if self.zmq_publisher is not None:
            logger.info("Stopping ZMQ publisher...")
            self.zmq_publisher.stop()

        # Close camera
        if self.camera is not None:
            logger.info("Closing camera...")
            self.camera.close()

        # Close Web UI
        if self.webui is not None:
            logger.info("Closing Web UI...")
            self.webui.close()

        logger.info("✓ Application stopped")

    def _detection_loop(self):
        """Main detection loop running in separate thread"""
        logger.info("Detection thread started")

        while self._running:
            try:
                if not self._live_mode:
                    # Not in live mode, just sleep
                    time.sleep(0.1)
                    continue

                # Capture frame from camera
                frame = self.camera.capture_frame() # pyright: ignore[reportOptionalMemberAccess]

                if frame is None:
                    time.sleep(0.01)
                    continue

                # Run detection
                result = self.detector.detect(frame) # pyright: ignore[reportOptionalMemberAccess]

                # Publish result via ZMQ
                if self.zmq_publisher is not None:
                    self.zmq_publisher.publish(result)

                # Update statistics
                self._frame_count += 1

                # Rate limiting
                time.sleep(1.0 / self.config.max_fps)

            except Exception as e:
                logger.error(f"Error in detection loop: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("Detection thread stopped")

    def _update_statistics(self):
        """Update FPS and other statistics"""
        current_time = time.time()
        elapsed = current_time - self._last_fps_time

        if elapsed >= 1.0:  # Update every second
            self._fps = self._frame_count / elapsed
            logger.debug(f"FPS: {self._fps:.2f}")

            # Reset counters
            self._frame_count = 0
            self._last_fps_time = current_time

    def _handle_preview(self, detection_enabled: bool = True) -> Optional[np.ndarray]:
        """Handle preview request from UI
        
        Args:
            detection_enabled: Whether to run detection and draw results on the frame
        """
        try:
            if self.camera is not None and not self.camera.is_open():  # pyright: ignore[reportOptionalMemberAccess]
                self.camera.open()

            if self.camera is None or not self.camera.is_open():
                logger.warning("Camera not available for preview")
                return None

            frame = self.camera.capture_frame()

            if frame is not None and detection_enabled and self.detector is not None:
                # Run detection on preview
                result = self.detector.detect(frame)
                # Draw detections on frame
                frame = self.detector.draw_detections(frame, result)

            return frame

        except Exception as e:
            logger.error(f"Preview error: {e}")
            return None

    def _handle_live_toggle(self, is_live: bool):
        """Handle live mode toggle from UI"""
        try:
            self._live_mode = is_live

            if is_live:
                logger.info("Live mode ENABLED")
            else:
                logger.info("Live mode DISABLED")

        except Exception as e:
            logger.error(f"Live toggle error: {e}")

    def _handle_config_update(self, config):
        """Handle configuration update from UI"""
        try:
            logger.info("Updating camera configuration...")

            if self.camera is not None:
                self.camera.set_exposure(config.exposure_time_ms)
                self.camera.set_gain(config.gain)
                self.camera.set_gamma(config.gamma)
                self.camera.set_wb_mode(config.wb_mode)
                self.camera.set_rgb_balance(
                    config.red_balance,
                    config.green_balance,
                    config.blue_balance
                )
                self.camera.set_mode(config.mode)

            # Update system config
            self.config.camera = config

            logger.info("✓ Configuration updated")

        except Exception as e:
            logger.error(f"Config update error: {e}")

    def _handle_save_config(self):
        """Save current configuration to file"""
        try:
            logger.info(f"Saving configuration to {self.config_file}...")
            self.config.to_yaml(self.config_file)
            logger.info("✓ Configuration saved")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    def _handle_camera_reset(self) -> str:
        """Handle camera reset request from UI"""
        try:
            logger.info("Resetting camera...")
            
            # Stop live mode if active
            was_live = self._live_mode
            if was_live:
                self._live_mode = False
                time.sleep(0.2)  # Give detection thread time to stop
            
            if self.camera is not None:
                # Close the camera
                logger.info("Closing camera...")
                self.camera.close()
                time.sleep(0.5)  # Wait for camera to fully close
                
                # Reopen the camera
                logger.info("Reopening camera...")
                if self.camera.open():
                    logger.info("✓ Camera reset successful")
                    
                    # Restore live mode if it was active
                    if was_live:
                        self._live_mode = True
                    
                    return "✓ Camera reset successful"
                else:
                    logger.error("Failed to reopen camera")
                    return "✗ Failed to reopen camera"
            else:
                return "✗ No camera available"
                
        except Exception as e:
            logger.error(f"Camera reset error: {e}")
            return f"✗ Error: {str(e)}"

    def _handle_wb_calibrate(self) -> tuple[str, float, float, float]:
        """Handle white balance calibration request from UI
        
        Returns:
            Tuple of (status_message, red_balance, green_balance, blue_balance)
        """
        try:
            logger.info("Starting white balance calibration...")

            if self.camera is None or not self.camera.is_open():
                logger.error("Camera not available for white balance calibration")
                # Return current config values instead of defaults on error
                return (
                    "✗ Camera not available",
                    self.config.camera.red_balance,
                    self.config.camera.green_balance,
                    self.config.camera.blue_balance
                )

            # Perform calibration (updates camera.config RGB balance values)
            success = self.camera.calibrate_white_balance()  # pyright: ignore[reportOptionalMemberAccess]
            
            # Check if calibration was successful
            if not success or success is None:
                logger.error("White balance calibration failed")
                # Return current config values instead of defaults on error
                return (
                    "✗ Calibration failed",
                    self.config.camera.red_balance,
                    self.config.camera.green_balance,
                    self.config.camera.blue_balance
                )

            # Get the updated RGB balance values from camera config
            # Defensive check to ensure camera.config exists
            if not hasattr(self.camera, 'config') or self.camera.config is None:
                logger.error("Camera config not available")
                return (
                    "✗ Camera config not available",
                    self.config.camera.red_balance,
                    self.config.camera.green_balance,
                    self.config.camera.blue_balance
                )
            
            red = self.camera.config.red_balance
            green = self.camera.config.green_balance
            blue = self.camera.config.blue_balance
            
            # Update system config to match camera config
            self.config.camera.red_balance = red
            self.config.camera.green_balance = green
            self.config.camera.blue_balance = blue
            
            # Save configuration to disk
            self._handle_save_config()
            
            logger.info(f"✓ White balance calibrated: R={red:.2f}, G={green:.2f}, B={blue:.2f}")
            return (f"✓ White balance calibrated: R={red:.2f}, G={green:.2f}, B={blue:.2f}", red, green, blue)

        except Exception as e:
            logger.error(f"White balance calibration error: {e}")
            # Return current config values instead of defaults on error
            return (
                f"✗ Error: {str(e)}",
                self.config.camera.red_balance,
                self.config.camera.green_balance,
                self.config.camera.blue_balance
            )

def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("RGBTrack Multi-threaded Framework")
    logger.info("=" * 60)

    # Create and start application
    app = RGBTrackApplication(use_dummy_camera=False)

    try:
        app.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
