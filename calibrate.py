"""Main application - Camera adjustment and calibration tool"""

import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

from src.camera import CameraBase, create_camera
from src.config import SystemConfig
from src.webui import WebUI

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
    Camera adjustment and calibration application.

    Architecture:
    - Main thread: Coordinates all components and runs Gradio web interface
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
        self.webui: WebUI | None = None

        # State
        self._running = False
        self._frozen = False

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

            # 2. Initialize Web UI
            logger.info("Initializing Web UI...")
            self.webui = WebUI(
                config=self.config,
                calibration_config=self.config.calibration,
                camera_cls=type(self.camera),
                on_preview=self._handle_preview,
                on_freeze_toggle=self._handle_freeze_toggle,
                on_config_update=self._handle_config_update,
                on_calibration_config_update=self._handle_calibration_config_update,
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
            # Start Web UI (this will block in the main thread)
            logger.info("✓ Starting Web UI...")
            self.webui.launch(share=False)  # pyright: ignore[reportOptionalMemberAccess]

            # Keep main thread alive
            logger.info("✓ Application running. Press Ctrl+C to stop.")
            while self._running:
                time.sleep(1)

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
        self._frozen = True

        # Close camera
        if self.camera is not None:
            logger.info("Closing camera...")
            self.camera.close()

        # Close Web UI
        if self.webui is not None:
            logger.info("Closing Web UI...")
            self.webui.close()

        logger.info("✓ Application stopped")

    def _apply_undistortion(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply camera undistortion to the frame.

        Args:
            frame: Input frame

        Returns:
            Undistorted frame
        """
        try:
            # Build camera matrix and distortion coefficients
            h, w = frame.shape[:2]

            # Use simple camera matrix (assuming focal length ~ image width)
            # In a real application, you would calibrate and save these values
            fx = fy = w  # Approximate focal length
            cx, cy = w / 2, h / 2  # Principal point at image center

            camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)

            # Distortion coefficients (k1, k2, p1, p2, k3)
            dist_coeffs = np.array([
                self.config.calibration.dist_coef[0],
                self.config.calibration.dist_coef[1],
                self.config.calibration.dist_coef[2],
                self.config.calibration.dist_coef[3],
                self.config.calibration.dist_coef[4]
            ], dtype=np.float32)

            # Apply undistortion
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
            return undistorted

        except Exception as e:
            logger.error(f"Undistortion failed: {e}")
            return frame  # Return original frame on error

    def _handle_preview(self) -> Optional[np.ndarray]:
        """Handle preview request from UI"""
        try:
            if self.camera is not None and not self.camera.is_open():
                self.camera.open()

            if self.camera is None or not self.camera.is_open():
                logger.warning("Camera not available for preview")
                return None

            frame = self.camera.capture_frame()

            if frame is None:
                return None

            # Apply undistortion if enabled
            if self.config.camera.undistort:
                frame = self._apply_undistortion(frame)

            return frame

        except Exception as e:
            logger.error(f"Preview error: {e}")
            return None

    def _handle_freeze_toggle(self, frozen: bool):
        """Handle freeze toggle from UI"""
        try:
            self._frozen = frozen
            logger.info(f"Processing {'FROZEN' if frozen else 'RESUMED'}")
        except Exception as e:
            logger.error(f"Freeze toggle error: {e}")

    def _handle_config_update(self, config):
        """Handle configuration update from UI"""
        try:
            logger.info("Updating camera configuration...")

            if self.camera is not None and self.camera.is_open():
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

    def _handle_calibration_config_update(self, config):
        """Handle calibration configuration update from UI"""
        try:
            logger.info("Updating calibration configuration...")

            # Update system config
            self.config.calibration = config

            logger.info("✓ Calibration configuration updated")

        except Exception as e:
            logger.error(f"Calibration config update error: {e}")

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

            # Pause processing during camera reset
            was_frozen = self._frozen
            if not was_frozen:
                self._frozen = True
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

                    # Restore processing state
                    if not was_frozen:
                        self._frozen = False

                    return "✓ Camera reset successful"
                else:
                    logger.error("Failed to reopen camera")
                    return "✗ Failed to reopen camera"
            else:
                return "✗ No camera available"

        except Exception as e:
            logger.error(f"Camera reset error: {e}")
            return f"✗ Error: {str(e)}"

    def _handle_wb_calibrate(self) -> str:
        """Handle white balance calibration request from UI"""
        try:
            logger.info("Starting white balance calibration...")

            if self.camera is None or not self.camera.is_open():
                logger.error("Camera not available for white balance calibration")
                return "✗ Camera not available"

            # Capture frame for calibration
            self.camera.calibrate_white_balance()  # pyright: ignore[reportOptionalMemberAccess]
            frame = self.camera.capture_frame()
            self._handle_save_config()

            if frame is None:
                logger.error("Failed to capture frame for white balance calibration")
                return "✗ Failed to capture frame"

            return "✓ White balance calibration successful"

        except Exception as e:
            logger.error(f"White balance calibration error: {e}")
            return f"✗ Error: {str(e)}"


def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("RGBTrack Camera Calibration Tool")
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
