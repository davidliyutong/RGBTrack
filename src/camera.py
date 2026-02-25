"""Camera abstraction layer for MindVision SDK or other camera systems"""

import mvsdk
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

import av
import cv2
import numpy as np

from .config import CameraConfig

logger = logging.getLogger(__name__)


class CameraBase(ABC):
    """Abstract base class for camera implementations"""

    def __init__(self, config: CameraConfig):
        self.config = config
        self._master_lock = threading.Lock()
        self._config_lock = threading.Lock()
        self._is_open = False

        # FPS tracking
        self._frame_timestamps: deque = deque(maxlen=5)
        self._fps_lock = threading.Lock()

    @property
    def fps(self) -> float:
        """Get current camera FPS (averaged over last 5 frames)"""
        with self._fps_lock:
            if len(self._frame_timestamps) < 2:
                return 0.0
            dt = self._frame_timestamps[-1] - self._frame_timestamps[0]
            if dt > 0:
                return (len(self._frame_timestamps) - 1) / dt
            return 0.0

    def _record_frame_capture(self):
        """Record timestamp for FPS calculation (call after successful capture)"""
        with self._fps_lock:
            self._frame_timestamps.append(time.time())

    @staticmethod
    @abstractmethod
    def enumerate_cameras() -> List[Dict[str, Any]]:
        """Enumerate available MindVision cameras."""
        return []

    @abstractmethod
    def open(self) -> bool:
        """Open the camera"""
        pass

    @abstractmethod
    def close(self):
        """Close the camera"""
        pass

    @abstractmethod
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame (RGB format)"""
        pass

    @abstractmethod
    def set_exposure(self, exposure_ms: int):
        """Set exposure time in milliseconds"""
        pass

    @abstractmethod
    def set_gain(self, gain: float):
        """Set camera gain"""
        pass

    @abstractmethod
    def set_gamma(self, gamma: float):
        """Set camera gamma value"""
        pass

    @abstractmethod
    def set_wb_mode(self, mode: str):
        """Set white balance mode (auto or manual)"""
        pass

    @abstractmethod
    def set_rgb_balance(self, red: float, green: float, blue: float):
        """Set RGB white balance (manual mode only)"""
        pass

    @abstractmethod
    def set_mode(self, mode: str):
        """Set camera mode (high_speed or normal)"""
        pass

    @abstractmethod
    def calibrate_white_balance(self) -> bool:
        """Perform automatic white balance calibration"""
        pass

    def is_open(self) -> bool:
        """Check if camera is open"""
        return self._is_open


class MindVisionCamera(CameraBase):
    """
    MindVision Camera implementation.
    This is a placeholder - adapt this to actual MindVision SDK API.
    """

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self.camera_handle = None
        self.capacity = None
        self.frame_buffer_size = 0
        self.p_frame_buffer = None
        self.resolution: mvsdk.tSdkImageResolution
        logger.info(f"Initializing MindVision camera with device_sn: {config.device_sn}")

    @staticmethod
    def enumerate_cameras() -> List[Dict[str, Any]]:
        """Enumerate available MindVision cameras.

        Returns:
            List of camera info dictionaries with keys: 'sn', 'name', 'sensor', 'port'
        """
        try:
            # Enumerate devices
            camera_list = mvsdk.CameraEnumerateDevice()

            cameras = []
            for i, cam_info in enumerate(camera_list):
                cameras.append({
                    'sn': cam_info.GetSn(),
                    'name': cam_info.GetProductName(),
                    'sensor': cam_info.GetSensorType(),
                    'port': cam_info.GetPortType(),
                    'friendly_name': cam_info.GetFriendlyName(),
                    'instance_index': i
                })

            logger.info(f"Found {len(cameras)} camera(s)")
            return cameras

        except ImportError:
            logger.warning("mvsdk module not found, returning empty camera list")
            return []
        except Exception as e:
            logger.error(f"Failed to enumerate cameras: {e}")
            return []

    def open(self) -> bool:
        """Open MindVision camera"""
        with self._master_lock:
            try:
                for camera in self.enumerate_cameras():
                    if camera['sn'] == self.config.device_sn:
                        # Initialize camera here
                        camera_list = mvsdk.CameraEnumerateDevice()
                        try:
                            self.camera_handle = mvsdk.CameraInit(camera_list[camera['instance_index']], -1, -1)
                            break
                        except mvsdk.CameraException as e:
                            logger.error(f"Camera initialization failed: {e}")
                            return False

                if self.camera_handle is None:
                    logger.error(f"Camera with SN {self.config.device_sn} not found")
                    return False

                logger.info(f"MindVision camera {self.config.device_sn} opened successfully")
                self._is_open = True

                # Apply initial settings
                self.set_isp_output_format()
                self.set_exposure(self.config.exposure_time_ms)
                self.set_gain(self.config.gain)
                self.set_gamma(self.config.gamma)
                self.set_wb_mode(self.config.wb_mode)
                self.set_rgb_balance(
                    self.config.red_balance,
                    self.config.green_balance,
                    self.config.blue_balance
                )
                self.set_mode(self.config.mode)
                mvsdk.CameraPlay(self.camera_handle)

                return True
            except Exception as e:
                logger.error(f"Failed to open MindVision camera: {e}")
                return False

    def close(self):
        """Close MindVision camera"""
        with self._master_lock:
            if self.camera_handle is not None:
                try:
                    if self.camera_handle is not None and self.camera_handle > 0:
                        mvsdk.CameraStop(self.camera_handle)
                        mvsdk.CameraUnInit(self.camera_handle)
                        self.camera_handle = None
                    if self.p_frame_buffer is not None and self.p_frame_buffer != 0:
                        mvsdk.CameraAlignFree(self.p_frame_buffer)
                        self.p_frame_buffer = None
                    logger.info("MindVision camera closed")
                    self._is_open = False
                except Exception as e:
                    logger.error(f"Error closing camera: {e}")

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from MindVision camera"""
        if not self._is_open:
            return None
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.camera_handle, 1000)
            mvsdk.CameraImageProcess(self.camera_handle, pRawData, self.p_frame_buffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(self.camera_handle, pRawData)
            frame = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.p_frame_buffer)
            frame = np.frombuffer(frame, dtype=np.uint8).reshape((FrameHead.iHeight, FrameHead.iWidth, 3))  # FIXME: hardcoded to 3 channels

            # Record FPS
            self._record_frame_capture()

            return frame
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return None

    def set_isp_output_format(self):
        with self._config_lock:
            if self._is_open and self.camera_handle is not None:
                # pixel and buffer
                mvsdk.CameraSetIspOutFormat(self.camera_handle, mvsdk.CAMERA_MEDIA_TYPE_YUV422_8)  # FIXME: hardcoded for color
                self.capacity = mvsdk.CameraGetCapability(self.camera_handle)
                self.frame_buffer_size = self.capacity.sResolutionRange.iWidthMax * self.capacity.sResolutionRange.iHeightMax * 3
                self.p_frame_buffer = mvsdk.CameraAlignMalloc(self.frame_buffer_size, 16)

                # resolution
                self.resolution: mvsdk.tSdkImageResolution = self.capacity.pImageSizeDesc[0]  # FIXME: use the first resolution by default
                mvsdk.CameraSetImageResolution(self.camera_handle, self.resolution)

                # trigger
                mvsdk.CameraSetTriggerMode(self.camera_handle, 0)  # 0: auto, 1 software, 2 hardware  # FIXME: hardcoded to auto

    def set_exposure(self, exposure_ms: int):
        """Set exposure time"""
        with self._config_lock:
            self.config.exposure_time_ms = exposure_ms
            if self._is_open and self.camera_handle is not None:
                if self.config.mode == "high_speed":
                    mvsdk.CameraSetFrameSpeed(self.camera_handle, 1)  # high speed
                else:
                    mvsdk.CameraSetFrameSpeed(self.camera_handle, 0)  # normal speed
                mvsdk.CameraSetAeState(self.camera_handle, 0)  # Disable auto exposure
                mvsdk.CameraSetExposureTime(self.camera_handle, int(exposure_ms * 1000))
                logger.debug(f"Set exposure to {exposure_ms}ms")

    def set_gain(self, gain: float):
        """Set camera gain"""
        with self._config_lock:
            self.config.gain = gain
            if self._is_open and self.camera_handle is not None:
                mvsdk.CameraSetAnalogGain(self.camera_handle, int(gain * 10))  # FIXME: try compare with mvsdk.CameraSetAeTarget
                logger.debug(f"Set gain to {gain}")

    def set_gamma(self, gamma: float):
        """Set camera gamma value"""
        with self._config_lock:
            self.config.gamma = gamma
            if self._is_open and self.camera_handle is not None:
                mvsdk.CameraSetGamma(self.camera_handle, int(gamma * 100))
                logger.debug(f"Set gamma to {gamma}")

    def set_wb_mode(self, mode: str):
        """Set white balance mode (auto or manual)"""
        with self._config_lock:
            self.config.wb_mode = mode  # pyright: ignore[reportAttributeAccessIssue]
            if self._is_open and self.camera_handle is not None:
                if mode == "auto":
                    # Enable auto white balance
                    mvsdk.CameraSetWbMode(self.camera_handle, 1)
                    logger.debug("Set white balance to auto mode")
                else:
                    # Disable auto white balance for manual control
                    mvsdk.CameraSetWbMode(self.camera_handle, 0)
                    # Apply current manual RGB balance values
                    mvsdk.CameraSetGain(
                        self.camera_handle,
                        iRGain=int(self.config.red_balance * 100),
                        iGGain=int(self.config.green_balance * 100),
                        iBGain=int(self.config.blue_balance * 100)
                    )
                    logger.debug("Set white balance to manual mode")

    def set_rgb_balance(self, red: float, green: float, blue: float):
        """Set RGB white balance (manual mode only)"""
        with self._config_lock:
            self.config.red_balance = red
            self.config.green_balance = green
            self.config.blue_balance = blue
            if self._is_open and self.camera_handle is not None:
                # Only apply if in manual mode
                if self.config.wb_mode == "manual":
                    mvsdk.CameraSetGain(self.camera_handle, iRGain=int(red * 100), iGGain=int(green * 100), iBGain=int(blue * 100))
                    logger.debug(f"Set RGB balance to R:{red}, G:{green}, B:{blue}")
                else:
                    logger.debug(f"RGB balance updated in config but not applied (auto WB is active)")

    def set_mode(self, mode: str):
        """Set camera mode"""
        with self._config_lock:
            self.config.mode = mode  # pyright: ignore[reportAttributeAccessIssue]
            if self._is_open and self.camera_handle is not None:
                # TODO: Replace with actual MindVision SDK call
                # Different modes might affect frame rate, resolution, etc.
                logger.debug(f"Set mode to {mode}")

    def calibrate_white_balance(self) -> bool:
        with self._config_lock:
            if self._is_open and self.camera_handle is not None:
                try:
                    # Perform auto white balance calibration
                    mvsdk.CameraSetOnceWB(self.camera_handle)  # Trigger one-time WB
                    time.sleep(1)  # Wait for a moment to let camera adjust
                    # Retrieve calibrated gains
                    gains = mvsdk.CameraGetGain(self.camera_handle)
                    r_gain = gains[0] / 100.0
                    g_gain = gains[1] / 100.0
                    b_gain = gains[2] / 100.0

                    # Update config
                    self.config.red_balance = r_gain
                    self.config.green_balance = g_gain
                    self.config.blue_balance = b_gain

                    # gamma correction using opencv
                    frame = self.capture_frame()
                    if frame is not None:
                        # compute the gamma value based on the average brightness
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        mean_brightness = gray.mean()
                        if mean_brightness > 0:
                            gamma = np.log10(0.5) / np.log10(mean_brightness / 255.0)
                            self.config.gamma = gamma
                            logger.info(f"Auto gamma adjusted to {gamma:.2f} based on mean brightness {mean_brightness:.2f}")

                    logger.info(f"White balance calibrated: R:{r_gain}, G:{g_gain}, B:{b_gain}")
                    return True
                except Exception as e:
                    logger.error(f"White balance calibration failed: {e}")
                    return False
            else:
                logger.error("Cannot calibrate - camera is not open")
                return False


class DummyCamera(CameraBase):
    """
    Dummy camera for testing without actual hardware.
    Generates synthetic frames with a moving circle.
    """

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self.frame_count = 0
        logger.info("Initializing Dummy camera for testing")

    @staticmethod
    def enumerate_cameras() -> List[Dict[str, Any]]:
        """Enumerate dummy cameras.

        Returns:
            List of dummy camera info dictionaries.
        """
        return [{
            'sn': 'DUMMY123456',
            'name': 'Dummy Camera',
            'sensor': 'N/A',
            'port': 'N/A',
            'friendly_name': 'Dummy Camera 1',
            'instance_index': 0
        }]

    def open(self) -> bool:
        """Open dummy camera"""
        with self._master_lock:
            self._is_open = True
            logger.info("Dummy camera opened")
            return True

    def close(self):
        """Close dummy camera"""
        with self._master_lock:
            self._is_open = False
            logger.info("Dummy camera closed")

    def capture_frame(self) -> Optional[np.ndarray]:
        """Generate a synthetic frame"""
        if not self._is_open:
            return None

        # Create a frame with a moving circle
        frame = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)

        # Background gradient based on exposure
        brightness = min(255, int(self.config.exposure_time_ms * 2))
        frame[:, :] = brightness // 3

        # Moving circle
        center_x = int(self.config.width / 2 + 200 * np.sin(self.frame_count * 0.1))
        center_y = int(self.config.height / 2 + 100 * np.cos(self.frame_count * 0.1))

        # Apply RGB balance to the circle color
        color = (
            int(100 * self.config.blue_balance),
            int(100 * self.config.green_balance),
            int(255 * self.config.red_balance)
        )

        cv2.circle(frame, (center_x, center_y), 50, color, -1)

        # Add text
        cv2.putText(
            frame,
            f"Frame: {self.frame_count} | Mode: {self.config.mode}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Exp: {self.config.exposure_time_ms}ms | Gain: {self.config.gain:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

        self.frame_count += 1

        # Simulate camera capture time
        time.sleep(0.01 if self.config.mode == "high_speed" else 0.03)

        # Record FPS
        self._record_frame_capture()

        return frame

    def set_exposure(self, exposure_ms: int):
        """Set exposure time"""
        with self._master_lock:
            self.config.exposure_time_ms = exposure_ms
            logger.debug(f"Dummy camera: Set exposure to {exposure_ms}ms")

    def set_gain(self, gain: float):
        """Set camera gain"""
        with self._master_lock:
            self.config.gain = gain
            logger.debug(f"Dummy camera: Set gain to {gain}")

    def set_gamma(self, gamma: float):
        """Set camera gamma"""
        with self._master_lock:
            self.config.gamma = gamma
            logger.debug(f"Dummy camera: Set gamma to {gamma}")

    def set_wb_mode(self, mode: str):
        """Set white balance mode"""
        with self._master_lock:
            self.config.wb_mode = mode  # pyright: ignore[reportAttributeAccessIssue]
            logger.debug(f"Dummy camera: Set white balance mode to {mode}")

    def set_rgb_balance(self, red: float, green: float, blue: float):
        """Set RGB white balance"""
        with self._master_lock:
            self.config.red_balance = red
            self.config.green_balance = green
            self.config.blue_balance = blue
            logger.debug(f"Dummy camera: Set RGB balance to R:{red}, G:{green}, B:{blue}")

    def set_mode(self, mode: str):
        """Set camera mode"""
        with self._master_lock:
            self.config.mode = mode  # pyright: ignore[reportAttributeAccessIssue]
            logger.debug(f"Dummy camera: Set mode to {mode}")

    def calibrate_white_balance(self) -> bool:
        """Perform dummy white balance calibration"""
        with self._master_lock:
            if not self._is_open:
                logger.error("Dummy camera: Cannot calibrate - camera is not open")
                return False

            # Simulate calibration by adjusting RGB balance
            self.config.red_balance = 1.0
            self.config.green_balance = 1.0
            self.config.blue_balance = 1.0
            logger.info("Dummy camera: White balance calibrated (reset to 1.0, 1.0, 1.0)")
            return True


def create_camera(config: CameraConfig, use_dummy: bool = True) -> CameraBase:
    """
    Factory function to create a camera instance.

    Args:
        config: Camera configuration
        use_dummy: If True, creates a DummyCamera for testing

    Returns:
        Camera instance
    """
    # If file parameter is set, use VideoFileCamera
    if config.file:
        return VideoFileCamera(config)
    elif use_dummy:
        return DummyCamera(config)
    else:
        return MindVisionCamera(config)


class VideoFileCamera(CameraBase):
    """
    Camera implementation that reads frames from a video file using PyAV.
    When file parameter is set, hardware controls are disabled.
    """

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self.container: Optional[av.container.InputContainer] = None  # type: ignore
        self.video_stream: Optional[av.video.stream.VideoStream] = None  # type: ignore
        self.frame_iterator = None
        self.current_frame_idx = 0
        self.loop_video = True  # Loop the video when it reaches the end

        # Extract file path from file:// URL
        if config.file.startswith("file://"):
            self.file_path = urlparse(config.file).path
        else:
            self.file_path = config.file

        logger.info(f"Initializing VideoFileCamera with file: {self.file_path}")

    @staticmethod
    def enumerate_cameras() -> List[Dict[str, Any]]:
        """Video file camera doesn't enumerate devices"""
        return []

    def open(self) -> bool:
        """Open video file and get video properties"""
        with self._master_lock:
            try:
                if not Path(self.file_path).exists():
                    logger.error(f"Video file not found: {self.file_path}")
                    return False

                # Open video container
                self.container = av.open(self.file_path)
                self.video_stream = self.container.streams.video[0]  # type: ignore

                # Auto-fill width and height from video properties
                self.config.width = self.video_stream.width  # type: ignore
                self.config.height = self.video_stream.height  # type: ignore

                logger.info(f"Video opened: {self.config.width}x{self.config.height}, "
                            f"fps={self.video_stream.average_rate}, "  # type: ignore
                            f"frames={self.video_stream.frames}")  # type: ignore

                # Create frame iterator
                self.frame_iterator = self.container.decode(video=0)  # type: ignore

                self._is_open = True
                return True

            except Exception as e:
                logger.error(f"Failed to open video file: {e}")
                return False

    def close(self):
        """Close video file"""
        with self._master_lock:
            if self.container is not None:
                try:
                    self.container.close()
                    self.container = None
                    self.video_stream = None
                    self.frame_iterator = None
                    logger.info("Video file closed")
                    self._is_open = False
                except Exception as e:
                    logger.error(f"Error closing video file: {e}")

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture next frame from video file"""
        if not self._is_open or self.frame_iterator is None:
            return None

        try:
            # Get next frame
            frame = next(self.frame_iterator)

            # Convert to numpy array (RGB format)
            img = frame.to_ndarray(format='rgb24')

            self.current_frame_idx += 1

            # Record FPS
            self._record_frame_capture()

            return img

        except StopIteration:
            # Video ended, loop back to start if enabled
            if self.loop_video:
                logger.info("Video ended, looping back to start")
                self.container.seek(0)  # type: ignore
                self.frame_iterator = self.container.decode(video=0)  # type: ignore
                self.current_frame_idx = 0
                return self.capture_frame()
            else:
                logger.info("Video ended")
                return None

        except Exception as e:
            logger.error(f"Error capturing frame from video: {e}")
            return None

    # Hardware controls are disabled for video file playback
    def set_exposure(self, exposure_ms: int):
        """Exposure control disabled for video files"""
        logger.debug("Exposure control not available for video file input")

    def set_gain(self, gain: float):
        """Gain control disabled for video files"""
        logger.debug("Gain control not available for video file input")

    def set_gamma(self, gamma: float):
        """Gamma control disabled for video files"""
        logger.debug("Gamma control not available for video file input")

    def set_wb_mode(self, mode: str):
        """White balance control disabled for video files"""
        logger.debug("White balance control not available for video file input")

    def set_rgb_balance(self, red: float, green: float, blue: float):
        """RGB balance control disabled for video files"""
        logger.debug("RGB balance control not available for video file input")

    def set_mode(self, mode: str):
        """Mode control disabled for video files"""
        logger.debug("Mode control not available for video file input")

    def calibrate_white_balance(self) -> bool:
        """White balance calibration disabled for video files"""
        logger.debug("White balance calibration not available for video file input")
        return False
