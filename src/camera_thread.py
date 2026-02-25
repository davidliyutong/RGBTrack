"""Camera thread module for continuous frame capture"""

import logging
import threading
import time
from queue import Queue, Empty
from typing import Optional

import numpy as np

from .camera import CameraBase

logger = logging.getLogger(__name__)


class CameraThread:
    """
    Dedicated thread for continuous camera frame capture.

    Architecture:
    - Runs in its own thread
    - Continuously captures frames when camera is open
    - Puts frames into a queue for detection thread to consume
    - Thread-safe start/stop controls
    """

    def __init__(self, camera: CameraBase, queue_size: int = 5):
        """
        Initialize camera thread.

        Args:
            camera: Camera instance to use for capture
            queue_size: Maximum size of frame queue (default 5)
        """
        self.camera = camera
        self._queue: Queue[tuple[np.ndarray, int]] = Queue(maxsize=queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._dropped_frames = 0
        self._lock = threading.Lock()
        self._paused = False
        self._latest_frame: Optional[tuple[np.ndarray, int]] = None

    def start(self):
        """Start the camera capture thread"""
        if self._running:
            logger.warning("Camera thread already running")
            return

        logger.info("Starting camera capture thread...")
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="camera_thread"
        )
        self._thread.start()
        logger.info("✓ Camera thread started")

    def stop(self, timeout: float = 2.0):
        """Stop the camera capture thread"""
        if not self._running:
            return

        logger.info("Stopping camera thread...")
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=timeout)

        logger.info("✓ Camera thread stopped")

    def get_frame(self, timeout: float = 0.1) -> Optional[tuple[np.ndarray, int]]:
        """
        Get latest frame from queue (non-blocking).

        Args:
            timeout: How long to wait for frame (default 0.1s)

        Returns:
            Frame as numpy array and timestamp, or None if timeout/queue empty
        """
        try:
            frame = self._queue.get(timeout=timeout)
            return frame
        except Empty:
            return None
        except Exception as e:
            logger.error(f"Error getting frame from queue: {e}")
            return None

    def get_latest_frame(self) -> Optional[tuple[np.ndarray, int]]:
        """Get the most recently captured frame."""
        with self._lock:
            return self._latest_frame

    def pause(self):
        """Pause frame enqueueing while keeping camera thread alive."""
        with self._lock:
            self._paused = True

    def resume(self):
        """Resume frame enqueueing."""
        with self._lock:
            self._paused = False

    def is_paused(self) -> bool:
        """Check if enqueueing is paused."""
        with self._lock:
            return self._paused

    def _capture_loop(self):
        """Main camera capture loop (runs in thread)"""
        logger.info("Camera capture loop started")

        while self._running:
            try:
                # Check if camera is open
                if not self.camera.is_open():
                    time.sleep(0.01)
                    continue

                # Capture frame
                frame = self.camera.capture_frame()
                frame_ts = time.time_ns()

                if frame is None:
                    time.sleep(0.01)
                    continue

                with self._lock:
                    self._latest_frame = (frame, frame_ts)
                    paused = self._paused

                if paused:
                    time.sleep(0.01)
                    continue

                self._frame_count += 1

                # Put frame in queue (non-blocking)
                try:
                    # If queue is full, drop old frame and add new one
                    if self._queue.full():
                        try:
                            self._queue.get_nowait()
                            self._dropped_frames += 1
                        except Empty:
                            pass

                    self._queue.put_nowait((frame, frame_ts))

                except Exception as e:
                    logger.debug(f"Queue error: {e}")
                    self._dropped_frames += 1

            except Exception as e:
                logger.error(f"Camera capture error: {e}")
                time.sleep(0.01)

        logger.info(
            f"Camera capture loop stopped (frames: {self._frame_count}, dropped: {self._dropped_frames})")

    def get_stats(self) -> dict:
        """Get camera thread statistics"""
        with self._lock:
            return {
                'running': self._running,
                'frame_count': self._frame_count,
                'dropped_frames': self._dropped_frames,
                'queue_size': self._queue.qsize(),
                'camera_fps': self.camera.fps if hasattr(self.camera, 'fps') else 0.0
            }

    def reset_stats(self):
        """Reset statistics counters"""
        with self._lock:
            self._frame_count = 0
            self._dropped_frames = 0
