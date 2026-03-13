"""RGBTrack inference service main entrypoint."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import cv2

from src.camera import create_camera
from src.camera_thread import CameraThread
from src.config import SystemConfig
from src.detection import DetectionAlgorithm, DetectionResult
from src.viser_process import ViserProcess
from src.zmq_common import (
    CMD_PAUSE,
    CMD_RESET,
    CMD_RESUME,
    CMD_SET_NMS_THRESHOLD,
    CMD_SET_PROMPT,
    CMD_START,
    CMD_START_RECORDING,
    CMD_STOP_RECORDING,
    KEY_COMMAND,
    KEY_MESSAGE,
    KEY_NMS_THRESHOLD,
    KEY_PAYLOAD,
    KEY_PROMPT,
    KEY_RECORDING,
    KEY_SUCCESS,
    STATUS_DETECTING,
    STATUS_IDLE,
    STATUS_PAUSED,
    STATUS_TRACKING,
)
from src.zmq_publisher import ZMQPublisher


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("rgbtrack_inference.log"),
    ],
)

logger = logging.getLogger(__name__)


class TrackingState(Enum):
    """Tracking state machine states."""

    IDLE = auto()
    DETECTING = auto()
    TRACKING = auto()
    PAUSED = auto()


class RGBTrackInferenceService:
    """RGBTrack inference service with camera, detection, publisher, and optional Viser."""

    def __init__(self, config: SystemConfig, backend: str = "zeromq"):
        self.config = config
        self._backend = backend
        self.state = TrackingState.IDLE
        self._state_lock = threading.Lock()

        self.camera: Optional[CameraThread] = None
        self.detection: Optional[DetectionAlgorithm] = None
        self.publisher: Optional[Union[ZMQPublisher, Any]] = None
        self.viser_process: Optional[ViserProcess] = None

        self._running = False
        self._detection_thread: Optional[threading.Thread] = None
        self._shutdown_in_progress = False
        self._shutdown_lock = threading.Lock()

        self._current_pose: Optional[np.ndarray] = None
        self._current_mask: Optional[np.ndarray] = None
        self._frame_id = 0

        self._use_viser = True
        self._preview_enabled = False
        self._preview_window_name = "RGBTrack Preview"

        # Recording state
        self._recording = False
        self._recording_lock = threading.Lock()
        self._recording_container = None   # av.OutputContainer
        self._recording_stream = None      # av.VideoStream
        self._recording_path: Optional[Path] = None
        self._recording_frame_count = 0

        self.camera_intrinsics = np.array(self.config.calibration.K, dtype=np.float32)
        self.camera_distortion = np.array(self.config.calibration.dist_coef, dtype=np.float32)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def initialize(self) -> bool:
        """Initialize camera, detector, ZMQ, and optional Viser process."""
        try:
            logger.info("Initializing RGBTrack inference service")

            camera_instance = create_camera(self.config.camera, use_dummy=False)
            if not camera_instance.open():
                logger.error("Failed to open camera")
                return False
            self.camera = CameraThread(camera_instance)
            self.camera.start()
            logger.info("Camera initialized")

            self.detection = DetectionAlgorithm(
                detection_config=self.config.detection,
                calibration_config=self.config.calibration,
            )
            if not self.detection.initialize():
                logger.error("Failed to initialize detection algorithm")
                return False
            logger.info("Detection algorithm initialized")

            if self._backend == "ros2":
                from src.ros2_publisher import ROS2Publisher
                self.publisher = ROS2Publisher(self.config.ros2)
            else:
                self.publisher = ZMQPublisher(self.config.zmq)
            self.publisher.set_command_handler(self._handle_control_command)
            self.publisher.update_status(
                status=self._state_to_string(TrackingState.IDLE),
                prompt=self.config.detection.prompt,
                nms_threshold=self.config.detection.nms_threshold,
            )
            self.publisher.start()
            logger.info("Publisher service initialized (backend: %s)", self._backend)

            if self._use_viser:
                self.viser_process = ViserProcess(self.config)
                self.viser_process.start()
                logger.info("Viser process started with PID %s", self.viser_process.pid)

            return True
        except Exception as exc:
            logger.error("Initialization failed: %s", exc, exc_info=True)
            return False

    def start(self) -> None:
        """Start background detection loop and hold main process."""
        if not self.initialize():
            return

        self._running = True
        self._detection_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True,
            name="detection_thread",
        )
        self._detection_thread.start()

        logger.info("Service started")
        last_stats_log_ts = 0.0
        try:
            while self._running:
                time.sleep(0.1)
                now = time.time()
                if self.camera is not None and now - last_stats_log_ts >= 2.0:
                    logger.info("CameraThread stats: %s", self.camera.get_stats())
                    last_stats_log_ts = now
                if self.viser_process is not None and not self.viser_process.is_alive():
                    logger.warning("Viser process exited unexpectedly")
                    self.stop()
                    break
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop service and release resources."""
        if not self._running and self._shutdown_in_progress:
            return

        self._running = False
        logger.info("Stopping service")

        if self._detection_thread is not None:
            self._detection_thread.join(timeout=2.0)

        if self.publisher is not None:
            self.publisher.stop()
            self.publisher = None

        if self.viser_process is not None:
            self.viser_process.terminate()
            self.viser_process.join(timeout=3.0)
            if self.viser_process.is_alive():
                self.viser_process.kill()
                self.viser_process.join(timeout=1.0)
            self.viser_process = None

        if self.camera is not None:
            self.camera.stop(timeout=2.0)
            self.camera.camera.close()
            self.camera = None

        if self._preview_enabled:
            try:
                cv2.destroyWindow(self._preview_window_name)
                cv2.waitKey(1)
            except Exception:
                pass

        logger.info("Service stopped")

    def _detection_loop(self) -> None:
        logger.info("Detection loop started")
        while self._running:
            if self.camera is None or self.detection is None:
                time.sleep(0.05)
                continue

            frame = self.camera.get_frame(timeout=0.05)
            if frame is None:
                frame = self.camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            frame_data, frame_ts = frame

            if self.config.camera.undistort:
                frame_data = cv2.undistort(frame_data, self.camera_intrinsics, self.camera_distortion)

            # Save frame if recording is active
            with self._recording_lock:
                if self._recording and self._recording_stream is not None:
                    import av
                    av_frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
                    for packet in self._recording_stream.encode(av_frame):
                        self._recording_container.mux(packet)
                    self._recording_frame_count += 1

            with self._state_lock:
                state = self.state

            self._show_preview(frame_data, state)

            if state == TrackingState.IDLE:
                self._run_idle(frame_data, frame_ts)
                continue

            if state == TrackingState.PAUSED:
                continue

            if state == TrackingState.DETECTING:
                self._run_detecting(frame_data, frame_ts)
                continue

            if state == TrackingState.TRACKING:
                self._run_tracking(frame_data, frame_ts)

        logger.info("Detection loop stopped")

    def _show_preview(self, frame: np.ndarray, state: TrackingState) -> None:
        if not self._preview_enabled:
            return

        try:
            if frame.ndim != 3 or frame.shape[2] != 3:
                return

            preview = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            camera_fps = self.camera.camera.fps if self.camera is not None else 0.0
            inference_fps = self.detection.fps if self.detection is not None else 0.0

            cv2.putText(
                preview,
                f"State: {self._state_to_string(state)}",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                preview,
                f"Camera FPS: {camera_fps:.2f}",
                (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                preview,
                f"Inference FPS: {inference_fps:.2f}",
                (20, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                preview,
                "Press q / ESC to quit",
                (20, 122),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            cv2.imshow(self._preview_window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                logger.info("Preview window requested quit")
                self._running = False
        except Exception as exc:
            logger.warning("OpenCV preview disabled due to error: %s", exc)
            self._preview_enabled = False

    def _run_idle(self, frame: np.ndarray, frame_ts: int) -> None:
        """Publish dummy data in the same format when idle."""
        t0 = time.time_ns()
        dummy_pose = np.eye(4, dtype=np.float64)
        dummy_linvel = np.array([0.0, 0.0, 0.0])
        dummy_angvel = np.array([0.0, 0.0, 0.0])
        result = self._build_result(frame=frame, frame_ts=frame_ts, pose=dummy_pose, t0=t0, linvel=dummy_linvel, angvel=dummy_angvel, valid=False)
        self._publish_result(result=result, frame=frame)

    def _run_detecting(self, frame: np.ndarray, frame_ts: int) -> None:

        if self.detection is None:
            return
        t0 = time.time_ns()
        mask = self.detection.detect_first_frame(frame)
        if mask is None:
            logger.warning("First-frame detection failed")
            with self._state_lock:
                self.state = TrackingState.IDLE
            self._update_pub_status()
            return

        pose = self.detection.current_pose
        if pose is None:
            with self._state_lock:
                self.state = TrackingState.IDLE
            self._update_pub_status()
            return

        self._current_pose = pose
        self._current_mask = mask
        self._current_linvel = self.detection.current_linvel
        self._current_angvel = self.detection.current_angvel
        result = self._build_result(frame=frame, frame_ts=frame_ts, pose=pose, t0=t0, linvel=self._current_linvel, angvel=self._current_angvel)
        self._publish_result(result=result, frame=frame)

        with self._state_lock:
            self.state = TrackingState.TRACKING
        self._update_pub_status()

    def _run_tracking(self, frame: np.ndarray, frame_ts: int) -> None:
        if self.detection is None:
            return
        t0 = time.time_ns()
        pose, linvel, angvel = self.detection.track(frame, frame_ts)
        if pose is None:
            return

        self._current_pose = pose
        self._current_linvel = linvel
        self._current_angvel = angvel
        self._current_mask = self.detection.current_mask
        result = self._build_result(frame=frame, frame_ts=frame_ts, pose=pose, t0=t0, linvel=linvel, angvel=angvel)
        self._publish_result(result=result, frame=frame)
        self._frame_id += 1

    def _build_result(self, frame: np.ndarray, frame_ts: int, pose: np.ndarray, t0: int, linvel: np.ndarray, angvel: np.ndarray, valid: bool = True) -> DetectionResult:
        camera_fps = self.camera.camera.fps if self.camera is not None else 0.0
        inference_fps = self.detection.fps if self.detection is not None else 0.0
        return DetectionResult(
            timestamp=frame_ts,
            frame_id=self._frame_id,
            pose=pose,
            linvel=linvel,
            angvel=angvel,
            processing_time_ms=(time.time_ns() - t0) / 1_000_000.0,
            frame_shape=frame.shape,
            camera_intrinsics=np.array(self.config.calibration.K, dtype=np.float64),
            camera_fps=camera_fps,
            inference_fps=inference_fps,
            mask_png=None,
            valid=valid,
        )

    def _publish_result(self, result: DetectionResult, frame: np.ndarray) -> None:
        if self.publisher is None:
            return
        if self.publisher.is_frame_buffer_enabled():
            self.publisher.publish_result(result=result, frame=frame)
        else:
            self.publisher.publish_result(result=result, frame=None)

    def _handle_control_command(self, command_msg: Dict[str, Any]) -> Dict[str, Any]:
        command = command_msg.get(KEY_COMMAND)
        payload = command_msg.get(KEY_PAYLOAD) or {}

        if command == CMD_START:
            ok = self._handle_start_detection()
            return {KEY_SUCCESS: ok, KEY_MESSAGE: "started" if ok else "invalid state"}

        if command == CMD_PAUSE:
            ok = self._handle_pause()
            return {KEY_SUCCESS: ok, KEY_MESSAGE: "paused" if ok else "invalid state"}

        if command == CMD_RESUME:
            ok = self._handle_resume()
            return {KEY_SUCCESS: ok, KEY_MESSAGE: "resumed" if ok else "invalid state"}

        if command == CMD_RESET:
            ok = self._handle_reset()
            return {KEY_SUCCESS: ok, KEY_MESSAGE: "reset" if ok else "reset failed"}

        if command == CMD_SET_PROMPT:
            prompt = payload.get(KEY_PROMPT, "")
            self.config.detection.prompt = str(prompt)
            if self.publisher is not None:
                self.publisher.update_status(prompt=self.config.detection.prompt)
            return {KEY_SUCCESS: True, KEY_MESSAGE: "prompt updated"}

        if command == CMD_SET_NMS_THRESHOLD:
            value = payload.get(KEY_NMS_THRESHOLD)
            try:
                if value is None:
                    raise ValueError("nms_threshold is required")
                nms = float(value)
                self.config.detection.nms_threshold = nms
                if self.publisher is not None:
                    self.publisher.update_status(nms_threshold=nms)
                return {KEY_SUCCESS: True, KEY_MESSAGE: "nms updated"}
            except Exception:
                return {KEY_SUCCESS: False, KEY_MESSAGE: "invalid nms_threshold"}

        if command == CMD_START_RECORDING:
            if self._backend != "zeromq":
                return {KEY_SUCCESS: False, KEY_MESSAGE: "recording not supported in ros2 backend; use ros2 bag record"}
            ok, msg = self._handle_start_recording()
            return {KEY_SUCCESS: ok, KEY_MESSAGE: msg, KEY_RECORDING: ok}

        if command == CMD_STOP_RECORDING:
            if self._backend != "zeromq":
                return {KEY_SUCCESS: False, KEY_MESSAGE: "recording not supported in ros2 backend; use ros2 bag record"}
            ok, msg = self._handle_stop_recording()
            return {KEY_SUCCESS: ok, KEY_MESSAGE: msg, KEY_RECORDING: False}

        return {KEY_SUCCESS: False, KEY_MESSAGE: f"unknown command: {command}"}

    def _handle_start_recording(self) -> tuple:
        """Start recording undistorted camera frames to MP4."""
        import av
        with self._recording_lock:
            if self._recording:
                return False, "already recording"
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            rec_dir = Path("recordings")
            rec_dir.mkdir(parents=True, exist_ok=True)
            out_path = rec_dir / f"{timestamp}.mp4"

            container = av.open(str(out_path), mode="w")
            stream = container.add_stream("h264", rate=self.config.camera.fps)
            stream.width = self.config.camera.width
            stream.height = self.config.camera.height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "fast"}

            self._recording_container = container
            self._recording_stream = stream
            self._recording_path = out_path
            self._recording_frame_count = 0
            self._recording = True
            logger.info("Started recording to %s", out_path)
            return True, f"recording to {out_path}"

    def _handle_stop_recording(self) -> tuple:
        """Stop recording, flush and close MP4 file."""
        with self._recording_lock:
            if not self._recording:
                return False, "not recording"
            self._recording = False
            count = self._recording_frame_count
            out_path = self._recording_path
            try:
                # Flush remaining packets
                for packet in self._recording_stream.encode():
                    self._recording_container.mux(packet)
                self._recording_container.close()
            except Exception as exc:
                logger.warning("Error closing recording container: %s", exc)
            self._recording_container = None
            self._recording_stream = None
            self._recording_path = None
            self._recording_frame_count = 0
            logger.info("Stopped recording. %d frames saved to %s", count, out_path)
            return True, f"stopped. {count} frames saved to {out_path}"

    def _handle_start_detection(self) -> bool:
        with self._state_lock:
            if self.state != TrackingState.IDLE:
                return False
            self.state = TrackingState.DETECTING
        self._update_pub_status()
        return True

    def _handle_pause(self) -> bool:
        with self._state_lock:
            if self.state != TrackingState.TRACKING:
                return False
            self.state = TrackingState.PAUSED
        self._update_pub_status()
        return True

    def _handle_resume(self) -> bool:
        with self._state_lock:
            if self.state != TrackingState.PAUSED:
                return False
            self.state = TrackingState.TRACKING
        self._update_pub_status()
        return True

    def _handle_reset(self) -> bool:
        try:
            with self._state_lock:
                self.state = TrackingState.IDLE
            self._current_pose = None
            self._current_mask = None
            self._frame_id = 0
            if self.detection is not None:
                self.detection.reset()
            self._update_pub_status()
            return True
        except Exception:
            return False

    def _update_pub_status(self) -> None:
        if self.publisher is None:
            return
        with self._state_lock:
            status = self._state_to_string(self.state)
        self.publisher.update_status(status=status)

    @staticmethod
    def _state_to_string(state: TrackingState) -> str:
        if state == TrackingState.DETECTING:
            return STATUS_DETECTING
        if state == TrackingState.TRACKING:
            return STATUS_TRACKING
        if state == TrackingState.PAUSED:
            return STATUS_PAUSED
        return STATUS_IDLE

    def _signal_handler(self, sig, frame) -> None:
        logger.info("Received signal %s, shutting down", sig)
        with self._shutdown_lock:
            if self._shutdown_in_progress:
                return
            self._shutdown_in_progress = True
        self.stop()
        sys.exit(0)


def main() -> None:
    """Main entrypoint."""
    parser = argparse.ArgumentParser(description="RGBTrack inference service")
    parser.add_argument(
        "--backend",
        choices=["zeromq", "ros2"],
        default="zeromq",
        help="Communication backend (default: zeromq)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    config = SystemConfig.from_yaml(config_path)
    service = RGBTrackInferenceService(config, backend=args.backend)
    service.start()


if __name__ == "__main__":
    main()
