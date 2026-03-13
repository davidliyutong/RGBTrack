"""ROS2 publisher and control service for RGBTrack inference."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from .config import ROS2Config
from .detection import DetectionResult
from .zmq_common import (
    CMD_DISABLE_FRAME_BUFFER,
    CMD_ENABLE_FRAME_BUFFER,
    CMD_GET_STATUS,
    CMD_PAUSE,
    CMD_RESET,
    CMD_RESUME,
    CMD_SET_NMS_THRESHOLD,
    CMD_SET_PROMPT,
    CMD_START,
    KEY_COMMAND,
    KEY_FRAME_BUFFER_ENABLED,
    KEY_MESSAGE,
    KEY_NMS_THRESHOLD,
    KEY_PAYLOAD,
    KEY_PROMPT,
    KEY_RECORDING,
    KEY_STATUS,
    KEY_SUCCESS,
    KEY_TIMESTAMP,
)

logger = logging.getLogger(__name__)


class ROS2Publisher:
    """ROS2 service with topics for results and services for control commands.

    Mirrors the ZMQPublisher interface exactly so main.py can swap backends.
    Published topics (under node namespace):
      ~/pose          - geometry_msgs/msg/PoseStamped  (object pose in camera frame)
      ~/twist         - geometry_msgs/msg/TwistStamped (linear and angular velocity)
      ~/camera_info   - sensor_msgs/msg/CameraInfo    (camera intrinsics)
      ~/status        - std_msgs/msg/String            (JSON status snapshot)
      ~/preview_image - sensor_msgs/msg/Image          (RGB8 frame, when frame buffer enabled)

    Control services (under node namespace):
      ~/start                - std_srvs/srv/Trigger  (IDLE → DETECTING)
      ~/pause                - std_srvs/srv/Trigger  (TRACKING → PAUSED)
      ~/resume               - std_srvs/srv/Trigger  (PAUSED → TRACKING)
      ~/reset                - std_srvs/srv/Trigger  (any → IDLE)
      ~/enable_frame_buffer  - std_srvs/srv/SetBool  (enable/disable preview_image topic)
      ~/get_status           - std_srvs/srv/Trigger  (returns JSON status in message field)

    Node parameters (settable via ros2 param set):
      prompt         (string) - CLIP prompt for object selection
      nms_threshold  (double) - NMS threshold for detection
    """

    def __init__(self, config: ROS2Config):
        self.config = config

        self._running = False
        self._pub_thread: Optional[threading.Thread] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()

        self._latest_payload: Optional[Dict[str, Any]] = None
        self._new_payload_event = threading.Event()
        self._frame_buffer_enabled = False
        self._status = "IDLE"
        self._prompt = ""
        self._nms_threshold = 0.4
        self._command_handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

        # rclpy objects — created in start()
        self._node = None
        self._executor = None
        self._pub_pose = None
        self._pub_twist = None
        self._pub_camera_info = None
        self._pub_status = None
        self._pub_image = None

    def set_command_handler(self, handler: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """Register external command handler from service layer."""
        self._command_handler = handler

    def start(self) -> None:
        """Start ROS2 node, publishers, services, and background loops."""
        if self._running:
            return

        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rcl_interfaces.msg import SetParametersResult

        rclpy.init()

        node_name = self.config.node_name
        namespace = self.config.namespace or ""
        self._node = rclpy.create_node(node_name, namespace=namespace)
        node = self._node

        # Publishers
        from geometry_msgs.msg import PoseStamped, TwistStamped
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import String

        self._pub_pose = node.create_publisher(PoseStamped, "pose", 10)
        self._pub_twist = node.create_publisher(TwistStamped, "twist", 10)
        self._pub_camera_info = node.create_publisher(CameraInfo, "camera_info", 10)
        self._pub_status = node.create_publisher(String, "status", 10)
        self._pub_image = node.create_publisher(Image, "preview_image", 10)

        # Services
        from std_srvs.srv import SetBool, Trigger

        node.create_service(Trigger, "start", self._srv_start)
        node.create_service(Trigger, "pause", self._srv_pause)
        node.create_service(Trigger, "resume", self._srv_resume)
        node.create_service(Trigger, "reset", self._srv_reset)
        node.create_service(SetBool, "enable_frame_buffer", self._srv_enable_frame_buffer)
        node.create_service(Trigger, "get_status", self._srv_get_status)

        # Node parameters
        node.declare_parameter("prompt", self._prompt)
        node.declare_parameter("nms_threshold", self._nms_threshold)

        def _on_set_parameters(params):
            for param in params:
                if param.name == "prompt":
                    self._dispatch_command({KEY_COMMAND: CMD_SET_PROMPT, KEY_PAYLOAD: {KEY_PROMPT: param.value}})
                elif param.name == "nms_threshold":
                    self._dispatch_command({KEY_COMMAND: CMD_SET_NMS_THRESHOLD, KEY_PAYLOAD: {KEY_NMS_THRESHOLD: param.value}})
            return SetParametersResult(successful=True)

        node.add_on_set_parameters_callback(_on_set_parameters)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(node)

        self._running = True

        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            daemon=True,
            name="ros2_spin_thread",
        )
        self._spin_thread.start()

        self._pub_thread = threading.Thread(
            target=self._publish_loop,
            daemon=True,
            name="ros2_pub_thread",
        )
        self._pub_thread.start()

        logger.info("ROS2 node '%s' started (namespace: '%s')", node_name, namespace)

    def stop(self) -> None:
        """Stop background threads, shut down ROS2 node."""
        if not self._running:
            return

        self._running = False
        self._new_payload_event.set()

        if self._pub_thread is not None:
            self._pub_thread.join(timeout=1.0)

        if self._executor is not None:
            self._executor.shutdown()

        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)

        if self._node is not None:
            self._node.destroy_node()
            self._node = None

        import rclpy
        try:
            rclpy.shutdown()
        except Exception:
            pass

        logger.info("ROS2 publisher stopped")

    def update_status(self, *, status: Optional[str] = None, prompt: Optional[str] = None, nms_threshold: Optional[float] = None) -> None:
        """Update status snapshot (mirrors ZMQPublisher.update_status)."""
        with self._state_lock:
            if status is not None:
                self._status = status
            if prompt is not None:
                self._prompt = prompt
            if nms_threshold is not None:
                self._nms_threshold = float(nms_threshold)

    def publish_result(self, result: DetectionResult, frame: Optional[np.ndarray] = None) -> None:
        """Store latest result and frame for publish loop."""
        payload = {"result": result, "frame": frame}
        with self._state_lock:
            self._latest_payload = payload
        self._new_payload_event.set()

    def is_frame_buffer_enabled(self) -> bool:
        """Get frame-buffer enabled state."""
        with self._state_lock:
            return self._frame_buffer_enabled

    # ------------------------------------------------------------------
    # Internal publish loop
    # ------------------------------------------------------------------

    def _publish_loop(self) -> None:
        interval = self.config.publish_interval_sec
        while self._running:
            if interval > 0:
                time.sleep(interval)
                payload = self._get_latest_payload()
                if payload is not None:
                    self._send_pub(payload)
            else:
                self._new_payload_event.wait(timeout=0.2)
                self._new_payload_event.clear()
                payload = self._get_latest_payload()
                if payload is not None:
                    self._send_pub(payload)

    def _get_latest_payload(self) -> Optional[Dict[str, Any]]:
        with self._state_lock:
            if self._latest_payload is None:
                return None
            return dict(self._latest_payload)

    def _send_pub(self, payload: Dict[str, Any]) -> None:
        result: DetectionResult = payload["result"]
        frame: Optional[np.ndarray] = payload["frame"]

        try:
            stamp = self._ns_to_stamp(result.timestamp)
            self._pub_pose_msg(result, stamp)
            self._pub_twist_msg(result, stamp)
            self._pub_camera_info_msg(result, stamp)
            self._pub_status_msg(result)
            if self.is_frame_buffer_enabled() and frame is not None:
                self._pub_image_msg(frame, stamp)
        except Exception as exc:
            logger.debug("ROS2 publish failed: %s", exc)

    def _pub_pose_msg(self, result: DetectionResult, stamp) -> None:
        from geometry_msgs.msg import PoseStamped
        from scipy.spatial.transform import Rotation

        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "camera"

        t = result.pose[:3, 3]
        msg.pose.position.x = float(t[0])
        msg.pose.position.y = float(t[1])
        msg.pose.position.z = float(t[2])

        r = Rotation.from_matrix(result.pose[:3, :3])
        q = r.as_quat()  # [x, y, z, w]
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])

        self._pub_pose.publish(msg)

    def _pub_twist_msg(self, result: DetectionResult, stamp) -> None:
        from geometry_msgs.msg import TwistStamped

        msg = TwistStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "camera"

        lv = result.linvel if result.linvel is not None else np.zeros(3)
        av = result.angvel if result.angvel is not None else np.zeros(3)

        msg.twist.linear.x = float(lv[0])
        msg.twist.linear.y = float(lv[1])
        msg.twist.linear.z = float(lv[2])
        msg.twist.angular.x = float(av[0])
        msg.twist.angular.y = float(av[1])
        msg.twist.angular.z = float(av[2])

        self._pub_twist.publish(msg)

    def _pub_camera_info_msg(self, result: DetectionResult, stamp) -> None:
        from sensor_msgs.msg import CameraInfo

        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = "camera"

        if result.camera_intrinsics is not None:
            K = result.camera_intrinsics
            msg.k = [
                float(K[0, 0]), float(K[0, 1]), float(K[0, 2]),
                float(K[1, 0]), float(K[1, 1]), float(K[1, 2]),
                float(K[2, 0]), float(K[2, 1]), float(K[2, 2]),
            ]

        if result.frame_shape is not None and len(result.frame_shape) >= 2:
            msg.height = int(result.frame_shape[0])
            msg.width = int(result.frame_shape[1])

        self._pub_camera_info.publish(msg)

    def _pub_status_msg(self, result: DetectionResult) -> None:
        from std_msgs.msg import String

        with self._state_lock:
            status_str = self._status

        data = {
            KEY_STATUS: status_str,
            KEY_TIMESTAMP: result.timestamp,
            "valid": result.valid,
            "frame_id": result.frame_id,
            "processing_time_ms": result.processing_time_ms,
            "camera_fps": result.camera_fps,
            "inference_fps": result.inference_fps,
        }
        msg = String()
        msg.data = json.dumps(data)
        self._pub_status.publish(msg)

    def _pub_image_msg(self, frame: np.ndarray, stamp) -> None:
        from sensor_msgs.msg import Image

        if frame.ndim != 3 or frame.shape[2] != 3:
            return

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = "camera"
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = "rgb8"
        msg.step = frame.shape[1] * 3
        msg.data = frame.tobytes()
        self._pub_image.publish(msg)

    # ------------------------------------------------------------------
    # Service callbacks
    # ------------------------------------------------------------------

    def _srv_start(self, request, response):
        result = self._dispatch_command({KEY_COMMAND: CMD_START, KEY_PAYLOAD: {}})
        response.success = bool(result.get(KEY_SUCCESS, False))
        response.message = str(result.get(KEY_MESSAGE, ""))
        return response

    def _srv_pause(self, request, response):
        result = self._dispatch_command({KEY_COMMAND: CMD_PAUSE, KEY_PAYLOAD: {}})
        response.success = bool(result.get(KEY_SUCCESS, False))
        response.message = str(result.get(KEY_MESSAGE, ""))
        return response

    def _srv_resume(self, request, response):
        result = self._dispatch_command({KEY_COMMAND: CMD_RESUME, KEY_PAYLOAD: {}})
        response.success = bool(result.get(KEY_SUCCESS, False))
        response.message = str(result.get(KEY_MESSAGE, ""))
        return response

    def _srv_reset(self, request, response):
        result = self._dispatch_command({KEY_COMMAND: CMD_RESET, KEY_PAYLOAD: {}})
        response.success = bool(result.get(KEY_SUCCESS, False))
        response.message = str(result.get(KEY_MESSAGE, ""))
        return response

    def _srv_enable_frame_buffer(self, request, response):
        with self._state_lock:
            self._frame_buffer_enabled = bool(request.data)
        cmd = CMD_ENABLE_FRAME_BUFFER if request.data else CMD_DISABLE_FRAME_BUFFER
        # This is an internal command — no external handler dispatch needed
        response.success = True
        response.message = "enabled" if request.data else "disabled"
        logger.info("Frame buffer %s", response.message)
        return response

    def _srv_get_status(self, request, response):
        with self._state_lock:
            data = {
                KEY_STATUS: self._status,
                KEY_PROMPT: self._prompt,
                KEY_NMS_THRESHOLD: self._nms_threshold,
                KEY_FRAME_BUFFER_ENABLED: self._frame_buffer_enabled,
                KEY_RECORDING: False,
                KEY_TIMESTAMP: time.time(),
            }
        response.success = True
        response.message = json.dumps(data)
        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dispatch_command(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Forward command to the registered handler (same as ZMQ control flow)."""
        if self._command_handler is None:
            return {KEY_SUCCESS: False, KEY_MESSAGE: "no command handler registered"}
        try:
            return self._command_handler(msg)
        except Exception as exc:
            return {KEY_SUCCESS: False, KEY_MESSAGE: str(exc)}

    @staticmethod
    def _ns_to_stamp(timestamp_ns: int):
        """Convert nanosecond timestamp to rclpy Time stamp struct."""
        from rclpy.time import Time
        t = Time(nanoseconds=int(timestamp_ns))
        return t.to_msg()
