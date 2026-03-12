"""ZeroMQ publisher and control server for RGBTrack inference."""

from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
import zmq

from .config import ZMQConfig
from .detection import DetectionResult
from .zmq_common import (
    CMD_DISABLE_FRAME_BUFFER,
    CMD_ENABLE_FRAME_BUFFER,
    CMD_GET_STATUS,
    CMD_START_RECORDING,
    CMD_STOP_RECORDING,
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
    KEY_TYPE,
    MSG_TYPE_COMMAND,
    MSG_TYPE_ERROR,
    MSG_TYPE_RESULT,
    MSG_TYPE_STATUS,
)

logger = logging.getLogger(__name__)


class ZMQPublisher:
    """ZMQ service with PUB for results and REP for control commands."""

    def __init__(self, config: ZMQConfig):
        self.config = config
        self.context = zmq.Context()
        self.pub_socket: Optional[zmq.Socket] = None
        self.rep_socket: Optional[zmq.Socket] = None

        self._running = False
        self._pub_thread: Optional[threading.Thread] = None
        self._rep_thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()

        self._latest_payload: Optional[Dict[str, Any]] = None
        self._new_payload_event = threading.Event()
        self._frame_buffer_enabled = False
        self._recording = False
        self._status = "IDLE"
        self._prompt = ""
        self._nms_threshold = 0.4
        self._command_handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

    def set_command_handler(self, handler: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """Register external command handler from service layer."""
        self._command_handler = handler

    def start(self) -> None:
        """Start PUB and REP sockets with background loops."""
        if self._running:
            return

        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.setsockopt(zmq.SNDHWM, 1)
        self.pub_socket.bind(self.config.results_address)

        self.rep_socket = self.context.socket(zmq.REP)
        self.rep_socket.bind(self.config.control_address)
        self.rep_socket.setsockopt(zmq.RCVTIMEO, 200)

        self._running = True
        self._pub_thread = threading.Thread(target=self._publish_loop, daemon=True, name="zmq_pub_thread")
        self._rep_thread = threading.Thread(target=self._control_loop, daemon=True, name="zmq_rep_thread")
        self._pub_thread.start()
        self._rep_thread.start()

        logger.info("ZMQ PUB bound to %s", self.config.results_address)
        logger.info("ZMQ REP bound to %s", self.config.control_address)

    def stop(self) -> None:
        """Stop background threads and close sockets."""
        if not self._running:
            return

        self._running = False
        self._new_payload_event.set()

        if self._pub_thread is not None:
            self._pub_thread.join(timeout=1.0)
        if self._rep_thread is not None:
            self._rep_thread.join(timeout=1.0)

        if self.pub_socket is not None:
            self.pub_socket.close(0)
            self.pub_socket = None
        if self.rep_socket is not None:
            self.rep_socket.close(0)
            self.rep_socket = None

        self.context.term()
        logger.info("ZMQ publisher stopped")

    def update_status(self, *, status: Optional[str] = None, prompt: Optional[str] = None, nms_threshold: Optional[float] = None) -> None:
        """Update status snapshot for control replies."""
        with self._state_lock:
            if status is not None:
                self._status = status
            if prompt is not None:
                self._prompt = prompt
            if nms_threshold is not None:
                self._nms_threshold = float(nms_threshold)

    def publish_result(self, result: DetectionResult, frame: Optional[np.ndarray] = None) -> None:
        """Update latest result payload for publisher loop."""
        payload = self._build_result_payload(result=result, frame=frame)
        with self._state_lock:
            self._latest_payload = payload
        self._new_payload_event.set()

    def is_frame_buffer_enabled(self) -> bool:
        """Get global frame-buffer enabled state."""
        with self._state_lock:
            return self._frame_buffer_enabled

    def _build_result_payload(self, result: DetectionResult, frame: Optional[np.ndarray]) -> Dict[str, Any]:
        payload = result.to_dict()

        preview_jpeg = payload.get("preview_jpeg")
        if isinstance(preview_jpeg, (bytes, bytearray)):
            payload["preview_jpeg"] = base64.b64encode(preview_jpeg).decode("ascii")

        mask_png = payload.get("mask_png")
        if isinstance(mask_png, (bytes, bytearray)):
            payload["mask_png"] = base64.b64encode(mask_png).decode("ascii")

        include_frame = self.is_frame_buffer_enabled()
        if include_frame and frame is not None:
            ok, encoded = cv2.imencode(".jpg", frame)
            if ok:
                payload["frame_jpeg"] = base64.b64encode(encoded.tobytes()).decode("ascii")

        return {
            KEY_TYPE: MSG_TYPE_RESULT,
            KEY_TIMESTAMP: time.time(),
            KEY_PAYLOAD: payload,
        }

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

    def _send_pub(self, payload: Dict[str, Any]) -> None:
        if self.pub_socket is None:
            return
        try:
            self.pub_socket.send_json(payload)
        except Exception as exc:
            logger.debug("PUB send failed: %s", exc)

    def _get_latest_payload(self) -> Optional[Dict[str, Any]]:
        with self._state_lock:
            if self._latest_payload is None:
                return None
            return dict(self._latest_payload)

    def _control_loop(self) -> None:
        while self._running:
            if self.rep_socket is None:
                return

            try:
                req = self.rep_socket.recv_json()
            except zmq.Again:
                continue
            except Exception as exc:
                logger.debug("REP recv failed: %s", exc)
                continue

            if not isinstance(req, dict):
                response = {
                    KEY_TYPE: MSG_TYPE_ERROR,
                    KEY_SUCCESS: False,
                    KEY_MESSAGE: "invalid request payload",
                    KEY_TIMESTAMP: time.time(),
                }
                try:
                    self.rep_socket.send_json(response)
                except Exception as exc:
                    logger.debug("REP send failed: %s", exc)
                continue

            try:
                response = self._process_control_request(req)
            except Exception as exc:
                response = {
                    KEY_TYPE: MSG_TYPE_ERROR,
                    KEY_SUCCESS: False,
                    KEY_MESSAGE: str(exc),
                    KEY_TIMESTAMP: time.time(),
                }

            try:
                self.rep_socket.send_json(response)
            except Exception as exc:
                logger.debug("REP send failed: %s", exc)

    def _process_control_request(self, req: Dict[str, Any]) -> Dict[str, Any]:
        if req.get(KEY_TYPE) != MSG_TYPE_COMMAND:
            return {
                KEY_TYPE: MSG_TYPE_ERROR,
                KEY_SUCCESS: False,
                KEY_MESSAGE: "Invalid message type",
                KEY_TIMESTAMP: time.time(),
            }

        command = req.get(KEY_COMMAND)
        payload = req.get(KEY_PAYLOAD) or {}

        with self._state_lock:
            if command == CMD_ENABLE_FRAME_BUFFER:
                self._frame_buffer_enabled = True
            elif command == CMD_DISABLE_FRAME_BUFFER:
                self._frame_buffer_enabled = False
            elif command == CMD_GET_STATUS:
                pass

        external_result: Dict[str, Any] = {}
        if self._command_handler is not None and command not in {CMD_ENABLE_FRAME_BUFFER, CMD_DISABLE_FRAME_BUFFER, CMD_GET_STATUS}:
            external_result = self._command_handler({KEY_COMMAND: command, KEY_PAYLOAD: payload})

        # Update recording state after external handler processes it
        if command in {CMD_START_RECORDING, CMD_STOP_RECORDING}:
            with self._state_lock:
                self._recording = external_result.get(KEY_RECORDING, self._recording)

        with self._state_lock:
            return {
                KEY_TYPE: MSG_TYPE_STATUS,
                KEY_SUCCESS: external_result.get(KEY_SUCCESS, True),
                KEY_STATUS: self._status,
                KEY_PROMPT: self._prompt,
                KEY_NMS_THRESHOLD: self._nms_threshold,
                KEY_FRAME_BUFFER_ENABLED: self._frame_buffer_enabled,
                KEY_RECORDING: self._recording,
                KEY_MESSAGE: external_result.get(KEY_MESSAGE, "ok"),
                KEY_TIMESTAMP: time.time(),
            }

