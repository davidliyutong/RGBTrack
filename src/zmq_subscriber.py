"""ZeroMQ subscriber client for RGBTrack inference service."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

import zmq

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
    KEY_PAYLOAD,
    KEY_TIMESTAMP,
    KEY_TYPE,
    MSG_TYPE_COMMAND,
)

logger = logging.getLogger(__name__)


class ZMQSubscriber:
    """Client with SUB for results and REQ for control commands."""

    def __init__(self, result_address: str, control_address: str):
        self.context = zmq.Context()
        self.result_address = result_address
        self.control_address = control_address

        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(self.result_address)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub_socket.setsockopt(zmq.RCVHWM, 1)
        self.sub_socket.setsockopt(zmq.CONFLATE, 1)
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 200)

        self.req_socket = self.context.socket(zmq.REQ)
        self.req_socket.connect(self.control_address)
        self.req_socket.setsockopt(zmq.RCVTIMEO, 1000)
        self.req_socket.setsockopt(zmq.LINGER, 0)

        self._lock = threading.Lock()
        self._disconnected = False

    def disconnect(self) -> None:
        """Close sockets and context."""
        if self._disconnected:
            return
        self._disconnected = True
        self.sub_socket.close(0)
        self.req_socket.close(0)
        self.context.term()

    def __enter__(self) -> "ZMQSubscriber":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    def recv_result(self, timeout_ms: int = 200) -> Optional[Dict[str, Any]]:
        """Receive one result message from SUB socket."""
        self.sub_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        try:
            message = self.sub_socket.recv_json()
            if isinstance(message, dict):
                return message
            return None
        except zmq.Again:
            return None
        except Exception as exc:
            logger.debug("SUB recv failed: %s", exc)
            return None

    def recv_latest_result(self, timeout_ms: int = 50, max_drain: int = 100) -> Optional[Dict[str, Any]]:
        """Receive and return only the latest available result message."""
        latest = self.recv_result(timeout_ms=timeout_ms)
        if latest is None:
            return None

        drained = 0
        while drained < max_drain:
            try:
                message = self.sub_socket.recv_json(flags=zmq.DONTWAIT)
            except zmq.Again:
                break
            except Exception:
                break

            if isinstance(message, dict):
                latest = message
            drained += 1

        return latest

    def send_command(self, command: str, payload: Optional[Dict[str, Any]] = None, timeout_ms: Optional[int] = None) -> Dict[str, Any]:
        """Send one control command over REQ and wait for REP response."""
        request = {
            KEY_TYPE: MSG_TYPE_COMMAND,
            KEY_COMMAND: command,
            KEY_PAYLOAD: payload or {},
            KEY_TIMESTAMP: time.time(),
        }

        with self._lock:
            try:
                if timeout_ms is not None:
                    self.req_socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
                self.req_socket.send_json(request)
                response = self.req_socket.recv_json()
                if isinstance(response, dict):
                    return response
                return {"success": False, "message": "invalid response", "timestamp": time.time()}
            except zmq.Again:
                return {"success": False, "message": "timeout", "timestamp": time.time()}
            except Exception as exc:
                self._reset_req_socket()
                return {"success": False, "message": f"zmq error: {exc}", "timestamp": time.time()}
            finally:
                if timeout_ms is not None:
                    self.req_socket.setsockopt(zmq.RCVTIMEO, 1000)

    def start_detection(self) -> Dict[str, Any]:
        return self.send_command(CMD_START)

    def pause(self) -> Dict[str, Any]:
        return self.send_command(CMD_PAUSE)

    def resume(self) -> Dict[str, Any]:
        return self.send_command(CMD_RESUME)

    def reset(self) -> Dict[str, Any]:
        return self.send_command(CMD_RESET)

    def enable_frame_buffer(self) -> Dict[str, Any]:
        return self.send_command(CMD_ENABLE_FRAME_BUFFER)

    def disable_frame_buffer(self) -> Dict[str, Any]:
        return self.send_command(CMD_DISABLE_FRAME_BUFFER)

    def set_prompt(self, prompt: str) -> Dict[str, Any]:
        return self.send_command(CMD_SET_PROMPT, {"prompt": prompt})

    def set_nms_threshold(self, value: float) -> Dict[str, Any]:
        return self.send_command(CMD_SET_NMS_THRESHOLD, {"nms_threshold": float(value)})

    def get_status(self) -> Dict[str, Any]:
        return self.send_command(CMD_GET_STATUS, timeout_ms=100)

    def _reset_req_socket(self) -> None:
        try:
            self.req_socket.close(0)
        except Exception:
            pass
        self.req_socket = self.context.socket(zmq.REQ)
        self.req_socket.connect(self.control_address)
        self.req_socket.setsockopt(zmq.RCVTIMEO, 1000)
        self.req_socket.setsockopt(zmq.LINGER, 0)

