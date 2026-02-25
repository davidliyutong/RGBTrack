"""Viser-based inference UI for RGBTrack."""

from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from .config import SystemConfig
from .zmq_common import (
    KEY_FRAME_BUFFER_ENABLED,
    KEY_MESSAGE,
    KEY_NMS_THRESHOLD,
    KEY_PAYLOAD,
    KEY_PROMPT,
    KEY_STATUS,
)
from .zmq_subscriber import ZMQSubscriber

logger = logging.getLogger(__name__)


class ViserInterface:
    """Minimal inference visualization and control interface."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.running = False

        try:
            import viser
        except Exception as exc:
            raise RuntimeError(f"viser import failed: {exc}") from exc

        self._viser = viser
        self.server = viser.ViserServer(host=config.viser.host, port=config.viser.port)
        self.client = ZMQSubscriber(config.zmq)

        self._update_thread: Optional[threading.Thread] = None
        self._latest_pose: Optional[np.ndarray] = None
        self._camera_to_world = np.array(self.config.calibration.extrinsic_matrix, dtype=np.float64)
        self._camera_frustum = None
        self._object_mesh_handle = None
        self._last_status_poll_ts = 0.0

    def _compute_camera_fov_aspect(self) -> tuple[float, float]:
        k = np.array(self.config.calibration.K, dtype=np.float64)
        height = max(1.0, float(self.config.camera.height))
        width = max(1.0, float(self.config.camera.width))
        fy = max(1e-6, float(k[1, 1]))
        fov = 2.0 * np.arctan(height / (2.0 * fy))
        aspect = width / height
        return float(fov), float(aspect)

    def _camera_pose_from_config(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        pose = self._camera_to_world
        position = tuple(pose[:3, 3].tolist())
        wxyz = (1.0, 0.0, 0.0, 0.0)
        try:
            import viser.transforms as tf

            wxyz = tuple(tf.SO3.from_matrix(pose[:3, :3]).wxyz)
        except Exception:
            pass
        return position, wxyz

    def _set_object_pose_from_camera_pose(self, pose_camera: np.ndarray) -> None:
        pose_world = self._camera_to_world @ pose_camera
        self.object_frame.position = tuple(pose_world[:3, 3].tolist())
        try:
            import viser.transforms as tf

            self.object_frame.wxyz = tuple(tf.SO3.from_matrix(pose_world[:3, :3]).wxyz)
        except Exception:
            pass

    def _update_camera_frustum_image(self, frame_b64: Optional[str]) -> None:
        if self._camera_frustum is None:
            return
        if not frame_b64:
            self._camera_frustum.image = None
            return

        try:
            img_bytes = base64.b64decode(frame_b64)
            np_buf = np.frombuffer(img_bytes, dtype=np.uint8)
            rgb = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
            self._camera_frustum.image = rgb
        except Exception as exc:
            logger.debug("Failed to decode frame buffer for frustum: %s", exc)

    def setup_ui(self) -> None:
        with self.server.gui.add_folder("Tracking Control"):
            self.status_text = self.server.gui.add_text("State", initial_value="IDLE", disabled=True)
            self.camera_fps_text = self.server.gui.add_text("Camera FPS", initial_value="0.00", disabled=True)
            self.inference_fps_text = self.server.gui.add_text("Inference FPS", initial_value="0.00", disabled=True)
            self.prompt_input = self.server.gui.add_text("Prompt", initial_value=self.config.detection.prompt)
            self.nms_slider = self.server.gui.add_slider(
                "NMS Threshold",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=self.config.detection.nms_threshold,
            )
            self.frame_toggle = self.server.gui.add_checkbox("Enable Frame Buffer", initial_value=False)
            self.start_button = self.server.gui.add_button("Start Detection")
            self.pause_resume_button = self.server.gui.add_button("Pause")
            self.reset_button = self.server.gui.add_button("Reset")
            self.message_text = self.server.gui.add_text("Message", initial_value="", disabled=True)

        self.world_frame = self.server.scene.add_frame("/world", axes_length=0.2, axes_radius=0.004)
        self.camera_frame = self.server.scene.add_frame("/camera", axes_length=0.15, axes_radius=0.003)
        self.object_frame = self.server.scene.add_frame("/object", axes_length=0.12, axes_radius=0.003)
        cam_pos, cam_wxyz = self._camera_pose_from_config()
        self.camera_frame.position = cam_pos
        self.camera_frame.wxyz = cam_wxyz

        if hasattr(self.server.scene, "world_axes") and self.server.scene.world_axes is not None:
            self.server.scene.world_axes.visible = False

        if self.config.viser.show_camera_cone and hasattr(self.server.scene, "add_camera_frustum"):
            try:
                fov, aspect = self._compute_camera_fov_aspect()
                self._camera_frustum = self.server.scene.add_camera_frustum(
                    name="/camera_cone",
                    fov=fov,
                    aspect=aspect,
                    scale=self.config.viser.camera_cone_scale,
                    color=(90, 170, 255),
                    position=cam_pos,
                    wxyz=cam_wxyz,
                    image=None,
                    format="jpeg",
                )
            except Exception as exc:
                logger.warning("Failed to create camera frustum: %s", exc)

        try:
            import trimesh

            mesh = trimesh.load(self.config.detection.mesh_path)
            if isinstance(mesh, trimesh.Trimesh) and hasattr(self.server.scene, "add_mesh_trimesh"):
                self._object_mesh_handle = self.server.scene.add_mesh_trimesh(name="/object/mesh", mesh=mesh)
        except Exception as exc:
            logger.warning("Failed to load mesh for Viser: %s", exc)

        @self.start_button.on_click
        def _(_event):
            response = self.client.start_detection()
            self._apply_status_response(response)

        @self.pause_resume_button.on_click
        def _(_event):
            if self.pause_resume_button.label == "Pause":
                response = self.client.pause()
                if response.get("success"):
                    self.pause_resume_button.label = "Resume"
            else:
                response = self.client.resume()
                if response.get("success"):
                    self.pause_resume_button.label = "Pause"
            self._apply_status_response(response)

        @self.reset_button.on_click
        def _(_event):
            response = self.client.reset()
            self.pause_resume_button.label = "Pause"
            self._apply_status_response(response)

        @self.prompt_input.on_update
        def _(_event):
            response = self.client.set_prompt(self.prompt_input.value)
            self._apply_status_response(response)

        @self.nms_slider.on_update
        def _(_event):
            response = self.client.set_nms_threshold(float(self.nms_slider.value))
            self._apply_status_response(response)

        @self.frame_toggle.on_update
        def _(_event):
            if self.frame_toggle.value:
                response = self.client.enable_frame_buffer()
            else:
                response = self.client.disable_frame_buffer()
            self._apply_status_response(response)

    def start(self) -> None:
        self.running = True
        self.setup_ui()
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True, name="viser_update_thread")
        self._update_thread.start()
        logger.info("Viser UI running at http://%s:%s", self.config.viser.host, self.config.viser.port)

    def stop(self) -> None:
        self.running = False
        if self._update_thread is not None:
            self._update_thread.join(timeout=1.0)
        self.client.disconnect()

    def _update_loop(self) -> None:
        while self.running:
            result_msg = self.client.recv_latest_result(timeout_ms=30, max_drain=200)
            if result_msg is not None:
                payload = result_msg.get(KEY_PAYLOAD, {})
                camera_fps = payload.get("camera_fps")
                if camera_fps is not None:
                    self.camera_fps_text.value = f"{float(camera_fps):.2f}"
                inference_fps = payload.get("inference_fps")
                if inference_fps is not None:
                    self.inference_fps_text.value = f"{float(inference_fps):.2f}"

                if bool(self.frame_toggle.value):
                    self._update_camera_frustum_image(payload.get("frame_jpeg"))
                else:
                    self._update_camera_frustum_image(None)

                pose_list = payload.get("pose")
                if pose_list is not None:
                    pose = np.array(pose_list, dtype=np.float64)
                    if pose.shape == (4, 4):
                        self._latest_pose = pose
                        self._set_object_pose_from_camera_pose(pose)

            now = time.time()
            if now - self._last_status_poll_ts >= 1.0:
                status = self.client.get_status()
                self._apply_status_response(status)
                self._last_status_poll_ts = now

            time.sleep(0.01)

    def _apply_status_response(self, response: dict) -> None:
        status = response.get(KEY_STATUS)
        if status is not None:
            self.status_text.value = str(status)
        message = response.get(KEY_MESSAGE)
        if message is not None:
            self.message_text.value = str(message)
        prompt = response.get(KEY_PROMPT)
        if prompt is not None and prompt != self.prompt_input.value:
            self.prompt_input.value = str(prompt)
        nms = response.get(KEY_NMS_THRESHOLD)
        if nms is not None:
            nms_value = float(nms)
            if abs(float(self.nms_slider.value) - nms_value) > 1e-6:
                self.nms_slider.value = nms_value
        frame_enabled = response.get(KEY_FRAME_BUFFER_ENABLED)
        if frame_enabled is not None:
            frame_value = bool(frame_enabled)
            if bool(self.frame_toggle.value) != frame_value:
                self.frame_toggle.value = frame_value


def start_viser_interface(config: SystemConfig) -> None:
    """Start the inference Viser interface."""
    interface = ViserInterface(config)
    interface.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        interface.stop()

