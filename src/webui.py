"""Gradio-based Web UI for camera configuration and control - Improved Layout"""

import logging
import time
from pathlib import Path
from typing import Optional, Callable

import cv2
import gradio as gr
import numpy as np

from .config import CameraConfig, DetectionConfig, CalibrationConfig
from .camera import CameraBase
from .calibration import (
    CameraCalibrator,
    AprilTagBoardConfig,
    generate_apriltag_board
)

logger = logging.getLogger(__name__)


class WebUI:
    """
    Gradio-based web interface for camera configuration and control.
    Layout: Left column for configuration tabs, Right column for camera preview & control.
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        detection_config: DetectionConfig,
        calibration_config: CalibrationConfig,
        camera_cls: type[CameraBase],
        on_preview: Callable[[bool], Optional[np.ndarray]],
        on_live_toggle: Callable[[bool], None],
        on_config_update: Callable[[CameraConfig], None],
        on_detection_config_update: Callable[[DetectionConfig], None],
        on_calibration_config_update: Callable[[CalibrationConfig], None],
        on_camera_reset: Callable[[], str] = lambda: "Camera reset not implemented",
        on_save_config: Callable[[], None] = lambda: None,
        on_wb_calibrate: Callable[[], str] = lambda: "White balance calibration not implemented",
        config_path: Optional[Path] = None,
        host: str = "0.0.0.0",
        port: int = 7860
    ):
        self.config = camera_config
        self.detection_config = detection_config
        self.calibration_config = calibration_config
        self.camera_cls = camera_cls
        self.on_preview = on_preview
        self.on_live_toggle = on_live_toggle
        self.on_config_update = on_config_update
        self.on_detection_config_update = on_detection_config_update
        self.on_calibration_config_update = on_calibration_config_update
        self.on_camera_reset = on_camera_reset
        self.on_save_config = on_save_config
        self.on_wb_calibrate = on_wb_calibrate
        self.config_path = config_path or Path("config.yaml")
        self.host = host
        self.port = port

        self.app = None
        self.is_live = False
        self.detection_enabled = True

        # Initialize calibrator
        self.calibrator: Optional[CameraCalibrator] = None
        self._init_calibrator()

    def _init_calibrator(self):
        """Initialize calibrator with current configuration"""
        try:
            board_config = AprilTagBoardConfig(
                family=self.calibration_config.apriltag_family,
                tags_x=self.calibration_config.apriltag_tags_x,
                tags_y=self.calibration_config.apriltag_tags_y,
                tag_size=self.calibration_config.apriltag_tag_size,
                tag_spacing=self.calibration_config.apriltag_tag_spacing,
                first_marker_id=self.calibration_config.apriltag_first_marker
            )
            self.calibrator = CameraCalibrator(board_config)
            logger.info(f"Calibrator initialized successfully with AprilTag board")
        except Exception as e:
            logger.error(f"Failed to initialize calibrator: {e}")
            self.calibrator = None

    def _get_camera_choices(self) -> list[str]:
        """Get list of available camera serial numbers for dropdown"""
        cameras = self.camera_cls.enumerate_cameras()
        if not cameras:
            return ["No cameras found"]

        choices = []
        for cam in cameras:
            choices.append(cam['sn'])
        return choices

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface with left-right layout"""

        with gr.Blocks(title="RGBTrack Camera Control") as app:
            gr.Markdown("# 📷 RGBTrack Camera Control Panel")
            gr.Markdown(f"**Config**: `{self.config_path}`")

            with gr.Row():
                # ==================== LEFT COLUMN: Configuration Tabs ====================
                with gr.Column(scale=1):
                    with gr.Tabs():
                        # --------------- TAB 1: Camera Configuration ---------------
                        with gr.Tab("📹 Camera"):
                            gr.Markdown("### Camera Configuration")

                            device_sn = gr.Dropdown(
                                label="Camera Serial Number",
                                value=self.config.device_sn,
                                choices=self._get_camera_choices(),
                                allow_custom_value=True
                            )
                            refresh_cameras_btn = gr.Button("🔄 Refresh", size="sm")

                            exposure_slider = gr.Slider(
                                minimum=self.config.exposure_min,
                                maximum=self.config.exposure_max,
                                value=self.config.exposure_time_ms,
                                step=1,
                                label="Exposure (ms)"
                            )

                            gain_slider = gr.Slider(
                                minimum=self.config.gain_min,
                                maximum=self.config.gain_max,
                                value=self.config.gain,
                                step=0.1,
                                label="Gain"
                            )

                            gamma_slider = gr.Slider(
                                minimum=self.config.gamma_min,
                                maximum=self.config.gamma_max,
                                value=self.config.gamma,
                                step=0.1,
                                label="Gamma"
                            )

                            gr.Markdown("**White Balance**")

                            with gr.Row():
                                wb_mode_radio = gr.Radio(
                                    choices=["auto", "manual"],
                                    value=self.config.wb_mode,
                                    label="Mode"
                                )
                                wb_calibrate_btn = gr.Button("Calibrate", size="sm")

                            red_slider = gr.Slider(
                                minimum=self.config.balance_min,
                                maximum=self.config.balance_max,
                                value=self.config.red_balance,
                                step=0.01,
                                label="Red",
                                interactive=(self.config.wb_mode == "manual")
                            )

                            green_slider = gr.Slider(
                                minimum=self.config.balance_min,
                                maximum=self.config.balance_max,
                                value=self.config.green_balance,
                                step=0.01,
                                label="Green",
                                interactive=(self.config.wb_mode == "manual")
                            )

                            blue_slider = gr.Slider(
                                minimum=self.config.balance_min,
                                maximum=self.config.balance_max,
                                value=self.config.blue_balance,
                                step=0.01,
                                label="Blue",
                                interactive=(self.config.wb_mode == "manual")
                            )

                            mode_radio = gr.Radio(
                                choices=["normal", "high_speed"],
                                value=self.config.mode,
                                label="Mode"
                            )

                            gr.Markdown("**Distortion**")

                            with gr.Row():
                                distortion_k1 = gr.Number(label="k1", value=self.config.distortion_k1, precision=6)
                                distortion_k2 = gr.Number(label="k2", value=self.config.distortion_k2, precision=6)
                            with gr.Row():
                                distortion_p1 = gr.Number(label="p1", value=self.config.distortion_p1, precision=6)
                                distortion_p2 = gr.Number(label="p2", value=self.config.distortion_p2, precision=6)
                            distortion_k3 = gr.Number(label="k3", value=self.config.distortion_k3, precision=6)

                            undistort_checkbox = gr.Checkbox(
                                label="Enable Undistortion",
                                value=self.config.undistort
                            )

                            apply_btn = gr.Button("💾 Apply Configuration", variant="primary")
                            config_status = gr.Textbox(label="Status", interactive=False, lines=2)

                        # --------------- TAB 2: Detection Configuration ---------------
                        with gr.Tab("🔍 Detection"):
                            gr.Markdown("### Detection Configuration")

                            detection_prompt = gr.Textbox(
                                label="SAM Prompt",
                                value=self.detection_config.prompt,
                                lines=2
                            )

                            confidence_slider = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=self.detection_config.confidence_threshold,
                                step=0.01,
                                label="Confidence Threshold"
                            )

                            nms_slider = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=self.detection_config.nms_threshold,
                                step=0.01,
                                label="NMS Threshold"
                            )

                            mesh_path_text = gr.Textbox(
                                label="Mesh Path",
                                value=self.detection_config.mesh_path
                            )

                            apply_detection_btn = gr.Button("💾 Apply Detection Config", variant="primary")
                            detection_status = gr.Textbox(label="Status", interactive=False, lines=2)

                        # --------------- TAB 3: Calibration ---------------
                        with gr.Tab("🎯 Calibration"):
                            gr.Markdown("### AprilTag Board Setup")

                            apriltag_family = gr.Dropdown(
                                label="Family",
                                choices=AprilTagBoardConfig.get_available_families(),
                                value=self.calibration_config.apriltag_family
                            )

                            with gr.Row():
                                apriltag_tags_x = gr.Number(
                                    label="Tags X",
                                    value=self.calibration_config.apriltag_tags_x,
                                    precision=0
                                )
                                apriltag_tags_y = gr.Number(
                                    label="Tags Y",
                                    value=self.calibration_config.apriltag_tags_y,
                                    precision=0
                                )

                            with gr.Row():
                                apriltag_tag_size = gr.Number(
                                    label="Tag Size (m)",
                                    value=self.calibration_config.apriltag_tag_size,
                                    precision=4
                                )
                                apriltag_tag_spacing = gr.Number(
                                    label="Spacing (m)",
                                    value=self.calibration_config.apriltag_tag_spacing,
                                    precision=4
                                )

                            apriltag_first_marker = gr.Number(
                                label="First Marker ID",
                                value=self.calibration_config.apriltag_first_marker,
                                precision=0
                            )

                            update_apriltag_btn = gr.Button("Update AprilTag Board")

                            generate_board_btn = gr.Button("📄 Generate Board")
                            board_file = gr.File(label="Download", visible=False)
                            board_status = gr.Textbox(label="Status", interactive=False)

                            gr.Markdown("### Capture Images")

                            image_count = gr.Textbox(label="Images", value="0", interactive=False)

                            with gr.Row():
                                capture_btn = gr.Button("📸 Capture", variant="primary")
                                clear_images_btn = gr.Button("🗑️ Clear")

                            upload_images = gr.File(
                                label="Upload Images",
                                file_count="multiple",
                                file_types=["image"],
                                type="filepath"
                            )

                            with gr.Row():
                                save_images_btn = gr.Button("Save")
                                load_images_btn = gr.Button("Load")

                            gr.Markdown("### Run Calibration")

                            calibrate_btn = gr.Button("🎯 Calibrate", variant="primary", size="lg")
                            calibration_status = gr.Textbox(label="Status", interactive=False, lines=3)

                            gr.Markdown("**Results**")
                            with gr.Row():
                                K_00 = gr.Number(label="fx", precision=2, interactive=False)
                                K_02 = gr.Number(label="cx", precision=2, interactive=False)
                            with gr.Row():
                                K_11 = gr.Number(label="fy", precision=2, interactive=False)
                                K_12 = gr.Number(label="cy", precision=2, interactive=False)

                            gr.Markdown("**Distortion**")
                            with gr.Row():
                                dist_k1 = gr.Number(label="k1", precision=6, interactive=False)
                                dist_k2 = gr.Number(label="k2", precision=6, interactive=False)
                            with gr.Row():
                                dist_p1 = gr.Number(label="p1", precision=6, interactive=False)
                                dist_p2 = gr.Number(label="p2", precision=6, interactive=False)
                            dist_k3 = gr.Number(label="k3", precision=6, interactive=False)

                            save_calibration_btn = gr.Button("💾 Save to Config", variant="primary")
                            save_calib_status = gr.Textbox(label="Save Status", interactive=False)

                # ==================== RIGHT COLUMN: Camera Preview & Control ====================
                with gr.Column(scale=2):
                    gr.Markdown("### 📺 Camera Preview")

                    preview_image = gr.Image(
                        label="Live Preview",
                        type="numpy",
                        height=600
                    )

                    with gr.Row():
                        preview_btn = gr.Button("📷 Capture", size="lg", variant="primary")
                        reset_camera_btn = gr.Button("🔄 Reset Camera", size="lg")

                    with gr.Row():
                        distortion_toggle = gr.Checkbox(label="Distortion Correction", value=False)
                        detection_toggle = gr.Checkbox(label="Detection", value=False)
                        live_toggle = gr.Checkbox(label="Live Mode", value=False)

                    with gr.Row():
                        status_text = gr.Textbox(label="Status", value="Ready", interactive=False)
                        fps_text = gr.Textbox(label="FPS", value="0.0", interactive=False)

            # ==================== Event Handlers ====================

            def update_config(dev_sn, exp, gain, gamma, wb_mode, red, green, blue, mode, k1, k2, p1, p2, k3, undistort):
                try:
                    self.config.device_sn = str(dev_sn)
                    self.config.exposure_time_ms = int(exp)
                    self.config.gain = float(gain)
                    self.config.gamma = float(gamma)
                    self.config.wb_mode = wb_mode  # type: ignore
                    self.config.red_balance = float(red)
                    self.config.green_balance = float(green)
                    self.config.blue_balance = float(blue)
                    self.config.mode = mode
                    self.config.distortion_k1 = float(k1)
                    self.config.distortion_k2 = float(k2)
                    self.config.distortion_p1 = float(p1)
                    self.config.distortion_p2 = float(p2)
                    self.config.distortion_k3 = float(k3)
                    self.config.undistort = bool(undistort)
                    self.on_config_update(self.config)
                    if self.on_save_config:
                        self.on_save_config()
                    return f"✓ Configuration saved to {self.config_path}"
                except Exception as e:
                    logger.error(f"Failed to update config: {e}")
                    return f"✗ Error: {str(e)}"

            def update_detection_config(prompt, confidence, nms, mesh_path):
                try:
                    self.detection_config.prompt = str(prompt)
                    self.detection_config.confidence_threshold = float(confidence)
                    self.detection_config.nms_threshold = float(nms)
                    self.detection_config.mesh_path = str(mesh_path)
                    self.on_detection_config_update(self.detection_config)
                    if self.on_save_config:
                        self.on_save_config()
                    return f"✓ Detection config saved to {self.config_path}"
                except Exception as e:
                    logger.error(f"Failed to update detection config: {e}")
                    return f"✗ Error: {str(e)}"

            def take_preview(detection_enabled, distortion_enabled):
                try:
                    frame = self.on_preview(detection_enabled)
                    if frame is not None:
                        return frame
                    else:
                        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(placeholder, "No camera available", (180, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        return placeholder
                except Exception as e:
                    logger.error(f"Preview failed: {e}")
                    return None

            def refresh_cameras():
                return gr.Dropdown(choices=self._get_camera_choices())

            def reset_camera():
                try:
                    return self.on_camera_reset()
                except Exception as e:
                    logger.error(f"Camera reset failed: {e}")
                    return f"✗ Error: {str(e)}"

            def toggle_wb_mode(mode):
                is_manual = (mode == "manual")
                return [gr.Slider(interactive=is_manual)] * 3

            def toggle_live(is_live):
                try:
                    self.is_live = is_live
                    self.on_live_toggle(is_live)
                    return "Live mode: ACTIVE" if is_live else "Live mode: INACTIVE"
                except Exception as e:
                    logger.error(f"Live toggle failed: {e}")
                    return f"Error: {str(e)}"

            def calibrate_wb():
                try:
                    result = self.on_wb_calibrate()
                    status = f"✓ WB calibrated: R={self.config.red_balance:.2f}, G={self.config.green_balance:.2f}, B={self.config.blue_balance:.2f}"
                    return (status, self.config.red_balance, self.config.green_balance, self.config.blue_balance)
                except Exception as e:
                    logger.error(f"WB calibration failed: {e}")
                    return (f"✗ Error: {str(e)}", self.config.red_balance, self.config.green_balance, self.config.blue_balance)

            # Calibration handlers
            def update_apriltag_config(family, tags_x, tags_y, tag_size, tag_spacing, first_marker):
                try:
                    self.calibration_config.apriltag_family = str(family)
                    self.calibration_config.apriltag_tags_x = int(tags_x)
                    self.calibration_config.apriltag_tags_y = int(tags_y)
                    self.calibration_config.apriltag_tag_size = float(tag_size)
                    self.calibration_config.apriltag_tag_spacing = float(tag_spacing)
                    self.calibration_config.apriltag_first_marker = int(first_marker)
                    self._init_calibrator()
                    return "✓ AprilTag board configuration updated"
                except Exception as e:
                    return f"✗ Error: {str(e)}"

            def generate_board_image():
                try:
                    if self.calibrator is None:
                        return None, "✗ Calibrator not initialized"

                    output_path = Path("apriltag_board.png")
                    success = generate_apriltag_board(
                        self.calibrator.board_config,  # type: ignore
                        output_path,
                        dpi=300
                    )

                    return (str(output_path), f"✓ Generated: {output_path}") if success else (None, "✗ Failed")
                except Exception as e:
                    return None, f"✗ Error: {str(e)}"

            def capture_calib_image():
                try:
                    if self.calibrator is None:
                        return None, "0", "✗ Calibrator not initialized"
                    frame = self.on_preview(False)
                    if frame is None:
                        return None, str(self.calibrator.get_image_count()), "✗ No frame"
                    success, annotated = self.calibrator.add_image(frame)
                    count = self.calibrator.get_image_count()
                    return (annotated, str(count), f"✓ Captured! Total: {count}") if success else (annotated, str(count), "✗ Board not detected")
                except Exception as e:
                    return None, str(self.calibrator.get_image_count() if self.calibrator else 0), f"✗ Error: {str(e)}"

            def clear_calib_images():
                try:
                    if self.calibrator is None:
                        return "0", "✗ Calibrator not initialized"
                    self.calibrator.clear_images()
                    return "0", "✓ Cleared"
                except Exception as e:
                    return "0", f"✗ Error: {str(e)}"

            def upload_calib_images(files):
                try:
                    if self.calibrator is None or not files:
                        return None, "0", "✗ Error"
                    success_count = 0
                    last_annotated = None
                    for file_path in files:
                        img = cv2.imread(file_path)
                        if img is not None:
                            success, annotated = self.calibrator.add_image(img)
                            if success:
                                success_count += 1
                                last_annotated = annotated
                    count = self.calibrator.get_image_count()
                    return last_annotated, str(count), f"✓ {success_count}/{len(files)} detected. Total: {count}"
                except Exception as e:
                    return None, str(self.calibrator.get_image_count() if self.calibrator else 0), f"✗ Error: {str(e)}"

            def save_calib_images():
                try:
                    if self.calibrator is None:
                        return "✗ Calibrator not initialized"
                    output_dir = Path(self.calibration_config.calibration_images_dir)
                    count = self.calibrator.save_images(output_dir)
                    return f"✓ Saved {count} images to {output_dir}"
                except Exception as e:
                    return f"✗ Error: {str(e)}"

            def load_calib_images():
                try:
                    if self.calibrator is None:
                        return "0", "✗ Calibrator not initialized"
                    image_dir = Path(self.calibration_config.calibration_images_dir)
                    total, detected = self.calibrator.load_images(image_dir)
                    count = self.calibrator.get_image_count()
                    return str(count), f"✓ Loaded {total}, detected {detected}. Total: {count}"
                except Exception as e:
                    return "0", f"✗ Error: {str(e)}"

            def run_calibration():
                try:
                    if self.calibrator is None:
                        return ("✗ Error", 0, 0, 0, 0, 0, 0, 0, 0, 0)
                    success, K, dist_coef, message = self.calibrator.calibrate()
                    if success and K is not None and dist_coef is not None:
                        fx, fy = float(K[0, 0]), float(K[1, 1])
                        cx, cy = float(K[0, 2]), float(K[1, 2])
                        k1, k2, p1, p2, k3 = [float(x) for x in dist_coef.ravel()[:5]]
                        return (message, fx, cx, fy, cy, k1, k2, p1, p2, k3)
                    else:
                        return (message, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                except Exception as e:
                    return (f"✗ Error: {str(e)}", 0, 0, 0, 0, 0, 0, 0, 0, 0)

            def save_calibration_results(fx, cx, fy, cy, k1, k2, p1, p2, k3):
                try:
                    self.calibration_config.K = [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]]
                    self.calibration_config.dist_coef = [float(k1), float(k2), float(p1), float(p2), float(k3)]
                    self.on_calibration_config_update(self.calibration_config)
                    if self.on_save_config:
                        self.on_save_config()
                    return f"✓ Saved to {self.config_path}"
                except Exception as e:
                    return f"✗ Error: {str(e)}"

            # ==================== Connect Events ====================
            apply_btn.click(update_config,
                            inputs=[device_sn, exposure_slider, gain_slider, gamma_slider, wb_mode_radio,
                                    red_slider, green_slider, blue_slider, mode_radio,
                                    distortion_k1, distortion_k2, distortion_p1, distortion_p2, distortion_k3, undistort_checkbox],
                            outputs=config_status)

            apply_detection_btn.click(update_detection_config,
                                      inputs=[detection_prompt, confidence_slider, nms_slider, mesh_path_text],
                                      outputs=detection_status)

            preview_btn.click(take_preview, inputs=[detection_toggle, distortion_toggle], outputs=preview_image)
            reset_camera_btn.click(reset_camera, outputs=status_text)
            refresh_cameras_btn.click(refresh_cameras, outputs=device_sn)
            wb_mode_radio.change(toggle_wb_mode, inputs=wb_mode_radio, outputs=[red_slider, green_slider, blue_slider])
            live_toggle.change(toggle_live, inputs=live_toggle, outputs=status_text)
            wb_calibrate_btn.click(calibrate_wb, outputs=[config_status, red_slider, green_slider, blue_slider])

            # Calibration events
            update_apriltag_btn.click(update_apriltag_config,
                                      inputs=[apriltag_family, apriltag_tags_x, apriltag_tags_y, apriltag_tag_size, apriltag_tag_spacing, apriltag_first_marker],
                                      outputs=board_status)

            generate_board_btn.click(generate_board_image, outputs=[board_file, board_status])
            capture_btn.click(capture_calib_image, outputs=[preview_image, image_count, board_status])
            clear_images_btn.click(clear_calib_images, outputs=[image_count, board_status])
            upload_images.upload(upload_calib_images, inputs=upload_images, outputs=[preview_image, image_count, board_status])
            save_images_btn.click(save_calib_images, outputs=board_status)
            load_images_btn.click(load_calib_images, outputs=[image_count, board_status])
            calibrate_btn.click(run_calibration,
                                outputs=[calibration_status, K_00, K_02, K_11, K_12, dist_k1, dist_k2, dist_p1, dist_p2, dist_k3])
            save_calibration_btn.click(save_calibration_results,
                                       inputs=[K_00, K_02, K_11, K_12, dist_k1, dist_k2, dist_p1, dist_p2, dist_k3],
                                       outputs=save_calib_status)

        self.app = app
        return app

    def launch(self, share: bool = False):
        """Launch the Gradio interface"""
        if self.app is None:
            self.create_interface()
        logger.info(f"Launching WebUI on {self.host}:{self.port}")
        try:
            self.app.launch(  # type: ignore
                server_name=self.host,
                server_port=self.port,
                share=share,
                prevent_thread_lock=True
            )
        except Exception as e:
            logger.error(f"Failed to launch WebUI: {e}")
            raise

    def close(self):
        """Close the Gradio interface"""
        if self.app is not None:
            try:
                self.app.close()
                logger.info("WebUI closed")
            except Exception as e:
                logger.error(f"Error closing WebUI: {e}")
