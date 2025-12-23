"""Gradio-based Web UI for camera configuration and control"""

import logging
import time
from pathlib import Path
from typing import Optional, Callable

import cv2
import gradio as gr
import numpy as np

from .config import CameraConfig
from .camera import CameraBase

logger = logging.getLogger(__name__)


class WebUI:
    """
    Gradio-based web interface for camera configuration and control.
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        camera_cls: type[CameraBase],
        on_preview: Callable[[bool], Optional[np.ndarray]],
        on_live_toggle: Callable[[bool], None],
        on_config_update: Callable[[CameraConfig], None],
        on_camera_reset: Callable[[], str] = lambda: "Camera reset not implemented",
        on_save_config: Callable[[], None] = lambda: None,
        on_wb_calibrate: Callable[[], tuple[str, float, float, float]] = lambda: ("White balance calibration not implemented", 1.0, 1.0, 1.0),
        config_path: Optional[Path] = None,
        host: str = "0.0.0.0",
        port: int = 7860
    ):
        """
        Initialize WebUI.

        Args:
            camera_config: Initial camera configuration
            on_preview: Callback to get a preview frame (takes detection_enabled bool)
            on_live_toggle: Callback when live mode is toggled
            on_config_update: Callback when configuration is updated
            on_camera_reset: Callback to reset/reopen the camera
            on_save_config: Callback to save configuration to disk
            on_wb_calibrate: Callback to perform white balance calibration
            config_path: Path to configuration file for display
            host: Server host
            port: Server port
        """
        self.config = camera_config
        self.camera_cls = camera_cls
        self.on_preview = on_preview
        self.on_live_toggle = on_live_toggle
        self.on_config_update = on_config_update
        self.on_camera_reset = on_camera_reset
        self.on_save_config = on_save_config
        self.on_wb_calibrate = on_wb_calibrate
        self.config_path = config_path or Path("config.yaml")
        self.host = host
        self.port = port

        self.app = None
        self.is_live = False
        self.detection_enabled = True

    def _get_camera_choices(self) -> list[str]:
        """Get list of available camera serial numbers for dropdown"""
        cameras = self.camera_cls.enumerate_cameras()
        if not cameras:
            return ["No cameras found"]

        # Format: "SN - Name (Sensor)"
        choices = []
        for cam in cameras:
            label = f"{cam['sn']} - {cam['name']} ({cam['sensor']})"
            choices.append(cam['sn'])  # Use SN as value

        return choices

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""

        with gr.Blocks(title="RGBTrack Camera Control") as app:
            gr.Markdown("# RGBTrack Camera Control Panel")
            gr.Markdown(f"**Config file**: `{self.config_path}`")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Camera Configuration")

                    # Device Serial Number
                    with gr.Row():
                        device_sn = gr.Dropdown(
                            label="Camera Serial Number",
                            value=self.config.device_sn,
                            choices=self._get_camera_choices(),
                            allow_custom_value=True,
                            info="Select camera by serial number"
                        )
                        refresh_cameras_btn = gr.Button("🔄", scale=0, min_width=50)

                    # Exposure
                    exposure_slider = gr.Slider(
                        minimum=self.config.exposure_min,
                        maximum=self.config.exposure_max,
                        value=self.config.exposure_time_ms,
                        step=1,
                        label=f"Exposure Time (ms)",
                        info=f"Range: {self.config.exposure_min}-{self.config.exposure_max}ms"
                    )

                    # Gain
                    gain_slider = gr.Slider(
                        minimum=self.config.gain_min,
                        maximum=self.config.gain_max,
                        value=self.config.gain,
                        step=1,
                        label="Gain",
                        info=f"Range: {self.config.gain_min}-{self.config.gain_max}"
                    )

                    # Gamma
                    gamma_slider = gr.Slider(
                        minimum=self.config.gamma_min,
                        maximum=self.config.gamma_max,
                        value=self.config.gamma,
                        step=0.1,
                        label="Gamma",
                        info=f"Range: {self.config.gamma_min}-{self.config.gamma_max}"
                    )

                    gr.Markdown("### White Balance")

                    # White balance mode
                    with gr.Row():
                        wb_mode_radio = gr.Radio(
                            choices=["auto", "manual"],
                            value=self.config.wb_mode,
                            label="White Balance Mode",
                            info="Auto mode uses camera's automatic white balance",
                            scale=3
                        )
                        with gr.Column(scale=1):
                            gr.Markdown("")
                            wb_calibrate_btn = gr.Button("Calibrate WB", variant="secondary", size="sm")

                    gr.Markdown("#### Manual RGB Balance")
                    gr.Markdown("Only active when white balance mode is set to 'manual'")

                    # RGB Balance
                    red_slider = gr.Slider(
                        minimum=self.config.balance_min,
                        maximum=self.config.balance_max,
                        value=self.config.red_balance,
                        step=0.01,
                        label="Red Balance",
                        info=f"Range: {self.config.balance_min}-{self.config.balance_max}",
                        interactive=(self.config.wb_mode == "manual")
                    )

                    green_slider = gr.Slider(
                        minimum=self.config.balance_min,
                        maximum=self.config.balance_max,
                        value=self.config.green_balance,
                        step=0.01,
                        label="Green Balance",
                        info=f"Range: {self.config.balance_min}-{self.config.balance_max}",
                        interactive=(self.config.wb_mode == "manual")
                    )

                    blue_slider = gr.Slider(
                        minimum=self.config.balance_min,
                        maximum=self.config.balance_max,
                        value=self.config.blue_balance,
                        step=0.01,
                        label="Blue Balance",
                        info=f"Range: {self.config.balance_min}-{self.config.balance_max}",
                        interactive=(self.config.wb_mode == "manual")
                    )

                    # Mode
                    mode_radio = gr.Radio(
                        choices=["normal", "high_speed"],
                        value=self.config.mode,
                        label="Camera Mode",
                        info="High speed mode may reduce quality for faster frame rate"
                    )

                    gr.Markdown("### Distortion Parameters")
                    gr.Markdown("Camera lens distortion coefficients (k1, k2, p1, p2, k3)")

                    distortion_k1 = gr.Number(
                        label="k1 (Radial Distortion)",
                        value=self.config.distortion_k1,
                        precision=6,
                        info=f"Range: {self.config.distortion_min} to {self.config.distortion_max}"
                    )

                    distortion_k2 = gr.Number(
                        label="k2 (Radial Distortion)",
                        value=self.config.distortion_k2,
                        precision=6,
                        info=f"Range: {self.config.distortion_min} to {self.config.distortion_max}"
                    )

                    distortion_p1 = gr.Number(
                        label="p1 (Tangential Distortion)",
                        value=self.config.distortion_p1,
                        precision=6,
                        info=f"Range: {self.config.distortion_min} to {self.config.distortion_max}"
                    )

                    distortion_p2 = gr.Number(
                        label="p2 (Tangential Distortion)",
                        value=self.config.distortion_p2,
                        precision=6,
                        info=f"Range: {self.config.distortion_min} to {self.config.distortion_max}"
                    )

                    distortion_k3 = gr.Number(
                        label="k3 (Radial Distortion)",
                        value=self.config.distortion_k3,
                        precision=6,
                        info=f"Range: {self.config.distortion_min} to {self.config.distortion_max}"
                    )

                    # Apply button
                    apply_btn = gr.Button("Apply Configuration", variant="primary")
                    config_status = gr.Textbox(label="Configuration Status", interactive=False)

                with gr.Column(scale=2):
                    gr.Markdown("## Camera Preview & Control")

                    # Preview image
                    preview_image = gr.Image(
                        label="Camera Preview",
                        type="numpy",
                        height=480
                    )

                    with gr.Row():
                        # Preview button
                        preview_btn = gr.Button("Take Preview", size="lg")

                        # Reset camera button
                        reset_camera_btn = gr.Button("Reset Camera", size="lg", variant="secondary")

                        # Detection enable toggle
                        detection_toggle = gr.Checkbox(
                            label="Enable Detection",
                            value=True,
                            info="Show detection results on preview"
                        )

                        # Live mode toggle
                        live_toggle = gr.Checkbox(
                            label="Live Mode",
                            value=False,
                            info="Enable to start continuous capture and detection"
                        )

                    # Status display
                    with gr.Row():
                        with gr.Column():
                            status_text = gr.Textbox(
                                label="System Status",
                                value="Ready",
                                interactive=False
                            )
                        with gr.Column():
                            fps_text = gr.Textbox(
                                label="Frame Rate (FPS)",
                                value="0.0",
                                interactive=False
                            )

            # Event handlers
            def update_config(dev_sn, exp, gain, gamma, wb_mode, red, green, blue, mode, k1, k2, p1, p2, k3):
                """Update camera configuration"""
                try:
                    self.config.device_sn = str(dev_sn)
                    self.config.exposure_time_ms = int(exp)
                    self.config.gain = float(gain)
                    self.config.gamma = float(gamma)
                    self.config.wb_mode = wb_mode  # pyright: ignore[reportAttributeAccessIssue]
                    self.config.red_balance = float(red)
                    self.config.green_balance = float(green)
                    self.config.blue_balance = float(blue)
                    self.config.mode = mode
                    self.config.distortion_k1 = float(k1)
                    self.config.distortion_k2 = float(k2)
                    self.config.distortion_p1 = float(p1)
                    self.config.distortion_p2 = float(p2)
                    self.config.distortion_k3 = float(k3)

                    # Notify parent
                    self.on_config_update(self.config)

                    # Save configuration to disk
                    if self.on_save_config:
                        self.on_save_config()

                    return f"✓ Configuration updated and saved to {self.config_path}"
                except Exception as e:
                    logger.error(f"Failed to update config: {e}")
                    return f"✗ Error: {str(e)}"

            def take_preview(detection_enabled):
                """Capture a single preview frame"""
                try:
                    frame = self.on_preview(detection_enabled)
                    if frame is not None:
                        return frame # assume the frame is a RGB image
                    else:
                        # Return a placeholder
                        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(
                            placeholder,
                            "No camera available",
                            (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2
                        )
                        return placeholder
                except Exception as e:
                    logger.error(f"Preview failed: {e}")
                    return None

            def refresh_cameras():
                """Refresh camera list"""
                return gr.Dropdown(choices=self._get_camera_choices())

            def reset_camera():
                """Reset/reopen the camera"""
                try:
                    result = self.on_camera_reset()
                    return result
                except Exception as e:
                    logger.error(f"Camera reset failed: {e}")
                    return f"✗ Error: {str(e)}"

            def toggle_wb_mode(mode):
                """Toggle RGB sliders based on white balance mode"""
                is_manual = (mode == "manual")
                return [
                    gr.Slider(interactive=is_manual),  # red_slider
                    gr.Slider(interactive=is_manual),  # green_slider
                    gr.Slider(interactive=is_manual),  # blue_slider
                ]

            def toggle_live(is_live):
                """Toggle live mode"""
                try:
                    self.is_live = is_live
                    self.on_live_toggle(is_live)

                    if is_live:
                        return "Live mode: ACTIVE"
                    else:
                        return "Live mode: INACTIVE"
                except Exception as e:
                    logger.error(f"Live toggle failed: {e}")
                    return f"Error: {str(e)}"
            
            def calibrate_wb():
                """Perform white balance calibration and update UI with new RGB balance values"""
                try:
                    # Call the calibration callback which returns (message, red, green, blue)
                    status_msg, red, green, blue = self.on_wb_calibrate()
                    
                    # Update internal config
                    self.config.red_balance = red
                    self.config.green_balance = green
                    self.config.blue_balance = blue
                    
                    # Return status message and updated RGB slider values
                    return status_msg, red, green, blue
                except Exception as e:
                    logger.error(f"WB calibration failed: {e}")
                    # Return error message and current values
                    return f"✗ Error: {str(e)}", self.config.red_balance, self.config.green_balance, self.config.blue_balance

            # Connect events
            apply_btn.click(
                fn=update_config,
                inputs=[
                    device_sn,
                    exposure_slider,
                    gain_slider,
                    gamma_slider,
                    wb_mode_radio,
                    red_slider,
                    green_slider,
                    blue_slider,
                    mode_radio,
                    distortion_k1,
                    distortion_k2,
                    distortion_p1,
                    distortion_p2,
                    distortion_k3
                ],
                outputs=config_status
            )

            preview_btn.click(
                fn=take_preview,
                inputs=detection_toggle,
                outputs=preview_image
            )

            reset_camera_btn.click(
                fn=reset_camera,
                outputs=status_text
            )

            refresh_cameras_btn.click(
                fn=refresh_cameras,
                outputs=device_sn
            )

            wb_mode_radio.change(
                fn=toggle_wb_mode,
                inputs=wb_mode_radio,
                outputs=[red_slider, green_slider, blue_slider]
            )

            live_toggle.change(
                fn=toggle_live,
                inputs=live_toggle,
                outputs=status_text
            )

            wb_calibrate_btn.click(
                fn=calibrate_wb,
                outputs=[config_status, red_slider, green_slider, blue_slider]
            )

        self.app = app
        return app

    def launch(self, share: bool = False):
        """
        Launch the Gradio interface.

        Args:
            share: Whether to create a public link
        """
        if self.app is None:
            self.create_interface()

        logger.info(f"Launching WebUI on {self.host}:{self.port}")

        try:
            self.app.launch(  # pyright: ignore[reportOptionalMemberAccess]
                server_name=self.host,
                server_port=self.port,
                share=share,
                prevent_thread_lock=True  # Important for running in a thread
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
