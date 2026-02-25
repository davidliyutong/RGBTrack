"""Viser-based 3D visualization web interface"""

import logging
import threading
from typing import Optional, Callable, Tuple
import time

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation

from .config import SystemConfig
from .detection import DetectionAlgorithm

logger = logging.getLogger(__name__)


class ViserWebUI:
    """
    Viser-based 3D visualization web interface.
    Shows camera, object mesh, tracking results, and provides user controls.
    """
    
    def __init__(
        self,
        config: SystemConfig,
        detection: DetectionAlgorithm,
        on_start_detection: Callable[[], bool],
        on_pause: Callable[[], None],
        on_resume: Callable[[], None],
        on_reset: Callable[[], None]
    ):
        """
        Initialize viser UI.
        
        Args:
            config: System configuration
            detection: Detection algorithm instance
            on_start_detection: Callback for start detection button
            on_pause: Callback for pause button
            on_resume: Callback for resume button
            on_reset: Callback for reset button
        """
        self.config = config
        self.detection = detection
        self.on_start_detection = on_start_detection
        self.on_pause = on_pause
        self.on_resume = on_resume
        self.on_reset = on_reset
        
        # Server and scene objects
        self.server: Optional[viser.ViserServer] = None
        self.object_mesh_handle: Optional[viser.MeshHandle] = None
        self.camera_frustum_handle: Optional[viser.CameraFrustumHandle] = None
        self.trajectory_handle: Optional[viser.PointCloudHandle] = None
        
        # GUI handles for updates
        self.status_label: Optional[viser.GuiMarkdownHandle] = None
        self.fps_label: Optional[viser.GuiMarkdownHandle] = None
        self.pose_label: Optional[viser.GuiMarkdownHandle] = None
        self.prompt_input: Optional[viser.GuiInputHandle[str]] = None
        self.nms_slider: Optional[viser.GuiSliderHandle] = None
        
        # Trajectory storage
        self._trajectory: list[np.ndarray] = []
        self._trajectory_lock = threading.Lock()
        
        # Update lock for thread-safe scene updates
        self._update_lock = threading.Lock()
    
    def launch(self):
        """Start viser server (blocking call)"""
        try:
            logger.info(f"Starting viser server on {self.config.viser.host}:{self.config.viser.port}")
            
            # Create viser server
            self.server = viser.ViserServer(
                host=self.config.viser.host,
                port=self.config.viser.port
            )
            
            # Configure scene
            self.server.scene.set_up_direction("+y")
            
            # Build scene
            self._build_scene()
            
            # Create GUI panel
            self._create_gui_panel()
            
            logger.info("✓ Viser UI launched successfully")
            logger.info(f"  Open http://localhost:{self.config.viser.port} in your browser")
            
            # Keep server running (blocking)
            while True:
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Failed to launch viser UI: {e}", exc_info=True)
            raise
    
    def _build_scene(self):
        """Build 3D scene with coordinate frames, camera, and object mesh"""
        assert self.server is not None
        
        # Add world coordinate frame
        self.server.scene.add_frame(
            name="/world",
            show_axes=True,
            axes_length=0.5
        )
        
        # Add camera frame at origin
        self.server.scene.add_frame(
            name="/camera",
            show_axes=True,
            axes_length=0.3
        )
        
        # Add camera frustum (pyramid)
        if self.config.viser.show_camera_cone:
            self.camera_frustum_handle = self._create_camera_frustum()
        
        # Load and add object mesh
        try:
            logger.info(f"Loading object mesh from {self.config.detection.mesh_path}")
            self.object_mesh = trimesh.load(self.config.detection.mesh_path)
            
            # Add mesh to scene
            self.object_mesh_handle = self.server.scene.add_mesh_trimesh(
                name="/object",
                mesh=self.object_mesh,
            )
            logger.info("✓ Object mesh loaded and added to scene")
            
        except Exception as e:
            logger.error(f"Failed to load object mesh: {e}")
            # Create a simple box as placeholder
            box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
            self.object_mesh_handle = self.server.scene.add_mesh_trimesh(
                name="/object",
                mesh=box,
            )
    
    def _create_camera_frustum(self) -> viser.CameraFrustumHandle:
        """
        Create camera frustum visualization.
        
        Args:
            K: 3x3 camera intrinsic matrix
            scale: Scale factor for frustum size
        
        Returns:
            Handle for camera frustum
        """
        assert self.server is not None
        
        # Compute FOV from intrinsics
        # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        fx, fy = K[0, 0], K[1, 1]
        # Assume image width of 1 for normalization
        hfov = 2 * np.arctan(0.5 / fx)
        
        # Add camera frustum at origin pointing along -Z axis
        frustum_handle = self.server.scene.add_camera_frustum(
            name="/camera/frustum",
            fov=hfov,
            aspect=1.333,  # 4:3 aspect ratio (adjust based on your camera)
            scale=scale,
            color=(0, 0, 255),
        )
        
        return frustum_handle
    
    def _create_gui_panel(self):
        """Create GUI control panel"""
        assert self.server is not None
        
        # Use gui folder for organization
        with self.server.gui.add_folder("Controls"):
            # Detection prompt input
            self.prompt_input = self.server.gui.add_text(
                label="Detection Prompt",
                initial_value=self.config.detection.prompt,
                hint="Object name for CLIP"
            )
            
            @self.prompt_input.on_update
            def _(_):
                self.config.detection.prompt = self.prompt_input.value
                logger.info(f"Detection prompt updated: {self.config.detection.prompt}")
            
            # NMS threshold slider
            self.nms_slider = self.server.gui.add_slider(
                label="NMS Threshold",
                min=0.0,
                max=1.0,
                step=0.05,
                initial_value=self.config.detection.nms_threshold
            )
            
            @self.nms_slider.on_update
            def _(_):
                self.config.detection.nms_threshold = self.nms_slider.value
                logger.info(f"NMS threshold updated: {self.config.detection.nms_threshold}")
            
            # Control buttons
            start_btn = self.server.gui.add_button("🎯 Start Detection")
            
            @start_btn.on_click
            def _(_):
                success = self.on_start_detection()
                if self.status_label is not None:
                    if success:
                        self.status_label.content = "Status: **✓ DETECTING**"
                    else:
                        self.status_label.content = "Status: **✗ Failed** (already running?)"
            
            self.pause_btn = self.server.gui.add_button("⏸️ Pause")
            
            @self.pause_btn.on_click
            def _(_):
                self.on_pause()
                if self.status_label is not None:
                    self.status_label.content = "Status: **⏸️ PAUSED**"
            
            self.resume_btn = self.server.gui.add_button("▶️ Resume")
            
            @self.resume_btn.on_click
            def _(_):
                self.on_resume()
                if self.status_label is not None:
                    self.status_label.content = "Status: **▶️ TRACKING**"
            
            reset_btn = self.server.gui.add_button("🔄 Reset")
            
            @reset_btn.on_click
            def _(_):
                self.on_reset()
                if self.status_label is not None:
                    self.status_label.content = "Status: **🟢 IDLE**"
                # Clear trajectory
                with self._trajectory_lock:
                    self._trajectory.clear()
            
            # Status displays (using markdown)
            self.status_label = self.server.gui.add_markdown("Status: **🟢 IDLE**")
            self.fps_label = self.server.gui.add_markdown("FPS: 0.0")
            self.pose_label = self.server.gui.add_markdown("Pose: N/A")
    
    def update_visualization(
        self,
        pose: Optional[np.ndarray],
        mask: Optional[np.ndarray],
        frame_shape: Tuple[int, int, int],
        fps: float,
        status: str
    ):
        """
        Update scene with new tracking results (thread-safe).
        
        Args:
            pose: 4x4 pose matrix (object-to-camera), or None
            mask: Object mask (H, W), or None
            frame_shape: Camera frame shape (H, W, C)
            fps: Current FPS
            status: Status string (IDLE, DETECTING, TRACKING, PAUSED)
        """
        if self.server is None:
            return
        
        with self._update_lock:
            try:
                # Update status labels
                if self.fps_label is not None:
                    self.fps_label.content = f"FPS: {fps:.1f}"
                
                if self.pose_label is not None:
                    if pose is not None:
                        pos = pose[:3, 3]
                        rot = Rotation.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True)
                        self.pose_label.content = (
                            f"Pose: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m\\n"
                            f"Rot: [{rot[0]:.1f}°, {rot[1]:.1f}°, {rot[2]:.1f}°]"
                        )
                    else:
                        self.pose_label.content = "Pose: N/A"
                
                # Update object mesh pose
                if pose is not None and self.object_mesh_handle is not None:
                    # Decompose pose into position and quaternion
                    position = pose[:3, 3]
                    rotation = Rotation.from_matrix(pose[:3, :3])
                    # scipy returns quaternions in xyzw format, viser needs wxyz
                    quat_xyzw = rotation.as_quat()
                    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
                    
                    # Apply pose to mesh
                    self.object_mesh_handle.position = position
                    self.object_mesh_handle.wxyz = quat_wxyz
                
                # Update trajectory
                if self.config.viser.show_trajectory and pose is not None:
                    with self._trajectory_lock:
                        self._trajectory.append(pose[:3, 3].copy())
                        
                        # Limit trajectory length
                        max_length = self.config.viser.trajectory_length
                        if len(self._trajectory) > max_length:
                            self._trajectory.pop(0)
                        
                        # Update trajectory visualization
                        self._update_trajectory_visualization()
                
            except Exception as e:
                logger.error(f"Failed to update visualization: {e}", exc_info=True)
    
    def _update_trajectory_visualization(self):
        """Update trajectory line visualization"""
        assert self.server is not None
        
        if len(self._trajectory) < 1:
            return
        
        # Convert trajectory points to array
        points = np.array(self._trajectory)
        
        # Create point cloud for trajectory
        if self.trajectory_handle is not None:
            self.trajectory_handle.remove()
        
        # Add trajectory as spheres
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        colors[:, 0] = 255  # Red
        colors[:, 2] = 165  # Orange tint
        
        self.trajectory_handle = self.server.scene.add_point_cloud(
            name="/trajectory",
            points=points.astype(np.float32),
            colors=colors,
            point_size=0.02,
        )
