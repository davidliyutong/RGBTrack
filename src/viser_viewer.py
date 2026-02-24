"""Viser 3D viewer for visualizing camera poses and object 6D poses in real-time"""

import argparse
import logging
import signal
import sys
import time
from typing import Optional

import cv2
import numpy as np
import viser
import viser.transforms as vtf

from .config import ZMQConfig
from .zmq_subscriber import ZMQSubscriber

logger = logging.getLogger(__name__)


class ViserViewer:
    """
    Real-time 3D viewer using Viser for visualizing:
    - Camera pose (extrinsics) with frustum
    - Target 6D pose
    - Detection results from ZMQ
    """

    def __init__(
        self,
        zmq_config: ZMQConfig,
        viser_host: str = "0.0.0.0",
        viser_port: int = 8080,
        frustum_scale: float = 0.3,
        update_rate_hz: float = 30.0,
        invert_camera_pose: bool = False,
    ):
        """
        Initialize Viser viewer.

        Args:
            zmq_config: ZMQ configuration for subscribing to detection results
            viser_host: Viser server host
            viser_port: Viser server port
            frustum_scale: Scale factor for camera frustum visualization
            update_rate_hz: Update rate for visualization (Hz)
        """
        self.zmq_config = zmq_config
        self.viser_host = viser_host
        self.viser_port = viser_port
        self.frustum_scale = frustum_scale
        self.update_interval = 1.0 / update_rate_hz
        self.invert_camera_pose = invert_camera_pose

        # Viser server and handles
        self.server: Optional[viser.ViserServer] = None
        self.camera_frustum: Optional[viser.CameraFrustumHandle] = None
        self.target_frame: Optional[viser.FrameHandle] = None
        self.target_mesh: Optional[viser.MeshHandle] = None

        # ZMQ subscriber
        self.subscriber: Optional[ZMQSubscriber] = None

        # State
        self.running = False
        self.message_count = 0
        self.last_update_time = 0.0

        # Latest data
        self.latest_camera_pose: Optional[np.ndarray] = None
        self.latest_target_pose: Optional[np.ndarray] = None
        self.latest_frame_shape: Optional[tuple[int, ...]] = None
        self.latest_camera_intrinsics: Optional[np.ndarray] = None
        self.latest_preview_rgb: Optional[np.ndarray] = None

    def start(self):
        """Start the Viser viewer"""
        self._print_header()

        try:
            # Initialize Viser server
            logger.info(f"Starting Viser server on {self.viser_host}:{self.viser_port}")
            self.server = viser.ViserServer(host=self.viser_host, port=self.viser_port)
            logger.info(f"✓ Viser server started at http://{self.viser_host}:{self.viser_port}")

            # Setup scene
            self._setup_scene()

            # Connect to ZMQ publisher
            logger.info(f"Connecting to ZMQ publisher at {self.zmq_config.address}")
            self.subscriber = ZMQSubscriber(self.zmq_config.address)
            self.subscriber.connect()
            logger.info("✓ Connected to ZMQ publisher")

            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            self.running = True
            print("\nViewer is running. Open your browser to view the visualization.")
            print("Press Ctrl+C to stop.\n")

            # Main loop
            self._run_loop()

        except Exception as e:
            logger.error(f"Failed to start viewer: {e}", exc_info=True)
            raise
        finally:
            self._shutdown()

    def _setup_scene(self):
        """Setup the initial 3D scene"""
        if self.server is None:
            return

        # Add world coordinate frame (origin)
        self.server.add_frame( # type: ignore
            "/world",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            axes_length=0.5,
            axes_radius=0.01,
        )

        # Add grid
        self.server.add_grid( # type: ignore
            "/world/grid",
            width=10.0,
            height=10.0,
            width_segments=20,
            height_segments=20,
            plane="xz",  # Grid on XZ plane (horizontal)
            cell_color=(200, 200, 200),
            cell_thickness=0.5,
        )

        # Add camera frustum (will be updated with actual pose)
        self.camera_frustum = self.server.add_camera_frustum( # type: ignore
            "/world/camera",
            fov=float(np.deg2rad(60.0)),  # radians, updated from intrinsics
            aspect=16.0 / 9.0,  # Will be updated from image size
            scale=self.frustum_scale,
            color=(0, 255, 0),  # Green
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
        )

        # Add target frame (will be updated with actual pose)
        self.target_frame = self.server.add_frame( # type: ignore
            "/world/target",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            axes_length=0.2,
            axes_radius=0.01,
        )

        # Add a simple box mesh for the target object
        # This will be replaced with the actual mesh if provided
        self._add_target_mesh()

        logger.info("✓ Scene setup complete")

    def _add_target_mesh(self):
        """Add a simple box mesh to represent the target object"""
        if self.server is None or self.target_frame is None:
            return

        # Create a simple box mesh
        vertices = np.array([
            [-0.05, -0.05, -0.05],
            [0.05, -0.05, -0.05],
            [0.05, 0.05, -0.05],
            [-0.05, 0.05, -0.05],
            [-0.05, -0.05, 0.05],
            [0.05, -0.05, 0.05],
            [0.05, 0.05, 0.05],
            [-0.05, 0.05, 0.05],
        ], dtype=np.float32)

        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # front
            [2, 3, 7], [2, 7, 6],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 2, 6], [1, 6, 5],  # right
        ], dtype=np.uint32)

        self.target_mesh = self.server.add_mesh_simple( # type: ignore
            "/world/target/mesh",
            vertices=vertices,
            faces=faces,
            color=(255, 0, 0),  # Red
            wireframe=False,
        )

    def _run_loop(self):
        """Main visualization loop"""
        while self.running:
            try:
                # Receive detection result from ZMQ
                result = self.subscriber.receive(timeout_ms=100) if self.subscriber else None

                if result is not None:
                    self.message_count += 1
                    self._process_result(result)

                # Limit update rate
                current_time = time.time()
                if current_time - self.last_update_time >= self.update_interval:
                    self._update_visualization()
                    self.last_update_time = current_time

                # Small sleep to prevent busy waiting
                time.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in visualization loop: {e}", exc_info=True)
                time.sleep(0.1)

    def _process_result(self, result: dict):
        """
        Process detection result and extract pose information.

        Expected result structure:
        {
            'timestamp': float,
            'frame_id': int,
            'pose': list (4x4 transformation matrix),  # Target pose in camera frame
            'camera_pose': list (4x4 transformation matrix),  # Camera pose in world frame (optional)
            ...
        }
        """
        try:
            # Extract target pose (object pose in camera frame)
            if 'pose' in result:
                pose_list = result['pose']
                self.latest_target_pose = np.array(pose_list).reshape(4, 4)

            # Extract camera pose (camera extrinsics in world frame)
            # If not provided, assume camera is at origin
            if 'camera_pose' in result:
                camera_pose_list = result['camera_pose']
                camera_pose = np.array(camera_pose_list).reshape(4, 4)
                if self.invert_camera_pose:
                    try:
                        camera_pose = np.linalg.inv(camera_pose)
                    except Exception as inv_error:
                        logger.warning(f"Failed to invert camera pose: {inv_error}")
                self.latest_camera_pose = camera_pose
            else:
                # Default: camera at origin looking along +Z
                self.latest_camera_pose = np.eye(4)

            if 'frame_shape' in result:
                frame_shape = result['frame_shape']
                if isinstance(frame_shape, (list, tuple)) and len(frame_shape) >= 2:
                    self.latest_frame_shape = tuple(int(v) for v in frame_shape)

            if 'camera_intrinsics' in result:
                try:
                    self.latest_camera_intrinsics = np.array(result['camera_intrinsics'], dtype=np.float64).reshape(3, 3)
                except Exception as e:
                    logger.debug(f"Invalid camera_intrinsics payload: {e}")

            preview_bytes = result.get('preview_jpeg', None)
            if isinstance(preview_bytes, (bytes, bytearray)) and len(preview_bytes) > 0:
                arr = np.frombuffer(preview_bytes, dtype=np.uint8)
                preview_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if preview_bgr is not None:
                    self.latest_preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
            elif 'mask_png' in result and self.latest_preview_rgb is None:
                mask_bytes = result.get('mask_png', None)
                if isinstance(mask_bytes, (bytes, bytearray)) and len(mask_bytes) > 0:
                    arr = np.frombuffer(mask_bytes, dtype=np.uint8)
                    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        self.latest_preview_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

            if self.message_count % 30 == 0:
                logger.debug(f"Processed {self.message_count} messages")

        except Exception as e:
            logger.error(f"Error processing result: {e}", exc_info=True)

    def _update_visualization(self):
        """Update the 3D visualization with latest poses"""
        if self.server is None:
            return

        try:
            # Update camera frustum
            if self.latest_camera_pose is not None and self.camera_frustum is not None:
                # Convert 4x4 pose matrix to position and quaternion
                position = self.latest_camera_pose[:3, 3]
                rotation_matrix = self.latest_camera_pose[:3, :3]

                if self.latest_camera_intrinsics is not None and self.latest_frame_shape is not None:
                    try:
                        h = float(self.latest_frame_shape[0])
                        w = float(self.latest_frame_shape[1])
                        fy = float(self.latest_camera_intrinsics[1, 1])
                        if h > 1e-6 and w > 1e-6 and fy > 1e-6:
                            self.camera_frustum.aspect = w / h
                            self.camera_frustum.fov = float(2.0 * np.arctan(h / (2.0 * fy)))
                    except Exception as e:
                        logger.debug(f"Failed to update frustum intrinsics: {e}")

                if self.latest_preview_rgb is not None:
                    try:
                        self.camera_frustum.image = self.latest_preview_rgb
                    except Exception as e:
                        logger.debug(f"Failed to update frustum image: {e}")

                # Convert rotation matrix to quaternion (wxyz format for viser)
                wxyz = vtf.SO3.from_matrix(rotation_matrix).wxyz

                self.camera_frustum.wxyz = wxyz
                self.camera_frustum.position = position

            # Update target object pose
            if self.latest_target_pose is not None and self.target_frame is not None:
                # If camera pose is provided, transform target from camera frame to world frame
                if self.latest_camera_pose is not None:
                    # target_in_world = camera_pose @ target_in_camera
                    target_world_pose = self.latest_camera_pose @ self.latest_target_pose
                else:
                    target_world_pose = self.latest_target_pose

                position = target_world_pose[:3, 3]
                rotation_matrix = target_world_pose[:3, :3]
                wxyz = vtf.SO3.from_matrix(rotation_matrix).wxyz # type: ignore

                self.target_frame.wxyz = wxyz
                self.target_frame.position = position

        except Exception as e:
            logger.error(f"Error updating visualization: {e}", exc_info=True)

    def _signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        print("\n\nReceived shutdown signal...")
        self.running = False

    def _shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down viewer...")

        if self.subscriber is not None:
            self.subscriber.disconnect()

        # Viser server will be closed automatically

        # Display statistics
        logger.info(f"Total messages received: {self.message_count}")
        logger.info("✓ Viewer stopped")

    def _print_header(self):
        """Print viewer header"""
        print("=" * 70)
        print("RGBTrack Viser 3D Viewer")
        print("=" * 70)
        print(f"Viser server: http://{self.viser_host}:{self.viser_port}")
        print(f"ZMQ address:  {self.zmq_config.address}")
        print(f"Invert camera pose: {self.invert_camera_pose}")
        print("-" * 70)


def main():
    """Command-line interface for the Viser viewer"""
    parser = argparse.ArgumentParser(
        description="3D viewer for RGBTrack detection results using Viser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default settings (IPC transport)
  %(prog)s
  
  # Connect to TCP publisher
  %(prog)s --transport tcp --host localhost --port 5555
  
  # Custom Viser server settings
  %(prog)s --viser-host 0.0.0.0 --viser-port 8080
  
  # Adjust camera frustum size
  %(prog)s --frustum-scale 0.5

    # Invert camera pose if incoming extrinsics are world->camera
    %(prog)s --invert-camera-pose
        """
    )

    # ZMQ settings
    parser.add_argument(
        '--transport', '-t',
        choices=['ipc', 'tcp'],
        default='ipc',
        help='Transport protocol for ZMQ (default: ipc)'
    )

    parser.add_argument(
        '--socket', '-s',
        default='/tmp/rgbtrack.sock',
        help='Unix socket path for IPC transport (default: /tmp/rgbtrack.sock)'
    )

    parser.add_argument(
        '--host', '-H',
        default='localhost',
        help='Host for TCP transport (default: localhost)'
    )

    parser.add_argument(
        '--port', '-p',
        type=int,
        default=5555,
        help='Port for TCP transport (default: 5555)'
    )

    # Viser settings
    parser.add_argument(
        '--viser-host',
        default='0.0.0.0',
        help='Viser server host (default: 0.0.0.0)'
    )

    parser.add_argument(
        '--viser-port',
        type=int,
        default=8080,
        help='Viser server port (default: 8080)'
    )

    parser.add_argument(
        '--frustum-scale',
        type=float,
        default=0.3,
        help='Scale factor for camera frustum visualization (default: 0.3)'
    )

    parser.add_argument(
        '--update-rate',
        type=float,
        default=30.0,
        help='Visualization update rate in Hz (default: 30.0)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--invert-camera-pose',
        action='store_true',
        help='Invert incoming camera_pose (use when data is world->camera)'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create ZMQ configuration
    zmq_config = ZMQConfig(
        transport=args.transport,
        socket_path=args.socket,
        host=args.host,
        port=args.port
    )

    # Create and start viewer
    try:
        viewer = ViserViewer(
            zmq_config=zmq_config,
            viser_host=args.viser_host,
            viser_port=args.viser_port,
            frustum_scale=args.frustum_scale,
            update_rate_hz=args.update_rate,
            invert_camera_pose=args.invert_camera_pose,
        )
        viewer.start()

    except KeyboardInterrupt:
        print("\n✓ Stopped by user")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
