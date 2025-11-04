import os
import sys
import time
import argparse
from typing import Optional, Tuple

import numpy as np
import cv2

# Camera SDK
try:
    import mvsdk  # Requires MindVision SDK
except Exception:
    # Fallback to bundled SDK under demo/python_demo/mvsdk.py
    _this_dir = os.path.dirname(os.path.realpath(__file__))
    _project_root = os.path.abspath(os.path.join(_this_dir, '..'))
    _mv_path = os.path.join(_project_root, 'demo', 'python_demo')
    if _mv_path not in sys.path:
        sys.path.insert(0, _mv_path)
    import importlib
    try:
        mvsdk = importlib.import_module('mvsdk')  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Failed to import mvsdk. Ensure MindVision SDK is installed or that '{_mv_path}' is on PYTHONPATH."
        ) from e

# Reuse RGBTrack internals (keep same import style as test_demo_without_depth)
from estimater import * # noqa: F401,F403
from tools import *      # noqa: F401,F403

# SAM2-based mask bootstrap
from generate_image_marker import (
    build_sam2_model,
    generate_masks_with_sam2,
    rank_masks_by_prompt,
)


def load_intrinsics(
    k_txt: Optional[str],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    if k_txt:
        K = np.loadtxt(k_txt)
        if K.shape != (3, 3):
            raise RuntimeError("--K-txt must contain a 3x3 matrix")
        return K
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K


class HighSpeedCamera:
    def __init__(self, exposure_ms: float, max_video_fps: int):
        self.hCamera = None
        self.cap = None
        self.pFrameBuffer = None
        self.frame_bytes = 0
        self.monoCamera = False

        # Enumerate and open
        DevList = mvsdk.CameraEnumerateDevice()
        if len(DevList) < 1:
            raise RuntimeError("No camera was found!")
        DevInfo = DevList[0]
        try:
            self.hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
        except mvsdk.CameraException as e:
            raise RuntimeError(f"CameraInit Failed({e.error_code}): {e.message}")

        self.cap = mvsdk.CameraGetCapability(self.hCamera)
        self.monoCamera = (self.cap.sIspCapacity.bMonoSensor != 0)
        if self.monoCamera:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # Continuous mode
        mvsdk.CameraSetTriggerMode(self.hCamera, 0)

        # Exposure and fps
        mvsdk.CameraSetAeState(self.hCamera, 0)
        mvsdk.CameraSetExposureTime(self.hCamera, float(exposure_ms) * 1000.0)
        safe_fps = int(min(max_video_fps, max(1, int(1000.0 / (exposure_ms * 1.2)))))
        self.safe_fps = safe_fps

        # Start streaming
        mvsdk.CameraPlay(self.hCamera)

        # White balance once if supported
        if (not self.monoCamera) and (self.cap.sIspCapacity.bWbOnce != 0):
            mvsdk.CameraSetOnceWB(self.hCamera)

        # Pre-allocate aligned buffer
        FrameBufferSize = (
            self.cap.sResolutionRange.iWidthMax
            * self.cap.sResolutionRange.iHeightMax
            * (1 if self.monoCamera else 3)
        )
        self.pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    def read(self) -> Tuple[bool, np.ndarray]:
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
            mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)

            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            channels = 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, channels))
            # Uniform to 640x480 for pipeline speed/consistency
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            if channels == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            return True, frame
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print(f"CameraGetImageBuffer failed({e.error_code}): {e.message}")
            return False, None

    def white_balance_once(self):
        if (not self.monoCamera) and (self.cap.sIspCapacity.bWbOnce != 0):
            mvsdk.CameraSetOnceWB(self.hCamera)

    def release(self):
        try:
            if self.hCamera is not None:
                mvsdk.CameraUnInit(self.hCamera)
        finally:
            if self.pFrameBuffer is not None:
                mvsdk.CameraAlignFree(self.pFrameBuffer)


def main():
    parser = argparse.ArgumentParser(description="Real-time RGBTrack inference from high-speed camera with first-frame SAM2 mask bootstrap")
    code_dir = os.path.dirname(os.path.realpath(__file__))

    # Camera
    parser.add_argument('--exposure-ms', type=float, default=30.0, help='Exposure time in milliseconds')
    parser.add_argument('--max-video-fps', type=int, default=120, help='Upper limit for video FPS')

    # Intrinsics
    parser.add_argument('--K-txt', type=str, default='/home/xxz/code/toss-reorient/RGBTrack/demo_data/test/cam_K.txt', help='Path to 3x3 intrinsics txt (overrides fx/fy/cx/cy)')
    parser.add_argument('--fx', type=float, default=600.0)
    parser.add_argument('--fy', type=float, default=600.0)
    parser.add_argument('--cx', type=float, default=320.0)
    parser.add_argument('--cy', type=float, default=240.0)

    # Object/estimator
    parser.add_argument('--mesh_file', type=str, default=f"{code_dir}/demo_data/test/mesh/beachball_single.obj")
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=1)
    parser.add_argument('--mode', type=int, default=0, choices=[0, 1], help='0: no depth prior, 1: render last depth prior')
    parser.add_argument('--debug', type=int, default=1)

    # SAM2 + CLIP
    parser.add_argument('--prompt', type=str, default='beachball', help='Text prompt for target object')
    parser.add_argument('--sam2_repo', type=str, default=f"{code_dir}/segment-anything-2-real-time")
    parser.add_argument('--sam2_checkpoint', type=str, default='/home/xxz/code/toss-reorient/sam2.1_hiera_small.pt')
    parser.add_argument('--sam2_cfg', type=str, default='configs/sam2.1/sam2.1_hiera_s.yaml')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

    # Outputs
    parser.add_argument('--output_dir', type=str, default=f"{code_dir}/output_live")
    parser.add_argument('--save_video', action='store_true', default=True)
    parser.add_argument('--video_dir', type=str, default=f"{code_dir}/output_live")
    parser.add_argument('--video_name', type=str, default='pose_overlay_live.mp4')
    parser.add_argument('--max_steps', type=int, default=5000, help='Process at most N frames; 0 means unlimited')
    parser.add_argument('--mask_path', type=str, default=f"{code_dir}/output_live/masks/first_mask.png", help='Path to binary mask file to use for initial pose; if missing, it will be generated and saved here')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build intrinsics
    K = load_intrinsics(args.K_txt if args.K_txt else None, args.fx, args.fy, args.cx, args.cy)

    # Start camera
    cam = HighSpeedCamera(args.exposure_ms, args.max_video_fps)
    print(f"[INFO] Exposure: {args.exposure_ms:.2f} ms  →  Recommended video_fps: {cam.safe_fps} fps")

    # Prepare estimator (match style of test_demo_without_depth)
    mesh = trimesh.load(args.mesh_file)  # trimesh provided via tools/estimater imports
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()  # provided via estimater/tools namespace
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=args.output_dir,
        debug=args.debug,
        glctx=glctx,
    )

    # Build SAM2 model once
    cfg_rel = args.sam2_cfg
    ckpt_name = os.path.basename(args.sam2_checkpoint).lower()
    if 'sam2.1' in ckpt_name:
        if 'tiny' in ckpt_name:
            cfg_rel = 'configs/sam2.1/sam2.1_hiera_t.yaml'
        elif 'small' in ckpt_name:
            cfg_rel = 'configs/sam2.1/sam2.1_hiera_s.yaml'
        elif 'base_plus' in ckpt_name or 'b_plus' in ckpt_name or 'b+.pt' in ckpt_name or '_b+.' in ckpt_name:
            cfg_rel = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
        elif 'large' in ckpt_name or 'hiera_l' in ckpt_name:
            cfg_rel = 'configs/sam2.1/sam2.1_hiera_l.yaml'
    sam2_model = build_sam2_model(args.sam2_repo, args.sam2_checkpoint, args.device, cfg_rel)

    # Video writer
    video_writer = None
    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)
        video_path = os.path.join(args.video_dir, args.video_name if args.video_name.endswith('.mp4') else args.video_name + '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, cam.safe_fps, (640, 480))
        print(f"[INFO] Saving video to: {video_path}")

    # Pose log
    poses_txt_path = os.path.join(args.output_dir, 'poses_live.txt')
    open(poses_txt_path, 'w').close()

    have_initial_pose = False
    last_mask = None
    pose = None
    step = 0

    print("[INFO] Press 'q' to quit, 'w' to white-balance-once")
    try:
        while True:
            ok, frame_bgr = cam.read()
            if not ok:
                continue

            t1 = time.time()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            step += 1
            if not have_initial_pose:
                # Ensure directory for mask_path exists
                mask_dir = os.path.dirname(args.mask_path) if args.mask_path else os.path.join(args.output_dir, 'masks')
                if mask_dir:
                    os.makedirs(mask_dir, exist_ok=True)

                # If a mask file already exists, use it; otherwise generate and save
                mask_path_to_use = args.mask_path if args.mask_path else os.path.join(mask_dir, 'first_mask.png')
                if not os.path.isfile(mask_path_to_use):
                    # First-frame mask via SAM2 + CLIP
                    masks = generate_masks_with_sam2(frame_rgb, sam2_model)
                    best_mask_u8, score = rank_masks_by_prompt(frame_rgb, masks, args.prompt, args.device)
                    try:
                        cv2.imwrite(mask_path_to_use, best_mask_u8)
                        # Also write visualization alongside
                        mask_bgr = cv2.cvtColor(best_mask_u8, cv2.COLOR_GRAY2BGR)
                        h1, w1 = frame_bgr.shape[:2]
                        h2, w2 = mask_bgr.shape[:2]
                        if (h1, w1) != (h2, w2):
                            mask_bgr = cv2.resize(mask_bgr, (w1, h1), interpolation=cv2.INTER_NEAREST)
                        vis_first = np.hstack([frame_bgr, mask_bgr])
                        vis_mask_path = os.path.join(mask_dir, 'first_mask_vis.png')
                        cv2.imwrite(vis_mask_path, vis_first)
                        print(f"[INFO] Saved first mask to: {mask_path_to_use}")
                        print(f"[INFO] Saved first mask visualization to: {vis_mask_path}")
                    except Exception as e:
                        print(f"[WARN] Failed to save first-frame mask: {e}")

                # Load mask from disk for initial registration
                mask_gray = cv2.imread(mask_path_to_use, cv2.IMREAD_GRAYSCALE)
                if mask_gray is None:
                    raise RuntimeError(f"Failed to read mask from {mask_path_to_use}")
                if mask_gray.shape[:2] != frame_bgr.shape[:2]:
                    mask_gray = cv2.resize(mask_gray, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                last_mask = (mask_gray > 0)

                # Initial pose by binary-search-depth registration
                pose = binary_search_depth(est, mesh, frame_rgb, last_mask, K, debug=True)
                have_initial_pose = True
            else:
                if args.mode == 0:
                    last_depth = np.zeros_like(last_mask, dtype=np.uint8)
                else:
                    last_depth = render_cad_depth(pose, mesh, K)
                pose = est.track_one(rgb=frame_rgb, depth=last_depth, K=K, iteration=args.track_refine_iter)

            # Save pose to log (timestamped)
            ts = time.time()
            with open(poses_txt_path, 'a') as f:
                f.write(f"{ts}\n")
                np.savetxt(f, pose.reshape(4, 4))
                f.write("\n")

            # Overlay
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(K, img=frame_rgb.copy(), ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
            fps = int(1.0 / max(1e-6, (time.time() - t1)))
            vis = cv2.putText(vis, f"fps {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

            if video_writer is not None:
                video_writer.write(vis_bgr)

            # Stop on user input
            cv2.imshow('RGBTrack Live (q to quit, w WB once)', vis_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):
                cam.white_balance_once()

            # Stop automatically if max_steps reached
            if args.max_steps > 0 and step >= args.max_steps:
                print(f"[INFO] Reached max_steps={args.max_steps}; stopping...")
                break
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user (Ctrl+C); stopping gracefully...")
    finally:
        # Cleanup
        cam.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Ensure RGBTrack root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    main()


