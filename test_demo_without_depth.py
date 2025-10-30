# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import time
import os
from estimater import *
from datareader import *
import argparse
from tools import *
import numpy as np
from tqdm import tqdm


SAVE_VIDEO=False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument(
        "--mesh_file",
        type=str,
        default=f"{code_dir}/demo_data/test/mesh/beachball_single.obj",
    )
    parser.add_argument(
        "--test_scene_dir", type=str, default=f"{code_dir}/demo_data/test"
    )
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=1)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--save_video", type=int, default=1)
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    # Print input sources
    print(f"Input mesh: {args.mesh_file}")
    print(f"Input sequence directory: {args.test_scene_dir}")
    print(f"  RGB frames from: {os.path.join(args.test_scene_dir, 'rgb')}")
    print(f"  Masks from: {os.path.join(args.test_scene_dir, 'masks')}")

    mesh = trimesh.load(args.mesh_file)

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(
        f"rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam"
    )

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=debug,
        glctx=glctx,
    )
    logging.info("estimator initializ ation done")

    reader = YcbineoatReader(
        video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf
    )
    # ensure output directory exists and open pose log file
    output_dir = os.path.join(code_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    poses_txt_path = os.path.join(output_dir, "poses.txt")
    video_path = os.path.join(output_dir, "pose_overlay.mp4")
    video_writer = None
    SAVE_VIDEO = bool(args.save_video)

    # Print planned outputs
    print(f"Outputs will be saved under: {output_dir}")
    print(f"  Pose log path: {poses_txt_path}")
    if SAVE_VIDEO:
        print(f"  Overlay video path: {video_path}")

    for i in tqdm(range(len(reader.color_files)), total=len(reader.color_files), desc="Processing images"):
        color = reader.get_color(i)
        if i == 0:
            mask = reader.get_mask(0).astype(bool)
            last_mask= mask
            t1=time.time()
            pose= binary_search_depth(est, mesh, color, mask, reader.K, debug=True)

            # pose = est.register_without_depth(
            #     K=reader.K,
            #     rgb=color,
            #     ob_mask=mask,
            #     iteration=args.est_refine_iter,
            # )
            logging.info(f"Initial pose:\n{pose}")
        
            t2=time.time()
        else:
            t1=time.time()
            if args.mode==0:
                last_depth = np.zeros_like(last_mask)
            elif args.mode==1:
                last_depth = render_cad_depth(pose, mesh, reader.K)
            pose = est.track_one(
                rgb=color, depth=last_depth, K=reader.K, iteration=args.track_refine_iter
            )
            t2=time.time()
        os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
        np.savetxt(f"{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt", pose.reshape(4, 4))
        # append pose to a single poses.txt in output directory
        with open(poses_txt_path, "a") as f:
            f.write(f"{reader.id_strs[i]}\n")
            np.savetxt(f, pose.reshape(4, 4))
            f.write("\n")

        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            color=cv2.putText(color, f"fps {int(1/(t2-t1))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            vis = draw_posed_3d_box(
                reader.K, img=color, ob_in_cam=center_pose, bbox=bbox
            )
            vis = draw_xyz_axis(
                color,
                ob_in_cam=center_pose,
                scale=0.1,
                K=reader.K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            # initialize video writer on first frame with correct size and codec
            if SAVE_VIDEO:
                if video_writer is None:
                    h, w = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, 27, (w, h))
                frame_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)

        if debug >= 2:
            os.makedirs(f"{debug_dir}/track_vis", exist_ok=True)
            imageio.imwrite(f"{debug_dir}/track_vis/{reader.id_strs[i]}.png", vis)
    if SAVE_VIDEO and video_writer is not None:
        video_writer.release()
    # Print summary of outputs
    print(f"Saved pose log: {poses_txt_path}")
    if SAVE_VIDEO:
        if os.path.isfile(video_path):
            print(f"Saved overlay video: {video_path}")
        else:
            print("Video saving was enabled, but no video file was written.")
