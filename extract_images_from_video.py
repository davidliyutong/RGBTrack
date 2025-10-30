import os
import shutil
import argparse
import cv2
from tqdm import tqdm


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def extract_frames(video_path: str, out_dir: str, prefix: str = "", start_index: int = 1) -> int:
    print("Video path: ", video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    rgb_dir = os.path.join(out_dir, "rgb")
    # Clear previous images if exist
    if os.path.isdir(rgb_dir):
        shutil.rmtree(rgb_dir)
    ensure_dir(rgb_dir)

    idx = start_index
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total, desc="Extracting frames") if tqdm is not None else None
    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out_path = os.path.join(rgb_dir, f"{prefix}{idx:07d}.png")
        cv2.imwrite(out_path, frame_rgb[:, :, ::-1])  # save as BGR for cv2
        idx += 1
        if pbar is not None:
            pbar.update(1)

    cap.release()
    if pbar is not None:
        pbar.close()
    return idx - start_index


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video to rgb/ images.")
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument("--video", type=str, default=f"/home/xxz/code/toss-reorient/RGBTrack/demo_data/test/video/far_static_beachball.mp4", help="Path to input .mov video")
    # parser.add_argument("--video", type=str, default=f"/home/xxz/code/toss-reorient/RGBTrack/demo_data/calib_images/chessboard_f1_0.mp4", help="Path to input .mov video")
    parser.add_argument("--out_dir", type=str, default=f"{code_dir}/demo_data/test", help="Output root directory; rgb/ will be created under it")
    parser.add_argument("--prefix", default="", type=str, help="Filename prefix for frames (optional)")
    parser.add_argument("--start_index", default=1, type=int, help="Starting index for frame filenames")
    args = parser.parse_args()

    out_dir = os.path.expanduser(args.out_dir)
    num = extract_frames(args.video, out_dir, args.prefix, args.start_index)
    print(f"Saved {num} frames to {os.path.join(out_dir, 'rgb')}")


if __name__ == "__main__":
    main()


