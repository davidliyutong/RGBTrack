#!/usr/bin/env bash

set -euo pipefail

CODE_DIR="/home/xxz/code/toss-reorient/RGBTrack"

echo "[1/3] Extract frames from video..."
python "$CODE_DIR/extract_images_from_video.py" \
  --video="/home/xxz/code/toss-reorient/RGBTrack/demo_data/test/video/high_speed_camera_beachball.mp4"

echo "[2/3] Generate first-frame mask with SAM2 + CLIP..."
python "$CODE_DIR/generate_image_marker.py"

echo "[3/3] Run pose estimation without depth..."
python "$CODE_DIR/test_demo_without_depth.py"

echo "Done."
 