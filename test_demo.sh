#!/usr/bin/env bash

set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/3] Extract frames from video..."
python "$CODE_DIR/extract_images_from_video.py" \
  --video="./demo_data/test/video/Lark20260223-223627.mp4"

echo "[2/3] Generate first-frame mask with SAM2 + CLIP..."
python "$CODE_DIR/generate_image_marker.py" --image ./demo_data/test/rgb --sam2_checkpoint $CODE_DIR/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt

echo "[3/3] Run pose estimation without depth..."
python "$CODE_DIR/test_demo_without_depth.py"

echo "Done."
 