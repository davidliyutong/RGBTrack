import os
import sys
import argparse
from typing import List, Tuple

import cv2
import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_sam2_model(sam2_repo: str, sam2_checkpoint: str, device: str, cfg_rel: str):
    if not os.path.isdir(sam2_repo):
        raise RuntimeError(f"SAM2 repo not found: {sam2_repo}")
    sys.path.insert(0, sam2_repo)
    from sam2.build_sam import build_sam2  # type: ignore

    cwd = os.getcwd()
    try:
        os.chdir(sam2_repo)
        model = build_sam2(config_file=cfg_rel, ckpt_path=sam2_checkpoint, device=device)
    finally:
        os.chdir(cwd)
    return model


def generate_masks_with_sam2(image_rgb: np.ndarray, model) -> List[dict]:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
    mask_generator = SAM2AutomaticMaskGenerator(model)
    masks = mask_generator.generate(image_rgb)
    return masks


def load_clip(device: str):
    try:
        import clip  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: 'clip'. Install via: pip install git+https://github.com/openai/CLIP.git"
        ) from e
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, clip


def rank_masks_by_prompt(image_rgb: np.ndarray, masks: List[dict], prompt: str, device: str) -> Tuple[np.ndarray, float]:
    if len(masks) == 0:
        raise RuntimeError("No masks produced by SAM2.")

    model, preprocess, clip = load_clip(device)

    import torch
    text = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    from PIL import Image

    best_score = -1.0
    best_mask = None

    for m in masks:
        seg = m.get("segmentation", None)
        if seg is None:
            continue
        seg = seg.astype(bool)
        if seg.sum() == 0:
            continue

        masked = np.zeros_like(image_rgb)
        masked[seg] = image_rgb[seg]

        ys, xs = np.where(seg)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        crop = masked[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        pil_img = Image.fromarray(crop)
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(input_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            score = float((image_features @ text_features.T).squeeze().item())

        if score > best_score:
            best_score = score
            best_mask = seg

    if best_mask is None:
        raise RuntimeError("Failed to select a mask with CLIP scoring.")

    return best_mask.astype(np.uint8) * 255, best_score


def main():
    parser = argparse.ArgumentParser(
        description="Segment an object from an RGB image with a text prompt and save a binary mask (white=object, black=background)."
    )
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument("--image", type=str, default="/home-ldap/yutong/Projects/RGBTrack/demo_data/test/rgb", required=False, help="Path to input RGB image")
    parser.add_argument("--prompt", type=str, default="beachball", required=False, help="Text prompt describing the target object")
    parser.add_argument("--out_mask", type=str, default=f"{code_dir}/demo_data/test/masks/beachball_masks.png", required=False, help="Path to save the binary mask PNG")
    parser.add_argument("--out_vis", type=str, default=f"{code_dir}/demo_data/test/masks/beachball_vis.png", required=False, help="Path to save side-by-side visualization (rgb | mask)")
    parser.add_argument("--sam2_repo", type=str, default=f"{code_dir}/segment-anything-2-real-time", help="Path to SAM2 repo root")
    parser.add_argument("--sam2_checkpoint", type=str, default=f"{code_dir}/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt", required=False, help="Path to SAM2 checkpoint (.pt)")
    parser.add_argument("--sam2_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_s.yaml", help="Relative config path in SAM2 repo")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()

    # Support passing a directory; pick the first image file inside it
    image_path = args.image
    if os.path.isdir(image_path):
        candidates = sorted(
            f for f in os.listdir(image_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"))
        )
        if not candidates:
            raise RuntimeError(f"No image files found under directory: {image_path}")
        image_path = os.path.join(image_path, candidates[0])

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Auto-select config by checkpoint name if possible
    cfg_rel = args.sam2_cfg
    if args.sam2_checkpoint:
        ckpt_name = os.path.basename(args.sam2_checkpoint).lower()
        if "sam2.1" in ckpt_name:
            if "tiny" in ckpt_name:
                cfg_rel = "configs/sam2.1/sam2.1_hiera_t.yaml"
            elif "small" in ckpt_name:
                cfg_rel = "configs/sam2.1/sam2.1_hiera_s.yaml"
            elif "base_plus" in ckpt_name or "b_plus" in ckpt_name or "b+.pt" in ckpt_name or "_b+." in ckpt_name:
                cfg_rel = "configs/sam2.1/sam2.1_hiera_b+.yaml"
            elif "large" in ckpt_name or "hiera_l" in ckpt_name:
                cfg_rel = "configs/sam2.1/sam2.1_hiera_l.yaml"

    model = build_sam2_model(args.sam2_repo, args.sam2_checkpoint, args.device, cfg_rel)
    masks = generate_masks_with_sam2(image_rgb, model)
    best_mask_u8, score = rank_masks_by_prompt(image_rgb, masks, args.prompt, args.device)

    # Derive output filenames to match input image basename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    # Determine mask output directory from provided arg (treat arg as dir if it is a dir or endswith '/')
    if args.out_mask.endswith("/") or os.path.isdir(args.out_mask):
        out_mask_dir = args.out_mask
    else:
        out_mask_dir = os.path.dirname(args.out_mask)
    if out_mask_dir:
        ensure_dir(out_mask_dir)
    out_mask_path = os.path.join(out_mask_dir if out_mask_dir else ".", f"{base_name}.png")
    cv2.imwrite(out_mask_path, best_mask_u8)
    print(f"Saved mask to {out_mask_path} (CLIP score={score:.4f})")

    # Save a side-by-side visualization of original image and binary mask
    try:
        mask_bgr = cv2.cvtColor(best_mask_u8, cv2.COLOR_GRAY2BGR)
        h1, w1 = image_bgr.shape[:2]
        h2, w2 = mask_bgr.shape[:2]
        if (h1, w1) != (h2, w2):
            mask_bgr = cv2.resize(mask_bgr, (w1, h1), interpolation=cv2.INTER_NEAREST)
        vis = np.hstack([image_bgr, mask_bgr])
        # Determine vis output directory similarly and name as <basename>_vis.png
        if args.out_vis.endswith("/") or os.path.isdir(args.out_vis):
            out_vis_dir = args.out_vis
        else:
            out_vis_dir = os.path.dirname(args.out_vis)
        if out_vis_dir:
            ensure_dir(out_vis_dir)
        out_vis_path = os.path.join(out_vis_dir if out_vis_dir else ".", f"{base_name}_vis.png")
        cv2.imwrite(out_vis_path, vis)
        print(f"Saved visualization to {out_vis_path}")
    except Exception as e:
        print(f"Warning: failed to save visualization: {e}")


if __name__ == "__main__":
    main()