import cv2
import numpy as np
import glob
import os
import random
import sys
import matplotlib.pyplot as plt

# ================== 用户输入参数 ==================
checker_width = 0.015     # 每个小方格的宽度 (15mm)
board_rows = 6
board_cols = 8
image_dir = "/home/xxz/code/toss-reorient/captures"
num_samples = 50
output_dir = "/home/xxz/code/toss-reorient/RGBTrack/demo_data/calib_images"
# ==================================================

pattern_size = (board_cols - 1, board_rows - 1)

# 构建棋盘格三维点
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= checker_width

objpoints, imgpoints = [], []
img_size = None

images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) +
                glob.glob(os.path.join(image_dir, "*.png")))
total_images = len(images)
print(f"Total images found: {total_images}")

if total_images > num_samples:
    T = total_images // num_samples
    images = [images[i] for i in range(0, total_images, T)][:num_samples]
else:
    T = 1
print(f"Sampling every {T} image(s): total {len(images)} images for calibration...\n")

# === 检测角点 ===
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if ret:
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        )
        objpoints.append(objp)
        imgpoints.append(corners2)
        print(f"{os.path.basename(fname)}: corners detected")
    else:
        print(f"{os.path.basename(fname)}: no corners detected")

print(f"\nValid frames used: {len(objpoints)}")
if len(objpoints) < 5:
    raise RuntimeError("Not enough valid images for calibration!")

# === 初次标定 ===
print("\nRunning initial pinhole calibration...")
rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None,
    flags=cv2.CALIB_RATIONAL_MODEL
)

print(f"\nInitial RMS: {rms:.5f}")
print("K (camera matrix):\n", K)
print("D (distortion coefficients):\n", D.ravel())

# === 计算重投影误差 ===
errors = []
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
    err = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    errors.append(err)

mean_err = np.mean(errors)
std_err = np.std(errors)
thresh = mean_err + 2 * std_err
print(f"\nMean per-image error: {mean_err:.4f}, Std: {std_err:.4f}, Threshold: {thresh:.4f}")

good_indices = [i for i, e in enumerate(errors) if e < thresh]
print(f"Removing {len(errors) - len(good_indices)} outlier(s)...")

# === 剔除异常帧重新标定 ===
objpoints = [objpoints[i] for i in good_indices]
imgpoints = [imgpoints[i] for i in good_indices]
rms_refined, K, D, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None,
    flags=cv2.CALIB_RATIONAL_MODEL
)

print("\n=== Refined Calibration Results ===")
print("RMS Reprojection Error:", rms_refined)
print("Camera Matrix (K):\n", K)
print("Distortion Coefficients (k1,k2,p1,p2,k3,...):\n", D.ravel())

# === 保存参数 ===
np.savez(os.path.join(output_dir, "high_speed_camera_checkerboard_pinhole_refined.npz"),
         K=K, D=D, rms=rms_refined)
print(f"\nSaved refined parameters to high_speed_camera_checkerboard_pinhole_refined.npz")

# === 绘制误差图 ===
plt.figure(figsize=(10, 4))
plt.bar(range(len(errors)), errors, color='gray', label='All Images')
plt.axhline(mean_err, color='r', linestyle='--', label='Mean Error')
plt.axhline(thresh, color='orange', linestyle='--', label='Threshold')
plt.xlabel("Image Index")
plt.ylabel("Reprojection Error (px)")
plt.title("Pinhole Per-Image Reprojection Errors")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pinhole_error_distribution.png"))
print(f"Saved error distribution to pinhole_error_distribution.png")

# === 去畸变示例 ===
test_img = cv2.imread(images[0])
h, w = test_img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
undistorted = cv2.undistort(test_img, K, D, None, newcameramtx)

combined = np.hstack((test_img, undistorted))
cv2.putText(combined, "Original", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
cv2.putText(combined, "Undistorted", (w + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
output_path = os.path.join(output_dir, "calibration_result_pinhole_refined.png")
cv2.imwrite(output_path, combined)
print(f"Saved comparison image to: {output_path}")

sys.exit(0)
