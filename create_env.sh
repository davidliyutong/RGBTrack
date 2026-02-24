#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-RGBTrack}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAM2_DIR="${SAM2_DIR:-${WORKSPACE_DIR}/segment-anything-2-real-time}"
INSTALL_KAOLIN="${INSTALL_KAOLIN:-0}"
DOWNLOAD_SAM2_CKPT="${DOWNLOAD_SAM2_CKPT:-1}"
INSTALL_PYTORCH3D="${INSTALL_PYTORCH3D:-1}"
CUDA_TAG="${CUDA_TAG:-cu117}"
PYTORCH_TAG="${PYTORCH_TAG:-pyt200}"

echo "[INFO] Workspace: ${WORKSPACE_DIR}"
echo "[INFO] Target conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"

if ! command -v conda >/dev/null 2>&1; then
	echo "[ERROR] conda not found. Please install Miniconda/Anaconda first."
	exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
	echo "[INFO] Conda environment '${ENV_NAME}' already exists. Reusing it."
else
	echo "[INFO] Creating conda environment '${ENV_NAME}'..."
	conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

echo "[INFO] Installing Eigen..."
conda install -y -c conda-forge eigen=3.4.0

echo "[INFO] Installing Python dependencies from requirements.txt..."
python -m pip install --upgrade pip
python -m pip install "setuptools<81" wheel
python -m pip install --no-build-isolation "visdom==0.2.4"
python -m pip install -r "${WORKSPACE_DIR}/requirements.txt"

echo "[INFO] Installing NVDiffRast..."
python -m pip install --quiet --no-cache-dir --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git

if [[ "${INSTALL_KAOLIN}" == "1" ]]; then
	echo "[INFO] Installing Kaolin..."
	python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu128.html
else
	echo "[INFO] Skipping Kaolin (INSTALL_KAOLIN=0)."
fi

if [[ "${INSTALL_PYTORCH3D}" == "1" ]]; then
	echo "[INFO] Installing PyTorch3D wheel..."
	python -m pip install --quiet --no-index --no-cache-dir --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
else
	echo "[INFO] Skipping PyTorch3D (INSTALL_PYTORCH3D=0)."
fi

if [[ -d "${SAM2_DIR}" ]]; then
	echo "[INFO] Installing SAM2 from local checkout: ${SAM2_DIR}"
	python -m pip install -e "${SAM2_DIR}"

	if [[ "${DOWNLOAD_SAM2_CKPT}" == "1" ]]; then
		if [[ -x "${SAM2_DIR}/checkpoints/download_ckpts.sh" ]]; then
			echo "[INFO] Downloading SAM2 checkpoints..."
			(
				cd "${SAM2_DIR}/checkpoints"
				./download_ckpts.sh
			)
		else
			echo "[WARN] SAM2 checkpoint downloader not executable: ${SAM2_DIR}/checkpoints/download_ckpts.sh"
		fi
	fi
else
	echo "[ERROR] SAM2 directory not found: ${SAM2_DIR}"
	echo "        Please ensure 'segment-anything-2-real-time' is checked out in workspace."
	exit 1
fi

if [[ -d "${WORKSPACE_DIR}/bop_toolkit" ]]; then
	echo "[INFO] Installing BOP Toolkit from local checkout..."
	python -m pip install -e "${WORKSPACE_DIR}/bop_toolkit"
else
	echo "[WARN] BOP Toolkit directory not found: ${WORKSPACE_DIR}/bop_toolkit"
fi

echo "[INFO] Building FoundationPose extensions (conda mode)..."
CMAKE_PREFIX_PATH="${CONDA_PREFIX}/lib/python${PYTHON_VERSION}/site-packages/pybind11/share/cmake/pybind11" \
	bash "${WORKSPACE_DIR}/build_all_conda.sh"

# TensorRT
echo "[INFO] Installing TensorRT (CUDA 12)..."
if ! python -m pip install --quiet --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt-cu12; then
	echo "[WARN] tensorrt-cu12 install failed, trying legacy nvidia-tensorrt package..."
	python -m pip install --quiet --no-cache-dir --extra-index-url https://pypi.nvidia.com nvidia-tensorrt
fi

# Filterpy
echo "[INFO] Installing FilterPy..."
python -m pip install --quiet --no-cache-dir filterpy==1.4.5

# MVSDK
echo "[INFO] Installing MVSDK..."
python -m pip install --quiet third_party/mvsdk-python

# Server requirements
echo "[INFO] Installing server requirements..."
python -m pip install --quiet --no-cache-dir -r "${WORKSPACE_DIR}/requirements.server.txt"

echo
echo "[DONE] Environment '${ENV_NAME}' is ready."
echo "       Monocular depth modules (Metric3D / ZoeDepth) were intentionally not installed."
echo "       Activate with: conda activate ${ENV_NAME}"
