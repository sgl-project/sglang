#!/bin/bash
# Download NVIDIA Math-DX (cuBLASDx + cuSolverDx) headers and the cuSolverDx
# device-side library, place them under ./mathdx/ next to this script. The
# JIT path in cuda_solver.py picks them up via get_mathdx_root().
#
# Override MATHDX_VERSION to pin a specific release.

set -euo pipefail

cd "$(dirname "$0")"

: "${MATHDX_VERSION:=25.06.0}"
MATHDX_MINOR="${MATHDX_VERSION%.*}"
URL="https://developer.download.nvidia.com/compute/cuSOLVERDx/redist/cuSOLVERDx/nvidia-mathdx-${MATHDX_VERSION}.tar.gz"

if [[ -d mathdx/include && -d mathdx/lib ]]; then
    echo "[download-mathdx] mathdx/ already present at $(pwd)/mathdx — skipping"
    exit 0
fi

echo "[download-mathdx] fetching ${URL}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

curl -fSL "${URL}" -o "${TMP}/mathdx.tar.gz"
tar -xzf "${TMP}/mathdx.tar.gz" -C "${TMP}"

# Archive layout: nvidia-mathdx-<full>/nvidia/mathdx/<minor>/{include,lib,external}
SRC="${TMP}/nvidia-mathdx-${MATHDX_VERSION}/nvidia/mathdx/${MATHDX_MINOR}"
[[ -d "${SRC}/include" ]] || {
    echo "[download-mathdx] expected ${SRC}/include not found" >&2
    exit 1
}

rm -rf mathdx
mv "${SRC}" mathdx

echo "[download-mathdx] installed:"
ls -la mathdx/
echo "[download-mathdx] done. Set MATHDX_HOME=$(pwd)/mathdx if calling from outside this directory."
