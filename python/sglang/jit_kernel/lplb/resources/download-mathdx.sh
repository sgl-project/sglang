#!/bin/bash
# Download NVIDIA Math-DX (cuBLASDx + cuSolverDx) headers and the cuSolverDx
# device-side library, place them under ./mathdx/ next to this script. The
# JIT path in cuda_solver.py picks them up via get_mathdx_root().
#
# Override MATHDX_VERSION / MATHDX_URL to pin a specific release.

set -euo pipefail

cd "$(dirname "$0")"

: "${MATHDX_VERSION:=25.06.0}"
: "${MATHDX_URL:=https://developer.download.nvidia.com/compute/cublasdx/redist/cublasdx/nvidia-mathdx-${MATHDX_VERSION}-Linux.tar.gz}"

if [[ -d mathdx/include && -d mathdx/lib ]]; then
    echo "[download-mathdx] mathdx/ already present at $(pwd)/mathdx — skipping"
    exit 0
fi

echo "[download-mathdx] fetching ${MATHDX_URL}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

curl -fSL "${MATHDX_URL}" -o "${TMP}/mathdx.tar.gz"

mkdir -p mathdx
tar -xzf "${TMP}/mathdx.tar.gz" -C "${TMP}"

# The archive top-level is a versioned directory; flatten it.
SRC="$(find "${TMP}" -maxdepth 2 -mindepth 1 -type d -name 'nvidia-mathdx*' | head -1)"
[[ -n "${SRC}" ]] || { echo "[download-mathdx] could not locate extracted dir" >&2; exit 1; }

cp -r "${SRC}/include"          mathdx/
[[ -d "${SRC}/lib"      ]] && cp -r "${SRC}/lib"      mathdx/ || true
[[ -d "${SRC}/external" ]] && cp -r "${SRC}/external" mathdx/ || true

echo "[download-mathdx] installed:"
ls -la mathdx/
echo "[download-mathdx] done."
