#!/bin/bash
# Build sgl-deep-gemm wheel inside a CUDA-versioned container.
#
# Usage: build_sgl_deep_gemm.sh <PYTHON_VERSION> <CUDA_VERSION> <DEEPGEMM_SRC> [ARCH]
#   PYTHON_VERSION: e.g. 3.10
#   CUDA_VERSION:   e.g. 12.9 or 13.0
#   DEEPGEMM_SRC:   path to a checkout of sgl-project/DeepGEMM
#   ARCH:           x86_64 (default) or aarch64
#
# The wheel is written into <DEEPGEMM_SRC>/dist by build_sgl_deep_gemm.sh.
set -ex

if [ $# -lt 3 ]; then
  echo "Usage: $0 <PYTHON_VERSION> <CUDA_VERSION> <DEEPGEMM_SRC> [ARCH]"
  exit 1
fi

PYTHON_VERSION="$1"
CUDA_VERSION="$2"
DEEPGEMM_SRC="$(cd "$3" && pwd)"
ARCH="${4:-$(uname -i)}"

if [ "${ARCH}" = "aarch64" ]; then
  BASE_IMG="pytorch/manylinuxaarch64-builder"
else
  BASE_IMG="pytorch/manylinux2_28-builder"
fi

PY_TAG="cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE="${REPO_ROOT}/docker/sgl-deep-gemm.Dockerfile"

DEPS_TAG="sgl-deep-gemm-deps:cuda${CUDA_VERSION}-${PY_TAG}-${ARCH}"

echo "----------------------------------------"
echo "PYTHON_VERSION: ${PYTHON_VERSION}"
echo "CUDA_VERSION:   ${CUDA_VERSION}"
echo "ARCH:           ${ARCH}"
echo "BASE_IMG:       ${BASE_IMG}"
echo "DEEPGEMM_SRC:   ${DEEPGEMM_SRC}"
echo "DEPS_TAG:       ${DEPS_TAG}"
echo "----------------------------------------"

docker build \
  -f "${DOCKERFILE}" "$(dirname "${DOCKERFILE}")" \
  --build-arg BASE_IMG="${BASE_IMG}" \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg ARCH="${ARCH}" \
  --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
  --build-arg PYTHON_TAG="${PY_TAG}" \
  -t "${DEPS_TAG}" \
  --network=host

mkdir -p "${DEEPGEMM_SRC}/dist"

docker run --rm \
  --network=host \
  -v "${DEEPGEMM_SRC}:/deepgemm" \
  -w /deepgemm \
  "${DEPS_TAG}" \
  bash build_sgl_deep_gemm.sh

echo "Wheels written to ${DEEPGEMM_SRC}/dist:"
ls -lh "${DEEPGEMM_SRC}/dist"/*.whl 2>/dev/null || true
