#!/bin/bash
# Build sgl-deep-gemm wheel inside a CUDA-versioned container.
#
# Usage: build_sgl_deep_gemm.sh <PYTHON_VERSION> <CUDA_VERSION> <DEEPGEMM_SRC> [ARCH]
#   PYTHON_VERSION: e.g. 3.10
#   CUDA_VERSION:   e.g. 12.9 or 13.0
#   DEEPGEMM_SRC:   path to a checkout of sgl-project/DeepGEMM
#   ARCH:           x86_64 (default) or aarch64
#
# Writes:
#   <DEEPGEMM_SRC>/dist/      — wheel(s) tagged +cu129 / +cu130 and manylinux
#   <DEEPGEMM_SRC>/dist-pypi/ — cu130 only: same wheel(s) with +cu130 stripped
#                              (PyPI rejects local-version segments)
set -ex

if [ $# -lt 3 ]; then
  echo "Usage: $0 <PYTHON_VERSION> <CUDA_VERSION> <DEEPGEMM_SRC> [ARCH]"
  exit 1
fi

PYTHON_VERSION="$1"
CUDA_VERSION="$2"
DEEPGEMM_SRC="$(cd "$3" && pwd)"
ARCH="${4:-$(uname -i)}"

case "${CUDA_VERSION}" in
  13.0) CU_TAG=cu130 ;;
  12.9) CU_TAG=cu129 ;;
  *)
    echo "Unsupported CUDA_VERSION: ${CUDA_VERSION}" >&2
    exit 1
    ;;
esac

if [ "${ARCH}" = "aarch64" ]; then
  BASE_IMG="pytorch/manylinuxaarch64-builder"
else
  BASE_IMG="pytorch/manylinux2_28-builder"
fi

PY_TAG="cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE="${REPO_ROOT}/docker/sgl-deep-gemm.Dockerfile"
RENAME_SCRIPT="${SCRIPT_DIR}/rename_sgl_deep_gemm_whl.sh"

DEPS_TAG="sgl-deep-gemm-deps:cuda${CUDA_VERSION}-${PY_TAG}-${ARCH}"

echo "----------------------------------------"
echo "PYTHON_VERSION: ${PYTHON_VERSION}"
echo "CUDA_VERSION:   ${CUDA_VERSION}"
echo "CU_TAG:         ${CU_TAG}"
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

# 1) Build the wheel inside the deps container.
docker run --rm \
  --network=host \
  -v "${DEEPGEMM_SRC}:/deepgemm" \
  -w /deepgemm \
  "${DEPS_TAG}" \
  bash build_sgl_deep_gemm.sh

# 2) Rename inside the same image so we have a working pip / wheel CLI and can
#    rewrite the root-owned wheel files written by the build container above.
docker run --rm \
  -v "${DEEPGEMM_SRC}:/deepgemm" \
  -v "${RENAME_SCRIPT}:/rename_sgl_deep_gemm_whl.sh:ro" \
  -w /deepgemm \
  "${DEPS_TAG}" \
  bash /rename_sgl_deep_gemm_whl.sh dist "${CU_TAG}" "${ARCH}"

# 3) cu130 only: produce a sibling dist-pypi/ with the +cu130 local-version
#    stripped (PyPI rejects local versions).
if [ "${CU_TAG}" = "cu130" ]; then
  docker run --rm \
    -v "${DEEPGEMM_SRC}:/deepgemm" \
    -w /deepgemm \
    "${DEPS_TAG}" \
    bash -c '
set -eux
mkdir -p dist-pypi
for w in dist/*.whl; do
  tmp=$(mktemp -d)
  python3 -m wheel unpack "$w" --dest "$tmp"
  unpacked=$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | head -1)
  info=$(find "$unpacked" -maxdepth 1 -type d -name "*.dist-info" | head -1)
  meta="$info/METADATA"
  orig=$(grep "^Version:" "$meta" | head -1 | sed "s/^Version:[[:space:]]*//")
  new=$(echo "$orig" | sed "s/+cu[0-9]\+$//")
  if [ "$orig" != "$new" ]; then
    sed -i "s/^Version:.*/Version: ${new}/" "$meta"
    old_base=$(basename "$info")
    new_base="${old_base/${orig}/${new}}"
    mv "$info" "$(dirname "$info")/${new_base}"
  fi
  python3 -m wheel pack "$unpacked" --dest-dir dist-pypi
  rm -rf "$tmp"
done
ls -lh dist-pypi/
'
fi

echo "Wheels in ${DEEPGEMM_SRC}/dist:"
ls -lh "${DEEPGEMM_SRC}/dist"/*.whl 2>/dev/null || true
if [ "${CU_TAG}" = "cu130" ]; then
  echo "PyPI-ready wheels in ${DEEPGEMM_SRC}/dist-pypi:"
  ls -lh "${DEEPGEMM_SRC}/dist-pypi"/*.whl 2>/dev/null || true
fi
