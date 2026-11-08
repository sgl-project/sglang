#!/bin/bash
# Build the sgl-flash-mla wheel inside a CUDA-versioned container.
#
# Usage: build_sgl_flash_mla.sh <PYTHON_VERSION> <CUDA_VERSION> <FLASHMLA_SRC> [ARCH]
#   PYTHON_VERSION: e.g. 3.12 -- only picks the interpreter that drives the build.
#                   The wheel is abi3 and serves CPython 3.10+ whichever is used.
#   CUDA_VERSION:   13.4
#   FLASHMLA_SRC:   path to a checkout of sgl-project/FlashMLA (submodules included)
#   ARCH:           x86_64 (default) or aarch64
#
# Writes:
#   <FLASHMLA_SRC>/dist/      — wheel tagged +cu130 and manylinux2014
#   <FLASHMLA_SRC>/dist-pypi/ — the same wheel with +cu130 stripped
#                               (PyPI rejects local-version segments)
#
# On the two version numbers: the kernels are compiled by CUDA 13.4, because they
# use `cvt` with `.scaled::n2::ue8m0` (nvcc >= 13.1) and older toolkits emit
# suboptimal instructions for them. The +cu130 tag describes the *runtime* the
# wheel is published for: the extensions link libcudart.so.13, whose SONAME
# carries only the major version, and call no cudart API newer than 13.0, so a
# 13.4-built wheel loads against the cu130 torch SGLang ships.
set -ex

if [ $# -lt 3 ]; then
  echo "Usage: $0 <PYTHON_VERSION> <CUDA_VERSION> <FLASHMLA_SRC> [ARCH]"
  exit 1
fi

PYTHON_VERSION="$1"
CUDA_VERSION="$2"
FLASHMLA_SRC="$(cd "$3" && pwd)"
ARCH="${4:-$(uname -i)}"

case "${CUDA_VERSION}" in
  13.4) CU_TAG=cu130 ;;
  *)
    echo "Unsupported CUDA_VERSION: ${CUDA_VERSION} (the kernels require 13.4)" >&2
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
DOCKERFILE="${REPO_ROOT}/docker/sgl-flash-mla.Dockerfile"
RENAME_SCRIPT="${SCRIPT_DIR}/rename_sgl_flash_mla_whl.sh"

DEPS_TAG="sgl-flash-mla-deps:cuda${CUDA_VERSION}-${PY_TAG}-${ARCH}"

echo "----------------------------------------"
echo "PYTHON_VERSION: ${PYTHON_VERSION}"
echo "CUDA_VERSION:   ${CUDA_VERSION}"
echo "CU_TAG:         ${CU_TAG}"
echo "ARCH:           ${ARCH}"
echo "BASE_IMG:       ${BASE_IMG}"
echo "FLASHMLA_SRC:   ${FLASHMLA_SRC}"
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

mkdir -p "${FLASHMLA_SRC}/dist"

# 1) Build. --no-build-isolation because setup.py imports torch, which the image
#    provides; FlashMLA declares no PEP 518 build requirements.
#    NVCC_THREADS defaults to 32 in setup.py; MAX_JOBS is what actually
#    parallelises, and without ninja on PATH torch silently ignores it.
docker run --rm \
  --network=host \
  -v "${FLASHMLA_SRC}:/flashmla" \
  -w /flashmla \
  -e MAX_JOBS="${MAX_JOBS:-$(nproc)}" \
  "${DEPS_TAG}" \
  bash -c 'set -eux; python3 setup.py bdist_wheel'

# 2) Retag inside the same image, which has a working pip/wheel and can rewrite
#    the root-owned files the build container just wrote.
docker run --rm \
  -v "${FLASHMLA_SRC}:/flashmla" \
  -v "${RENAME_SCRIPT}:/rename_sgl_flash_mla_whl.sh:ro" \
  -w /flashmla \
  "${DEPS_TAG}" \
  bash /rename_sgl_flash_mla_whl.sh dist "${CU_TAG}" "${ARCH}"

# 3) Sibling dist-pypi/ with the local version stripped.
docker run --rm \
  -v "${FLASHMLA_SRC}:/flashmla" \
  -w /flashmla \
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

echo "Wheels in ${FLASHMLA_SRC}/dist:"
ls -lh "${FLASHMLA_SRC}/dist"/*.whl 2>/dev/null || true
echo "PyPI-ready wheels in ${FLASHMLA_SRC}/dist-pypi:"
ls -lh "${FLASHMLA_SRC}/dist-pypi"/*.whl 2>/dev/null || true
