#!/bin/bash
# Build sgl-deepep wheel inside a CUDA-versioned container.
#
# Usage: build_sgl_deepep.sh <PYTHON_VERSION> <CUDA_VERSION> <DEEPEP_WORKSPACE> [ARCH]
set -ex

if [ $# -lt 3 ]; then
  echo "Usage: $0 <PYTHON_VERSION> <CUDA_VERSION> <DEEPEP_WORKSPACE> [ARCH]"
  exit 1
fi

PYTHON_VERSION="$1"
CUDA_VERSION="$2"
DEEPEP_WORKSPACE="$(cd "$3" && pwd)"
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
DOCKERFILE="${REPO_ROOT}/docker/sgl-deepep.Dockerfile"
RENAME_SCRIPT="${SCRIPT_DIR}/rename_sgl_deepep_whl.sh"
PATCH_SCRIPT="${SCRIPT_DIR}/apply_deepep_k3_patch.sh"

DEPS_TAG="sgl-deepep-deps:cuda${CUDA_VERSION}-${PY_TAG}-${ARCH}"

echo "----------------------------------------"
echo "PYTHON_VERSION: ${PYTHON_VERSION}"
echo "CUDA_VERSION:   ${CUDA_VERSION}"
echo "CU_TAG:         ${CU_TAG}"
echo "ARCH:           ${ARCH}"
echo "BASE_IMG:       ${BASE_IMG}"
echo "DEEPEP_WORKSPACE: ${DEEPEP_WORKSPACE}"
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

mkdir -p "${DEEPEP_WORKSPACE}/dist"

# 1) Build the wheel inside the deps container.
docker run --rm \
  --network=host \
  -v "${DEEPEP_WORKSPACE}:/sgl-workspace/DeepEP" \
  -v "${PATCH_SCRIPT}:/apply_deepep_k3_patch.sh:ro" \
  -w /sgl-workspace/DeepEP \
  -e DEEPEP_DIR=/sgl-workspace/DeepEP \
  "${DEPS_TAG}" \
  bash /apply_deepep_k3_patch.sh

# 2) Rename inside the same image so we have a working pip / wheel CLI and can
#    rewrite the root-owned wheel files written by the build container above.
docker run --rm \
  -v "${DEEPEP_WORKSPACE}:/sgl-workspace/DeepEP" \
  -v "${RENAME_SCRIPT}:/rename_sgl_deepep_whl.sh:ro" \
  -w /sgl-workspace/DeepEP \
  "${DEPS_TAG}" \
  bash /rename_sgl_deepep_whl.sh dist "${CU_TAG}" "${ARCH}"

# 3) cu130 only: produce a sibling dist-pypi/ with the +cu130 local-version
#    stripped (PyPI rejects local versions).
if [ "${CU_TAG}" = "cu130" ]; then
  docker run --rm \
    -v "${DEEPEP_WORKSPACE}:/sgl-workspace/DeepEP" \
    -w /sgl-workspace/DeepEP \
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

echo "Wheels in ${DEEPEP_WORKSPACE}/dist:"
ls -lh "${DEEPEP_WORKSPACE}/dist"/*.whl 2>/dev/null || true
if [ "${CU_TAG}" = "cu130" ]; then
  echo "PyPI-ready wheels in ${DEEPEP_WORKSPACE}/dist-pypi:"
  ls -lh "${DEEPEP_WORKSPACE}/dist-pypi"/*.whl 2>/dev/null || true
fi
