#!/usr/bin/env bash
# Build sgl-deep-ep wheels in a CUDA-versioned manylinux container.
#
# Usage:
#   build_sgl_deepep.sh <python-version> <cuda-version> <deepep-source> <packaging-overlay> [architecture]
#
# Writes CUDA-tagged wheels to <deepep-source>/dist. CUDA 13 builds also write
# PyPI-ready wheels without the local CUDA version to <deepep-source>/dist-pypi.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: build_sgl_deepep.sh <python-version> <cuda-version> <deepep-source> <packaging-overlay> [architecture]

  python-version:     3.10, 3.11, 3.12, or 3.13
  cuda-version:       12.9 or 13.0
  deepep-source:      checkout of the selected DeepEP implementation branch
  packaging-overlay: path to the shared DeepEP sgl_deep_ep directory
  architecture:       x86_64 or aarch64 (defaults to the current machine)
EOF
}

if [[ $# -lt 4 || $# -gt 5 ]]; then
    usage >&2
    exit 2
fi

PYTHON_VERSION="$1"
CUDA_VERSION="$2"
DEEPEP_SOURCE="$(cd "$3" && pwd)"
PACKAGING_OVERLAY="$(cd "$4" && pwd)"
ARCHITECTURE="${5:-$(uname -m)}"

if [[ "${ARCHITECTURE}" == arm64 ]]; then
    ARCHITECTURE=aarch64
fi

case "${PYTHON_VERSION}" in
    3.10|3.11|3.12|3.13) ;;
    *)
        echo "Unsupported Python version: ${PYTHON_VERSION}" >&2
        exit 2
        ;;
esac

case "${CUDA_VERSION}" in
    12.9)
        CUDA_TAG=cu129
        ;;
    13.0)
        CUDA_TAG=cu130
        ;;
    *)
        echo "Unsupported CUDA version: ${CUDA_VERSION}" >&2
        exit 2
        ;;
esac

case "${ARCHITECTURE}" in
    x86_64)
        BASE_IMAGE=pytorch/manylinux2_28-builder
        ;;
    aarch64)
        BASE_IMAGE=pytorch/manylinuxaarch64-builder
        ;;
    *)
        echo "Unsupported architecture: ${ARCHITECTURE}" >&2
        exit 2
        ;;
esac

for required_file in setup.py deep_ep/__init__.py; do
    if [[ ! -f "${DEEPEP_SOURCE}/${required_file}" ]]; then
        echo "DeepEP source is missing ${required_file}: ${DEEPEP_SOURCE}" >&2
        exit 1
    fi
done
for required_file in build_sgl_deep_ep.sh setup.py VERSION; do
    if [[ ! -f "${PACKAGING_OVERLAY}/${required_file}" ]]; then
        echo "Packaging overlay is missing ${required_file}: ${PACKAGING_OVERLAY}" >&2
        exit 1
    fi
done
if ! command -v docker >/dev/null; then
    echo "docker is required to build sgl-deep-ep" >&2
    exit 1
fi

PYTHON_TAG="cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE="${REPOSITORY_ROOT}/docker/sgl-deep-ep.Dockerfile"
IMAGE_TAG="sgl-deep-ep-builder:cuda${CUDA_VERSION}-${PYTHON_TAG}-${ARCHITECTURE}"
DIST_DIR="${DEEPEP_SOURCE}/dist"
PYPI_DIST_DIR="${DEEPEP_SOURCE}/dist-pypi"

mkdir -p "${DIST_DIR}" "${PYPI_DIST_DIR}"

echo "----------------------------------------"
echo "Python:            ${PYTHON_VERSION} (${PYTHON_TAG})"
echo "CUDA:              ${CUDA_VERSION} (${CUDA_TAG})"
echo "Architecture:      ${ARCHITECTURE}"
echo "Base image:        ${BASE_IMAGE}:cuda${CUDA_VERSION}"
echo "DeepEP source:     ${DEEPEP_SOURCE}"
echo "Packaging overlay: ${PACKAGING_OVERLAY}"
echo "Builder image:     ${IMAGE_TAG}"
echo "----------------------------------------"

docker build \
    --file "${DOCKERFILE}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg CUDA_VERSION="${CUDA_VERSION}" \
    --build-arg CUDA_TAG="${CUDA_TAG}" \
    --build-arg PYTHON_TAG="${PYTHON_TAG}" \
    --build-arg ARCHITECTURE="${ARCHITECTURE}" \
    --build-arg TORCH_VERSION="${TORCH_VERSION:-2.13.0}" \
    --tag "${IMAGE_TAG}" \
    --network=host \
    "${REPOSITORY_ROOT}/docker"

docker run --rm \
    --network=host \
    --env ARCHITECTURE="${ARCHITECTURE}" \
    --env CUDA_TAG="${CUDA_TAG}" \
    --env CUDA_VERSION="${CUDA_VERSION}" \
    --env MAX_JOBS="${MAX_JOBS:-8}" \
    --volume "${DEEPEP_SOURCE}:/deepep:ro" \
    --volume "${PACKAGING_OVERLAY}:/sgl-deep-ep-packaging:ro" \
    --volume "${DIST_DIR}:/output/dist" \
    --volume "${PYPI_DIST_DIR}:/output/dist-pypi" \
    "${IMAGE_TAG}" \
    bash -euo pipefail -c '
find /output/dist -maxdepth 1 -type f -name "sgl_deep_ep-*.whl" -delete
find /output/dist-pypi -maxdepth 1 -type f -name "sgl_deep_ep-*.whl" -delete
raw_dir="$(mktemp -d -t sgl-deep-ep-raw.XXXXXX)"
trap '\''rm -rf -- "${raw_dir}"'\'' EXIT

bash /sgl-deep-ep-packaging/build_sgl_deep_ep.sh \
    /deepep /sgl-deep-ep-packaging "${raw_dir}" "${CUDA_VERSION}" "${ARCHITECTURE}"

shopt -s nullglob
raw_wheels=("${raw_dir}"/*.whl)
if [[ ${#raw_wheels[@]} -ne 1 ]]; then
    echo "Expected exactly one raw wheel, found ${#raw_wheels[@]}" >&2
    exit 1
fi

auditwheel repair \
    --plat "manylinux_2_28_${ARCHITECTURE}" \
    --wheel-dir /output/dist \
    --exclude libcuda.so.1 \
    --exclude libcudart.so.12 \
    --exclude libcudart.so.13 \
    --exclude libc10.so \
    --exclude libc10_cuda.so \
    --exclude libtorch.so \
    --exclude libtorch_cpu.so \
    --exclude libtorch_cuda.so \
    --exclude libtorch_python.so \
    --exclude libnvshmem_host.so.1 \
    --exclude libnvshmem_host.so.2 \
    --exclude libnvshmem_host.so.3 \
    --exclude libnccl.so.2 \
    --exclude libgdrapi.so.2 \
    --exclude libnvToolsExt.so.1 \
    "${raw_wheels[0]}"

if [[ "${CUDA_TAG}" == cu130 ]]; then
    tagged_wheels=(/output/dist/sgl_deep_ep-*+cu130-*.whl)
    if [[ ${#tagged_wheels[@]} -ne 1 ]]; then
        echo "Expected exactly one CUDA 13 wheel, found ${#tagged_wheels[@]}" >&2
        exit 1
    fi
    unpack_root="$(mktemp -d -t sgl-deep-ep-pypi.XXXXXX)"
    python -m wheel unpack "${tagged_wheels[0]}" --dest "${unpack_root}"
    unpacked="$(find "${unpack_root}" -mindepth 1 -maxdepth 1 -type d | head -1)"
    dist_info="$(find "${unpacked}" -maxdepth 1 -type d -name "*.dist-info" | head -1)"
    metadata="${dist_info}/METADATA"
    original_version="$(sed -n "s/^Version:[[:space:]]*//p" "${metadata}" | head -1)"
    public_version="${original_version%+cu130}"
    if [[ "${original_version}" == "${public_version}" ]]; then
        echo "CUDA 13 wheel metadata lacks the +cu130 local version" >&2
        exit 1
    fi
    sed -i "s/^Version:.*/Version: ${public_version}/" "${metadata}"
    old_dist_info="$(basename "${dist_info}")"
    new_dist_info="${old_dist_info/${original_version}/${public_version}}"
    mv "${dist_info}" "$(dirname "${dist_info}")/${new_dist_info}"
    python -m wheel pack "${unpacked}" --dest-dir /output/dist-pypi
fi

ls -lh /output/dist/*.whl
if [[ "${CUDA_TAG}" == cu130 ]]; then
    ls -lh /output/dist-pypi/*.whl
fi
'
