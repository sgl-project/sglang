#!/bin/bash
set -euo pipefail

ROCM_VERSION=${1:-}

if [[ "${ROCM_VERSION}" == "711" ]]; then
  IMAGE="rocm/pytorch:rocm7.1.1_ubuntu22.04_py3.10_pytorch_release_2.9.1"
else
  echo "ERROR: Unsupported ROCM_VERSION='${ROCM_VERSION}'. Only '711' is supported." >&2
  exit 1
fi

PYTHON_ROOT_PATH="/opt/venv/bin"
AMDGPU_TARGET="gfx942;gfx950"

# Pull and run the latest image
echo "Pulling Docker image: ${IMAGE}"
docker pull "${IMAGE}"

docker run --rm \
  -v $(pwd):/sgl-kernel \
  -e AMDGPU_TARGET="${AMDGPU_TARGET}" \
  -e PYTORCH_ROCM_ARCH="${AMDGPU_TARGET}" \
  ${IMAGE} \
  bash -c "
  # Install CMake (version >= 3.26) - Robust Installation
  export CMAKE_VERSION_MAJOR=3.31
  export CMAKE_VERSION_MINOR=1
  echo \"Downloading CMake from: https://cmake.org/files/v\${CMAKE_VERSION_MAJOR}/cmake-\${CMAKE_VERSION_MAJOR}.\${CMAKE_VERSION_MINOR}-linux-x86_64.tar.gz\"
  wget https://cmake.org/files/v\${CMAKE_VERSION_MAJOR}/cmake-\${CMAKE_VERSION_MAJOR}.\${CMAKE_VERSION_MINOR}-linux-x86_64.tar.gz
  tar -xzf cmake-\${CMAKE_VERSION_MAJOR}.\${CMAKE_VERSION_MINOR}-linux-x86_64.tar.gz
  mv cmake-\${CMAKE_VERSION_MAJOR}.\${CMAKE_VERSION_MINOR}-linux-x86_64 /opt/cmake
  export PATH=/opt/cmake/bin:\$PATH

  ${PYTHON_ROOT_PATH}/pip install --no-cache-dir ninja setuptools wheel numpy uv scikit-build-core && \

  cd /sgl-kernel && \
  rm -rf CMakeLists.txt && mv CMakeLists_rocm.txt CMakeLists.txt && \
  ${PYTHON_ROOT_PATH}/python rocm_hipify.py && \
  ${PYTHON_ROOT_PATH}/python -m uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation && \
  ./rename_wheels_rocm.sh
"
