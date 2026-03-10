#!/bin/bash
set -euo pipefail

ROCM_VERSION=${1:-}

if [[ "${ROCM_VERSION}" == "700" ]]; then
  IMAGE="lmsysorg/sglang:v0.5.8.post1-rocm700-mi35x"
elif [[ "${ROCM_VERSION}" == "720" ]]; then
  IMAGE="rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1"
else
  echo "ERROR: Unsupported ROCM_VERSION='${ROCM_VERSION}'. Only '700' and '720' are supported." >&2
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
  # Install torch, triton, and friends, depending on the ROCm version
  if [[ "${ROCM_VERSION}" == "700" ]]; then
    ${PYTHON_ROOT_PATH}/pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/torch-2.9.1.dev20251204%2Brocm7.0.2.lw.git351ff442-cp310-cp310-linux_x86_64.whl https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/triton-3.5.1%2Brocm7.0.2.gita272dfa8-cp310-cp310-linux_x86_64.whl https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/torchaudio-2.9.0%2Brocm7.0.2.gite3c6ee2b-cp310-cp310-linux_x86_64.whl https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/torchvision-0.24.0%2Brocm7.0.2.gitb919bd0c-cp310-cp310-linux_x86_64.whl
  elif [[ "${ROCM_VERSION}" == "720" ]]; then
    ${PYTHON_ROOT_PATH}/pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp310-cp310-linux_x86_64.whl https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp310-cp310-linux_x86_64.whl https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchaudio-2.9.0%2Brocm7.2.0.gite3c6ee2b-cp310-cp310-linux_x86_64.whl https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchvision-0.24.0%2Brocm7.2.0.gitb919bd0c-cp310-cp310-linux_x86_64.whl
  fi
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
