#!/bin/bash
set -euo pipefail

ROCM_VERSION=${1:-}

if [[ "${ROCM_VERSION}" == "720" ]]; then
  IMAGE="rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1"
elif [[ "${ROCM_VERSION}" == "1000" ]]; then
  # Ubuntu 24.04 / Python 3.12 / torch 2.11 matches the stack the released
  # ROCm 10 images carry, and this image is built for every device arch, so one
  # wheel can still cover both gfx942 and gfx950.
  IMAGE="rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.11.0"
else
  echo "ERROR: Unsupported ROCM_VERSION='${ROCM_VERSION}'. Only '720' and '1000' are supported." >&2
  exit 1
fi

PYTHON_ROOT_PATH="/opt/venv/bin"
AMDGPU_TARGET="gfx942;gfx950"

# ROCm 10.0.0 is distributed as pip packages that unpack into site-packages, so
# the image has neither the /opt/rocm tree CMakeLists_rocm.txt looks for
# hip-lang under nor, being a runtime image, a devel tree to compile against.
# These are the same fixups docker/rocm.Dockerfile's rocm1000-base stage makes
# for the released images.
ROCM_SETUP=""
if [[ "${ROCM_VERSION}" == "1000" ]]; then
  ROCM_SETUP=$(cat <<'ROCM1000_SETUP'
  set -e
  apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates wget
  rocm_sdk_version=$(pip show rocm-sdk-core | awk '/^Version:/ {print $2}')
  test -n "${rocm_sdk_version}"
  pip install --no-cache-dir --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm-sdk-devel==${rocm_sdk_version}"
  rocm-sdk init
  export ROCM_HOME=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/_rocm_sdk_devel
  export ROCM_PATH="${ROCM_HOME}"
  export CPATH="${ROCM_HOME}/include"
  export LIBRARY_PATH="${ROCM_HOME}/lib"
  export LD_LIBRARY_PATH="${ROCM_HOME}/lib"
  export PATH="${ROCM_HOME}/llvm/bin:${ROCM_HOME}/bin:${PATH}"
  ln -sfn "${ROCM_HOME}" /opt/rocm
  # The SDK's hsakmtTargets.cmake hardcodes /usr/lib64/libc.so from its own
  # build host; Ubuntu keeps libc in /lib/x86_64-linux-gnu.
  mkdir -p /usr/lib64 && ln -sf /lib/x86_64-linux-gnu/libc.so /usr/lib64/libc.so
ROCM1000_SETUP
)
fi

# ROCm 10 has no /opt/rocm-<version> directory for rename_wheels_rocm.sh to read
# the local version tag from, so name it here instead. Keeping it in step with
# the `--rocm` argument release-whl-kernel.yml passes to
# scripts/update_kernel_whl_index.py is what puts the wheel in the index it
# gets published under.
WHEEL_ROCM_VERSION=""
if [[ "${ROCM_VERSION}" == "1000" ]]; then
  WHEEL_ROCM_VERSION="1000"
fi

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
  if [[ "${ROCM_VERSION}" == "720" ]]; then
    ${PYTHON_ROOT_PATH}/pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp310-cp310-linux_x86_64.whl https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp310-cp310-linux_x86_64.whl https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchaudio-2.9.0%2Brocm7.2.0.gite3c6ee2b-cp310-cp310-linux_x86_64.whl https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchvision-0.24.0%2Brocm7.2.0.gitb919bd0c-cp310-cp310-linux_x86_64.whl
  fi
${ROCM_SETUP}
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
  ./rename_wheels_rocm.sh ${WHEEL_ROCM_VERSION}
"
