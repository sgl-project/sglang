#!/bin/bash
set -euo pipefail
ROCM_VERSION=$1

PYTHON_ROOT_PATH="/opt/venv/bin"
AMDGPU_TARGET="gfx942;gfx950"

echo "Python root path is: $PYTHON_ROOT_PATH"

# Get version from git tags
SGLANG_VERSION="v0.5.6"   # Default version, will be overridden if git tags are found

# Fetch tags from origin to ensure we have the latest
if git fetch --tags origin; then
  # Get the latest version tag sorted by version number (e.g., v0.5.7)
  VERSION_FROM_TAG=$(git tag -l 'v[0-9]*' --sort=-v:refname | head -1)
  if [ -n "$VERSION_FROM_TAG" ]; then
    SGLANG_VERSION="$VERSION_FROM_TAG"
    echo "Using SGLang version from git tags: $SGLANG_VERSION"
  else
    echo "Warning: No version tags found; using default $SGLANG_VERSION" >&2
  fi
else
  echo "Warning: Failed to fetch tags from origin; using default $SGLANG_VERSION" >&2
fi

# Default base tags (can be overridden by command line arguments)
DEFAULT_MI30X_BASE_TAG="${SGLANG_VERSION}-rocm700-mi30x"
DEFAULT_MI35X_BASE_TAG="${SGLANG_VERSION}-rocm700-mi35x"

# Parse command line arguments
MI30X_BASE_TAG="${DEFAULT_MI30X_BASE_TAG}"
MI35X_BASE_TAG="${DEFAULT_MI35X_BASE_TAG}"

# Detect GPU architecture from the Kubernetes runner hostname
HOSTNAME_VALUE=$(hostname)
GPU_ARCH="mi30x"   # default

# Host names look like: linux-mi35x-gpu-1-xxxxx-runner-zzzzz
if [[ "${HOSTNAME_VALUE}" =~ ^linux-(mi[0-9]+[a-z]*)-gpu-[0-9]+ ]]; then
  GPU_ARCH="${BASH_REMATCH[1]}"
  echo "Detected GPU architecture from hostname: ${GPU_ARCH}"
else
  echo "Warning: could not parse GPU architecture from '${HOSTNAME_VALUE}', defaulting to ${GPU_ARCH}"
fi

case "${GPU_ARCH}" in
  mi35x)
    echo "Runner uses ${GPU_ARCH}; will fetch mi35x image."
    ;;
  mi30x|mi300|mi325)
    echo "Runner uses ${GPU_ARCH}; will fetch mi30x image."
    GPU_ARCH="mi30x"
    ;;
  *)
    echo "Runner architecture '${GPU_ARCH}' unrecognised; defaulting to mi30x image." >&2
    GPU_ARCH="mi30x"
    ;;
esac

if [[ -f /etc/podinfo/gha-render-devices ]]; then
  DEVICE_FLAG=$(cat /etc/podinfo/gha-render-devices)
else
  DEVICE_FLAG="--device /dev/dri"
fi

# Find the latest image
find_latest_image() {
  local gpu_arch=$1
  local base_tag days_back image_tag

  case "${gpu_arch}" in
      mi30x) base_tag="${MI30X_BASE_TAG}" ;;
      mi35x) base_tag="${MI35X_BASE_TAG}" ;;
      *)     echo "Error: unsupported GPU architecture '${gpu_arch}'" >&2; return 1 ;;
  esac

  for days_back in {0..6}; do
    image_tag="${base_tag}-$(date -d "${days_back} days ago" +%Y%m%d)"
    echo "Checking for image: rocm/sgl-dev:${image_tag}" >&2
    if docker manifest inspect "rocm/sgl-dev:${image_tag}" >/dev/null 2>&1; then
      echo "Found available image: rocm/sgl-dev:${image_tag}" >&2
      echo "rocm/sgl-dev:${image_tag}"
      return 0
    fi
  done

  echo "Error: no ${gpu_arch} image found in the last 7 days for base ${base_tag}" >&2
  echo "Using hard-coded fallback…" >&2
  if [[ "${gpu_arch}" == "mi35x" ]]; then
    echo "rocm/sgl-dev:v0.5.3-rocm700-mi35x-20251009"
  else
    echo "rocm/sgl-dev:v0.5.3-rocm700-mi30x-20251009"
  fi
}

# Pull and run the latest image
IMAGE=$(find_latest_image "${GPU_ARCH}")
echo "Pulling Docker image: ${IMAGE}"
docker pull "${IMAGE}"

docker run --rm \
  -v $(pwd):/sgl-kernel \
  -e AMDGPU_TARGET="${AMDGPU_TARGET}" \
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
