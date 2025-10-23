#!/bin/bash
set -euo pipefail

# Get version from SGLang version.py file
SGLANG_VERSION_FILE="$(dirname "$0")/../../python/sglang/version.py"
SGLANG_VERSION="v0.5.0rc0"   # Default version, will be overridden if version.py is found

if [ -f "$SGLANG_VERSION_FILE" ]; then
  VERSION_FROM_FILE=$(python3 -c '
import re, sys
with open(sys.argv[1], "r") as f:
    content = f.read()
    match = re.search(r"__version__\s*=\s*[\"'"'"'](.*?)[\"'"'"']", content)
    if match:
        print("v" + match.group(1))
' "$SGLANG_VERSION_FILE" 2>/dev/null || echo "")

  if [ -n "$VERSION_FROM_FILE" ]; then
      SGLANG_VERSION="$VERSION_FROM_FILE"
      echo "Using SGLang version from version.py: $SGLANG_VERSION"
  else
      echo "Warning: Could not parse version from $SGLANG_VERSION_FILE, using default: $SGLANG_VERSION" >&2
  fi
else
  echo "Warning: version.py not found, using default version: $SGLANG_VERSION" >&2
fi


# Default base tags (can be overridden by command line arguments)
DEFAULT_MI30X_BASE_TAG="${SGLANG_VERSION}-rocm630-mi30x"
DEFAULT_MI35X_BASE_TAG="${SGLANG_VERSION}-rocm700-mi35x"

# Parse command line arguments
MI30X_BASE_TAG="${DEFAULT_MI30X_BASE_TAG}"
MI35X_BASE_TAG="${DEFAULT_MI35X_BASE_TAG}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --mi30x-base-tag) MI30X_BASE_TAG="$2"; shift 2;;
    --mi35x-base-tag) MI35X_BASE_TAG="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 [--mi30x-base-tag TAG] [--mi35x-base-tag TAG]"
      exit 0
      ;;
    *) echo "Unknown option $1"; exit 1;;
  esac
done



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

# Normalise / collapse architectures we don’t yet build specifically for
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


# Set up DEVICE_FLAG based on Kubernetes pod info
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

  # First, check local cache
  for days_back in {0..6}; do
    image_tag="${base_tag}-$(date -d "${days_back} days ago" +%Y%m%d)"
    local local_image="rocm/sgl-dev:${image_tag}"
    image_id=$(docker images -q "${local_image}")
    if [[ -n "$image_id" ]]; then
        echo "Found cached image locally: ${local_image}" >&2
        echo "${local_image}"
        return 0
    fi
  done

  # If not found locally, fall back to pulling from public registry
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
    echo "rocm/sgl-dev:v0.5.0rc0-rocm700-mi35x-20250812"
  else
    echo "rocm/sgl-dev:v0.5.0rc0-rocm630-mi30x-20250812"
  fi
}

# Pull and run the latest image
IMAGE=$(find_latest_image "${GPU_ARCH}")
echo "Pulling Docker image: ${IMAGE}"
docker pull "${IMAGE}"

HF_CACHE_HOST=/home/runner/sgl-data/hf-cache
if [[ -d "$HF_CACHE_HOST" ]]; then
    CACHE_VOLUME="-v $HF_CACHE_HOST:/hf_home"
else
    CACHE_VOLUME=""
fi

echo "Launching container: ci_sglang"
docker run -dt --user root --device=/dev/kfd ${DEVICE_FLAG} \
  -v "${GITHUB_WORKSPACE:-$PWD}:/sglang-checkout" \
  $CACHE_VOLUME \
  --ipc=host --group-add video \
  --shm-size 32g \
  --cap-add=SYS_PTRACE \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HOME=/hf_home \
  --security-opt seccomp=unconfined \
  -w /sglang-checkout \
  --name ci_sglang \
  "${IMAGE}"
