#!/bin/bash
set -euo pipefail

# Default base tags (can be overridden by command line arguments)
DEFAULT_MI30X_BASE_TAG="v0.4.9.post2-rocm630-mi30x"
DEFAULT_MI35X_BASE_TAG="v0.4.9.post2-rocm700-mi35x"

# Parse command line arguments
MI30X_BASE_TAG="$DEFAULT_MI30X_BASE_TAG"
MI35X_BASE_TAG="$DEFAULT_MI35X_BASE_TAG"

while [[ $# -gt 0 ]]; do
  case $1 in
    --mi30x-base-tag)
      MI30X_BASE_TAG="$2"
      shift 2
      ;;
    --mi35x-base-tag)
      MI35X_BASE_TAG="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--mi30x-base-tag TAG] [--mi35x-base-tag TAG]"
      echo "  --mi30x-base-tag TAG    Base tag for mi30x images (default: $DEFAULT_MI30X_BASE_TAG)"
      echo "  --mi35x-base-tag TAG    Base tag for mi35x images (default: $DEFAULT_MI35X_BASE_TAG)"
      exit 0
      ;;
    *)
      echo "Unknown option $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Set up DEVICE_FLAG based on Kubernetes pod info
if [ -f "/etc/podinfo/gha-render-devices" ]; then
  DEVICE_FLAG=$(cat /etc/podinfo/gha-render-devices)
else
  DEVICE_FLAG="--device /dev/dri"
fi

# Function to find latest available image for a given GPU architecture
find_latest_image() {
  local gpu_arch=$1
  local base_tag

  if [ "$gpu_arch" == "mi30x" ]; then
    base_tag="$MI30X_BASE_TAG"
  elif [ "$gpu_arch" == "mi35x" ]; then
    base_tag="$MI35X_BASE_TAG"
  else
    echo "Error: Unsupported GPU architecture '$gpu_arch'" >&2
    return 1
  fi

  local days_back=0

  while [ $days_back -lt 30 ]; do
    local check_date=$(date -d "$days_back days ago" +%Y%m%d)
    local image_tag="${base_tag}-${check_date}"

    echo "Checking for image: rocm/sgl-dev:${image_tag}" >&2

    # Check if the image exists by trying to get its manifest
    if docker manifest inspect "rocm/sgl-dev:${image_tag}" >/dev/null 2>&1; then
      echo "Found available image: rocm/sgl-dev:${image_tag}" >&2
      echo "rocm/sgl-dev:${image_tag}"
      return 0
    fi

    days_back=$((days_back + 1))
  done

  echo "Error: No ${gpu_arch} image found in the last 30 days" >&2
  return 1
}

# Determine image finder and fallback based on runner
# In Kubernetes, the hostname contains the GPU type (e.g., linux-mi300-gpu-1-bgg8r-runner-vknlb)
# Extract the GPU type from hostname
HOSTNAME_VALUE=$(hostname)
RUNNER_NAME="unknown"

if [[ "${HOSTNAME_VALUE}" =~ ^(linux-mi[0-9]+-gpu-[0-9]+) ]]; then
  RUNNER_NAME="${BASH_REMATCH[1]}"
  echo "Extracted runner from hostname: ${RUNNER_NAME}"
else
  echo "Could not extract runner info from hostname: ${HOSTNAME_VALUE}"
fi

echo "The runner is: ${RUNNER_NAME}"
GPU_ARCH="mi30x"
FALLBACK_IMAGE="rocm/sgl-dev:${MI30X_BASE_TAG}-20250715"
FALLBACK_MSG="No mi30x image found in last 30 days, using fallback image"

# Check for mi350/mi355 runners
if [[ "${RUNNER_NAME}" =~ ^linux-mi350-gpu-[0-9]+$ ]] || [[ "${RUNNER_NAME}" =~ ^linux-mi355-gpu-[0-9]+$ ]]; then
  echo "Runner is ${RUNNER_NAME}, will find mi35x image."
  GPU_ARCH="mi35x"
  FALLBACK_IMAGE="rocm/sgl-dev:${MI35X_BASE_TAG}-20250715"
  FALLBACK_MSG="No mi35x image found in last 30 days, using fallback image"
# Check for mi300/mi325 runners
elif [[ "${RUNNER_NAME}" =~ ^linux-mi300-gpu-[0-9]+$ ]] || [[ "${RUNNER_NAME}" =~ ^linux-mi325-gpu-[0-9]+$ ]]; then
  echo "Runner is ${RUNNER_NAME}, will find mi30x image."
else
  echo "Runner type not recognized: '${RUNNER_NAME}'"
  echo "Defaulting to find mi30x image"
fi

# Find and pull the latest image
IMAGE=$(find_latest_image "${GPU_ARCH}")
if [ $? -eq 0 ]; then
  echo "Pulling Docker image: $IMAGE"
else
  echo "$FALLBACK_MSG" >&2
  IMAGE="$FALLBACK_IMAGE"
  echo "Pulling fallback Docker image: $IMAGE"
fi
docker pull "$IMAGE"

# Run the container
echo "Starting container: ci_sglang"
docker run -dt --user root --device=/dev/kfd $DEVICE_FLAG \
  -v "${GITHUB_WORKSPACE:-$PWD}:/sglang-checkout" \
  --ipc=host --group-add video \
  --cap-add=SYS_PTRACE \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  --security-opt seccomp=unconfined \
  -w /sglang-checkout \
  --name ci_sglang \
  "$IMAGE"
