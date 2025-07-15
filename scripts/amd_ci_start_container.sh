#!/bin/bash
set -euo pipefail

# Set up DEVICE_FLAG based on Kubernetes pod info
if [ -f "/etc/podinfo/gha-render-devices" ]; then
  DEVICE_FLAG=$(cat /etc/podinfo/gha-render-devices)
else
  DEVICE_FLAG="--device /dev/dri"
fi

# Function to find latest available mi30x image
find_latest_mi30x_image() {
  local base_tag="v0.4.9.post2-rocm630-mi30x"
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

  echo "Error: No mi30x image found in the last 30 days" >&2
  return 1
}

# Function to find latest available mi35x image
find_latest_mi35x_image() {
  local base_tag="v0.4.9.post2-rocm700-mi35x"
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

  echo "Error: No mi35x image found in the last 30 days" >&2
  return 1
}

# Determine image finder and fallback based on runner
echo "The runner is: ${RUNNER_NAME:="unknown"}"
FIND_IMAGE_CMD="find_latest_mi30x_image"
FALLBACK_IMAGE="rocm/sgl-dev:v0.4.9.post2-rocm630-mi30x-20250715"
FALLBACK_MSG="No mi30x image found in last 30 days, using fallback image"

if [[ "${RUNNER_NAME}" == "linux-mi35x-gpu-x" ]]; then
  echo "Runner is ${RUNNER_NAME}, will find mi35x image."
  FIND_IMAGE_CMD="find_latest_mi35x_image"
  FALLBACK_IMAGE="rocm/sgl-dev:v0.4.9.post2-rocm700-mi35x-20250715"
  FALLBACK_MSG="No mi35x image found in last 30 days, using fallback image"
elif [[ "${RUNNER_NAME}" == "linux-mi300-gpu-x" || "${RUNNER_NAME}" == "linux-mi325-gpu-x" ]]; then
  echo "Runner is ${RUNNER_NAME}, will find mi30x image."
else
  echo "RUNNER_NAME env is not set or not recognized: '${RUNNER_NAME}'"
  echo "Defaulting to find mi30x image"
fi

# Find and pull the latest image
IMAGE=$($FIND_IMAGE_CMD)
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
