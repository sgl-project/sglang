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
  local current_date=$(date +%Y%m%d)
  local days_back=0

  while [ $days_back -lt 30 ]; do
    local check_date=$(date -d "$days_back days ago" +%Y%m%d)
    local image_tag="${base_tag}-${check_date}"

    echo "Checking for image: rocm/sgl-dev:${image_tag}"

    # Check if the image exists by trying to get its manifest
    if docker manifest inspect "rocm/sgl-dev:${image_tag}" >/dev/null 2>&1; then
      echo "Found available image: rocm/sgl-dev:${image_tag}"
      echo "rocm/sgl-dev:${image_tag}"
      return 0
    fi

    days_back=$((days_back + 1))
  done

  echo "Error: No mi30x image found in the last 30 days" >&2
  return 1
}

# Find and pull the latest image
IMAGE=$(find_latest_mi30x_image)
if [ $? -eq 0 ]; then
  echo "Pulling Docker image: $IMAGE"
  docker pull "$IMAGE"
else
  echo "No mi30x image found in last 30 days, using fallback image"
  IMAGE="rocm/sgl-dev:v0.4.9.post2-rocm630-mi30x-20250715"
  echo "Pulling fallback Docker image: $IMAGE"
  docker pull "$IMAGE"
fi

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
