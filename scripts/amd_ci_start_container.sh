#!/bin/bash
set -euo pipefail

# Set up DEVICE_FLAG based on Kubernetes pod info
if [ -f "/etc/podinfo/gha-render-devices" ]; then
  DEVICE_FLAG=$(cat /etc/podinfo/gha-render-devices)
else
  DEVICE_FLAG="--device /dev/dri"
fi

# Pull the image
IMAGE="lmsysorg/sglang:v0.4.6.post5-rocm630"
echo "Pulling Docker image: $IMAGE"
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
