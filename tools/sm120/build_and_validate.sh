#!/usr/bin/env bash
# Build the SM120 tilelang paged-MQA-logits validator image for a given sglang
# fork branch and run it. Run this from tools/sm120/.
#
#   ./build_and_validate.sh [BRANCH] [BASE_IMAGE]
#
# Examples:
#   # default: the experiment branch, override base image to your working one
#   ./build_and_validate.sh sm120-tilelang-experiment lmsysorg/sglang:v0.5.5-cu128
#
#   # build only (e.g. on a box without a GPU), then push to a registry:
#   NO_RUN=1 ./build_and_validate.sh sm120-tilelang-experiment <base-image>
#
set -euo pipefail

BRANCH="${1:-sm120-tilelang-experiment}"
BASE_IMAGE="${2:-${SGLANG_IMAGE:-lmsysorg/sglang:latest}}"
REPO="${SGLANG_REPO:-https://github.com/shivajid/sglang.git}"
IMAGE_TAG="${IMAGE_TAG:-sm120-validator:${BRANCH}}"

echo ">> Building ${IMAGE_TAG}"
echo "   repo=${REPO} branch=${BRANCH} base=${BASE_IMAGE}"
docker build -f Dockerfile \
  --build-arg SGLANG_IMAGE="${BASE_IMAGE}" \
  --build-arg SGLANG_REPO="${REPO}" \
  --build-arg SGLANG_BRANCH="${BRANCH}" \
  --build-arg CACHEBUST="$(date +%s)" \
  -t "${IMAGE_TAG}" .

if [ "${NO_RUN:-0}" = "1" ]; then
  echo ">> NO_RUN=1 set; skipping run. Image built: ${IMAGE_TAG}"
  echo "   To run on the SM120 box:  docker run --rm --gpus all ${IMAGE_TAG}"
  exit 0
fi

echo ">> Running validator (requires an SM120 GPU on this host)"
docker run --rm --gpus all "${IMAGE_TAG}"
