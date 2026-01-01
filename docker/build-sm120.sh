#!/bin/bash
# Build script for SM120 MXFP4 support image
#
# Prerequisites:
#   1. Build wheels first: ./docker/build-wheels.sh
#
# Usage:
#   ./docker/build-sm120.sh [--push]
#
# Options:
#   --push    Push to registry after building

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEELS_DIR="${SCRIPT_DIR}/wheels"

REGISTRY="${REGISTRY:-registry.k8s.hq.droidcraft.org/droidcraft}"
IMAGE_NAME="${IMAGE_NAME:-sglang}"
TAG="${TAG:-sm120}"
BASE_IMAGE="${BASE_IMAGE:-public.ecr.aws/droidcraft/sglang:base}"

echo "=== SM120 MXFP4 Docker Build ==="
echo "Repository root: ${REPO_ROOT}"
echo "Wheels dir:      ${WHEELS_DIR}"
echo "Base image:      ${BASE_IMAGE}"
echo "Target image:    ${REGISTRY}/${IMAGE_NAME}:${TAG}"
echo ""

# Verify wheels exist
if [[ ! -d "${WHEELS_DIR}" ]] || [[ -z "$(ls -A ${WHEELS_DIR}/*.whl 2>/dev/null)" ]]; then
    echo "ERROR: No wheels found in ${WHEELS_DIR}"
    echo ""
    echo "Build wheels first with:"
    echo "  ./docker/build-wheels.sh"
    exit 1
fi

echo "Found wheels:"
ls -1 "${WHEELS_DIR}"/*.whl
echo ""

# Build the image
echo ""
echo "Building Docker image..."
cd "${REPO_ROOT}"

docker build \
    -f docker/Dockerfile.sm120 \
    -t "${REGISTRY}/${IMAGE_NAME}:${TAG}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    .

echo ""
echo "Build complete: ${REGISTRY}/${IMAGE_NAME}:${TAG}"

# Push if requested
if [[ "${1:-}" == "--push" ]]; then
    echo ""
    echo "Pushing to registry..."
    docker push "${REGISTRY}/${IMAGE_NAME}:${TAG}"
    echo "Push complete!"
fi
