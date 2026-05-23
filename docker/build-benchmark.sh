#!/usr/bin/env bash
# Build the sglang-benchmark client image.
#
# Usage:
#   docker/build-benchmark.sh             # build for host arch, load locally
#   docker/build-benchmark.sh multiarch   # multi-arch (linux/amd64,linux/arm64); requires PUSH=1
#
# Env:
#   TAG        image tag (default: latest)
#   IMG_NAME   image name (default: sglang-benchmark)
#   PUSH=1     push instead of --load (required for multi-arch builds)
#   REGISTRY   prepended to image name when set (e.g. ghcr.io/yourname)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

TAG="${TAG:-latest}"
IMG_NAME="${IMG_NAME:-sglang-benchmark}"
PUSH="${PUSH:-0}"
REGISTRY="${REGISTRY:-}"

host_platform() {
  case "$(uname -m)" in
    x86_64|amd64)  echo "linux/amd64" ;;
    aarch64|arm64) echo "linux/arm64" ;;
    *)             echo "linux/$(uname -m)" ;;
  esac
}

full_image_name() {
  if [[ -n "${REGISTRY}" ]]; then
    echo "${REGISTRY}/${IMG_NAME}:${TAG}"
  else
    echo "${IMG_NAME}:${TAG}"
  fi
}

output_flag() {
  if [[ "${PUSH}" == "1" ]]; then echo "--push"; else echo "--load"; fi
}

build_host() {
  docker buildx build \
    --platform "$(host_platform)" \
    --file "${HERE}/sglang-benchmark-client.Dockerfile" \
    --tag "$(full_image_name)" \
    $(output_flag) \
    "${ROOT}"
}

build_multiarch() {
  if [[ "${PUSH}" != "1" ]]; then
    echo "multiarch builds require PUSH=1 since --load cannot accept multi-platform manifests" >&2
    exit 1
  fi
  docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --file "${HERE}/sglang-benchmark-client.Dockerfile" \
    --tag "$(full_image_name)" \
    --push \
    "${ROOT}"
}

case "${1:-host}" in
  host|"")    build_host ;;
  multiarch)  build_multiarch ;;
  *)          sed -n '2,8p' "$0" >&2; exit 2 ;;
esac
