#!/bin/bash
set -euo pipefail

# Get version from git tags
SGLANG_VERSION="v0.5.5"   # Default version, will be overridden if git tags are found

# Fetch tags from origin to ensure we have the latest
if git fetch --tags origin; then
  # Use the shared helper so stable/post releases sort above rc tags.
  VERSION_FROM_TAG=$(python3 python/tools/get_version_tag.py --tag-only || true)
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
XPU_REGISTRY="gar-registry.caas.intel.com/sglang/sglang-xpu"
DEFAULT_BMG_BASE_TAG="${SGLANG_VERSION}-xpu-bmg"

# Parse command line arguments
BMG_BASE_TAG="${DEFAULT_BMG_BASE_TAG}"
CUSTOM_IMAGE=""
BUILD_FROM_DOCKERFILE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --bmg-base-tag) BMG_BASE_TAG="$2"; shift 2;;
    --custom-image) CUSTOM_IMAGE="$2"; shift 2;;
    --build-from-dockerfile) BUILD_FROM_DOCKERFILE="1"; shift;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  --bmg-base-tag TAG         Override BMG base image tag"
      echo "  --custom-image IMAGE       Use a specific Docker image directly"
      echo "  --build-from-dockerfile    Build image from docker/xpu.Dockerfile"
      exit 0
      ;;
    *) echo "Unknown option $1"; exit 1;;
  esac
done


# Detect GPU architecture (single-arch today, but kept for parity with AMD)
GPU_ARCH="bmg"

# Retry a command with exponential backoff. Usage: retry_with_backoff <max_attempts> <cmd...>
retry_with_backoff() {
  local max_attempts=$1; shift
  local attempt=1
  local wait_secs=30
  # Add jitter (0-30s) so concurrent jobs don't all retry at the same instant
  local jitter=$(( RANDOM % 30 ))
  while true; do
    if "$@"; then
      return 0
    fi
    if (( attempt >= max_attempts )); then
      echo "Error: '$*' failed after ${max_attempts} attempts" >&2
      return 1
    fi
    local sleep_time=$(( wait_secs + jitter ))
    echo "Attempt ${attempt}/${max_attempts} failed. Retrying in ${sleep_time}s…" >&2
    sleep "${sleep_time}"
    (( attempt++ ))
    (( wait_secs = wait_secs * 2 > 300 ? 300 : wait_secs * 2 ))
    jitter=$(( RANDOM % 30 ))
  done
}

# Authenticate to the Intel Harbor registry to avoid anonymous pull issues.
# Credentials are optional; when absent we fall back to unauthenticated pulls.
if [[ -n "${INTEL_HARBOR_USERNAME:-}" && -n "${INTEL_HARBOR_TOKEN:-}" ]]; then
  echo "Logging in to gar-registry.caas.intel.com…"
  if retry_with_backoff 6 sh -c 'echo "${INTEL_HARBOR_TOKEN}" | docker login gar-registry.caas.intel.com -u "${INTEL_HARBOR_USERNAME}" --password-stdin >/dev/null 2>&1'; then
    echo "Intel Harbor login successful"
  else
    echo "Warning: Intel Harbor login failed after retries; continuing with unauthenticated pulls" >&2
  fi
fi

# Find the latest image
find_latest_image() {
  local gpu_arch=$1
  local base_tag days_back image_tag image_id

  case "${gpu_arch}" in
      bmg)   base_tag="${BMG_BASE_TAG}" ;;
      *)     echo "Error: unsupported GPU architecture '${gpu_arch}'" >&2; return 1 ;;
  esac

  # First, check local cache on the runner.
  for days_back in {0..6}; do
    image_tag="${base_tag}-$(date -d "${days_back} days ago" +%Y%m%d)"
    image_id=$(docker images -q "${XPU_REGISTRY}:${image_tag}")
    if [[ -n "$image_id" ]]; then
      echo "Found cached image locally: ${XPU_REGISTRY}:${image_tag}" >&2
      echo "${XPU_REGISTRY}:${image_tag}"
      return 0
    fi
  done

  # If not found locally, probe the remote registry by manifest.
  for days_back in {0..6}; do
    image_tag="${base_tag}-$(date -d "${days_back} days ago" +%Y%m%d)"
    echo "Checking for image: ${XPU_REGISTRY}:${image_tag}" >&2
    if docker manifest inspect "${XPU_REGISTRY}:${image_tag}" >/dev/null 2>&1; then
      echo "Found available image: ${XPU_REGISTRY}:${image_tag}" >&2
      echo "${XPU_REGISTRY}:${image_tag}"
      return 0
    fi
  done

  echo "No recent images found. Searching any cached local images matching ${gpu_arch}…" >&2
  local any_local
  any_local=$(docker images --format '{{.Repository}}:{{.Tag}}' --filter "reference=${XPU_REGISTRY}:*xpu-${gpu_arch}*" | sort -r | head -n 1)
  if [[ -n "$any_local" ]]; then
      echo "Using cached fallback image: ${any_local}" >&2
      echo "${any_local}"
      return 0
  fi

  echo "Error: no ${gpu_arch} image found in the last 7 days for base ${base_tag}" >&2
  return 1
}

# Determine which image to use
if [[ -n "${CUSTOM_IMAGE}" ]]; then
  # Use explicitly provided custom image
  IMAGE="${CUSTOM_IMAGE}"
  echo "Using custom image: ${IMAGE}"
  retry_with_backoff 6 docker pull "${IMAGE}"
elif [[ -n "${BUILD_FROM_DOCKERFILE}" ]]; then
  # Build image from Dockerfile
  DOCKERFILE_DIR="${GITHUB_WORKSPACE:-$PWD}/docker"
  DOCKERFILE="${DOCKERFILE_DIR}/xpu.Dockerfile"

  if [[ ! -f "${DOCKERFILE}" ]]; then
    echo "Error: Dockerfile not found at ${DOCKERFILE}" >&2
    exit 1
  fi

  IMAGE="sglang-xpu-ci:${GPU_ARCH}-$(date +%Y%m%d)"
  echo "Building Docker image from ${DOCKERFILE}..."

  PR_REPO="${PR_REPO:-}"
  PR_HEAD_REF="${PR_HEAD_REF:-}"
  docker build \
    ${PR_REPO:+--build-arg SG_LANG_REPO=$PR_REPO} \
    ${PR_HEAD_REF:+--build-arg SG_LANG_BRANCH=$PR_HEAD_REF} \
    -t "${IMAGE}" \
    -f "${DOCKERFILE}" \
    --no-cache --progress=plain \
    "${GITHUB_WORKSPACE:-$PWD}"
  echo "Successfully built image: ${IMAGE}"
else
  # Find the latest pre-built image
  IMAGE=$(find_latest_image "${GPU_ARCH}")
  retry_with_backoff 6 docker pull "${IMAGE}"
fi

# HF token can be provided via env or via the token file used by pr-test-xpu.yml.
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/huggingface_token.txt" ]]; then
  HF_TOKEN="$(cat "$HOME/huggingface_token.txt")"
fi

echo "Launching container: ci_sglang_xpu"
docker run -dt \
  --group-add 992 \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path \
  -v "${GITHUB_WORKSPACE:-$PWD}:/sglang-checkout" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --shm-size 32g \
  --cap-add=SYS_PTRACE \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HUB_ETAG_TIMEOUT=300 \
  -e HF_HUB_DOWNLOAD_TIMEOUT=300 \
  --security-opt seccomp=unconfined \
  -w /sglang-checkout \
  --name ci_sglang_xpu \
  "${IMAGE}"

# Mark the mounted checkout safe so setuptools-scm / vcs_versioning can resolve
# the package version (the runner mount is owned by a non-root user but the
# container may run as root for some operations).
docker exec ci_sglang_xpu git config --global --add safe.directory /sglang-checkout
