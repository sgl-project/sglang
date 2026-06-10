#!/bin/bash
set -euo pipefail

# Start the Intel XPU CI container (ci_sglang_xpu) using the latest nightly
# intel/sglang-dev image published by .github/workflows/release-docker-intel-xpu-nightly.yml.
#
# Walks back N days through nightly date-stamped tags, pulls the first match,
# then starts a long-running container that subsequent steps `docker exec` into.

CONTAINER_NAME="ci_sglang_xpu"
IMAGE_REPO="intel/sglang-dev"
TAG_PREFIX="nightly-dev-xpu-bmg"
LOOKBACK_DAYS=7
CUSTOM_IMAGE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --custom-image) CUSTOM_IMAGE="$2"; shift 2;;
    --container-name) CONTAINER_NAME="$2"; shift 2;;
    --lookback-days) LOOKBACK_DAYS="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  --custom-image IMAGE     Use a specific Docker image directly"
      echo "  --container-name NAME    Override container name (default: ${CONTAINER_NAME})"
      echo "  --lookback-days N        Days of nightly tags to scan (default: ${LOOKBACK_DAYS})"
      exit 0
      ;;
    *) echo "Unknown option $1"; exit 1;;
  esac
done

# Retry a command with exponential backoff. Usage: retry_with_backoff <max_attempts> <cmd...>
retry_with_backoff() {
  local max_attempts=$1; shift
  local attempt=1
  local wait_secs=30
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
    echo "Attempt ${attempt}/${max_attempts} failed. Retrying in ${sleep_time}s..." >&2
    sleep "${sleep_time}"
    (( attempt++ ))
    (( wait_secs = wait_secs * 2 > 300 ? 300 : wait_secs * 2 ))
    jitter=$(( RANDOM % 30 ))
  done
}

# Authenticate to Docker Hub when credentials are present (avoids anonymous pull
# rate limits). Both vars are optional; falls back to unauthenticated pulls.
if [[ -n "${DOCKERHUB_INTEL_USERNAME:-}" && -n "${DOCKERHUB_INTEL_TOKEN:-}" ]]; then
  echo "Logging in to Docker Hub..."
  if retry_with_backoff 6 sh -c 'echo "${DOCKERHUB_INTEL_TOKEN}" | docker login -u "${DOCKERHUB_INTEL_USERNAME}" --password-stdin >/dev/null 2>&1'; then
    echo "Docker Hub login successful"
  else
    echo "Warning: Docker Hub login failed after retries; continuing with unauthenticated pulls" >&2
  fi
fi

# Locate the latest published nightly image. Walks back LOOKBACK_DAYS days of
# date-stamped tags (commit-hash suffix is unknown, so we query Docker Hub).
find_latest_image() {
  local days_back target_date matched_tag

  # Local cache first.
  for (( days_back=0; days_back<LOOKBACK_DAYS; days_back++ )); do
    target_date=$(date -d "${days_back} days ago" +%Y%m%d)
    matched_tag=$(docker images --format '{{.Repository}}:{{.Tag}}' \
      --filter "reference=${IMAGE_REPO}:${TAG_PREFIX}-${target_date}-*" \
      | sort -r | head -n 1)
    if [[ -n "${matched_tag}" ]]; then
      echo "Found cached image locally: ${matched_tag}" >&2
      echo "${matched_tag}"
      return 0
    fi
  done

  # Then query Docker Hub for any tag matching the date prefix.
  for (( days_back=0; days_back<LOOKBACK_DAYS; days_back++ )); do
    target_date=$(date -d "${days_back} days ago" +%Y%m%d)
    echo "Checking Docker Hub for: ${IMAGE_REPO}:${TAG_PREFIX}-${target_date}-*" >&2
    matched_tag=$(curl -fsSL \
      "https://registry.hub.docker.com/v2/repositories/${IMAGE_REPO}/tags?page_size=100&name=${TAG_PREFIX}-${target_date}" \
      2>/dev/null \
      | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | head -n 1 || true)
    if [[ -n "${matched_tag}" ]]; then
      echo "Found published image: ${IMAGE_REPO}:${matched_tag}" >&2
      echo "${IMAGE_REPO}:${matched_tag}"
      return 0
    fi
  done

  # Final fallback: any locally cached image matching the prefix.
  matched_tag=$(docker images --format '{{.Repository}}:{{.Tag}}' \
    --filter "reference=${IMAGE_REPO}:${TAG_PREFIX}-*" \
    | sort -r | head -n 1)
  if [[ -n "${matched_tag}" ]]; then
    echo "Using cached fallback image: ${matched_tag}" >&2
    echo "${matched_tag}"
    return 0
  fi

  echo "Error: no ${IMAGE_REPO}:${TAG_PREFIX}-* image found in the last ${LOOKBACK_DAYS} days" >&2
  return 1
}

if [[ -n "${CUSTOM_IMAGE}" ]]; then
  IMAGE="${CUSTOM_IMAGE}"
  echo "Using custom image: ${IMAGE}"
else
  IMAGE=$(find_latest_image)
fi
# Always pull so each stage runs the registry's current image; the cleanup
# step removes the image after the stage so the runner doesn't accumulate
# stale layers across runs.
retry_with_backoff 6 docker pull "${IMAGE}"

# Export the resolved image so the cleanup step can rmi the exact tag used.
if [[ -n "${GITHUB_ENV:-}" ]]; then
  echo "CI_SGLANG_XPU_IMAGE=${IMAGE}" >> "${GITHUB_ENV}"
fi

# Remove any stale container of the same name so re-runs are idempotent.
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Removing existing container: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)

HF_TOKEN_FILE="${HOME}/huggingface_token.txt"
HF_TOKEN_VALUE=""
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_VALUE="${HF_TOKEN}"
elif [[ -r "${HF_TOKEN_FILE}" ]]; then
  HF_TOKEN_VALUE=$(cat "${HF_TOKEN_FILE}")
fi

echo "Launching container: ${CONTAINER_NAME} from ${IMAGE}"
docker run -dt \
  --group-add 992 \
  ${VIDEO_GID:+--group-add "${VIDEO_GID}"} \
  ${RENDER_GID:+--group-add "${RENDER_GID}"} \
  --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -v "${GITHUB_WORKSPACE:-$PWD}:/sglang-checkout" \
  -e HF_TOKEN="${HF_TOKEN_VALUE}" \
  --name "${CONTAINER_NAME}" \
  "${IMAGE}"

# Mark the workspace mount as a safe directory so git operations as root
# inside the container don't trip the cross-user repo guard.
docker exec "${CONTAINER_NAME}" git config --global --add safe.directory /sglang-checkout || true
