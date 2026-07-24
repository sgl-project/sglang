#!/bin/bash
set -euo pipefail

# Start the Intel XPU CI container (ci_sglang_xpu) using the intel/sglang-dev:latest
# image published by .github/workflows/release-docker-intel-xpu-nightly.yml.
#
# Pulls the :latest tag and starts a long-running container that subsequent
# steps `docker exec` into.

CONTAINER_NAME="ci_sglang_xpu"
IMAGE_REPO="intel/sglang-dev"
IMAGE_TAG="latest"
CUSTOM_IMAGE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --custom-image) CUSTOM_IMAGE="$2"; shift 2;;
    --container-name) CONTAINER_NAME="$2"; shift 2;;
    --image-tag) IMAGE_TAG="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  --custom-image IMAGE     Use a specific Docker image directly"
      echo "  --container-name NAME    Override container name (default: ${CONTAINER_NAME})"
      echo "  --image-tag TAG          Tag of ${IMAGE_REPO} to pull (default: ${IMAGE_TAG})"
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

if [[ -n "${CUSTOM_IMAGE}" ]]; then
  IMAGE="${CUSTOM_IMAGE}"
  echo "Using custom image: ${IMAGE}"
else
  IMAGE="${IMAGE_REPO}:${IMAGE_TAG}"
  echo "Using image: ${IMAGE}"
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
# SGLANG_SERVER_LAUNCH_TIMEOUT=36000 matches /data/pgirijal/scripts/setup_upstream_env.sh:
# 4-GPU MoE loads (Qwen3.5-35B-A3B, gemma-4-26B-A4B, ...) on Arc Pro B60 can
# take >1h from a cold HF cache, so give sglang server startup a 10h ceiling.
docker run -dt \
  --shm-size 8g \
  --group-add 992 \
  ${VIDEO_GID:+--group-add "${VIDEO_GID}"} \
  ${RENDER_GID:+--group-add "${RENDER_GID}"} \
  --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -v "${GITHUB_WORKSPACE:-$PWD}:/sglang-checkout" \
  -e HF_TOKEN="${HF_TOKEN_VALUE}" \
  -e SGLANG_SERVER_LAUNCH_TIMEOUT=36000 \
  --name "${CONTAINER_NAME}" \
  "${IMAGE}"

# Mark the workspace mount as a safe directory so git operations as root
# inside the container don't trip the cross-user repo guard.
docker exec "${CONTAINER_NAME}" git config --global --add safe.directory /sglang-checkout || true
