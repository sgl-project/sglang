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

# Forward ZE_AFFINITY_MASK so each runner pins to its own GPU (else all pile onto L0 dev 0).
# ONEAPI_DEVICE_SELECTOR keeps SYCL consistent with the L0-filtered device.
GPU_AFFINITY_ARGS=()
if [[ -n "${ZE_AFFINITY_MASK:-}" ]]; then
  echo "Pinning container to GPU via ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}"
  GPU_AFFINITY_ARGS+=(-e "ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}")
  GPU_AFFINITY_ARGS+=(-e "ONEAPI_DEVICE_SELECTOR=level_zero:0")
else
  echo "Warning: ZE_AFFINITY_MASK is not set; container will default to GPU 0." >&2
  echo "         Set ZE_AFFINITY_MASK per runner to spread jobs across GPUs." >&2
fi

HF_TOKEN_FILE="${HOME}/huggingface_token.txt"
HF_TOKEN_VALUE=""
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_VALUE="${HF_TOKEN}"
elif [[ -r "${HF_TOKEN_FILE}" ]]; then
  HF_TOKEN_VALUE=$(cat "${HF_TOKEN_FILE}")
fi

# Persistent JIT kernel cache (Triton/Inductor/NEO/SYCL) keyed by GPU mask.
# Cold JIT compile can push test_xpu_basic past its 1200s timeout on B580.
XPU_KERNEL_CACHE_HOST="${XPU_KERNEL_CACHE_DIR:-${HOME}/.cache/sglang-xpu-ci/kernel-cache-gpu${ZE_AFFINITY_MASK:-shared}}"
mkdir -p "${XPU_KERNEL_CACHE_HOST}"/{triton,inductor,neo,sycl}
echo "Using persistent XPU kernel cache: ${XPU_KERNEL_CACHE_HOST}"

# Cap the cache (default 5 GiB); over-cap resets it (misses just recompile).
XPU_KERNEL_CACHE_MAX_MB="${XPU_KERNEL_CACHE_MAX_MB:-5120}"
cache_mb=$(du -sm "${XPU_KERNEL_CACHE_HOST}" 2>/dev/null | cut -f1)
if [[ -n "${cache_mb}" && "${cache_mb}" -gt "${XPU_KERNEL_CACHE_MAX_MB}" ]]; then
  echo "XPU kernel cache is ${cache_mb} MiB (> ${XPU_KERNEL_CACHE_MAX_MB} MiB cap); resetting it."
  rm -rf "${XPU_KERNEL_CACHE_HOST:?}"/{triton,inductor,neo,sycl}
  mkdir -p "${XPU_KERNEL_CACHE_HOST}"/{triton,inductor,neo,sycl}
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
  -v "${XPU_KERNEL_CACHE_HOST}:/root/.cache/sglang-xpu" \
  -e HF_TOKEN="${HF_TOKEN_VALUE}" \
  -e SGLANG_SERVER_LAUNCH_TIMEOUT=36000 \
  -e TRITON_CACHE_DIR=/root/.cache/sglang-xpu/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/sglang-xpu/inductor \
  -e NEO_CACHE_DIR=/root/.cache/sglang-xpu/neo \
  -e NEO_CACHE_PERSISTENT=1 \
  -e SYCL_CACHE_DIR=/root/.cache/sglang-xpu/sycl \
  -e SYCL_CACHE_PERSISTENT=1 \
  "${GPU_AFFINITY_ARGS[@]}" \
  --name "${CONTAINER_NAME}" \
  "${IMAGE}"

# Mark the workspace mount as a safe directory so git operations as root
# inside the container don't trip the cross-user repo guard.
docker exec "${CONTAINER_NAME}" git config --global --add safe.directory /sglang-checkout || true

# Pre-warm the HF cache for models used by tests that time out on cold download.
# popen_launch_server's inner timeout counts network time, so a slow HF Hub can
# eat the whole window before shard loading begins. Best-effort: on failure the
# test still tries a live download.
if [[ -n "${HF_TOKEN_VALUE}" ]]; then
  docker exec "${CONTAINER_NAME}" /bin/bash -c \
    "/opt/venv/bin/hf auth login --token '${HF_TOKEN_VALUE}' >/dev/null 2>&1 || true"
fi
for model in \
    "meta-llama/Llama-3.2-1B-Instruct" \
    "rescommons/SpecForge-EAGLE3-Llama-3.2-1B-Instruct"; do
  echo "Pre-downloading HF model: ${model}"
  docker exec "${CONTAINER_NAME}" /opt/venv/bin/hf download "${model}" \
    >/dev/null 2>&1 || echo "Warning: pre-download of ${model} failed; test will retry online" >&2
done
