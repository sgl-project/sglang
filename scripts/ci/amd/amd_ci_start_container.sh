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
ROCM_VERSION="rocm700"
DEFAULT_MI30X_BASE_TAG="${SGLANG_VERSION}-${ROCM_VERSION}-mi30x"
DEFAULT_MI35X_BASE_TAG="${SGLANG_VERSION}-${ROCM_VERSION}-mi35x"
LOCAL_DOCKER_REGISTRY="10.44.14.109:5000"

# Parse command line arguments
MI30X_BASE_TAG="${DEFAULT_MI30X_BASE_TAG}"
MI35X_BASE_TAG="${DEFAULT_MI35X_BASE_TAG}"
CUSTOM_IMAGE=""
BUILD_FROM_DOCKERFILE=""
GPU_ARCH_BUILD=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --mi30x-base-tag) MI30X_BASE_TAG="$2"; shift 2;;
    --mi35x-base-tag) MI35X_BASE_TAG="$2"; shift 2;;
    --custom-image) CUSTOM_IMAGE="$2"; shift 2;;
    --build-from-dockerfile) BUILD_FROM_DOCKERFILE="1"; shift;;
    --gpu-arch) GPU_ARCH_BUILD="$2"; shift 2;;
    --rocm-version)
      ROCM_VERSION="$2"
      MI30X_BASE_TAG="${SGLANG_VERSION}-${ROCM_VERSION}-mi30x"
      MI35X_BASE_TAG="${SGLANG_VERSION}-${ROCM_VERSION}-mi35x"
      echo "Using ROCm version override: ${ROCM_VERSION}"
      shift 2;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  --mi30x-base-tag TAG       Override MI30x base image tag"
      echo "  --mi35x-base-tag TAG       Override MI35x base image tag"
      echo "  --custom-image IMAGE       Use a specific Docker image directly"
      echo "  --build-from-dockerfile    Build image from docker/rocm.Dockerfile"
      echo "  --gpu-arch ARCH            GPU architecture for Dockerfile build (e.g., gfx950-rocm720)"
      echo "  --rocm-version VERSION     Override ROCm version for image lookup (e.g., rocm720)"
      echo ""
      echo "Environment:"
      echo "  ENABLE_CACHE_HOST=auto|1|0"
      echo "      Mount AMD_CI_CACHE_HOST to /sgl-data. Defaults to auto (enabled on MI300/MI35x runners)."
      echo "  AMD_CI_CACHE_HOST=/path"
      echo "      Host cache directory. Defaults to /home/runner/sglang-data."
      exit 0
      ;;
    *) echo "Unknown option $1"; exit 1;;
  esac
done



# Detect GPU architecture from the runner hostname.
HOSTNAME_VALUE=$(hostname)
RUNNER_IDENTITY="${HOSTNAME_VALUE} ${RUNNER_NAME:-} ${RUNNER_LABELS:-}"
RUNNER_GPU_ARCH="mi30x"   # default runner architecture
GPU_ARCH="mi30x"          # default image architecture

# Kubernetes host names look like: linux-mi35x-gpu-1-xxxxx-runner-zzzzz
# Self-hosted MI300 runner names look like: linux-mi300-1gpu-sglang
if [[ "${RUNNER_IDENTITY}" =~ linux-(mi[0-9]+[a-z]*)-gpu-[0-9]+ ]]; then
  RUNNER_GPU_ARCH="${BASH_REMATCH[1]}"
  echo "Detected GPU architecture from runner identity: ${RUNNER_GPU_ARCH}"
elif [[ "${RUNNER_IDENTITY}" =~ linux-(mi[0-9]+[a-z]*)-[0-9]+gpu($|[^[:alnum:]]) ]]; then
  RUNNER_GPU_ARCH="${BASH_REMATCH[1]}"
  echo "Detected GPU architecture from runner identity: ${RUNNER_GPU_ARCH}"
else
  echo "Warning: could not parse GPU architecture from '${RUNNER_IDENTITY}', defaulting to ${RUNNER_GPU_ARCH}"
fi

GPU_ARCH="${RUNNER_GPU_ARCH}"

# Normalise / collapse architectures we don't yet build specifically for
case "${GPU_ARCH}" in
  mi35x)
    echo "Runner uses ${GPU_ARCH}; will fetch mi35x image."
    ;;
  mi30x|mi300|mi325)
    echo "Runner uses ${GPU_ARCH}; will fetch mi30x image."
    GPU_ARCH="mi30x"
    ;;
  *)
    echo "Runner architecture '${GPU_ARCH}' unrecognised; defaulting to mi30x image." >&2
    GPU_ARCH="mi30x"
    ;;
esac


# Set up DEVICE_FLAG based on Kubernetes pod info
if [[ -f /etc/podinfo/gha-render-devices ]]; then
  DEVICE_FLAG=$(cat /etc/podinfo/gha-render-devices)
else
  DEVICE_FLAG="--device /dev/dri"
fi

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

network_diagnostics_enabled() {
  case "${AMD_NETWORK_DIAGNOSTICS:-0}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

run_network_diagnostics_phase() {
  if ! network_diagnostics_enabled; then
    return 0
  fi

  local mode="$1"
  local phase="$2"
  local image="${3:-}"
  local image_tag="${NETWORK_DIAG_IMAGE_TAG:-}"

  if [[ -n "${image}" && "${image}" == *":"* ]]; then
    image_tag="${image##*:}"
  fi

  NETWORK_DIAG_MODE="${mode}" \
  NETWORK_DIAG_SOURCE="inline" \
  NETWORK_DIAG_PHASE="${phase}" \
  NETWORK_DIAG_ROCM_VERSION="${ROCM_VERSION}" \
  NETWORK_DIAG_IMAGE_TAG="${image_tag}" \
    bash scripts/ci/amd/network_diagnostics.sh || true
}

timed_docker_pull_capture() {
  local label="$1"
  shift
  local start end rc tmp
  tmp="$(mktemp)"

  echo "::group::network diagnostic: ${label}" >&2
  echo "docker_pull_label=${label}" >&2
  echo "docker_pull_command=$*" >&2
  echo "docker_pull_start=$(date -Is)" >&2
  start=$(date +%s)
  "$@" >"${tmp}" 2>&1
  rc=$?
  end=$(date +%s)
  cat "${tmp}" >&2
  cat "${tmp}"
  rm -f "${tmp}"
  echo "docker_pull_rc=${rc}" >&2
  echo "docker_pull_elapsed_sec=$((end - start))" >&2
  echo "docker_pull_end=$(date -Is)" >&2
  echo "::endgroup::" >&2
  return "${rc}"
}

timed_public_docker_pull() {
  local image="$1"
  timed_docker_pull_capture "public docker pull ${image}" docker pull "${image}"
}

# Authenticate to Docker Hub to avoid anonymous pull rate limits.
# Credentials are optional; when absent we fall back to unauthenticated pulls.
if [[ -n "${DOCKERHUB_AMD_USERNAME:-}" && -n "${DOCKERHUB_AMD_TOKEN:-}" ]]; then
  echo "Logging in to Docker Hub…"
  if retry_with_backoff 6 sh -c 'echo "${DOCKERHUB_AMD_TOKEN}" | docker login -u "${DOCKERHUB_AMD_USERNAME}" --password-stdin >/dev/null 2>&1'; then
    echo "Docker Hub login successful"
  else
    echo "Warning: Docker Hub login failed after retries; continuing with unauthenticated pulls" >&2
  fi
fi

# Find the latest image
find_latest_image() {
  local gpu_arch=$1
  local base_tag days_back image_tag image_id remote_tags

  case "${gpu_arch}" in
      mi30x) base_tag="${MI30X_BASE_TAG}" ;;
      mi35x) base_tag="${MI35X_BASE_TAG}" ;;
      *)     echo "Error: unsupported GPU architecture '${gpu_arch}'" >&2; return 1 ;;
  esac

  # First, check local cache on the runner.
  for days_back in {0..6}; do
    image_tag="${base_tag}-$(date -d "${days_back} days ago" +%Y%m%d)"
    image_id=$(docker images -q "rocm/sgl-dev:${image_tag}")
    if [[ -n "$image_id" ]]; then
      echo "Found cached image locally: rocm/sgl-dev:${image_tag}" >&2
      echo "rocm/sgl-dev:${image_tag}"
      return 0
    fi
  done

  # If not found locally, fall back to pulling from public registry.
  # We intentionally do not probe ${LOCAL_DOCKER_REGISTRY} here with
  # `docker manifest inspect --insecure` because that command runs in the
  # runner pod's network namespace, which on every observed AMD scale set
  # cannot reach 10.44.14.109:5000 (every probe either fast-fails with TLS
  # reject or hits a 30s TCP timeout, multiplied across 7 daily candidates).
  # The actual local-registry pull still happens in the call site below via
  # `docker pull "${LOCAL_DOCKER_REGISTRY}/${IMAGE}"`, which goes through the
  # docker daemon on the host and inherits its insecure-registries config.
  for days_back in {0..6}; do
    image_tag="${base_tag}-$(date -d "${days_back} days ago" +%Y%m%d)"
    echo "Checking for image: rocm/sgl-dev:${image_tag}" >&2
    if docker manifest inspect "rocm/sgl-dev:${image_tag}" >/dev/null 2>&1; then
      echo "Found available image: rocm/sgl-dev:${image_tag}" >&2
      echo "rocm/sgl-dev:${image_tag}"
      return 0
    fi
  done

  # Docker Hub's `name=` filter is fuzzy; only accept official version tags.
  echo "Exact version not found. Searching remote registry for versioned ${ROCM_VERSION}-${gpu_arch} images…" >&2
  for days_back in {0..6}; do
    local target_date=$(date -d "${days_back} days ago" +%Y%m%d)
    local sgl_tag_regex="^v[0-9][A-Za-z0-9._-]*-${ROCM_VERSION}-${gpu_arch}-${target_date}$"
    remote_tags=$(curl -s "https://registry.hub.docker.com/v2/repositories/rocm/sgl-dev/tags?page_size=100&name=${ROCM_VERSION}-${gpu_arch}-${target_date}" 2>/dev/null | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | while read -r tag; do
      if [[ "${tag}" =~ ${sgl_tag_regex} ]]; then
        echo "${tag}"
        break
      fi
    done || true)
    if [[ -n "$remote_tags" ]]; then
      echo "Found available image: rocm/sgl-dev:${remote_tags}" >&2
      echo "rocm/sgl-dev:${remote_tags}"
      return 0
    fi
  done

  echo "No recent images found. Searching cached local versioned images matching ROCm+arch…" >&2
  local any_local
  any_local=$(docker images --format '{{.Repository}}:{{.Tag}}' --filter "reference=rocm/sgl-dev:v*-${ROCM_VERSION}-${gpu_arch}-*" | while read -r image; do
    local tag="${image#rocm/sgl-dev:}"
    if [[ "${tag}" =~ ^v[0-9][A-Za-z0-9._-]*-${ROCM_VERSION}-${gpu_arch}-[0-9]{8}$ ]]; then
      echo "${image}"
    fi
  done | sort -r | head -n 1)
  if [[ -n "$any_local" ]]; then
      echo "Using cached fallback image: ${any_local}" >&2
      echo "${any_local}"
      return 0
  fi

  echo "Error: no ${gpu_arch} image found in the last 7 days for base ${base_tag}" >&2
  echo "Using hard-coded fallback for ${ROCM_VERSION}…" >&2
  case "${ROCM_VERSION}" in
    rocm720)
      if [[ "${gpu_arch}" == "mi35x" ]]; then
        echo "rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260211-preview"
      else
        echo "rocm/sgl-dev:v0.5.8.post1-rocm720-mi30x-20260211-preview"
      fi
      ;;
    rocm700)
      if [[ "${gpu_arch}" == "mi35x" ]]; then
        echo "rocm/sgl-dev:v0.5.8.post1-rocm700-mi35x-20260211"
      else
        echo "rocm/sgl-dev:v0.5.8.post1-rocm700-mi30x-20260211"
      fi
      ;;
    *)
      echo "Error: no hard-coded fallback available for ${ROCM_VERSION}" >&2
      return 1
      ;;
  esac
}

# Determine which image to use
if [[ -n "${CUSTOM_IMAGE}" ]]; then
  # Use explicitly provided custom image
  IMAGE="${CUSTOM_IMAGE}"
  echo "Using custom image: ${IMAGE}"
  run_network_diagnostics_phase prepull start_container_custom_prepull "${IMAGE}"
  if [[ "${IMAGE}" == "${LOCAL_DOCKER_REGISTRY}/"* ]]; then
    timed_docker_pull_capture "custom local docker pull ${IMAGE}" docker pull "${IMAGE}"
  else
    retry_with_backoff 6 timed_public_docker_pull "${IMAGE}"
  fi
elif [[ -n "${BUILD_FROM_DOCKERFILE}" ]]; then
  # Build image from Dockerfile
  if [[ -z "${GPU_ARCH_BUILD}" ]]; then
    echo "Error: --gpu-arch is required when using --build-from-dockerfile" >&2
    exit 1
  fi

  DOCKERFILE_DIR="${GITHUB_WORKSPACE:-$PWD}/docker"
  DOCKERFILE="${DOCKERFILE_DIR}/rocm.Dockerfile"

  if [[ ! -f "${DOCKERFILE}" ]]; then
    echo "Error: Dockerfile not found at ${DOCKERFILE}" >&2
    exit 1
  fi

  IMAGE="sglang-ci:${GPU_ARCH_BUILD}-$(date +%Y%m%d)"
  echo "Building Docker image from ${DOCKERFILE} with GPU_ARCH=${GPU_ARCH_BUILD}..."

  # Pass full GPU_ARCH (e.g., gfx950-rocm720) - Dockerfile handles stripping suffix
  docker build \
    --build-arg GPU_ARCH="${GPU_ARCH_BUILD}" \
    --build-arg SGL_BRANCH="main" \
    -t "${IMAGE}" \
    -f "${DOCKERFILE}" \
    "${DOCKERFILE_DIR}"
  echo "Successfully built image: ${IMAGE}"
else
  # Find the latest pre-built image
  IMAGE=$(find_latest_image "${GPU_ARCH}")
  run_network_diagnostics_phase prepull start_container_prepull "${IMAGE}"
  # Try the local docker registry first (avoids Docker Hub rate limits and is
  # faster on the LAN); if that fails for any reason, fall back to the
  # public registry with exponential-backoff retries. Capture stderr so the
  # real failure reason (TLS handshake, 404, connection refused, etc.) is
  # visible in the job log instead of being silently swallowed.
  if local_pull_output=$(timed_docker_pull_capture "local docker pull ${LOCAL_DOCKER_REGISTRY}/${IMAGE}" docker pull "${LOCAL_DOCKER_REGISTRY}/${IMAGE}"); then
    echo "Pulled from local docker registry: ${LOCAL_DOCKER_REGISTRY}/${IMAGE}"
    docker tag "${LOCAL_DOCKER_REGISTRY}/${IMAGE}" "${IMAGE}"
  else
    echo "Local docker registry pull failed; falling back to public registry: ${IMAGE}" >&2
    printf '%s\n' "${local_pull_output}" | sed 's/^/  [local-pull] /' >&2
    retry_with_backoff 6 timed_public_docker_pull "${IMAGE}"
  fi
fi

CACHE_HOST="${AMD_CI_CACHE_HOST:-/home/runner/sglang-data}"
ENABLE_CACHE_HOST="${ENABLE_CACHE_HOST:-auto}"
case "${ENABLE_CACHE_HOST,,}" in
  auto)
    RUNNER_CACHE_IDENTITY="${RUNNER_IDENTITY,,}"
    if [[ "${RUNNER_CACHE_IDENTITY}" == *mi300* || "${RUNNER_CACHE_IDENTITY}" == *mi35x* ]]; then
      ENABLE_CACHE_HOST="1"
    else
      ENABLE_CACHE_HOST="0"
    fi
    ;;
esac
case "${ENABLE_CACHE_HOST,,}" in
  1|true|yes|on|pvc|persistent)
    if [[ ! -d "$CACHE_HOST" ]]; then
      echo "Error: ENABLE_CACHE_HOST=1 but ${CACHE_HOST} does not exist." >&2
      exit 1
    fi
    CACHE_VOLUME="-v $CACHE_HOST:/sgl-data"
    echo "Mounting persistent CI data: ${CACHE_HOST} -> /sgl-data"
    ;;
  0|false|no|off|"")
    CACHE_VOLUME=""
    echo "Not mounting ${CACHE_HOST}; /sgl-data will be container-local."
    ;;
  *)
    echo "Error: unsupported ENABLE_CACHE_HOST='${ENABLE_CACHE_HOST}'" >&2
    echo "Use auto, 1/true/pvc/persistent, or 0/false/off." >&2
    exit 1
    ;;
esac

echo "Launching container: ci_sglang"
docker run -dt --user root --device=/dev/kfd ${DEVICE_FLAG} \
  --ulimit nofile=65536:65536 \
  -v "${GITHUB_WORKSPACE:-$PWD}:/sglang-checkout" \
  $CACHE_VOLUME \
  --group-add video \
  --shm-size 32g \
  --cap-add=SYS_PTRACE \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HOME=/sgl-data/hf-cache \
  -e HF_HUB_ETAG_TIMEOUT=300 \
  -e HF_HUB_DOWNLOAD_TIMEOUT=300 \
  -e MIOPEN_USER_DB_PATH=/sgl-data/miopen-cache \
  -e MIOPEN_CUSTOM_CACHE_DIR=/sgl-data/miopen-cache \
  -e PYTHONPATH="/opt/tilelang:${PYTHONPATH:-}" \
  --security-opt seccomp=unconfined \
  -w /sglang-checkout \
  --name ci_sglang \
  "${IMAGE}"

docker exec ci_sglang mkdir -p \
  /sgl-data/hf-cache/hub \
  /sgl-data/pip-cache \
  /sgl-data/miopen-cache \
  /sgl-data/aiter-kernels

# The checkout is owned by the runner (non-root) but the container runs as
# root.  Git >= 2.35.2 rejects cross-user repos; mark the mount as safe so
# setuptools-scm / vcs_versioning can resolve the package version.
docker exec ci_sglang git config --global --add safe.directory /sglang-checkout
