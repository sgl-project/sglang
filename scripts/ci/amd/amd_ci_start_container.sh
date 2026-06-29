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
LOCAL_DOCKER_REGISTRY="10.245.143.50:5000"

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
      exit 0
      ;;
    *) echo "Unknown option $1"; exit 1;;
  esac
done



# Detect GPU architecture from the Kubernetes runner hostname
HOSTNAME_VALUE=$(hostname)
GPU_ARCH="mi30x"   # default

# Host names look like: linux-mi35x-gpu-1-xxxxx-runner-zzzzz
if [[ "${HOSTNAME_VALUE}" =~ ^linux-(mi[0-9]+[a-z]*)-gpu-[0-9]+ ]]; then
  GPU_ARCH="${BASH_REMATCH[1]}"
  echo "Detected GPU architecture from hostname: ${GPU_ARCH}"
else
  echo "Warning: could not parse GPU architecture from '${HOSTNAME_VALUE}', defaulting to ${GPU_ARCH}"
fi

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
  # cannot reach 10.245.143.50:5000 (every probe either fast-fails with TLS
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
  if [[ "${IMAGE}" == "${LOCAL_DOCKER_REGISTRY}/"* ]]; then
    docker pull "${IMAGE}"
  else
    retry_with_backoff 6 docker pull "${IMAGE}"
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
  # Try the local docker registry first (avoids Docker Hub rate limits and is
  # faster on the LAN); if that fails for any reason, fall back to the
  # public registry with exponential-backoff retries. Capture stderr so the
  # real failure reason (TLS handshake, 404, connection refused, etc.) is
  # visible in the job log instead of being silently swallowed.
  if local_pull_output=$(docker pull "${LOCAL_DOCKER_REGISTRY}/${IMAGE}" 2>&1); then
    echo "Pulled from local docker registry: ${LOCAL_DOCKER_REGISTRY}/${IMAGE}"
    docker tag "${LOCAL_DOCKER_REGISTRY}/${IMAGE}" "${IMAGE}"
  else
    echo "Local docker registry pull failed; falling back to public registry: ${IMAGE}" >&2
    printf '%s\n' "${local_pull_output}" | sed 's/^/  [local-pull] /' >&2
    retry_with_backoff 6 docker pull "${IMAGE}"
  fi
fi

# CACHE_HOST=/home/runner/sgl-data
CACHE_HOST=/home/runner/temp-sglang-data
if [[ -d "$CACHE_HOST" ]]; then
    CACHE_VOLUME="-v $CACHE_HOST:/sgl-data"
else
    CACHE_VOLUME=""
fi

echo "=========================================="
echo "Host cache mount diagnostics"
echo "=========================================="
echo "Hostname:        $(hostname)"
echo "Expected cache:  ${CACHE_HOST}"
if [[ -d "$CACHE_HOST" ]]; then
    echo "Status:          PRESENT - will mount ${CACHE_HOST} -> /sgl-data"
    ls -ld "$CACHE_HOST" 2>/dev/null || true
    df -h "$CACHE_HOST" 2>/dev/null || true
    echo "Top-level entries (first 20):"
    ls -la "$CACHE_HOST" 2>/dev/null | head -20 || true
else
    echo "Status:          MISSING - /sgl-data inside container will be ephemeral"
    echo "                 HF models / MIOPEN cache will NOT persist between runs"
fi

echo ""
echo "Other candidate cache locations on host:"
for candidate in \
    /home/runner/sgl-data \
    /home/runner/sglang-data \
    /home/runner/cache \
    /home/runner/.cache \
    /home/runner/_work/_tool \
    /home/runner/_work/_actions \
    /sgl-data \
    /sglang-data \
    /mnt/sgl-data \
    /mnt/sglang-data \
    /mnt/cache \
    /data \
    /data/sgl-data \
    /data/sglang-data \
    /data2 \
    /data2/sgl-data \
    /data2/sglang-data \
    /data2/models \
    /data2/models/huggingface \
    /data2/models/huggingface/hub \
    /run/sgl-data \
    /run/sglang-data \
    /run/cache \
    /run/runner/sgl-data \
    /run/runner/sglang-data \
    /scratch \
    /cache \
    /opt/cache \
    /var/cache/sglang; do
    if [[ -e "$candidate" ]]; then
        if [[ -d "$candidate" ]]; then
            size=$(du -sh "$candidate" 2>/dev/null | cut -f1)
            echo "  EXISTS  ${candidate} (dir, size=${size:-?})"
        else
            echo "  EXISTS  ${candidate} (not a dir)"
        fi
    fi
done

echo ""
echo "Host /home contents:"
ls -la /home 2>/dev/null || echo "  (cannot list /home)"
echo ""
echo "Host /home/runner contents:"
ls -la /home/runner 2>/dev/null || echo "  (cannot list /home/runner)"
echo ""
echo "Host /mnt contents:"
ls -la /mnt 2>/dev/null || echo "  (cannot list /mnt)"
echo ""
echo "Host /run contents (often the persistent NVMe on MI runners):"
ls -la /run 2>/dev/null || echo "  (cannot list /run)"
echo ""
echo "Host /run subdirectory sizes (top-level only):"
du -sh /run/* 2>/dev/null | sort -hr | head -20 || true
echo ""
echo "Host /run/* one level deeper (max-depth 2):"
find /run -maxdepth 2 -type d 2>/dev/null | head -40 || true
echo ""
echo "Host /data contents:"
ls -la /data 2>/dev/null || echo "  (cannot list /data or does not exist)"
echo ""
echo "Host /data2 contents:"
ls -la /data2 2>/dev/null || echo "  (cannot list /data2 or does not exist)"
echo ""
echo "Host filesystem usage:"
df -h 2>/dev/null | head -25 || true
echo ""
echo "Persistent-looking mounts (excluding tmpfs/overlay/proc/sys/cgroup):"
awk '$3 != "tmpfs" && $3 != "overlay" && $3 != "proc" && $3 != "sysfs" && $3 != "cgroup" && $3 != "cgroup2" && $3 != "devpts" && $3 != "mqueue" {print $1, "->", $2, "type", $3}' /proc/mounts 2>/dev/null || true
echo "=========================================="

echo "=========================================="
echo "Host network diagnostics"
echo "=========================================="
echo "DNS servers (/etc/resolv.conf):"
grep -E '^(nameserver|search)' /etc/resolv.conf 2>/dev/null || echo "  (no /etc/resolv.conf)"
echo ""
echo "Default route:"
ip route show default 2>/dev/null || echo "  (no ip command)"
echo ""
echo "Interfaces (brief):"
ip -br addr 2>/dev/null || ifconfig -a 2>/dev/null || echo "  (no ip/ifconfig)"
echo ""

if ! command -v curl >/dev/null 2>&1; then
    echo "curl not available on host; skipping latency/throughput tests"
else
    HOST_CURL_AUTH=()
    if [[ -n "${HF_TOKEN:-}" ]]; then
        HOST_CURL_AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
    fi

    echo "Endpoint latency (curl --max-time 15):"
    for url in \
        https://huggingface.co/api/models/gpt2 \
        https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct \
        https://cdn-lfs.huggingface.co/ \
        https://github.com \
        https://pypi.org/simple/ \
        http://10.245.143.50:5000/v2/; do
        result=$(curl -sS -o /dev/null --max-time 15 "${HOST_CURL_AUTH[@]}" \
            -w "dns=%{time_namelookup}s conn=%{time_connect}s ssl=%{time_appconnect}s ttfb=%{time_starttransfer}s total=%{time_total}s speed=%{speed_download}B/s http=%{http_code}" \
            "$url" 2>&1) || result="FAILED (timeout / network error)"
        printf '  %-70s %s\n' "$url" "$result"
    done
    echo ""

    echo "HF CDN throughput test: first 10MB of public gpt2 weights (follow redirects)"
    speed_url="https://huggingface.co/gpt2/resolve/main/pytorch_model.bin"
    curl -sS -L -o /dev/null --range 0-10485759 --max-time 60 \
        -w "  total=%{time_total}s speed=%{speed_download}B/s bytes=%{size_download} http=%{http_code} effective_url=%{url_effective}\n" \
        "$speed_url" 2>&1 || echo "  FAILED (timeout / connection error)"
fi

echo ""
echo "Link MTU:"
ip -o link show 2>/dev/null | awk -F': ' '{print "  " $2}' | head -10 || echo "  (no ip command)"

echo ""
echo "Path MTU to huggingface.co (tracepath):"
if command -v tracepath >/dev/null 2>&1; then
    tracepath -n huggingface.co 2>&1 | head -15 || true
else
    echo "  (tracepath not installed)"
fi

echo ""
echo "Key TCP sysctls (host):"
for k in \
    net.ipv4.tcp_congestion_control \
    net.core.default_qdisc \
    net.core.rmem_max \
    net.core.wmem_max \
    net.ipv4.tcp_rmem \
    net.ipv4.tcp_wmem \
    net.ipv4.tcp_window_scaling \
    net.ipv4.tcp_mtu_probing \
    net.ipv4.tcp_timestamps; do
    val=$(sysctl -n "$k" 2>/dev/null) && printf "  %-40s %s\n" "$k" "$val" || printf "  %-40s (unavailable)\n" "$k"
done

echo ""
echo "Path / RTT to huggingface.co (mtr -rnzc 10, ~10s):"
if command -v mtr >/dev/null 2>&1; then
    mtr -rnzbc 10 --no-dns huggingface.co 2>&1 | head -30 || true
elif command -v traceroute >/dev/null 2>&1; then
    echo "  mtr not installed; using traceroute -n -w 2 (max 15 hops):"
    traceroute -n -w 2 -m 15 huggingface.co 2>&1 | head -20 || true
else
    echo "  (neither mtr nor traceroute installed)"
fi
echo "=========================================="

echo "Launching container: ci_sglang"
# EXPERIMENT: --network=host to bypass Docker bridge NAT overhead.
# Previous in-container TLS handshake to huggingface.co was ~2x slower than host
# (228ms vs 107ms), suggesting NAT/bridge adds ~120ms latency. Disagg variant
# (amd_ci_start_container_disagg.sh) already uses --network=host as precedent.
docker run -dt --user root --device=/dev/kfd ${DEVICE_FLAG} \
  --ulimit nofile=65536:65536 \
  -v "${GITHUB_WORKSPACE:-$PWD}:/sglang-checkout" \
  $CACHE_VOLUME \
  --privileged \
  --network=host \
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

docker exec ci_sglang bash -lc '
  numa_balancing_path=/proc/sys/kernel/numa_balancing

  if [[ ! -r "${numa_balancing_path}" ]]; then
    echo "WARNING: ${numa_balancing_path} is not readable; skipping NUMA balancing check" >&2
    exit 0
  fi

  echo "kernel.numa_balancing=$(cat "${numa_balancing_path}")"

  if [[ "$(cat "${numa_balancing_path}")" != "0" ]]; then
    if [[ -w "${numa_balancing_path}" ]]; then
      if echo 0 > "${numa_balancing_path}"; then
        echo "Disabled kernel.numa_balancing for AMD CI"
      else
        echo "WARNING: failed to disable kernel.numa_balancing" >&2
      fi
    else
      echo "WARNING: ${numa_balancing_path} is not writable; unable to disable NUMA balancing" >&2
    fi
  fi

  echo "kernel.numa_balancing=$(cat "${numa_balancing_path}")"
'

docker exec ci_sglang bash -lc '
  echo "=========================================="
  echo "In-container /sgl-data mount diagnostics"
  echo "=========================================="

  if [[ ! -e /sgl-data ]]; then
    echo "WARNING: /sgl-data does NOT exist inside container"
    echo "         HF / MIOPEN cache paths will fail or be created in container layer"
  else
    is_mount=0
    if command -v mountpoint >/dev/null 2>&1 && mountpoint -q /sgl-data 2>/dev/null; then
      is_mount=1
    elif [[ "$(stat -c %d / 2>/dev/null)" != "$(stat -c %d /sgl-data 2>/dev/null)" ]]; then
      is_mount=1
    fi

    if [[ "$is_mount" == "1" ]]; then
      echo "Status: /sgl-data IS a host volume mount (cache will persist)"
    else
      echo "WARNING: /sgl-data is NOT a host mount (same device as /)"
      echo "         Cache is ephemeral - HF downloads will repeat every run"
    fi

    ls -ld /sgl-data 2>/dev/null || true
    df -h /sgl-data 2>/dev/null || true
    echo "Top-level entries (first 20):"
    ls -la /sgl-data 2>/dev/null | head -20 || true
    echo "HF hub cache (first 20):"
    ls -la /sgl-data/hf-cache/hub 2>/dev/null | head -20 || echo "  (hf-cache/hub not present)"
  fi
  echo "=========================================="
'

docker exec ci_sglang bash -lc '
  echo "=========================================="
  echo "In-container network diagnostics"
  echo "=========================================="
  echo "DNS servers (/etc/resolv.conf):"
  grep -E "^(nameserver|search)" /etc/resolv.conf 2>/dev/null || echo "  (no /etc/resolv.conf)"
  echo ""
  echo "Default route:"
  ip route show default 2>/dev/null || echo "  (no ip command)"
  echo ""
  echo "Interfaces (brief):"
  ip -br addr 2>/dev/null || ifconfig -a 2>/dev/null || echo "  (no ip/ifconfig)"
  echo ""

  if ! command -v curl >/dev/null 2>&1; then
    echo "curl not available in container; skipping latency/throughput tests"
  else
    CURL_AUTH=()
    if [[ -n "${HF_TOKEN:-}" ]]; then
      CURL_AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
    fi

    echo "Endpoint latency (curl --max-time 15):"
    for url in \
      https://huggingface.co/api/models/gpt2 \
      https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct \
      https://cdn-lfs.huggingface.co/ \
      https://github.com \
      https://pypi.org/simple/ \
      http://10.245.143.50:5000/v2/; do
      result=$(curl -sS -o /dev/null --max-time 15 "${CURL_AUTH[@]}" \
        -w "dns=%{time_namelookup}s conn=%{time_connect}s ssl=%{time_appconnect}s ttfb=%{time_starttransfer}s total=%{time_total}s speed=%{speed_download}B/s http=%{http_code}" \
        "$url" 2>&1) || result="FAILED (timeout / network error)"
      printf "  %-70s %s\n" "$url" "$result"
    done
    echo ""

    echo "HF CDN throughput test: first 10MB of public gpt2 weights (follow redirects)"
    speed_url="https://huggingface.co/gpt2/resolve/main/pytorch_model.bin"
    curl -sS -L -o /dev/null --range 0-10485759 --max-time 60 \
      -w "  total=%{time_total}s speed=%{speed_download}B/s bytes=%{size_download} http=%{http_code} effective_url=%{url_effective}\n" \
      "$speed_url" 2>&1 || echo "  FAILED (timeout / connection error)"
  fi

  echo ""
  echo "Link MTU (container):"
  ip -o link show 2>/dev/null | awk -F": " "{print \"  \" \$2}" | head -10 || echo "  (no ip command)"

  echo ""
  echo "Path MTU to huggingface.co (tracepath):"
  if command -v tracepath >/dev/null 2>&1; then
    tracepath -n huggingface.co 2>&1 | head -15 || true
  else
    echo "  (tracepath not installed)"
  fi

  echo ""
  echo "Key TCP sysctls (container):"
  for k in \
    net.ipv4.tcp_congestion_control \
    net.core.default_qdisc \
    net.core.rmem_max \
    net.core.wmem_max \
    net.ipv4.tcp_rmem \
    net.ipv4.tcp_wmem \
    net.ipv4.tcp_window_scaling \
    net.ipv4.tcp_mtu_probing \
    net.ipv4.tcp_timestamps; do
    val=$(sysctl -n "$k" 2>/dev/null) && printf "  %-40s %s\n" "$k" "$val" || printf "  %-40s (unavailable)\n" "$k"
  done

  echo ""
  echo "Path / RTT to huggingface.co (mtr -rnzc 10, ~10s):"
  if command -v mtr >/dev/null 2>&1; then
    mtr -rnzbc 10 --no-dns huggingface.co 2>&1 | head -30 || true
  elif command -v traceroute >/dev/null 2>&1; then
    echo "  mtr not installed; using traceroute -n -w 2 (max 15 hops):"
    traceroute -n -w 2 -m 15 huggingface.co 2>&1 | head -20 || true
  else
    echo "  (neither mtr nor traceroute installed)"
  fi
  echo "=========================================="
'

# The checkout is owned by the runner (non-root) but the container runs as
# root.  Git >= 2.35.2 rejects cross-user repos; mark the mount as safe so
# setuptools-scm / vcs_versioning can resolve the package version.
docker exec ci_sglang git config --global --add safe.directory /sglang-checkout
