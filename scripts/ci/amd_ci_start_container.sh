#!/bin/bash
set -euo pipefail

# Get version from SGLang version.py file
SGLANG_VERSION_FILE="$(dirname "$0")/../../python/sglang/version.py"
SGLANG_VERSION="v0.5.5"   # Default version, will be overridden if version.py is found

TMP_VERSION_FILE=$(mktemp)
if git fetch --depth=1 origin main; then
  if git show origin/main:python/sglang/version.py >"$TMP_VERSION_FILE" 2>/dev/null; then
    VERSION_FROM_FILE="v$(cat "$SGLANG_VERSION_FILE" | cut -d'"' -f2)"
    if [ -n "$VERSION_FROM_FILE" ]; then
      SGLANG_VERSION="$VERSION_FROM_FILE"
      echo "Using SGLang version from origin/main: $SGLANG_VERSION"
    else
      echo "Warning: Could not parse version from origin/main; using default $SGLANG_VERSION" >&2
    fi
  else
    echo "Warning: version.py not found on origin/main; using default $SGLANG_VERSION" >&2
  fi
else
  echo "Warning: failed to fetch origin/main; using default $SGLANG_VERSION" >&2
fi
rm -f "$TMP_VERSION_FILE"


# Default base tags (can be overridden by command line arguments)
ROCM_VERSION="rocm700"
DEFAULT_MI30X_BASE_TAG="${SGLANG_VERSION}-${ROCM_VERSION}-mi30x"
DEFAULT_MI35X_BASE_TAG="${SGLANG_VERSION}-${ROCM_VERSION}-mi35x"

# Parse command line arguments
MI30X_BASE_TAG="${DEFAULT_MI30X_BASE_TAG}"
MI35X_BASE_TAG="${DEFAULT_MI35X_BASE_TAG}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --mi30x-base-tag) MI30X_BASE_TAG="$2"; shift 2;;
    --mi35x-base-tag) MI35X_BASE_TAG="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 [--mi30x-base-tag TAG] [--mi35x-base-tag TAG]"
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


# Find the latest image
find_latest_image() {
  local gpu_arch=$1
  local base_tag days_back image_tag

  case "${gpu_arch}" in
      mi30x) base_tag="${MI30X_BASE_TAG}" ;;
      mi35x) base_tag="${MI35X_BASE_TAG}" ;;
      *)     echo "Error: unsupported GPU architecture '${gpu_arch}'" >&2; return 1 ;;
  esac

  # First, check local cache
  for days_back in {0..6}; do
    image_tag="${base_tag}-$(date -d "${days_back} days ago" +%Y%m%d)"
    local local_image="rocm/sgl-dev:${image_tag}"
    image_id=$(docker images -q "${local_image}")
    if [[ -n "$image_id" ]]; then
        echo "Found cached image locally: ${local_image}" >&2
        echo "${local_image}"
        return 0
    fi
  done

  # If not found locally, fall back to pulling from public registry
  for days_back in {0..6}; do
    image_tag="${base_tag}-$(date -d "${days_back} days ago" +%Y%m%d)"
    echo "Checking for image: rocm/sgl-dev:${image_tag}" >&2
    if docker manifest inspect "rocm/sgl-dev:${image_tag}" >/dev/null 2>&1; then
      echo "Found available image: rocm/sgl-dev:${image_tag}" >&2
      echo "rocm/sgl-dev:${image_tag}"
      return 0
    fi
  done

  echo "No recent images found. Searching any cached local images matching ROCm+arch…" >&2
  local any_local
  any_local=$(docker images --format '{{.Repository}}:{{.Tag}}' --filter "reference=rocm/sgl-dev:*${ROCM_VERSION}*${gpu_arch}*" | sort -r | head -n 1)
  if [[ -n "$any_local" ]]; then
      echo "Using cached fallback image: ${any_local}" >&2
      echo "${any_local}"
      return 0
  fi

  echo "Error: no ${gpu_arch} image found in the last 7 days for base ${base_tag}" >&2
  echo "Using hard-coded fallback…" >&2
  if [[ "${gpu_arch}" == "mi35x" ]]; then
    echo "rocm/sgl-dev:v0.5.5-rocm700-mi35x-20251110"
  else
    echo "rocm/sgl-dev:v0.5.5-rocm700-mi30x-20251110"
  fi
}

# Pull and run the latest image
IMAGE=$(find_latest_image "${GPU_ARCH}")
echo "Pulling Docker image: ${IMAGE}"
docker pull "${IMAGE}"

# Check for NFS cache directory on host
CACHE_HOST=/home/runner/sgl-data
echo "=== Shared Storage Detection ==="
echo "Checking for shared storage: $CACHE_HOST"

if [[ -d "$CACHE_HOST" ]]; then
    # Check if it's a mountpoint (separate filesystem, not local)
    if mountpoint -q "$CACHE_HOST" 2>/dev/null; then
        echo "✓ $CACHE_HOST is a mountpoint (shared storage)"
        IS_SHARED_STORAGE="1"
    else
        # Check if it's on a different device than root
        root_dev=$(df / | tail -1 | awk '{print $1}')
        cache_dev=$(df "$CACHE_HOST" | tail -1 | awk '{print $1}')
        if [[ "$root_dev" != "$cache_dev" ]]; then
            echo "✓ $CACHE_HOST is on different device than root (shared storage)"
            IS_SHARED_STORAGE="1"
        else
            echo "⚠ $CACHE_HOST is on same device as root (local storage)"
            IS_SHARED_STORAGE="0"
        fi
    fi
    
    echo ""
    echo "=== Storage Info ==="
    df -h "$CACHE_HOST"
    ls -la "$CACHE_HOST" 2>/dev/null | head -5 || true
    
    CACHE_VOLUME="-v $CACHE_HOST:/sgl-data"
    HAS_NFS_CACHE="1"
else
    echo "✗ Shared storage NOT found: $CACHE_HOST"
    echo "Available directories in /home/runner:"
    ls -la /home/runner 2>/dev/null || echo "Cannot list /home/runner"
    CACHE_VOLUME=""
    HAS_NFS_CACHE="0"
    IS_SHARED_STORAGE="0"
fi
echo "==============================="

# Measure NFS bandwidth using a safetensor file read test
# If bandwidth > 10GB/s, use NFS for HF cache; otherwise use local storage
# This handles NFS performance "cliff" where busy NFS is slower than direct download
measure_nfs_bandwidth() {
    local test_dir="${CACHE_HOST}/hf-cache/hub"
    local safetensor_file=""

    echo "Looking for safetensor test file in: $test_dir"

    # Find a safetensor file to test bandwidth (look for any .safetensors file)
    if [[ -d "$test_dir" ]]; then
        safetensor_file=$(find "$test_dir" -name "*.safetensors" -type f -size +10M 2>/dev/null | head -1)
        if [[ -n "$safetensor_file" ]]; then
            echo "Found test file: $safetensor_file"
        else
            echo "No suitable .safetensors files found (need >10MB)"
            # List what's in the cache for debugging
            echo "HF cache contents:"
            ls -la "$test_dir" 2>/dev/null | head -5 || echo "  (empty or not accessible)"
        fi
    else
        echo "HF cache directory does not exist yet: $test_dir"
    fi

    if [[ -z "$safetensor_file" ]]; then
        echo "No safetensor file for bandwidth test - defaulting to NFS"
        echo "10.1"  # Default to "good" bandwidth if no test file exists
        return
    fi

    # Get file size in bytes
    local file_size=$(stat -c %s "$safetensor_file" 2>/dev/null || echo "0")
    local file_size_mb=$((file_size / 1048576))
    echo "Test file size: ${file_size_mb} MB"

    if [[ "$file_size" -lt 1048576 ]]; then  # Less than 1MB
        echo "Test file too small (<1MB) - defaulting to NFS"
        echo "10.1"
        return
    fi

    # Clear page cache to get real disk/NFS read speed (requires root)
    echo "Clearing page cache for accurate measurement..."
    sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "  (skipped - not root)"

    # Time how long it takes to read the file
    echo "Reading file to measure bandwidth..."
    local start_time=$(date +%s.%N)
    dd if="$safetensor_file" of=/dev/null bs=4M 2>/dev/null
    local end_time=$(date +%s.%N)

    # Calculate bandwidth in GB/s
    local elapsed=$(echo "$end_time - $start_time" | bc)
    echo "Read time: ${elapsed} seconds"

    if [[ $(echo "$elapsed > 0" | bc) -eq 1 ]]; then
        local bandwidth=$(echo "scale=2; $file_size / $elapsed / 1073741824" | bc)
        echo "Calculated bandwidth: ${bandwidth} GB/s"
        echo "$bandwidth"
    else
        echo "Timing error - defaulting to NFS"
        echo "10.1"  # Default if timing failed
    fi
}

# Check NFS bandwidth and decide HF cache strategy
USE_NFS_HF_CACHE="0"  # Default to local if no NFS available
if [[ "$HAS_NFS_CACHE" == "1" ]]; then
    echo ""
    echo "=== NFS Bandwidth Test ==="
    NFS_BANDWIDTH=$(measure_nfs_bandwidth)
    echo ""
    echo "Result: NFS bandwidth = ${NFS_BANDWIDTH} GB/s (threshold: 10 GB/s)"

    if (( $(echo "$NFS_BANDWIDTH < 10" | bc -l) )); then
        echo "DECISION: Bandwidth BELOW threshold"
        echo "ACTION: Using LOCAL storage (/tmp/hf-cache) for HF cache"
        USE_NFS_HF_CACHE="0"
    else
        echo "DECISION: Bandwidth ABOVE threshold"
        echo "ACTION: Using NFS (/sgl-data/hf-cache) for HF cache"
        USE_NFS_HF_CACHE="1"
    fi
    echo "=========================="
else
    echo ""
    echo "=== NFS Not Available ==="
    echo "No NFS cache directory - using local storage for all caches"
    echo "=========================="
fi

# AITER and MIOpen caches should ALWAYS use NFS if available
# These are kernel caches that are expensive to rebuild and benefit from persistence
if [[ "$HAS_NFS_CACHE" == "1" ]]; then
    AITER_CACHE_DIR="/sgl-data/aiter-kernels"
    MIOPEN_CACHE_DIR="/sgl-data/miopen-cache"
else
    AITER_CACHE_DIR="/tmp/aiter-kernels"
    MIOPEN_CACHE_DIR="/tmp/miopen-cache"
fi

# HF cache location depends on bandwidth check
if [[ "$USE_NFS_HF_CACHE" == "1" ]]; then
    HF_CACHE_DIR="/sgl-data/hf-cache"
else
    HF_CACHE_DIR="/tmp/hf-cache"
fi

echo ""
echo "=== Final Cache Configuration ==="
echo "  AITER cache:  ${AITER_CACHE_DIR}"
echo "  MIOpen cache: ${MIOPEN_CACHE_DIR}"
echo "  HF cache:     ${HF_CACHE_DIR}"
echo "  NFS HF cache: ${USE_NFS_HF_CACHE} (1=NFS, 0=local)"
echo "================================="

echo ""
echo "Launching container: ci_sglang"
docker run -dt --user root --device=/dev/kfd ${DEVICE_FLAG} \
  -v "${GITHUB_WORKSPACE:-$PWD}:/sglang-checkout" \
  $CACHE_VOLUME \
  --group-add video \
  --shm-size 32g \
  --cap-add=SYS_PTRACE \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HOME="${HF_CACHE_DIR}" \
  -e AITER_JIT_DIR="${AITER_CACHE_DIR}" \
  -e MIOPEN_USER_DB_PATH="${MIOPEN_CACHE_DIR}" \
  -e MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CACHE_DIR}" \
  -e SGLANG_NFS_HF_CACHE="${USE_NFS_HF_CACHE}" \
  --security-opt seccomp=unconfined \
  -w /sglang-checkout \
  --name ci_sglang \
  "${IMAGE}"

echo ""
echo "Container started successfully"
