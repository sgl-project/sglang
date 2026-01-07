#!/bin/bash
set -euo pipefail
HOSTNAME_VALUE=$(hostname)
GPU_ARCH="mi30x"   # default
OPTIONAL_DEPS="${1:-}"

# Build python extras
EXTRAS="dev_hip"
if [ -n "$OPTIONAL_DEPS" ]; then
    EXTRAS="dev_hip,${OPTIONAL_DEPS}"
fi
echo "Installing python extras: [${EXTRAS}]"

# Host names look like: linux-mi35x-gpu-1-xxxxx-runner-zzzzz
if [[ "${HOSTNAME_VALUE}" =~ ^linux-(mi[0-9]+[a-z]*)-gpu-[0-9]+ ]]; then
  GPU_ARCH="${BASH_REMATCH[1]}"
  echo "Detected GPU architecture from hostname: ${GPU_ARCH}"
else
  echo "Warning: could not parse GPU architecture from '${HOSTNAME_VALUE}', defaulting to ${GPU_ARCH}"
fi

# Install the required dependencies in CI.
# Fix permissions on pip cache, ignore errors from concurrent access or missing temp files
docker exec ci_sglang chown -R root:root /sgl-data/pip-cache 2>/dev/null || true
docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache --upgrade pip
docker exec ci_sglang pip uninstall sgl-kernel -y || true
docker exec ci_sglang pip uninstall sglang -y || true
# Clear Python cache to ensure latest code is used
docker exec ci_sglang find /opt/venv -name "*.pyc" -delete || true
docker exec ci_sglang find /opt/venv -name "__pycache__" -type d -exec rm -rf {} + || true
# Also clear cache in sglang-checkout
docker exec ci_sglang find /sglang-checkout -name "*.pyc" -delete || true
docker exec ci_sglang find /sglang-checkout -name "__pycache__" -type d -exec rm -rf {} + || true
docker exec -w /sglang-checkout/sgl-kernel ci_sglang bash -c "rm -f pyproject.toml && mv pyproject_rocm.toml pyproject.toml && python3 setup_rocm.py install"

# Helper function to install with retries and fallback PyPI mirror
install_with_retry() {
  local max_attempts=3
  local cmd="$@"

  for attempt in $(seq 1 $max_attempts); do
    echo "Attempt $attempt/$max_attempts: $cmd"
    if eval "$cmd"; then
      echo "Success!"
      return 0
    fi

    if [ $attempt -lt $max_attempts ]; then
      echo "Failed, retrying in 5 seconds..."
      sleep 5
      # Try with alternative PyPI index on retry
      if [[ "$cmd" =~ "pip install" ]] && [ $attempt -eq 2 ]; then
        cmd="$cmd --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
        echo "Using fallback PyPI mirror: $cmd"
      fi
    fi
  done

  echo "Failed after $max_attempts attempts"
  return 1
}

# Helper function to git clone with retries
git_clone_with_retry() {
  local repo_url="$1"
  local dest_dir="${2:-}"
  local branch_args="${3:-}"
  local max_attempts=3

  for attempt in $(seq 1 $max_attempts); do
    echo "Git clone attempt $attempt/$max_attempts: $repo_url"

    # prevent from partial clone
    if [ -n "$dest_dir" ] && [ -d "$dest_dir" ]; then
      rm -rf "$dest_dir"
    fi

    if git \
      -c http.lowSpeedLimit=1000 \
      -c http.lowSpeedTime=30 \
      clone --depth 1 ${branch_args:+$branch_args} "$repo_url" "$dest_dir"; then
      echo "Git clone succeeded."
      return 0
    fi

    if [ $attempt -lt $max_attempts ]; then
      echo "Git clone failed, retrying in 5 seconds..."
      sleep 5
    fi
  done

  echo "Git clone failed after $max_attempts attempts: $repo_url"
  return 1
}



case "${GPU_ARCH}" in
  mi35x)
    echo "Runner uses ${GPU_ARCH}; will fetch mi35x image."
    docker exec ci_sglang rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
    install_with_retry docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache -e "python[${EXTRAS}]" --no-deps # TODO: only for mi35x
    # For lmms_evals evaluating MMMU
    docker exec -w / ci_sglang git clone --branch v0.4.1 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    install_with_retry docker exec -w /lmms-eval ci_sglang pip install --cache-dir=/sgl-data/pip-cache -e . --no-deps
    ;;
  mi30x|mi300|mi325)
    echo "Runner uses ${GPU_ARCH}; will fetch mi30x image."
    docker exec ci_sglang rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
    install_with_retry docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache -e "python[${EXTRAS}]"
    # For lmms_evals evaluating MMMU
    docker exec -w / ci_sglang git clone --branch v0.4.1 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    install_with_retry docker exec -w /lmms-eval ci_sglang pip install --cache-dir=/sgl-data/pip-cache -e .
    ;;
  *)
    echo "Runner architecture '${GPU_ARCH}' unrecognised;" >&2
    ;;
esac

#docker exec -w / ci_sglang git clone https://github.com/merrymercy/human-eval.git
git_clone_with_retry https://github.com/merrymercy/human-eval.git human-eval
docker cp human-eval ci_sglang:/
install_with_retry docker exec -w /human-eval ci_sglang pip install --cache-dir=/sgl-data/pip-cache -e .

docker exec -w / ci_sglang mkdir -p /dummy-grok
mkdir -p dummy-grok && wget https://sharkpublic.blob.core.windows.net/sharkpublic/sglang/dummy_grok.json -O dummy-grok/config.json
docker cp ./dummy-grok ci_sglang:/

docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache huggingface_hub[hf_xet]
docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache pytest

# Install tvm-ffi for JIT kernel support (QK-norm, etc.)
echo "Installing tvm-ffi for JIT kernel support..."
docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache git+https://github.com/apache/tvm-ffi.git || echo "tvm-ffi installation failed, JIT kernels will use fallback"

# Install cache-dit for qwen_image_t2i_cache_dit_enabled test (added in PR 16204)
docker exec ci_sglang pip install --cache-dir=/sgl-data/pip-cache cache-dit || echo "cache-dit installation failed"

# Detect AITER version
#############################################
# Detect correct AITER_COMMIT for this runner
# + Check mismatch
# + Rebuild AITER if needed
#############################################

echo "[CI-AITER-CHECK] === AITER VERSION CHECK START ==="

DOCKERFILE="docker/rocm.Dockerfile"

# GPU_ARCH
GPU_ARCH="${GPU_ARCH:-mi30x}"
echo "[CI-AITER-CHECK] Runner GPU_ARCH=${GPU_ARCH}"

#############################################
# 1. Extract AITER_COMMIT from correct Dockerfile block
#############################################
if [[ "${GPU_ARCH}" == "mi35x" ]]; then
    echo "[CI-AITER-CHECK] Using gfx950 block from Dockerfile..."
    REPO_AITER_COMMIT=$(grep -F -A20 'FROM $BASE_IMAGE_950 AS gfx950' docker/rocm.Dockerfile \
                        | grep 'AITER_COMMIT=' \
                        | head -n1 \
                        | sed 's/.*AITER_COMMIT="\([^"]*\)".*/\1/')
else
    echo "[CI-AITER-CHECK] Using gfx942-rocm700 block from Dockerfile..."
    REPO_AITER_COMMIT=$(grep -F -A20 'FROM $BASE_IMAGE_942_ROCM700 AS gfx942-rocm700' docker/rocm.Dockerfile \
                        | grep 'AITER_COMMIT=' \
                        | head -n1 \
                        | sed 's/.*AITER_COMMIT="\([^"]*\)".*/\1/')
fi


if [[ -z "${REPO_AITER_COMMIT}" ]]; then
    echo "[CI-AITER-CHECK] ERROR: Failed to extract AITER_COMMIT from Dockerfile."
    exit 1
fi

echo "[CI-AITER-CHECK] Dockerfile expects AITER_COMMIT=${REPO_AITER_COMMIT}"

#############################################
# 2. Check container pre-installed AITER version
#############################################
IMAGE_AITER_VERSION=$(docker exec ci_sglang bash -c "pip show aiter 2>/dev/null | grep '^Version:' | awk '{print \$2}'" || echo "none")
IMAGE_AITER_VERSION="v${IMAGE_AITER_VERSION}"
echo "[CI-AITER-CHECK] AITER version inside CI image: ${IMAGE_AITER_VERSION}"

#############################################
# 3. Decide rebuild
#############################################
NEED_REBUILD="false"

if [[ "${IMAGE_AITER_VERSION}" == "none" ]]; then
    echo "[CI-AITER-CHECK] No AITER found in image"
    NEED_REBUILD="true"
elif [[ "${IMAGE_AITER_VERSION}" != "${REPO_AITER_COMMIT}" ]]; then
    echo "[CI-AITER-CHECK] Version mismatch:"
    echo "     Image: ${IMAGE_AITER_VERSION}"
    echo "     Repo : ${REPO_AITER_COMMIT}"
    NEED_REBUILD="true"
else
    echo "[CI-AITER-CHECK] AITER version matches → using image's version."
fi


#############################################
# 4. Rebuild AITER if needed
#############################################
if [[ "${NEED_REBUILD}" == "true" ]]; then
    echo "[CI-AITER-CHECK] === AITER REBUILD START ==="

    # uninstall existing aiter
    docker exec ci_sglang pip uninstall -y aiter || true

    # delete old aiter directory
    docker exec ci_sglang rm -rf /sgl-workspace/aiter

    # clone a fresh copy to /sgl-workspace/aiter
    docker exec ci_sglang git clone https://github.com/ROCm/aiter.git /sgl-workspace/aiter

    # checkout correct version
    docker exec ci_sglang bash -c "
        cd /sgl-workspace/aiter && \
        git fetch --all && \
        git checkout ${REPO_AITER_COMMIT} && \
        git submodule update --init --recursive
    "

    if [[ "${GPU_ARCH}" == "mi35x" ]]; then
        GPU_ARCH_LIST="gfx950"
    else
        GPU_ARCH_LIST="gfx942"
    fi
    echo "[CI-AITER-CHECK] GPU_ARCH_LIST=${GPU_ARCH_LIST}"

    # build AITER
    docker exec ci_sglang bash -c "
        cd /sgl-workspace/aiter && \
        GPU_ARCHS=${GPU_ARCH_LIST} python3 setup.py develop
    "

    echo "[CI-AITER-CHECK] === AITER REBUILD COMPLETE ==="
fi

echo "[CI-AITER-CHECK] === AITER VERSION CHECK END ==="


# Clear pre-built AITER kernels from Docker image to avoid segfaults
# The Docker image may contain pre-compiled kernels incompatible with the current environment
echo "Clearing pre-built AITER kernels from Docker image..."
docker exec ci_sglang find /sgl-workspace/aiter/aiter/jit -name "*.so" -delete 2>/dev/null || true
docker exec ci_sglang ls -la /sgl-workspace/aiter/aiter/jit/ 2>/dev/null || echo "jit dir empty or not found"

# Pre-build AITER kernels to avoid timeout during tests
echo "Warming up AITER JIT kernels..."
docker exec -e SGLANG_USE_AITER=1 ci_sglang python3 /sglang-checkout/scripts/ci/amd_ci_warmup_aiter.py || echo "AITER warmup completed (some kernels may not be available)"
