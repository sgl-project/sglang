#!/bin/bash
set -euo pipefail
HOSTNAME_VALUE=$(hostname)
GPU_ARCH="mi30x"   # default

# Host names look like: linux-mi35x-gpu-1-xxxxx-runner-zzzzz
if [[ "${HOSTNAME_VALUE}" =~ ^linux-(mi[0-9]+[a-z]*)-gpu-[0-9]+ ]]; then
  GPU_ARCH="${BASH_REMATCH[1]}"
  echo "Detected GPU architecture from hostname: ${GPU_ARCH}"
else
  echo "Warning: could not parse GPU architecture from '${HOSTNAME_VALUE}', defaulting to ${GPU_ARCH}"
fi

# Check disk space and conditionally manage cache
echo "Checking available disk space..."
cache_parent_dir="${CACHE_DIR:-$HOME/.sglang-ci-cache}"

# Robust disk space check with fallbacks
available_gb=0
if df "$cache_parent_dir" >/dev/null 2>&1; then
  # Try to get available space in GB
  available_gb=$(df "$cache_parent_dir" | awk 'NR==2 {print int($4/1024/1024)}' 2>/dev/null || echo "0")
elif [ -d "$cache_parent_dir" ]; then
  # Fallback: check parent directory if cache dir doesn't exist yet
  available_gb=$(df "$(dirname "$cache_parent_dir")" | awk 'NR==2 {print int($4/1024/1024)}' 2>/dev/null || echo "0")
else
  # Final fallback: check root filesystem
  available_gb=$(df / | awk 'NR==2 {print int($4/1024/1024)}' 2>/dev/null || echo "0")
fi

echo "Available disk space: ${available_gb}GB"

# Cache management is disabled by default, only enable if ENABLE_CACHE=true
if [[ "${ENABLE_CACHE:-false}" == "true" ]]; then
  echo "Cache management enabled via ENABLE_CACHE=true"
  echo "Initializing model cache..."
  bash "$(dirname "$0")/amd_manage_cache.sh" init

  if [ "${available_gb}" -lt 20 ]; then
    echo "Low disk space detected (${available_gb}GB). Running cleanup..."
    bash "$(dirname "$0")/amd_manage_cache.sh" clean-old

    # Re-check after cleanup
    available_gb=$(df "$cache_parent_dir" | awk 'NR==2 {print int($4/1024/1024)}' 2>/dev/null || echo "0")
    echo "Available disk space after cleanup: ${available_gb}GB"

    if [ "${available_gb}" -lt 10 ]; then
      echo "Warning: Still low on disk space (${available_gb}GB). Cache operations may be limited." >&2
    fi
  else
    echo "Disk space is sufficient for cache operations"
  fi
else
  echo "Cache management disabled by default (set ENABLE_CACHE=true to enable)"
  if [ "${available_gb}" -lt 10 ]; then
    echo "Warning: Low disk space detected (${available_gb}GB). Consider enabling cache cleanup." >&2
  fi
fi

# Install the required dependencies in CI.
docker exec ci_sglang pip install --upgrade pip
docker exec ci_sglang pip uninstall sgl-kernel -y || true
docker exec -w /sglang-checkout/sgl-kernel ci_sglang bash -c "rm -f pyproject.toml && mv pyproject_rocm.toml pyproject.toml && python3 setup_rocm.py install"

case "${GPU_ARCH}" in
  mi35x)
    echo "Runner uses ${GPU_ARCH}; will fetch mi35x image."
    docker exec ci_sglang rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
    docker exec ci_sglang pip install -e "python[dev_hip]" --no-deps # TODO: only for mi35x
    # For lmms_evals evaluating MMMU
    docker exec -w / ci_sglang git clone --branch v0.3.3 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    docker exec -w /lmms-eval ci_sglang pip install -e . --no-deps # TODO: only for mi35x
    ;;
  mi30x|mi300|mi325)
    echo "Runner uses ${GPU_ARCH}; will fetch mi30x image."
    docker exec ci_sglang rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
    docker exec ci_sglang pip install -e "python[dev_hip]"
    # For lmms_evals evaluating MMMU
    docker exec -w / ci_sglang git clone --branch v0.3.3 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    docker exec -w /lmms-eval ci_sglang pip install -e .
    ;;
  *)
    echo "Runner architecture '${GPU_ARCH}' unrecognised;" >&2
    ;;
esac

docker exec -w / ci_sglang git clone https://github.com/merrymercy/human-eval.git
docker exec -w /human-eval ci_sglang pip install -e .

docker exec -w / ci_sglang mkdir -p /dummy-grok
mkdir -p dummy-grok && wget https://sharkpublic.blob.core.windows.net/sharkpublic/sglang/dummy_grok.json -O dummy-grok/config.json
docker cp ./dummy-grok ci_sglang:/

docker exec ci_sglang pip install huggingface_hub[hf_xet]
docker exec ci_sglang pip install pytest

# Report Hugging Face cache status
echo "=== Hugging Face Cache Status ==="
if [[ "${ENABLE_CACHE:-false}" == "true" ]]; then
  bash "$(dirname "$0")/amd_manage_cache.sh" status
else
  echo "Cache management disabled - no cache status available"
fi
docker exec ci_sglang bash -c "
echo 'Container cache info:'
echo 'HF_HOME=' \$HF_HOME
echo 'TRANSFORMERS_CACHE=' \$TRANSFORMERS_CACHE
"
