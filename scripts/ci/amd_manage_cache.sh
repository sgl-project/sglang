#!/bin/bash
# Script to manage AMD CI model cache
set -euo pipefail

CACHE_DIR="${CACHE_DIR:-/opt/sglang-ci-cache}"

case "${1:-}" in
  init)
    echo "Initializing AMD CI model cache..."
    if [[ "$EUID" -eq 0 ]]; then
      # Running as root
      mkdir -p "${CACHE_DIR}/huggingface"
      chmod -R 777 "${CACHE_DIR}"
    else
      # Running as non-root, use sudo
      sudo mkdir -p "${CACHE_DIR}/huggingface"
      sudo chmod -R 777 "${CACHE_DIR}"
    fi
    echo "Cache directory initialized at ${CACHE_DIR}"
    echo "Permissions set to allow Docker container access"
    ;;

  status)
    echo "=== AMD CI Cache Status ==="
    echo "Cache location: ${CACHE_DIR}"
    if [ -d "${CACHE_DIR}/huggingface/hub" ]; then
      cache_size=$(du -sh "${CACHE_DIR}/huggingface/hub" 2>/dev/null | cut -f1 || echo "unknown")
      model_count=$(ls -1 "${CACHE_DIR}/huggingface/hub" 2>/dev/null | grep -c 'models--' || echo "0")
      echo "Cache size: ${cache_size}"
      echo "Cached models: ${model_count}"

      if [ "${model_count}" -gt 0 ]; then
        echo "Model list:"
        ls -1 "${CACHE_DIR}/huggingface/hub" 2>/dev/null | grep 'models--' | head -10 | sed 's/models--/  - /' || true
        if [ "${model_count}" -gt 10 ]; then
          echo "  ... and $((model_count - 10)) more"
        fi
      fi
    else
      echo "Cache is empty or not initialized"
    fi

    echo "Disk usage in cache directory:"
    df -h "${CACHE_DIR}" 2>/dev/null || echo "  (unable to check disk usage)"
    ;;

  clean-old)
    echo "Cleaning up old cache files (not accessed in 30 days)..."
    if [ -d "${CACHE_DIR}" ]; then
      old_files_count=$(find "${CACHE_DIR}" -type f -atime +30 2>/dev/null | wc -l || echo "0")
      if [ "${old_files_count}" -gt 0 ]; then
        echo "Found ${old_files_count} old files to remove"
        if [[ "$EUID" -eq 0 ]]; then
          find "${CACHE_DIR}" -type f -atime +30 -delete 2>/dev/null || true
        else
          sudo find "${CACHE_DIR}" -type f -atime +30 -delete 2>/dev/null || true
        fi
        echo "Old files removed"
      else
        echo "No old files found"
      fi
    else
      echo "Cache directory does not exist"
    fi
    ;;

  preload)
    echo "Pre-loading common test models..."
    if ! command -v docker >/dev/null 2>&1; then
      echo "Error: Docker not found. Cannot preload models without Docker." >&2
      exit 1
    fi

    # Ensure cache directory exists
    mkdir -p "${CACHE_DIR}/huggingface"

    # Use a temporary container to preload models
    echo "Starting temporary container for model preloading..."
    docker run --rm \
      -v "${CACHE_DIR}/huggingface:/root/.cache/huggingface" \
      -e HF_HOME="/root/.cache/huggingface" \
      -e TRANSFORMERS_CACHE="/root/.cache/huggingface/hub" \
      python:3.10-slim bash -c "
      pip install huggingface_hub transformers torch --quiet
      python3 -c \"
from huggingface_hub import snapshot_download
import os

models = [
    'meta-llama/Llama-3.1-8B-Instruct',
    'meta-llama/Llama-3.2-1B-Instruct',
    'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
    'mistralai/Mixtral-8x7B-Instruct-v0.1',
]

for model in models:
    print(f'Downloading {model}...')
    try:
        snapshot_download(model, cache_dir=os.environ['HF_HOME'])
        print(f'✓ {model} cached successfully')
    except Exception as e:
        print(f'✗ Failed to cache {model}: {e}')
        continue
      \"
      "
    echo "Model preloading completed"
    ;;

  help|--help|-h)
    echo "AMD CI Model Cache Management"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  init       Initialize cache directory with proper permissions"
    echo "  status     Show cache status and statistics"
    echo "  clean-old  Remove files not accessed in 30 days"
    echo "  preload    Pre-download common test models"
    echo "  help       Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  CACHE_DIR  Cache directory path (default: /opt/sglang-ci-cache)"
    ;;

  *)
    echo "Usage: $0 {init|status|clean-old|preload|help}"
    echo "Use '$0 help' for more information"
    exit 1
    ;;
esac
