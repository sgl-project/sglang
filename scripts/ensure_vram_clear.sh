#!/bin/bash

# Source the VRAM checking function
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/check_vram_clear.sh"

ensure_vram_clear() {
    local max_retries=3
    local retry_count=0

    # Log host information for debugging
    echo "=== Host Information ==="
    echo "Hostname: $(hostname)"
    echo "Host IP: $(hostname -I 2>/dev/null || echo 'N/A')"
    echo "Date: $(date)"
    echo "Mode: rocm"
    echo "========================"
    echo "Running in ROCm mode"

    # Show initial GPU status
    echo "=== Initial GPU Memory Status ==="
    rocm-smi --showmemuse
    echo "=================================="

    while [ $retry_count -lt $max_retries ]; do
        echo "=== Cleanup Attempt $((retry_count + 1))/$max_retries ==="

        # Clean SGLang processes
        echo "Killing SGLang processes..."
        pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt' | xargs -r kill -9 || true

        if [ $retry_count -gt 0 ]; then
            echo "Performing aggressive cleanup..."
            # Kill all processes using KFD
            lsof /dev/kfd 2>/dev/null | awk 'NR>1 {print $2}' | xargs -r kill -9 2>/dev/null || true
            # Wait a bit for cleanup to take effect
            echo "Waiting 10 seconds for VRAM to clear..."
            sleep 10
        fi

        # Check VRAM
        echo "Checking VRAM status..."
        if check_vram_clear; then
            echo "✓ VRAM cleanup successful after $((retry_count + 1)) attempts"
            return 0
        else
            echo "✗ VRAM still not clear after attempt $((retry_count + 1))"
            retry_count=$((retry_count + 1))
        fi
    done

    # Failed after all retries
    echo "=== FAILED: VRAM cleanup unsuccessful after $max_retries attempts ==="
    echo "Final GPU status:"
    rocm-smi --showmemuse
    echo "Processes using GPU:"
    lsof /dev/kfd 2>/dev/null || echo "No processes found using /dev/kfd"
    echo "=================================================================="
    return 1
}

# If this script is run directly (not sourced), run the ensure function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -e
    ensure_vram_clear "$@"
fi
