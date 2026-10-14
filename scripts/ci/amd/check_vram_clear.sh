#!/bin/bash

check_vram_clear() {
    local vram_threshold_percent=5  # Allow up to 5% VRAM usage
    local memory_threshold_mb=500   # Allow up to 500MB memory usage

    if command -v rocm-smi >/dev/null 2>&1; then
        echo "Checking ROCm GPU VRAM usage..."
        # Check if any GPU has more than threshold VRAM allocated
        local high_usage=$(rocm-smi --showmemuse | grep -E "GPU Memory Allocated \(VRAM%\): ([6-9]|[1-9][0-9]|100)")
        if [ -n "$high_usage" ]; then
            echo "ERROR: VRAM usage exceeds threshold (${vram_threshold_percent}%) on some GPUs:"
            echo "$high_usage"
            rocm-smi --showmemuse
            return 1
        else
            echo "✓ VRAM usage is within acceptable limits on all GPUs"
            return 0
        fi
   fi
}

# If this script is run directly (not sourced), run the check
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -e
    check_vram_clear
fi
