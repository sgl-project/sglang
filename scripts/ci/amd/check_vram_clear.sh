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

# Print the indices of GPUs whose VRAM usage is above threshold, one per line.
# Used by ensure_vram_clear.sh to scope a last-resort `rocm-smi --gpureset` to
# only the leaked GPUs, so we never touch a device another job is still using.
#
# rocm-smi --showmemuse emits one line per card, e.g.:
#   GPU[0]          : GPU Memory Allocated (VRAM%): 95
# We extract the card index whenever that percentage is >= 6.
get_dirty_gpu_indices() {
    command -v rocm-smi >/dev/null 2>&1 || return 0
    # Keep only the per-card lines whose VRAM% is >= 6 (same regex as the
    # check above), then pull the card index out of the "GPU[N]" prefix.
    # Uses only grep/sort so it works with mawk-only runners too.
    rocm-smi --showmemuse 2>/dev/null \
        | grep -E "GPU Memory Allocated \(VRAM%\): ([6-9]|[1-9][0-9]|100)" \
        | grep -oE 'GPU\[[0-9]+\]' \
        | grep -oE '[0-9]+' \
        | sort -un
}

# If this script is run directly (not sourced), run the check
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -e
    check_vram_clear
fi
