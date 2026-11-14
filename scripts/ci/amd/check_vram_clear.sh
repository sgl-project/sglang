#!/bin/bash

# Return 0 for clear, 1 for verified busy, and 2 for unavailable telemetry.
check_vram_clear() {
    local vram_threshold_percent=5  # Allow up to 5% VRAM usage
    local memory_usage usage_status=0

    if ! command -v rocm-smi >/dev/null 2>&1; then
        echo "ERROR: rocm-smi is unavailable; cannot verify GPU VRAM usage."
        return 2
    fi

    echo "Checking ROCm GPU VRAM usage..."
    if ! memory_usage=$(rocm-smi --showmemuse); then
        echo "ERROR: rocm-smi failed; cannot verify GPU VRAM usage."
        return 2
    fi
    printf '%s\n' "$memory_usage"
    awk -v threshold="$vram_threshold_percent" '
        /GPU Memory Allocated \(VRAM%\):/ {
            count++
            sub(/^.*GPU Memory Allocated \(VRAM%\):[[:space:]]*/, "")
            if ($0 !~ /^[0-9]+([.][0-9]+)?[[:space:]]*$/ || $0 + 0 > 100) {
                invalid = 1
            } else if ($0 + 0 > threshold) {
                busy = 1
            }
        }
        END {
            if (!count || invalid) exit 2
            if (busy) exit 1
        }
    ' <<< "$memory_usage" || usage_status=$?

    case "$usage_status" in
        0) echo "✓ VRAM usage is within acceptable limits on all GPUs" ;;
        1) echo "ERROR: VRAM usage exceeds threshold (${vram_threshold_percent}%) on some GPUs." ;;
        *) echo "ERROR: rocm-smi returned unrecognized VRAM usage; cannot verify GPUs."
           return 2 ;;
    esac
    return "$usage_status"
}

# If this script is run directly (not sourced), run the check
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -e
    check_vram_clear
fi
