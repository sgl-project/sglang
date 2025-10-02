#!/bin/bash

# Source the VRAM checking function
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/check_vram_clear.sh"

if [ "$1" = "rocm" ]; then
    echo "Running in ROCm mode"

    # Show current GPU status
    rocm-smi --showmemuse

    # Clean SGLang processes
    pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt' | xargs -r kill -9

    # If VRAM is still allocated, try more aggressive cleanup
    if ! check_vram_clear; then
        echo "Attempting aggressive cleanup..."
        # Kill all processes using KFD
        lsof /dev/kfd 2>/dev/null | awk 'NR>1 {print $2}' | xargs -r kill -9 2>/dev/null || true
        # Try to reset GPUs (use with caution)
        # rocm-smi --gpureset (commented out as it might be too aggressive)
    fi

    # Show GPU status after clean up
    rocm-smi --showmemuse

else
    # Show current GPU status
    nvidia-smi

    # Clean SGLang processes
    pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt' | xargs -r kill -9

    # Clean all GPU processes if any argument is provided or if VRAM is still allocated
    if [ $# -gt 0 ] || ! check_vram_clear; then
        echo "Attempting aggressive cleanup..."
        # Check if sudo is available
        if command -v sudo >/dev/null 2>&1; then
            sudo apt-get update
            sudo apt-get install -y lsof
        else
            apt-get update
            apt-get install -y lsof
        fi
        kill -9 $(nvidia-smi | sed -n '/Processes:/,$p' | grep "   [0-9]" | awk '{print $5}') 2>/dev/null
        lsof /dev/nvidia* | awk '{print $2}' | xargs kill -9 2>/dev/null
    fi

    # Show GPU status after clean up
    nvidia-smi
fi
