#!/bin/bash

check_vram_clear() {
    if command -v rocm-smi >/dev/null 2>&1; then
        echo "Checking ROCm GPU VRAM usage..."
        # Check if any GPU has VRAM allocated
        if rocm-smi --showmemuse | grep -q "GPU Memory Allocated (VRAM%): [1-9]"; then
            echo "ERROR: VRAM is still allocated on some GPUs"
            rocm-smi --showmemuse
            return 1
        else
            echo "✓ VRAM is clear on all GPUs"
            return 0
        fi
    elif command -v nvidia-smi >/dev/null 2>&1; then
        echo "Checking NVIDIA GPU memory usage..."
        # Check NVIDIA GPU memory usage
        if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | grep -q "^[1-9]"; then
            echo "ERROR: GPU memory is still allocated on some GPUs"
            nvidia-smi
            return 1
        else
            echo "✓ GPU memory is clear on all GPUs"
            return 0
        fi
    else
        echo "No GPU monitoring tool found, assuming VRAM is clear"
        return 0
    fi
}

# If this script is run directly (not sourced), run the check
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -e
    check_vram_clear
fi
