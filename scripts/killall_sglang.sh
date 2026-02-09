#!/bin/bash

# Usage:
#   ./killall_sglang.sh              - Kill SGLang processes only (NVIDIA mode)
#   ./killall_sglang.sh rocm         - Kill SGLang processes only (ROCm mode)
#   ./killall_sglang.sh all          - Kill all GPU processes (NVIDIA mode)
#   ./killall_sglang.sh gpus 0,1,2,3 - Kill all processes on specific GPUs

if [ "$1" = "rocm" ]; then
    echo "Running in ROCm mode"

    # Clean SGLang processes
    pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt|sgl_diffusion::' | xargs -r kill -9

elif [ "$1" = "gpus" ] && [ -n "$2" ]; then
    # Kill all processes on specific GPUs only
    echo "Killing all processes on GPUs: $2"

    # Show current GPU status
    nvidia-smi

    # Build device file list from GPU IDs (e.g., "0,1,2,3" -> "/dev/nvidia0 /dev/nvidia1 ...")
    devices=$(echo "$2" | tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sed 's|^|/dev/nvidia|' | tr '\n' ' ')
    echo "Targeting devices: $devices"

    # Kill all processes using specified GPU devices
    [ -n "$devices" ] && lsof $devices 2>/dev/null | awk 'NR>1 {print $2}' | sort -u | xargs -r kill -9 2>/dev/null

    # Show GPU status after clean up
    nvidia-smi

else
    # Show current GPU status
    nvidia-smi

    # Clean SGLang processes
    pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt|sgl_diffusion::' | xargs -r kill -9

    # Clean all GPU processes if "all" argument is provided
    if [ "$1" = "all" ]; then
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
