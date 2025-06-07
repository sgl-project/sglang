#!/bin/bash

if [ "$1" = "rocm" ]; then
    echo "Running in ROCm mode"

    # Clean SGLang processes
    pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt' | xargs -r kill -9

else
    # Show current GPU status
    nvidia-smi

    # Clean SGLang processes
    pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt' | xargs -r kill -9

    # Clean all GPU processes if any argument is provided
    if [ $# -gt 0 ]; then
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
