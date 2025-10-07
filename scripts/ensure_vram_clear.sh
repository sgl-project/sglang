#!/bin/bash

# Source the VRAM checking function
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/check_vram_clear.sh"

ensure_vram_clear() {
    local max_retries=3
    local retry_count=0

    # Stop and remove any existing ci_sglang container
    echo "Stopping any existing ci_sglang container..."
    docker stop ci_sglang || true
    docker rm ci_sglang || true

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
            rocm-smi --showpids 2>/dev/null | grep 'PID:' | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
            # Wait a bit for cleanup to take effect
            echo "Waiting 30 seconds for VRAM to clear..."
            sleep 30
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
    timeout 30 rocm-smi --showmemuse || echo "rocm-smi timed out"
    echo "Processes using GPU:"
    rocm-smi --showpids 2>/dev/null | grep -q 'PID:' || echo "No processes found using /dev/kfd"

    # Print detailed information about suspicious processes
    echo "=== Detailed Process Information ==="
    if command -v rocm-smi >/dev/null 2>&1; then
        # For AMD GPUs, get processes from rocm-smi --showpids
        kfd_pids=$(rocm-smi --showpids 2>/dev/null | grep 'PID:' | awk '{print $2}' | sort -u)
        if [ -n "$kfd_pids" ]; then
            echo "Processes accessing /dev/kfd (AMD GPU device):"
            for pid in $kfd_pids; do
                if ps -p $pid -o pid,ppid,cmd --no-headers 2>/dev/null; then
                    echo "  └─ Command line: $(ps -p $pid -o cmd --no-headers 2>/dev/null | head -1)"
                else
                    echo "  └─ PID $pid: Process not found or already terminated"
                fi
            done
        else
            echo "No processes found accessing /dev/kfd"
        fi
    fi

    # Check for any remaining sglang-related processes
    echo "Checking for any remaining sglang-related processes:"
    sglang_procs=$(pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt' 2>/dev/null)
    if [ -n "$sglang_procs" ]; then
        echo "Found sglang processes still running:"
        for pid in $sglang_procs; do
            ps -p $pid -o pid,ppid,cmd --no-headers 2>/dev/null || echo "PID $pid not found"
        done
    else
        echo "No sglang-related processes found."
    fi

    echo "=================================================================="
    return 1
}

# If this script is run directly (not sourced), run the ensure function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -e
    ensure_vram_clear "$@"
fi
