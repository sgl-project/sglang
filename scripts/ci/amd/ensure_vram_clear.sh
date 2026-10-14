#!/bin/bash

# Source the VRAM checking function
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/check_vram_clear.sh"

# Stop and remove every container that holds any /dev/kfd or /dev/dri device.
# Some failing CI runs leave behind containers other than `ci_sglang` (e.g.
# from previous AMD jobs that were force-killed mid-run); those still hold
# VRAM via KFD even though the host pgrep finds nothing.
stop_all_gpu_containers() {
    if ! command -v docker >/dev/null 2>&1; then
        return 0
    fi

    local all_ids
    all_ids=$(docker ps -aq 2>/dev/null || true)
    if [ -z "$all_ids" ]; then
        echo "No docker containers found on host"
        return 0
    fi

    local gpu_ids=""
    local cid
    for cid in $all_ids; do
        # A container is "GPU-attached" if its inspect output mentions any
        # GPU device or NVIDIA/ROCm GPU capability. Inspecting the raw JSON
        # (instead of a specific field) survives docker version differences.
        if docker inspect "$cid" 2>/dev/null \
            | grep -qE '"PathOnHost":"/dev/(kfd|dri)|"Capabilities":\[\["gpu"\]\]'; then
            gpu_ids+=" $cid"
        fi
    done

    gpu_ids=$(echo "$gpu_ids" | tr ' ' '\n' | grep -E '^[a-f0-9]+$' || true)
    if [ -z "$gpu_ids" ]; then
        echo "No GPU-attached docker containers found on host"
        return 0
    fi

    echo "Found GPU-attached containers, stopping them:"
    for cid in $gpu_ids; do
        docker ps -a --filter "id=$cid" --format '  {{.ID}} {{.Image}} {{.Status}} {{.Names}}' 2>/dev/null || true
    done
    echo "$gpu_ids" | xargs -r docker stop --time 5 2>/dev/null || true
    echo "$gpu_ids" | xargs -r docker rm -f 2>/dev/null || true
}

# Find and kill any host process that holds an open handle to /dev/kfd or
# /dev/dri/renderD*. This is far more reliable than `rocm-smi --showpids`,
# which only sees processes that registered a HSA queue (zombies and
# processes that crashed mid-init are invisible to it).
kill_processes_holding_gpu_devices() {
    local signal=${1:-TERM}
    local pids=""

    # If neither tool is present we silently degrade to a no-op, which used
    # to look identical in the log to "no holders found" and made the
    # script's failures very confusing. Emit a loud warning so the runner
    # owner knows why GPU device cleanup isn't happening.
    if ! command -v fuser >/dev/null 2>&1 && ! command -v lsof >/dev/null 2>&1; then
        echo "WARNING: neither fuser nor lsof installed on the host;" \
             "cannot detect processes holding /dev/kfd or /dev/dri/renderD*."
        echo "         Install psmisc (for fuser) or lsof on the runner host" \
             "to enable device-fd-based cleanup."
        return 0
    fi

    if command -v fuser >/dev/null 2>&1; then
        # `fuser` prints PIDs to stdout, names to stderr; collect everything
        # that has any handle on KFD or render nodes.
        pids+=" $(fuser /dev/kfd 2>/dev/null || true)"
        for dev in /dev/dri/renderD*; do
            [ -e "$dev" ] || continue
            pids+=" $(fuser "$dev" 2>/dev/null || true)"
        done
    fi

    if command -v lsof >/dev/null 2>&1; then
        pids+=" $(lsof -t /dev/kfd 2>/dev/null || true)"
        for dev in /dev/dri/renderD*; do
            [ -e "$dev" ] || continue
            pids+=" $(lsof -t "$dev" 2>/dev/null || true)"
        done
    fi

    pids=$(echo "$pids" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u || true)
    if [ -z "$pids" ]; then
        return 0
    fi

    local self_pid=$$
    echo "Processes holding /dev/kfd or /dev/dri/renderD*:"
    for pid in $pids; do
        # Skip our own PID and any of our ancestors so we don't suicide.
        if [ "$pid" = "$self_pid" ]; then
            continue
        fi
        local cmd
        cmd=$(ps -p "$pid" -o pid,ppid,stat,cmd --no-headers 2>/dev/null || true)
        if [ -z "$cmd" ]; then
            continue
        fi
        echo "  $cmd"
        # If it's a zombie, kill the parent instead — kill -9 on a zombie is
        # a no-op, the only way to reap it is to make its parent reap it.
        local stat
        stat=$(ps -p "$pid" -o stat= 2>/dev/null | tr -d ' ')
        if [[ "$stat" == Z* ]]; then
            local ppid
            ppid=$(ps -p "$pid" -o ppid= 2>/dev/null | tr -d ' ')
            if [ -n "$ppid" ] && [ "$ppid" != "1" ] && [ "$ppid" != "$self_pid" ]; then
                echo "    -> $pid is a zombie, sending SIG$signal to parent $ppid"
                kill "-$signal" "$ppid" 2>/dev/null || true
            fi
            continue
        fi
        kill "-$signal" "$pid" 2>/dev/null || true
    done
}

# Print rich diagnostics that explain *why* VRAM is still allocated when no
# obvious owner exists. Helpful when zombies / other namespaces hold memory.
dump_gpu_diagnostics() {
    echo "=== GPU device file holders (fuser) ==="
    if command -v fuser >/dev/null 2>&1; then
        fuser -v /dev/kfd 2>&1 || true
        for dev in /dev/dri/renderD* /dev/dri/card*; do
            [ -e "$dev" ] || continue
            fuser -v "$dev" 2>&1 || true
        done
    else
        echo "fuser not installed"
    fi

    echo "=== GPU device file holders (lsof) ==="
    if command -v lsof >/dev/null 2>&1; then
        lsof /dev/kfd 2>/dev/null || true
        lsof /dev/dri/renderD* 2>/dev/null || true
    else
        echo "lsof not installed"
    fi

    echo "=== Zombie processes on host ==="
    ps -eo pid,ppid,stat,etime,cmd 2>/dev/null | awk 'NR==1 || $3 ~ /^Z/ {print}'

    echo "=== Docker containers on host ==="
    if command -v docker >/dev/null 2>&1; then
        docker ps -a --format 'table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}' 2>/dev/null || true
    fi

    echo "=== rocm-smi --showpids ==="
    timeout 30 rocm-smi --showpids 2>&1 || echo "rocm-smi --showpids timed out"

    echo "=== rocm-smi --showmemuse ==="
    timeout 30 rocm-smi --showmemuse 2>&1 || echo "rocm-smi --showmemuse timed out"
}

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

    # Always stop the well-known CI container first (best-effort).
    echo "Stopping any existing ci_sglang container..."
    docker stop ci_sglang 2>/dev/null || true
    docker rm -f ci_sglang 2>/dev/null || true

    # Show initial GPU status
    echo "=== Initial GPU Memory Status ==="
    rocm-smi --showmemuse
    echo "=================================="

    # Fast path: if the runner is already clean, skip the cleanup loop
    # entirely so healthy jobs don't pay the ~35s/attempt cleanup cost.
    if check_vram_clear; then
        echo "✓ VRAM is already clear; skipping cleanup."
        return 0
    fi

    while [ $retry_count -lt $max_retries ]; do
        echo "=== Cleanup Attempt $((retry_count + 1))/$max_retries ==="

        # Step 1: kill SGLang-named processes on the host (cheap, fast).
        # NOTE: host pgrep cannot see PIDs inside a container's PID
        # namespace, so in CI this almost never matches anything; the
        # heavy lifting is done by step 2 below. Kept as a fast early
        # cleanup for the rare case where something runs on the host.
        echo "Killing SGLang processes..."
        pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt' \
            | xargs -r kill -9 2>/dev/null || true

        # Step 2: aggressive cleanup. Run on EVERY attempt — the previous
        # version skipped this on attempt 1, which made attempt 1 a near
        # no-op for the most common failure mode (a leftover container
        # holding VRAM, invisible to host pgrep).
        echo "Performing aggressive cleanup..."

        # 2a. Stop ALL GPU-attached containers, not just ci_sglang. A
        # leftover container from a previous job will keep VRAM held even
        # though `pgrep` on the host shows nothing.
        stop_all_gpu_containers

        # 2b. SIGTERM anything that has /dev/kfd or /dev/dri/renderD* open.
        # `lsof`/`fuser` see processes that `rocm-smi --showpids` misses
        # (notably zombies and processes outside our PID namespace).
        echo "Sending SIGTERM to processes holding GPU device files..."
        kill_processes_holding_gpu_devices TERM
        sleep 5

        # 2c. SIGKILL anything still holding GPU device files.
        echo "Sending SIGKILL to remaining holders..."
        kill_processes_holding_gpu_devices KILL

        # 2d. Best-effort: also kill anything `rocm-smi --showpids` reports.
        # Handles both the legacy "PID: <n>" line format and the modern
        # tabular format (`<pid>\t<name>\t<gpus>\t...`); the previous
        # `grep 'PID:'` matched nothing on ROCm 5+ tabular output.
        rocm-smi --showpids 2>/dev/null \
            | awk '/^PID:[[:space:]]*[0-9]+/ {print $2} /^[0-9]+/ {print $1}' \
            | xargs -r kill -9 2>/dev/null || true

        echo "Waiting 30 seconds for VRAM to clear..."
        sleep 30

        # Step 3: re-check.
        echo "Checking VRAM status..."
        if check_vram_clear; then
            echo "✓ VRAM cleanup successful after $((retry_count + 1)) attempts"
            return 0
        else
            echo "✗ VRAM still not clear after attempt $((retry_count + 1))"
            # Step 4: dump diagnostics on every failed attempt so the next
            # attempt's logs already explain WHY cleanup didn't work.
            # Without this we'd only see what's holding the GPU at the very
            # end, which makes triage much harder.
            echo "--- Diagnostics for failed attempt $((retry_count + 1)) ---"
            dump_gpu_diagnostics
            echo "--- End of diagnostics for attempt $((retry_count + 1)) ---"
            retry_count=$((retry_count + 1))
        fi
    done

    # Failed after all retries — diagnostics for the last cleanup attempt
    # were already dumped above; just print the actionable hint.
    echo "=== FAILED: VRAM cleanup unsuccessful after $max_retries attempts ==="
    echo "(See diagnostics above for the final attempt.)"
    echo "=================================================================="
    echo "Hint: if no host process / container holds the GPU but VRAM is"
    echo "still allocated, this is almost certainly a zombie KFD context"
    echo "(see ROCm/aiter#2061). The node will need to be rebooted before"
    echo "subsequent jobs can succeed."
    echo "=================================================================="
    return 1
}

# If this script is run directly (not sourced), run the ensure function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -e
    ensure_vram_clear "$@"
fi
