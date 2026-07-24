#!/bin/bash
# Ensure the Intel XPU runner is ready to accept a new CI job.
#
# Ordered like AMD's ensure_vram_clear.sh:
#   1. Fast path: if the runner already looks clean, exit immediately.
#   2. Cleanup path: stop stale GPU-attached containers, kill any host
#      process holding /dev/dri/renderD*, wait, re-check.
#   3. Retry up to XPU_CLEAR_MAX_RETRIES times; give up loudly if not clean.
#
# This is the single missing piece that lets one job's device-lost /
# leaked-process leave the next job in a state where model weight-load
# silently hangs for 600s. See the "Root cause" section in this repo's
# XPU CI stability notes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./check_xpu_clear.sh
source "$SCRIPT_DIR/check_xpu_clear.sh"

XPU_CLEAR_MAX_RETRIES=${XPU_CLEAR_MAX_RETRIES:-3}
XPU_CLEAR_WAIT_SECS=${XPU_CLEAR_WAIT_SECS:-15}
XPU_CI_CONTAINER=${XPU_CI_CONTAINER:-ci_sglang_xpu}

_have() { command -v "$1" >/dev/null 2>&1; }

# Stop any container that holds /dev/dri (GPU-attached), not just our
# named one. A crash in the previous job can leave a foreign container
# holding the render node; the host's fuser will see its PID but not its
# name, so `docker rm -f ci_sglang_xpu` alone is not sufficient.
stop_all_xpu_containers() {
    if ! _have docker; then
        return 0
    fi
    local all_ids
    all_ids=$(docker ps -aq 2>/dev/null || true)
    [ -z "$all_ids" ] && return 0

    local gpu_ids=""
    for cid in $all_ids; do
        # `docker inspect` output survives docker CLI version drift, unlike
        # relying on a specific JSON path.
        if docker inspect "$cid" 2>/dev/null | grep -q '"PathOnHost":"/dev/dri'; then
            gpu_ids+=" $cid"
        fi
    done
    gpu_ids=$(echo "$gpu_ids" | tr ' ' '\n' | grep -E '^[a-f0-9]+$' || true)
    [ -z "$gpu_ids" ] && return 0

    echo "Stopping GPU-attached containers:"
    for cid in $gpu_ids; do
        docker ps -a --filter "id=$cid" --format '  {{.ID}} {{.Image}} {{.Status}} {{.Names}}' 2>/dev/null || true
    done
    echo "$gpu_ids" | xargs -r docker stop --time 5 2>/dev/null || true
    echo "$gpu_ids" | xargs -r docker rm -f 2>/dev/null || true
}

# Kill any host process with an open fd to /dev/dri/renderD*.
# fuser / lsof see zombies and cross-PID-namespace holders that xpu-smi
# does not, which is the whole point.
kill_processes_holding_dri() {
    local signal=${1:-TERM}

    if ! _have fuser && ! _have lsof; then
        echo "WARNING: neither fuser nor lsof installed; cannot kill /dev/dri holders." >&2
        echo "         Install psmisc (fuser) or lsof on the runner host." >&2
        return 0
    fi

    local pids=""
    if _have fuser; then
        for dev in /dev/dri/renderD*; do
            [ -e "$dev" ] || continue
            pids+=" $(fuser "$dev" 2>/dev/null || true)"
        done
    fi
    if _have lsof; then
        for dev in /dev/dri/renderD*; do
            [ -e "$dev" ] || continue
            pids+=" $(lsof -t "$dev" 2>/dev/null || true)"
        done
    fi
    pids=$(echo "$pids" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u || true)
    [ -z "$pids" ] && return 0

    local self_pid=$$
    echo "Processes holding /dev/dri/renderD*:"
    for pid in $pids; do
        # Skip our own PID — killing ourselves obviously doesn't help.
        [ "$pid" = "$self_pid" ] && continue
        local cmd stat ppid
        cmd=$(ps -p "$pid" -o pid,ppid,stat,cmd --no-headers 2>/dev/null || true)
        [ -z "$cmd" ] && continue
        echo "  $cmd"
        stat=$(ps -p "$pid" -o stat= 2>/dev/null | tr -d ' ')
        # Zombies ignore SIGKILL; the only way to reap them is via the parent.
        if [[ "$stat" == Z* ]]; then
            ppid=$(ps -p "$pid" -o ppid= 2>/dev/null | tr -d ' ')
            if [ -n "$ppid" ] && [ "$ppid" != "1" ] && [ "$ppid" != "$self_pid" ]; then
                echo "    -> zombie; sending SIG${signal} to parent ${ppid}"
                kill "-${signal}" "$ppid" 2>/dev/null || true
            fi
            continue
        fi
        kill "-${signal}" "$pid" 2>/dev/null || true
    done
}

ensure_xpu_ready() {
    echo "=== XPU runner readiness ==="
    echo "Hostname: $(hostname)"
    echo "Date: $(date)"

    # Always remove our own well-known container first (best-effort).
    if _have docker; then
        docker rm -f "${XPU_CI_CONTAINER}" 2>/dev/null || true
    fi

    # Fast path: healthy runners skip the expensive cleanup loop entirely.
    if check_xpu_clear; then
        echo "✓ XPU runner already clean; skipping cleanup."
        return 0
    fi

    local attempt=0
    while [ $attempt -lt "$XPU_CLEAR_MAX_RETRIES" ]; do
        attempt=$((attempt + 1))
        echo "=== Cleanup attempt ${attempt}/${XPU_CLEAR_MAX_RETRIES} ==="

        stop_all_xpu_containers

        echo "Sending SIGTERM to /dev/dri holders..."
        kill_processes_holding_dri TERM
        sleep 5
        echo "Sending SIGKILL to remaining /dev/dri holders..."
        kill_processes_holding_dri KILL

        echo "Waiting ${XPU_CLEAR_WAIT_SECS}s for driver to release the device..."
        sleep "$XPU_CLEAR_WAIT_SECS"

        if check_xpu_clear; then
            echo "✓ XPU runner cleaned after ${attempt} attempt(s)."
            return 0
        fi
        echo "✗ Still not clean after attempt ${attempt}."
    done

    # Final diagnostic dump so the failure is triage-able without SSH.
    "$SCRIPT_DIR/dump_xpu_diagnostics.sh" || true

    echo "=================================================================="
    echo "FAILED: XPU runner could not be brought to a clean state."
    echo "This host is likely holding a Level-Zero device-lost state that"
    echo "needs a driver reload or reboot; further jobs on it will fail."
    echo "=================================================================="
    return 1
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -e
    ensure_xpu_ready "$@"
fi
