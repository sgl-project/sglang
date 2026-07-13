#!/bin/bash
#
# Detect and (optionally) recover *hung / unresponsive* AMD GPUs during CI
# pre-flight.
#
# Relationship to ensure_vram_clear.sh:
#   * ensure_vram_clear.sh targets *leaked VRAM* -- memory allocated with no
#     owning process (the ROCm/aiter#2061 zombie-KFD signature).
#   * THIS script targets a *hung* GPU -- one that is unresponsive or has logged
#     a hardware fault -- which can wedge a node even when VRAM reads clean.
#     A hung GPU is the usual cause of the exit-134 "HW Exception ... reason:
#     GPU Hang" failures seen on the MI35x runners; landing a job on such a GPU
#     just fails again until the device is reset or the node is rebooted.
#
# Detection is host-level (no container / HIP runtime required):
#   1. rocm-smi responsiveness -- a per-device query that times out or errors is
#      a strong signal the GPU/driver is not responding. Repeated to avoid
#      transient false positives under load.
#   2. Kernel log (dmesg) signatures -- best-effort, node-level: recent amdgpu
#      ring timeouts / GPU reset / page faults / "GPU hang". Used only to
#      corroborate / bias toward recovery, never as the sole gate (dmesg lines
#      map to PCI/DRM ids, not rocm-smi indices, and may be unreadable in a pod).
#
# Recovery: a scoped `rocm-smi --gpureset -d <idx>`, applied ONLY to a GPU that
#   (a) looks hung AND (b) has no process holding its device files. The
#   device-file guard means we never reset a GPU another tenant is actively
#   using -- a GPU running someone else's job always has a holder. (A hung GPU
#   whose stuck process is still attached is intentionally NOT auto-reset here;
#   we warn and let the operator / ensure_vram_clear kill the holder first.)
#
# Behaviour is gated by AMD_CI_GPU_HANG_RESET:
#   auto (default) : detect + attempt guarded reset; abort (exit 1) only if a
#                    confirmed hang could not be recovered.
#   warn           : detect + report only, never reset, never fail the job.
#   0/off/false    : disable entirely (no-op, exit 0).
#
# Exit codes:
#   0 - all target GPUs healthy, or a hung GPU was recovered, or disabled/warn.
#   1 - a GPU is hung and could not be recovered (node needs drain / reboot).

set -uo pipefail

# ------------------------------------------------------------------ config ----
SMI_QUERY_TIMEOUT="${SMI_QUERY_TIMEOUT:-20}"   # per rocm-smi probe, seconds
GPU_RESET_TIMEOUT="${GPU_RESET_TIMEOUT:-90}"   # per rocm-smi --gpureset, seconds
PROBE_RETRIES="${PROBE_RETRIES:-3}"            # confirmations before "hung"
PROBE_RETRY_WAIT="${PROBE_RETRY_WAIT:-5}"      # seconds between probe retries
DMESG_TAIL_LINES="${DMESG_TAIL_LINES:-2000}"

# --------------------------------------------------------------- utilities ----
log() { echo "[gpu-health] $*"; }

# All GPU indices rocm-smi knows about (one per line).
_all_gpu_indices() {
    command -v rocm-smi >/dev/null 2>&1 || return 0
    timeout "$SMI_QUERY_TIMEOUT" rocm-smi --showid 2>/dev/null \
        | grep -oE 'GPU\[[0-9]+\]' | grep -oE '[0-9]+' | sort -un
}

# Probe a single GPU's responsiveness once.
# Returns 0 if the device answered a lightweight query, non-zero otherwise
# (124 = rocm-smi itself timed out => strongly suggests a wedged device/driver).
_probe_gpu_once() {
    local idx="$1" out rc
    out=$(timeout "$SMI_QUERY_TIMEOUT" rocm-smi -d "$idx" --showuse --showmemuse 2>&1)
    rc=$?
    if [ "$rc" -eq 124 ]; then
        return 124   # timed out -> unresponsive
    fi
    if [ "$rc" -ne 0 ]; then
        return 1     # rocm-smi errored for this device
    fi
    # A responsive device reports a numeric "GPU use (%)" and/or VRAM% line.
    # Treat explicit error strings as unhealthy even on rc==0.
    if echo "$out" | grep -qiE 'unable to|not responding|error|failed'; then
        return 1
    fi
    if ! echo "$out" | grep -qE 'GPU (use|Memory Allocated)'; then
        return 1     # no usable telemetry came back
    fi
    return 0
}

# Confirm a GPU is hung across PROBE_RETRIES attempts (avoids transient blips).
_gpu_is_hung() {
    local idx="$1" attempt rc
    for attempt in $(seq 1 "$PROBE_RETRIES"); do
        if _probe_gpu_once "$idx"; then
            return 1   # answered at least once -> not hung
        fi
        rc=$?
        log "GPU ${idx}: unresponsive probe ${attempt}/${PROBE_RETRIES} (rc=${rc})"
        [ "$attempt" -lt "$PROBE_RETRIES" ] && sleep "$PROBE_RETRY_WAIT"
    done
    return 0   # never answered -> hung
}

# Best-effort: does the kernel log show recent GPU-hang/reset signatures?
# Node-level only (not per-index). Returns 0 if signatures are present.
_dmesg_shows_hang() {
    command -v dmesg >/dev/null 2>&1 || return 1
    dmesg 2>/dev/null | tail -n "$DMESG_TAIL_LINES" \
        | grep -qiE 'ring .*timeout|gpu reset|GPU hang|amdgpu.*(reset|fault)|VM_L2.*fault|MES.*failed'
}

# PIDs (host namespace) holding any GPU device file. Empty => no host holder.
# NOTE: cannot see into container PID namespaces; in CI the container is stopped
# before this runs, so an empty result means the GPU is genuinely unowned.
_gpu_device_holders() {
    local pids=""
    if command -v fuser >/dev/null 2>&1; then
        pids+=" $(fuser /dev/kfd 2>/dev/null || true)"
        for dev in /dev/dri/renderD*; do
            [ -e "$dev" ] || continue
            pids+=" $(fuser "$dev" 2>/dev/null || true)"
        done
    elif command -v lsof >/dev/null 2>&1; then
        pids+=" $(lsof -t /dev/kfd 2>/dev/null || true)"
        for dev in /dev/dri/renderD*; do
            [ -e "$dev" ] || continue
            pids+=" $(lsof -t "$dev" 2>/dev/null || true)"
        done
    fi
    echo "$pids" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u || true
}

# --------------------------------------------------------------- main flow ----
check_gpu_health() {
    local mode="${AMD_CI_GPU_HANG_RESET:-auto}"
    case "${mode,,}" in
        0|false|off|no|disable|disabled)
            log "GPU hang probe disabled (AMD_CI_GPU_HANG_RESET=${mode}); skipping."
            return 0
            ;;
    esac

    if ! command -v rocm-smi >/dev/null 2>&1; then
        log "rocm-smi not available; cannot probe GPU health (skipping)."
        return 0
    fi

    log "=== GPU health probe on $(hostname) ($(date -u +%FT%TZ)) ==="

    local gpus
    gpus=$(_all_gpu_indices)
    if [ -z "$gpus" ]; then
        log "No GPUs enumerated by rocm-smi; nothing to probe."
        return 0
    fi

    if _dmesg_shows_hang; then
        log "NOTE: kernel log shows recent GPU hang/reset/fault signatures on this node."
    fi

    # Phase 1: identify hung GPUs.
    local hung=() idx
    for idx in $gpus; do
        if _gpu_is_hung "$idx"; then
            log "GPU ${idx}: HUNG (unresponsive after ${PROBE_RETRIES} probes)."
            hung+=("$idx")
        fi
    done

    if [ "${#hung[@]}" -eq 0 ]; then
        log "All GPUs responsive; no hang detected."
        return 0
    fi

    # warn mode: report only.
    if [ "${mode,,}" = "warn" ]; then
        log "WARN mode: hung GPU(s) [${hung[*]}] detected; not resetting."
        log "Set AMD_CI_GPU_HANG_RESET=auto to enable scoped recovery."
        return 1
    fi

    # Phase 2: attempt guarded, scoped recovery.
    local holders unrecovered=()
    for idx in "${hung[@]}"; do
        holders=$(_gpu_device_holders)
        if [ -n "$holders" ]; then
            # A stuck process still owns a GPU device file. Resetting now could
            # disturb it (or another tenant). Leave it to the operator /
            # ensure_vram_clear to clear holders first.
            log "GPU ${idx}: hung but device-file holders still present (PIDs: $(echo "$holders" | tr '\n' ' ')); NOT resetting."
            unrecovered+=("$idx")
            continue
        fi

        log "GPU ${idx}: hung with no owning process -> rocm-smi --gpureset -d ${idx}"
        if timeout "$GPU_RESET_TIMEOUT" rocm-smi --gpureset -d "$idx" 2>&1; then
            sleep 10
            if _gpu_is_hung "$idx"; then
                log "GPU ${idx}: still hung after reset."
                unrecovered+=("$idx")
            else
                log "GPU ${idx}: recovered after scoped reset."
            fi
        else
            log "GPU ${idx}: reset command failed/timed out."
            unrecovered+=("$idx")
        fi
    done

    if [ "${#unrecovered[@]}" -gt 0 ]; then
        log "=== FAILED: GPU(s) [${unrecovered[*]}] hung and NOT recovered ==="
        log "Node needs to be drained/rebooted before it can run jobs reliably."
        return 1
    fi

    log "=== All hung GPUs recovered via scoped reset ==="
    return 0
}

# Run when executed directly (not when sourced, so tests can import helpers).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    check_gpu_health "$@"
fi
