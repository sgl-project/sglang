#!/bin/bash
# Verify the Intel XPU (Battlemage) runner is in a clean, usable state.
#
# Returns 0 if:
#   - xpu-smi can enumerate at least one device
#   - GPU memory usage is below XPU_MEM_THRESHOLD_PCT (default 5)
#   - no process on the host has an open handle to /dev/dri/renderD*
#
# Returns 1 if any of the above fails. The caller (ensure_xpu_ready.sh)
# uses this as a fast-path health check before the expensive cleanup path,
# and as the post-cleanup verifier.

XPU_MEM_THRESHOLD_PCT=${XPU_MEM_THRESHOLD_PCT:-5}

_have() { command -v "$1" >/dev/null 2>&1; }

check_xpu_clear() {
    # xpu-smi lives in the Intel driver stack on the host runner. If it isn't
    # installed we can still do a fd-holder check, but we cannot verify
    # memory state — degrade to fd-only and warn loudly.
    local mem_ok=0
    if _have xpu-smi; then
        # `xpu-smi discovery` prints a boxed table:
        #   | 0         | Device Name: Intel(R) Arc(TM) B580 Graphics
        #   |           | PCI BDF Address: 0000:18:00.0
        # Row starts with `|`, then whitespace, then the id, then `|`.
        # Extract that id column and dedupe (each device spans multiple lines).
        local discovery
        discovery=$(timeout 15 xpu-smi discovery 2>/dev/null || true)
        local dev_ids
        dev_ids=$(echo "$discovery" \
            | awk -F'|' '$2 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/ {
                gsub(/[[:space:]]/,"",$2); print $2
              }' \
            | sort -un)
        if [ -z "$dev_ids" ]; then
            echo "ERROR: xpu-smi enumerated 0 devices — driver likely wedged." >&2
            echo "$discovery" | sed 's/^/  xpu-smi: /' >&2
            return 1
        fi
        echo "✓ xpu-smi sees device(s): $(echo "$dev_ids" | tr '\n' ' ')"

        # Memory check: iterate EVERY discovered device. Hardcoding `-d 0`
        # would miss dGPU on a mixed iGPU+dGPU host (dGPU is often id 1),
        # and would silently pass on multi-XPU hosts where only device 1
        # is contaminated.
        local id mem_report high
        for id in $dev_ids; do
            mem_report=$(timeout 15 xpu-smi stats -d "$id" 2>/dev/null || true)
            [ -z "$mem_report" ] && continue
            high=$(echo "$mem_report" \
                | awk -v t="$XPU_MEM_THRESHOLD_PCT" \
                    'tolower($0) ~ /memory util/ {
                        for (i=1;i<=NF;i++) if ($i ~ /^[0-9.]+$/ && $i+0 > t) print $0
                    }')
            if [ -n "$high" ]; then
                echo "ERROR: device ${id} memory utilization exceeds ${XPU_MEM_THRESHOLD_PCT}%:" >&2
                echo "$high" | sed 's/^/  /' >&2
                return 1
            fi
            mem_ok=1
        done
    else
        echo "WARNING: xpu-smi not installed on host; skipping enumeration + memory check." >&2
    fi

    # File-descriptor check: any process holding /dev/dri/renderD* keeps the
    # Level-Zero device attached and can wedge the next job. This catches
    # zombies + processes from other PID namespaces that xpu-smi misses.
    local holders=""
    if _have fuser; then
        for dev in /dev/dri/renderD*; do
            [ -e "$dev" ] || continue
            holders+=" $(fuser "$dev" 2>/dev/null || true)"
        done
    elif _have lsof; then
        for dev in /dev/dri/renderD*; do
            [ -e "$dev" ] || continue
            holders+=" $(lsof -t "$dev" 2>/dev/null || true)"
        done
    else
        echo "WARNING: neither fuser nor lsof present; cannot detect /dev/dri holders." >&2
        # If we couldn't check memory AND couldn't check fds, we know nothing.
        if [ "$mem_ok" -eq 0 ]; then
            echo "ERROR: no viable health-check tool available on runner." >&2
            return 1
        fi
    fi
    holders=$(echo "$holders" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u || true)
    if [ -n "$holders" ]; then
        echo "ERROR: /dev/dri/renderD* still held by processes:" >&2
        for pid in $holders; do
            ps -p "$pid" -o pid,ppid,stat,cmd --no-headers 2>/dev/null \
                | sed 's/^/  /' >&2 || true
        done
        return 1
    fi

    echo "✓ /dev/dri render nodes have no open holders."
    return 0
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -e
    check_xpu_clear
fi
