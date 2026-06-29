#!/usr/bin/env bash
# preflight_nodes.sh - fail early and clearly on environment issues before
# the orchestrator burns time launching SGLang.
#
# Runs a battery of checks against PREFILL_NODE and DECODE_NODE (SSH-side)
# plus the docker container expected on each. Every check prints PASS/FAIL/WARN;
# the script exits non-zero if any FAIL was recorded.
#
# Required env (same as run_cross_node_pd.sh):
#   PREFILL_NODE  DECODE_NODE  PREFILL_IP  DECODE_IP
#   CONTAINER     REMOTE_WORKDIR
#
# Optional:
#   ROUTER_NODE        (default: PREFILL_NODE)
#   IB_DEVS            comma-list of RDMA devices to verify (e.g. rdma0,rdma1,...);
#                      if unset, just asserts >=1 rdma* device is present
#   PREFILL_PORT       default 30025
#
# NOTE: some nodes are Slurm-gated (pam_slurm_adopt). If an SSH check fails with
# "Connection closed" / "no active jobs on this node", allocate first:
#   salloc -A <account> -p <partition> -w <node> -t 02:00:00 --no-shell
#   DECODE_PORT        default 30100
#   BOOTSTRAP_PORT     default 8998
#   ROUTER_PORT        default 8000
#   DECODE_BOOT_PORT   default 9001

set -uo pipefail

: "${PREFILL_NODE:?}"
: "${DECODE_NODE:?}"
: "${PREFILL_IP:?}"
: "${DECODE_IP:?}"
: "${CONTAINER:?}"
: "${REMOTE_WORKDIR:?}"

ROUTER_NODE="${ROUTER_NODE:-$PREFILL_NODE}"
IB_DEVS="${IB_DEVS:-}"
PREFILL_PORT="${PREFILL_PORT:-30025}"
DECODE_PORT="${DECODE_PORT:-30100}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
DECODE_BOOT_PORT="${DECODE_BOOT_PORT:-9001}"

SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes)

FAILS=0
WARNS=0

pass() { printf '  [PASS] %s\n' "$*"; }
fail() { printf '  [FAIL] %s\n' "$*"; FAILS=$((FAILS+1)); }
warn() { printf '  [WARN] %s\n' "$*"; WARNS=$((WARNS+1)); }
hdr()  { printf '\n=== %s ===\n' "$*"; }

# Run a remote command; capture combined output. Returns the remote exit code.
rsh() {
    local tgt="$1"; shift
    ssh "${SSH_OPTS[@]}" "$tgt" "$@"
}

# Run inside the container on a node; returns the docker exec exit code.
in_container() {
    local tgt="$1" cmd="$2"
    ssh "${SSH_OPTS[@]}" "$tgt" "docker exec ${CONTAINER} bash -c $(printf '%q' "$cmd")"
}

# ---------------------------------------------------------------------------
check_node() {
    local role="$1" tgt="$2" peer_ip="$3"
    hdr "$role  ($tgt)"

    # 1. SSH reachability
    if out=$(rsh "$tgt" 'echo OK; hostname; hostname -I' 2>&1); then
        pass "ssh OK; remote hostname=$(echo "$out" | sed -n 2p)"
        echo "         remote IPs: $(echo "$out" | sed -n 3p)"
    else
        fail "ssh to $tgt failed: $out"
        return  # no point running the rest
    fi

    # 2. rocm-smi + compute partition
    # Count physical GPUs via VRAM "Total Used" lines (one per device) rather than
    # --showid, which over-reports on these nodes (XCD/partition rows inflate it).
    if out=$(rsh "$tgt" 'rocm-smi --showmeminfo vram 2>&1' 2>&1); then
        gpu_count=$(echo "$out" | grep -cE 'Total Used' || true)
        if (( gpu_count >= 1 )); then
            pass "rocm-smi reports $gpu_count GPU(s)"
            if (( gpu_count > 8 )); then
                warn "  >8 GPUs means compute-partition (CPX). TP=8 will fail. Need SPX."
            fi
        else
            fail "rocm-smi present but no GPUs visible (amdgpu blacklisted?)"
        fi
    else
        fail "rocm-smi not runnable on host: $out"
    fi
    rsh "$tgt" 'rocm-smi --showcomputepartition 2>&1 | head -20' \
        2>&1 | sed 's/^/         /'

    # 3. /dev/kfd, /dev/dri/renderD*, /dev/infiniband
    out=$(rsh "$tgt" 'ls -l /dev/kfd 2>&1 | head -1; ls /dev/dri/renderD* 2>/dev/null | wc -l; ls /dev/infiniband 2>/dev/null | tr "\n" " "')
    echo "$out" | sed 's/^/         /'
    if echo "$out" | grep -q '/dev/kfd'; then pass "/dev/kfd present"; else fail "/dev/kfd missing"; fi
    renderd=$(echo "$out" | sed -n 2p)
    if [ "${renderd:-0}" -gt 0 ]; then
        pass "/dev/dri/renderD* count = $renderd"
        if [ "$renderd" -gt 8 ]; then
            warn "  >8 renderD nodes -> compute-partitioned. Need SPX flip."
        fi
    else
        fail "no /dev/dri/renderD* devices"
    fi
    if echo "$out" | sed -n 3p | grep -q '.'; then
        pass "/dev/infiniband populated"
    else
        fail "/dev/infiniband empty or missing"
    fi

    # 4. docker available
    if rsh "$tgt" 'docker version --format "client {{.Client.Version}} / server {{.Server.Version}}"' 2>&1; then
        pass "docker reachable"
    else
        fail "docker not available or daemon down"
    fi

    # 5. container existence + state
    cstate=$(rsh "$tgt" "docker inspect -f '{{.State.Status}}' ${CONTAINER} 2>/dev/null" || true)
    if [ -z "$cstate" ]; then
        fail "container '$CONTAINER' does not exist on $tgt (setup_container.sh will fail loudly)"
    else
        echo "         container state: $cstate"
        case "$cstate" in
            running) pass "container '$CONTAINER' running" ;;
            exited|created|paused) warn "container exists but state=$cstate (setup_container.sh will start it)" ;;
            *) warn "container in unexpected state: $cstate" ;;
        esac
    fi

    # 6. REMOTE_WORKDIR inside container (skip if container not running)
    if [ "$cstate" = "running" ]; then
        if in_container "$tgt" "test -d $REMOTE_WORKDIR" 2>/dev/null; then
            pass "REMOTE_WORKDIR=$REMOTE_WORKDIR exists in container"
        else
            fail "REMOTE_WORKDIR=$REMOTE_WORKDIR missing in container"
        fi
        if in_container "$tgt" "test -f $REMOTE_WORKDIR/cache_bench.py" 2>/dev/null; then
            pass "cache_bench.py present in REMOTE_WORKDIR"
        else
            warn "cache_bench.py NOT in $REMOTE_WORKDIR (orchestrator will fail at benchmark step)"
        fi
    fi

    # 7. expected ports free (check INSIDE the container, since that's where sglang binds)
    if [ "$cstate" = "running" ]; then
        local pcheck
        case "$role" in
            prefill) pcheck="$PREFILL_PORT $BOOTSTRAP_PORT" ;;
            decode)  pcheck="$DECODE_PORT $DECODE_BOOT_PORT" ;;
            router)  pcheck="$ROUTER_PORT" ;;
        esac
        for p in $pcheck; do
            if in_container "$tgt" "ss -ltn 2>/dev/null | awk '{print \$4}' | grep -q ':${p}\$'"; then
                fail "port $p already in use inside container on $tgt"
            else
                pass "port $p free inside container"
            fi
        done
    fi

    # 8. RDMA devices present (host side). Pensando RoCE enumerates as rdma0..rdma7.
    found_devs=$(rsh "$tgt" "ibv_devices 2>/dev/null | awk 'NR>2 && \$1 ~ /^rdma/ {print \$1}' | sort -V | paste -sd,")
    if [ -n "$found_devs" ]; then
        ndev=$(echo "$found_devs" | tr ',' '\n' | grep -c .)
        pass "RDMA devices on host: $found_devs ($ndev)"
        if [ -n "$IB_DEVS" ]; then
            for d in ${IB_DEVS//,/ }; do
                if echo ",$found_devs," | grep -q ",$d,"; then
                    pass "  requested IB device $d present"
                else
                    fail "  requested IB device $d NOT present on $tgt"
                fi
            done
        fi
    else
        fail "no rdma* devices found on $tgt (ibv_devices empty) — RoCE fabric down?"
    fi

    # 9. reachability of PEER ip from this node (ICMP only; service not up yet)
    if rsh "$tgt" "ping -c 2 -W 2 $peer_ip" >/dev/null 2>&1; then
        pass "peer IP $peer_ip pingable from $tgt"
    else
        fail "peer IP $peer_ip NOT pingable from $tgt"
    fi
}

# ---------------------------------------------------------------------------
hdr "preflight start"
echo "PREFILL_NODE = $PREFILL_NODE      PREFILL_IP = $PREFILL_IP"
echo "DECODE_NODE  = $DECODE_NODE       DECODE_IP  = $DECODE_IP"
echo "ROUTER_NODE  = $ROUTER_NODE"
echo "CONTAINER    = $CONTAINER"
echo "REMOTE_WORKDIR = $REMOTE_WORKDIR"
echo "IB_DEVS      = ${IB_DEVS:-(auto-discover rdma*)}"

check_node prefill "$PREFILL_NODE" "$DECODE_IP"
check_node decode  "$DECODE_NODE"  "$PREFILL_IP"

# Router-specific check only if router is on a third node
if [ "$ROUTER_NODE" != "$PREFILL_NODE" ] && [ "$ROUTER_NODE" != "$DECODE_NODE" ]; then
    check_node router "$ROUTER_NODE" "$PREFILL_IP"
fi

hdr "preflight summary"
echo "FAILS=$FAILS  WARNS=$WARNS"
if [ "$FAILS" -gt 0 ]; then
    echo "preflight FAILED — fix the items above before running run_cross_node_pd.sh"
    exit 1
fi
if [ "$WARNS" -gt 0 ]; then
    echo "preflight passed with warnings — review before continuing"
fi
echo "preflight OK"
