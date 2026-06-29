#!/usr/bin/env bash
# discover_network.sh - dump per-node networking / RDMA info so we can decide
# whether PREFILL_IP / DECODE_IP should be the SSH-side IP (10.245.x.x) or a
# separate bnxt/RDMA-side address.
#
# Pure read-only. No changes. Output is dumped to stdout; redirect to a file
# if you want to diff between nodes.
#
# Required env: any of NODES="node1 node2 ..." or the standard PREFILL_NODE/
# DECODE_NODE/ROUTER_NODE trio.

set -uo pipefail

SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes)

if [ -z "${NODES:-}" ]; then
    NODES="${PREFILL_NODE:-} ${DECODE_NODE:-} ${ROUTER_NODE:-}"
fi
# de-dup, strip empties
NODES=$(echo $NODES | tr ' ' '\n' | awk 'NF && !seen[$0]++' | tr '\n' ' ')
if [ -z "$NODES" ]; then
    echo "set NODES=\"a@1.2.3.4 b@5.6.7.8\" (or PREFILL_NODE / DECODE_NODE / ROUTER_NODE)" >&2
    exit 2
fi

section() { printf '\n--- %s ---\n' "$*"; }

for tgt in $NODES; do
    echo
    echo "============================================================"
    echo "=== node: $tgt"
    echo "============================================================"

    ssh "${SSH_OPTS[@]}" "$tgt" 'bash -s' <<'REMOTE'
set +e

section() { printf '\n--- %s ---\n' "$*"; }

section "hostname / uname"
hostname
hostname -I
uname -r

section "ip -br addr"
ip -br addr

section "ip -br link"
ip -br link

section "ip route (default)"
ip route show default

section "rdma link (kernel rdma subsystem)"
if command -v rdma >/dev/null; then
    rdma link 2>&1
else
    echo "rdma cli not installed"
fi

section "ibv_devinfo (libibverbs)"
if command -v ibv_devinfo >/dev/null; then
    ibv_devinfo 2>&1 | head -200
else
    echo "ibv_devinfo not installed"
fi

section "/sys/class/infiniband"
if [ -d /sys/class/infiniband ]; then
    for dev in /sys/class/infiniband/*; do
        [ -e "$dev" ] || continue
        d=$(basename "$dev")
        echo "[$d]"
        for f in node_guid sys_image_guid fw_ver hca_type board_id; do
            [ -r "$dev/$f" ] && printf "  %-15s %s\n" "$f" "$(cat $dev/$f)"
        done
        # ports + their IPv4 GID-mapped address (RoCE shows IPs in gid_attrs/ndevs)
        for port in "$dev"/ports/*; do
            [ -d "$port" ] || continue
            pnum=$(basename "$port")
            state=$(cat "$port/state" 2>/dev/null)
            rate=$(cat "$port/rate" 2>/dev/null)
            phys=$(cat "$port/phys_state" 2>/dev/null)
            echo "  port $pnum: state=$state  phys=$phys  rate=$rate"
            if [ -d "$port/gid_attrs/ndevs" ]; then
                for gn in "$port/gid_attrs/ndevs"/*; do
                    [ -r "$gn" ] || continue
                    nd=$(cat "$gn" 2>/dev/null)
                    [ -n "$nd" ] || continue
                    idx=$(basename "$gn")
                    ipv4=$(ip -4 -br addr show "$nd" 2>/dev/null | awk '{print $3}')
                    echo "    gid[$idx] -> netdev=$nd  ipv4=$ipv4"
                done
            fi
        done
    done
else
    echo "no /sys/class/infiniband"
fi

section "RDMA driver modules / dmesg"
lsmod 2>/dev/null | grep -E '^(ionic|bnxt|mlx5|irdma)' || echo "no known RDMA driver modules matched"
dmesg 2>/dev/null | grep -iE 'ionic|bnxt_re|mlx5|RoCE' | tail -10 || true
REMOTE

done
