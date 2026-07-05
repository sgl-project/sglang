#!/bin/bash
# Dump diagnostics after a suspected Level-Zero device-lost or hang.
# Called from the workflow with `if: failure()`, and from ensure_xpu_ready.sh
# as the last-resort artifact before we give up on the runner.
#
# We deliberately do NOT `set -e` here — every probe is best-effort. A dead
# driver often makes xpu-smi itself hang, so every command is wrapped in
# `timeout` to keep this bounded.

echo "===================================================================="
echo "== XPU diagnostics dump — $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "== Host: $(hostname)"
echo "===================================================================="

echo "--- xpu-smi discovery -----------------------------------------------"
if command -v xpu-smi >/dev/null 2>&1; then
    timeout 20 xpu-smi discovery 2>&1 || echo "(xpu-smi discovery timed out)"
else
    echo "(xpu-smi not installed on host)"
fi

echo "--- xpu-smi stats (device 0) ---------------------------------------"
if command -v xpu-smi >/dev/null 2>&1; then
    timeout 20 xpu-smi stats -d 0 2>&1 || echo "(xpu-smi stats timed out)"
fi

echo "--- xpu-smi dump ---------------------------------------------------"
# xpu-smi dump captures the internal driver state — the equivalent of a
# CUDA coredump for Intel GPUs. Directed at stdout so it lands in the
# GitHub Actions log without needing artifact upload plumbing.
if command -v xpu-smi >/dev/null 2>&1; then
    timeout 30 xpu-smi dump -m 0,1,2,3,4,5 -n 1 2>&1 || echo "(xpu-smi dump failed)"
fi

echo "--- /dev/dri holders (fuser) ---------------------------------------"
if command -v fuser >/dev/null 2>&1; then
    for dev in /dev/dri/renderD* /dev/dri/card*; do
        [ -e "$dev" ] || continue
        echo "fuser $dev:"
        timeout 10 fuser -v "$dev" 2>&1 || true
    done
else
    echo "(fuser not installed)"
fi

echo "--- /dev/dri holders (lsof) ----------------------------------------"
if command -v lsof >/dev/null 2>&1; then
    timeout 15 lsof /dev/dri/renderD* /dev/dri/card* 2>/dev/null || true
else
    echo "(lsof not installed)"
fi

echo "--- Zombie processes on host ---------------------------------------"
ps -eo pid,ppid,stat,etime,cmd 2>/dev/null | awk 'NR==1 || $3 ~ /^Z/ {print}'

echo "--- Docker containers on host --------------------------------------"
if command -v docker >/dev/null 2>&1; then
    timeout 15 docker ps -a --format 'table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}' 2>&1 || true
fi

echo "--- kernel ring buffer tail (last 200 lines) -----------------------"
# dmesg often surfaces Level-Zero / i915 / xe kernel messages that explain
# UR_RESULT_ERROR_DEVICE_LOST better than any userspace tool.
if command -v dmesg >/dev/null 2>&1; then
    timeout 10 dmesg --ctime 2>/dev/null | tail -n 200 \
        || timeout 10 dmesg 2>/dev/null | tail -n 200 \
        || echo "(dmesg requires elevated privileges)"
fi

echo "--- /proc/driver/dri (if present) ----------------------------------"
if [ -d /proc/driver/dri ]; then
    ls -la /proc/driver/dri 2>/dev/null || true
fi

echo "===================================================================="
echo "== End of XPU diagnostics dump"
echo "===================================================================="
