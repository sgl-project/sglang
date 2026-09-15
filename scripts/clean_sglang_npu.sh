#!/bin/bash
set -u

echo "=================================================="
echo " SGLang NPU process cleanup"
echo "=================================================="

# Extract only SGLang-related PIDs shown by npu-smi.
PIDS=$(npu-smi info 2>/dev/null | awk -F'|' '
/sglang|sglangschedul/ {
    pid=$4
    gsub(/[[:space:]]/, "", pid)
    if (pid ~ /^[0-9]+$/) print pid
}' | sort -u)

if [ -z "${PIDS}" ]; then
    echo "[OK] No SGLang-related NPU processes found."
    npu-smi info
    exit 0
fi

echo "[INFO] SGLang-related NPU PIDs:"
echo "${PIDS}"
echo

echo "[INFO] Process details:"
ps -o user,pid,ppid,etime,cmd -p $(echo "${PIDS}" | tr '\n' ' ') 2>/dev/null || true
echo

echo "[1/3] Sending SIGTERM..."
kill -TERM ${PIDS} 2>/dev/null || true

# Give SGLang/HCCL workers a moment to exit cleanly.
sleep 5

LEFT=""
for pid in ${PIDS}; do
    if kill -0 "${pid}" 2>/dev/null; then
        LEFT="${LEFT} ${pid}"
    fi
done

if [ -n "${LEFT}" ]; then
    echo "[2/3] Processes still alive:${LEFT}"
    echo "[INFO] Sending SIGKILL..."
    kill -KILL ${LEFT} 2>/dev/null || true
    sleep 2
else
    echo "[2/3] All processes exited normally."
fi

echo
echo "[3/3] Current NPU status:"
npu-smi info

echo
echo "[DONE] Cleanup finished."
