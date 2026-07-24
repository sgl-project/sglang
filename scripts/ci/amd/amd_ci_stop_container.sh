#!/bin/bash
# Gracefully tear down the ci_sglang container at the end of a job.
#
# Why this exists: the AMD CI jobs launch `ci_sglang` but historically never
# stopped it -- cleanup was left entirely to the *next* job's
# "Ensure VRAM is clear" step. When the ephemeral runner pod is recycled (or
# force-deleted) between jobs, the still-running sglang server is SIGKILLed
# mid-HIP-operation, which leaves an unreclaimable KFD context and leaks VRAM
# with no owning process (ROCm/aiter#2061). Running this as an `if: always()`
# teardown step lets sglang free its GPU memory *before* the pod terminates.
#
# This script is intentionally best-effort and never fails the job: a teardown
# problem must not turn a green test run red. Set -e is deliberately NOT used.

set -uo pipefail

CONTAINER="${1:-ci_sglang}"
STOP_TIMEOUT="${DOCKER_STOP_TIMEOUT:-60}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker not available; nothing to tear down."
    exit 0
fi

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "Container '$CONTAINER' not found; nothing to tear down."
    exit 0
fi

echo "=== Gracefully stopping container '$CONTAINER' ==="

# Step 1: ask the sglang processes inside the container to shut down cleanly.
# SIGINT lets sglang run its normal teardown (kill_process_tree + torch
# CUDA/HIP context teardown) so VRAM is released. pkill is best-effort; the
# container may already be idle.
echo "Signalling sglang processes inside the container (SIGINT)..."
docker exec "$CONTAINER" bash -c '
    pkill -INT -f "sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt|sglang serve" 2>/dev/null || true
' 2>/dev/null || true

# Give sglang a few seconds to unwind before the harder stop.
sleep 5

# Step 2: `docker stop` with a generous grace period. --init (see
# amd_ci_start_container.sh) forwards this SIGTERM to the process group and
# reaps children, so remaining GPU contexts are torn down cleanly instead of
# being SIGKILLed.
echo "docker stop --time ${STOP_TIMEOUT} ${CONTAINER}..."
docker stop --time "$STOP_TIMEOUT" "$CONTAINER" 2>/dev/null || true

# Step 3: remove the container so the name is free for the next job.
docker rm -f "$CONTAINER" 2>/dev/null || true

# Step 4: best-effort visibility into whether VRAM actually came back.
if command -v rocm-smi >/dev/null 2>&1; then
    echo "=== Post-teardown VRAM status ==="
    timeout 30 rocm-smi --showmemuse 2>&1 || echo "rocm-smi --showmemuse timed out"
fi

echo "=== Teardown of '$CONTAINER' complete ==="
exit 0
