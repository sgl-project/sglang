#!/bin/bash
# Entrypoint for containerized CI runner on 5090 machines.
#
# Required environment variables:
#   RUNNER_NAME   - Unique runner name (e.g. "5090a-gpu-0")
#   RUNNER_TOKEN  - GitHub Actions runner registration token
#
# Optional environment variables:
#   RUNNER_ORG    - GitHub org (default: sgl-project)
#   RUNNER_REPO   - GitHub repo (default: sglang)
#   RUNNER_LABELS - Comma-separated labels (default: 1-gpu-5090)
#   RUNNER_GROUP  - Runner group (default: Default)
#   RUNNER_EPHEMERAL - Set to "1" for ephemeral mode (default: 0)
set -euo pipefail

RUNNER_ORG="${RUNNER_ORG:-sgl-project}"
RUNNER_REPO="${RUNNER_REPO:-sglang}"
RUNNER_LABELS="${RUNNER_LABELS:-1-gpu-5090}"
RUNNER_GROUP="${RUNNER_GROUP:-Default}"
RUNNER_EPHEMERAL="${RUNNER_EPHEMERAL:-0}"

if [ -z "${RUNNER_NAME:-}" ]; then
    echo "ERROR: RUNNER_NAME is required"
    exit 1
fi

if [ -z "${RUNNER_TOKEN:-}" ]; then
    echo "ERROR: RUNNER_TOKEN is required"
    exit 1
fi

# Ensure HF cache directory exists and is writable
mkdir -p "${HF_HOME:-/hf_home}"

echo "=== CI Runner Container ==="
echo "Runner name:  ${RUNNER_NAME}"
echo "Labels:       ${RUNNER_LABELS}"
echo "Repo:         ${RUNNER_ORG}/${RUNNER_REPO}"
echo "GPU:          CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo "==========================="

# Verify GPU access
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

cd /home/runner/actions-runner

# Remove any stale runner config from a previous container run
if [ -f .runner ]; then
    echo "Removing stale runner configuration..."
    ./config.sh remove --token "${RUNNER_TOKEN}" 2>/dev/null || true
fi

# Configure the runner
EPHEMERAL_FLAG=""
if [ "${RUNNER_EPHEMERAL}" = "1" ]; then
    EPHEMERAL_FLAG="--ephemeral"
fi

./config.sh \
    --url "https://github.com/${RUNNER_ORG}/${RUNNER_REPO}" \
    --token "${RUNNER_TOKEN}" \
    --name "${RUNNER_NAME}" \
    --labels "${RUNNER_LABELS}" \
    --runnergroup "${RUNNER_GROUP}" \
    --work _work \
    --replace \
    --unattended \
    ${EPHEMERAL_FLAG}

# Graceful shutdown: deregister runner on SIGTERM/SIGINT
# We run run.sh in the background and use `wait` so the bash process stays
# alive to receive signals (exec would replace bash and bypass the trap).
cleanup() {
    echo "Caught signal, removing runner..."
    kill -TERM "$RUNNER_PID" 2>/dev/null || true
    wait "$RUNNER_PID" 2>/dev/null || true
    ./config.sh remove --token "${RUNNER_TOKEN}" 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# Start the runner
echo "Starting GitHub Actions runner..."
./run.sh &
RUNNER_PID=$!
wait $RUNNER_PID
