#!/bin/bash
# Management script for containerized CI runners on 5090 host machines.
#
# Usage:
#   ./manage-runners.sh start          - Launch all 8 runner containers
#   ./manage-runners.sh stop           - Stop all runner containers
#   ./manage-runners.sh restart        - Restart all runner containers
#   ./manage-runners.sh status         - Show container/runner status
#   ./manage-runners.sh rebuild        - Rebuild image and restart all
#   ./manage-runners.sh logs <N>       - View logs for runner N (0-7)
#
# Required environment variables:
#   RUNNER_TOKEN  - GitHub Actions runner registration token
#
# Optional environment variables:
#   MACHINE_NAME  - Machine identifier (default: auto-detect from hostname)
#   NUM_GPUS      - Number of GPUs/runners (default: 8)
#   IMAGE         - Docker image (default: ghcr.io/sgl-project/sglang/ci-5090:latest)
#   HF_CACHE_DIR  - Host HuggingFace cache (default: /tmp/huggingface)
set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-8}"
IMAGE="${IMAGE:-ghcr.io/sgl-project/sglang/ci-5090:latest}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/tmp/huggingface}"
CONTAINER_PREFIX="sglang-ci-runner"

# Auto-detect machine name from hostname (e.g. "5090a" or "5090b")
if [ -z "${MACHINE_NAME:-}" ]; then
    HOSTNAME_SHORT=$(hostname -s)
    case "$HOSTNAME_SHORT" in
        *5090a*) MACHINE_NAME="5090a" ;;
        *5090b*) MACHINE_NAME="5090b" ;;
        *)       MACHINE_NAME="$HOSTNAME_SHORT" ;;
    esac
fi

container_name() {
    echo "${CONTAINER_PREFIX}-${MACHINE_NAME}-gpu-${1}"
}

runner_name() {
    echo "${MACHINE_NAME}-gpu-${1}"
}

# ── Commands ────────────────────────────────────────────────────────────

cmd_start() {
    if [ -z "${RUNNER_TOKEN:-}" ]; then
        echo "ERROR: RUNNER_TOKEN is required"
        echo "Generate one at: https://github.com/sgl-project/sglang/settings/actions/runners/new"
        exit 1
    fi

    mkdir -p "${HF_CACHE_DIR}"

    echo "Starting ${NUM_GPUS} runner containers on ${MACHINE_NAME}..."
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        local cname
        cname=$(container_name "$gpu_id")
        local rname
        rname=$(runner_name "$gpu_id")

        # Skip if already running
        if docker ps --format '{{.Names}}' | grep -q "^${cname}$"; then
            echo "  [${gpu_id}] ${cname} already running, skipping"
            continue
        fi

        # Remove stopped container with same name
        docker rm -f "${cname}" 2>/dev/null || true

        echo "  [${gpu_id}] Starting ${cname} (GPU ${gpu_id})..."
        docker run -d \
            --name "${cname}" \
            --gpus "device=${gpu_id}" \
            --network host \
            --restart unless-stopped \
            --shm-size 16g \
            -v "${HF_CACHE_DIR}:/hf_home" \
            -e RUNNER_NAME="${rname}" \
            -e RUNNER_TOKEN="${RUNNER_TOKEN}" \
            -e RUNNER_LABELS="1-gpu-5090" \
            -e CUDA_VISIBLE_DEVICES=0 \
            -e HF_TOKEN="${HF_TOKEN:-}" \
            "${IMAGE}"
    done

    echo "All runners started."
}

cmd_stop() {
    echo "Stopping runner containers on ${MACHINE_NAME}..."
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        local cname
        cname=$(container_name "$gpu_id")
        if docker ps -a --format '{{.Names}}' | grep -q "^${cname}$"; then
            echo "  [${gpu_id}] Stopping ${cname}..."
            docker stop -t 30 "${cname}" 2>/dev/null || true
            docker rm "${cname}" 2>/dev/null || true
        fi
    done
    echo "All runners stopped."
}

cmd_restart() {
    cmd_stop
    cmd_start
}

cmd_status() {
    echo "=== Runner Status (${MACHINE_NAME}) ==="
    printf "%-5s %-35s %-12s %-20s\n" "GPU" "CONTAINER" "STATUS" "UPTIME"
    echo "--------------------------------------------------------------------"
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        local cname
        cname=$(container_name "$gpu_id")
        local status uptime
        status=$(docker inspect --format '{{.State.Status}}' "${cname}" 2>/dev/null || echo "not found")
        if [ "$status" = "running" ]; then
            uptime=$(docker inspect --format '{{.State.StartedAt}}' "${cname}" 2>/dev/null | cut -d. -f1 || echo "?")
        else
            uptime="-"
        fi
        printf "%-5s %-35s %-12s %-20s\n" "${gpu_id}" "${cname}" "${status}" "${uptime}"
    done
    echo ""
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
}

cmd_rebuild() {
    echo "Rebuilding CI image..."

    # Find repo root (this script lives in scripts/ci/5090/)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

    docker build -f "${REPO_ROOT}/docker/Dockerfile.ci-5090" \
        -t "${IMAGE}" \
        "${REPO_ROOT}"

    echo "Image rebuilt. Restarting runners..."
    cmd_restart
}

cmd_logs() {
    local gpu_id="${1:-}"
    if [ -z "$gpu_id" ]; then
        echo "Usage: $0 logs <GPU_ID>"
        exit 1
    fi
    local cname
    cname=$(container_name "$gpu_id")
    docker logs -f "${cname}"
}

# ── Main ────────────────────────────────────────────────────────────────
case "${1:-}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_restart ;;
    status)  cmd_status ;;
    rebuild) cmd_rebuild ;;
    logs)    cmd_logs "${2:-}" ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|rebuild|logs <N>}"
        exit 1
        ;;
esac
