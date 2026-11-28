#!/bin/bash
# Entrypoint for the DeepSWE eval container (amd64 pod).
#
# Runs next to a `docker:dind` sidecar: install the docker CLI + compose plugin
# (pier shells out to both), wait for the daemon, then drive evalscope via
# run_deepswe_eval.py. Config comes from env vars injected by
# test_npu_accuracy_utils.run_evalscope_on_x86_pod.

set -euo pipefail

export DOCKER_HOST="${DOCKER_HOST:-tcp://127.0.0.1:2375}"

echo "===== DeepSWE eval container ====="
echo "arch=$(uname -m) python=$(python3 -V 2>&1) DOCKER_HOST=${DOCKER_HOST}"
echo "api_url=${DEEPSWE_API_URL:-} datasets=${DEEPSWE_DATASETS:-} limit=${DEEPSWE_LIMIT:-} timeout=${DEEPSWE_TIMEOUT:-} metrics_dir=${METRICS_DATA_FILE:-}"
echo "sglang commit: $(git -C /root/sglang rev-parse --short HEAD 2>/dev/null || echo unknown)"

if ! command -v docker >/dev/null 2>&1; then
    echo "Installing docker CLI and compose plugin ..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
        docker.io docker-compose-v2 ca-certificates curl git
fi

echo "Waiting for the Docker daemon at ${DOCKER_HOST} ..."
docker_ready=0
for _ in $(seq 1 150); do
    if docker version >/dev/null 2>&1; then
        docker_ready=1
        break
    fi
    sleep 2
done
if [ "${docker_ready}" != "1" ]; then
    echo "ERROR: Docker daemon is not reachable at ${DOCKER_HOST}"
    docker version || true
    exit 1
fi
docker version
docker compose version || echo "WARNING: docker compose plugin is still unavailable"

python3 /root/sglang/python/sglang/test/ascend/e2e/run_deepswe_eval.py
