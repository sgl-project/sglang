#!/bin/bash
# Stop the ci_sglang_xpu container and wipe what the run wrote into the workspace.
# Usage: xpu_ci_cleanup.sh   (run from the workspace root)
CONTAINER_NAME="${CONTAINER_NAME:-ci_sglang_xpu}"
WORKSPACE="${GITHUB_WORKSPACE:-$PWD}"

# pip install ran as root inside the container against the
# bind-mounted workspace, so build artifacts are root-owned on
# the host. Chown them back before rm to avoid Permission denied.
docker run --rm -v "${WORKSPACE}:/w" busybox:latest \
  chown -R "$(id -u):$(id -g)" /w || true
rm -rf \
  python/build \
  python/dist \
  python/sglang.egg-info \
  python/sglang/*.egg-info \
  test/result.jsonl \
  test/results \
  test/.pytest_cache \
  .pytest_cache || true
find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
find . -type f -name "*.pyc" -delete || true
# SIGTERM sglang and drain GPU context before `docker rm -f`;
# SIGKILL leaves the xe/GuC exec queue registered and triggers a
# GT reset (+ devcoredump) on B580.
if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker exec "${CONTAINER_NAME}" bash -c '
    pkill -TERM -f "sglang|run_suite|python3.*test_" 2>/dev/null || true
    for _ in $(seq 1 30); do
      pgrep -f "sglang::|sglang.launch_server" >/dev/null || break
      sleep 1
    done
    pkill -KILL -f "sglang|run_suite" 2>/dev/null || true
  ' || true
fi
docker rm -f "${CONTAINER_NAME}" || true
if [[ -n "${CI_SGLANG_XPU_IMAGE:-}" ]]; then
  docker rmi -f "${CI_SGLANG_XPU_IMAGE}" || true
fi
