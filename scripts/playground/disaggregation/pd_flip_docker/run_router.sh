#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${ENV_FILE:-${SCRIPT_DIR}/env.example}"

mounts=(-v "${SGLANG_REPO}:/sgl-workspace/sglang")
if [[ -d "${MODEL_PATH}" ]]; then
  mounts+=(-v "${MODEL_PATH}:${MODEL_PATH}:ro")
fi

args=(
  --host 0.0.0.0
  --port "${ROUTER_PORT}"
  --model-id "${MODEL_ID}"
)
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
  args+=(--tokenizer-path "${TOKENIZER_PATH}")
fi
args+=(--worker-urls "${NODE0}" "${NODE1}" "${NODE2}" "${NODE3}")

if [[ -n "${EXTRA_ROUTER_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_router_args=(${EXTRA_ROUTER_ARGS})
  args+=("${extra_router_args[@]}")
fi

printf -v router_args '%q ' "${args[@]}"

router_cmd="cd /sgl-workspace/sglang/experimental/sgl-router && \
if [ -x target/release/sgl-router ]; then \
  exec target/release/sgl-router ${router_args}; \
elif command -v cargo >/dev/null 2>&1; then \
  exec cargo run --release -- ${router_args}; \
else \
  echo 'sgl-router binary not found and cargo is unavailable. Build experimental/sgl-router/target/release/sgl-router first or use an image with cargo.' >&2; \
  exit 1; \
fi"

# shellcheck disable=SC2206
extra_docker_args=(${EXTRA_DOCKER_ARGS:-})

exec docker run --rm \
  --network host \
  "${extra_docker_args[@]}" \
  "${mounts[@]}" \
  "${IMAGE}" \
  bash -lc "${router_cmd}"
