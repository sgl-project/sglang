#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${ENV_FILE:-${SCRIPT_DIR}/env.example}"

case "${ADMIN_API_KEY:-}" in
  ""|replace-with-*|changeme|CHANGE_ME)
    echo "ADMIN_API_KEY must be set to a non-placeholder secret" >&2
    exit 2
    ;;
esac

ROLE="${1:?usage: run_worker.sh prefill|decode <local-bind-ip-or-0.0.0.0>}"
LOCAL_IP="${2:?usage: run_worker.sh prefill|decode <local-bind-ip-or-0.0.0.0>}"

if [[ "${ROLE}" != "prefill" && "${ROLE}" != "decode" ]]; then
  echo "ROLE must be prefill or decode, got: ${ROLE}" >&2
  exit 2
fi

mounts=(-v "${SGLANG_REPO}:/sgl-workspace/sglang")
if [[ -d "${MODEL_PATH}" ]]; then
  mounts+=(-v "${MODEL_PATH}:${MODEL_PATH}:ro")
fi
if [[ -d /dev/infiniband ]]; then
  mounts+=(-v /dev/infiniband:/dev/infiniband)
fi

server_args=(
  python3 -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --host "${LOCAL_IP}"
  --port "${PORT}"
  --tp "${TP_SIZE}"
  --dp "${DP_SIZE}"
  --enable-dp-attention
  --disaggregation-mode "${ROLE}"
  --disaggregation-transfer-backend "${TRANSFER_BACKEND}"
  --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}"
  --disaggregation-ib-device "${IB_DEVICE}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
)

if [[ "${ENABLE_PD_FLIP_STATE_MACHINE:-1}" == "1" ]]; then
  server_args+=(--enable-pd-flip-state-machine)
fi

if [[ "${ENABLE_PD_RUNTIME_ROLE_SWITCH:-${ENABLE_PD_FLIP_STATE_MACHINE:-1}}" == "1" ]]; then
  server_args+=(--enable-pd-runtime-role-switch)
fi

if [[ "${ENABLE_PD_FLIP_HICACHE_STITCH:-1}" == "1" ]]; then
  server_args+=(
    --enable-pd-flip-hicache-stitch
    --disaggregation-decode-enable-radix-cache
    --enable-hierarchical-cache
    --hicache-storage-backend "${HICACHE_STORAGE_BACKEND:-mooncake}"
    --hicache-write-policy "${HICACHE_WRITE_POLICY:-write_through}"
  )
fi

server_args+=(--admin-api-key "${ADMIN_API_KEY}")

if [[ -n "${EXTRA_SGLANG_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${EXTRA_SGLANG_ARGS})
  server_args+=("${extra_args[@]}")
fi

printf -v server_cmd '%q ' "${server_args[@]}"
launch_cmd="cd /sgl-workspace/sglang && PYTHONPATH=python exec ${server_cmd}"

# shellcheck disable=SC2206
extra_docker_args=(${EXTRA_DOCKER_ARGS:-})
for name in \
  MOONCAKE_MASTER \
  MOONCAKE_TE_META_DATA_SERVER \
  MOONCAKE_GLOBAL_SEGMENT_SIZE \
  MOONCAKE_PROTOCOL \
  MOONCAKE_DEVICE; do
  if [[ -n "${!name:-}" && "${EXTRA_DOCKER_ARGS:-}" != *"${name}"* ]]; then
    extra_docker_args+=(-e "${name}=${!name}")
  fi
done
if [[ "${EXTRA_DOCKER_ARGS:-}" != *"MOONCAKE_LOCAL_HOSTNAME"* ]]; then
  host_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  if [[ -n "${host_ip}" ]]; then
    extra_docker_args+=(-e "MOONCAKE_LOCAL_HOSTNAME=${host_ip}")
  fi
fi

exec docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  --privileged \
  "${extra_docker_args[@]}" \
  "${mounts[@]}" \
  "${IMAGE}" \
  bash -lc "${launch_cmd}"
