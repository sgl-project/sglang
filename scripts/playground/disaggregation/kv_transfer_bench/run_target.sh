#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_IMAGE="${SGLANG_IMAGE:-sglang-pd-switch:tianciJ}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/kv-transfer-bench}"
HOST_IP="${HOST_IP:-$(hostname -I | awk '{print $1}')}"
GPU_ID="${GPU_ID:-0}"
IB_DEVICE="${IB_DEVICE:-mlx5_0}"
PROTOCOL="${PROTOCOL:-${MOONCAKE_PROTOCOL:-rdma}}"
MAX_BYTES="${MAX_BYTES:-2GB}"
TARGET_INFO_FILE="${TARGET_INFO_FILE:-/tmp/kv-transfer-bench/target-info.json}"

mkdir -p "${OUTPUT_DIR}"

if [[ -n "${SGLANG_REPO:-}" && -d "${SGLANG_REPO}" ]]; then
  mounts=(-v "${SGLANG_REPO}:/workspace/sglang:ro")
  workdir=/workspace/sglang
  pythonpath=/workspace/sglang/python:/workspace/sglang
  bench_script=scripts/playground/disaggregation/kv_transfer_bench/kv_transfer_latency.py
else
  mounts=(-v "${SCRIPT_DIR}:/workspace/kv_transfer_bench:ro")
  workdir=/workspace/kv_transfer_bench
  pythonpath=
  bench_script=kv_transfer_latency.py
fi
mounts+=(-v "${OUTPUT_DIR}:/tmp/kv-transfer-bench")
if [[ -d /dev/infiniband ]]; then
  mounts+=(-v /dev/infiniband:/dev/infiniband)
fi

extra_env=(-e "MOONCAKE_PROTOCOL=${PROTOCOL}" -e "IB_DEVICE=${IB_DEVICE}")
if [[ -n "${MC_FORCE_TCP:-}" ]]; then
  extra_env+=(-e "MC_FORCE_TCP=${MC_FORCE_TCP}")
fi

exec docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  --privileged \
  --ulimit memlock=-1:-1 \
  "${extra_env[@]}" \
  "${mounts[@]}" \
  "${SGLANG_IMAGE}" \
  bash -lc "cd '${workdir}' && PYTHONPATH='${pythonpath}' python3 '${bench_script}' --role target --host '${HOST_IP}' --gpu-id '${GPU_ID}' --ib-device '${IB_DEVICE}' --protocol '${PROTOCOL}' --max-bytes '${MAX_BYTES}' --target-info-file '${TARGET_INFO_FILE}'"
