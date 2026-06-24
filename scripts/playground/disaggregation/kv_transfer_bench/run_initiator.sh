#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_IMAGE="${SGLANG_IMAGE:-sglang-pd-switch:tianciJ}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/kv-transfer-bench}"
HOST_IP="${HOST_IP:-$(hostname -I | awk '{print $1}')}"
GPU_ID="${GPU_ID:-0}"
IB_DEVICE="${IB_DEVICE:-mlx5_0}"
PROTOCOL="${PROTOCOL:-${MOONCAKE_PROTOCOL:-rdma}}"
TARGET_INFO_FILE="${TARGET_INFO_FILE:-/tmp/kv-transfer-bench/target-info.json}"
SIZES="${SIZES:-1MB:1GB:x2}"
WARMUP="${WARMUP:-3}"
REPEAT="${REPEAT:-20}"
SUMMARY_CSV="${SUMMARY_CSV:-/tmp/kv-transfer-bench/summary.csv}"
SAMPLES_JSONL="${SAMPLES_JSONL:-/tmp/kv-transfer-bench/samples.jsonl}"
extra_cli_args=()
if [[ -n "${RATE_LIMIT_GBPS:-}" ]]; then
  extra_cli_args+=(--rate-limit-gbps "${RATE_LIMIT_GBPS}")
fi
if [[ -n "${CHUNK_SIZE:-}" ]]; then
  extra_cli_args+=(--chunk-size "${CHUNK_SIZE}")
fi
if [[ -n "${BACKGROUND_DURATION_SECONDS:-}" ]]; then
  extra_cli_args+=(--background-duration-seconds "${BACKGROUND_DURATION_SECONDS}")
fi
if [[ -n "${BACKGROUND_BYTES:-}" ]]; then
  extra_cli_args+=(--background-bytes "${BACKGROUND_BYTES}")
fi
extra_cli_args_quoted=""
if ((${#extra_cli_args[@]} > 0)); then
  printf -v extra_cli_args_quoted ' %q' "${extra_cli_args[@]}"
fi

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
if [[ -n "${TARGET_INFO_JSON:-}" ]]; then
  extra_env+=(-e "TARGET_INFO_JSON=${TARGET_INFO_JSON}")
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
  bash -lc "cd '${workdir}' && PYTHONPATH='${pythonpath}' python3 '${bench_script}' --role initiator --host '${HOST_IP}' --gpu-id '${GPU_ID}' --ib-device '${IB_DEVICE}' --protocol '${PROTOCOL}' --target-info-file '${TARGET_INFO_FILE}' --sizes '${SIZES}' --warmup '${WARMUP}' --repeat '${REPEAT}' --summary-csv '${SUMMARY_CSV}' --samples-jsonl '${SAMPLES_JSONL}'${extra_cli_args_quoted}"
