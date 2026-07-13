#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/sglang}"
SUITE_DIR="${1:-${REPO}/experiments/pd_flip_hicache_stitch_$(date +%Y%m%d_%H%M%S)}"

export RUN_NAME="${RUN_NAME:-01_1p3d_to_2p2d_hicache_stitch}"
export MODE="${MODE:-one_p_three_d_hicache_stitch}"
export TRACE_DIR_NAME="${TRACE_DIR_NAME:-trace_1p3d_to_2p2d_hicache_stitch}"
export REQUESTS="${REQUESTS:-40}"
export INTERVAL_SECONDS="${INTERVAL_SECONDS:-0.5}"
export SHORT_CHARS="${SHORT_CHARS:-1000}"
export LONG_CHARS="${LONG_CHARS:-10000}"
export SHORT_COUNT="${SHORT_COUNT:-20}"
export LONG_COUNT="${LONG_COUNT:-20}"

export EXTRA_SGLANG_ARGS_SUFFIX="${EXTRA_SGLANG_ARGS_SUFFIX:-\
--enable-hierarchical-cache \
--hicache-storage-backend mooncake \
--disaggregation-decode-enable-radix-cache}"

export EXTRA_DOCKER_ARGS_SUFFIX="${EXTRA_DOCKER_ARGS_SUFFIX:-\
-e SGLANG_PD_FLIP_HICACHE_STITCH=1 \
-e MOONCAKE_MASTER=192.168.0.42:50051 \
-e MOONCAKE_TE_META_DATA_SERVER=http://192.168.0.42:8080/metadata \
-e MOONCAKE_PROTOCOL=rdma \
-e MOONCAKE_DEVICE=mlx5_0 \
-e MOONCAKE_GLOBAL_SEGMENT_SIZE=4gb}"

exec bash "${REPO}/experiments/pd_flip_1p3d_to_2p2d_runner.sh" "${SUITE_DIR}"
