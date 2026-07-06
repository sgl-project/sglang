#!/bin/bash
# Benchmark symmetric-memory collectives: AllGather, ReduceScatter, All-to-All.
#
# Usage:
#   bash bench_symm.sh              # 8 GPUs, all ops
#   bash bench_symm.sh 4            # 4 GPUs, all ops
#   bash bench_symm.sh 8 ag rs      # 8 GPUs, AG + RS only
#
# Results (H100 8-GPU, bf16, 500 iters, graph-loop=100):
#
# | msg_size   |   AG eager (us) |   AG symm graph (us) |   RS eager (us) |   RS symm graph (us) |   A2A eager (us) |   A2A symm graph (us) |
# |------------|-----------------|----------------------|-----------------|----------------------|------------------|-----------------------|
# | 2.0 KiB    |           14.57 |                 2.68 |           16.06 |                 2.82 |            18.34 |                  5.45 |
# | 4.0 KiB    |           14.58 |                 2.84 |           17.48 |                 2.86 |            17.87 |                  5.51 |
# | 8.0 KiB    |           14.42 |                 3.11 |           15.82 |                 3.00 |            17.84 |                  5.57 |
# | 16.0 KiB   |           14.03 |                 4.70 |           18.66 |                 3.01 |            17.70 |                  5.57 |
# | 32.0 KiB   |           14.37 |                 5.07 |           18.74 |                 3.25 |            17.90 |                  6.25 |
# | 64.0 KiB   |           15.94 |                 5.35 |           15.89 |                 3.41 |            24.99 |                  6.82 |
# | 128.0 KiB  |           18.14 |                 6.90 |           17.12 |                 4.24 |            21.18 |                  7.00 |
#
# Symm-mem CUDA graph speedup over torch eager:
#   AllGather:     2.6-5.4x
#   ReduceScatter: 4.0-6.2x
#   All-to-All:    3.0-3.6x

set -euo pipefail

NGPU=${1:-8}
shift 1 2>/dev/null || true
OPS="${@:-ag rs a2a}"

export NCCL_GRAPH_MIXING_SUPPORT=0
export NCCL_CUMEM_ENABLE=1
export NCCL_NVLS_ENABLE=2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

torchrun --nproc_per_node "$NGPU" \
    "$SCRIPT_DIR/benchmark_symm_mem.py" \
    --ops $OPS \
    --warmup 10 \
    --iters 1000 \
    --graph-loop 100
