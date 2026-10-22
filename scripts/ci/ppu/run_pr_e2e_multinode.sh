#!/usr/bin/env bash
# PR-only colocated multi-node setup, shared by Answer and performance.
set -euo pipefail
: "${PPU_E2E_PLAN:?selection plan is required}"
: "${PPU_E2E_KIND:?evaluation kind is required}"
case "$PPU_E2E_KIND" in
  ANSWER|PERF) ;;
  *) echo 'Only colocated Answer/performance use this adapter.' >&2; exit 2 ;;
esac
cd /workspace/source
results_variable="SGLANG_PPU_${PPU_E2E_KIND}_RESULTS_DIR"
results="${!results_variable}"
address=$(bash scripts/ci/ppu/answer_rendezvous.sh "$results/ranks")
export "SGLANG_PPU_${PPU_E2E_KIND}_DIST_INIT_ADDR=$address"
NCCL_IB_GID_INDEX=$(bash scripts/ci/ppu/answer_gid_index.sh)
GLOO_SOCKET_IFNAME=$(bash scripts/ci/ppu/answer_gloo_iface.sh "$address")
export NCCL_IB_GID_INDEX GLOO_SOCKET_IFNAME
export PCCL_MNNVL_ENABLE=0 SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE=0
export SGLANG_PPU_ANSWER_INCLUDE_RAW_OUTPUTS=0
bash scripts/ci/ppu/ppu_install_dependency.sh
JIT_MTIME=$(git log -1 --format=%ct)
if [[ -d python/sglang/kernels/jit ]]; then
  find python/sglang/kernels/jit -type f -exec touch -d @"$JIT_MTIME" {} +
fi
. scripts/ci/ppu/use_local_jit_cache.sh
# PP-sharded workers must not each warm the entire checkpoint on the NAS.
exec python3 scripts/ci/ppu/pr_e2e_worker.py suite
