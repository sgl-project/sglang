#!/usr/bin/env bash
# Run the V4.1 top-k validation suites on the devbox, one suite per GPU.
#   bash run_tests.sh <tree: sglang|sglang-base> <log dir>
set -uo pipefail

TREE=${1:?tree}
LOGS=$(realpath -m "${2:?log dir}")
ROOT=/hai-workspace/dsv41-fi-topk
cd "$ROOT/$TREE"
mkdir -p "$LOGS"
export PYTHONPATH="$ROOT/$TREE/python"
export SGLANG_CACHE_DIR="$ROOT/cache/$TREE"
export FLASHINFER_WORKSPACE_BASE="$SGLANG_CACHE_DIR/flashinfer"
unset FLASHINFER_TOPK_ALGO SGLANG_DSA_FUSE_TOPK SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC \
  SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK SGLANG_TOPK_V2_MODE SGLANG_OPT_USE_TOPK_V2

declare -A SUITES=(
  [unit]="test/registered/unit/layers/test_dsv41_indexer_arch_gates.py test/registered/unit/layers/test_dsv4_nonpaged_indexer.py test/registered/unit/layers/test_dsv41_candidate_blocks.py test/registered/unit/layers/test_dsv41_indexer_tp_shard.py"
  [selector]="test/registered/kernels/ops/attention/test_dsv41_topk_selector.py"
  [dense_prefill]="test/registered/kernels/ops/attention/test_dense_prefill_indexer.py"
  [sparse_prefill]="test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py"
  [fp4_indexer]="test/registered/kernels/ops/attention/test_fp4_indexer.py test/registered/kernels/ops/attention/test_dsv41_candidate_table_boundary.py test/registered/kernels/ops/attention/test_topk_v2.py"
  [dsv4_attention]="test/registered/attention/unittests/dsv4/test_deepseek_v4.py"
  [dsa_indexer]="test/registered/kernels/ops/attention/test_dsa_indexer.py"
)

read -r -a GPU_POOL <<< "${TEST_GPUS:-0 1 2 3 4 5 6 7}"
slot=0
for name in unit selector dense_prefill sparse_prefill fp4_indexer dsv4_attention dsa_indexer; do
  files=()
  for f in ${SUITES[$name]}; do
    [[ -f "$f" ]] && files+=("$f")
  done
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "skip $name: no files in $TREE" > "$LOGS/$name.log"
    continue
  fi
  gpu=${GPU_POOL[$((slot % ${#GPU_POOL[@]}))]}
  (
    echo "tree=$TREE suite=$name gpu=$gpu files=${files[*]}"
    CUDA_VISIBLE_DEVICES=$gpu python3 -m pytest -q -ra -p no:cacheprovider "${files[@]}"
    echo "exit=$?"
  ) > "$LOGS/$name.log" 2>&1 &
  slot=$((slot + 1))
  # One suite per GPU at a time: wait for the batch once every GPU is busy.
  if (( slot % ${#GPU_POOL[@]} == 0 )); then
    wait
  fi
done
wait
for name in unit selector dense_prefill sparse_prefill fp4_indexer dsv4_attention dsa_indexer; do
  echo "== $name: $(grep -E '^(exit=|skip )' "$LOGS/$name.log" | tail -1) $(grep -E '(passed|failed|error)' "$LOGS/$name.log" | tail -1)"
done
