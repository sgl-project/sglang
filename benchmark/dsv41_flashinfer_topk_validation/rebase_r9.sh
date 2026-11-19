#!/usr/bin/env bash
# Validation after rebasing onto a new base: `RUN=$RUN scope_r6.sh` (the seven suites
# on the PR tree, negative and mode-2 controls, graph legs, previous-vs-this kernel
# traces), then the seven suites on the new base tree (sglang-base, GPUs 0-6) next to
# the sgl-kernel base-vs-branch kernel trace on GPU 7 (base, branch, base again).
#   RUN=<name> nohup bash tools/rebase_r9.sh > logs/<name>-all.out 2>&1 &
set -uo pipefail
ROOT=/hai-workspace/dsv41-fi-topk
RUN=${RUN:-r9}
cd "$ROOT"
unset FLASHINFER_TOPK_ALGO SGLANG_DSA_FUSE_TOPK SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC \
  SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK SGLANG_TOPK_V2_MODE SGLANG_OPT_USE_TOPK_V2
echo "$(date -u +%FT%TZ) start"
RUN=$RUN bash tools/scope_r6.sh > "logs/$RUN.out" 2>&1
echo "$(date -u +%FT%TZ) scope done"

BASE_TRACE=$ROOT/trace-$RUN-base
mkdir -p "$BASE_TRACE"
(TEST_GPUS="0 1 2 3 4 5 6" bash tools/run_tests.sh sglang-base "$ROOT/logs/$RUN/tests-base" \
  > "$ROOT/logs/$RUN/tests-base.out" 2>&1) &
(
  for step in "sgl-kernel-base sglang-base sgl-kernel-base" "sgl-kernel sglang sgl-kernel" \
    "sgl-kernel-base sglang-base sgl-kernel-base-again"; do
    read -r variant tree out <<< "$step"
    env CUDA_VISIBLE_DEVICES=7 PYTHONPATH="$ROOT/$tree/python" SGLANG_CACHE_DIR="$ROOT/cache/$tree" \
      FLASHINFER_WORKSPACE_BASE="$ROOT/cache/$tree/flashinfer" \
      python3 tools/trace_sgl_kernel.py record --variant "$variant" --tree "$ROOT/$tree" \
      --out "$BASE_TRACE/$out.jsonl" > "$BASE_TRACE/$out.log" 2>&1
    status=$?
    echo "$(date -u +%FT%TZ) base trace $out exit=$status"
  done
  python3 tools/trace_sgl_kernel.py compare "$BASE_TRACE/sgl-kernel-base.jsonl" \
    "$BASE_TRACE/sgl-kernel.jsonl" --repeat "$BASE_TRACE/sgl-kernel-base-again.jsonl" \
    > "$BASE_TRACE/compare.md" 2>&1
) &
wait
echo "$(date -u +%FT%TZ) done"
touch "$ROOT/logs/$RUN/all-done"
