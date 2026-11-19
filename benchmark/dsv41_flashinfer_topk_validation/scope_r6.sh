#!/usr/bin/env bash
# Fused-only revision on the devbox: the seven suites (GPUs 0-6), then on GPU 7 the
# negative controls, the mode-2 parity controls and the graph legs; then, for every branch
# variant on the previous head's tree (sglang-prev) and on this one (sglang), the kernel
# launches of every benchmark shape (trace_sgl_kernel.py) and of every TopKSelector entry
# point (trace_entry_points.py), compared per variant.
#   RUN=<name> nohup bash tools/scope_r6.sh > logs/<name>.out 2>&1 &   (RUN defaults to r6)
set -uo pipefail
ROOT=/hai-workspace/dsv41-fi-topk
RUN=${RUN:-r6}
cd "$ROOT"
LOGS=$ROOT/logs/$RUN
TRACE=$ROOT/trace-$RUN
mkdir -p "$LOGS" "$TRACE"
unset FLASHINFER_TOPK_ALGO SGLANG_DSA_FUSE_TOPK SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC \
  SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK SGLANG_TOPK_V2_MODE SGLANG_OPT_USE_TOPK_V2
echo "$(date -u +%FT%TZ) start"

(TEST_GPUS="0 1 2 3 4 5 6" bash tools/run_tests.sh sglang "$LOGS/tests" > "$LOGS/tests.out" 2>&1) &
(
  cd sglang
  export PYTHONPATH=$ROOT/sglang/python SGLANG_CACHE_DIR=$ROOT/cache/sglang \
    FLASHINFER_WORKSPACE_BASE=$ROOT/cache/sglang/flashinfer CUDA_VISIBLE_DEVICES=7
  python3 ../tools/negative_controls.py > "$LOGS/negative-controls.log" 2>&1
  echo "exit=$?" >> "$LOGS/negative-controls.log"
  python3 ../tools/mode2_controls.py "$ROOT/sglang" > "$LOGS/mode2-controls.log" 2>&1
  echo "exit=$?" >> "$LOGS/mode2-controls.log"
  python3 ../tools/verify_graph_backends.py > "$LOGS/graph-backends.log" 2>&1
  echo "exit=$?" >> "$LOGS/graph-backends.log"
) &
wait
echo "$(date -u +%FT%TZ) tests and controls done"

record() {  # gpu tree variant kind (trace | entry)
  local gpu=$1 tree=$2 variant=$3 kind=$4
  local env_args=(CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT/$tree/python"
    SGLANG_CACHE_DIR="$ROOT/cache/$tree" FLASHINFER_WORKSPACE_BASE="$ROOT/cache/$tree/flashinfer")
  case $variant in
    sgl-kernel-mode2) env_args+=(SGLANG_TOPK_V2_MODE=2) ;;
    flashinfer-small) env_args+=(SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK=small) ;;
    flashinfer-small-det)
      env_args+=(SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK=small SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC=1) ;;
  esac
  if [[ $kind == trace ]]; then
    env "${env_args[@]}" python3 tools/trace_sgl_kernel.py record --variant "$variant" \
      --tree "$ROOT/$tree" --out "$TRACE/$tree-$variant.jsonl" > "$TRACE/$tree-$variant.log" 2>&1
  else
    env "${env_args[@]}" python3 tools/trace_entry_points.py record --variant "$variant" \
      --out "$TRACE/entry-$tree-$variant.jsonl" > "$TRACE/entry-$tree-$variant.log" 2>&1
  fi
  local status=$?
  echo "$(date -u +%FT%TZ) $kind $tree $variant exit=$status"
}

VARIANTS="sgl-kernel sgl-kernel-mode2 flashinfer flashinfer-small flashinfer-small-det"
gpu=0
for kind in trace entry; do
  for variant in $VARIANTS; do
    for tree in sglang-prev sglang; do
      record "$gpu" "$tree" "$variant" "$kind" &
      gpu=$((gpu + 1))
      if (( gpu == 8 )); then
        wait
        gpu=0
      fi
    done
  done
done
wait
for variant in $VARIANTS; do
  python3 tools/trace_sgl_kernel.py compare "$TRACE/sglang-prev-$variant.jsonl" \
    "$TRACE/sglang-$variant.jsonl" --names previous this > "$TRACE/compare-$variant.md" 2>&1
  python3 tools/trace_entry_points.py compare "$TRACE/entry-sglang-prev-$variant.jsonl" \
    "$TRACE/entry-sglang-$variant.jsonl" --names previous this > "$TRACE/entry-compare-$variant.md" 2>&1
done
echo "$(date -u +%FT%TZ) done"
touch "$LOGS/done"
