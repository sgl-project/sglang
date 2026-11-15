#!/usr/bin/env bash
# Final operator benchmark: the balanced four-run invocation, then #47960's re-run rule
# (60 rounds, ABBA, for every cell whose sgl-kernel branch-vs-base delta exceeds 2%).
set -uo pipefail

ROOT=/hai-workspace/dsv41-fi-topk
cd "$ROOT"
OUT=$ROOT/bench/final-c
rm -rf "$OUT"
BENCH_GPU=0 bash tools/run_bench_balanced.sh "$OUT" --rounds 20 > bench/final-c.out 2>&1
python3 tools/analyze_bench.py "$OUT" > bench/final-c-analysis-pre-rerun.md 2>&1
python3 - "$ROOT" <<'PY' > bench/final-c-reruns.txt
import json, sys
last = open(f"{sys.argv[1]}/bench/final-c-analysis-pre-rerun.md").read().strip().splitlines()[-1]
for suite, label in json.loads(last.split(":", 1)[1]):
    print(f"{suite}\t{label}")
PY
index=0
while IFS=$'\t' read -r suite label; do
  [[ -z $suite ]] && continue
  index=$((index + 1))
  for step in "1 sgl-kernel-base" "1 sgl-kernel" "2 sgl-kernel" "2 sgl-kernel-base"; do
    read -r run variant <<< "$step"
    tree=sglang
    [[ $variant == sgl-kernel-base ]] && tree=sglang-base
    echo "== rerun$run $variant $suite / $label $(date -u +%FT%TZ)" >> bench/final-c.out
    env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT/$tree/python" SGLANG_CACHE_DIR="$ROOT/cache/$tree" \
      FLASHINFER_WORKSPACE_BASE="$ROOT/cache/$tree/flashinfer" \
      python3 tools/bench_topk.py --variant "$variant" --tree "$ROOT/$tree" --suite "$suite" \
      --only "$label" --rounds 60 --out "$OUT/rerun$run-$variant-$index.jsonl" >> bench/final-c.out 2>&1
  done
done < bench/final-c-reruns.txt
python3 tools/analyze_bench.py "$OUT" > bench/final-c-analysis.md 2>&1
echo done > bench/final-c.done
