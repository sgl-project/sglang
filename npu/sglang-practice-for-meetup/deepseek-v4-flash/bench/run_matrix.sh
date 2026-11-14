#!/usr/bin/env bash
# run_matrix.sh — the 5-cell benchmark matrix. ~20 minutes on a healthy 4-NPU slice.
#
# Why these cells: they separate the three regimes that behave differently on any accelerator.
#   C1 decode-dominated  (in 128  / out 512, concurrency 1)   -> single-stream TPOT, the bandwidth floor
#   C2 balanced          (in 2048 / out 256, concurrency 8)   -> the realistic chat serving point
#   C3 prefill-heavy     (in 16384/ out 128, concurrency 4)   -> TTFT under long prompts
#   C4 long-context      (in 32768/ out 256, concurrency 1)   -> KV-cache growth vs decode speed
#   C5 open-loop arrival (in 2048 / out 256, rate 2 req/s)    -> queueing behaviour, tail latency
#
# Notes on semantics (verified against python/sglang/benchmark/serving.py):
#   * EOS is IGNORED by default; there is no --ignore-eos flag, the opt-in is --disable-ignore-eos.
#   * --random-range-ratio 1 keeps prompt lengths fixed at input_len.
#   * output is APPENDED JSONL (the runner truncates the file first).
#   * --tokenize-prompt tokenizes client-side, so the recorded input length is real tokens.
#
# Each cell writes:
#   results/<cell>.jsonl       per-request records (SGLang bench_serving --output-file)
#   results/<cell>.meta.json   wall time + exact command + configuration (used by summarize.py)
#
# Usage: bash bench/run_matrix.sh [--cells c1,c2] [--num-prompts-scale 1.0]
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJ=$(cd "$HERE/.." && pwd)
# shellcheck source=../env/00_common.sh
source "$PROJ/env/00_common.sh"

BASE="http://127.0.0.1:$PORT"
MODEL_NAME=${MODEL_NAME:-deepseek-v4-flash}
SEED=${SEED:-1}
SCALE=${SCALE:-1.0}
CELLS_FILTER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cells) CELLS_FILTER=$2; shift 2 ;;
    --num-prompts-scale) SCALE=$2; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

command -v python3 >/dev/null || { echo "python3 required"; exit 1; }
curl -sf "$BASE/health" >/dev/null || { echo "server not healthy at $BASE — run deploy/03_launch.sh first"; exit 1; }

# cell name | in_len | out_len | num_prompts | concurrency | request_rate | note
CELLS=(
  "c1-decode|128|512|32|1|inf|single-stream decode: TPOT floor"
  "c2-balanced|2048|256|64|8|inf|chat serving: TTFT/TPOT together"
  "c3-prefill|16384|128|16|4|inf|long prompt: TTFT under prefill load"
  "c4-longctx|32768|256|4|1|inf|long context: KV growth vs decode"
  "c5-openloop|2048|256|64|64|2|open loop 2 req/s: queueing & tail"
)

want() { [ -z "$CELLS_FILTER" ] && return 0; case ",$CELLS_FILTER," in *",$1,"*) return 0;; *) return 1;; esac; }

for spec in "${CELLS[@]}"; do
  IFS='|' read -r name in out base_n conc rate note <<< "$spec"
  want "$name" || continue
  n=$(python3 -c "print(max(4, int(round($base_n * $SCALE))))")

  # cold-ish start for every cell: flush prefix cache so TTFT is a real prefill number
  curl -s -X POST "$BASE/flush_cache" >/dev/null 2>&1 || true

  out_file="$RESULTS_DIR/${name}.jsonl"
  meta_file="$RESULTS_DIR/${name}.meta.json"
  : > "$out_file"

  echo
  echo "=== [$name] $note"
  echo "    in=$in out=$out prompts=$n conc=$conc rate=$rate"

  cmd=(python3 -m sglang.benchmark.serving
    --backend sglang
    --host 127.0.0.1 --port "$PORT"
    --model "$MODEL_NAME"
    --dataset-name random
    --random-input-len "$in"
    --random-output-len "$out"
    --random-range-ratio 1
    --num-prompts "$n"
    --request-rate "$rate"
    --warmup-requests 2
    --seed "$SEED"
    --tokenize-prompt
    --disable-tqdm
    --flush-cache
    --output-details
    --cache-report
    --output-file "$out_file"
    --tag "$name")

  # --max-concurrency only for the closed-loop cells (open-loop is governed by request-rate)
  if [ "$rate" = "inf" ]; then cmd+=(--max-concurrency "$conc"); fi

  printf '# %s\n' "${cmd[*]}" > "$meta_file"
  start=$(date +%s)
  set +e
  "${cmd[@]}" 2>&1 | tee "$LOG_DIR/bench-${name}.log"
  rc=${PIPESTATUS[0]}
  set -e
  end=$(date +%s)

  python3 - "$meta_file" "$name" "$in" "$out" "$n" "$conc" "$rate" "$start" "$end" "$rc" "$CHIP" "$NPUS" "$TP_SIZE" "$DP_SIZE" "$MODEL_DIR" "$IMAGE" "${cmd[*]}" <<'PY'
import json, sys
(mf, name, i, o, n, c, rate, start, end, rc, chip, npus, tp, dp, model_dir, image, cmd) = sys.argv[1:]
json.dump({
    "cell": name, "input_len": int(i), "output_len": int(o), "num_prompts": int(n),
    "max_concurrency": None if rate == "inf" else int(c),
    "request_rate": rate,
    "start_epoch": int(start), "end_epoch": int(end), "duration_s": int(end) - int(start),
    "return_code": int(rc),
    "hardware": {"chip": chip, "npus": int(npus), "tp": int(tp), "dp": int(dp)},
    "model_dir": model_dir, "image": image, "command": cmd,
}, open(mf, "w"), indent=2)
PY

  echo "    rc=$rc duration=$((end - start))s -> $out_file"
  [ "$rc" -ne 0 ] && echo "    WARN: benchmark exited non-zero; see $LOG_DIR/bench-${name}.log"
  sleep 5
done

echo
echo "matrix done. next: python3 bench/summarize.py"
