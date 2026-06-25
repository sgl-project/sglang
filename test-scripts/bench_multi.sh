#!/usr/bin/env bash
# Run multiple bench_serving combos against ONE shared server.
# Same (mode, input_len), varying output_len — server is launched once
# and reused, since output_len is a client-side bench param only.
#
# Env vars:
#   MODE          dense | sparse              (required)
#   INPUT_LEN     input tokens               (required)
#   OUTPUT_LENS   space-separated list       (required, e.g. "1024 2048")
#   MAX_CONC      max concurrency            (auto by INPUT_LEN)
#   NUM_PROMPTS   num prompts                (auto: MAX_CONC * 4)
#   PORT          server port                (default 30000)
#   RESULTS_DIR   where to drop logs/json    (default /tmp/glm_bench_matrix)
#   EXTRA_LAUNCH_ARGS  extra args for launch_server
#
# Outputs: same CSV/jsonl/log layout as bench_one.sh, one row per output_len.

set -uo pipefail
source "${VENV:-/root/paddlejob/inference-public/denghaodong/code/sglang/.venv/bin/activate}"

MODEL_PATH="/root/paddlejob/inference-public/denghaodong/code/model/GLM_v2"
DENSE_DIR="/root/paddlejob/inference-public/denghaodong/code/sglang-main"
SPARSE_DIR="/root/paddlejob/inference-public/denghaodong/code/sglang"

MODE="${MODE:?MODE required (dense|sparse)}"
INPUT_LEN="${INPUT_LEN:?INPUT_LEN required}"
OUTPUT_LENS="${OUTPUT_LENS:?OUTPUT_LENS required (e.g. \"1024 2048\")}"
PORT="${PORT:-30000}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/glm_bench_matrix}"
EXTRA_LAUNCH_ARGS="${EXTRA_LAUNCH_ARGS:-}"
mkdir -p "$RESULTS_DIR"

scale_concurrency() {
  case "$INPUT_LEN" in
    16384)   echo 64 ;;
    32768)   echo 32 ;;
    65536)   echo 16 ;;
    102400)  echo 16 ;;
    *)       echo 16 ;;
  esac
}
MAX_CONC="${MAX_CONC:-$(scale_concurrency)}"
NUM_PROMPTS="${NUM_PROMPTS:-$(( MAX_CONC * 4 ))}"
CSV="$RESULTS_DIR/results.csv"
SERVER_TAG="${MODE}_in${INPUT_LEN}"
SERVER_LOG="$RESULTS_DIR/server_${SERVER_TAG}.log"

echo "[$(date '+%F %T')] === bench_multi: mode=$MODE in=$INPUT_LEN outs=[$OUTPUT_LENS] conc=$MAX_CONC n=$NUM_PROMPTS ==="

# ---- pick worktree by mode ----
if [[ "$MODE" == "dense" ]]; then
  SGLANG_DIR="$DENSE_DIR"
else
  SGLANG_DIR="$SPARSE_DIR"
fi
cd "$SGLANG_DIR"
export PYTHONPATH="$SGLANG_DIR/python"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export FLASHINFER_USE_CUDA_NORM=1
echo "[bench_multi] using worktree: $SGLANG_DIR ($(git rev-parse --abbrev-ref HEAD))"

cleanup() {
  local rc=$?
  set +e
  echo "[bench_multi] cleanup: killing server, restoring config"
  pkill -9 -f "sglang.launch_server" 2>/dev/null || true
  for i in $(seq 1 60); do
    if ! pgrep -f "sglang.launch_server" >/dev/null; then break; fi
    sleep 5
  done
  if [[ -f "${MODEL_PATH}/config.json.bak_${SERVER_TAG}" ]]; then
    mv "${MODEL_PATH}/config.json.bak_${SERVER_TAG}" "${MODEL_PATH}/config.json"
  fi
  exit $rc
}
trap cleanup EXIT INT TERM

# ---- sparse only: write sparse config ----
if [[ "$MODE" != "dense" ]]; then
  cp "${MODEL_PATH}/config.json" "${MODEL_PATH}/config.json.bak_${SERVER_TAG}"
  ENABLED=true TOPK=2048 FORCE_LEFT=64 FORCE_RIGHT=128 FREQ=4 \
  PATTERN="FFFSSSSSSSSSSSSSSSFSSFSSSSSFSFFSSFSSSSFSSSFFSS" \
  python3 - "$MODEL_PATH/config.json" <<'PYEOF'
import json, os, sys
p = sys.argv[1]
c = json.load(open(p))
def as_bool(s): return str(s).strip().lower() in ("1","true","yes","on")
pat = os.environ["PATTERN"]
c["glm_sparse_indexer_enabled"] = as_bool(os.environ["ENABLED"])
c["glm_sparse_indexer_topk"]    = int(os.environ["TOPK"])
c["glm_sparse_force_left"]      = int(os.environ["FORCE_LEFT"])
c["glm_sparse_force_right"]     = int(os.environ["FORCE_RIGHT"])
c["glm_sparse_index_topk_freq"] = int(os.environ["FREQ"])
c["glm_sparse_index_topk_pattern"] = None if pat == "None" else pat
json.dump(c, open(p,"w"), indent=2)
print("[bench_multi] config.json sparse fields written")
PYEOF
fi

# ---- start server ONCE ----
echo "[bench_multi] launching sglang server (max-running-requests=$MAX_CONC)"
nohup python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 8 \
  --context-length 131072 \
  --max-running-requests "$MAX_CONC" \
  --quantization fp8 \
  --reasoning-parser glm45 \
  $EXTRA_LAUNCH_ARGS \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[bench_multi] server pid=$SERVER_PID  log=$SERVER_LOG"

# ---- wait for readiness ----
READY_TIMEOUT="${READY_TIMEOUT:-1200}"
echo "[bench_multi] waiting up to ${READY_TIMEOUT}s for server readiness on port $PORT"
ready=0
for ((i=0; i<READY_TIMEOUT; i+=5)); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[bench_multi] ERROR: server process died during startup (see $SERVER_LOG)"
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 \
     || curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    echo "[bench_multi] server ready after ~${i}s"
    break
  fi
  sleep 5
done
if [[ "$ready" != "1" ]]; then
  echo "[bench_multi] ERROR: server not ready within ${READY_TIMEOUT}s (see $SERVER_LOG)"
  exit 1
fi

COMMIT="$(git rev-parse --short HEAD)"

# ---- loop output_lens against the SAME server ----
for OUTPUT_LEN in $OUTPUT_LENS; do
  TAG="${MODE}_in${INPUT_LEN}_out${OUTPUT_LEN}"
  BENCH_LOG="$RESULTS_DIR/bench_${TAG}.log"
  RESULT_JSONL="$RESULTS_DIR/bench_${TAG}.jsonl"
  echo "=========================================================="
  echo "[bench_multi] bench out=$OUTPUT_LEN @ $(date '+%F %T')"
  echo "=========================================================="

  python3 -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 \
    --port "$PORT" \
    --ready-check-timeout-sec 120 \
    --dataset-name random-ids \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$MAX_CONC" \
    --seed 1 \
    --output-file "$RESULT_JSONL" \
    2>&1 | tee "$BENCH_LOG"

  python3 - "$RESULT_JSONL" "$CSV" "$MODE" "$INPUT_LEN" "$OUTPUT_LEN" \
           "$MAX_CONC" "$NUM_PROMPTS" "$COMMIT" <<'PYEOF'
import json, sys, os, time
src, csv, mode, il, ol, mc, np, commit = sys.argv[1:9]
row = None
with open(src) as f:
    for line in f:
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "output_throughput" in d:
            row = d
if row is None:
    print("[csv] no result line found in", src)
    sys.exit(0)
new = not os.path.exists(csv)
cols = [
    ("ts", time.strftime("%F %T")),
    ("mode", mode),
    ("input_len", il),
    ("output_len", ol),
    ("max_conc", mc),
    ("num_prompts", np),
    ("commit", commit),
    ("input_throughput",  f"{row.get('input_throughput',0):.1f}"),
    ("output_throughput", f"{row.get('output_throughput',0):.1f}"),
    ("total_throughput",  f"{row.get('total_throughput',0):.1f}"),
    ("mean_ttft_ms",      f"{row.get('mean_ttft_ms',0):.1f}"),
    ("mean_tpot_ms",      f"{row.get('mean_tpot_ms',0):.2f}"),
    ("mean_e2e_ms",       f"{row.get('mean_e2e_latency_ms',0):.1f}"),
    ("p99_ttft_ms",       f"{row.get('p99_ttft_ms',0):.1f}"),
    ("completed",         row.get('completed', 0)),
]
with open(csv, "a") as f:
    if new:
        f.write(",".join(k for k,_ in cols) + "\n")
    f.write(",".join(str(v) for _,v in cols) + "\n")
print("[csv] appended row:", ",".join(str(v) for _,v in cols))
PYEOF

  echo "[bench_multi] DONE out=$OUTPUT_LEN"
done

echo "[bench_multi] ALL DONE for mode=$MODE in=$INPUT_LEN"
