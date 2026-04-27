#!/usr/bin/env bash
# Run the same IFBench calibration twice — once with Qwen3 thinking mode ON,
# once with it OFF — so you can A/B-compare KV envelope, output lengths,
# and response quality.
#
# Each run's artifacts are renamed with _think / _nothink suffixes so the two
# sets don't clobber each other.  At the end, prompt/ifbench.jsonl is
# restored to the thinking-ON recipe so it's ready for the main sweep.
#
# Default knobs (override via env vars):
#   NUM_PROMPTS     = 16           # keep small; this is a calib smoke, not the sweep
#   CALIB_GPU       = 0
#   CALIB_MC        = 32           # max concurrent requests
#   CALIB_PORT      = 31400
#   CALIB_NCCL_PORT = 41400
#   THINK_MAX_TOK   = 8192         # thinking-mode output cap
#   NOTHINK_MAX_TOK = 2048         # non-thinking output cap (no reasoning → no need for 8192)
#
# Usage:
#   bash pipeline/run_calib_think_compare.sh
#   NUM_PROMPTS=32 CALIB_GPU=1 bash pipeline/run_calib_think_compare.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
EXPERIMENTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

NUM_PROMPTS="${NUM_PROMPTS:-16}"
CALIB_GPU="${CALIB_GPU:-0}"
CALIB_MC="${CALIB_MC:-32}"
CALIB_PORT="${CALIB_PORT:-31400}"
CALIB_NCCL_PORT="${CALIB_NCCL_PORT:-41400}"
THINK_MAX_TOK="${THINK_MAX_TOK:-8192}"
NOTHINK_MAX_TOK="${NOTHINK_MAX_TOK:-2048}"

PY="${PY:-$(command -v python3 || command -v python)}"
TASK=ifbench
KV_DIR="$EXPERIMENTS_DIR/data/kv_calib"
PROMPT_DIR="$SCRIPT_DIR/prompt"
mkdir -p "$KV_DIR"

# Use the raw calib jsonl name that run_calib.sh produces: calib_<task>_n<N>.jsonl
CALIB_JSONL="$KV_DIR/calib_${TASK}_n${NUM_PROMPTS}.jsonl"
CALIB_JSON="$KV_DIR/${TASK}.json"
SERVER_LOG="$KV_DIR/server_${TASK}_bf16.log"

run_one() {
    # $1 = tag (think | nothink)
    local tag="$1"
    local run_log="$KV_DIR/run_${TASK}_${tag}.log"

    echo
    echo "============================================================"
    echo "  PHASE: $tag   (NUM_PROMPTS=$NUM_PROMPTS GPU=$CALIB_GPU MC=$CALIB_MC)"
    echo "  logging stdout → $run_log"
    echo "============================================================"

    # Snapshot the prompts file so `show_calib.py --rendered` can later
    # reconstruct the exact chat-template string this run fed to the model.
    cp -v "$PROMPT_DIR/${TASK}.jsonl"      "$PROMPT_DIR/${TASK}_${tag}.jsonl"
    cp -v "$PROMPT_DIR/${TASK}.meta.jsonl" "$PROMPT_DIR/${TASK}_${tag}.meta.jsonl"

    CALIB_GPU="$CALIB_GPU" CALIB_MC="$CALIB_MC" \
    CALIB_PORT="$CALIB_PORT" CALIB_NCCL_PORT="$CALIB_NCCL_PORT" \
    NUM_PROMPTS="$NUM_PROMPTS" \
    bash "$SCRIPT_DIR/kv_calib/run_calib.sh" "$TASK" 2>&1 | tee "$run_log"

    # Rename the three fixed-path artifacts.
    mv -v "$CALIB_JSONL" "$KV_DIR/calib_${TASK}_n${NUM_PROMPTS}_${tag}.jsonl"
    mv -v "$CALIB_JSON"  "$KV_DIR/${TASK}_${tag}.json"
    mv -v "$SERVER_LOG"  "$KV_DIR/server_${TASK}_bf16_${tag}.log"
}

# --- Phase 1: thinking ON --------------------------------------------------
"$PY" "$PROMPT_DIR/prepare_prompts_ifbench.py" --max_tokens "$THINK_MAX_TOK"
run_one think

# --- Phase 2: thinking OFF -------------------------------------------------
"$PY" "$PROMPT_DIR/prepare_prompts_ifbench.py" --no-enable_thinking --max_tokens "$NOTHINK_MAX_TOK"
run_one nothink

# --- Restore prompt/ifbench.jsonl to the thinking-ON recipe ----------------
"$PY" "$PROMPT_DIR/prepare_prompts_ifbench.py" --max_tokens "$THINK_MAX_TOK" > /dev/null
echo "(restored prompt/ifbench.jsonl to thinking=True, max_tokens=$THINK_MAX_TOK)"

cat <<EOF

============================================================
  DONE.  Artifacts:

  think mode:
    trace       $KV_DIR/calib_${TASK}_n${NUM_PROMPTS}_think.jsonl
    summary     $KV_DIR/${TASK}_think.json
    server log  $KV_DIR/server_${TASK}_bf16_think.log
    stdout      $KV_DIR/run_${TASK}_think.log

  no-think mode:
    trace       $KV_DIR/calib_${TASK}_n${NUM_PROMPTS}_nothink.jsonl
    summary     $KV_DIR/${TASK}_nothink.json
    server log  $KV_DIR/server_${TASK}_bf16_nothink.log
    stdout      $KV_DIR/run_${TASK}_nothink.log

  Next (paths are absolute; show_calib lives at experiments/monitor/):
    python $EXPERIMENTS_DIR/monitor/show_calib.py $KV_DIR/calib_${TASK}_n${NUM_PROMPTS}_think.jsonl   --row 1
    python $EXPERIMENTS_DIR/monitor/show_calib.py $KV_DIR/calib_${TASK}_n${NUM_PROMPTS}_nothink.jsonl --row 1

    diff <(jq . $KV_DIR/${TASK}_think.json) <(jq . $KV_DIR/${TASK}_nothink.json)
============================================================
EOF
