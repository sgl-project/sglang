#!/usr/bin/env bash
# Capture a BS=1 decode trace (with full call stacks) from an ALREADY-RUNNING
# sglang server. Does NOT start or stop the server -- you launch the server
# yourself (e.g. the MiMo DFLASH FP8 command), then run this against it.
#
# BS=1: the profiled bench is sent with --max-concurrency 1 so only one request
# is ever in flight, i.e. the decode stage runs at batch size 1.
#
# profile_by_stage skips idle scheduler batches; num_steps caps the decode stage
# so it auto-stops mid-request (no trailing idle). bench_serving sends its own
# warmup request after start_profile, which (under profile_by_stage) opens the
# decode profiler on the *first* decode; the real request's decode then lands in
# that running profiler. Inspect the per-rank "*-DECODE.trace.json.gz"; the
# "*-EXTEND" file is just the warmup prefill and can be ignored.
#
# The accept length (if pinned via SGLANG_SIMULATE_ACC_LEN) is controlled on the
# SERVER side at launch -- this client script does not set it.
#
# Usage:
#   bash scripts/run_profile_bs1.sh
#   SERVER_PORT=29999 STACK_PROFILE_STEPS=20 bash scripts/run_profile_bs1.sh

set -euo pipefail

# Repo root = parent of this script's dir, so `uv run` uses this repo's venv
# regardless of the caller's cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="${SGLANG_DIR:-$(dirname "$SCRIPT_DIR")}"

# ----------------------------------------------------------------------------
# Config -- point these at your running server
# ----------------------------------------------------------------------------
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"           # server bound --host 0.0.0.0 -> IPv4 loopback
SERVER_PORT="${SERVER_PORT:-29999}"
BASE_URL="http://${SERVER_HOST}:${SERVER_PORT}"

# Tokenizer MUST match the running server's model so prompt token counts line up.
# Default: the local MiMo-V2.5-Pro-FP4-DFlash snapshot.
TOKENIZER="${TOKENIZER:-$HOME/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro-FP4-DFlash/snapshots/b754e6c86008bdb5cc901308dda5a38173ec7276}"

PROFILE_ROOT="${PROFILE_ROOT:-/data/users/${USER}/codesign_profile}"
PROFILE_DIR="${PROFILE_DIR:-${PROFILE_ROOT}/mimo_v2_fp4_dflash_bs1}"

INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
# Cap on profiled decode steps. Keep strictly below OUTPUT_LEN so the cap fires
# before the single request ends (counts the ~1 bench warmup decode step too).
STACK_PROFILE_STEPS="${STACK_PROFILE_STEPS:-20}"
# Prompts for the profiled run. With --max-concurrency 1 these are sequential and
# every decode is BS=1; 2 is enough (bench's own warmup opens the profiler, the
# next request is captured). num_steps stops the decode early regardless.
NUM_PROMPTS="${NUM_PROMPTS:-2}"

mkdir -p "$PROFILE_DIR"
echo "==> Repo:        $SGLANG_DIR"
echo "==> Server:      $BASE_URL"
echo "==> Tokenizer:   $TOKENIZER"
echo "==> Profile dir: $PROFILE_DIR"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
wait_for_server() {
    local start=$SECONDS
    while ! curl -s -g -o /dev/null "${BASE_URL}/v1/models" 2>/dev/null; do
        if (( SECONDS - start > 60 )); then
            echo "ERROR: no server reachable at ${BASE_URL} -- start it first." >&2
            exit 1
        fi
        sleep 2
    done
    echo "Server reachable at ${BASE_URL}"
}

run_bench() {
    local output_file="$1" num_prompts="$2" output_len="$3"
    ( cd "$SGLANG_DIR" && uv run --no-sync python -m sglang.bench_serving \
        --backend sglang \
        --base-url "$BASE_URL" \
        --dataset-name random-ids \
        --num-prompts "$num_prompts" \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$output_len" \
        --random-range-ratio 1.0 \
        --max-concurrency 1 \
        --request-rate inf \
        --tokenizer "$TOKENIZER" \
        --tokenize-prompt \
        --output-file "$output_file" ) ||
        echo "WARN: bench_serving failed"
}

# ----------------------------------------------------------------------------
# Attach to the running server
# ----------------------------------------------------------------------------
wait_for_server

# Optional warmup (CUDA graphs are already captured on a server that's been
# serving; harmless to re-send one short BS=1 request).
echo "Warmup: 1 short request..."
run_bench /dev/null 1 64

# ----------------------------------------------------------------------------
# BS=1 trace: profiler WITH stack
# ----------------------------------------------------------------------------
echo ""
echo "=== BS=1 trace: with stack (${STACK_PROFILE_STEPS} decode steps) ==="
curl -s -g -X POST "${BASE_URL}/start_profile" \
    -H "Content-Type: application/json" \
    -d "{
    \"output_dir\": \"$PROFILE_DIR\",
    \"with_stack\": true,
    \"profile_prefix\": \"stack_bs1\",
    \"activities\": [\"CPU\", \"GPU\"],
    \"profile_by_stage\": true,
    \"num_steps\": $STACK_PROFILE_STEPS
}"
echo ""
sleep 5
run_bench "${PROFILE_DIR}/bench_profiler_stack_bs1.jsonl" "$NUM_PROMPTS" "$OUTPUT_LEN"
# NOTE: do NOT send /stop_profile here. With num_steps set, profile_by_stage
# auto-stops the profiler and writes the trace after STACK_PROFILE_STEPS decode
# steps. A redundant /stop_profile would raise "Profiling is not in progress" and
# crash the scheduler in this sglang version. (Requires STACK_PROFILE_STEPS > 0
# and OUTPUT_LEN > STACK_PROFILE_STEPS so the cap reliably fires; both hold by
# default.) Give the auto-stop a moment to flush the trace to disk.
sleep 5

echo ""
echo "Traces saved to: $PROFILE_DIR"
ls -lh "$PROFILE_DIR"/*.trace.json.gz 2>/dev/null || echo "(no trace files found yet)"
echo ""
echo "Inspect the per-rank *-DECODE.trace.json.gz (the *-EXTEND file is warmup prefill)."
echo "Profile complete. Server left running."
