#!/usr/bin/env bash
#
# Hybrid serving-benchmark launcher for the AMD nightly benchmark workflow.
#
#   - The sglang server is launched HERE (config-driven flags/env), under the
#     sglang build under test, so the launch is fully controllable and a single
#     launch sweeps every concurrency.
#   - The load is driven by the SemiAnalysis InferenceX client
#     (utils/bench_serving/benchmark_serving.py via run_benchmark_serving in
#     benchmark_lib.sh), so the metrics stay apples-to-apples with the published
#     dashboard (and the dsv4 / EAGLE chat framing via --dsv4 is preserved).
#
# Used by .github/workflows/nightly-benchmark-amd-rocm720.yml, run inside the
# ci_sglang container via amd_ci_exec.sh. Inputs arrive as env vars; the launch
# spec (env + server/client args) is a base64(JSON) blob built by
# generate_benchmark_matrix.py from benchmark-configs.yaml.
#
# To benchmark a new model / change how the server launches, edit
# scripts/ci/amd/benchmark-configs.yaml -- not this script.

set -o pipefail

# ---- inputs (env) -----------------------------------------------------------
BENCHMARK_DIR="${BENCHMARK_DIR:-/sglang-checkout/benchmark-recipes}"  # InferenceX checkout (client + benchmark_lib.sh)
MODEL_PREFIX="${MODEL_PREFIX:?MODEL_PREFIX is required (e.g. dsv4)}"
PRECISION="${PRECISION:?PRECISION is required (e.g. fp4)}"
RUNNER="${RUNNER:?RUNNER is required (e.g. mi355x)}"
FRAMEWORK="${FRAMEWORK:-sglang}"
VARIANT="${VARIANT:-base}"                 # base | mtp (already merged into LAUNCH_SPEC)
BENCH_MODEL="${BENCH_MODEL:?BENCH_MODEL is required (HF id or local /path)}"
TP="${TP:-8}"
EP_SIZE="${EP_SIZE:-1}"
DP_ATTENTION="${DP_ATTENTION:-false}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
CONC_LIST="${CONC_LIST:-32}"               # space/comma separated; ONE launch sweeps all
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
LAUNCH_SPEC_B64="${LAUNCH_SPEC_B64:-}"     # base64(JSON): {env, server_args, client_args, chat_template, bench_backend}
RESULT_OUT_DIR="${RESULT_OUT_DIR:-/sglang-checkout/benchmark_results}"
PORT="${PORT:-30000}"
export PORT
export EVAL_ONLY="${EVAL_ONLY:-false}"     # benchmark_lib.sh references this unguarded

SPEC_DECODING=$([ "$VARIANT" = "mtp" ] && echo "mtp" || echo "none")

# ---- decode the launch spec -------------------------------------------------
# Emits `export K=V` for env plus SERVER_ARGS / CLIENT_ARGS / CHAT_TEMPLATE /
# BENCH_BACKEND, all shell-quoted by python (stdlib only; no pyyaml needed here).
SERVER_ARGS=""
CLIENT_ARGS=""
CHAT_TEMPLATE=""
BENCH_BACKEND="vllm"
spec_sh="$(LAUNCH_SPEC_B64="$LAUNCH_SPEC_B64" python3 - <<'PY'
import base64
import json
import os
import shlex

raw = os.environ.get("LAUNCH_SPEC_B64", "") or ""
spec = json.loads(base64.b64decode(raw)) if raw else {}
lines = []
for k, v in (spec.get("env") or {}).items():
    lines.append(f"export {k}={shlex.quote(str(v))}")
lines.append(f"SERVER_ARGS={shlex.quote(spec.get('server_args', '') or '')}")
lines.append(f"CLIENT_ARGS={shlex.quote(spec.get('client_args', '') or '')}")
lines.append(f"CHAT_TEMPLATE={shlex.quote(spec.get('chat_template', '') or '')}")
lines.append(f"BENCH_BACKEND={shlex.quote(spec.get('bench_backend', 'vllm') or 'vllm')}")
print("\n".join(lines))
PY
)"
eval "$spec_sh"

read -ra SERVER_ARGS_ARR <<< "$SERVER_ARGS"
read -ra CLIENT_ARGS_ARR <<< "$CLIENT_ARGS"

# Resolve chat template relative to the benchmark repo unless absolute.
CHAT_TEMPLATE_FLAG=()
if [[ -n "$CHAT_TEMPLATE" ]]; then
    case "$CHAT_TEMPLATE" in
        /*) ct_path="$CHAT_TEMPLATE" ;;
        *)  ct_path="$BENCHMARK_DIR/$CHAT_TEMPLATE" ;;
    esac
    CHAT_TEMPLATE_FLAG=(--chat-template "$ct_path")
fi

# Parallelism flags derived from the (override-able) knobs.
PARALLEL_ARGS=(--tensor-parallel-size "$TP")
if [[ "$DP_ATTENTION" == "true" ]]; then
    PARALLEL_ARGS+=(--dp "$TP" --enable-dp-attention --enable-prefill-delayer)
fi
if [[ "${EP_SIZE:-1}" -gt 1 ]]; then
    PARALLEL_ARGS+=(--ep-size "$EP_SIZE")
fi

CONC_LIST="${CONC_LIST//,/ }"
MAX_MODEL_LEN=$(( ISL + OSL + 256 ))
MAX_CONC=1
for c in $CONC_LIST; do [[ "$c" -gt "$MAX_CONC" ]] && MAX_CONC="$c"; done

mkdir -p /workspace "$RESULT_OUT_DIR"
SERVER_LOG="/workspace/server.log"
export MODEL="$BENCH_MODEL"

# benchmark_lib.sh provides start_gpu_monitor / wait_for_server_ready /
# run_benchmark_serving / stop_gpu_monitor (the InferenceX client wrapper).
source "$BENCHMARK_DIR/benchmarks/benchmark_lib.sh"

SERVER_PID=""
cleanup() {
    [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
    sleep 3
    pkill -f "sglang.launch_server" 2>/dev/null || true
    stop_gpu_monitor 2>/dev/null || true
}
trap cleanup EXIT

echo "==================================================================="
echo " ${MODEL_PREFIX} ${PRECISION} ${RUNNER} ${FRAMEWORK} (${VARIANT})"
echo "   model=${MODEL} tp=${TP} ep=${EP_SIZE} dp_attn=${DP_ATTENTION}"
echo "   isl=${ISL} osl=${OSL} conc_list='${CONC_LIST}' max_conc=${MAX_CONC} port=${PORT}"
echo "   server-args: ${SERVER_ARGS}"
echo "   client-args: ${CLIENT_ARGS}  backend=${BENCH_BACKEND}"
echo "==================================================================="

start_gpu_monitor

# Launch the server ONCE; all concurrencies are swept against it.
set -x
python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    --context-length "$MAX_MODEL_LEN" \
    --max-running-requests "$MAX_CONC" \
    "${CHAT_TEMPLATE_FLAG[@]}" \
    "${SERVER_ARGS_ARR[@]}" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
set +x

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

SUMMARY_FILE="${GITHUB_STEP_SUMMARY:-/dev/stdout}"
{
    echo "### ${MODEL_PREFIX} ${PRECISION} ${RUNNER} ${FRAMEWORK} (${VARIANT}) - SGLang under test"
    echo ""
    echo "model=\`${MODEL}\` tp=${TP} ep=${EP_SIZE} dp_attn=${DP_ATTENTION} isl=${ISL} osl=${OSL} spec=${SPEC_DECODING}"
    echo ""
    echo "| conc | completed | total tput (tok/s) | total tput/gpu (tok/s/gpu) | output tput (tok/s) | median TTFT (ms) | median TPOT (ms) | interactivity (tok/s/user) | median ITL (ms) | median E2EL (ms) |"
    echo "| ---- | --------- | ------------------ | -------------------------- | ------------------- | ---------------- | ---------------- | -------------------------- | --------------- | ---------------- |"
} >> "$SUMMARY_FILE"

overall_rc=0
for CONC in $CONC_LIST; do
    RESULT_FILENAME="${MODEL_PREFIX}_${PRECISION}_${RUNNER}_${FRAMEWORK}_${VARIANT}_tp${TP}-ep${EP_SIZE}-dpa${DP_ATTENTION}_isl${ISL}_osl${OSL}_conc${CONC}"
    echo "--- benchmark: conc=${CONC} -> ${RESULT_FILENAME}"

    rc=0
    run_benchmark_serving \
        --model "$MODEL" \
        --port "$PORT" \
        --backend "$BENCH_BACKEND" \
        --input-len "$ISL" \
        --output-len "$OSL" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$((CONC * 10))" \
        --max-concurrency "$CONC" \
        --result-filename "$RESULT_FILENAME" \
        --result-dir /workspace \
        --bench-serving-dir "$BENCHMARK_DIR" \
        --server-pid "$SERVER_PID" \
        "${CLIENT_ARGS_ARR[@]}" || rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "WARN: benchmark client exited rc=$rc for conc=${CONC}" >&2
        overall_rc=$rc
    fi

    cp -f "/workspace/${RESULT_FILENAME}.json" "$RESULT_OUT_DIR/" 2>/dev/null \
        || echo "WARN: no result json produced for conc=${CONC}" >&2

    python3 - "$RESULT_OUT_DIR/${RESULT_FILENAME}.json" "$CONC" "$SUMMARY_FILE" "$TP" <<'PY' || true
import json
import sys

path, conc, summary = sys.argv[1], sys.argv[2], sys.argv[3]
gpus = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 0
try:
    with open(path) as f:
        d = json.load(f)
except Exception as e:  # noqa: BLE001 - summary row is best-effort
    row = f"| {conc} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
    print(f"WARN: could not read {path}: {e}", file=sys.stderr)
else:
    def g(key):
        v = d.get(key)
        return f"{v:.2f}" if isinstance(v, (int, float)) else "n/a"

    # total throughput per GPU (single-node: GPUs == tensor-parallel size)
    tt = d.get("total_token_throughput")
    tput_per_gpu = f"{tt / gpus:.2f}" if isinstance(tt, (int, float)) and gpus > 0 else "n/a"
    # interactivity = 1s / TPOT = output tokens/s for a single user
    tpot = d.get("median_tpot_ms")
    interactivity = f"{1000.0 / tpot:.2f}" if isinstance(tpot, (int, float)) and tpot > 0 else "n/a"

    row = (
        f"| {d.get('max_concurrency', conc)} | {d.get('completed', 'n/a')} | "
        f"{g('total_token_throughput')} | {tput_per_gpu} | {g('output_throughput')} | "
        f"{g('median_ttft_ms')} | {g('median_tpot_ms')} | {interactivity} | "
        f"{g('median_itl_ms')} | {g('median_e2el_ms')} |"
    )
with open(summary, "a") as f:
    f.write(row + "\n")
print(row)
PY

    cp -f "/workspace/server.log" "$RESULT_OUT_DIR/server_${MODEL_PREFIX}_${VARIANT}.log" 2>/dev/null || true
done

echo "Results staged in: $RESULT_OUT_DIR"
ls -la "$RESULT_OUT_DIR" || true
exit "$overall_rc"
