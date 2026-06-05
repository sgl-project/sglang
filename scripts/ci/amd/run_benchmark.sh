#!/usr/bin/env bash
#
# Run a SemiAnalysis serving-benchmark recipe against the sglang build *under
# test*, inside the AMD CI container (ci_sglang).
#
# Generic launcher for .github/workflows/nightly-benchmark-amd-rocm720.yml. It
# does NOT reimplement the benchmark: it resolves the recipe script for the
# requested (model-prefix, precision, runner, framework, variant) from the
# benchmark repo checkout and invokes it verbatim, so throughput / TTFT / TPOT
# / E2EL stay apples-to-apples with the published dashboard. The server binary
# comes from the under-test sglang that amd_ci_install_dependency.sh
# editable-installed into ci_sglang.
#
# To benchmark a new model, add an entry to scripts/ci/amd/benchmark-configs.yaml
# (and make sure the recipe exists in the benchmark repo) -- no changes to this
# script are needed.
#
# Runs inside ci_sglang via amd_ci_exec.sh. All inputs arrive as env vars.

set -uo pipefail

# ---- inputs (env) -----------------------------------------------------------
BENCHMARK_DIR="${BENCHMARK_DIR:-/sglang-checkout/benchmark-recipes}"
MODEL_PREFIX="${MODEL_PREFIX:?MODEL_PREFIX is required (e.g. dsv4)}"
PRECISION="${PRECISION:?PRECISION is required (e.g. fp4)}"
RUNNER="${RUNNER:?RUNNER is required (e.g. mi355x)}"
FRAMEWORK="${FRAMEWORK:-sglang}"
VARIANT="${VARIANT:-base}"               # base | mtp
BENCH_MODEL="${BENCH_MODEL:?BENCH_MODEL is required (HF id or local /path)}"
TP="${TP:-8}"
EP_SIZE="${EP_SIZE:-1}"
DP_ATTENTION="${DP_ATTENTION:-false}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
CONC_LIST="${CONC_LIST:-32}"             # space/comma separated; each relaunches the server
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
RUN_EVAL="${RUN_EVAL:-false}"
RESULT_OUT_DIR="${RESULT_OUT_DIR:-/sglang-checkout/benchmark_results}"
# ----------------------------------------------------------------------------

case "$VARIANT" in
    base) SPEC_SUFFIX="";     SPEC_DECODING="none" ;;
    mtp)  SPEC_SUFFIX="_mtp"; SPEC_DECODING="mtp"  ;;
    *) echo "ERROR: VARIANT must be 'base' or 'mtp', got '$VARIANT'" >&2; exit 2 ;;
esac

# Recipe scripts live under single_node/fixed_seq_len/ (base) or single_node/
# (some MTP variants). Probe both.
SCRIPT_BASE="${MODEL_PREFIX}_${PRECISION}_${RUNNER}_${FRAMEWORK}${SPEC_SUFFIX}"
BENCH_SCRIPT=""
for cand in \
    "$BENCHMARK_DIR/benchmarks/single_node/fixed_seq_len/${SCRIPT_BASE}.sh" \
    "$BENCHMARK_DIR/benchmarks/single_node/${SCRIPT_BASE}.sh"; do
    if [[ -f "$cand" ]]; then
        BENCH_SCRIPT="$cand"
        break
    fi
done
if [[ -z "$BENCH_SCRIPT" ]]; then
    echo "ERROR: no recipe script found for '${SCRIPT_BASE}.sh' under" >&2
    echo "       $BENCHMARK_DIR/benchmarks/single_node[/fixed_seq_len]/" >&2
    echo "       Is BENCHMARK_DIR a valid benchmark-repo checkout, and does the recipe exist?" >&2
    exit 2
fi
echo "Using recipe: $BENCH_SCRIPT"

# The recipes hardcode /workspace as the result + server-log dir.
mkdir -p /workspace "$RESULT_OUT_DIR"
CONC_LIST="${CONC_LIST//,/ }"
MAX_MODEL_LEN=$(( ISL + OSL + 256 ))
SUMMARY_FILE="${GITHUB_STEP_SUMMARY:-/dev/stdout}"

{
    echo "### ${MODEL_PREFIX} ${PRECISION} ${RUNNER} ${FRAMEWORK} (${VARIANT}) - SGLang under test"
    echo ""
    echo "model=\`${BENCH_MODEL}\` tp=${TP} ep=${EP_SIZE} dp_attn=${DP_ATTENTION} isl=${ISL} osl=${OSL} spec=${SPEC_DECODING}"
    echo ""
    echo "| conc | completed | total tput (tok/s) | output tput (tok/s) | mean TTFT (ms) | mean TPOT (ms) | mean E2EL (ms) |"
    echo "| ---- | --------- | ------------------ | ------------------- | -------------- | -------------- | -------------- |"
} >> "$SUMMARY_FILE"

overall_rc=0
for CONC in $CONC_LIST; do
    RESULT_FILENAME="${MODEL_PREFIX}_${PRECISION}_${RUNNER}_${FRAMEWORK}_${VARIANT}_tp${TP}-ep${EP_SIZE}-dpa${DP_ATTENTION}_isl${ISL}_osl${OSL}_conc${CONC}"
    echo "==================================================================="
    echo " ${MODEL_PREFIX}/${VARIANT}: conc=${CONC} isl=${ISL} osl=${OSL} dp_attn=${DP_ATTENTION}"
    echo " recipe=${BENCH_SCRIPT}"
    echo "==================================================================="

    # Env contract consumed by the recipe (check_env_vars) and, if invoked
    # downstream, the benchmark repo's utils/process_result.py.
    export MODEL="$BENCH_MODEL"
    export TP EP_SIZE DP_ATTENTION ISL OSL RANDOM_RANGE_RATIO MAX_MODEL_LEN
    export CONC="$CONC"
    export RESULT_FILENAME SPEC_DECODING RUN_EVAL
    export MODEL_PREFIX RUNNER_TYPE="$RUNNER" FRAMEWORK PRECISION DISAGG="false"
    export IMAGE="${IMAGE:-sglang-under-test}"

    rc=0
    bash "$BENCH_SCRIPT" || rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "WARN: recipe exited rc=$rc for conc=${CONC}" >&2
        overall_rc=$rc
    fi

    # Preserve artifacts into the mounted workspace dir.
    cp -f "/workspace/${RESULT_FILENAME}.json" "$RESULT_OUT_DIR/" 2>/dev/null \
        || echo "WARN: no result json produced for conc=${CONC}" >&2
    cp -f "/workspace/server.log" "$RESULT_OUT_DIR/server_${RESULT_FILENAME}.log" 2>/dev/null || true

    # Append a summary row parsed from the benchmark_serving JSON.
    python3 - "$RESULT_OUT_DIR/${RESULT_FILENAME}.json" "$CONC" "$SUMMARY_FILE" <<'PY' || true
import json
import sys

path, conc, summary = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(path) as f:
        d = json.load(f)
except Exception as e:  # noqa: BLE001 - summary row is best-effort
    row = f"| {conc} | n/a | n/a | n/a | n/a | n/a | n/a |"
    print(f"WARN: could not read {path}: {e}", file=sys.stderr)
else:
    def g(key):
        v = d.get(key)
        return f"{v:.2f}" if isinstance(v, (int, float)) else "n/a"

    row = (
        f"| {d.get('max_concurrency', conc)} | {d.get('completed', 'n/a')} | "
        f"{g('total_token_throughput')} | {g('output_throughput')} | "
        f"{g('mean_ttft_ms')} | {g('mean_tpot_ms')} | {g('mean_e2el_ms')} |"
    )
with open(summary, "a") as f:
    f.write(row + "\n")
print(row)
PY

    # The recipe launches the server in the background and relies on job
    # teardown to reap it. When sweeping multiple concurrencies in one job we
    # stop it ourselves so the next iteration gets a clean port + VRAM.
    pkill -f "sglang.launch_server" 2>/dev/null || true
    sleep 30
done

echo "Results staged in: $RESULT_OUT_DIR"
ls -la "$RESULT_OUT_DIR" || true
exit "$overall_rc"
