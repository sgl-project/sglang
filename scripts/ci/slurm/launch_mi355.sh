#!/usr/bin/env bash
# Launch a multi-node SGLang PD-disaggregation benchmark on the AMD MI355
# cluster via Slurm. In-repo analog of launch_gb200.sh: instead of cloning an
# external orchestrator (NVIDIA/srt-slurm), the whole recipe lives under
# scripts/ci/slurm/mi355/ and is submitted with a single sbatch.
#
# Required environment variables (set by the GitHub Actions workflow):
#   FRAMEWORK         - "sglang-disagg"
#   MODEL_PREFIX      - short prefix, e.g. "qwen3-8b"
#   PRECISION         - "bf16", "fp8", ...
#   ISL / OSL         - input / output sequence lengths (documentation only;
#                       the authoritative values come from the recipe YAML)
#   CONFIG_FILE       - recipe YAML path, repo-relative
#                       (e.g. scripts/ci/slurm/mi355/recipes/1p1d-tp8.yaml)
#   RESULT_FILENAME   - prefix for output JSON filenames
#   RUNNER_NAME       - GitHub Actions runner name (used to tag the Slurm job)
#   IMAGE             - ROCm SGLang docker image (mooncake + AITER built in)
#   GITHUB_WORKSPACE  - set automatically by GitHub Actions (repo checkout root)
#
# Optional:
#   SLURM_PARTITION   - default: amd-sglang
#   SLURM_ACCOUNT     - default: amd-sglang
#   TIME_LIMIT        - default: 02:00:00
#   MODEL_CACHE_DIR   - HF cache root, bind-mounted to /models inside the
#                       container (default: /it-share/model_coverage)
#   NODELIST          - explicit comma-separated nodelist (size must equal the
#                       recipe's prefill_nodes + decode_nodes)
#   SLURM_EXCLUDE_NODES - comma-separated nodes to keep the job off of
#                       (default: mia1-p01-g20). g20's ionic (AMD Pollara)
#                       RDMA driver ABI is currently out of sync with the
#                       container userspace, so the NICs enumerate as 0 devices
#                       inside the container. g09 + g29 are a natively-matching
#                       clean pair; excluding g20 keeps PD jobs on good NICs
#                       until the fleet driver/firmware stack is standardized.

set -euo pipefail
set -x

: "${FRAMEWORK:?}"
: "${MODEL_PREFIX:?}"
: "${PRECISION:?}"
: "${CONFIG_FILE:?}"
: "${RESULT_FILENAME:?}"
: "${RUNNER_NAME:?}"
: "${IMAGE:?IMAGE (ROCm SGLang docker image) must be set}"
: "${GITHUB_WORKSPACE:?}"

SLURM_PARTITION="${SLURM_PARTITION:-amd-sglang}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-amd-sglang}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/it-share/model_coverage}"

cd "$GITHUB_WORKSPACE"
RECIPE="$CONFIG_FILE"
[ -f "$RECIPE" ] || { echo "ERROR: recipe not found: $RECIPE"; exit 1; }

# ---------------------------------------------------------------------------
# Parse the recipe YAML into shell variables (incl. CLI flag strings built from
# the common/prefill/decode dicts). tensor-parallel-size is emitted separately
# (PREFILL_TP/DECODE_TP) because it is passed explicitly as --tp.
# ---------------------------------------------------------------------------
RECIPE_ENV="$(python3 - "$RECIPE" <<'PY'
import shlex, sys, yaml

with open(sys.argv[1]) as f:
    r = yaml.safe_load(f)

def flags(d, skip=()):
    out = []
    for k, v in (d or {}).items():
        if k in skip:
            continue
        if isinstance(v, bool):
            if v:
                out.append(f"--{k}")
        else:
            out += [f"--{k}", str(v)]
    return " ".join(out)

sgl = r.get("backend", {}).get("sglang_config", {})
common, p, d = sgl.get("common", {}), sgl.get("prefill", {}), sgl.get("decode", {})
res, topo = r.get("resources", {}), r.get("topology", {})

def emit(k, v):
    print(f"{k}={shlex.quote(str(v))}")

emit("MODEL", r["model"])
emit("SERVED_MODEL_NAME", r.get("served_model_name", r["model"]))
emit("R_ISL", r.get("isl", 1024))
emit("R_OSL", r.get("osl", 1024))
emit("RANDOM_RANGE_RATIO", r.get("random_range_ratio", 1.0))
emit("CONCURRENCIES", " ".join(str(c) for c in r.get("concurrencies", [])))
emit("PREFILL_NODES", topo.get("prefill_nodes", 1))
emit("DECODE_NODES", topo.get("decode_nodes", 1))
emit("PREFILL_WORKERS", res.get("prefill_workers", 1))
emit("DECODE_WORKERS", res.get("decode_workers", 1))
emit("GPUS_PER_NODE", res.get("gpus_per_node", 8))
emit("PREFILL_TP", p.get("tensor-parallel-size", 8))
emit("DECODE_TP", d.get("tensor-parallel-size", 8))
emit("COMMON_FLAGS", flags(common))
emit("PREFILL_FLAGS", flags(p, skip=("tensor-parallel-size",)))
emit("DECODE_FLAGS", flags(d, skip=("tensor-parallel-size",)))
PY
)"
eval "$RECIPE_ENV"

NNODES=$((PREFILL_NODES + DECODE_NODES))
PREFILL_GPUS=$((PREFILL_NODES * GPUS_PER_NODE))
DECODE_GPUS=$((DECODE_NODES * GPUS_PER_NODE))

echo "--- recipe parsed ---"
echo "model=$MODEL nnodes=$NNODES prefill_nodes=$PREFILL_NODES decode_nodes=$DECODE_NODES"
echo "prefill_tp=$PREFILL_TP decode_tp=$DECODE_TP concurrencies=[$CONCURRENCIES]"

# ---------------------------------------------------------------------------
# Per-runner workspace on the shared filesystem (readable by compute nodes).
# ---------------------------------------------------------------------------
WORKDIR="$HOME/sglang-ci/$RUNNER_NAME"
LOGS_DIR="$WORKDIR/logs"
RESULTS_DIR="$WORKDIR/results"
rm -rf "$WORKDIR"
mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

# Job environment file sourced by pd_disagg.slurm (handles flag strings that
# contain spaces, which sbatch --export does not pass cleanly).
JOB_ENV="$WORKDIR/job_env.sh"
cat > "$JOB_ENV" <<EOF
export IMAGE=$(printf '%q' "$IMAGE")
export REPO_DIR=$(printf '%q' "$GITHUB_WORKSPACE")
export MODEL_CACHE_DIR=$(printf '%q' "$MODEL_CACHE_DIR")
export MODEL=$(printf '%q' "$MODEL")
export SERVED_MODEL_NAME=$(printf '%q' "$SERVED_MODEL_NAME")
export PREFILL_NODES=$PREFILL_NODES
export DECODE_NODES=$DECODE_NODES
export PREFILL_WORKERS=$PREFILL_WORKERS
export DECODE_WORKERS=$DECODE_WORKERS
export PREFILL_TP=$PREFILL_TP
export DECODE_TP=$DECODE_TP
export GPUS_PER_NODE=$GPUS_PER_NODE
export ISL=$R_ISL
export OSL=$R_OSL
export RANDOM_RANGE_RATIO=$(printf '%q' "$RANDOM_RANGE_RATIO")
export CONCURRENCIES=$(printf '%q' "$CONCURRENCIES")
export COMMON_FLAGS=$(printf '%q' "$COMMON_FLAGS")
export PREFILL_FLAGS=$(printf '%q' "$PREFILL_FLAGS")
export DECODE_FLAGS=$(printf '%q' "$DECODE_FLAGS")
export RESULTS_DIR=$(printf '%q' "$RESULTS_DIR")
export ROUTER_PORT=${ROUTER_PORT:-8000}
export PREFILL_SERVER_PORT=${PREFILL_SERVER_PORT:-30000}
export DECODE_SERVER_PORT=${DECODE_SERVER_PORT:-30001}
export PREFILL_BOOTSTRAP_PORT=${PREFILL_BOOTSTRAP_PORT:-8998}
export DECODE_BOOTSTRAP_PORT=${DECODE_BOOTSTRAP_PORT:-9001}
EOF
echo "--- job_env.sh ---"; cat "$JOB_ENV"

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
NODELIST_OPT=()
if [ -n "${NODELIST:-}" ]; then
    NODELIST_OPT=(--nodelist "$NODELIST")
fi

# Keep PD jobs off nodes with a known-bad in-container RDMA stack. g20's ionic
# driver ABI is out of sync (0 NIC devices in-container); g09 + g29 are clean.
EXCLUDE_OPT=()
SLURM_EXCLUDE_NODES="${SLURM_EXCLUDE_NODES:-mia1-p01-g20}"
if [ -n "$SLURM_EXCLUDE_NODES" ] && [ -z "${NODELIST:-}" ]; then
    EXCLUDE_OPT=(--exclude "$SLURM_EXCLUDE_NODES")
fi

JOB_ID=$(sbatch --parsable \
    --exclusive \
    -N "$NNODES" -n "$NNODES" --ntasks-per-node=1 \
    --gres="gpu:${GPUS_PER_NODE}" \
    --time "$TIME_LIMIT" \
    --partition "$SLURM_PARTITION" \
    --account "$SLURM_ACCOUNT" \
    --job-name "$RUNNER_NAME" \
    --output "$LOGS_DIR/slurm-%j.out" \
    --error "$LOGS_DIR/slurm-%j.err" \
    "${NODELIST_OPT[@]}" \
    "${EXCLUDE_OPT[@]}" \
    --export=ALL,JOB_ENV="$JOB_ENV" \
    "$GITHUB_WORKSPACE/scripts/ci/slurm/mi355/pd_disagg.slurm")

[ -n "$JOB_ID" ] || { echo "ERROR: sbatch did not return a job id"; exit 1; }
echo "Submitted Slurm job: $JOB_ID"

set +x
LOG_FILE="$LOGS_DIR/slurm-${JOB_ID}.out"
while ! ls "$LOG_FILE" &>/dev/null; do
    if ! squeue -j "$JOB_ID" --noheader 2>/dev/null | grep -q "$JOB_ID"; then
        echo "ERROR: Job $JOB_ID left the queue before creating $LOG_FILE"
        scontrol show job "$JOB_ID" || true
        cat "$LOGS_DIR/slurm-${JOB_ID}.err" 2>/dev/null || true
        exit 1
    fi
    echo "Waiting for job $JOB_ID to start..."
    sleep 5
done

( while squeue -j "$JOB_ID" --noheader 2>/dev/null | grep -q "$JOB_ID"; do sleep 10; done ) &
POLL_PID=$!
tail -F -s 2 -n+1 "$LOG_FILE" --pid=$POLL_PID 2>/dev/null
wait $POLL_PID
set -x

echo "Job $JOB_ID finished. Collecting results from $RESULTS_DIR ..."

# ---------------------------------------------------------------------------
# Collect results + logs back into the workspace for the workflow's
# Process/Upload steps. Filenames embed ctx=<prefill gpus> gen=<decode gpus>
# so process_result.py can recover the GPU split (same convention as GB200).
# ---------------------------------------------------------------------------
tar czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" -C "$LOGS_DIR" . 2>/dev/null || true

RECIPE_NAME="$(basename "$RECIPE" .yaml)"
RESULT_COUNT=0
for result_file in "$RESULTS_DIR"/results_concurrency_*.json; do
    [ -f "$result_file" ] || continue
    conc=$(basename "$result_file" | sed -n 's/results_concurrency_\([0-9]*\)\.json/\1/p')
    [ -n "$conc" ] || continue
    DEST="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${RECIPE_NAME}_conc${conc}_gpus_$((PREFILL_GPUS + DECODE_GPUS))_ctx_${PREFILL_GPUS}_gen_${DECODE_GPUS}.json"
    cp "$result_file" "$DEST"
    echo "Saved: $DEST"
    RESULT_COUNT=$((RESULT_COUNT + 1))
done

if [ "$RESULT_COUNT" -eq 0 ]; then
    echo "ERROR: no result JSONs produced — benchmark failed"
    exit 1
fi
echo "Done. Collected $RESULT_COUNT result file(s)."
