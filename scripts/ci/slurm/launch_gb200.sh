#!/usr/bin/env bash
# Launch a dynamo-sglang benchmark job on the GB200 cluster via srt-slurm.
#
# Required environment variables (set by the GitHub Actions workflow):
#   FRAMEWORK         - must be "dynamo-sglang"
#   MODEL             - HuggingFace model ID (used as fallback if no local path)
#   MODEL_PREFIX      - short prefix: "dsr1"
#   PRECISION         - "fp8" or "fp4"
#   ISL               - input sequence length (e.g. "1024")
#   OSL               - output sequence length (e.g. "1024")
#   CONFIG_FILE       - path relative to srt-slurm repo root (e.g. recipes/gb200-fp8/1k1k/low-latency.yaml)
#   RESULT_FILENAME   - prefix for output JSON filenames
#   RUNNER_NAME       - GitHub Actions runner name (used to tag the Slurm job)
#   SQUASH_FILE       - path to pre-imported sglang enroot squash file on Lustre
#   NGINX_SQUASH_FILE - path to pre-imported nginx enroot squash file on Lustre
#   SLURM_PARTITION   - Slurm partition (default: batch)
#   SLURM_ACCOUNT     - Slurm account  (default: sglang)
#   SRT_SLURM_BRANCH  - branch of srt-slurm repo to check out
#   GITHUB_WORKSPACE  - set automatically by GitHub Actions
#   MATRIX_CONFIG_NAME- matrix entry name (e.g. dsr1-fp4-1k1k-mid-curve); used in S3 prefix
#   S3_BUCKET         - MinIO bucket for benchmark log uploads
#   S3_ENDPOINT_URL   - MinIO endpoint URL (e.g. https://minio.<host>.nip.io)
#   AWS_ACCESS_KEY_ID - writer access key for S3_BUCKET (via GH secrets)
#   AWS_SECRET_ACCESS_KEY - writer secret key for S3_BUCKET (via GH secrets)

set -euo pipefail
set -x

# ---------------------------------------------------------------------------
# Validate required vars
# ---------------------------------------------------------------------------
: "${FRAMEWORK:?}"
: "${MODEL_PREFIX:?}"
: "${PRECISION:?}"
: "${ISL:?}"
: "${OSL:?}"
: "${CONFIG_FILE:?}"
: "${RESULT_FILENAME:?}"
: "${RUNNER_NAME:?}"
: "${SQUASH_FILE:?}"
: "${NGINX_SQUASH_FILE:?}"
: "${GITHUB_WORKSPACE:?}"

SLURM_PARTITION="${SLURM_PARTITION:-batch}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-sglang}"
SRT_SLURM_BRANCH="${SRT_SLURM_BRANCH:-sglang-nightly-regression}"

# ---------------------------------------------------------------------------
# Resolve local model paths on Lustre (avoids re-downloading on each run)
# ---------------------------------------------------------------------------
if [[ "$MODEL_PREFIX" == "dsr1" && "$PRECISION" == "fp8" ]]; then
    MODEL_PATH="/mnt/lustre01/models/deepseek-r1-0528"
    SRT_SLURM_MODEL_PREFIX="dsr1-fp8"
elif [[ "$MODEL_PREFIX" == "dsr1" && "$PRECISION" == "fp4" ]]; then
    MODEL_PATH="/mnt/lustre01/models/deepseek-r1-0528-fp4-v2/"
    SRT_SLURM_MODEL_PREFIX="dsr1-fp4"
else
    MODEL_PATH="$MODEL"
    SRT_SLURM_MODEL_PREFIX="$MODEL_PREFIX"
fi

# ---------------------------------------------------------------------------
# Set up per-runner Lustre workspace (cleaned before each run, accessible
# to both the runner and compute nodes)
# ---------------------------------------------------------------------------
LUSTRE_WORKSPACE="/mnt/lustre01/users-public/sglang-ci/workspace/${RUNNER_NAME}"
rm -rf "$LUSTRE_WORKSPACE"
mkdir -p "$LUSTRE_WORKSPACE"

# ---------------------------------------------------------------------------
# Clone and set up srt-slurm
# ---------------------------------------------------------------------------
SRT_REPO_DIR="$LUSTRE_WORKSPACE/srt-slurm"

git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
cd "$SRT_REPO_DIR"
git checkout "$SRT_SLURM_BRANCH"
echo "--- srt-slurm last commit ---"
git log -1 --format="commit %H%nauthor %an%ndate   %ad%nsubject %s" --date=iso

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

uv venv
source .venv/bin/activate
uv pip install -e .

if ! command -v srtctl &>/dev/null; then
    echo "ERROR: srtctl installation failed"
    exit 1
fi

# ---------------------------------------------------------------------------
# Generate srtslurm.yaml
# ---------------------------------------------------------------------------
SRTCTL_ROOT="$SRT_REPO_DIR"

: "${S3_BUCKET:?S3_BUCKET must be set}"
: "${S3_ENDPOINT_URL:?S3_ENDPOINT_URL must be set}"
: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID must be set}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY must be set}"
: "${MATRIX_CONFIG_NAME:?MATRIX_CONFIG_NAME must be set}"
: "${GITHUB_RUN_ID:?GITHUB_RUN_ID must be set}"
: "${GITHUB_RUN_ATTEMPT:?GITHUB_RUN_ATTEMPT must be set}"

# Map the GitHub trigger into a friendlier top-level prefix: cron/manual.
case "${GITHUB_EVENT_NAME:-}" in
    schedule)          TRIGGER=cron ;;
    workflow_dispatch) TRIGGER=manual ;;
    *)                 TRIGGER="${GITHUB_EVENT_NAME:-unknown}" ;;
esac

# Format ISL/OSL as "1k1k" / "1k8k" / "8k1k" etc. for the S3 prefix, so logs
# group naturally by sequence-length bucket under each run.
fmt_seq_len() {
    local n=$1
    if (( n % 1024 == 0 )); then echo "$((n / 1024))k"; else echo "$n"; fi
}
SEQ_LEN="$(fmt_seq_len "$ISL")$(fmt_seq_len "$OSL")"

S3_PREFIX="${TRIGGER}/${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}/${SEQ_LEN}/${MATRIX_CONFIG_NAME}"

cat > srtslurm.yaml <<EOF
# SRT SLURM configuration for SGLang GB200 nightly CI
default_account: "${SLURM_ACCOUNT}"
default_partition: "${SLURM_PARTITION}"
default_time_limit: "6:00:00"

gpus_per_node: 4
network_interface: ""

srtctl_root: "${SRTCTL_ROOT}"

model_paths:
  "${SRT_SLURM_MODEL_PREFIX}": "${MODEL_PATH}"

containers:
  dynamo-sglang: ${SQUASH_FILE}
  nginx: ${NGINX_SQUASH_FILE}
  nginx-sqsh: ${NGINX_SQUASH_FILE}

# srt-slurm postprocess uploads /logs to this bucket after each Slurm job.
# Credentials are read from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars,
# not written to disk. srt-slurm appends /<date>/<slurm-job-id>/ after prefix.
reporting:
  s3:
    bucket: "${S3_BUCKET}"
    prefix: "${S3_PREFIX}"
    endpoint_url: "${S3_ENDPOINT_URL}"
EOF

echo "--- srtslurm.yaml ---"
cat srtslurm.yaml
echo "--- S3 log upload: s3://${S3_BUCKET}/${S3_PREFIX}/ ---"

make setup ARCH=aarch64

# ---------------------------------------------------------------------------
# Patch job name and submit via srtctl
# ---------------------------------------------------------------------------
sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "$CONFIG_FILE"

SRTCTL_OUTPUT=$(srtctl apply -f "$CONFIG_FILE" \
    --tags "gb200,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},sglang-nightly-$(date +%Y%m%d)" \
    --setup-script install-torchao.sh 2>&1)
echo "$SRTCTL_OUTPUT"

JOB_ID=$(echo "$SRTCTL_OUTPUT" | grep -oP '✅ Job \K[0-9]+' || echo "$SRTCTL_OUTPUT" | grep -oP 'Job \K[0-9]+' || true)

if [ -z "$JOB_ID" ]; then
    echo "ERROR: Could not extract JOB_ID from srtctl output"
    exit 1
fi

echo "Submitted Slurm job: $JOB_ID"

set +x

# ---------------------------------------------------------------------------
# Wait for job and stream logs
# ---------------------------------------------------------------------------
LOGS_DIR="outputs/$JOB_ID/logs"
LOG_FILE="$LOGS_DIR/sweep_${JOB_ID}.log"

mkdir -p "$LOGS_DIR"

while ! ls "$LOG_FILE" &>/dev/null; do
    if ! squeue -j "$JOB_ID" --noheader 2>/dev/null | grep -q "$JOB_ID"; then
        echo "ERROR: Job $JOB_ID failed before creating log file"
        scontrol show job "$JOB_ID" || true
        exit 1
    fi
    echo "Waiting for job $JOB_ID to start and $LOG_FILE to appear..."
    sleep 5
done

(
    while squeue -j "$JOB_ID" --noheader 2>/dev/null | grep -q "$JOB_ID"; do
        sleep 10
    done
) &
POLL_PID=$!

tail -F -s 2 -n+1 "$LOG_FILE" --pid=$POLL_PID 2>/dev/null

wait $POLL_PID

set -x

echo "Job $JOB_ID completed. Collecting results..."

# ---------------------------------------------------------------------------
# Collect results
# ---------------------------------------------------------------------------
if [ ! -d "$LOGS_DIR" ]; then
    echo "WARNING: Logs directory not found at $LOGS_DIR"
    exit 1
fi

cp -r "$LOGS_DIR" "$GITHUB_WORKSPACE/LOGS"
tar czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" -C "$LOGS_DIR" .

RESULT_SUBDIRS=$(find "$LOGS_DIR" -maxdepth 1 -type d -name "*isl*osl*" 2>/dev/null || true)

if [ -z "$RESULT_SUBDIRS" ]; then
    echo "ERROR: No result subdirectories found in $LOGS_DIR — benchmark did not produce any output"
    exit 1
else
    RESULT_COUNT=0
    for result_subdir in $RESULT_SUBDIRS; do
        CONFIG_NAME=$(basename "$result_subdir")
        RESULT_FILES=$(find "$result_subdir" -name "results_concurrency_*.json" 2>/dev/null || true)
        for result_file in $RESULT_FILES; do
            if [ -f "$result_file" ]; then
                filename=$(basename "$result_file")
                concurrency=$(echo "$filename" | sed -n 's/results_concurrency_\([0-9]*\)_gpus_.*/\1/p')
                gpus=$(echo "$filename" | sed -n 's/results_concurrency_[0-9]*_gpus_\([0-9]*\)_ctx_.*/\1/p')
                ctx=$(echo "$filename" | sed -n 's/.*_ctx_\([0-9]*\)_gen_.*/\1/p')
                gen=$(echo "$filename" | sed -n 's/.*_gen_\([0-9]*\)\.json/\1/p')
                DEST="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${CONFIG_NAME}_conc${concurrency}_gpus_${gpus}_ctx_${ctx}_gen_${gen}.json"
                cp "$result_file" "$DEST"
                echo "Saved: $DEST"
                RESULT_COUNT=$((RESULT_COUNT + 1))
            fi
        done
    done
    if [ "$RESULT_COUNT" -eq 0 ]; then
        echo "ERROR: Result subdirectories found but no result JSON files produced — benchmark failed"
        exit 1
    fi
fi

echo "Done."
