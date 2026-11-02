#!/usr/bin/env bash
# Run one AMD CI suite on a single MI35x node scheduled through Spur.
#
# The GitHub runner lives on the cluster's login node. This script submits a
# one-node, whole-node batch job running amd_ci_job_mi35x.sh -- the same
# container steps a linux-mi35x-gpu-* ARC job runs in pr-test-amd.yml -- streams
# the job's log into the step output, and exits with the suite's exit code.
#
# Spur behaviours handled here (see launch_mi355x.sh, #40113 and #41025):
#   * batch output is not streamed back, so the job logs to shared storage and
#     this script re-reads that file by offset each poll (`tail -F` misses
#     appends made on another host over NFS);
#   * a job can be cancelled or held (JobHoldMaxRequeue) before it starts, so a
#     job that never started is resubmitted, up to SUBMIT_ATTEMPTS times;
#   * the controller sometimes answers with an empty queue, so one missing
#     state reading is not taken as "job finished";
#   * a two-field time limit parses as HH:MM, so TIME_LIMIT must be HH:MM:SS;
#   * the job's exit-code file can stay invisible on the login host for up to
#     a minute after the node writes it (NFS negative lookup caching), so a
#     finished job is polled for it rather than checked once.
#
# Usage:
#   launch_amd_ci_mi35x.sh --suite NAME [--rocm-version rocm10] [-- run_suite.py args...]
#
# Required environment (set in the runner's .env, like the DI nightly runner):
#   SLURM_PARTITION, SLURM_ACCOUNT, SLURM_QOS
#   CI_NODE_SCRATCH  node-local directory for the checkout copy and the
#                    persistent /sgl-data cache (models, pip, MIOpen)
# Optional:
#   SLURM_EXCLUDE    comma-separated nodes to keep the scheduler off
#   TIME_LIMIT       default 01:30:00
#   CI_SHARED_DIR    shared-storage dir for job logs, default $HOME/.amd_ci_spur
#   SUBMIT_ATTEMPTS  default 3

set -euo pipefail

SUITE=""
ROCM_VERSION="rocm10"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite) SUITE="$2"; shift 2 ;;
        --rocm-version) ROCM_VERSION="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "ERROR: unknown argument '$1'" >&2; exit 2 ;;
    esac
done
SUITE_ARGS="$*"

: "${SUITE:?--suite is required}"
: "${SLURM_PARTITION:?set in the runner environment}"
: "${SLURM_ACCOUNT:?set in the runner environment}"
: "${SLURM_QOS:?set in the runner environment}"
: "${CI_NODE_SCRATCH:?set in the runner environment}"
: "${GITHUB_WORKSPACE:?}"
TIME_LIMIT="${TIME_LIMIT:-01:30:00}"
SUBMIT_ATTEMPTS="${SUBMIT_ATTEMPTS:-3}"
if [[ ! "$TIME_LIMIT" =~ ^([0-9]+-)?[0-9]+:[0-9]{2}:[0-9]{2}$ ]]; then
    echo "ERROR: TIME_LIMIT must be HH:MM:SS or D-HH:MM:SS (got '$TIME_LIMIT'); Spur reads '30:00' as 30 hours" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The workflow's cleanup step cancels by this prefix, so keep the two in sync.
JOB_PREFIX="amd-ci-${RUNNER_NAME:-local}-${GITHUB_RUN_ID:-0}-"
JOB_NAME="${JOB_PREFIX}${GITHUB_JOB:-job}"
RUN_DIR="${CI_SHARED_DIR:-$HOME/.amd_ci_spur}/${JOB_NAME}"
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"

# Read by amd_ci_job_mi35x.sh; sbatch exports the submitting environment.
export CI_SRC="$GITHUB_WORKSPACE" CI_SUITE="$SUITE" CI_SUITE_ARGS="$SUITE_ARGS"
export CI_ROCM_VERSION="$ROCM_VERSION" CI_NODE_SCRATCH CI_EXIT_FILE="$RUN_DIR/exit_code"

JOB_ID=""
cleanup() {
    if [[ -n "$JOB_ID" && ! -f "$CI_EXIT_FILE" ]]; then
        echo "Cancelling cluster job $JOB_ID"
        scancel "$JOB_ID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

LOG_OFFSET=0
stream_log() {  # print what the job appended to $LOG since the last call
    local size
    size="$(stat -c %s "$LOG" 2>/dev/null || echo 0)"
    if (( size > LOG_OFFSET )); then
        tail -c +"$((LOG_OFFSET + 1))" "$LOG" | head -c "$((size - LOG_OFFSET))"
        LOG_OFFSET="$size"
    fi
}

wait_for_exit_file() {  # up to 90s; listing the directory forces a fresh NFS lookup
    local _
    for _ in $(seq 1 18); do
        ls "$RUN_DIR" > /dev/null 2>&1 || true
        [[ -s "$CI_EXIT_FILE" ]] && return 0
        sleep 5
    done
    return 1
}

job_state() {  # "<STATE> <REASON>", or empty if the controller does not know the job
    scontrol show job "$1" 2>/dev/null \
        | awk '{for (i = 1; i <= NF; i++) if ($i ~ /^(JobState|Reason)=/) {sub(/^[^=]*=/, "", $i); printf "%s ", $i}}'
}

for attempt in $(seq 1 "$SUBMIT_ATTEMPTS"); do
    LOG="$RUN_DIR/attempt${attempt}.log"
    : > "$LOG"
    rm -f "$CI_EXIT_FILE"
    SBATCH_ARGS=(-p "$SLURM_PARTITION" -A "$SLURM_ACCOUNT" -q "$SLURM_QOS"
                 -N1 --exclusive -t "$TIME_LIMIT" -J "$JOB_NAME" --output "$LOG")
    [[ -n "${SLURM_EXCLUDE:-}" ]] && SBATCH_ARGS+=(--exclude="$SLURM_EXCLUDE")
    JOB_ID="$(sbatch --parsable "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/amd_ci_job_mi35x.sh")"
    JOB_ID="${JOB_ID%%;*}"
    echo "Submitted job $JOB_ID (attempt $attempt/$SUBMIT_ATTEMPTS): suite=$SUITE rocm=$ROCM_VERSION"

    LOG_OFFSET=0
    misses=0
    started=0
    while true; do
        sleep 15
        stream_log
        # The job writes its exit code last; COMPLETING can linger after that.
        [[ -f "$CI_EXIT_FILE" ]] && break
        read -r state reason <<< "$(job_state "$JOB_ID")" || true
        case "${state:-}" in
            PENDING)
                misses=0
                if [[ "${reason:-}" == JobHold* ]]; then
                    echo "Job $JOB_ID held ($reason) before starting; cancelling"
                    scancel "$JOB_ID" 2>/dev/null || true
                    break
                fi ;;
            RUNNING|CONFIGURING|COMPLETING) misses=0; started=1 ;;
            "") misses=$((misses + 1)); (( misses >= 4 )) && break ;;
            *) break ;;
        esac
    done
    if [[ "${state:-}" == PENDING ]]; then
        sleep 5          # held before starting: no exit code is coming
    else
        wait_for_exit_file || true
    fi
    stream_log           # also picks up the last lines of the log

    if [[ -s "$CI_EXIT_FILE" ]]; then
        rc="$(tr -dc 0-9 < "$CI_EXIT_FILE")"
        echo "Job $JOB_ID finished with exit code ${rc:-unknown}"
        exit "${rc:-1}"
    fi
    if (( started )) || grep -q '^\[amd-ci-job\] start' "$LOG"; then
        echo "ERROR: job $JOB_ID started but ended (${state:-gone} ${reason:-}) without writing an exit code" >&2
        exit 1
    fi
    echo "Job $JOB_ID never started (${state:-gone} ${reason:-}); resubmitting"
    JOB_ID=""
done
echo "ERROR: no job started after $SUBMIT_ATTEMPTS attempts" >&2
exit 1
