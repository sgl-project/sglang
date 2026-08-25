#!/bin/bash
# Report HuggingFace cache headroom before a large checkpoint is used, and
# clear stale download artifacts.
#
# Usage (inside the ci_sglang container, where /sgl-data is the cache mount):
#   check_hf_cache_space.sh <model_repo_id> [required_gib]
#
# Why this exists: run 32196787596 died 40 minutes into a 1.2 TB download with
# "OSError: [Errno 28] No space left on device", and the only way to find that
# out was reading 8,500 lines of job log -- free space was never reported
# anywhere. This puts the number in the log every time, before the download
# rather than after it fails.
#
# What it deliberately does NOT do is free space by deleting other checkpoints.
# /sgl-data is not a per-runner disk: it is `amdprj3-k8s-2`, a 15 TB volume
# shared by the whole AMD fleet, and it sits at 100% used. An earlier version of
# this script evicted least-recently-used checkpoints until it hit a free-space
# target; on that filesystem it removed 48 of them and free space went from
# 298 MB to 227 MB, because concurrent jobs consume anything released as fast as
# it appears. So the eviction destroyed other jobs' caches fleet-wide, forcing
# them to re-download, and bought nothing. A volume at capacity is an
# infrastructure problem and a per-job script cannot fix it by deleting things
# other jobs still need.
#
# Never fails the job: a full cache is not necessarily fatal (the checkpoint may
# already be cached, which is the common case), and when it is fatal the
# download says so itself -- now against a log that already explained why.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_REPO_ID="${1:?model repo id, e.g. amd/Qwen3.8-2.4T-A95B-Quark-MXFP4}"
REQUIRED_GIB="${2:-0}"

HF_CACHE="${HF_HOME:-/sgl-data/hf-cache}/hub"
# HuggingFace stores `org/name` as `models--org--name`.
MODEL_DIR="$HF_CACHE/models--${MODEL_REPO_ID//\//--}"

avail_gib() {
    df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'
}

report() {
    echo "=== HF cache space ($1) ==="
    df -h "$HF_CACHE" 2>/dev/null || df -h /sgl-data 2>/dev/null || true
    echo "==========================="
}

check_hf_cache_space() {
    if [[ ! -d "$HF_CACHE" ]]; then
        echo "HF cache $HF_CACHE does not exist yet; nothing to report."
        return 0
    fi

    report "before"

    if [[ -d "$MODEL_DIR" ]]; then
        echo "✓ ${MODEL_REPO_ID} is already cached at ${MODEL_DIR};" \
             "no download needed regardless of free space."
    else
        echo "${MODEL_REPO_ID} is NOT cached; it must be downloaded."
    fi

    # Abandoned partial downloads are pure waste and safe to drop. This is the
    # shared helper the CUDA runner prep already uses; it only touches
    # *.incomplete / *.tmp older than two hours, so it cannot pull the rug from
    # under a download running right now.
    python3 "${SCRIPT_DIR}/../utils/cleanup_hf_cache.py" || true

    report "after"

    local avail
    avail=$(avail_gib "$HF_CACHE")
    if [[ -z "$avail" ]]; then
        echo "WARNING: could not read free space from df."
        return 0
    fi
    echo "Free space: ${avail} GiB."

    if [[ -d "$MODEL_DIR" ]] || (( REQUIRED_GIB == 0 )) || (( avail >= REQUIRED_GIB )); then
        return 0
    fi

    echo "=============================================================="
    echo "WARNING: ${MODEL_REPO_ID} is not cached and only ${avail} GiB is"
    echo "         free, against roughly ${REQUIRED_GIB} GiB of weights. The"
    echo "         download will likely fail with ENOSPC partway through."
    echo ""
    echo "         /sgl-data is shared by the whole AMD fleet, so this is a"
    echo "         capacity problem rather than something this job can clear:"
    echo "         deleting other checkpoints to make room just moves the"
    echo "         failure onto whichever job needed them next. Raising it"
    echo "         needs the runner owners."
    echo "=============================================================="
    return 0
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    check_hf_cache_space "$@"
fi
