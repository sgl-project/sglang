#!/bin/bash
# Report HuggingFace cache headroom before a large checkpoint is used, and
# clear stale download artifacts.
#
# Usage (inside the ci_sglang container, where /sgl-data is the cache mount):
#   check_hf_cache_space.sh <model_repo_id> [required_gib] [--strict]
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
# Warn-only by default: a full cache is not necessarily fatal (the checkpoint may
# already be cached, which is the common case), and when it is fatal the
# download says so itself -- now against a log that already explained why.
#
# `--strict` makes a predicted shortfall fail the job instead. Warning and
# proceeding is only harmless when the download's failure is confined to this
# job, and for a checkpoint far larger than the free space it is not: run
# 34487053976 warned that amd/GLM-5.2-MXFP4 needed 314 GiB against 120 GiB free,
# proceeded anyway, and spent eleven minutes turning that 120 GiB into a
# checkpoint that was still incomplete -- ending with 832 MB free on a volume the
# whole AMD fleet shares. Every byte it consumed came out of some other job's
# headroom, and it could not have finished no matter how long it ran. Pass
# --strict when the caller knows the download is all-or-nothing, so the job stops
# at the check instead of draining the volume on the way to the same failure.
#
# "Cached" is decided by bytes on disk, not by the model directory existing. An
# ENOSPC failure leaves a partial checkpoint behind, so the directory test alone
# reports the next run's 300 GiB shortfall as "already cached, no download
# needed" and suppresses the one warning that run needs.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STRICT=0
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --strict) STRICT=1 ;;
        # An empty positional is how a caller says "no expected size".
        "") ;;
        -*)
            echo "Unknown argument: $arg (expected --strict)" >&2
            exit 2
            ;;
        *) ARGS+=("$arg") ;;
    esac
done

MODEL_REPO_ID="${ARGS[0]:?model repo id, e.g. amd/Qwen3.8-2.4T-A95B-Quark-MXFP4}"
REQUIRED_GIB="${ARGS[1]:-0}"
if [[ ! "$REQUIRED_GIB" =~ ^[0-9]+$ ]]; then
    echo "Expected size in GiB to be a whole number, got: $REQUIRED_GIB" >&2
    exit 2
fi

HF_CACHE="${HF_HOME:-/sgl-data/hf-cache}/hub"
# HuggingFace stores `org/name` as `models--org--name`.
MODEL_DIR="$HF_CACHE/models--${MODEL_REPO_ID//\//--}"

avail_gib() {
    df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'
}

# Only the blobs hold real data -- snapshot entries are symlinks into them, and
# du counts a symlink's target once -- so this is the checkpoint's true
# footprint. Timed out because it walks a fleet-shared network volume.
cached_gib() {
    timeout 300 du -sBG "$1" 2>/dev/null | cut -f1 | tr -dc '0-9'
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

    local cached=0
    if [[ -d "$MODEL_DIR" ]]; then
        cached=$(cached_gib "$MODEL_DIR")
        # A size we cannot read is indistinguishable from a complete download,
        # so assume the common case rather than warn about a download that may
        # never be attempted.
        if [[ -z "$cached" ]]; then
            echo "✓ ${MODEL_REPO_ID} is cached at ${MODEL_DIR}, size unreadable;" \
                 "assuming it is complete."
            return 0
        fi
    fi

    local remaining=$(( REQUIRED_GIB > cached ? REQUIRED_GIB - cached : 0 ))

    if (( cached == 0 )); then
        echo "${MODEL_REPO_ID} is NOT cached; it must be downloaded."
    elif (( remaining == 0 )); then
        echo "✓ ${MODEL_REPO_ID} is already cached at ${MODEL_DIR} (${cached} GiB);" \
             "no download needed regardless of free space."
    else
        echo "${MODEL_REPO_ID} is only PARTIALLY cached at ${MODEL_DIR}:" \
             "${cached} GiB of roughly ${REQUIRED_GIB} GiB, so about ${remaining} GiB" \
             "still has to be downloaded."
    fi

    if (( remaining == 0 )) || (( avail >= remaining )); then
        return 0
    fi

    echo "=============================================================="
    echo "WARNING: ${MODEL_REPO_ID} still needs roughly ${remaining} GiB against"
    echo "         only ${avail} GiB free. The download will likely fail with"
    echo "         ENOSPC partway through."
    echo ""
    echo "         /sgl-data is shared by the whole AMD fleet, so this is a"
    echo "         capacity problem rather than something this job can clear:"
    echo "         deleting other checkpoints to make room just moves the"
    echo "         failure onto whichever job needed them next. Raising it"
    echo "         needs the runner owners."
    echo "=============================================================="

    if (( STRICT )); then
        echo "Stopping here (--strict). Starting the download would spend the"
        echo "remaining ${avail} GiB on a checkpoint that would still be"
        echo "incomplete, leaving the shared volume with nothing for any other"
        echo "job and failing anyway."
        return 1
    fi
    return 0
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    check_hf_cache_space "$@"
fi
