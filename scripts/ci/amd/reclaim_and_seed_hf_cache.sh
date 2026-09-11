#!/bin/bash
# Reclaim space on the shared AMD HuggingFace cache and download a large
# checkpoint into the space that was just freed, in one job.
#
# Usage (inside the ci_sglang container, where /sgl-data is the cache mount):
#   RECLAIM_DIRS="models--org--name ..." \
#   SEED_MODEL=amd/GLM-5.2-MXFP4 SEED_MIN_GIB=408 \
#     bash scripts/ci/amd/reclaim_and_seed_hf_cache.sh
#
# Env:
#   RECLAIM_DIRS  Whitespace-separated `models--org--name` directories to delete.
#                 Empty means audit only, which is the safe default.
#   SEED_MODEL    Repo id to download once the space is free. Empty means the
#                 job only reclaims.
#   SEED_MIN_GIB  Expected size of SEED_MODEL. The seed fails below it, so a
#                 truncated download cannot be mistaken for a complete one.
#   HF_HOME       Cache root, default /sgl-data/hf-cache.
#
# Why the two halves are one job: freeing space and then queueing the job that
# needs it does not work on this volume. Run 34481735602 reclaimed 1.42 TB from
# it; six hours later, when the GLM-5.2 job finally got an MI35x runner, 120 GiB
# of that was left and the download died at the same ENOSPC as before. /sgl-data
# is `amdprj3-k8s-2`, a 15 TB volume the whole AMD fleet shares with no eviction,
# and it sits at capacity, so demand absorbs anything released within hours.
# The bytes have to be claimed by the download in the same job that frees them.
#
# Why a separate job downloads weights a test would otherwise fetch itself: a
# checkpoint larger than the free space cannot be fetched in-band at all. Runs
# 34446866991, 34448628528 and 34487053976 each spent an 8-GPU MI35x slot
# discovering that. Seeding from a 1-GPU runner costs a cheap slot, and the
# consumer job then finds the checkpoint cached.

set -uo pipefail

HF_CACHE_ROOT="${HF_HOME:-/sgl-data/hf-cache}"
HUB="${HF_CACHE_ROOT}/hub"
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/null}"

RECLAIM_DIRS="${RECLAIM_DIRS:-}"
SEED_MODEL="${SEED_MODEL:-}"
SEED_MIN_GIB="${SEED_MIN_GIB:-0}"

# HuggingFace stores `org/name` under `models--org--name`.
seed_dir_name() {
    printf 'models--%s\n' "${1//\//--}"
}

avail_gib() {
    df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'
}

# Only the blobs hold real data -- snapshot entries are symlinks into them, and
# du counts a symlink's target once -- so this is the true footprint. Timed out
# because it walks a fleet-shared network volume.
dir_gib() {
    timeout 600 du -sBG "$1" 2>/dev/null | cut -f1 | tr -dc '0-9'
}

report_df() {
    echo "=== ${HUB} ($1) ==="
    df -h "$HUB" 2>/dev/null || df -h /sgl-data 2>/dev/null || true
    echo "======================================"
}

if [[ ! -d "$HUB" ]]; then
    echo "HF cache $HUB does not exist; nothing to reclaim or seed." >&2
    exit 1
fi

echo "hostname=$(hostname) runner=${RUNNER_NAME:-unknown}"
echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

# ----------------------------------------------------------------------------
# Audit. Always runs: choosing what to delete needs the sizes, and after a
# reclaim the same listing shows what the fleet refilled.
# ----------------------------------------------------------------------------
report_df "before"
echo
echo "=== /sgl-data top level ==="
timeout 900 du -sh /sgl-data/* 2>/dev/null | sort -rh || true
echo
echo "=== ${HUB} by size ==="
# Captured as an artifact so the reachability join (which checkpoints any AMD
# suite still names) can run off-runner against `amd_cache_audit.py`.
: > hf-cache-sizes.txt
timeout 1800 du -BG --max-depth=1 "$HUB" 2>/dev/null | sort -rn > hf-cache-sizes.txt
cat hf-cache-sizes.txt
: > hf-cache-listing.txt
ls -1 "$HUB" > hf-cache-listing.txt 2>/dev/null
echo
echo "$(wc -l < hf-cache-listing.txt) entries in ${HUB}"
echo

{
    echo "### AMD HF cache reclaim and seed"
    echo
    echo "- runner: \`${RUNNER_NAME:-unknown}\` (\`$(hostname)\`)"
    echo "- free before: $(avail_gib "$HUB") GiB"
} >> "$SUMMARY"

# ----------------------------------------------------------------------------
# Reclaim.
# ----------------------------------------------------------------------------
removed=0
missing=0
failed=0

if [[ -n "${RECLAIM_DIRS// /}" ]]; then
    echo "=== reclaim ==="
    {
        echo
        echo "| Checkpoint | Freed |"
        echo "| ---------- | ----- |"
    } >> "$SUMMARY"

    seed_keep=""
    if [[ -n "$SEED_MODEL" ]]; then
        seed_keep="$(seed_dir_name "$SEED_MODEL")"
    fi

    # shellcheck disable=SC2086  # RECLAIM_DIRS is a whitespace-separated list.
    for name in $RECLAIM_DIRS; do
        # An allowlist is only an allowlist if it cannot name a path. Anything
        # that is not a plain `models--org--name` under this hub is a bug in the
        # caller, and on a volume the whole fleet shares a bug here is expensive.
        if [[ ! "$name" =~ ^models--[A-Za-z0-9._-]+$ ]]; then
            echo "Refusing unsafe checkpoint name: ${name}" >&2
            exit 2
        fi
        if [[ -n "$seed_keep" && "$name" == "$seed_keep" ]]; then
            echo "Refusing to delete the seed target itself: ${name}" >&2
            exit 2
        fi

        target="${HUB}/${name}"
        lock="${HUB}/${name}.lock"
        if [[ ! -e "$target" && ! -e "$lock" ]]; then
            echo "absent   ${target}"
            echo "| \`${name}\` | absent |" >> "$SUMMARY"
            missing=$((missing + 1))
            continue
        fi

        before="$(dir_gib "$target")"
        echo "removing ${target} (${before:-?} GiB)"
        if rm -rf -- "$target" "$lock" && [[ ! -e "$target" && ! -e "$lock" ]]; then
            echo "removed  ${target}"
            echo "| \`${name}\` | ${before:-?} GiB |" >> "$SUMMARY"
            removed=$((removed + 1))
        else
            echo "FAILED   ${target}" >&2
            echo "| \`${name}\` | FAILED |" >> "$SUMMARY"
            failed=$((failed + 1))
        fi
    done

    echo
    echo "removed=${removed} missing=${missing} failed=${failed}"
    report_df "after reclaim"
    echo
else
    echo "RECLAIM_DIRS is empty: audit only, nothing deleted."
    echo >> "$SUMMARY"
    echo "Audit only; nothing deleted." >> "$SUMMARY"
fi

if (( failed != 0 )); then
    echo "Not seeding: a reclaim failed, so the free space is not what was planned." >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# Seed.
# ----------------------------------------------------------------------------
if [[ -z "$SEED_MODEL" ]]; then
    echo "SEED_MODEL is empty: nothing to download."
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
seed_target="${HUB}/$(seed_dir_name "$SEED_MODEL")"
cached_before="$(dir_gib "$seed_target")"
avail_before="$(avail_gib "$HUB")"
echo "=== seed ${SEED_MODEL} ==="
echo "already cached: ${cached_before:-0} GiB of ${SEED_MIN_GIB} GiB expected"
echo "free: ${avail_before:-?} GiB"

# A partial checkpoint is worth nothing, so a download that cannot finish is
# only a slower way to reach ENOSPC -- and it takes the rest of the volume with
# it on the way. Stop before spending any of it.
remaining=$(( SEED_MIN_GIB > ${cached_before:-0} ? SEED_MIN_GIB - ${cached_before:-0} : 0 ))
if (( remaining > 0 )) && [[ -n "${avail_before}" ]] && (( avail_before < remaining )); then
    echo "Refusing to start: ${SEED_MODEL} still needs ${remaining} GiB against" >&2
    echo "only ${avail_before} GiB free. Reclaim more before seeding." >&2
    echo >> "$SUMMARY"
    echo "Seed refused: needs ${remaining} GiB, ${avail_before} GiB free." >> "$SUMMARY"
    exit 1
fi

python3 "${SCRIPT_DIR}/seed_hf_checkpoint.py" "$SEED_MODEL"
seed_rc=$?

cached_after="$(dir_gib "$seed_target")"
report_df "after seed"
echo "${SEED_MODEL}: ${cached_before:-0} GiB -> ${cached_after:-0} GiB"
{
    echo
    echo "- seeded \`${SEED_MODEL}\`: ${cached_before:-0} GiB -> ${cached_after:-0} GiB"
    echo "- free after: $(avail_gib "$HUB") GiB"
} >> "$SUMMARY"

if (( seed_rc != 0 )); then
    echo "Seed download failed (exit ${seed_rc})." >&2
    exit "$seed_rc"
fi

# du rounds up per file, so the measured size runs a little over the nominal
# one; only a shortfall means shards are missing.
if (( SEED_MIN_GIB > 0 )) && [[ -n "${cached_after}" ]] && (( cached_after < SEED_MIN_GIB )); then
    echo "Seed incomplete: ${cached_after} GiB on disk against ${SEED_MIN_GIB} GiB expected." >&2
    exit 1
fi

echo "${SEED_MODEL} is seeded."
