#!/bin/bash
# Report free disk before a CUDA CI install, and reclaim space when it runs low.
#
# Every runner: print free space on the filesystems the install and the tests
# write to, and prune unreachable uv cache entries when space is low.
# SGLANG_CI_DISK_GUARD=strict also evicts least-recently-used Hugging Face
# checkpoints and fails the job when the disk is still too full afterwards.
# Use strict only where the HF cache is local to the host: on a volume shared
# by many hosts, eviction just moves the failure onto whichever job needed the
# evicted checkpoint (see scripts/ci/amd/check_hf_cache_space.sh).

set -uo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Estimate, not a measurement: a cold install unpacks about 15 GiB of CUDA
# wheels, and one 8B bf16 checkpoint needs another 16 GiB.
TARGET_FREE_GIB=40
# Arbitrary floor; below it even an incremental install rarely finishes.
MIN_FREE_GIB=10

MODE="${SGLANG_CI_DISK_GUARD:-report}"
HF_ROOT="${HF_HOME:-${HOME}/.cache/huggingface}"
HF_HUB_DIR="${HF_HUB_CACHE:-${HF_ROOT}/hub}"

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return
    fi
    python3 -c 'import uv; print(uv.find_uv_bin())' 2>/dev/null || true
}

UV_BIN="$(find_uv)"
if [ -n "$UV_BIN" ]; then
    UV_CACHE="$("$UV_BIN" cache dir 2>/dev/null || true)"
fi
UV_CACHE="${UV_CACHE:-${UV_CACHE_DIR:-${HOME}/.cache/uv}}"

# /opt holds the kept venv (ci_install_dependency.sh). /tmp is left out: it may be
# a small tmpfs that evicting checkpoints cannot relieve.
WATCHED_PATHS=(/opt "$UV_CACHE" "$HF_HUB_DIR")

existing_ancestor() {
    local path="$1"
    while [ ! -e "$path" ]; do
        path="$(dirname "$path")"
    done
    echo "$path"
}

lowest_free_gib() {
    local lowest="" path free
    for path in "${WATCHED_PATHS[@]}"; do
        free="$(df -BG --output=avail "$(existing_ancestor "$path")" 2>/dev/null | tail -1 | tr -dc '0-9')"
        [ -n "$free" ] || continue
        if [ -z "$lowest" ] || [ "$free" -lt "$lowest" ]; then
            lowest="$free"
        fi
    done
    echo "$lowest"
}

report() {
    local path
    local existing=()
    for path in "${WATCHED_PATHS[@]}"; do
        existing+=("$(existing_ancestor "$path")")
    done
    echo "=== Disk space ($1) ==="
    # One row per filesystem: several watched paths usually share one.
    df -h "${existing[@]}" 2>/dev/null | awk 'NR == 1 || !seen[$NF]++'
}

prune_uv_cache() {
    [ -n "$UV_BIN" ] || return 0
    # Bounded so a slow prune cannot eat the install step's timeout.
    timeout 300 "$UV_BIN" cache prune || echo "WARNING: uv cache prune failed or timed out"
}

show_largest_caches() {
    echo "=== Largest cache directories ==="
    timeout 120 du -sh "$HF_ROOT" "$UV_CACHE" /opt/sglang-ci-* "${HOME}/.cache/flashinfer" \
        /tmp/sglang-ci-* 2>/dev/null | sort -rh || true
}

main() {
    echo "Disk guard mode: ${MODE}"
    report "before"

    local free
    free="$(lowest_free_gib)"
    if [ -z "$free" ]; then
        echo "WARNING: could not read free space; skipping the disk guard"
        return 0
    fi
    if [ "$free" -ge "$TARGET_FREE_GIB" ]; then
        return 0
    fi

    echo "Lowest free space is ${free} GiB, below the ${TARGET_FREE_GIB} GiB target; reclaiming space"
    prune_uv_cache
    if [ "$MODE" = "strict" ]; then
        python3 "${SCRIPT_DIR}/../utils/evict_hf_cache.py" \
            --cache-dir "$HF_HUB_DIR" \
            --target-free-gib "$TARGET_FREE_GIB" || echo "WARNING: HF cache eviction failed"
        show_largest_caches
    fi
    report "after"

    free="$(lowest_free_gib)"
    if [ "$MODE" = "strict" ] && [ -n "$free" ] && [ "$free" -lt "$MIN_FREE_GIB" ]; then
        echo "::error::Only ${free} GiB of disk is free on runner ${RUNNER_NAME:-unknown} after cleanup (need ${MIN_FREE_GIB} GiB). The disk is too small or filled by data this job cannot evict; the runner needs attention."
        return 1
    fi
    return 0
}

main "$@"
