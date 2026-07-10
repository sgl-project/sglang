#!/bin/bash
# Remove the per-job uv venv created by ci_install_dependency.sh.
#
# Meant to run in a post-job workflow step with `if: always()` so the venv is
# destroyed even on job failure/cancel. Runner-level safety net: a cron or
# startup task should also purge stale /tmp/sglang-ci-* directories to catch
# cancelled or crashed jobs that never reached this cleanup.

# Best-effort cleanup: never fail the job.
set +e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Reclaim leaked sglang /dev/shm segments at teardown. SGLang processes are torn
# down with SIGKILL, which skips their unlink paths, so sgl_shm_* /
# multi_tokenizer_args_* segments accumulate and eventually fill /dev/shm — the
# next scheduler then SIGBUSes at init ("Fatal Python error: Bus error",
# scheduler died exit code -7).
#
# Reuse the canonical sweep (stale_shm_cleanup.py, the same one ci_install_
# dependency.sh runs at job startup) rather than a flat age delete: it is
# name-scoped and removes a segment only if its embedded creator pid is dead. So
# a concurrent or long-running job's live segments — and any non-job files the
# CI user owns (the runner agent, daemons, reparented supervisors) — are never
# touched, and there is no age race to re-trigger the very SIGBUS this guards
# against. Runs independent of venv mode (before the USE_VENV early-exit below).
# Best effort: never fails the job.
SHM_CLEANUP="${REPO_ROOT}/python/sglang/srt/utils/stale_shm_cleanup.py"
if [ -f "$SHM_CLEANUP" ]; then
    SGLANG_IS_IN_CI=true python3 "$SHM_CLEANUP" || true
fi

# Skip entirely when venv mode is disabled — no /tmp/sglang-ci-* dir exists
# and there's nothing to sweep. Matches the USE_VENV parsing in
# ci_install_dependency.sh (accepts 1/true/yes, case-insensitive).
USE_VENV_RAW="${USE_VENV:-true}"
case "$(printf '%s' "$USE_VENV_RAW" | tr '[:upper:]' '[:lower:]')" in
    1 | true | yes) ;;
    *)
        echo "USE_VENV=${USE_VENV_RAW}: skipping venv cleanup"
        exit 0
        ;;
esac

# Prefer the path propagated via GITHUB_ENV. Fallback: glob for any venv from
# this run+job (covers the case where install crashed before exporting the path).
if [ -n "${SGLANG_CI_VENV_PATH:-}" ] && [ -d "$SGLANG_CI_VENV_PATH" ]; then
    if rm -rf "$SGLANG_CI_VENV_PATH"; then
        echo "Cleaned up venv: $SGLANG_CI_VENV_PATH"
    else
        echo "::warning::Failed to remove $SGLANG_CI_VENV_PATH — runner cron should sweep /tmp/sglang-ci-*"
    fi
else
    matched=0
    for venv in /tmp/sglang-ci-${GITHUB_RUN_ID:-unknownrun}-${GITHUB_JOB:-unknownjob}-*; do
        [ -d "$venv" ] || continue
        matched=1
        if rm -rf "$venv"; then
            echo "Cleaned up venv (via glob): $venv"
        else
            echo "::warning::Failed to remove $venv — runner cron should sweep /tmp/sglang-ci-*"
        fi
    done
    [ "$matched" -eq 0 ] && echo "No venv to clean for run=${GITHUB_RUN_ID:-?} job=${GITHUB_JOB:-?}"
fi

# Sweep stale venvs from cancelled/crashed jobs that never reached cleanup.
# Any /tmp/sglang-ci-* dir older than 4 hours is considered orphaned.
stale_count=0
for venv in /tmp/sglang-ci-*; do
    [ -d "$venv" ] || continue
    if find "$venv" -maxdepth 0 -mmin +240 -print -quit | grep -q .; then
        rm -rf "$venv" && stale_count=$((stale_count + 1))
    fi
done
[ "$stale_count" -gt 0 ] && echo "Swept $stale_count stale venv(s) older than 4h"

exit 0
