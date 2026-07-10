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

# Sweep stale /dev/shm files left by crashed/cancelled/leaky jobs. Runs
# independent of venv mode (before the USE_VENV early-exit below). Leaked shm —
# loky pool semaphores (sem.loky-*), sglang shm segments (sglang_loads_*),
# cuda.shm.* — accumulates across jobs and, on a runner with a small --shm-size,
# eventually fills /dev/shm so the next scheduler SIGBUSes at init ("Fatal
# Python error: Bus error", scheduler died exit code -7).
#
# A flat age floor alone is unsafe on a shared runner: /dev/shm is host-wide, so
# a concurrent job — or a long-running one (nightly can exceed 4h) — that mapped
# a segment at startup would have that still-live segment unlinked once its mtime
# (set at creation, never on access) crosses the floor, re-triggering the exact
# SIGBUS this guards against. So a file is deleted only if it is BOTH older than
# 4h AND not currently mapped/opened by any process. The in-use scan reads
# /proc/*/maps and /proc/*/fd (runner runs as root); a file we can't prove is
# free is kept. Only files actually unlinked are counted (EPERM etc. don't
# inflate the total), so the summary never claims a cleanup that didn't happen.
if [ -d /dev/shm ]; then
    shm_swept=$(python3 - <<'PY'
import glob, os, time
SHM = "/dev/shm"
FLOOR = 4 * 3600
now = time.time()

# Paths of /dev/shm segments any live process still has mapped or open.
in_use = set()
for maps in glob.glob("/proc/[0-9]*/maps"):
    try:
        with open(maps) as fh:
            for line in fh:
                if "/dev/shm/" in line:
                    in_use.add(line.rstrip("\n").split(" ", 5)[-1])
    except OSError:
        pass
for fd in glob.glob("/proc/[0-9]*/fd/*"):
    try:
        tgt = os.readlink(fd)
    except OSError:
        continue
    if tgt.startswith("/dev/shm/"):
        in_use.add(tgt)

swept = 0
for path in glob.glob(os.path.join(SHM, "*")):
    if not os.path.isfile(path) or path in in_use:
        continue
    try:
        if now - os.path.getmtime(path) < FLOOR:
            continue
        os.remove(path)          # count only on a real unlink
        swept += 1
    except OSError:
        pass                     # gone already, or not ours (EPERM) — skip
print(swept)
PY
)
    [ "${shm_swept:-0}" -gt 0 ] && echo "Swept $shm_swept stale /dev/shm file(s) (>4h, not in use)"
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
