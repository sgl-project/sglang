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

# Matches the USE_VENV parsing in ci_install_dependency.sh (accepts
# 1/true/yes, case-insensitive).
USE_VENV_RAW="${USE_VENV:-true}"
case "$(printf '%s' "$USE_VENV_RAW" | tr '[:upper:]' '[:lower:]')" in
    1 | true | yes) ;;
    *)
        # USE_VENV=false path: there is no /tmp/sglang-ci-* venv to drop,
        # but system site-packages can carry over stale trees between jobs.
        # Observed: `uv pip uninstall flashinfer-python` leaves flashinfer/data/
        # behind because flashinfer-cubin owns files beneath it, so the next
        # job's `uv pip install -e python[...]` fails with
        # "failed to create directory flashinfer/data/: File exists (os error 17)".
        # See https://github.com/sgl-project/sglang/actions/runs/24634237642/job/72027123887
        #
        # Fully uninstall the flashinfer trio and rm -rf any residual package
        # dirs so the next setup starts from a clean slate. Cached wheels
        # under ~/.cache/flashinfer-wheels/ keep the reinstall fast.
        echo "USE_VENV=${USE_VENV_RAW}: purging flashinfer leftovers from system site-packages"
        python3 -m pip uninstall -y \
            flashinfer-python flashinfer-cubin flashinfer-jit-cache \
            >/dev/null 2>&1 || true

        SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)"
        if [ -n "${SITE_PACKAGES:-}" ] && [ -d "$SITE_PACKAGES" ]; then
            for pkg in flashinfer flashinfer_cubin flashinfer_jit_cache; do
                stale="${SITE_PACKAGES}/${pkg}"
                if [ -e "$stale" ] || [ -L "$stale" ]; then
                    if rm -rf "$stale"; then
                        echo "Purged ${stale}"
                    else
                        echo "::warning::Failed to remove ${stale}"
                    fi
                fi
            done
        fi
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
