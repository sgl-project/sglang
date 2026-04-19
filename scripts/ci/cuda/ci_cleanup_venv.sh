#!/bin/bash
# Remove the uv venv created by ci_install_dependency.sh.
#
# The venv lives at a stable path (/tmp/sglang-ci-venv) so cached JIT kernels
# keyed on include paths (deep_gemm, flashinfer) stay valid across runs.
# Freshness comes from a wipe-and-recreate at the start of each job's install
# step, not from this cleanup — so this script is purely for disk hygiene.
#
# Meant to run in a post-job workflow step with `if: always()` so the venv
# doesn't outlive a failed/cancelled job any longer than necessary.

# Best-effort cleanup: never fail the job.
set +e
set -u

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

# Target the stable venv path. Prefer SGLANG_CI_VENV_PATH (set by install),
# fall back to the hardcoded default so cleanup still works even if GITHUB_ENV
# propagation dropped the export (e.g., install crashed very early).
VENV_PATH="${SGLANG_CI_VENV_PATH:-/tmp/sglang-ci-venv}"
if [ -d "$VENV_PATH" ]; then
    if rm -rf "$VENV_PATH"; then
        echo "Cleaned up venv: $VENV_PATH"
    else
        echo "::warning::Failed to remove $VENV_PATH"
    fi
else
    echo "No venv to clean at $VENV_PATH"
fi

exit 0
