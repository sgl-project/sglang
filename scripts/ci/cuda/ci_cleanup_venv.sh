#!/bin/bash
# Disk hygiene for the uv venv created by ci_install_dependency.sh.
# Freshness is the install side's job (wipe-recreate); this just frees /tmp
# after each job. Run as a post-job step with `if: always()`.

# Best-effort cleanup: never fail the job.
set +e
set -u

# Same parse as ci_install_dependency.sh.
USE_VENV_RAW="${USE_VENV:-true}"
case "$(printf '%s' "$USE_VENV_RAW" | tr '[:upper:]' '[:lower:]')" in
    1 | true | yes) ;;
    *)
        echo "USE_VENV=${USE_VENV_RAW}: skipping venv cleanup"
        exit 0
        ;;
esac

# Fall back to the hardcoded default if the install step crashed before
# exporting SGLANG_CI_VENV_PATH via GITHUB_ENV.
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
