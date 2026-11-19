#!/bin/bash
# Runner job-started hook: clear what the last job left, then refuse this job and
# take the runner offline if the GPUs are unfit. The entrypoint keeps it offline
# until the health check passes again.
set -uo pipefail

# Kills leftover sglang processes; exits non-zero only if the GPUs stay dirty.
python3 /opt/sglang-ci/killall.py || true

if ! /usr/local/bin/ci-runner-healthcheck.sh; then
    echo "::error::Runner ${RUNNER_NAME:-} failed its health check and is going offline."
    # PID 1 is the runner; stopping it ends the container, and the restart policy
    # brings it back to the entrypoint's health loop.
    (sleep 5 && kill -TERM 1) &
    exit 1
fi
