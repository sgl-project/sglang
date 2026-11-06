#!/bin/bash
# Install zstd when missing, for any job that reads or writes an actions/cache entry.
# actions/cache identifies an entry by key *and* a version derived from the
# compression tool, so a runner without zstd cannot see what one with it saved -
# a silent miss every run. Warn, not fail: both sides work without the cache.
set -uo pipefail
if ! command -v zstd >/dev/null 2>&1; then
    if [ "$(id -u)" = "0" ]; then SUDO=""
    elif command -v sudo >/dev/null 2>&1; then SUDO="sudo"
    else SUDO=""; fi
    # Host lock shared with ci_install_dependency.sh, for runners sharing one host.
    HOST_LOCK=(flock --timeout 1800 /tmp/sglang-ci-host.lock)
    command -v flock >/dev/null 2>&1 || HOST_LOCK=()
    ${SUDO} "${HOST_LOCK[@]}" apt-get update || true
    ${SUDO} "${HOST_LOCK[@]}" apt-get install -y --no-install-recommends zstd || true
fi
command -v zstd >/dev/null 2>&1 \
    || echo "::warning::zstd unavailable on ${RUNNER_NAME:-this runner}; actions/cache entries here will not match the ones saved with it"
