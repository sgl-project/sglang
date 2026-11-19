#!/bin/bash
# Start a self-hosted sgl-project/sglang runner; setup and upgrades are in docker/ci-runner.md.
set -euo pipefail

: "${RUNNER_NAME:?RUNNER_NAME is required, e.g. acme-gb300-host01-0123}"
: "${RUNNER_LABELS:?RUNNER_LABELS is required, e.g. 4-gpu-gb300}"
[[ "${RUNNER_NAME}" =~ ^[a-z0-9]+-[a-z0-9]+-[a-z0-9]+-[a-z0-9]+$ ]] \
    || { echo "RUNNER_NAME must be <org>-<gpu>-<host>-<gpu ids>, lowercase, e.g. acme-gb300-host01-0123" >&2; exit 1; }
[[ "${RUNNER_LABELS}" =~ ^[0-9]+-gpu-[a-z0-9]+$ ]] \
    || { echo "RUNNER_LABELS must be one <count>-gpu-<gpu> label, e.g. 4-gpu-gb300" >&2; exit 1; }

# The registration lives in this runner's own volume, so recreating the container
# on a new image reuses it and needs no new token.
STATE=/root/.cache/sglang-ci-runner
mkdir -p "${STATE}"
cd /actions-runner
for f in .runner .credentials .credentials_rsaparams; do
    ln -sfn "${STATE}/${f}" "${f}"
done

until /usr/local/bin/ci-runner-healthcheck.sh; do
    echo "Not taking jobs until the health check passes; retrying in 60s." >&2
    sleep 60
done

if [ ! -s "${STATE}/.runner" ]; then
    : "${RUNNER_TOKEN:?RUNNER_TOKEN is required for the first start}"
    # New runners only join smoke-* queues; a maintainer moves them to the real
    # label after a smoke run, and the change persists with the registration.
    ./config.sh --unattended --replace \
        --url "${RUNNER_URL:-https://github.com/sgl-project/sglang}" \
        --token "${RUNNER_TOKEN}" \
        --name "${RUNNER_NAME}" \
        --labels "smoke-${RUNNER_LABELS}" \
        --work "${STATE}/_work"
fi
# Jobs inherit this environment; the registration token must not reach PR code.
unset RUNNER_TOKEN

for model in ${PREFETCH_MODELS:-}; do
    /opt/sglang-ci-py3.12/bin/hf download "${model}"
done

exec ./run.sh
