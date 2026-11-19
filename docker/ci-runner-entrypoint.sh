#!/bin/bash
# Register this container as a self-hosted runner for sgl-project/sglang and run jobs.
#
#   docker run -d --restart unless-stopped --gpus all --ipc=host --cap-add SYS_PTRACE \
#     -e RUNNER_TOKEN=<registration token from a maintainer> \
#     -e RUNNER_LABELS=4-gpu-gb300 \
#     -v sglang-ci-cache:/root/.cache \
#     lmsysorg/sglang:dev-ci
#
# The named volume holds models and compiled kernels: tests download what they
# miss into it, so only the first job per model pays. Optional variables:
#   RUNNER_NAME      runner name (default: container hostname)
#   RUNNER_URL       repository URL (default: https://github.com/sgl-project/sglang)
#   PREFETCH_MODELS  space-separated Hugging Face repo IDs to download before taking jobs
#   HF_TOKEN         for gated models; jobs see it, so use a read-only token
# SYS_PTRACE is required: CUDA IPC and FlashInfer's MNNVL all-to-all share handles
# between sibling processes, which the default ptrace scope blocks.
set -euo pipefail

cd /actions-runner

# .runner survives container restarts but not recreation, so a restart reuses the
# registration and a recreated container needs a fresh token.
if [ ! -f .runner ]; then
    : "${RUNNER_TOKEN:?RUNNER_TOKEN is required to register the runner}"
    : "${RUNNER_LABELS:?RUNNER_LABELS is required, e.g. 4-gpu-gb300}"
    ./config.sh --unattended --replace \
        --url "${RUNNER_URL:-https://github.com/sgl-project/sglang}" \
        --token "${RUNNER_TOKEN}" \
        --name "${RUNNER_NAME:-$(hostname)}" \
        --labels "${RUNNER_LABELS}" \
        --work _work
fi
# Jobs inherit this environment; the registration token must not reach PR code.
unset RUNNER_TOKEN

for model in ${PREFETCH_MODELS:-}; do
    /opt/sglang-ci-py3.12/bin/hf download "${model}"
done

exec ./run.sh
