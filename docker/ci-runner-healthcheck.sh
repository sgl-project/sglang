#!/bin/bash
# Exit non-zero unless this runner's GPUs are fit to take a job: every GPU the
# label promises is visible, Blackwell, idle, and runs a small matmul.
# Memory, not process IDs, decides "idle": containers without --pid=host see none.
set -uo pipefail

PYTHON=/opt/sglang-ci-py3.12/bin/python3
# Arbitrary; well above a driver context, well below any model.
MAX_IDLE_MEMORY_MIB=2048
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-50}"

fail() {
    echo "HEALTHCHECK FAIL: $*" >&2
    exit 1
}

if ! gpus=$(nvidia-smi --query-gpu=index,compute_cap,memory.used --format=csv,noheader,nounits 2>&1); then
    fail "nvidia-smi failed: ${gpus}"
fi

# The label's leading count (4-gpu-gb300 -> 4) is the contract the scheduler relies on.
expected=$(printf '%s' "${RUNNER_LABELS:-}" | sed -n 's/^\([0-9]\+\)-gpu-.*/\1/p')
visible=$(printf '%s\n' "${gpus}" | grep -c .)
[ -n "${expected}" ] || fail "RUNNER_LABELS=${RUNNER_LABELS:-} does not start with <count>-gpu-"
[ "${visible}" -eq "${expected}" ] || fail "label promises ${expected} GPUs, ${visible} visible"

while IFS=', ' read -r index cap used; do
    # The image is only pre-installed for SM100+; others would reinstall every job.
    [ "${cap%%.*}" -ge 10 ] || fail "GPU ${index} has compute capability ${cap}; this image is Blackwell-only"
    [ "${used}" -le "${MAX_IDLE_MEMORY_MIB}" ] || fail "GPU ${index} has ${used} MiB in use; something else is running on it"
done <<< "${gpus}"

for path in /opt/sglang-ci-models /root/.cache; do
    free_gb=$(df --output=avail -BG "${path}" | tail -1 | tr -dc '0-9')
    [ "${free_gb}" -ge "${MIN_FREE_DISK_GB}" ] || fail "${path} has ${free_gb} GB free, below ${MIN_FREE_DISK_GB} GB"
done

if ! out=$(timeout 120 "${PYTHON}" -c '
import torch
for i in range(torch.cuda.device_count()):
    a = torch.ones(256, 256, device=f"cuda:{i}")
    assert torch.equal(a @ a, torch.full_like(a, 256)), f"GPU {i} returned a wrong matmul"
print(f"{torch.cuda.device_count()} GPUs passed a matmul")
' 2>&1); then
    fail "GPU compute check failed: ${out}"
fi

echo "HEALTHCHECK PASS: ${out}"
