#!/bin/bash
# Prepare the CI runner by cleaning up stale HuggingFace cache artifacts and validating models
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Preparing CI runner..."
echo ""

# Reap orphaned POSIX shm segments from crashed prior CI jobs. A worker that
# died with SIGBUS / SIGKILL leaves /dev/shm/* behind; over many runs these
# fill /dev/shm and cause the next scheduler's mmap to ENOSPC -> SIGBUS during
# init_ipc_channels (observed on run 26912909759 job 79396724256).
# Touch only entries owned by the current user, older than 30 minutes — well
# past any active job's IPC lifetime on this runner.
echo "==[shm cleanup]== /dev/shm BEFORE:"
df -h /dev/shm 2>/dev/null || true
find /dev/shm -maxdepth 1 -user "$(id -u)" -mmin +30 \
  \( -name 'sglang*' -o -name 'torch_*' -o -name 'pym-*' -o -name 'psm_*' -o -name 'sem.*' \) \
  -print -delete 2>/dev/null || true
echo "==[shm cleanup]== /dev/shm AFTER:"
df -h /dev/shm /tmp /root/.cache 2>/dev/null || true
echo ""

# Clean up stale HuggingFace cache artifacts from previous failed downloads
python3 "${SCRIPT_DIR}/../utils/cleanup_hf_cache.py"
echo ""

# Pre-validate cached models and write markers for offline mode
# This allows tests to run with HF_HUB_OFFLINE=1 for models that are fully cached
python3 "${SCRIPT_DIR}/../utils/prevalidate_cached_models.py"
echo ""

echo "CI runner preparation complete!"
