#!/bin/bash
# Download flashinfer cubins if the local set is incomplete.
#
# The flashinfer-cubin pip package may not include cubins for newer architectures
# (e.g. sm_100, sm_120) due to PyPI size limits. This script checks the local
# cubin status against the flashinfer artifact repository and downloads any
# missing files.
#
# This script is best-effort: if the status check or download times out (e.g.
# due to a GPU in error state blocking CUDA init), we warn and continue.
# The pip package already includes cubins for common architectures (sm_80, sm_90).
set -uxo pipefail

# Early exit: the pip package already includes cubins for sm_80 and sm_90.
# Only sm_100+ (Blackwell) needs extra cubins downloaded. Skip the expensive
# Python status check entirely if no such GPU is present.
if COMPUTE_CAPS=$(timeout 10 nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null); then
    NEEDS_EXTRA_CUBINS=false
    while IFS= read -r cap; do
        major="${cap%%.*}"
        if [ "$major" -ge 10 ] 2>/dev/null; then
            NEEDS_EXTRA_CUBINS=true
            break
        fi
    done <<< "$COMPUTE_CAPS"
    if [ "$NEEDS_EXTRA_CUBINS" = false ]; then
        echo "All GPUs are sm_9x or older (compute caps: $(echo $COMPUTE_CAPS | tr '\n' ' ')), pip cubins sufficient — skipping download"
        exit 0
    fi
fi

# Use timeout to prevent hangs when GPUs are in error state (the flashinfer
# import can trigger CUDA init which blocks on bad GPUs).
CUBIN_STATUS=$(timeout 60 python3 -c "
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
from flashinfer.artifacts import get_artifacts_status
status = get_artifacts_status()
total = len(status)
downloaded = sum(1 for _, exists in status if exists)
print(f'{downloaded}/{total}')
" 2>/dev/null) || CUBIN_STATUS="unknown"

echo "Flashinfer cubin status: ${CUBIN_STATUS}"

if echo "$CUBIN_STATUS" | grep -qE '^[0-9]+/[0-9]+$'; then
    CUBIN_DOWNLOADED="${CUBIN_STATUS%/*}"
    CUBIN_TOTAL="${CUBIN_STATUS#*/}"
    if [ "$CUBIN_DOWNLOADED" = "$CUBIN_TOTAL" ] && [ "$CUBIN_TOTAL" != "0" ]; then
        echo "All flashinfer cubins already present (${CUBIN_STATUS}), skipping download"
    else
        echo "Cubins incomplete (${CUBIN_STATUS}), downloading..."
        if ! timeout 300 env FLASHINFER_LOGGING_LEVEL=warning python3 -m flashinfer --download-cubin; then
            echo "WARNING: flashinfer cubin download failed or timed out, continuing with existing cubins"
        fi
    fi
else
    echo "Could not determine cubin status (status check timed out or failed), attempting download..."
    if ! timeout 300 env FLASHINFER_LOGGING_LEVEL=warning python3 -m flashinfer --download-cubin; then
        echo "WARNING: flashinfer cubin download failed or timed out, continuing with existing cubins"
    fi
fi
