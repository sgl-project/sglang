#!/bin/bash
# Download flashinfer cubins if the local set is incomplete.
#
# The flashinfer-cubin pip package may not include cubins for newer architectures
# (e.g. sm_100, sm_120) due to PyPI size limits. This script checks the local
# cubin status against the flashinfer artifact repository and downloads any
# missing files.
set -euxo pipefail

CUBIN_STATUS=$(FLASHINFER_LOGGING_LEVEL=warning python3 -c "
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
        FLASHINFER_LOGGING_LEVEL=warning python3 -m flashinfer --download-cubin
    fi
else
    echo "Could not determine cubin status, downloading as fallback..."
    FLASHINFER_LOGGING_LEVEL=warning python3 -m flashinfer --download-cubin
fi
