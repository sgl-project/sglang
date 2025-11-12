#!/bin/bash
# Prepare the CI runner by cleaning up stale HuggingFace cache artifacts and validating models
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Preparing CI runner..."
echo ""

# Clean up stale HuggingFace cache artifacts from previous failed downloads
python3 "${SCRIPT_DIR}/cleanup_hf_cache.py"
echo ""

# Validate model integrity for configured runners
echo "Validating model integrity..."

# Enable accelerated HuggingFace downloads (10x faster on high-bandwidth networks)
export HF_HUB_ENABLE_HF_TRANSFER=1

python3 "${SCRIPT_DIR}/validate_and_download_models.py"
VALIDATION_EXIT_CODE=$?

if [ $VALIDATION_EXIT_CODE -ne 0 ]; then
    echo "Model validation failed with exit code: $VALIDATION_EXIT_CODE"
    exit $VALIDATION_EXIT_CODE
fi

echo ""
echo "CI runner preparation complete!"
