#!/bin/bash
# Prepare the CI runner by cleaning up stale HuggingFace cache artifacts and validating models
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Preparing CI runner..."
echo ""

# Clean up stale HuggingFace cache artifacts from previous failed downloads
python3 "${SCRIPT_DIR}/../utils/cleanup_hf_cache.py"
echo ""

# Pre-validate cached models and write markers for offline mode
# This allows tests to run with HF_HUB_OFFLINE=1 for models that are fully cached
python3 "${SCRIPT_DIR}/../utils/prevalidate_cached_models.py"
echo ""

# Validate weight checksums for critical models to detect bit-flip corruption
# This catches corruption that safetensors header validation might miss
python3 "${SCRIPT_DIR}/../utils/validate_weight_checksums.py"
echo ""

echo "CI runner preparation complete!"
