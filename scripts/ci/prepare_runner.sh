#!/bin/bash
# Prepare the CI runner by cleaning up stale HuggingFace cache artifacts and validating models
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Preparing CI runner..."
echo ""

# Clean up stale HuggingFace cache artifacts from previous failed downloads
python3 "${SCRIPT_DIR}/cleanup_hf_cache.py"
echo ""

echo "CI runner preparation complete!"
