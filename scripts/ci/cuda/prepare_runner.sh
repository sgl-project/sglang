#!/bin/bash
# Prepare the CI runner by cleaning up stale HuggingFace cache artifacts
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Preparing CI runner..."
echo ""

# Clean up stale HuggingFace cache artifacts from previous failed downloads.
# No prevalidation pass here: each test run validates (and repairs) its own
# model's cache at launch/load time, which a boot-time sweep cannot substitute
# for and only duplicates.
python3 "${SCRIPT_DIR}/../utils/cleanup_hf_cache.py"
echo ""

echo "CI runner preparation complete!"
