#!/bin/bash
# Prepare the CI runner by cleaning up incomplete HuggingFace download files
set -euo pipefail

echo "Preparing CI runner..."

# Clean up incomplete HuggingFace download files
echo "Cleaning up incomplete HuggingFace download files..."
python3 << 'EOF'
import os
import glob

try:
    from huggingface_hub import constants
    hf_cache_dir = constants.HF_HUB_CACHE
except Exception as e:
    print(f"Warning: Could not import huggingface_hub constants: {e}")
    # Fallback to checking HF_HOME env var or default location
    hf_home = os.environ.get('HF_HOME', os.path.expanduser("~/.cache/huggingface"))
    hf_cache_dir = os.path.join(hf_home, "hub")

if os.path.exists(hf_cache_dir):
    print(f"Checking HuggingFace cache directory: {hf_cache_dir}")

    # Clean up incomplete marker files, temporary files, and lock files
    patterns = ['**/*.incomplete', '**/*.tmp', '**/*.lock']
    cleaned_count = 0

    for pattern in patterns:
        files = glob.glob(os.path.join(hf_cache_dir, pattern), recursive=True)
        for file_path in files:
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
                cleaned_count += 1
            except Exception as e:
                print(f"Warning: Could not remove {file_path}: {e}")

    if cleaned_count > 0:
        print(f"Cleaned up {cleaned_count} incomplete HuggingFace download file(s)")
    else:
        print("No incomplete HuggingFace download files found")
else:
    print(f"HuggingFace cache directory does not exist: {hf_cache_dir}")
EOF

echo "CI runner preparation complete!"
