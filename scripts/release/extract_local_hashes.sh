#!/bin/bash

# Extract hashes for local releases

if [ $# -lt 1 ]; then
  echo "Usage: $0 <package_name> [version] [output_file]"
  exit 1
fi

PACKAGE_NAME=$1
VERSION=$2
OUTPUT_FILE="${3:-hash.txt}"

# Try to calculate hashes locally first
if [ -z "$VERSION" ]; then
  python3 $(dirname "$0")/calculate_local_hashes.py "$PACKAGE_NAME" --output "$OUTPUT_FILE"
  
  # If local calculation fails, extract from PyPI as fallback
  if [ $? -ne 0 ]; then
    python3 $(dirname "$0")/extract_pypi_hashes.py "$PACKAGE_NAME" --output "$OUTPUT_FILE" --wait
  fi
else
  python3 $(dirname "$0")/calculate_local_hashes.py "$PACKAGE_NAME" --version "$VERSION" --output "$OUTPUT_FILE"
  
  # If local calculation fails, extract from PyPI as fallback
  if [ $? -ne 0 ]; then
    python3 $(dirname "$0")/extract_pypi_hashes.py "$PACKAGE_NAME" --version "$VERSION" --output "$OUTPUT_FILE" --wait
  fi
fi
