#!/bin/bash

# Extract hashes for local releases

if [ $# -lt 1 ]; then
  echo "Usage: $0 <package_name> [version] [output_file]"
  exit 1
fi

PACKAGE_NAME=$1
VERSION=$2
OUTPUT_FILE="${3:-hash.txt}"

# Extract hashes
if [ -z "$VERSION" ]; then
  python3 $(dirname "$0")/extract_pypi_hashes.py "$PACKAGE_NAME" --output "$OUTPUT_FILE"
else
  python3 $(dirname "$0")/extract_pypi_hashes.py "$PACKAGE_NAME" --version "$VERSION" --output "$OUTPUT_FILE"
fi
