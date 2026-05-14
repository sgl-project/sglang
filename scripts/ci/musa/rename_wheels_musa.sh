#!/usr/bin/env bash
set -euo pipefail

# Rename MUSA wheels to include a +musa<suffix> build tag.
# Usage:
#   rename_wheels_musa.sh <musa_suffix> [wheel_dir]
# Example:
#   rename_wheels_musa.sh 43 sgl-kernel/dist

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <musa_suffix> [wheel_dir]" >&2
  exit 1
fi

MUSA_SUFFIX="$1"
WHEEL_DIR="${2:-dist}"

wheel_files=("$WHEEL_DIR"/*.whl)

if [[ ! -e "${wheel_files[0]}" ]]; then
  echo "No wheel files found in ${WHEEL_DIR}/, nothing to rename."
  exit 0
fi

for wheel in "${wheel_files[@]}"; do
  # Normalize platform tag to manylinux2014
  intermediate_wheel="${wheel/linux/manylinux2014}"

  # Extract Python ABI version (e.g. cp310)
  if [[ $intermediate_wheel =~ -cp([0-9]+)- ]]; then
    cp_version="${BASH_REMATCH[1]}"
  else
    echo "Could not extract Python version from wheel name: $intermediate_wheel" >&2
    continue
  fi

  # Insert +musa<suffix> before the Python ABI tag
  new_wheel="${intermediate_wheel/-cp${cp_version}/+musa${MUSA_SUFFIX}-cp${cp_version}}"

  if [[ "$wheel" != "$new_wheel" ]]; then
    echo "Renaming $wheel -> $new_wheel"
    mv -- "$wheel" "$new_wheel"
  fi
done

echo "MUSA wheel renaming completed."
