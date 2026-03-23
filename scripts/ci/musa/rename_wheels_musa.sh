#!/usr/bin/env bash
set -euo pipefail

# Rename MUSA wheels to include a +musa<suffix> build tag.
# Usage:
#   rename_wheels_musa.sh <musa_suffix> [wheel_dir]
# Example:
#   rename_wheels_musa.sh 43 sgl-kernel/dist
#
# Idempotency contract (same rules as sgl-kernel/rename_wheels.sh):
#   - Platform suffix: only rewrite exact PEP 427 end patterns (*-linux_x86_64.whl /
#     *-linux_aarch64.whl). Do NOT use ${path/linux/manylinux2014} — "linux" is a
#     substring of "manylinux2014", so repeated runs produce manymanylinux20142014.
#   - +musa suffix: skip wheels whose filename already contains "+musa", so a second
#     run is a no-op instead of inserting +musa+musa.

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
  base="$(basename "$wheel")"

  # Idempotency guard: skip if +musa suffix already present in filename.
  if [[ "$base" == *"+musa"* ]]; then
    echo "Skipping $base: already has +musa suffix."
    continue
  fi

  # Normalize platform suffix (PEP 427 end patterns only — not substring replace).
  intermediate_wheel="$wheel"
  case "$wheel" in
    *-linux_x86_64.whl)
      intermediate_wheel="${wheel%-linux_x86_64.whl}-manylinux2014_x86_64.whl" ;;
    *-linux_aarch64.whl)
      intermediate_wheel="${wheel%-linux_aarch64.whl}-manylinux2014_aarch64.whl" ;;
  esac
  if [[ "$wheel" != "$intermediate_wheel" ]]; then
    mv -- "$wheel" "$intermediate_wheel"
    wheel="$intermediate_wheel"
  fi

  # Extract Python ABI version (e.g. cp310) from the (possibly renamed) path.
  if [[ "$wheel" =~ -cp([0-9]+)- ]]; then
    cp_version="${BASH_REMATCH[1]}"
  else
    echo "Could not extract Python version from wheel name: $(basename "$wheel")" >&2
    continue
  fi

  # Insert +musa<suffix> before the Python ABI tag.
  new_wheel="${wheel/-cp${cp_version}/+musa${MUSA_SUFFIX}-cp${cp_version}}"

  if [[ "$wheel" != "$new_wheel" ]]; then
    echo "Renaming $(basename "$wheel") -> $(basename "$new_wheel")"
    mv -- "$wheel" "$new_wheel"
  fi
done

echo "MUSA wheel renaming completed."
