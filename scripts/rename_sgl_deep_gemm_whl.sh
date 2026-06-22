#!/usr/bin/env bash
# Rename a freshly-built sgl-deep-gemm wheel so its filename, METADATA Version,
# and WHEEL platform tag carry the +cuXXX local version and manylinux2014_<arch>
# tag expected by the sgl-whl index and PyPI upload step.
#
# Input:  dist/sgl_deep_gemm-<VERSION>-py3-none-any.whl
# Output: dist/sgl_deep_gemm-<VERSION>+<CU_TAG>-py3-none-manylinux2014_<ARCH>.whl
#
# Usage: rename_wheels.sh <WHEEL_DIR> <CU_TAG> <ARCH>
#   WHEEL_DIR: directory containing the *.whl file (e.g. DeepGEMM/dist)
#   CU_TAG:    cu129 | cu130
#   ARCH:      x86_64 | aarch64
set -ex

if [ $# -lt 3 ]; then
  echo "Usage: $0 <WHEEL_DIR> <CU_TAG> <ARCH>"
  exit 1
fi

WHEEL_DIR="$1"
CU_TAG="$2"
ARCH="$3"
PLAT_TAG="manylinux2014_${ARCH}"

PYTHON="${PYTHON:-python3}"
"${PYTHON}" -m pip install --quiet wheel

shopt -s nullglob
wheel_files=("${WHEEL_DIR}"/sgl_deep_gemm-*.whl)
if [ ${#wheel_files[@]} -eq 0 ]; then
  echo "No sgl_deep_gemm wheel found under ${WHEEL_DIR}" >&2
  exit 1
fi

for wheel in "${wheel_files[@]}"; do
  TMPDIR=$(mktemp -d)
  trap 'rm -rf -- "$TMPDIR"' ERR

  "${PYTHON}" -m wheel unpack "$wheel" --dest "$TMPDIR"
  UNPACKED=$(find "$TMPDIR" -mindepth 1 -maxdepth 1 -type d | head -1)
  DIST_INFO=$(find "$UNPACKED" -maxdepth 1 -type d -name "*.dist-info" | head -1)
  WHEEL_META="${DIST_INFO}/WHEEL"
  METADATA_FILE="${DIST_INFO}/METADATA"

  # Replace the py3-none-any tag with a platform-specific one.
  sed -i "s/^Tag: py3-none-any$/Tag: py3-none-${PLAT_TAG}/" "$WHEEL_META"

  ORIG_VERSION=$(grep '^Version:' "$METADATA_FILE" | head -1 | sed 's/^Version:[[:space:]]*//')
  if [[ "$ORIG_VERSION" == *"+${CU_TAG}"* ]]; then
    NEW_VERSION="$ORIG_VERSION"
  else
    NEW_VERSION="${ORIG_VERSION}+${CU_TAG}"
    sed -i "s/^Version:.*/Version: ${NEW_VERSION}/" "$METADATA_FILE"
    OLD_BASE=$(basename "$DIST_INFO")
    NEW_BASE="${OLD_BASE/${ORIG_VERSION}/${NEW_VERSION}}"
    mv "$DIST_INFO" "${UNPACKED}/${NEW_BASE}"
  fi

  rm -f "$wheel"
  "${PYTHON}" -m wheel pack "$UNPACKED" --dest-dir "$WHEEL_DIR"
  rm -rf "$TMPDIR"
  trap - ERR
done

echo "Renamed wheels in ${WHEEL_DIR}:"
ls -lh "${WHEEL_DIR}"/*.whl
