#!/usr/bin/env bash
# Retag a freshly-built sgl-fa3 wheel for the sgl-whl index.
#
# scikit-build-core tags the wheel `cp310-abi3-linux_x86_64`. `linux_x86_64` is
# not installable from an index -- pip and PyPI both reject it -- so the
# platform becomes manylinux2014_<arch>, and the version gains the +cuXXX local
# segment the index keys on. The abi3 tag is left alone.
#
# Input:  <WHEEL_DIR>/sgl_fa3-<VERSION>-cp310-abi3-linux_<ARCH>.whl
# Output: <WHEEL_DIR>/sgl_fa3-<VERSION>+<CU_TAG>-cp310-abi3-manylinux2014_<ARCH>.whl
#
# Usage: rename_sgl_fa3_whl.sh <WHEEL_DIR> <CU_TAG> <ARCH>
# (sgl_fa3/scripts/build.sh in sgl-project/sgl-flash-attn runs it inside the
# build container, mounted from this checkout.)
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
wheel_files=("${WHEEL_DIR}"/sgl_fa3-*.whl)
if [ ${#wheel_files[@]} -eq 0 ]; then
  echo "No sgl_fa3 wheel found under ${WHEEL_DIR}" >&2
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

  # Only the platform part of the tag changes; the interpreter/ABI part stays abi3.
  sed -i -E "s/^(Tag: [^-]+-[^-]+)-linux_${ARCH}$/\1-${PLAT_TAG}/" "$WHEEL_META"
  grep -q "^Tag: .*-${PLAT_TAG}$" "$WHEEL_META" || {
    echo "Failed to retag ${wheel}; WHEEL says:" >&2
    cat "$WHEEL_META" >&2
    exit 1
  }

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
