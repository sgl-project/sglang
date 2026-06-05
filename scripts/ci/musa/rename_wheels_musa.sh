#!/usr/bin/env bash
# Align MUSA wheel filenames (+musa43/...) with internal METADATA Version and
# WHEEL tags after build. Two drifts need fixing in lockstep:
#   - METADATA `Version:` must carry the `+musa<suffix>` local version, or
#     recent pip versions reject the wheel with "inconsistent version".
#   - WHEEL `Tag:` must be `manylinux2014_*` when the filename says so;
#     leaving it as `linux_*` can trip installers that re-derive the platform.
# Unpack → patch WHEEL/METADATA → wheel pack (RECORD regenerated; no hand-editing).
#
# Usage:
#   rename_wheels_musa.sh <musa_suffix> [wheel_dir]
# Example:
#   rename_wheels_musa.sh 43 sgl-kernel/dist
set -euxo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <musa_suffix> [wheel_dir]" >&2
  exit 1
fi

MUSA_SUFFIX="+musa$1"
WHEEL_DIR="${2:-dist}"

patch_wheel_platform_tags() {
    local wheel_file="$1"
    # Line-end anchors: "linux_x86_64" is a substring of "manylinux2014_x86_64", so
    # unanchored global replace corrupts tags on a second run.
    sed -i \
        -e 's/-linux_x86_64$/-manylinux2014_x86_64/' \
        -e 's/-linux_aarch64$/-manylinux2014_aarch64/' \
        "$wheel_file"
}

wheel_files=("$WHEEL_DIR"/*.whl)
if [[ ! -e "${wheel_files[0]}" ]]; then
  echo "No wheel files found in ${WHEEL_DIR}/, nothing to rename."
  exit 0
fi

for wheel in "${wheel_files[@]}"; do
    [[ -f "$wheel" ]] || continue

    intermediate_wheel="$wheel"
    case "$wheel" in
        *-linux_x86_64.whl)
            intermediate_wheel="${wheel%-linux_x86_64.whl}-manylinux2014_x86_64.whl"
            ;;
        *-linux_aarch64.whl)
            intermediate_wheel="${wheel%-linux_aarch64.whl}-manylinux2014_aarch64.whl"
            ;;
    esac
    if [[ "$wheel" != "$intermediate_wheel" ]]; then
        mv -- "$wheel" "$intermediate_wheel"
        wheel="$intermediate_wheel"
    fi

    TMPDIR=$(mktemp -d)
    trap 'rm -rf -- "$TMPDIR"' ERR

    "${PYTHON:-python3}" -m wheel unpack "$wheel" --dest "$TMPDIR"
    # `find | head -1` succeeds with empty stdout when there are no matches —
    # `set -e` won't catch that. Assert each path is real so a malformed wheel
    # surfaces with a useful message instead of a downstream `sed: /WHEEL` error.
    UNPACKED=$(find "$TMPDIR" -mindepth 1 -maxdepth 1 -type d | head -1)
    [[ -d "$UNPACKED" ]] || { echo "ERROR: wheel unpack produced no top-level dir for $wheel" >&2; exit 1; }
    DIST_INFO=$(find "$UNPACKED" -maxdepth 1 -type d -name "*.dist-info" | head -1)
    [[ -d "$DIST_INFO" ]] || { echo "ERROR: no *.dist-info under $UNPACKED (malformed wheel?): $wheel" >&2; exit 1; }
    WHEEL_META="${DIST_INFO}/WHEEL"
    METADATA_FILE="${DIST_INFO}/METADATA"
    [[ -f "$WHEEL_META" && -f "$METADATA_FILE" ]] || { echo "ERROR: missing WHEEL or METADATA in $DIST_INFO" >&2; exit 1; }

    patch_wheel_platform_tags "$WHEEL_META"

    ORIG_VERSION=$(grep '^Version:' "$METADATA_FILE" | head -1 | sed 's/^Version:[[:space:]]*//')
    # Empty ORIG_VERSION would fall through the `+musa` check below and silently
    # produce `Version: +musa43` — a broken release. Fail loud instead.
    [[ -n "$ORIG_VERSION" ]] || { echo "ERROR: no 'Version:' line in $METADATA_FILE" >&2; exit 1; }
    if [[ "$ORIG_VERSION" == *"$MUSA_SUFFIX"* ]]; then
        echo "Skipping $wheel: version in METADATA is already suffixed."
        rm -rf "$TMPDIR"
        trap - ERR
        continue
    fi
    NEW_VERSION="${ORIG_VERSION}${MUSA_SUFFIX}"
    sed -i "s/^Version:.*/Version: ${NEW_VERSION}/" "$METADATA_FILE"
    # `sed -i` exits 0 even when the pattern matched zero lines. Verify the
    # rewrite actually landed before we publish.
    grep -qx "Version: ${NEW_VERSION}" "$METADATA_FILE" || { echo "ERROR: METADATA Version rewrite did not land in $METADATA_FILE" >&2; exit 1; }

    OLD_BASE=$(basename "$DIST_INFO")
    NEW_BASE="${OLD_BASE/${ORIG_VERSION}/${NEW_VERSION}}"
    # `${var/pat/repl}` silently leaves var unchanged if pat is empty or absent.
    [[ "$NEW_BASE" != "$OLD_BASE" ]] || { echo "ERROR: dist-info dir '$OLD_BASE' did not contain ORIG_VERSION='$ORIG_VERSION'" >&2; exit 1; }
    mv "$DIST_INFO" "${UNPACKED}/${NEW_BASE}"

    rm -f "$wheel"
    "${PYTHON:-python3}" -m wheel pack "$UNPACKED" --dest-dir "$WHEEL_DIR"
    rm -rf "$TMPDIR"
    trap - ERR
done

echo "MUSA wheel renaming completed."
