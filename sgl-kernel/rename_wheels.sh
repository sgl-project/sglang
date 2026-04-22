#!/usr/bin/env bash
# Align CUDA wheel filenames (+cu124/+cu128/+cu130) with internal METADATA Version and
# WHEEL tags after build (fixes pip "inconsistent version" when only the .whl name changed).
# Unpack → patch WHEEL/METADATA → wheel pack (RECORD regenerated; no hand-editing).
set -ex

WHEEL_DIR="dist"

detect_cuda_suffix() {
    if ls /usr/local/ 2>/dev/null | grep -q "12.4"; then
        echo "+cu124"
    elif ls /usr/local/ 2>/dev/null | grep -q "12.8"; then
        echo "+cu128"
    elif ls /usr/local/ 2>/dev/null | grep -q "13.0"; then
        echo "+cu130"
    else
        echo ""
    fi
}

CUDA_SUFFIX=$(detect_cuda_suffix)

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

    if [[ -z "$CUDA_SUFFIX" ]]; then
        continue
    fi

    TMPDIR=$(mktemp -d)
    trap 'rm -rf -- "$TMPDIR"' ERR

    "${PYTHON:-python3}" -m wheel unpack "$wheel" --dest "$TMPDIR"
    UNPACKED=$(find "$TMPDIR" -mindepth 1 -maxdepth 1 -type d | head -1)
    DIST_INFO=$(find "$UNPACKED" -maxdepth 1 -type d -name "*.dist-info" | head -1)
    WHEEL_META="${DIST_INFO}/WHEEL"
    METADATA_FILE="${DIST_INFO}/METADATA"

    patch_wheel_platform_tags "$WHEEL_META"

    ORIG_VERSION=$(grep '^Version:' "$METADATA_FILE" | head -1 | sed 's/^Version:[[:space:]]*//')
    if [[ "$ORIG_VERSION" == *"$CUDA_SUFFIX"* ]]; then
        echo "Skipping $wheel: version in METADATA is already suffixed."
        rm -rf "$TMPDIR"
        trap - ERR
        continue
    fi
    NEW_VERSION="${ORIG_VERSION}${CUDA_SUFFIX}"
    sed -i "s/^Version:.*/Version: ${NEW_VERSION}/" "$METADATA_FILE"

    OLD_BASE=$(basename "$DIST_INFO")
    NEW_BASE="${OLD_BASE/${ORIG_VERSION}/${NEW_VERSION}}"
    mv "$DIST_INFO" "${UNPACKED}/${NEW_BASE}"

    rm -f "$wheel"
    "${PYTHON:-python3}" -m wheel pack "$UNPACKED" --dest-dir "$WHEEL_DIR"
    rm -rf "$TMPDIR"
    trap - ERR
done
echo "Wheel renaming completed."
