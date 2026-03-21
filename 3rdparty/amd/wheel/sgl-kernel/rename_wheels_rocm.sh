#!/usr/bin/env bash
# Repack wheels so +rocm* in the filename matches METADATA/WHEEL (same strategy as sgl-kernel/rename_wheels.sh).
set -ex

WHEEL_DIR="dist"

resolve_python() {
    if [[ -n "${PYTHON_ROOT_PATH:-}" ]]; then
        if [[ -x "${PYTHON_ROOT_PATH}/bin/python" ]]; then
            echo "${PYTHON_ROOT_PATH}/bin/python"
            return
        fi
        if [[ -x "${PYTHON_ROOT_PATH}/python" ]]; then
            echo "${PYTHON_ROOT_PATH}/python"
            return
        fi
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return
    fi
    echo "ERROR: no python interpreter found (set PYTHON_ROOT_PATH or install python)." >&2
    exit 1
}

PYTHON="$(resolve_python)"

# Optional: SGL_KERNEL_ROCM_SUFFIX_OVERRIDE=+rocm700 for tests without /opt/rocm-*.
detect_rocm_suffix() {
    if [[ -n "${SGL_KERNEL_ROCM_SUFFIX_OVERRIDE:-}" ]]; then
        echo "${SGL_KERNEL_ROCM_SUFFIX_OVERRIDE}"
        return
    fi
    local ver_abrv
    ver_abrv=$(realpath /opt/rocm-* 2>/dev/null | head -1 | sed -e 's/.*-//' -e 's/\.//g')
    [[ -z "$ver_abrv" ]] && echo "" && return
    echo "+rocm${ver_abrv}"
}

ROCM_SUFFIX=$(detect_rocm_suffix)

patch_wheel_platform_tags() {
    local wheel_file="$1"
    sed -i \
        -e 's/linux_x86_64/manylinux2014_x86_64/g' \
        -e 's/linux_aarch64/manylinux2014_aarch64/g' \
        "$wheel_file"
}

wheel_files=("$WHEEL_DIR"/*.whl)
for wheel in "${wheel_files[@]}"; do
    [[ -f "$wheel" ]] || continue

    intermediate_wheel="${wheel/linux/manylinux2014}"
    [[ "$intermediate_wheel" == *"+rocm"* ]] && continue

    if [[ "$wheel" != "$intermediate_wheel" ]]; then
        mv -- "$wheel" "$intermediate_wheel"
    fi
    wheel="$intermediate_wheel"

    if [[ -z "$ROCM_SUFFIX" ]]; then
        continue
    fi

    TMPDIR=$(mktemp -d)
    trap 'rm -rf -- "$TMPDIR"' ERR

    "${PYTHON}" -m wheel unpack "$wheel" --dest "$TMPDIR"
    UNPACKED=$(find "$TMPDIR" -mindepth 1 -maxdepth 1 -type d | head -1)
    DIST_INFO=$(find "$UNPACKED" -maxdepth 1 -type d -name "*.dist-info" | head -1)
    WHEEL_META="${DIST_INFO}/WHEEL"
    METADATA_FILE="${DIST_INFO}/METADATA"

    patch_wheel_platform_tags "$WHEEL_META"

    ORIG_VERSION=$(grep '^Version:' "$METADATA_FILE" | head -1 | sed 's/^Version:[[:space:]]*//')
    if [[ "$ORIG_VERSION" == *"$ROCM_SUFFIX"* ]]; then
        echo "Skipping $wheel: version in METADATA is already suffixed."
        rm -rf "$TMPDIR"
        trap - ERR
        continue
    fi
    NEW_VERSION="${ORIG_VERSION}${ROCM_SUFFIX}"
    sed -i "s/^Version:.*/Version: ${NEW_VERSION}/" "$METADATA_FILE"

    OLD_BASE=$(basename "$DIST_INFO")
    NEW_BASE="${OLD_BASE/${ORIG_VERSION}/${NEW_VERSION}}"
    mv "$DIST_INFO" "${UNPACKED}/${NEW_BASE}"

    rm -f "$wheel"
    "${PYTHON}" -m wheel pack "$UNPACKED" --dest-dir "$WHEEL_DIR"
    rm -rf "$TMPDIR"
    trap - ERR
done
echo "Wheel renaming completed."
