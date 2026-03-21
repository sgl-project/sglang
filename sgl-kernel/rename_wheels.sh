#!/usr/bin/env bash
# Repack wheels so the filename local version (+cu124/+cu128/+cu130) matches METADATA/WHEEL.
# Uses `wheel unpack` / `wheel pack` only (no manual RECORD); patches WHEEL Tag so pack emits manylinux2014_*, not linux_*.
set -ex

WHEEL_DIR="dist"

# Prefer manylinux Python (Dockerfile sets PYTHON_ROOT_PATH=/opt/python/cp310-cp310); ROCm images may use /opt/venv/bin/python.
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
    # Prefer `python` before `python3` when PATH is ambiguous (e.g. Git Bash: Store python3 stub).
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

# Optional: SGL_KERNEL_CUDA_SUFFIX_OVERRIDE=+cu130 for tests/CI without /usr/local/cuda-* layout.
detect_cuda_suffix() {
    if [[ -n "${SGL_KERNEL_CUDA_SUFFIX_OVERRIDE:-}" ]]; then
        echo "${SGL_KERNEL_CUDA_SUFFIX_OVERRIDE}"
        return
    fi
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
    # wheel pack reads Tag lines to build the filename; must match manylinux2014_* not linux_*.
    sed -i \
        -e 's/linux_x86_64/manylinux2014_x86_64/g' \
        -e 's/linux_aarch64/manylinux2014_aarch64/g' \
        "$wheel_file"
}

wheel_files=("$WHEEL_DIR"/*.whl)
for wheel in "${wheel_files[@]}"; do
    [[ -f "$wheel" ]] || continue

    intermediate_wheel="${wheel/linux/manylinux2014}"
    if [[ "$wheel" != "$intermediate_wheel" ]]; then
        mv -- "$wheel" "$intermediate_wheel"
    fi
    wheel="$intermediate_wheel"

    if [[ -z "$CUDA_SUFFIX" ]]; then
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
    "${PYTHON}" -m wheel pack "$UNPACKED" --dest-dir "$WHEEL_DIR"
    rm -rf "$TMPDIR"
    trap - ERR
done
echo "Wheel renaming completed."
