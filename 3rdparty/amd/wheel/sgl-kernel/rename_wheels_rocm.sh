#!/usr/bin/env bash
set -ex

WHEEL_DIR="dist"

# ROCm version for the wheel's local version tag (e.g. 720 -> +rocm720). It is
# read from the /opt/rocm-<version> tree when the caller does not name one; the
# pip-installed ROCm 10 SDK has no such tree, so that flavor passes it in.
ROCM_WHEEL_VERSION="${1:-}"

wheel_files=($WHEEL_DIR/*.whl)
for wheel in "${wheel_files[@]}"; do
    intermediate_wheel="${wheel/linux/manylinux2014}"
    [[ "$intermediate_wheel" == *"+rocm"* ]] && continue

    # Extract the current python version from the wheel name
    if [[ $intermediate_wheel =~ -cp([0-9]+)- ]]; then
        cp_version="${BASH_REMATCH[1]}"
    else
        echo "Could not extract Python version from wheel name: $intermediate_wheel"
        continue
    fi

    # Detect ROCm version and add appropriate suffix
    ver_abrv="${ROCM_WHEEL_VERSION}"
    if [[ -z "$ver_abrv" ]]; then
        ver_abrv=$(realpath /opt/rocm-* | sed -e 's/.*-//' -e 's/\.//g')
    fi
    new_wheel=${intermediate_wheel/-cp${cp_version}/+rocm${ver_abrv}-cp${cp_version}}

    if [[ "$wheel" != "$new_wheel" ]]; then
        echo "Renaming $wheel to $new_wheel"
        mv -- "$wheel" "$new_wheel"
    fi
done
echo "Wheel renaming completed."
