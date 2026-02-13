#!/usr/bin/env bash
set -ex

WHEEL_DIR="dist"

wheel_files=($WHEEL_DIR/*.whl)
for wheel in "${wheel_files[@]}"; do
    intermediate_wheel="${wheel/linux/manylinux2014}"

    # Extract the current python version from the wheel name
    if [[ $intermediate_wheel =~ -cp([0-9]+)- ]]; then
        cp_version="${BASH_REMATCH[1]}"
    else
        echo "Could not extract Python version from wheel name: $intermediate_wheel"
        continue
    fi

    # Detect ROCm version and add appropriate suffix
    if ls /opt | grep -q "7.0"; then
        new_wheel="${intermediate_wheel/-cp${cp_version}/+rocm700-cp${cp_version}}"
    else
        new_wheel="$intermediate_wheel"
    fi

    if [[ "$wheel" != "$new_wheel" ]]; then
        echo "Renaming $wheel to $new_wheel"
        mv -- "$wheel" "$new_wheel"
    fi
done
echo "Wheel renaming completed."
