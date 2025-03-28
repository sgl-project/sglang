#!/usr/bin/env bash
set -ex

WHEEL_DIR="dist"

wheel_files=($WHEEL_DIR/*.whl)
for wheel in "${wheel_files[@]}"; do
    new_wheel="${wheel/linux/manylinux2014}"

    if [[ "$wheel" != "$new_wheel" ]]; then
        echo "Renaming $wheel to $new_wheel"
        mv -- "$wheel" "$new_wheel"

    fi
done

echo "Wheel renaming completed."
