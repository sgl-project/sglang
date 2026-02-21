#!/usr/bin/env bash
set -ex

WHEEL_DIR="dist"

# Function to update the version inside a wheel's METADATA and RECORD files
update_wheel_metadata() {
    local wheel_path="$1"
    local version_suffix="$2"

    if [[ -z "$version_suffix" ]]; then
        return 0
    fi

    local temp_dir
    temp_dir=$(mktemp -d)
    trap "rm -rf '$temp_dir'" RETURN

    # Unzip the wheel
    unzip -q "$wheel_path" -d "$temp_dir"

    # Find the dist-info directory
    local dist_info_dir
    dist_info_dir=$(find "$temp_dir" -maxdepth 1 -type d -name "*.dist-info" | head -1)

    if [[ -z "$dist_info_dir" ]]; then
        echo "Warning: Could not find .dist-info directory in $wheel_path"
        return 1
    fi

    local metadata_file="$dist_info_dir/METADATA"
    local record_file="$dist_info_dir/RECORD"

    if [[ ! -f "$metadata_file" ]]; then
        echo "Warning: METADATA file not found in $wheel_path"
        return 1
    fi

    # Update the Version field in METADATA
    # Match "Version: X.Y.Z" and append the suffix
    sed -i "s/^Version: \(.*\)$/Version: \1${version_suffix}/" "$metadata_file"

    # Regenerate the RECORD file (contains checksums of all files)
    # The RECORD file itself has an empty hash entry
    local dist_info_name
    dist_info_name=$(basename "$dist_info_dir")

    # Remove old RECORD and regenerate
    rm -f "$record_file"

    # Generate new RECORD entries
    while IFS= read -r -d '' file; do
        local rel_path="${file#$temp_dir/}"
        if [[ "$rel_path" == "$dist_info_name/RECORD" ]]; then
            continue
        fi
        local hash
        hash=$(openssl dgst -sha256 -binary "$file" | openssl base64 -A | tr '+/' '-_' | tr -d '=')
        local size
        size=$(wc -c < "$file" | tr -d ' ')
        echo "$rel_path,sha256=$hash,$size" >> "$record_file"
    done < <(find "$temp_dir" -type f -print0)

    # Add RECORD itself with empty hash
    echo "$dist_info_name/RECORD,," >> "$record_file"

    # Repack the wheel
    rm -f "$wheel_path"
    (cd "$temp_dir" && zip -q -r "$wheel_path" .)
}

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

    # Detect CUDA version and add appropriate suffix
    version_suffix=""
    if ls /usr/local/ | grep -q "12.4"; then
        version_suffix="+cu124"
        new_wheel="${intermediate_wheel/-cp${cp_version}/+cu124-cp${cp_version}}"
    elif ls /usr/local/ | grep -q "12.8"; then
        version_suffix="+cu128"
        new_wheel="${intermediate_wheel/-cp${cp_version}/+cu128-cp${cp_version}}"
    elif ls /usr/local/ | grep -q "13.0"; then
        version_suffix="+cu130"
        new_wheel="${intermediate_wheel/-cp${cp_version}/+cu130-cp${cp_version}}"
    else
        new_wheel="$intermediate_wheel"
    fi

    # First rename linux -> manylinux2014 if needed
    if [[ "$wheel" != "$intermediate_wheel" ]]; then
        mv -- "$wheel" "$intermediate_wheel"
    fi

    # Update the metadata inside the wheel if we're adding a version suffix
    if [[ -n "$version_suffix" ]]; then
        echo "Updating METADATA in $intermediate_wheel with version suffix $version_suffix"
        update_wheel_metadata "$(pwd)/$intermediate_wheel" "$version_suffix"
    fi

    # Rename to final name with CUDA suffix
    if [[ "$intermediate_wheel" != "$new_wheel" ]]; then
        echo "Renaming $intermediate_wheel to $new_wheel"
        mv -- "$intermediate_wheel" "$new_wheel"
    fi
done
echo "Wheel renaming completed."
