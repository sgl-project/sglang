#!/bin/bash

#define input and output direction
input_dir="./boolq/data"
output_dir="./boolq/data"

#define files needed to be handled
files=(
        "train-00000-of-00001.parquet"
        "validation-00000-of-00001.parquet"
)

#foe files above, use python script to convert the form
for file in "${files[@]}"; do
    input_file="${input_dir}/${file}"
    output_file="${output_dir}/${file%.parquet}.json"

    echo "Converting ${input_file} to ${output_file} ..."
    python3 convert_parquet_to_json.py "${input_file}" "${output_file}"

    if [ $? -eq 0 ]; then
        echo "Conversion successful: ${output_file}"
    else
        echo "Conversion failed: ${input_file}"
    fi
done
