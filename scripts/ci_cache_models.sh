#!/bin/bash
set -euxo pipefail

mapfile -t models < <(python3 -c "from sglang.test.test_utils import _get_default_models; print(_get_default_models())" | jq -r '.[]')

if [ ${#models[@]} -eq 0 ]; then
    echo "Failed to get default models."
    exit 1
fi

cache_dir="${DEFAULT_MODEL_CACHE_DIR:-}"

if [ -z "$cache_dir" ]; then
    echo "DEFAULT_MODEL_CACHE_DIR environment variable is not set."
    exit 1
fi

failed_models=()
for model in "${models[@]}"; do
    local_model_dir="$cache_dir/$model"
    echo "Caching model: $model to $local_model_dir"
    mkdir -p "$local_model_dir"

    if ! huggingface-cli download "$model" \
        --local-dir "$local_model_dir" \
        --local-dir-use-symlinks False 2>/dev/null; then
        echo "WARNING: Failed to cache model: $model"
        rm -rf "$local_model_dir"
        failed_models+=("$model")
        continue
    fi
    echo "Successfully cached model: $model"
done

if [ ${#failed_models[@]} -gt 0 ]; then
    echo -e "\n[Summary] Failed to cache following models:"
    printf ' - %s\n' "${failed_models[@]}"
else
    echo -e "\n[Summary] All models cached successfully"
fi
