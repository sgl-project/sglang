#!/bin/bash
set -euxo pipefail

models=$(python3 -c "from sglang.test.test_utils import _get_default_models; print(_get_default_models())" | jq -r '.[]')

if [ -z "$models" ]; then
    echo "Failed to get default models."
    exit 1
fi

# 获取 DEFAULT_MODEL_CACHE_DIR 环境变量的值
cache_dir="${DEFAULT_MODEL_CACHE_DIR:-}"

if [ -z "$cache_dir" ]; then
    echo "DEFAULT_MODEL_CACHE_DIR environment variable is not set."
    exit 1
fi

for model in $models; do
    # 构建每个模型的本地下载路径
    local_model_dir="$cache_dir/$model"
    echo "Caching model: $model to $local_model_dir"
    # 使用 local_model_dir 作为下载路径
    mkdir -p "$local_model_dir"
    huggingface-cli download "$model" --local-dir "$local_model_dir" --local-dir-use-symlinks False
    if [ $? -ne 0 ]; then
        echo "Failed to cache model: $model"
        continue 
    else
        echo "Successfully cached model: $model"
    fi
done
