#!/usr/bin/env bash
# Cache the files of the DeepSeek checkpoints the openai_parity fixtures pin, as
# a pinned commit's snapshot in the HF cache, where `resolve_model_file` finds
# them: the tokenizer the DeepSeek tests decode with, and the config that
# selects the checkpoint's renderer.
set -euo pipefail

fixtures="$(dirname "$0")/../fixtures/openai_parity"
cache="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
for fixture in "$fixtures"/deepseek-*.json; do
    model=$(jq -r .model "$fixture")
    revision=$(jq -r .revision "$fixture")
    dir="$cache/models--${model//\//--}/snapshots/$revision"
    mkdir -p "$dir"
    for file in tokenizer.json config.json; do
        curl -fsSL --retry 3 -o "$dir/$file" \
            "https://huggingface.co/$model/resolve/$revision/$file"
    done
done
