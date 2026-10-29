#!/usr/bin/env bash
# Run inside the serving image.
# Runtime-only binary: do not add it to git.
set -euo pipefail
q38_package_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
q38_binary=mem_cache.cpython-312-aarch64-linux-gnu.so
q38_runtime_src=/sgl-workspace/sglang/python/sglang/srt/mem_cache/rust_tree_core/$q38_binary
q38_runtime_dst=$q38_package_dir/../mem_cache/rust_tree_core/$q38_binary
if [[ ! -e $q38_runtime_dst ]]; then
  cp -- "$q38_runtime_src" "$q38_runtime_dst"
fi
