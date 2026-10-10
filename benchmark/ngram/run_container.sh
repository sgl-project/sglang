#!/usr/bin/env bash
# Run the benchmark checkout in a prebuilt SGLang environment. No image build.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
inputs=${BENCH_INPUTS_DIR:-"$HOME/sa-bench"}
image=${SGLANG_BENCHMARK_IMAGE:-docker.io/lmsysorg/sglang:v0.5.19}
: "${BENCH_GPU_UUID:?Set BENCH_GPU_UUID to your assigned GPU UUID (GPU-...)}"
if [[ $BENCH_GPU_UUID != GPU-* ]]; then
    echo 'BENCH_GPU_UUID must be a GPU UUID, not a numeric device index.' >&2
    exit 2
fi
if [[ ! -d $repo/.git ]]; then
    echo 'Use a standalone clone so Git provenance is available inside the container.' >&2
    exit 2
fi
for path in manifest.json dataset.jsonl model trt-inputs.json; do
    if [[ ! -e $inputs/$path ]]; then
        echo "Missing $inputs/$path; prepare inputs and export TRT tokens first." >&2
        exit 2
    fi
done
inputs=$(cd "$inputs" && pwd)

engine=${CONTAINER_ENGINE:-}
if [[ -z $engine ]]; then
    if command -v podman >/dev/null 2>&1; then
        engine=podman
    else
        engine=docker
    fi
fi
gpu_args=(--gpus "device=$BENCH_GPU_UUID")
if [[ $("$engine" --version) == *[Pp]odman* ]]; then
    gpu_args=(--device "nvidia.com/gpu=$BENCH_GPU_UUID")
fi
run_args=(--rm)
if [[ -t 0 && -t 1 ]]; then
    run_args+=(-it)
fi
if [[ $# == 0 ]]; then
    set -- bash
fi

exec "$engine" run "${run_args[@]}" "${gpu_args[@]}" \
    --ipc=host \
    -v "$repo:/workspace/sglang:ro" \
    -v "$inputs:/work" \
    -w /workspace/sglang \
    -e PYTHONPATH=/workspace/sglang/python \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e SGLANG_RUST_BUILD_MODE=never \
    -e GIT_CONFIG_COUNT=1 \
    -e GIT_CONFIG_KEY_0=safe.directory \
    -e GIT_CONFIG_VALUE_0=/workspace/sglang \
    -e "SGLANG_BENCHMARK_IMAGE=$image" \
    --entrypoint /bin/bash "$image" -c \
    'python benchmark/ngram/check_env.py && exec "$@"' benchmark-container "$@"
