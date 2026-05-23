# sglang-benchmark
#
# Multi-arch (linux/amd64 + linux/arm64) slim image for the *client-only*
# benchmarks under `benchmark/` — those that hit an sglang server over HTTP
# and never `import sglang.srt` or touch CUDA.
#
# Verified-runnable client dirs: asr/, prefill_only/.
# benchmark_batch/ imports `sglang.lang.backend.runtime_endpoint` and
# `sglang.srt.utils.patch_tokenizer` — the latter pulls in the GPU stack,
# so benchmark_batch is NOT runnable here. Run it from the main
# `lmsysorg/sglang` image instead.
#
# Build via docker/build-benchmark.sh, or directly:
#   docker buildx build --platform linux/amd64,linux/arm64 \
#     -f docker/sglang-benchmark-client.Dockerfile \
#     -t <registry>/sglang-benchmark:latest --push .

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Client-side bench deps only. No torch, no CUDA, no sglang server runtime.
RUN pip install \
        aiohttp \
        datasets \
        evaluate \
        httpx \
        "huggingface-hub[hf_transfer]" \
        librosa \
        numpy \
        openai \
        requests \
        soundfile \
        tqdm \
        transformers

COPY benchmark /sglang/benchmark
WORKDIR /sglang/benchmark
