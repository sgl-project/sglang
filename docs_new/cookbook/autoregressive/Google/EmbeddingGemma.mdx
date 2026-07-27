---
title: EmbeddingGemma
description: Serve Google's EmbeddingGemma text embedding model with SGLang.
tag: NEW
---

## Overview

[EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m) is Google's 300M-parameter text embedding model. SGLang detects its bidirectional Gemma 3 encoder, applies normalized mean pooling, and serves embeddings through the OpenAI-compatible `/v1/embeddings` endpoint.

On NVIDIA CUDA, SGLang uses breakable CUDA graph (BCG) for its complete prefill by default. It also disables prefix caching and chunked prefill, which are incompatible with this bidirectional encoder.

## Prerequisites

- NVIDIA CUDA GPU.
- A Hugging Face account that has accepted the [EmbeddingGemma license](https://huggingface.co/google/embeddinggemma-300m).
- A Hugging Face access token. Export it before starting the server so it can download the gated checkpoint:

```bash
export HF_TOKEN=<your-hugging-face-token>
```

Install an SGLang build that includes EmbeddingGemma support:

```bash
pip install 'git+https://github.com/sgl-project/sglang.git#subdirectory=python'
```

## Start the server

The standard configuration detects EmbeddingGemma and enables embedding mode,
BCG, and the checkpoint's BF16 dtype automatically:

```bash
sglang serve \
  --model-path google/embeddinggemma-300m \
  --host 0.0.0.0
```

### Hopper performance defaults

On H100 and H200, SGLang automatically selects FA3 and captures BCG through
16,384 tokens, covering eight 2K embedding requests in one replay. No extra
performance flags are required for this workload.

To capture larger aggregate prefills, raise the BCG tier explicitly:

```bash
sglang serve \
  --model-path google/embeddinggemma-300m \
  --cuda-graph-max-bs-prefill 32768 \
  --host 0.0.0.0
```

EmbeddingGemma automatically enables batch tokenization for list-valued
embedding requests, so do not add a separate tokenizer batching flag.

## Create embeddings

Send one string or a batch of strings to the OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:30000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/embeddinggemma-300m",
    "input": [
      "A short guide to serving text embeddings.",
      "Vector search retrieves semantically similar documents."
    ],
    "encoding_format": "float"
  }'
```

See [OpenAI-compatible embedding APIs](/docs/basic_usage/openai_api_embeddings) for Python and OpenAI client examples.

## Deployment behavior

EmbeddingGemma performs bidirectional attention over the complete input, so reusing a prefix KV cache or splitting the input into chunked prefills would produce incorrect attention states. SGLang applies the required settings automatically:

- disables RadixAttention prefix caching;
- disables chunked prefill;
- disables the decode CUDA graph because this is an embedding-only model;
- uses BCG for CUDA prefill;
- uses the FlashAttention raw-K/V path when the prefill backend is FA3 or FA4 on supported Hopper and Blackwell CUDA GPUs.

No prefill CUDA-graph override is required for this recipe. Keep BCG enabled to use the optimized EmbeddingGemma path.
