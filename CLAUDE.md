# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SGLang is a high-performance serving framework for large language models (LLMs) and vision-language models (VLMs). It provides low-latency and high-throughput inference across single GPUs to large distributed clusters.

**Key Features**: RadixAttention (prefix caching), continuous batching, paged attention, tensor/pipeline/expert parallelism, speculative decoding, quantization (FP4/FP8/INT4/AWQ/GPTQ), multi-LoRA batching.

**Hardware Support**: NVIDIA GPUs (GB200/H100/A100), AMD GPUs (MI355/MI300), Intel Xeon CPUs, Google TPUs, Ascend NPUs.

## Build & Development Commands

### Python Package (main)
```bash
cd python
pip install -e .                    # Development install
pip install -e ".[test]"            # With test dependencies
```

### CUDA Kernels (sgl-kernel)
```bash
cd sgl-kernel
make build                          # Build with CMake, uses MAX_JOBS=$(nproc)
make install                        # Development install with pip -e
make test                           # Run kernel tests
make format                         # Format C++/CUDA/Python files
```

### Model Gateway (Rust)
```bash
cd sgl-model-gateway
make build                          # cargo build --release
make test                           # cargo test
make python-dev                     # Build Python bindings (maturin develop)
```

### Running Tests
```bash
# Single test file
python3 test/srt/test_srt_endpoint.py

# Single test method
python3 test/srt/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# Test suite (per-commit, nightly, etc.)
python3 test/srt/run_suite.py --suite per-commit-1-gpu
```

### Code Formatting (pre-commit)
```bash
isort python/
black python/
ruff check --select=F401,F821 --fix python/
clang-format -i sgl-kernel/csrc/**/*.cu sgl-kernel/csrc/**/*.cuh
```

## Architecture

### Three-Component Pipeline

```
TokenizerManager (Python)  →  Scheduler (subprocess)  →  DetokenizerManager (subprocess)
      ↓                            ↓
   FastAPI/HTTP              ModelRunner (GPU)
   gRPC Server               CUDA Graph Execution
```

**IPC**: ZMQ-based asynchronous messaging between processes.

### Directory Structure

```
python/sglang/
├── lang/                    # Frontend DSL (gen, select, chat templates)
│   ├── api.py               # Public API functions
│   ├── ir.py                # Intermediate representation
│   └── backend/             # OpenAI, Anthropic, VertexAI backends
└── srt/                     # Runtime backend (serving)
    ├── entrypoints/
    │   ├── engine.py        # Main Engine class, process management
    │   ├── http_server.py   # FastAPI server, OpenAI-compatible routes
    │   └── openai/          # protocol.py, serving_chat.py, serving_completions.py
    ├── managers/
    │   ├── tokenizer_manager.py   # Request entry, tokenization
    │   ├── scheduler.py           # Batch scheduling, KV cache allocation
    │   └── detokenizer_manager.py # Output post-processing
    ├── model_executor/
    │   ├── model_runner.py        # Model forward pass
    │   └── cuda_graph_runner.py   # CUDA graph capture/replay
    ├── models/              # 100+ model implementations (llama, qwen, deepseek, etc.)
    ├── layers/attention/    # Attention backends (flashinfer, triton, etc.)
    ├── mem_cache/
    │   ├── radix_cache.py   # RadixAttention prefix caching (trie-based)
    │   └── memory_pool.py   # GPU memory pool management
    ├── sampling/            # Temperature, top-p, top-k sampling
    ├── constrained/         # Structured outputs (llguidance, outlines, xgrammar)
    ├── distributed/         # Tensor/pipeline parallelism
    ├── lora/                # LoRA adapter support
    ├── speculative/         # Speculative decoding (Eagle, Medusa)
    └── server_args.py       # 200+ server configuration options

sgl-kernel/                  # C++/CUDA kernels (CMake build)
sgl-model-gateway/           # Rust-based request router (Cargo build)
```

### Request Flow

1. HTTP/gRPC request → TokenizerManager (tokenize, validate)
2. → Scheduler (allocate KV cache, plan batches)
3. → ModelRunner (forward pass, sample tokens)
4. → DetokenizerManager (convert to text)
5. → HTTP/gRPC response (streaming or full)

### Key Innovations

**RadixAttention**: Trie-based KV cache enabling prefix sharing. Dramatically reduces memory with repeated prompts. Implementation in `mem_cache/radix_cache.py` and `mem_cache/radix_cache_cpp.py`.

**Attention Backend Registry**: Runtime selection of optimal kernel based on hardware/model/precision. See `layers/attention/attention_registry.py`.

**CUDA Graph Execution**: Captures and replays forward passes for reduced kernel launch overhead. See `model_executor/cuda_graph_runner.py`.

## Configuration

Server configuration via `ServerArgs` in `server_args.py`. Key parameters:

- Model: `--model`, `--tokenizer-path`, `--dtype`
- Memory: `--mem-fraction-static`, `--max-running-requests`
- Parallelism: `--tp` (tensor), `--pp` (pipeline), `--dp` (data)
- Attention: `--attention-backend` (flashinfer, triton, etc.)
- Quantization: `--quantization` (fp8, awq, gptq)

## Test Suites

Tests in `test/srt/run_suite.py` organized by GPU requirements:
- `per-commit-1-gpu`: Fast tests run per commit
- `per-commit-2-gpu`, `per-commit-4-gpu`: Multi-GPU tests
- `per-commit-8-gpu-h200`: Large model tests (DeepSeek V3)
- `nightly-*`: Performance benchmarks

## Current Branch Context

**Branch**: `feature/attention-token-visualization`

Recent commits on this branch add top-k attention token capture for interpretability/visualization:
- Modified: `protocol.py`, `serving_chat.py`, `serving_completions.py`, `cuda_graph_runner.py`, `piecewise_cuda_graph_runner.py`
- New tests: `test_attention_tokens.py`, `test_topk_attention.py`
- Server args: `--return-attention-tokens`, `--attention-tokens-top-k`
- API usage: `extra_body={"return_attention_tokens": True}`

## Common Development Tasks

### Adding a New Model
1. Create `python/sglang/srt/models/my_model.py`
2. Implement model class with architecture and forward pass
3. Register in model registry
4. Add tests in `test/srt/models/`

### Adding OpenAI API Extension
1. Add protocol in `entrypoints/openai/protocol.py`
2. Implement in `entrypoints/openai/serving_*.py`
3. Add route in `entrypoints/http_server.py`
4. Add tests in `test/srt/openai_server/`

### Debugging
```bash
SGLANG_LOG_LEVEL=DEBUG python -m sglang.launch_server ...
SGLANG_LOG_REQ=1    # Log request details
```
