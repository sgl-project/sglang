# SGLang Project Overview

## Purpose
SGLang is a fast serving framework for large language models (LLMs) and vision language models (VLMs). It provides efficient serving with features like RadixAttention for prefix caching, zero-overhead CPU scheduler, prefill-decode disaggregation, and more.

## Tech Stack
- **Programming Languages**: Python (primary), Rust (for sgl-router), C++ (for sgl-kernel)
- **Framework**: Built on PyTorch 2.8
- **GPU Support**: NVIDIA (CUDA), AMD (ROCm), Intel XPU, and other accelerators
- **Key Dependencies**:
  - torch==2.8.0
  - flashinfer_python==0.2.11.post3
  - transformers==4.55.2
  - FastAPI for API server
  - sgl-kernel for custom CUDA/compute kernels

## Model Support
- Generative models: Llama, Qwen, DeepSeek, Kimi, GPT, Gemma, Mistral, etc.
- Embedding models: e5-mistral, gte, mcdse
- Reward models: Skywork
- Vision models: LLaVA and other VLMs

## Key Features
- Fast backend runtime with various optimizations
- Flexible frontend language for programming LLM applications
- Extensive model support with easy extensibility
- Active community with wide industry adoption (deployed on over 1,000,000 GPUs worldwide)