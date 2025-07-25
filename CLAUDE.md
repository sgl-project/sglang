# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SGLang is a high-performance serving framework for large language models (LLMs) and vision language models (VLMs). It consists of a co-designed backend runtime (SRT) and frontend language for efficient LLM programming.

## Key Architecture Components

### Python Package Structure (python/sglang/)
- **srt/**: Backend serving engine - handles model serving, scheduling, memory management
- **lang/**: Frontend language APIs - provides intuitive programming interfaces (gen, select, function)
- **test/**: Test utilities organized by backend (srt) and frontend (lang) components

### Multi-Language Components
- **sgl-router/**: Rust-based high-performance load balancer with cache-aware routing
- **sgl-kernel/**: C++/CUDA kernels for performance-critical operations
- **benchmark/**: Performance testing suites for throughput and latency measurements

### Core Design Principles
- RadixAttention for efficient prefix caching and memory management
- Support for 100+ models with various parallelism strategies (tensor, pipeline, data, expert)
- Multiple quantization methods (FP8, INT4, AWQ, GPTQ) for different deployment scenarios
- Hardware flexibility: NVIDIA GPUs, AMD GPUs, Intel CPUs/GPUs, AWS Neuron, TPUs

## Essential Development Commands

### Building and Installation
```bash
# Install from source with development dependencies
pip install -e "python[dev]"

# Install for specific hardware backends
pip install "sglang[srt_hip]"  # AMD GPUs
pip install "sglang[srt_cpu]"  # CPU only
```

### Running Tests
```bash
# Run per-commit test suite (fast, essential tests)
cd test/srt
python run_suite.py --suite per-commit

# Run specific test file
python test_serving_throughput.py

# Run multi-GPU tests
python run_suite.py --suite multi-gpu
```

### Code Quality
```bash
# Format code (uses black and isort)
make format

# Update version across all components
make update 0.4.10
```

### Launching Server
```bash
# Basic server launch
python -m sglang.launch_server --model-path meta-llama/Llama-3-8B-Instruct

# With specific configurations
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3-8B-Instruct \
  --port 30000 \
  --mem-fraction-static 0.8 \
  --tp 2  # tensor parallelism
```

## Testing Strategy

- Tests use Python's `unittest` framework
- Test suites in `test/srt/run_suite.py`: per-commit, nightly, multi-gpu
- Always run per-commit tests before submitting changes
- Backend tests in `test/srt/`, frontend tests in `test/lang/`
- Mock models available for quick testing without downloading full weights

## Key Files and Entry Points

- **Server Entry**: `python/sglang/srt/server.py` - Main serving engine
- **Model Registry**: `python/sglang/srt/models/` - Model implementations
- **Router**: `sgl-router/src/main.rs` - Load balancer entry point
- **Benchmarks**: `benchmark/throughput_latency/launch_benchmark.py` - Performance testing

## Performance-Critical Code Areas

When modifying these areas, ensure thorough benchmarking:
- `python/sglang/srt/managers/scheduler.py` - Request scheduling logic
- `python/sglang/srt/mem_cache/radix_cache.py` - RadixAttention implementation
- `sgl-kernel/` - Low-level CUDA/C++ kernels
- `python/sglang/srt/model_executor/` - Model execution pipeline

## Common Development Patterns

### Adding New Models
1. Create model file in `python/sglang/srt/models/`
2. Register in model registry
3. Add tests in `test/srt/models/`
4. Update documentation

### Implementing New Features
1. Backend features go in `python/sglang/srt/`
2. Frontend APIs in `python/sglang/lang/`
3. Always include unit tests
4. Update relevant benchmarks

### Debugging Tips
- Use `--log-level debug` for verbose server logs
- Check `crashed_rank_*.log` for multi-GPU debugging
- Use `--disable-cuda-graph` to debug CUDA issues
- Enable `--enable-torch-compile` for performance debugging
