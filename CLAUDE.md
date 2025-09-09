# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SGLang is a fast serving framework for large language models and vision language models. It consists of two main components:

1. **Frontend Language API** (`sglang/lang/`) - Python API for programming LLM applications with chained generation, control flow, and multi-modal inputs
2. **Backend Runtime** (`sglang/srt/`) - High-performance serving engine with RadixAttention, batching, parallelism, and optimization features

## Key Commands

### Development Setup
```bash
# Install from source (after cloning)
pip install -e ".[dev]"

# Install pre-commit hooks for code formatting
pip install pre-commit
pre-commit install

# Format code before committing
pre-commit run --all-files
make format  # Alternative: formats only modified Python files
```

### Testing
```bash
# Run specific test suites
python test/lang/run_suite.py  # Frontend language tests
python test/srt/run_suite.py   # Backend runtime tests (if exists)

# Run accuracy tests for model changes
python -m sglang.launch_server --model Qwen/Qwen2-7B-Instruct
# Then run specific benchmarks from benchmark/ directory
```

### Linting and Code Quality
```bash
# The project uses pre-commit with these tools:
# - black (Python formatting)
# - isort (import sorting) 
# - ruff (linting, F401 unused imports)
# - clang-format (C++/CUDA code)
# - codespell (spell checking)
```

### Building and Installation
```bash
# Install different runtime variants:
pip install -e ".[srt]"      # NVIDIA GPU runtime
pip install -e ".[srt_hip]"  # AMD GPU runtime  
pip install -e ".[srt_cpu]"  # CPU runtime
pip install -e ".[all]"      # Full installation
```

## Architecture Overview

### Core Directories
- `python/sglang/lang/` - Frontend language implementation with backends for OpenAI, Anthropic, etc.
- `python/sglang/srt/` - SGLang Runtime with model execution, memory management, distributed serving
- `python/sglang/srt/models/` - Model implementations (Llama, Qwen, DeepSeek, etc.)
- `python/sglang/srt/managers/` - Request scheduling and batching logic
- `python/sglang/srt/layers/` - Custom CUDA kernels and attention layers
- `sgl-kernel/` - Custom CUDA/Triton kernels (separate package)
- `sgl-router/` - Load balancing and routing component
- `benchmark/` - Performance benchmarks and evaluation scripts
- `test/` - Unit tests for both frontend and runtime components

### Key Components
- **RadixAttention**: Prefix caching system for efficient serving
- **BatchManager**: Handles continuous batching and request scheduling  
- **ModelExecutor**: Manages model loading, sharding, and inference
- **MemoryPool**: GPU memory management for KV cache
- **Tokenizer**: Text processing and token management
- **LoRA**: Low-rank adaptation support for efficient fine-tuning

### Multi-Platform Support
The codebase supports multiple hardware platforms with different installation profiles:
- NVIDIA GPUs (default `srt` profile)
- AMD GPUs (`srt_hip` with ROCm)
- Intel CPUs (`srt_cpu`)
- Intel XPU/Gaudi (`srt_xpu`, `srt_hpu`)
- Ascend NPU (`srt_npu`)

## Development Workflow

1. **Code Changes**: Always create feature branches, never commit to main
2. **Testing**: Add unit tests for new features, run accuracy tests for model changes
3. **Documentation**: Update docs for new features or API changes
4. **Pre-commit**: Code must pass all pre-commit checks before PR submission
5. **Performance**: Provide benchmarks for changes affecting inference speed

## Important Notes

- The project uses PyTorch 2.8.0 and specific versions of transformers (4.56.0) and other ML libraries
- Model outputs are deterministic - changes to kernels or forward passes require accuracy validation
- The codebase includes optimizations for specific model architectures (DeepSeek MLA, etc.)
- Docker images and multi-GPU setups are supported for production deployments