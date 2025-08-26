# SGLang Codebase Structure

## Main Directories

### `/python/sglang/`
The core Python package containing:
- `srt/` - Server Runtime: Core serving engine implementation
  - `layers/` - Neural network layers and optimizations
  - `models/` - Model implementations
  - `mem_cache/` - Memory caching systems
  - `openai_api/` - OpenAI-compatible API server
- `lang/` - Frontend language API for programming LLM applications
- `test/` - Unit tests for Python components
- `eval/` - Evaluation utilities
- Core files: `launch_server.py`, `bench_serving.py`, `utils.py`

### `/sgl-router/`
Rust-based router for data parallelism and load balancing:
- `src/` - Rust source code
- `py_src/` - Python bindings
- `tests/` - Rust unit tests
- Built with Cargo and setuptools-rust

### `/sgl-kernel/`
CUDA/compute kernels for performance optimization:
- Custom GPU kernels
- Optimized operations
- Hardware-specific implementations

### `/test/`
Comprehensive test suite:
- `srt/` - Server runtime tests
- `lang/` - Frontend language tests
- Integration and performance tests

### `/scripts/`
- `ci/` - CI/CD scripts
- Utility scripts for development
- Deployment helpers

### `/docs/`
Documentation source files (rendered at docs.sglang.ai)

### `/benchmark/`
Performance benchmarking scripts and configurations

### `/examples/`
Example code and usage demonstrations

### `/docker/`
Docker configurations and Kubernetes deployment files

## Key Configuration Files
- `pyproject.toml` - Python package configuration
- `.pre-commit-config.yaml` - Code quality checks
- `Makefile` - Development shortcuts
- `.isort.cfg` - Import sorting configuration