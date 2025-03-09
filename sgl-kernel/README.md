# SGL Kernel

[Kernel Library](https://github.com/sgl-project/sglang/tree/main/sgl-kernel) for SGLang

[![PyPI](https://img.shields.io/pypi/v/sgl-kernel)](https://pypi.org/project/sgl-kernel)

## Installation

For CUDA 11.8:

```bash
pip3 install sgl-kernel -i https://docs.sglang.ai/whl/cu118
```

For CUDA 12.1 or CUDA 12.4:

```bash
pip3 install sgl-kernel
```

# Developer Guide

## Development Environment Setup

Use Docker to set up the development environment. See [Docker setup guide](https://github.com/sgl-project/sglang/blob/main/docs/developer/development_guide_using_docker.md#setup-docker-container).

Create and enter development container:
```bash
docker run -itd --shm-size 32g --gpus all -v $HOME/.cache:/root/.cache --ipc=host --name sglang_zhyncs lmsysorg/sglang:dev /bin/zsh
docker exec -it sglang_zhyncs /bin/zsh
```

## Project Structure

### Dependencies

Third-party libraries:

- [CCCL](https://github.com/NVIDIA/cccl)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)

### Kernel Development

Steps to add a new kernel:

1. Implement the kernel in [csrc](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc)
2. Expose the interface in [include/sgl_kernel_ops.h](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/include/sgl_kernel_ops.h)
3. Create torch extension in [csrc/torch_extension.cc](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/torch_extension.cc)
4. Update [setup.py](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/setup.py) to include new CUDA source
5. Expose Python interface in [python](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel)

### Build & Install

Development build:

```bash
make build
```

Note:

The `sgl-kernel` is rapidly evolving. If you experience a compilation failure, try using `make rebuild`.

### Testing & Benchmarking

1. Add pytest tests in [tests/](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/tests)
2. Add benchmarks using [triton benchmark](https://triton-lang.org/main/python-api/generated/triton.testing.Benchmark.html) in [benchmark/](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/benchmark)
3. Run test suite

### Release new version

Update version in [pyproject.toml](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/pyproject.toml) and [version.py](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/version.py)
