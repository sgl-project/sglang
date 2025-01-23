# Developer Guide for sgl-kernel

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

1. Implement in `sgl-kernel/src/sgl-kernel/csrc`
2. Expose interface in `sgl-kernel/csrc/sgl_kernel_ops.cu` with pybind11
3. Create Python wrapper in `sgl-kernel/src/sgl-kernel/ops/__init__.py`
4. Expose Python interface in `sgl-kernel/src/sgl-kernel/__init__.py`

### Build & Install

Development build:

```bash
make build
pip3 install dist/*whl --force-reinstall --no-deps
# Or use: make install (runs pip install -e .)
```

### Testing & Benchmarking

1. Add pytest tests in `sgl-kernel/tests/`
2. Add benchmarks using [triton benchmark](https://triton-lang.org/main/python-api/generated/triton.testing.Benchmark.html) in `sgl-kernel/benchmark/`
3. Run test suite
