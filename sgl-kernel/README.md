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

- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)

### Kernel Development

Steps to add a new kernel:

1. Implement the kernel in [csrc](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc)
2. Expose the interface in [include/sgl_kernel_ops.h](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/include/sgl_kernel_ops.h)
3. Create torch extension in [csrc/torch_extension.cc](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/torch_extension.cc)
4. Update [CMakeLists.txt](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/CMakeLists.txt) to include new CUDA source
5. Expose Python interface in [python](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel)

### Development Tips

1. When implementing kernels in [csrc](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc), only define pure CUDA files and C++ interfaces. If you need to use `Torch::tensor`, use `<torch/all.h>` instead of `<torch/extension.h>`. Using `<torch/extension.h>` will cause compilation errors when using SABI.

2. When creating torch extensions, simply add the function definition with `m.def`:
   ```cpp
   m.def("register_graph_buffers", register_graph_buffers);
   ```

3. When exposing Python interfaces, avoid using kwargs in C++ interface kernels.

    **Avoid this:**

    ```cpp
    torch.ops.sgl_kernel.apply_rope_pos_ids_cos_sin_cache.default(
        q=query.view(query.shape[0], -1, head_size),
        k=key.view(key.shape[0], -1, head_size),
        q_rope=query.view(query.shape[0], -1, head_size),
        k_rope=key.view(key.shape[0], -1, head_size),
        cos_sin_cache=cos_sin_cache,
        pos_ids=positions.long(),
        interleave=(not is_neox),
        cuda_stream=get_cuda_stream(),
    )
    ```

    **Use this instead:**

    ```cpp
    torch.ops.sgl_kernel.apply_rope_pos_ids_cos_sin_cache.default(
        query.view(query.shape[0], -1, head_size),
        key.view(key.shape[0], -1, head_size),
        query.view(query.shape[0], -1, head_size),
        key.view(key.shape[0], -1, head_size),
        cos_sin_cache,
        positions.long(),
        (not is_neox),
        get_cuda_stream(),
    )
    ```

### Integrating Third-Party Libraries with Data Type Conversion

When integrating new third-party libraries like flash-attention, you may encounter data type compatibility issues between the C++ interface and PyTorch bindings. For example, the third-party code might use `float` or `int` types, while PyTorch requires `double` and `int64_t`.

To address this issue, we provide the `make_pytorch_shim` function in [sgl_kernel_torch_shim](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/include/sgl_kernel_torch_shim.h) that handles data type conversions automatically.

When you need to support new data type conversions, you can easily add conversion functions like this:

```cpp
// Map `int` -> `int64_t`
template <>
struct pytorch_library_compatible_type<int> {
  using type = int64_t;
  static int convert_from_type(int64_t arg) {
    TORCH_CHECK(arg <= std::numeric_limits<int>::max(), "int64_t value is too large to be converted  to int");
    TORCH_CHECK(arg >= std::numeric_limits<int>::min(), "int64_t value is too small to be converted to int");
    return arg;
  }
};
```

To use this with your library functions, simply wrap them with make_pytorch_shim:

```cpp
/*
 * From flash-attention
 */
 m.def("fwd", make_pytorch_shim(mha_fwd));
```

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
