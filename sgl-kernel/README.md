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
## Build from source

Development build:

```bash
make build
```

Note:

The `sgl-kernel` is rapidly evolving. If you experience a compilation failure, try using `make rebuild`.

### Build with [ccache](https://github.com/ccache/ccache)
```bash
# or `yum install -y ccache`.
apt-get install -y ccache
# Building with ccache is enabled when ccache is installed and CCACHE_DIR is set.
export CCACHE_DIR=/path/to/your/ccache/dir
export CCACHE_BACKEND=""
export CCACHE_KEEP_LOCAL_STORAGE="TRUE"
unset CCACHE_READONLY
python -m uv build --wheel -Cbuild-dir=build --color=always .
```

### Configuring CMake Build Options
Cmake options can be configuring by adding `-Ccmake.define.<option>=<value>` to the `uv build` flags.
For example, to enable building FP4 kernels, use:
```bash
python -m uv build --wheel -Cbuild-dir=build -Ccmake.define.SGL_KERNEL_ENABLE_FP4=1 --color=always .
```
See CMakeLists.txt for more options.

### Parallel Build

We highly recommend you build sgl-kernel with Ninja. Ninja can automatically build sgl-kernel in parallel.
And if you build the sgl-kernel with cmake, you need to add `CMAKE_BUILD_PARALLEL_LEVEL` for parallel build like:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) python -m uv build --wheel -Cbuild-dir=build --color=always .
```

# Developer Guide

## Development Environment Setup

Use Docker to set up the development environment. See [Docker setup guide](https://github.com/sgl-project/sglang/blob/main/docs/references/development_guide_using_docker.md#setup-docker-container).

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

### FlashAttention FYI

  FA3 can fail without a enough shared memory for a some shapes, such as higher hidden_dim or some special cases. Right now, fa3 is supported for sm80/sm87 and sm86/sm89.

  The main different Between sm80/sm87 and sm86/sm89 is the shared memory size. you can follow the link below for more information https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-8-x.

  And for sgl-kernel right now, we can build fa3 on sm80/sm86/sm89/sm90a. That means if you use **A100(tested)**/A*0/**L20(tested)**/L40/L40s/**3090(tested)** you can use fa3.

### Kernel Development

Steps to add a new kernel:

1. Implement the kernel in [csrc](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc)
2. Expose the interface in [include/sgl_kernel_ops.h](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/include/sgl_kernel_ops.h)
3. Create torch extension in [csrc/common_extension.cc](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/common_extension.cc)
4. Update [CMakeLists.txt](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/CMakeLists.txt) to include new CUDA source
5. Expose Python interface in [python](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel)

### Development Tips

1. When implementing kernels in [csrc](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc), only define pure CUDA files and C++ interfaces. If you need to use `Torch::tensor`, use `<torch/all.h>` instead of `<torch/extension.h>`. Using `<torch/extension.h>` will cause compilation errors when using SABI.

2. When creating torch extensions, add the function definition with `m.def`, and device binding with `m.impl`:
- Using torch.compile need `m.def` with schema, it helps auto capture the custom kernel. Reference: [How to add FakeTensor](https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.ptttacy8y1u9)

- How to write schema: [Schema reference](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func)

   ```cpp
   // We need def with schema here for torch.compile
   m.def(
    "bmm_fp8(Tensor A, Tensor B, Tensor! D, Tensor A_scale, Tensor B_scale, Tensor workspace_buffer, int "
    "cublas_handle, int cuda_stream) -> ()");
   m.impl("bmm_fp8", torch::kCUDA, &bmm_fp8);
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

> The reason we need `double` and `int64_t` in torch binding is that TORCH_LIBRARY handles the `Python-to-C++` conversion process. Python's `float` data type actually corresponds to `double` in C++, while Python's `int` corresponds to `int64_t` in C++.

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
 m.impl("fwd", torch::kCUDA, make_pytorch_shim(&mha_fwd));
```

### Testing & Benchmarking

1. Add pytest tests in [tests/](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/tests), if you need to skip some test, please use `@pytest.mark.skipif`

```python
@pytest.mark.skipif(
    skip_condition, reason="Nvfp4 Requires compute capability of 10 or above."
)
```

2. Add benchmarks using [triton benchmark](https://triton-lang.org/main/python-api/generated/triton.testing.Benchmark.html) in [benchmark/](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/benchmark)
3. Run test suite

### FAQ

- When encountering this error while compiling using ccache: `ImportError: /usr/local/lib/python3.10/dist-packages/sgl_kernel/common_ops.abi3.so: undefined symbol: _ZN3c108ListType3getERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEENS_4Type24SingletonOrSharedTypePtrIS9_EE`, please modify the last command as follows to resolve it: `python3 -m uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation` .

### Release new version

Update version in [pyproject.toml](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/pyproject.toml) and [version.py](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel/version.py)
