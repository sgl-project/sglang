# sgl-kernel Metal kernels

Custom Apple Metal kernels for the MLX backend on Apple Silicon. Shader sources (`*.metal`) and C++ host / nanobind sources (`*.cpp`) in this directory are compiled by [`sgl-kernel/setup_metal.py`](../../setup_metal.py) into the `sgl_kernel._metal` extension and the `sgl_metal_kernels.metallib` archive, and exposed through Python wrappers in [`python/sgl_kernel/metal.py`](../../python/sgl_kernel/metal.py).

## Kernels

| Kernel | Description | Tested on |
| --- | --- | --- |
| `rope_pool_fused` | Fused NeoX RoPE for Q/K plus K/V scatter into the MLX KV pool. | Apple Silicon / MLX |


## Adding a new Metal kernel

1. Add the shader under `csrc/metal/<kernel>.metal`.
2. Add the C++ host / nanobind binding under `csrc/metal/<kernel>.cpp`, exporting the entry point on the `sgl_kernel._metal` module.
3. Append both files to `metal_shader_sources` and `cxx_sources` in [`sgl-kernel/setup_metal.py`](../../setup_metal.py).
4. Add a Python wrapper in [`python/sgl_kernel/metal.py`](../../python/sgl_kernel/metal.py) that validates input shapes/dtypes and calls `mx.eval` on its operands before invoking the AOT C++ entry point.
5. Add a test under [`sgl-kernel/tests/`](../../tests) and update the **Kernels** table above with a short description and the hardware / OS / MLX version the kernel was validated on.
