# JIT Kernel Support for AMD SGLang

The sglang wheel includes JIT (Just-In-Time) kernel compilation support. JIT kernels allow for dynamic compilation of optimized CUDA/HIP kernels at runtime.

## Requirements

JIT kernel compilation requires:

1. **apache-tvm-ffi** - Included in the `runtime_common` dependencies (installed with `sglang[all-hip,...]`)
2. **System compiler toolchain** - A C++ compiler compatible with your ROCm installation
   - For ROCm environments, this is typically provided by the ROCm installation
   - Ensure `hipcc` is available in your PATH

The JIT kernel source files (`.cuh`, `.cu` headers) are bundled with the wheel and will be available at runtime for compilation.

## Verification

To verify JIT kernel support is working:
```python
from sglang.jit_kernel.utils import KERNEL_PATH
print(f"JIT kernel path: {KERNEL_PATH}")
# Should print the path to site-packages/sglang/jit_kernel
```

## Installation

The `apache-tvm-ffi` dependency is automatically installed when you install sglang with the runtime dependencies:

```bash
pip install "sglang[all-hip,rocm720]" --extra-index-url https://aioss-pypi-prod.s3.amazonaws.com/sglang/rocm720/simple/
```

This ensures all necessary dependencies for JIT kernel compilation are available.