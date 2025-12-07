# Development Guide for JIT Kernels

## Environment Setup

We strongly recommend using `clangd` as the language server for JIT kernel development.
For Ubuntu/Debian, you can download clangd from [apt.llvm.org](https://apt.llvm.org/).
If you are using VS Code, we recommend installing the `clangd` extension for better IDE integration.

All JIT-related files are located in `python/sglang/jit_kernel`.
Unlike `sgl-kernel`, which compiles CUDA/C++ binaries ahead of time (AOT), just-in-time (JIT) kernels are compiled at runtime.
Consequently, a static `compile_commands.json` cannot be generated.
To enable code completion with `clangd`, run `python -m sglang.jit_kernel` to generate a `.clangd` configuration file in your current directory.
After generating the file, restart the clangd language server. It should now recognize all JIT kernel files.

## Adding a New Kernel

### C++ Implementation

C++ source code is located in `python/sglang/jit_kernel/csrc`.
Reusable functions should be placed in `python/sglang/jit_kernel/include`.

We use [tvm-ffi](https://github.com/apache/tvm-ffi) for efficient foreign language bindings.
Refer to the [documentation](https://tvm.apache.org/ffi/) for advanced usage, such as exporting C++ objects.
Typically, `tvm::ffi::TensorView` is sufficient for passing PyTorch Tensors from Python.

### Python Interface

Python interfaces are defined in `python/sglang/jit_kernel`.
The `load_jit` utility function in `python/sglang/jit_kernel/utils.py` loads and returns the compiled module.
To export a C++ function (e.g., `cpp_func`), pass `cuda_wrappers=[("func", "cpp_func")]` to `load_jit`.
The function can then be called in Python as `module.func`.

### C++ Utilities

The following C++ utilities are available:

#### Runtime Checking

`RuntimeCheck` validates conditions at runtime. It accepts optional arguments for error reporting.
If the check fails, these arguments are output to aid debugging.
`RuntimeDeviceCheck` verifies the status of the last kernel launch.

```C++
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

void test() {
  host::RuntimeCheck(1 + 1 == 2, 1 + 1, " != ", 2);
  host::RuntimeDeviceCheck();
  // check the provided `cudaError_t`
  host::RuntimeDeviceCheck(cudaGetLastError());
}

```

#### Tensor Checking

`TensorMatcher` provides a readable way to validate and extract tensor shape information.

```cpp
#include <sgl_kernel/tensor.h>

void test(const tvm::ffi::TensorView k_cache, const tvm::ffi::TensorView v_cache) {
  using namespace host;

  auto D = SymbolicSize{"D"};  // cache dimension
  auto N = SymbolicSize{"N"};  // kvcache stride
  auto dtype = SymbolicDType{};
  auto device = SymbolicDevice{};

  TensorMatcher({-1, D})  //
      .with_strides({N, 1})
      .with_dtype<int32_t, int64_t>(dtype)
      .with_device<kDLCUDA, kDLCPU>(device)
      .verify(k_cache)
      .verify(v_cache);
}
```

Configure the `TensorMatcher` with expected stride, dtype, and device properties before verification.
- If `with_strides` is omitted, the tensor is expected to be contiguous.
- Template arguments in `with_dtype` restrict the allowed data types.
- Template arguments in `with_device` restrict the allowed devices.
- Values passed to `with_xxx` methods enforce equality checks.
- Passing `-1` for size or stride allows matching any value.

A `Symbolic` variable must resolve to the same value across all verifications.
Use `.unwrap()` to retrieve the matched value after verification.

> Note: `TensorMatcher` is a temporary expression and should not be stored in a variable.
> Tip: Add `//` at the end of the `TensorMatcher` chain to enforce proper indentation.

#### Kernel Launching

`LaunchKernel::resolve_device` retrieves the current `cudaStream` from PyTorch.
Kernels can be launched directly using `LaunchKernel`.

```cpp
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>

__global__ void kernel() {}

void test() {
  const auto num_blocks = 1;
  const auto num_threads = 32;
  const auto dynamic_smem = 0;

  DLDevice dev;  // suppose this is initialized properly
  host::LaunchKernel(num_blocks, num_threads, dev)(kernel);

  cudaStream_t stream = host::LaunchKernel::resolve_device(dev);
  host::LaunchKernel(num_blocks, num_threads, stream, dynamic_smem)(kernel);
}

```
