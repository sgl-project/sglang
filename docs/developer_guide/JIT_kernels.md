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

## Code Structure

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

#### Integer Range

Similar to PyTorch, we provide an `irange` function to represent an integer range.

```C++
#include <sgl_kernel/utils.h>

void test() {
  for (auto i : host::irange(100)) { // [0, 100)
    // do something
  }
  for (auto i : host::irange(0, 100)) { // [0, 100)
    // do something
  }
}

```

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
Kernels can also be launched directly using `LaunchKernel`.

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

## Add new kernels

This section walks through a complete, end-to-end example of adding a new JIT kernel to the system.
We use a simple add_constant kernel as a running example, which adds a constant integer value to every element of an input tensor.

Conceptually, the Python interface looks like this:

```python
def add_constant(src: torch.Tensor, c: int):
    return src + c
```

### STEP 1: Write the C++ kernel

Write your CUDA kernel in [jit_kernel/csrc/add_constant.cuh](../../python/sglang/jit_kernel/csrc/add_constant.cuh). For demonstration purposes, we pass the constant value as a template parameter.

```cpp
#include <sgl_kernel/tensor.h>   // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.cuh>  // For LaunchKernel
#include <sgl_kernel/utils.h>    // For div_ceil, RuntimeCheck

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

template <int32_t kConstant>
__global__ void add_constant_kernel(int32_t* dst, const int32_t* src, size_t length) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    dst[idx] = src[idx] + kConstant;
  }
}

constexpr size_t kBlockSize = 256;

// You can also use struct with static method as an alternative
template <int32_t kConstant>
void add_constant(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) {
  using namespace host;

  // 1. Validate input tensors
  SymbolicSize N = {"num_elements"};
  SymbolicDevice device_;
  TensorMatcher({N})                  // 1D tensor, must be contiguous
      .with_dtype<int32_t>()          // must be int32
      .with_device<kDLCUDA>(device_)  // must be on CUDA device
      .verify(dst)                    // check tensor dst
      .verify(src);                   // check tensor src

  // 2. Extract required parameters, prepare for kernel launch
  const size_t num_elements = N.unwrap();
  const size_t grid_size = div_ceil(num_elements, kBlockSize);
  const DLDevice device = device_.unwrap();
  // some extra runtime checks using host::RuntimeCheck
  RuntimeCheck(num_elements > 0, "We only support non-empty tensors, got num_elements = ", num_elements);

  // 3. Launch the kernel. Error code will be automatically checked.
  LaunchKernel(grid_size, kBlockSize, device /*, dynamic_smem*/)(
      // kernel function
      add_constant_kernel<kConstant>,
      // kernel arguments
      static_cast<int32_t*>(dst.data_ptr()),
      static_cast<int32_t*>(src.data_ptr()),
      num_elements);
}

}  // namespace

```

### STEP 2: Create Python Interfaces

Next, expose the kernel through a Python wrapper.
Create a new file at [jit_kernel/add_constant.py](../../python/sglang/jit_kernel/add_constant.py) and expose the needed interfaces.

```python
from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@functools.cache
def _jit_add_constant_module(constant: int) -> Module:
    args = make_cpp_args(constant)  # pass all the template argument
    return load_jit(
        "add_constant",
        *args,
        cuda_files=["add_constant.cuh"],
        cuda_wrappers=[("add_constant", f"add_constant<{args}>")],
    )


def add_constant(src: torch.Tensor, constant: int) -> torch.Tensor:
    dst = torch.empty_like(src)
    module = _jit_add_constant_module(constant)
    module.add_constant(dst, src)
    return dst

```

### STEP 3: Use your kernel

Finally, import and use the kernel like a regular Python function:

```python
from sglang.jit_kernel.add_constant import add_constant
```

For a complete, runnable example, refer to [test_add_constant.py](../../python/sglang/jit_kernel/test_add_constant.py).
