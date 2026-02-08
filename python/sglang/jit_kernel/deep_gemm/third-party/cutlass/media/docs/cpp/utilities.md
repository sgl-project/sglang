![ALT](../../images/gemm-hierarchy-with-epilogue-no-labels.png "CUTLASS Code Organization")


Note: This document discusses utilities commonly used with code that targets CUTLASS 2.x.
Although CUTLASS 3.0's primary entry point APIs do not transact in these `cutlass::*` tensor types anymore,
users can still find them convenient for managing allocations with trivial affine layouts.
For more advanced host side tensor management, [`cute::Tensor`](cute/03_tensor.md)s
can be used on either host or device for any memory space and full expressive power of
[`cute::Layout`](cute/01_layout.md)s.

# CUTLASS Utilities

CUTLASS utilities are additional template classes that facilitate recurring tasks. These are
flexible implementations of needed functionality, but they are not expected to be efficient.

Applications should configure their builds to list `/tools/util/include` in their include
paths.

Source code is in [`/tools/util/include/cutlass/util/`](https://github.com/NVIDIA/cutlass/tree/main/tools/util/include/cutlass/util).

## Tensor Allocation and I/O

To allocate a tensor with storage in both host and device memory, use `HostTensor` in
[`cutlass/util/host_tensor.h`](https://github.com/NVIDIA/cutlass/tree/main/tools/util/include/cutlass/util/host_tensor.h)

```c++
template <typename Element, typename Layout>
class HostTensor;
```

This class is compatible with all CUTLASS numeric data types and layouts.

**Example:** column-major matrix storage of single-precision elements.
```c++
#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>

int main() {
  int rows = 32;
  int columns = 16;

  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> tensor({rows, columns});

  return 0;
}
```

Internal host-side storage may be accessed via the following methods.
```c++
float *host_ptr = tensor.host_data();
cutlass::TensorRef<float, cutlass::layout::ColumnMajor> host_ref = tensor.host_ref();
cutlass::TensorView<float, cutlass::layout::ColumnMajor> host_view = tensor.host_view();
```

Device memory may be accessed similarly.
```c++
float *device_ptr = tensor.device_data();
cutlass::TensorRef<float, cutlass::layout::ColumnMajor> device_ref = tensor.device_ref();
cutlass::TensorView<float, cutlass::layout::ColumnMajor> device_view = tensor.device_view();
```

Printing to human-readable CSV output is accoplished with `std::ostream::operator<<()` defined in
[`cutlass/util/tensor_view_io.h`](https://github.com/NVIDIA/cutlass/tree/main/tools/util/include/cutlass/util/tensor_view_io.h). 
Note, this assumes all views refer to host memory.
```c++
#include <cutlass/util/tensor_view_io.h>

int main() {
  // Obtain a TensorView into host memory
  cutlass::TensorView<float, cutlass::layout::ColumnMajor> view = tensor.host_view();

  // Print to std::cout
  std::cout << view << std::endl;

  return 0;
}
```

Host and device memory must be explicitly synchronized by the application.
```c++
float idx = 0;

for (int i = 0; i < rows; ++i) {
  for (int j = 0; j < columns; ++j) {

    // Write the element at location {i, j} in host memory
    tensor.host_ref().at({i, j}) = idx;

    idx += 0.5f;
  } 
}

// Copy host memory to device memory
tensor.sync_device();

// Obtain a device pointer usable in CUDA kernels
float *device_ptr = tensor.device_data();
```

`HostTensor<>` is usable by all CUTLASS layouts including interleaved layouts.
```c++
int rows = 4;
int columns = 3;

cutlass::HostTensor<float, cutlass::layout::ColumnMajorInterleaved<4>> tensor({rows, columns});

for (int i = 0; i < rows; ++i) {
  for (int j = 0; j < columns; ++j) {

    // Write the element at location {i, j} in host memory
    tensor.host_ref().at({i, j}) = float(i) * 1.5f - float(j) * 2.25f;
  } 
}

std::cout << tensor.host_view() << std::endl;
```

## Device Allocations

To strictly allocate memory on the device using the smart pointer pattern to manage allocation and deallocation,
use `cutlass::DeviceAllocation<>`. 

**Example:** allocating an array in device memory.
```c++
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor_view.h>
#include <cutlass/util/device_memory.h>

__global__ void kernel(float *device_ptr) {

}

int main() {

  size_t N = 1024;

  cutlass::DeviceAllocation<float> device_alloc(N);

  // Call a CUDA kernel passing device memory as a pointer argument
  kernel<<< grid, block >>>(alloc.get());

  if (cudaGetLastError() != cudaSuccess) {
    return -1;
  }

  // Device memory is automatically freed when device_alloc goes out of scope

  return 0;
}
```

## Tensor Initialization

CUTLASS defines several utility functions to initialize tensors to uniform, procedural,
or randomly generated elements. These have implementations using strictly host code and
implementations using strictly CUDA device code.

`TensorFill()` for uniform elements throughout a tensor.
```c++
#include <cutlass/layout/matrix.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/host_tensor.h>

int main() {
  int rows = 128;
  int columns = 64;

  float x = 3.14159f;

  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> tensor({rows, columns});

  // Initialize in host memory
  cutlass::reference::host::TensorFill(tensor.host_view(), x);

  // Initialize in device memory
  cutlass::reference::device::TensorFill(tensor.device_view(), x);

  return 0;
}
```

`TensorFillRandomUniform()` for initializing elements to a random uniform distribution.
The device-side implementation uses CURAND to generate random numbers.
```c++
#include <cutlass/layout/matrix.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/host_tensor.h>

int main() {
  int rows = 128;
  int columns = 64;

  double maximum = 4;
  double minimum = -4;
  uint64_t seed = 0x2019;

  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> tensor({rows, columns});

  // Initialize in host memory
  cutlass::reference::host::TensorFillRandomUniform(
    tensor.host_view(),
    seed,
    maximum,
    minimum);

  // Initialize in device memory
  cutlass::reference::device::TensorFillRandomUniform(
    tensor.device_view(),
    seed,
    maximum,
    minimum);

  return 0;
}
```


`TensorFillRandomGaussian()` for initializing elements to a random gaussian distribution.
The device-side implementation uses CURAND to generate random numbers.
```c++
#include <cutlass/layout/matrix.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/host_tensor.h>

int main() {

  int rows = 128;
  int columns = 64;

  double mean = 0.5;
  double stddev = 2.0;
  uint64_t seed = 0x2019;

  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> tensor({rows, columns});

  // Initialize in host memory
  cutlass::reference::host::TensorFillRandomGaussian(
    tensor.host_view(),
    seed,
    mean,
    stddev);

  // Initialize in device memory
  cutlass::reference::device::TensorFillRandomGaussian(
    tensor.device_view(),
    seed,
    mean,
    stddev);

  return 0;
}
```

Each of these functions accepts an additional argument to specify how many bits of
the mantissa less than 1 are non-zero. This simplifies functional comparisons when
exact random distributions are not necessary, since elements may be restricted to
integers or values with exact fixed-point representations.

```c++
#include <cutlass/layout/matrix.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/host_tensor.h>

int main() {

  int rows = 128;
  int columns = 64;

  double mean = 0.5;
  double stddev = 2.0;
  uint64_t seed = 0x2019;

  int bits_right_of_binary_decimal = 2;

  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> tensor({rows, columns});

  // Initialize in host memory
  cutlass::reference::host::TensorFillRandomGaussian(
    tensor.host_view(),
    seed,
    mean,
    stddev,
    bits_right_of_binary_decimal);

  // Initialize in device memory
  cutlass::reference::device::TensorFillRandomGaussian(
    tensor.device_view(),
    seed,
    mean,
    stddev,
    bits_right_of_binary_decimal);

  return 0;
}
```

These utilities may be used for all data types.

**Example:** random half-precision tensor with Gaussian distribution.
```c++
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/host_tensor.h>

int main() {
  int rows = 128;
  int columns = 64;

  double mean = 0.5;
  double stddev = 2.0;
  uint64_t seed = 0x2019;

  // Allocate a column-major tensor with half-precision elements
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> tensor({rows, columns});

  // Initialize in host memory
  cutlass::reference::host::TensorFillRandomGaussian(
    tensor.host_view(),
    seed,
    mean,
    stddev);

  // Initialize in device memory
  cutlass::reference::device::TensorFillRandomGaussian(
    tensor.device_view(),
    seed,
    mean,
    stddev);

  return 0;
}
```

## Reference Implementations

CUTLASS defines reference implementations usable with all data types and layouts. These are
used throughout the unit tests.

**Example:** Reference GEMM implementation with mixed precision internal computation.
```c++
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>

#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm.h>

int main() {

  int M = 64;
  int N = 32;
  int K = 16;

  float alpha = 1.5f;
  float beta = -1.25f;

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A({M, K});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B({K, N});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C({M, N});

  cutlass::reference::host::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,   // ElementA and LayoutA
    cutlass::half_t, cutlass::layout::ColumnMajor,   // ElementB and LayoutB
    cutlass::half_t, cutlass::layout::ColumnMajor,   // ElementC and LayoutC
    float,                                           // scalar type (alpha and beta)
    float> gemm_op;                                  // internal accumulation type

  gemm_op(
    {M, N, K},             // problem size
    alpha,                 // alpha scalar
    A.host_view(),         // TensorView to host memory
    B.host_view(),         // TensorView to host memory
    beta,                  // beta scalar
    C.host_view(),         // TensorView to host memory
    D.host_view());        // TensorView to device memory

  return 0;
}
```

## Debugging Asynchronous Kernels with CUTLASS's Built-in `synclog` Tool

CUTLASS provides a built-in tool called `synclog` that enables printing runtime information useful for debugging asynchronous CUTLASS kernels. With the introduction of Warp Specialization in CUTLASS 3.0 for Hopper GPUs, kernel designs now incorporate synchronization among warps. The `synclog` tool simplifies debugging efforts for these asynchronous programs by recording and displaying timing information for synchronization events.

### Enabling `synclog`
To enable `synclog`, add the -DCUTLASS_ENABLE_SYNCLOG=1 flag during compilation. From the CUTLASS root directory:

```
$ mkdir build && cd build && 
$ cmake .. -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_ENABLE_SYNCLOG=1
```

### Building and Running with `synclog`
After enabling `synclog`, build your CUTLASS example. For instance, to build example 54:

```
$ cd examples/54_hopper_fp8_warp_specialized_gemm
$ make
```

Run the example, setting the profiling iteration count to 0 to ensure `synclog` information is printed only for the reference run:

```
$ ./54_hopper_fp8_warp_specialized_gemm --iterations=0 &> synclog.txt
```

### Interpreting `synclog` output
The synclog.txt file will contain runtime information about synchronization events. Here's a sample output snippet:

```
synclog start
synclog at 1: cluster_barrier_init line=281 time=1725400116233388736 thread=0,0,0 block=0,0,0 smem_addr=197632 arrive_count=1
synclog at 13: fence_barrier_init line=583 time=1725400116233388768 thread=32,0,0 block=0,0,0 
...
```

Each line in the main body follows this format:
```
synclog at [synclog_at]: [header] line=[line] thread=[threadIdx.xyz] block=[blockIdx.xyz] 
```
* `synclog at`: Address in the `synclog` output buffer (in bytes). Output exceeding 2^26 bytes is discarded.
* `header`: Name of the synchronization event.
* `line`: Code line number of the synchronization operation calling into `synclog`.

Additional information may appear at the end of each line, such as shared memory address, phase bit, and arrive count. For more detailed information on `synclog` output, refer to [synclog.hpp](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/arch/synclog.hpp) in the CUTLASS source code. 

Please note that `synclog` is an experimental feature, and its functionality is not always guaranteed. We encourage its use in custom kernels and CUTLASS examples, though it is known to be incompatible with profiler kernels.

### Copyright

Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
