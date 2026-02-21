#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck
#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

template <int32_t kN>
struct InputArray {
  int32_t values[kN];
};

template <int32_t kN>
__global__ void copy_to_gpu_no_ce_kernel(InputArray<kN> input_array, int32_t* output) {
  int32_t idx = static_cast<int32_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (idx < kN) {
    output[idx] = input_array.values[idx];
  }
}

// Copy a small int32 CPU tensor to GPU without using the copy engine.
// The CPU data is packed into a kernel-argument struct (passed by value),
// so the transfer goes through the kernel launch path instead of DMA.
template <int32_t kN>
void copy_to_gpu_no_ce(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
  using namespace host;

  // Validate CPU input tensor
  TensorMatcher({kN})
      .with_dtype<int32_t>()
      .with_device<kDLCPU>()
      .verify(input);

  // Validate CUDA output tensor, capture device for stream resolution
  SymbolicDevice cuda_device;
  TensorMatcher({kN})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(cuda_device)
      .verify(output);

  // Pack CPU data into a by-value struct to avoid copy engine
  InputArray<kN> input_array;
  const int32_t* input_ptr = static_cast<const int32_t*>(input.data_ptr());
  for (int32_t i = 0; i < kN; ++i) {
    input_array.values[i] = input_ptr[i];
  }

  // Launch kernel: one block, kN threads
  const DLDevice device = cuda_device.unwrap();
  LaunchKernel(dim3(1), dim3(static_cast<uint32_t>(kN)), device)(
      copy_to_gpu_no_ce_kernel<kN>,
      input_array,
      static_cast<int32_t*>(output.data_ptr()));
}

}  // namespace
