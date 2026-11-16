// Shared parts of the tiny GEMM family `out[m, n] = sum_k x[m, k] * w[n, k]` (bf16 x, bf16 or fp32 w,
// fp32 accumulation): the kernel parameters and the host-side input checks. `simt.cuh` holds the CUDA-core
// kernels (CUDA and ROCm), `mma.cuh` the tensor-core kernel (CUDA only).
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

struct TinyGEMMParams {
  const bf16_t* __restrict__ x;
  const void* __restrict__ w;
  void* __restrict__ out;
  int64_t stride_x;  // row stride of x, in elements
  uint32_t num_tokens;
};

namespace device::tiny_gemm {

// nvcc deficiency: after inline asm (the PDL wait) a generic pointer loses its address space and 32-byte
// loads split into generic LD.E.128s. Call this on the pointer right before the load, after the asm.
template <typename T>
SGL_DEVICE const T* assume_global(const T* ptr) {
#ifndef USE_ROCM
  __builtin_assume(__isGlobal(ptr));
#endif
  return ptr;
}

}  // namespace device::tiny_gemm

namespace host::tiny_gemm {

// Checks x [M, K] (row-strided), w [N, K] and out [M, N]; x rows are loaded as whole `x_align`-byte vectors.
template <uint32_t N, uint32_t K, typename Weight, typename Out>
inline TinyGEMMParams make_params(
    const tvm::ffi::TensorView x,
    const tvm::ffi::TensorView w,
    const tvm::ffi::TensorView out,
    int64_t x_align,
    int64_t w_align,
    DLDevice& device) {
  auto M = SymbolicSize{"num_tokens"};
  auto dev = SymbolicDevice{};
  dev.set_options<kDLGPU>();
  TensorMatcher({M, K})  //
      .with_strides({-1, 1})
      .with_dtype<bf16_t>()
      .with_device(dev)
      .ensure_alignment(x_align)
      .verify(x);
  TensorMatcher({N, K})  //
      .with_dtype<Weight>()
      .with_device(dev)
      .ensure_alignment(w_align)
      .verify(w);
  TensorMatcher({M, N})  //
      .with_dtype<Out>()
      .with_device(dev)
      .verify(out);
  device = dev.unwrap();
  return TinyGEMMParams{
      .x = static_cast<const bf16_t*>(x.data_ptr()),
      .w = w.data_ptr(),
      .out = out.data_ptr(),
      .stride_x = x.stride(0),
      .num_tokens = static_cast<uint32_t>(M.unwrap()),
  };
}

}  // namespace host::tiny_gemm

}  // namespace sglang
