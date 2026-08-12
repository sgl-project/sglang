#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

// Keep the CUDA implementation byte-for-byte identical to the AOT baseline.
#include "gptq_kernel_impl.cuh"

namespace sglang {

void gptq_gemm_jit(
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b_q_weight,
    tvm::ffi::TensorView b_gptq_qzeros,
    tvm::ffi::TensorView b_gptq_scales,
    tvm::ffi::TensorView b_g_idx,
    tvm::ffi::TensorView c,
    tvm::ffi::TensorView temp_dq,
    bool use_shuffle,
    int64_t bit) {
  using namespace host;
  auto M = SymbolicSize{"M"};
  auto K = SymbolicSize{"K"};
  auto N = SymbolicSize{"N"};
  auto packed_k = SymbolicSize{"packed_k"};
  auto groups = SymbolicSize{"groups"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({M, K}).with_dtype<fp16_t>().with_device<kDLCUDA>(device).verify(a);
  TensorMatcher({packed_k, N}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(b_q_weight);
  TensorMatcher({groups, -1}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(b_gptq_qzeros);
  TensorMatcher({groups, N}).with_dtype<fp16_t>().with_device<kDLCUDA>(device).verify(b_gptq_scales);
  TensorMatcher({K}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(b_g_idx);
  TensorMatcher({M, N}).with_dtype<fp16_t>().with_device<kDLCUDA>(device).verify(c);
  TensorMatcher({K, N}).with_dtype<fp16_t>().with_device<kDLCUDA>(device).verify(temp_dq);
  CHECK_HOST(bit == 2 || bit == 3 || bit == 4 || bit == 8) << "unsupported GPTQ bit width " << bit;
  CHECK_HOST(packed_k.unwrap() * 32 == K.unwrap() * bit) << "inconsistent packed K dimension";

  static cublasHandle_t handle = [] {
    cublasHandle_t value;
    CHECK_HOST(cublasCreate(&value) == CUBLAS_STATUS_SUCCESS) << "failed to create cuBLAS handle";
    return value;
  }();
  auto stream = LaunchKernel::resolve_device(device.unwrap());
  CHECK_HOST(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS) << "failed to set cuBLAS stream";
  gptq::gemm_half_q_half_cuda(
      handle,
      static_cast<const half*>(a.data_ptr()),
      static_cast<const uint32_t*>(b_q_weight.data_ptr()),
      static_cast<const uint32_t*>(b_gptq_qzeros.data_ptr()),
      static_cast<const half*>(b_gptq_scales.data_ptr()),
      static_cast<const int*>(b_g_idx.data_ptr()),
      static_cast<half*>(c.data_ptr()),
      static_cast<half*>(temp_dq.data_ptr()),
      M.unwrap(),
      N.unwrap(),
      K.unwrap(),
      groups.unwrap(),
      use_shuffle,
      bit);
}

void gptq_shuffle_jit(tvm::ffi::TensorView q_weight, tvm::ffi::TensorView q_perm, int64_t bit) {
  using namespace host;
  auto packed_k = SymbolicSize{"packed_k"};
  auto N = SymbolicSize{"N"};
  auto K = SymbolicSize{"K"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({packed_k, N}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(q_weight);
  TensorMatcher({K}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(q_perm);
  CHECK_HOST(packed_k.unwrap() * 32 == K.unwrap() * bit) << "inconsistent packed K dimension";
  gptq::shuffle_exllama_weight(
      static_cast<uint32_t*>(q_weight.data_ptr()), static_cast<int*>(q_perm.data_ptr()), K.unwrap(), N.unwrap(), bit);
}

}  // namespace sglang
