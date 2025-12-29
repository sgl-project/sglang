#include <sgl_kernel/fp8_utils.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <flashinfer/vec_dtypes.cuh>

namespace {

using device::atomicMaxFloat;
using device::blockReduceMax;
using device::FP8_E4M3_MAX;

template <typename T>
__global__ void
per_tensor_absmax_kernel(const T* __restrict__ input, float* __restrict__ output_s, const int64_t num_elements) {
  float max_value = 0.0f;
  unsigned int tid = threadIdx.x;
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_size = blockDim.x * gridDim.x;

  constexpr uint32_t vec_size = 16 / sizeof(T);
  using vec_t = flashinfer::vec_t<T, vec_size>;

  const int32_t num_vec_elems = num_elements / vec_size;

  for (int32_t i = gid; i < num_vec_elems; i += grid_size) {
    vec_t input_vec;
    input_vec.cast_load(input + i * vec_size);

#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      float val = static_cast<float>(input_vec[j]);
      max_value = fmaxf(max_value, fabsf(val));
    }
  }

  const int32_t remaining_start = num_vec_elems * vec_size;
  for (int32_t idx = remaining_start + gid; idx < num_elements; idx += grid_size) {
    float val = static_cast<float>(input[idx]);
    max_value = fmaxf(max_value, fabsf(val));
  }

  max_value = blockReduceMax(max_value);

  if (tid == 0) {
    atomicMaxFloat(output_s, max_value / FP8_E4M3_MAX);
  }
}

template <typename T, typename DST_DTYPE>
__global__ void per_tensor_quant_fp8_kernel(
    const T* __restrict__ input,
    DST_DTYPE* __restrict__ output,
    const float* __restrict__ scale,
    const int64_t num_elements) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_size = blockDim.x * gridDim.x;
  const float scale_val = 1.0f / (*scale);

  const uint32_t VEC_SIZE = 16;
  using vec_t = flashinfer::vec_t<T, VEC_SIZE>;

  const int32_t num_vec_elems = num_elements / VEC_SIZE;

  for (int32_t i = gid; i < num_vec_elems; i += grid_size) {
    vec_t input_vec;
    input_vec.cast_load(input + i * VEC_SIZE);

    DST_DTYPE output_arr[VEC_SIZE];
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      float val = fmax(fmin(static_cast<float>(input_vec[j]) * scale_val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
      output_arr[j] = static_cast<DST_DTYPE>(val);
#else
      output_arr[j] = c10::Float8_e4m3fnuz(
          __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
          c10::Float8_e4m3fnuz::from_bits());
#endif
    }
    *(uint4*)(output + i * VEC_SIZE) = *(uint4*)output_arr;
  }

  const int32_t remaining_start = num_vec_elems * VEC_SIZE;
  for (int32_t idx = remaining_start + gid; idx < num_elements; idx += grid_size) {
    float val = fmax(-FP8_E4M3_MAX, fmin(static_cast<float>(input[idx]) * scale_val, FP8_E4M3_MAX));
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
    output[idx] = static_cast<DST_DTYPE>(val);
#else
    output[idx] = c10::Float8_e4m3fnuz(
        __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
        c10::Float8_e4m3fnuz::from_bits());
#endif
  }
}

constexpr size_t kBlockSize = 256;

template <bool kIsStatic>
void per_tensor_quant_fp8(tvm::ffi::TensorView input, tvm::ffi::TensorView output_q, tvm::ffi::TensorView output_s) {
  using namespace host;

  SymbolicSize num_tokens = {"num_tokens"};
  SymbolicSize hidden_dim = {"hidden_dim"};
  SymbolicDevice device_;
  SymbolicDType input_dtype;

  TensorMatcher({num_tokens, hidden_dim})  //
      .with_dtype<float, __half, __nv_bfloat16>(input_dtype)
      .with_device<kDLCUDA>(device_)
      .verify(input);

  TensorMatcher({num_tokens, hidden_dim})  //
      .with_dtype<__nv_fp8_e4m3>()
      .with_device<kDLCUDA>(device_)
      .verify(output_q);

  TensorMatcher({1})  //
      .with_dtype<float>()
      .with_device<kDLCUDA>(device_)
      .verify(output_s);

  const size_t total_elements = num_tokens.unwrap() * hidden_dim.unwrap();
  const size_t num_blocks = std::min((total_elements + kBlockSize - 1) / kBlockSize, size_t(1024));
  const DLDevice device = device_.unwrap();

  RuntimeCheck(total_elements > 0, "Input tensor must be non-empty");

  auto launch_kernels = [&]<typename T>() {
    if constexpr (!kIsStatic) {
      LaunchKernel(num_blocks, kBlockSize, device)(
          per_tensor_absmax_kernel<T>,
          static_cast<const T*>(input.data_ptr()),
          static_cast<float*>(output_s.data_ptr()),
          static_cast<int64_t>(total_elements));
    }

    LaunchKernel(num_blocks, kBlockSize, device)(
        per_tensor_quant_fp8_kernel<T, __nv_fp8_e4m3>,
        static_cast<const T*>(input.data_ptr()),
        static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
        static_cast<const float*>(output_s.data_ptr()),
        static_cast<int64_t>(total_elements));
  };

  const DLDataType dtype = input_dtype.unwrap();
  if (dtype.code == kDLFloat && dtype.bits == 32) {
    launch_kernels.template operator()<float>();
  } else if (dtype.code == kDLBfloat && dtype.bits == 16) {
    launch_kernels.template operator()<__nv_bfloat16>();
  } else if (dtype.code == kDLFloat && dtype.bits == 16) {
    launch_kernels.template operator()<__half>();
  }
}

}  // namespace
