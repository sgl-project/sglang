#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace {

template <typename T>
__device__ __forceinline__ float cast_to_float(T v) {
  if constexpr (std::is_same_v<T, nv_bfloat16>) {
    return __bfloat162float(v);
  } else if constexpr (std::is_same_v<T, half>) {
    return __half2float(v);
  } else {
    return static_cast<float>(v);
  }
}

template <bool kFlipSinToCos, typename TIn>
__global__ void timestep_embedding_kernel(
    const TIn* __restrict__ t_ptr,
    float* __restrict__ output_ptr,
    int dim,
    float neg_log_max_period,
    float scale,
    int batch_size) {
  int row_idx = static_cast<int>(blockIdx.x * blockDim.y + threadIdx.y);
  if (row_idx >= batch_size) {
    return;
  }

  float t_val = cast_to_float(t_ptr[row_idx]);
  float* output_batch_base_ptr = output_ptr + row_idx * dim;

  int half_dim = dim / 2;
  int thread_offset = static_cast<int>(threadIdx.x);
  while (thread_offset * 4 < half_dim) {
    float4* top_half;
    float4* bottom_half;
    if constexpr (!kFlipSinToCos) {
      bottom_half = reinterpret_cast<float4*>(output_batch_base_ptr + thread_offset * 4);
      top_half = reinterpret_cast<float4*>(output_batch_base_ptr + half_dim + thread_offset * 4);
    } else {
      top_half = reinterpret_cast<float4*>(output_batch_base_ptr + thread_offset * 4);
      bottom_half = reinterpret_cast<float4*>(output_batch_base_ptr + half_dim + thread_offset * 4);
    }

    float4 vals;
    vals.x = scale * t_val * expf(neg_log_max_period * __int2float_rn(thread_offset * 4 + 0));
    vals.y = scale * t_val * expf(neg_log_max_period * __int2float_rn(thread_offset * 4 + 1));
    vals.z = scale * t_val * expf(neg_log_max_period * __int2float_rn(thread_offset * 4 + 2));
    vals.w = scale * t_val * expf(neg_log_max_period * __int2float_rn(thread_offset * 4 + 3));

    float4 sin_vals;
    sin_vals.x = cosf(vals.x);
    sin_vals.y = cosf(vals.y);
    sin_vals.z = cosf(vals.z);
    sin_vals.w = cosf(vals.w);
    *top_half = sin_vals;

    float4 cos_vals;
    cos_vals.x = sinf(vals.x);
    cos_vals.y = sinf(vals.y);
    cos_vals.z = sinf(vals.z);
    cos_vals.w = sinf(vals.w);
    *bottom_half = cos_vals;

    thread_offset += static_cast<int>(blockDim.x);
  }
}

template <typename TIn>
inline void launch_timestep_embedding(
    const tvm::ffi::TensorView t,
    const tvm::ffi::TensorView output,
    int dim,
    bool flip_sin_to_cos,
    float downscale_freq_shift,
    float scale,
    int max_period) {
  using namespace host;

  const int batch_size = static_cast<int>(t.shape()[0]);
  const int half_dim = dim / 2;

  constexpr int kMaxThreadsPerBlock = 1024;
  constexpr int kMinThreadsPerBlock = 128;

  const int num_threads_per_row = std::min(kMaxThreadsPerBlock, half_dim / 4);
  const int num_rows = (kMinThreadsPerBlock + num_threads_per_row - 1) / num_threads_per_row;

  dim3 grid((batch_size + num_rows - 1) / num_rows);
  dim3 block(num_threads_per_row, num_rows);

  const float neg_log_max_period =
      std::log(static_cast<float>(max_period)) * (-1.0f) / (static_cast<float>(half_dim) - downscale_freq_shift);

  const DLDevice device = output.device();

  if (flip_sin_to_cos) {
    LaunchKernel(grid, block, device)(
        timestep_embedding_kernel<true, TIn>,
        static_cast<const TIn*>(t.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        dim,
        neg_log_max_period,
        scale,
        batch_size);
  } else {
    LaunchKernel(grid, block, device)(
        timestep_embedding_kernel<false, TIn>,
        static_cast<const TIn*>(t.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        dim,
        neg_log_max_period,
        scale,
        batch_size);
  }
}

void timestep_embedding(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView output,
    int dim,
    bool flip_sin_to_cos,
    float downscale_freq_shift,
    float scale,
    int max_period) {
  using namespace host;

  auto B = SymbolicSize{"batch_size"};
  auto D = SymbolicSize{"dim"};
  auto device = SymbolicDevice{};

  TensorMatcher({B}).with_strides({1}).template with_device<kDLCUDA>(device).verify(input);

  TensorMatcher({B, D}).with_strides({D, 1}).with_dtype<float>().template with_device<kDLCUDA>(device).verify(output);

  RuntimeCheck(D.unwrap() == dim, "Output dim mismatch: ", D.unwrap(), " vs ", dim);
  RuntimeCheck(dim % 8 == 0, "dim must align to 8, got ", dim);

  const DLDataType in_dtype = input.dtype();

  const bool input_dtype_supported = (in_dtype.code == kDLFloat && in_dtype.bits == 16) ||
                                     (in_dtype.code == kDLBfloat && in_dtype.bits == 16) ||
                                     (in_dtype.code == kDLFloat && in_dtype.bits == 32);
  RuntimeCheck(input_dtype_supported, "input dtype must be fp16/bf16/fp32, but got ", in_dtype);

  auto launch = [&]<typename TIn>() {
    launch_timestep_embedding<TIn>(input, output, dim, flip_sin_to_cos, downscale_freq_shift, scale, max_period);
  };

  if (in_dtype.code == kDLFloat && in_dtype.bits == 32) {
    launch.template operator()<float>();
  } else if (in_dtype.code == kDLBfloat && in_dtype.bits == 16) {
    launch.template operator()<nv_bfloat16>();
  } else if (in_dtype.code == kDLFloat && in_dtype.bits == 16) {
    launch.template operator()<half>();
  }
}

}  // namespace
