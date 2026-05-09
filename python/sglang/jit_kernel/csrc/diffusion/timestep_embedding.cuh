#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <type_traits>

namespace {

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

  float t_val = device::cast<float>(t_ptr[row_idx]);
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
    vals.x = scale * t_val * device::math::exp(neg_log_max_period * __int2float_rn(thread_offset * 4 + 0));
    vals.y = scale * t_val * device::math::exp(neg_log_max_period * __int2float_rn(thread_offset * 4 + 1));
    vals.z = scale * t_val * device::math::exp(neg_log_max_period * __int2float_rn(thread_offset * 4 + 2));
    vals.w = scale * t_val * device::math::exp(neg_log_max_period * __int2float_rn(thread_offset * 4 + 3));

    float4 cos_vals;
    cos_vals.x = device::math::cos(vals.x);
    cos_vals.y = device::math::cos(vals.y);
    cos_vals.z = device::math::cos(vals.z);
    cos_vals.w = device::math::cos(vals.w);
    *top_half = cos_vals;

    float4 sin_vals;
    sin_vals.x = device::math::sin(vals.x);
    sin_vals.y = device::math::sin(vals.y);
    sin_vals.z = device::math::sin(vals.z);
    sin_vals.w = device::math::sin(vals.w);
    *bottom_half = sin_vals;

    thread_offset += static_cast<int>(blockDim.x);
  }
}

template <bool kFlipSinToCos, typename TIn>
__global__ void timestep_embedding_sincos_kernel(
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

  float t_val = device::cast<float>(t_ptr[row_idx]);
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
    vals.x = scale * t_val * device::math::exp(neg_log_max_period * __int2float_rn(thread_offset * 4 + 0));
    vals.y = scale * t_val * device::math::exp(neg_log_max_period * __int2float_rn(thread_offset * 4 + 1));
    vals.z = scale * t_val * device::math::exp(neg_log_max_period * __int2float_rn(thread_offset * 4 + 2));
    vals.w = scale * t_val * device::math::exp(neg_log_max_period * __int2float_rn(thread_offset * 4 + 3));

    float4 sin_vals;
    float4 cos_vals;
    sincosf(vals.x, &sin_vals.x, &cos_vals.x);
    sincosf(vals.y, &sin_vals.y, &cos_vals.y);
    sincosf(vals.z, &sin_vals.z, &cos_vals.z);
    sincosf(vals.w, &sin_vals.w, &cos_vals.w);
    *top_half = cos_vals;
    *bottom_half = sin_vals;

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

  // sincosf is profitable for large batches where trig work dominates launch
  // overhead. Keep smaller and very wide cases on the original sinf/cosf path.
  const bool use_sincos = batch_size >= 16384 && dim <= 1024;

  if (flip_sin_to_cos && use_sincos) {
    LaunchKernel(grid, block, device)(
        timestep_embedding_sincos_kernel<true, TIn>,
        static_cast<const TIn*>(t.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        dim,
        neg_log_max_period,
        scale,
        batch_size);
  } else if (flip_sin_to_cos) {
    LaunchKernel(grid, block, device)(
        timestep_embedding_kernel<true, TIn>,
        static_cast<const TIn*>(t.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        dim,
        neg_log_max_period,
        scale,
        batch_size);
  } else if (use_sincos) {
    LaunchKernel(grid, block, device)(
        timestep_embedding_sincos_kernel<false, TIn>,
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

template <typename TIn>
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

  TensorMatcher({B})  // input
      .with_strides({1})
      .with_dtype<TIn>()
      .template with_device<kDLCUDA>(device)
      .verify(input);

  TensorMatcher({B, D}).with_strides({D, 1}).with_dtype<float>().template with_device<kDLCUDA>(device).verify(output);

  RuntimeCheck(D.unwrap() == dim, "Output dim mismatch: ", D.unwrap(), " vs ", dim);
  RuntimeCheck(dim % 8 == 0, "dim must align to 8, got ", dim);

  launch_timestep_embedding<TIn>(input, output, dim, flip_sin_to_cos, downscale_freq_shift, scale, max_period);
}

}  // namespace
