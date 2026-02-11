#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>
#include <tvm/ffi/container/tensor.h>

#define kBitsToLoad 128
#define kBytesToLoad (kBitsToLoad / 8)

namespace {

namespace detail {

template <typename T>
SGL_DEVICE float to_f32(const T& x) {
#if USE_ROCM
  return castToFloat(x);
#else
  return static_cast<float>(x);
#endif
}

template <typename T>
SGL_DEVICE T from_f32(float f32) {
#if USE_ROCM
  return castFromFloat<T>(f32);
#else
  return static_cast<T>(f32);
#endif
}

}  // namespace detail

template <typename T>
SGL_DEVICE T silu(const T& x) {
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val / (1.0f + expf(-f32_val)));
}

template <typename T>
SGL_DEVICE T gelu(const T& x) {
  constexpr float kAlpha = M_SQRT1_2;
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val * (0.5f * (1.0f + erf(f32_val * kAlpha))));
}

// gelu_quick(x) = x * torch.sigmoid(1.702 * x)
template <typename T>
SGL_DEVICE T gelu_quick_act(const T& x) {
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val / (1.0f + expf(-f32_val * 1.702f)));
}

template <typename T>
SGL_DEVICE T gelu_tanh(const T& x) {
  constexpr float kAlpha = 0.044715f;
  constexpr float kBeta = 0.7978845608028654f;
  float f32_val = detail::to_f32(x);
  const float cdf = 0.5f * (1.0f + tanhf((kBeta * (f32_val + kAlpha * f32_val * f32_val * f32_val))));
  return detail::from_f32<T>(f32_val * cdf);
}

template <typename T, T (*Activation)(const T&)>
__global__ void act_and_mul_kernel(T* __restrict__ out_ptr, const T* __restrict__ input_ptr, int64_t d,
                                   int64_t num_tokens) {
  constexpr uint32_t vec_size = kBytesToLoad / sizeof(T);
  const int64_t token_idx = blockIdx.x;
  const int64_t thread_idx = threadIdx.x;
  const int64_t stride = blockDim.x;
  const int64_t offset = token_idx * 2 * d;

  if (token_idx >= num_tokens) return;

#pragma unroll 1
  for (uint32_t idx = thread_idx; idx < d / vec_size; idx += stride) {
    device::AlignedVector<T, vec_size> x_vec, y_vec, out_vec;
    x_vec.load(input_ptr + offset + idx * vec_size);
    y_vec.load(input_ptr + offset + d + idx * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      out_vec[i] = Activation(x_vec[i]) * y_vec[i];
    }
    out_vec.store(out_ptr + token_idx * d + idx * vec_size);
  }

  const int64_t remaining_offset = d - d % (stride * vec_size);
  // process the remaining elements
#pragma unroll 1
  for (int64_t idx = thread_idx; idx < d % (stride * vec_size); idx += stride) {
    T x = input_ptr[offset + remaining_offset + idx], y = input_ptr[offset + remaining_offset + d + idx];
    out_ptr[token_idx * d + remaining_offset + idx] = Activation(x) * y;
  }
}

template <typename T, T (*Activation)(const T&)>
__global__ void act_only_kernel(T* __restrict__ out_ptr, const T* __restrict__ input_ptr, int64_t d,
                                int64_t num_tokens) {
  constexpr uint32_t vec_size = kBytesToLoad / sizeof(T);
  const int64_t token_idx = blockIdx.x;
  const int64_t thread_idx = threadIdx.x;
  const int64_t stride = blockDim.x;
  const int64_t offset = token_idx * d;

  if (token_idx >= num_tokens) return;

#pragma unroll 1
  for (uint32_t idx = thread_idx; idx < d / vec_size; idx += stride) {
    device::AlignedVector<T, vec_size> x_vec, out_vec;
    x_vec.load(input_ptr + offset + idx * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      out_vec[i] = Activation(x_vec[i]);
    }
    out_vec.store(out_ptr + token_idx * d + idx * vec_size);
  }

  const int64_t remaining_offset = d - d % (stride * vec_size);
  // process the remaining elements
#pragma unroll 1
  for (int64_t idx = thread_idx; idx < d % (stride * vec_size); idx += stride) {
    T x = input_ptr[offset + remaining_offset + idx];
    out_ptr[token_idx * d + remaining_offset + idx] = Activation(x);
  }
}

template <typename T, T (*Activation)(const T&)>
struct ActivationAndMul {
  static constexpr auto kernel = act_and_mul_kernel<T, Activation>;

  static void run(tvm::ffi::TensorView output, tvm::ffi::TensorView input) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D_half = SymbolicSize{"d"};
    auto D_full = SymbolicSize{"2*d"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D_full})  //
        .with_dtype<T>()
        .with_device(device)
        .verify(input);

    TensorMatcher({N, D_half})  //
        .with_dtype<T>()
        .with_device(device)
        .verify(output);

    const int64_t num_tokens = N.unwrap();
    const int64_t d = D_half.unwrap();

    RuntimeCheck(D_full.unwrap() == 2 * d, "Input dimension must be 2 * output dimension");

    const uint32_t block_size = std::min<uint32_t>(d, 1024);

    LaunchKernel(num_tokens, block_size, device.unwrap())(
        kernel, static_cast<T*>(output.data_ptr()), static_cast<T*>(input.data_ptr()), d, num_tokens);
  }
};

template <typename T, T (*Activation)(const T&)>
struct ActivationOnly {
  static constexpr auto kernel = act_only_kernel<T, Activation>;

  static void run(tvm::ffi::TensorView output, tvm::ffi::TensorView input) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"d"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D})  //
        .with_dtype<T>()
        .with_device(device)
        .verify(input);

    TensorMatcher({N, D})  //
        .with_dtype<T>()
        .with_device(device)
        .verify(output);

    const int64_t num_tokens = N.unwrap();
    const int64_t d = D.unwrap();

    const uint32_t block_size = std::min<uint32_t>(d, 1024);

    LaunchKernel(num_tokens, block_size, device.unwrap())(
        kernel, static_cast<T*>(output.data_ptr()), static_cast<T*>(input.data_ptr()), d, num_tokens);
  }
};

template <typename T>
using silu_and_mul = ActivationAndMul<T, silu>;

template <typename T>
using gelu_and_mul = ActivationAndMul<T, gelu>;

template <typename T>
using gelu_tanh_and_mul = ActivationAndMul<T, gelu_tanh>;

template <typename T>
using gelu_quick = ActivationOnly<T, gelu_quick_act>;

}  // namespace
