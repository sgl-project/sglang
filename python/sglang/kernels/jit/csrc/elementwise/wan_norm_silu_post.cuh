// SPDX-License-Identifier: Apache-2.0
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/activation.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace sglang {

/// \brief Fuse divide, scale, affine, and SiLU without changing aten's L2 reduction.
/// \tparam kHalfChain The chain of a half-precision activation outside autocast:
///   the denominator and the output are X, and every op rounds to X as aten's
///   separate kernels do. Otherwise the chain is FP32 end to end.
template <typename X, typename A, bool kChannelsLast, bool kHalfChain>
__global__ void wan_norm_silu_post_kernel(
    const X* __restrict__ input,
    const std::conditional_t<kHalfChain, X, float>* __restrict__ denominator,
    const A* __restrict__ gamma,
    const A* __restrict__ bias,
    std::conditional_t<kHalfChain, X, float>* __restrict__ output,
    int64_t num_vecs,
    int64_t channels,
    int64_t spatial,
    float scale,
    bool has_bias) {
  using D = std::conditional_t<kHalfChain, X, float>;
  constexpr int kVec = 4;
  // aten computes each op in FP32 and rounds once, to the op's result dtype.
  const auto rounded = [](float value) { return kHalfChain ? device::cast<float>(device::cast<X>(value)) : value; };
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t vec = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; vec < num_vecs; vec += step) {
    device::AlignedVector<X, kVec> x;
    device::AlignedVector<D, kVec> d;
    device::AlignedVector<A, kVec> g;
    device::AlignedVector<A, kVec> b;
    device::AlignedVector<D, kVec> result;
    x.load(input, vec);
    float shared_d = 0.f, shared_g = 0.f, shared_b = 0.f;
    if constexpr (kChannelsLast) {
      const int64_t channel = vec * kVec % channels;
      shared_d = device::cast<float>(denominator[vec * kVec / channels]);
      g.load(gamma, channel / kVec);
      if (has_bias) b.load(bias, channel / kVec);
    } else {
      const int64_t channel = vec * kVec / spatial % channels;
      const int64_t batch = vec * kVec / (spatial * channels);
      const int64_t pixel = vec * kVec % spatial;
      d.load(denominator, (batch * spatial + pixel) / kVec);
      shared_g = device::cast<float>(gamma[channel]);
      if (has_bias) shared_b = device::cast<float>(bias[channel]);
    }
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float denom = kChannelsLast ? shared_d : device::cast<float>(d[i]);
      const float weight = kChannelsLast ? device::cast<float>(g[i]) : shared_g;
      const float offset = has_bias ? (kChannelsLast ? device::cast<float>(b[i]) : shared_b) : 0.f;
      float y = rounded(__fdiv_rn(device::cast<float>(x[i]), denom));
      y = rounded(__fmul_rn(y, scale));
      y = rounded(__fmul_rn(y, weight));
      y = rounded(__fadd_rn(y, offset));
      result[i] = device::cast<D>(device::activation::silu<false>(y));
    }
    result.store(output, vec);
  }
}

/// \brief Launch normalization post-ops for dense NCDHW or channels-last storage.
/// \param denominator Native L2 norm, clamped to the normalization epsilon:
///   FP32, or X when kHalfChain.
template <typename X, typename A, bool kChannelsLast, bool kHalfChain>
void wan_norm_silu_post(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView denominator,
    tvm::ffi::TensorView gamma,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView output,
    int64_t channels,
    int64_t spatial,
    double scale,
    bool has_bias) {
  using namespace host;
  using D = std::conditional_t<kHalfChain, X, float>;
  auto elements = SymbolicSize{"elements"};
  auto pixels = SymbolicSize{"pixels"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLGPU>();
  TensorMatcher({elements}).with_dtype<X>().with_device(device_).verify(input);
  TensorMatcher({pixels}).with_dtype<D>().with_device(device_).verify(denominator);
  TensorMatcher({channels}).with_dtype<A>().with_device(device_).verify(gamma);
  TensorMatcher({channels}).with_dtype<A>().with_device(device_).verify(bias);
  TensorMatcher({elements}).with_dtype<D>().with_device(device_).verify(output);
  const int64_t count = elements.unwrap();
  CHECK_HOST(channels > 0 && channels % 4 == 0 && spatial > 0 && spatial % 4 == 0);
  CHECK_HOST(count > 0 && count % (channels * spatial) == 0);
  CHECK_HOST(pixels.unwrap() == count / channels);
  constexpr int64_t threads = 256;
  const auto kernel = wan_norm_silu_post_kernel<X, A, kChannelsLast, kHalfChain>;
  const auto blocks_per_sm = runtime::get_blocks_per_sm(kernel, threads);
  const auto sms = runtime::get_sm_count(device_.unwrap().device_id);
  const auto blocks = std::min<int64_t>(sms * blocks_per_sm, div_ceil(count / 4, threads));
  LaunchKernel(blocks, threads, device_.unwrap())(
      kernel,
      static_cast<const X*>(input.data_ptr()),
      static_cast<const D*>(denominator.data_ptr()),
      static_cast<const A*>(gamma.data_ptr()),
      static_cast<const A*>(bias.data_ptr()),
      static_cast<D*>(output.data_ptr()),
      count / 4,
      channels,
      spatial,
      static_cast<float>(scale),
      has_bias);
}

}  // namespace sglang
