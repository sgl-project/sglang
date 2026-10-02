#pragma once
#include "hc_combine.cuh"

namespace sglang {
// Expose the existing two stock stages separately; all arithmetic is unchanged.
struct HcCombineDecode {
  static void gate(tvm::ffi::TensorView normed, tvm::ffi::TensorView weight, tvm::ffi::TensorView partials) {
    using namespace host;
    SymbolicSize m{"rows"};
    SymbolicDevice dev;
    dev.set_options<kDLCUDA>();
    TensorMatcher({m, 10240}).with_dtype<bf16_t>().with_device(dev).verify(normed);
    TensorMatcher({4, 10240}).with_dtype<bf16_t>().with_device(dev).verify(weight);
    TensorMatcher({m, 8, 4}).with_dtype<fp32_t>().with_device(dev).verify(partials);
    const HcCombineSplitParams params{
        .block_output = nullptr,
        .residual = nullptr,
        .normed_residual = normed.data_ptr(),
        .inject_weight = weight.data_ptr(),
        .output = nullptr,
        .partials = static_cast<float*>(partials.data_ptr())};
    LaunchKernel(dim3(m.unwrap(), 32, 1), 32, dev.unwrap())
        .enable_pdl(true)(hc_combine_gate_kernel<4, 2560, true, bf16_t>, params);
  }
  static void apply(
      tvm::ffi::TensorView block,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView partials,
      tvm::ffi::TensorView out) {
    using namespace host;
    SymbolicSize m{"rows"};
    SymbolicDevice dev;
    dev.set_options<kDLCUDA>();
    TensorMatcher({m, 2560}).with_dtype<bf16_t>().with_device(dev).verify(block);
    TensorMatcher({m, 10240}).with_dtype<bf16_t>().with_device(dev).verify(residual).verify(out);
    TensorMatcher({m, 8, 4}).with_dtype<fp32_t>().with_device(dev).verify(partials);
    const HcCombineSplitParams params{
        .block_output = block.data_ptr(),
        .residual = residual.data_ptr(),
        .normed_residual = nullptr,
        .inject_weight = nullptr,
        .output = out.data_ptr(),
        .partials = static_cast<float*>(partials.data_ptr())};
    LaunchKernel(dim3(m.unwrap(), 8, 1), 160, dev.unwrap())
        .enable_pdl(true)(hc_combine_apply_kernel<4, 2560, true, bf16_t>, params);
  }
};
}  // namespace sglang

namespace sglang {
struct HcCombineNormDecodeParams {
  const bf16_t* block;
  const bf16_t* residual;
  const float* partials;
  const bf16_t* weight;
  bf16_t* combined;
  bf16_t* normalized;
  float eps;
};

template <int kParts>
__global__
__launch_bounds__(160) void hc_combine_norm_decode_kernel(const HcCombineNormDecodeParams __grid_constant__ p) {
  using namespace device;
  using Float2 = packed_t<bf16_t>;
  using Storage = AlignedVector<Float2, 8>;
  constexpr uint32_t kThreads = 160;
  constexpr uint32_t kWarps = 5;
  static_assert(kThreads % kParts == 0);
  const uint32_t row = blockIdx.x / (4 * kParts);
  const uint32_t branch = (blockIdx.x / kParts) % 4;
  const uint32_t part = blockIdx.x % kParts;
  const uint32_t tx = threadIdx.x;
  const auto gmem = tile::Memory<Storage>::cta(kThreads);
  const auto r = p.residual + (row * 4 + branch) * 2560;
  const auto y = p.block + row * 2560;
  const auto w = p.weight + branch * 2560;
  __shared__ float smem[kWarpThreads];
  PDLWaitPrimary<true>();
  float total = 0.0f;
#pragma unroll
  for (uint32_t split = 0; split < 8; ++split)
    total += p.partials[(row * 8 + split) * 4 + branch];
  const float a = 2.0f / (1.0f + math::exp(-total / 4));
  const auto rv = gmem.load(r, 0);
  const auto yv = gmem.load(y, 0);
  const auto wv = gmem.load(w, 0);
  Storage combined;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const auto [rx, ry] = cast<fp32x2_t>(rv[i]);
    const auto [yx, yy] = cast<fp32x2_t>(yv[i]);
    // Match the stock apply's FMA and its BF16 store before normalization.
    combined[i] = cast<Float2>(fp32x2_t{rx + a * yx, ry + a * yy});
  }
  float sum = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const auto [x, y] = cast<fp32x2_t>(combined[i]);
    sum += x * x + y * y;
  }
  sum = warp::reduce_sum(sum);
  const auto warp_id = tx / kWarpThreads;
  smem[warp_id] = sum;
  __syncthreads();
  if (warp_id == 0) {
    const auto local_sum = tx < kWarps ? smem[tx] : 0.0f;
    sum = warp::reduce_sum(local_sum);
    smem[tx] = math::rsqrt(sum / 2560 + p.eps);
  }
  __syncthreads();
  const float norm = smem[warp_id];
  if (tx >= part * (kThreads / kParts) && tx < (part + 1) * (kThreads / kParts)) {
    Storage normalized;
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      const auto [x, y] = cast<fp32x2_t>(combined[i]);
      const auto [wx, wy] = cast<fp32x2_t>(wv[i]);
      normalized[i] = cast<Float2>(fp32x2_t{x * norm * (1.0f + wx), y * norm * (1.0f + wy)});
    }
    gmem.store(p.combined + (row * 4 + branch) * 2560, combined, 0);
    gmem.store(p.normalized + (row * 4 + branch) * 2560, normalized, 0);
  }
  PDLTriggerSecondary<true>();
}

template <int kParts>
struct HcCombineNormDecode {
  static void
  run(tvm::ffi::TensorView block,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView partials,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView combined,
      tvm::ffi::TensorView normalized,
      float eps) {
    using namespace host;
    SymbolicSize m{"rows"};
    SymbolicDevice dev;
    dev.set_options<kDLCUDA>();
    TensorMatcher({m, 2560}).with_dtype<bf16_t>().with_device(dev).verify(block);
    TensorMatcher({m, 10240})
        .with_dtype<bf16_t>()
        .with_device(dev)
        .verify(residual)
        .verify(combined)
        .verify(normalized);
    TensorMatcher({10240}).with_dtype<bf16_t>().with_device(dev).verify(weight);
    TensorMatcher({m, 8, 4}).with_dtype<fp32_t>().with_device(dev).verify(partials);
    CHECK_HOST(m.unwrap() > 0 && m.unwrap() <= 4);
    const HcCombineNormDecodeParams p{
        static_cast<const bf16_t*>(block.data_ptr()),
        static_cast<const bf16_t*>(residual.data_ptr()),
        static_cast<const float*>(partials.data_ptr()),
        static_cast<const bf16_t*>(weight.data_ptr()),
        static_cast<bf16_t*>(combined.data_ptr()),
        static_cast<bf16_t*>(normalized.data_ptr()),
        eps};
    LaunchKernel(m.unwrap() * 4 * kParts, 160, dev.unwrap()).enable_pdl(true)(hc_combine_norm_decode_kernel<kParts>, p);
  }
};
}  // namespace sglang
