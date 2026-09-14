#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cooperative_groups.h>
#include <cstdint>

namespace sglang {

/// Fused DeepSeek-V4.1 mHC sublayer boundary (HC = 4, hidden = 5120), the two
/// Triton kernels `mhc_post_split_h` + `hc_combine_norm` in one launch:
///
///   residual[t, i, :] = bf16( post[t, i] * x[t, :] + sum_j comb[t, j, i] * residual[t, j, :] )
///   v[t, :]           = bf16( sum_i pre[t, i] * residual[t, i, :] )
///   y[t, :]           = bf16( v * rsqrt(mean(v * v) + eps) * w )
///
/// `residual` is updated in place; the two bf16 roundings are the ones the
/// unfused pair performs (residual stored as bf16, combine stored as bf16
/// before the RMSNorm reads it). Accumulation order inside fp32 is not
/// preserved. `x` is the only input produced by the immediately preceding
/// kernel (the TP all-reduce), so everything else is loaded before the PDL wait.
///
/// One cluster of kCGASize CTAs owns one token and each thread owns kVecSize
/// consecutive hidden elements of every stream, so the whole boundary is
/// register-resident. The RMS statistic is the only cross-CTA exchange: one
/// float per warp pushed through DSMEM.
template <uint32_t kCGASize_, uint32_t kVecSize_>
struct MHCConfig {
  static constexpr uint32_t kWidth = 4;
  static constexpr uint32_t kHiddenDim = 5120;
  static constexpr uint32_t kCGASize = kCGASize_;
  static constexpr uint32_t kVecSize = kVecSize_;  // bf16 elements per thread
  static constexpr uint32_t kElemsPerCTA = kHiddenDim / kCGASize;
  static constexpr uint32_t kNumWarps = kElemsPerCTA / (kVecSize * device::kWarpThreads);
  static constexpr uint32_t kCTASize = kNumWarps * device::kWarpThreads;
  static constexpr uint32_t kNumCGAWarps = kNumWarps * kCGASize;
  static_assert(kCGASize >= 1 && kCGASize <= 8, "portable cluster sizes only");
  static_assert(kVecSize == 8 || kVecSize == 16, "16 B or 32 B per thread");
  static_assert(kHiddenDim % kCGASize == 0, "the cluster must split the row evenly");
  static_assert(kElemsPerCTA % (kVecSize * device::kWarpThreads) == 0);
  static_assert(kWidth * kWidth <= device::kWarpThreads, "comb is broadcast from one lane each");
  static_assert(kNumCGAWarps <= device::kWarpThreads, "one RMS partial slot per lane");
};

struct MHCParams {
  const bf16_t* __restrict__ x;     // [m, 5120]      sublayer output (all-reduced)
  bf16_t* __restrict__ residual;    // [m, 4, 5120]   in: streams before, out: streams after
  const fp32_t* __restrict__ post;  // [m, 4]
  const fp32_t* __restrict__ comb;  // [m, 4, 4]      comb[t, j, i]: source stream j -> target stream i
  const fp32_t* __restrict__ pre;   // [m, 4]         previous sublayer's pre-mix
  const bf16_t* __restrict__ w;     // [5120]         RMSNorm weight
  bf16_t* __restrict__ y;           // [m, 5120]      normalized sublayer input
  float eps;
};

/// Two independent fp32 FMAs in one instruction on Blackwell (`fma.rn.f32x2`,
/// PTX ISA 8.6, sm_100+); plain FMAs elsewhere.
SGL_DEVICE fp32x2_t fma_f32x2(fp32x2_t a, fp32x2_t b, fp32x2_t c) {
#if SGL_ARCH_BLACKWELL_OR_GREATER
  uint64_t d;
  const auto a_ = reinterpret_cast<const uint64_t&>(a);
  const auto b_ = reinterpret_cast<const uint64_t&>(b);
  const auto c_ = reinterpret_cast<const uint64_t&>(c);
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a_), "l"(b_), "l"(c_));
  return reinterpret_cast<const fp32x2_t&>(d);
#else
  return {fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y)};
#endif
}

/// TODO: rewrite with TMA for CGA = 1
template <typename C, bool kUsePDL>
__global__ __launch_bounds__(C::kCTASize)  //
    void mhc_post_combine_norm_kernel(const __grid_constant__ MHCParams params) {
  using namespace device;
  constexpr uint32_t kWidth = C::kWidth;
  constexpr uint32_t kHiddenDim = C::kHiddenDim;
  constexpr uint32_t kCGASize = C::kCGASize;
  constexpr uint32_t kVecSize = C::kVecSize;
  constexpr uint32_t kNumPairs = kVecSize / 2;
  using vec_t = AlignedVector<bf16x2_t, kVecSize / 2>;

  const auto token = blockIdx.y;
  const auto cta_rank = blockIdx.x;  // grid.x == cluster.x, so this is the cluster rank
  const auto tx = threadIdx.x;
  const auto vid = cta_rank * C::kCTASize + tx;  // vector index inside the row
  const auto lane_id = tx % kWarpThreads;
  const auto warp_id = vid / kWarpThreads;

  const auto row = static_cast<int64_t>(token);
  const auto x_ptr = params.x + row * kHiddenDim;
  const auto res_ptr = params.residual + row * (kWidth * kHiddenDim);
  const auto y_ptr = params.y + row * kHiddenDim;
  const auto comb_ptr = params.comb + row * (kWidth * kWidth);
  const auto post_ptr = params.post + row * kWidth;
  const auto pre_ptr = params.pre + row * kWidth;

  // Everything but x predates the preceding kernel: pull it in before the wait.
  vec_t res[kWidth];
#pragma unroll
  for (uint32_t j = 0; j < kWidth; ++j) {
    res[j].load(res_ptr + j * kHiddenDim, vid);
  }
  static_assert(kWidth == 4);
  const auto lane_ptr = lane_id < 16   ? comb_ptr + lane_id
                        : lane_id < 20 ? post_ptr + lane_id - 16
                        : lane_id < 24 ? pre_ptr + lane_id - 20
                                       : nullptr;
  const auto lane_val = lane_id < 24 ? *lane_ptr : 0.0f;

  PDLWaitPrimary<kUsePDL>();
  vec_t x;
  x.load(x_ptr, vid);
  vec_t w;

  // Post-mix one target stream at a time, store it, and fold it into the
  // combine right away so only one output vector is live.
  fp32x2_t acc[kNumPairs] = {};
#pragma unroll
  for (uint32_t i = 0; i < kWidth; ++i) {
    fp32x2_t mix[kNumPairs] = {};
    const auto post_i = __shfl_sync(0xffffffffu, lane_val, i + 16);
    const auto pre_i = __shfl_sync(0xffffffffu, lane_val, i + 20);
#pragma unroll
    for (uint32_t j = 0; j < kWidth; ++j) {
      const auto comb_ji = __shfl_sync(0xffffffffu, lane_val, j * kWidth + i);
      const auto comb2 = fp32x2_t{comb_ji, comb_ji};
#pragma unroll
      for (uint32_t k = 0; k < kNumPairs; ++k) {
        mix[k] = fma_f32x2(cast<fp32x2_t>(res[j][k]), comb2, mix[k]);
      }
    }
    if (i == kWidth - 1) w.load(params.w, vid);

    const auto post2 = fp32x2_t{post_i, post_i};
#pragma unroll
    for (uint32_t k = 0; k < kNumPairs; ++k) {
      mix[k] = fma_f32x2(cast<fp32x2_t>(x[k]), post2, mix[k]);
    }
    vec_t out;
#pragma unroll
    for (uint32_t k = 0; k < kNumPairs; ++k) {
      out[k] = cast<bf16x2_t>(mix[k]);
    }
    out.store(res_ptr + i * kHiddenDim, vid);
    const auto pre2 = fp32x2_t{pre_i, pre_i};
#pragma unroll
    for (uint32_t k = 0; k < kNumPairs; ++k) {
      acc[k] = fma_f32x2(cast<fp32x2_t>(out[k]), pre2, acc[k]);
    }
  }

  // rounding 2: the unfused combine stores bf16 before the RMSNorm reads it.
  float sq = 0.0f;
#pragma unroll
  for (uint32_t k = 0; k < kNumPairs; ++k) {
    acc[k] = cast<fp32x2_t>(cast<bf16x2_t>(acc[k]));
    sq = fmaf(acc[k].x, acc[k].x, sq);
    sq = fmaf(acc[k].y, acc[k].y, sq);
  }
  sq = warp::reduce_sum(sq);
  PDLTriggerSecondary<kUsePDL>();

  // One float per warp of the whole cluster lands in every CTA's copy of s_sq.
  __shared__ float s_sq[C::kNumCGAWarps];
  if constexpr (kCGASize > 1) {
    const auto cluster = cooperative_groups::this_cluster();
    if (lane_id < kCGASize) *cluster.map_shared_rank(&s_sq[warp_id], lane_id) = sq;
    cluster.sync();  // release / acquire: the DSMEM stores above are visible after this
  } else {
    if (lane_id == 0) s_sq[warp_id] = sq;
    __syncthreads();
  }

  float total = 0.0f;
  if constexpr (C::kNumWarps <= 8) {
#pragma unroll
    for (uint32_t i = 0; i < C::kNumCGAWarps; ++i) {
      total += s_sq[i];
    }
  } else {
    total = warp::reduce_sum(lane_id < C::kNumCGAWarps ? s_sq[lane_id] : 0.0f);
  }
  constexpr float kInvHidden = 1.0f / static_cast<float>(kHiddenDim);
  const float scale = math::rsqrt(fmaf(total, kInvHidden, params.eps));

  vec_t out;
#pragma unroll
  for (uint32_t k = 0; k < kNumPairs; ++k) {
    const auto [wa, wb] = cast<fp32x2_t>(w[k]);
    out[k] = cast<bf16x2_t>(fp32x2_t{acc[k].x * scale * wa, acc[k].y * scale * wb});
  }
  out.store(y_ptr, vid);
}

/// Host entry. Every tensor is contiguous on one CUDA device; `residual` is
/// rewritten in place, `y` must not alias `x` or `residual`.
template <uint32_t kCGASize, uint32_t kVecSize, bool kUsePDL>
struct MHCFusedKernel {
  using C = MHCConfig<kCGASize, kVecSize>;

  static void post_combine_norm(
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView post,
      const tvm::ffi::TensorView comb,
      const tvm::ffi::TensorView pre,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView y,
      const float eps) {
    using namespace host;
    constexpr auto kH = static_cast<int64_t>(C::kHiddenDim);
    constexpr auto kW = static_cast<int64_t>(C::kWidth);
    auto M = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({M, kH})  // x, y
        .with_dtype<bf16_t>()
        .with_device(device_)
        .verify(x)
        .verify(y);
    TensorMatcher({M, kW, kH})  // residual
        .with_dtype<bf16_t>()
        .with_device(device_)
        .verify(residual);
    TensorMatcher({M, kW})  // post, pre
        .with_dtype<fp32_t>()
        .with_device(device_)
        .verify(post)
        .verify(pre);
    TensorMatcher({M, kW, kW})  // comb
        .with_dtype<fp32_t>()
        .with_device(device_)
        .verify(comb);
    TensorMatcher({kH})  // w
        .with_dtype<bf16_t>()
        .with_device(device_)
        .verify(w);
    constexpr auto kAlign = static_cast<uintptr_t>(C::kVecSize * sizeof(bf16_t));
    CHECK_HOST(
        reinterpret_cast<uintptr_t>(x.data_ptr()) % kAlign == 0 &&
        reinterpret_cast<uintptr_t>(residual.data_ptr()) % kAlign == 0 &&
        reinterpret_cast<uintptr_t>(w.data_ptr()) % kAlign == 0 &&
        reinterpret_cast<uintptr_t>(y.data_ptr()) % kAlign == 0)
        << "x, residual, w and y must be " << kAlign << " B aligned";
    const int64_t m = M.unwrap();
    CHECK_HOST(m <= 65535) << "num_tokens exceeds grid.y, got " << m;
    if (m == 0) return;  // empty tensors may legitimately share a data pointer
    CHECK_HOST(y.data_ptr() != x.data_ptr() && y.data_ptr() != residual.data_ptr()) << "y must not alias x or residual";
    const auto params = MHCParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .residual = static_cast<bf16_t*>(residual.data_ptr()),
        .post = static_cast<const fp32_t*>(post.data_ptr()),
        .comb = static_cast<const fp32_t*>(comb.data_ptr()),
        .pre = static_cast<const fp32_t*>(pre.data_ptr()),
        .w = static_cast<const bf16_t*>(w.data_ptr()),
        .y = static_cast<bf16_t*>(y.data_ptr()),
        .eps = eps,
    };
    auto launcher = LaunchKernel(dim3(kCGASize, static_cast<uint32_t>(m)), C::kCTASize, device_.unwrap());
    launcher.enable_pdl(kUsePDL);
    if constexpr (kCGASize > 1) launcher.enable_cluster(dim3(kCGASize, 1, 1));
    launcher.launch(mhc_post_combine_norm_kernel<C, kUsePDL>, params);
  }
};

}  // namespace sglang
