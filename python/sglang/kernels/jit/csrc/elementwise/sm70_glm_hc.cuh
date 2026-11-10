// SPDX-License-Identifier: Apache-2.0
// GLM-5.3-Flash mHC on Volta. hidden=4096, hc_mult=4, mix width=24.
// Sinkhorn matches sm70_dsv41_hc_sinkhorn (sequential 4-wide, post_mult=2).
// pre takes the 24-wide mixes from torch; mix_partial + pre_fused compute
// them here, with fp64 accumulation.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang::sm70_glm_hc {

inline constexpr int kHc = 4;
inline constexpr int kHidden = 4096;
inline constexpr int kMix = 24;
inline constexpr int kVec = 8;
inline constexpr int kBlock = 256;
inline constexpr int kHiddenVecs = kHidden / kVec;
inline constexpr int kMixK = kHc * kHidden;
// 24 dots with fn plus the sum of squares.
inline constexpr int kMixOut = kMix + 1;
inline constexpr int kMaxSplits = kMixK / kBlock;
// Decode-side pre_fused/post: small blocks spread over the SMs, one vector
// (or one (column, vector) pair for post) per thread.
inline constexpr int kPreThreads = 128;
inline constexpr int kPreCombineBlocks = kHiddenVecs / kPreThreads;
inline constexpr int kPostThreads = 128;
inline constexpr int kPostBlocks = kHc * kHiddenVecs / kPostThreads;
// pre_fused_norm: one block combines and normalizes the whole row, one vector
// per thread, which is flashinfer RMSNormKernel<8, half>'s launch at d=4096.
inline constexpr int kNormThreads = kHiddenVecs;

static_assert(kHidden % kVec == 0, "hidden must be a multiple of the fp16 vector");

// torch.sum on a strided size-4 reduction (mHC columns) is left-to-right.
SGL_DEVICE float sum4_seq(float a, float b, float c, float d) {
  float s = a + b;
  s = s + c;
  s = s + d;
  return s;
}

// torch.sum on a contiguous size-4 vector (mHC rows) is pairwise.
SGL_DEVICE float sum4_pair(float a, float b, float c, float d) {
  return (a + b) + (c + d);
}

// torch.softmax on 4 elements uses a 4-thread warp reduction:
// (e0 + e2) + (e1 + e3). Sequential and adjacent-pair sums do not match.
SGL_DEVICE float sum4_softmax(float e0, float e1, float e2, float e3) {
  return (e0 + e2) + (e1 + e3);
}

SGL_DEVICE float max4(float a, float b, float c, float d) {
  float m = a;
  m = fmaxf(m, b);
  m = fmaxf(m, c);
  m = fmaxf(m, d);
  return m;
}

using hvec_t = device::AlignedVector<fp16_t, kVec>;

SGL_DEVICE hvec_t combine_load(const fp16_t* __restrict__ x_row, float p0, float p1, float p2, float p3, uint32_t vi) {
  hvec_t a, b, c, d, out;
  a.load(x_row + 0 * kHidden, vi);
  b.load(x_row + 1 * kHidden, vi);
  c.load(x_row + 2 * kHidden, vi);
  d.load(x_row + 3 * kHidden, vi);
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    float acc = p0 * static_cast<float>(a[i]);
    acc = acc + p1 * static_cast<float>(b[i]);
    acc = acc + p2 * static_cast<float>(c[i]);
    acc = acc + p3 * static_cast<float>(d[i]);
    out[i] = DTypeTrait<fp16_t>::from(acc);
  }
  return out;
}

SGL_DEVICE void combine_vec(fp16_t* __restrict__ y_row,
                            const fp16_t* __restrict__ x_row,
                            float p0,
                            float p1,
                            float p2,
                            float p3,
                            uint32_t vi) {
  combine_load(x_row, p0, p1, p2, p3, vi).store(y_row, vi);
}

SGL_DEVICE void combine_one(fp16_t* __restrict__ y_row,
                            const fp16_t* __restrict__ x_row,
                            float p0,
                            float p1,
                            float p2,
                            float p3) {
  for (uint32_t vi = threadIdx.x; vi < kHiddenVecs; vi += kBlock) {
    combine_vec(y_row, x_row, p0, p1, p2, p3, vi);
  }
}

SGL_DEVICE float hc_pre_weight(const float* __restrict__ mix,
                               const float* __restrict__ hc_scale,
                               const float* __restrict__ hc_base,
                               int j,
                               float eps) {
  const float logit = mix[j] * hc_scale[0] + hc_base[j];
  return 1.0f / (1.0f + expf(-logit)) + eps;
}

/// One token on one warp. mix is pre[4], post[4], comb[16] row-major;
/// lane l holds comb element (l & 15) = (j, k).
/// Row and column sums gather four values by shuffle and add them in the
/// torch reference order (sum4_*), so outputs match the reference bitwise.
SGL_DEVICE void sinkhorn_warp(const float* __restrict__ mix,
                              const float* __restrict__ hc_scale,
                              const float* __restrict__ hc_base,
                              float* __restrict__ pre_row,
                              float* __restrict__ post_row,
                              float* __restrict__ comb_row,
                              int sinkhorn_iters,
                              float eps) {
  constexpr unsigned kFull = 0xffffffffu;
  const int lane = static_cast<int>(threadIdx.x & 31);
  const int k = lane & 3;
  const int row_base = lane & ~3;
  const int col_base = (lane & 16) | k;
  if (lane < kHc) {
    pre_row[lane] = hc_pre_weight(mix, hc_scale, hc_base, lane, eps);
  } else if (lane < 2 * kHc) {
    const float logit = mix[lane] * hc_scale[1] + hc_base[lane];
    post_row[lane - kHc] = 2.0f * (1.0f / (1.0f + expf(-logit)));
  }
  const int idx = 2 * kHc + (lane & 15);
  float c = mix[idx] * hc_scale[2] + hc_base[idx];

  const float row_max = max4(__shfl_sync(kFull, c, row_base + 0),
                             __shfl_sync(kFull, c, row_base + 1),
                             __shfl_sync(kFull, c, row_base + 2),
                             __shfl_sync(kFull, c, row_base + 3));
  const float ex = expf(c - row_max);
  const float row_sum = sum4_softmax(__shfl_sync(kFull, ex, row_base + 0),
                                     __shfl_sync(kFull, ex, row_base + 1),
                                     __shfl_sync(kFull, ex, row_base + 2),
                                     __shfl_sync(kFull, ex, row_base + 3));
  c = ex / row_sum + eps;
  float col_sum = sum4_seq(__shfl_sync(kFull, c, col_base + 0),
                           __shfl_sync(kFull, c, col_base + 4),
                           __shfl_sync(kFull, c, col_base + 8),
                           __shfl_sync(kFull, c, col_base + 12));
  c = c / (col_sum + eps);
  for (int it = 0; it < sinkhorn_iters - 1; ++it) {
    const float rs = sum4_pair(__shfl_sync(kFull, c, row_base + 0),
                               __shfl_sync(kFull, c, row_base + 1),
                               __shfl_sync(kFull, c, row_base + 2),
                               __shfl_sync(kFull, c, row_base + 3));
    c = c / (rs + eps);
    col_sum = sum4_seq(__shfl_sync(kFull, c, col_base + 0),
                       __shfl_sync(kFull, c, col_base + 4),
                       __shfl_sync(kFull, c, col_base + 8),
                       __shfl_sync(kFull, c, col_base + 12));
    c = c / (col_sum + eps);
  }
  if (lane < kHc * kHc) {
    comb_row[lane] = c;
  }
}

/// One token per block. Warp 0 runs Sinkhorn while every warp combines; the
/// combine only needs pre, which each thread recomputes with the same expression.
SGL_DEVICE void pre_token(uint32_t t,
                          const float* mix,
                          fp32_t* __restrict__ pre,
                          fp32_t* __restrict__ post,
                          fp32_t* __restrict__ comb,
                          fp16_t* __restrict__ y,
                          const fp16_t* __restrict__ residual,
                          const fp32_t* __restrict__ hc_scale,
                          const fp32_t* __restrict__ hc_base,
                          int32_t sinkhorn_iters,
                          float eps) {
  if (threadIdx.x < 32) {
    sinkhorn_warp(mix,
                  hc_scale,
                  hc_base,
                  pre + static_cast<int64_t>(t) * kHc,
                  post + static_cast<int64_t>(t) * kHc,
                  comb + static_cast<int64_t>(t) * (kHc * kHc),
                  sinkhorn_iters,
                  eps);
  }
  combine_one(y + static_cast<int64_t>(t) * kHidden,
              residual + static_cast<int64_t>(t) * (kHc * kHidden),
              hc_pre_weight(mix, hc_scale, hc_base, 0, eps),
              hc_pre_weight(mix, hc_scale, hc_base, 1, eps),
              hc_pre_weight(mix, hc_scale, hc_base, 2, eps),
              hc_pre_weight(mix, hc_scale, hc_base, 3, eps));
}

/// mixes [T, 24] fp32, residual [T, 16384] fp16. Writes pre/post/comb/y.
__global__ __launch_bounds__(kBlock) void pre_kernel(
    fp32_t* __restrict__ pre,
    fp32_t* __restrict__ post,
    fp32_t* __restrict__ comb,
    fp16_t* __restrict__ y,
    const fp32_t* __restrict__ mixes,
    const fp16_t* __restrict__ residual,
    const fp32_t* __restrict__ hc_scale,
    const fp32_t* __restrict__ hc_base,
    int32_t sinkhorn_iters,
    float eps) {
  const uint32_t t = blockIdx.x;
  pre_token(t, mixes + static_cast<int64_t>(t) * kMix, pre, post, comb, y, residual, hc_scale, hc_base,
            sinkhorn_iters, eps);
}

/// One step of warp_sum_scatter: keep the half of v this lane owns, add the
/// partner's copy of it, and hand the partner the other half.
template <int kN>
SGL_DEVICE void sum_scatter_step(const double (&v)[kN], double (&out)[kN / 2], int lane) {
  constexpr int kHalf = kN / 2;
  const bool upper = (lane & kHalf) != 0;
#pragma unroll
  for (int i = 0; i < kHalf; ++i) {
    const double keep = upper ? v[i + kHalf] : v[i];
    const double send = upper ? v[i] : v[i + kHalf];
    out[i] = keep + __shfl_xor_sync(0xffffffffu, send, kHalf);
  }
}

/// Warp sum of each of the kMixOut values, returned on lane j for value j.
/// The adds pair lanes exactly as an xor butterfly over offsets 16..1 does, so
/// the sums are bitwise the butterfly's, with 31 shuffles instead of 125.
SGL_DEVICE double warp_sum_scatter(const double (&acc)[kMixOut], int lane) {
  double v32[32], v16[16], v8[8], v4[4], v2[2], v1[1];
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    v32[j] = j < kMixOut ? acc[j] : 0.0;
  }
  sum_scatter_step(v32, v16, lane);
  sum_scatter_step(v16, v8, lane);
  sum_scatter_step(v8, v4, lane);
  sum_scatter_step(v4, v2, lane);
  sum_scatter_step(v2, v1, lane);
  return v1[0];
}

/// Grid (splits, T). partials[t, s, :] over columns [s * span, (s + 1) * span):
/// 24 dots of the fp16 residual with fp32 fn, then the sum of squares, in fp64.
/// Products of fp16 and fp32 are exact in fp64; block sums run in warp order.
__global__ __launch_bounds__(kBlock) void mix_partial_kernel(double* __restrict__ partials,
                                                             const fp16_t* __restrict__ residual,
                                                             const fp32_t* __restrict__ fn,
                                                             int32_t splits) {
  const int s = static_cast<int>(blockIdx.x);
  const int64_t t = blockIdx.y;
  const int span = kMixK / splits;
  const fp16_t* x_row = residual + t * kMixK + static_cast<int64_t>(s) * span;
  const fp32_t* fn_col = fn + static_cast<int64_t>(s) * span;
  double acc[kMixOut];
#pragma unroll
  for (int j = 0; j < kMixOut; ++j) {
    acc[j] = 0.0;
  }
  for (int k = static_cast<int>(threadIdx.x); k < span; k += kBlock) {
    const double x = static_cast<double>(static_cast<float>(x_row[k]));
#pragma unroll
    for (int j = 0; j < kMix; ++j) {
      acc[j] = fma(x, static_cast<double>(fn_col[static_cast<int64_t>(j) * kMixK + k]), acc[j]);
    }
    acc[kMix] = fma(x, x, acc[kMix]);
  }
  __shared__ double s_part[kBlock / 32][kMixOut];
  const int warp = static_cast<int>(threadIdx.x >> 5);
  const int lane = static_cast<int>(threadIdx.x & 31);
  const double sum = warp_sum_scatter(acc, lane);
  if (lane < kMixOut) {
    s_part[warp][lane] = sum;
  }
  __syncthreads();
  if (threadIdx.x < kMixOut) {
    double v = 0.0;
#pragma unroll
    for (int w = 0; w < kBlock / 32; ++w) {
      v += s_part[w][threadIdx.x];
    }
    partials[(t * splits + s) * kMixOut + threadIdx.x] = v;
  }
}

/// mix_partial_kernel at splits == kMaxSplits (one column per thread) for up to
/// kMixRowsMaxTokens tokens: one block per split keeps its fn column in
/// registers and walks the tokens with the same per-token arithmetic, so each
/// token's partials are bitwise those of mix_partial_kernel.
inline constexpr int kMixRowsMaxTokens = 4;
static_assert(kMixK / kMaxSplits == kBlock);

__global__ __launch_bounds__(kBlock) void mix_partial_rows_kernel(double* __restrict__ partials,
                                                                  const fp16_t* __restrict__ residual,
                                                                  const fp32_t* __restrict__ fn,
                                                                  int32_t tokens) {
  const int s = static_cast<int>(blockIdx.x);
  const int64_t k = static_cast<int64_t>(s) * kBlock + threadIdx.x;
  fp32_t f[kMix];
#pragma unroll
  for (int j = 0; j < kMix; ++j) {
    f[j] = fn[static_cast<int64_t>(j) * kMixK + k];
  }
  __shared__ double s_part[kBlock / 32][kMixOut];
  const int warp = static_cast<int>(threadIdx.x >> 5);
  const int lane = static_cast<int>(threadIdx.x & 31);
  for (int t = 0; t < tokens; ++t) {
    const double x = static_cast<double>(static_cast<float>(residual[static_cast<int64_t>(t) * kMixK + k]));
    double acc[kMixOut];
#pragma unroll
    for (int j = 0; j < kMix; ++j) {
      acc[j] = fma(x, static_cast<double>(f[j]), 0.0);
    }
    acc[kMix] = fma(x, x, 0.0);
    const double sum = warp_sum_scatter(acc, lane);
    if (lane < kMixOut) {
      s_part[warp][lane] = sum;
    }
    __syncthreads();
    if (threadIdx.x < kMixOut) {
      double v = 0.0;
#pragma unroll
      for (int w = 0; w < kBlock / 32; ++w) {
        v += s_part[w][threadIdx.x];
      }
      partials[(static_cast<int64_t>(t) * kMaxSplits + s) * kMixOut + threadIdx.x] = v;
    }
    __syncthreads();
  }
}

/// mixes = dots * rsqrt(mean(x^2) + rms_eps) for token t, reduced from
/// mix_partial_kernel's [T, splits, 25] in split order and rounded once to fp32.
/// Every block runs this in the same order, so all see the same mixes.
template <int kThreads>
SGL_DEVICE void reduce_mixes(
    float* __restrict__ s_mix, const double* __restrict__ partials, int32_t splits, uint32_t t, double rms_eps) {
  __shared__ double s_part[kMaxSplits * kMixOut];
  __shared__ double s_tot[kMixOut];
  const double* p = partials + static_cast<int64_t>(t) * splits * kMixOut;
  for (int i = static_cast<int>(threadIdx.x); i < splits * kMixOut; i += kThreads) {
    s_part[i] = p[i];
  }
  __syncthreads();
  if (threadIdx.x < kMixOut) {
    double v = 0.0;
    if (splits == kMaxSplits) {
#pragma unroll
      for (int s = 0; s < kMaxSplits; ++s) {
        v += s_part[s * kMixOut + threadIdx.x];
      }
    } else {
      for (int s = 0; s < splits; ++s) {
        v += s_part[s * kMixOut + threadIdx.x];
      }
    }
    s_tot[threadIdx.x] = v;
  }
  __syncthreads();
  if (threadIdx.x < kMix) {
    const double r = 1.0 / sqrt(s_tot[kMix] / kMixK + rms_eps);
    s_mix[threadIdx.x] = static_cast<float>(s_tot[threadIdx.x] * r);
  }
  __syncthreads();
}

SGL_DEVICE void sinkhorn_token(const float* __restrict__ s_mix,
                               fp32_t* __restrict__ pre,
                               fp32_t* __restrict__ post,
                               fp32_t* __restrict__ comb,
                               const fp32_t* __restrict__ hc_scale,
                               const fp32_t* __restrict__ hc_base,
                               uint32_t t,
                               int32_t sinkhorn_iters,
                               float eps) {
  if (threadIdx.x < 32) {
    sinkhorn_warp(s_mix,
                  hc_scale,
                  hc_base,
                  pre + static_cast<int64_t>(t) * kHc,
                  post + static_cast<int64_t>(t) * kHc,
                  comb + static_cast<int64_t>(t) * (kHc * kHc),
                  sinkhorn_iters,
                  eps);
  }
}

/// pre_kernel with the mixes from reduce_mixes.
/// Grid (1 + kPreCombineBlocks, T): block 0 runs Sinkhorn on its first warp
/// while the others combine one vector per thread.
__global__ __launch_bounds__(kPreThreads) void pre_fused_kernel(
    fp32_t* __restrict__ pre,
    fp32_t* __restrict__ post,
    fp32_t* __restrict__ comb,
    fp16_t* __restrict__ y,
    const double* __restrict__ partials,
    int32_t splits,
    const fp16_t* __restrict__ residual,
    const fp32_t* __restrict__ hc_scale,
    const fp32_t* __restrict__ hc_base,
    int32_t sinkhorn_iters,
    float eps,
    double rms_eps) {
  const uint32_t t = blockIdx.y;
  __shared__ float s_mix[kMix];
  reduce_mixes<kPreThreads>(s_mix, partials, splits, t, rms_eps);
  if (blockIdx.x == 0) {
    sinkhorn_token(s_mix, pre, post, comb, hc_scale, hc_base, t, sinkhorn_iters, eps);
    return;
  }
  combine_vec(y + static_cast<int64_t>(t) * kHidden,
              residual + static_cast<int64_t>(t) * (kHc * kHidden),
              hc_pre_weight(s_mix, hc_scale, hc_base, 0, eps),
              hc_pre_weight(s_mix, hc_scale, hc_base, 1, eps),
              hc_pre_weight(s_mix, hc_scale, hc_base, 2, eps),
              hc_pre_weight(s_mix, hc_scale, hc_base, 3, eps),
              (blockIdx.x - 1) * kPreThreads + threadIdx.x);
}

SGL_DEVICE float rsqrt_approx_ftz(float x) {
  float y;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/// pre_fused_kernel followed by RMSNorm of the fp16 layer input, bitwise equal
/// to flashinfer RMSNormKernel<8, half> (sgl_kernel rmsnorm) on that row:
/// per-thread fma sum of squares, xor-butterfly within and across 16 warps,
/// IEEE divide by d, rsqrt.approx.ftz, then (x * rcp) * (0 + w).
/// Grid (2, T): block 0 runs Sinkhorn, block 1 combines and normalizes.
__global__ __launch_bounds__(kNormThreads) void pre_fused_norm_kernel(
    fp32_t* __restrict__ pre,
    fp32_t* __restrict__ post,
    fp32_t* __restrict__ comb,
    fp16_t* __restrict__ y,
    const double* __restrict__ partials,
    int32_t splits,
    const fp16_t* __restrict__ residual,
    const fp32_t* __restrict__ hc_scale,
    const fp32_t* __restrict__ hc_base,
    const fp16_t* __restrict__ norm_weight,
    int32_t sinkhorn_iters,
    float eps,
    double rms_eps,
    float norm_eps) {
  constexpr int kWarps = kNormThreads / 32;
  static_assert(kWarps <= 32, "cross-warp reduction runs on one warp");
  const uint32_t t = blockIdx.y;
  __shared__ float s_mix[kMix];
  __shared__ float s_sq[kWarps];
  reduce_mixes<kNormThreads>(s_mix, partials, splits, t, rms_eps);
  if (blockIdx.x == 0) {
    sinkhorn_token(s_mix, pre, post, comb, hc_scale, hc_base, t, sinkhorn_iters, eps);
    return;
  }
  const uint32_t vi = threadIdx.x;
  const hvec_t x = combine_load(residual + static_cast<int64_t>(t) * (kHc * kHidden),
                                hc_pre_weight(s_mix, hc_scale, hc_base, 0, eps),
                                hc_pre_weight(s_mix, hc_scale, hc_base, 1, eps),
                                hc_pre_weight(s_mix, hc_scale, hc_base, 2, eps),
                                hc_pre_weight(s_mix, hc_scale, hc_base, 3, eps),
                                vi);
  float sum_sq = 0.0f;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    const float v = static_cast<float>(x[i]);
    sum_sq = __fmaf_rn(v, v, sum_sq);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xffffffffu, sum_sq, off));
  }
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t warp = threadIdx.x >> 5;
  s_sq[warp] = sum_sq;
  __syncthreads();
  if (warp == 0) {
    sum_sq = lane < kWarps ? s_sq[lane] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xffffffffu, sum_sq, off));
    }
    if (lane == 0) {
      s_sq[0] = sum_sq;
    }
  }
  __syncthreads();
  const float rcp = rsqrt_approx_ftz(__fadd_rn(__fdiv_rn(s_sq[0], static_cast<float>(kHidden)), norm_eps));
  hvec_t w, out;
  w.load(norm_weight, vi);
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    const float scaled = __fmul_rn(static_cast<float>(x[i]), rcp);
    out[i] = DTypeTrait<fp16_t>::from(__fmul_rn(scaled, __fadd_rn(0.0f, static_cast<float>(w[i]))));
  }
  out.store(y + static_cast<int64_t>(t) * kHidden, vi);
}

/// out[t, j, h] = post[t, j] * x[t, h] + sum_k comb[t, k, j] * residual[t, k, h]
/// Grid (kPostBlocks, T); thread -> (column j, hidden vector vi).
__global__ __launch_bounds__(kPostThreads) void post_kernel(
    fp16_t* __restrict__ out,
    const fp16_t* __restrict__ x,
    const fp16_t* __restrict__ residual,
    const fp32_t* __restrict__ post,
    const fp32_t* __restrict__ comb) {
  using vec_t = device::AlignedVector<fp16_t, kVec>;
  const uint32_t t = blockIdx.y;
  const uint32_t idx = blockIdx.x * kPostThreads + threadIdx.x;
  const int j = static_cast<int>(idx / kHiddenVecs);
  const uint32_t vi = idx % kHiddenVecs;
  const fp16_t* x_row = x + static_cast<int64_t>(t) * kHidden;
  const fp16_t* res_row = residual + static_cast<int64_t>(t) * (kHc * kHidden);
  fp16_t* out_row = out + static_cast<int64_t>(t) * (kHc * kHidden);
  const fp32_t* comb_row = comb + static_cast<int64_t>(t) * (kHc * kHc);
  const float post_j = post[static_cast<int64_t>(t) * kHc + j];
  float comb_kj[kHc];
#pragma unroll
  for (int k = 0; k < kHc; ++k) {
    comb_kj[k] = comb_row[k * kHc + j];
  }
  vec_t xv;
  vec_t rv[kHc];
  xv.load(x_row, vi);
#pragma unroll
  for (int k = 0; k < kHc; ++k) {
    rv[k].load(res_row + k * kHidden, vi);
  }
  vec_t ov;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    // Match `post * x + sum_k`, with the size-4 sum left-to-right.
    float acc = comb_kj[0] * static_cast<float>(rv[0][i]);
    acc = acc + comb_kj[1] * static_cast<float>(rv[1][i]);
    acc = acc + comb_kj[2] * static_cast<float>(rv[2][i]);
    acc = acc + comb_kj[3] * static_cast<float>(rv[3][i]);
    acc = post_j * static_cast<float>(xv[i]) + acc;
    ov[i] = DTypeTrait<fp16_t>::from(acc);
  }
  ov.store(out_row + j * kHidden, vi);
}

void pre(tvm::ffi::TensorView pre_out,
         tvm::ffi::TensorView post_out,
         tvm::ffi::TensorView comb_out,
         tvm::ffi::TensorView y,
         tvm::ffi::TensorView mixes,
         tvm::ffi::TensorView residual,
         tvm::ffi::TensorView hc_scale,
         tvm::ffi::TensorView hc_base,
         int64_t sinkhorn_iters,
         double eps) {
  using namespace host;
  SymbolicSize n_tokens = {"num_tokens"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({n_tokens, kMix}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(mixes);
  TensorMatcher({n_tokens, kHc * kHidden})
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(residual);
  TensorMatcher({n_tokens, kHc}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(pre_out).verify(post_out);
  TensorMatcher({n_tokens, kHc, kHc}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(comb_out);
  TensorMatcher({n_tokens, kHidden}).with_dtype<fp16_t>().with_device<kDLCUDA>(device_).verify(y);
  TensorMatcher({3}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(hc_scale);
  TensorMatcher({kMix}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(hc_base);
  const uint32_t n = static_cast<uint32_t>(n_tokens.unwrap());
  CHECK_HOST(n > 0);
  CHECK_HOST(sinkhorn_iters >= 1);
  LaunchKernel(n, kBlock, device_.unwrap())(
      pre_kernel,
      static_cast<fp32_t*>(pre_out.data_ptr()),
      static_cast<fp32_t*>(post_out.data_ptr()),
      static_cast<fp32_t*>(comb_out.data_ptr()),
      static_cast<fp16_t*>(y.data_ptr()),
      static_cast<const fp32_t*>(mixes.data_ptr()),
      static_cast<const fp16_t*>(residual.data_ptr()),
      static_cast<const fp32_t*>(hc_scale.data_ptr()),
      static_cast<const fp32_t*>(hc_base.data_ptr()),
      static_cast<int32_t>(sinkhorn_iters),
      static_cast<float>(eps));
}

/// pre with the mixes computed here: fn [24, 16384] fp32, partials [T, S, 25]
/// fp64 workspace, where S is a power of two dividing 16384 / 256.
/// A non-null norm_weight [4096] fp16 writes RMSNorm(layer input) to y instead.
inline void launch_pre_fused(tvm::ffi::TensorView pre_out,
                             tvm::ffi::TensorView post_out,
                             tvm::ffi::TensorView comb_out,
                             tvm::ffi::TensorView y,
                             tvm::ffi::TensorView partials,
                             tvm::ffi::TensorView residual,
                             tvm::ffi::TensorView fn,
                             tvm::ffi::TensorView hc_scale,
                             tvm::ffi::TensorView hc_base,
                             const fp16_t* norm_weight,
                             int64_t sinkhorn_iters,
                             double eps,
                             double rms_eps,
                             double norm_eps) {
  using namespace host;
  SymbolicSize n_tokens = {"num_tokens"};
  SymbolicSize n_splits = {"splits"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({n_tokens, n_splits, kMixOut})
      .with_dtype<double>()
      .with_device<kDLCUDA>(device_)
      .verify(partials);
  TensorMatcher({kMix, kMixK}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(fn);
  TensorMatcher({n_tokens, kHc * kHidden})
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(residual);
  TensorMatcher({n_tokens, kHc}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(pre_out).verify(post_out);
  TensorMatcher({n_tokens, kHc, kHc}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(comb_out);
  TensorMatcher({n_tokens, kHidden}).with_dtype<fp16_t>().with_device<kDLCUDA>(device_).verify(y);
  TensorMatcher({3}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(hc_scale);
  TensorMatcher({kMix}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(hc_base);
  const uint32_t n = static_cast<uint32_t>(n_tokens.unwrap());
  const int32_t splits = static_cast<int32_t>(n_splits.unwrap());
  CHECK_HOST(n > 0);
  CHECK_HOST(sinkhorn_iters >= 1);
  CHECK_HOST(splits >= 1 && splits <= kMixK / kBlock && (splits & (splits - 1)) == 0);
  if (splits == kMaxSplits && n <= kMixRowsMaxTokens) {
    LaunchKernel(dim3(kMaxSplits), kBlock, device_.unwrap())(
        mix_partial_rows_kernel,
        static_cast<double*>(partials.data_ptr()),
        static_cast<const fp16_t*>(residual.data_ptr()),
        static_cast<const fp32_t*>(fn.data_ptr()),
        static_cast<int32_t>(n));
  } else {
    LaunchKernel(dim3(splits, n), kBlock, device_.unwrap())(
        mix_partial_kernel,
        static_cast<double*>(partials.data_ptr()),
        static_cast<const fp16_t*>(residual.data_ptr()),
        static_cast<const fp32_t*>(fn.data_ptr()),
        splits);
  }
  if (norm_weight == nullptr) {
    LaunchKernel(dim3(1 + kPreCombineBlocks, n), kPreThreads, device_.unwrap())(
        pre_fused_kernel,
        static_cast<fp32_t*>(pre_out.data_ptr()),
        static_cast<fp32_t*>(post_out.data_ptr()),
        static_cast<fp32_t*>(comb_out.data_ptr()),
        static_cast<fp16_t*>(y.data_ptr()),
        static_cast<const double*>(partials.data_ptr()),
        splits,
        static_cast<const fp16_t*>(residual.data_ptr()),
        static_cast<const fp32_t*>(hc_scale.data_ptr()),
        static_cast<const fp32_t*>(hc_base.data_ptr()),
        static_cast<int32_t>(sinkhorn_iters),
        static_cast<float>(eps),
        rms_eps);
    return;
  }
  LaunchKernel(dim3(2, n), kNormThreads, device_.unwrap())(
      pre_fused_norm_kernel,
      static_cast<fp32_t*>(pre_out.data_ptr()),
      static_cast<fp32_t*>(post_out.data_ptr()),
      static_cast<fp32_t*>(comb_out.data_ptr()),
      static_cast<fp16_t*>(y.data_ptr()),
      static_cast<const double*>(partials.data_ptr()),
      splits,
      static_cast<const fp16_t*>(residual.data_ptr()),
      static_cast<const fp32_t*>(hc_scale.data_ptr()),
      static_cast<const fp32_t*>(hc_base.data_ptr()),
      norm_weight,
      static_cast<int32_t>(sinkhorn_iters),
      static_cast<float>(eps),
      rms_eps,
      static_cast<float>(norm_eps));
}

void pre_fused(tvm::ffi::TensorView pre_out,
               tvm::ffi::TensorView post_out,
               tvm::ffi::TensorView comb_out,
               tvm::ffi::TensorView y,
               tvm::ffi::TensorView partials,
               tvm::ffi::TensorView residual,
               tvm::ffi::TensorView fn,
               tvm::ffi::TensorView hc_scale,
               tvm::ffi::TensorView hc_base,
               int64_t sinkhorn_iters,
               double eps,
               double rms_eps) {
  launch_pre_fused(
      pre_out, post_out, comb_out, y, partials, residual, fn, hc_scale, hc_base, nullptr, sinkhorn_iters, eps, rms_eps, 0.0);
}

void pre_fused_norm(tvm::ffi::TensorView pre_out,
                    tvm::ffi::TensorView post_out,
                    tvm::ffi::TensorView comb_out,
                    tvm::ffi::TensorView y,
                    tvm::ffi::TensorView partials,
                    tvm::ffi::TensorView residual,
                    tvm::ffi::TensorView fn,
                    tvm::ffi::TensorView hc_scale,
                    tvm::ffi::TensorView hc_base,
                    tvm::ffi::TensorView norm_weight,
                    int64_t sinkhorn_iters,
                    double eps,
                    double rms_eps,
                    double norm_eps) {
  using namespace host;
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({kHidden}).with_dtype<fp16_t>().with_device<kDLCUDA>(device_).verify(norm_weight);
  CHECK_HOST(reinterpret_cast<uintptr_t>(norm_weight.data_ptr()) % 16 == 0);
  launch_pre_fused(pre_out,
                   post_out,
                   comb_out,
                   y,
                   partials,
                   residual,
                   fn,
                   hc_scale,
                   hc_base,
                   static_cast<const fp16_t*>(norm_weight.data_ptr()),
                   sinkhorn_iters,
                   eps,
                   rms_eps,
                   norm_eps);
}

void post(tvm::ffi::TensorView out,
          tvm::ffi::TensorView x,
          tvm::ffi::TensorView residual,
          tvm::ffi::TensorView post_mix,
          tvm::ffi::TensorView comb) {
  using namespace host;
  SymbolicSize n_tokens = {"num_tokens"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({n_tokens, kHc * kHidden})
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(out)
      .verify(residual);
  TensorMatcher({n_tokens, kHidden}).with_dtype<fp16_t>().with_device<kDLCUDA>(device_).verify(x);
  TensorMatcher({n_tokens, kHc}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(post_mix);
  TensorMatcher({n_tokens, kHc, kHc}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(comb);
  const uint32_t n = static_cast<uint32_t>(n_tokens.unwrap());
  CHECK_HOST(n > 0);
  LaunchKernel(dim3(kPostBlocks, n), kPostThreads, device_.unwrap())(
      post_kernel,
      static_cast<fp16_t*>(out.data_ptr()),
      static_cast<const fp16_t*>(x.data_ptr()),
      static_cast<const fp16_t*>(residual.data_ptr()),
      static_cast<const fp32_t*>(post_mix.data_ptr()),
      static_cast<const fp32_t*>(comb.data_ptr()));
}

}  // namespace sglang::sm70_glm_hc
