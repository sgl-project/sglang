// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SGLang project
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// K31HT_DRAFT_SITE_EPI (default off): one epilogue kernel per DFlash draft-layer site on the HT drafter. The stock all-reduce
// stays; the kernel replaces
//     DFlashGroupedConv.finish (Triton, side 1)  ->  RMSNorm(x, residual) (sgl_kernel fused_add_rmsnorm = flashinfer CuTe
//     FusedAddRMSNormKernel)
// on the all-reduced rows with the SAME arithmetic, so the outputs (normed rows and the updated residual) are bit-identical to the
// chain:
//   * conv side 1: c_k = f32(base_k) + f32(delta_k); out = fma(c0, y[t], c1 * (y[t-1] or +0)); RN to bf16 (the served Triton
//     form: mul c1 * shifted, then fma c0, x, that. The other contraction, fma(c1, y[t-1], c0 * y[t]), differs in rare words);
//   * add + norm: h = f32(x) + f32(residual); residual <- bf16(h); sum of h*h in flashinfer's thread mapping
//     (128 threads per row, thread k owns columns 8k + v + 1024b) and reduction order; rstd = rsqrt.approx(sum / H + eps);
//     out <- bf16((h * rstd) * w).
// The exact in-thread order / mean / rsqrt / conv forms match flashinfer 0.7.2 (
// 128 threads x 7 vector blocks per row, cluster 1; in-thread fma sequence, butterfly 1..16, div.rn mean, rsqrt.approx; Triton conv
// PTX = fma.rn.f32 form), bit-exact on 12 cells against the served Triton conv + sgl_kernel fused_add_rmsnorm.
#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/extra/stl.h>

#include <cstdint>

namespace sglang {

namespace k31ht_arsite {

inline constexpr uint32_t kHidden = 7168;
inline constexpr uint32_t kRowVecs = kHidden / 8;  // 896 vectors of 8 bf16
inline constexpr uint32_t kTaps = 2;
inline constexpr uint32_t kGroup = 16;
inline constexpr uint32_t kGroups = kHidden / kGroup;  // 448
inline constexpr uint32_t kNormThreads = 128;           // flashinfer threads_per_row for H 7168
inline constexpr uint32_t kNormVecBlocks = kRowVecs / kNormThreads;  // 7

using site_vec_t = device::AlignedVector<bf16x2_t, 4>;

SGL_DEVICE float bf1(const bf16_t v) { return __bfloat162float(v); }

// conv value before the bf16 store: c0 * y[t] + c1 * p (p = y[t-1], or +0 at block position 0) in the served Triton contraction
SGL_DEVICE float conv_pair(const float c0, const float y, const float c1, const float p) {
  return __fmaf_rn(c0, y, __fmul_rn(c1, p));
}

SGL_DEVICE float rsqrt_approx_ftz(float x) {
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// ---- K31HT_DRAFT_SITE_EPI (default off): the stock all-reduce stays; one kernel replaces DFlashGroupedConv.finish (side 1) +
// fused_add_rmsnorm on the reduced rows y. Per row the elements of flashinfer's thread k are its column vectors k + 128 b, b < 7;
// conv = fma(c0, y[t], c1 * (y[t-1] or +0)) rounded to bf16 (the served Triton form), then the flashinfer add + norm
// arithmetic. Reads y exactly as the stock conv does, so it is bit-identical with no zero-sign edge.
inline constexpr uint32_t kEpiSlices = 4;                          // vector-block slices per row
inline constexpr uint32_t kEpiThreads = kNormThreads * kEpiSlices;  // 512 threads per row

// v3 (the 896-thread v2 runs 1 CTA per SM): thread i = (k = i % 128, slice c = i / 128) owns the
// column vectors k + 128 b for b = c and b = c + 4 (b < 7), so a row needs 512 threads and two rows fit an SM; every load of the row
// is still in flight at once. The sum of squares still runs in flashinfer's order: thread k (< 128) chains fma over (b, v) from
// shared memory, then the same butterfly / 4-warp reduction.
template <bool kUsePDL>
__global__ __launch_bounds__(kEpiThreads, 2) void k31ht_draft_site_epi_kernel(
    const bf16_t* __restrict__ y, const bf16_t* __restrict__ delta, const bf16_t* __restrict__ base, bf16_t* __restrict__ residual,
    const bf16_t* __restrict__ weight, bf16_t* __restrict__ out, int64_t d_row, int64_t d_tap, int64_t b_tap, uint32_t q, float eps) {
  using namespace device;
  const uint32_t t = blockIdx.x;
  const uint32_t i = threadIdx.x;
  const uint32_t k = i % kNormThreads, c = i / kNormThreads;
  constexpr uint32_t kPer = 2;
  __shared__ float sh[kNormVecBlocks * 8][kNormThreads];  // h, element (b, v) of flashinfer thread k at sh[b * 8 + v][k]
  __shared__ float red[kNormThreads / kWarpThreads];
  __shared__ float rstd_s;
  const bool has_prev = (t % q) >= 1;
  float h[kPer][8];
  site_vec_t yv[kPer], ypv[kPer], bv0[kPer], bv1[kPer], res[kPer];
#pragma unroll
  for (uint32_t s = 0; s < kPer; ++s) {
    const uint32_t bb = c + kEpiSlices * s;
    if (bb < kNormVecBlocks) {
      const uint32_t cv = k + kNormThreads * bb;
      bv0[s].load(base, cv);  // inputs no earlier kernel writes: before the PDL wait
      bv1[s].load(base + b_tap, cv);
    }
  }
  PDLWaitPrimary<kUsePDL>();
  const bf16_t* dl = delta + int64_t(t) * d_row;
#pragma unroll
  for (uint32_t s = 0; s < kPer; ++s) {
    const uint32_t bb = c + kEpiSlices * s;
    if (bb < kNormVecBlocks) {
      const uint32_t cv = k + kNormThreads * bb;
      yv[s].load(y, int64_t(t) * kRowVecs + cv);
      if (has_prev) {
        ypv[s].load(y, int64_t(t - 1) * kRowVecs + cv);
      } else {
#pragma unroll
        for (int e = 0; e < 4; ++e) reinterpret_cast<uint32_t*>(ypv[s].data())[e] = 0u;  // +0.0 (the masked Triton load)
      }
      res[s].load(residual, int64_t(t) * kRowVecs + cv);
    }
  }
#pragma unroll
  for (uint32_t s = 0; s < kPer; ++s) {
    const uint32_t bb = c + kEpiSlices * s;
    if (bb < kNormVecBlocks) {
      const uint32_t cv = k + kNormThreads * bb;
      const float d0 = bf1(dl[cv / 2]), d1 = bf1(dl[d_tap + cv / 2]);
      site_vec_t hb;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const auto [ya, yb] = cast<fp32x2_t>(yv[s][e]);
        const auto [pa, pb] = cast<fp32x2_t>(ypv[s][e]);
        const auto [ba0, bb0] = cast<fp32x2_t>(bv0[s][e]);
        const auto [ba1, bb1] = cast<fp32x2_t>(bv1[s][e]);
        const auto [ra, rb] = cast<fp32x2_t>(res[s][e]);
        const float oa = conv_pair(__fadd_rn(ba0, d0), ya, __fadd_rn(ba1, d1), pa);
        const float ob = conv_pair(__fadd_rn(bb0, d0), yb, __fadd_rn(bb1, d1), pb);
        const auto [xa, xb] = cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{oa, ob}));  // the conv output tensor is bf16
        h[s][2 * e] = __fadd_rn(xa, ra);
        h[s][2 * e + 1] = __fadd_rn(xb, rb);
        hb[e] = cast<bf16x2_t>(fp32x2_t{h[s][2 * e], h[s][2 * e + 1]});
      }
      hb.store(residual, int64_t(t) * kRowVecs + cv);
#pragma unroll
      for (int v = 0; v < 8; ++v) sh[bb * 8 + v][k] = h[s][v];
    }
  }
  __syncthreads();
  if (c == 0) {  // warps 0-3: flashinfer thread k, its warp = k / 32
    const uint32_t lane = k % kWarpThreads, warp = k / kWarpThreads;
    float acc = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kNormVecBlocks * 8; ++j) acc = __fmaf_rn(sh[j][k], sh[j][k], acc);
#pragma unroll
    for (uint32_t off = 1; off < kWarpThreads; off <<= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, off));
    if (lane == 0) red[warp] = acc;
    asm volatile("bar.sync 1, %0;" ::"r"(kNormThreads) : "memory");
    if (warp == 0) {
      float tot = lane < kNormThreads / kWarpThreads ? red[lane] : 0.0f;
#pragma unroll
      for (uint32_t off = 1; off < kWarpThreads; off <<= 1) tot = __fadd_rn(tot, __shfl_xor_sync(0xffffffffu, tot, off));
      if (lane == 0) rstd_s = rsqrt_approx_ftz(__fadd_rn(__fdiv_rn(tot, float(kHidden)), eps));
    }
  }
  __syncthreads();
  const float rstd = rstd_s;
#pragma unroll
  for (uint32_t s = 0; s < kPer; ++s) {
    const uint32_t bb = c + kEpiSlices * s;
    if (bb < kNormVecBlocks) {
      site_vec_t o, wv;
      wv.load(weight, k + kNormThreads * bb);  // after the reduction: keeps the load phase within 64 registers
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const auto [wa, wb] = cast<fp32x2_t>(wv[e]);
        const float oa = __fmul_rn(__fmul_rn(h[s][2 * e], rstd), __fadd_rn(wa, 0.0f));
        const float ob = __fmul_rn(__fmul_rn(h[s][2 * e + 1], rstd), __fadd_rn(wb, 0.0f));
        o[e] = cast<bf16x2_t>(fp32x2_t{oa, ob});
      }
      o.store(out, int64_t(t) * kRowVecs + k + kNormThreads * bb);
    }
  }
  PDLTriggerSecondary<kUsePDL>();
}

inline constexpr uint32_t kEpi896Threads = kNormThreads * kNormVecBlocks;  // 896: one thread per (flashinfer thread k, vector block b)

// v2 (kept for rows > 2 x SMs, where v3's 2-CTA-per-SM wave splits): 896 threads per row, thread i = (k = i % 128, b = i / 128) owns
// column vector k + 128 b, so every load of the row is in flight at once; the sum of squares still runs in flashinfer's order:
// thread k (< 128) chains fma over b = 0..6, v = 0..7 from shared memory, then the same butterfly / 4-warp reduction.
template <bool kUsePDL>
__global__ __launch_bounds__(kEpi896Threads, 1) void k31ht_draft_site_epi896_kernel(
    const bf16_t* __restrict__ y, const bf16_t* __restrict__ delta, const bf16_t* __restrict__ base, bf16_t* __restrict__ residual,
    const bf16_t* __restrict__ weight, bf16_t* __restrict__ out, int64_t d_row, int64_t d_tap, int64_t b_tap, uint32_t q, float eps) {
  using namespace device;
  const uint32_t t = blockIdx.x;
  const uint32_t i = threadIdx.x;
  const uint32_t k = i % kNormThreads, b = i / kNormThreads;
  const uint32_t cv = k + kNormThreads * b;  // columns 8 cv .. 8 cv + 7 (group cv / 2)
  __shared__ float sh[kNormVecBlocks * 8][kNormThreads];  // h, element (b, v) of flashinfer thread k at sh[b * 8 + v][k]
  __shared__ float red[kNormThreads / kWarpThreads];
  __shared__ float rstd_s;
  site_vec_t bv0, bv1, wv;
  bv0.load(base, cv);  // inputs that no earlier kernel writes: before the PDL wait
  bv1.load(base + b_tap, cv);
  wv.load(weight, cv);
  PDLWaitPrimary<kUsePDL>();
  const bool has_prev = (t % q) >= 1;
  const bf16_t* dl = delta + int64_t(t) * d_row;
  site_vec_t yv, ypv, res, hb;
  yv.load(y, int64_t(t) * kRowVecs + cv);
  if (has_prev) {
    ypv.load(y, int64_t(t - 1) * kRowVecs + cv);
  } else {
#pragma unroll
    for (int e = 0; e < 4; ++e) reinterpret_cast<uint32_t*>(ypv.data())[e] = 0u;  // +0.0 (the masked Triton load)
  }
  res.load(residual, int64_t(t) * kRowVecs + cv);
  const float d0 = bf1(dl[cv / 2]), d1 = bf1(dl[d_tap + cv / 2]);
  float h[8];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const auto [ya, yb] = cast<fp32x2_t>(yv[e]);
    const auto [pa, pb] = cast<fp32x2_t>(ypv[e]);
    const auto [ba0, bb0] = cast<fp32x2_t>(bv0[e]);
    const auto [ba1, bb1] = cast<fp32x2_t>(bv1[e]);
    const auto [ra, rb] = cast<fp32x2_t>(res[e]);
    const float oa = conv_pair(__fadd_rn(ba0, d0), ya, __fadd_rn(ba1, d1), pa);
    const float ob = conv_pair(__fadd_rn(bb0, d0), yb, __fadd_rn(bb1, d1), pb);
    const auto [xa, xb] = cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{oa, ob}));  // the conv output tensor is bf16
    h[2 * e] = __fadd_rn(xa, ra);
    h[2 * e + 1] = __fadd_rn(xb, rb);
    hb[e] = cast<bf16x2_t>(fp32x2_t{h[2 * e], h[2 * e + 1]});
  }
  hb.store(residual, int64_t(t) * kRowVecs + cv);
#pragma unroll
  for (int v = 0; v < 8; ++v) sh[b * 8 + v][k] = h[v];
  __syncthreads();
  if (b == 0) {  // warps 0-3: flashinfer thread k, its warp = k / 32
    const uint32_t lane = k % kWarpThreads, warp = k / kWarpThreads;
    float acc = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kNormVecBlocks * 8; ++j) acc = __fmaf_rn(sh[j][k], sh[j][k], acc);
#pragma unroll
    for (uint32_t off = 1; off < kWarpThreads; off <<= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, off));
    if (lane == 0) red[warp] = acc;
    asm volatile("bar.sync 1, %0;" ::"r"(kNormThreads) : "memory");
    if (warp == 0) {
      float tot = lane < kNormThreads / kWarpThreads ? red[lane] : 0.0f;
#pragma unroll
      for (uint32_t off = 1; off < kWarpThreads; off <<= 1) tot = __fadd_rn(tot, __shfl_xor_sync(0xffffffffu, tot, off));
      if (lane == 0) rstd_s = rsqrt_approx_ftz(__fadd_rn(__fdiv_rn(tot, float(kHidden)), eps));
    }
  }
  __syncthreads();
  const float rstd = rstd_s;
  site_vec_t o;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const auto [wa, wb] = cast<fp32x2_t>(wv[e]);
    const float oa = __fmul_rn(__fmul_rn(h[2 * e], rstd), __fadd_rn(wa, 0.0f));
    const float ob = __fmul_rn(__fmul_rn(h[2 * e + 1], rstd), __fadd_rn(wb, 0.0f));
    o[e] = cast<bf16x2_t>(fp32x2_t{oa, ob});
  }
  o.store(out, int64_t(t) * kRowVecs + cv);
  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct DraftSiteEpilogue {
  using Tensor = tvm::ffi::Tensor;

  static void
  run(const Tensor y,
      const Tensor delta,
      const Tensor base,
      const Tensor residual,
      const Tensor weight,
      const Tensor out,
      const double eps,
      const int64_t block_size) {
    using namespace host;
    auto N = SymbolicSize{"rows"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({N, kHidden}).with_strides({kHidden, 1}).with_dtype<bf16_t>().with_device(dev).verify(y);
    TensorMatcher({N, kHidden}).with_strides({kHidden, 1}).with_dtype<bf16_t>().with_device(dev).verify(residual);
    TensorMatcher({N, kHidden}).with_strides({kHidden, 1}).with_dtype<bf16_t>().with_device(dev).verify(out);
    TensorMatcher({N, kTaps, kGroups}).with_strides({-1, -1, 1}).with_dtype<bf16_t>().with_device(dev).verify(delta);
    TensorMatcher({kTaps, kHidden}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(dev).verify(base);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(dev).verify(weight);
    const auto rows = static_cast<uint32_t>(N.unwrap());
    CHECK_HOST(rows > 0 && block_size >= 1);
    CHECK_HOST(base.stride(0) % 8 == 0) << "base tap stride must keep 16 B alignment";
    const auto stream = LaunchKernel::resolve_device(y.device());
    // v4 dispatch (rows-aware): v3 (512 threads, 2 CTAs per SM) while all rows fit one wave, else v2 (896 threads); both are the
    // same arithmetic (bitwise).
    static const uint32_t kSm = [] {
      int dev = 0;
      CHECK_CUDA(cudaGetDevice(&dev));
      return static_cast<uint32_t>(host::runtime::get_sm_count(dev));
    }();
    const bool use896 = rows > 2 * kSm;
    const auto kern = use896 ? k31ht_draft_site_epi896_kernel<kUsePDL> : k31ht_draft_site_epi_kernel<kUsePDL>;
    LaunchKernel(rows, use896 ? kEpi896Threads : kEpiThreads, stream)
        .enable_pdl(kUsePDL)(
            kern,
            static_cast<const bf16_t*>(y.data_ptr()), static_cast<const bf16_t*>(delta.data_ptr()),
            static_cast<const bf16_t*>(base.data_ptr()), static_cast<bf16_t*>(residual.data_ptr()),
            static_cast<const bf16_t*>(weight.data_ptr()), static_cast<bf16_t*>(out.data_ptr()), delta.stride(0), delta.stride(1),
            base.stride(0), static_cast<uint32_t>(block_size), static_cast<float>(eps));
  }
};

}  // namespace k31ht_arsite
}  // namespace sglang
