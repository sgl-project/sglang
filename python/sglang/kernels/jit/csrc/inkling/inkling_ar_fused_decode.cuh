// Fused decode {all-reduce -> mlp/attn sconv -> residual-add + RMSNorm} for the
// Inkling (Moonrise) small-batch decode path -- the v5 push one-shot all-reduce
// (inkling_all_reduce.cuh) with the EPILOGUE SEAM filled in by the decode short-conv
// (fused_decode_update.cuh semantics) and the fused-add RMSNorm.
//
// Replaces THREE kernels (AR + fused_decode_update + fused_add_rmsnorm) and
// their intermediate HBM round trips with ONE launch per (AR, sconv, norm)
// chain. Layout: ONE BLOCK PER TOKEN (decode rows are few and the RMSNorm needs
// a per-row cross-hidden reduction), VPT 16B vecs (8 channels each) per thread
// -- a TUNED knob: fewer/fatter threads buy load ILP and a cheaper block
// reduction; more threads buy parallelism. Phases:
//
//   0. prefetch: sconv metadata, conv history and conv weights load FIRST --
//      none depend on the producer kernel's output, and their HBM latency
//      hides under the cross-GPU barrier below. (This -- not PDL -- is where
//      the fused kernel's latency win comes from: the producer GEMMs never
//      trigger programmatic launch early, so the PDL wait is a no-op in
//      practice and the launch attribute only pipelines the launch tail.)
//   1. push:    griddepcontrol.wait, then multicast-store this rank's partial
//               row into staging slot (rank*T + t)*D; issue the residual load.
//   2. barrier: per-block peer handshake (block t <-> peers' block t).
//   3. reduce:  fp32 sum of the kNumGPU staged shards; round to bf16 `xb`
//               (bit-identical to what the unfused AR would have stored).
//   4. sconv:   decode causal_conv1d on xb (W-1 cached taps gated by
//               cache_mask + current token), optional SiLU, optional +xb
//               residual; cache shift-update (+ optional track-copy) --
//               identical semantics to fused_decode_update_kernel.
//   5. norm:    r = residual_in + y (fp32); block-reduce sum(r^2); write
//               residual_out = bf16(r) and hs_out = bf16(r * rsqrt(mean+eps)
//               * gamma)  (fused_add_rmsnorm semantics).
//
// Staging reuse is caller-managed (A/B rotation shared with v5 -- this kernel
// IS a v5 AR occupying one rotation slot). PAD rows (cache_indices == -1)
// still compute y/hs but never write the cache, matching the unfused kernel.
// bf16-only.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "inkling_ar_barrier.cuh"
#include <bit>
#include <cstdint>
#include <cuda_bf16.h>
#include <type_traits>

namespace {

constexpr int kPadSlot = -1;
constexpr uint32_t kVecElems = 8;  // bf16x8 = 16 B

// Register-level fused add of two bf16x8 vecs (fp32 math, ONE round to bf16)
// -- torch.add numerics, so folding the shared-expert partials into the push
// stays bit-identical to the unfused {torch.add -> AR} chain.
__device__ __forceinline__ uint4 add_bf16x8_rn(const uint4 a, const uint4 b) {
  const auto* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
  const auto* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
  uint4 out;
  auto* o2 = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 x = __bfloat1622float2(a2[j]);
    const float2 y = __bfloat1622float2(b2[j]);
    o2[j] = __floats2bfloat162_rn(x.x + y.x, x.y + y.y);
  }
  return out;
}

struct ArSconvNormParams {
  // AR
  const void* __restrict__ in;      // [T, D] partial sums (LOCAL tensor)
  const void* __restrict__ shared;  // optional [T, D] shared-expert partials (LOCAL)
  void* __restrict__ mc_stage;      // multicast staging base (>= kNumGPU*T*D elems)
  const void* __restrict__ stage;   // this GPU's local view of the staging base
  void* const* __restrict__ flag_ptrs;
  uint32_t* __restrict__ state;
  // sconv (fused_decode_update semantics)
  void* __restrict__ cache;                // [pool, W-1, D], in-place update
  const void* __restrict__ cache_indices;  // int32 [T] (PAD == -1)
  const void* __restrict__ cache_mask;     // bool  [T]
  const void* __restrict__ conv_weight;    // [D, W]
  const void* __restrict__ track_mask;     // bool  [T]  (DO_TRACK only)
  const void* __restrict__ track_indices;  // int64 [T]  (DO_TRACK only)
  // norm
  const void* __restrict__ residual_in;  // [T, D]
  void* __restrict__ residual_out;       // [T, D]
  void* __restrict__ hs_out;             // [T, D]
  const void* __restrict__ norm_weight;  // [D]
  float eps;
  // strides (elements)
  int64_t in_stride_t;
  int64_t shared_stride_t;
  int64_t res_in_stride_t;
  int64_t res_out_stride_t;
  int64_t hs_stride_t;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t conv_weight_stride_d;
  int64_t track_idx_stride;
  uint32_t rank;
  uint32_t T;
  uint32_t D;
};

// VPT = 16B vecs handled per thread (tuning knob; see the header comment).
// Vec i of a thread is at index threadIdx.x + i*blockDim.x (warp-coalesced).
template <typename DType, uint32_t kNumGPU, int W, bool USE_SILU, bool USE_RESIDUAL, bool DO_TRACK, int VPT>
__global__ __launch_bounds__(1024, 1) void inkling_ar_sconv_norm_kernel(const __grid_constant__ ArSconvNormParams p) {
  static_assert(std::is_same_v<DType, __nv_bfloat16>, "multimem push path is bf16-only");
  constexpr int W1 = W - 1;
  const uint32_t t = blockIdx.x;
  const uint32_t vecs = p.D / kVecElems;

  uint32_t c0[VPT];
  bool act[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const uint32_t v = threadIdx.x + i * blockDim.x;
    act[i] = v < vecs;
    c0[i] = (act[i] ? v : 0) * kVecElems;  // clamp: inactive lanes never store
  }

  // ---- 0. prefetch (independent of the producer's output) ----
  const int ci = static_cast<const int32_t*>(p.cache_indices)[t];
  const bool valid = ci != kPadSlot;
  const int slot_id = valid ? ci : 0;  // PAD lanes still emit y, never write cache
  const float cm = static_cast<const bool*>(p.cache_mask)[t] ? 1.0f : 0.0f;
  auto* cp = static_cast<__nv_bfloat16*>(p.cache);
  const auto* wp = static_cast<const __nv_bfloat16*>(p.conv_weight);
  uint4 hist_raw[VPT][W1];
  __nv_bfloat16 wtaps[VPT][kVecElems][W];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    if (!act[i]) continue;
    const int64_t cache_base = static_cast<int64_t>(slot_id) * p.cache_stride_slot + c0[i];
#pragma unroll
    for (int w = 0; w < W1; ++w) {
      hist_raw[i][w] = *reinterpret_cast<const uint4*>(&cp[cache_base + w * p.cache_stride_w]);
    }
#pragma unroll
    for (int j = 0; j < static_cast<int>(kVecElems); ++j) {
      const int64_t wrow = static_cast<int64_t>(c0[i] + j) * p.conv_weight_stride_d;
      if constexpr (W == 4) {
        // One 8B load per channel row (bf16 x4, 8B-aligned for contiguous [D, W]).
        if (p.conv_weight_stride_d == W) {
          *reinterpret_cast<uint2*>(wtaps[i][j]) = *reinterpret_cast<const uint2*>(wp + wrow);
          continue;
        }
      }
#pragma unroll
      for (int w = 0; w < W; ++w)
        wtaps[i][j][w] = wp[wrow + w];
    }
  }

  // ---- 1. push: wait for the producer's output (PDL; no-op without a PDL
  // launch or an early-triggering producer), multicast-store this rank's
  // partial row, and issue the residual load (it lands under the barrier). ----
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const auto* in_row = static_cast<const __nv_bfloat16*>(p.in) + t * p.in_stride_t;
  const auto* sh_row =
      p.shared == nullptr ? nullptr : static_cast<const __nv_bfloat16*>(p.shared) + t * p.shared_stride_t;
  auto* slot = static_cast<__nv_bfloat16*>(p.mc_stage) + (static_cast<uint64_t>(p.rank) * p.T + t) * p.D;
  uint4 res_raw[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    if (!act[i]) continue;
    uint4 d = *reinterpret_cast<const uint4*>(in_row + c0[i]);
    if (sh_row != nullptr) {
      d = add_bf16x8_rn(d, *reinterpret_cast<const uint4*>(sh_row + c0[i]));
    }
    asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(slot + c0[i]),
                 "r"(d.x),
                 "r"(d.y),
                 "r"(d.z),
                 "r"(d.w)
                 : "memory");
    res_raw[i] = *reinterpret_cast<const uint4*>(
        static_cast<const __nv_bfloat16*>(p.residual_in) + t * p.res_in_stride_t + c0[i]);
  }

  // ---- 2. per-block barrier: all ranks' row-t pushes have landed locally ----
  inkling_ar::block_system_barrier<kNumGPU>(p.state, p.flag_ptrs, p.rank);
  // Inactive lanes must NOT exit: they participate in the norm's __syncthreads
  // and full-mask warp shuffles below (sumsq contribution 0).

  float r[VPT][kVecElems];
  float sumsq = 0.0f;
  const auto* stage = static_cast<const __nv_bfloat16*>(p.stage);
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    if (!act[i]) continue;
    // ---- 3. reduce: fp32 sum of the kNumGPU staged shards; round to bf16 ----
    float xf[kVecElems];
#pragma unroll
    for (int j = 0; j < static_cast<int>(kVecElems); ++j)
      xf[j] = 0.0f;
#pragma unroll
    for (uint32_t rr = 0; rr < kNumGPU; ++rr) {
      const uint4 d = *reinterpret_cast<const uint4*>(stage + (static_cast<uint64_t>(rr) * p.T + t) * p.D + c0[i]);
      const auto* h2 = reinterpret_cast<const __nv_bfloat162*>(&d);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(h2[j]);
        xf[2 * j] += f.x;
        xf[2 * j + 1] += f.y;
      }
    }
    // Round to bf16 exactly as the unfused AR's store would (the sconv below
    // and the cache append must see the same bits the unfused path sees).
    __nv_bfloat162 xb2[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      xb2[j] = __floats2bfloat162_rn(xf[2 * j], xf[2 * j + 1]);

    // ---- 4. sconv: conv over W-1 cached taps (prefetched) + current token ----
    float y[kVecElems];
#pragma unroll
    for (int j = 0; j < static_cast<int>(kVecElems); ++j) {
      const float xj = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(xb2)[j]);
      float acc = 0.0f;
#pragma unroll
      for (int w = 0; w < W1; ++w) {
        const float h = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&hist_raw[i][w])[j]);
        acc += h * cm * __bfloat162float(wtaps[i][j][w]);
      }
      acc += xj * __bfloat162float(wtaps[i][j][W1]);
      if constexpr (USE_SILU) acc = __fdividef(acc, 1.0f + __expf(-acc));
      if constexpr (USE_RESIDUAL) acc += xj;
      y[j] = acc;
    }

    if (valid) {
      // Shift state left (gated by cache_mask), append current token (xb).
      const int64_t cache_base = static_cast<int64_t>(slot_id) * p.cache_stride_slot + c0[i];
      int64_t track_base = 0;
      bool do_tr = false;
      if constexpr (DO_TRACK) {
        do_tr = static_cast<const bool*>(p.track_mask)[t];
        if (do_tr) {
          const int64_t tslot =
              static_cast<const int64_t*>(p.track_indices)[static_cast<int64_t>(t) * p.track_idx_stride];
          track_base = tslot * p.cache_stride_slot + c0[i];
        }
      }
      const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int w = 0; w < W1; ++w) {
        const uint4 nv =
            (w < W1 - 1) ? ((cm != 0.0f) ? hist_raw[i][w + 1] : zero) : *reinterpret_cast<const uint4*>(xb2);
        *reinterpret_cast<uint4*>(&cp[cache_base + w * p.cache_stride_w]) = nv;
        if constexpr (DO_TRACK) {
          if (do_tr) {
            *reinterpret_cast<uint4*>(&cp[track_base + w * p.cache_stride_w]) = nv;
          }
        }
      }
    }

    // ---- 5a. residual add (fused_add_rmsnorm semantics) ----
    // yb: round the sconv output to bf16 first -- the unfused path writes y to
    // HBM as bf16 before the norm kernel reads it back.
#pragma unroll
    for (int j = 0; j < static_cast<int>(kVecElems); ++j) {
      const float yb = __bfloat162float(__float2bfloat16_rn(y[j]));
      r[i][j] = yb + __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&res_raw[i])[j]);
      sumsq += r[i][j] * r[i][j];
    }
  }

  // ---- 5b. block reduction of sumsq (warp shuffle + one smem slot/warp) ----
  __shared__ float s_warp[32];
  __shared__ float s_inv;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    sumsq += __shfl_down_sync(~0u, sumsq, off);
  if (lane == 0) s_warp[warp] = sumsq;
  __syncthreads();
  if (warp == 0) {
    const uint32_t nwarps = (blockDim.x + 31u) >> 5;
    float total = (lane < nwarps && lane < 32u) ? s_warp[lane] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      total += __shfl_down_sync(~0u, total, off);
    if (lane == 0) s_inv = rsqrtf(total / static_cast<float>(p.D) + p.eps);
  }
  __syncthreads();
  const float inv = s_inv;

  const auto* gw = static_cast<const __nv_bfloat16*>(p.norm_weight);
  auto* res_out = static_cast<__nv_bfloat16*>(p.residual_out) + t * p.res_out_stride_t;
  auto* hs_out = static_cast<__nv_bfloat16*>(p.hs_out) + t * p.hs_stride_t;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    if (!act[i]) continue;
    __nv_bfloat162 ro[4], ho[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float g0 = __bfloat162float(gw[c0[i] + 2 * j]);
      const float g1 = __bfloat162float(gw[c0[i] + 2 * j + 1]);
      ro[j] = __floats2bfloat162_rn(r[i][2 * j], r[i][2 * j + 1]);
      ho[j] = __floats2bfloat162_rn(r[i][2 * j] * inv * g0, r[i][2 * j + 1] * inv * g1);
    }
    *reinterpret_cast<uint4*>(res_out + c0[i]) = *reinterpret_cast<const uint4*>(ro);
    *reinterpret_cast<uint4*>(hs_out + c0[i]) = *reinterpret_cast<const uint4*>(ho);
  }
}

// ---------------------------------------------------------------------------
// Target-verify variant: {AR -> extend-style causal_conv1d ->
// save_intermediate_conv_windows -> add+RMSNorm} in one launch. Every sequence
// has exactly `q` (draft_token_num) consecutive tokens; token t belongs to
// seq = t/q with bos = seq*q. The conv's cross-token taps are RE-REDUCED from
// the v5 staging buffer (any block can rebuild any token's reduced row by
// summing the staged shards -- ~kNumGPU x 16B extra local L2 reads per tap, no
// cross-block dependency). The conv does NOT update the working cache at
// verify; instead the per-position windows are written to intermediate_out
// (consumed by update_conv_state_after_mtp_verify), whose values are exactly
// the cache prefix rows and the re-reduced x this kernel already holds.
struct ArSconvNormVerifyParams {
  const void* __restrict__ in;      // [T, D] partial sums (LOCAL tensor)
  const void* __restrict__ shared;  // optional [T, D] shared-expert partials (LOCAL)
  void* __restrict__ mc_stage;      // multicast staging base
  const void* __restrict__ stage;   // this GPU's local view of the staging base
  void* const* __restrict__ flag_ptrs;
  uint32_t* __restrict__ state;
  const void* __restrict__ cache;          // [pool, W-1, D] (read-only here)
  const void* __restrict__ cache_indices;  // int32 [B] per-SEQ slot (PAD == -1)
  const void* __restrict__ cache_mask;     // bool  [B] per-SEQ prefix gate
  const void* __restrict__ conv_weight;    // [D, W]
  void* __restrict__ inter_out;            // [max_bs, q, W-1, D]
  const void* __restrict__ residual_in;    // [T, D]
  void* __restrict__ residual_out;         // [T, D]
  void* __restrict__ hs_out;               // [T, D]
  const void* __restrict__ norm_weight;    // [D]
  float eps;
  int64_t in_stride_t;
  int64_t shared_stride_t;
  int64_t res_in_stride_t;
  int64_t res_out_stride_t;
  int64_t hs_stride_t;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t conv_weight_stride_d;
  int64_t inter_stride_b;
  int64_t inter_stride_t;
  int64_t inter_stride_w;
  uint32_t rank;
  uint32_t T;
  uint32_t D;
  uint32_t q;  // draft_token_num
};

template <typename DType, uint32_t kNumGPU, int W, bool USE_SILU, bool USE_RESIDUAL>
__global__
__launch_bounds__(1024, 1) void inkling_ar_sconv_norm_verify_kernel(const __grid_constant__ ArSconvNormVerifyParams p) {
  static_assert(std::is_same_v<DType, __nv_bfloat16>, "multimem push path is bf16-only");
  constexpr int W1 = W - 1;
  const uint32_t vecs = p.D / kVecElems;
  const uint32_t v = threadIdx.x;  // one 16B vec (8 channels) per thread
  const bool active = v < vecs;
  const uint32_t c0 = (active ? v : 0) * kVecElems;
  const uint32_t stride_t = gridDim.x;  // grid-stride over tokens

  // Conv weights are token-independent (per channel) -- load once.
  const auto* wp = static_cast<const __nv_bfloat16*>(p.conv_weight);
  __nv_bfloat16 wtaps[kVecElems][W];
  if (active) {
#pragma unroll
    for (int j = 0; j < static_cast<int>(kVecElems); ++j) {
      const int64_t wrow = static_cast<int64_t>(c0 + j) * p.conv_weight_stride_d;
      if constexpr (W == 4) {
        if (p.conv_weight_stride_d == W) {
          *reinterpret_cast<uint2*>(wtaps[j]) = *reinterpret_cast<const uint2*>(wp + wrow);
          continue;
        }
      }
#pragma unroll
      for (int w = 0; w < W; ++w)
        wtaps[j][w] = wp[wrow + w];
    }
  }

  // ---- Phase 1: push every assigned row into staging (PDL-gated input). ----
  // A single grid barrier (below) then makes ALL rows' pushes visible on this
  // GPU, so Phase 2's cross-token (neighbor) staging reads are race-free -- the
  // per-block barrier only synchronized the same blockIdx across ranks and did
  // NOT order block t-j's push before block t's read.
  asm volatile("griddepcontrol.wait;" ::: "memory");
  auto* mc = static_cast<__nv_bfloat16*>(p.mc_stage);
  const auto* in = static_cast<const __nv_bfloat16*>(p.in);
  const auto* sh = static_cast<const __nv_bfloat16*>(p.shared);
  if (active) {
    for (uint32_t t = blockIdx.x; t < p.T; t += stride_t) {
      uint4 d = *reinterpret_cast<const uint4*>(in + t * p.in_stride_t + c0);
      if (sh != nullptr) {
        // Fold the shared-expert partials in registers (torch.add numerics);
        // the staged value then matches the unfused pre-added input, so the
        // cross-token re-reduces below stay bit-identical too.
        d = add_bf16x8_rn(d, *reinterpret_cast<const uint4*>(sh + t * p.shared_stride_t + c0));
      }
      auto* slot = mc + (static_cast<uint64_t>(p.rank) * p.T + t) * p.D + c0;
      asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(slot),
                   "r"(d.x),
                   "r"(d.y),
                   "r"(d.z),
                   "r"(d.w)
                   : "memory");
    }
  }

  // ---- Grid barrier: all pushes done + system-visible across all ranks. ----
  inkling_ar::grid_system_barrier<kNumGPU>(
      p.state,
      p.flag_ptrs,
      p.rank,
      0,
      /*publish_writes=*/true);

  // ---- Phase 2: reduce + conv + save_windows + add-RMSNorm per row. ----
  const auto* stage = static_cast<const __nv_bfloat16*>(p.stage);
  const auto* cp = static_cast<const __nv_bfloat16*>(p.cache);
  const auto* gw = static_cast<const __nv_bfloat16*>(p.norm_weight);
  __shared__ float s_warp[32];
  __shared__ float s_inv;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = threadIdx.x >> 5;

  auto reduce_row = [&](uint32_t row, __nv_bfloat162* out2) {
    float xf[kVecElems];
#pragma unroll
    for (int j = 0; j < static_cast<int>(kVecElems); ++j)
      xf[j] = 0.0f;
#pragma unroll
    for (uint32_t rr = 0; rr < kNumGPU; ++rr) {
      const uint4 d = *reinterpret_cast<const uint4*>(stage + (static_cast<uint64_t>(rr) * p.T + row) * p.D + c0);
      const auto* h2 = reinterpret_cast<const __nv_bfloat162*>(&d);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(h2[j]);
        xf[2 * j] += f.x;
        xf[2 * j + 1] += f.y;
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
      out2[j] = __floats2bfloat162_rn(xf[2 * j], xf[2 * j + 1]);
  };

  for (uint32_t t = blockIdx.x; t < p.T; t += stride_t) {
    const uint32_t seq = t / p.q;
    const uint32_t tq = t - seq * p.q;
    const int bos = static_cast<int>(seq * p.q);
    const int ci = static_cast<const int32_t*>(p.cache_indices)[seq];
    const bool valid = ci != kPadSlot;
    const int slot_id = valid ? ci : 0;
    const float cm = (valid && static_cast<const bool*>(p.cache_mask)[seq]) ? 1.0f : 0.0f;
    const int64_t cache_base = static_cast<int64_t>(slot_id) * p.cache_stride_slot + c0;

    float r[kVecElems];
    float sumsq = 0.0f;
    if (active) {
      uint4 pref_raw[W1];
#pragma unroll
      for (int w = 0; w < W1; ++w) {
        pref_raw[w] = *reinterpret_cast<const uint4*>(&cp[cache_base + w * p.cache_stride_w]);
      }
      const uint4 res_raw = *reinterpret_cast<const uint4*>(
          static_cast<const __nv_bfloat16*>(p.residual_in) + t * p.res_in_stride_t + c0);

      __nv_bfloat162 xb2[4];      // own row
      __nv_bfloat162 xn2[W1][4];  // neighbors t-1 .. t-(W-1), where in-seq
      reduce_row(t, xb2);
#pragma unroll
      for (int j = 1; j <= W1; ++j) {
        const int n = static_cast<int>(t) - j;
        if (n >= bos) reduce_row(static_cast<uint32_t>(n), xn2[j - 1]);
      }

      // conv (jit causal_conv1d semantics, fp32 accum, ascending tap order).
      float y[kVecElems];
#pragma unroll
      for (int j = 0; j < static_cast<int>(kVecElems); ++j) {
        const float xj = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(xb2)[j]);
        float acc = 0.0f;
#pragma unroll
        for (int iw = 0; iw < W1; ++iw) {
          const int shifted = static_cast<int>(t) - W1 + iw;
          float tap = 0.0f;
          if (shifted >= bos) {
            tap = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(xn2[W1 - 1 - iw])[j]);
          } else {
            const int prefix_pos = shifted - bos + W1;
            if (prefix_pos >= 0) {
              tap = cm * __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&pref_raw[prefix_pos])[j]);
            }
          }
          acc += tap * __bfloat162float(wtaps[j][iw]);
        }
        acc += xj * __bfloat162float(wtaps[j][W1]);
        if constexpr (USE_SILU) acc = __fdividef(acc, 1.0f + __expf(-acc));
        if constexpr (USE_RESIDUAL) acc += xj;
        y[j] = acc;
      }

      // save_intermediate_conv_windows: window after draft position tq is raw
      // copies of {cache prefix rows | reduced x rows} (no cm gating).
      if (valid) {
        auto* op = static_cast<__nv_bfloat16*>(p.inter_out) + static_cast<int64_t>(seq) * p.inter_stride_b +
                   static_cast<int64_t>(tq) * p.inter_stride_t + c0;
#pragma unroll
        for (int w = 0; w < W1; ++w) {
          const int position = static_cast<int>(tq) + 1 + w;
          uint4 val;
          if (position < W1) {
            val = pref_raw[position];
          } else {
            const int g = bos + position - W1;
            val = (g == static_cast<int>(t)) ? *reinterpret_cast<const uint4*>(xb2)
                                             : *reinterpret_cast<const uint4*>(xn2[t - g - 1]);
          }
          *reinterpret_cast<uint4*>(op + w * p.inter_stride_w) = val;
        }
      }

      // residual add.
#pragma unroll
      for (int j = 0; j < static_cast<int>(kVecElems); ++j) {
        const float yb = __bfloat162float(__float2bfloat16_rn(y[j]));
        r[j] = yb + __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&res_raw)[j]);
        sumsq += r[j] * r[j];
      }
    }

    // block reduction of sumsq (all threads participate; inactive contribute 0).
    __syncthreads();  // protect s_warp/s_inv reuse across the token loop
    float ss = sumsq;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      ss += __shfl_down_sync(~0u, ss, off);
    if (lane == 0) s_warp[warp] = ss;
    __syncthreads();
    if (warp == 0) {
      const uint32_t nwarps = (blockDim.x + 31u) >> 5;
      float total = (lane < nwarps && lane < 32u) ? s_warp[lane] : 0.0f;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        total += __shfl_down_sync(~0u, total, off);
      if (lane == 0) s_inv = rsqrtf(total / static_cast<float>(p.D) + p.eps);
    }
    __syncthreads();
    const float inv = s_inv;

    if (active) {
      auto* res_out = static_cast<__nv_bfloat16*>(p.residual_out) + t * p.res_out_stride_t;
      auto* hs_out = static_cast<__nv_bfloat16*>(p.hs_out) + t * p.hs_stride_t;
      __nv_bfloat162 ro[4], ho[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float g0 = __bfloat162float(gw[c0 + 2 * j]);
        const float g1 = __bfloat162float(gw[c0 + 2 * j + 1]);
        ro[j] = __floats2bfloat162_rn(r[2 * j], r[2 * j + 1]);
        ho[j] = __floats2bfloat162_rn(r[2 * j] * inv * g0, r[2 * j + 1] * inv * g1);
      }
      *reinterpret_cast<uint4*>(res_out + c0) = *reinterpret_cast<const uint4*>(ro);
      *reinterpret_cast<uint4*>(hs_out + c0) = *reinterpret_cast<const uint4*>(ho);
    }
  }
}

template <typename DType, uint32_t kNumGPU, int W, bool USE_SILU, bool USE_RESIDUAL, bool DO_TRACK>
struct ArSconvNormKernel {
  template <int VPT>
  static void launch(const ArSconvNormParams& params, uint32_t t_num, uint32_t vecs, DLDevice dev, bool pdl) {
    using namespace host;
    const uint32_t block = min(1024u, div_ceil(div_ceil(vecs, VPT), 32u) * 32u);
    constexpr auto kernel = inkling_ar_sconv_norm_kernel<DType, kNumGPU, W, USE_SILU, USE_RESIDUAL, DO_TRACK, VPT>;
    LaunchKernel(dim3{t_num}, dim3{block}, dev).enable_pdl(pdl)(kernel, params);
  }

  static void
  run(tvm::ffi::TensorView in,
      tvm::ffi::TensorView residual_in,
      tvm::ffi::TensorView residual_out,
      tvm::ffi::TensorView hs_out,
      tvm::ffi::TensorView norm_weight,
      double eps,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView cache_indices,
      tvm::ffi::TensorView cache_mask,
      tvm::ffi::TensorView conv_weight,
      tvm::ffi::TensorView track_mask,
      tvm::ffi::TensorView track_indices,
      int64_t mc_stage_ptr,
      int64_t local_stage_ptr,
      int64_t flag_ptrs_dev,
      int64_t state_ptr,
      int64_t rank,
      int64_t enable_pdl,
      int64_t vecs_per_thread,
      tvm::ffi::TensorView shared) {
    using namespace host;
    auto T = SymbolicSize{"T"};
    auto D = SymbolicSize{"D"};
    auto Wd = SymbolicSize{"W"};
    auto W1s = SymbolicSize{"W_minus_1"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    Wd.set_value(W);
    W1s.set_value(W - 1);

    TensorMatcher({T, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(in);
    const bool do_shared = shared.numel() > 0;
    if (do_shared) {
      TensorMatcher({T, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(shared);
      RuntimeCheck(shared.stride(0) % kVecElems == 0, "shared row stride must keep 16B alignment");
      RuntimeCheck(std::bit_cast<intptr_t>(shared.data_ptr()) % 16 == 0, "shared not 16B aligned");
    }
    TensorMatcher({T, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(residual_in);
    TensorMatcher({T, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(residual_out);
    TensorMatcher({T, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(hs_out);
    TensorMatcher({D}).with_dtype<DType>().with_device(dev).verify(norm_weight);
    TensorMatcher({-1, W1s, D}).with_dtype<DType>().with_device(dev).verify(cache);
    TensorMatcher({T}).with_dtype<int32_t>().with_device(dev).verify(cache_indices);
    TensorMatcher({T}).with_device(dev).verify(cache_mask);
    TensorMatcher({D, Wd}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(conv_weight);
    const uint32_t t_num = static_cast<uint32_t>(T.unwrap());
    const uint32_t d_num = static_cast<uint32_t>(D.unwrap());
    const uint32_t vecs = d_num / kVecElems;
    RuntimeCheck(
        t_num >= 1 && t_num <= inkling_ar::kMaxBarrierBlocks,
        "T must be in [1, kMaxBarrierBlocks] (one barrier slot per token)");
    RuntimeCheck(d_num % kVecElems == 0, "D must be a multiple of 8");
    RuntimeCheck(cache.stride(2) == 1, "cache must be channel-contiguous");
    RuntimeCheck(mc_stage_ptr % 16 == 0, "mc_stage_ptr not 16B aligned");
    RuntimeCheck(local_stage_ptr != 0 && local_stage_ptr % 16 == 0, "bad local_stage_ptr");
    RuntimeCheck(flag_ptrs_dev != 0, "flag_ptrs_dev is null");
    RuntimeCheck(state_ptr != 0, "state_ptr is null");
    RuntimeCheck(rank >= 0 && rank < kNumGPU, "rank out of range");
    RuntimeCheck(in.stride(0) % kVecElems == 0, "in row stride must keep 16B alignment");
    RuntimeCheck(std::bit_cast<intptr_t>(in.data_ptr()) % 16 == 0, "in not 16B aligned");

    const auto params = ArSconvNormParams{
        .in = in.data_ptr(),
        .shared = do_shared ? shared.data_ptr() : nullptr,
        .mc_stage = reinterpret_cast<void*>(mc_stage_ptr),
        .stage = reinterpret_cast<const void*>(local_stage_ptr),
        .flag_ptrs = reinterpret_cast<void* const*>(flag_ptrs_dev),
        .state = reinterpret_cast<uint32_t*>(state_ptr),
        .cache = cache.data_ptr(),
        .cache_indices = cache_indices.data_ptr(),
        .cache_mask = cache_mask.data_ptr(),
        .conv_weight = conv_weight.data_ptr(),
        .track_mask = DO_TRACK ? track_mask.data_ptr() : nullptr,
        .track_indices = DO_TRACK ? track_indices.data_ptr() : nullptr,
        .residual_in = residual_in.data_ptr(),
        .residual_out = residual_out.data_ptr(),
        .hs_out = hs_out.data_ptr(),
        .norm_weight = norm_weight.data_ptr(),
        .eps = static_cast<float>(eps),
        .in_stride_t = in.stride(0),
        .shared_stride_t = do_shared ? shared.stride(0) : 0,
        .res_in_stride_t = residual_in.stride(0),
        .res_out_stride_t = residual_out.stride(0),
        .hs_stride_t = hs_out.stride(0),
        .cache_stride_slot = cache.stride(0),
        .cache_stride_w = cache.stride(1),
        .conv_weight_stride_d = conv_weight.stride(0),
        .track_idx_stride = DO_TRACK ? track_indices.stride(0) : 0,
        .rank = static_cast<uint32_t>(rank),
        .T = t_num,
        .D = d_num,
    };

    // vecs_per_thread (VPT) is the tuned knob; 0 -> 1. Each VPT must still fit
    // one block (div_ceil(vecs, VPT) <= 1024).
    const int vpt = vecs_per_thread > 0 ? static_cast<int>(vecs_per_thread) : 1;
    const bool pdl = enable_pdl != 0;
    switch (vpt) {
      case 1:
        RuntimeCheck(vecs <= 1024, "D/8 must fit one block at VPT=1");
        launch<1>(params, t_num, vecs, dev.unwrap(), pdl);
        break;
      case 2:
        launch<2>(params, t_num, vecs, dev.unwrap(), pdl);
        break;
      case 3:
        launch<3>(params, t_num, vecs, dev.unwrap(), pdl);
        break;
      case 4:
        launch<4>(params, t_num, vecs, dev.unwrap(), pdl);
        break;
      case 6:
        launch<6>(params, t_num, vecs, dev.unwrap(), pdl);
        break;
      default:
        RuntimeCheck(false, "unsupported vecs_per_thread (use 1/2/3/4/6)");
    }
  }
};

// Host wrapper for the target-verify variant. DO_TRACK is accepted (to share
// the module's template-arg string) but unused -- verify never tracks.
template <typename DType, uint32_t kNumGPU, int W, bool USE_SILU, bool USE_RESIDUAL, bool DO_TRACK>
struct ArSconvNormVerifyKernel {
  static void
  run(tvm::ffi::TensorView in,
      tvm::ffi::TensorView residual_in,
      tvm::ffi::TensorView residual_out,
      tvm::ffi::TensorView hs_out,
      tvm::ffi::TensorView norm_weight,
      double eps,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView cache_indices,
      tvm::ffi::TensorView cache_mask,
      tvm::ffi::TensorView conv_weight,
      tvm::ffi::TensorView inter_out,
      int64_t q,
      int64_t mc_stage_ptr,
      int64_t local_stage_ptr,
      int64_t flag_ptrs_dev,
      int64_t state_ptr,
      int64_t rank,
      int64_t enable_pdl,
      tvm::ffi::TensorView shared) {
    using namespace host;
    auto T = SymbolicSize{"T"};
    auto B = SymbolicSize{"B"};
    auto D = SymbolicSize{"D"};
    auto Wd = SymbolicSize{"W"};
    auto W1s = SymbolicSize{"W_minus_1"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    Wd.set_value(W);
    W1s.set_value(W - 1);

    TensorMatcher({T, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(in);
    const bool do_shared = shared.numel() > 0;
    if (do_shared) {
      TensorMatcher({T, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(shared);
      RuntimeCheck(shared.stride(0) % kVecElems == 0, "shared row stride must keep 16B alignment");
      RuntimeCheck(std::bit_cast<intptr_t>(shared.data_ptr()) % 16 == 0, "shared not 16B aligned");
    }
    TensorMatcher({T, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(residual_in);
    TensorMatcher({T, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(residual_out);
    TensorMatcher({T, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(hs_out);
    TensorMatcher({D}).with_dtype<DType>().with_device(dev).verify(norm_weight);
    TensorMatcher({-1, W1s, D}).with_dtype<DType>().with_device(dev).verify(cache);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(dev).verify(cache_indices);
    TensorMatcher({B}).with_device(dev).verify(cache_mask);
    TensorMatcher({D, Wd}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(conv_weight);
    const uint32_t t_num = static_cast<uint32_t>(T.unwrap());
    const uint32_t b_num = static_cast<uint32_t>(B.unwrap());
    const uint32_t d_num = static_cast<uint32_t>(D.unwrap());
    RuntimeCheck(q > 0 && t_num == b_num * static_cast<uint32_t>(q), "T must equal B * draft_token_num");
    RuntimeCheck(
        t_num >= 1 && t_num <= inkling_ar::kMaxBarrierBlocks,
        "T must be in [1, kMaxBarrierBlocks] (one barrier slot per token)");
    RuntimeCheck(d_num % kVecElems == 0, "D must be a multiple of 8");
    RuntimeCheck(d_num / kVecElems <= 1024, "D/8 must fit one block");
    RuntimeCheck(cache.stride(2) == 1, "cache must be channel-contiguous");
    // inter_out: [max_bs, q, W-1, D], channel-contiguous, batch B rows used.
    auto MB = SymbolicSize{"max_bs"};
    auto Qs = SymbolicSize{"q"};
    Qs.set_value(q);
    TensorMatcher({MB, Qs, W1s, D}).with_dtype<DType>().with_device(dev).verify(inter_out);
    RuntimeCheck(MB.unwrap() >= b_num, "inter_out batch dim too small");
    RuntimeCheck(inter_out.stride(3) == 1, "inter_out must be channel-contiguous");
    RuntimeCheck(mc_stage_ptr % 16 == 0, "mc_stage_ptr not 16B aligned");
    RuntimeCheck(local_stage_ptr != 0 && local_stage_ptr % 16 == 0, "bad local_stage_ptr");
    RuntimeCheck(flag_ptrs_dev != 0 && state_ptr != 0, "null barrier resources");
    RuntimeCheck(rank >= 0 && rank < kNumGPU, "rank out of range");
    RuntimeCheck(in.stride(0) % kVecElems == 0, "in row stride must keep 16B alignment");
    RuntimeCheck(std::bit_cast<intptr_t>(in.data_ptr()) % 16 == 0, "in not 16B aligned");

    const auto params = ArSconvNormVerifyParams{
        .in = in.data_ptr(),
        .shared = do_shared ? shared.data_ptr() : nullptr,
        .mc_stage = reinterpret_cast<void*>(mc_stage_ptr),
        .stage = reinterpret_cast<const void*>(local_stage_ptr),
        .flag_ptrs = reinterpret_cast<void* const*>(flag_ptrs_dev),
        .state = reinterpret_cast<uint32_t*>(state_ptr),
        .cache = cache.data_ptr(),
        .cache_indices = cache_indices.data_ptr(),
        .cache_mask = cache_mask.data_ptr(),
        .conv_weight = conv_weight.data_ptr(),
        .inter_out = inter_out.data_ptr(),
        .residual_in = residual_in.data_ptr(),
        .residual_out = residual_out.data_ptr(),
        .hs_out = hs_out.data_ptr(),
        .norm_weight = norm_weight.data_ptr(),
        .eps = static_cast<float>(eps),
        .in_stride_t = in.stride(0),
        .shared_stride_t = do_shared ? shared.stride(0) : 0,
        .res_in_stride_t = residual_in.stride(0),
        .res_out_stride_t = residual_out.stride(0),
        .hs_stride_t = hs_out.stride(0),
        .cache_stride_slot = cache.stride(0),
        .cache_stride_w = cache.stride(1),
        .conv_weight_stride_d = conv_weight.stride(0),
        .inter_stride_b = inter_out.stride(0),
        .inter_stride_t = inter_out.stride(1),
        .inter_stride_w = inter_out.stride(2),
        .rank = static_cast<uint32_t>(rank),
        .T = t_num,
        .D = d_num,
        .q = static_cast<uint32_t>(q),
    };

    const uint32_t block = min(1024u, div_ceil(d_num / kVecElems, 32u) * 32u);
    constexpr auto kernel = inkling_ar_sconv_norm_verify_kernel<DType, kNumGPU, W, USE_SILU, USE_RESIDUAL>;
    // The kernel grid-strides over tokens with ONE grid_system_barrier between
    // the push and the neighbor-reading reduce, so all blocks must be
    // co-resident (else the leader waits forever). Cap the grid at the
    // occupancy limit; the token loop covers any remaining rows.
    const uint32_t bps = host::runtime::get_blocks_per_sm(kernel, block);
    const uint32_t cap = host::runtime::get_sm_count(dev.unwrap().device_id) * max(1u, bps);
    const uint32_t grid = min(t_num, cap);
    LaunchKernel(dim3{grid}, dim3{block}, dev.unwrap()).enable_pdl(enable_pdl != 0)(kernel, params);
  }
};

}  // namespace
