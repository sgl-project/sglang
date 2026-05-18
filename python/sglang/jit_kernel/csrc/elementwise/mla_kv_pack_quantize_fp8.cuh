#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct MlaKVPackQuantizeFp8Params {
  void* __restrict__ k_out;
  void* __restrict__ v_out;
  const void* __restrict__ k_nope;
  const void* __restrict__ k_pe;
  const void* __restrict__ v;
  float k_scale_inv;
  float v_scale_inv;
  uint32_t s_total;
  uint32_t num_heads;
  int64_t k_nope_stride_t;
  int64_t k_nope_stride_h;
  int64_t k_pe_stride_t;
  int64_t v_stride_t;
  int64_t v_stride_h;
  int64_t k_out_stride_t;
  int64_t k_out_stride_h;
  int64_t v_out_stride_t;
  int64_t v_out_stride_h;
};

// Each CTA handles kBlockS tokens × 1 head. Work is split into three phases
// (k_nope, k_pe, v), and within each phase the (sl, vi) work items are
// distributed *flatly* across all kThreads — every thread participates in
// every phase, eliminating the wasted-lanes problem of one-warp-per-phase
// designs when kThreads exceeds the per-token vec count.
//
// Fast / slow path: when all kBlockS tokens are in-range (the common case for
// non-tail CTAs), the per-thread iteration count is a compile-time constant.
// We split that path off and batch all in-phase loads into a register array
// before any stores, so the compiler emits all `LDG.E.*` instructions
// back-to-back, raising the count of outstanding HBM transactions per thread
// and hiding more latency. The tail-block path keeps the original
// variable-iter loop so we don't pay extra register pressure when masking.
template <
    typename TIn,
    typename TOut,
    int kQkNope,
    int kQkRope,
    int kVHead,
    int kVecN,
    int kBlockS,
    int kThreads,
    bool kUsePDL>
__global__ void mla_kv_pack_quantize_fp8_kernel(__grid_constant__ const MlaKVPackQuantizeFp8Params params) {
  using namespace device;
  using vec_in_t = AlignedVector<TIn, kVecN>;
  using vec_out_t = AlignedVector<TOut, kVecN>;

  static_assert(kQkNope % kVecN == 0, "kQkNope must be a multiple of kVecN");
  static_assert(kQkRope % kVecN == 0, "kQkRope must be a multiple of kVecN");
  static_assert(kVHead % kVecN == 0, "kVHead must be a multiple of kVecN");

  constexpr int kKNopeVecs = kQkNope / kVecN;
  constexpr int kKPeVecs = kQkRope / kVecN;
  constexpr int kVVecs = kVHead / kVecN;

  constexpr int kKNopeTotalFast = kBlockS * kKNopeVecs;
  constexpr int kKPeTotalFast = kBlockS * kKPeVecs;
  constexpr int kVTotalFast = kBlockS * kVVecs;
  constexpr int kKNopeItersFast = (kKNopeTotalFast + kThreads - 1) / kThreads;
  constexpr int kKPeItersFast = (kKPeTotalFast + kThreads - 1) / kThreads;
  constexpr int kVItersFast = (kVTotalFast + kThreads - 1) / kThreads;

  const uint32_t s_block = blockIdx.x;
  const uint32_t head = blockIdx.y;
  const uint32_t tid = threadIdx.x;

  const uint32_t s_block_start = s_block * kBlockS;
  if (s_block_start >= params.s_total) return;
  const uint32_t valid_s =
      (params.s_total - s_block_start) < static_cast<uint32_t>(kBlockS) ? (params.s_total - s_block_start) : kBlockS;

  const TIn* __restrict__ const k_nope_base = static_cast<const TIn*>(params.k_nope);
  const TIn* __restrict__ const k_pe_base = static_cast<const TIn*>(params.k_pe);
  const TIn* __restrict__ const v_base = static_cast<const TIn*>(params.v);
  TOut* __restrict__ const k_out_base = static_cast<TOut*>(params.k_out);
  TOut* __restrict__ const v_out_base = static_cast<TOut*>(params.v_out);

  const float k_scale = params.k_scale_inv;
  const float v_scale = params.v_scale_inv;

  PDLWaitPrimary<kUsePDL>();

  if (valid_s == static_cast<uint32_t>(kBlockS)) {
    // ============ K_nope phase (fast path: all in-range) ============
    {
      vec_in_t bufs[kKNopeItersFast];
#pragma unroll
      for (int it = 0; it < kKNopeItersFast; ++it) {
        const uint32_t wi = tid + static_cast<uint32_t>(it) * kThreads;
        if constexpr (kKNopeItersFast * kThreads > kKNopeTotalFast) {
          if (wi >= static_cast<uint32_t>(kKNopeTotalFast)) continue;
        }
        const uint32_t sl = wi / static_cast<uint32_t>(kKNopeVecs);
        const uint32_t vi = wi - sl * static_cast<uint32_t>(kKNopeVecs);
        const uint32_t s_idx = s_block_start + sl;
        const TIn* k_nope_row = k_nope_base + s_idx * params.k_nope_stride_t + head * params.k_nope_stride_h;
        bufs[it].load(k_nope_row, vi);
      }
#pragma unroll
      for (int it = 0; it < kKNopeItersFast; ++it) {
        const uint32_t wi = tid + static_cast<uint32_t>(it) * kThreads;
        if constexpr (kKNopeItersFast * kThreads > kKNopeTotalFast) {
          if (wi >= static_cast<uint32_t>(kKNopeTotalFast)) continue;
        }
        const uint32_t sl = wi / static_cast<uint32_t>(kKNopeVecs);
        const uint32_t vi = wi - sl * static_cast<uint32_t>(kKNopeVecs);
        const uint32_t s_idx = s_block_start + sl;
        vec_out_t vout;
#pragma unroll
        for (int i = 0; i < kVecN; ++i) {
          vout[i] = static_cast<TOut>(static_cast<float>(bufs[it][i]) * k_scale);
        }
        TOut* k_out_row = k_out_base + s_idx * params.k_out_stride_t + head * params.k_out_stride_h;
        vout.store(k_out_row, vi);
      }
    }

    // ============ K_pe phase (fast path) ============
    {
      vec_in_t bufs[(kKPeItersFast > 0) ? kKPeItersFast : 1];
#pragma unroll
      for (int it = 0; it < kKPeItersFast; ++it) {
        const uint32_t wi = tid + static_cast<uint32_t>(it) * kThreads;
        if constexpr (kKPeItersFast * kThreads > kKPeTotalFast) {
          if (wi >= static_cast<uint32_t>(kKPeTotalFast)) continue;
        }
        const uint32_t sl = wi / static_cast<uint32_t>(kKPeVecs);
        const uint32_t vi = wi - sl * static_cast<uint32_t>(kKPeVecs);
        const uint32_t s_idx = s_block_start + sl;
        const TIn* k_pe_row = k_pe_base + s_idx * params.k_pe_stride_t;
        bufs[it].load(k_pe_row, vi);
      }
#pragma unroll
      for (int it = 0; it < kKPeItersFast; ++it) {
        const uint32_t wi = tid + static_cast<uint32_t>(it) * kThreads;
        if constexpr (kKPeItersFast * kThreads > kKPeTotalFast) {
          if (wi >= static_cast<uint32_t>(kKPeTotalFast)) continue;
        }
        const uint32_t sl = wi / static_cast<uint32_t>(kKPeVecs);
        const uint32_t vi = wi - sl * static_cast<uint32_t>(kKPeVecs);
        const uint32_t s_idx = s_block_start + sl;
        vec_out_t vout;
#pragma unroll
        for (int i = 0; i < kVecN; ++i) {
          vout[i] = static_cast<TOut>(static_cast<float>(bufs[it][i]) * k_scale);
        }
        TOut* k_out_row =
            k_out_base + s_idx * params.k_out_stride_t + head * params.k_out_stride_h + static_cast<int64_t>(kQkNope);
        vout.store(k_out_row, vi);
      }
    }

    // ============ V phase (fast path) ============
    {
      vec_in_t bufs[kVItersFast];
#pragma unroll
      for (int it = 0; it < kVItersFast; ++it) {
        const uint32_t wi = tid + static_cast<uint32_t>(it) * kThreads;
        if constexpr (kVItersFast * kThreads > kVTotalFast) {
          if (wi >= static_cast<uint32_t>(kVTotalFast)) continue;
        }
        const uint32_t sl = wi / static_cast<uint32_t>(kVVecs);
        const uint32_t vi = wi - sl * static_cast<uint32_t>(kVVecs);
        const uint32_t s_idx = s_block_start + sl;
        const TIn* v_row = v_base + s_idx * params.v_stride_t + head * params.v_stride_h;
        bufs[it].load(v_row, vi);
      }
#pragma unroll
      for (int it = 0; it < kVItersFast; ++it) {
        const uint32_t wi = tid + static_cast<uint32_t>(it) * kThreads;
        if constexpr (kVItersFast * kThreads > kVTotalFast) {
          if (wi >= static_cast<uint32_t>(kVTotalFast)) continue;
        }
        const uint32_t sl = wi / static_cast<uint32_t>(kVVecs);
        const uint32_t vi = wi - sl * static_cast<uint32_t>(kVVecs);
        const uint32_t s_idx = s_block_start + sl;
        vec_out_t vout;
#pragma unroll
        for (int i = 0; i < kVecN; ++i) {
          vout[i] = static_cast<TOut>(static_cast<float>(bufs[it][i]) * v_scale);
        }
        TOut* v_out_row = v_out_base + s_idx * params.v_out_stride_t + head * params.v_out_stride_h;
        vout.store(v_out_row, vi);
      }
    }

    PDLTriggerSecondary<kUsePDL>();
    return;
  }

  // ============ Slow path: tail CTA with valid_s < kBlockS. ============
  // K_nope phase. Layout: wi = sl * kKNopeVecs + vi for sl in [0, valid_s).
  {
    const uint32_t total = valid_s * static_cast<uint32_t>(kKNopeVecs);
    for (uint32_t wi = tid; wi < total; wi += kThreads) {
      const uint32_t sl = wi / static_cast<uint32_t>(kKNopeVecs);
      const uint32_t vi = wi - sl * static_cast<uint32_t>(kKNopeVecs);
      const uint32_t s_idx = s_block_start + sl;

      const TIn* k_nope_row = k_nope_base + s_idx * params.k_nope_stride_t + head * params.k_nope_stride_h;
      vec_in_t vin;
      vin.load(k_nope_row, vi);
      vec_out_t vout;
#pragma unroll
      for (int i = 0; i < kVecN; ++i) {
        vout[i] = static_cast<TOut>(static_cast<float>(vin[i]) * k_scale);
      }
      TOut* k_out_row = k_out_base + s_idx * params.k_out_stride_t + head * params.k_out_stride_h;
      vout.store(k_out_row, vi);
    }
  }

  // K_pe phase
  {
    const uint32_t total = valid_s * static_cast<uint32_t>(kKPeVecs);
    for (uint32_t wi = tid; wi < total; wi += kThreads) {
      const uint32_t sl = wi / static_cast<uint32_t>(kKPeVecs);
      const uint32_t vi = wi - sl * static_cast<uint32_t>(kKPeVecs);
      const uint32_t s_idx = s_block_start + sl;

      const TIn* k_pe_row = k_pe_base + s_idx * params.k_pe_stride_t;
      vec_in_t vin;
      vin.load(k_pe_row, vi);
      vec_out_t vout;
#pragma unroll
      for (int i = 0; i < kVecN; ++i) {
        vout[i] = static_cast<TOut>(static_cast<float>(vin[i]) * k_scale);
      }
      TOut* k_out_row =
          k_out_base + s_idx * params.k_out_stride_t + head * params.k_out_stride_h + static_cast<int64_t>(kQkNope);
      vout.store(k_out_row, vi);
    }
  }

  // V phase
  {
    const uint32_t total = valid_s * static_cast<uint32_t>(kVVecs);
    for (uint32_t wi = tid; wi < total; wi += kThreads) {
      const uint32_t sl = wi / static_cast<uint32_t>(kVVecs);
      const uint32_t vi = wi - sl * static_cast<uint32_t>(kVVecs);
      const uint32_t s_idx = s_block_start + sl;

      const TIn* v_row = v_base + s_idx * params.v_stride_t + head * params.v_stride_h;
      vec_in_t vin;
      vin.load(v_row, vi);
      vec_out_t vout;
#pragma unroll
      for (int i = 0; i < kVecN; ++i) {
        vout[i] = static_cast<TOut>(static_cast<float>(vin[i]) * v_scale);
      }
      TOut* v_out_row = v_out_base + s_idx * params.v_out_stride_t + head * params.v_out_stride_h;
      vout.store(v_out_row, vi);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

// ===========================================================================
// "Small / fat-warp" variant. One CTA = 1 warp (32 threads), grid = (s,h).
// Implicit kBlockS=1, no inner loop, straight-line code. Each phase activates
// the lanes needed to cover (head_dim / kVecN) vectors. Targeted at bs <= 16
// where launch + scheduling overhead dominate and the loop-based kernel
// can't amortize its per-CTA bookkeeping.
// ===========================================================================
template <typename TIn, typename TOut, int kQkNope, int kQkRope, int kVHead, int kVecN, bool kUsePDL>
__global__ void mla_kv_pack_quantize_fp8_small_kernel(
    __grid_constant__ const MlaKVPackQuantizeFp8Params params) {
  using namespace device;
  using vec_in_t = AlignedVector<TIn, kVecN>;
  using vec_out_t = AlignedVector<TOut, kVecN>;

  static_assert(kQkNope % kVecN == 0, "kQkNope must be a multiple of kVecN");
  static_assert(kQkRope % kVecN == 0, "kQkRope must be a multiple of kVecN");
  static_assert(kVHead % kVecN == 0, "kVHead must be a multiple of kVecN");

  constexpr int kKNopeVecs = kQkNope / kVecN;
  constexpr int kKPeVecs = kQkRope / kVecN;
  constexpr int kVVecs = kVHead / kVecN;
  static_assert(kKNopeVecs <= 32, "kQkNope/kVecN must fit in a warp");
  static_assert(kKPeVecs <= 32, "kQkRope/kVecN must fit in a warp");
  static_assert(kVVecs <= 32, "kVHead/kVecN must fit in a warp");

  // Cast strides to int; for bs <= 16 the row offsets always fit easily.
  const int s_idx = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x);

  const int k_nope_stride_t = static_cast<int>(params.k_nope_stride_t);
  const int k_nope_stride_h = static_cast<int>(params.k_nope_stride_h);
  const int k_pe_stride_t = static_cast<int>(params.k_pe_stride_t);
  const int v_stride_t = static_cast<int>(params.v_stride_t);
  const int v_stride_h = static_cast<int>(params.v_stride_h);
  const int k_out_stride_t = static_cast<int>(params.k_out_stride_t);
  const int k_out_stride_h = static_cast<int>(params.k_out_stride_h);
  const int v_out_stride_t = static_cast<int>(params.v_out_stride_t);
  const int v_out_stride_h = static_cast<int>(params.v_out_stride_h);

  const TIn* __restrict__ k_nope_row =
      static_cast<const TIn*>(params.k_nope) + s_idx * k_nope_stride_t + head * k_nope_stride_h;
  const TIn* __restrict__ k_pe_row = static_cast<const TIn*>(params.k_pe) + s_idx * k_pe_stride_t;
  const TIn* __restrict__ v_row = static_cast<const TIn*>(params.v) + s_idx * v_stride_t + head * v_stride_h;
  TOut* __restrict__ k_out_row = static_cast<TOut*>(params.k_out) + s_idx * k_out_stride_t + head * k_out_stride_h;
  TOut* __restrict__ v_out_row = static_cast<TOut*>(params.v_out) + s_idx * v_out_stride_t + head * v_out_stride_h;

  const float k_scale = params.k_scale_inv;
  const float v_scale = params.v_scale_inv;

  PDLWaitPrimary<kUsePDL>();

  // ---- Issue all loads first, then casts, then stores (latency hiding) ----
  vec_in_t k_nope_in, k_pe_in, v_in;
  if (lane < kKNopeVecs) {
    k_nope_in.load(k_nope_row, lane);
  }
  if (lane < kKPeVecs) {
    k_pe_in.load(k_pe_row, lane);
  }
  if (lane < kVVecs) {
    v_in.load(v_row, lane);
  }

  vec_out_t k_nope_out, k_pe_out, v_out;
  if (lane < kKNopeVecs) {
#pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      k_nope_out[i] = static_cast<TOut>(static_cast<float>(k_nope_in[i]) * k_scale);
    }
    k_nope_out.store(k_out_row, lane);
  }
  if (lane < kKPeVecs) {
#pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      k_pe_out[i] = static_cast<TOut>(static_cast<float>(k_pe_in[i]) * k_scale);
    }
    k_pe_out.store(k_out_row + static_cast<int>(kQkNope), lane);
  }
  if (lane < kVVecs) {
#pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      v_out[i] = static_cast<TOut>(static_cast<float>(v_in[i]) * v_scale);
    }
    v_out.store(v_out_row, lane);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <typename TIn, typename TOut, int kQkNope, int kQkRope, int kVHead, bool kUsePDL>
struct MlaKVPackQuantizeFp8Kernel {
  static_assert(kQkNope > 0 && kQkRope > 0 && kVHead > 0);

  template <int kVecN, int kBlockS, int kThreads>
  static constexpr auto kernel =
      mla_kv_pack_quantize_fp8_kernel<TIn, TOut, kQkNope, kQkRope, kVHead, kVecN, kBlockS, kThreads, kUsePDL>;

  template <int kVecN>
  static constexpr auto small_kernel =
      mla_kv_pack_quantize_fp8_small_kernel<TIn, TOut, kQkNope, kQkRope, kVHead, kVecN, kUsePDL>;

  static void
  run(tvm::ffi::TensorView k_out,
      tvm::ffi::TensorView v_out,
      tvm::ffi::TensorView k_nope,
      tvm::ffi::TensorView k_pe,
      tvm::ffi::TensorView v,
      double k_scale_inv,
      double v_scale_inv,
      int64_t vec_n,
      int64_t block_s,
      int64_t num_warps) {
    using namespace host;

    auto S = SymbolicSize{"s_total"};
    auto H = SymbolicSize{"num_heads"};
    auto QK_NOPE = SymbolicSize{"qk_nope"};
    auto QK_ROPE = SymbolicSize{"qk_rope"};
    auto V_HEAD = SymbolicSize{"v_head"};
    auto QK_TOTAL = SymbolicSize{"qk_total"};
    auto S0_k_nope = SymbolicSize{"k_nope_stride_t"};
    auto S1_k_nope = SymbolicSize{"k_nope_stride_h"};
    auto S0_k_pe = SymbolicSize{"k_pe_stride_t"};
    auto S0_v = SymbolicSize{"v_stride_t"};
    auto S1_v = SymbolicSize{"v_stride_h"};
    auto S0_k_out = SymbolicSize{"k_out_stride_t"};
    auto S1_k_out = SymbolicSize{"k_out_stride_h"};
    auto S0_v_out = SymbolicSize{"v_out_stride_t"};
    auto S1_v_out = SymbolicSize{"v_out_stride_h"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    QK_NOPE.set_value(kQkNope);
    QK_ROPE.set_value(kQkRope);
    V_HEAD.set_value(kVHead);
    QK_TOTAL.set_value(kQkNope + kQkRope);

    TensorMatcher({S, H, QK_NOPE})
        .with_strides({S0_k_nope, S1_k_nope, 1})
        .with_dtype<TIn>()
        .with_device(device)
        .verify(k_nope);
    TensorMatcher({S, QK_ROPE})
        .with_strides({S0_k_pe, 1})
        .with_dtype<TIn>()
        .with_device(device)
        .verify(k_pe);
    TensorMatcher({S, H, V_HEAD})
        .with_strides({S0_v, S1_v, 1})
        .with_dtype<TIn>()
        .with_device(device)
        .verify(v);
    TensorMatcher({S, H, QK_TOTAL})
        .with_strides({S0_k_out, S1_k_out, 1})
        .with_dtype<TOut>()
        .with_device(device)
        .verify(k_out);
    TensorMatcher({S, H, V_HEAD})
        .with_strides({S0_v_out, S1_v_out, 1})
        .with_dtype<TOut>()
        .with_device(device)
        .verify(v_out);

    const uint32_t s_total = static_cast<uint32_t>(S.unwrap());
    const uint32_t num_heads = static_cast<uint32_t>(H.unwrap());
    if (s_total == 0 || num_heads == 0) return;

    const auto params = MlaKVPackQuantizeFp8Params{
        .k_out = k_out.data_ptr(),
        .v_out = v_out.data_ptr(),
        .k_nope = k_nope.data_ptr(),
        .k_pe = k_pe.data_ptr(),
        .v = v.data_ptr(),
        .k_scale_inv = static_cast<float>(k_scale_inv),
        .v_scale_inv = static_cast<float>(v_scale_inv),
        .s_total = s_total,
        .num_heads = num_heads,
        .k_nope_stride_t = S0_k_nope.unwrap(),
        .k_nope_stride_h = S1_k_nope.unwrap(),
        .k_pe_stride_t = S0_k_pe.unwrap(),
        .v_stride_t = S0_v.unwrap(),
        .v_stride_h = S1_v.unwrap(),
        .k_out_stride_t = S0_k_out.unwrap(),
        .k_out_stride_h = S1_k_out.unwrap(),
        .v_out_stride_t = S0_v_out.unwrap(),
        .v_out_stride_h = S1_v_out.unwrap(),
    };

    auto launch = [&]<int kVecN, int kBlockS, int kThreads>() {
      static_assert(kThreads % device::kWarpThreads == 0);
      const uint32_t grid_x = div_ceil(s_total, static_cast<uint32_t>(kBlockS));
      LaunchKernel(dim3(grid_x, num_heads), kThreads, device.unwrap())
          .enable_pdl(kUsePDL)(kernel<kVecN, kBlockS, kThreads>, params);
    };

    auto launch_small = [&]<int kVecN>() {
      // One CTA = 1 warp per (token, head). No inner loop.
      LaunchKernel(dim3(s_total, num_heads), device::kWarpThreads, device.unwrap())
          .enable_pdl(kUsePDL)(small_kernel<kVecN>, params);
    };

    auto bad = [&]() {
      Panic("Unsupported (vec_n=", vec_n, ", block_s=", block_s, ", num_warps=", num_warps, ")");
    };

    // block_s == 0 is the sentinel for the small/fat-warp variant. One CTA =
    // one warp per (token, head). Vec width must fit one full row in a warp.
    if (block_s == 0) {
      constexpr bool kCanSmallVec4 = (kQkNope % 4 == 0) && (kQkRope % 4 == 0) && (kVHead % 4 == 0) &&
                                     (kQkNope / 4 <= 32) && (kQkRope / 4 <= 32) && (kVHead / 4 <= 32);
      constexpr bool kCanSmallVec8 = (sizeof(TIn) * 8 <= 16) && (kQkNope % 8 == 0) && (kQkRope % 8 == 0) &&
                                     (kVHead % 8 == 0) && (kQkNope / 8 <= 32) && (kQkRope / 8 <= 32) &&
                                     (kVHead / 8 <= 32);
      if (vec_n == 4) {
        if constexpr (kCanSmallVec4) {
          launch_small.template operator()<4>();
          return;
        }
      } else if (vec_n == 8) {
        if constexpr (kCanSmallVec8) {
          launch_small.template operator()<8>();
          return;
        }
      }
      bad();
      return;
    }

    // (vec_n, block_s, num_warps) dispatch. Templates are enumerated to keep
    // the kernel monomorphic and fully unrolled.
    auto dispatch_warps = [&]<int VN, int BS>() {
      switch (num_warps) {
        case 1: launch.template operator()<VN, BS, 32>(); break;
        case 2: launch.template operator()<VN, BS, 64>(); break;
        case 4: launch.template operator()<VN, BS, 128>(); break;
        case 8: launch.template operator()<VN, BS, 256>(); break;
        case 16: launch.template operator()<VN, BS, 512>(); break;
        default: bad();
      }
    };

    auto dispatch_block_s = [&]<int VN>() {
      switch (block_s) {
        case 1: dispatch_warps.template operator()<VN, 1>(); break;
        case 2: dispatch_warps.template operator()<VN, 2>(); break;
        case 4: dispatch_warps.template operator()<VN, 4>(); break;
        case 8: dispatch_warps.template operator()<VN, 8>(); break;
        case 16: dispatch_warps.template operator()<VN, 16>(); break;
        case 32: dispatch_warps.template operator()<VN, 32>(); break;
        case 64: dispatch_warps.template operator()<VN, 64>(); break;
        default: bad();
      }
    };

    constexpr int kVecN8 = 8;
    constexpr int kVecN16 = 16;
    static_assert(sizeof(TIn) * kVecN8 <= 16, "8-elem vec must fit in 128 bits for TIn");
    if (vec_n == kVecN8) {
      dispatch_block_s.template operator()<kVecN8>();
    } else if (vec_n == kVecN16 && sizeof(TIn) * kVecN16 <= device::kMaxVecBytes) {
      if constexpr (sizeof(TIn) * kVecN16 <= device::kMaxVecBytes) {
        dispatch_block_s.template operator()<kVecN16>();
      } else {
        bad();
      }
    } else {
      bad();
    }
  }
};

}  // namespace
