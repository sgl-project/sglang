// Low-latency MXFP8 wq_b GEMM + q RoPE for DeepSeek-V4.1-Flash TP4 decode (SM100).
//
//   q[T, 16, 512] = bf16(x[T, 1280] @ W[8192, 1280]^T), then RoPE on dims [448, 512) of every head, T <= 128
//   (dsv41_indexer_wq_b_ll: the same GEMM for the indexer's wq_b, W [4096, 1280], no RoPE;
//   dsv41_indexer_q_fp4_ll: that GEMM with index_q_rope_pack_weights fused into the epilogue)
//   x: e4m3 + UE8M0 per 32 in FlashInfer's 128x4 swizzled layout (rmsnorm_mxfp8's output); W: e4m3 with UE8M0 per
//   row per 32 in the same layout (the layer's weight and weight_scale_inv_swizzled, mm_mxfp8's operands).
//   RoPE is fused_rope_inplace's: interleaved (re, im) pairs of the bf16 GEMM output rotated in fp32 by
//   freqs_cis[pos_t], rounded to bf16 again. It differs from mm_mxfp8 + fused_rope_inplace only by the fp32 order of
//   the two K halves.
//
// Decomposition (swap-AB, one wave): the weight is MMA operand A (M_BLOCK = 128 output rows), the tokens are N
// (8 / 16 / 32 / 64 / 128, one pass). Split-K 2: the two K halves of a row tile form a 2-CTA cluster; cluster rank r
// finishes tokens [r N / 2, (r + 1) N / 2) of all 128 rows, receiving the peer's partials of them over DSMEM.
// Warp-specialized mainloop over 128-K subtiles, weight and activation in separate smem rings:
//   warp 0: the weight (TMA, SW128, L2 evict_first) and its scales, issued first; it never waits on PDL, so the slab
//           streams during the predecessor;
//   warp 1: griddepcontrol.wait, then the activation (TMA, rows past T zero-filled) and its scales;
//   warp 2: one lane copies the scale chunks into TMEM and issues 4 MMAs per subtile;
//   warp 3: TMEM alloc and dealloc;
//   warps 4-7: in RoPE tiles, the (cos, sin) rows of the T positions by cp.async while the mainloop runs (inputs
//              the predecessor does not write); then the epilogue: each warp pushes its TMEM quadrant's partials of
//              the peer's tokens (16 B st.async), adds the peer's partials of its own tokens, stages the bf16 rows
//              as [token][128] in the freed weight ring, and after one epilogue-group barrier each warp finishes
//              whole tokens: 4 adjacent features (2 RoPE pairs) per lane, RoPE, 8 B stores.
//   (indexer q: dims [64, 128) of every 128-dim head rotated, then each warp's 32 rows, one fp4 block, are
//   transposed through smem so lanes own tokens for both fp4 stages and the packed store.)
// PDL dependents are released at kernel start.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tcgen05.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp4_utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>

namespace sglang::device::wq_b {

namespace ptx {

using device::ptx::mbar_arrive;
using device::ptx::mbar_arrive_expect_tx;
using device::ptx::mbar_init;
using device::ptx::to_shared;

// The inbox wait orders the smem reads of what the peer st.async'd in, hence a local copy with a "memory" clobber.
SGL_DEVICE void mbar_wait_parity(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}\n" ::"r"(to_shared(bar)),
      "r"(parity)
      : "memory");
}
SGL_DEVICE uint64_t policy_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
// One 2D tensor-map box at coords (x, y) into smem, completing its bytes on `bar`.
SGL_DEVICE void tma_load_2d(void* dst, const CUtensorMap* map, int32_t x, int32_t y, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::
          "r"(to_shared(dst)),
      "l"(map),
      "r"(x),
      "r"(y),
      "r"(to_shared(bar))
      : "memory");
}
// The same with an L2 cache policy.
SGL_DEVICE void tma_load_2d_hint(void* dst, const CUtensorMap* map, int32_t x, int32_t y, uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(to_shared(dst)),
      "l"(map),
      "r"(x),
      "r"(y),
      "r"(to_shared(bar)),
      "l"(pol)
      : "memory");
}
// `bytes` (a multiple of 16) of contiguous global memory into smem, completing them on `bar`.
SGL_DEVICE void cp_async_bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(to_shared(dst)),
      "l"(src),
      "r"(bytes),
      "r"(to_shared(bar))
      : "memory");
}
SGL_DEVICE void cp_async_cg_16B(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(to_shared(dst)), "l"(src) : "memory");
}
SGL_DEVICE void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;" ::: "memory");
}
// 16 B of registers (4 fp32) to `dst` in a peer CTA, completing its bytes on the peer mbarrier `bar`.
SGL_DEVICE void st_async_16B(uint32_t dst, const void* src, uint32_t bar) {
  const auto v = static_cast<const uint32_t*>(src);
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(dst),
               "r"(v[0]),
               "r"(v[1]),
               "r"(v[2]),
               "r"(v[3]),
               "r"(bar)
               : "memory");
}
SGL_DEVICE uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t out;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(out) : "r"(addr), "r"(rank));
  return out;
}
SGL_DEVICE uint32_t cluster_ctarank() {
  uint32_t r;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
SGL_DEVICE void fence_mbarrier_init_release_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
SGL_DEVICE void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
SGL_DEVICE void cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
SGL_DEVICE void prefetch_tensormap(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory");
}
SGL_DEVICE void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
SGL_DEVICE void barrier_sync(uint32_t id, uint32_t threads) {
  asm volatile("barrier.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}
SGL_DEVICE void barrier_arrive(uint32_t id, uint32_t threads) {
  asm volatile("barrier.arrive %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

}  // namespace ptx

using tcgen05 = device::ptx::TCGen05<1>;

/// What the epilogue does with the bf16 tile: store it, store it with the q RoPE of dims [448, 512) of every
/// 512-dim head, or rotate dims [64, 128) of every 128-dim indexer head and store both fp4 stages of
/// index_q_rope_pack_weights (packed e2m1 payload, ue8m0 exponents) instead of the tile.
enum class Epilogue : int { kPlain = 0, kRope = 1, kIndexQ = 2 };

// From fp4_indexer_rope.cuh (index_rope_quant_pack's packer stage), kept bitwise.

/// The packer's ue8m0 exponent: its own floor (after the divide) and clamp, unlike the fake-quant stage's.
SGL_DEVICE uint32_t index_pack_exponent(float amax) {
  const auto exponent = deepseek_v4::fp8::cast_to_ue8m0(fmaxf(amax / 6.0f, 1.0e-4f));
  return static_cast<uint32_t>(min(max(exponent, 1), 254));
}

/// Clears the sign of every packed nibble whose magnitude rounded to zero (the reference packer stores no -0).
SGL_DEVICE uint32_t clear_negative_zero(uint32_t packed) {
  const auto any_magnitude = (packed | (packed >> 1) | (packed >> 2)) & 0x11111111u;
  return packed & ((any_magnitude << 3) | 0x77777777u);
}

/**
 * \brief Compile-time shape of the wq_b GEMM + RoPE.
 * \tparam N_       tokens, the MMA N (8 / 16 / 32 / 64 / 128); T <= N_.
 * \tparam kOutDim  weight rows (8192: wq_b, 4096: the indexer's wq_b).
 * \tparam kStages_ weight ring depth in 128-K subtiles (clamped to the CTA's subtile count); the activation ring
 *                  takes the smem left over, so a fully resident weight never waits on the activation.
 * \tparam kInflight_ weight subtiles in flight per CTA (clamped to kStages_); the next is issued when the oldest lands.
 * \tparam kRope    RoPE on dims [448, 512) of every 512-dim head (else a plain bf16 GEMM).
 */
template <int N_, int kOutDim, int kStages_, int kInflight_, bool kRope>
struct Trait {
  static constexpr int N = N_;
  static_assert(N == 8 || N == 16 || N == 32 || N == 64 || N == 128);
  static constexpr int kSplitK = 2;
  static constexpr int M = kOutDim;
  static_assert(M % 128 == 0);
  static constexpr int K = 1280;  // q_lora_rank
  static constexpr int M_BLOCK = 128;
  static constexpr int kHeadDim = 512;
  static constexpr int kRopeDim = 64;
  static constexpr int K_BLOCK = K / kSplitK;
  static constexpr int kSub = K_BLOCK / 128;  // 128-byte K subtiles == 4-block scale groups
  static constexpr int kStages = std::min(kStages_, kSub);
  static constexpr int kMaxInflight = std::min(kInflight_, kStages);
  static constexpr int kNumCtas = M / M_BLOCK * kSplitK;
  static constexpr int kThreads = 256;
  static constexpr int kOwnTokens = N / kSplitK;
  static_assert(kOwnTokens % 4 == 0);
  static constexpr int kWSubBytes = M_BLOCK * 128;
  static constexpr int kXSubBytes = N * 128;
  // A CTA's slab of a 128x4 scale image: kSub contiguous 512 B tiles, tcgen05.cp.32x128b.warpx4's order.
  static constexpr int kSFBytes = kSub * 512;
  static constexpr int kInboxFloats = 4 * 32 * kOwnTokens;
  // TMEM: accumulator, then A scales, then B scales.
  static constexpr uint32_t kTmemAcc = 0;
  static constexpr uint32_t kTmemSFA = N;
  static constexpr uint32_t kTmemSFB = kTmemSFA + 4 * kSub;
  static constexpr uint32_t kTmemUsed = kTmemSFB + 4 * kSub;
  static constexpr uint32_t kTmemCols = get_tmem_cols(kTmemUsed);
  static_assert(kTmemUsed <= kTmemCols);
  static_assert(kOwnTokens * M_BLOCK * 2 <= kStages * kWSubBytes);
  // Everything but the activation ring, plus barriers and alignment slack; the ring gets the rest.
  static constexpr int kSmemCapacity = 227 * 1024;  // SM100 max dynamic smem per CTA
  static constexpr int kRopeFloats = kRope ? N * kRopeDim : 4;
  static constexpr int kFixedBytes = kStages * kWSubBytes + 2 * kSFBytes + kInboxFloats * 4 + kRopeFloats * 4 + 3072;
  static constexpr int kXFit = (kSmemCapacity - kFixedBytes) / kXSubBytes;
  static constexpr int kXStages = std::min(kXFit, kSub);
  static_assert(kXStages >= 1, "no room for the activation");

  struct alignas(1024) Smem {
    alignas(1024) uint8_t w[kStages * kWSubBytes];   // SW128 K-major weight subtiles
    alignas(1024) uint8_t x[kXStages * kXSubBytes];  // SW128 K-major activation subtiles
    alignas(128) uint8_t sfa[kSFBytes];
    alignas(128) uint8_t sfb[kSFBytes];
    alignas(16) float inbox[kInboxFloats];
    alignas(16) float rope[kRopeFloats];  // (cos, sin) interleaved per token
    uint64_t w_full[kStages];
    uint64_t w_empty[kStages];
    uint64_t x_full[kXStages];
    uint64_t x_empty[kXStages];
    uint64_t mma_done;
    uint64_t red_ready;
    uint64_t drained;
    uint64_t rope_ready;
    uint32_t taddr;
  };
  static_assert(sizeof(Smem) <= kSmemCapacity - 1024, "exceeds the SM's dynamic smem");

  /// Byte offset of this CTA's scale slab in a 128x4 image of the weight rows.
  SGL_DEVICE static uint32_t sf_offset(uint32_t tile, uint32_t rank) {
    return (tile * (K / 128) + rank * kSub) * 512;
  }
};

struct Params {
  CUtensorMap x_map;       // x [T, 1280] e4m3 as 2D (k, t), box (128, N), 128 B swizzle
  CUtensorMap w_map;       // W [M, 1280] e4m3 as 2D (k, m), box (128, 128), 128 B swizzle
  const uint8_t* x_scale;  // its UE8M0 scales, FlashInfer 128x4 image (128 padded rows)
  const uint8_t* w_scale;  // the weight's UE8M0 per-32 scales, FlashInfer 128x4 image
  const float* freqs;      // [max_pos, 64] fp32, interleaved (cos, sin)
  const void* positions;   // [T] int32 or int64
  bf16_t* out;             // [T, M] bf16, rows out_stride elements apart
  uint32_t out_stride;
  uint32_t num_tokens;
  uint32_t pos_is_i64;
  // Epilogue::kIndexQ: [T * heads, 64] packed e2m1, [T * heads] four ue8m0 bytes, and the head weights
  // bf16(head_weights * weight_scale) as fp32 [T * heads].
  int8_t* payload;
  uint8_t* scale;
  const bf16_t* head_weights;
  float* weights_out;
  float weight_scale;
};

template <int N, int kOutDim, int kStages, int kInflight, Epilogue kEpi>
__global__ void __launch_bounds__(256, 1) wq_b_rope_kernel(const __grid_constant__ Params p) {
  constexpr bool kRope = kEpi != Epilogue::kPlain;
  using T = Trait<N, kOutDim, kStages, kInflight, kRope>;
  constexpr int kSplitK = T::kSplitK;
  constexpr int kSub = T::kSub;
  constexpr int kWStages = T::kStages;
  constexpr int kXStages = T::kXStages;
  extern __shared__ __align__(1024) uint8_t s_buf[];
  auto& smem = *reinterpret_cast<typename T::Smem*>(s_buf);

  enum Role { WARP_W = 0, WARP_X = 1, WARP_MMA = 2, WARP_TMEM = 3, WARP_EPI = 4 };
  enum NamedBarrier { BAR_INIT = 1, BAR_EPI = 2 };
  constexpr uint32_t kOwnTokens = T::kOwnTokens;
  // RoPE rows by the epilogue warps: warp w owns tokens 4 j + w (j < N / 4 <= 32) and lane j holds that token's
  // position. Each step copies 2 rows (512 B), a half-warp per 256 B row at 16 B per lane.
  constexpr uint32_t kRopeSteps = (N + 7) / 8;

  PDLTriggerSecondary<true>();

  const auto warp_id = warp::broadcast(threadIdx.x / 32, 0);
  const auto lane_id = warp::get_lane_id();
  const uint32_t rank = ptx::cluster_ctarank();
  const auto tile = blockIdx.x / static_cast<uint32_t>(kSplitK);
  const auto row0 = tile * T::M_BLOCK;
  const auto num_tokens = p.num_tokens;
  // The RoPE rows are quadrants 2 and 3 of every 4th tile (q dims 448..511), or of every tile (indexer dims 64..127).
  const bool rope_tile = kRope && (kEpi == Epilogue::kIndexQ || (row0 % T::kHeadDim) == T::kHeadDim - T::M_BLOCK);
  const auto own_t0 = rank * kOwnTokens;
  const auto rope_token = [&](uint32_t j) { return 4 * j + warp_id - WARP_EPI; };
  int32_t rope_pos = 0;

  // Weight subtile s into ring stage s % kWStages, completing w_full; stage 0 also takes the scales (kSub contiguous
  // 512 B tiles of the image: row tile, then this CTA's k4 groups).
  const auto load_w = [&](uint32_t s, uint64_t policy) {
    const auto stage = s % kWStages;
    ptx::mbar_arrive_expect_tx(&smem.w_full[stage], T::kWSubBytes + (s == 0 ? T::kSFBytes : 0));
    if (s == 0) ptx::cp_async_bulk_g2s(smem.sfa, p.w_scale + T::sf_offset(tile, rank), T::kSFBytes, &smem.w_full[0]);
    const auto k = static_cast<int32_t>(rank * T::K_BLOCK + 128 * s);
    ptx::tma_load_2d_hint(&smem.w[stage * T::kWSubBytes], &p.w_map, k, row0, &smem.w_full[stage], policy);
  };

  // ---------------------------- init ----------------------------
  if (warp_id == WARP_W) {
    // The weight depends on nothing: its first subtiles go out before the other barriers are initialized.
    if (warp::elect_one_lane()) {
      const auto policy = ptx::policy_evict_first();
#pragma unroll
      for (uint32_t s = 0; s < kWStages; ++s) {
        ptx::mbar_init(&smem.w_full[s], 1);
      }
      // CTA-local barriers completed by this CTA's own TMA: the proxy fence suffices before the first loads.
      ptx::fence_proxy_async_shared();
#pragma unroll
      for (uint32_t s = 0; s < T::kMaxInflight; ++s) {
        load_w(s, policy);
      }
#pragma unroll
      for (uint32_t s = 0; s < kWStages; ++s) {
        ptx::mbar_init(&smem.w_empty[s], 1);
      }
#pragma unroll
      for (uint32_t s = 0; s < kXStages; ++s) {
        ptx::mbar_init(&smem.x_full[s], 1);
        ptx::mbar_init(&smem.x_empty[s], 1);
      }
      ptx::mbar_init(&smem.mma_done, 1);
      ptx::mbar_init(&smem.red_ready, 1);
      ptx::mbar_init(&smem.drained, 4);
      ptx::mbar_init(&smem.rope_ready, T::kThreads - 32 * WARP_EPI);  // every epilogue thread
      ptx::fence_mbarrier_init_release_cluster();
      ptx::mbar_arrive_expect_tx(&smem.red_ready, T::kInboxFloats * 4);
    }
    __syncwarp();
    ptx::barrier_arrive(BAR_INIT, T::kThreads);
  } else {
    if (warp_id == WARP_TMEM) {
      tcgen05::alloc(&smem.taddr, T::kTmemCols);
      tcgen05::relinquish_alloc_permit();
      tcgen05::fence_before_thread_sync();
    } else if (warp_id >= WARP_EPI) {
      // Positions are non-negative and below 2^31, so an int64 position is read as its low int32 word. Only loaded
      // here: issuing the copies would stall this warp on the round trip, and the init barrier with it.
      const auto t = rope_token(lane_id);
      if (rope_tile && t < num_tokens) {
        rope_pos = static_cast<const int32_t*>(p.positions)[t * (p.pos_is_i64 ? 2u : 1u)];
      }
    }
    ptx::barrier_sync(BAR_INIT, T::kThreads);
  }
  ptx::cluster_arrive_relaxed();

  // ---------------------------- mainloop ----------------------------
  if (warp_id == WARP_W) {
    // ---- warp 0: the rest of the weight, each subtile once its ring stage is free ----
    if (warp::elect_one_lane()) {
      const auto policy = ptx::policy_evict_first();
#pragma unroll
      for (uint32_t s = T::kMaxInflight; s < kWStages; ++s) {
        ptx::mbar_wait_parity(&smem.w_full[s - T::kMaxInflight], 0);
        load_w(s, policy);
      }
#pragma unroll
      for (uint32_t s = kWStages; s < kSub; ++s) {
        ptx::mbar_wait_parity(&smem.w_empty[s % kWStages], (s / kWStages - 1) & 1);
        load_w(s, policy);
      }
    }
  } else if (warp_id == WARP_X) {
    // ---- warp 1: the activation, a TMA box per subtile and its scales by one bulk copy, after the PDL wait ----
    if (warp::elect_one_lane()) {
      PDLWaitPrimary<true>();
#pragma unroll
      for (uint32_t s = 0; s < kSub; ++s) {
        const auto stage = s % kXStages;
        if (s >= kXStages) ptx::mbar_wait_parity(&smem.x_empty[stage], (s / kXStages - 1) & 1);
        ptx::mbar_arrive_expect_tx(&smem.x_full[stage], T::kXSubBytes + (s == 0 ? T::kSFBytes : 0));
        if (s == 0) ptx::cp_async_bulk_g2s(smem.sfb, p.x_scale + rank * T::kSFBytes, T::kSFBytes, &smem.x_full[0]);
        ptx::tma_load_2d(&smem.x[stage * T::kXSubBytes], &p.x_map, rank * T::K_BLOCK + 128 * s, 0, &smem.x_full[stage]);
      }
    }
  } else if (warp_id == WARP_MMA) {
    // ---- warp 2: one lane copies the scales into TMEM and issues 4 MMAs per subtile ----
    if (warp::elect_one_lane()) {
      tcgen05::fence_after_thread_sync();
      const auto taddr = smem.taddr;
      const auto sfa = tcgen05::make_desc_sf_chunk(smem.sfa);
      const auto sfb = tcgen05::make_desc_sf_chunk(smem.sfb);
      const auto a = tcgen05::make_desc_kmajor_sw128(smem.w);
      const auto b = tcgen05::make_desc_kmajor_sw128(smem.x);
      const auto idesc = tcgen05::make_instr_desc_mxf8<T::M_BLOCK, N>();
#pragma unroll
      for (uint32_t s = 0; s < kSub; ++s) {
        const auto wstage = s % kWStages, xstage = s % kXStages;
        ptx::mbar_wait_parity(&smem.w_full[wstage], (s / kWStages) & 1);
        ptx::mbar_wait_parity(&smem.x_full[xstage], (s / kXStages) & 1);
        tcgen05::fence_after_thread_sync();
        if (s == 0) {
          // Both scale slabs landed with stage 0; tcgen05.cp and tcgen05.mma of one thread execute in order.
#pragma unroll
          for (uint32_t j = 0; j < kSub; ++j) {
            tcgen05::cp_32x128b_warpx4(taddr + T::kTmemSFA + 4 * j, sfa.advance_bytes(j * 512).bits);
            tcgen05::cp_32x128b_warpx4(taddr + T::kTmemSFB + 4 * j, sfb.advance_bytes(j * 512).bits);
          }
        }
        // K block (s, kb): 32 B into the subtile; the scales are byte kb of the subtile's TMEM scale columns.
#pragma unroll
        for (uint32_t kb = 0; kb < 4; ++kb) {
          tcgen05::mma_mxf8_ss(
              taddr + T::kTmemAcc,
              a.advance_bytes(wstage * T::kWSubBytes + 32 * kb).bits,
              b.advance_bytes(xstage * T::kXSubBytes + 32 * kb).bits,
              idesc.with_sf_id(kb).bits,
              taddr + T::kTmemSFA + 4 * s,
              taddr + T::kTmemSFB + 4 * s,
              s | kb);
        }
        if constexpr (kWStages < kSub) tcgen05::commit(&smem.w_empty[wstage]);
        if constexpr (kXStages < kSub) tcgen05::commit(&smem.x_empty[xstage]);
      }
      tcgen05::commit(&smem.mma_done);
    }
  } else if (warp_id >= WARP_EPI) {
    // ---- warps 4-7: the (cos, sin) rows of the T positions while the mainloop runs ----
    if (rope_tile) {
      const auto chunk = lane_id % 16 * 4;  // floats into the 64-float row
#pragma unroll
      for (uint32_t i = 0; i < kRopeSteps; ++i) {
        const auto j = 2 * i + lane_id / 16;
        const auto pos = __shfl_sync(0xffffffffu, rope_pos, j);
        const auto t = rope_token(j);
        if (t < num_tokens) {
          ptx::cp_async_cg_16B(&smem.rope[t * T::kRopeDim + chunk], p.freqs + pos * T::kRopeDim + chunk);
        }
      }
      ptx::cp_async_wait_all();
      ptx::mbar_arrive(&smem.rope_ready);
    }
  }

  // ---------------------------- epilogue ----------------------------
  __syncwarp();
  const auto quad = warp_id % 4;
  const auto row_local = quad * 32 + lane_id;
  const bool is_re = (lane_id & 1) == 0;
  const auto taddr = smem.taddr;
  const auto acc_addr = taddr + ((quad * 32) << 16) + T::kTmemAcc;
  ptx::cluster_wait_acquire();
  ptx::mbar_wait_parity(&smem.mma_done, 0);
  tcgen05::fence_after_thread_sync();

  if (warp_id < WARP_EPI) {
    if (warp_id == WARP_TMEM) {
      ptx::mbar_wait_parity(&smem.drained, 0);
      tcgen05::fence_after_thread_sync();
      tcgen05::dealloc(taddr, T::kTmemCols);
    }
    return;
  }

  // bf16 results packed by token pairs: packed cvt.rn.bf16x2 (a scalar F2F per value is quarter-rate and made the
  // T = 64 epilogue F2F-bound). RoPE rows are rotated in straight-line chunks, branch-free, so the (cos, sin) loads
  // and partner shuffles of a chunk issue back to back. Tokens past T are don't-care.
  const auto round_and_rope = [&](const auto& acc, auto& res, uint32_t c0) {
    constexpr uint32_t kCnt = sizeof(acc) / sizeof(acc[0]);
    const bool do_rope = rope_tile && quad >= 2;  // warp-uniform
    if (!do_rope) {
#pragma unroll
      for (uint32_t k = 0; k < kCnt / 2; ++k) {
        res[k] = cast<bf16x2_t>(fp32x2_t{acc[2 * k], acc[2 * k + 1]});
      }
      return;
    }
    ptx::mbar_wait_parity(&smem.rope_ready, 0);
    const auto pair = (row_local - 64) / 2;
    constexpr uint32_t kChunk = kCnt <= 32 ? kCnt : 16;
#pragma unroll
    for (uint32_t c = 0; c < kCnt; c += kChunk) {
      float self[kChunk], other[kChunk];
      AlignedVector<fp32_t, 2> cs[kChunk];
#pragma unroll
      for (uint32_t j = 0; j < kChunk; j += 2) {
        const auto [a, b] = cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{acc[c + j], acc[c + j + 1]}));
        self[j + 0] = a;
        self[j + 1] = b;
      }
#pragma unroll
      for (uint32_t j = 0; j < kChunk; ++j) {
        cs[j].load(&smem.rope[(c0 + c + j) * T::kRopeDim], pair);
        other[j] = __shfl_xor_sync(0xffffffffu, self[j], 1);
      }
#pragma unroll
      for (uint32_t j = 0; j < kChunk; ++j) {
        const float fr = cs[j][0], fi = cs[j][1];
        const float xr = is_re ? self[j] : other[j], xi = is_re ? other[j] : self[j];
        self[j] = is_re ? xr * fr - xi * fi : xr * fi + xi * fr;
      }
#pragma unroll
      for (uint32_t j = 0; j < kChunk; j += 2) {
        res[(c + j) / 2] = cast<bf16x2_t>(fp32x2_t{self[j], self[j + 1]});
      }
    }
  };

  // Epilogue::kIndexQ: the warp's 32 rows are one fp4 block of head `tile`, transposed through smem so kLanes adjacent
  // lanes own a token (32 / kLanes elements each); the stage reductions stay in registers (plus log2(kLanes)
  // shuffles) and the payload goes out in 4 / 8 / 16 B stores. A payload byte is an element pair, low nibble even.
  // Tokens past T are computed and dropped.
  const auto store_index_q = [&](const auto& res, uint32_t c0) {
    constexpr uint32_t kCnt = 2 * sizeof(res) / sizeof(res[0]);
    constexpr uint32_t kHeads = T::M / T::M_BLOCK;
    constexpr uint32_t kRowPad = 40;  // bf16 per token: 80 B rows, conflict-free 16 B reads
    // Measured at T <= 16: 8 lanes per token (or 4 at N = 16) is no faster; at least 8 elements per lane.
    constexpr uint32_t kLanes = kCnt >= 32 ? 1 : std::min(32 / kCnt, 4u);
    constexpr uint32_t kElems = 32 / kLanes;
    constexpr uint32_t kBytes = kElems / 2;
    constexpr uint32_t kBlockRows = std::max(kCnt, 32 / kLanes);
    static_assert(4 * kBlockRows * kRowPad * 2 <= T::kStages * T::kWSubBytes);
    // Rows are the warp's tokens [c0, c0 + kCnt) from 0; rows past kCnt (kCnt = 4) are read and dropped.
    const auto block = reinterpret_cast<bf16_t*>(smem.w) + quad * kBlockRows * kRowPad;
#pragma unroll
    for (uint32_t k = 0; k < kCnt / 2; ++k) {
      block[(2 * k) * kRowPad + lane_id] = res[k].x;
      block[(2 * k + 1) * kRowPad + lane_id] = res[k].y;
    }
    __syncwarp();
    const auto part = lane_id % kLanes;
    const auto group_max = [](float v) {
#pragma unroll
      for (uint32_t o = 1; o < kLanes; o <<= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
      }
      return v;
    };
#pragma unroll
    for (uint32_t i = 0; i < kCnt; i += 32 / kLanes) {
      if (c0 + i >= num_tokens) break;  // warp-uniform
      const auto tl = i + lane_id / kLanes;
      const auto t = c0 + tl;
      float d[kElems];
#pragma unroll
      for (uint32_t j = 0; j < kElems / 8; ++j) {
        AlignedVector<bf16x2_t, 4> v;
        v.load(block + tl * kRowPad + part * kElems + j * 8);
#pragma unroll
        for (uint32_t e = 0; e < 4; ++e) {
          const auto [lo, hi] = cast<fp32x2_t>(v[e]);
          d[j * 8 + 2 * e] = lo;
          d[j * 8 + 2 * e + 1] = hi;
        }
      }
      // Stage 1: the fake fp4 round trip (exact in bf16, so no rounding after it).
      float amax = 0.0f;
#pragma unroll
      for (uint32_t e = 0; e < kElems; ++e) {
        amax = fmaxf(amax, fabsf(d[e]));
      }
      const auto [scale, inv_scale] = deepseek_v4::fp4::block_scale(group_max(amax));
      amax = 0.0f;
#pragma unroll
      for (uint32_t e = 0; e < kElems; e += 2) {
        const auto q = deepseek_v4::fp4::fake_quant_x2(fp32x2_t{d[e], d[e + 1]}, scale, inv_scale);
        d[e] = q.x;
        d[e + 1] = q.y;
        amax = fmaxf(amax, fmaxf(fabsf(q.x), fabsf(q.y)));
      }
      // Stage 2: the packer, byte i = elements (2i, 2i + 1), the even one in the low nibble.
      const auto exponent = index_pack_exponent(group_max(amax));
      const auto inv = deepseek_v4::fp8::inv_scale_ue8m0(static_cast<int32_t>(exponent));
      AlignedVector<uint32_t, kBytes / 4> packed;
#pragma unroll
      for (uint32_t w = 0; w < kBytes / 4; ++w) {
        uint32_t word = 0;
#pragma unroll
        for (uint32_t b = 0; b < 4; ++b) {
          const auto e = (w * 4 + b) * 2;
          const auto code = __nv_cvt_float2_to_fp4x2(fp32x2_t{d[e] * inv, d[e + 1] * inv}, __NV_E2M1, cudaRoundNearest);
          word |= static_cast<uint32_t>(code) << (8 * b);
        }
        packed[w] = clear_negative_zero(word);
      }
      if (tl < kCnt && t < num_tokens) {
        const auto row = size_t(t) * kHeads + tile;
        packed.store(p.payload + row * 64 + quad * 16 + part * kBytes);
        if (part == 0) {
          p.scale[row * 4 + quad] = static_cast<uint8_t>(exponent);
          if (quad == 0) {
            // head_weights(x).float(): the bf16 round trip is part of the reference.
            p.weights_out[row] = cast<float>(cast<bf16_t>(cast<float>(p.head_weights[row]) * p.weight_scale));
          }
        }
      }
    }
  };

  float acc[kOwnTokens];
  {
    float part[kOwnTokens];
    tcgen05::ld_32x32b<kOwnTokens>(acc_addr + own_t0, acc);
    tcgen05::ld_32x32b<kOwnTokens>(acc_addr + (kOwnTokens - own_t0), part);
    const auto peer = rank ^ 1u;
    const auto bar = ptx::mapa(ptx::to_shared(&smem.red_ready), peer);
    const auto dst = ptx::mapa(ptx::to_shared(&smem.inbox[(quad * kOwnTokens / 4 * 32 + lane_id) * 4]), peer);
#pragma unroll
    for (uint32_t g = 0; g < kOwnTokens / 4; ++g) {
      ptx::st_async_16B(dst + g * 32 * 16, &part[4 * g], bar);
    }

    tcgen05::wait_ld();
    tcgen05::fence_before_thread_sync();
    __syncwarp();
    if (lane_id == 0) ptx::mbar_arrive(&smem.drained);

    ptx::mbar_wait_parity(&smem.red_ready, 0);
    const auto inbox = &smem.inbox[(quad * kOwnTokens / 4 * 32 + lane_id) * 4];
#pragma unroll
    for (uint32_t g = 0; g < kOwnTokens / 4; ++g) {
      AlignedVector<fp32_t, 4> other;
      other.load(inbox + g * 32 * 4);
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        acc[4 * g + j] += other[j];
      }
    }
  }

  if constexpr (kEpi == Epilogue::kIndexQ) {
    bf16x2_t res[kOwnTokens / 2];
    round_and_rope(acc, res, own_t0);
    store_index_q(res, own_t0);
  } else {
    const auto stage = reinterpret_cast<bf16_t*>(smem.w);
#pragma unroll
    for (uint32_t k = 0; k < kOwnTokens / 2; ++k) {
      const auto v = cast<bf16x2_t>(fp32x2_t{acc[2 * k], acc[2 * k + 1]});
      stage[(2 * k + 0) * T::M_BLOCK + row_local] = v.x;
      stage[(2 * k + 1) * T::M_BLOCK + row_local] = v.y;
    }
    ptx::barrier_sync(BAR_EPI, 128);
    const bool lane_rope = rope_tile && lane_id >= 16;
    if (rope_tile) ptx::mbar_wait_parity(&smem.rope_ready, 0);
    const auto f0 = lane_id * 4;
    const auto out = p.out + row0 + f0;
#pragma unroll
    for (uint32_t tl = quad; tl < kOwnTokens; tl += 4) {
      const auto t = own_t0 + tl;
      AlignedVector<bf16x2_t, 2> v;
      v.load(stage + tl * T::M_BLOCK + f0);
      if (lane_rope) {
        AlignedVector<fp32_t, 4> cs;
        cs.load(&smem.rope[t * T::kRopeDim + f0 - 64]);
#pragma unroll
        for (uint32_t q = 0; q < 2; ++q) {
          const auto [xr, xi] = cast<fp32x2_t>(v[q]);
          const float fr = cs[2 * q], fi = cs[2 * q + 1];
          v[q] = cast<bf16x2_t>(fp32x2_t{xr * fr - xi * fi, xr * fi + xi * fr});
        }
      }
      if (t < num_tokens) v.store(out + size_t(t) * p.out_stride);
    }
  }
}

}  // namespace sglang::device::wq_b

namespace sglang {

/// A 2D row-major tensor as an (inner, outer) tensor map with box (box_inner, box_outer).
inline CUtensorMap make_wq_b_map(
    const void* ptr,
    CUtensorMapDataType dtype,
    uint64_t inner,
    uint64_t outer,
    uint64_t row_stride_bytes,
    uint32_t box_inner,
    uint32_t box_outer,
    CUtensorMapSwizzle swizzle) {
  using namespace host;
  // Through the runtime's driver entry point, so the module needs no -lcuda.
  static const auto encode = [] {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult status;
    CHECK_CUDA(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &status));
    CHECK_HOST(status == cudaDriverEntryPointSuccess && fn != nullptr);
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
  }();
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {row_stride_bytes};
  const cuuint32_t box[2] = {box_inner, box_outer};
  const cuuint32_t elem_strides[2] = {1, 1};
  CUtensorMap map;
  const auto res = encode(
      &map,
      dtype,
      2,
      const_cast<void*>(ptr),
      dims,
      strides,
      box,
      elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CHECK_HOST(res == CUDA_SUCCESS) << int(res);
  return map;
}

namespace detail {

/// Verifies the operands both entries share, fills their part of `p` and launches.
template <int N, int kOutDim, int kStages, int kInflight, device::wq_b::Epilogue kEpi>
inline void launch_wq_b_rope(
    device::wq_b::Params& p,
    host::SymbolicSize& num_tokens_,
    host::SymbolicDevice& device_,
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView x_scale,
    tvm::ffi::TensorView w,
    tvm::ffi::TensorView w_scale,
    tvm::ffi::Optional<tvm::ffi::TensorView> freqs,
    tvm::ffi::Optional<tvm::ffi::TensorView> positions,
    tvm::ffi::Optional<tvm::ffi::TensorView> out) {
  using namespace host;
  constexpr bool kRope = kEpi != device::wq_b::Epilogue::kPlain;
  using T = device::wq_b::Trait<N, kOutDim, kStages, kInflight, kRope>;
  TensorMatcher({num_tokens_, T::K})  //
      .with_strides({-1, 1})
      .with_dtype<fp8_e4m3_t>()
      .with_device(device_)
      .ensure_alignment(16)
      .verify(x);
  TensorMatcher({-1})  //
      .with_dtype<uint8_t>()
      .with_device(device_)
      .ensure_alignment(16)
      .verify(x_scale);
  CHECK_HOST(x.stride(0) % 16 == 0) << x.stride(0);
  CHECK_HOST(x_scale.size(0) >= 128 * T::K / 32) << x_scale.size(0);
  TensorMatcher({T::M, T::K})  //
      .with_strides({T::K, 1})
      .with_dtype<fp8_e4m3_t>()
      .with_device(device_)
      .ensure_alignment(16)
      .verify(w);
  TensorMatcher({-1})  //
      .with_dtype<uint8_t>()
      .with_device(device_)
      .ensure_alignment(16)
      .verify(w_scale);
  CHECK_HOST(freqs.has_value() == kRope && positions.has_value() == kRope);
  auto pos_dtype = SymbolicDType{};
  if constexpr (kRope) {
    TensorMatcher({-1, T::kRopeDim})  //
        .with_strides({T::kRopeDim, 1})
        .with_dtype<float>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(freqs.value());
    TensorMatcher({num_tokens_})  //
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device_)
        .verify(positions.value());
    p.freqs = static_cast<const float*>(freqs.value().data_ptr());
    p.positions = positions.value().data_ptr();
    p.pos_is_i64 = pos_dtype.is_type<int64_t>() ? 1 : 0;
  }
  CHECK_HOST(out.has_value() == (kEpi != device::wq_b::Epilogue::kIndexQ));
  if (out.has_value()) {
    TensorMatcher({num_tokens_, T::M})  //
        .with_strides({-1, 1})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(out.value());
    CHECK_HOST(out.value().stride(0) % 8 == 0) << out.value().stride(0);
  }
  const auto num_tokens = num_tokens_.unwrap();
  CHECK_HOST(num_tokens <= N) << num_tokens;
  CHECK_HOST(w_scale.size(0) >= T::M * T::K / 32) << w_scale.size(0);
  if (num_tokens == 0) return;  // an empty exchange would wait forever on its inbox

  p.x_map = make_wq_b_map(
      x.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, T::K, num_tokens, x.stride(0), 128, N, CU_TENSOR_MAP_SWIZZLE_128B);
  p.x_scale = static_cast<const uint8_t*>(x_scale.data_ptr());
  p.w_map = make_wq_b_map(
      w.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, T::K, T::M, T::K, 128, T::M_BLOCK, CU_TENSOR_MAP_SWIZZLE_128B);
  p.w_scale = static_cast<const uint8_t*>(w_scale.data_ptr());
  if (out.has_value()) {
    p.out = static_cast<bf16_t*>(out.value().data_ptr());
    p.out_stride = static_cast<uint32_t>(out.value().stride(0));
  }
  p.num_tokens = static_cast<uint32_t>(num_tokens);

  const auto device = device_.unwrap();
  constexpr auto kernel = device::wq_b::wq_b_rope_kernel<N, kOutDim, kStages, kInflight, kEpi>;
  constexpr size_t kSmemBytes = sizeof(typename T::Smem);
  runtime::init_per_device(device, [] {
    runtime::set_max_dynamic_smem(kernel, kSmemBytes);
    runtime::set_smem_carveout(kernel, cudaSharedmemCarveoutMaxShared);
  });
  LaunchKernel(static_cast<uint32_t>(T::kNumCtas), T::kThreads, device, kSmemBytes)
      .config({.use_pdl = true, .cluster_dim = dim3{T::kSplitK, 1, 1}})(kernel, p);
}

}  // namespace detail

/**
 * \brief Low-latency wq_b + RoPE: q = bf16(x @ W^T) viewed as [T, 16, 512], then RoPE (fused_rope_inplace's
 *        arithmetic) on dims [448, 512) of every head at `positions`.
 *
 * Launched as a PDL secondary that releases its own dependents at start; the weight, `freqs` and `positions` are
 * read before the PDL wait, only `x` and `x_scale` after it.
 * \tparam N         MMA N (8 / 16 / 32 / 64 / 128); T must be <= N.
 * \tparam kStages   weight ring depth in 128-K subtiles (the activation ring takes the smem left).
 * \tparam kInflight weight subtiles in flight per CTA.
 * \param x          [T, 1280] e4m3, row stride a multiple of 16 bytes.
 * \param x_scale    its UE8M0 scales, FlashInfer 128x4 image, >= 128 * 40 bytes.
 * \param w          weight [8192, 1280] e4m3, contiguous.
 * \param w_scale    its UE8M0 per-32 scales, FlashInfer 128x4 image, >= 8192 * 40 bytes.
 * \param freqs      [max_pos, 64] fp32, interleaved (cos, sin) (view_as_real of the complex table).
 * \param positions  [T] int32 or int64.
 * \param out        [T, 8192] bf16, contiguous (the [T, 16, 512] q).
 */
template <int N, int kStages, int kInflight>
inline void dsv41_wq_b_rope_ll(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView x_scale,
    tvm::ffi::TensorView w,
    tvm::ffi::TensorView w_scale,
    tvm::ffi::TensorView freqs,
    tvm::ffi::TensorView positions,
    tvm::ffi::TensorView out) {
  using namespace host;
  auto num_tokens_ = SymbolicSize{"num_tokens"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();
  device::wq_b::Params p{};
  detail::launch_wq_b_rope<N, 8192, kStages, kInflight, device::wq_b::Epilogue::kRope>(
      p, num_tokens_, device_, x, x_scale, w, w_scale, freqs, positions, out);
}

/**
 * \brief The indexer's wq_b: q = bf16(x @ W^T), W [4096, 1280] (32 heads x 128), no RoPE (the indexer's fused
 *        RoPE + fp4 rounding follows). x is the same rmsnorm_mxfp8 output dsv41_wq_b_rope_ll takes, so the
 *        indexer needs no quantization of its own.
 * \tparam N         MMA N (8 / 16 / 32 / 64 / 128); T must be <= N.
 * \tparam kStages   weight ring depth in 128-K subtiles.
 * \tparam kInflight weight subtiles in flight per CTA.
 * \param out        [T, 4096] bf16, contiguous.
 * Other parameters as dsv41_wq_b_rope_ll.
 */
template <int N, int kStages, int kInflight>
inline void dsv41_indexer_wq_b_ll(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView x_scale,
    tvm::ffi::TensorView w,
    tvm::ffi::TensorView w_scale,
    tvm::ffi::TensorView out) {
  using namespace host;
  auto num_tokens_ = SymbolicSize{"num_tokens"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();
  device::wq_b::Params p{};
  detail::launch_wq_b_rope<N, 4096, kStages, kInflight, device::wq_b::Epilogue::kPlain>(
      p,
      num_tokens_,
      device_,
      x,
      x_scale,
      w,
      w_scale,
      tvm::ffi::Optional<tvm::ffi::TensorView>{},
      tvm::ffi::Optional<tvm::ffi::TensorView>{},
      out);
}

/**
 * \brief The indexer's queries in one launch: dsv41_indexer_wq_b_ll's GEMM with index_q_rope_pack_weights fused
 *        into the epilogue (RoPE on dims [64, 128) of every 128-dim head at `positions`, both fp4 stages, the
 *        packed payload and ue8m0 scales, the scaled head weights). The bf16 q is never stored.
 * \tparam N            MMA N (8 / 16 / 32 / 64 / 128); T must be <= N.
 * \tparam kStages      weight ring depth in 128-K subtiles.
 * \tparam kInflight    weight subtiles in flight per CTA.
 * \param payload       [T * 32, 64] int8, e2m1 pairs, the low nibble the even element.
 * \param scale         [T * 32] int32, the four block exponents little-endian.
 * \param head_weights  [T, 32] bf16, the raw weights_proj output (written before this launch).
 * \param weights_out   [T, 32] fp32, bf16(head_weights * weight_scale).
 * \param weight_scale  softmax_scale * heads^-0.5.
 * Other parameters as dsv41_wq_b_rope_ll (freqs is the whole [max_pos, 64] table, gathered at positions).
 */
template <int N, int kStages, int kInflight>
inline void dsv41_indexer_q_fp4_ll(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView x_scale,
    tvm::ffi::TensorView w,
    tvm::ffi::TensorView w_scale,
    tvm::ffi::TensorView freqs,
    tvm::ffi::TensorView positions,
    tvm::ffi::TensorView payload,
    tvm::ffi::TensorView scale,
    tvm::ffi::TensorView head_weights,
    tvm::ffi::TensorView weights_out,
    double weight_scale) {
  using namespace host;
  constexpr int kHeads = 32;
  auto num_tokens_ = SymbolicSize{"num_tokens"};
  auto num_rows_ = SymbolicSize{"num_rows"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();
  TensorMatcher({num_rows_, 64}).with_strides({64, 1}).with_dtype<int8_t>().with_device(device_).verify(payload);
  TensorMatcher({num_rows_}).with_dtype<int32_t>().with_device(device_).verify(scale);
  TensorMatcher({num_tokens_, kHeads})
      .with_strides({kHeads, 1})
      .with_dtype<bf16_t>()
      .with_device(device_)
      .verify(head_weights);
  TensorMatcher({num_tokens_, kHeads})
      .with_strides({kHeads, 1})
      .with_dtype<fp32_t>()
      .with_device(device_)
      .verify(weights_out);
  device::wq_b::Params p{};
  p.head_weights = static_cast<const bf16_t*>(head_weights.data_ptr());
  p.weights_out = static_cast<float*>(weights_out.data_ptr());
  p.payload = static_cast<int8_t*>(payload.data_ptr());
  p.scale = static_cast<uint8_t*>(scale.data_ptr());
  p.weight_scale = static_cast<float>(weight_scale);
  detail::launch_wq_b_rope<N, kHeads * 128, kStages, kInflight, device::wq_b::Epilogue::kIndexQ>(
      p, num_tokens_, device_, x, x_scale, w, w_scale, freqs, positions, tvm::ffi::Optional<tvm::ffi::TensorView>{});
  CHECK_HOST(num_rows_.unwrap() == num_tokens_.unwrap() * kHeads) << num_rows_.unwrap();
}

}  // namespace sglang
