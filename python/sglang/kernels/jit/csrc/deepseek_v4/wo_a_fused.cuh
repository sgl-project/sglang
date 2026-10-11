// Fused inverse-RoPE + grouped WO-A BF16 GEMM + MXFP8 quantization, SM100.
//
// Collapses three launches on the DSV4 decode/verify path (fused_rope_inplace,
// _wo_a_partial, _wo_a_reduce_quant) into one cluster-launched kernel.
//
//   x [T, 2, 4096] bf16 (a strided view of a [T, 64, 512] attention output)
//   W [2, 1024, 4096] bf16
//   -> q [T, 2048] e4m3 + scales [8192] ue8m0, and/or y [T, 2048] bf16
//
// Decomposition: the 2048 output columns split into 32 N-tiles of 64, one per
// cluster; K splits 8 ways across the cluster's CTAs, so each CTA owns exactly
// one of the group's 8 heads -- and therefore exactly one 64-wide RoPE window,
// which is exactly one staged K-tile. That is what folds the rope launch in
// without a grid-wide dependency.
//
// swapAB: the MMA's M axis is the 64 output columns (A = weights) and its N
// axis is the 16 tokens. Tokens as M would pad 8x and blow the A operand up to
// 64xK, as large as the weights themselves.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace sglang::device::ptx {

// ---- TMA -------------------------------------------------------------------
// Coordinate convention: the tensor map's globalDim is (inner, outer) -- dim 0
// is the stride-1 axis -- and the load takes (x = inner, y = outer). Swapping
// them loads a transposed tile with plausible magnitudes and scrambled pairing.
// Warm the cache line holding the tensor-map descriptor so the first TMA does
// not pay the descriptor fetch on top of the DRAM latency.
SGL_DEVICE void prefetch_tensormap(const void* tmap) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
}

SGL_DEVICE void
cp_async_bulk_tensor_2d(uint32_t dst_smem, const CUtensorMap* tmap, int32_t x, int32_t y, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst_smem),
      "l"(tmap),
      "r"(x),
      "r"(y),
      "r"(to_shared(bar))
      : "memory");
}

// Push partial sums into a peer's inbox and credit its mbarrier byte count.
SGL_DEVICE void st_async_b32(uint32_t dst_dsmem, float value, uint32_t dst_bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(dst_dsmem),
               "f"(value),
               "r"(dst_bar)
               : "memory");
}

// Generic stores land in a different proxy than the one the MMA and the bulk
// copy engine read through.
SGL_DEVICE void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// Retarget a local smem offset at CTA `rank` of this cluster (DSMEM).
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

// Publishes mbarrier initialization to the whole cluster. Because this fence
// carries the release, the cluster arrive below needs no ordering of its own.
SGL_DEVICE void fence_mbarrier_init_release_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// Split cluster handshake: every CTA arrives in the prologue, but only the code
// that actually touches a peer's memory pays for the wait.
SGL_DEVICE void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
SGL_DEVICE void cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

SGL_DEVICE void bar_sync(uint32_t id, uint32_t threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

// ---- tcgen05 ---------------------------------------------------------------
// Mandatory order: alloc (one whole warp, n_cols a power of 2 in [32,512],
// taddr written to smem) -> barrier + read taddr -> mma/ld -> dealloc.
// `relinquish_alloc_permit` only promises this CTA will not allocate again, so
// it is issued right after the single alloc: until it retires, no second CTA
// on the SM can allocate, which would pin the kernel to 1 CTA/SM.
SGL_DEVICE void tcgen05_alloc(uint32_t smem_dst, uint32_t n_cols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_dst), "r"(n_cols));
}
SGL_DEVICE void tcgen05_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
SGL_DEVICE void tcgen05_dealloc(uint32_t taddr, uint32_t n_cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(n_cols));
}
SGL_DEVICE void tcgen05_commit_arrive(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(to_shared(bar)));
}
SGL_DEVICE void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}
// The "memory" clobber is load-bearing: ptxas lowers the wait to scoreboard
// waits on the dependent register consumers, so anything that does not read the
// drained registers can be hoisted above the last drain.
SGL_DEVICE void tcgen05_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// 32 lanes x 16 TMEM columns -> 16 b32 per lane (one per token).
SGL_DEVICE void tcgen05_ld_32x32b_x16(uint32_t taddr, void* dst) {
  const auto d = static_cast<uint32_t*>(dst);
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=r"(d[0]),
        "=r"(d[1]),
        "=r"(d[2]),
        "=r"(d[3]),
        "=r"(d[4]),
        "=r"(d[5]),
        "=r"(d[6]),
        "=r"(d[7]),
        "=r"(d[8]),
        "=r"(d[9]),
        "=r"(d[10]),
        "=r"(d[11]),
        "=r"(d[12]),
        "=r"(d[13]),
        "=r"(d[14]),
        "=r"(d[15])
      : "r"(taddr));
}

SGL_DEVICE void tcgen05_ld_32x32b_x32(uint32_t taddr, void* dst) {
  const auto d = static_cast<uint32_t*>(dst);
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%"
      "30,%31}, [%32];"
      : "=r"(d[0]),
        "=r"(d[1]),
        "=r"(d[2]),
        "=r"(d[3]),
        "=r"(d[4]),
        "=r"(d[5]),
        "=r"(d[6]),
        "=r"(d[7]),
        "=r"(d[8]),
        "=r"(d[9]),
        "=r"(d[10]),
        "=r"(d[11]),
        "=r"(d[12]),
        "=r"(d[13]),
        "=r"(d[14]),
        "=r"(d[15]),
        "=r"(d[16]),
        "=r"(d[17]),
        "=r"(d[18]),
        "=r"(d[19]),
        "=r"(d[20]),
        "=r"(d[21]),
        "=r"(d[22]),
        "=r"(d[23]),
        "=r"(d[24]),
        "=r"(d[25]),
        "=r"(d[26]),
        "=r"(d[27]),
        "=r"(d[28]),
        "=r"(d[29]),
        "=r"(d[30]),
        "=r"(d[31])
      : "r"(taddr));
}

// Valid dense kind::f16 shapes: M in {64,128}, N in {8..256 step 8}, K = 16.
// ptxas does NOT reject off-table shapes; they fault at runtime.
SGL_DEVICE void tcgen05_mma_f16(uint32_t d, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, bool accumulate) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %4, 0;"
      "  tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }" ::"r"(d),
      "l"(desc_a),
      "l"(desc_b),
      "r"(idesc),
      "r"(uint32_t(accumulate)));
}

// Smem matrix descriptor. The public PTX spec is wrong at bits 46-60; bits
// 46-47 must be version=1 on Blackwell or 128B-swizzle BLOCK_K>16 yields
// garbage. 0-13 addr>>4 | 16-29 lbo>>4 | 32-45 sbo>>4 | 46-47 version
// | 49-51 base_offset | 61-63 layout (0=none, 2=128B, 4=64B, 6=32B).
// K-major mandates swizzle bytes == BLOCK_K * sizeof(T), i.e. 64 bf16.
SGL_DEVICE uint64_t smem_desc_k_major_128b(uint32_t addr) {
  auto enc = [](uint32_t v) -> uint64_t { return uint64_t((v & 0x3FFFFu) >> 4); };
  return enc(addr) | (enc(0u) << 16) | (enc(8u * 128u) << 32) | (uint64_t(1) << 46) | (uint64_t(2) << 61);
}

// The address sits in bits 0-13 as addr>>4, so a byte offset folds into a
// descriptor with a plain add; there is no need to rebuild it per tile.
SGL_DEVICE uint64_t smem_desc_add(uint64_t desc, uint32_t byte_offset) {
  return desc + (byte_offset >> 4);
}

// Instruction descriptor, kind::f16 (PTX ISA Table 44). Major::K = 0 for both
// operands, d_type F32, a/b_type BF16.
constexpr uint32_t inst_desc_bf16(uint32_t M, uint32_t N) {
  return (1u << 4) | (1u << 7) | (1u << 10) | (((N >> 3) & 0x3Fu) << 17) | (((M >> 4) & 0x1Fu) << 24);
}

SGL_DEVICE void barrier_sync(uint32_t id, uint32_t n) {
  asm volatile("barrier.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}

SGL_DEVICE void barrier_arrive(uint32_t id, uint32_t n) {
  asm volatile("barrier.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}

}  // namespace sglang::device::ptx

namespace sglang::wo_a_fused {

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------
/**
 * \brief Compile-time shape for the TP4 WO-A decomposition.
 *
 * \tparam kMaxTokens Token capacity; also the MMA's N axis, so a multiple of 8.
 *
 * 2048 output columns split into kClusters N-tiles of kNTile, one per cluster;
 * K splits kSplitK ways across the cluster's CTAs, giving each CTA exactly one
 * of the group's 8 heads and therefore exactly one RoPE window.
 */
template <int kMaxTokens>
struct WoATrait {
  static_assert(kMaxTokens == 16 || kMaxTokens == 32, "only the x16/x32 TMEM loads are wired up");

  static constexpr int kMMax = kMaxTokens;
  static constexpr int kNTile = 64;  // == 2 MXFP8 blocks
  static constexpr int kSplitK = 8;  // == cluster size
  static constexpr int kGroupK = 4096;
  static constexpr int kKSlab = kGroupK / kSplitK;
  static constexpr int kBlockK = 64;  // 128 B per row == the 128B swizzle width
  static constexpr int kStages = kKSlab / kBlockK;
  static constexpr int kHeadDim = 512;
  static constexpr int kHeadStages = kHeadDim / kBlockK;
  static constexpr int kHeadsPerCta = kKSlab / kHeadDim;
  static constexpr int kRopeDim = 64;
  static constexpr int kNOut = 2048;  // 2 groups x 1024
  // One UE8M0 byte per MXFP8 block, for 128 padded token rows.
  static constexpr int kScaleBytes = (kNOut / 32) * 128;
  static constexpr int kClusters = kNOut / kNTile;

  // 8 warps x 2 CTAs/SM = 512 threads/SM, i.e. a 128-register budget per thread.
  static constexpr int kWarps = 8;
  static constexpr int kThreads = kWarps * device::kWarpThreads;
  static constexpr int kDrainWarps = 4;  // the M=64 accumulator spans 4 TMEM lane bands
  // Layout F gives each 32-lane band kBandCols of the 64 output columns, in
  // lanes [0, kBandCols). A property of M, not of the token count.
  static constexpr int kBandCols = kNTile / kDrainWarps;
  static constexpr int kTmemCols = 32;  // alloc granularity, a power of 2 >= 32

  static constexpr int kWTile = kNTile * kBlockK * 2;
  static constexpr int kXTile = kMMax * kBlockK * 2;
  // Rank r owns tokens [r*kTokPerRank, ...), one warp each, so the epilogue
  // never needs more than 4 warps and the back half is free for cleanup.
  static constexpr int kTokPerRank = (kMMax + kSplitK - 1) / kSplitK;
  static constexpr int kEpiWarps = kTokPerRank;
  static constexpr int kCleanupWarp = kWarps / 2;

  // The RoPE window is a head's trailing 64 lanes == that head's last K-tile.
  static constexpr bool is_rope_stage(int s) {
    return s % kHeadStages == kHeadStages - 1;
  }

  struct alignas(1024) Smem {
    alignas(1024) uint8_t w[kStages][kWTile];
    alignas(1024) uint8_t x[kStages][kXTile];
    // Every rank pushes into the owner's inbox, itself included, so there is no
    // separate local staging buffer.
    alignas(16) float inbox[kSplitK][kTokPerRank][kNTile];
    uint64_t w_ready[kStages];
    uint64_t x_ready[kStages];
    uint64_t x_landed[kHeadsPerCta];
    uint64_t mma_done;
    uint64_t reduce_done;
    uint64_t tmem_ld_done;
    uint32_t taddr;
  };
  // 2 CTAs/SM needs <= 113 KiB each; the 32-cluster wave depends on it.
  static_assert(sizeof(Smem) <= 115712, "smem would drop the kernel to 1 CTA/SM");
};

// ---------------------------------------------------------------------------

// Byte offset inside a K-major, 128B-swizzled tile of 64 bf16 per row, in
// 8-row core groups of 1024 B. TMA writes this layout natively.
constexpr uint32_t kmajor_off(uint32_t r, uint32_t c) {
  return (r >> 3) * 1024u + (r & 7u) * 128u + ((c ^ ((r & 7u) << 3)) << 1);
}

// FlashInfer 128x4 scale swizzle: g = MXFP8 block index, r = token row.
constexpr uint32_t sf_off(uint32_t g, uint32_t r) {
  return (g >> 2) * 512u + ((r % 32u) * 4u + ((r / 32u) % 4u)) * 4u + (g & 3u);
}

// Positive round-to-next-power-of-two exponent, bit-identical to ue8m0_scale()
// in mxfp8_epilogue.py (subnormals included, clamped at 254).
SGL_DEVICE void ue8m0_scale(float amax, uint32_t& sf, float& inv) {
  // amax is a max of magnitudes, so it is non-negative and the sign bit is 0;
  // `bits == 0` is then exactly the `normalized <= 0` case, which lets both
  // results be selects instead of a branch.
  const uint32_t bits = __float_as_uint(amax * (1.0f / 448.0f));
  const uint32_t exponent = bits >> 23;
  const uint32_t mantissa = bits & 0x7FFFFFu;
  const bool bump = mantissa != 0u && !(exponent == 0u && mantissa <= 0x400000u);
  sf = bits == 0u ? 0u : min(exponent + uint32_t(bump), 254u);
  // Same 2^(127-sf) form as deepseek_v4::fp8::inv_scale_ue8m0.
  inv = bits == 0u ? 0.0f : __uint_as_float((254u - sf) << 23);
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

// `PosT` follows the caller's position buffer: DSV4 decode hands us int32, but
// the DSpark verify path hands us int64. Templating beats casting in Python,
// which would cost one launch per layer on the decode critical path.
template <typename Trait, typename PosT>
__global__ void __cluster_dims__(Trait::kSplitK, 1, 1) __launch_bounds__(Trait::kThreads, 2) wo_a_fused_kernel(
    const __grid_constant__ CUtensorMap w_map,
    const __grid_constant__ CUtensorMap x_map,
    const float* __restrict__ freqs,     // [S, 64] fp32, (cos0, sin0, cos1, ...)
    const PosT* __restrict__ positions,  // [T] int32 or int64
    fp8_e4m3_t* __restrict__ q,          // [T, 2048], may be null
    uint8_t* __restrict__ scales_out,    // [8192], may be null when q is
    bf16_t* __restrict__ y,              // [T, 2048], may be null
    uint32_t m) {
  using namespace device;
  // The trait's shape, unqualified, so the body below reads as plain constants.
  constexpr int kMMax = Trait::kMMax;
  constexpr int kNTile = Trait::kNTile;
  constexpr int kSplitK = Trait::kSplitK;
  constexpr int kGroupK = Trait::kGroupK;
  constexpr int kKSlab = Trait::kKSlab;
  constexpr int kBlockK = Trait::kBlockK;
  constexpr int kStages = Trait::kStages;
  constexpr int kHeadStages = Trait::kHeadStages;
  constexpr int kHeadsPerCta = Trait::kHeadsPerCta;
  constexpr int kRopeDim = Trait::kRopeDim;
  constexpr int kNOut = Trait::kNOut;
  constexpr int kWarps = Trait::kWarps;
  constexpr int kThreads = Trait::kThreads;
  constexpr int kDrainWarps = Trait::kDrainWarps;
  constexpr int kBandCols = Trait::kBandCols;
  constexpr int kTmemCols = Trait::kTmemCols;
  constexpr int kWTile = Trait::kWTile;
  constexpr int kXTile = Trait::kXTile;
  constexpr int kTokPerRank = Trait::kTokPerRank;
  constexpr int kEpiWarps = Trait::kEpiWarps;
  constexpr int kCleanupWarp = Trait::kCleanupWarp;
  constexpr auto is_rope_stage = Trait::is_rope_stage;
  using Smem = typename Trait::Smem;
  extern __shared__ Smem smem_raw[];
  Smem& smem = smem_raw[0];
  enum Warp : uint32_t { TMA_W = 0, TMA_X = 1, TMA_ROPE = 2, MMA = 3, ROPE = 4 };
  enum Barrier : uint32_t { BAR_MBAR = 2, BAR_MBAR_MMA = 3 };
  constexpr int kRopeWarps = kWarps - ROPE;
  constexpr int kRopeThreads = kRopeWarps * device::kWarpThreads;

  const uint32_t tx = threadIdx.x;
  const uint32_t warp_id = tx / kWarpThreads;
  const uint32_t lane_id = tx % kWarpThreads;
  const uint32_t rank = ptx::cluster_ctarank();  // == the head this CTA owns
  const uint32_t work_id = blockIdx.x / kSplitK;
  const uint32_t group = work_id / (1024 / kNTile);
  const uint32_t n0 = (work_id % (1024 / kNTile)) * kNTile;
  const uint32_t k0 = rank * kKSlab;

  // Before anything else: the descriptor fetch is serial with the first tile's
  // DRAM latency, which is the kernel's single largest term.

  // Claim TMEM before the barrier init, not after the sync. tcgen05.alloc costs
  // ~1.0 us (barrier init is free, ~0 us), and issuing it first lets that
  // latency overlap the init and the sync; deferring it past the sync measured
  // 0.43 us slower end to end. The permit is dropped immediately -- holding it
  // is what pins the kernel to one CTA per SM (measured: 2 co-resident with the
  // relinquish, 1 without).
  if (warp_id == MMA) {
    ptx::tcgen05_alloc(ptx::to_shared(&smem.taddr), kTmemCols);
    ptx::tcgen05_relinquish();
  } else {
    if (warp_id == TMA_W) {
      ptx::prefetch_tensormap(&w_map);
    } else if (warp_id == TMA_X) {
      ptx::prefetch_tensormap(&x_map);
    } else if (warp_id == TMA_ROPE) {
      if (lane_id < kStages) {
        ptx::mbar_init(&smem.w_ready[lane_id], 1);
        ptx::mbar_init(&smem.x_ready[lane_id], lane_id == kStages - 1 ? kRopeThreads : 1);
      }
      if (lane_id < kHeadsPerCta) ptx::mbar_init(&smem.x_landed[lane_id], 1);
      if (lane_id == 31) {
        ptx::mbar_init(&smem.mma_done, 1);
        ptx::mbar_init(&smem.reduce_done, 1);
        ptx::mbar_init(&smem.tmem_ld_done, kDrainWarps * kWarpThreads);
      }
      ptx::fence_mbarrier_init_release_cluster();
    }
    ptx::bar_sync(BAR_MBAR, kThreads - 32);
  }
  // signal mbarrier init done
  ptx::cluster_arrive_relaxed();

  if (warp_id == TMA_W) {
    // The weights are not the producer kernel's output, so this warp never waits
    // on PDL and the whole 16 MiB stream overlaps the attention tail.
    if (warp::elect_one_lane()) {
#pragma unroll
      for (int s = 0; s < kStages; ++s) {
        ptx::cp_async_bulk_tensor_2d(
            ptx::to_shared(smem.w[s]), &w_map, k0 + s * kBlockK, group * 1024 + n0, &smem.w_ready[s]);
        ptx::mbar_arrive_expect_tx(&smem.w_ready[s], kWTile);
      }
    }
  } else if (warp_id == TMA_X) {
    // Activation loader, plain tiles only; the rope group owns the rest. Rows
    // >= m are out of bounds in the tensor map and TMA zero-fills them, so
    // nothing downstream needs masking.
    if (warp::elect_one_lane()) {
      // x IS the producer kernel's output, so only the lanes that issue its TMA
      // need the PDL wait; freqs and positions were written long before.
      PDLWaitPrimary<true>();
#pragma unroll
      for (int s = 0; s < kStages; ++s) {
        if (is_rope_stage(s)) continue;
        ptx::cp_async_bulk_tensor_2d(
            ptx::to_shared(smem.x[s]), &x_map, group * kGroupK + k0 + s * kBlockK, 0, &smem.x_ready[s]);
        ptx::mbar_arrive_expect_tx(&smem.x_ready[s], kXTile);
      }
    }
  } else if (warp_id == TMA_ROPE) {
    if (warp::elect_one_lane()) {
      PDLWaitPrimary<true>();
#pragma unroll
      for (int h = 0; h < kHeadsPerCta; ++h) {
        const int s = (h + 1) * kHeadStages - 1;
        ptx::cp_async_bulk_tensor_2d(
            ptx::to_shared(smem.x[s]), &x_map, group * kGroupK + k0 + s * kBlockK, 0, &smem.x_landed[h]);
        ptx::mbar_arrive_expect_tx(&smem.x_landed[h], kXTile);
      }
    }
    ptx::barrier_arrive(BAR_MBAR_MMA, 64);
  } else if (warp_id == MMA) {
    ptx::barrier_sync(BAR_MBAR_MMA, 64);
    const uint32_t taddr = smem.taddr;
    if (warp::elect_one_lane()) {
      constexpr uint32_t idesc = ptx::inst_desc_bf16(kNTile, kMMax);
      // Build both descriptors once; stage and K steps are plain adds.
      const uint64_t w_desc0 = ptx::smem_desc_k_major_128b(ptx::to_shared(smem.w[0]));
      const uint64_t x_desc0 = ptx::smem_desc_k_major_128b(ptx::to_shared(smem.x[0]));
      // No fence::after_thread_sync in the loop: every stage has its own buffer,
      // so there is no WAR hazard against an earlier MMA that would need it.
#pragma unroll
      for (int s = 0; s < kStages; ++s) {
        ptx::mbar_wait_parity(&smem.x_ready[s], 0);
        ptx::mbar_wait_parity(&smem.w_ready[s], 0);
        const uint64_t w_desc = ptx::smem_desc_add(w_desc0, s * kWTile);
        const uint64_t x_desc = ptx::smem_desc_add(x_desc0, s * kXTile);
#pragma unroll
        for (int k = 0; k < kBlockK / 16; ++k) {
          ptx::tcgen05_mma_f16(
              taddr, ptx::smem_desc_add(w_desc, k * 32), ptx::smem_desc_add(x_desc, k * 32), idesc, s || k);
        }
      }
      ptx::tcgen05_commit_arrive(&smem.mma_done);
    }
  } else if (warp_id >= ROPE) {
    const uint32_t rope_tx = tx - ROPE * 32;
    const uint32_t pair = rope_tx % kWarpThreads;
    const uint32_t phase = rope_tx / kWarpThreads;
    // Prefetch the rotation table BEFORE issuing the tiles: neither positions
    // nor freqs is producer output, and __shfl_sync reconverges the warp, so
    // ordering this after lane 0's PDL wait would stall all 32 lanes on it. Read inside the
    // rotate loop instead, positions[t] -> freqs[...] is a two-level dependent
    // chain that a runtime-bounded loop cannot pipeline.
    constexpr int kTokPerThread = kMMax / kRopeWarps;
    float2 rot[kTokPerThread];
#pragma unroll
    for (int i = 0; i < kTokPerThread; ++i) {
      const uint32_t t = phase + i * kRopeWarps;
      if (t < m) rot[i] = reinterpret_cast<const float2*>(freqs + static_cast<size_t>(positions[t]) * kRopeDim)[pair];
    }
#pragma unroll
    for (int h = 0; h < kHeadsPerCta; ++h) {
      const int s = (h + 1) * kHeadStages - 1;
      ptx::mbar_wait_parity(&smem.x_landed[h], 0);
      uint8_t* tile = smem.x[s];
#pragma unroll
      for (int i = 0; i < kTokPerThread; ++i) {
        const uint32_t t = phase + i * kRopeWarps;
        if (t >= m) continue;
        const auto p = reinterpret_cast<__nv_bfloat162*>(tile + kmajor_off(t, 2 * pair));
        const float2 v = __bfloat1622float2(*p);
        const float2 f = rot[i];
        // (a + bi) * conj(c + di); rounded back to bf16 because the reference
        // stores the rotated tensor and the GEMM re-reads it.
        *p = __floats2bfloat162_rn(v.x * f.x + v.y * f.y, v.y * f.x - v.x * f.y);
      }
      // The MMA reads smem through the async proxy; the rope wrote it generically.
      ptx::fence_proxy_async_shared();
      ptx::mbar_arrive(&smem.x_ready[s]);
    }
  }

  // ---- all warps: drain TMEM ----------------------------------------------
  // [measured on B200] For M=64, N=16 the accumulator lands at
  //   (lane, column) -> (m = (lane/32)*16 + lane%32, n = column)
  // with only lane%32 < 16 populated: each 32-lane band carries 16 of the 64
  // output columns. So four warps drain, and one 32-column MXFP8 block spans
  // two of them -- which is why the epilogue reduces through smem.
  ptx::mbar_wait_parity(&smem.mma_done, 0);
  // Mandatory here: without it the register reads can observe stale TMEM even
  // though the mbarrier already signalled the MMA complete.
  ptx::tcgen05_fence_after_thread_sync();
  const uint32_t taddr = smem.taddr;

  float acc[kMMax];
  if (warp_id < kDrainWarps) {
    const uint32_t col = warp_id * kBandCols + lane_id;
    if constexpr (kMMax == 16) {
      ptx::tcgen05_ld_32x32b_x16(taddr + ((warp_id * 32) << 16), acc);
    } else {
      ptx::tcgen05_ld_32x32b_x32(taddr + ((warp_id * 32) << 16), acc);
    }
    ptx::tcgen05_wait_ld();
    ptx::mbar_arrive_relaxed(&smem.tmem_ld_done);
    ptx::cluster_wait_acquire();
    // layout F for tcgen05.mma: only lane [0, kBandCols) is active
    if (lane_id < kBandCols) {
#pragma unroll
      for (int owner = 0; owner < kSplitK; ++owner) {
        const auto bar = ptx::mapa(ptx::to_shared(&smem.reduce_done), owner);
        const auto base = ptx::mapa(ptx::to_shared(&smem.inbox[rank][0][col]), owner);
#pragma unroll
        for (int tl = 0; tl < kTokPerRank; ++tl) {
          // Skip tokens past m: they are padding, and the owner's expect_tx
          // below counts only the live ones, so the bytes are never owed.
          const uint32_t t = owner * kTokPerRank + tl;
          if (t < m) ptx::st_async_b32(base + tl * kNTile * 4, acc[t], bar);
        }
      }
    }

    // Only the first kEpiWarps warps own a token; inbox's middle dim is
    // kTokPerRank, and t would otherwise collide with the next rank's tokens.
    const auto col0 = group * 1024 + n0;
    const auto t = rank * kTokPerRank + warp_id;
    // Unconditional, and before the guard: this is what keeps the CTA alive
    // until every peer's st.async into our inbox has landed. A rank whose tokens
    // are all >= m has nothing to emit, but it is still a write target, and a
    // CTA that exits with peers still writing its smem faults the launch.
    ptx::mbar_wait_parity(&smem.reduce_done, 0);
    if (warp_id >= kEpiWarps || t >= m) return;
    float2 v = {
        smem.inbox[0][warp_id][lane_id + 0],
        smem.inbox[0][warp_id][lane_id + 32],
    };
#pragma unroll
    for (int p = 1; p < kSplitK; ++p) {
      v.x += smem.inbox[p][warp_id][lane_id + 0];
      v.y += smem.inbox[p][warp_id][lane_id + 32];
    }
    const auto packed = cast<bf16x2_t>(v);
    if (q != nullptr) {
      static_assert(kNTile == 64);
      // The reference rounds the fp32 sum to bf16 and back before quantizing.
      const auto [lo, hi] = cast<float2>(packed);
      uint32_t sf_lo, sf_hi;
      float inv_lo, inv_hi;
      // Both reductions need every lane; the warp is masked uniformly by the
      // guard above, which depends only on rank and warp_id.
      ue8m0_scale(warp::reduce_max(fabsf(lo)), sf_lo, inv_lo);
      ue8m0_scale(warp::reduce_max(fabsf(hi)), sf_hi, inv_hi);
      const auto out = t * kNOut + col0 + lane_id;
      // SATFINITE already saturates +-inf, so only an upper clamp is needed --
      // without it a NaN input would convert to an fp8 NaN code.
      q[out + 0] = cast<fp8_e4m3_t>(fminf(lo * inv_lo, kFP8E4M3Max));
      q[out + 32] = cast<fp8_e4m3_t>(fminf(hi * inv_hi, kFP8E4M3Max));
      if (lane_id == 0) {
        scales_out[sf_off(work_id * 2 + 0, t)] = uint8_t(sf_lo);
        scales_out[sf_off(work_id * 2 + 1, t)] = uint8_t(sf_hi);
      }
      if (y != nullptr) {
        y[out] = packed.x;
        y[out + 32] = packed.y;
      }
    } else if (y != nullptr) {
      const auto out = t * kNOut + col0 + lane_id;
      y[out] = packed.x;
      y[out + 32] = packed.y;
    }
  } else {
    if (tx == 128) {
      // Only this rank's live tokens are pushed, by every sender.
      const int32_t live = min(max(int32_t(m) - int32_t(rank * kTokPerRank), 0), kTokPerRank);
      ptx::mbar_arrive_expect_tx(&smem.reduce_done, kSplitK * live * kNTile * 4);
    }
    if (warp_id == kCleanupWarp) {
      ptx::mbar_wait_parity(&smem.tmem_ld_done, 0);
      ptx::tcgen05_dealloc(taddr, kTmemCols);
    }
    PDLTriggerSecondary<true>();  // trigger PDL in epilogue
    if (rank == kSplitK - 1 && q != nullptr) {
      const auto cleanup_tx = tx - kCleanupWarp * 32;
      const auto cleanup_threads = (kWarps - kCleanupWarp) * 32;
      const auto pad_rows = 128u - m;
      for (uint32_t idx = cleanup_tx; idx < 2 * pad_rows; idx += cleanup_threads) {
        const uint32_t blk = idx / pad_rows, r = m + idx % pad_rows;
        scales_out[sf_off(work_id * 2 + blk, r)] = 0u;
      }
    }
    ptx::cluster_wait_acquire();
  }
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

// `rows`/`row_stride_bytes` describe the outer axis; rows >= rows are out of
// bounds and TMA zero-fills them, which is how tokens >= m get masked.
inline CUtensorMap make_map(
    const void* base, uint64_t cols, uint64_t rows, uint64_t row_stride_bytes, uint32_t box_cols, uint32_t box_rows) {
  CUtensorMap map{};
  uint64_t dim[2] = {cols, rows};
  uint64_t stride[1] = {row_stride_bytes};
  uint32_t box[2] = {box_cols, box_rows};
  uint32_t elem_stride[2] = {1, 1};
  // The only driver-API call in the file; there is no runtime-API tensor-map
  // encoder, so the module links `-lcuda` for it.
  cuTensorMapEncodeTiled(
      &map,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      2,
      const_cast<void*>(base),
      dim,
      stride,
      box,
      elem_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return map;
}

}  // namespace sglang::wo_a_fused

namespace sglang {

/**
 * \brief Fused inverse-RoPE + grouped WO-A BF16 GEMM + MXFP8 quantization.
 *
 * \param x         [T, 2, 4096] bf16 attention output; a strided view of a
 *                  [T, 64, 512] buffer, so only the group and K axes need to be
 *                  contiguous and the row stride is read off the tensor.
 * \param w         [2, 1024, 4096] bf16, contiguous.
 * \param freqs     [S, 64] fp32 rotation table, (cos0, sin0, cos1, sin1, ...).
 * \param positions [T] absolute token positions, int32 or int64.
 * Exactly one output form must be supplied: either (\p q, \p scales) or \p y.
 *
 * \param q         [T, 2048] e4m3 quantized output; pairs with \p scales.
 * \param scales    [8192] uint8 UE8M0 scales in FlashInfer's 128x4 swizzle.
 * \param y         [T, 2048] bf16, the result before quantization.
 */
template <int kMaxTokens>
inline void wo_a_fused_run(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView w,
    tvm::ffi::TensorView freqs,
    tvm::ffi::TensorView positions,
    tvm::ffi::Optional<tvm::ffi::TensorView> q,
    tvm::ffi::Optional<tvm::ffi::TensorView> scales,
    tvm::ffi::Optional<tvm::ffi::TensorView> y) {
  using namespace host;
  using namespace wo_a_fused;  // make_map, wo_a_fused_kernel
  using Trait = WoATrait<kMaxTokens>;
  constexpr int kMMax = Trait::kMMax;
  constexpr int kNTile = Trait::kNTile;
  constexpr int kBlockK = Trait::kBlockK;
  constexpr int kGroupK = Trait::kGroupK;
  constexpr int kRopeDim = Trait::kRopeDim;
  constexpr int kNOut = Trait::kNOut;
  constexpr int kScaleBytes = Trait::kScaleBytes;
  constexpr int kClusters = Trait::kClusters;
  constexpr int kSplitK = Trait::kSplitK;
  constexpr int kThreads = Trait::kThreads;
  using Smem = typename Trait::Smem;

  SymbolicSize T = {"num_tokens"};
  // The row stride is captured rather than required: it is 32768 elements when
  // the attention backend pads to 64 heads and 8192 when it does not.
  SymbolicSize XS0 = {"x_row_stride"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({T, 2, kGroupK})  //
      .with_strides({XS0, kGroupK, 1})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(x);
  TensorMatcher({2, kNOut / 2, kGroupK})  //
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(w);
  TensorMatcher({-1, kRopeDim})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(freqs);
  TensorMatcher({T})  //
      .with_dtype<int32_t, int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(positions);
  // Exactly one output form: quantized, or the pre-quantization bf16. Both at
  // once would make the epilogue write twice for a result no caller wants.
  CHECK_HOST(q.has_value() != y.has_value()) << "wo_a_fused: pass either (q, scales) or y, not both and not neither";
  CHECK_HOST(q.has_value() == scales.has_value()) << "wo_a_fused: q and scales come as a pair";
  if (q.has_value()) {
    TensorMatcher({T, kNOut})  //
        .with_dtype<fp8_e4m3_t>()
        .with_device<kDLCUDA>(device_)
        .verify(q.value());
    TensorMatcher({kScaleBytes})  //
        .with_dtype<uint8_t>()
        .with_device<kDLCUDA>(device_)
        .verify(scales.value());
  } else {
    TensorMatcher({T, kNOut})  //
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device_)
        .verify(y.value());
  }

  const int64_t m = T.unwrap();
  const int64_t x_stride0 = XS0.unwrap();
  const DLDevice device = device_.unwrap();
  CHECK_HOST(m >= 1 && m <= kMMax) << "wo_a_fused supports 1 <= T <= " << kMMax << ", got " << m;
  CHECK_HOST(x_stride0 >= 2 * kGroupK) << "x row stride must cover both groups, got " << x_stride0;
  const auto w_map = make_map(w.data_ptr(), kGroupK, kNOut, kGroupK * 2, kBlockK, kNTile);
  const auto x_map = make_map(x.data_ptr(), 2 * kGroupK, m, x_stride0 * 2, kBlockK, kMMax);
  const auto launch = [&](const auto* pos) {
    using PosT = std::decay_t<decltype(*pos)>;
    constexpr auto kernel = wo_a_fused_kernel<Trait, PosT>;
    [[maybe_unused]] static const auto _ = [] {
      CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Smem)));
      CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      return 0;
    }();
    LaunchKernel(kClusters * kSplitK, kThreads, device, sizeof(Smem))
        .config({.use_pdl = true, .cluster_dim = dim3{kSplitK, 1, 1}})(
            kernel,
            w_map,
            x_map,
            static_cast<const fp32_t*>(freqs.data_ptr()),
            pos,
            q.has_value() ? static_cast<fp8_e4m3_t*>(q.value().data_ptr()) : nullptr,
            scales.has_value() ? static_cast<uint8_t*>(scales.value().data_ptr()) : nullptr,
            y.has_value() ? static_cast<bf16_t*>(y.value().data_ptr()) : nullptr,
            static_cast<int32_t>(m));
  };
  if (is_type<int32_t>(positions.dtype())) {
    launch(static_cast<const int32_t*>(positions.data_ptr()));
  } else {
    launch(static_cast<const int64_t*>(positions.data_ptr()));
  }
}

}  // namespace sglang
