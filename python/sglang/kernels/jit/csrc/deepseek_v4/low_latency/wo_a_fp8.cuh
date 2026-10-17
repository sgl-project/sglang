// Low-latency MXFP8 wo_a for DeepSeek-V4.1 decode (SM100): a drop-in for
//   deep_gemm.fp8_einsum("bhr,hdr->bhd", (o_fp8, o_s), (weight, weight_scale), out, recipe=(1, 1, 32))
// at T <= 64, taking the "mega-fp8" activation FlashMLA mega attention emits:
//   o_fp8 [T, G, 4096] e4m3, column c = chunk * 256 + head * 32 + j (interleave_output_columns order);
//   o_s   [T, G, 32] int32, 4 packed UE8M0 bytes per int32, byte b of k4 scales columns [32 (4 k4 + b), +32).
//   weight [G, 1024, 4096] e4m3 in the same column order with per-row-per-32 UE8M0 scales (quantize_wo_a),
//   read as is: the kernel shares DeepGEMM fp8_einsum's weight tensors.
// The MMA is tcgen05 kind::mxf8f6f4.block_scale (e4m3 x e4m3, UE8M0 per 32 K on both operands, fp32
// accumulation), i.e. fp8_einsum's math up to the fp32 accumulation order.
//
// Decomposition (swap-AB + cluster split-K, as wo_b_fp8.cuh): the weight is MMA operand A (M = 128
// output rows), the tokens are N (8 / 16 / 32 / 64, one pass). G * 1024 rows = 8 G row tiles; K = 4096
// splits 8 ways over an 8-CTA cluster by contiguous 512-column slabs (2 chunks x 8 heads), so a CTA's
// activation is 512 contiguous bytes per token plus 4 contiguous int32 scale rows. 64 G CTAs of 256
// threads, two per SM, each owning a 128 x 512 weight slab (64 KiB).
//
// Before griddepcontrol.wait warps 4-7 stream the slab and its scales into smem with LDGSTS + L2
// evict_first (a cp.async.bulk / TMA issued before the wait completes only after the primary grid does)
// and copy both into TMEM, so the MMAs do not re-read 64 KiB of A from smem and the slab's smem becomes
// the reduce inbox. After the wait only the activation is fetched: 4 TMA boxes from one lane of warp 0
// (straight into the SW128 layout, rows past T zero-filled), and its scale cells by warp 1, which also copies
// them into TMEM. The A copies, the TMA and the B-scale copies all complete one mbarrier, the only one one
// elected lane of warp 2 waits on before issuing the 16 MMAs. The 8 K partials meet over DSMEM (N = 8: st.async
// of the live tokens; otherwise one bulk copy per (source, owner)): cluster rank r owns rows [16 r, 16 r + 16)
// of the tile and stores them as bf16.
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

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>

namespace sglang::device::wo_a_mxf8 {

namespace ptx {

using device::ptx::mbar_arrive;
using device::ptx::mbar_arrive_expect_tx;
using device::ptx::mbar_init;
using device::ptx::to_shared;

// The inbox wait orders the smem reads of what peers st.async'd / bulk-copied in, hence a local copy of
// device::ptx's with a "memory" clobber.
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
// L2 evict_first: the 8 MiB weight stream otherwise evicts the co-running attention's working set
// (measured: +0.9 us on the v22 attention at b=1 without the hint, +0.4 us with it).
SGL_DEVICE void cp_async_16B(uint32_t dst, const void* src, uint64_t policy) {
#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "l"(policy)
               : "memory");
#else
  // nvcc 13.0's ptxas emits an illegal instruction for the hinted form here (measured on B200).
  (void)policy;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
#endif
}
SGL_DEVICE void cp_async_4B(uint32_t dst, const void* src, uint64_t policy) {
#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)
  asm volatile("cp.async.ca.shared.global.L2::cache_hint [%0], [%1], 4, %2;" ::"r"(dst), "l"(src), "l"(policy)
               : "memory");
#else
  (void)policy;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src) : "memory");
#endif
}
SGL_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
SGL_DEVICE void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;" ::: "memory");
}
SGL_DEVICE void prefetch_tensormap(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory");
}
// One 3D tensor-map box at coords (x, y, z) into smem, completing its bytes on `bar`.
SGL_DEVICE void tma_load_3d(void* dst, const CUtensorMap* map, int32_t x, int32_t y, int32_t z, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], "
      "[%5];" ::"r"(to_shared(dst)),
      "l"(map),
      "r"(x),
      "r"(y),
      "r"(z),
      "r"(to_shared(bar))
      : "memory");
}

// 4 B of a register to `dst` in a peer CTA, completing its bytes on the peer mbarrier `bar`.
SGL_DEVICE void st_async_4B(uint32_t dst, float v, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(dst),
               "r"(__float_as_uint(v)),
               "r"(bar)
               : "memory");
}
SGL_DEVICE void cp_async_bulk_s2cluster(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
      "r"(to_shared(src)),
      "r"(bytes),
      "r"(bar)
      : "memory");
}
SGL_DEVICE void bulk_commit() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
SGL_DEVICE void bulk_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
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
SGL_DEVICE void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
SGL_DEVICE void bar_sync(uint32_t id, uint32_t threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

}  // namespace ptx

using tcgen05 = device::ptx::TCGen05<1>;

/**
 * \brief Compile-time shape.
 * \tparam N_ tokens per CTA, the MMA N (8 / 16 / 32 / 64); T <= N_.
 */
template <int N_>
struct Trait {
  static constexpr int N = N_;
  static_assert(N == 8 || N == 16 || N == 32 || N == 64);
  static constexpr int M = 1024;  // output rows per group
  static constexpr int K = 4096;
  static constexpr int M_BLOCK = 128;
  static constexpr int kTilesPerGroup = M / M_BLOCK;
  static constexpr int kSplitK = 8;            // == cluster size
  static constexpr int K_BLOCK = K / kSplitK;  // 512
  static constexpr int kSub = K_BLOCK / 128;   // 128-byte K subtiles == int32 scale words
  static constexpr int kKB = K_BLOCK / 32;
  static constexpr int kThreads = 256;
  static constexpr int kRoleThreads = 128;
  static constexpr int kWSubBytes = M_BLOCK * 128;
  static constexpr int kWBytes = kSub * kWSubBytes;  // 64 KiB
  static constexpr int kXSubBytes = N * 128;
  static constexpr int kSFBytes = kSub * 512;  // [subtile][lane][quadrant] words, tcgen05.cp order
  static constexpr int kHalf = N / 2;          // TMEM columns per epilogue warp == tokens per owner
  // Split-K reduce: N = 8 st.asyncs the live tokens only; wider tiles send one bulk copy per (source, owner).
  static constexpr bool kSparseReduce = N <= 8;
  // Owner o holds rows [32 (o % 4), +32) x tokens [kHalf (o / 4), +kHalf): warp o's accumulator block.
  // Its inbox is [source rank][token][32 rows] fp32, in the slab's smem; so is the bulk staging, [warp][token][row].
  static constexpr int kInboxFloats = kSplitK * 32 * kHalf;
  static constexpr int kInboxBytes = kInboxFloats * 4;
  static_assert(2 * kInboxBytes <= kWBytes, "the inbox and the bulk staging share the slab's smem");
  // TMEM, one allocation (tcgen05.alloc is slow): accumulator, A scales, B scales, then A (4 e4m3 per column).
  static constexpr uint32_t kTmemAcc = 0;
  static constexpr uint32_t kTmemSFA = 64;
  static constexpr uint32_t kTmemSFB = kTmemSFA + 4 * kSub;
  static constexpr uint32_t kTmemA = 128;
  static constexpr uint32_t kTmemCols = 256;
  static_assert(kTmemAcc + N <= kTmemSFA && kTmemSFB + 4 * kSub <= kTmemA && kTmemA + K_BLOCK / 4 <= kTmemCols);

  struct alignas(1024) Smem {
    union {
      alignas(1024) uint8_t w[kWBytes];  // the weight slab, until it is in TMEM
      float inbox[kWBytes / 4];          // then the reduce inbox, and the bulk staging in its upper half
    };
    alignas(1024) uint8_t x[kSub * kXSubBytes];
    alignas(128) uint32_t sfa[kSFBytes / 4];
    alignas(128) uint32_t sfb[kSFBytes / 4];
    // The MMA's operands: the A slab + A scales (tcgen05.cp), the B tile (TMA) and the B scales (tcgen05.cp).
    uint64_t mma_ready;
    uint64_t mma_done;
    uint64_t red_ready;
    uint64_t a_loaded;
    uint64_t drained;
    uint32_t taddr;
  };
  // Two CTAs per SM: 2 x (smem + 1 KiB reserve) <= 228 KiB.
  static_assert(sizeof(Smem) <= 115712, "smem would drop to 1 CTA/SM");
};

struct Params {
  CUtensorMap x_map;  // o_fp8 [T, G, 4096] e4m3 as 3D (k, g, t), box (128, 1, N), 128 B swizzle
  const int32_t* xs;  // o_s [T, G, 32] int32, strides (s_st, s_sg, s_sk) elements
  const uint8_t* w;   // weight [G, 1024, 4096] e4m3, contiguous (DeepGEMM fp8_einsum's operand)
  const int32_t* ws;  // its scales [G, 1024, 32] int32 (4 UE8M0 per 128 columns), strides (ws_sg, ws_sr, ws_sk)
  // Output, act_quant(bf16(out)) as wo_b takes it: e4m3 [T, G * 1024] (row stride o_st bytes) and its UE8M0
  // per-32 scales in FlashInfer's 128x4 swizzled layout (rows = tokens, padded to 128).
  fp8_e4m3_t* out;
  uint8_t* out_scale;
  int64_t ws_sg, ws_sr, ws_sk;
  int64_t s_st, s_sg, s_sk;
  int64_t o_st;
  uint32_t num_tokens;
};

template <int N>
__global__ void __launch_bounds__(Trait<N>::kThreads, 2) wo_a_mxf8_ll_kernel(const __grid_constant__ Params p) {
  using T = Trait<N>;
  constexpr int kSub = T::kSub;
  constexpr int kSplitK = T::kSplitK;
  constexpr int kHalf = T::kHalf;
  constexpr bool kSparse = T::kSparseReduce;
  constexpr uint32_t kRole = T::kRoleThreads;
  extern __shared__ __align__(1024) uint8_t s_buf[];
  auto& smem = *reinterpret_cast<typename T::Smem*>(s_buf);

  // Warps 0-3: activation (after the PDL wait), MMA (warp 2) and the owner reduce. Warps 4-7: the weight
  // (stream, TMEM copies from warp 4).
  const auto tid = threadIdx.x;
  const auto warp_id = warp::broadcast(tid / 32, 0);
  const auto lane_id = warp::get_lane_id();
  const auto rank = ptx::cluster_ctarank();
  const auto tile = blockIdx.x / static_cast<uint32_t>(kSplitK);
  const auto group = tile / T::kTilesPerGroup;
  const auto row0 = (tile % T::kTilesPerGroup) * T::M_BLOCK;

  static_assert(kRole == 4 * 32);
  // Warps 0-3: activation (after the PDL wait), MMA (warp 2) and the owner reduce. Warps 4-7: the weight. The
  // three warps busy after the wait sit on different SMSPs (warp w runs on scheduler w % 4).
  enum Role { WARP_X = 0, WARP_SFB = 1, WARP_MMA = 2, WARP_WEIGHT = 4, WARP_TMEM_FREE = T::kThreads / 32 - 1 };
  enum NamedBarrier { BAR_WEIGHT = 1 };
  const auto num_tokens = p.num_tokens;

  // ---- init: the weight loads first (they depend on nothing), mbarriers, TMEM ----
  if (warp_id >= WARP_WEIGHT) {
    // The slab and its scales, read in DeepGEMM's layout: the 128 B swizzle and the tcgen05.cp scale order
    // are in the smem addresses.
    const auto wt = tid - kRole;
    const auto policy = ptx::policy_evict_first();
    // Weight: 128 rows x 32 chunks of 16 B; a warp reads one row's 512 contiguous bytes, so thread wt always
    // takes chunk wt % 32, of rows wt / 32 + 4 i.
    const auto chunk = wt % 32, row_base = wt / 32;
    const auto w_src = p.w + (size_t(group) * T::M + row0 + row_base) * T::K + rank * T::K_BLOCK + chunk * 16;
    // sw128_off(row_base + 4 i, chunk % 8) with row_base < 4: only i's parity reaches the swizzle.
    const auto w_dst = ptx::to_shared(&smem.w[(chunk / 8) * T::kWSubBytes]);
    const uint32_t w_dst_even = w_dst + tcgen05::sw128_off(row_base, chunk % 8);
    const uint32_t w_dst_odd = w_dst + tcgen05::sw128_off(row_base + 4, chunk % 8);
#pragma unroll
    for (uint32_t i = 0; i < T::M_BLOCK / 4; ++i) {
      ptx::cp_async_16B((i % 2 ? w_dst_odd : w_dst_even) + (i / 2) * 1024, w_src + i * 4 * T::K, policy);
    }
    // Scales: thread wt reads row wt's word k4 = 4 rank + s into word (s, lane wt % 32, quadrant wt / 32).
    const auto ws_src = p.ws + group * p.ws_sg + (row0 + wt) * p.ws_sr + rank * kSub * p.ws_sk;
#pragma unroll
    for (uint32_t s = 0; s < kSub; ++s) {
      ptx::cp_async_4B(ptx::to_shared(&smem.sfa[s * 128 + (wt % 32) * 4 + wt / 32]), ws_src + s * p.ws_sk, policy);
    }
    ptx::cp_async_commit();
  } else if (warp_id == WARP_X) {
    if (warp::elect_one_lane()) {
      ptx::prefetch_tensormap(&p.x_map);
      ptx::mbar_init(&smem.mma_ready, 3);  // weight warp's commit, TMA (expect_tx), x-scale warp's commit
      ptx::mbar_init(&smem.a_loaded, 1);
      ptx::mbar_init(&smem.mma_done, 1);
      ptx::mbar_init(&smem.red_ready, 1);
      ptx::mbar_init(&smem.drained, T::kThreads / 32);
      ptx::fence_mbarrier_init_release_cluster();
      // Every rank pushes its 32 x kHalf partials of our block (sparse: only the live tokens).
      const auto live = min(kHalf, max(int(num_tokens) - int(rank / 4) * kHalf, 0));
      ptx::mbar_arrive_expect_tx(&smem.red_ready, kSparse ? kSplitK * 32 * 4 * live : T::kInboxBytes);
    }
  } else if (warp_id == WARP_SFB) {
    tcgen05::alloc(&smem.taddr, T::kTmemCols);
    // Without this the SM's second CTA could never allocate.
    tcgen05::relinquish_alloc_permit();
    tcgen05::fence_before_thread_sync();
  } else {
    // Warps 2-3 have no init work: they release the dependents, so the next kernel can prefetch its weight too.
    PDLTriggerSecondary<true>();
  }
  __syncthreads();
  const auto taddr = smem.taddr;

  if (warp_id >= WARP_WEIGHT) {
    // ---- weight -> TMEM (warp 4), then the cluster arrive ------------------
    ptx::cp_async_wait_all();
    ptx::fence_proxy_async_shared();  // cp.async wrote in the generic proxy, tcgen05.cp reads in the async one
    ptx::bar_sync(BAR_WEIGHT, kRole);
    if (warp_id == WARP_WEIGHT) {
      if (warp::elect_one_lane()) {
        tcgen05::fence_after_thread_sync();
        const auto sfa = tcgen05::make_desc_sf_chunk(smem.sfa);
#pragma unroll
        for (uint32_t s = 0; s < kSub; ++s) {
          tcgen05::cp_32x128b_warpx4(taddr + T::kTmemSFA + 4 * s, sfa.advance_bytes(s * 512).bits);
        }
        // A in 32-K blocks: block b is 32 B into subtile b / 4 of the swizzled slab.
        const auto a = tcgen05::make_desc_kmajor_sw128(smem.w);
#pragma unroll
        for (uint32_t b = 0; b < T::kKB; ++b) {
          tcgen05::cp_128x256b(taddr + T::kTmemA + 8 * b, a.advance_bytes((b / 4) * T::kWSubBytes + 32 * (b % 4)).bits);
        }
        tcgen05::commit(&smem.a_loaded);   // for the cluster arrive below
        tcgen05::commit(&smem.mma_ready);  // for the MMA
      }
    }
    // The inbox reuses the slab's smem: arrive only once it is in TMEM.
    ptx::mbar_wait_parity(&smem.a_loaded, 0);
  }
  ptx::cluster_arrive_relaxed();

  if (warp_id == WARP_X) {
    // ---- activation: the B tile by TMA ----------------------------------
    if (warp::elect_one_lane()) {
      PDLWaitPrimary<true>();
      // Subtile s is the 128 K columns at rank * 512 + 128 s for the N tokens (rows past T are zeros),
      // landing in the SW128 K-major layout the MMA reads.
      ptx::mbar_arrive_expect_tx(&smem.mma_ready, kSub * T::kXSubBytes);
#pragma unroll
      for (uint32_t s = 0; s < kSub; ++s) {
        ptx::tma_load_3d(&smem.x[s * T::kXSubBytes], &p.x_map, rank * T::K_BLOCK + 128 * s, group, 0, &smem.mma_ready);
      }
    }
  } else if (warp_id == WARP_SFB) {
    // ---- activation scales ----------------------------------------------
    PDLWaitPrimary<true>();
    // Scale cell of (token t, subtile s): int32 k4 = 4 rank + s holds the 4 block exponents in MMA byte order.
    constexpr uint32_t kIters = (kSub * N + 31) / 32;
    int32_t sv[kIters];
#pragma unroll
    for (uint32_t it = 0; it < kIters; ++it) {
      const auto i = lane_id + 32 * it;
      const auto s = i / N, t = i % N;
      sv[it] = 0;
      if (s < kSub && t < num_tokens) sv[it] = p.xs[group * p.s_sg + (rank * kSub + s) * p.s_sk + t * p.s_st];
    }
#pragma unroll
    for (uint32_t it = 0; it < kIters; ++it) {
      const auto i = lane_id + 32 * it;
      const auto s = i / N, t = i % N;
      if (s < kSub) smem.sfb[s * 128 + (t % 32) * 4 + t / 32] = sv[it];
    }
    ptx::fence_proxy_async_shared();  // the tcgen05.cp reading them is in the async proxy
    __syncwarp();
    if (warp::elect_one_lane()) {
      tcgen05::fence_after_thread_sync();
      const auto sfb = tcgen05::make_desc_sf_chunk(smem.sfb);
#pragma unroll
      for (uint32_t s = 0; s < kSub; ++s) {
        tcgen05::cp_32x128b_warpx4(taddr + T::kTmemSFB + 4 * s, sfb.advance_bytes(s * 512).bits);
      }
      tcgen05::commit(&smem.mma_ready);
    }
  } else if (warp_id == WARP_MMA) {
    if (warp::elect_one_lane()) {
      // ---- MMA: one lane issues everything; the others go straight to the mma_done wait ----
      const auto b = tcgen05::make_desc_kmajor_sw128(smem.x);
      const auto idesc = tcgen05::make_instr_desc_mxf8<T::M_BLOCK, N>();
      ptx::mbar_wait_parity(&smem.mma_ready, 0);
      tcgen05::fence_after_thread_sync();
      // K block (s, kb): A is TMEM column 8 (4 s + kb), B is 32 B into subtile s, the scales are byte kb of the
      // subtile's TMEM scale columns.
#pragma unroll
      for (uint32_t s = 0; s < kSub; ++s) {
#pragma unroll
        for (uint32_t kb = 0; kb < 4; ++kb) {
          tcgen05::mma_mxf8_ts(
              taddr + T::kTmemAcc,
              taddr + T::kTmemA + 8 * (4 * s + kb),
              b.advance_bytes(s * T::kXSubBytes + 32 * kb).bits,
              idesc.with_sf_id(kb).bits,
              taddr + T::kTmemSFA + 4 * s,
              taddr + T::kTmemSFB + 4 * s,
              s | kb);
        }
      }
      tcgen05::commit(&smem.mma_done);
    }
  }

  // Paired with the cluster arrives above, while the MMAs run: the peers' mbarriers are initialized and their
  // slabs are in TMEM (the inbox reuses that smem). The acquire invalidates L1, so it stays off the path after
  // mma_done.
  ptx::cluster_wait_acquire();

  // ---- TMEM -> registers: warp w reads lane quadrant w % 4, column half w / 4
  ptx::mbar_wait_parity(&smem.mma_done, 0);
  tcgen05::fence_after_thread_sync();
  const auto quad = warp_id % 4, half = warp_id / 4;
  float acc[kHalf];
  tcgen05::ld_32x32b<kHalf>(taddr + ((quad * 32) << 16) + T::kTmemAcc + half * kHalf, acc);
  tcgen05::wait_ld();
  tcgen05::fence_before_thread_sync();
  __syncwarp();
  if (lane_id == 0) ptx::mbar_arrive(&smem.drained);

  // ---- split-K partials -> the block owner: warp w's block (quadrant w % 4, token half w / 4) is owner w's ----
  // Inbox and staging are [source][token][row]: lanes are rows, so every store and load is one 128 B row.
  const auto owner = warp_id;
  const auto bar = ptx::mapa(ptx::to_shared(&smem.red_ready), owner);
  const auto dst = ptx::mapa(ptx::to_shared(&smem.inbox[rank * kHalf * 32]), owner);
  if constexpr (kSparse) {
    // Each lane sends its row's value of every live token.
#pragma unroll
    for (uint32_t i = 0; i < kHalf; ++i) {
      if (half * kHalf + i >= num_tokens) break;
      ptx::st_async_4B(dst + (i * 32 + lane_id) * 4, acc[i], bar);
    }
  } else {
    // The block is staged and sent as one bulk copy, so each owner mbarrier sees 8 complete_tx.
    const auto stage = &smem.inbox[T::kInboxFloats + warp_id * kHalf * 32];
#pragma unroll
    for (uint32_t i = 0; i < kHalf; ++i)
      stage[i * 32 + lane_id] = acc[i];
    ptx::fence_proxy_async_shared();
    __syncwarp();
    if (warp::elect_one_lane()) {
      ptx::cp_async_bulk_s2cluster(dst, stage, kHalf * 32 * 4, bar);
      ptx::bulk_commit();
    }
  }

  // ---- owner: sum the 8 K slabs in a fixed order, round to bf16, quantize per 32 rows ----
  // A token's 32 rows are one act_quant group: lanes are rows, warp w takes tokens w, w + 8, ...
  {
    const auto token0 = (rank / 4) * kHalf;
    const auto col = group * T::M + row0 + (rank % 4) * 32;  // of the [T, G * 1024] output
    const auto live = min(kHalf, max(int(num_tokens) - int(token0), 0));
    if (int(warp_id) < live) {
      ptx::mbar_wait_parity(&smem.red_ready, 0);
      for (uint32_t i = warp_id; int(i) < live; i += T::kThreads / 32) {
        float sum = smem.inbox[i * 32 + lane_id];
#pragma unroll
        for (uint32_t src = 1; src < kSplitK; ++src)
          sum += smem.inbox[(src * kHalf + i) * 32 + lane_id];
        // The reference's act_quant of the bf16 output: amax clamped at 1e-4, scale 2^ceil(log2(amax / 448)).
        const float v = cast<fp32_t>(cast<bf16_t>(sum));
        const float amax = fmaxf(warp::reduce_max(fabsf(v)), 1e-4f);
        const int32_t e = deepseek_v4::fp8::cast_to_ue8m0(amax * (1.0f / 448.0f));
        const auto t = token0 + i;
        // fp8_e4m3 cast will automatically clip to FP8 range except for NAN
        p.out[t * p.o_st + col + lane_id] = fp8_e4m3_t{v * deepseek_v4::fp8::inv_scale_ue8m0(e)};
        if (lane_id == 0) {
          const auto g = col / 32;  // FlashInfer 128x4: (k4 tile, row % 32, row / 32 % 4, k % 4), rows = tokens
          p.out_scale[(g / 4) * 512 + ((t % 32) * 4 + (t / 32) % 4) * 4 + g % 4] = uint8_t(e);
        }
      }
    }
  }
  if constexpr (!kSparse) ptx::bulk_wait_read();
  // Every CTA owns rows, so it lives until all pushes into its inbox landed (the wait above, by warp 0).
  // The last warp frees TMEM: it is the one least likely to be an owner above.
  if (warp_id == WARP_TMEM_FREE) {
    ptx::mbar_wait_parity(&smem.drained, 0);
    tcgen05::fence_after_thread_sync();
    tcgen05::dealloc(taddr, T::kTmemCols);
  }
}

}  // namespace sglang::device::wo_a_mxf8

namespace sglang {

/// o_fp8 [T, G, 4096] as a 3D (k, g, t) e4m3 tensor map; box (128 B, 1, box_rows), 128 B swizzle.
inline CUtensorMap make_wo_a_x_map(tvm::ffi::TensorView x, uint32_t box_rows) {
  using namespace host;
  // Through the runtime's driver entry point, so the module needs no -lcuda.
  static const auto encode = [] {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult status;
    CHECK_CUDA(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &status));
    CHECK_HOST(status == cudaDriverEntryPointSuccess && fn != nullptr);
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
  }();
  const cuuint64_t dims[3] = {cuuint64_t(x.size(2)), cuuint64_t(x.size(1)), cuuint64_t(x.size(0))};
  const cuuint64_t strides[2] = {cuuint64_t(x.stride(1)), cuuint64_t(x.stride(0))};  // bytes
  const cuuint32_t box[3] = {128, 1, box_rows};
  const cuuint32_t elem_strides[3] = {1, 1, 1};
  CUtensorMap map;
  const auto res = encode(
      &map,
      CU_TENSOR_MAP_DATA_TYPE_UINT8,
      3,
      x.data_ptr(),
      dims,
      strides,
      box,
      elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CHECK_HOST(res == CUDA_SUCCESS) << int(res);
  return map;
}

/**
 * \brief Low-latency MXFP8 wo_a: W[g] (e4m3, UE8M0/32) x o[t, g, :] (e4m3, UE8M0/32), rounded to bf16 and
 *        quantized per 32 columns for wo_b (the reference's act_quant, which wo_b applies to this output).
 *
 * Launched as a PDL secondary that releases its own dependents at start.
 * \tparam N    MMA N (8 / 16 / 32 / 64); T must be <= N.
 * \param x      o_fp8 [T, G, 4096] e4m3; strides (st, sg, 1) multiples of 16.
 * \param xs     o_s [T, G, 32] int32, any strides (mega-fp8: (1, 32 T_aligned, T_aligned)).
 * \param w      weight [G, 1024, 4096] e4m3, contiguous (quantize_wo_a's output,
 *               the operand DeepGEMM fp8_einsum takes).
 * \param ws     its scales [G, 1024, 32] int32, 4 packed UE8M0 per 128 columns, any strides.
 * \param out    [T, G * 1024] e4m3, row stride a multiple of 16.
 * \param out_scale its UE8M0 scales in FlashInfer's 128x4 swizzled layout, >= 128 * G * 32 bytes (rows past T
 *               are left untouched).
 */
template <int N>
inline void wo_a_fp8_mxf8_ll(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView xs,
    tvm::ffi::TensorView w,
    tvm::ffi::TensorView ws,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView out_scale) {
  using namespace host;
  using T = device::wo_a_mxf8::Trait<N>;
  auto num_tokens_ = SymbolicSize{"num_tokens"};
  auto groups_ = SymbolicSize{"num_groups"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();
  TensorMatcher({num_tokens_, groups_, T::K})  //
      .with_strides({-1, -1, 1})
      .with_dtype<fp8_e4m3_t>()
      .with_device(device_)
      .ensure_alignment(16)
      .verify(x);
  TensorMatcher({num_tokens_, groups_, T::K / 128})  //
      .with_strides({-1, -1, -1})
      .with_dtype<int32_t>()
      .with_device(device_)
      .verify(xs);
  const int64_t w_row_stride = T::K, w_group_stride = T::M * T::K;  // contiguous
  TensorMatcher({groups_, T::M, T::K})                              //
      .with_strides({w_group_stride, w_row_stride, 1})
      .with_dtype<fp8_e4m3_t>()
      .with_device(device_)
      .ensure_alignment(16)
      .verify(w);
  TensorMatcher({groups_, T::M, T::K / 128})  //
      .with_strides({-1, -1, -1})
      .with_dtype<int32_t>()
      .with_device(device_)
      .verify(ws);
  TensorMatcher({num_tokens_, -1})  //
      .with_strides({-1, 1})
      .with_dtype<fp8_e4m3_t>()
      .with_device(device_)
      .ensure_alignment(16)
      .verify(out);
  TensorMatcher({-1})  //
      .with_dtype<uint8_t>()
      .with_device(device_)
      .verify(out_scale);
  const auto num_tokens = num_tokens_.unwrap();
  const auto num_ctas = groups_.unwrap() * T::kTilesPerGroup * T::kSplitK;
  CHECK_HOST(num_tokens <= N) << num_tokens;
  CHECK_HOST(out.size(1) == groups_.unwrap() * T::M) << out.size(1) << " " << groups_.unwrap();
  // FlashInfer's 128x4 image: 128 padded token rows x G * 1024 / 32 columns.
  CHECK_HOST(out_scale.size(0) >= 128 * out.size(1) / 32) << out_scale.size(0) << " " << out.size(1);

  if (num_tokens == 0) return;  // an empty reduce would wait forever on its inbox

  device::wo_a_mxf8::Params p{};
  p.x_map = make_wo_a_x_map(x, N);
  p.xs = static_cast<const int32_t*>(xs.data_ptr());
  p.w = static_cast<const uint8_t*>(w.data_ptr());
  p.ws = static_cast<const int32_t*>(ws.data_ptr());
  p.ws_sg = ws.stride(0);
  p.ws_sr = ws.stride(1);
  p.ws_sk = ws.stride(2);
  p.out = static_cast<fp8_e4m3_t*>(out.data_ptr());
  p.out_scale = static_cast<uint8_t*>(out_scale.data_ptr());
  p.s_st = xs.stride(0);
  p.s_sg = xs.stride(1);
  p.s_sk = xs.stride(2);
  p.o_st = out.stride(0);
  p.num_tokens = static_cast<uint32_t>(num_tokens);

  const auto device = device_.unwrap();
  constexpr auto kernel = device::wo_a_mxf8::wo_a_mxf8_ll_kernel<N>;
  constexpr size_t kSmemBytes = sizeof(typename T::Smem);
  runtime::init_per_device(device, [] {
    runtime::set_max_dynamic_smem(kernel, kSmemBytes);
    runtime::set_smem_carveout(kernel, cudaSharedmemCarveoutMaxShared);
  });
  LaunchKernel(static_cast<uint32_t>(num_ctas), T::kThreads, device, kSmemBytes)
      .config({.use_pdl = true, .cluster_dim = dim3{T::kSplitK, 1, 1}})(kernel, p);
}

}  // namespace sglang
