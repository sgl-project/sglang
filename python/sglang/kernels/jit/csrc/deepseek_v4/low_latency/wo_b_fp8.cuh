// Low-latency FP8 wo_b GEMM for DeepSeek-V4.1-Flash decode (SM100), publishing its rows into a Lamport all-reduce
// mailbox.
//
//   wo_b (TP4 rank): out[T, 5120] = x[T, 2048] @ W[5120, 2048]^T, T <= 64
//   x: e4m3 + UE8M0 per 32 in FlashInfer's 128x4 swizzled layout (what wo_a_fp8_mxf8_ll emits: the reference's
//   act_quant of its output); W: e4m3 with UE8M0 per row per 32 in the same layout (sglang's plain weight and
//   weight_scale_inv_swizzled, the operands FlashInfer's mm_mxfp8 takes). The MMA is tcgen05
//   kind::mxf8f6f4.block_scale, i.e. the reference fp8_gemm: FP32 accumulation of each 32-wide K block scaled by
//   s_x * s_w.
//
// Decomposition (FlashInfer's cute-dsl split-K, swap-AB): the weight is MMA operand A (M = 128 output rows), the
// tokens are N (8 / 16 / 32 / 64, one pass). 5120 rows = 40 row tiles; K = 2048 splits 4 ways over a 4-CTA
// cluster, so the grid is 160 CTAs of 256 threads, two per SM, each owning a 128 x 512 weight slab (64 KiB).
//
// Before griddepcontrol.wait warps 4-7 stream the slab and its scales into smem with LDGSTS + L2 evict_first (a
// cp.async.bulk / TMA issued before the wait completes only after the primary grid does), one cp.async group per
// 128-K subtile, and copy each subtile into TMEM as soon as it lands, so the MMAs read A from TMEM and the slab's
// smem becomes the reduce inbox. After the wait one lane of warp 0
// loads the activation by 4 TMA boxes (straight into the SW128 layout, rows past T zero-filled) and its scales by
// one 2 KiB bulk copy, completing the same mbarrier as the A copies; one lane of warp 2 then copies the B scales
// into TMEM and issues the 16 MMAs. The 4 K partials meet over DSMEM (N = 8: st.async of the live tokens;
// otherwise one bulk copy per warp): cluster rank r owns rows [32 r, 32 r + 32), exactly TMEM lane quadrant r, and
// publishes them as BF16. With kWorldSize > 0 the rows are also pushed into slot `rank` of every rank's push-plane
// workspace (CustomAllReduceV2's lamport protocol, see sgl_kernel/distributed/communicator.cuh), which
// low_latency/all_reduce_mhc.cuh consumes.
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

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>

namespace sglang::device::wo_b {

namespace ptx {

using device::ptx::mbar_arrive;
using device::ptx::mbar_arrive_expect_tx;
using device::ptx::mbar_init;
using device::ptx::st_relaxed_16B;
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
SGL_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
// Waits until at most `pending` (a constant after unrolling, 0..3) of this thread's cp.async groups are in flight.
SGL_DEVICE void cp_async_wait_pending(uint32_t pending) {
  if (pending == 0) {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
  } else if (pending == 1) {
    asm volatile("cp.async.wait_group 1;" ::: "memory");
  } else if (pending == 2) {
    asm volatile("cp.async.wait_group 2;" ::: "memory");
  } else {
    asm volatile("cp.async.wait_group 3;" ::: "memory");
  }
}
SGL_DEVICE void prefetch_tensormap(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory");
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
// `bytes` (a multiple of 16) of contiguous global memory into smem, completing them on `bar`.
SGL_DEVICE void cp_async_bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(to_shared(dst)),
      "l"(src),
      "r"(bytes),
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
// A push-plane phase counter, read at system scope: never a stale L1 line from an earlier kernel on this SM.
SGL_DEVICE uint32_t load_counter(const distributed::Counter* counter) {
  return device::ptx::load_relaxed_sys(reinterpret_cast<const uint32_t*>(counter));
}
// One 16 B store to the same offset of every rank's buffer behind a multicast address, relaxed at system scope so a
// peer's relaxed poll sees it (device::ptx's st_multimem_16B is weak).
template <typename V>
SGL_DEVICE void st_multimem_relaxed_16B(const V& x, void* mc_addr, int64_t vec_offset) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  const auto r = reinterpret_cast<const uint32_t*>(&x);
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%4], {%0, %1, %2, %3};" ::"r"(r[0]),
               "r"(r[1]),
               "r"(r[2]),
               "r"(r[3]),
               "l"(static_cast<uint8_t*>(mc_addr) + vec_offset * 16)
               : "memory");
}
SGL_DEVICE void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
SGL_DEVICE void barrier_sync(uint32_t id, uint32_t threads) {
  asm volatile("barrier.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

}  // namespace ptx

using tcgen05 = device::ptx::TCGen05<1>;

/**
 * \brief Compile-time shape of the wo_b producer.
 * \tparam N_ tokens per CTA, the MMA N (8 / 16 / 32 / 64); T <= N_.
 */
template <int N_>
struct Trait {
  static constexpr int N = N_;
  static_assert(N == 8 || N == 16 || N == 32 || N == 64);
  static constexpr int M = 5120;  // hidden
  static constexpr int K = 2048;  // 2 local o_groups x o_lora_rank
  static constexpr int M_BLOCK = 128;
  static constexpr int kSplitK = 4;            // == cluster size == TMEM lane quadrants
  static constexpr int K_BLOCK = K / kSplitK;  // 512
  static constexpr int kSub = K_BLOCK / 128;   // 128-byte K subtiles == 4-block scale groups
  static constexpr int kKB = K_BLOCK / 32;
  static constexpr int kNumCtas = M / M_BLOCK * kSplitK;  // 160
  static constexpr int kThreads = 256;
  static constexpr int kRoleThreads = 128;
  static constexpr int kWSubBytes = M_BLOCK * 128;
  static constexpr int kWBytes = kSub * kWSubBytes;  // 64 KiB
  static constexpr int kXSubBytes = N * 128;
  // A CTA's slab of a FlashInfer 128x4 scale image: 4 contiguous 512 B tiles (k4 tile, row % 32, row / 32 % 4,
  // k % 4), which is also tcgen05.cp.32x128b.warpx4's order.
  static constexpr int kSFBytes = kSub * 512;
  static constexpr int kHalf = N / 2;  // TMEM columns per epilogue warp
  // Split-K reduce: N = 8 st.asyncs the live tokens only; wider tiles send one bulk copy per warp.
  static constexpr bool kSparseReduce = N <= 8;
  // Rank r owns rows [32 r, +32) of the tile (TMEM lane quadrant r), from warps r and r + 4. Its inbox is
  // [source rank][token][32 rows] fp32, in the slab's smem; so is the bulk staging, [warp][token][row].
  static constexpr int kInboxFloats = kSplitK * N * 32;
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
      float inbox[kWBytes / 4];          // then the reduce inbox, and the bulk staging after it
    };
    alignas(1024) uint8_t x[kSub * kXSubBytes];  // the activation tile, SW128 K-major
    alignas(128) uint32_t sfa[kSFBytes / 4];
    alignas(128) uint32_t sfb[kSFBytes / 4];
    uint64_t mma_ready;  // A + A scales in TMEM (tcgen05.commit), x tile + x scales in smem (TMA / bulk copy)
    uint64_t a_loaded;   // A in TMEM: the slab's smem is free for the inbox
    uint64_t mma_done;
    uint64_t red_ready;
    uint64_t drained;
    uint32_t taddr;
  };
  // Two CTAs per SM: 2 x (smem + 1 KiB reserve) <= 228 KiB.
  static_assert(sizeof(Smem) <= 115712, "smem would drop to 1 CTA/SM");
};

template <uint32_t kWorldSize>
struct Params {
  CUtensorMap x_map;       // x [T, 2048] e4m3 as 2D (k, t), box (128, N), 128 B swizzle
  const uint8_t* x_scale;  // its UE8M0 scales, FlashInfer 128x4 image (128 padded rows)
  const uint8_t* w;        // weight [5120, 2048] e4m3, contiguous
  const uint8_t* w_scale;  // its UE8M0 per-32 scales, FlashInfer 128x4 image
  bf16_t* out;             // kWorldSize == 0: [T, 5120] bf16
  uint32_t num_tokens;
  uint32_t rank;                              // kWorldSize > 0: this rank in the push plane
  distributed::PushWorkSpace<kWorldSize> ws;  // kWorldSize > 0: slots of [T, 5120] bf16
};

template <int N, uint32_t kWorldSize, bool kMulticast>
__global__ void __launch_bounds__(Trait<N>::kThreads, 2) wo_b_ll_kernel(const __grid_constant__ Params<kWorldSize> p) {
  using T = Trait<N>;
  constexpr int kSub = T::kSub;
  constexpr int kSplitK = T::kSplitK;
  constexpr int kHalf = T::kHalf;
  constexpr bool kSparse = T::kSparseReduce;
  constexpr uint32_t kRole = T::kRoleThreads;
  extern __shared__ __align__(1024) uint8_t s_buf[];
  auto& smem = *reinterpret_cast<typename T::Smem*>(s_buf);

  const auto tid = threadIdx.x;
  const auto warp_id = warp::broadcast(tid / 32, 0);
  const auto lane_id = warp::get_lane_id();
  const auto rank = ptx::cluster_ctarank();  // K slab == owned row quadrant
  const auto tile = blockIdx.x / static_cast<uint32_t>(kSplitK);
  const auto row0 = tile * T::M_BLOCK;

  static_assert(kRole == 4 * 32);
  // Warps 0-3: activation (after the PDL wait), MMA (warp 2) and the owner reduce. Warps 4-7: the weight. The
  // warps busy after the wait sit on different SMSPs (warp w runs on scheduler w % 4).
  enum Role { WARP_X = 0, WARP_ALLOC = 1, WARP_MMA = 2, WARP_WEIGHT = 4, WARP_TMEM_FREE = T::kThreads / 32 - 1 };
  enum NamedBarrier { BAR_INIT = 1, BAR_WEIGHT = 2 };
  const auto num_tokens = p.num_tokens;
  // The owners' push phase, read up front so its latency hides in the prefetch: every counter of the plane holds the
  // same parity between kernels, and no predecessor back to the last griddepcontrol.wait-before-trigger kernel (the
  // attention) touches them.
  uint32_t phase_offset = 0;
  if constexpr (kWorldSize > 0) {
    if (tid < 4 * num_tokens) phase_offset = (ptx::load_counter(p.ws.counter) & 1) * p.ws.slot_bytes * kWorldSize;
  }

  // ---- init: the weight loads first (they depend on nothing), mbarriers, TMEM ----
  if (warp_id >= WARP_WEIGHT) {
    // Its scales first: a contiguous 2 KiB of the 128x4 image (row tile, then this slab's 4 k4 tiles).
    const auto wt = tid - kRole;
    const auto policy = ptx::policy_evict_first();
    static_assert(T::kSFBytes / 16 == kRole);
    const auto ws_src = p.w_scale + (size_t(tile) * (T::K / 128) + rank * kSub) * 512;
    ptx::cp_async_16B(ptx::to_shared(smem.sfa) + wt * 16, ws_src + wt * 16, policy);
    // The slab, one group per subtile, read row-major with the 128 B swizzle in the smem addresses: thread wt
    // takes chunk wt % 8 of rows wt / 8 + 16 i, so a warp reads 4 full 128 B row segments and the swizzle
    // (row % 8) is the thread's own.
    const auto chunk = wt % 8, row_base = wt / 8;
    const auto w_src = p.w + size_t(row0 + row_base) * T::K + rank * T::K_BLOCK + chunk * 16;
    const auto w_dst = ptx::to_shared(smem.w) + tcgen05::sw128_off(row_base, chunk);
#pragma unroll
    for (uint32_t s = 0; s < kSub; ++s) {
#pragma unroll
      for (uint32_t i = 0; i < T::M_BLOCK / 16; ++i) {
        ptx::cp_async_16B(w_dst + s * T::kWSubBytes + i * 2048, w_src + s * 128 + i * 16 * T::K, policy);
      }
      ptx::cp_async_commit();
    }
  } else if (warp_id == WARP_X) {
    if (warp::elect_one_lane()) {
      ptx::prefetch_tensormap(&p.x_map);
      ptx::mbar_init(&smem.mma_ready, 2);  // the weight warp's commit, the activation's expect_tx
      ptx::mbar_init(&smem.a_loaded, 1);
      ptx::mbar_init(&smem.mma_done, 1);
      ptx::mbar_init(&smem.red_ready, 1);
      ptx::mbar_init(&smem.drained, T::kThreads / 32);
      ptx::fence_mbarrier_init_release_cluster();
      // Every rank pushes its 32 x N partials of our rows (sparse: only the live tokens).
      ptx::mbar_arrive_expect_tx(&smem.red_ready, kSparse ? kSplitK * 32 * 4 * num_tokens : T::kInboxBytes);
    }
  } else if (warp_id == WARP_ALLOC) {
    tcgen05::alloc(&smem.taddr, T::kTmemCols);
    // Without this the SM's second CTA could never allocate.
    tcgen05::relinquish_alloc_permit();
    tcgen05::fence_before_thread_sync();
  }
  ptx::barrier_sync(BAR_INIT, T::kThreads);
  const auto taddr = smem.taddr;

  if (warp_id >= WARP_WEIGHT) {
    // ---- weight -> TMEM, subtile by subtile as the groups land (one lane of warp 4 copies) ----
    const bool copier = warp_id == WARP_WEIGHT && warp::elect_one_lane();
    const auto sfa = tcgen05::make_desc_sf_chunk(smem.sfa);
    const auto a = tcgen05::make_desc_kmajor_sw128(smem.w);
#pragma unroll
    for (uint32_t s = 0; s < kSub; ++s) {
      ptx::cp_async_wait_pending(kSub - 1 - s);
      ptx::fence_proxy_async_shared();  // cp.async wrote in the generic proxy, tcgen05.cp reads in the async one
      ptx::barrier_sync(BAR_WEIGHT, kRole);
      if (copier) {
        if (s == 0) {
          tcgen05::fence_after_thread_sync();
#pragma unroll
          for (uint32_t j = 0; j < kSub; ++j) {
            tcgen05::cp_32x128b_warpx4(taddr + T::kTmemSFA + 4 * j, sfa.advance_bytes(j * 512).bits);
          }
        }
        // A in 32-K blocks: block 4 s + kb is 32 B into subtile s of the swizzled slab.
#pragma unroll
        for (uint32_t kb = 0; kb < 4; ++kb) {
          tcgen05::cp_128x256b(taddr + T::kTmemA + 8 * (4 * s + kb), a.advance_bytes(s * T::kWSubBytes + 32 * kb).bits);
        }
      }
    }
    if (copier) {
      tcgen05::commit(&smem.a_loaded);   // for the cluster arrive below
      tcgen05::commit(&smem.mma_ready);  // for the MMA
    }
    // The inbox reuses the slab's smem: arrive only once it is in TMEM.
    ptx::mbar_wait_parity(&smem.a_loaded, 0);
  } else {
    // Release the dependents right away, so the next kernel can prefetch too.
    PDLTriggerSecondary<true>();
  }
  ptx::cluster_arrive_relaxed();

  if (warp_id == WARP_X) {
    // ---- activation: the B tile by TMA, its scales by one bulk copy -------
    if (warp::elect_one_lane()) {
      PDLWaitPrimary<true>();
      // Subtile s is the 128 K columns at rank * 512 + 128 s for the N tokens (rows past T are zeros), landing in
      // the SW128 K-major layout the MMA reads; the scales are this slab's 2 KiB of the 128x4 image.
      ptx::mbar_arrive_expect_tx(&smem.mma_ready, kSub * T::kXSubBytes + T::kSFBytes);
#pragma unroll
      for (uint32_t s = 0; s < kSub; ++s) {
        ptx::tma_load_2d(&smem.x[s * T::kXSubBytes], &p.x_map, rank * T::K_BLOCK + 128 * s, 0, &smem.mma_ready);
      }
      ptx::cp_async_bulk_g2s(smem.sfb, p.x_scale + rank * T::kSFBytes, T::kSFBytes, &smem.mma_ready);
    }
  } else if (warp_id == WARP_MMA && warp::elect_one_lane()) {
    // ---- MMA: one lane issues everything; the others go straight to the mma_done wait ----
    const auto sfb = tcgen05::make_desc_sf_chunk(smem.sfb);
    const auto b = tcgen05::make_desc_kmajor_sw128(smem.x);
    const auto idesc = tcgen05::make_instr_desc_mxf8<T::M_BLOCK, N>();
    ptx::mbar_wait_parity(&smem.mma_ready, 0);
    tcgen05::fence_after_thread_sync();
    // The B scales go in first: tcgen05.cp and tcgen05.mma of one thread execute in issue order.
#pragma unroll
    for (uint32_t s = 0; s < kSub; ++s) {
      tcgen05::cp_32x128b_warpx4(taddr + T::kTmemSFB + 4 * s, sfb.advance_bytes(s * 512).bits);
    }
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

  // Paired with the cluster arrives above, while the MMAs run: the peers' mbarriers are initialized and their
  // slabs are in TMEM (the inbox reuses that smem). The acquire invalidates L1, so it stays off the path after
  // mma_done.
  ptx::cluster_wait_acquire();

  // ---- TMEM -> registers: warp w reads lane quadrant w % 4, column half w / 4 ----
  ptx::mbar_wait_parity(&smem.mma_done, 0);
  tcgen05::fence_after_thread_sync();
  const auto quad = warp_id % 4, half = warp_id / 4;
  float acc[kHalf];
  tcgen05::ld_32x32b<kHalf>(taddr + ((quad * 32) << 16) + T::kTmemAcc + half * kHalf, acc);
  tcgen05::wait_ld();
  tcgen05::fence_before_thread_sync();
  __syncwarp();
  if (lane_id == 0) ptx::mbar_arrive(&smem.drained);

  // ---- split-K partials -> the quadrant's owner (cluster rank == quadrant) ----
  // Inbox and staging are [source][token][row]: lanes are rows, so every store and load is one 128 B row.
  const auto bar = ptx::mapa(ptx::to_shared(&smem.red_ready), quad);
  const auto dst = ptx::mapa(ptx::to_shared(&smem.inbox[(rank * N + half * kHalf) * 32]), quad);
  if constexpr (kSparse) {
    // Each lane sends its row's value of every live token.
#pragma unroll
    for (uint32_t i = 0; i < kHalf; ++i) {
      if (half * kHalf + i >= num_tokens) break;
      ptx::st_async_4B(dst + (i * 32 + lane_id) * 4, acc[i], bar);
    }
  } else {
    // The warp's block is staged and sent as one bulk copy, so each owner mbarrier sees 8 complete_tx.
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

  // ---- owner: sum the 4 K slabs in a fixed order, round, publish ----------
  // Item = (token, 8 rows): two LDS.128 per source, one 16 B store.
  static_assert(4 * N <= T::kThreads);
  if (tid < 4 * num_tokens) {
    const auto t = tid / 4, rg = tid % 4;
    ptx::mbar_wait_parity(&smem.red_ready, 0);
    AlignedVector<fp32_t, 4> sum[2];
#pragma unroll
    for (uint32_t src = 0; src < kSplitK; ++src) {
#pragma unroll
      for (uint32_t k = 0; k < 2; ++k) {
        AlignedVector<fp32_t, 4> v;
        v.load(&smem.inbox[(src * N + t) * 32 + rg * 8], k);
#pragma unroll
        for (uint32_t e = 0; e < 4; ++e) {
          sum[k][e] = src == 0 ? v[e] : sum[k][e] + v[e];
        }
      }
    }
    AlignedVector<bf16x2_t, 4> packed;
#pragma unroll
    for (uint32_t k = 0; k < 4; ++k) {
      packed[k] = cast<bf16x2_t>(fp32x2_t{sum[k / 2][2 * (k % 2)], sum[k / 2][2 * (k % 2) + 1]});
    }
    const auto row = row0 + rank * 32 + rg * 8;
    if constexpr (kWorldSize == 0) {
      packed.store(p.out + size_t(t) * T::M + row);
    } else {
      // +0 words would read as "not arrived": they go out as -0. Slot vector = (t, row / 8) of the [T, 5120] view.
      distributed::LamportTrait<bf16_t, 8, 4>::clear_pos_zero(packed.data());
      const auto vid = (t * T::M + row) / 8;
      const auto slot_offset = phase_offset + p.rank * p.ws.slot_bytes;
      if constexpr (kMulticast) {
        ptx::st_multimem_relaxed_16B(packed, p.ws.mc_workspace + slot_offset, vid);
      } else {
#pragma unroll
        for (uint32_t dst = 0; dst < kWorldSize; ++dst) {
          ptx::st_relaxed_16B(packed, p.ws.workspaces[dst] + slot_offset, vid);
        }
      }
    }
  }
  if constexpr (!kSparse) ptx::bulk_wait_read();

  // Every CTA owns rows, so it lives until all pushes into its inbox landed (the wait above). The last warp frees
  // TMEM: it is the one least likely to be an owner above.
  if (warp_id == WARP_TMEM_FREE) {
    ptx::mbar_wait_parity(&smem.drained, 0);
    tcgen05::fence_after_thread_sync();
    tcgen05::dealloc(taddr, T::kTmemCols);
  }
}

}  // namespace sglang::device::wo_b

namespace sglang {

/// x [T, 2048] as a 2D (k, t) e4m3 tensor map; box (128 B, box_rows), 128 B swizzle.
inline CUtensorMap make_wo_b_x_map(tvm::ffi::TensorView x, uint32_t box_rows) {
  using namespace host;
  // Through the runtime's driver entry point, so the module needs no -lcuda.
  static const auto encode = [] {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult status;
    CHECK_CUDA(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &status));
    CHECK_HOST(status == cudaDriverEntryPointSuccess && fn != nullptr);
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
  }();
  const cuuint64_t dims[2] = {cuuint64_t(x.size(1)), cuuint64_t(x.size(0))};
  const cuuint64_t strides[1] = {cuuint64_t(x.stride(0))};  // bytes
  const cuuint32_t box[2] = {128, box_rows};
  const cuuint32_t elem_strides[2] = {1, 1};
  CUtensorMap map;
  const auto res = encode(
      &map,
      CU_TENSOR_MAP_DATA_TYPE_UINT8,
      2,
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
 * \brief Low-latency wo_b producer: x[t, :] (e4m3, UE8M0/32) x W^T (e4m3, UE8M0/32) in bf16, written to `out`
 *        (kWorldSize == 0) or pushed into every rank's push-plane slot for all_reduce_mhc.cuh (kWorldSize > 0).
 *
 * Launched as a PDL secondary that releases its own dependents at start.
 * \tparam N          MMA N (8 / 16 / 32 / 64); T must be <= N.
 * \tparam kWorldSize 0: no push; 2 / 4: the communicator's world size.
 * \tparam kMulticast push with one multimem store through the plane's multicast address instead of kWorldSize
 *                    unicast stores.
 * \param x           [T, 2048] e4m3, row stride a multiple of 16.
 * \param x_scale     its UE8M0 scales, FlashInfer 128x4 image, >= 128 * 64 bytes.
 * \param w           weight [5120, 2048] e4m3, contiguous.
 * \param w_scale     its UE8M0 per-32 scales, FlashInfer 128x4 image (block_scale_interleave), >= 5120 * 64 bytes.
 * \param out         kWorldSize == 0: [T, 5120] bf16, contiguous.
 * \param comm        kWorldSize > 0: the CustomAllReduceV2 communicator whose push plane takes the rows.
 */
template <int N, uint32_t kWorldSize, bool kMulticast>
inline void dsv41_wo_b_ll(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView x_scale,
    tvm::ffi::TensorView w,
    tvm::ffi::TensorView w_scale,
    tvm::ffi::Optional<tvm::ffi::TensorView> out,
    tvm::ffi::Optional<host::distributed::CommunicatorRef> comm) {
  using namespace host;
  using T = device::wo_b::Trait<N>;
  auto num_tokens_ = SymbolicSize{"num_tokens"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();
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
  const int64_t w_row_stride = T::K, out_row_stride = T::M;  // contiguous
  TensorMatcher({T::M, T::K})                                //
      .with_strides({w_row_stride, 1})
      .with_dtype<fp8_e4m3_t>()
      .with_device(device_)
      .ensure_alignment(16)
      .verify(w);
  TensorMatcher({-1})  //
      .with_dtype<uint8_t>()
      .with_device(device_)
      .ensure_alignment(16)
      .verify(w_scale);
  const auto num_tokens = num_tokens_.unwrap();
  CHECK_HOST(num_tokens <= N) << num_tokens;
  CHECK_HOST(x_scale.size(0) >= 128 * T::K / 32) << x_scale.size(0);
  CHECK_HOST(w_scale.size(0) >= T::M * T::K / 32) << w_scale.size(0);
  CHECK_HOST(out.has_value() == (kWorldSize == 0));
  if constexpr (kWorldSize == 0) {
    TensorMatcher({num_tokens_, T::M})  //
        .with_strides({out_row_stride, 1})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(out.value());
  }
  if (num_tokens == 0) return;  // an empty reduce would wait forever on its inbox

  device::wo_b::Params<kWorldSize> p{};
  p.x_map = make_wo_b_x_map(x, N);
  p.x_scale = static_cast<const uint8_t*>(x_scale.data_ptr());
  p.w = static_cast<const uint8_t*>(w.data_ptr());
  p.w_scale = static_cast<const uint8_t*>(w_scale.data_ptr());
  if constexpr (kWorldSize == 0) p.out = static_cast<bf16_t*>(out.value().data_ptr());
  p.num_tokens = static_cast<uint32_t>(num_tokens);
  static_assert(kWorldSize == 0 || kWorldSize == 2 || kWorldSize == 4);
  static_assert(!kMulticast || kWorldSize > 0);
  if constexpr (kWorldSize > 0) {
    CHECK_HOST(comm.has_value());
    const auto& push = comm.value()->get_push_obj();
    CHECK_HOST(push.world_size == kWorldSize) << push.world_size;
    const int64_t num_bytes = num_tokens * T::M * int64_t(sizeof(bf16_t));
    CHECK_HOST(num_bytes <= push.slot_bytes) << num_bytes << " " << push.slot_bytes;
    p.rank = push.rank;
    p.ws = push.get_workspace<kWorldSize>(num_bytes);
    if constexpr (kMulticast) CHECK_HOST(p.ws.mc_workspace != nullptr) << "the push plane has no multicast mapping";
  }

  const auto device = device_.unwrap();
  constexpr auto kernel = device::wo_b::wo_b_ll_kernel<N, kWorldSize, kMulticast>;
  constexpr size_t kSmemBytes = sizeof(typename T::Smem);
  runtime::init_per_device(device, [] {
    runtime::set_max_dynamic_smem(kernel, kSmemBytes);
    runtime::set_smem_carveout(kernel, cudaSharedmemCarveoutMaxShared);
  });
  LaunchKernel(static_cast<uint32_t>(T::kNumCtas), T::kThreads, device, kSmemBytes)
      .config({.use_pdl = true, .cluster_dim = dim3{T::kSplitK, 1, 1}})(kernel, p);
}

}  // namespace sglang
