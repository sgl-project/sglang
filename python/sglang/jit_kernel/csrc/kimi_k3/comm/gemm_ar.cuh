// K3 fused o_proj GEMM + all-reduce for decode (bf16, TP row-parallel).
//
// Vendored from the autokernels gemm_ar family single-file runner
// (oproj_gemm_ar.cu, p2p-only slim build, 2026-07-23); the standalone
// test/bench harness is dropped and replaced by the tvm-ffi adapter at the
// bottom (GemmArKernel<K, R, kUsePDL>). Keep local changes diffable against
// the source runner.
//
// CONTRACT (per rank r of R):  out[M, 7168] = sum_r x_r[M, K] @ W_r[7168, K]^T
//   bf16 in/out, fp32 accumulate, partials round to bf16 pre-sum (same
//   semantics as the unfused GEMM + bf16 ring AR).
// M in [1, 512], rounded up to a tuned cell {8,16,32,64,128,256,512}; out
// must have `cell` rows — rows [M, cell) are clobbered with zeros.
// Comm plane: pure NVLink P2P (unicast pushes + per-rank flag reductions);
// one-shot AR below the two-shot threshold, two-shot RS+AG above.
// Requires SM100+ with full P2P; tuned on GB300 (sm_103a).
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <sgl_kernel/cuda_check.h>
#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/tensor_map.h>
#include <sgl_kernel/ptx_addr.cuh>
#include <sgl_kernel/ptx_mbarrier.cuh>
#include <sgl_kernel/ptx_mma.cuh>
#include <sgl_kernel/ptx_smem.cuh>
#include <sgl_kernel/ptx_sync.cuh>
#include <sgl_kernel/ptx_cvt.cuh>
#include <sgl_kernel/ptx_clc.cuh>
#include <sgl_kernel/ptx_tma.cuh>
#include <sgl_kernel/ptx_tcgen05.cuh>
#include <sgl_kernel/ptx_mma_desc.cuh>
#include <sgl_kernel/dense_gemm_mainloop.cuh>
#include <sgl_kernel/ptx_tcgen05_mma_dense.cuh>
#include <sgl_kernel/swizzle.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>
#include <stdexcept>

// ================= kernels/gemm_ar/oss_main.cu =================
// kernels/gemm_ar — MINIMAL OSS entry (assembled into oproj_gemm_ar_single.cu;
// the full gate/soak/baseline harness lives in test.cu, and the NVLS/multimem
// comm plane in the family kernels — this runner ships the pure-P2P plane,
// which measures >= the NVLS plane at every shape on 4x GB300 NVL4).
// Correctness tests + a simple benchmark; no cublas, no multicast objects.
//
// SHAPE CONTRACT (Kimi-K3 attention output projection, TP row-parallel):
//   per rank r:  partial_r[M, 7168] = x_r[M, 3072] @ W_r[7168, 3072]^T
//   out[M, 7168] = sum_r partial_r          (bf16 in/out, fp32 accumulate,
//                                            partials round to bf16 pre-sum)
//   R = 4 GPUs (compile-time; R=8 deployment shards K=12288/R=1536 — same
//   kernels, recompile with R/K changed).
//
// TOKEN SUPPORT: M = total decode tokens across the batch, ANY value in
//   [1, 512] (spec-decode: M = 8 verify tokens x batch size). Internally M
//   rounds up to the nearest tuned cell {8,16,32,64,128,256,512}; rows
//   [M, cell) of `out` are clobbered with zeros. M < 1 or M > 512 aborts
//   with an error (kMMax sizes the communication slots).


// ================= kernels/gemm_ar/kernel.cu =================
// kernels/gemm_ar/oproj_ar — fused low-latency GEMM + one-shot all-reduce
// (OPROJ_AR_DESIGN.md). Per rank r of R:
//   partial_r[M,N] = x_r[M,K] @ W_r[N,K]^T   (bf16 in, fp32 accum)
//   out[M,N]       = sum_r partial_r          (bf16, replicated everywhere)
// One kernel per rank per call: 1-wave grid (gridDim ≤ resident capacity —
// CTAs spin on a grid+rank-wide flag, a 2nd wave would deadlock by
// construction), TMA swizzle-128B W stream → mma.sync bf16, epilogue pushes
// each warp's finished tile into the comm plane, then ONE §H-shaped boundary
// (per-CTA acq_rel gather → last CTA fence.release.sys + relaxed flag →
// per-CTA local-replica acquire spin), then a tile-local reduce. Comm plane
// is a template axis; this build ships the pure-P2P planes:
//   kPeer   — per-peer unicast st push (no MC object anywhere; any P2P
//             NVLink set, the R=8 fallback); per-rank red.relaxed flags.
// GEMM_ON=false replaces the mainloop with a gmem partial read → the
// standalone AR kernel the unfused composite baseline launches.


// ================= kernels/gemm_ar/kernel.cu =================

namespace oproj_ar {

using device::distributed::atomic_add_acq_rel_gpu;
using device::distributed::fence_release_sys;
using device::distributed::load_acquire_sys;
using device::distributed::red_add_relaxed_sys;

// ---------------------------------------------------------------- constants
#ifndef OPROJ_N                 // output dim (columns of W / of out). The
#define OPROJ_N 7168            // default is the Kimi-K3 o_proj shape; other
#endif                          // shapes compile via -DOPROJ_N (see asserts).
constexpr int kN        = OPROJ_N;
static_assert(kN % 256 == 0 && kN >= 256,
              "N must be a multiple of 256 (member tile table: 128-row "
              "strips + BN up to 256); relaxing this needs a BN-table edit");
constexpr int kBK       = 64;   // K per stage (128 B rows → swizzle-128B)
constexpr int kBNRows   = 48;   // B-box rows per stage (6 n8-tiles)
constexpr int kTilesMax = 6;    // n8-tiles per CTA (5-tile CTAs pad, never push)
constexpr int kCWarps   = 6;    // consumer warp w owns n8-tile w
constexpr int kThreads  = (kCWarps + 1) * 32;   // +1 dedicated TMA producer warp
constexpr int kMMax     = 512;  // bs64 — sizes the shared slot layout
constexpr int kRing     = 64;   // epoch flag/gather ring (monotonic values)

enum class Comm { kNone, kMc, kPeer, kMcPull, kTwoShot, kTwoShotPeer };

// Shared-region layout (BYTES, M-independent: sized at kMMax so every arm and
// every bs cell reuses one region). Parity-double-buffered payloads; the flag
// ring lives on its own 2 MB page. Slot reuse across epochs e / e+2 is safe
// with 2 parities because each rank's launch e+1 spin-waits epoch e+1 AFTER
// its own e-reduce (per-rank stream order) — DESIGN §2.
// Slots are TILE-MAJOR ([n8-tile][m][8 cols]), NOT [m][n]: a warp's push for
// one tile is then a CONTIGUOUS 128 B fabric write instead of 32 scattered
// 4-16 B m-strided writes — lane-scatter runs ~0.26x on this fabric
// (cross_rank_sync §D) and the scattered form's ack-drain dominated the bs8
// boundary (stamped 22 us at idle). The reduce un-transposes locally.
constexpr size_t kSlotBytes1 = size_t(kMMax) * kN * 2;   // one [M,N] bf16
constexpr __host__ __device__ size_t slot_off(int parity, int src, int R) {
    return (size_t(parity) * R + src) * kSlotBytes1;
}
constexpr __host__ __device__ size_t pull_off(int parity, int R) {           // [2][M,N] above slots
    return (2 * size_t(R) + parity) * kSlotBytes1;
}
constexpr __host__ __device__ size_t flags_off(int R) {
    const size_t end = (2 * size_t(R) + 2) * kSlotBytes1;
    return (end + (size_t(2) << 20)) & ~((size_t(2) << 20) - 1);
}
// second flag family: "epoch e's slots fully REDUCED", at PER-CTA granularity.
// PDL's wait pairs with the prior grid's TRIGGER (not completion), and with
// 2-CTA/SM residency a fast rank's e+2 pushes can overwrite a slot replica a
// straggling rank's e-reduce still reads — so epoch e's push phase guards on
// done[e-2]. Per-CTA flags (the overwriter of tile t IS every rank's CTA-t)
// keep the publish fully parallel: no second grid-wide gather chain.
constexpr int kMaxCTA = 256;
constexpr int kFams   = 7;   // flag/gather/done ring FAMILY per dispatch
                             // CELL {8,16,32,64,128,256,512}. Monotonic ring
                             // targets assume every epoch of a ring came
                             // from the same gridDim (and only two-shot
                             // bumps the boundary-2 ring words), so one fam
                             // per cell removes the host-side ring reset a
                             // cell change needed under the original 3-fam
                             // grid-class split — the reset was a collective
                             // and blocked CUDA-graph capture.
constexpr __host__ __device__ size_t done_off(int R) {
    return flags_off(R) + size_t(kFams) * 512;
}
constexpr __host__ __device__ size_t region_bytes(int R) {
    return done_off(R) + size_t(kFams) * kRing * kMaxCTA * 4 + 512;
}


// PDL (programmatic dependent launch): the NEXT launch on the stream may
// start its feed while THIS grid sits in the boundary spin + reduce; its
// epilogue then griddepcontrol.wait's until this grid fully completes, so
// slots/flags/out stay race-free. Serving overlaps the next layer's kernels
// the same way; an unfused cublas composite cannot cooperate across the
// vendor-kernel boundary.
static __device__ __forceinline__ void pdl_launch_dependents() {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
static __device__ __forceinline__ void pdl_wait() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
}
// ldmatrix.x2 (non-trans) — one n8×k16 B fragment (b0,b1) from an [n][k]
// row-major tile: lane L's r_m = (row n=L/4, k-pair 2(L%4)) of matrix m —
// exactly the mma.m16n8k16 .col B fragment (k-pairs per lane, n = L/4).
// Lanes 0-7 address matrix 0 rows (k-half 0), 8-15 matrix 1 (k-half 1).
static __device__ __forceinline__ void ldmatrix_x2_b16(
        uint32_t smem_addr, uint32_t& r0, uint32_t& r1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 {%0, %1}, [%2];"
                 : "=r"(r0), "=r"(r1) : "r"(smem_addr));
}
static __device__ __forceinline__ uint32_t cluster_ctarank() {
    uint32_t r; asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r;
}
static __device__ __forceinline__ uint32_t bf2_u32(float2 f) {
    const __nv_bfloat162 p = __float22bfloat162_rn(f);
    return *reinterpret_cast<const uint32_t*>(&p);
}

// ------------------------------------------------------------------ params

template <int R>
struct Params {
    uint8_t*  mc_base;          // MC VA (null when no MC object — kPeer runs)
    uint8_t*  uc_base[R];       // per-rank unicast VAs of the shared region
    uint32_t* gather;           // device-local u32[kRing]
    __nv_bfloat16* out;         // [M,N] local output
    const __nv_bfloat16* partial_in;  // GEMM_ON=false input [M,N]
    // Device-resident per-fam CTA ticket counters. Every CTA takes one ticket
    // at entry and divides by the (family-stable) gridDim to recover the
    // launch epoch. All CTAs have taken their tickets before this grid
    // triggers a PDL successor, so successive launches receive disjoint,
    // contiguous ticket ranges without a separate bump kernel. Device state
    // instead of a launch arg keeps CUDA-graph replays advancing the epoch.
    uint32_t* epoch_base;
    int       my_rank;
    int       fam;              // ring family (dispatch cell) — see kFams
};

// -------------------------------------------------------------- CTA strips
// kN/8 = 896 n8-tiles over gridDim CTAs: first `rem` CTAs own base+1 tiles.
struct Strip {
    int t0, nt;
    static __device__ Strip make(int cta, int ncta) {
        const int kT = kN / 8;
        const int base = kT / ncta, rem = kT % ncta;
        Strip s;
        if (cta < rem) { s.nt = base + 1; s.t0 = cta * (base + 1); }
        else           { s.nt = base;     s.t0 = rem * (base + 1) + (cta - rem) * base; }
        return s;
    }
};

// ------------------------------------------------------------------ kernel
// mbar contract (recipes/mbar_handshake_design): full[s] count=1, arrive =
// producer's arrive_expect_tx(B+A bytes) + TMA complete_tx; empty[s] count =
// kCWarps (one elected lane per consumer warp after its last stage read).
// Producer = warp kCWarps lane 0; it never consumes, so the ring never
// self-blocks. Ring reuse distance = S stages, issue j waits empty parity
// ((j-S)/S)&1 — both derived from S.
// C = TMA cluster size for the A feed: ONE leader multicast per stage fills
// all C CTAs' A slots (cp_async_bulk_tensor_2d_load_multicast_cg1: each CTA's
// local full-bar gets its own tx-decrement, so expect_tx is unchanged). Per-CTA
// A bytes drop C-fold — the M-scaling A-tax (LEDGER O1) is A riding every
// CTA's capped TMA pipe. Contract deltas at C>1: the LEADER's empty[s] count
// = kCWarps*C (followers' consumers cluster-arrive it — release variant, so
// their slot reads are performed-before the leader's next multicast write);
// followers skip their own A issue but keep full expect_tx.
template <int M, int K, int R, Comm COMM, bool GEMM_ON, int S, int CH, int C>
__global__ void __launch_bounds__(kThreads) oproj_ar_kernel(
        const __grid_constant__ CUtensorMap w_map,
        const __grid_constant__ CUtensorMap x_map,
        const __grid_constant__ Params<R> prm) {
    constexpr int Mp      = (M + 15) & ~15;   // mma m16 padding (x buffer padded)
    constexpr int MT      = Mp / 16;
    constexpr int KSTEPS  = K / (kBK * CH);   // OUTER stages: CH k-chunks batched
    constexpr int kBBytes = kBNRows * kBK * 2, kABytes = Mp * kBK * 2;
    constexpr int kStB = CH * kBBytes, kStA = CH * kABytes;
    static_assert(K % (kBK * CH) == 0);

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const Strip strip = Strip::make(blockIdx.x, gridDim.x);
    __shared__ uint32_t cta_epoch;
    if (tid == 0)
        cta_epoch = atomicAdd(prm.epoch_base + prm.fam, 1u) / uint32_t(gridDim.x);
    __syncthreads();
    const uint32_t epoch = cta_epoch;
    const int parity = int(epoch & 1);
    const int ring   = int(epoch % kRing);
    // pinned wait-set (§I): flag VA + monotonic targets resolved before any spin
    const size_t foff = flags_off(R) + size_t(prm.fam) * 512;
    const size_t doff2 = done_off(R) + size_t(prm.fam) * kRing * kMaxCTA * 4;
    uint32_t* const flag_local = reinterpret_cast<uint32_t*>(
        prm.uc_base[prm.my_rank] + foff) + ring;
    uint32_t* const done_local = reinterpret_cast<uint32_t*>(
        prm.uc_base[prm.my_rank] + doff2) + size_t(blockIdx.x);
    uint32_t* const gather_fam = prm.gather + size_t(prm.fam) * 2 * kRing;
    const uint32_t wrap        = epoch / kRing + 1;
    const uint32_t flag_target = wrap * R;
    const uint32_t gath_target = wrap * gridDim.x;

    float4 acc[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) acc[i] = make_float4(0.f, 0.f, 0.f, 0.f);

    extern __shared__ __align__(1024) uint8_t smem[];
    uint8_t*  b_st   = smem;                                   // [S][CH][kBBytes]
    uint8_t*  a_st   = b_st + size_t(S) * kStB;                // [S][CH][kABytes]
    uint64_t* fullb  = reinterpret_cast<uint64_t*>(a_st + size_t(S) * kStA);
    uint64_t* emptyb = fullb + S;

    const uint32_t crank = C > 1 ? cluster_ctarank() : 0;
    {
        if (tid == 0) {
            ptx::prefetch_tensormap(&w_map);
            ptx::prefetch_tensormap(&x_map);
#pragma unroll
            for (int s = 0; s < S; ++s) {
                ptx::mbar_init(fullb + s, 1);
                ptx::mbar_init(emptyb + s, crank == 0 ? kCWarps * C : kCWarps);
            }
        }
        __syncthreads();
        if constexpr (C > 1) ptx::cluster_sync_rel_acq();

        if (warp == kCWarps) {
            // ---- producer: the whole K stream, one thread -----------------
            // k-phase rotation: spreads B's DRAM pages across CTAs; fp32
            // accumulation order changes per CTA — a sum, gate-covered.
            // CLUSTER-uniform: all members consume the same A flight.
            const int phase = ((int(blockIdx.x) / C) * KSTEPS) / (int(gridDim.x) / C);
            if (lane == 0) {
                for (int j = 0; j < KSTEPS; ++j) {
                    const int slot = j % S;
                    const int jj = (j + phase) % KSTEPS;
                    if (j >= S) ptx::mbar_wait_parity(emptyb + slot, ((j - S) / S) & 1);
                    ptx::mbar_arrive_expect_tx(fullb + slot, kStB + kStA);
#pragma unroll
                    for (int c = 0; c < CH; ++c) {
                        ptx::cp_async_bulk_tensor_2d_load(
                            ptx::to_shared(b_st + size_t(slot) * kStB + c * kBBytes),
                            &w_map, (jj * CH + c) * kBK, strip.t0 * 8, fullb + slot);
                        if constexpr (C > 1) {
                            if (crank == 0)
                                ptx::cp_async_bulk_tensor_2d_load_multicast_cg1(
                                    ptx::to_shared(a_st + size_t(slot) * kStA + c * kABytes),
                                    &x_map, (jj * CH + c) * kBK, 0, fullb + slot,
                                    uint16_t((1u << C) - 1));
                        } else {
                            ptx::cp_async_bulk_tensor_2d_load(
                                ptx::to_shared(a_st + size_t(slot) * kStA + c * kABytes),
                                &x_map, (jj * CH + c) * kBK, 0, fullb + slot);
                        }
                    }
                }
            }
        } else {
            // ---- consumers: warp w owns n8-tile (strip.t0 + w) ------------
            // A fragments load straight from gmem (x is L2-hot and tiny; the
            // LSU pipe is idle here) — the TMA path stays a pure-B stream.
            const int b_row = (lane & 7) + warp * 8;   // row within the B box
            const int b_ka  = (lane >> 3) & 1;         // k-atom half (x2)
            const int a_row = lane & 15, a_ka = lane >> 4;
            for (int s = 0; s < KSTEPS; ++s) {
                const int slot = s % S;
                ptx::mbar_wait_parity(fullb + slot, (s / S) & 1);
#pragma unroll
                for (int c = 0; c < CH; ++c) {
                const uint32_t b_base = ptx::to_shared(b_st + size_t(slot) * kStB + c * kBBytes);
#pragma unroll
                for (int k16 = 0; k16 < kBK / 16; ++k16) {
                    uint32_t b0, b1;
                    ldmatrix_x2_b16(
                        b_base + uint32_t(b_row) * (kBK * 2)
                               + swz::smem_col_128b_bf16(b_row, (k16 * 2 + b_ka) * 8) * 2,
                        b0, b1);
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        uint32_t a0, a1, a2, a3;
                        ptx::ldmatrix_x4_b16(
                            ptx::to_shared(a_st + size_t(slot) * kStA + c * kABytes
                                + uint32_t(mt * 16 + a_row) * (kBK * 2)
                                + swz::smem_col_128b_bf16(a_row, (k16 * 2 + a_ka) * 8) * 2),
                            a0, a1, a2, a3);
                        ptx::mma_m16n8k16_bf16f32(acc[mt], a0, a1, a2, a3, b0, b1);
                    }
                }
                }
                if (lane == 0) {
                    ptx::mbar_arrive(emptyb + slot);
                    if constexpr (C > 1)
                        if (crank != 0)
                            ptx::mbar_arrive_cluster_release(emptyb + slot, 0);
                }
            }
        }
    }

    // ---- epilogue: push ----------------------------------------------------
    __syncthreads();               // whole CTA past its smem/feed reads
    pdl_launch_dependents();       // next launch streams weights under our
                                   // push+boundary+reduce (needs 2-CTA/SM
                                   // co-residency: 100% smem carveout + this
                                   // kernel's smem ≤ ~113 KB)
    pdl_wait();   // prior grid reached ITS trigger (k-loop end) — NOT done
    {
        // guard: epoch e-2 (same parity) fully reduced everywhere before we
        // overwrite its slots. Steady-state this is already set (~one hot
        // acquire); it binds only when a boundary straggles.
        if (tid == 0 && epoch >= 2) {
            const uint32_t e2 = epoch - 2;
            const uint32_t tgt = (e2 / kRing + 1) * R;
            while (load_acquire_sys(done_local + size_t(e2 % kRing) * kMaxCTA) < tgt) { }
        }
        __syncthreads();
    }
    const int  tig = lane & 3, grp = lane >> 2;
    const int  n0  = (strip.t0 + warp) * 8;
    const bool own = warp < strip.nt;   // phantom 6th tile / producer never push

    // tile-major slot offset: tile t (global n8 index), row m, byte offset
    auto slot_tm = [&](uint8_t* base, size_t sb, int t, int m) {
        return base + sb + (size_t(t) * M + m) * 16;
    };



    {
        const size_t sb = slot_off(parity, prm.my_rank, R);
        if (own) {
            const int t = strip.t0 + warp;
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int m = mt * 16 + grp + half * 8;
                    if (m < M) {
                        const uint32_t v = half ? bf2_u32(make_float2(acc[mt].z, acc[mt].w))
                                                : bf2_u32(make_float2(acc[mt].x, acc[mt].y));
                        // lanes (grp,tig) of one warp land contiguous: 128 B
                        // per (mt,half) group
                        const size_t boff = size_t(m) * 16 + tig * 4;
                        {
#pragma unroll
                            for (int r = 0; r < R; ++r)
                                *reinterpret_cast<uint32_t*>(
                                    slot_tm(prm.uc_base[r], sb, t, 0) + boff) = v;
                        }
                    }
                }
            }
        }
    }



    // ---- boundary (§H gather + flag + per-CTA local-replica spin) ----------
    __syncthreads();
    if (tid == 0) {
        const uint32_t old = atomic_add_acq_rel_gpu(gather_fam + ring, 1);
        if (old + 1 == gath_target) {
            // ONE completing-CTA fence: publishes every CTA's pushes via the
            // acq_rel gather chain (§H). Measured 0.9 us better than per-CTA
            // fences at bs1 (11.04 vs 11.97) — the parallel-drain theory lost.
            fence_release_sys();
            {
#pragma unroll
                for (int r = 0; r < R; ++r)
                    red_add_relaxed_sys(reinterpret_cast<uint32_t*>(
                        prm.uc_base[r] + foff) + ring, 1);
            }
        }
        while (load_acquire_sys(flag_local) < flag_target) { }
    }
    __syncthreads();

    // ---- reduce: each CTA finishes its own tiles from the LOCAL replica ----
    const int units = strip.nt * M;
    for (int u = tid; u < units; u += kThreads) {
        const int t = u / M, m = u % M, c0 = (strip.t0 + t) * 8;
        const size_t soff = (size_t(strip.t0 + t) * M + m) * 16;   // tile-major
        uint4 res;
        {
            float2 s[4] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint4 v = *reinterpret_cast<const uint4*>(
                    prm.uc_base[prm.my_rank] + slot_off(parity, r, R) + soff);
                const uint32_t w4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(&w4[j]));
                    s[j].x += f.x; s[j].y += f.y;
                }
            }
            res = make_uint4(bf2_u32(s[0]), bf2_u32(s[1]), bf2_u32(s[2]), bf2_u32(s[3]));
        }
        *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(prm.out)
                                  + (size_t(m) * kN + c0) * 2) = res;
    }

    // ---- done-publish: per-CTA, fence-free. The beacon carries no payload:
    // the remote guard only needs its VALUE. This CTA's reduce loads are
    // data-flow-complete (their values fed the out stores) before the
    // syncthreads, so the relaxed beacon cannot pass an unfinished read; a
    // fence here stamped at ~4 us draining against the co-resident feed.
    __syncthreads();
    if (tid == 0) {
        const size_t off = size_t(ring) * kMaxCTA + blockIdx.x;
        {
#pragma unroll
            for (int r = 0; r < R; ++r)
                red_add_relaxed_sys(reinterpret_cast<uint32_t*>(
                    prm.uc_base[r] + doff2) + off, 1);
        }
    }
}


// ================= kernels/gemm_ar/kernel_dense_ar.cuh =================
// kernels/gemm_ar — member 3 `dense_ar` (LEDGER O9f): the tuned dense_1cta
// BF16 mainloop (fp8out-sibling SKELETON: simple persistent tile loop = grid
// ≤ SM count, 1-wave-resident — AR flag spin safe; 2-stage TMEM ping-pong
// via mainloop/epi mbars) with the AR comm epilogue in place of the C store.
// Forked from kernels/gemm/dense_1cta/kernel.cu (fp8out kernel skeleton +
// the BF16-out drain idiom) per the family fork rule; re-fold candidate once
// the member set settles. Slots use the DRAIN-NATURAL order (lane-int4 rows)
// so every push is 512 B-coalesced per warp (W7's lesson); the reduce
// descatters locally into out[m][n].
// NOTE: included from kernel.cu INSIDE namespace oproj_ar — the headers this
// member needs (ptx_cvt, dense_gemm_mainloop) are hoisted to kernel.cu's
// top-level include block.

constexpr int kD3BM = 128, kD3BK = 64;
constexpr __host__ __device__ int d3_ns(int bn) {   // ring depth by stage size
    // deepest ring the smem budget fits, capped at the mbar array size. The
    // old 12/11/4 step form silently ran M>=128 at NS=4 (BN64 stage 24.5 KB
    // fits 9) — a 4-deep ring can't hide DRAM latency and was the dominant
    // M=128 pole (21.4 us vs DG 17.2 at-shape).
    const int stage = kD3BM * kD3BK * 2 + bn * kD3BK * 2;
    const int ns = 220 * 1024 / stage;
    return ns > 12 ? 12 : ns;
}
constexpr int kD3GroupN = 16, kD3KPer = kD3BK / 16;
constexpr int kD3Threads = 8 * 32;                       // warps 0,1,4-7 live
constexpr int kD3ABytes = kD3BM * kD3BK * 2;
// per-M BLOCK_N: smaller BN = more tiles = more feed pipes (grid-starvation
// at BN=256 measured M-flat ~30.7 us: 28 CTAs); bigger BN = less A-dup.
// bytes/(min(tiles,152)) + tensor-floor optimum: 64 for M<=128, 128 above.
// re-swept 2026-07-23 at floored d3_ns: M=128 BN64/NS9 16.41 vs BN128/NS6
// 20.50 (ring depth, not BN, was the old 21.4 pole); 256 BN128/NS6 20.51;
// 512 BN256/NS4 30.9 (BN128's 224-on-152 nu-imbalance regressed to 38.9).
constexpr __host__ __device__ int d3_bn(int Mp) {
    return Mp <= 128 ? 64 : (Mp == 256 ? 128 : 256);
}

// drain-order slot offset: unit = (tile t, n-block nb, epi-warp w, lane l),
// 16 B each. Total = num_tiles * (BN/8) * 4 * 32 * 16 = Mpad*N*2 bytes.
__device__ __forceinline__ size_t d3_slot_off(int t, int nb, int w, int l,
                                              int nblk_per_tile) {
    return ((size_t(t) * nblk_per_tile + nb) * 4u + w) * 32u * 16u + size_t(l) * 16u;
}

// SWAP (M<=64): swapAB on the dense ring — A = W (M-slot carries the 7168
// n-rows in 128-row strips, NO small-M padding tax), B = x^T (N-slot = Mp,
// as small as 8). Tiles = 56 n-strips; the drain's lane-rows become n and
// its cols become m — the same drain-order slots work, only the out mapping
// transposes (local scatter, cheap at these payloads).
template <int M, int K, int R, Comm COMM>
__global__ void __launch_bounds__(kD3Threads) oproj_dense_ar_kernel(
        const __grid_constant__ CUtensorMap x_tmap,      // A = x [kMMax, K]
        const __grid_constant__ CUtensorMap w_tmap,      // B = W [kN, K]
        const __grid_constant__ Params<R> prm) {
    constexpr bool SWAP = (M <= 64);
    // two-shot planes share RS/reduce/out-copy; they differ only in the AG +
    // flag transport (kTwoShot = NVLS multimem, kTwoShotPeer = pure P2P).
    constexpr bool k2S =
        (COMM == Comm::kTwoShot || COMM == Comm::kTwoShotPeer);
    // SWAP = the DG decode recipe (O9h): token-slot tiles of 16/32 m-cols —
    // tiles = 56 strips × (Mp/tok) fills the grid (112 pipes at M≥32; the
    // 56-tile form measured flat ~19.6-20.5, R11) — with a 12-deep ring.
    // UMMA M=128 requires N ≥ 16 step 16 (bs1 pads the token slot to 16).
    constexpr int Mp = SWAP ? (M < 16 ? 16 : ((M + 15) & ~15))
                            : (M + kD3BM - 1) / kD3BM * kD3BM;
    constexpr int kD3BN = SWAP ? (Mp <= 32 ? 16 : 32) : d3_bn(Mp);
    constexpr int kD3BBytes = kD3BN * kD3BK * 2;
    constexpr int kGridM = SWAP ? kN / kD3BM : Mp / kD3BM;
    constexpr int kGridN = SWAP ? Mp / kD3BN : kN / kD3BN;
    constexpr int kTiles = kGridM * kGridN;
    constexpr int kIters = K / kD3BK;
    constexpr int kNBlk  = kD3BN / 8;

    constexpr int kD3NS = d3_ns(kD3BN);
    // tcgen05.alloc column count must be a power of two >= 32
    constexpr int kTmemCols = 2 * kD3BN <= 32 ? 32
        : 2 * kD3BN <= 64 ? 64 : 2 * kD3BN <= 128 ? 128
        : 2 * kD3BN <= 256 ? 256 : 512;
    extern __shared__ __align__(1024) uint8_t smem_buf[];
    const uint32_t smem_base = ptx::to_shared(smem_buf);
    constexpr uint32_t kSmemAOff = 0, kSmemBOff = kD3NS * kD3ABytes;

    __shared__ __align__(8) uint64_t tma_mbars[12];
    __shared__ __align__(8) uint64_t mma_mbars[12];
    __shared__ __align__(8) uint64_t mainloop_mbars[2];
    __shared__ __align__(8) uint64_t epi_mbars[2];
    __shared__ __align__(4) uint32_t s_taddr[1];
    __shared__ uint32_t cta_epoch;

    const int tid = threadIdx.x, warp_id = tid >> 5, lane_id = tid & 31;
    const int bid = int(blockIdx.x), num_bids = int(gridDim.x);
    if (tid == 0)
        cta_epoch = atomicAdd(prm.epoch_base + prm.fam, 1u) / uint32_t(gridDim.x);
    __syncthreads();
    const uint32_t epoch = cta_epoch;
    const int parity = int(epoch & 1);
    const int ring   = int(epoch % kRing);
    const size_t foff  = flags_off(R) + size_t(prm.fam) * 512;
    const size_t doff2 = done_off(R) + size_t(prm.fam) * kRing * kMaxCTA * 4;
    uint32_t* const flag_local = reinterpret_cast<uint32_t*>(
        prm.uc_base[prm.my_rank] + foff) + ring;
    uint32_t* const done_local = reinterpret_cast<uint32_t*>(
        prm.uc_base[prm.my_rank] + doff2) + size_t(blockIdx.x);
    uint32_t* const gather_fam = prm.gather + size_t(prm.fam) * 2 * kRing;
    const uint32_t wrap        = epoch / kRing + 1;
    const uint32_t flag_target = wrap * R;
    const uint32_t gath_target = wrap * gridDim.x;
    const size_t sb = slot_off(parity, prm.my_rank, R);

    if (warp_id == 0 && ptx::elect_one()) {
        for (int i = 0; i < kD3NS; ++i) {
            ptx::mbar_init(&tma_mbars[i], 1);
            ptx::mbar_init(&mma_mbars[i], 1);
        }
        for (int i = 0; i < 2; ++i) {
            ptx::mbar_init(&mainloop_mbars[i], 1);
            ptx::mbar_init(&epi_mbars[i], 4 * 32);
        }
    } else if (warp_id == 1) {
        ptx::tcgen05_alloc(ptx::to_shared(s_taddr), kTmemCols);
    }
    __syncthreads();
    const uint32_t taddr = s_taddr[0];

    constexpr uint32_t i_desc = ptx::mma_inst_desc_f16(
        kD3BM, kD3BN, ptx::F16Type::BF16, ptx::F16Type::BF16,
        ptx::DType::F32, ptx::Major::K, ptx::Major::K);
    auto tile_mn = [&](int linear) -> int2 {
        return dense_gemm_mainloop::group_n_swizzle<1, kD3GroupN>(
            linear, 0, kGridM, kGridN);
    };

    // Prefetch the first ring of input-independent weight stages before the
    // PDL dependency. SWAP changes which operand slot contains W, but never
    // changes which tensor map is safe to touch here.
    if (warp_id == 0 && ptx::elect_one()) {
        constexpr int kPrefetch = kIters < kD3NS ? kIters : kD3NS;
        const int2 mn = tile_mn(bid);
#pragma unroll
        for (int k = 0; k < kPrefetch; ++k) {
            ptx::mbar_arrive_expect_tx(
                &tma_mbars[k], kD3ABytes + kD3BBytes);
            ptx::cp_async_bulk_tensor_2d_load(
                smem_base + (SWAP ? kSmemAOff + k * kD3ABytes
                                  : kSmemBOff + k * kD3BBytes),
                &w_tmap, k * kD3BK,
                (SWAP ? mn.x * kD3BM : mn.y * kD3BN), &tma_mbars[k]);
        }
    }

    // x and all slot traffic remain behind the dependency.
    pdl_wait();

    // done-guard before any slot write (W3; PDL wait pairs with the trigger)
    {
        if (tid == 0 && epoch >= 2) {
            const uint32_t e2 = epoch - 2;
            const uint32_t tgt = (e2 / kRing + 1) * R;
            while (load_acquire_sys(done_local + size_t(e2 % kRing) * kMaxCTA) < tgt) { }
        }
        __syncthreads();
    }

    if (warp_id == 0 && ptx::elect_one()) {
        // TMA issuer (simple persistent) — verbatim dense_1cta fp8out shape
        int stage = 0, mma_phase = 1;
        for (int t = bid; t < kTiles; t += num_bids) {
            const int2 mn = tile_mn(t);
            for (int k = 0; k < kIters; ++k) {
                ptx::mbar_wait_parity(&mma_mbars[stage], mma_phase);
                constexpr bool kDropA = false;
                const bool prefetched = (t == bid && k < kD3NS);
                if (!prefetched)
                    ptx::mbar_arrive_expect_tx(&tma_mbars[stage],
                        (kDropA ? 0 : kD3ABytes) + kD3BBytes);
                if constexpr (!kDropA) {
                    // In SWAP, A is the prefetched weight; otherwise A is x.
                    if (!prefetched || !SWAP)
                        ptx::cp_async_bulk_tensor_2d_load(
                            smem_base + kSmemAOff + stage * kD3ABytes,
                            SWAP ? &w_tmap : &x_tmap,
                            k * kD3BK, mn.x * kD3BM, &tma_mbars[stage]);
                }
                // In SWAP, B is x; otherwise B is the prefetched weight.
                if (!prefetched || SWAP)
                    ptx::cp_async_bulk_tensor_2d_load(
                        smem_base + kSmemBOff + stage * kD3BBytes,
                        SWAP ? &x_tmap : &w_tmap,
                        k * kD3BK, mn.y * kD3BN, &tma_mbars[stage]);
                if (++stage == kD3NS) { stage = 0; mma_phase ^= 1; }
            }
        }
    } else if (warp_id == 1 && ptx::elect_one()) {
        // MMA issuer with 2-stage TMEM ping-pong
        int stage = 0, tma_phase = 0, ml_stage = 0, epi_phase = 1;
        for (int t = bid; t < kTiles; t += num_bids) {
            ptx::mbar_wait_parity(&epi_mbars[ml_stage], epi_phase);
            const uint32_t tmem_d = taddr + uint32_t(ml_stage) * kD3BN;
            for (int k = 0; k < kIters; ++k) {
                ptx::mbar_wait_parity(&tma_mbars[stage], tma_phase);
                ptx::tcgen05_fence_after_thread_sync();
                const uint32_t a_smem = smem_base + kSmemAOff + stage * kD3ABytes;
                const uint32_t b_smem = smem_base + kSmemBOff + stage * kD3BBytes;
#pragma unroll
                for (int k2 = 0; k2 < kD3KPer; ++k2) {
                    const uint64_t da = ptx::mma_smem_desc_k_major<uint16_t, kD3BK, 128>(
                        a_smem + uint32_t(k2) * 32);
                    const uint64_t db = ptx::mma_smem_desc_k_major<uint16_t, kD3BK, 128>(
                        b_smem + uint32_t(k2) * 32);
                    ptx::tcgen05_mma_f16(tmem_d, da, db, i_desc,
                                         (k == 0 && k2 == 0) ? 0u : 1u);
                }
                ptx::tcgen05_commit_arrive(&mma_mbars[stage]);
                if (++stage == kD3NS) { stage = 0; tma_phase ^= 1; }
            }
            ptx::tcgen05_commit_arrive(&mainloop_mbars[ml_stage]);
            ml_stage ^= 1;
            if (ml_stage == 0) epi_phase ^= 1;
        }
    } else if (warp_id >= 4) {
        // epilogue: BF16 drain (dense_1cta idiom) → coalesced comm push
        const int epi_warp = warp_id & 3;
        const uint32_t taddr_lane = uint32_t(epi_warp * 32) << 16;
        int ml_stage = 0, ml_phase = 0;
        for (int t = bid; t < kTiles; t += num_bids) {
            const int2 mn = tile_mn(t);
            ptx::mbar_wait_parity(&mainloop_mbars[ml_stage], ml_phase);
            ptx::tcgen05_fence_after_thread_sync();
            const uint32_t tmem_d_base = taddr + uint32_t(ml_stage) * kD3BN;
            const int row = mn.x * kD3BM + epi_warp * 32 + lane_id;   // SWAP: n
#pragma unroll 4
            for (int nb = 0; nb < kNBlk; ++nb) {
                const uint32_t taddr_n = tmem_d_base + uint32_t(nb) * 8 + taddr_lane;
                uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                ptx::tcgen05_ld_32x32b_x8(taddr_n, r0, r1, r2, r3, r4, r5, r6, r7);
                ptx::tcgen05_wait_ld();
                uint4 v;
                v.x = ptx::cvt_pack_f32x2_to<ptx::bf16>(__int_as_float(r1), __int_as_float(r0));
                v.y = ptx::cvt_pack_f32x2_to<ptx::bf16>(__int_as_float(r3), __int_as_float(r2));
                v.z = ptx::cvt_pack_f32x2_to<ptx::bf16>(__int_as_float(r5), __int_as_float(r4));
                v.w = ptx::cvt_pack_f32x2_to<ptx::bf16>(__int_as_float(r7), __int_as_float(r6));
                if (SWAP || row < M) {               // SWAP masks pad m-cols below
                    {
                        const size_t off = sb + d3_slot_off(t, nb, epi_warp, lane_id, kNBlk);
                        if constexpr (k2S)
                            // RS: unicast to tile-owner only — 1x egress and,
                            // spread over the tile loop, absorbed under the
                            // mainloop (O9b: one-shot is R x-payload INGRESS-
                            // bound; the composite pays its RS serially)
                            *reinterpret_cast<uint4*>(
                                prm.uc_base[t % R] + off) = v;
                        else {
#pragma unroll
                            for (int r = 0; r < R; ++r)
                                *reinterpret_cast<uint4*>(prm.uc_base[r] + off) = v;
                        }
                    }
                }
            }
            (void)ptx::mbar_arrive(&epi_mbars[ml_stage]);
            ml_stage ^= 1;
            if (ml_stage == 0) ml_phase ^= 1;
        }
    }
    __syncthreads();
    pdl_launch_dependents();
    if (warp_id == 1) {
        ptx::tcgen05_dealloc(taddr, kTmemCols);
        ptx::tcgen05_relinquish();
    }


    // ---- boundary (§H, fam rings) — verbatim member-1 contract ------------
    if (tid == 0) {
        const uint32_t old = atomic_add_acq_rel_gpu(gather_fam + ring, 1);
        if (old + 1 == gath_target) {
            fence_release_sys();
            {
#pragma unroll
                for (int r = 0; r < R; ++r)
                    red_add_relaxed_sys(reinterpret_cast<uint32_t*>(
                        prm.uc_base[r] + foff) + ring, 1);
            }
        }
        while (load_acquire_sys(flag_local) < flag_target) { }
    }
    __syncthreads();

    if constexpr (k2S) {
        // ---- owner-reduce + AG: reduce MY tiles, store the result to all
        // replicas' pull region (kTwoShot: one mm.st, fabric replicates;
        // kTwoShotPeer: R unicast stores — (R-1)× the egress, same ingress);
        // then boundary 2 gates the out-copy ---------------------------------
        for (int u = tid + bid * kD3Threads; u < kTiles * kNBlk * 4 * 32;
             u += num_bids * kD3Threads) {
            const int l = u & 31, w = (u >> 5) & 3, nb = (u >> 7) % kNBlk;
            const int t = u / (kNBlk * 128);
            if (t % R != prm.my_rank) continue;          // not my slab
            const int2 mn = tile_mn(t);
            if (!SWAP && mn.x * kD3BM + w * 32 + l >= M) continue;
            const size_t soff = d3_slot_off(t, nb, w, l, kNBlk);
            float2 acc2[4] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint4 vv = *reinterpret_cast<const uint4*>(
                    prm.uc_base[prm.my_rank] + slot_off(parity, r, R) + soff);
                const uint32_t w4[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(&w4[j]));
                    acc2[j].x += f.x; acc2[j].y += f.y;
                }
            }
            const uint4 res = make_uint4(bf2_u32(acc2[0]), bf2_u32(acc2[1]),
                                         bf2_u32(acc2[2]), bf2_u32(acc2[3]));
            {
#pragma unroll
                for (int r = 0; r < R; ++r)
                    *reinterpret_cast<uint4*>(
                        prm.uc_base[r] + pull_off(parity, R) + soff) = res;
            }
        }
        // boundary 2 (second flag/gather ring words at +256 B / +kRing)
        __syncthreads();
        if (tid == 0) {
            uint32_t* const g2 = gather_fam + kRing;       // AG gather ring
            const uint32_t old = atomic_add_acq_rel_gpu(g2 + ring, 1);
            if (old + 1 == gath_target) {
                fence_release_sys();
                {
#pragma unroll
                    for (int r = 0; r < R; ++r)
                        red_add_relaxed_sys(reinterpret_cast<uint32_t*>(
                            prm.uc_base[r] + foff + 256) + ring, 1);
                }
            }
            while (load_acquire_sys(reinterpret_cast<uint32_t*>(
                prm.uc_base[prm.my_rank] + foff + 256) + ring) < flag_target) { }
        }
        __syncthreads();
        // out-copy: every CTA writes its grid-partition of out from the
        // LOCAL reduced replica
        for (int u = tid + bid * kD3Threads; u < kTiles * kNBlk * 4 * 32;
             u += num_bids * kD3Threads) {
            const int l = u & 31, w = (u >> 5) & 3, nb = (u >> 7) % kNBlk;
            const int t = u / (kNBlk * 128);
            const int2 mn = tile_mn(t);
            const int row = mn.x * kD3BM + w * 32 + l;   // SWAP: row = n, full
            if (!SWAP && row >= M) continue;             // range; m masked below
            const uint4 res = *reinterpret_cast<const uint4*>(
                prm.uc_base[prm.my_rank] + pull_off(parity, R)
                + d3_slot_off(t, nb, w, l, kNBlk));
            if constexpr (SWAP) {
                const uint32_t u4[4] = {res.x, res.y, res.z, res.w};
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int mm = mn.y * kD3BN + nb * 8 + j;
                    if (mm < M)
                        *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(prm.out)
                            + (size_t(mm) * kN + row) * 2) =
                            uint16_t((u4[j >> 1] >> ((j & 1) * 16)) & 0xFFFFu);
                }
            } else {
                *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(prm.out)
                    + (size_t(row) * kN + size_t(mn.y) * kD3BN + nb * 8) * 2) = res;
            }
        }
        __syncthreads();
        if (tid == 0) {
            const size_t off = size_t(ring) * kMaxCTA + blockIdx.x;
            {
#pragma unroll
                for (int r = 0; r < R; ++r)
                    red_add_relaxed_sys(reinterpret_cast<uint32_t*>(
                        prm.uc_base[r] + doff2) + off, 1);
            }
        }
        return;
    }

    // ---- reduce: descatter drain-order slots → out[m][n] -------------------
    // unit = (t, nb, w, l): row = mn.x*BM + w*32 + l; cols = mn.y*BN + nb*8.
    const int units = kTiles * kNBlk * 4 * 32;
    for (int u = tid + bid * kD3Threads; u < units; u += num_bids * kD3Threads) {
        const int l = u & 31, w = (u >> 5) & 3, nb = (u >> 7) % kNBlk;
        const int t = u / (kNBlk * 128);
        const int2 mn = tile_mn(t);
        const int row = mn.x * kD3BM + w * 32 + l;
        if (!SWAP && row >= M) continue;
        const size_t soff = d3_slot_off(t, nb, w, l, kNBlk);
        uint4 res;
        {
            float2 acc2[4] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint4 vv = *reinterpret_cast<const uint4*>(
                    prm.uc_base[prm.my_rank] + slot_off(parity, r, R) + soff);
                const uint32_t w4[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(&w4[j]));
                    acc2[j].x += f.x; acc2[j].y += f.y;
                }
            }
            res = make_uint4(bf2_u32(acc2[0]), bf2_u32(acc2[1]),
                             bf2_u32(acc2[2]), bf2_u32(acc2[3]));
        }
        if constexpr (SWAP) {
            const uint32_t u4[4] = {res.x, res.y, res.z, res.w};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int mm = mn.y * kD3BN + nb * 8 + j;
                if (mm < M)
                    *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(prm.out)
                        + (size_t(mm) * kN + row) * 2) =
                        uint16_t((u4[j >> 1] >> ((j & 1) * 16)) & 0xFFFFu);
            }
        } else {
            *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(prm.out)
                + (size_t(row) * kN + size_t(mn.y) * kD3BN + nb * 8) * 2) = res;
        }
    }

    // ---- done publish (fence-free beacon, W4) ------------------------------
    __syncthreads();
    if (tid == 0) {
        const size_t off = size_t(ring) * kMaxCTA + blockIdx.x;
        {
#pragma unroll
            for (int r = 0; r < R; ++r)
                red_add_relaxed_sys(reinterpret_cast<uint32_t*>(
                    prm.uc_base[r] + doff2) + off, 1);
        }
    }
}

template <int M, int K, int R, Comm COMM>
struct Launcher3 {
    static constexpr bool kSwap = (M <= 64);
    static constexpr int Mp = kSwap ? (M < 16 ? 16 : ((M + 15) & ~15))
                                    : (M + kD3BM - 1) / kD3BM * kD3BM;
    static constexpr int kBN = kSwap ? (Mp <= 32 ? 16 : 32) : d3_bn(Mp);
    static constexpr int kTiles = kSwap ? (kN / kD3BM) * (Mp / kBN)
                                        : (Mp / kD3BM) * (kN / kBN);
    static constexpr int kGrid = kTiles < 152 ? kTiles : 152;
    static constexpr size_t kSmem =
        size_t(d3_ns(kBN)) * (kD3ABytes + kBN * kD3BK * 2);
    static void set_smem_attr() {
        CUDA_CHECK(cudaFuncSetAttribute(oproj_dense_ar_kernel<M, K, R, COMM>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        int(kSmem)));
        CUDA_CHECK(cudaFuncSetAttribute(oproj_dense_ar_kernel<M, K, R, COMM>,
                                        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    }
    static void launch(const CUtensorMap& x_tmap, const CUtensorMap& w_tmap,
                       const Params<R>& prm, cudaStream_t stream, bool pdl) {
        cudaLaunchConfig_t cfg{};
        cudaLaunchAttribute attr[1];
        int na = 0;
        if (pdl) {
            attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attr[na].val.programmaticStreamSerializationAllowed = 1;
            ++na;
        }
        cfg.gridDim = dim3(unsigned(kGrid));
        cfg.blockDim = dim3(kD3Threads);
        cfg.dynamicSmemBytes = kSmem;
        cfg.stream = stream;
        cfg.attrs = attr;
        cfg.numAttrs = unsigned(na);
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, oproj_dense_ar_kernel<M, K, R, COMM>,
                                      x_tmap, w_tmap, prm));
    }
};


// ================= kernels/gemm_ar/kernel.cu =================

// ------------------------------------------------------------- launch glue
// Ring geometry per Mp (smem-budget-fit): batching CH k-chunks per stage
// amortizes the per-stage mbar/wake overhead (recipes/dram_mixed_stream_walls
// ring model: tau = B_stage/R_deliv + F/NS); ring depth S divides the fixed
// feed latency F.
// smem ≤ ~113 KB/CTA so the NEXT PDL launch co-resides with this grid's
// boundary spin (2 CTA/SM transiently) — the tail hides under its feed.
// Ring depth shrinks with Mp to stay under the co-residency budget: the
// F/NS solo-feed penalty (~1.9/3.8 us at bs4/bs8) trades against hiding
// their 10/16 us tails.
template <int M, int K, int R, Comm COMM, bool GEMM_ON,
          int S = ((M + 15) & ~15) <= 16 ? 6 : (((M + 15) & ~15) == 32 ? 4 : 3),
          int CH = 2,
          int C = 1>   // cluster-multicast A axis: C=2 REFUTED as-built at
                       // S=3/4 rings (LEDGER R9 — pair-lockstep pacing beats
                       // the halved A bytes); flip here to test siblings
struct Launcher {
    static constexpr int Mp = (M + 15) & ~15;
    static constexpr size_t kSmem = GEMM_ON
        ? size_t(S) * CH * (kBNRows * kBK * 2 + Mp * kBK * 2) + 2 * S * sizeof(uint64_t)
        : 4096;   // AR-only path never touches the feed ring
    // once per device, before first launch
    static void set_smem_attr() {
        CUDA_CHECK(cudaFuncSetAttribute(oproj_ar_kernel<M, K, R, COMM, GEMM_ON, S, CH, C>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        int(kSmem)));
        // 100% carveout → the SM smem config fits TWO CTAs (this grid's tail
        // + the next PDL grid's feed); the default config blocks dual
        // residency and with it the whole tail-hiding scheme.
        CUDA_CHECK(cudaFuncSetAttribute(oproj_ar_kernel<M, K, R, COMM, GEMM_ON, S, CH, C>,
                                        cudaFuncAttributePreferredSharedMemoryCarveout,
                                        100));
    }
    // pdl=false = the NON-COOPERATIVE-neighbor regime: no programmatic
    // serialization attribute, so successive kernels fully serialize — the
    // AR tail is exposed, as it is in a serving stack whose adjacent kernels
    // don't PDL-cooperate or can't co-reside. In-kernel griddepcontrol ops
    // are no-ops without the attribute; the done-guard is trivially met.
    static void launch(const CUtensorMap& w_map, const CUtensorMap& x_map,
                       const Params<R>& prm, int ncta, cudaStream_t stream,
                       bool pdl) {
        cudaLaunchConfig_t cfg{};
        cudaLaunchAttribute attr[2];
        int na = 0;
        if (pdl) {
            attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attr[na].val.programmaticStreamSerializationAllowed = 1;
            ++na;
        }
        if constexpr (C > 1) {
            attr[na].id = cudaLaunchAttributeClusterDimension;
            attr[na].val.clusterDim.x = C;
            attr[na].val.clusterDim.y = 1;
            attr[na].val.clusterDim.z = 1;
            ++na;
        }
        cfg.gridDim = dim3(unsigned(ncta));
        cfg.blockDim = dim3(kThreads);
        cfg.dynamicSmemBytes = kSmem;
        cfg.stream = stream;
        cfg.attrs = attr;
        cfg.numAttrs = unsigned(na);
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, oproj_ar_kernel<M, K, R, COMM, GEMM_ON, S, CH, C>,
                                      w_map, x_map, prm));
    }
};

}  // namespace oproj_ar

// ================= sglang tvm-ffi adapter =================

#include <sgl_kernel/tensor.h>   // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>    // For CHECK_HOST
#include <sgl_kernel/utils.cuh>  // For bf16_t, TVMFFIEnvGetStream

#include <tvm/ffi/container/tensor.h>

#include <array>
#include <mutex>
#include <unordered_map>

namespace oproj_ar_ffi {

using namespace oproj_ar;
using tvm::ffi::TensorView;

constexpr int kCellList[7] = {8, 16, 32, 64, 128, 256, 512};

inline int cell_of(int m) {
  for (int c : kCellList)
    if (m <= c) return c;
  return -1;
}

template <int K, int R, bool kUsePDL>
struct GemmArKernel {
  static_assert(R >= 2 && R <= 8, "R outside the validated 2..8 range");
  static_assert(K % 128 == 0 && K >= 128, "K must be a multiple of 128");

  static constexpr int kTwoShotMinM = R >= 8 ? 128 : 256;

  // one ring family per dispatch cell (see kFams)
  static int64_t fam_of(int64_t m) {
    const int cell = cell_of(int(m));
    CHECK_HOST(cell > 0) << "gemm_ar: M=" << m << " outside [1, " << kMMax << "]";
    for (int i = 0; i < 7; ++i)
      if (kCellList[i] == cell) return i;
    return -1;
  }

  static int64_t cell_of_ffi(int64_t m) { return cell_of(int(m)); }
  static int64_t region_nbytes() { return int64_t(region_bytes(R)); }
  static int64_t gather_words() { return int64_t(kFams) * 2 * kRing; }
  static int64_t num_fams() { return kFams; }
  static int64_t max_tokens() { return kMMax; }

  // Per-weight-pointer W tensor maps (encode once; weights are static).
  struct WMaps {
    CUtensorMap w48, w64, w128, w256;
  };

  static const WMaps& w_maps(void* w) {
    static std::unordered_map<void*, WMaps> cache;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(w);
    if (it == cache.end()) {
      auto enc = [&](uint32_t box_rows) {
        return tmap::encode_tiled_2d(
            w, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, kN, K, size_t(K) * 2, box_rows,
            kBK, CU_TENSOR_MAP_SWIZZLE_128B);
      };
      it = cache.emplace(w, WMaps{enc(kBNRows), enc(64), enc(128), enc(256)}).first;
    }
    return it->second;
  }

  // x tensor map over the caller's [M, K] tensor: global rows = M, TMA
  // zero-fills the [M, cell) padding rows out-of-bounds.
  static CUtensorMap x_map(void* x, int m, uint32_t box_rows) {
    return tmap::encode_tiled_2d(
        x, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, uint64_t(m), K, size_t(K) * 2,
        box_rows, kBK, CU_TENSOR_MAP_SWIZZLE_128B);
  }

  static void set_smem_attrs_once() {
    static bool done = [] {
      Launcher<8, K, R, Comm::kPeer, true>::set_smem_attr();
      Launcher<16, K, R, Comm::kPeer, true>::set_smem_attr();
      Launcher3<32, K, R, Comm::kPeer>::set_smem_attr();
      Launcher3<64, K, R, Comm::kPeer>::set_smem_attr();
      if constexpr (kTwoShotMinM > 128) Launcher3<128, K, R, Comm::kPeer>::set_smem_attr();
      if constexpr (kTwoShotMinM <= 128) Launcher3<128, K, R, Comm::kTwoShotPeer>::set_smem_attr();
      Launcher3<256, K, R, Comm::kTwoShotPeer>::set_smem_attr();
      Launcher3<512, K, R, Comm::kTwoShotPeer>::set_smem_attr();
      return true;
    }();
    (void)done;
  }

  template <int CELL>
  static void enqueue_cell(const WMaps& wm, void* x, int m, const Params<R>& prm,
                           cudaStream_t stream, bool pdl) {
    if constexpr (CELL <= 16) {
      const CUtensorMap xm = x_map(x, m, 16);
      Launcher<CELL, K, R, Comm::kPeer, true>::launch(wm.w48, xm, prm, kM1Grid_(), stream, pdl);
    } else if constexpr (CELL < kTwoShotMinM) {
      const CUtensorMap xm = x_map(x, m, CELL <= 32 ? 16 : (CELL <= 64 ? 32 : 128));
      const CUtensorMap& wmap = CELL <= 64 ? wm.w128 : wm.w64;
      Launcher3<CELL, K, R, Comm::kPeer>::launch(xm, wmap, prm, stream, pdl);
    } else {
      const CUtensorMap xm = x_map(x, m, 128);
      const CUtensorMap& wmap = CELL == 128 ? wm.w64 : (CELL == 256 ? wm.w128 : wm.w256);
      Launcher3<CELL, K, R, Comm::kTwoShotPeer>::launch(xm, wmap, prm, stream, pdl);
    }
  }

  static constexpr int kM1Grid_() { return 152; }

  // per-rank UC VAs of the comm region, stashed host-side ONCE at init:
  // per-call CPU-tensor derefs from inside the op are not reliable in every
  // execution context (observed dangling under the sglang scheduler).
  static std::array<uint8_t*, R>& bases_store() {
    static std::array<uint8_t*, R> a{};
    return a;
  }

  static void set_bases(TensorView uc_bases) {
    using namespace host;
    TensorMatcher({R}).with_dtype<int64_t>().verify(uc_bases);
    const int64_t* b = static_cast<const int64_t*>(uc_bases.data_ptr());
    for (int r = 0; r < R; ++r) bases_store()[r] = reinterpret_cast<uint8_t*>(b[r]);
  }

  static void run(
      TensorView out,
      TensorView x,
      TensorView w,
      TensorView gather,    // [kFams * 2 * kRing] int32 CUDA, device-local
      TensorView epochs,    // [kFams] int32 CUDA: device-resident CTA ticket counters
      int64_t my_rank) {
    using namespace host;

    auto M = SymbolicSize{"num_tokens"};
    auto CellRows = SymbolicSize{"cell_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({kN, K}).with_dtype<bf16_t>().with_device(device).verify(w);
    TensorMatcher({CellRows, kN}).with_dtype<bf16_t>().with_device(device).verify(out);

    const int m = int(M.unwrap());
    const int cell = cell_of(m);
    CHECK_HOST(cell > 0) << "gemm_ar: M=" << m << " outside [1, " << kMMax << "]";
    CHECK_HOST(int64_t(CellRows.unwrap()) == cell)
        << "out must have cell(M)=" << cell << " rows, got " << CellRows.unwrap();
    CHECK_HOST(my_rank >= 0 && my_rank < R);
    CHECK_HOST(bases_store()[0] != nullptr) << "gemm_ar: set_bases not called";
    TensorMatcher({int64_t(kFams) * 2 * kRing}).with_dtype<int32_t>().verify(gather);
    TensorMatcher({kFams}).with_dtype<int32_t>().verify(epochs);

    set_smem_attrs_once();

    const DLDevice dev = device.unwrap();
    const auto stream =
        static_cast<cudaStream_t>(::TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    Params<R> prm{};
    prm.mc_base = nullptr;  // pure-P2P plane
    for (int r = 0; r < R; ++r) prm.uc_base[r] = bases_store()[r];
    prm.gather = static_cast<uint32_t*>(gather.data_ptr());
    prm.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    prm.partial_in = nullptr;
    prm.epoch_base = static_cast<uint32_t*>(epochs.data_ptr());
    prm.my_rank = int(my_rank);
    prm.fam = int(fam_of(m));

    const bool pdl = kUsePDL;
    const WMaps& wm = w_maps(w.data_ptr());
    switch (cell) {
      case 8:   enqueue_cell<8>(wm, x.data_ptr(), m, prm, stream, pdl); break;
      case 16:  enqueue_cell<16>(wm, x.data_ptr(), m, prm, stream, pdl); break;
      case 32:  enqueue_cell<32>(wm, x.data_ptr(), m, prm, stream, pdl); break;
      case 64:  enqueue_cell<64>(wm, x.data_ptr(), m, prm, stream, pdl); break;
      case 128: enqueue_cell<128>(wm, x.data_ptr(), m, prm, stream, pdl); break;
      case 256: enqueue_cell<256>(wm, x.data_ptr(), m, prm, stream, pdl); break;
      default:  enqueue_cell<512>(wm, x.data_ptr(), m, prm, stream, pdl); break;
    }
    CHECK_CUDA(cudaGetLastError()) << "gemm_ar launch (cell=" << cell << ")";
  }
};

}  // namespace oproj_ar_ffi

using oproj_ar_ffi::GemmArKernel;
