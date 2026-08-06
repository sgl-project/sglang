#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include "../../distributed/custom_all_reduce.cuh"
#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdint>
#include <utility>

// Local PTX primitives (mbarrier / bulk TMA / tcgen05 / warp-group sync)

namespace ptx {

// ---- bulk 1D TMA (PTX ISA §9.7.9.25) ---------------------------------------

// global -> shared::cluster, completed by an smem mbarrier. Arm `bar` with
// `mbar_arrive_expect_tx(bar, bytes)` before issuing; `bytes` and both
// endpoints must be 16-byte aligned.
static SGL_DEVICE void cp_async_bulk_1d_load(void* smem_dst, const void* gmem_src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];" ::"r"(to_shared(smem_dst)),
      "l"(gmem_src),
      "r"(bytes),
      "r"(to_shared(bar))
      : "memory");
}

// Publish mbarrier initialization from the generic proxy before an async engine
// uses the barrier.
static SGL_DEVICE void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;");
}

// ---- warp / warp-group sync (PTX ISA §9.7.4, §9.7.12.6, §9.7.13) -----------

// Partial-CTA rendezvous. `id` must be in [1, 15]; barrier 0 is reserved for
// the full-CTA barrier behind __syncthreads().
static SGL_DEVICE void named_barrier_sync(uint32_t id, uint32_t num_threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(num_threads) : "memory");
}

// True on exactly one lane of the issuing warp — guards single-issuer sites
// (mbar init, TMA issue, MMA issue, TMEM alloc) without gating on lane_id.
static SGL_DEVICE bool elect_one() {
  uint32_t pred;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "elect.sync _|p, 0xffffffff;\n\t"
      "selp.b32 %0, 1, 0, p;\n\t}\n"
      : "=r"(pred));
  return pred != 0;
}

// Runtime warp-group register-budget reallocation: widen the epilogue's
// per-thread budget (so it holds a larger primary array without spilling) by
// narrowing the mainloop warps, which need few registers.
//
// Both forms are `.sync.aligned`: all 128 threads of the issuing warp-group
// must execute the SAME instruction with the SAME N, from a warp-group
// boundary (issuing from only one warp of the group hangs). N in [24, 256],
// multiple of 8, per thread; the CTA total must satisfy
// sum(warp_group_threads * N) <= 64512 (the safe allocatable cap on B100/B300
// after ~1024 reserved regs). Caller owns the budgeting — there is no
// compile-time check, since N per warp-group is orthogonal.
//
// This pays only for ASYMMETRIC budgets that ptxas cannot infer from the
// source. For a symmetric cap, `__launch_bounds__(NUM_THREADS, 1)` is cleaner
// and measured faster on B100/B300.
template <int N>
static SGL_DEVICE void setmaxnreg_dec() {
  static_assert(N >= 24 && N <= 256, "setmaxnreg N must be in [24, 256]");
  static_assert((N & 7) == 0, "setmaxnreg N must be a multiple of 8");
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(N));
}

template <int N>
static SGL_DEVICE void setmaxnreg_inc() {
  static_assert(N >= 24 && N <= 256, "setmaxnreg N must be in [24, 256]");
  static_assert((N & 7) == 0, "setmaxnreg N must be a multiple of 8");
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(N));
}

// ---- tcgen05 (PTX ISA §9.7.16) ---------------------------------------------
//
// Lifecycle (mandatory order, §9.7.16.7.1): alloc (one warp, n_cols a power of
// 2 in [32, 512], TMEM address written to smem) -> __syncthreads + read taddr
// -> ld/st -> dealloc -> relinquish before kernel exit.
//
// Each warp can only touch its own 32-lane TMEM band (§9.7.16.8.1): warp 0 ->
// lanes 0-31, warp 1 -> 32-63, and so on. Use `tcgen05_wait_st` /
// `tcgen05_wait_ld` before consuming the other side of a store / drain.
static SGL_DEVICE void tcgen05_alloc(uint32_t smem_addr_for_taddr, uint32_t n_cols) {
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_addr_for_taddr), "r"(n_cols));
}

static SGL_DEVICE void tcgen05_dealloc(uint32_t taddr, uint32_t n_cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(n_cols));
}

static SGL_DEVICE void tcgen05_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}

// .32x32b.x8: 8 b32 per lane = 8 TMEM columns. Per-lane 8 FP32 -> 4 bf16x2
// packs = one int4, the natural fit for a BF16 epilogue moving a column band
// with 16-byte smem accesses.
static SGL_DEVICE void tcgen05_ld_32x32b_x8(
    uint32_t taddr,
    uint32_t& r0,
    uint32_t& r1,
    uint32_t& r2,
    uint32_t& r3,
    uint32_t& r4,
    uint32_t& r5,
    uint32_t& r6,
    uint32_t& r7) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
      " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7)
      : "r"(taddr));
}

static SGL_DEVICE void tcgen05_ld_32x32b_x8(uint32_t taddr, uint32_t* dst) {
  tcgen05_ld_32x32b_x8(taddr, dst[0], dst[1], dst[2], dst[3], dst[4], dst[5], dst[6], dst[7]);
}

static SGL_DEVICE void tcgen05_st_32x32b_x8(uint32_t taddr, const uint32_t* src) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x8.b32 "
      " [%8], {%0, %1, %2, %3, %4, %5, %6, %7};"
      :
      : "r"(src[0]),
        "r"(src[1]),
        "r"(src[2]),
        "r"(src[3]),
        "r"(src[4]),
        "r"(src[5]),
        "r"(src[6]),
        "r"(src[7]),
        "r"(taddr));
}

static SGL_DEVICE void tcgen05_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

}  // namespace ptx

namespace sglang {

struct AttnResTMAParams {
  const bf16_t* __restrict__ prefix_sum;  // [T, H]
  const bf16_t* __restrict__ bank;        // [T, NB_total, H]
  const bf16_t* __restrict__ cw;          // [H] score norm * proj weight
  const bf16_t* __restrict__ ow;          // [H] out norm weight
  bf16_t* __restrict__ out;               // [T, H]
  // Fused bank write (nullptr = off): per-token destination of the prefix
  // row snapshot, bank row nvb (strided by stride_bm like the read rows).
  // The kernel never reads row nvb, so the write races with nothing.
  bf16_t* __restrict__ prefix_dst;
  // Optional fused NVLS reduce-scatter source. When input_mc is non-null,
  // the producer warp materializes this rank's reduced token shard plus the
  // local residual into prefix_sum/prefix_out before TMA consumes it.
  const uint8_t* input_mc;
  const bf16_t* residual;
  bf16_t* prefix_out;
  device::distributed::Semaphore* sem_local;
  uint8_t* sem_mc;
  uint8_t* output_mc;
  uint32_t world_size;
  uint32_t rank;
  int64_t stride_bm;  // bank stride along T (in elements)
  float eps;
  uint32_t num_tokens;
};

template <int64_t kDim_, uint32_t kNumBankRows_, uint32_t kChunkRows_, uint32_t kConsumerRegs_ = 0>
struct KimiK3AttnResTrait {
 public:
  static constexpr int64_t kDim = kDim_;
  static constexpr int64_t kTile = 1024;               // one warp-group-wide 16B sweep
  static constexpr uint32_t kNumRows = kNumBankRows_;  // bank rows; +1 prefix row
  static constexpr uint32_t kChunkRows = kChunkRows_;  // rows per chunk (one barrier pair per chunk)
  // Chunk slots in the smem ring. Frozen at 2 (double buffering): 1 stalls
  // the producer behind the consumers (~10% slower), >2 gains nothing and
  // costs smem at small T.
  static constexpr uint32_t kNumStages = 2;
  static constexpr uint32_t kNumChunks = (kNumRows + 1 + kChunkRows - 1) / kChunkRows;
  static constexpr uint32_t kNumConsumerWarps = 8;
  static constexpr uint32_t kConsumerRegs = kConsumerRegs_;
  static constexpr uint32_t kProducerRegs = 40;
  static constexpr uint32_t kNumProducerWarps = kConsumerRegs > 0 ? 4 : 1;
  static constexpr uint32_t kNumWarps = kNumConsumerWarps + kNumProducerWarps;
  static constexpr uint32_t kNumThreads = kNumWarps * device::kWarpThreads;
  static constexpr uint32_t kNumConsumerThreads = kNumConsumerWarps * device::kWarpThreads;
  static_assert(
      kConsumerRegs == 0 || (kConsumerRegs % 8 == 0 && 24 <= kConsumerRegs && kConsumerRegs <= 256 &&
                             2 * kConsumerRegs + kProducerRegs <= 512),
      "consumer register budget exceeds the SM sub-partition file");

  // Consumer tiling (v1 layout): two 128-thread warp groups; group g owns
  // tiles g, g + 2, ... of the row; each thread owns one 16B vector per tile.
  static constexpr uint32_t kNumGroups = 2;
  static constexpr uint32_t kGroupThreads = kNumConsumerThreads / kNumGroups;
  static constexpr uint32_t kVecElems = 16 / sizeof(bf16_t);  // smem ld/st are 16B max
  static constexpr uint32_t kNumTiles = kDim / kTile;
  static constexpr uint32_t kSlicesPerGroup = (kNumTiles + kNumGroups - 1) / kNumGroups;
  static constexpr uint32_t kAccPerThread = kSlicesPerGroup * kVecElems;

  // TMEM: per group, kTmemColsPerGroup columns of cw then of ow.
  static constexpr uint32_t kTmemColsPerGroup = 32;
  static constexpr uint32_t kTmemCols = 2 * kNumGroups * kTmemColsPerGroup;
  static constexpr uint32_t kConsumerBarId = 1;  // barrier 0 stays __syncthreads'

  static_assert(kDim % kTile == 0, "kDim must be a whole number of tiles");
  static_assert(kTile == kGroupThreads * kVecElems, "a tile is one group-wide 16B sweep");
  static_assert(kNumTiles <= kNumGroups * kSlicesPerGroup, "slices must cover all tiles");
  static_assert(kSlicesPerGroup * kVecElems <= kTmemColsPerGroup, "weight slices must fit their TMEM columns");
  static_assert(kNumRows >= 1, "need at least one bank row");
  static_assert(kChunkRows >= 1, "need at least one chunk row");

  struct Smem {
    uint64_t bar_full[kNumStages];
    uint64_t bar_free[kNumStages];
    float warp_rms[kNumConsumerWarps][kChunkRows];
    float warp_dot[kNumConsumerWarps][kChunkRows];
    // The out-norm reduction gets its own buffer: it can overlap the next
    // token's first score reduction.
    float warp_ssq[kNumConsumerWarps];
    uint32_t tmem_base;
    alignas(128) bf16_t buf[kNumStages][kChunkRows][kDim];
  };

  static SGL_DEVICE void forward(const AttnResTMAParams& params, Smem* smem);
};

SGL_DEVICE float2 fma_f32x2(float2 a, float2 b, float2 c) {
  const uint64_t a_bits = reinterpret_cast<const uint64_t&>(a);
  const uint64_t b_bits = reinterpret_cast<const uint64_t&>(b);
  const uint64_t c_bits = reinterpret_cast<const uint64_t&>(c);
  uint64_t result;
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(result) : "l"(a_bits), "l"(b_bits), "l"(c_bits));
  return reinterpret_cast<const float2&>(result);
}

SGL_DEVICE float2 mul_f32x2(float2 a, float2 b) {
  const uint64_t a_bits = reinterpret_cast<const uint64_t&>(a);
  const uint64_t b_bits = reinterpret_cast<const uint64_t&>(b);
  uint64_t result;
  asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(result) : "l"(a_bits), "l"(b_bits));
  return reinterpret_cast<const float2&>(result);
}

template <int64_t kDim_, uint32_t kNumBankRows_, uint32_t kChunkRows_, uint32_t kConsumerRegs_>
SGL_DEVICE void KimiK3AttnResTrait<kDim_, kNumBankRows_, kChunkRows_, kConsumerRegs_>::forward(
    const AttnResTMAParams& params, Smem* smem) {
  using namespace device;
  using row_vec_t = AlignedVector<bf16x2_t, kVecElems / 2>;  // 16 bytes
  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;

  if (warp_id == 0 && lane_id < kNumStages) {
    ::ptx::mbar_init(&smem->bar_full[lane_id], 1);
    ::ptx::mbar_init(&smem->bar_free[lane_id], kNumConsumerWarps * kWarpThreads);
    ::ptx::fence_mbarrier_init();
  } else if (warp_id == 1) {
    ::ptx::tcgen05_alloc(::ptx::to_shared(&smem->tmem_base), kTmemCols);
    ::ptx::tcgen05_relinquish();
  }

  __syncthreads();
  if (warp_id >= kNumConsumerWarps) {  // producer warp (group); first warp works
    if constexpr (kConsumerRegs > 0) ::ptx::setmaxnreg_dec<kProducerRegs>();
    // TODO: reduce the register usage
    if (warp_id == kNumConsumerWarps && ::ptx::elect_one()) {
      uint32_t global_chunks = 0;
      constexpr uint32_t kRowBytes = kDim * sizeof(bf16_t);
      for (auto token = blockIdx.x; token < params.num_tokens; token += gridDim.x) {
#pragma unroll
        for (uint32_t ci = 0; ci < kNumChunks; ++ci, ++global_chunks) {
          const uint32_t base_row = ci * kChunkRows;
          const uint32_t an = (kNumRows + 1 - base_row) < kChunkRows ? (kNumRows + 1 - base_row) : kChunkRows;
          const auto slot = global_chunks % kNumStages;
          const auto phase = (global_chunks / kNumStages) & 1;
          if (global_chunks >= kNumStages) {
            ::ptx::mbar_wait_parity(&smem->bar_free[slot], phase ^ 1);
          }
          // One barrier per chunk; each row still gets its own bulk copy.
          ::ptx::mbar_arrive_expect_tx(&smem->bar_full[slot], an * kRowBytes);
#pragma unroll
          for (uint32_t r = 0; r < an; ++r) {
            const auto row = base_row + r;
            const auto src = row == kNumRows ? params.prefix_sum + token * kDim  //
                                             : params.bank + token * params.stride_bm + row * kDim;
            // Only prefix_sum is written by the immediately-preceding kernel;
            // one wait before the first token's prefix load covers the rest.
            if (token == blockIdx.x && row == kNumRows) PDLWaitPrimary<true>();
            ::ptx::cp_async_bulk_1d_load(&smem->buf[slot][r], src, kRowBytes, &smem->bar_full[slot]);
          }
        }
      }
      PDLTriggerSecondary<true>();
    }
  } else {  // 2 consumer warp groups; one chunk per rendezvous
    if constexpr (kConsumerRegs > 0) ::ptx::setmaxnreg_inc<kConsumerRegs>();
    const auto group = warp_id / (kNumConsumerWarps / kNumGroups);
    const auto tid_in_group = tx % kGroupThreads;
    const auto tmem_cw = smem->tmem_base + group * kTmemColsPerGroup;
    const auto tmem_ow = tmem_cw + kNumGroups * kTmemColsPerGroup;

    // Stage this thread's cw / ow slices into TMEM (read once from gmem).
    {
      float staged[kAccPerThread];
#pragma unroll
      for (uint32_t si = 0; si < kSlicesPerGroup; ++si) {
        const auto tile = si * kNumGroups + group;
        if (tile >= kNumTiles) continue;
        const auto h_base = tile * kTile + tid_in_group * kVecElems;
#pragma unroll
        for (uint32_t j = 0; j < kVecElems; ++j) {
          staged[si * kVecElems + j] = __bfloat162float(params.cw[h_base + j]);
        }
      }
#pragma unroll
      for (uint32_t si = 0; si < kSlicesPerGroup; ++si) {
        ::ptx::tcgen05_st_32x32b_x8(
            tmem_cw + si * kVecElems, reinterpret_cast<const uint32_t*>(&staged[si * kVecElems]));
      }
#pragma unroll
      for (uint32_t si = 0; si < kSlicesPerGroup; ++si) {
        const auto tile = si * kNumGroups + group;
        if (tile >= kNumTiles) continue;
        const auto h_base = tile * kTile + tid_in_group * kVecElems;
#pragma unroll
        for (uint32_t j = 0; j < kVecElems; ++j) {
          staged[si * kVecElems + j] = __bfloat162float(params.ow[h_base + j]);
        }
      }
#pragma unroll
      for (uint32_t si = 0; si < kSlicesPerGroup; ++si) {
        ::ptx::tcgen05_st_32x32b_x8(
            tmem_ow + si * kVecElems, reinterpret_cast<const uint32_t*>(&staged[si * kVecElems]));
      }
      ::ptx::tcgen05_wait_st();
    }

    uint32_t global_chunks = 0;  // mirrors the producer's chunk counter
    for (auto token = blockIdx.x; token < params.num_tokens; token += gridDim.x) {
      float run_max = -FLT_MAX;  // online-softmax state
      float run_sum = 0.f;
      float2 acc[kAccPerThread / 2] = {};  // packed fp32x2 accumulator

#pragma unroll
      for (uint32_t ci = 0; ci < kNumChunks; ++ci, ++global_chunks) {
        const uint32_t base_row = ci * kChunkRows;
        // Active rows of this chunk; folds per unrolled iteration.
        const uint32_t an = (kNumRows + 1 - base_row) < kChunkRows ? (kNumRows + 1 - base_row) : kChunkRows;
        const auto slot = global_chunks % kNumStages;
        const auto phase = (global_chunks / kNumStages) & 1;
        ::ptx::mbar_wait_parity(&smem->bar_full[slot], phase);

        // Score pass: the cw slice is loaded once and reused across the
        // chunk's rows; each row's 16B slices land in registers. rms/dot
        // accumulate as packed fp32x2 lanes, folded to scalars just before
        // the warp reduction.
        row_vec_t rows[kSlicesPerGroup][kChunkRows];
        float2 acc_rms2[kChunkRows] = {};
        float2 acc_dot2[kChunkRows] = {};
#pragma unroll
        for (uint32_t si = 0; si < kSlicesPerGroup; ++si) {
          const auto tile = si * kNumGroups + group;
          if (tile >= kNumTiles) continue;
          float q[kVecElems];
          ::ptx::tcgen05_ld_32x32b_x8(tmem_cw + si * kVecElems, reinterpret_cast<uint32_t*>(q));
          const auto* q2 = reinterpret_cast<const float2*>(q);
          const auto offset = tile * kTile + tid_in_group * kVecElems;
#pragma unroll
          for (uint32_t r = 0; r < an; ++r) {
            rows[si][r].load(&smem->buf[slot][r][offset]);
          }
#pragma unroll
          for (uint32_t r = 0; r < an; ++r) {
#pragma unroll
            for (uint32_t j = 0; j < kVecElems / 2; ++j) {
              const auto f = cast<float2>(rows[si][r][j]);
              acc_rms2[r] = fma_f32x2(f, f, acc_rms2[r]);
              acc_dot2[r] = fma_f32x2(f, q2[j], acc_dot2[r]);
            }
          }
        }
        ::ptx::mbar_arrive(&smem->bar_free[slot]);

        // Fused bank write: the prefix row (last row of the last chunk) is
        // already in registers; snapshot it to bank row nvb with plain
        // stores — the .write() copy kernel disappears. Placed after the
        // arrive so the slot handoff is not delayed.
        if (params.prefix_dst != nullptr && base_row + an == kNumRows + 1) {
          const uint32_t pr = kNumRows - base_row;
          auto* dst = params.prefix_dst + static_cast<int64_t>(token) * params.stride_bm;
#pragma unroll
          for (uint32_t si = 0; si < kSlicesPerGroup; ++si) {
            const auto tile = si * kNumGroups + group;
            if (tile >= kNumTiles) continue;
            rows[si][pr].store(dst, tile * (kTile / kVecElems) + tid_in_group);
          }
        }

        float acc_rms[kChunkRows];
        float acc_dot[kChunkRows];
#pragma unroll
        for (uint32_t r = 0; r < an; ++r) {
          acc_rms[r] = acc_rms2[r].x + acc_rms2[r].y;
          acc_dot[r] = acc_dot2[r].x + acc_dot2[r].y;
        }

#pragma unroll
        for (int n = 0; n < an; n++) {
          acc_rms[n] = warp::reduce_sum(acc_rms[n]);
          acc_dot[n] = warp::reduce_sum(acc_dot[n]);
        }
        if (lane_id == 0) {
#pragma unroll
          for (uint32_t r = 0; r < an; ++r) {
            smem->warp_rms[warp_id][r] = acc_rms[r];
            smem->warp_dot[warp_id][r] = acc_dot[r];
          }
        }
        ::ptx::named_barrier_sync(kConsumerBarId, kNumConsumerThreads);
        // Lane r totals row r, then broadcasts: an*16 smem loads per warp
        // instead of per thread.
        float lane_logit = 0.f;
        if (lane_id < an) {
          float total_rms = 0.f;
          float total_dot = 0.f;
#pragma unroll
          for (uint32_t w = 0; w < kNumConsumerWarps; ++w) {
            total_rms += smem->warp_rms[w][lane_id];
            total_dot += smem->warp_dot[w][lane_id];
          }
          constexpr float kScale = 1.f / static_cast<float>(kDim);
          lane_logit = total_dot * rsqrtf(total_rms * kScale + params.eps);
        }
        float logit[kChunkRows];
#pragma unroll
        for (uint32_t r = 0; r < an; ++r) {
          logit[r] = __shfl_sync(0xffffffffu, lane_logit, r);
        }

        // Online-softmax fold of the chunk into the running accumulator.
        float chunk_max = -FLT_MAX;
#pragma unroll
        for (uint32_t r = 0; r < an; ++r) {
          chunk_max = fmaxf(chunk_max, logit[r]);
        }
        const float new_max = fmaxf(run_max, chunk_max);
        const float correction = exp2f((run_max - new_max) * math::log2e);
        float weight[kChunkRows];
        float weight_sum = 0.f;
#pragma unroll
        for (uint32_t r = 0; r < an; ++r) {
          weight[r] = exp2f((logit[r] - new_max) * math::log2e);
          weight_sum += weight[r];
        }
        run_sum = run_sum * correction + weight_sum;
        run_max = new_max;

        // Fold the chunk into the packed accumulator (v1 loop order: scale
        // once, then rows outer / vector lanes inner, all fp32x2 FMAs).
        const float2 correction2 = make_float2(correction, correction);
        float2 weight2[kChunkRows];
#pragma unroll
        for (uint32_t r = 0; r < an; ++r) {
          weight2[r] = make_float2(weight[r], weight[r]);
        }
#pragma unroll
        for (uint32_t si = 0; si < kSlicesPerGroup; ++si) {
          const auto tile = si * kNumGroups + group;
          if (tile >= kNumTiles) continue;
          float2 a[kVecElems / 2];
#pragma unroll
          for (uint32_t j = 0; j < kVecElems / 2; ++j) {
            a[j] = mul_f32x2(acc[si * (kVecElems / 2) + j], correction2);
          }
#pragma unroll
          for (uint32_t r = 0; r < an; ++r) {
#pragma unroll
            for (uint32_t j = 0; j < kVecElems / 2; ++j) {
              a[j] = fma_f32x2(weight2[r], cast<float2>(rows[si][r][j]), a[j]);
            }
          }
#pragma unroll
          for (uint32_t j = 0; j < kVecElems / 2; ++j) {
            acc[si * (kVecElems / 2) + j] = a[j];
          }
        }
      }

      // Fused out norm: mixed = acc / run_sum, out = rmsnorm(mixed) * ow.
      const float inv_sum = 1.f / run_sum;
      float2 acc_sq2 = make_float2(0.f, 0.f);
#pragma unroll
      for (uint32_t j = 0; j < kAccPerThread / 2; ++j) {
        acc_sq2 = fma_f32x2(acc[j], acc[j], acc_sq2);
      }
      float acc_sq = warp::reduce_sum(acc_sq2.x + acc_sq2.y);
      if (lane_id == 0) smem->warp_ssq[warp_id] = acc_sq;
      ::ptx::named_barrier_sync(kConsumerBarId, kNumConsumerThreads);
      float total_sq = 0.f;
#pragma unroll
      for (uint32_t w = 0; w < kNumConsumerWarps; ++w) {
        total_sq += smem->warp_ssq[w];
      }
      const float scale = inv_sum * rsqrtf(total_sq * inv_sum * inv_sum / static_cast<float>(kDim) + params.eps);
      const float2 scale2 = make_float2(scale, scale);

      auto* out_ptr = params.out + static_cast<int64_t>(token) * kDim;
#pragma unroll
      for (uint32_t si = 0; si < kSlicesPerGroup; ++si) {
        const auto tile = si * kNumGroups + group;
        if (tile >= kNumTiles) continue;
        float q[kVecElems];
        ::ptx::tcgen05_ld_32x32b_x8(tmem_ow + si * kVecElems, reinterpret_cast<uint32_t*>(q));
        const auto* q2 = reinterpret_cast<const float2*>(q);
        row_vec_t out_vec;
#pragma unroll
        for (uint32_t j = 0; j < kVecElems / 2; ++j) {
          const auto scaled = mul_f32x2(acc[si * (kVecElems / 2) + j], scale2);
          out_vec[j] = cast<bf16x2_t>(mul_f32x2(scaled, q2[j]));
        }
        const auto row_vid = tile * (kTile / kVecElems) + tid_in_group;
        if (params.output_mc != nullptr) {
          const auto global_token = static_cast<int64_t>(params.rank) * params.num_tokens + token;
          const auto global_vid = global_token * (kDim / kVecElems) + row_vid;
          st_multimem_16B(out_vec, params.output_mc, global_vid);
        } else {
          out_vec.store(out_ptr, row_vid);
        }
      }
    }
    ::ptx::named_barrier_sync(kConsumerBarId, kNumConsumerThreads);
    if (warp_id == 1) {
      ::ptx::tcgen05_dealloc(smem->tmem_base, kTmemCols);
    }
  }
}

// kOccupancy > 1 caps the register budget (65536 / (kOccupancy * kNumThreads))
// so that many CTAs actually co-reside; smem must also fit kOccupancy copies.
template <typename Trait, uint32_t kOccupancy>
__global__ void __launch_bounds__(Trait::kNumThreads, kOccupancy)
    attn_res_fused_tma_kernel(const __grid_constant__ AttnResTMAParams params) {
  extern __shared__ char smem_raw[];
  Trait::forward(params, reinterpret_cast<typename Trait::Smem*>(smem_raw));
}

SGL_DEVICE uint32_t* attn_res_sem_mc_flag(uint8_t* sem_mc, uint32_t block) {
  static_assert(sizeof(device::distributed::Semaphore) == 128);
  return reinterpret_cast<uint32_t*>(sem_mc + block * sizeof(device::distributed::Semaphore));
}

SGL_DEVICE void attn_res_sem_arrive_relaxed(uint32_t* flag) {
#if SGL_ARCH_HOPPER_OR_GREATER
  asm volatile("multimem.red.relaxed.sys.global.add.u32 [%0], 1;" ::"l"(flag) : "memory");
#else
  assert(false && "multimem red requires Hopper or later");
#endif
}

SGL_DEVICE void attn_res_sem_arrive_release(uint32_t* flag) {
#if SGL_ARCH_HOPPER_OR_GREATER
  asm volatile("multimem.red.release.sys.global.add.u32 [%0], 1;" ::"l"(flag) : "memory");
#else
  assert(false && "multimem red requires Hopper or later");
#endif
}

// Fused NVLS pull RS + local residual + attention-residual aggregation.
// The entry/exit barriers make local o_proj writes visible before the
// producer's multimem reduction and preserve the shared pull-semaphore
// protocol used by the neighboring K3 collectives.
template <typename Trait, uint32_t kOccupancy>
__global__ void __launch_bounds__(Trait::kNumThreads, kOccupancy)
    attn_res_fused_pull_rs_kernel(const __grid_constant__ AttnResTMAParams params) {
  __shared__ uint32_t exit_base;
  if (threadIdx.x == 0) {
    auto* semaphore = &params.sem_local[blockIdx.x];
    const auto reserved = semaphore->counter_ptr()->inc(2 * params.world_size);
    exit_base = reserved + params.world_size;
    device::PDLWaitPrimary<true>();
    attn_res_sem_arrive_relaxed(attn_res_sem_mc_flag(params.sem_mc, blockIdx.x));
    while (semaphore->get_relaxed() - reserved < params.world_size)
      ;
  }
  __syncthreads();

  // Cooperative NVLS materialization: all TMA producer + consumer threads
  // participate, so the remote reduction exposes hundreds of outstanding
  // 16-byte loads per CTA instead of serializing the row through one warp.
  using pull_vec_t = device::AlignedVector<bf16x2_t, 4>;
  using SumOp = device::ReductionTrait<device::ReductionOp::SUM, bf16x2_t>;
  constexpr uint32_t kRowVecs = Trait::kDim * sizeof(bf16_t) / sizeof(pull_vec_t);
  for (auto token = blockIdx.x; token < params.num_tokens; token += gridDim.x) {
    auto* prefix = params.prefix_out + static_cast<int64_t>(token) * Trait::kDim;
    const auto* input_mc = params.input_mc + static_cast<int64_t>(token) * Trait::kDim * sizeof(bf16_t);
    const auto* residual =
        params.residual == nullptr ? nullptr : params.residual + static_cast<int64_t>(token) * Trait::kDim;
    for (uint32_t vid = threadIdx.x; vid < kRowVecs; vid += blockDim.x) {
      pull_vec_t vec;
      ld_multimem_16B(vec, input_mc, vid);
      if (residual != nullptr) {
        pull_vec_t res;
        res.load(residual, vid);
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          vec[j] = SumOp::reduce(vec[j], res[j]);
        }
      }
      vec.store(prefix, vid);
    }
  }
  __threadfence();
  __syncthreads();

  extern __shared__ char smem_raw[];
  Trait::forward(params, reinterpret_cast<typename Trait::Smem*>(smem_raw));

  __syncthreads();
  if (threadIdx.x == 0) {
    auto* semaphore = &params.sem_local[blockIdx.x];
    attn_res_sem_arrive_release(attn_res_sem_mc_flag(params.sem_mc, blockIdx.x));
    while (semaphore->get_acquire() - exit_base < params.world_size)
      ;
  }
}

// Local attention-residual aggregation + direct AG epilogue. The consumer
// threads already hold each normalized 16B output vector in registers, so
// Trait::forward multicast-stores those vectors into every peer's symmetric
// full-token output instead of launching a separate all-gather.
template <typename Trait, uint32_t kOccupancy>
__global__ void __launch_bounds__(Trait::kNumThreads, kOccupancy)
    attn_res_fused_direct_ag_kernel(const __grid_constant__ AttnResTMAParams params) {
  __shared__ uint32_t exit_base;
  if (threadIdx.x == 0) {
    auto* semaphore = &params.sem_local[blockIdx.x];
    const auto reserved = semaphore->counter_ptr()->inc(2 * params.world_size);
    exit_base = reserved + params.world_size;
    attn_res_sem_arrive_relaxed(attn_res_sem_mc_flag(params.sem_mc, blockIdx.x));
    while (semaphore->get_relaxed() - reserved < params.world_size)
      ;
  }
  __syncthreads();

  extern __shared__ char smem_raw[];
  Trait::forward(params, reinterpret_cast<typename Trait::Smem*>(smem_raw));

  __syncthreads();
  if (threadIdx.x == 0) {
    auto* semaphore = &params.sem_local[blockIdx.x];
    attn_res_sem_arrive_release(attn_res_sem_mc_flag(params.sem_mc, blockIdx.x));
    while (semaphore->get_acquire() - exit_base < params.world_size)
      ;
  }
}

}  // namespace sglang

using namespace sglang;
using host::distributed::CommunicatorRef;

// Host launcher: constexpr kernel table over nvb.

template <int64_t kDim, uint32_t kMaxBankRows, uint32_t kChunkRows, uint32_t kOccupancy, uint32_t kConsumerRegs>
struct AttnResFusedTmaKernel {
  using KernelFn = void (*)(const AttnResTMAParams);
  template <uint32_t kNvb>
  using Trait = KimiK3AttnResTrait<kDim, kNvb, kChunkRows, kConsumerRegs>;
  static constexpr uint32_t kNumThreads = Trait<1>::kNumThreads;
  static constexpr size_t kSmemBytes = sizeof(typename Trait<1>::Smem);
  // kOccupancy copies of the smem ring must fit one SM (228KB on SM100).
  static_assert(kOccupancy >= 1 && kOccupancy * kSmemBytes <= 233472 - 1024, "occupancy exceeds the smem budget");

  template <std::size_t... I>
  static constexpr auto make_table(std::index_sequence<I...>) {
    return std::array<KernelFn, kMaxBankRows + 1>{nullptr, attn_res_fused_tma_kernel<Trait<I + 1>, kOccupancy>...};
  }
  static constexpr auto kTable = make_table(std::make_index_sequence<kMaxBankRows>{});
  template <std::size_t... I>
  static constexpr auto make_pull_table(std::index_sequence<I...>) {
    return std::array<KernelFn, kMaxBankRows + 1>{nullptr, attn_res_fused_pull_rs_kernel<Trait<I + 1>, kOccupancy>...};
  }
  static constexpr auto kPullTable = make_pull_table(std::make_index_sequence<kMaxBankRows>{});
  template <std::size_t... I>
  static constexpr auto make_ag_table(std::index_sequence<I...>) {
    return std::array<KernelFn, kMaxBankRows + 1>{
        nullptr, attn_res_fused_direct_ag_kernel<Trait<I + 1>, kOccupancy>...};
  }
  static constexpr auto kAgTable = make_ag_table(std::make_index_sequence<kMaxBankRows>{});

  static void
  run(const tvm::ffi::TensorView prefix_sum,
      const tvm::ffi::TensorView bank,
      const tvm::ffi::TensorView cw,
      const tvm::ffi::TensorView ow,
      const tvm::ffi::TensorView out,
      int64_t nvb,
      double eps,
      bool write_prefix) {
    using namespace host;

    auto T_ = SymbolicSize{"num_tokens"};
    auto H_ = SymbolicSize{"hidden_size"};
    auto NB_ = SymbolicSize{"num_bank_slots"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device(device).verify(prefix_sum).verify(out);
    TensorMatcher({T_, NB_, H_}).with_dtype<bf16_t>().with_device(device).verify(bank);
    TensorMatcher({H_}).with_dtype<bf16_t>().with_device(device).verify(cw).verify(ow);

    const auto num_tokens = static_cast<int64_t>(T_.unwrap());
    const auto H = static_cast<int64_t>(H_.unwrap());
    const auto NB = static_cast<int64_t>(NB_.unwrap());

    RuntimeCheck(H == kDim, "attn_res_fused_tma: H must be ", kDim, ", got ", H);
    RuntimeCheck(
        1 <= nvb && nvb <= kMaxBankRows && nvb <= NB,
        "attn_res_fused_tma: nvb must be in [1, ",
        kMaxBankRows,
        "] and <= NB, got nvb=",
        nvb,
        " NB=",
        NB);
    RuntimeCheck(
        !write_prefix || nvb < NB,
        "attn_res_fused_tma: write_prefix targets bank row nvb, needs nvb < NB, got nvb=",
        nvb,
        " NB=",
        NB);

    if (num_tokens == 0) return;

    [[maybe_unused]] static const bool attrs_set = [] {
      for (uint32_t i = 1; i <= kMaxBankRows; ++i) {
        RuntimeDeviceCheck(cudaFuncSetAttribute(kTable[i], cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
      }
      return true;
    }();

    const auto num_sm = runtime::get_sm_count(device.unwrap().device_id);
    const auto grid = std::min<int64_t>((int64_t)num_sm * kOccupancy, num_tokens);
    const auto params = AttnResTMAParams{
        .prefix_sum = static_cast<const bf16_t*>(prefix_sum.data_ptr()),
        .bank = static_cast<const bf16_t*>(bank.data_ptr()),
        .cw = static_cast<const bf16_t*>(cw.data_ptr()),
        .ow = static_cast<const bf16_t*>(ow.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .prefix_dst = write_prefix ? static_cast<bf16_t*>(bank.data_ptr()) + nvb * H : nullptr,
        .input_mc = nullptr,
        .residual = nullptr,
        .prefix_out = nullptr,
        .sem_local = nullptr,
        .sem_mc = nullptr,
        .output_mc = nullptr,
        .world_size = 0,
        .rank = 0,
        .stride_bm = NB * H,
        .eps = static_cast<float>(eps),
        .num_tokens = static_cast<uint32_t>(num_tokens),
    };
    LaunchKernel(grid, kNumThreads, device.unwrap(), kSmemBytes).enable_pdl(true)(kTable[nvb], params);
  }

  static void run_pull_rs(
      CommunicatorRef ref,
      const tvm::ffi::TensorView input,
      std::optional<tvm::ffi::TensorView> residual,
      const tvm::ffi::TensorView bank,
      const tvm::ffi::TensorView cw,
      const tvm::ffi::TensorView ow,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView prefix_out,
      int64_t nvb,
      double eps,
      int64_t input_mc_ptr,
      int64_t sem_mc_ptr,
      int64_t max_blocks) {
    using namespace host;
    const auto& data = *ref.get();
    auto GT_ = SymbolicSize{"global_tokens"};
    auto T_ = SymbolicSize{"local_tokens"};
    auto H_ = SymbolicSize{"hidden_size"};
    auto NB_ = SymbolicSize{"num_bank_slots"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({GT_, H_}).with_dtype<bf16_t>().with_device(device).verify(input);
    TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device(device).verify(out).verify(prefix_out);
    if (residual.has_value()) {
      TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device(device).verify(residual.value());
    }
    TensorMatcher({T_, NB_, H_}).with_dtype<bf16_t>().with_device(device).verify(bank);
    TensorMatcher({H_}).with_dtype<bf16_t>().with_device(device).verify(cw).verify(ow);

    const auto global_tokens = static_cast<int64_t>(GT_.unwrap());
    const auto num_tokens = static_cast<int64_t>(T_.unwrap());
    const auto H = static_cast<int64_t>(H_.unwrap());
    const auto NB = static_cast<int64_t>(NB_.unwrap());
    RuntimeCheck(data.world_size > 1, "fused pull RS requires world_size > 1");
    RuntimeCheck(global_tokens == num_tokens * data.world_size, "global tokens must equal local tokens * world size");
    RuntimeCheck(H == kDim, "fused pull RS: H must be ", kDim, ", got ", H);
    RuntimeCheck(1 <= nvb && nvb <= kMaxBankRows && nvb <= NB, "fused pull RS: invalid nvb=", nvb, " NB=", NB);
    RuntimeCheck(input_mc_ptr != 0, "fused pull RS requires multicast input");
    RuntimeCheck(sem_mc_ptr != 0, "fused pull RS requires multicast semaphores");
    RuntimeCheck(max_blocks > 0, "fused pull RS requires max_blocks > 0");
    if (num_tokens == 0) return;

    [[maybe_unused]] static const bool attrs_set = [] {
      for (uint32_t i = 1; i <= kMaxBankRows; ++i) {
        RuntimeDeviceCheck(
            cudaFuncSetAttribute(kPullTable[i], cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
      }
      return true;
    }();

    const auto num_sm = runtime::get_sm_count(device.unwrap().device_id);
    const auto grid = std::min<int64_t>(
        {static_cast<int64_t>(num_sm) * kOccupancy,
         num_tokens,
         max_blocks,
         static_cast<int64_t>(data.num_pull_blocks)});
    const auto local_elems = num_tokens * H;
    const auto params = AttnResTMAParams{
        .prefix_sum = static_cast<const bf16_t*>(prefix_out.data_ptr()),
        .bank = static_cast<const bf16_t*>(bank.data_ptr()),
        .cw = static_cast<const bf16_t*>(cw.data_ptr()),
        .ow = static_cast<const bf16_t*>(ow.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .prefix_dst = nullptr,
        .input_mc = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(input_mc_ptr)) +
                    data.rank * local_elems * sizeof(bf16_t),
        .residual = residual.has_value() ? static_cast<const bf16_t*>(residual.value().data_ptr()) : nullptr,
        .prefix_out = static_cast<bf16_t*>(prefix_out.data_ptr()),
        .sem_local = data.pull_semaphores[data.rank],
        .sem_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(sem_mc_ptr)),
        .output_mc = nullptr,
        .world_size = data.world_size,
        .rank = data.rank,
        .stride_bm = NB * H,
        .eps = static_cast<float>(eps),
        .num_tokens = static_cast<uint32_t>(num_tokens),
    };
    LaunchKernel(grid, kNumThreads, device.unwrap(), kSmemBytes).enable_pdl(true)(kPullTable[nvb], params);
  }

  static void run_direct_ag(
      CommunicatorRef ref,
      const tvm::ffi::TensorView prefix_sum,
      const tvm::ffi::TensorView bank,
      const tvm::ffi::TensorView cw,
      const tvm::ffi::TensorView ow,
      const tvm::ffi::TensorView out,
      int64_t nvb,
      double eps,
      int64_t output_mc_ptr,
      int64_t sem_mc_ptr,
      int64_t max_blocks,
      bool write_prefix) {
    using namespace host;
    const auto& data = *ref.get();
    auto T_ = SymbolicSize{"local_tokens"};
    auto GT_ = SymbolicSize{"global_tokens"};
    auto H_ = SymbolicSize{"hidden_size"};
    auto NB_ = SymbolicSize{"num_bank_slots"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device(device).verify(prefix_sum);
    TensorMatcher({T_, NB_, H_}).with_dtype<bf16_t>().with_device(device).verify(bank);
    TensorMatcher({H_}).with_dtype<bf16_t>().with_device(device).verify(cw).verify(ow);
    TensorMatcher({GT_, H_}).with_dtype<bf16_t>().with_device(device).verify(out);

    const auto num_tokens = static_cast<int64_t>(T_.unwrap());
    const auto global_tokens = static_cast<int64_t>(GT_.unwrap());
    const auto H = static_cast<int64_t>(H_.unwrap());
    const auto NB = static_cast<int64_t>(NB_.unwrap());
    RuntimeCheck(data.world_size > 1, "fused direct AG requires world_size > 1");
    RuntimeCheck(global_tokens == num_tokens * data.world_size, "global tokens must equal local tokens * world size");
    RuntimeCheck(H == kDim, "fused direct AG: H must be ", kDim, ", got ", H);
    RuntimeCheck(1 <= nvb && nvb <= kMaxBankRows && nvb <= NB, "fused direct AG: invalid nvb=", nvb, " NB=", NB);
    RuntimeCheck(!write_prefix || nvb < NB, "fused direct AG: write_prefix targets bank row nvb, needs nvb < NB");
    RuntimeCheck(output_mc_ptr != 0, "fused direct AG requires multicast output");
    RuntimeCheck(sem_mc_ptr != 0, "fused direct AG requires multicast semaphores");
    RuntimeCheck(max_blocks > 0, "fused direct AG requires max_blocks > 0");
    if (num_tokens == 0) return;

    [[maybe_unused]] static const bool attrs_set = [] {
      for (uint32_t i = 1; i <= kMaxBankRows; ++i) {
        RuntimeDeviceCheck(cudaFuncSetAttribute(kAgTable[i], cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
      }
      return true;
    }();

    const auto num_sm = runtime::get_sm_count(device.unwrap().device_id);
    const auto grid = std::min<int64_t>(
        {static_cast<int64_t>(num_sm) * kOccupancy,
         num_tokens,
         max_blocks,
         static_cast<int64_t>(data.num_pull_blocks)});
    const auto params = AttnResTMAParams{
        .prefix_sum = static_cast<const bf16_t*>(prefix_sum.data_ptr()),
        .bank = static_cast<const bf16_t*>(bank.data_ptr()),
        .cw = static_cast<const bf16_t*>(cw.data_ptr()),
        .ow = static_cast<const bf16_t*>(ow.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .prefix_dst = write_prefix ? static_cast<bf16_t*>(bank.data_ptr()) + nvb * H : nullptr,
        .input_mc = nullptr,
        .residual = nullptr,
        .prefix_out = nullptr,
        .sem_local = data.pull_semaphores[data.rank],
        .sem_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(sem_mc_ptr)),
        .output_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(output_mc_ptr)),
        .world_size = data.world_size,
        .rank = data.rank,
        .stride_bm = NB * H,
        .eps = static_cast<float>(eps),
        .num_tokens = static_cast<uint32_t>(num_tokens),
    };
    LaunchKernel(grid, kNumThreads, device.unwrap(), kSmemBytes).enable_pdl(true)(kAgTable[nvb], params);
  }
};
