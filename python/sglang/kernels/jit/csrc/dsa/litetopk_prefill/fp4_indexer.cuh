// fp4 (MX e2m1) DSA prefill indexer with sampled-threshold candidate filtering (SM100).
//
// The scoring core follows DeepGEMM's `sm100_mqa_logits` MX path: a 256-key KV tile is the A operand of two
// M128 x N128 x K64 `kind::mxf4` MMAs per math warpgroup (B = the block's `BLOCK_Q x heads` Q rows), the
// accumulator lives in TMEM, every math thread owns one key of the tile and reduces the heads in registers.
// Instead of writing logits, each row sets a threshold T_r from a strided sample of its own scores and only the
// (score, key) pairs at or above T_r reach a per-row global slab; blocks of short rows store every score instead.
#pragma once

#include <sgl_kernel/once.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/ptx/async.cuh>
#include <sgl_kernel/ptx/barrier.cuh>
#include <sgl_kernel/ptx/tcgen05.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <type_traits>

namespace sglang {

namespace device::ptx {

template <uint32_t kNumRegs>
SGL_DEVICE void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(kNumRegs));
}
template <uint32_t kNumRegs>
SGL_DEVICE void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(kNumRegs));
}

// Predicated stores keep an emission branch-free.
SGL_DEVICE void st_global_if(bool pred, void* addr, uint32_t value) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %0, 0;\n\t@p st.global.u32 [%1], %2;\n\t}" ::"r"(
                   static_cast<uint32_t>(pred)),
               "l"(addr),
               "r"(value)
               : "memory");
}
SGL_DEVICE void st_global_u16_if(bool pred, void* addr, uint32_t value) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %0, 0;\n\t@p st.global.u16 [%1], %2;\n\t}" ::"r"(
                   static_cast<uint32_t>(pred)),
               "l"(addr),
               "h"(static_cast<uint16_t>(value))
               : "memory");
}
// Bit position of the most significant set bit, 0xffffffff for 0.
SGL_DEVICE uint32_t bfind(uint32_t x) {
  uint32_t r;
  asm("bfind.u32 %0, %1;" : "=r"(r) : "r"(x));
  return r;
}
// `cvt.rn.relu.bf16x2.f32`: (v.x, v.y) clamped at 0 into the (low, high) halves.
SGL_DEVICE bf16x2_t cvt_relu_bf16x2_f32(float2 v) {
  bf16x2_t packed;
  asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;" : "=r"(*reinterpret_cast<uint32_t*>(&packed)) : "f"(v.y), "f"(v.x));
  return packed;
}

}  // namespace device::ptx

struct Fp4IndexerParams {
  CUtensorMap tmap_q;        // q_fp4 as [rows * heads, 64] u8, 64 B swizzle, 128 rows per box
  CUtensorMap tmap_q_sf;     // q_sf as [rows, heads] i32, BLOCK_Q rows per box
  CUtensorMap tmap_weights;  // weights as [rows, heads] bf16, BLOCK_Q rows per box
  CUtensorMap tmap_kv;       // kv_fp4 as [keys, 64] u8, 64 B swizzle, 256 rows per box
  CUtensorMap tmap_sf_kv;    // kv_sf as [1, keys] i32, 256 keys per box
  // threshold samples: kv_fp4 as [keys / S, S, 64] (key = row * S + phase), 256 rows of one phase per box, for each
  // odd sample stride S (see sample_stride)
  CUtensorMap tmap_kv_strided[6];
  const int8_t* __restrict__ q_fp4;     // [rows, heads, 64] packed e2m1
  const int32_t* __restrict__ q_sf;     // [rows, heads] four ue8m0 bytes per (row, head)
  const int8_t* __restrict__ kv_fp4;    // [keys, 64] packed e2m1
  const int32_t* __restrict__ kv_sf;    // [keys] four ue8m0 bytes per key
  const void* __restrict__ weights;     // [rows, heads] bf16
  const int32_t* __restrict__ ks;       // [rows] first key of the row
  const int32_t* __restrict__ ke;       // [rows] one past the last key
  void* __restrict__ cand_scores;       // [rows, kCap] bf16
  int32_t* __restrict__ cand_indices;   // [rows, kCap] key - ks[row]
  int32_t* __restrict__ cand_count;     // [rows] keys at or above T (may exceed kCap)
  int32_t* __restrict__ cand_overflow;  // [rows] 1 when count > kCap
  float* __restrict__ row_thresholds;   // [rows] T_r
  int32_t* __restrict__ row_samples;    // [rows] samples S_r the threshold was taken from
  uint32_t num_rows;
  uint32_t num_kv;
  uint32_t sample_limit[6];  // keys < limit[m] = (num_kv / S_m) * S_m are reachable through tmap_kv_strided[m]
  uint32_t topk;
  // the dynamic block counter: zero before the first launch, every launch leaves it zero (one per stream: launches
  // sharing it must not overlap)
  uint32_t* __restrict__ sched;
};

namespace device::fp4_indexer {

// Scores are DeepGEMM's bf16 head reduce (bf16 weights) and the slab holds them as bf16 (the model selects in bf16):
// the threshold is exactly the j'-th largest bf16 sample and the candidate test is on the bf16 value.
// The math warps only reduce and stage every (row, key) bf16 score in a ring of kDrainSlots tile slots; two select
// warps take the thresholds from the staged sample tiles and two drain warps filter the staged main tiles into the
// slab (half of the rows each). The sample tiles stay in their slots until the thresholds are known. A dense block
// stores only the scores, in key order (a T = -inf row's index of slot s is s).
// kNumHeads    Index heads (32 or 64); BLOCK_Q = 128 / kNumHeads query rows per CTA block.
// kNumKVStages KV ring depth (16 KB fp4 + 1 KB scales per stage).
// kDrainSlots  Staging-ring depth in 256-key tiles (BLOCK_Q x 512 B each).
template <uint32_t kNumHeads, uint32_t kNumKVStages, uint32_t kDrainSlots>
struct Fp4IndexerImpl {
  static constexpr uint32_t BLOCK_Q = 128 / kNumHeads;
  static_assert(kNumHeads == 32 || kNumHeads == 64);
  static constexpr uint32_t kMathUnroll = 2;  // math tile loop (as DeepGEMM's)
  // Main tiles [0, kMaskStart) of a block are filtered by the drains from the staged scores; from tile kMaskStart
  // on the math warps compute the hit masks (they wait for the block's thresholds there, once). The select warps
  // take the thresholds while the math warps reduce those tiles. Measured: 4 at 32 heads, 8 at 64 heads.
  static constexpr uint32_t kMaskStart = kNumHeads == 64 ? 8 : 4;
  static constexpr uint32_t kDrainPass = 4;    // main tiles per drain pass
  static constexpr uint32_t kRoundUnroll = 2;  // drain store rounds per loop iteration
  // Most 256-key sample tiles per block (S <= 256 * kSampleTilesMax samples per row); the ring keeps 8 more slots.
  static constexpr uint32_t kSampleTilesMax = std::min(kDrainSlots - 8, 32u);
  static_assert(kDrainSlots > 8 && kSampleTilesMax + kMaskStart <= kDrainSlots, "samples and the unmasked tiles fit");
  static constexpr uint32_t kHeadDim = 128;
  static constexpr uint32_t kKeyBytes = kHeadDim / 2;  // packed e2m1
  static constexpr uint32_t kNumWG = 2;                // math warpgroups
  static constexpr uint32_t UMMA_M = 128, UMMA_N = 128;
  static constexpr uint32_t kTileKeys = UMMA_M * kNumWG;  // one tile = one UMMA_M part per math warpgroup
  static constexpr uint32_t kTileBytes = kTileKeys * kKeyBytes;
  static constexpr uint32_t kQBytes = UMMA_N * kKeyBytes;
  static constexpr uint32_t kNumQStages = 3;
  // 3 accumulator stages: one per math warpgroup and one of lookahead
  static constexpr uint32_t kNumTmemStages = 3;
  static constexpr uint32_t kCap = 12288;
  // Spans up to this many keys sample every 33rd key, longer ones the stride that fills kSampleTilesMax tiles.
  static constexpr uint32_t kShortSampleSpan = 49152;
  // Blocks whose rows are all at most this long skip the sample and the drain and store every score: measured
  // faster end to end up to ~6K keys, slower from ~8K on (the top-k then reads every key instead of ~2.5 topk).
  static constexpr uint32_t kDenseLen = 6144;
  static_assert(kDenseLen <= kCap);
  static constexpr uint32_t kNumMathWarps = 4 * kNumWG, kNumMathThreads = 128 * kNumWG;
  // Warp group 2: load, MMA, scale warps and a register donor; warp group 3: two drains (warps 0 and 3, on
  // sub-partitions 0 and 3) and two select warps (warps 1 and 2) that take the thresholds from the staged sample tiles
  // as soon as they land, so neither the math warps nor the drains wait for the other's backlog. A drain and a select
  // warp own the same half of the block's rows.
  static constexpr uint32_t kNumDrainWarps = 2;
  static constexpr uint32_t kNumSelectWarps = 2;
  static_assert(kNumSelectWarps == kNumDrainWarps, "a select warp owns a drain warp's rows");
  static constexpr uint32_t kNumThreads = kNumMathThreads + 256;
  // 65536 registers = 256 math threads x kMathRegs + 128 x kLowRegs (warp group 2, which needs <= 32) + 128 x
  // kSpecRegs (warp group 3: the drains' store loops rematerialise their addresses below ~72)
  static constexpr uint32_t kMathRegs = 200, kLowRegs = 40, kSpecRegs = 72;
  static_assert(kNumMathThreads * kMathRegs + 128 * (kLowRegs + kSpecRegs) == 65536);
  // TMEM columns: the accumulator stages, the Q scales (4 per Q stage), the KV scales (4 per tile half and KV
  // stage, so the KV stage's empty barrier also frees them).
  static constexpr uint32_t kSfKvStageCols = 4 * kNumWG;
  static constexpr uint32_t kSfQCol = kNumTmemStages * UMMA_N;
  static constexpr uint32_t kSfKvCol = kSfQCol + kNumQStages * 4;
  static constexpr uint32_t kNumTmemCols = 512;
  static_assert(kSfKvCol + kNumKVStages * kSfKvStageCols <= kNumTmemCols);
  static constexpr uint32_t kNumWeightRegs = kNumHeads / 2;  // bf16x2 pairs
  // columns per TMEM load: a tile part's UMMA_N columns (BLOCK_Q rows x kNumHeads heads) in UMMA_N / kLoadCols loads
  static constexpr uint32_t kLoadCols = 16;
  using tcgen05 = device::ptx::TCGen05<1>;

  static constexpr uint32_t kSchedDepth = 4, kSchedEnd = 0xffffffffu;  // the block ring (see take_block)
  // + the MMA and scale warps
  static constexpr uint32_t kSchedConsumers = kNumMathWarps + kNumDrainWarps + kNumSelectWarps + 2;
  enum Barrier { BAR_MATH = 1 };

  struct alignas(16) BlockPlan {            // 16 B aligned: the sched ring copies it with vector accesses
    uint32_t kv_base;                       // first key the tiles cover (multiple of 4)
    uint32_t span;                          // keys covered
    uint32_t num_tiles;                     // 256-key main tiles
    uint32_t num_samples;                   // sample tiles
    uint32_t sample_stride, sample_base;    // single-key samples: key of sample i = sample_base + i * sample_stride
    uint32_t sample_map, sample_limit;      // tensor map index of the stride, keys below the limit are reachable
    uint32_t start[BLOCK_Q], len[BLOCK_Q];  // len = 0 past the last row
    bool dense;  // every row is at most kDenseLen keys: the math warps store all its scores, no sample or drain
  };

  struct Smem {
    alignas(1024) uint8_t kv[kNumKVStages][kTileBytes];
    alignas(1024) uint8_t q[kNumQStages][kQBytes];
    alignas(128) uint32_t sf_kv[kNumKVStages][kTileKeys];
    alignas(128) uint32_t sf_q[kNumQStages][UMMA_N];
    alignas(16) bf16x2_t weights[kNumQStages][BLOCK_Q * kNumWeightRegs];  // head pairs
    // per sampled-block parity: [0] high byte of the sample codes (the math warps), [1] low byte of the pivot bin
    // (the select warps)
    alignas(16) uint32_t hist[2][2][BLOCK_Q][256];
    alignas(16) uint16_t drain_stage[kDrainSlots][BLOCK_Q][kTileKeys];  // bf16 score bits
    // per slot and math warp: the warp's 32-key hit masks of every row (main tiles >= kMaskStart of a block)
    alignas(16) uint32_t drain_mask[kDrainSlots][kNumMathWarps][BLOCK_Q];
    uint64_t drain_full[kDrainSlots], drain_empty[kDrainSlots];
    alignas(16) float thr_vals[2][BLOCK_Q];  // the thresholds of the sampled block with ordinal parity p
    uint64_t thr_full[2];                    // the select warps published thr_vals[p]
    uint64_t thr_empty[2];                   // the drains read thr_vals[p]
    // the math warps started sampled block p: the select warps skip the main tiles' slots, so they wait for the math
    // warps' block before waiting on its sample slots (a parity wait more than one phase ahead would pass at once)
    uint64_t blk_start[2];
    uint32_t sched_ids[kSchedDepth];     // the CTA's k-th block (kSchedEnd past the last)
    BlockPlan sched_plans[kSchedDepth];  // and its plan, computed once by the load warp
    uint64_t sched_full[kSchedDepth], sched_empty[kSchedDepth];
    // full_*_data: the operands landed in smem; full_*: their scales also sit in TMEM (the MMA warp's condition);
    // empty_q: the MMAs of the Q stage completed (its smem and TMEM scale columns are free) and the math warps read
    // its weights
    uint64_t full_q_data[kNumQStages], full_q[kNumQStages], empty_q[kNumQStages];
    uint64_t full_kv_data[kNumKVStages], full_kv[kNumKVStages], empty_kv[kNumKVStages];
    uint64_t full_tmem[kNumTmemStages], empty_tmem[kNumTmemStages];
    uint32_t tmem_base;
  };
  static constexpr uint32_t kSmemBytes = sizeof(Smem);
  static_assert(kSmemBytes <= 232448, "shared memory exceeds the 227 KB opt-in limit");

  template <uint32_t kNumStages>
  struct Ring {
    uint32_t stage = 0, phase = 0;
    SGL_DEVICE void advance(uint32_t step = 1) {
      stage += step;
      if (stage >= kNumStages) {
        stage -= kNumStages;
        phase ^= 1u;
      }
    }
    // Any number of steps (the select warps skip a block's main tiles).
    SGL_DEVICE void advance_far(uint32_t step) {
      stage += step;
      phase ^= (stage / kNumStages) & 1u;
      stage %= kNumStages;
    }
  };

  SGL_DEVICE static BlockPlan make_block_plan(const Fp4IndexerParams& p, uint32_t block_idx) {
    BlockPlan plan;
    uint32_t lo = 0xffffffffu, hi = 0, max_len = 0;
#pragma unroll
    for (uint32_t i = 0; i < BLOCK_Q; ++i) {
      const uint32_t row = block_idx * BLOCK_Q + i;
      uint32_t s = 0, e = 0;
      if (row < p.num_rows) {
        s = static_cast<uint32_t>(__ldg(p.ks + row));
        e = max(static_cast<uint32_t>(__ldg(p.ke + row)), s);
      }
      plan.start[i] = s;
      plan.len[i] = e - s;
      lo = min(lo, s);
      hi = max(hi, e);
      max_len = max(max_len, e - s);
    }
    plan.kv_base = lo & ~3u;
    plan.span = hi > plan.kv_base ? hi - plan.kv_base : 0u;
    plan.num_tiles = div_ceil(plan.span, kTileKeys);
    plan.num_samples = 0;
    plan.sample_stride = 0;
    plan.sample_base = 0;
    plan.sample_map = 0;
    plan.sample_limit = 0;
    plan.dense = max_len <= kDenseLen;
    if (!plan.dense) {
      // Single keys on an odd stride (top keys sit in runs: an 8-key block lands in one or misses it, which makes
      // the threshold noisy): every 33rd key up to kShortSampleSpan keys, else the smallest stride that keeps the
      // sample within kSampleTilesMax tiles; a per-block phase keeps the grid off any fixed alignment of the keys.
      uint32_t m = 0;
      if (plan.span > kShortSampleSpan) {
        m = 1;
        while (m < 5 && div_ceil(plan.span, sample_stride(m)) > kSampleTilesMax * kTileKeys)
          ++m;
      }
      plan.sample_stride = sample_stride(m);
      plan.sample_map = m;
      plan.sample_limit = p.sample_limit[m];
      plan.sample_base = plan.kv_base + ((block_idx * 2654435761u) >> 8) % plan.sample_stride;
      const uint32_t n = plan.sample_base < hi ? div_ceil(hi - plan.sample_base, plan.sample_stride) : 1u;
      plan.num_samples = min(div_ceil(n, kTileKeys), kSampleTilesMax);
    }
    return plan;
  }

  // Blocks are scheduled dynamically: the load warp takes the next block index from a global counter (longest
  // blocks first: the counter walks the blocks backwards unless the first block is the longer end), publishes it and
  // its plan in a small smem ring, and every other warp reads the same sequence from there.

  // The CTA's k-th block, published by the load warp (no release: the load warp's own reads).
  SGL_DEVICE static uint32_t sched_peek(Smem& smem, uint32_t k) {
    ptx::mbar_wait_parity_suspend(&smem.sched_full[k % kSchedDepth], (k / kSchedDepth) & 1u);
    return smem.sched_ids[k % kSchedDepth];
  }

  // A consumer warp's k-th block and its plan: the k-th published entry, released after the read.
  SGL_DEVICE static uint32_t take_block(Smem& smem, uint32_t k, uint32_t lane_id, BlockPlan& plan) {
    const uint32_t b = sched_peek(smem, k);
    if (b != kSchedEnd) plan = smem.sched_plans[k % kSchedDepth];
    __syncwarp();
    if (lane_id == 0) ptx::mbar_arrive(&smem.sched_empty[k % kSchedDepth]);
    return b;
  }

  // Block source of the load warp: entry 0 is the CTA's own block (blockIdx.x), entry j > 0 the block of the counter
  // value taken when it is published (the counter hands out blocks from gridDim.x on; taking ahead would let a CTA
  // hold several of the longest blocks).
  struct BlockProducer {
    uint32_t* sched = nullptr;
    uint32_t published = 0;
    bool forward = false, ended = false;
    SGL_DEVICE void init(const Fp4IndexerParams& params, uint32_t num_blocks) {
      sched = params.sched;
      // longest blocks first: rows get longer towards the end of a request unless the first block is the longer end
      forward = make_block_plan(params, 0).num_tiles > make_block_plan(params, num_blocks - 1).num_tiles;
    }
    // The block of schedule position i.
    SGL_DEVICE uint32_t block_at(uint32_t i, uint32_t num_blocks) const {
      return forward ? i : num_blocks - 1 - i;
    }
    // Publishes entries up to j (one per call after the first).
    SGL_DEVICE void
    publish_to(const Fp4IndexerParams& params, Smem& smem, uint32_t j, uint32_t lane_id, uint32_t num_blocks) {
      for (; !ended && published <= j; ++published) {
        const uint32_t s = published % kSchedDepth;
        ptx::mbar_wait_parity_suspend(&smem.sched_empty[s], ((published / kSchedDepth) & 1u) ^ 1u);
        uint32_t b = kSchedEnd;
        if (lane_id == 0) {
          const uint32_t i = published == 0 ? blockIdx.x : gridDim.x + atomicAdd(sched, 1u);
          if (i < num_blocks) {
            b = block_at(i, num_blocks);
            smem.sched_plans[s] = make_block_plan(params, b);
          }
          smem.sched_ids[s] = b;
          ptx::mbar_arrive(&smem.sched_full[s]);  // release: the id and the plan are visible to the waiters
        }
        ended = __shfl_sync(warp::kFullMask, b, 0) == kSchedEnd;
      }
    }
  };

  // Odd sample strides by index (see make_block_plan).
  static constexpr uint32_t sample_stride(uint32_t m) {
    return m == 0 ? 33u : m == 1 ? 65u : m == 2 ? 101u : m == 3 ? 129u : m == 4 ? 193u : 257u;
  }
  // Key of sample `i` of the block.
  SGL_DEVICE static uint32_t sample_key(const BlockPlan& plan, uint32_t i) {
    return plan.sample_base + i * plan.sample_stride;
  }

  // `rank`-th largest element of a 256-bin histogram (one warp): the bin and the rank within it.
  // Returns false when the histogram holds fewer than `rank` elements.
  SGL_DEVICE static bool
  find_rank_bin(const uint32_t* hist, uint32_t rank, uint32_t lane_id, uint32_t& bin, uint32_t& rank_in_bin) {
    AlignedVector<uint32_t, 4> lo, hi;
    lo.load(hist + 8 * lane_id);
    hi.load(hist + 8 * lane_id + 4);
    uint32_t count[8];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      count[i] = lo[i];
      count[i + 4] = hi[i];
    }
    uint32_t local = 0;
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      local += count[i];
    }
    // above[j]: elements in bins > 8 * lane_id + j; the highest bins live on the highest lanes
    uint32_t above = local;
#pragma unroll
    for (uint32_t offset = 1; offset < kWarpThreads; offset <<= 1) {
      const auto t = __shfl_down_sync(warp::kFullMask, above, offset);
      if (lane_id + offset < kWarpThreads) above += t;
    }
    const uint32_t total = __shfl_sync(warp::kFullMask, above, 0);
    above -= local;
    if (rank > total || rank == 0) return false;
    const bool found = above < rank && rank <= above + local;
    uint32_t found_bin = 0, found_rank = 0;
    uint32_t c = above;
#pragma unroll
    for (int32_t j = 7; j >= 0; --j) {
      const uint32_t next = c + count[j];
      if (c < rank && rank <= next) {
        found_bin = 8 * lane_id + j;
        found_rank = rank - c;
      }
      c = next;
    }
    const uint32_t src = __ffs(__ballot_sync(warp::kFullMask, found)) - 1;
    bin = __shfl_sync(warp::kFullMask, found_bin, src);
    rank_in_bin = __shfl_sync(warp::kFullMask, found_rank, src);
    return true;
  }

  // kChunks groups of 128 scale words (one per row), in place: row r of each 32 x 16 B UTCCP chunk holds rows r,
  // r + 32, r + 64, r + 96. Every load is issued before the one warp sync.
  template <uint32_t kChunks>
  SGL_DEVICE static void transpose_sf(uint32_t* sf, uint32_t lane_id) {
    uint32_t values[kChunks][4];
#pragma unroll
    for (uint32_t c = 0; c < kChunks; ++c) {
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        values[c][i] = sf[c * 128 + i * 32 + lane_id];
      }
    }
    __syncwarp();
#pragma unroll
    for (uint32_t c = 0; c < kChunks; ++c) {
      device::AlignedVector<uint32_t, 4> v;
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        v[i] = values[c][i];
      }
      v.store(sf + c * 128 + lane_id * 4);
    }
  }

  SGL_DEVICE static void run(const Fp4IndexerParams& params, Smem& smem) {
    const uint32_t warp_id = __shfl_sync(warp::kFullMask, threadIdx.x / kWarpThreads, 0);
    const uint32_t lane_id = warp::get_lane_id();
    const uint32_t num_blocks = div_ceil(params.num_rows, BLOCK_Q);
    // Warp group 2: the producers and a register donor; warp group 3: the drains (sub-partitions 0 and 3, away from
    // the MMA and scale warps) and the select warps.
    enum WarpRole : uint32_t {
      WARP_LOAD = kNumMathWarps + 0,
      WARP_MMA = kNumMathWarps + 1,
      WARP_SF = kNumMathWarps + 2,
      WARP_DRAIN_0 = kNumMathWarps + 4,
      WARP_SELECT_0 = kNumMathWarps + 5,
      WARP_SELECT_1 = kNumMathWarps + 6,
      WARP_DRAIN_1 = kNumMathWarps + 7,
    };

    if (warp_id == 0) {
      tcgen05::alloc(&smem.tmem_base, kNumTmemCols);
      tcgen05::fence_before_thread_sync();
    } else if (warp_id == 1) {
      if (lane_id < kNumQStages) {
        ptx::mbar_init(&smem.full_q_data[lane_id], 1);
        ptx::mbar_init(&smem.full_q[lane_id], 1);
        ptx::mbar_init(&smem.empty_q[lane_id], kNumMathWarps + 1);
      }
      if (lane_id < kNumKVStages) {
        ptx::mbar_init(&smem.full_kv_data[lane_id], 1);
        ptx::mbar_init(&smem.full_kv[lane_id], 1);
        ptx::mbar_init(&smem.empty_kv[lane_id], 1);
      }
      if (lane_id < kNumTmemStages) {
        ptx::mbar_init(&smem.full_tmem[lane_id], 1);
        ptx::mbar_init(&smem.empty_tmem[lane_id], 128);
      }
      for (uint32_t i = lane_id; i < kDrainSlots; i += kWarpThreads) {
        ptx::mbar_init(&smem.drain_full[i], kNumMathWarps);
        ptx::mbar_init(&smem.drain_empty[i], kNumDrainWarps);
      }
      if (lane_id < 2) ptx::mbar_init(&smem.thr_full[lane_id], kNumSelectWarps);
      if (lane_id < 2) ptx::mbar_init(&smem.thr_empty[lane_id], kNumDrainWarps);
      if (lane_id < 2) ptx::mbar_init(&smem.blk_start[lane_id], 1);
      for (uint32_t i = lane_id; i < sizeof(smem.hist) / 4; i += kWarpThreads) {
        (&smem.hist[0][0][0][0])[i] = 0;
      }
      if (lane_id < kSchedDepth) {
        ptx::mbar_init(&smem.sched_full[lane_id], 1);
        ptx::mbar_init(&smem.sched_empty[lane_id], kSchedConsumers);
      }
      ptx::fence_barrier_init();
    } else if (warp_id == 2 && lane_id == 0) {
      ptx::prefetch_tensormap(&params.tmap_q);
      ptx::prefetch_tensormap(&params.tmap_q_sf);
      ptx::prefetch_tensormap(&params.tmap_weights);
      ptx::prefetch_tensormap(&params.tmap_kv);
      ptx::prefetch_tensormap(&params.tmap_sf_kv);
    }
    __syncthreads();
    tcgen05::fence_after_thread_sync();
    const auto tmem_base = smem.tmem_base;

    if (warp_id == WARP_LOAD) {
      // The load warp: schedules and plans the blocks; per block it loads the next block's Q, Q scales and weights
      // (TMA; rows past the end are zero-filled, their len is 0), then this block's sample tiles (cp.async) and main
      // tiles (TMA).
      ptx::setmaxnreg_dec<kLowRegs>();
      Ring<kNumQStages> q_ring;
      Ring<kNumKVStages> kv_ring;
      BlockProducer producer;
      producer.init(params, num_blocks);
      // Entry k + 1 is published (a global atomic and the next plan) and its Q loaded during block k, once block k's
      // sample tiles and first main tile are issued, so the atomic's round trip is off the sample gather's path.
      producer.publish_to(params, smem, 0, lane_id, num_blocks);
      const auto next_block = [&](uint32_t k) { return sched_peek(smem, k); };
      const auto load_q = [&](uint32_t block_idx) {
        ptx::mbar_wait_parity_suspend(&smem.empty_q[q_ring.stage], q_ring.phase ^ 1);
        if (warp::elect_one_lane()) {
          const auto q_stage = q_ring.stage;
          const auto bar = &smem.full_q_data[q_stage];
          const auto row0 = static_cast<int32_t>(block_idx * BLOCK_Q);
          ptx::cp_async_bulk_tensor_2d(ptx::to_shared(smem.q[q_stage]), &params.tmap_q, 0, row0 * kNumHeads, bar);
          ptx::cp_async_bulk_tensor_2d(ptx::to_shared(smem.sf_q[q_stage]), &params.tmap_q_sf, 0, row0, bar);
          ptx::cp_async_bulk_tensor_2d(ptx::to_shared(smem.weights[q_stage]), &params.tmap_weights, 0, row0, bar);
          ptx::mbar_arrive_expect_tx(bar, kQBytes + sizeof(smem.sf_q[0]) + sizeof(smem.weights[0]));
        }
        __syncwarp();
        q_ring.advance();
      };
      const auto prepare_next = [&](uint32_t k) {
        producer.publish_to(params, smem, k + 1, lane_id, num_blocks);
        const auto next = sched_peek(smem, k + 1);
        if (next < num_blocks) load_q(next);
      };
      load_q(producer.block_at(blockIdx.x, num_blocks));  // the grid never exceeds the blocks
      for (uint32_t k = 0, block_idx = next_block(0); block_idx < num_blocks; block_idx = next_block(++k)) {
        const BlockPlan plan = smem.sched_plans[k % kSchedDepth];
        // Sample tiles: block b of each tile is one TMA box of 8 keys (512 B, one 64 B swizzle atom) issued by one
        // lane; its scales come by cp.async from lane b (a TMA destination needs 128 B alignment).
        for (uint32_t t = 0; t < plan.num_samples; ++t) {
          ptx::mbar_wait_parity_suspend(&smem.empty_kv[kv_ring.stage], kv_ring.phase ^ 1);
          const auto stage = kv_ring.stage;
          const auto bar = &smem.full_kv_data[stage];
          // lane l: the scales of samples 8 l .. 8 l + 7 (keys past the map read key 0: not a valid sample)
#pragma unroll
          for (uint32_t j = 0; j < 8; ++j) {
            const uint32_t key = sample_key(plan, t * kTileKeys + lane_id * 8 + j);
            ptx::cp_async_ca_4B(
                ptx::to_shared(&smem.sf_kv[stage][lane_id * 8 + j]),
                params.kv_sf + (key < plan.sample_limit ? key : 0u));
          }
          ptx::cp_async_mbarrier_arrive(bar);  // counted before the elected lane's arrive below
          __syncwarp();
          if (warp::elect_one_lane()) {
            ptx::mbar_arrive_expect_tx(bar, kTileBytes);
            const uint32_t key0 = sample_key(plan, t * kTileKeys);
            ptx::cp_async_bulk_tensor_3d(
                ptx::to_shared(smem.kv[stage]),
                &params.tmap_kv_strided[plan.sample_map],
                0,
                static_cast<int32_t>(key0 % plan.sample_stride),
                static_cast<int32_t>(key0 / plan.sample_stride),
                bar);
          }
          __syncwarp();
          kv_ring.advance();
        }
        // Main tiles by TMA (rows past the KV tensor are zero-filled).
        for (uint32_t t = 0; t < plan.num_tiles; ++t) {
          ptx::mbar_wait_parity_suspend(&smem.empty_kv[kv_ring.stage], kv_ring.phase ^ 1);
          if (warp::elect_one_lane()) {
            const auto stage = kv_ring.stage;
            const auto key = static_cast<int32_t>(plan.kv_base + t * kTileKeys);
            ptx::mbar_arrive_expect_tx(&smem.full_kv_data[stage], kTileBytes + kTileKeys * 4);
            ptx::cp_async_bulk_tensor_2d(
                ptx::to_shared(smem.kv[stage]), &params.tmap_kv, 0, key, &smem.full_kv_data[stage]);
            ptx::cp_async_bulk_tensor_2d(
                ptx::to_shared(smem.sf_kv[stage]), &params.tmap_sf_kv, key, 0, &smem.full_kv_data[stage]);
          }
          __syncwarp();
          kv_ring.advance();
          if (t == 0) prepare_next(k);
        }
        if (plan.num_tiles == 0) prepare_next(k);
      }
      // Every CTA takes blocks until one past the end, so the counter ends at num_blocks - gridDim.x (the blocks past
      // the first wave) + gridDim.x; the CTA whose compare sees that value (the last to take) resets it.
      if (lane_id == 0) atomicCAS(params.sched, num_blocks, 0u);
    } else if (warp_id == WARP_SF) {
      // The scale warp: transposes each landed stage's scales into the UTCCP layout and copies them into the stage's
      // TMEM columns, so the MMA warp only waits and issues.
      ptx::setmaxnreg_dec<kLowRegs>();
      Ring<kNumQStages> q_ring;
      Ring<kNumKVStages> kv_ring;
      BlockPlan plan;
      for (uint32_t k = 0, block_idx = take_block(smem, 0, lane_id, plan); block_idx < num_blocks;
           block_idx = take_block(smem, ++k, lane_id, plan)) {
        const uint32_t q_stage = q_ring.stage;
        // the MMAs of the previous use of the stage's TMEM columns completed before the stage was reloaded
        ptx::mbar_wait_parity_suspend(&smem.full_q_data[q_stage], q_ring.phase);
        tcgen05::fence_after_thread_sync();
        transpose_sf<1>(smem.sf_q[q_stage], lane_id);
        ptx::fence_proxy_async_shared();
        __syncwarp();
        if (warp::elect_one_lane()) {
          tcgen05::cp_32x128b_warpx4(
              tmem_base + kSfQCol + q_stage * 4, tcgen05::make_desc_sf_chunk(smem.sf_q[q_stage]).bits);
          tcgen05::fence_before_thread_sync();
          ptx::mbar_arrive(&smem.full_q[q_stage]);
        }
        __syncwarp();
        q_ring.advance();
        const uint32_t num_tiles = plan.num_samples + plan.num_tiles;
        for (uint32_t t = 0; t < num_tiles; ++t) {
          const uint32_t stage = kv_ring.stage;
          // the MMAs of the stage's previous tile (its TMEM scale columns too) completed before it was reloaded
          ptx::mbar_wait_parity_suspend(&smem.full_kv_data[stage], kv_ring.phase);
          tcgen05::fence_after_thread_sync();
          transpose_sf<kNumWG>(smem.sf_kv[stage], lane_id);
          ptx::fence_proxy_async_shared();
          __syncwarp();
          if (warp::elect_one_lane()) {
#pragma unroll
            for (uint32_t part = 0; part < kNumWG; ++part) {
              tcgen05::cp_32x128b_warpx4(
                  tmem_base + kSfKvCol + stage * kSfKvStageCols + part * 4,
                  tcgen05::make_desc_sf_chunk(smem.sf_kv[stage] + part * UMMA_M).bits);
            }
            tcgen05::fence_before_thread_sync();
            ptx::mbar_arrive(&smem.full_kv[stage]);
          }
          __syncwarp();
          kv_ring.advance();
        }
      }
    } else if (warp_id == WARP_MMA) {
      ptx::setmaxnreg_dec<kLowRegs>();
      Ring<kNumQStages> q_ring;
      Ring<kNumKVStages> kv_ring;
      Ring<kNumTmemStages> tmem_ring;
      const uint32_t idesc_0 = tcgen05::make_instr_desc_mxf4<UMMA_M, UMMA_N>().with_sf_id(0).bits;
      const uint32_t idesc_1 = tcgen05::make_instr_desc_mxf4<UMMA_M, UMMA_N>().with_sf_id(2).bits;
      // stage descriptors: the stage-0 descriptor advanced by the stage's byte offset
      const auto a_desc_0 = tcgen05::make_desc_kmajor_sw64(smem.kv[0]);
      const auto b_desc_0 = tcgen05::make_desc_kmajor_sw64(smem.q[0]);
      BlockPlan plan;
      for (uint32_t k = 0, block_idx = take_block(smem, 0, lane_id, plan); block_idx < num_blocks;
           block_idx = take_block(smem, ++k, lane_id, plan)) {
        const uint32_t q_stage = q_ring.stage;
        ptx::mbar_wait_parity_suspend(&smem.full_q[q_stage], q_ring.phase);
        tcgen05::fence_after_thread_sync();
        const uint32_t num_tiles = plan.num_samples + plan.num_tiles;
        // The whole warp waits on the barriers (converged, suspended) and an elect.sync lane issues: one lane polling
        // in a diverged warp is slow, and `lane_id == 0` instead of elect.sync makes ptxas wrap every tcgen05 op in
        // a waterfall loop (64 heads: the MMA issue took ~4x longer)
        const auto b_0 = b_desc_0.advance_bytes(q_stage * kQBytes);
        const auto b_1 = b_0.advance_bytes(32);
        const uint32_t taddr_sfb = tmem_base + kSfQCol + q_stage * 4;
        for (uint32_t t = 0; t < num_tiles; ++t) {
          const uint32_t stage = kv_ring.stage;
          ptx::mbar_wait_parity_suspend(&smem.full_kv[stage], kv_ring.phase);
          tcgen05::fence_after_thread_sync();
          const auto a_desc = a_desc_0.advance_bytes(stage * kTileBytes);
#pragma unroll
          for (uint32_t half = 0; half < kNumWG; ++half) {
            ptx::mbar_wait_parity_suspend(&smem.empty_tmem[tmem_ring.stage], tmem_ring.phase ^ 1);
            tcgen05::fence_after_thread_sync();
            if (warp::elect_one_lane()) {
              const auto taddr_d = tmem_base + tmem_ring.stage * UMMA_N;
              const auto taddr_sfa = tmem_base + kSfKvCol + stage * kSfKvStageCols + half * 4;
              const auto a_0 = a_desc.advance_bytes(half * UMMA_M * kKeyBytes);
              const auto a_1 = a_0.advance_bytes(32);
              tcgen05::mma_mxf4_ss(taddr_d, a_0.bits, b_0.bits, idesc_0, taddr_sfa, taddr_sfb, 0);
              tcgen05::mma_mxf4_ss(taddr_d, a_1.bits, b_1.bits, idesc_1, taddr_sfa, taddr_sfb, 1);
              tcgen05::commit(&smem.full_tmem[tmem_ring.stage]);
            }
            __syncwarp();
            tmem_ring.advance();
          }
          // the smem stage and its TMEM scale columns
          if (warp::elect_one_lane()) tcgen05::commit(&smem.empty_kv[stage]);
          __syncwarp();
          kv_ring.advance();
        }
        if (warp::elect_one_lane()) tcgen05::commit(&smem.empty_q[q_stage]);
        __syncwarp();
        q_ring.advance();
      }
    } else if (warp_id == WARP_DRAIN_0 || warp_id == WARP_DRAIN_1) {
      ptx::setmaxnreg_dec<kSpecRegs>();
      drain(params, smem, lane_id, num_blocks, warp_id == WARP_DRAIN_0 ? 0u : 1u);
    } else if (warp_id == WARP_SELECT_0 || warp_id == WARP_SELECT_1) {
      ptx::setmaxnreg_dec<kSpecRegs>();
      select(params, smem, lane_id, num_blocks, warp_id - WARP_SELECT_0);
    } else if (warp_id >= kNumMathWarps) {
      ptx::setmaxnreg_dec<kLowRegs>();  // warp group 2's register donor
    } else {
      ptx::setmaxnreg_inc<kMathRegs>();
      math_warps(params, smem, tmem_base, warp_id, lane_id, num_blocks);
      ptx::bar_sync(BAR_MATH, kNumMathThreads);
      if (warp_id == 0) {
        tcgen05::fence_after_thread_sync();
        tcgen05::dealloc(tmem_base, kNumTmemCols);
      }
    }
  }

  // Math warps: per tile, the bf16 score of this thread's key for every row of the block goes to the tile's staging
  // slot for the drain (plus the hit masks once the thresholds are known, or the sample histogram), or straight to
  // the slab in a dense block. The weights of a block are read from its Q stage, which is then released; the tiles
  // (samples first) arrive in the MMA warp's order.
  SGL_DEVICE static void math_warps(
      const Fp4IndexerParams& params,
      Smem& smem,
      uint32_t tmem_base,
      uint32_t warp_id,
      uint32_t lane_id,
      uint32_t num_blocks) {
    enum class Mask {
      NO_MASK = 0,
      BOUND_MASK = 1,
      INTERIOR_MASK = 2,
      DENSE_MASK = 3,
    };

    const uint32_t wg = warp_id / 4;
    const uint32_t tile_pos = warp_id * kWarpThreads + lane_id;
    const uint32_t tmem_lane = ((warp_id % 4) * kWarpThreads) << 16;
    Ring<kNumTmemStages> tmem_ring;
    tmem_ring.advance(wg);
    Ring<kDrainSlots> slot_ring;
    BlockPlan plan;
    uint32_t ns = 0;  // sampled blocks so far (thr_vals / hist parity)
    for (uint32_t k = 0, block_idx = take_block(smem, 0, lane_id, plan); block_idx < num_blocks;
         block_idx = take_block(smem, ++k, lane_id, plan)) {
      const auto q_iter = k;
      const auto q_stage = q_iter % kNumQStages, q_phase = (q_iter / kNumQStages) & 1u;
      ptx::mbar_wait_parity_suspend(&smem.full_q_data[q_stage], q_phase);
      bf16x2_t weights[BLOCK_Q][kNumWeightRegs];
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < kNumWeightRegs; ++j) {
          weights[i][j] = smem.weights[q_stage][i * kNumWeightRegs + j];
        }
      }
      __syncwarp();
      if (lane_id == 0) ptx::mbar_arrive(&smem.empty_q[q_stage]);
      float threshold[BLOCK_Q];
      // One tile: TMEM -> reduce -> staged bf16 scores, plus the 32-key hit masks of every row (kMasks): none,
      // masks with the row bounds, masks of a tile inside every row's range, or a dense block (the scores go straight
      // to the slab, slot = key - start)
      const auto tile = [&](uint32_t t, auto masks_t) {
        constexpr Mask kMasks = decltype(masks_t)::value;
        ptx::mbar_wait_parity_suspend(&smem.full_tmem[tmem_ring.stage], tmem_ring.phase);
        tcgen05::fence_after_thread_sync();
        const uint32_t taddr = tmem_base + tmem_ring.stage * UMMA_N + tmem_lane;
        float accum[UMMA_N];  // (row, head) columns
#pragma unroll
        for (uint32_t c = 0; c < UMMA_N; c += kLoadCols) {
          tcgen05::ld_32x32b<kLoadCols>(taddr + c, accum + c);
        }
        tcgen05::wait_ld();
        tcgen05::fence_before_thread_sync();
        ptx::mbar_arrive(&smem.empty_tmem[tmem_ring.stage]);
        tmem_ring.advance(kNumWG);
        // DeepGEMM's bf16 head reduce: relu folded into the bf16x2 conversion, even head pairs on one bf16x2 FMA
        // chain and odd ones on the other
        uint32_t bits[BLOCK_Q];
#pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++i) {
          const auto w = weights[i];
          const auto score = accum + i * kNumHeads;
          bf16x2_t sum_0 = __floats2bfloat162_rn(0.0f, 0.0f), sum_1 = sum_0;
#pragma unroll
          for (uint32_t p = 0; p < kNumHeads / 2; p += 2) {
            const auto a_0 = ptx::cvt_relu_bf16x2_f32(make_float2(score[2 * p + 0], score[2 * p + 1]));
            const auto a_1 = ptx::cvt_relu_bf16x2_f32(make_float2(score[2 * p + 2], score[2 * p + 3]));
            sum_0 = __hfma2(a_0, w[p + 0], sum_0);
            sum_1 = __hfma2(a_1, w[p + 1], sum_1);
          }
          const auto sum = __hadd2_rn(sum_0, sum_1);
          // the bf16 result x + y in both halves of the word (one HADD2): the low half is the stored score, the high
          // half with the low bits cleared its float value for the hit test (no PRMT / shift per row)
          const bf16x2_t total = __hadd2_rn(__low2bfloat162(sum), __high2bfloat162(sum));
          bits[i] = *reinterpret_cast<const uint32_t*>(&total);
        }
        if constexpr (kMasks == Mask::DENSE_MASK) {
          const uint32_t key = plan.kv_base + t * kTileKeys + tile_pos;
#pragma unroll
          for (uint32_t i = 0; i < BLOCK_Q; ++i) {
            const uint32_t slot = key - plan.start[i];
            if (slot < plan.len[i]) {
              const auto off = static_cast<int64_t>(block_idx * BLOCK_Q + i) * kCap + slot;
              // T = -inf rows hold every key in key order: the index of slot s is s, not stored
              static_cast<uint16_t*>(params.cand_scores)[off] = static_cast<uint16_t>(bits[i]);
            }
          }
          return;
        }
        ptx::mbar_wait_parity_suspend(&smem.drain_empty[slot_ring.stage], slot_ring.phase ^ 1);
#pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++i) {
          smem.drain_stage[slot_ring.stage][i][tile_pos] = static_cast<uint16_t>(bits[i]);
        }
        if constexpr (kMasks != Mask::NO_MASK) {
          const uint32_t key = plan.kv_base + t * kTileKeys + tile_pos;
          AlignedVector<uint32_t, BLOCK_Q> m;
#pragma unroll
          for (uint32_t i = 0; i < BLOCK_Q; ++i) {
            bool pass = __uint_as_float(bits[i] & 0xffff0000u) >= threshold[i];
            if constexpr (kMasks == Mask::BOUND_MASK) pass = pass && key - plan.start[i] < plan.len[i];
            m[i] = __ballot_sync(warp::kFullMask, pass);
          }
          if (lane_id == 0) m.store(&smem.drain_mask[slot_ring.stage][warp_id][0]);
        } else {
          if (t < plan.num_samples) {
            // the sample's high code byte into the threshold histogram (the drain's first level)
            const uint32_t key = sample_key(plan, t * kTileKeys + tile_pos);
#pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++i) {
              if (key - plan.start[i] < plan.len[i] && key < plan.sample_limit) {
                atomicAdd(&smem.hist[ns & 1u][0][i][bf16_code(bits[i] >> 16) >> 8], 1u);
              }
            }
          }
        }
        __syncwarp();
        if (lane_id == 0) ptx::mbar_arrive(&smem.drain_full[slot_ring.stage]);
        slot_ring.advance();
      };
      using NoMask = std::integral_constant<Mask, Mask::NO_MASK>;
      using BoundMask = std::integral_constant<Mask, Mask::BOUND_MASK>;
      using InteriorMask = std::integral_constant<Mask, Mask::INTERIOR_MASK>;
      using DenseMask = std::integral_constant<Mask, Mask::DENSE_MASK>;
      if (plan.dense) {
#pragma unroll kMathUnroll
        for (uint32_t t = 0; t < plan.num_tiles; ++t) {
          tile(t, DenseMask{});
        }
        continue;
      }
      if (warp_id == 0 && lane_id == 0) ptx::mbar_arrive(&smem.blk_start[ns & 1u]);
      // the sample tiles, then main tiles [0, mask_start): one loop (no masks, the tile index is unused)
      const uint32_t mask_start = min(kMaskStart, plan.num_tiles);
#pragma unroll 1
      for (uint32_t t = 0; t < plan.num_samples + mask_start; ++t) {
        tile(t, NoMask{});
      }
      if (mask_start < plan.num_tiles) {
        ptx::mbar_wait_parity_suspend(&smem.thr_full[ns & 1u], (ns >> 1) & 1u);
#pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++i) {
          threshold[i] = smem.thr_vals[ns & 1u][i];
        }
      }
      // tiles [lo, hi) lie inside every row's range and skip the bounds test
      uint32_t row_lo = 0, row_hi = 0xffffffffu;
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
        if (plan.len[i] > 0) row_lo = max(row_lo, plan.start[i]), row_hi = min(row_hi, plan.start[i] + plan.len[i]);
      }
      const uint32_t in_lo = row_hi > row_lo
                                 ? min(max(div_ceil(row_lo - plan.kv_base, kTileKeys), mask_start), plan.num_tiles)
                                 : plan.num_tiles;
      const uint32_t in_hi =
          row_hi > row_lo ? max(min((row_hi - plan.kv_base) / kTileKeys, plan.num_tiles), in_lo) : plan.num_tiles;
      // rows with len 0 (past the last row) take the bounds path everywhere
      bool all_rows = true;
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
        all_rows &= plan.len[i] > 0;
      }
      const uint32_t lo = all_rows ? in_lo : plan.num_tiles, hi = all_rows ? in_hi : plan.num_tiles;
      // unrolled: one tile's reduce tail overlaps the next tile's TMEM wait and loads (as DeepGEMM's unroll 2)
#pragma unroll kMathUnroll
      for (uint32_t t = mask_start; t < lo; ++t) {
        tile(t, BoundMask{});
      }
#pragma unroll kMathUnroll
      for (uint32_t t = lo; t < hi; ++t) {
        tile(t, InteriorMask{});
      }
#pragma unroll kMathUnroll
      for (uint32_t t = max(hi, lo); t < plan.num_tiles; ++t) {
        tile(t, BoundMask{});
      }
      ++ns;
    }
  }

  // Hit bits of 8 staged bf16 scores (4 packed pairs) against a bf16 threshold pair: bit j for element j.
  SGL_DEVICE static uint32_t hits8(const device::AlignedVector<uint32_t, 4>& v, uint32_t thr2) {
    uint32_t m[4];
#pragma unroll
    for (uint32_t k = 0; k < 4; ++k) {
      const uint32_t pair = v[k];
      m[k] = __hge2_mask(*reinterpret_cast<const bf16x2_t*>(&pair), *reinterpret_cast<const bf16x2_t*>(&thr2));
    }
    // one flag byte per element (0xFF / 0x00) in element order, then signed dot products pack them into bits
    const auto lo = __byte_perm(m[0], m[1], 0x7531);
    const auto hi = __byte_perm(m[2], m[3], 0x7531);
    const int bits_lo = __dp4a(static_cast<int>(lo), static_cast<int>(0xF8FCFEFFu), 0);
    const int bits_hi = __dp4a(static_cast<int>(hi), static_cast<int>(0xF8FCFEFFu), 0);
    return static_cast<uint32_t>(bits_lo) | (static_cast<uint32_t>(bits_hi) << 4);
  }

  // Ordered 16-bit code of bf16 bits (ascending in value, negatives below positives).
  SGL_DEVICE static uint32_t bf16_code(uint32_t h) {
    return h ^ ((h & 0x8000u) ? 0xffffu : 0x8000u);
  }

  // A select warp (owns the rows of drain warp `part`): per block, its rows' thresholds from the staged sample tiles
  // (a 2-level 256-bin select of their bf16 codes), published for the math warps and the drains once the drains read
  // the previous use of thr_vals[p]; then it frees the sample slots. In the selection lane l owns sample block l of a
  // tile (one 16 B vector per row).
  SGL_DEVICE static void
  select(const Fp4IndexerParams& params, Smem& smem, uint32_t lane_id, uint32_t num_blocks, uint32_t part) {
    constexpr uint32_t kRows = BLOCK_Q / kNumDrainWarps;
    const uint32_t row_base = part * kRows;
    uint32_t ns = 0;  // sampled blocks so far
    Ring<kDrainSlots> slot_ring;
    BlockPlan plan;
    for (uint32_t k = 0, block_idx = take_block(smem, 0, lane_id, plan); block_idx < num_blocks;
         block_idx = take_block(smem, ++k, lane_id, plan)) {
      const uint32_t row0 = block_idx * BLOCK_Q + row_base;  // my first row
      // my rows' bounds (part is warp-uniform: a select per row)
      uint32_t start[kRows], len[kRows];
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        start[r] = part == 0 ? plan.start[r] : plan.start[(kNumDrainWarps - 1) * kRows + r];
        len[r] = part == 0 ? plan.len[r] : plan.len[(kNumDrainWarps - 1) * kRows + r];
      }
      if (plan.dense) continue;  // no thresholds
      // sampled blocks alternate thr_vals / hist parities (ns counts them); thr_vals[par] is free once the drains read
      // it for sampled block ns - 2
      const uint32_t par = ns & 1u, phase = (ns >> 1) & 1u;
      ++ns;
      ptx::mbar_wait_parity_suspend(&smem.thr_empty[par], phase ^ 1u);
      ptx::mbar_wait_parity_suspend(&smem.blk_start[par], phase);
      // ---- threshold ----
      auto& hist = smem.hist[par];
      const auto sample_slot = [&](uint32_t t) {
        Ring<kDrainSlots> r = slot_ring;
        r.advance(t);
        return r;
      };
      // the math warps counted every sample tile's high bytes before releasing it
      for (uint32_t t = 0; t < plan.num_samples; ++t) {
        const auto r = sample_slot(t);
        ptx::mbar_wait_parity_suspend(&smem.drain_full[r.stage], r.phase);
      }
      float thr[kRows];
      uint32_t pivot[kRows], pivot_rank[kRows], samples[kRows];
#pragma unroll
      for (uint32_t i = 0; i < kRows; ++i) {
        AlignedVector<uint32_t, 4> lo, hi;
        lo.load(&hist[0][row_base + i][8 * lane_id]);
        hi.load(&hist[0][row_base + i][8 * lane_id + 4]);
        uint32_t local = 0;
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          local += lo[j] + hi[j];
        }
        samples[i] = warp::reduce_sum(local);
        thr[i] = -INFINITY;
        pivot[i] = 0xffffffffu;
        pivot_rank[i] = 0;
        if (len[i] > params.topk && samples[i] > 0) {
          const float j = static_cast<float>(params.topk) * static_cast<float>(samples[i]) / static_cast<float>(len[i]);
          const float alpha = 4.5f * sqrtf(2.0f / j);
          const uint32_t rank = static_cast<uint32_t>(ceilf((1.0f + alpha) * j));
          uint32_t bin = 0, rank_in_bin = 0;
          if (find_rank_bin(&hist[0][row_base + i][0], rank, lane_id, bin, rank_in_bin)) {
            pivot[i] = bin;
            pivot_rank[i] = rank_in_bin;
          }
        }
      }
      bool any_pivot = false;
#pragma unroll
      for (uint32_t i = 0; i < kRows; ++i) {
        any_pivot |= pivot[i] != 0xffffffffu;
      }
      if (any_pivot) {
        for (uint32_t t = 0; t < plan.num_samples; ++t) {
          const auto r = sample_slot(t);
#pragma unroll
          for (uint32_t i = 0; i < kRows; ++i) {
            AlignedVector<uint32_t, 4> v;
            v.load(&smem.drain_stage[r.stage][row_base + i][8 * lane_id]);
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
              const uint32_t code = bf16_code((v[j / 2] >> (16 * (j % 2))) & 0xffffu);
              const uint32_t key = sample_key(plan, t * kTileKeys + 8 * lane_id + j);
              if (key - start[i] < len[i] && key < plan.sample_limit && (code >> 8) == pivot[i]) {
                atomicAdd(&hist[1][row_base + i][code & 0xffu], 1u);
              }
            }
          }
        }
        __syncwarp();
#pragma unroll
        for (uint32_t i = 0; i < kRows; ++i) {
          uint32_t bin = 0, rank_in_bin = 0;
          if (pivot[i] != 0xffffffffu &&
              find_rank_bin(&hist[1][row_base + i][0], pivot_rank[i], lane_id, bin, rank_in_bin)) {
            const uint32_t code = (pivot[i] << 8) | bin;
            thr[i] = __uint_as_float((code ^ ((code & 0x8000u) ? 0x8000u : 0xffffu)) << 16);  // bf16 code -> value
          }
        }
      }
      // my rows of the histograms are zero again before the thresholds are published: the math warps count the next
      // samples of this parity only after waiting for them
      {
        AlignedVector<uint32_t, 4> zeros;
        zeros.fill(0);
#pragma unroll
        for (uint32_t level = 0; level < 2; ++level) {
#pragma unroll
          for (uint32_t i = 0; i < kRows; ++i) {
#pragma unroll
            for (uint32_t v = lane_id; v < 256 / 4; v += kWarpThreads) {
              zeros.store(&hist[level][row_base + i][v * 4]);
            }
          }
        }
      }
      __syncwarp();  // every lane's zeros before lane 0's release
      // publish the thresholds for the math warps and the drains, then free the sample slots
      if (lane_id == 0) {
#pragma unroll
        for (uint32_t i = 0; i < kRows; ++i) {
          smem.thr_vals[par][row_base + i] = thr[i];
        }
        ptx::mbar_arrive(&smem.thr_full[par]);  // release: thr_vals is visible to the waiting math warps
      }
      __syncwarp();
      for (uint32_t t = 0; t < plan.num_samples; ++t) {
        if (lane_id == 0) ptx::mbar_arrive(&smem.drain_empty[slot_ring.stage]);
        slot_ring.advance();
      }
      slot_ring.advance_far(plan.num_tiles);  // the drains' slots
      if (lane_id < kRows) {
        const uint32_t row = row0 + lane_id;
        uint32_t num_samples = 0;
        float th = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < kRows; ++i) {
          if (i == lane_id) th = thr[i], num_samples = samples[i];
        }
        if (row < params.num_rows) {
          params.row_thresholds[row] = th;
          params.row_samples[row] = static_cast<int32_t>(num_samples);
        }
      }
    }
  }

  // A drain warp (part 0 in warp group 2, part 1 in warp group 3; each owns the kRows rows [part * kRows, (part + 1)
  // * kRows) of every block): per block, its rows' hits of every staged main tile filtered into the slab, against the
  // select warps' thresholds. Register arrays are indexed by r in [0, kRows); smem and the slab by the block row
  // row_base + r.
  SGL_DEVICE static void
  drain(const Fp4IndexerParams& params, Smem& smem, uint32_t lane_id, uint32_t num_blocks, uint32_t part) {
    constexpr uint32_t kRows = BLOCK_Q / kNumDrainWarps;
    const uint32_t row_base = part * kRows;
    uint32_t ns = 0;  // sampled blocks so far
    Ring<kDrainSlots> slot_ring;
    BlockPlan plan;
    for (uint32_t k = 0, block_idx = take_block(smem, 0, lane_id, plan); block_idx < num_blocks;
         block_idx = take_block(smem, ++k, lane_id, plan)) {
      const uint32_t row0 = block_idx * BLOCK_Q + row_base;  // my first row
      // my rows' bounds (part is warp-uniform: a select per row)
      uint32_t start[kRows], len[kRows];
#pragma unroll
      for (uint32_t r = 0; r < kRows; ++r) {
        start[r] = part == 0 ? plan.start[r] : plan.start[(kNumDrainWarps - 1) * kRows + r];
        len[r] = part == 0 ? plan.len[r] : plan.len[(kNumDrainWarps - 1) * kRows + r];
      }
      if (plan.dense) {
        // the math warps stored every score: count = len, T = -inf
        if (lane_id < kRows) {
          const uint32_t row = row0 + lane_id;
          uint32_t c = 0;
#pragma unroll
          for (uint32_t i = 0; i < kRows; ++i) {
            if (i == lane_id) c = len[i];
          }
          if (row < params.num_rows) {
            params.cand_count[row] = static_cast<int32_t>(c);
            params.cand_overflow[row] = 0;
            params.row_thresholds[row] = -INFINITY;
            params.row_samples[row] = 0;
          }
        }
        continue;
      }
      const uint32_t par = ns & 1u;
      ptx::mbar_wait_parity_suspend(&smem.thr_full[par], (ns >> 1) & 1u);
      ++ns;
      float thr[kRows];
#pragma unroll
      for (uint32_t i = 0; i < kRows; ++i) {
        thr[i] = smem.thr_vals[par][row_base + i];
      }
      __syncwarp();
      if (lane_id == 0) ptx::mbar_arrive(&smem.thr_empty[par]);
      slot_ring.advance(plan.num_samples);  // the select warps free the sample slots
      // ---- main tiles ----
      uint32_t thr2[kRows];
#pragma unroll
      for (uint32_t i = 0; i < kRows; ++i) {
        const uint32_t h = __float_as_uint(thr[i]) >> 16;  // bf16-exact (a code value or -inf)
        thr2[i] = h | (h << 16);
      }
      uint32_t count[kRows];  // my rows' candidates
      if constexpr (kRows == 1) {
        // One row per drain warp (64 heads). A pass covers kDrainPass tiles; lane l owns the row's 32-key word of
        // tile l / 8 that math warp l % 8 reduced: its hit bits come from that warp's mask (or, for tiles staged
        // before the threshold was published, from the staged scores), a scan over the lanes gives the slab slots,
        // and max(hits per word) rounds of one score load and two predicated stores write them (highest key first,
        // from the word's last slot down).
        static_assert(kDrainPass * kNumMathWarps == kWarpThreads, "one mask word per lane and pass");
        const uint32_t my_u = lane_id / kNumMathWarps, my_w = lane_id % kNumMathWarps;
#pragma unroll
        for (uint32_t i = 0; i < kRows; ++i) {
          count[i] = 0;
        }
        // kDrainPass tiles per pass, tiles [0, kMaskStart) (filtered from the staged scores) in passes of their own.
        const uint32_t mask_start = min(kMaskStart, plan.num_tiles);
        for (uint32_t t = 0; t < plan.num_tiles;) {
          const bool from_scores = t < mask_start;
          const uint32_t nt = min(kDrainPass, (from_scores ? mask_start : plan.num_tiles) - t);
          for (uint32_t u = 0; u < nt; ++u) {
            Ring<kDrainSlots> r = slot_ring;
            r.advance(u);
            // suspended: a spinning wait takes issue slots from the math warps of this SM sub-partition
            ptx::mbar_wait_parity_suspend(&smem.drain_full[r.stage], r.phase);
          }
          Ring<kDrainSlots> my_ring = slot_ring;
          my_ring.advance(my_u);
          const uint32_t slot = my_ring.stage;
          const uint32_t word_key = plan.kv_base + (t + my_u) * kTileKeys + my_w * kWarpThreads;
#pragma unroll
          for (uint32_t i = 0; i < kRows; ++i) {
            uint32_t m = 0;
            if (my_u < nt) {
              if (from_scores) {
#pragma unroll
                for (uint32_t c = 0; c < kWarpThreads / 8; ++c) {
                  AlignedVector<uint32_t, 4> v;
                  v.load(&smem.drain_stage[slot][row_base + i][my_w * kWarpThreads + 8 * c]);
                  const uint32_t key0 = word_key + 8 * c;
                  const int32_t lo = min(max(static_cast<int32_t>(start[i] - key0), 0), 8);
                  const int32_t hi = min(max(static_cast<int32_t>(start[i] + len[i] - key0), 0), 8);
                  m |= (hits8(v, thr2[i]) & ((1u << hi) - 1u) & ~((1u << lo) - 1u)) << (8 * c);
                }
              } else {
                m = smem.drain_mask[slot][my_w][row_base + i];
              }
            }
            const uint32_t n = __popc(m);
            const uint32_t inc = warp::inclusive_sum(n);
            const uint32_t top = count[i] + inc - 1;  // the word's last slot
            count[i] += __shfl_sync(warp::kFullMask, inc, kWarpThreads - 1);
            const uint32_t rounds = __reduce_max_sync(warp::kFullMask, n);
            const uint32_t key = word_key - start[i];
            const uint16_t* const sp = &smem.drain_stage[slot][row_base + i][my_w * kWarpThreads];
            uint16_t* const scores = static_cast<uint16_t*>(params.cand_scores) + static_cast<int64_t>(row0 + i) * kCap;
            int32_t* const indices = params.cand_indices + static_cast<int64_t>(row0 + i) * kCap;
            // round r writes the word's (r+1)-th highest hit to slot top - r; a row past kCap (rare) also tests lo
            const uint32_t lo = top + 1 > kCap ? top + 1 - kCap : 0u;
            const auto store_rounds = [&](auto capped_t) {
              constexpr bool kCapped = decltype(capped_t)::value;
              int32_t left = static_cast<int32_t>(n);  // hits of the word not yet written
              for (uint32_t r0 = 0; r0 < rounds; r0 += kRoundUnroll) {
                uint16_t* const ps = scores + (top - r0);
                int32_t* const pi = indices + (top - r0);
#pragma unroll
                for (uint32_t k = 0; k < kRoundUnroll; ++k) {
                  const uint32_t j = ptx::bfind(m | 1u);  // key 0 once the word is empty (the store is off)
                  m &= ~(1u << j);
                  const uint32_t h = sp[j];
                  const bool store = left > static_cast<int32_t>(k) && (!kCapped || r0 + k >= lo);
                  ptx::st_global_u16_if(store, ps - k, h);
                  ptx::st_global_if(store, pi - k, key + j);
                }
                left -= kRoundUnroll;
              }
            };
            if (count[i] > kCap) {  // uniform
              store_rounds(std::true_type{});
            } else {
              store_rounds(std::false_type{});
            }
          }
          __syncwarp();  // every score of the slots was read
          if (lane_id == 0) {
            Ring<kDrainSlots> r = slot_ring;
            for (uint32_t u = 0; u < nt; ++u, r.advance()) {
              ptx::mbar_arrive(&smem.drain_empty[r.stage]);
            }
          }
          slot_ring.advance(nt);
          t += nt;
        }
      } else {
        // Several rows per drain warp (32 heads: two); one word per lane serializes them, so per tile lane l owns
        // kLaneKeys keys of one of my rows (l / kRowLanes): their hit bits come from the math warp's mask (or, for
        // tiles staged before the threshold was published, from the staged scores), a scan over the row's lanes gives
        // the slab slots, and max(hits per lane) rounds of one score load and two predicated stores write them
        // (highest key first, from the chain's last slot down).
        constexpr uint32_t kRowLanes = kWarpThreads / kRows;   // 16
        constexpr uint32_t kLaneKeys = kTileKeys / kRowLanes;  // 16
        const uint32_t my_r = lane_id / kRowLanes, my_row = row_base + my_r, seg_lane = lane_id % kRowLanes;
        const uint32_t first = seg_lane * kLaneKeys;  // my keys within the tile
        const uint32_t word = first / kWarpThreads, shift = first % kWarpThreads;
        uint32_t my_start = 0, my_len = 0, my_thr2 = 0, my_count = 0;
#pragma unroll
        for (uint32_t i = 0; i < kRows; ++i) {
          if (i == my_r) my_start = start[i], my_len = len[i], my_thr2 = thr2[i];
        }
        uint16_t* const my_scores =
            static_cast<uint16_t*>(params.cand_scores) + static_cast<int64_t>(row0 + my_r) * kCap;
        int32_t* const my_indices = params.cand_indices + static_cast<int64_t>(row0 + my_r) * kCap;
        constexpr uint32_t kLaneMask = kLaneKeys == 32 ? 0xffffffffu : (1u << kLaneKeys) - 1u;
        // Tiles [0, kMaskStart): staged before the math warps had the thresholds, filtered here from the scores.
        const uint32_t mask_start = min(kMaskStart, plan.num_tiles);
        for (uint32_t t = 0; t < mask_start; ++t) {
          // suspended: a spinning wait takes issue slots from the math warps of this SM sub-partition
          ptx::mbar_wait_parity_suspend(&smem.drain_full[slot_ring.stage], slot_ring.phase);
          const uint32_t slot_idx = slot_ring.stage;
          const uint32_t tile_key = plan.kv_base + t * kTileKeys;
          uint32_t m = 0;
#pragma unroll
          for (uint32_t c = 0; c < kLaneKeys / 8; ++c) {
            AlignedVector<uint32_t, 4> v;
            v.load(&smem.drain_stage[slot_idx][my_row][first + 8 * c]);
            const uint32_t key0 = tile_key + first + 8 * c;
            const int32_t lo = min(max(static_cast<int32_t>(my_start - key0), 0), 8);
            const int32_t hi = min(max(static_cast<int32_t>(my_start + my_len - key0), 0), 8);
            m |= (hits8(v, my_thr2) & ((1u << hi) - 1u) & ~((1u << lo) - 1u)) << (8 * c);
          }
          const uint32_t n = __popc(m);
          const uint32_t inc = warp::inclusive_sum<kRowLanes>(n, seg_lane);
          const uint32_t total = __shfl_sync(warp::kFullMask, inc, kRowLanes - 1, kRowLanes);
          uint32_t sl = my_count + inc - n;
          my_count += total;
          const uint32_t rounds = __reduce_max_sync(warp::kFullMask, n);
          for (uint32_t r = 0; r < rounds; ++r) {
            const bool hit = m != 0;
            const uint32_t j = (__ffs(m) - 1) & (kLaneKeys - 1);
            m &= m - 1;
            const uint32_t h = smem.drain_stage[slot_idx][my_row][first + j];
            const bool store = hit && sl < kCap;
            ptx::st_global_u16_if(store, my_scores + sl, h);
            ptx::st_global_if(store, my_indices + sl, tile_key + first + j - my_start);
            sl += hit ? 1u : 0u;
          }
          __syncwarp();  // every score of the slot was read
          if (lane_id == 0) ptx::mbar_arrive(&smem.drain_empty[slot_idx]);
          slot_ring.advance();
        }
        // Tiles [kMaskStart, end): the math warps' masks, kDrainPass tiles per pass (their chains interleave and
        // share one round count).
        for (uint32_t t = mask_start; t < plan.num_tiles; t += kDrainPass) {
          const uint32_t nt = min(kDrainPass, plan.num_tiles - t);
          uint32_t m[kDrainPass], slot_idx[kDrainPass];
#pragma unroll
          for (uint32_t u = 0; u < kDrainPass; ++u) {
            Ring<kDrainSlots> r = slot_ring;
            r.advance(u);
            slot_idx[u] = r.stage;
            m[u] = 0;
            if (u < nt) {
              ptx::mbar_wait_parity_suspend(&smem.drain_full[r.stage], r.phase);
              m[u] = (smem.drain_mask[r.stage][word][my_row] >> shift) & kLaneMask;
            }
          }
          // per chain: hits n, last slot top (the highest key goes there first), rounds >= lo land below kCap
          uint32_t n[kDrainPass], top[kDrainPass], lo[kDrainPass], key[kDrainPass];
          uint32_t most = 0;
#pragma unroll
          for (uint32_t u = 0; u < kDrainPass; ++u) {
            n[u] = __popc(m[u]);
            const uint32_t inc = warp::inclusive_sum<kRowLanes>(n[u], seg_lane);
            const uint32_t total = __shfl_sync(warp::kFullMask, inc, kRowLanes - 1, kRowLanes);
            top[u] = my_count + inc - 1;
            lo[u] = top[u] + 1 > kCap ? top[u] + 1 - kCap : 0u;
            key[u] = plan.kv_base + (t + u) * kTileKeys + first - my_start;
            my_count += total;
            most = max(most, n[u]);
          }
          const uint32_t rounds = __reduce_max_sync(warp::kFullMask, most);
          // round r writes each chain's (r+1)-th highest hit to slot top - r; a row past kCap (rare) also tests lo
          const auto store_rounds = [&](auto capped_t) {
            constexpr bool kCapped = decltype(capped_t)::value;
            const uint16_t* sp[kDrainPass];
#pragma unroll
            for (uint32_t u = 0; u < kDrainPass; ++u) {
              sp[u] = &smem.drain_stage[slot_idx[u]][my_row][first];
            }
            int32_t left[kDrainPass];  // hits of the chain not yet written
#pragma unroll
            for (uint32_t u = 0; u < kDrainPass; ++u) {
              left[u] = static_cast<int32_t>(n[u]);
            }
            for (uint32_t r0 = 0; r0 < rounds; r0 += kRoundUnroll) {
#pragma unroll
              for (uint32_t u = 0; u < kDrainPass; ++u) {
                uint16_t* const ps = my_scores + (top[u] - r0);
                int32_t* const pi = my_indices + (top[u] - r0);
#pragma unroll
                for (uint32_t k = 0; k < kRoundUnroll; ++k) {
                  const uint32_t j = ptx::bfind(m[u] | 1u);  // key 0 once the chain is empty (the store is off)
                  m[u] &= ~(1u << j);
                  const uint32_t h = sp[u][j];
                  const bool store = left[u] > static_cast<int32_t>(k) && (!kCapped || r0 + k >= lo[u]);
                  ptx::st_global_u16_if(store, ps - k, h);
                  ptx::st_global_if(store, pi - k, key[u] + j);
                }
                left[u] -= kRoundUnroll;
              }
            }
          };
          if (__any_sync(warp::kFullMask, my_count > kCap)) {
            store_rounds(std::true_type{});
          } else {
            store_rounds(std::false_type{});
          }
          __syncwarp();  // every score of the slots was read
          if (lane_id == 0) {
            for (uint32_t u = 0; u < nt; ++u) {
              ptx::mbar_arrive(&smem.drain_empty[slot_idx[u]]);
            }
          }
          slot_ring.advance(nt);
        }
#pragma unroll
        for (uint32_t i = 0; i < kRows; ++i) {
          count[i] = __shfl_sync(warp::kFullMask, my_count, i * kRowLanes);
        }
      }
      if (lane_id < kRows) {
        const uint32_t row = row0 + lane_id;
        uint32_t c = 0;
#pragma unroll
        for (uint32_t i = 0; i < kRows; ++i) {
          if (i == lane_id) c = count[i];
        }
        if (row < params.num_rows) {
          params.cand_count[row] = static_cast<int32_t>(c);
          params.cand_overflow[row] = c > kCap ? 1 : 0;
        }
      }
    }
  }
};

template <typename Impl>
__global__ void __launch_bounds__(Impl::kNumThreads, 1)
    fp4_indexer_kernel(const __grid_constant__ Fp4IndexerParams params) {
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  Impl::run(params, *reinterpret_cast<typename Impl::Smem*>(smem_buffer));
}

}  // namespace device::fp4_indexer

template <uint32_t kNumHeads, uint32_t kNumKVStages, uint32_t kDrainSlots>
struct Fp4IndexerKernel {
  using Impl = device::fp4_indexer::Fp4IndexerImpl<kNumHeads, kNumKVStages, kDrainSlots>;

  static void
  run(tvm::ffi::TensorView q_fp4,
      tvm::ffi::TensorView q_sf,
      tvm::ffi::TensorView kv_fp4,
      tvm::ffi::TensorView kv_sf,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView ks,
      tvm::ffi::TensorView ke,
      tvm::ffi::TensorView cand_scores,
      tvm::ffi::TensorView cand_indices,
      tvm::ffi::TensorView cand_count,
      tvm::ffi::TensorView cand_overflow,
      tvm::ffi::TensorView row_thresholds,
      tvm::ffi::TensorView row_samples,
      tvm::ffi::TensorView sched,
      int64_t topk,
      int64_t num_ctas) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto N = SymbolicSize{"keys"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({R, kNumHeads, Impl::kKeyBytes})  //
        .with_dtype<int8_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(q_fp4);
    TensorMatcher({R, kNumHeads})  //
        .with_dtype<int32_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(q_sf);
    TensorMatcher({N, Impl::kKeyBytes})  //
        .with_dtype<int8_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(kv_fp4);
    TensorMatcher({N})  //
        .with_dtype<int32_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(kv_sf);
    TensorMatcher({R, kNumHeads})  //
        .with_dtype<bf16_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(weights);
    TensorMatcher({R})  //
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(ks)
        .verify(ke)
        .verify(cand_count)
        .verify(cand_overflow)
        .verify(row_samples);
    TensorMatcher({R, Impl::kCap})  //
        .with_dtype<bf16_t>()
        .with_device(device_)
        .verify(cand_scores);
    TensorMatcher({R, Impl::kCap})  //
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(cand_indices);
    TensorMatcher({R})  //
        .with_dtype<fp32_t>()
        .with_device(device_)
        .verify(row_thresholds);
    const auto num_rows = static_cast<uint32_t>(R.unwrap());
    const auto num_kv = static_cast<uint32_t>(N.unwrap());
    const DLDevice device = device_.unwrap();
    CHECK_HOST(num_rows > 0 && num_kv > 0 && topk > 0);
    CHECK_HOST(num_kv % 4 == 0) << "the scale tensor map needs a multiple of 4 keys";

    Fp4IndexerParams params{};
    params.tmap_q = tma::make_map(
        q_fp4.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        {Impl::kKeyBytes, num_rows * kNumHeads},
        {Impl::kKeyBytes},
        {Impl::kKeyBytes, Impl::UMMA_N},
        {.swizzle = CU_TENSOR_MAP_SWIZZLE_64B});
    params.tmap_q_sf = tma::make_map(
        q_sf.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_INT32,
        {kNumHeads, num_rows},
        {kNumHeads * 4ull},
        {kNumHeads, Impl::BLOCK_Q});
    params.tmap_weights = tma::make_map(
        weights.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        {kNumHeads, num_rows},
        {kNumHeads * 2ull},
        {kNumHeads, Impl::BLOCK_Q});
    params.tmap_kv = tma::make_map(
        kv_fp4.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        {Impl::kKeyBytes, num_kv},
        {Impl::kKeyBytes},
        {Impl::kKeyBytes, Impl::kTileKeys},
        {.swizzle = CU_TENSOR_MAP_SWIZZLE_64B});
    params.tmap_sf_kv = tma::make_map(
        kv_sf.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_INT32, {num_kv, 1}, {num_kv * 4ull}, {Impl::kTileKeys, 1});
    for (uint32_t m = 0; m < 6; ++m) {
      const uint64_t stride = Impl::sample_stride(m);
      params.tmap_kv_strided[m] = tma::make_map(
          kv_fp4.data_ptr(),
          CU_TENSOR_MAP_DATA_TYPE_UINT8,
          {Impl::kKeyBytes, stride, std::max<uint64_t>(num_kv / stride, 1)},
          {Impl::kKeyBytes, stride * Impl::kKeyBytes},
          {Impl::kKeyBytes, 1, Impl::kTileKeys},
          {.swizzle = CU_TENSOR_MAP_SWIZZLE_64B});
      params.sample_limit[m] = static_cast<uint32_t>(num_kv / stride * stride);
    }
    params.q_fp4 = static_cast<const int8_t*>(q_fp4.data_ptr());
    params.q_sf = static_cast<const int32_t*>(q_sf.data_ptr());
    params.kv_fp4 = static_cast<const int8_t*>(kv_fp4.data_ptr());
    params.kv_sf = static_cast<const int32_t*>(kv_sf.data_ptr());
    params.weights = weights.data_ptr();
    params.ks = static_cast<const int32_t*>(ks.data_ptr());
    params.ke = static_cast<const int32_t*>(ke.data_ptr());
    params.cand_scores = cand_scores.data_ptr();
    params.cand_indices = static_cast<int32_t*>(cand_indices.data_ptr());
    params.cand_count = static_cast<int32_t*>(cand_count.data_ptr());
    params.cand_overflow = static_cast<int32_t*>(cand_overflow.data_ptr());
    params.row_thresholds = static_cast<float*>(row_thresholds.data_ptr());
    params.row_samples = static_cast<int32_t*>(row_samples.data_ptr());
    params.num_rows = num_rows;
    params.num_kv = num_kv;
    params.topk = static_cast<uint32_t>(topk);
    TensorMatcher({1})  // sched
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(sched);
    params.sched = static_cast<uint32_t*>(sched.data_ptr());

    const auto kernel = device::fp4_indexer::fp4_indexer_kernel<Impl>;
    const auto num_sms = init_once([&] {
      runtime::set_max_dynamic_smem(kernel, Impl::kSmemBytes);
      return runtime::get_sm_count(device.device_id);
    });
    const uint32_t num_blocks = div_ceil(num_rows, Impl::BLOCK_Q);
    const uint32_t grid = std::min(num_ctas > 0 ? static_cast<uint32_t>(num_ctas) : num_sms, num_blocks);
    LaunchKernel(grid, Impl::kNumThreads, device, Impl::kSmemBytes)(kernel, params);
  }
};

}  // namespace sglang
