// SPDX-License-Identifier: Apache-2.0
// Batched int8 DSpark markov walk (serving: bs 5..64 at the standard vocabulary, every bs 1..64 for big vocabularies):
// wgmma with u as the register A operand and W2 as the SMEM B operand, so the register file holds no W2 and W2 may
// partly stream from L2.
//
//   for every request b:  prev_b = anchor_b;  for k < K:  logit = bf16(base[b, k] + W2 W1[prev_b]);
//                         tok[b, k] = argmax(logit)  or  a sample of softmax(logit / T_b);  prev_b = tok[b, k]
//
// GEMM per (32-request block, 64-row W2 tile): wgmma m64n64k32 s8 over 8 k-chunks, A from registers. M row 16 w + g is
// q_hi of request 8 w + g of the block and row 16 w + g + 8 its q_lo, so a thread's A fragment for chunk j is
// {hi[32j + 4tig ..], lo[..], hi[32j + 16 + 4tig ..], lo[..]}; the host stores W1 rows in that byte order (w1f), and
// the per-step gather is 8 LDG.128 straight into the fragment registers. Accumulator d[4i + e] / d[4i + 2 + e] is the
// hi / lo dot with tile column n = 8i + 2tig + e. The host permutes every tile's rows (column n is tile row
// i < 4 ? 8tig + 2i + e : 32 + 8tig + 2(i - 4) + e), so a thread owns one request and two aligned 8-row runs
// (8tig .. +8 and 32 + 8tig .. +8): base and corrected move as 16-B accesses and the running best is one (value, row).
// bias = s_lo float(256 d_hi + d_lo) (exact int32), logit = fma(row scale, bias, base), rounded to bf16 once (RNE);
// argmax and sampler read the rounded values and corrected receives exactly those bits. Sampling per 8-row item: the
// row by inverse CDF inside the item (uniform23), the item by Gumbel-max on log2 of its mass (gumbel2_fast).
//
// W2 per CTA: kTiles tiles of 64 rows (16 KiB, canonical K-major B layout); kRes are SMEM-resident (TMA once per
// launch), the others (bit t of kStreamMask = tile t) stream every step from L2 through a kRing-slot TMA ring.
// Ring protocol: streamed tiles form one FIFO q = step * num_streamed + rank of the tile among the streamed ones; each
// consuming warpgroup takes its tiles in increasing q. The warp whose arrival completes q's consumption (monotone
// per-slot count, 4 arrivals per consuming warpgroup) refills the slot with q + kRing: it writes the slot's tag
// s_slot_q = q + kRing, arms the full barrier and issues the TMA. A consumer waits for its tag before the parity wait,
// since warpgroups skip each other's fills and a parity cannot tell fill f from fill f - 2.
// Ordering (CTA scope): arrivals are acq_rel RMWs, so the refill's TMA write follows every consumer's completed wgmma
// reads of the slot; the tag is a release exchange polled by acquire loads, so a consumer that sees tag q also sees the
// completion of fill q - kRing, which the refiller observed. Deadlock-free for any mask: the smallest unconsumed q is
// always filled, and every warpgroup reaches it without waiting on a larger q.
//
// Work split: ceil(nb / 32) request blocks, 4 / blocks warpgroups per block; warpgroup w serves block w / wpb and
// tiles t = w (mod wpb). Rows reach a thread in increasing order, so strict > keeps the smallest row. After the tiles,
// warp 0 runs the cross-CTA exchange for every request (lane l: requests l and l + 32, polled together). Vocabulary
// rows [valid_rows, rows_pad), the last CTA's last tiles, are never loaded or stored. State: as
// dspark_markov_walk_small_batch.cuh.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include "dspark_markov_walk_common.cuh"
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>

namespace sglang::device::ptx {

SGL_DEVICE void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

SGL_DEVICE void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

template <int32_t N>
SGL_DEVICE void wgmma_wait_group() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}

// d (+)= a b, m64n64k32 s8 -> s32 with A from registers; scale_d = 0 overwrites d
SGL_DEVICE void
wgmma_m64n64k32_s8_rs(int32_t (&d)[32], const uint4& a, uint32_t desc_lo, uint32_t desc_hi, int32_t scale_d) {
  asm volatile(
      "{\n.reg .pred p;\n.reg .b64 bd;\nsetp.ne.b32 p, %37, 0;\nmov.b64 bd, {%36, %38};\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "{%32, %33, %34, %35}, bd, p;\n}"
      // clang-format off
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
        "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
        "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
        "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
      // clang-format on
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(desc_lo), "r"(scale_d), "r"(desc_hi));
}

// No instruction: keeps every later read of d below the wgmma wait, as the compiler does not see the async write
SGL_DEVICE void wgmma_fence_operand(int32_t (&d)[32]) {
#pragma unroll
  for (int32_t i = 0; i < 32; ++i) {
    asm volatile("" : "+r"(d[i])::"memory");
  }
}

// The read-only path: u rows are shared by many CTAs and L1 merges them
SGL_DEVICE uint4 ld_nc_v4(const void* ptr) {
  uint4 value;
  asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(ptr));
  return value;
}

SGL_DEVICE uint4 ld_hint_v4(const void* ptr, uint64_t policy) {
  uint4 value;
  asm volatile("ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(ptr), "l"(policy));
  return value;
}

SGL_DEVICE void st_hint_v4(void* ptr, uint4 value, uint64_t policy) {
  asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;" ::"l"(ptr),
               "r"(value.x),
               "r"(value.y),
               "r"(value.z),
               "r"(value.w),
               "l"(policy)
               : "memory");
}

SGL_DEVICE uint64_t create_policy_evict_last() {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
  return policy;
}

// The ring hand-off; the writer sides are RMWs, which racecheck does not report against the polls
SGL_DEVICE int32_t ld_acquire_cta_s32(const int32_t* ptr) {
  int32_t value;
  asm volatile("ld.acquire.cta.shared::cta.s32 %0, [%1];" : "=r"(value) : "r"(to_shared(ptr)) : "memory");
  return value;
}

SGL_DEVICE void exch_release_cta_s32(int32_t* ptr, int32_t value) {
  asm volatile("{ .reg .b32 o; atom.release.cta.shared::cta.exch.b32 o, [%0], %1; }" ::"r"(to_shared(ptr)), "r"(value)
               : "memory");
}

SGL_DEVICE int32_t add_acq_rel_cta_s32(int32_t* ptr, int32_t value) {
  int32_t old;
  asm volatile("atom.acq_rel.cta.shared::cta.add.u32 %0, [%1], %2;"
               : "=r"(old)
               : "r"(to_shared(ptr)), "r"(value)
               : "memory");
  return old;
}

}  // namespace sglang::device::ptx

namespace sglang::dspark_markov_walk::wgmma {

inline constexpr int32_t kCtaThreads = 512;
inline constexpr int32_t kWarpgroups = 4;
inline constexpr int32_t kW2TileRows = 64;  // the wgmma N
inline constexpr int32_t kW2TileBytes = kW2TileRows * 256;
inline constexpr int32_t kChunkBytes = kW2TileBytes / 8;  // one k32 chunk
inline constexpr int32_t kBlockRequests = 32;             // M = 64: the hi and lo rows of 32 requests
// w1f rows have the size and the scale words of W1q rows, with the 512 q bytes in A-fragment order

// K-major, no swizzle, LBO 128, SBO 256: the high word is the constant SBO field, the low word start address | LBO.
// Offsets inside the 228 KiB window never carry out of the 14-bit address field, so they are 32-bit adds.
inline constexpr uint32_t kDescHi = 256 >> 4;
SGL_DEVICE uint32_t smem_desc_lo(uint32_t addr) {
  return ((addr & 0x3FFFF) >> 4) | ((128 >> 4) << 16);
}

template <int32_t kTiles, int32_t kRes, int32_t kRing>
struct Smem {
  uint8_t res[kRes * kW2TileBytes];    // resident tiles
  uint8_t ring[kRing * kW2TileBytes];  // the streamed tiles' TMA ring
  float scale[kTiles * kW2TileRows];   // natural row order
};

template <int32_t kTiles, int32_t kRes, int32_t kRing, uint32_t kStreamMask>
__global__ void __launch_bounds__(kCtaThreads, 1) markov_walk_wgmma_kernel(
    const uint8_t* __restrict__ w2_res,
    const uint8_t* __restrict__ w2_str,
    const float* __restrict__ row_scale,
    const uint32_t* __restrict__ w1f,
    const bf16_t* __restrict__ base,
    const int64_t* __restrict__ anchor,
    int64_t* __restrict__ tokens,
    bf16_t* __restrict__ corrected,
    uint64_t* __restrict__ state,
    const float* __restrict__ temps,
    int32_t nb,
    int32_t num_steps,
    int32_t state_steps,
    int32_t kb,
    int32_t ld,
    int32_t valid_rows,
    uint64_t seed) {
  constexpr int32_t kRowsCta = kTiles * kW2TileRows;
  constexpr int32_t kStr = kTiles - kRes;
  extern __shared__ __align__(128) uint8_t smem_raw[];
  auto& smem = *reinterpret_cast<Smem<kTiles, kRes, kRing>*>(smem_raw);
  __shared__ uint64_t s_mbar_res[kRes + 1];  // one per resident tile (its first use waits for it), then the scales
  __shared__ uint64_t s_mbar_full[kRing];
  __shared__ int32_t s_slot_q[kRing];      // the FIFO index the slot holds or is being filled with
  __shared__ int32_t s_slot_count[kRing];  // monotone: 4 warps per consuming warpgroup per fill
  __shared__ uint64_t s_wg_best[kMaxBatch][kWarpgroups];
  __shared__ uint32_t s_token[kMaxBatch];
  __shared__ float s_inv_t[kMaxBatch];
  const int32_t lane_id = threadIdx.x & 31, warp_id = threadIdx.x >> 5;
  const int32_t wg_id = warp_id >> 2, wg_warp = warp_id & 3, g = lane_id >> 2, tig = lane_id & 3;
  const int32_t row_base = blockIdx.x * kRowsCta;

  const auto round = ptx::ld_relaxed_u64(state);
  const int32_t set = static_cast<int32_t>(round & 1);
  const int32_t set_words = state_steps * kStepWords;
  const auto pairs = state + 16 + set * set_words;
  const auto pair = [&](int32_t k, int32_t b) { return pairs + kPairWords * (k * kMaxBatch + b); };
  if (blockIdx.x == 0) {
    for (int32_t i = threadIdx.x; i < state_steps * kMaxBatch * 2; i += kCtaThreads) {
      ptx::st_relaxed_u64(state + 16 + (set ^ 1) * set_words + (i >> 1) * kPairWords + (i & 1), 0);
    }
  }
  const auto philox_key = make_uint2(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32));

  const int32_t num_blocks = (nb + kBlockRequests - 1) / kBlockRequests, wpb = kWarpgroups / num_blocks;
  const int32_t block = wg_id / wpb, phase = wg_id % wpb;
  const int32_t b = block * kBlockRequests + 8 * wg_warp + g;  // this thread's request
  const bool live = b < nb;
  // Run 0 (tile rows 8 tig .. +8) / run 1 (32 + 8 tig ..) of tile t is a valid vocabulary run iff t < lim0 / lim1; a
  // run is all valid or all padding. Dead requests get 0, which also skips their base loads.
  const int32_t local_rows = valid_rows - row_base;
  const int32_t lim0 = live ? min(kTiles, max(0, local_rows - 8 * tig + kW2TileRows - 1) / kW2TileRows) : 0;
  const int32_t lim1 = live ? min(kTiles, max(0, local_rows - 32 - 8 * tig + kW2TileRows - 1) / kW2TileRows) : 0;
  const int32_t consumers = 4 * num_blocks;  // ring arrivals per fill, 4 warps per warpgroup
  const int32_t num_fills = kStr * num_steps;
  const int32_t last_step_q = num_fills - kStr;  // fills q >= this load the last step's tiles
  const auto stream_rank = [&](int32_t t) { return __popc(kStreamMask & ((1u << t) - 1u)); };

  // Once per round: resident tiles and row scales by TMA, then the ring prefill
  const auto evict_last = ptx::create_policy_evict_last(), evict_first = ptx::create_policy_evict_first();
  const auto issue_fill = [&](int32_t q) {  // thread-local: tag, arm, copy (q < num_fills)
    const int32_t slot = q % kRing;
    ptx::exch_release_cta_s32(&s_slot_q[slot], q);
    ptx::mbar_arrive_expect_tx(&s_mbar_full[slot], kW2TileBytes);
    ptx::cp_async_bulk_g2s(
        smem.ring + slot * kW2TileBytes,
        w2_str + (static_cast<size_t>(blockIdx.x) * kStr + q % kStr) * kW2TileBytes,
        kW2TileBytes,
        &s_mbar_full[slot],
        q >= last_step_q ? evict_first : evict_last);  // last step: demote the lines for whatever runs next
  };
  if (threadIdx.x == 0) {
    for (int32_t i = 0; i <= kRes; ++i) {
      ptx::mbar_init(&s_mbar_res[i], 1);
    }
    for (int32_t s = 0; s < kRing; ++s) {
      ptx::mbar_init(&s_mbar_full[s], 1);
      s_slot_count[s] = 0;
    }
    ptx::fence_mbarrier_init();
    // Scales first (every epilogue needs them), then the ring prefill, then the resident tiles in first-use order
    ptx::mbar_arrive_expect_tx(&s_mbar_res[kRes], sizeof(smem.scale));
    ptx::cp_async_bulk_g2s(smem.scale, row_scale + row_base, sizeof(smem.scale), &s_mbar_res[kRes], evict_first);
    for (int32_t q = 0; q < kRing && q < num_fills; ++q) {
      issue_fill(q);
    }
    for (int32_t i = 0; i < kRes; ++i) {
      ptx::mbar_arrive_expect_tx(&s_mbar_res[i], kW2TileBytes);
      ptx::cp_async_bulk_g2s(
          smem.res + i * kW2TileBytes,
          w2_res + (static_cast<size_t>(blockIdx.x) * kRes + i) * kW2TileBytes,
          kW2TileBytes,
          &s_mbar_res[i],
          evict_first);
    }
  }
  for (int32_t i = threadIdx.x; i < nb; i += kCtaThreads) {
    s_token[i] = clamp_row(anchor[i], valid_rows);
    s_inv_t[i] = inv_temperature(temps[i]);
  }
  __syncthreads();  // barriers initialized, as no thread touches one before this; the loads are still in flight
  ptx::mbar_wait(&s_mbar_res[kRes], 0);

  const float inv_t = live ? s_inv_t[b] : 0.f;
  const float inv_t_log2e = inv_t * 1.4426950408889634f;
  const auto res_desc = smem_desc_lo(ptx::to_shared(smem.res)), ring_desc = smem_desc_lo(ptx::to_shared(smem.ring));
  auto next0 = make_uint4(0, 0, 0, 0), next1 = make_uint4(0, 0, 0, 0);  // the next tile's base runs (bf16 x 8)

  for (int32_t k = 0; k < num_steps; ++k) {
    // u fragments of this thread's request; dead requests take zeros, so the warpgroup's MMA stays well defined
    uint4 a[8];
    float s_lo = 0.f;
    if (live) {
      const auto src = w1f + static_cast<size_t>(s_token[b]) * kURowWords;
#pragma unroll
      for (int32_t j = 0; j < 8; ++j) {
        a[j] = ptx::ld_nc_v4(src + j * 16 + tig * 4);
      }
      s_lo = __uint_as_float(__ldg(src + 129));
    } else {
#pragma unroll
      for (int32_t j = 0; j < 8; ++j) {
        a[j] = make_uint4(0, 0, 0, 0);
      }
    }
    const int32_t base_off = (b * kb + k) * ld + row_base + 8 * tig;  // (b, k, run 0 of tile 0)
    if (k == 0) {  // later steps load the first tile's base before the previous step's exchange
      if (phase < lim0) next0 = ptx::ld_hint_v4(base + base_off + phase * kW2TileRows, evict_first);
      if (phase < lim1) next1 = ptx::ld_hint_v4(base + base_off + phase * kW2TileRows + 32, evict_first);
    }
    float cur_v = -INFINITY;
    int32_t cur_r = -1;

#pragma unroll 2
    for (int32_t t = phase; t < kTiles; t += wpb) {
      // mask & bit, not (kStreamMask >> t) & 1: the shift form compiles to a longer bit test (2 more SASS per tile)
      const bool streamed = (kStreamMask & (1u << t)) != 0u;
      uint32_t desc;
      int32_t q = 0;
      if (streamed) {
        q = k * kStr + stream_rank(t);
        const int32_t slot = q % kRing;
        while (ptx::ld_acquire_cta_s32(&s_slot_q[slot]) != q) {
        }
        ptx::mbar_wait(&s_mbar_full[slot], static_cast<uint32_t>((q / kRing) & 1));
        desc = ring_desc + slot * (kW2TileBytes >> 4);
      } else {
        const int32_t ri = t - stream_rank(t);
        if (k == 0) ptx::mbar_wait(&s_mbar_res[ri], 0);  // first use of a resident tile: its own TMA overlaps step 0
        desc = res_desc + ri * (kW2TileBytes >> 4);
      }
      int32_t d[32];
      ptx::wgmma_fence();
#pragma unroll
      for (int32_t j = 0; j < 8; ++j) {
        ptx::wgmma_m64n64k32_s8_rs(d, a[j], desc + j * (kChunkBytes >> 4), kDescHi, j);
      }
      ptx::wgmma_commit_group();
      // This tile's base was loaded one tile ago and the next tile's goes out under this MMA. Runs past valid_rows
      // (and dead requests) load nothing: their registers keep a stale tile, masked below.
      const int32_t o = base_off + t * kW2TileRows;
      const auto cur0 = next0, cur1 = next1;
      if (t + wpb < lim0) next0 = ptx::ld_hint_v4(base + o + wpb * kW2TileRows, evict_first);
      if (t + wpb < lim1) next1 = ptx::ld_hint_v4(base + o + wpb * kW2TileRows + 32, evict_first);
      ptx::wgmma_wait_group<0>();
      ptx::wgmma_fence_operand(d);
      if (streamed && lane_id == 0) {  // this warp is done reading the slot
        const int32_t slot = q % kRing;
        const int32_t done = ptx::add_acq_rel_cta_s32(&s_slot_count[slot], 1) + 1;
        if (done == (q / kRing + 1) * consumers && q + kRing < num_fills) issue_fill(q + kRing);
      }
      if (!live) continue;
      device::AlignedVector<float, 4> scale_v[4];  // run 0 rows, then run 1 rows
      scale_v[0].load(smem.scale + t * kW2TileRows + 8 * tig);
      scale_v[1].load(smem.scale + t * kW2TileRows + 8 * tig + 4);
      scale_v[2].load(smem.scale + t * kW2TileRows + 32 + 8 * tig);
      scale_v[3].load(smem.scale + t * kW2TileRows + 32 + 8 * tig + 4);
      const uint32_t base_w[8] = {cur0.x, cur0.y, cur0.z, cur0.w, cur1.x, cur1.y, cur1.z, cur1.w};
      float lg[16];  // [run][row]: run 0 = i < 4, row 2 (i % 4) + e
#pragma unroll
      for (int32_t i = 0; i < 8; ++i) {
#pragma unroll
        for (int32_t e = 0; e < 2; ++e) {
          const int32_t idx = 8 * (i >> 2) + 2 * (i & 3) + e;
          const float bias = s_lo * static_cast<float>(d[4 * i + e] * 256 + d[4 * i + 2 + e]);
          const uint32_t w = base_w[idx >> 1];
          lg[idx] = fmaf(scale_v[idx / 4][idx % 4], bias, __uint_as_float(e ? (w & 0xFFFF0000u) : (w << 16)));
        }
      }
      // The walk's logits are the bf16 roundings (RNE) of these, which the verifier reads back from corrected:
      // pk[4 run + i] packs rows 2i, 2i + 1 of the run, corrected's 16 B in memory order. Greedy works on pk directly;
      // the sampler unpacks (bf16 -> fp32 is a shift).
      uint32_t pk[8];
#pragma unroll
      for (int32_t i = 0; i < 8; ++i) {
        pk[i] = bf16x2_bits(__floats2bfloat162_rn(lg[2 * i], lg[2 * i + 1]));
      }
      const int32_t r = row_base + t * kW2TileRows + 8 * tig;  // run 0 rows r .. r + 8, run 1 rows r + 32 .. r + 40
      if (row_base + (t + 1) * kW2TileRows > valid_rows) {  // the last CTA's padding runs, uniform per tile: bf16 -inf
#pragma unroll
        for (int32_t run = 0; run < 2; ++run) {
          if (r + 32 * run >= valid_rows) {
#pragma unroll
            for (int32_t i = 0; i < 4; ++i) {
              pk[4 * run + i] = 0xFF80FF80u;
            }
          }
        }
      }
      if (inv_t > 0.f) {
        // corrected: sampling requests (the verifier's q), valid runs only
        if (corrected != nullptr) {
          if (t < lim0) ptx::st_hint_v4(corrected + o, make_uint4(pk[0], pk[1], pk[2], pk[3]), evict_first);
          if (t < lim1) ptx::st_hint_v4(corrected + o + 32, make_uint4(pk[4], pk[5], pk[6], pk[7]), evict_first);
        }
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
          lg[2 * i] = __uint_as_float(pk[i] << 16);
          lg[2 * i + 1] = __uint_as_float(pk[i] & 0xFFFF0000u);
        }
        // Uniforms: philox(run 0's item index = global row / 8, step | b << 8, round), x and y for run 0, z and w for
        // run 1 (run-1 items 8 T + 4 + tig never coincide with a run-0 counter)
        const auto x = philox(
            make_uint4(
                static_cast<uint32_t>(r >> 3),
                static_cast<uint32_t>(k) | (static_cast<uint32_t>(b) << 8),
                static_cast<uint32_t>(round),
                static_cast<uint32_t>(round >> 32)),
            philox_key);
#pragma unroll
        for (int32_t run = 0; run < 2; ++run) {
          const auto l8 = lg + 8 * run;
          const float m =
              __uint_as_float(ptx::bf16x8_max(pk[4 * run], pk[4 * run + 1], pk[4 * run + 2], pk[4 * run + 3]) << 16);
          const float m_scaled = m * inv_t_log2e;
          float cdf[8], mass = 0.f;
#pragma unroll
          for (int32_t i = 0; i < 8; ++i) {
            mass += ptx::ex2_approx_ftz(fmaf(l8[i], inv_t_log2e, -m_scaled));
            cdf[i] = mass;
          }
          const float target = uniform23(run == 0 ? x.x : x.z) * mass;  // < mass (uniform23)
          int32_t row = 7;
#pragma unroll
          for (int32_t i = 6; i >= 0; --i) {
            row = target < cdf[i] ? i : row;
          }
          const float key_v = m_scaled + ptx::lg2_approx_ftz(mass) + gumbel2_fast(run == 0 ? x.y : x.w);
          if (key_v > cur_v) {
            cur_v = key_v;
            cur_r = r + 32 * run + row;
          }
        }
      } else {
#pragma unroll
        for (int32_t run = 0; run < 2; ++run) {
          const auto p8 = pk + 4 * run;
          const auto m2 = ptx::bf16x8_max(p8[0], p8[1], p8[2], p8[3]);
          const int32_t row = ptx::bf16x8_first_eq(p8[0], p8[1], p8[2], p8[3], m2);
          const float m = __uint_as_float(m2 << 16);
          if (m > cur_v) {
            cur_v = m;
            cur_r = r + 32 * run + row;
          }
        }
      }
    }
    {  // the request's 4 tig lanes: this warpgroup's best for it
      auto key = cur_r >= 0 ? pack_key(cur_v, static_cast<uint32_t>(cur_r)) : uint64_t{0};
      key = max(key, __shfl_xor_sync(0xffffffffu, key, 1));
      key = max(key, __shfl_xor_sync(0xffffffffu, key, 2));
      if (tig == 0 && live) s_wg_best[b][phase] = key;
    }
    // The next step's base does not depend on the tokens: its first tile into registers under the exchange
    if (k + 1 < num_steps) {
      const int32_t o1 = base_off + ld + phase * kW2TileRows;
      if (phase < lim0) next0 = ptx::ld_hint_v4(base + o1, evict_first);
      if (phase < lim1) next1 = ptx::ld_hint_v4(base + o1 + 32, evict_first);
    }
    __syncthreads();
    if (warp_id == 0) {
      uint64_t key[2] = {0, 0};
#pragma unroll
      for (int32_t i = 0; i < 2; ++i) {
        const int32_t bq = lane_id + 32 * i;
        if (bq < nb) {
          uint64_t best = 0;
          for (int32_t p = 0; p < wpb; ++p) {
            best = max(best, s_wg_best[bq][p]);
          }
          ptx::red_relaxed_max_u64(pair(k, bq), best);
          ptx::red_release_add_u64(pair(k, bq) + 1, 1);
        }
      }
      // Both of a lane's requests polled in the same round trip; a request the lane does not hold starts arrived
      const uint64_t num_ctas = gridDim.x;
      uint64_t v0[2] = {0, lane_id < nb ? 0 : num_ctas}, v1[2] = {0, lane_id + 32 < nb ? 0 : num_ctas};
      do {
        if (v0[1] < num_ctas) ptx::ld_relaxed_v2_u64(pair(k, lane_id), v0);
        if (v1[1] < num_ctas) ptx::ld_relaxed_v2_u64(pair(k, lane_id + 32), v1);
      } while (v0[1] < num_ctas || v1[1] < num_ctas);
      key[0] = v0[0];
      key[1] = v1[0];
#pragma unroll
      for (int32_t i = 0; i < 2; ++i) {
        const int32_t bq = lane_id + 32 * i;
        if (bq < nb) {
          // Key 0: no candidate row anywhere, every logit NaN or -inf (a padded CUDA-graph row). key_row(0) would
          // make the next W1 gather an illegal address; emit row 0, torch.argmax's answer.
          const auto token = key[i] != 0 ? key_row(key[i]) : 0u;
          s_token[bq] = token;
          if (blockIdx.x == 0) tokens[static_cast<size_t>(bq) * num_steps + k] = token;
          // The next gather's u row into L2 before the barrier releases it, from CTA 0 only: every CTA gathers the
          // same rows, and 132 copies of the prefetch made bs 64 2-3% slower on H100.
          if (blockIdx.x == 0 && k + 1 < num_steps) {
            const auto src = w1f + static_cast<size_t>(token) * kURowWords;
#pragma unroll
            for (int32_t line = 0; line < 5; ++line) {
              ptx::prefetch_l2_evict_last(src + line * 32);  // 128-B lines
            }
          }
        }
      }
    }
    __syncthreads();
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) ptx::st_relaxed_u64(state, round + 1);
}

// One round of the wgmma walk (bs 1..64); grid = rows_pad / (64 kTiles):
//   w2_res     int8 [grid, kRes, 16384], each CTA's resident tiles in tile order, canonical K-major B layout
//              [k32 chunk][8-column group][k16 core][column][16 B], tile rows permuted as in the header
//   w2_str     int8 [grid, kTiles - kRes, 16384], the streamed tiles (kStreamMask), same layout
//   row_scale  fp32 [rows_pad], natural row order; w1f uint8 [V, 528], s_lo = s_hi / 256
//   base       bf16 [B, >= num_steps, ld]; corrected: none, or shaped like base and written for T > 0 requests only
//   anchor     int64 [B]; tokens int64 [B * num_steps], row-major; temps fp32 [B], <= 0 or NaN for greedy
//   state      as dspark_markov_walk_small_batch.cuh
template <int32_t kTiles, int32_t kRes, int32_t kRing, uint32_t kStreamMask>
inline void walk(
    tvm::ffi::TensorView w2_res,
    tvm::ffi::TensorView w2_str,
    tvm::ffi::TensorView row_scale,
    tvm::ffi::TensorView w1f,
    tvm::ffi::TensorView base,
    tvm::ffi::TensorView anchor,
    tvm::ffi::TensorView tokens,
    tvm::ffi::Optional<tvm::ffi::TensorView> corrected,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView temps,
    int64_t num_steps,
    int64_t valid_rows,
    int64_t seed) {
  using namespace host;
  static_assert(kTiles <= 32, "kStreamMask is a 32-bit word");
  static_assert(kRes >= 1 && kRing >= 1 && kTiles - kRes >= kRing, "ring depth");
  static_assert(std::popcount(kStreamMask) == kTiles - kRes, "kStreamMask must name kTiles - kRes tiles");
  static_assert(kTiles == 32 || (kStreamMask >> (kTiles % 32)) == 0, "kStreamMask bits must be below bit kTiles");
  using Layout = Smem<kTiles, kRes, kRing>;
  auto grid = SymbolicSize{"grid"};
  auto batch = SymbolicSize{"batch"};
  auto base_steps = SymbolicSize{"base_steps"};
  auto ld = SymbolicSize{"ld"};
  auto device = SymbolicDevice{};
  TensorMatcher({grid, kRes, kW2TileBytes})  //
      .with_dtype<int8_t>()
      .with_device<kDLCUDA>(device)
      .ensure_alignment(16)
      .verify(w2_res);
  TensorMatcher({grid, kTiles - kRes, kW2TileBytes})  //
      .with_dtype<int8_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(w2_str);
  const auto rows_pad = grid.unwrap() * kTiles * kW2TileRows;
  TensorMatcher({rows_pad})  //
      .with_dtype<float>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(row_scale);
  TensorMatcher({-1, kURowBytes})  //
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(w1f);
  TensorMatcher({batch, base_steps, ld})  //
      .with_dtype<bf16_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(base);
  if (corrected.has_value()) {
    TensorMatcher({batch, base_steps, ld})  //
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(corrected.value());
  }
  TensorMatcher({batch})  //
      .with_dtype<int64_t>()
      .with_device(device)
      .verify(anchor);
  TensorMatcher({batch})  //
      .with_dtype<float>()
      .with_device(device)
      .verify(temps);
  TensorMatcher({batch.unwrap() * num_steps})  //
      .with_dtype<int64_t>()
      .with_device(device)
      .verify(tokens);
  TensorMatcher({-1})  //
      .with_dtype<int64_t>()
      .with_device(device)
      .ensure_alignment(128)
      .verify(state);
  const auto nb = batch.unwrap();
  const auto state_steps = (state.size(0) - 16) / (2 * kStepWords);
  CHECK_HOST(
      1 <= num_steps && num_steps <= state_steps && state_steps <= kMaxSteps &&
      state.size(0) == 16 + 2 * kStepWords * state_steps)
      << "markov walk: need int64 state [16 + " << 2 * kStepWords << " S] with 1 <= num_steps " << num_steps
      << " <= S <= " << kMaxSteps;
  CHECK_HOST(1 <= nb && nb <= kMaxBatch) << "markov walk: batch " << nb << " outside [1, " << kMaxBatch << "]";
  CHECK_HOST(base_steps.unwrap() >= num_steps);
  CHECK_HOST(valid_rows % 8 == 0 && 8 <= valid_rows && valid_rows <= std::min({ld.unwrap(), rows_pad, w1f.size(0)}))
      << "markov walk: vocab " << valid_rows << " must be a multiple of 8 within the base logits and the weights";
  // 32-bit element offsets: every lane of the 32-request blocks forms (b kb + k) ld + row, dead lanes included
  const auto lanes = kBlockRequests * ((nb + kBlockRequests - 1) / kBlockRequests);
  CHECK_HOST(ld.unwrap() % 8 == 0 && lanes * base_steps.unwrap() * ld.unwrap() + rows_pad < (int64_t{1} << 31));
  launch_cooperative<markov_walk_wgmma_kernel<kTiles, kRes, kRing, kStreamMask>, sizeof(Layout)>(
      device.unwrap(),
      static_cast<uint32_t>(grid.unwrap()),
      kCtaThreads,
      sizeof(Layout),
      static_cast<const uint8_t*>(w2_res.data_ptr()),
      static_cast<const uint8_t*>(w2_str.data_ptr()),
      static_cast<const float*>(row_scale.data_ptr()),
      static_cast<const uint32_t*>(w1f.data_ptr()),
      static_cast<const bf16_t*>(base.data_ptr()),
      static_cast<const int64_t*>(anchor.data_ptr()),
      static_cast<int64_t*>(tokens.data_ptr()),
      corrected.has_value() ? static_cast<bf16_t*>(corrected.value().data_ptr()) : nullptr,
      static_cast<uint64_t*>(state.data_ptr()),
      static_cast<const float*>(temps.data_ptr()),
      static_cast<int32_t>(nb),
      static_cast<int32_t>(num_steps),
      static_cast<int32_t>(state_steps),
      static_cast<int32_t>(base_steps.unwrap()),
      static_cast<int32_t>(ld.unwrap()),
      static_cast<int32_t>(valid_rows),
      static_cast<uint64_t>(seed));
}

}  // namespace sglang::dspark_markov_walk::wgmma
