// SPDX-License-Identifier: Apache-2.0
// Batched int8 DSpark markov walk (serving: bs 5..64 at the standard vocabulary, every bs 1..64 for big vocabularies):
// wgmma with u as the register A operand and W2 as the SMEM B operand -- the register file holds no W2, and W2 may
// partly stream from L2.
//
//   for every request b:  prev_b = anchor_b;  for k < K:  logit = bf16(base[b, k] + W2 W1[prev_b]);
//                         tok[b, k] = argmax(logit)  or  a sample of softmax(logit / T_b);  prev_b = tok[b, k]
//
// GEMM per (32-request block, 64-row W2 tile): wgmma m64n64k32 s8 x 8 k-chunks, A from registers.  M row 16 w + g =
// q_hi of request 8 w + g of the block, row 16 w + g + 8 = its q_lo: a thread's A fragment for chunk j is {hi[32j +
// 4tig ..], lo[..], hi[32j + 16 + 4tig ..], lo[..]} -- the host stores W1 rows in exactly this byte order (w1f), so the
// per-step gather is 8 x LDG.128 straight into the fragment registers.  Accumulator d[4i + e] / d[4i + 2 + e] = hi / lo
// dot of the thread's request with tile column n = 8i + 2tig + e; the host permutes the rows of every tile (column n
// <-> tile row rho(n) = i < 4 ? 8tig + 2i + e : 32 + 8tig + 2(i - 4) + e), so a thread owns ONE request and two aligned
// 8-row runs (8tig .. +8 and 32 + 8tig .. +8): base / corrected move as 16-B loads / stores, the sampler's 8-row items
// are in-thread, the running best is one (value, row) pair.  bias = s_lo float(256 d_hi + d_lo) (exact int32), logit =
// fma(row scale, bias, base), rounded to bf16 once (RNE); argmax and sampler read the rounded values (ties -> smallest
// row), corrected receives exactly those bits.  Sampling: per 8-row item, the row by inverse CDF inside the item
// (uniform23), the item by Gumbel-max on log2 of its mass (gumbel2_fast).
//
// W2 placement per CTA (kTiles tiles of 64 rows, 16 KiB each, canonical K-major B layout): kRes tiles
// SMEM-resident (TMA once per launch), the other kTiles - kRes streamed every step from L2 through a kRing-slot TMA
// ring; which tiles stream is kStreamMask (bit t = tile t).  Ring protocol: streamed tiles form one FIFO
// q = step * nstr + index (nstr = kTiles - kRes, index = rank of the tile among the streamed ones); each consumer
// warpgroup takes its tiles in increasing q.  A slot is refilled with q + kRing by the warp whose arrival completes
// q's consumption (monotone per-slot counter, 4 arrivals per consuming warpgroup).  The refiller first writes the
// slot's tag slot_q = q + kRing, then arms the full barrier and issues the TMA; a consumer waits for its tag BEFORE
// the parity wait: warpgroups skip other warpgroups' fills, and a parity alone cannot tell fill f from fill f - 2.
// Ordering (CTA scope): arrivals are acq_rel RMWs, so the refill's TMA write follows every consumer's (completed) wgmma
// reads of the slot; the tag is a release exchange polled by acquire loads, so a consumer that sees tag q also sees
// the completion of fill q - kRing, which the refiller observed.
// Deadlock-free for any mask: the smallest unconsumed q is always filled, and every warpgroup reaches it without
// waiting on a larger q.
//
// Work split: nblk = ceil(nb / 32) request blocks, wpb = 4 / nblk warpgroups per block; warpgroup w serves block
// w / wpb and tiles t = w % wpb (mod wpb).  Rows reach a thread in increasing order: strict > keeps the smallest row.
// After the tiles, warp 0 runs the cross-CTA exchange of the common header for every request (lane l: requests l and
// l + 32, polled together).  Vocabulary rows [valid_rows, rows_pad) -- the last CTA's last tiles -- are never loaded
// (each thread's per-run tile limits lim0 / lim1 predicate its base loads) and never stored.  A request with no
// candidate row at all (every logit NaN / -inf, e.g. a padded CUDA-graph row) gets token 0, as torch.argmax would.
// SMEM per CTA = (kRes + kRing) x 16 KiB + kRowsCta x 4 B (row scales) + ~2.7 KiB static, within the 227 KiB opt-in for
// every kTiles <= 32.  State: as dspark_markov_walk_small_batch.cuh (round | pad | two sets of 128-B {key, count}
// pairs; CTA 0 clears the other set and closes the round).
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include "dspark_markov_walk_common.cuh"
#include <cstdint>

#define DSPARK_MW_WG_D32                                                    \
  "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, " \
  "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}"
#define DSPARK_MW_WG_D32_OUT(d)                                                                               \
  "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), \
      "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]),  \
      "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), \
      "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])

// PTX only this kernel uses (wgmma, cache-hinted 16-B accesses, the ring hand-off), in the common header's
// `sglang::device::ptx`
namespace sglang::device::ptx {

SGL_DEVICE void wg_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
SGL_DEVICE void wg_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
SGL_DEVICE void wg_wait0() {
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
}
// d (+)= A B, m64n64k32 s8 -> s32, A from registers; scale_d = acc ? accumulate : overwrite
SGL_DEVICE void wgmma_rs(int32_t (&d)[32], const uint4& a, uint32_t desc_lo, uint32_t desc_hi, int32_t acc) {
  asm volatile(
      "{\n.reg .pred p;\n.reg .b64 bd;\nsetp.ne.b32 p, %37, 0;\nmov.b64 bd, {%36, %38};\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 " DSPARK_MW_WG_D32 ", {%32, %33, %34, %35}, bd, p;\n}"
      : DSPARK_MW_WG_D32_OUT(d)
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(desc_lo), "r"(acc), "r"(desc_hi));
}
// no instruction: every later read of d stays below the wgmma wait (the compiler does not know d is written async)
SGL_DEVICE void wg_fence_operand(int32_t (&d)[32]) {
  asm volatile("" : DSPARK_MW_WG_D32_OUT(d)::"memory");
}
SGL_DEVICE uint4 ldg_nc(const void* p) {  // read-only path (u rows: shared by many CTAs, L1 merges)
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
SGL_DEVICE uint4 ldg_ef(const void* p, uint64_t pol) {  // streamed once (base): L2 evict_first
  uint4 v;
  asm volatile("ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p), "l"(pol));
  return v;
}
SGL_DEVICE void stg_ef(void* p, uint4 v, uint64_t pol) {
  asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;" ::"l"(p),
               "r"(v.x),
               "r"(v.y),
               "r"(v.z),
               "r"(v.w),
               "l"(pol)
               : "memory");
}
SGL_DEVICE uint64_t evict_last_policy() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
// ring hand-off (see the protocol above); the writer sides are RMWs, which racecheck does not report against the polls
SGL_DEVICE int32_t ld_acquire_s32(const int32_t* p) {
  int32_t v;
  asm volatile("ld.acquire.cta.shared::cta.s32 %0, [%1];" : "=r"(v) : "r"(to_shared(p)) : "memory");
  return v;
}
SGL_DEVICE void exch_release_s32(int32_t* p, int32_t v) {
  asm volatile("{ .reg .b32 o; atom.release.cta.shared::cta.exch.b32 o, [%0], %1; }" ::"r"(to_shared(p)), "r"(v)
               : "memory");
}
SGL_DEVICE int32_t add_acq_rel_s32(int32_t* p, int32_t v) {
  int32_t old;
  asm volatile("atom.acq_rel.cta.shared::cta.add.u32 %0, [%1], %2;" : "=r"(old) : "r"(to_shared(p)), "r"(v) : "memory");
  return old;
}

}  // namespace sglang::device::ptx

namespace sglang::dspark_markov_walk::wgmma {

constexpr int32_t kMaxB = 64;
constexpr int32_t kThr = 512, kNumWG = 4;
constexpr int32_t kTileRows = 64;
constexpr int32_t kTileBytes = 16384, kChunkBytes = 2048;
constexpr int32_t kBlockReq = 32;
constexpr int32_t kW1fBytes = 528;              // 512 B fragment-ordered q_hi | q_lo + s_hi, s_lo, pad
constexpr int32_t kPairU64 = 16;                // one {key, count} pair per 128-B line
constexpr int32_t kStepU64 = kMaxB * kPairU64;  // one step's pairs in a set

// K-major, no swizzle, LBO 128, SBO 256: the high word is the constant SBO field, the low word start address | LBO.
// Offsets inside the 228 KiB window never carry out of the 14-bit address field, so they are 32-bit adds.
constexpr uint32_t kDescHi = 256 >> 4;
SGL_DEVICE uint32_t wg_desc_lo(uint32_t addr) {
  return ((addr & 0x3FFFF) >> 4) | ((128 >> 4) << 16);
}

// W2 placement: kRes of the kTiles tiles SMEM-resident, the others (kStreamMask) streamed through a kRing-slot TMA ring
template <int32_t kTiles, int32_t kRes, int32_t kRing, uint32_t kStreamMask>
__global__ void __launch_bounds__(kThr, 1) markov_walk_wgmma_kernel(
    const uint8_t* __restrict__ w2_res,
    const uint8_t* __restrict__ w2_str,
    const float* __restrict__ row_scale,
    const uint8_t* __restrict__ w1f,
    const __nv_bfloat16* __restrict__ base,
    const int64_t* __restrict__ anchor,
    int64_t* __restrict__ tokens,
    __nv_bfloat16* __restrict__ corrected,
    uint64_t* __restrict__ state,
    const float* __restrict__ temps,
    int32_t nb,
    int32_t num_steps,
    int32_t state_steps,
    int32_t kb,
    int32_t ld,
    int32_t valid_rows,
    uint64_t seed) {
  constexpr int32_t kRowsCta = kTiles * kTileRows;
  constexpr int32_t kStr = kTiles - kRes;
  static_assert(kRing >= 1 && kRing <= kStr, "ring depth");
  extern __shared__ __align__(128) uint8_t smem_raw[];
  uint8_t* res_s = smem_raw;                                               // [kRes][16 KiB]
  uint8_t* ring_s = res_s + kRes * kTileBytes;                             // [kRing][16 KiB]
  float* scale_s = reinterpret_cast<float*>(ring_s + kRing * kTileBytes);  // [kRowsCta] natural row order
  __shared__ __align__(8)
      uint64_t mbar_res[kRes + 1];  // one per resident tile (first use waits for its own tile) + scales
  __shared__ __align__(8) uint64_t mbar_full[kRing];
  __shared__ int32_t slot_q[kRing];    // the FIFO index the slot holds / is being filled with
  __shared__ int32_t slot_cnt[kRing];  // monotone: 4 warps x consumers per fill
  __shared__ uint64_t wbest_s[kMaxB][kNumWG];
  __shared__ uint32_t tok_s[kMaxB];
  __shared__ float inv_t_s[kMaxB];
  const int32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int32_t wg = warp >> 2, wq = warp & 3, g = lane >> 2, tig = lane & 3;
  const int32_t row_base = blockIdx.x * kRowsCta;

  const uint64_t round = ptx::ld_relaxed(state);
  const int32_t set = static_cast<int32_t>(round & 1);
  const int32_t set_u64 = state_steps * kStepU64;
  uint64_t* pairs = state + 16 + set * set_u64;
  auto pair = [&](int32_t k, int32_t b) { return pairs + kPairU64 * (k * kMaxB + b); };
  if (blockIdx.x == 0)
    for (int32_t i = threadIdx.x; i < state_steps * kMaxB * 2; i += kThr)
      ptx::st_relaxed(state + 16 + (set ^ 1) * set_u64 + (i >> 1) * kPairU64 + (i & 1), 0);
  const uint2 seed2 = make_uint2(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32));

  // ---- work split
  const int32_t nblk = (nb + kBlockReq - 1) / kBlockReq, wpb = kNumWG / nblk;
  const int32_t blk = wg / wpb, phase = wg % wpb;
  const int32_t b = blk * kBlockReq + 8 * wq + g;  // this thread's request
  const bool live = b < nb;
  // tile limits: run 0 (tile rows 8 tig .. +8) / run 1 (32 + 8 tig .. +8) of tile t is a valid vocabulary row run iff
  // t < lim0 / lim1 (valid_rows % 8 == 0: a run is all valid or all padding).  Only the last CTA's last tiles fall
  // outside.  A dead request gets 0: the same predicate that keeps the base loads inside the valid rows also skips
  // them for dead requests.
  const int32_t vloc = valid_rows - row_base;
  const int32_t lim0 = live ? min(kTiles, max(0, vloc - 8 * tig + kTileRows - 1) / kTileRows) : 0;
  const int32_t lim1 = live ? min(kTiles, max(0, vloc - 32 - 8 * tig + kTileRows - 1) / kTileRows) : 0;
  const int32_t consumers = 4 * nblk;   // ring arrivals per fill (4 warps per warpgroup)
  const int32_t nq = kStr * num_steps;  // streamed fills per round
  auto str_idx = [&](int32_t t) { return __popc(kStreamMask & ((1u << t) - 1u)); };  // t < kTiles <= 32: shift < 32

  // ---- once per round: resident tiles + row scales by TMA, ring prefill
  const uint64_t pol_last = ptx::evict_last_policy(), pol_first = ptx::evict_first_policy();
  auto res_mbar = [&](int32_t i) { return ptx::to_shared(&mbar_res[i]); };
  auto issue_fill = [&](int32_t q) {  // thread-local: tag, arm, copy (q < nq)
    const int32_t slot = q % kRing;
    ptx::exch_release_s32(&slot_q[slot], q);
    const uint32_t mb = ptx::to_shared(&mbar_full[slot]);
    ptx::mbar_expect_tx(mb, kTileBytes);
    ptx::bulk_g2s_hint(
        ptx::to_shared(ring_s + slot * kTileBytes),
        w2_str + (static_cast<size_t>(blockIdx.x) * kStr + q % kStr) * kTileBytes,
        kTileBytes,
        mb,
        q >= nq - kStr ? pol_first : pol_last);  // last step: demote the lines for whatever runs next
  };
  if (threadIdx.x == 0) {
    for (int32_t i = 0; i <= kRes; ++i)
      ptx::mbar_init(res_mbar(i), 1);
    for (int32_t s = 0; s < kRing; ++s) {
      ptx::mbar_init(ptx::to_shared(&mbar_full[s]), 1);
      slot_cnt[s] = 0;
    }
    ptx::fence_mbarrier_init();
    // scales first (every epilogue needs them), then the ring prefill, then the resident tiles in first-use order
    ptx::mbar_expect_tx(res_mbar(kRes), kRowsCta * 4);
    ptx::bulk_g2s_hint(ptx::to_shared(scale_s), row_scale + row_base, kRowsCta * 4, res_mbar(kRes), pol_first);
    for (int32_t q = 0; q < kRing && q < nq; ++q)
      issue_fill(q);
    for (int32_t i = 0; i < kRes; ++i) {
      ptx::mbar_expect_tx(res_mbar(i), kTileBytes);
      ptx::bulk_g2s_hint(
          ptx::to_shared(res_s + i * kTileBytes),
          w2_res + (static_cast<size_t>(blockIdx.x) * kRes + i) * kTileBytes,
          kTileBytes,
          res_mbar(i),
          pol_first);
    }
  }
  for (int32_t i = threadIdx.x; i < nb; i += kThr) {
    tok_s[i] = clamp_row(anchor[i], valid_rows);
    inv_t_s[i] = inv_temperature(temps[i]);
  }
  __syncthreads();  // barriers initialized (no thread touches one before this); loads still in flight
  ptx::mbar_wait_cta(res_mbar(kRes), 0);

  const float inv_t = live ? inv_t_s[b] : 0.f;
  const float t2 = inv_t * 1.4426950408889634f;
  const uint32_t res_desc0 = wg_desc_lo(ptx::to_shared(res_s)), ring_desc0 = wg_desc_lo(ptx::to_shared(ring_s));
  uint4 nb0 = make_uint4(0, 0, 0, 0), nb1 = make_uint4(0, 0, 0, 0);  // base of this thread's next tile (bf16 x 8 x 2)

  for (int32_t k = 0; k < num_steps; ++k) {
    // u fragments of this thread's request (dead requests: zeros, so the warpgroup's MMA stays well defined)
    uint4 a[8];
    float s_lo = 0.f;
    if (live) {
      const uint8_t* src = w1f + static_cast<size_t>(tok_s[b]) * kW1fBytes;
#pragma unroll
      for (int32_t j = 0; j < 8; ++j)
        a[j] = ptx::ldg_nc(src + j * 64 + tig * 16);
      s_lo = __ldg(reinterpret_cast<const float*>(src + 516));
    } else {
#pragma unroll
      for (int32_t j = 0; j < 8; ++j)
        a[j] = make_uint4(0, 0, 0, 0);
    }
    const int32_t ob = (b * kb + k) * ld + row_base + 8 * tig;  // element offset of (b, k, run 0 of tile 0)
    if (k == 0) {  // the first tile's base (later steps load it before the previous step's exchange)
      if (phase < lim0) nb0 = ptx::ldg_ef(base + ob + phase * kTileRows, pol_first);
      if (phase < lim1) nb1 = ptx::ldg_ef(base + ob + phase * kTileRows + 32, pol_first);
    }
    float cur_v = -INFINITY;
    int32_t cur_r = -1;

#pragma unroll 2
    for (int32_t t = phase; t < kTiles; t += wpb) {
      // mask & bit, not (kStreamMask >> t) & 1: the shift form compiles to a longer bit test (2 more SASS per tile)
      const bool streamed = (kStreamMask & (1u << t)) != 0u;
      uint32_t bd;
      int32_t q = 0;
      if (streamed) {
        q = k * kStr + str_idx(t);
        const int32_t slot = q % kRing;
        while (ptx::ld_acquire_s32(&slot_q[slot]) != q) {
        }
        ptx::mbar_wait_cta(ptx::to_shared(&mbar_full[slot]), static_cast<uint32_t>((q / kRing) & 1));
        bd = ring_desc0 + slot * (kTileBytes >> 4);
      } else {
        const int32_t ri = t - str_idx(t);
        if (k == 0) ptx::mbar_wait_cta(res_mbar(ri), 0);  // first use of a resident tile: its own TMA (overlaps step 0)
        bd = res_desc0 + ri * (kTileBytes >> 4);
      }
      int32_t d[32];
      ptx::wg_fence();
#pragma unroll
      for (int32_t j = 0; j < 8; ++j)
        ptx::wgmma_rs(d, a[j], bd + j * (kChunkBytes >> 4), kDescHi, j);
      ptx::wg_commit();
      // this tile's base was loaded one tile ago; the next tile's goes out under this MMA.  Runs past valid_rows (and
      // dead requests) load nothing: their registers keep a stale tile, masked below.
      const int32_t o = ob + t * kTileRows;
      const uint4 bb0 = nb0, bb1 = nb1;
      if (t + wpb < lim0) nb0 = ptx::ldg_ef(base + o + wpb * kTileRows, pol_first);
      if (t + wpb < lim1) nb1 = ptx::ldg_ef(base + o + wpb * kTileRows + 32, pol_first);
      ptx::wg_wait0();
      ptx::wg_fence_operand(d);     // keep every read of d below the wait (d is written async)
      if (streamed && lane == 0) {  // this warp is done reading the slot
        const int32_t slot = q % kRing;
        const int32_t done = ptx::add_acq_rel_s32(&slot_cnt[slot], 1) + 1;
        if (done == (q / kRing + 1) * consumers && q + kRing < nq) issue_fill(q + kRing);
      }
      if (!live) continue;
      // logits: bias = s_lo float(256 d_hi + d_lo) (exact int32), logit = fma(row scale, bias, base)
      const float4 sa = *reinterpret_cast<const float4*>(scale_s + t * kTileRows + 8 * tig);
      const float4 sb = *reinterpret_cast<const float4*>(scale_s + t * kTileRows + 8 * tig + 4);
      const float4 sc = *reinterpret_cast<const float4*>(scale_s + t * kTileRows + 32 + 8 * tig);
      const float4 sd = *reinterpret_cast<const float4*>(scale_s + t * kTileRows + 32 + 8 * tig + 4);
      const float sr[16] = {
          sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w, sc.x, sc.y, sc.z, sc.w, sd.x, sd.y, sd.z, sd.w};
      const uint32_t bw[8] = {bb0.x, bb0.y, bb0.z, bb0.w, bb1.x, bb1.y, bb1.z, bb1.w};
      float lg[16];  // [run][q]: run 0 = i < 4, element q = 2 (i % 4) + e
#pragma unroll
      for (int32_t i = 0; i < 8; ++i)
#pragma unroll
        for (int32_t e = 0; e < 2; ++e) {
          const int32_t idx = 8 * (i >> 2) + 2 * (i & 3) + e;
          const float bias = s_lo * static_cast<float>(d[4 * i + e] * 256 + d[4 * i + 2 + e]);
          const uint32_t w = bw[idx >> 1];
          lg[idx] = fmaf(sr[idx], bias, __uint_as_float(e ? (w & 0xFFFF0000u) : (w << 16)));
        }
      // the walk's logits are the bf16 roundings (RNE) of these -- the values sglang's verifier reads back from
      // corrected: pk[4 run + qq] = {lg[8 run + 2 qq] (low half), lg[8 run + 2 qq + 1] (high half)}, i.e. corrected's
      // 16 B of run `run` in memory order.  Greedy works on pk directly; the sampler unpacks (bf16 -> fp32 is a shift).
      uint32_t pk[8];
#pragma unroll
      for (int32_t p2 = 0; p2 < 8; ++p2) {
        const __nv_bfloat162 h = __floats2bfloat162_rn(lg[2 * p2], lg[2 * p2 + 1]);
        pk[p2] = *reinterpret_cast<const uint32_t*>(&h);
      }
      const int32_t r = row_base + t * kTileRows + 8 * tig;  // run 0 rows r .. r + 8, run 1 rows r + 32 .. r + 40
      if (row_base + (t + 1) * kTileRows > valid_rows) {  // the last CTA's padding runs (uniform per tile): bf16 -inf
#pragma unroll
        for (int32_t run = 0; run < 2; ++run)
          if (r + 32 * run >= valid_rows) {  // valid_rows % 8 == 0: whole runs
#pragma unroll
            for (int32_t qq = 0; qq < 4; ++qq)
              pk[4 * run + qq] = 0xFF80FF80u;
          }
      }
      if (inv_t > 0.f) {
        // corrected (sampling requests only: the verifier's q), valid runs only
        if (corrected != nullptr) {
          if (t < lim0) ptx::stg_ef(corrected + o, make_uint4(pk[0], pk[1], pk[2], pk[3]), pol_first);
          if (t < lim1) ptx::stg_ef(corrected + o + 32, make_uint4(pk[4], pk[5], pk[6], pk[7]), pol_first);
        }
#pragma unroll
        for (int32_t p2 = 0; p2 < 8; ++p2) {
          lg[2 * p2] = __uint_as_float(pk[p2] << 16);
          lg[2 * p2 + 1] = __uint_as_float(pk[p2] & 0xFFFF0000u);
        }
        // exact two-level sample per 8-row item: row by inverse CDF inside the item, item by
        // Gumbel-max on log2 of its mass.  Uniforms: philox(item index of run 0 = global row / 8, step | b << 8,
        // round): x, y for run 0, z, w for run 1 (run-1 items 8 T + 4 + tig never coincide with a run-0 counter)
        const uint4 xr = philox(
            make_uint4(
                static_cast<uint32_t>(r >> 3),
                static_cast<uint32_t>(k) | (static_cast<uint32_t>(b) << 8),
                static_cast<uint32_t>(round),
                static_cast<uint32_t>(round >> 32)),
            seed2);
#pragma unroll
        for (int32_t run = 0; run < 2; ++run) {
          const float* l8 = lg + 8 * run;
          const float m =
              __uint_as_float(ptx::bf16x8_max(pk[4 * run], pk[4 * run + 1], pk[4 * run + 2], pk[4 * run + 3]) << 16);
          const float mz = m * t2;
          float cs[8], c = 0.f;
#pragma unroll
          for (int32_t qq = 0; qq < 8; ++qq) {
            c += ptx::ex2_ftz(fmaf(l8[qq], t2, -mz));
            cs[qq] = c;
          }
          const uint32_t xu = run == 0 ? xr.x : xr.z, xg = run == 0 ? xr.y : xr.w;
          const float u = uniform23(xu);
          const float tt = u * c;  // < c (uniform23)
          int32_t qi = 7;
#pragma unroll
          for (int32_t qq = 6; qq >= 0; --qq)
            qi = tt < cs[qq] ? qq : qi;
          const float g2 = gumbel2_fast(xg);
          const float key_v = mz + ptx::lg2_ftz(c) + g2;
          if (key_v > cur_v) {
            cur_v = key_v;
            cur_r = r + 32 * run + qi;
          }
        }
      } else {  // greedy on the packed bf16 values
#pragma unroll
        for (int32_t run = 0; run < 2; ++run) {
          const uint32_t* p8 = pk + 4 * run;
          const uint32_t m2 = ptx::bf16x8_max(p8[0], p8[1], p8[2], p8[3]);
          const int32_t qi = ptx::bf16x8_first_eq(p8[0], p8[1], p8[2], p8[3], m2);  // ties -> smallest row
          const float m = __uint_as_float(m2 << 16);
          if (m > cur_v) {
            cur_v = m;
            cur_r = r + 32 * run + qi;
          }
        }
      }
    }
    // the 4 tig lanes of a request -> this warpgroup's best for it
    {
      uint64_t key = cur_r >= 0 ? pack_key(cur_v, static_cast<uint32_t>(cur_r)) : 0;
      key = max(key, __shfl_xor_sync(0xffffffffu, key, 1));
      key = max(key, __shfl_xor_sync(0xffffffffu, key, 2));
      if (tig == 0 && live) wbest_s[b][phase] = key;
    }
    // next step's base does not depend on the tokens: its first tile into registers, the next ones into L2, all
    // under the exchange
    if (k + 1 < num_steps) {
      const int32_t o1 = ob + ld + phase * kTileRows;
      if (phase < lim0) nb0 = ptx::ldg_ef(base + o1, pol_first);
      if (phase < lim1) nb1 = ptx::ldg_ef(base + o1 + 32, pol_first);
    }
    __syncthreads();
    if (warp == 0) {
      uint64_t key[2] = {0, 0};
#pragma unroll
      for (int32_t qd = 0; qd < 2; ++qd) {
        const int32_t bq = lane + 32 * qd;
        if (bq < nb) {
          uint64_t bst = 0;
          for (int32_t p = 0; p < wpb; ++p)
            bst = max(bst, wbest_s[bq][p]);
          ptx::red_relaxed_max_u64(pair(k, bq), bst);
          ptx::red_add_release(pair(k, bq) + 1, 1);
        }
      }
      // both of a lane's requests polled in the same round trip; a request the lane does not hold starts "arrived"
      const uint64_t n = gridDim.x;
      uint64_t v0[2] = {0, lane < nb ? 0 : n}, v1[2] = {0, lane + 32 < nb ? 0 : n};
      do {
        if (v0[1] < n) ptx::ld_relaxed_v2(pair(k, lane), v0);
        if (v1[1] < n) ptx::ld_relaxed_v2(pair(k, lane + 32), v1);
      } while (v0[1] < n || v1[1] < n);
      key[0] = v0[0];
      key[1] = v1[0];
#pragma unroll
      for (int32_t qd = 0; qd < 2; ++qd) {
        const int32_t bq = lane + 32 * qd;
        if (bq < nb) {
          // key 0 = no row was a candidate anywhere: every logit of the request NaN / -inf (greedy: NaN never wins
          // `>`, -inf never beats the -inf start; sampling: an item with a NaN or all -inf has a NaN key) -- e.g. a
          // padded CUDA-graph row.  key_row(0) = 0xFFFFFFFF would make the next step's W1 gather an illegal address:
          // emit row 0, torch.argmax's answer for such a row.
          const uint32_t tok = key[qd] != 0 ? key_row(key[qd]) : 0u;
          tok_s[bq] = tok;
          if (blockIdx.x == 0) tokens[static_cast<size_t>(bq) * num_steps + k] = tok;
          // The next gather's u row (528 B, 5 lines) into L2 before the barrier releases it, from CTA 0 only: every
          // CTA gathers the same rows, and 132 copies of the prefetch made bs 64 2-3% slower on H100.
          if (blockIdx.x == 0 && k + 1 < num_steps) {
            const uint8_t* src = w1f + static_cast<size_t>(tok) * kW1fBytes;
#pragma unroll
            for (int32_t l = 0; l < 5; ++l)
              ptx::prefetch_l2_evict_last(src + 128 * l);
          }
        }
      }
    }
    __syncthreads();
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) ptx::st_relaxed(state, round + 1);
}

/**
 * \brief One round of the wgmma markov walk (bs 1..64) in one cooperative launch.
 *
 * \tparam kTiles      64-row W2 tiles per CTA: rows per CTA = 64 kTiles, grid = rows_pad / (64 kTiles).
 * \tparam kRes        Tiles per CTA kept SMEM-resident (TMA once per launch).
 * \tparam kRing       TMA ring slots for the kTiles - kRes streamed tiles.
 * \tparam kStreamMask Bit t = tile t streams; kTiles - kRes bits below bit kTiles.
 * \param w2_res     int8 [grid, kRes, 16384], each CTA's resident 64-row W2 tiles in tile order, canonical K-major B
 *                   layout [k32 chunk][8-col group][k16 core][col][16 B], columns permuted by rho (header).
 * \param w2_str     int8 [grid, kTiles - kRes, 16384], the streamed tiles, same layout.
 * \param row_scale  fp32 [rows_pad], natural row order, rows_pad = grid x 64 kTiles.
 * \param w1f        uint8 [V, 528], per row [chunk j][tig][q_hi 4 | q_lo 4 | q_hi(+16) 4 | q_lo(+16) 4] then s_hi,
 *                   s_lo (fp32) with s_lo = s_hi / 256.
 * \param base       bf16 [B, kb, ld], kb >= num_steps, ld >= valid_rows, ld % 8 == 0 (sglang: [bs, K, V]); rows
 *                   >= valid_rows are never read.
 * \param anchor     int64 [B], clamped into [0, valid_rows).
 * \param tokens     int64 [B * num_steps], row-major b * num_steps + k.
 * \param corrected  None, or bf16 shaped like base: the bf16 logits the walk used, rows k < num_steps, vocabulary rows
 *                   < valid_rows, written for requests with temps > 0 only -- zero-initialise it once.
 * \param state      int64 [16 + 2 S kStepU64] for S >= num_steps steps: round | pad to 128 B | two sets (round
 *                   parity) of 128-B {key, count} pairs per (step, request); 128-B aligned, zeroed once and
 *                   never reset between calls; serves any num_steps <= S.
 * \param temps      fp32 [B]: <= 0 or NaN: greedy for that request; T > 0 is clamped to [1e-5, 1e4].
 * \param valid_rows The vocabulary size: % 8 == 0, <= min(ld, rows_pad, V).
 * \param seed       Philox key; the round counter in state advances every launch.
 */
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
  constexpr int32_t kRowsCta = kTiles * kTileRows;
  auto grid = SymbolicSize{"grid"};
  auto w1_rows = SymbolicSize{"w1_rows"};
  auto nb = SymbolicSize{"batch_size"};
  auto kb = SymbolicSize{"base_steps"};
  auto ld = SymbolicSize{"base_row_stride"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  CHECK_HOST(num_steps >= 1 && num_steps <= kMaxSteps) << "num_steps must be in [1, " << kMaxSteps << "]";
  TensorMatcher({grid, kRes, kTileBytes}).with_dtype<int8_t>().with_device(device).ensure_alignment(16).verify(w2_res);
  TensorMatcher({grid, kTiles - kRes, kTileBytes})
      .with_dtype<int8_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(w2_str);
  const int64_t rows_pad = grid.unwrap() * kRowsCta;
  TensorMatcher({rows_pad}).with_dtype<float>().with_device(device).ensure_alignment(16).verify(row_scale);
  TensorMatcher({w1_rows, kW1fBytes}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w1f);
  TensorMatcher({nb, kb, ld}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(base);
  if (corrected.has_value()) {
    TensorMatcher({nb, kb, ld}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(corrected.value());
  }
  TensorMatcher({nb}).with_dtype<int64_t>().with_device(device).ensure_alignment(8).verify(anchor);
  TensorMatcher({nb}).with_dtype<float>().with_device(device).ensure_alignment(4).verify(temps);
  TensorMatcher({nb.unwrap() * num_steps}).with_dtype<int64_t>().with_device(device).ensure_alignment(8).verify(tokens);
  auto state_words = SymbolicSize{"state_words"};
  TensorMatcher({state_words}).with_dtype<int64_t>().with_device(device).ensure_alignment(128).verify(state);
  const int64_t state_steps = (state_words.unwrap() - 16) / (2 * kStepU64);
  CHECK_HOST(
      state_words.unwrap() == 16 + 2 * kStepU64 * state_steps && state_steps >= num_steps && state_steps <= kMaxSteps)
      << "state must be int64 [16 + " << 2 * kStepU64 << " S] with num_steps <= S <= " << kMaxSteps << "; got "
      << state_words.unwrap() << " words for num_steps " << num_steps;
  const int64_t nb_v = nb.unwrap(), kb_v = kb.unwrap(), ld_v = ld.unwrap();
  CHECK_HOST(nb_v >= 1 && nb_v <= kMaxB) << "batch must be in [1, " << kMaxB << "], got " << nb_v;
  CHECK_HOST(kb_v >= num_steps) << "base has " << kb_v << " steps < num_steps " << num_steps;
  CHECK_HOST(
      ld_v % 8 == 0 && valid_rows % 8 == 0 && valid_rows >= 8 && valid_rows <= ld_v && valid_rows <= rows_pad &&
      valid_rows <= w1_rows.unwrap())
      << "need ld % 8 == 0, valid_rows % 8 == 0, 8 <= valid_rows <= min(ld, rows_pad, W1 rows); got ld " << ld_v
      << ", valid_rows " << valid_rows << ", rows_pad " << rows_pad << ", W1 rows " << w1_rows.unwrap();
  // 32-bit element offsets: every lane of the 32-request blocks forms (b kb + k) ld + row in [0, rows_pad) (dead lanes
  // b in [nb, 32 nblk) too, never dereferenced)
  const int64_t nb_lanes = kBlockReq * ((nb_v + kBlockReq - 1) / kBlockReq);
  CHECK_HOST(nb_lanes * kb_v * ld_v + rows_pad < (int64_t{1} << 31))
      << "32-bit element offsets: " << nb_lanes << " request lanes x kb " << kb_v << " x ld " << ld_v << " + rows_pad "
      << rows_pad;
  constexpr std::size_t kSmem = (kRes + kRing) * kTileBytes + kRowsCta * 4;
  launch_cooperative(
      markov_walk_wgmma_kernel<kTiles, kRes, kRing, kStreamMask>,
      device.unwrap(),
      static_cast<uint32_t>(grid.unwrap()),
      kThr,
      kSmem,
      kSmem,
      static_cast<const uint8_t*>(w2_res.data_ptr()),
      static_cast<const uint8_t*>(w2_str.data_ptr()),
      static_cast<const float*>(row_scale.data_ptr()),
      static_cast<const uint8_t*>(w1f.data_ptr()),
      static_cast<const __nv_bfloat16*>(base.data_ptr()),
      static_cast<const int64_t*>(anchor.data_ptr()),
      static_cast<int64_t*>(tokens.data_ptr()),
      corrected.has_value() ? static_cast<__nv_bfloat16*>(corrected.value().data_ptr()) : nullptr,
      static_cast<uint64_t*>(state.data_ptr()),
      static_cast<const float*>(temps.data_ptr()),
      static_cast<int32_t>(nb_v),
      static_cast<int32_t>(num_steps),
      static_cast<int32_t>(state_steps),
      static_cast<int32_t>(kb_v),
      static_cast<int32_t>(ld_v),
      static_cast<int32_t>(valid_rows),
      static_cast<uint64_t>(seed));
}

#undef DSPARK_MW_WG_D32_OUT
#undef DSPARK_MW_WG_D32

}  // namespace sglang::dspark_markov_walk::wgmma
