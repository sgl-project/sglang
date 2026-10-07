// SPDX-License-Identifier: Apache-2.0
// Batched int8 DSpark markov walk (serving: bs 2..4; the launcher takes 1..8): B independent walks share one on-chip
// int8 W2.  Same W2 / W1q formats, tiling and load path as dspark_markov_walk_single.cuh.
//
//   for every request b:  prev_b = anchor_b;  for k < K:  logit = bf16(base[b, k] + W2 W1[prev_b]);
//                         tok[b, k] = argmax(logit)  or  argmax(logit / T_b + Gumbel);  prev_b = tok[b, k]
//
// One mma pass serves 4 requests: B column n carries request 4p + n/2 (q_hi for even n, q_lo for odd), so lane tig
// holds request 4p + tig's (hi, lo) int32 dots of its rows -- up to 4 requests cost the tensor work of one.  Per step:
// per pass p (requests 4p .. 4p + 3; the next pass's u rows already in flight): GEMV over the CTA's tiles, bias into a
// double-buffered SMEM slab, barrier, then the epilogue of those requests (base + bias, bf16 rounding, corrected for
// sampling requests, Gumbel noise computed in place, per-request warp argmax: whole warps per request, no shared
// atomics); the base rows of the next pass are prefetched (cp.async) behind it.  After the last pass one warp runs the
// cross-CTA exchange of the common header for every request (lane b: request b's {key, count} pair).
// SMEM does not grow with B beyond the u rows (528 B per request).
//
// Logits: the fp32 op sequence before the bf16 rounding is pinned (__fmul_rn / __fmaf_rn / __fadd_rn), so an exact
// reference replays it bit for bit; argmax and Gumbel-max read the rounded values (ties -> smallest row) and
// corrected receives exactly those bits.
// State: int64 [16 + 2 S kStepU64] = round | pad to 128 B | two sets of [S >= num_steps steps][64 requests] pairs,
// one 128-B line each
// (atomics to one line serialize in its L2 slice).  Round r uses set r & 1; CTA 0 clears the other set at the start
// and stores round + 1 after the last step.  Nothing is reset between launches.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include "dspark_markov_walk_common.cuh"
#include <cstdint>

namespace sglang::dspark_markov_walk::small_batch {

constexpr int kMaxB = 64;                   // state / SMEM array sizing
constexpr int kPairMaxB = 8;                // launcher limit (two mma passes)
constexpr int kGrp = 4;                     // requests per mma pass (column pairs)
constexpr int kPairU64 = 16;                // one {key, count} pair per 128-B line
constexpr int kStepU64 = kMaxB * kPairU64;  // one step's pairs in a set
// W2 slice per CTA: TS SMEM tiles + NG register tiles per warp, 72 16-row tiles = 1152 rows
constexpr int NG = 6, TS = 3;
constexpr int kTilesSmem = TS * kWarps, kTilesPerCta = kTilesSmem + NG * kWarps, kRowsCta = kTilesPerCta * 16;

// The fp32 bias of rows l0 + g and + 8 with its rounding sequence pinned: bias = sr (u_hi c_hi + u_lo c_lo) as
// t = u_lo c_lo (RN), fma(u_hi, c_hi, t), times sr (RN) -- a last-bit difference in fp32 can cross a bf16 rounding
// boundary, so the exact reference must replay this order.
SGL_DEVICE void
store_bias_rn(const int (&c)[4], float sr_lo, float sr_hi, float u_hi, float u_lo, float* bias_s, int l0, int g) {
  bias_s[l0 + g] =
      __fmul_rn(sr_lo, __fmaf_rn(u_hi, static_cast<float>(c[0]), __fmul_rn(u_lo, static_cast<float>(c[1]))));
  bias_s[l0 + g + 8] =
      __fmul_rn(sr_hi, __fmaf_rn(u_hi, static_cast<float>(c[2]), __fmul_rn(u_lo, static_cast<float>(c[3]))));
}

__global__ void __launch_bounds__(kThreads, 1) markov_walk_small_batch_kernel(
    const uint4* __restrict__ frag,
    const float* __restrict__ row_scale,
    const uint32_t* __restrict__ w1q,
    const __nv_bfloat16* __restrict__ base,
    const int64_t* __restrict__ anchor,
    int64_t* __restrict__ tokens,
    __nv_bfloat16* __restrict__ corrected,
    u64* __restrict__ state,
    const float* __restrict__ temps,
    int nb,
    int num_steps,
    int state_steps,
    int kb,
    int ld,
    int valid_rows,
    u64 seed) {
  constexpr int tiles_smem = kTilesSmem;
  constexpr int tiles_per_cta = kTilesPerCta;
  constexpr int rows_cta = kRowsCta;
  constexpr int kChunks = rows_cta / 8;  // epilogue items per request (8 rows each)
  extern __shared__ __align__(16) unsigned char smem_raw[];
  unsigned char* p = smem_raw;
  __nv_bfloat16* base_s = reinterpret_cast<__nv_bfloat16*>(p);
  p += 2 * kGrp * rows_cta * 2;  // [2][grp][rows]
  float* bias_s = reinterpret_cast<float*>(p);
  p += 2 * kGrp * rows_cta * 4;  // [2][grp][rows]
  float* scale_s = reinterpret_cast<float*>(p);
  p += 64 * tiles_per_cta;
  uint4* w_s = reinterpret_cast<uint4*>(p);
  p += tiles_smem * kTileVecs * 16;
  uint32_t* u_s = reinterpret_cast<uint32_t*>(p);  // (nb + 1) W1q rows, row nb = zeros
  __shared__ u64 wbest_s[kMaxB][kWarps];           // per request: its epilogue warps' best keys
  __shared__ unsigned tok_s[kMaxB];
  __shared__ float inv_t_s[kMaxB];
  __shared__ __align__(8) u64 mbar_load;
  __shared__ __align__(8) u64 mbar_u[kMaxB / kGrp];  // u rows of pass p landed (phase = step parity)
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, tig = lane & 3;
  const int tile0 = blockIdx.x * tiles_per_cta;
  const int row_base = tile0 * 16;
  const uint4* g_frag = frag + static_cast<size_t>(tile0) * kTileVecs;

  const u64 round = ptx::ld_relaxed(state);
  const int set = static_cast<int>(round & 1);
  const int set_u64 = state_steps * kStepU64;
  u64* pairs = state + 16 + set * set_u64;  // pair (k, b) at pairs + kPairU64 * (k * kMaxB + b)
  auto pair = [&](int k, int b) { return pairs + kPairU64 * (k * kMaxB + b); };
  if (blockIdx.x == 0)  // clear the set round r+1 uses (it may differ in B, and in K <= state_steps)
    for (int i = threadIdx.x; i < state_steps * kMaxB * 2; i += kThreads)
      ptx::st_relaxed(state + 16 + (set ^ 1) * set_u64 + (i >> 1) * kPairU64 + (i & 1), 0);
  const uint2 seed2 = make_uint2(static_cast<unsigned>(seed), static_cast<unsigned>(seed >> 32));
  const int npass = (nb + kGrp - 1) / kGrp;
  const int n_items = num_steps * npass;  // (step, pass) in walk order

  // base rows of item it = (k, pass) -> base_s[it & 1]: the pass's requests, this CTA's rows.  Chunks at or past
  // valid_rows (the last CTA's tail; valid_rows % 8 == 0) copy the last valid chunk instead -- in bounds whatever the
  // row stride; the epilogue skips them (r < valid_rows)
  const int last8 = valid_rows - 8;
  auto prefetch_base = [&](int it) {
    const int k = it / npass, r0 = (it % npass) * kGrp, nr = min(kGrp, nb - r0);
    __nv_bfloat16* dst = base_s + (it & 1) * kGrp * rows_cta;
    for (int i = threadIdx.x; i < nr * kChunks; i += kThreads) {
      const int rr = i / kChunks, c = i % kChunks;
      ptx::cp_async16(
          dst + rr * rows_cta + c * 8,
          base + (static_cast<size_t>(r0 + rr) * kb + k) * ld + min(row_base + c * 8, last8));
    }
    ptx::cp_async_commit();
  };

  // ---- once per round: base of item 0, W2 slice (+ row scales) on-chip
  prefetch_base(0);
  const u64 pol = ptx::evict_first_policy();
  const unsigned a_mbar = ptx::to_shared(&mbar_load);
  if (threadIdx.x == 0) {
    ptx::mbar_init(a_mbar, 1);
    for (int ps = 0; ps < npass; ++ps)
      ptx::mbar_init(ptx::to_shared(&mbar_u[ps]), kThreads);  // every thread arrives
    ptx::fence_mbarrier_init();
    constexpr unsigned kW2Bytes = tiles_smem * kTileVecs * 16, kScaleBytes = tiles_per_cta * 64, kChunk = 16384;
    ptx::mbar_expect_tx(a_mbar, kW2Bytes + kScaleBytes);
    for (unsigned off = 0; off < kW2Bytes; off += kChunk)
      ptx::bulk_g2s_hint(
          ptx::to_shared(reinterpret_cast<unsigned char*>(w_s) + off),
          reinterpret_cast<const unsigned char*>(g_frag) + off,
          off + kChunk <= kW2Bytes ? kChunk : kW2Bytes - off,
          a_mbar,
          pol);
    ptx::bulk_g2s_hint(ptx::to_shared(scale_s), row_scale + row_base, kScaleBytes, a_mbar, pol);
  }
  for (int b = threadIdx.x; b < nb; b += kThreads) {
    tok_s[b] = clamp_row(anchor[b], valid_rows);
    inv_t_s[b] = inv_temperature(temps[b]);
  }
  for (int i = threadIdx.x; i < kURowWords; i += kThreads)
    u_s[nb * kURowWords + i] = 0;  // padding request
  uint4 a_reg[NG][8];
#pragma unroll
  for (int gi = 0; gi < NG; ++gi) {
    const int t = tiles_smem + warp * NG + gi;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      a_reg[gi][j] = ptx::ldcg_hint(g_frag + (t * 8 + j) * 32 + lane, pol);
  }
  // only thread 0 (which initialized it) waits on the load barrier: another thread could reach a try_wait before the
  // init (nothing orders them); the barrier below then publishes the landed tiles to everyone
  if (threadIdx.x == 0) ptx::mbar_wait_cta(a_mbar, 0);
  __syncthreads();

  // GEMV over this warp's NG + TS tiles; ub = this lane's B column source (a W1q row: q_hi | q_lo words),
  // column n takes q_hi for even n, q_lo for odd n
  auto gemv = [&](int (&acc)[NG + TS][4], const uint32_t* ub) {
#pragma unroll
    for (int i = 0; i < NG + TS; ++i)
      acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint4 as[TS];
#pragma unroll
      for (int ts = 0; ts < TS; ++ts)
        as[ts] = w_s[((warp + kWarps * ts) * 8 + j) * 32 + lane];
      const uint32_t b0 = ub[(g & 1) * 64 + j * 8 + tig];
      const uint32_t b1 = ub[(g & 1) * 64 + j * 8 + 4 + tig];
#pragma unroll
      for (int gi = 0; gi < NG; ++gi)
        ptx::mma_s8(acc[gi], a_reg[gi][j], b0, b1);
#pragma unroll
      for (int ts = 0; ts < TS; ++ts)
        ptx::mma_s8(acc[NG + ts], as[ts], b0, b1);
    }
  };

  // u rows of pass ps (requests 4 ps ..): cp.async spread over the threads, completion tracked on the pass's mbarrier
  // (cp.async.mbarrier.arrive: every thread arrives once per pass and step), fetched one pass ahead
  auto fetch_u = [&](int ps) {
    constexpr int kRowVecs = kURowBytes / 16;
    const int r0 = ps * kGrp, nr = min(kGrp, nb - r0);
    // %tid.x re-read here: no threadIdx-derived index kept live across the step (255 registers, 0 spill)
    const unsigned tq = ptx::tid_x();
    if (tq < nr * kRowVecs) {
      const int b = r0 + tq / kRowVecs, v = tq % kRowVecs;
      ptx::cp_async16(
          reinterpret_cast<uint4*>(u_s) + b * kRowVecs + v,
          reinterpret_cast<const uint4*>(w1q) + static_cast<size_t>(tok_s[b]) * kRowVecs + v);
    }
    ptx::cp_async_mbar_arrive_noinc(ptx::to_shared(&mbar_u[ps]));
  };
  constexpr int kXchgWarp = kWarps - 1;
  const u64 n = gridDim.x;
  for (int k = 0; k < num_steps; ++k) {
    // u rows of all requests (prefetched into L2 by the CTAs that held a local winner)
    fetch_u(0);  // pass 0's u rows; pass p fetches pass p+1's
    for (int ps = 0; ps < npass; ++ps) {
      const int it = k * npass + ps, r0 = ps * kGrp, nr = min(kGrp, nb - r0);
      if (ps + 1 < npass) fetch_u(ps + 1);
      ptx::mbar_wait_cta(ptx::to_shared(&mbar_u[ps]), k & 1);
      {
        const int rq = r0 + (g >> 1);
        int acc[NG + TS][4];
        gemv(acc, u_s + (rq < nb ? rq : nb) * kURowWords);
        if (tig < nr) {  // lane tig: request r0 + tig, rows g / g+8 of every tile of this warp
          const uint32_t* ur = u_s + (r0 + tig) * kURowWords;
          const float u_hi = __uint_as_float(ur[128]), u_lo = __uint_as_float(ur[129]);
          float* dst = bias_s + ((it & 1) * kGrp + tig) * rows_cta;
#pragma unroll
          for (int i = 0; i < NG + TS; ++i) {
            const int t = i < NG ? tiles_smem + warp * NG + i : warp + kWarps * (i - NG);
            store_bias_rn(acc[i], scale_s[t * 16 + g], scale_s[t * 16 + g + 8], u_hi, u_lo, dst, t * 16, g);
          }
        }
      }
      ptx::cp_async_wait<0>();  // this item's base (the only group in flight)
      __syncthreads();          // bias + base of item it visible; every thread is past item it-1's epilogue
      if (it + 1 < n_items) prefetch_base(it + 1);  // into the buffer item it-1 used
      // epilogue: wpr = kWarps / (nr rounded up to a power of 2) whole warps per request (8 for a lone request:
      // sampling's noise is the bulk of the work), lane's 8-row chunks (warp % wpr) * 32 + lane + 32 wpr j --
      // a plain warp max per request, no shared atomics
      const int wpr = nr == 1 ? kWarps : nr == 2 ? kWarps / 2 : kWarps / 4;
      const int rr = warp / wpr;
      if (rr < nr) {
        const int b = r0 + rr;
        const __nv_bfloat16* bsrc = base_s + ((it & 1) * kGrp + rr) * rows_cta;
        const float* bias = bias_s + ((it & 1) * kGrp + rr) * rows_cta;
        const float inv_t = inv_t_s[b];
        u64 best = 0;
#pragma unroll 1
        for (int c = (warp % wpr) * 32 + lane; c < kChunks; c += wpr * 32) {
          const int l = c * 8;
          const uint4 bb = *reinterpret_cast<const uint4*>(bsrc + l);
          const float4 b0 = *reinterpret_cast<const float4*>(bias + l);
          const float4 b1 = *reinterpret_cast<const float4*>(bias + l + 4);
          const __nv_bfloat162* bp = reinterpret_cast<const __nv_bfloat162*>(&bb);
          float lg[8];
          {
            const float2 x0 = __bfloat1622float2(bp[0]), x1 = __bfloat1622float2(bp[1]);
            const float2 x2 = __bfloat1622float2(bp[2]), x3 = __bfloat1622float2(bp[3]);
            lg[0] = __fadd_rn(x0.x, b0.x);
            lg[1] = __fadd_rn(x0.y, b0.y);
            lg[2] = __fadd_rn(x1.x, b0.z);
            lg[3] = __fadd_rn(x1.y, b0.w);
            lg[4] = __fadd_rn(x2.x, b1.x);
            lg[5] = __fadd_rn(x2.y, b1.y);
            lg[6] = __fadd_rn(x3.x, b1.z);
            lg[7] = __fadd_rn(x3.y, b1.w);
          }
          // the walk's logits are the bf16 roundings (RNE) of these: o = the 8 rounded values in memory
          // order (corrected's 16 B).  Greedy takes max / first argmax on the packed values; the sampler unpacks them
          // (bf16 -> fp32 is a shift).  valid_rows % 8 == 0: a chunk is all valid or all padding (skipped below).
          uint4 o;
          {
            unsigned* op = reinterpret_cast<unsigned*>(&o);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const __nv_bfloat162 hv = __floats2bfloat162_rn(lg[2 * i], lg[2 * i + 1]);
              op[i] = *reinterpret_cast<const unsigned*>(&hv);
            }
          }
          const size_t out = (static_cast<size_t>(b) * kb + k) * ld + row_base + l;
          const int r = row_base + l;
          float m;
          int idx;
          if (inv_t > 0.f) {  // Gumbel-max; Philox counter (row / 4, step | request << 8, round), small_batch's own key
            // corrected: sampling requests only (the verifier's q), valid rows only
            if (corrected && r < valid_rows) *reinterpret_cast<uint4*>(corrected + out) = o;
            lg[0] = __uint_as_float(o.x << 16);
            lg[1] = __uint_as_float(o.x & 0xFFFF0000u);
            lg[2] = __uint_as_float(o.y << 16);
            lg[3] = __uint_as_float(o.y & 0xFFFF0000u);
            lg[4] = __uint_as_float(o.z << 16);
            lg[5] = __uint_as_float(o.z & 0xFFFF0000u);
            lg[6] = __uint_as_float(o.w << 16);
            lg[7] = __uint_as_float(o.w & 0xFFFF0000u);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const uint4 x = philox(
                  make_uint4(
                      static_cast<unsigned>((r + 4 * h) >> 2),
                      static_cast<unsigned>(k) | (static_cast<unsigned>(b) << 8),
                      static_cast<unsigned>(round),
                      static_cast<unsigned>(round >> 32)),
                  seed2);
              const float gv[4] = {gumbel_fast(x.x), gumbel_fast(x.y), gumbel_fast(x.z), gumbel_fast(x.w)};
#pragma unroll
              for (int i = 0; i < 4; ++i)
                lg[4 * h + i] = __fmaf_rn(lg[4 * h + i], inv_t, gv[i]);
            }
            m = fmaxf(fmaxf(fmaxf(lg[0], lg[1]), fmaxf(lg[2], lg[3])), fmaxf(fmaxf(lg[4], lg[5]), fmaxf(lg[6], lg[7])));
            idx = 7;
#pragma unroll
            for (int i = 6; i >= 0; --i)
              idx = lg[i] == m ? i : idx;  // ties -> smallest row, as torch.argmax
          } else {
            const unsigned m2 = ptx::bf16x8_max(o.x, o.y, o.z, o.w);
            idx = ptx::bf16x8_first_eq(o.x, o.y, o.z, o.w, m2);
            m = __uint_as_float(m2 << 16);
          }
          if (r < valid_rows) best = max(best, pack_key(m, r + idx));
        }
        const unsigned hi = static_cast<unsigned>(best >> 32);
        const unsigned mx = __reduce_max_sync(0xffffffffu, hi);
        const unsigned lo = __reduce_max_sync(0xffffffffu, hi == mx ? static_cast<unsigned>(best) : 0u);
        if (lane == 0) wbest_s[b][warp % wpr] = (static_cast<u64>(mx) << 32) | lo;
      }
    }
    __syncthreads();  // every request's warp bests are in wbest_s
    if (warp == kXchgWarp) {
      // post: W1q[local winner] -> L2 (the global winner's row is then an L2 hit for every CTA), atomicMax of the
      // key, then the arrival count with release semantics (orders this lane's max before its arrival)
      for (int b = lane; b < nb; b += 32) {
        const int r0 = b / kGrp * kGrp, nr = min(kGrp, nb - r0);  // b's pass: how many warps wrote a best
        const int wpr = nr == 1 ? kWarps : nr == 2 ? kWarps / 2 : kWarps / 4;
        u64 best = wbest_s[b][0];
        for (int w = 1; w < wpr; ++w)
          best = max(best, wbest_s[b][w]);
        const char* row = reinterpret_cast<const char*>(w1q + static_cast<size_t>(key_row(best)) * kURowWords);
#pragma unroll
        for (int line = 0; line < 5; ++line)
          ptx::prefetch_l2(row + line * 128);
        atomicMax(pair(k, b), best);
      }
      u64 key[kMaxB / 32] = {};  // the winner key of requests lane + 32 q (zero-initialized: kept in registers)
      for (int b = lane; b < nb; b += 32)
        ptx::red_add_release(pair(k, b) + 1, 1);
#pragma unroll
      for (int q = 0; q < kMaxB / 32; ++q) {
        const int b = lane + 32 * q;
        if (b < nb) {
          ulonglong2 v;
          do {
            v = ptx::ld_relaxed_v2(pair(k, b));
          } while (v.y < n);
          key[q] = v.x;
        }
      }
#pragma unroll
      for (int q = 0; q < kMaxB / 32; ++q) {
        const int b = lane + 32 * q;
        if (b < nb) {
          const unsigned tok = key_row(key[q]);
          tok_s[b] = tok;
          if (blockIdx.x == 0) tokens[static_cast<size_t>(b) * num_steps + k] = tok;
        }
      }
    }
    __syncthreads();
  }
  // every CTA has seen every final count (no CTA still reads this round's state): CTA 0 closes the round
  if (blockIdx.x == 0 && threadIdx.x == 0) ptx::st_relaxed(state, round + 1);
}

/**
 * \brief One round of the batched (bs 2..4; the launcher takes 1..8) markov walk in one cooperative launch.
 *
 * \param frag       int8 [rows_pad * 256], W2 in m16n8k32 A-fragment order (as dspark_markov_walk_single.cuh).
 * \param row_scale  fp32 [rows_pad], W2 row scales; rows_pad = grid x 1152 sets the grid.
 * \param w1q        uint8 [V, 528], W1 rows: q_hi | q_lo | s_hi f32 | s_lo f32 | pad.
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
inline void walk(
    tvm::ffi::TensorView frag,
    tvm::ffi::TensorView row_scale,
    tvm::ffi::TensorView w1q,
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
  auto rows_pad = SymbolicSize{"rows_pad"};
  auto w1_rows = SymbolicSize{"w1_rows"};
  auto nb = SymbolicSize{"batch_size"};
  auto kb = SymbolicSize{"base_steps"};
  auto ld = SymbolicSize{"base_row_stride"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  CHECK_HOST(num_steps >= 1 && num_steps <= kMaxSteps) << "num_steps must be in [1, " << kMaxSteps << "]";
  TensorMatcher({rows_pad}).with_dtype<float>().with_device(device).ensure_alignment(16).verify(row_scale);
  CHECK_HOST(rows_pad.unwrap() >= kRowsCta && rows_pad.unwrap() % kRowsCta == 0)
      << "rows_pad (row_scale.numel()) " << rows_pad.unwrap() << " not a multiple of the CTA rows " << kRowsCta;
  TensorMatcher({rows_pad.unwrap() * 256}).with_dtype<int8_t>().with_device(device).ensure_alignment(16).verify(frag);
  TensorMatcher({w1_rows, kURowBytes}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w1q);
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
  CHECK_HOST(nb_v >= 1 && nb_v <= kPairMaxB) << "batch must be in [1, " << kPairMaxB << "], got " << nb_v;
  CHECK_HOST(kb_v >= num_steps) << "base has " << kb_v << " steps < num_steps " << num_steps;
  CHECK_HOST(
      ld_v % 8 == 0 && valid_rows % 8 == 0 && valid_rows >= 8 && valid_rows <= ld_v &&
      valid_rows <= rows_pad.unwrap() && valid_rows <= w1_rows.unwrap())
      << "need ld % 8 == 0, valid_rows % 8 == 0, 8 <= valid_rows <= min(ld, rows_pad, W1 rows); got ld " << ld_v
      << ", valid_rows " << valid_rows << ", rows_pad " << rows_pad.unwrap() << ", W1 rows " << w1_rows.unwrap();
  CHECK_HOST(nb_v * kb_v * ld_v < (int64_t{1} << 31)) << "base: element offsets must fit in 31 bits";
  // base / bias double buffers, row scales, SMEM tiles, then nb + 1 u rows
  constexpr std::size_t kSmemFixed = 2 * kGrp * kRowsCta * 6 + 64 * kTilesPerCta + kTilesSmem * kTileVecs * 16;
  launch_cooperative(
      markov_walk_small_batch_kernel,
      device.unwrap(),
      static_cast<uint32_t>(rows_pad.unwrap() / kRowsCta),
      kThreads,
      kSmemFixed + static_cast<std::size_t>(nb_v + 1) * kURowBytes,
      kSmemFixed + (kPairMaxB + 1) * kURowBytes,
      static_cast<const uint4*>(frag.data_ptr()),
      static_cast<const float*>(row_scale.data_ptr()),
      static_cast<const uint32_t*>(w1q.data_ptr()),
      static_cast<const __nv_bfloat16*>(base.data_ptr()),
      static_cast<const int64_t*>(anchor.data_ptr()),
      static_cast<int64_t*>(tokens.data_ptr()),
      corrected.has_value() ? static_cast<__nv_bfloat16*>(corrected.value().data_ptr()) : nullptr,
      static_cast<u64*>(state.data_ptr()),
      static_cast<const float*>(temps.data_ptr()),
      static_cast<int>(nb_v),
      static_cast<int>(num_steps),
      static_cast<int>(state_steps),
      static_cast<int>(kb_v),
      static_cast<int>(ld_v),
      static_cast<int>(valid_rows),
      static_cast<u64>(seed));
}

}  // namespace sglang::dspark_markov_walk::small_batch
