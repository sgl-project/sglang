// SPDX-License-Identifier: MIT
// Single-warp top-k selection of one row's global-slab candidates (fp24 codes,
// a higher code is a higher score). select_stream_row brackets the k-th
// largest code with sampled codes, places the codes above the bracket directly
// and resolves the bracket with histogram levels on register-resident codes;
// select_global_row is its exact radix fallback.
#pragma once

#include "candidates.cuh"
#include "common.cuh"
#include <cstdint>

namespace sglang::fused_mqa_topk::row_select {

// Shared-memory scratch is sized for kMaxK; the row's top-k K is a runtime value <= kMaxK.
inline constexpr uint32_t kMaxK = kTopK;
// Bins of the radix levels of select_global_row and of the levels that resolve
// the exact threshold inside a bracket.
inline constexpr uint32_t kBins = 1024, kMidBins = 512;

// Bin (descending) where the cumulative count first reaches `rank`;
// `above` = elements in higher bins. Lane l owns bins kBins-1-32l .. -31.
__device__ __forceinline__ uint32_t find_bin(const uint32_t* hist, uint32_t rank, uint32_t& above) {
  const uint32_t lane = threadIdx.x & 31;
  constexpr uint32_t kPerLane = kBins / 32;
  uint32_t sum = 0;
#pragma unroll
  for (uint32_t j = 0; j < kPerLane; ++j)
    sum += hist[kBins - 1 - lane * kPerLane - j];
  uint32_t inclusive = sum;
  inclusive = warp_inclusive_scan(inclusive, lane);
  const uint32_t before = inclusive - sum;
  const uint32_t winner = __ffs(__ballot_sync(kFullMask, before < rank && inclusive >= rank)) - 1;
  uint32_t chosen = 0, acc = before;
  if (lane == winner) {
    for (uint32_t j = 0; j < kPerLane; ++j) {
      const uint32_t bin = kBins - 1 - lane * kPerLane - j, h = hist[bin];
      if (acc + h >= rank) {
        chosen = bin;
        break;
      }
      acc += h;
    }
  }
  above = __shfl_sync(kFullMask, acc, winner);
  return __shfl_sync(kFullMask, chosen, winner);
}

__device__ __forceinline__ uint32_t warp_exclusive_scan(uint32_t v, uint32_t& total) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t inc = v;
  inc = warp_inclusive_scan(inc, lane);
  total = __shfl_sync(kFullMask, inc, 31);
  return inc - v;
}

// find_bin for kNumBins bins, each lane's kPer contiguous bins loaded as
// 16-byte vectors and searched in registers.
template <uint32_t kNumBins>
__device__ __forceinline__ uint32_t find_bin_v(const uint32_t* hist, uint32_t rank, uint32_t& above) {
  const uint32_t lane = threadIdx.x & 31;
  constexpr uint32_t kPer = kNumBins / 32, kVec = kPer / 4;
  static_assert(kPer % 4 == 0);
  // lane l owns bins [kNumBins - (l + 1) * kPer, kNumBins - l * kPer), highest first.
  const uint4* hv = reinterpret_cast<const uint4*>(hist + kNumBins - (lane + 1) * kPer);
  uint32_t h[kPer];
#pragma unroll
  for (uint32_t v = 0; v < kVec; ++v) {
    const uint4 x = hv[v];
    h[4 * v] = x.x;
    h[4 * v + 1] = x.y;
    h[4 * v + 2] = x.z;
    h[4 * v + 3] = x.w;
  }
  uint32_t sum = 0;
#pragma unroll
  for (uint32_t j = 0; j < kPer; ++j)
    sum += h[j];
  uint32_t inclusive = sum;
  inclusive = warp_inclusive_scan(inclusive, lane);
  const uint32_t before = inclusive - sum;
  const uint32_t winner = __ffs(__ballot_sync(kFullMask, before < rank && inclusive >= rank)) - 1;
  // Descending walk inside the lane: bin kNumBins - lane * kPer - 1 - j is h[kPer - 1 - j].
  uint32_t acc = before, chosen = 0, found = 0, acc_at = before;
#pragma unroll
  for (uint32_t j = 0; j < kPer; ++j) {
    const uint32_t x = h[kPer - 1 - j];
    const bool hit = !found && acc + x >= rank;
    if (hit) {
      chosen = kNumBins - lane * kPer - 1 - j;
      acc_at = acc;
      found = 1;
    }
    acc += x;
  }
  above = __shfl_sync(kFullMask, acc_at, winner);
  return __shfl_sync(kFullMask, chosen, winner);
}

// Streaming selection: no staging pass.
// Access unit: a lane reads 4 consecutive candidates (16 bytes of packed
// indices, 8 bytes of values), a warp 128 (one "group": 4 + 2 cache lines).
//  1. bracket: 32 clusters of 32 consecutive candidates (8 lanes per
//     cluster) at stratified positions give 1024 sample codes and the
//     bracket [lo, hi] of the k-th largest fp24 code (512 bins relative to
//     the sample range, so rows whose scores share a large common offset --
//     GLM-5.2 layer 1 -- still get a narrow bracket);
//  2. one coalesced pass over the row's global candidates (8 groups per
//     iteration, the next iteration's loads in flight): codes above the
//     bracket -> their keys to the stage, bracket elements (code, key) to the
//     mid list, as per-lane runs placed by one warp scan per iteration;
//  3. the k-th code inside the bracket: 512-bin levels over the register-
//     resident bracket codes until the chosen bin holds <= 32 elements (or is
//     one code wide), resolved by pairwise ranks;
//  4. the stage (K keys) is copied to out with 16-byte stores.
// The slab entries [count, round128(count)) are zeroed first (code 0 is below
// every candidate's code and every bracket), so partial groups need no
// per-element test.
// scratch: kStreamWords words per warp (16-byte aligned). wide: the retry's
// wider bracket margin. Returns false (out untouched) when the bracket misses;
// the caller then retries once with wide = true and, on a second miss, runs
// select_global_row.
inline constexpr uint32_t kStreamMidCap = 1536, kStreamWords = kMaxK + 2 * kStreamMidCap + 512 + 64;

// Per-element classification against the bracket [lo, hi] (PTX, so every
// element costs two compares and predicated adds / stores, no branches):
// count: cnt += (c > hi) + ((lo <= c <= hi) << 16)
__device__ __forceinline__ void classify_count(uint32_t c, uint32_t lo, uint32_t hi, uint32_t& cnt) {
  asm("{\n\t.reg .pred a, m;\n\t"
      "setp.gt.u32 a, %1, %3;\n\t"
      "setp.ge.and.u32 m, %1, %2, !a;\n\t"
      "@a add.u32 %0, %0, 1;\n\t"
      "@m add.u32 %0, %0, 65536;\n\t}"
      : "+r"(cnt)
      : "r"(c), "r"(lo), "r"(hi));
}
// store: c > hi -> key at [pa] (pa += 4); lo <= c <= hi -> (c, key) at [pm]
// (pm += 8). pa / pm are shared-memory byte addresses.
__device__ __forceinline__ void
classify_store(uint32_t c, uint32_t key, uint32_t lo, uint32_t hi, uint32_t& pa, uint32_t& pm) {
  asm volatile(
      "{\n\t.reg .pred a, m;\n\t"
      "setp.gt.u32 a, %2, %5;\n\t"
      "setp.ge.and.u32 m, %2, %4, !a;\n\t"
      "@a st.shared.u32 [%0], %3;\n\t"
      "@a add.u32 %0, %0, 4;\n\t"
      "@m st.shared.v2.u32 [%1], {%2, %3};\n\t"
      "@m add.u32 %1, %1, 8;\n\t}"
      : "+r"(pa), "+r"(pm)
      : "r"(c), "r"(key), "r"(lo), "r"(hi)
      : "memory");
}
// fp24 code of element k (0..3) of a lane's quad: packed bits 20..27 above
// the 16 value bits
__device__ __forceinline__ uint32_t quad_code(const uint2& v, const uint4& p, uint32_t k) {
  const uint32_t vw = (k < 2) ? v.x : v.y;
  const uint32_t q = k == 0 ? p.x : k == 1 ? p.y : k == 2 ? p.z : p.w;
  return __byte_perm(vw, q >> 4, (k & 1) ? 0x7632u : 0x7610u);
}
__device__ __forceinline__ uint32_t quad_packed(const uint4& p, uint32_t k) {
  return k == 0 ? p.x : k == 1 ? p.y : k == 2 ? p.z : p.w;
}

// Bins (descending) of ranks r_hi <= r_lo in a 512-bin histogram; above_hi:
// samples in bins above b_hi; n_hi / n_lo: the two bins' counts.
__device__ __forceinline__ void find_two_bins(
    const uint32_t* hist,
    uint32_t r_hi,
    uint32_t r_lo,
    uint32_t& b_hi,
    uint32_t& b_lo,
    uint32_t& above_hi,
    uint32_t& n_hi,
    uint32_t& n_lo) {
  const uint32_t lane = threadIdx.x & 31;
  // lane l owns bins [512 - 16 (l + 1), 512 - 16 l), highest first
  const uint4* hv = reinterpret_cast<const uint4*>(hist + 512 - (lane + 1) * 16);
  uint32_t h[16];
#pragma unroll
  for (uint32_t v = 0; v < 4; ++v) {
    const uint4 x = hv[v];
    h[4 * v] = x.x;
    h[4 * v + 1] = x.y;
    h[4 * v + 2] = x.z;
    h[4 * v + 3] = x.w;
  }
  uint32_t sum = 0;
#pragma unroll
  for (uint32_t j = 0; j < 16; ++j)
    sum += h[j];
  uint32_t incl = sum;
  incl = warp_inclusive_scan(incl, lane);
  const uint32_t before = incl - sum;
  uint32_t bh = 0, bl = 0, ah = 0, nh = 0, nl = 0;
  {
    uint32_t acc = before;
    bool fh = false, fl = false;
#pragma unroll
    for (uint32_t j = 0; j < 16; ++j) {
      const uint32_t x = h[15 - j], bin = 512 - lane * 16 - 1 - j;
      const bool hh = !fh && acc + x >= r_hi, hl = !fl && acc + x >= r_lo;
      bh = hh ? bin : bh;
      ah = hh ? acc : ah;
      nh = hh ? x : nh;
      bl = hl ? bin : bl;
      nl = hl ? x : nl;
      fh |= hh;
      fl |= hl;
      acc += x;
    }
  }
  const uint32_t wh = __ffs(__ballot_sync(kFullMask, before < r_hi && incl >= r_hi)) - 1,
                 wl = __ffs(__ballot_sync(kFullMask, before < r_lo && incl >= r_lo)) - 1;
  b_hi = __shfl_sync(kFullMask, bh, wh);
  above_hi = __shfl_sync(kFullMask, ah, wh);
  n_hi = __shfl_sync(kFullMask, nh, wh);
  b_lo = __shfl_sync(kFullMask, bl, wl);
  n_lo = __shfl_sync(kFullMask, nl, wl);
}

// Bracket [lo, hi] (lo >= 1) of the k-th largest code from NS sample codes
// per lane (0 = no sample). hist: 512 words. Bins relative to the sample
// range; when the two rank bins are crowded (an outlier sample stretches the
// range), a second level re-bins the samples between them.
template <uint32_t NS>
__device__ __forceinline__ void sample_bracket(
    const uint32_t (&smp)[NS], uint32_t count, uint32_t K, uint32_t* hist, uint32_t& lo, uint32_t& hi, bool wide) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t mn = 0xffffffffu, mx = 0, ns = 0;
#pragma unroll
  for (uint32_t i = 0; i < NS; ++i) {
    const bool v = smp[i] != 0u;
    mn = v ? min(mn, smp[i]) : mn;
    mx = max(mx, smp[i]);
    ns += v;
  }
#pragma unroll
  for (uint32_t b = lane; b < 128; b += 32)
    reinterpret_cast<uint4*>(hist)[b] = make_uint4(0, 0, 0, 0);
  mn = __reduce_min_sync(kFullMask, mn);
  mx = __reduce_max_sync(kFullMask, mx);
  const uint32_t S = __reduce_add_sync(kFullMask, ns);
  uint32_t base = mn, width = mx - mn;
  uint32_t shift = (width >> 9) ? 32 - __clz(width >> 9) : 0;
  __syncwarp();
#pragma unroll
  for (uint32_t i = 0; i < NS; ++i)
    if (smp[i]) atomicAdd(hist + ((smp[i] - base) >> shift), 1u);
  // Expected sample rank of the k-th largest and a 5 sd margin (9 sd for the
  // retry after a miss); sd from the hypergeometric model (S of count drawn,
  // K marked), which the clustered sample exceeds by ~20% on some layers.
  const float p = float(K) / float(count);
  const float var = float(S) * p * (1.0f - p) * (1.0f - float(S) / float(count));
  const uint32_t D = wide ? uint32_t(9.0f * sqrtf(var)) + 24u : uint32_t(5.0f * sqrtf(var)) + 8u;
  const uint32_t expect = (K * S + count / 2) / count;
  uint32_t r_hi = expect > D ? expect - D : 1u, r_lo = min(expect + D, S);
  const bool hi_open = r_hi == 1u, lo_open = r_lo == S;
  __syncwarp();
  uint32_t b_hi, b_lo, a_hi, n_hi, n_lo;
  find_two_bins(hist, r_hi, r_lo, b_hi, b_lo, a_hi, n_hi, n_lo);
  if (shift > 0 && n_hi + n_lo > 16) {
    // zoom: [edge(b_lo), edge(b_hi + 1)) in 512 finer bins
    const uint32_t zlo = base + (b_lo << shift), zhi = base + ((b_hi + 1) << shift) - 1;
    __syncwarp();
#pragma unroll
    for (uint32_t b = lane; b < 128; b += 32)
      reinterpret_cast<uint4*>(hist)[b] = make_uint4(0, 0, 0, 0);
    base = zlo;
    width = zhi - zlo;
    shift = (width >> 9) ? 32 - __clz(width >> 9) : 0;
    __syncwarp();
#pragma unroll
    for (uint32_t i = 0; i < NS; ++i)
      if (smp[i] >= zlo && smp[i] <= zhi) atomicAdd(hist + ((smp[i] - base) >> shift), 1u);
    r_hi -= a_hi;
    r_lo -= a_hi;
    __syncwarp();
    find_two_bins(hist, r_hi, r_lo, b_hi, b_lo, a_hi, n_hi, n_lo);
  }
  hi = hi_open ? 0xffffffffu : base + ((b_hi + 1) << shift) - 1;
  lo = lo_open ? 1u : max(base + (b_lo << shift), 1u);
  __syncwarp();
}

__device__ __noinline__ bool select_stream_row(
    uint32_t* __restrict__ scratch,
    uint32_t count,
    uint16_t* __restrict__ values,
    int32_t* __restrict__ packed,
    int32_t* __restrict__ out,
    uint32_t K,
    bool wide) {
  const uint32_t lane = threadIdx.x & 31, lt = (1u << lane) - 1u;
  if (count <= K) return false;
  uint32_t* const stage = scratch;                             // K keys
  uint32_t* const mid2 = scratch + kMaxK;                      // (code, key)
  uint32_t* const hist = scratch + kMaxK + 2 * kStreamMidCap;  // 512 bins
  uint32_t* const fin = hist + 512;                            // 32 pairs
  const uint32_t stage_s = static_cast<uint32_t>(__cvta_generic_to_shared(stage));
  const uint32_t mid_s = static_cast<uint32_t>(__cvta_generic_to_shared(mid2));
  constexpr uint32_t G = 8, kPer = 4 * G;  // groups per iteration, per lane
  const uint32_t groups = (count + 127) / 128;
  {
    // zero [count, 128 * groups) (within the row's candidate slab)
    const uint32_t zend = 128 * groups;
#pragma unroll
    for (uint32_t z = count + lane; z < zend; z += 32) {
      __stcg(values + z, uint16_t(0));
      __stcg(packed + z, 0);
    }
    __syncwarp();
  }
  const uint2* v2p = reinterpret_cast<const uint2*>(values);
  const uint4* p4p = reinterpret_cast<const uint4*>(packed);
  uint2 vv[G];
  uint4 pp[G];
  const auto load = [&](uint32_t g0) {
#pragma unroll
    for (uint32_t u = 0; u < G; ++u) {
      const uint32_t q = min(g0 + u, groups - 1) * 32 + lane;  // quad index
      vv[u] = __ldcg(v2p + q);
      pp[u] = __ldcg(p4p + q);
    }
  };
  // ---- 1. bracket from 32 stratified clusters of 32 candidates ----
  uint32_t lo, hi;
  {
    uint2 sv[8];
    uint4 sp[8];
#pragma unroll
    for (uint32_t u = 0; u < 8; ++u) {
      // cluster i = 4 u + lane / 8 at the (aligned) center of stratum i
      const uint32_t i = 4 * u + lane / 8;
      const uint32_t c = ((((2 * i + 1) * count) >> 6) - 16u) & ~3u;
      const uint32_t q = (min(c, count - 32u) & ~3u) / 4 + (lane % 8);
      sv[u] = __ldcg(v2p + q);
      sp[u] = __ldcg(p4p + q);
    }
    load(0);  // in flight during the bracket
    uint32_t smp[32];
#pragma unroll
    for (uint32_t u = 0; u < 8; ++u)
#pragma unroll
      for (uint32_t k = 0; k < 4; ++k)
        smp[4 * u + k] = quad_code(sv[u], sp[u], k);
    sample_bracket<32>(smp, count, K, hist, lo, hi, wide);
  }
  // ---- 2. classify (coalesced) ----
  uint32_t above = 0, mid = 0;
  uint32_t cw[kPer], pw[kPer];
#pragma unroll 1
  for (uint32_t g0 = 0; g0 < groups; g0 += G) {
#pragma unroll
    for (uint32_t u = 0; u < G; ++u)
#pragma unroll
      for (uint32_t k = 0; k < 4; ++k) {
        cw[4 * u + k] = quad_code(vv[u], pp[u], k);
        pw[4 * u + k] = quad_packed(pp[u], k);
      }
    if (g0 + G < groups) load(g0 + G);
    // per-lane runs placed by one warp scan per iteration
    uint32_t cnt = 0;  // above | mid << 16 of this lane's elements
#pragma unroll
    for (uint32_t u = 0; u < G; ++u)
      if (g0 + u < groups) {
#pragma unroll
        for (uint32_t k = 0; k < 4; ++k)
          classify_count(cw[4 * u + k], lo, hi, cnt);
      }
    uint32_t tot;
    const uint32_t ex = warp_exclusive_scan(cnt, tot);
    uint32_t pa = stage_s + 4u * (above + (ex & 0xffffu)), pm = mid_s + 8u * (mid + (ex >> 16));
    above += tot & 0xffffu;
    mid += tot >> 16;
    if (above >= K || mid > kStreamMidCap) {
      return false;
    }
#pragma unroll
    for (uint32_t u = 0; u < G; ++u)
      if (g0 + u < groups) {
#pragma unroll
        for (uint32_t k = 0; k < 4; ++k)
          classify_store(cw[4 * u + k], pw[4 * u + k], lo, hi, pa, pm);
      }
  }
  __syncwarp();
  if (above + mid < K) return false;
  // ---- 3. the need-th largest of the mid codes ----
  constexpr uint32_t R = kStreamMidCap / 32;
  uint32_t mc[R];
#pragma unroll
  for (uint32_t u = 0; u < R; ++u) {
    const uint32_t i = lane + 32 * u;
    mc[u] = i < mid ? mid2[2 * i] : 0u;  // 0: below lo
  }
  // levels: the need-th largest lies in [base, top] (r-th largest there)
  uint32_t r = K - above, base = lo, top = hi, nb = mid;
#pragma unroll 1
  while (nb > 32 && top > base) {
    const uint32_t width = top - base + 1, spread = (width - 1) / kMidBins;
    // Unguarded: __clz(0) = 32 and spread == 0 selects shift 0 below. A guard
    // changes this function's SASS (nvcc 13.0) from the build the kernel was
    // measured with.
    const uint32_t lz = __clz(spread);
    const uint32_t shift = spread ? 32 - lz : 0;
#pragma unroll
    for (uint32_t b = lane; b < kMidBins / 4; b += 32)
      reinterpret_cast<uint4*>(hist)[b] = make_uint4(0, 0, 0, 0);
    __syncwarp();
#pragma unroll
    for (uint32_t u = 0; u < R; ++u) {
      if (u * 32 >= mid) break;
      if (mc[u] >= base && mc[u] <= top) atomicAdd(hist + ((mc[u] - base) >> shift), 1u);
    }
    __syncwarp();
    uint32_t above_bin;
    const uint32_t chosen = find_bin_v<kMidBins>(hist, r, above_bin);
    nb = hist[chosen];
    r -= above_bin;
    base += chosen << shift;
    top = min(base + (1u << shift) - 1, top);
    __syncwarp();
  }
  // ---- placement: codes > top all selected (one pass also gathers the
  // nb codes in [base, top]); r of those by (code desc, position asc) ----
  {
    uint32_t pg = above, pf = 0;
#pragma unroll
    for (uint32_t u = 0; u < R; ++u) {
      if (u * 32 >= mid) break;
      const uint32_t i = lane + 32 * u;
      // unconditional load (i < kStreamMidCap): no branch around it
      const uint32_t key = mid2[2 * i + 1];
      const bool g = mc[u] > top, f = mc[u] >= base && !g;
      const uint32_t mg = __ballot_sync(kFullMask, g), mf = __ballot_sync(kFullMask, f);
      const uint32_t ag = pg + __popc(mg & lt), af = pf + __popc(mf & lt);
      if (g) stage[ag] = key;
      if (nb <= 32) {
        if (f) {
          fin[2 * af] = mc[u];
          fin[2 * af + 1] = key;
        }
      } else if (f && af < r) {  // one code wide, > 32 ties: first r
        stage[K - r + af] = key;
      }
      pg += __popc(mg);
      pf += __popc(mf);
    }
    if (nb <= 32) {
      __syncwarp();
      const uint32_t c = lane < nb ? fin[2 * lane] : 0u;
      uint32_t rk = 0;
#pragma unroll 8
      for (uint32_t j = 0; j < 32; ++j) {
        const uint32_t cj = __shfl_sync(kFullMask, c, j);
        rk += cj > c || (cj == c && j < lane);
      }
      const bool sel = lane < nb && rk < r;
      const uint32_t ms = __ballot_sync(kFullMask, sel);
      if (sel) stage[K - r + __popc(ms & lt)] = fin[2 * lane + 1];
    }
  }
  __syncwarp();
  // ---- 4. copy to out ----
#pragma unroll 4
  for (uint32_t j = lane * 4; j < K; j += 128) {
    const uint4 v = *reinterpret_cast<const uint4*>(stage + j);
    *reinterpret_cast<int4*>(out + j) = make_int4(
        int(v.x & kCandidateIndexMask),
        int(v.y & kCandidateIndexMask),
        int(v.z & kCandidateIndexMask),
        int(v.w & kCandidateIndexMask));
  }
  return true;
}

// Exact fallback of select_stream_row without staging (bracket miss): radix
// levels with kBins-bin histograms over the codes read from global memory,
// then the placement of the codes above / at the threshold. hist: kBins words.
__device__ __noinline__ void select_global_row(
    uint32_t count,
    const uint16_t* __restrict__ values,
    const int32_t* __restrict__ packed,
    int32_t* __restrict__ out,
    uint32_t K,
    uint32_t* hist) {
  const uint32_t lane = threadIdx.x & 31, lt = (1u << lane) - 1u;
  const uint32_t chunks = (count + 7) / 8;
  const uint4* v4p = reinterpret_cast<const uint4*>(values);
  const uint4* p4p = reinterpret_cast<const uint4*>(packed);
  constexpr uint32_t C = 4;
  // f(code, key, valid) over all elements, C chunks of 8 in flight per lane.
  const auto pass = [&](auto&& f) {
#pragma unroll 1
    for (uint32_t c0 = 0; c0 < chunks; c0 += 32 * C) {
      uint4 vv[C], pp0[C], pp1[C];
#pragma unroll
      for (uint32_t u = 0; u < C; ++u) {
        const uint32_t c = min(c0 + 32 * u + lane, chunks - 1);
        vv[u] = __ldcg(v4p + c);
        pp0[u] = __ldcg(p4p + 2 * c);
        pp1[u] = __ldcg(p4p + 2 * c + 1);
      }
#pragma unroll
      for (uint32_t u = 0; u < C; ++u) {
        if (c0 + 32 * u >= chunks) break;
        const uint32_t c = c0 + 32 * u + lane;
        const uint32_t vw[4] = {vv[u].x, vv[u].y, vv[u].z, vv[u].w};
        const uint32_t pw[8] = {pp0[u].x, pp0[u].y, pp0[u].z, pp0[u].w, pp1[u].x, pp1[u].y, pp1[u].z, pp1[u].w};
#pragma unroll
        for (uint32_t k = 0; k < 8; ++k) {
          const uint32_t code = candidate_load_score_code(uint16_t(vw[k / 2] >> (16 * (k & 1))), int32_t(pw[k]));
          f(code, candidate_decode_index(int32_t(pw[k])), 8 * c + k < count);
        }
      }
    }
  };
  uint32_t lo = 0xffffffffu, top = 0;
  pass([&](uint32_t code, int32_t, bool valid) {
    if (valid) {
      lo = min(lo, code);
      top = max(top, code);
    }
  });
  lo = __reduce_min_sync(kFullMask, lo);
  top = __reduce_max_sync(kFullMask, top);
  uint32_t rank = K;
#pragma unroll 1
  while (top > lo) {
    const uint32_t spread = (top - lo) / kBins;
    const uint32_t shift = spread ? 32 - __clz(spread) : 0;
    for (uint32_t b = lane; b < kBins; b += 32)
      hist[b] = 0;
    __syncwarp();
    const uint32_t base = lo, t = top;
    pass([&](uint32_t code, int32_t, bool valid) {
      if (valid && code >= base && code <= t) atomicAdd(hist + ((code - base) >> shift), 1u);
    });
    __syncwarp();
    uint32_t above;
    const uint32_t chosen = find_bin(hist, rank, above);
    rank -= above;
    lo += chosen << shift;
    top = min(lo + (1u << shift) - 1, top);
    __syncwarp();
  }
  const uint32_t th = lo;
  uint32_t pg = 0, pt = K - rank;
  pass([&](uint32_t code, int32_t key, bool valid) {
    const bool g = valid && code > th, tq = valid && code == th;
    const uint32_t mg = __ballot_sync(kFullMask, g), mt = __ballot_sync(kFullMask, tq);
    const uint32_t ag = pg + __popc(mg & lt), at = pt + __popc(mt & lt);
    if (g) out[ag] = key;
    if (tq && at < K) out[at] = key;
    pg += __popc(mg);
    pt += __popc(mt);
  });
}

}  // namespace sglang::fused_mqa_topk::row_select
