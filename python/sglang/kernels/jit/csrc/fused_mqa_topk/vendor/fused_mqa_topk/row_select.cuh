// SPDX-License-Identifier: MIT
// Single-warp top-K selection of one row's global-slab candidates for the
// persistent kernel. The streaming selection brackets the K-th largest fp24
// code with sampled codes, places the codes above the bracket directly, and
// resolves the bracket with histogram levels on register-resident codes;
// select_global_row is its exact radix fallback.
#pragma once
#include "fast_select.cuh"

namespace row_select {
using namespace fast_select;

// fast_select::find_bin for Bins bins, each lane's kPer contiguous bins loaded
// as 16-byte vectors and searched in registers.
template <unsigned Bins>
__device__ __forceinline__ unsigned find_bin_v(const unsigned* hist,
                                               unsigned rank, unsigned& above) {
  const unsigned lane = threadIdx.x & 31;
  constexpr unsigned kPer = Bins / 32, kVec = kPer / 4;
  static_assert(kPer % 4 == 0);
  // lane l owns bins [Bins - (l + 1) * kPer, Bins - l * kPer), highest first.
  const uint4* hv =
      reinterpret_cast<const uint4*>(hist + Bins - (lane + 1) * kPer);
  unsigned h[kPer];
#pragma unroll
  for (unsigned v = 0; v < kVec; ++v) {
    const uint4 x = hv[v];
    h[4 * v] = x.x;
    h[4 * v + 1] = x.y;
    h[4 * v + 2] = x.z;
    h[4 * v + 3] = x.w;
  }
  unsigned sum = 0;
#pragma unroll
  for (unsigned j = 0; j < kPer; ++j) sum += h[j];
  unsigned inclusive = sum;
#pragma unroll
  for (unsigned d = 1; d < 32; d *= 2) {
    const unsigned v = __shfl_up_sync(Full, inclusive, d);
    if (lane >= d) inclusive += v;
  }
  const unsigned before = inclusive - sum;
  const unsigned winner =
      __ffs(__ballot_sync(Full, before < rank && inclusive >= rank)) - 1;
  // Descending walk inside the lane: bin index Bins - lane * kPer - 1 - j  <->
  // h[kPer - 1 - j].
  unsigned acc = before, chosen = 0, found = 0, acc_at = before;
#pragma unroll
  for (unsigned j = 0; j < kPer; ++j) {
    const unsigned x = h[kPer - 1 - j];
    const bool hit = !found && acc + x >= rank;
    if (hit) {
      chosen = Bins - lane * kPer - 1 - j;
      acc_at = acc;
      found = 1;
    }
    acc += x;
  }
  above = __shfl_sync(Full, acc_at, winner);
  return __shfl_sync(Full, chosen, winner);
}

// Streaming selection: no staging pass.
// Access unit: a lane reads 4 consecutive candidates (16 bytes of packed
// indices, 8 bytes of values), a warp 128 (one "group": 4 + 2 cache lines).
//  1. bracket: 32 clusters of 32 consecutive candidates (8 lanes per
//     cluster) at stratified positions give 1024 sample codes and the
//     bracket [lo, hi] of the K-th largest fp24 code (512 bins relative to
//     the sample range, so rows whose scores share a large common offset --
//     GLM-5.2 layer 1 -- still get a narrow bracket);
//  2. one coalesced pass over the row's global candidates (8 groups per
//     iteration, the next iteration's loads in flight): codes above the
//     bracket -> their keys to the stage, bracket elements (code, key) to the
//     mid list, as per-lane runs placed by one warp scan per iteration;
//  3. the K-th code inside the bracket: 512-bin levels over the register-
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
constexpr unsigned kStreamMidCap = 1536,
                   kStreamWords = kMaxK + 2 * kStreamMidCap + 512 + 64;

// Per-element classification against the bracket [lo, hi] (PTX, so every
// element costs two compares and predicated adds / stores, no branches):
// count: cnt += (c > hi) + ((lo <= c <= hi) << 16)
__device__ __forceinline__ void classify_count(unsigned c, unsigned lo,
                                               unsigned hi, unsigned& cnt) {
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
__device__ __forceinline__ void classify_store(unsigned c, unsigned key,
                                               unsigned lo, unsigned hi,
                                               uint32_t& pa, uint32_t& pm) {
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
__device__ __forceinline__ unsigned quad_code(const uint2& v, const uint4& p,
                                              unsigned k) {
  const unsigned vw = (k < 2) ? v.x : v.y;
  const unsigned q = k == 0 ? p.x : k == 1 ? p.y : k == 2 ? p.z : p.w;
  return __byte_perm(vw, q >> 4, (k & 1) ? 0x7632u : 0x7610u);
}
__device__ __forceinline__ unsigned quad_packed(const uint4& p, unsigned k) {
  return k == 0 ? p.x : k == 1 ? p.y : k == 2 ? p.z : p.w;
}

// Bins (descending) of ranks r_hi <= r_lo in a 512-bin histogram; above_hi:
// samples in bins above b_hi; n_hi / n_lo: the two bins' counts.
__device__ __forceinline__ void find_two_bins(const unsigned* hist,
                                              unsigned r_hi, unsigned r_lo,
                                              unsigned& b_hi, unsigned& b_lo,
                                              unsigned& above_hi,
                                              unsigned& n_hi, unsigned& n_lo) {
  const unsigned lane = threadIdx.x & 31;
  // lane l owns bins [512 - 16 (l + 1), 512 - 16 l), highest first
  const uint4* hv = reinterpret_cast<const uint4*>(hist + 512 - (lane + 1) * 16);
  unsigned h[16];
#pragma unroll
  for (unsigned v = 0; v < 4; ++v) {
    const uint4 x = hv[v];
    h[4 * v] = x.x;
    h[4 * v + 1] = x.y;
    h[4 * v + 2] = x.z;
    h[4 * v + 3] = x.w;
  }
  unsigned sum = 0;
#pragma unroll
  for (unsigned j = 0; j < 16; ++j) sum += h[j];
  unsigned incl = sum;
#pragma unroll
  for (unsigned d = 1; d < 32; d *= 2) {
    const unsigned t = __shfl_up_sync(Full, incl, d);
    if (lane >= d) incl += t;
  }
  const unsigned before = incl - sum;
  unsigned bh = 0, bl = 0, ah = 0, nh = 0, nl = 0;
  {
    unsigned acc = before;
    bool fh = false, fl = false;
#pragma unroll
    for (unsigned j = 0; j < 16; ++j) {
      const unsigned x = h[15 - j], bin = 512 - lane * 16 - 1 - j;
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
  const unsigned wh =
                     __ffs(__ballot_sync(Full, before < r_hi && incl >= r_hi)) - 1,
                 wl =
                     __ffs(__ballot_sync(Full, before < r_lo && incl >= r_lo)) - 1;
  b_hi = __shfl_sync(Full, bh, wh);
  above_hi = __shfl_sync(Full, ah, wh);
  n_hi = __shfl_sync(Full, nh, wh);
  b_lo = __shfl_sync(Full, bl, wl);
  n_lo = __shfl_sync(Full, nl, wl);
}

// Bracket [lo, hi] (lo >= 1) of the K-th largest code from NS sample codes
// per lane (0 = no sample). hist: 512 words. Bins relative to the sample
// range; when the two rank bins are crowded (an outlier sample stretches the
// range), a second level re-bins the samples between them.
template <unsigned NS>
__device__ __forceinline__ void sample_bracket(const unsigned (&smp)[NS],
                                               unsigned count, unsigned K,
                                               unsigned* hist, unsigned& lo,
                                               unsigned& hi, bool wide) {
  const unsigned lane = threadIdx.x & 31;
  unsigned mn = 0xffffffffu, mx = 0, ns = 0;
#pragma unroll
  for (unsigned i = 0; i < NS; ++i) {
    const bool v = smp[i] != 0u;
    mn = v ? min(mn, smp[i]) : mn;
    mx = max(mx, smp[i]);
    ns += v;
  }
#pragma unroll
  for (unsigned b = lane; b < 128; b += 32)
    reinterpret_cast<uint4*>(hist)[b] = make_uint4(0, 0, 0, 0);
  mn = __reduce_min_sync(Full, mn);
  mx = __reduce_max_sync(Full, mx);
  const unsigned S = __reduce_add_sync(Full, ns);
  unsigned base = mn, width = mx - mn;
  unsigned shift = (width >> 9) ? 32 - __clz(width >> 9) : 0;
  __syncwarp();
#pragma unroll
  for (unsigned i = 0; i < NS; ++i)
    if (smp[i]) atomicAdd(hist + ((smp[i] - base) >> shift), 1u);
  // Expected sample rank of the K-th largest and a 5 sd margin (9 sd for the
  // retry after a miss); sd from the hypergeometric model (S of count drawn,
  // K marked), which the clustered sample exceeds by ~20% on some layers.
  const float p = float(K) / float(count);
  const float var =
      float(S) * p * (1.0f - p) * (1.0f - float(S) / float(count));
  const unsigned D = wide ? unsigned(9.0f * sqrtf(var)) + 24u
                          : unsigned(5.0f * sqrtf(var)) + 8u;
  const unsigned expect = (K * S + count / 2) / count;
  unsigned r_hi = expect > D ? expect - D : 1u, r_lo = min(expect + D, S);
  const bool hi_open = r_hi == 1u, lo_open = r_lo == S;
  __syncwarp();
  unsigned b_hi, b_lo, a_hi, n_hi, n_lo;
  find_two_bins(hist, r_hi, r_lo, b_hi, b_lo, a_hi, n_hi, n_lo);
  if (shift > 0 && n_hi + n_lo > 16) {
    // zoom: [edge(b_lo), edge(b_hi + 1)) in 512 finer bins
    const unsigned zlo = base + (b_lo << shift),
                   zhi = base + ((b_hi + 1) << shift) - 1;
    __syncwarp();
#pragma unroll
    for (unsigned b = lane; b < 128; b += 32)
      reinterpret_cast<uint4*>(hist)[b] = make_uint4(0, 0, 0, 0);
    base = zlo;
    width = zhi - zlo;
    shift = (width >> 9) ? 32 - __clz(width >> 9) : 0;
    __syncwarp();
#pragma unroll
    for (unsigned i = 0; i < NS; ++i)
      if (smp[i] >= zlo && smp[i] <= zhi)
        atomicAdd(hist + ((smp[i] - base) >> shift), 1u);
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
    unsigned* __restrict__ scratch, unsigned count,
    uint16_t* __restrict__ values, int32_t* __restrict__ packed,
    int32_t* __restrict__ out, unsigned K, bool wide) {
  const unsigned lane = threadIdx.x & 31, lt = (1u << lane) - 1u;
  if (count <= K) return false;
  unsigned* const stage = scratch;                             // K keys
  unsigned* const mid2 = scratch + kMaxK;                      // (code, key)
  unsigned* const hist = scratch + kMaxK + 2 * kStreamMidCap;  // 512 bins
  unsigned* const fin = hist + 512;                            // 32 pairs
  const uint32_t stage_s =
      static_cast<uint32_t>(__cvta_generic_to_shared(stage));
  const uint32_t mid_s = static_cast<uint32_t>(__cvta_generic_to_shared(mid2));
  constexpr unsigned G = 8, kPer = 4 * G;  // groups per iteration, per lane
  const unsigned groups = (count + 127) / 128;
  {
    // zero [count, 128 * groups) (within the 16384-entry slab row)
    const unsigned zend = 128 * groups;
#pragma unroll
    for (unsigned z = count + lane; z < zend; z += 32) {
      __stcg(values + z, uint16_t(0));
      __stcg(packed + z, 0);
    }
    __syncwarp();
  }
  const uint2* v2p = reinterpret_cast<const uint2*>(values);
  const uint4* p4p = reinterpret_cast<const uint4*>(packed);
  uint2 vv[G];
  uint4 pp[G];
  const auto load = [&](unsigned g0) {
#pragma unroll
    for (unsigned u = 0; u < G; ++u) {
      const unsigned q = min(g0 + u, groups - 1) * 32 + lane;  // quad index
      vv[u] = __ldcg(v2p + q);
      pp[u] = __ldcg(p4p + q);
    }
  };
  // ---- 1. bracket from 32 stratified clusters of 32 candidates ----
  unsigned lo, hi;
  {
    uint2 sv[8];
    uint4 sp[8];
#pragma unroll
    for (unsigned u = 0; u < 8; ++u) {
      // cluster i = 4 u + lane / 8 at the (aligned) center of stratum i
      const unsigned i = 4 * u + lane / 8;
      const unsigned c = ((((2 * i + 1) * count) >> 6) - 16u) & ~3u;
      const unsigned q = (min(c, count - 32u) & ~3u) / 4 + (lane % 8);
      sv[u] = __ldcg(v2p + q);
      sp[u] = __ldcg(p4p + q);
    }
    load(0);  // in flight during the bracket
    unsigned smp[32];
#pragma unroll
    for (unsigned u = 0; u < 8; ++u)
#pragma unroll
      for (unsigned k = 0; k < 4; ++k) smp[4 * u + k] = quad_code(sv[u], sp[u], k);
    sample_bracket<32>(smp, count, K, hist, lo, hi, wide);
  }
  // ---- 2. classify (coalesced) ----
  unsigned above = 0, mid = 0;
  unsigned cw[kPer], pw[kPer];
#pragma unroll 1
  for (unsigned g0 = 0; g0 < groups; g0 += G) {
#pragma unroll
    for (unsigned u = 0; u < G; ++u)
#pragma unroll
      for (unsigned k = 0; k < 4; ++k) {
        cw[4 * u + k] = quad_code(vv[u], pp[u], k);
        pw[4 * u + k] = quad_packed(pp[u], k);
      }
    if (g0 + G < groups) load(g0 + G);
    // per-lane runs placed by one warp scan per iteration
    unsigned cnt = 0;  // above | mid << 16 of this lane's elements
#pragma unroll
    for (unsigned u = 0; u < G; ++u)
      if (g0 + u < groups) {
#pragma unroll
        for (unsigned k = 0; k < 4; ++k)
          classify_count(cw[4 * u + k], lo, hi, cnt);
      }
    unsigned tot;
    const unsigned ex = warp_exclusive_scan(cnt, tot);
    uint32_t pa = stage_s + 4u * (above + (ex & 0xffffu)),
             pm = mid_s + 8u * (mid + (ex >> 16));
    above += tot & 0xffffu;
    mid += tot >> 16;
    if (above >= K || mid > kStreamMidCap) {
      return false;
    }
#pragma unroll
    for (unsigned u = 0; u < G; ++u)
      if (g0 + u < groups) {
#pragma unroll
        for (unsigned k = 0; k < 4; ++k)
          classify_store(cw[4 * u + k], pw[4 * u + k], lo, hi, pa, pm);
      }
  }
  __syncwarp();
  if (above + mid < K) return false;
  // ---- 3. the need-th largest of the mid codes ----
  constexpr unsigned R = kStreamMidCap / 32;
  unsigned mc[R];
#pragma unroll
  for (unsigned u = 0; u < R; ++u) {
    const unsigned i = lane + 32 * u;
    mc[u] = i < mid ? mid2[2 * i] : 0u;  // 0: below lo
  }
  // levels: the need-th largest lies in [base, top] (r-th largest there)
  unsigned r = K - above, base = lo, top = hi, nb = mid;
#pragma unroll 1
  while (nb > 32 && top > base) {
    const unsigned width = top - base + 1, spread = (width - 1) / kMidBins;
    // Unguarded on purpose (__clz(0) = 32): guarding it changes this function's
    // SASS (nvcc 13.0) from the build the speed numbers were measured with;
    // drop when re-measuring.
    const unsigned lz = __clz(spread);
    const unsigned shift = spread ? 32 - lz : 0;
#pragma unroll
    for (unsigned b = lane; b < kMidBins / 4; b += 32)
      reinterpret_cast<uint4*>(hist)[b] = make_uint4(0, 0, 0, 0);
    __syncwarp();
#pragma unroll
    for (unsigned u = 0; u < R; ++u) {
      if (u * 32 >= mid) break;
      if (mc[u] >= base && mc[u] <= top)
        atomicAdd(hist + ((mc[u] - base) >> shift), 1u);
    }
    __syncwarp();
    unsigned above_bin;
    const unsigned chosen = find_bin_v<kMidBins>(hist, r, above_bin);
    nb = hist[chosen];
    r -= above_bin;
    base += chosen << shift;
    top = min(base + (1u << shift) - 1, top);
    __syncwarp();
  }
  // ---- placement: codes > top all selected (one pass also gathers the
  // nb codes in [base, top]); r of those by (code desc, position asc) ----
  {
    unsigned pg = above, pf = 0;
#pragma unroll
    for (unsigned u = 0; u < R; ++u) {
      if (u * 32 >= mid) break;
      const unsigned i = lane + 32 * u;
      // unconditional load (i < kStreamMidCap): no branch around it
      const unsigned key = mid2[2 * i + 1];
      const bool g = mc[u] > top, f = mc[u] >= base && !g;
      const unsigned mg = __ballot_sync(Full, g), mf = __ballot_sync(Full, f);
      const unsigned ag = pg + __popc(mg & lt), af = pf + __popc(mf & lt);
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
      const unsigned c = lane < nb ? fin[2 * lane] : 0u;
      unsigned rk = 0;
#pragma unroll 8
      for (unsigned j = 0; j < 32; ++j) {
        const unsigned cj = __shfl_sync(Full, c, j);
        rk += cj > c || (cj == c && j < lane);
      }
      const bool sel = lane < nb && rk < r;
      const unsigned ms = __ballot_sync(Full, sel);
      if (sel) stage[K - r + __popc(ms & lt)] = fin[2 * lane + 1];
    }
  }
  __syncwarp();
  // ---- 4. copy to out ----
#pragma unroll 4
  for (unsigned j = lane * 4; j < K; j += 128) {
    const uint4 v = *reinterpret_cast<const uint4*>(stage + j);
    *reinterpret_cast<int4*>(out + j) =
        make_int4(int(v.x & fused_candidates::kCandidateIndexMask),
                  int(v.y & fused_candidates::kCandidateIndexMask),
                  int(v.z & fused_candidates::kCandidateIndexMask),
                  int(v.w & fused_candidates::kCandidateIndexMask));
  }
  return true;
}

// Exact fallback of select_stream_row without staging (bracket miss): radix
// levels with kBins-bin histograms over the codes read from global memory,
// then the placement of the codes above / at the threshold. hist: kBins words.
__device__ __noinline__ void select_global_row(
    unsigned count, const uint16_t* __restrict__ values,
    const int32_t* __restrict__ packed, int32_t* __restrict__ out, unsigned K,
    unsigned* hist) {
  const unsigned lane = threadIdx.x & 31, lt = (1u << lane) - 1u;
  const unsigned chunks = (count + 7) / 8;
  const uint4* v4p = reinterpret_cast<const uint4*>(values);
  const uint4* p4p = reinterpret_cast<const uint4*>(packed);
  constexpr unsigned C = 4;
  // f(code, key, valid) over all elements, C chunks of 8 in flight per lane.
  const auto pass = [&](auto&& f) {
#pragma unroll 1
    for (unsigned c0 = 0; c0 < chunks; c0 += 32 * C) {
      uint4 vv[C], pp0[C], pp1[C];
#pragma unroll
      for (unsigned u = 0; u < C; ++u) {
        const unsigned c = min(c0 + 32 * u + lane, chunks - 1);
        vv[u] = __ldcg(v4p + c);
        pp0[u] = __ldcg(p4p + 2 * c);
        pp1[u] = __ldcg(p4p + 2 * c + 1);
      }
#pragma unroll
      for (unsigned u = 0; u < C; ++u) {
        if (c0 + 32 * u >= chunks) break;
        const unsigned c = c0 + 32 * u + lane;
        const unsigned vw[4] = {vv[u].x, vv[u].y, vv[u].z, vv[u].w};
        const unsigned pw[8] = {pp0[u].x, pp0[u].y, pp0[u].z, pp0[u].w,
                                pp1[u].x, pp1[u].y, pp1[u].z, pp1[u].w};
#pragma unroll
        for (unsigned k = 0; k < 8; ++k) {
          const unsigned code = fused_candidates::candidate_load_score_code(
              uint16_t(vw[k / 2] >> (16 * (k & 1))), int32_t(pw[k]));
          f(code, fused_candidates::candidate_decode_index(int32_t(pw[k])),
            8 * c + k < count);
        }
      }
    }
  };
  unsigned lo = 0xffffffffu, top = 0;
  pass([&](unsigned code, int32_t, bool valid) {
    if (valid) {
      lo = min(lo, code);
      top = max(top, code);
    }
  });
  lo = __reduce_min_sync(Full, lo);
  top = __reduce_max_sync(Full, top);
  unsigned rank = K;
#pragma unroll 1
  while (top > lo) {
    const unsigned spread = (top - lo) / kBins;
    const unsigned shift = spread ? 32 - __clz(spread) : 0;
    for (unsigned b = lane; b < kBins; b += 32) hist[b] = 0;
    __syncwarp();
    const unsigned base = lo, t = top;
    pass([&](unsigned code, int32_t, bool valid) {
      if (valid && code >= base && code <= t)
        atomicAdd(hist + ((code - base) >> shift), 1u);
    });
    __syncwarp();
    unsigned above;
    const unsigned chosen = find_bin(hist, rank, above);
    rank -= above;
    lo += chosen << shift;
    top = min(lo + (1u << shift) - 1, top);
    __syncwarp();
  }
  const unsigned th = lo;
  unsigned pg = 0, pt = K - rank;
  pass([&](unsigned code, int32_t key, bool valid) {
    const bool g = valid && code > th, tq = valid && code == th;
    const unsigned mg = __ballot_sync(Full, g), mt = __ballot_sync(Full, tq);
    const unsigned ag = pg + __popc(mg & lt), at = pt + __popc(mt & lt);
    if (g) out[ag] = key;
    if (tq && at < K) out[at] = key;
    pg += __popc(mg);
    pt += __popc(mt);
  });
}
}  // namespace row_select
