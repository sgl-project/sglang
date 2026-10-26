// SPDX-License-Identifier: MIT
// Persistent fused FP8 MQA logits + per-row top-K (K = 2048; SM100),
// DeepGEMM/FA4-style: one CTA per SM, barriers and TMEM initialised once, every
// warp role runs its own loop over blocks claimed (heaviest first) from a
// global counter, and the sample and scan tiles of a block flow through ONE
// continuous Q/KV/TMEM pipeline whose stage/phase state is carried across
// blocks.
//
// Roles (384 threads):
//   warp 8   claimer + Q/weights TMA + drain of math WG1 rows
//   warp 9   KV TMA (sample tiles via the strided map, then the scan tiles)
//   warp 10  UMMA issuer
//   warp 11  linear-term rows + drain of math WG0 rows
//   warps 0-7 math: sample histogram -> gate -> scan/emit -> selection
// A block whose candidate count leaves [K, 16384] pauses the role loops; the
// CTA rescans it exactly (fallback.cuh) and resumes the loops with freshly
// initialised barriers.
// Only the global candidate slab path (KVS = 3) is vendored; PR #60618's
// resident kernel and shared-memory candidate path (KVS = 2) are not.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cuda_fp8.h>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/arch/cluster_sm90.hpp>
#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/ring_pipeline.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

#include "candidates.cuh"
#include "deep_gemm_compat.cuh"
#include "fallback.cuh"
#include "fast_select.cuh"
#include "output_map.cuh"
#include "row_select.cuh"
#include "sample.cuh"

namespace fused_persist {
using namespace deep_gemm;
using TBar = cutlass::arch::ClusterTransactionBarrier;

constexpr unsigned kTerm = 0xffffffffu, kPause = 0xfffffffeu;
constexpr unsigned kQBytes = 8 * 32 * 128, kKVBytes = 256 * 128,
                   kWBytes = 8 * 32 * 4;

template <unsigned KVS>
struct PStorage;
// Rebalanced scan: the 16 linear-term rows of each math warpgroup bracket Q,
// so WG0's N=144 UMMA reads B = [lin WG0 | Q rows 0..127] (D = [lin | heads])
// and WG1's reads B = [Q rows 128..255 | lin WG1] (D = [heads | lin]).
template <>
struct PStorage<3> {
  alignas(1024) uint8_t lin_lo[16 * 128];
  cutlass::float_e4m3_t q[kQBytes];
  uint8_t lin_hi[16 * 128];
  alignas(128) float weights[8 * 32];
  float lin_scale[8];
  alignas(1024) cutlass::float_e4m3_t kv[3][kKVBytes];
  alignas(128) float sf_kv[3][256];
};
static_assert(offsetof(PStorage<3>, q) == 16 * 128 &&
                  offsetof(PStorage<3>, lin_hi) == 16 * 128 + kQBytes,
              "lin rows must bracket Q");
// Selection scratch starts after lin + Q + lin + weights + row scales (the
// next block's linear term is written during this block's selection).
constexpr unsigned kSelBase3 = offsetof(PStorage<3>, kv);

// q'_r = sum_h (w_rh / 2) q_rh, split into 4 FP8 e4m3 terms: terms 0,1 scale
// s_r, terms 2,3 s_r * 2^-8 (lin_scale[r]); term t of row r is B row (r % 4) *
// 4 + t of lin_lo (rows 0-3) / lin_hi (4-7), 128B-swizzled K-major like Q. One
// warp; lane l owns dims 4l..4l+3.
__device__ __forceinline__ void compute_lin_rows(PStorage<3>& sm,
                                                 unsigned lane) {
  const uint8_t* qbytes = reinterpret_cast<const uint8_t*>(sm.q);
#pragma unroll 1
  for (unsigned r = 0; r < 8; ++r) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
    for (unsigned h = 0; h < 32; ++h) {
      const unsigned R = r * 32 + h;
      const unsigned off = R * 128u + ((((lane * 4u) >> 4) ^ (R & 7u)) << 4) +
                           ((lane * 4u) & 15u);
      const uint32_t packed =
          ptx::ld_shared(reinterpret_cast<const uint32_t*>(qbytes + off));
      const float wv = ptx::ld_shared(sm.weights + r * 32 + h) * 0.5f;
      const __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2(
          static_cast<__nv_fp8x2_storage_t>(packed & 0xffffu), __NV_E4M3);
      const __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2(
          static_cast<__nv_fp8x2_storage_t>(packed >> 16), __NV_E4M3);
      const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&lo));
      const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&hi));
      acc[0] = fmaf(wv, f0.x, acc[0]);
      acc[1] = fmaf(wv, f0.y, acc[1]);
      acc[2] = fmaf(wv, f1.x, acc[2]);
      acc[3] = fmaf(wv, f1.y, acc[3]);
    }
    float m = fmaxf(fmaxf(fabsf(acc[0]), fabsf(acc[1])),
                    fmaxf(fabsf(acc[2]), fabsf(acc[3])));
#pragma unroll
    for (unsigned o = 16; o > 0; o >>= 1)
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    int e = -120;  // pow2 scale s >= m / 448, clamped so 1/s stays finite
    if (m > 0.f && isfinite(m)) {
      frexpf(m * (1.0f / 448.0f), &e);
      e = max(e, -120);
    }
    const float s = ldexpf(1.0f, e), inv_s = ldexpf(1.0f, -e);
    float res[4];
#pragma unroll
    for (unsigned dd = 0; dd < 4; ++dd) res[dd] = acc[dd] * inv_s;
    uint8_t* lin_base = r < 4 ? sm.lin_lo : sm.lin_hi;
#pragma unroll
    for (unsigned tt = 0; tt < 4; ++tt) {
      const __nv_fp8x2_storage_t c01 = __nv_cvt_float2_to_fp8x2(
          make_float2(res[0], res[1]), __NV_SATFINITE, __NV_E4M3);
      const __nv_fp8x2_storage_t c23 = __nv_cvt_float2_to_fp8x2(
          make_float2(res[2], res[3]), __NV_SATFINITE, __NV_E4M3);
      const __half2_raw d01 = __nv_cvt_fp8x2_to_halfraw2(c01, __NV_E4M3);
      const __half2_raw d23 = __nv_cvt_fp8x2_to_halfraw2(c23, __NV_E4M3);
      const float2 g01 =
          __half22float2(*reinterpret_cast<const __half2*>(&d01));
      const float2 g23 =
          __half22float2(*reinterpret_cast<const __half2*>(&d23));
      res[0] -= g01.x;
      res[1] -= g01.y;
      res[2] -= g23.x;
      res[3] -= g23.y;
      // Terms 2, 3 in units of s * 2^-8: the residual after two terms is <=
      // 0.5, so x256 <= 128 (no saturation), and the e4m3 subnormal floor drops
      // from s * 2^-10 to s * 2^-18 per dim.
      if (tt == 1) {
        res[0] *= 256.0f;
        res[1] *= 256.0f;
        res[2] *= 256.0f;
        res[3] *= 256.0f;
      }
      const unsigned n = (r % 4u) * 4u + tt;
      const unsigned off = n * 128u + ((((lane * 4u) >> 4) ^ (n & 7u)) << 4) +
                           ((lane * 4u) & 15u);
      ptx::st_shared(
          reinterpret_cast<uint32_t*>(lin_base + off),
          static_cast<uint32_t>(c01) | (static_cast<uint32_t>(c23) << 16));
    }
    if (lane == 0) sm.lin_scale[r] = s;
  }
}

// The only top-k this build instantiates (fused_persist::kTopK below).
constexpr unsigned kShortKeys = 2048;
// Gate: the k-th largest sample (k = count * target / visible, count = the
// histogram total) of one row from its 4096-bin histogram (bins of the sample
// minus the row reference `ref`, 16-bit counters two per word); one warp per
// row. The gate is ref + the lower edge of the bin holding the k-th sample (a
// lower bound of it). -inf when the row sees at most `target` keys or has no
// samples; +inf when it sees at most top-k keys.
__device__ __forceinline__ unsigned sample_pair_total(unsigned v) {
  return (v & 0xffffu) + (v >> 16);
}
__device__ __noinline__ void finish_sample_histogram(const unsigned* hist,
                                                     float* gate,
                                                     unsigned visible,
                                                     unsigned target, float ref,
                                                     unsigned lane) {
  if (visible <= target) {
    // Rows that see at most top-k keys (padding rows past the end included)
    // are written directly (write_short) and need no candidates: a +inf gate
    // keeps them out of the any-pass vote.
    if (lane == 0) *gate = visible <= kShortKeys ? CUDART_INF_F : -CUDART_INF_F;
    __syncwarp();
    return;
  }
  // Lane l owns the 64 words (128 bins) of chunks 63-2l and 62-2l (bins
  // descending with the lane).
  const unsigned* const h0 = hist + (63 - 2 * lane) * 33;
  const unsigned* const h1 = hist + (62 - 2 * lane) * 33;
  unsigned sum = 0;
#pragma unroll 8
  for (unsigned j = 0; j < 32; ++j)
    sum += sample_pair_total(h0[j]) + sample_pair_total(h1[j]);
  unsigned incl = sum;
#pragma unroll
  for (unsigned dd = 1; dd < 32; dd *= 2) {
    const unsigned u = __shfl_up_sync(0xffffffffu, incl, dd);
    if (lane >= dd) incl += u;
  }
  const unsigned tot = __shfl_sync(0xffffffffu, incl, 31);
  if (tot == 0) {
    if (lane == 0) *gate = -CUDART_INF_F;
    __syncwarp();
    return;
  }
  const unsigned rank =
      max(1u, min(tot, (tot * target + visible - 1) / visible));
  const unsigned w = __ffs(__ballot_sync(0xffffffffu, incl >= rank)) - 1;
  unsigned above = __shfl_sync(0xffffffffu, incl - sum, w), chosen = 0;
#pragma unroll 1
  for (unsigned cc = 0; cc < 2; ++cc) {
    const unsigned c = 63 - 2 * w - cc;
    // lane 0 = highest word of the chunk; its high half is the higher bin
    const unsigned word = hist[c * 33 + 31 - lane];
    const unsigned total = sample_pair_total(word);
    unsigned v = total;
#pragma unroll
    for (unsigned dd = 1; dd < 32; dd *= 2) {
      const unsigned u = __shfl_up_sync(0xffffffffu, v, dd);
      if (lane >= dd) v += u;
    }
    const unsigned mask = __ballot_sync(0xffffffffu, above + v >= rank);
    if (mask) {
      const unsigned sel = __ffs(mask) - 1;
      const unsigned before = above + __shfl_sync(0xffffffffu, v - total, sel),
                     high = __shfl_sync(0xffffffffu, word >> 16, sel);
      chosen = 2 * (c * 32 + 31 - sel) + (before + high >= rank ? 1u : 0u);
      break;
    }
    above += __shfl_sync(0xffffffffu, v, 31);
  }
  if (lane == 0) *gate = ref + sample_floor(chosen);
  __syncwarp();
}

// Candidate staging: per (bank, math warp, row) a compacted list of the
// interval's passing keys. Record = bits 8-31 of the fp32 score (in place) |
// (half index in the interval) << 5 | lane, so an interval spans at most
// kStageCap / 64 tiles and a list can never overflow (64 keys per warp-tile).
// cnt: the lists' end shared addresses at the handoff.
constexpr unsigned kStageCap = 256;
struct alignas(16) PStageData {
  uint32_t recs[2][8][4][kStageCap];
  uint32_t cnt[2][8][4];
  uint32_t bases[2][8];
};

// Ordered fp32 code (candidate_ordered_fp32_code) of a staged record's score
// bits, in bits 8-31; bits 0-7 of the result are not the record's rel.
__device__ __forceinline__ uint32_t staged_code(uint32_t record) {
  return record ^ (static_cast<uint32_t>(static_cast<int32_t>(record) >> 31) |
                   0x80000000u);
}

// samples: gate sample tiles of the block (0 when no row sees more keys than
// the gate target, so every gate is -inf).
// out_base: output index of each row's first visible key (rows that see at
// most top-k keys output all of them).
struct ClaimEntry {
  unsigned vblock, all_short, kv_base, splits, slot, samples;
  unsigned starts[8], ends[8];
  int out_base[8];
};

// Barrier words (static shared, never aliased with data).
template <unsigned KVS>
struct BarIdx {
  // 3 per-warpgroup TMEM slots + the linear-term barrier.
  static constexpr unsigned NT = 3;
  static constexpr unsigned full_q = 0, empty_q = 1, full_kv = 2,
                            empty_kv = 2 + KVS, full_tmem = 2 + 2 * KVS,
                            empty_tmem = full_tmem + NT,
                            claim_full = empty_tmem + NT,
                            claim_empty = claim_full + 2,
                            sel_done = claim_empty + 2, h_full = sel_done + 1,
                            h_empty = h_full + 16, full_lin = h_empty + 16,
                            total = full_lin + 1;
};

__device__ __forceinline__ void mbar_inval(const void* b) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(
                   static_cast<uint32_t>(__cvta_generic_to_shared(b)))
               : "memory");
}

// Expect `bytes` of TMA transactions and arrive `count` times on one barrier.
__device__ __forceinline__ void mbar_arrive_expect_tx(const void* b,
                                                      uint32_t bytes,
                                                      uint32_t count) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(b));
  asm volatile(
      "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(a),
      "r"(bytes)
      : "memory");
  asm volatile(
      "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(a),
      "r"(count)
      : "memory");
}

// Volatile shared load of a value the epoch's barrier init stored: the scan
// loop keeps the shared addresses it loads this way in registers, where ptxas
// would otherwise rematerialize them from the shared window base (S2UR
// SR_CgaCtaId + ULEA) in every tile and the barrier probes would wait on it.
__device__ __forceinline__ uint32_t opaque_ld_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.volatile.shared.u32 %0, [%1];"
               : "=r"(v)
               : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p)))
               : "memory");
  return v;
}
// = ClusterBarrier::wait on a shared address
__device__ __forceinline__ void mbar_wait_addr(uint32_t addr, uint32_t phase) {
  asm volatile(
      "{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n\t"
      "@P1 bra DONE;\n\tbra LAB_WAIT;\n\tDONE:\n\t}"
      :
      : "r"(addr), "r"(phase), "r"(0x989680u)
      : "memory");
}
// = ClusterBarrier::test_wait on a shared address (non-blocking; acquire on success)
__device__ __forceinline__ bool mbar_test_addr(uint32_t addr, uint32_t phase) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred P1;\n\t"
      "mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
      "selp.b32 %0, 1, 0, P1;\n\t}"
      : "=r"(ok)
      : "r"(addr), "r"(phase)
      : "memory");
  return ok != 0;
}
// = ClusterBarrier::arrive on a shared address
__device__ __forceinline__ void mbar_arrive_addr(uint32_t addr) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(addr)
               : "memory");
}
// = ptx::ld_shared(const float*) on a shared address
__device__ __forceinline__ float ld_shared_f32_addr(uint32_t addr) {
  float v;
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(addr));
  return v;
}

// st.shared.u32 under a predicate (keeps the emission branch-free).
__device__ __forceinline__ void st_shared_if(bool p, uint32_t addr,
                                             uint32_t v) {
  asm volatile(
      "{\n\t.reg .pred q;\n\tsetp.ne.u32 q, %0, 0;\n\t"
      "@q st.shared.u32 [%1], %2;\n\t}" ::"r"(uint32_t(p)),
      "r"(addr), "r"(v)
      : "memory");
}

// Global slab position of a staged record: key = warp_kv (kv_base + 32 * (warp
// % 4)) + (interval start * 2 + half index) * 128 + lane.
__device__ __forceinline__ void store_staged(uint16_t* __restrict__ vals,
                                             int32_t* __restrict__ indices,
                                             uint64_t off, uint32_t record,
                                             uint32_t warp_kv,
                                             uint32_t flush_base) {
  const uint32_t physical = warp_kv +
                            (flush_base * 2 + ((record & 255u) >> 5)) * 128 +
                            (record & 31u);
  fused_candidates::store_candidate_record(vals + off, indices + off,
                                           staged_code(record) >> 8, physical);
}

// Self-drain, out of line: stores this warp's staged records of the last
// handoff interval (recs = recs[bank][warp], [4][kStageCap]; n0..n3 are
// warp-uniform): one shared-memory atomic per row (cnt_s: the warpgroup's 4
// per-row candidate counters), coalesced stores.
__device__ __noinline__ void self_drain_last(
    const uint32_t* __restrict__ recs, unsigned n0, unsigned n1, unsigned n2,
    unsigned n3, unsigned* cnt_s, unsigned rbase, uint32_t warp_kv,
    unsigned flush_base, int64_t row_delta, uint16_t* __restrict__ vals,
    int32_t* __restrict__ indices) {
  constexpr uint32_t kCap = 16384;
  const unsigned lane = threadIdx.x & 31;
  const unsigned n =
      lane == 0 ? n0
                : (lane == 1 ? n1 : (lane == 2 ? n2 : (lane == 3 ? n3 : 0u)));
  unsigned start = 0;
  if (n) start = atomicAdd(cnt_s + lane, n);
#pragma unroll 1
  for (unsigned row = 0; row < 4; ++row) {
    const unsigned nr = __shfl_sync(0xffffffffu, n, row),
                   sr = __shfl_sync(0xffffffffu, start, row);
    for (unsigned j = lane; j < nr; j += 32) {
      const unsigned slot2 = sr + j;
      if (slot2 < kCap)
        store_staged(
            vals, indices,
            static_cast<uint64_t>(int64_t(rbase + row) + row_delta) * kCap +
                slot2,
            recs[row * kStageCap + j], warp_kv, flush_base);
    }
  }
  __syncwarp();
}

// Drain of one interval of the 4 math warps p0..p0+3 (one warpgroup, rows
// row_base..row_base+3; slab row of row x: x + row_delta; kv_base: the block's
// first key). Per row one shared-memory atomic on the warpgroup's per-row
// counter (totals[0..3]) reserves one contiguous slab range for the row's 4
// lists (warp order). Lanes 8r..8r+7 then store row r's records, 8 consecutive
// slots per step: a lane finds its list with 3 compares against its row's
// list ends and selects of per-list constants (branch-free; no shuffle or
// atomic inside the steps; one loop for the 4 rows). skip: the last interval,
// self-drained by the warps.
__device__ __forceinline__ void ho_drain4(
    PStageData& d, TBar* hfull, TBar* hempty, unsigned p0, unsigned epoch,
    unsigned row_base, unsigned* totals, unsigned cap, bool skip,
    uint16_t* __restrict__ vals, int32_t* __restrict__ indices,
    int64_t row_delta, unsigned kv_base) {
  const unsigned lane = threadIdx.x & 31, bank = epoch & 1;
#pragma unroll
  for (unsigned p = 0; p < 4; ++p)
    hfull[bank * 8 + p0 + p].wait((epoch >> 1) & 1);
  __syncwarp();
  if (!skip) {
    // lane l < 16: list (warp p0 + l % 4, row l / 4) of n records (its end
    // address minus its base); incl: inclusive prefix of the counts within the
    // row (lane 4r + 3: the row total).
    const unsigned n =
        lane < 16 ? (d.cnt[bank][p0 + lane % 4][lane / 4] -
                     static_cast<uint32_t>(__cvta_generic_to_shared(
                         &d.recs[bank][p0 + lane % 4][lane / 4][0]))) /
                        4u
                  : 0u;
    unsigned incl = n;
    {
      unsigned u = __shfl_up_sync(0xffffffffu, incl, 1);
      if (lane & 3) incl += u;
      u = __shfl_up_sync(0xffffffffu, incl, 2);
      if (lane & 2) incl += u;
    }
    unsigned rstart = 0;
    if ((lane & 3) == 3 && lane < 16 && incl)
      rstart = atomicAdd(totals + lane / 4, incl);
    // Row-local index h of list (p, r) -> record recs[h + rdelta]
    // (recs = recs[bank][p0], lists at (p * 4 + r) * kStageCap).
    const unsigned rdelta =
        ((lane % 4) * 4 + lane / 4) * kStageCap - (incl - n);
    const unsigned r = lane / 8, src = 4 * r;  // this lane's row
    const unsigned e1 = __shfl_sync(0xffffffffu, incl, src),
                   e2 = __shfl_sync(0xffffffffu, incl, src + 1),
                   e3 = __shfl_sync(0xffffffffu, incl, src + 2),
                   nr = __shfl_sync(0xffffffffu, incl, src + 3),
                   sr = __shfl_sync(0xffffffffu, rstart, src + 3),
                   d0 = __shfl_sync(0xffffffffu, rdelta, src),
                   d1 = __shfl_sync(0xffffffffu, rdelta, src + 1),
                   d2 = __shfl_sync(0xffffffffu, rdelta, src + 2),
                   d3 = __shfl_sync(0xffffffffu, rdelta, src + 3);
    const unsigned lim = min(nr, cap - min(sr, cap));  // slots < cap
    const unsigned steps = __reduce_max_sync(0xffffffffu, lim);
    // Key of a record of warp p0 + p: kb_p + (half << 7) + lane, kb_p =
    // kv_base + 32 p + 256 * (the interval's first tile).
    const unsigned kb0 = kv_base + d.bases[bank][p0] * 256u;
    const uint32_t* const recs = &d.recs[bank][p0][0][0];
    const uint64_t roff =
        static_cast<uint64_t>(int64_t(row_base + r) + row_delta) * cap + sr;
    // 64-bit slab pointers of this lane's row (one IMAD.WIDE per record and
    // array).
    const uint64_t vrow = reinterpret_cast<uint64_t>(vals + roff);
    const uint64_t irow = reinterpret_cast<uint64_t>(indices + roff);
    // Two records (h0, h0 + 8) per iteration, both loaded before either is
    // stored: the predicated asm stores are compiler barriers, so an unrolled
    // loop cannot move a record's load above the previous record's stores
    // (blocks whose keys all pass the gate are drain-bound).
#pragma unroll 1
    for (unsigned h0 = lane & 7; h0 < steps; h0 += 16) {
      uint32_t rec[2], kb[2], hc[2], ok[2];
#pragma unroll
      for (unsigned u = 0; u < 2; ++u) {
        // Branch-free: out-of-range lanes compute on a clamped index and do
        // not store (predicated st.global.cg).
        const unsigned h = h0 + 8u * u;
        ok[u] = h < lim;
        hc[u] = ok[u] ? h : 0u;
        const bool b1 = hc[u] >= e1, b2 = hc[u] >= e2, b3 = hc[u] >= e3;
        const unsigned dd = b3 ? d3 : (b2 ? d2 : (b1 ? d1 : d0));
        kb[u] = kb0 + (b3 ? 96u : (b2 ? 64u : (b1 ? 32u : 0u)));
        rec[u] = recs[hc[u] + dd];
      }
#pragma unroll
      for (unsigned u = 0; u < 2; ++u) {
        const uint32_t code = staged_code(rec[u]);
        // = store_staged: value = code bits 0-15, index = key (< 2^20) | code
        // bits 16-23 << 20.
        const uint32_t key = kb[u] + ((rec[u] << 2) & 0x380u) + (rec[u] & 31u);
        const uint32_t idx = key | ((code >> 4) & 0x0ff00000u);
        asm volatile(
            "{\n\t.reg .pred q;\n\tsetp.ne.u32 q, %0, 0;\n\t"
            "@q st.global.cg.u16 [%1], %2;\n\t"
            "@q st.global.cg.u32 [%3], %4;\n\t}" ::"r"(ok[u]),
            "l"(vrow + uint64_t(hc[u]) * 2u), "h"(static_cast<uint16_t>(code >> 8)),
            "l"(irow + uint64_t(hc[u]) * 4u), "r"(idx)
            : "memory");
      }
    }
  }
  __syncwarp();
#pragma unroll
  for (unsigned p = 0; p < 4; ++p) hempty[bank * 8 + p0 + p].arrive();
}

template <unsigned Interval, unsigned KVS, unsigned TopK>
__global__ __launch_bounds__(384, 1) void fused_mqa_topk_fp8_prefill_kernel(
    unsigned rows, unsigned keys, const unsigned* starts_g,
    const unsigned* ends_g, uint16_t* vals, int* indices, int* counts,
    FallbackBuffers fallback, unsigned phase_bytes, unsigned sample_target,
    unsigned sample_stride, unsigned sample_groups, unsigned sample_tiles,
    const float* scales, const __grid_constant__ cute::TmaDescriptor tks,
    const __grid_constant__ cute::TmaDescriptor tq,
    const __grid_constant__ cute::TmaDescriptor ts,
    const __grid_constant__ cute::TmaDescriptor tk,
    const __grid_constant__ cute::TmaDescriptor tw, bool relative,
    const int* out_offsets, unsigned nblocks, unsigned* counter,
    int* defer_next, PageTable paged) {
  // The exact rescan (dense_scan.cuh) expects this 224 / 56 register split.
  constexpr uint32_t kMathRegs = 224, kSpecRegs = 168 * 3 - 2 * kMathRegs;
  constexpr uint32_t kCandGlobalCap = 16384;
  using St = PStorage<KVS>;
  using B = BarIdx<KVS>;
  // Compile-time: a runtime top-k costs ~0.5% in the scan (codegen).
  constexpr unsigned topk = TopK;
  static_assert(topk == kShortKeys, "short-row gate assumes the build top-k");
  // Rebalanced scan: linear term on the tensor core, per-warpgroup N=144 MMAs.
  static_assert(KVS == 3, "rebalanced scan needs the 3-KV-stage lin layout");
  constexpr uint32_t kSlotCols =
      160;  // 4 rows x 32 heads + 16 linear-term columns
  constexpr unsigned kSelB = kSelBase3;
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  auto& sm = *reinterpret_cast<St*>(smem_buffer);
  uint8_t* const region = smem_buffer + sizeof(St);
  auto& stream = *reinterpret_cast<SampleShared*>(region);
  auto& sdata = *reinterpret_cast<PStageData*>(region);
  static_assert(Interval * 64 <= kStageCap,
                "an interval must fit the staging lists");

  __shared__ alignas(8) uint64_t bar_words[B::total];
  __shared__ uint32_t tmem_ptr;
  __shared__ ClaimEntry entries[2];
  __shared__ float gate_s[8];
  __shared__ unsigned row_count[8];
  __shared__ unsigned sel_recover, sel_short;
  __shared__ unsigned defer_head, unsafe_s, pause_req, epoch_kind;
  __shared__ uint32_t opaque_bases[3];  // scan: see opaque_ld_u32
  __shared__ uint32_t lane_tab[2][32];  // emission: lane id, lanemask_lt

  TBar* const bars = reinterpret_cast<TBar*>(bar_words);
  TBar& full_q = bars[B::full_q];
  TBar& empty_q = bars[B::empty_q];
  TBar* const full_kv = bars + B::full_kv;
  // The 32 producer lanes stage the sample scales, so each arrives on full_kv.
  constexpr unsigned kFullKvArrivals = 32;
  TBar* const empty_kv = bars + B::empty_kv;
  TBar* const full_tmem = bars + B::full_tmem;
  TBar* const empty_tmem = bars + B::empty_tmem;
  TBar* const claim_full = bars + B::claim_full;
  TBar* const claim_empty = bars + B::claim_empty;
  TBar& sel_done = bars[B::sel_done];
  TBar* const h_full = bars + B::h_full;
  TBar* const h_empty = bars + B::h_empty;
  TBar* const full_lin = bars + B::full_lin;

  const unsigned warp_idx = cutlass::canonical_warp_idx_sync();
  const unsigned lane = ptx::get_lane_idx();
  constexpr unsigned kClaimConsumers = 11;  // warps 9, 10, 11, 0-7

  if (warp_idx == 8 && lane == 0) {
    cute::prefetch_tma_descriptor(&tq);
    cute::prefetch_tma_descriptor(&tw);
    cute::prefetch_tma_descriptor(&tk);
    cute::prefetch_tma_descriptor(&ts);
    cute::prefetch_tma_descriptor(&tks);
  }
  if (warp_idx == 9 && cute::elect_one_sync()) {
    defer_head = kTerm;
    pause_req = 0;
  }
  // Epochs: the persistent role loops run until the work runs out (kTerm) or a
  // block needs the exact rescan (kPause); rare blocks are then rescanned with
  // the fallback core and the loops resume with freshly initialised barriers.
  // The first epoch is emitted outside any loop (nesting the role loops in an
  // epoch loop costs ~1.5 percent in the scan: uniform address math is
  // rematerialised every tile). Later epochs only run after a recovery pause.
  __shared__ unsigned row_bounds[2][8];
  __shared__ unsigned fslot;
  bool more = false;
  {
    constexpr unsigned epoch = 0;
#include "persistent_epoch.inc"
  }
#pragma unroll 1
  for (unsigned epoch = 1; more; ++epoch) {
#include "persistent_epoch.inc"
  }
  // counter = {next block, finished CTAs}: the last CTA to finish (every claim
  // is done by then) zeroes both, so the caller's workspace is ready for the
  // next launch / graph replay.
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(counter + 1, 1u) == gridDim.x - 1) {
      atomicExch(counter, 0u);
      atomicExch(counter + 1, 0u);
    }
  }
}

// ---------------- host: launch configuration ----------------
// The only top-k this build instantiates (GLM-5.2's indexer).
constexpr unsigned kTopK = 2048;
using PersistentKernel =
    decltype(&fused_mqa_topk_fp8_prefill_kernel<4, 3, kTopK>);

// Fewest keys launched: the gates below are tuned and tested above 20480 keys
// only (PR #60618's resident and shared-memory paths are not vendored).
constexpr unsigned kMinKeys = 20481;
// Global slab: handoff interval of 4 tiles (the staging lists hold one
// interval's worst case).
inline PersistentKernel persistent_kernel() {
  return fused_mqa_topk_fp8_prefill_kernel<4, 3, kTopK>;
}

namespace detail {
// cudaFuncGetAttributes / cudaFuncSetAttribute cost microseconds per call and
// their results change only with the kernel (and, for the shared-memory
// opt-in, the device): both are cached per process. Thread-safe; failed calls
// are not cached.
struct AttributeCache {
  struct Entry {
    int static_smem = -1;  // cudaFuncAttributes::sharedSizeBytes (-1: unknown)
    int opt_in = 0;        // dynamic shared memory bytes opted in to
    uint64_t devices = 0;  // bit d: opted in on device d (< 64; others retry)
  };
  std::mutex mutex;
  std::unordered_map<const void*, Entry> entries;
};
inline AttributeCache& attribute_cache() {
  static AttributeCache cache;
  return cache;
}
}  // namespace detail

// Static shared memory bytes of `kernel`.
inline cudaError_t static_smem(const void* kernel, int& bytes) {
  auto& cache = detail::attribute_cache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  auto& entry = cache.entries[kernel];
  if (entry.static_smem < 0) {
    cudaFuncAttributes attr;
    if (const cudaError_t e = cudaFuncGetAttributes(&attr, kernel);
        e != cudaSuccess)
      return e;
    entry.static_smem = int(attr.sharedSizeBytes);
  }
  bytes = entry.static_smem;
  return cudaSuccess;
}

// Opts `kernel` in to `bytes` of dynamic shared memory on the current device
// (the attribute is per function and device).
inline cudaError_t opt_in_smem(const void* kernel, int bytes) {
  int device = 0;
  if (const cudaError_t e = cudaGetDevice(&device); e != cudaSuccess) return e;
  const uint64_t bit = device < 64 ? uint64_t(1) << device : 0;
  auto& cache = detail::attribute_cache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  auto& entry = cache.entries[kernel];
  if (entry.opt_in == bytes && (entry.devices & bit)) return cudaSuccess;
  if (const cudaError_t e = cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
      e != cudaSuccess)
    return e;
  if (entry.opt_in != bytes) entry.devices = 0;
  entry.opt_in = bytes;
  entry.devices |= bit;
  return cudaSuccess;
}

struct LaunchConfig {
  PersistentKernel kernel;
  int smem;              // dynamic shared memory bytes
  unsigned phase_bytes;  // selection scratch bytes past the Q / weights region
  // Gate sample: sample_tiles tiles of 256 keys on a uniform grid (stride
  // sample_stride).
  unsigned sample_target, sample_stride, sample_groups, sample_tiles;
};

// Sizes the kernel's shared memory (setting the opt-in limit on the current
// device, cached) and the gate sample for `keys` keys (top-kTopK rows).
inline cudaError_t launch_config(unsigned keys, LaunchConfig& c) {
  if (keys < kMinKeys) return cudaErrorInvalidValue;
  // The gate aims for `target` candidates per row and sizes the sample so its
  // quantile rank is `rank` (count error ~1/sqrt(rank)); a row whose count
  // falls below the top-k is rescanned exactly.
  // Above 128K the sample is clamped to 16-48 tiles (rank ~48 at 1M keys): a
  // sample tile costs about two scan tiles. A rescan runs as the last work of
  // its CTA (one 8-row block at 1M keys costs ~1/3 of the call), so the slab
  // gates were chosen on B200 for no rescan over GLM-5.2 layers 0-14 (real
  // dumps, 8192 rows, and layer-0 captures, 16384 rows causal / full) at the
  // smallest worst-layer cost.
  struct Gate {
    unsigned target, rank;
  };
  constexpr Gate kGates[4] = {
      // <= 32K     <= 64K      <= 128K     > 128K
      {2816, 128}, {3328, 64}, {3584, 64}, {4096, 64}};
  const bool long_ctx = keys > 131072;
  const Gate gate = kGates[keys <= 32768   ? 0
                           : keys <= 65536 ? 1
                           : !long_ctx     ? 2
                                           : 3];
  c.sample_target = gate.target;
  const unsigned rank = gate.rank;
  const unsigned rank_tiles = std::max(
      1u, unsigned((uint64_t(rank) * keys + 256ull * c.sample_target - 1) /
                   (256ull * c.sample_target)));
  constexpr unsigned max_tiles = 48;
  c.sample_tiles =
      long_ctx ? std::min(max_tiles, std::max(16u, rank_tiles)) : rank_tiles;
  // 16-bit sample histogram counters: at most 255 * 256 samples per row (the
  // gates above need <= 48 tiles).
  c.sample_tiles = std::min(c.sample_tiles, 255u);
  c.sample_stride = std::max(
      1u, (keys + 256u * c.sample_tiles - 1) / (256u * c.sample_tiles));
  c.sample_groups =
      (keys - c.sample_stride / 2 + c.sample_stride - 1) / c.sample_stride;
  c.kernel = persistent_kernel();

  // The recovery rescan reuses the same dynamic shared memory with the fallback
  // core's layout.
  constexpr int kFallbackSmem =
      int(sizeof(fused_mqa_dense::SharedStorage) + sizeof(FallbackShared));
  constexpr int kLongSmem = int(
      sizeof(PStorage<3>) + std::max(sizeof(PStageData), sizeof(SampleShared)));
  // selection scratch past Q + weights (checked below at run time)
  constexpr int kStreamSmem =
      int(kSelBase3) + int(8 * row_select::kStreamWords * 4);
  constexpr int kMaxSmem = 232448;  // sm_100 opt-in limit per block
  // Everything up to the opt-in limit.
  int static_bytes = 0;
  if (const cudaError_t e =
          static_smem(reinterpret_cast<const void*>(c.kernel), static_bytes);
      e != cudaSuccess)
    return e;
  c.smem = std::max(kFallbackSmem, kMaxSmem - static_bytes);
  if (c.smem < std::max(kLongSmem, kStreamSmem))
    return cudaErrorInvalidConfiguration;
  c.phase_bytes = unsigned(c.smem) - kSelBase3;
  return opt_in_smem(reinterpret_cast<const void*>(c.kernel), c.smem);
}

}  // namespace fused_persist
