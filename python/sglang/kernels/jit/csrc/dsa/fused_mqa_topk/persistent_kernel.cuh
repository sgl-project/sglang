// SPDX-License-Identifier: MIT
// Persistent fused FP8 MQA logits + per-row top-k (SM100), DeepGEMM/FA4-style:
// one CTA per SM, barriers and TMEM initialised once, every warp role runs its
// own loop over blocks claimed (heaviest first) from a global counter, and the
// sample and scan tiles of a block flow through ONE continuous Q/KV/TMEM
// pipeline whose stage/phase state is carried across blocks.
//
// Roles (384 threads):
//   warp 8   claimer + Q/weights TMA + drain of math WG1 rows
//   warp 9   KV TMA (sample tiles via the strided map, then the scan tiles)
//   warp 10  UMMA issuer
//   warp 11  linear-term rows + drain of math WG0 rows
//   warps 0-7 math: sample histogram -> gate -> scan/emit -> selection
// A block whose candidate count leaves [K, kCandidatesPerRow] pauses the role
// loops; the CTA rescans it exactly (rescan.cuh) and resumes the loops with
// freshly initialised barriers.
#pragma once

#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>

#include "candidates.cuh"
#include "common.cuh"
#include "output.cuh"
#include "rescan.cuh"
#include "sample.cuh"
#include "select.cuh"
#include <cstddef>
#include <cstdint>
#include <cuda_fp8.h>
#include <type_traits>

namespace sglang::fused_mqa_topk {

using rescan::RescanBuffers;

// Claim results that end the role loops: no work left / pause for a rescan.
inline constexpr uint32_t kTerm = 0xffffffffu, kPause = 0xfffffffeu;
// Bytes of a block's Q, of one KV tile and its per-key scales, of a block's
// weights.
inline constexpr uint32_t kQBytes = kBlockRows * kNumHeads * kHeadDim, kKVBytes = kTileKeys * kHeadDim,
                          kScaleBytes = kTileKeys * sizeof(float), kWBytes = kBlockRows * kNumHeads * sizeof(float);
// KV (and per-key scale) pipeline stages.
inline constexpr uint32_t kKVStages = 3;
// Linear-term B rows per math warpgroup: kWgRows query rows x 4 FP8 terms.
inline constexpr uint32_t kLinRows = kWgRows * 4;
// TMEM: per KV half, one UMMA per math warpgroup writes its 4 rows x 32 heads
// plus the kLinRows linear-term columns (N = 144) into a slot of kSlotCols.
inline constexpr uint32_t kUmmaN = kWgRows * kNumHeads + kLinRows, kSlotCols = 160;
static_assert(kSlotCols >= kUmmaN && kSlotCols % 32 == 0);
// Scan tiles per candidate handoff from the math warps to the drain warps.
inline constexpr uint32_t kInterval = 4;

// Rebalanced scan: the linear-term rows of each math warpgroup bracket Q, so
// WG0's N=144 UMMA reads B = [lin WG0 | Q rows 0..127] (D = [lin | heads]) and
// WG1's reads B = [Q rows 128..255 | lin WG1] (D = [heads | lin]).
struct KernelSmem {
  alignas(1024) uint8_t lin_lo[kLinRows * kHeadDim];
  cutlass::float_e4m3_t q[kQBytes];
  uint8_t lin_hi[kLinRows * kHeadDim];
  alignas(128) float weights[kBlockRows * kNumHeads];
  float lin_scale[kBlockRows];
  alignas(1024) cutlass::float_e4m3_t kv[kKVStages][kKVBytes];
  alignas(128) float sf_kv[kKVStages][kTileKeys];
};
static_assert(
    offsetof(KernelSmem, q) == kLinRows * kHeadDim && offsetof(KernelSmem, lin_hi) == kLinRows * kHeadDim + kQBytes,
    "lin rows must bracket Q");
// Selection scratch starts after lin + Q + lin + weights + row scales (the
// next block's linear term is written during this block's selection).
inline constexpr uint32_t kSelBase = offsetof(KernelSmem, kv);

// q'_r = sum_h (w_rh / 2) q_rh, split into 4 FP8 e4m3 terms: terms 0,1 scale
// s_r, terms 2,3 s_r * 2^-8 (lin_scale[r]); term t of row r is B row (r % 4) *
// 4 + t of lin_lo (rows 0-3) / lin_hi (4-7), 128B-swizzled K-major like Q. One
// warp; lane l owns dims 4l..4l+3.
__device__ __forceinline__ void compute_lin_rows(KernelSmem& sm, uint32_t lane) {
  const uint8_t* qbytes = reinterpret_cast<const uint8_t*>(sm.q);
#pragma unroll 1
  for (uint32_t r = 0; r < kBlockRows; ++r) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
    for (uint32_t h = 0; h < kNumHeads; ++h) {
      const uint32_t R = r * kNumHeads + h;
      const uint32_t off = R * 128u + ((((lane * 4u) >> 4) ^ (R & 7u)) << 4) + ((lane * 4u) & 15u);
      const uint32_t packed = ptx::ld_shared(reinterpret_cast<const uint32_t*>(qbytes + off));
      const float wv = ptx::ld_shared(sm.weights + r * kNumHeads + h) * 0.5f;
      const __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packed & 0xffffu), __NV_E4M3);
      const __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packed >> 16), __NV_E4M3);
      const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&lo));
      const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&hi));
      acc[0] = fmaf(wv, f0.x, acc[0]);
      acc[1] = fmaf(wv, f0.y, acc[1]);
      acc[2] = fmaf(wv, f1.x, acc[2]);
      acc[3] = fmaf(wv, f1.y, acc[3]);
    }
    float m = fmaxf(fmaxf(fabsf(acc[0]), fabsf(acc[1])), fmaxf(fabsf(acc[2]), fabsf(acc[3])));
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1)
      m = fmaxf(m, __shfl_xor_sync(kFullMask, m, o));
    int32_t e = -120;  // pow2 scale s >= m / 448, clamped so 1/s stays finite
    if (m > 0.f && isfinite(m)) {
      frexpf(m * (1.0f / 448.0f), &e);
      e = max(e, -120);
    }
    const float s = ldexpf(1.0f, e), inv_s = ldexpf(1.0f, -e);
    float res[4];
#pragma unroll
    for (uint32_t dd = 0; dd < 4; ++dd)
      res[dd] = acc[dd] * inv_s;
    uint8_t* lin_base = r < kWgRows ? sm.lin_lo : sm.lin_hi;
#pragma unroll
    for (uint32_t tt = 0; tt < 4; ++tt) {
      const __nv_fp8x2_storage_t c01 = __nv_cvt_float2_to_fp8x2(make_float2(res[0], res[1]), __NV_SATFINITE, __NV_E4M3);
      const __nv_fp8x2_storage_t c23 = __nv_cvt_float2_to_fp8x2(make_float2(res[2], res[3]), __NV_SATFINITE, __NV_E4M3);
      const __half2_raw d01 = __nv_cvt_fp8x2_to_halfraw2(c01, __NV_E4M3);
      const __half2_raw d23 = __nv_cvt_fp8x2_to_halfraw2(c23, __NV_E4M3);
      const float2 g01 = __half22float2(*reinterpret_cast<const __half2*>(&d01));
      const float2 g23 = __half22float2(*reinterpret_cast<const __half2*>(&d23));
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
      const uint32_t n = (r % 4u) * 4u + tt;
      const uint32_t off = n * 128u + ((((lane * 4u) >> 4) ^ (n & 7u)) << 4) + ((lane * 4u) & 15u);
      ptx::st_shared(
          reinterpret_cast<uint32_t*>(lin_base + off), static_cast<uint32_t>(c01) | (static_cast<uint32_t>(c23) << 16));
    }
    if (lane == 0) sm.lin_scale[r] = s;
  }
}

// Candidate staging: per (bank, math warp, row) a compacted list of the
// interval's passing keys. Record = bits 8-31 of the fp32 score (in place) |
// (half index in the interval) << 5 | lane, so an interval spans at most
// kStageCap / 64 tiles and a list can never overflow (64 keys per warp-tile).
// cnt: the lists' end shared addresses at the handoff.
inline constexpr uint32_t kStageCap = 256;
struct alignas(16) StageLists {
  uint32_t recs[2][kMathWarps][kWgRows][kStageCap];
  uint32_t cnt[2][kMathWarps][kWgRows];
  uint32_t bases[2][kMathWarps];
};

// Ordered fp32 code of a staged record's score bits (~bits if the sign is set,
// else bits ^ 0x80000000), in bits 8-31; bits 0-7 of the result are unused.
__device__ __forceinline__ uint32_t staged_code(uint32_t record) {
  return record ^ (static_cast<uint32_t>(static_cast<int32_t>(record) >> 31) | 0x80000000u);
}

// samples: gate sample tiles of the block (0 when no row sees more keys than
// the gate target, so every gate is -inf).
// out_base: output index of each row's first visible key (rows that see at
// most top-k keys output all of them).
struct ClaimHeader {
  uint32_t vblock, all_short, kv_base, splits, slot, samples;
};
struct ClaimEntry : ClaimHeader {
  uint32_t starts[kBlockRows], ends[kBlockRows];
  int32_t out_base[kBlockRows];
};

// Claim k in a producer role (warps 9-11): waits for the entry, copies its
// header and releases the entry (every lane arrives). False once the claimer
// ends the epoch (kTerm / kPause).
__device__ __forceinline__ bool
consume_claim(uint32_t k, Barrier* claim_full, Barrier* claim_empty, const ClaimEntry* entries, ClaimHeader& claim) {
  claim_full[k & 1].wait((k >> 1) & 1);
  claim = entries[k & 1];
  __syncwarp();
  if (claim.vblock >= kPause) return false;
  claim_empty[k & 1].arrive();
  return true;
}

// Barrier words (static shared, never aliased with data).
struct BarIdx {
  // TMEM slots shared by the two math warpgroups (WG0 even, WG1 odd positions).
  static constexpr uint32_t kTmemSlots = 3;
  static constexpr uint32_t full_q = 0, empty_q = 1, full_kv = 2, empty_kv = 2 + kKVStages,
                            full_tmem = 2 + 2 * kKVStages, empty_tmem = full_tmem + kTmemSlots,
                            claim_full = empty_tmem + kTmemSlots, claim_empty = claim_full + 2,
                            sel_done = claim_empty + 2, h_full = sel_done + 1, h_empty = h_full + 2 * kMathWarps,
                            full_lin = h_empty + 2 * kMathWarps, total = full_lin + 1;
};
static_assert(BarIdx::kTmemSlots * kSlotCols <= 512, "TMEM slots exceed the allocation");

// ---- PTX the CUTLASS / DeepGEMM headers do not provide ----

__device__ __forceinline__ void mbar_inval(const void* b) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(b)))
               : "memory");
}

// Expect `bytes` of TMA transactions and arrive `count` times on one barrier.
__device__ __forceinline__ void mbar_arrive_expect_tx(const void* b, uint32_t bytes, uint32_t count) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(b));
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(bytes) : "memory");
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(count) : "memory");
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
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(addr) : "memory");
}
// = ptx::ld_shared(const float*) on a shared address
__device__ __forceinline__ float ld_shared_f32_addr(uint32_t addr) {
  float v;
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(addr));
  return v;
}

// st.shared.u32 under a predicate (keeps the emission branch-free).
__device__ __forceinline__ void st_shared_if(bool p, uint32_t addr, uint32_t v) {
  asm volatile(
      "{\n\t.reg .pred q;\n\tsetp.ne.u32 q, %0, 0;\n\t"
      "@q st.shared.u32 [%1], %2;\n\t}" ::"r"(uint32_t(p)),
      "r"(addr),
      "r"(v)
      : "memory");
}

// ---- candidate drain ----

// Global slab position of a staged record: key = warp_kv (kv_base + 32 * (warp
// % 4)) + (interval start * 2 + half index) * 128 + lane.
__device__ __forceinline__ void store_staged(
    uint16_t* __restrict__ vals,
    int32_t* __restrict__ indices,
    uint64_t off,
    uint32_t record,
    uint32_t warp_kv,
    uint32_t flush_base) {
  const uint32_t physical = warp_kv + (flush_base * 2 + ((record & 255u) >> 5)) * 128 + (record & 31u);
  store_candidate_record(vals + off, indices + off, staged_code(record) >> 8, physical);
}

// Self-drain, out of line: stores this warp's staged records of the last
// handoff interval (recs = recs[bank][warp], [4][kStageCap]; n0..n3 are
// warp-uniform): one shared-memory atomic per row (cnt_s: the warpgroup's 4
// per-row candidate counters), coalesced stores.
__device__ __noinline__ void self_drain_last(
    const uint32_t* __restrict__ recs,
    uint32_t n0,
    uint32_t n1,
    uint32_t n2,
    uint32_t n3,
    uint32_t* cnt_s,
    uint32_t rbase,
    uint32_t warp_kv,
    uint32_t flush_base,
    int64_t row_delta,
    uint16_t* __restrict__ vals,
    int32_t* __restrict__ indices) {
  constexpr uint32_t kCap = kCandidatesPerRow;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t n = lane == 0 ? n0 : (lane == 1 ? n1 : (lane == 2 ? n2 : (lane == 3 ? n3 : 0u)));
  uint32_t start = 0;
  if (n) start = atomicAdd(cnt_s + lane, n);
#pragma unroll 1
  for (uint32_t row = 0; row < 4; ++row) {
    const uint32_t nr = __shfl_sync(kFullMask, n, row), sr = __shfl_sync(kFullMask, start, row);
    for (uint32_t j = lane; j < nr; j += 32) {
      const uint32_t slot2 = sr + j;
      if (slot2 < kCap)
        store_staged(
            vals,
            indices,
            static_cast<uint64_t>(int64_t(rbase + row) + row_delta) * kCap + slot2,
            recs[row * kStageCap + j],
            warp_kv,
            flush_base);
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
__device__ __forceinline__ void drain_interval(
    StageLists& d,
    Barrier* hfull,
    Barrier* hempty,
    uint32_t p0,
    uint32_t epoch,
    uint32_t row_base,
    uint32_t* totals,
    uint32_t cap,
    bool skip,
    uint16_t* __restrict__ vals,
    int32_t* __restrict__ indices,
    int64_t row_delta,
    uint32_t kv_base) {
  const uint32_t lane = threadIdx.x & 31, bank = epoch & 1;
#pragma unroll
  for (uint32_t p = 0; p < 4; ++p)
    hfull[bank * kMathWarps + p0 + p].wait((epoch >> 1) & 1);
  __syncwarp();
  if (!skip) {
    // lane l < 16: list (warp p0 + l % 4, row l / 4) of n records (its end
    // address minus its base); incl: inclusive prefix of the counts within the
    // row (lane 4r + 3: the row total).
    const uint32_t n =
        lane < 16 ? (d.cnt[bank][p0 + lane % 4][lane / 4] -
                     static_cast<uint32_t>(__cvta_generic_to_shared(&d.recs[bank][p0 + lane % 4][lane / 4][0]))) /
                        4u
                  : 0u;
    uint32_t incl = n;
    {
      uint32_t u = __shfl_up_sync(kFullMask, incl, 1);
      if (lane & 3) incl += u;
      u = __shfl_up_sync(kFullMask, incl, 2);
      if (lane & 2) incl += u;
    }
    uint32_t rstart = 0;
    if ((lane & 3) == 3 && lane < 16 && incl) rstart = atomicAdd(totals + lane / 4, incl);
    // Row-local index h of list (p, r) -> record recs[h + rdelta]
    // (recs = recs[bank][p0], lists at (p * 4 + r) * kStageCap).
    const uint32_t rdelta = ((lane % 4) * 4 + lane / 4) * kStageCap - (incl - n);
    const uint32_t r = lane / 8, src = 4 * r;  // this lane's row
    const uint32_t e1 = __shfl_sync(kFullMask, incl, src), e2 = __shfl_sync(kFullMask, incl, src + 1),
                   e3 = __shfl_sync(kFullMask, incl, src + 2), nr = __shfl_sync(kFullMask, incl, src + 3),
                   sr = __shfl_sync(kFullMask, rstart, src + 3), d0 = __shfl_sync(kFullMask, rdelta, src),
                   d1 = __shfl_sync(kFullMask, rdelta, src + 1), d2 = __shfl_sync(kFullMask, rdelta, src + 2),
                   d3 = __shfl_sync(kFullMask, rdelta, src + 3);
    const uint32_t lim = min(nr, cap - min(sr, cap));  // slots < cap
    const uint32_t steps = __reduce_max_sync(kFullMask, lim);
    // Key of a record of warp p0 + p: kb_p + (half << 7) + lane, kb_p =
    // kv_base + 32 p + 256 * (the interval's first tile).
    const uint32_t kb0 = kv_base + d.bases[bank][p0] * 256u;
    const uint32_t* const recs = &d.recs[bank][p0][0][0];
    const uint64_t roff = static_cast<uint64_t>(int64_t(row_base + r) + row_delta) * cap + sr;
    // 64-bit slab pointers of this lane's row (one IMAD.WIDE per record and
    // array).
    const uint64_t vrow = reinterpret_cast<uint64_t>(vals + roff);
    const uint64_t irow = reinterpret_cast<uint64_t>(indices + roff);
    // Two records (h0, h0 + 8) per iteration, both loaded before either is
    // stored: the predicated asm stores are compiler barriers, so an unrolled
    // loop cannot move a record's load above the previous record's stores
    // (blocks whose keys all pass the gate are drain-bound).
#pragma unroll 1
    for (uint32_t h0 = lane & 7; h0 < steps; h0 += 16) {
      uint32_t rec[2], kb[2], hc[2], ok[2];
#pragma unroll
      for (uint32_t u = 0; u < 2; ++u) {
        // Branch-free: out-of-range lanes compute on a clamped index and do
        // not store (predicated st.global.cg).
        const uint32_t h = h0 + 8u * u;
        ok[u] = h < lim;
        hc[u] = ok[u] ? h : 0u;
        const bool b1 = hc[u] >= e1, b2 = hc[u] >= e2, b3 = hc[u] >= e3;
        const uint32_t dd = b3 ? d3 : (b2 ? d2 : (b1 ? d1 : d0));
        kb[u] = kb0 + (b3 ? 96u : (b2 ? 64u : (b1 ? 32u : 0u)));
        rec[u] = recs[hc[u] + dd];
      }
#pragma unroll
      for (uint32_t u = 0; u < 2; ++u) {
        const uint32_t code = staged_code(rec[u]);
        // = store_staged on c = code >> 8 (the fp24 code): value = c bits 0-15,
        // index = key (< 2^20) | c bits 16-23 << 20.
        const uint32_t key = kb[u] + ((rec[u] << 2) & 0x380u) + (rec[u] & 31u);
        const uint32_t idx = key | ((code >> 4) & 0x0ff00000u);
        asm volatile(
            "{\n\t.reg .pred q;\n\tsetp.ne.u32 q, %0, 0;\n\t"
            "@q st.global.cg.u16 [%1], %2;\n\t"
            "@q st.global.cg.u32 [%3], %4;\n\t}" ::"r"(ok[u]),
            "l"(vrow + uint64_t(hc[u]) * 2u),
            "h"(static_cast<uint16_t>(code >> 8)),
            "l"(irow + uint64_t(hc[u]) * 4u),
            "r"(idx)
            : "memory");
      }
    }
  }
  __syncwarp();
#pragma unroll
  for (uint32_t p = 0; p < 4; ++p)
    hempty[bank * kMathWarps + p0 + p].arrive();
}

/// \brief The persistent kernel: one CTA per SM, kNumThreads threads (see the file comment).
__global__ __launch_bounds__(kNumThreads, 1) void fused_mqa_topk_prefill_kernel(
    uint32_t rows,
    uint32_t keys,
    const uint32_t* starts_g,
    const uint32_t* ends_g,
    uint16_t* vals,
    int32_t* indices,
    int32_t* counts,
    RescanBuffers fallback,
    // Unused (the launcher passes 0); dropping it shifts the offsets of the
    // later parameters in the kernel's SASS.
    uint32_t phase_bytes,
    uint32_t sample_target,
    uint32_t sample_stride,
    uint32_t sample_groups,
    uint32_t sample_tiles,
    const float* scales,
    const __grid_constant__ cute::TmaDescriptor tks,
    const __grid_constant__ cute::TmaDescriptor tq,
    const __grid_constant__ cute::TmaDescriptor ts,
    const __grid_constant__ cute::TmaDescriptor tk,
    const __grid_constant__ cute::TmaDescriptor tw,
    bool relative,
    const int32_t* out_offsets,
    uint32_t nblocks,
    uint32_t* counter,
    int32_t* defer_next,
    PageTable paged) {
  constexpr uint32_t kCandGlobalCap = kCandidatesPerRow;
  // Compile-time: a runtime top-k costs ~0.5% in the scan (codegen).
  constexpr uint32_t topk = kTopK;
  constexpr uint32_t Interval = kInterval;
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  auto& sm = *reinterpret_cast<KernelSmem*>(smem_buffer);
  uint8_t* const region = smem_buffer + sizeof(KernelSmem);
  auto& stream = *reinterpret_cast<SampleShared*>(region);
  auto& sdata = *reinterpret_cast<StageLists*>(region);
  static_assert(Interval * 64 <= kStageCap, "an interval must fit the staging lists");

  __shared__ alignas(8) uint64_t bar_words[BarIdx::total];
  __shared__ uint32_t tmem_ptr;
  __shared__ ClaimEntry entries[2];
  __shared__ float gate_s[kBlockRows];
  __shared__ uint32_t row_count[kBlockRows];
  __shared__ uint32_t sel_recover, sel_short;
  __shared__ uint32_t defer_head, unsafe_s, pause_req, epoch_kind;
  __shared__ uint32_t opaque_bases[3];  // scan: see opaque_ld_u32
  __shared__ uint32_t lane_tab[2][32];  // emission: lane id, lanemask_lt

  using B = BarIdx;
  Barrier* const bars = reinterpret_cast<Barrier*>(bar_words);
  Barrier& full_q = bars[B::full_q];
  Barrier& empty_q = bars[B::empty_q];
  Barrier* const full_kv = bars + B::full_kv;
  // The 32 producer lanes stage the sample scales, so each arrives on full_kv.
  constexpr uint32_t kFullKvArrivals = 32;
  Barrier* const empty_kv = bars + B::empty_kv;
  Barrier* const full_tmem = bars + B::full_tmem;
  Barrier* const empty_tmem = bars + B::empty_tmem;
  Barrier* const claim_full = bars + B::claim_full;
  Barrier* const claim_empty = bars + B::claim_empty;
  Barrier& sel_done = bars[B::sel_done];
  Barrier* const h_full = bars + B::h_full;
  Barrier* const h_empty = bars + B::h_empty;
  Barrier* const full_lin = bars + B::full_lin;

  const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
  const uint32_t lane = ptx::get_lane_idx();
  constexpr uint32_t kClaimConsumers = 11;  // warps 9, 10, 11, 0-7

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
  // the dense core and the loops resume with freshly initialised barriers.
  // The first epoch is emitted outside any loop: nesting the role loops in an
  // epoch loop costs ~1.5% in the scan (uniform address math is rematerialised
  // every tile). Later epochs only run after a recovery pause.
  __shared__ uint32_t row_bounds[2][kBlockRows];
  __shared__ uint32_t fslot;
  bool more = false;
  {
    constexpr uint32_t epoch = 0;
#include "persistent_epoch.inc"
  }
#pragma unroll 1
  for (uint32_t epoch = 1; more; ++epoch) {
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

}  // namespace sglang::fused_mqa_topk
