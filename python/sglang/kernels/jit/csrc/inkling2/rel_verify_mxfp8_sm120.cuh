// SPDX-License-Identifier: Apache-2.0

#if defined(SGL_CUDA_ARCH) && (SGL_CUDA_ARCH < 1200 || SGL_CUDA_ARCH >= 1300)
#error "inkling2/rel_verify_mxfp8_sm120.cuh requires SM12x (1200 <= SGL_CUDA_ARCH < 1300)"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits>

namespace sglang::inkling2_rel_verify_mxfp8_sm120 {

namespace {

constexpr int kHeads = 24;
constexpr int kKvHeads = 8;
constexpr int kGroup = 3;
constexpr int kHeadDim = 128;
constexpr int kPageSize = 128;
constexpr int kRelExtent = 1024;
constexpr int kQlen = 8;
constexpr int kMRows = kQlen * kGroup;
constexpr int kMTiles = kMRows / 8;
constexpr int kMaxBatch = 8;
constexpr int kMaxSplits = 64;
constexpr int kBN = 64;
constexpr int kThreads = 128;
constexpr int kPRow = 72;
constexpr float kLog2E = 1.4426950408889634f;
constexpr float kNegInf = -INFINITY;

SGL_DEVICE uint32_t smem_u32(void const* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

SGL_DEVICE void cp_async_cg_16(void* dst_smem, void const* src_gmem) {
  uint32_t dst = smem_u32(dst_smem);
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst), "l"(src_gmem) : "memory");
}

SGL_DEVICE void cp_async_cg_16_pred(void* dst_smem, void const* src_gmem, int src_size) {
  uint32_t dst = smem_u32(dst_smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src_gmem), "r"(src_size) : "memory");
}

SGL_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
SGL_DEVICE void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

SGL_DEVICE uint32_t cvt_e4m3x2_to_bf16x2(uint16_t x) {
  uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(x));
  float2 f2 = __half22float2(*reinterpret_cast<__half2 const*>(&h2));
  __nv_bfloat162 b2 = __float22bfloat162_rn(f2);
  return *reinterpret_cast<uint32_t const*>(&b2);
}

SGL_DEVICE uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
  uint32_t res;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b));
  return res;
}

SGL_DEVICE uint32_t ue8m0_to_bf16_u16(uint32_t e) {
  return (e == 0u) ? 0x0040u : ((e == 255u) ? 0x7FC0u : (e << 7));
}

SGL_DEVICE uint32_t ue8m0_to_bf16x2(uint32_t e) {
  uint32_t h = ue8m0_to_bf16_u16(e);
  return h | (h << 16);
}

SGL_DEVICE void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, void const* smem_ptr) {
  uint32_t addr = smem_u32(smem_ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

SGL_DEVICE void ldmatrix_x2_trans_b8(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, void const* smem_ptr) {
  uint32_t addr = smem_u32(smem_ptr);
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

SGL_DEVICE void
mma_m16n8k16_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

SGL_DEVICE float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

SGL_DEVICE int swiz_c16(int row, int c16) {
  return c16 ^ (row & 7);
}

SGL_DEVICE float col_reduce_max8(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
  return v;
}

SGL_DEVICE float col_reduce_sum8(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  return v;
}

SGL_DEVICE void cp_async_mxfp8_tile(
    uint4* __restrict__ dst_kv,
    uint32_t* __restrict__ dst_sc,
    uint8_t const* __restrict__ KV,
    uint32_t const* __restrict__ ScWords,
    int kv_head,
    int page,
    int slot_base,
    int k0,
    int k_lo_abs,
    int k_hi_abs,
    int64_t sk0,
    int64_t sk1,
    int64_t sk2,
    int64_t ss0,
    int64_t ss1,
    int tid,
    bool full_valid) {
  int const c16 = tid & 7;
  int const r0 = tid >> 3;
  int const col_byte = c16 << 4;
  bool const page_ok = (page >= 0);
  bool const tile_full_ok = full_valid && (k0 >= k_lo_abs) && (k0 + kBN <= k_hi_abs);
  int64_t const sp = page_ok ? static_cast<int64_t>(page) : 0LL;
  uint8_t const* base =
      KV + sp * sk0 + static_cast<int64_t>(slot_base) * sk1 + static_cast<int64_t>(kv_head) * sk2 + col_byte;
#pragma unroll
  for (int step = 0; step < 4; ++step) {
    int const r = r0 + step * 16;
    uint8_t const* src = base + static_cast<int64_t>(r) * sk1;
    if (tile_full_ok && page_ok) {
      cp_async_cg_16(&dst_kv[r * 8 + swiz_c16(r, c16)], src);
    } else {
      int const k_pos = k0 + r;
      int const sz = (page_ok && k_pos >= k_lo_abs && k_pos < k_hi_abs) ? 16 : 0;
      cp_async_cg_16_pred(&dst_kv[r * 8 + swiz_c16(r, c16)], src, sz);
    }
  }
  if (tid < kBN) {
    int const k_pos = k0 + tid;
    if (page_ok && (tile_full_ok || (k_pos >= k_lo_abs && k_pos < k_hi_abs))) {
      int const slot = slot_base + tid;
      int const widx = (slot & 31) * 4 + (slot >> 5);
      dst_sc[tid] = __ldg(ScWords + sp * ss0 + static_cast<int64_t>(kv_head) * ss1 + widx);
    } else {
      dst_sc[tid] = 0u;
    }
  }
}

SGL_DEVICE void cp_async_rel_tile(
    bf16_t (&dst)[kMRows][kPRow],
    bf16_t const* __restrict__ Rel,
    int q_start,
    int kv_head,
    int past,
    int k0,
    int64_t sb0,
    int64_t sb1,
    int tid) {
  constexpr int total = kMRows * 9;
#pragma unroll
  for (int idx = tid; idx < total; idx += kThreads) {
    int const m = idx / 9;
    int const chunk = idx - m * 9;
    int const tok = m / kGroup;
    int const head = kv_head * kGroup + (m - tok * kGroup);
    int const d_lo = (past + tok) - (k0 + kBN - 1);
    int const d_align = d_lo & ~7;
    int const d_c = d_align + (chunk << 3);
    int64_t const row_off = static_cast<int64_t>(q_start + tok) * sb0 + static_cast<int64_t>(head) * sb1;
    bool const ok = (static_cast<unsigned>(d_c) < static_cast<unsigned>(kRelExtent));
    int const safe_dc = ok ? d_c : 0;
    cp_async_cg_16_pred(&dst[m][chunk << 3], Rel + row_off + safe_dc, ok ? 16 : 0);
  }
}

SGL_DEVICE void load_q_direct_reg(
    uint32_t (&q_b0)[8][kMTiles],
    uint32_t (&q_b1)[8][kMTiles],
    bf16_t const* __restrict__ Q,
    int q_start,
    int kv_head,
    int64_t sq0,
    int64_t sq1,
    int gid,
    int tig) {
#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
    int const m = mt * 8 + gid;
    int const tok = m / kGroup;
    int const head = kv_head * kGroup + (m - tok * kGroup);
    bf16_t const* q_row = Q + static_cast<int64_t>(q_start + tok) * sq0 + static_cast<int64_t>(head) * sq1;
#pragma unroll
    for (int ds = 0; ds < 8; ++ds) {
      uint2 const qv = *reinterpret_cast<uint2 const*>(q_row + ds * 16 + tig * 4);
      q_b0[ds][mt] = qv.x;
      q_b1[ds][mt] = qv.y;
    }
  }
}

SGL_DEVICE void compute_kq_mxfp8_ldm(
    float (&s)[kMTiles][4],
    uint4 const* __restrict__ smem_k,
    uint32_t const* __restrict__ smem_ks,
    uint32_t const (&q_b0)[8][kMTiles],
    uint32_t const (&q_b1)[8][kMTiles],
    int w,
    int lane) {
#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
    s[mt][0] = 0.0f;
    s[mt][1] = 0.0f;
    s[mt][2] = 0.0f;
    s[mt][3] = 0.0f;
  }
  int const k_ld_row = w * 16 + ((lane >> 4) << 3) + (lane & 7);
  int const k_ld_c16_off = (lane >> 3) & 1;
  int const k_row0 = w * 16 + (lane >> 2);
  uint32_t const ks0 = smem_ks[k_row0];
  uint32_t const ks1 = smem_ks[k_row0 + 8];

#pragma unroll
  for (int ds2 = 0; ds2 < 4; ++ds2) {
    uint32_t b0, b1, b2, b3;
    ldmatrix_x4(b0, b1, b2, b3, &smem_k[k_ld_row * 8 + swiz_c16(k_ld_row, ds2 * 2 + k_ld_c16_off)]);
    uint32_t const sc0 = ue8m0_to_bf16x2((ks0 >> (ds2 * 8)) & 0xFFu);
    uint32_t const sc1 = ue8m0_to_bf16x2((ks1 >> (ds2 * 8)) & 0xFFu);
    int const ds_lo = ds2 * 2;
    int const ds_hi = ds_lo + 1;

    uint32_t a0 = mul_bf16x2(cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b0)), sc0);
    uint32_t a1 = mul_bf16x2(cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b2)), sc1);
    uint32_t a2 = mul_bf16x2(cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b0 >> 16)), sc0);
    uint32_t a3 = mul_bf16x2(cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b2 >> 16)), sc1);
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt) {
      mma_m16n8k16_bf16(s[mt], a0, a1, a2, a3, q_b0[ds_lo][mt], q_b1[ds_lo][mt]);
    }
    uint32_t c0 = mul_bf16x2(cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b1)), sc0);
    uint32_t c1 = mul_bf16x2(cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b3)), sc1);
    uint32_t c2 = mul_bf16x2(cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b1 >> 16)), sc0);
    uint32_t c3 = mul_bf16x2(cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b3 >> 16)), sc1);
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt) {
      mma_m16n8k16_bf16(s[mt], c0, c1, c2, c3, q_b0[ds_hi][mt], q_b1[ds_hi][mt]);
    }
  }
}

template <int KS_BEGIN>
SGL_DEVICE void compute_vp_mxfp8_ldm_half(
    float (&acc)[2][kMTiles][4],
    uint4 const* __restrict__ smem_v,
    uint32_t const* __restrict__ smem_vs,
    bf16_t const (&smem_p)[kMRows][kPRow],
    int w,
    int lane,
    int gid,
    int tig) {
  int const v_dp = w * 2 + (lane >> 4);
  int const v_r_off = lane & 15;
  int const shift = w * 8;
#pragma unroll
  for (int ks = KS_BEGIN; ks < KS_BEGIN + 2; ++ks) {
    int const k_base = ks * 16;
    int const vr = k_base + v_r_off;
    uint32_t vb0, vb1, vb2, vb3;
    ldmatrix_x2_trans_b8(vb0, vb1, vb2, vb3, &smem_v[vr * 8 + swiz_c16(vr, v_dp)]);

    int const k_t = k_base + tig * 4;
    uint32_t const s0 = ue8m0_to_bf16_u16((smem_vs[k_t + 0] >> shift) & 0xFFu);
    uint32_t const s1 = ue8m0_to_bf16_u16((smem_vs[k_t + 1] >> shift) & 0xFFu);
    uint32_t const s2 = ue8m0_to_bf16_u16((smem_vs[k_t + 2] >> shift) & 0xFFu);
    uint32_t const s3 = ue8m0_to_bf16_u16((smem_vs[k_t + 3] >> shift) & 0xFFu);
    uint32_t const sp01 = s0 | (s1 << 16);
    uint32_t const sp23 = s2 | (s3 << 16);

    uint32_t b0[kMTiles], b1[kMTiles];
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt) {
      int const m = mt * 8 + gid;
      uint2 const pv = *reinterpret_cast<uint2 const*>(&smem_p[m][k_t]);
      b0[mt] = pv.x;
      b1[mt] = pv.y;
    }

    uint32_t a0 = mul_bf16x2(cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb0)), sp01);
    uint32_t a1 = mul_bf16x2(cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb1)), sp01);
    uint32_t a2 = mul_bf16x2(cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb0 >> 16)), sp23);
    uint32_t a3 = mul_bf16x2(cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb1 >> 16)), sp23);
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt) {
      mma_m16n8k16_bf16(acc[0][mt], a0, a1, a2, a3, b0[mt], b1[mt]);
    }
    uint32_t c0 = mul_bf16x2(cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb2)), sp01);
    uint32_t c1 = mul_bf16x2(cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb3)), sp01);
    uint32_t c2 = mul_bf16x2(cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb2 >> 16)), sp23);
    uint32_t c3 = mul_bf16x2(cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb3 >> 16)), sp23);
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt) {
      mma_m16n8k16_bf16(acc[1][mt], c0, c1, c2, c3, b0[mt], b1[mt]);
    }
  }
}

struct VerifyMxfp8Smem {
  alignas(16) uint4 k[2][kBN * 8];
  alignas(16) uint4 v[kBN * 8];
  alignas(16) bf16_t rel[kMRows][kPRow];
  alignas(16) bf16_t p[kMRows][kPRow];
  alignas(16) uint32_t ks[2][kBN];
  alignas(16) uint32_t vs[kBN];
  alignas(16) float m[4][kMRows];
  alignas(16) float l[4][kMRows];
};

static_assert(sizeof(VerifyMxfp8Smem) <= 101376, "VerifyMxfp8Smem must fit within SM120/SM121 99 KiB opt-in smem");
static_assert(
    kMRows * kHeadDim * sizeof(float) <= sizeof(VerifyMxfp8Smem::k),
    "VerifyMxfp8Smem::k must be large enough to stage FP32 partial outputs");

__global__ void __launch_bounds__(kThreads, 3) rel_verify_full_mxfp8_kernel(
    bf16_t const* __restrict__ Q,
    uint8_t const* __restrict__ K,
    uint8_t const* __restrict__ V,
    uint32_t const* __restrict__ KWords,
    uint32_t const* __restrict__ VWords,
    int32_t const* __restrict__ PageTable,
    int32_t const* __restrict__ SeqLens,
    bf16_t const* __restrict__ Rel,
    float* __restrict__ OPart,
    float* __restrict__ MPart,
    float* __restrict__ LPart,
    int batch,
    float sm_scale,
    int num_splits,
    int64_t sq0,
    int64_t sq1,
    int64_t sk0,
    int64_t sk1,
    int64_t sk2,
    int64_t ss0,
    int64_t ss1,
    int64_t sp_stride,
    int64_t sb0,
    int64_t sb1) {
  device::PDLWaitPrimary<true>();

  int const bh = static_cast<int>(blockIdx.x);
  int const req = bh / kKvHeads;
  int const kv_head = bh - req * kKvHeads;
  int const split = static_cast<int>(blockIdx.y);
  int const tid = threadIdx.x;
  int const w = tid >> 5;
  int const lane = static_cast<int>(device::get_lane_id());
  int const gid = lane >> 2;
  int const tig = lane & 3;
  int const num_tokens = batch * kQlen;
  int const q_start = req * kQlen;

  int64_t const seqlen_k64 = static_cast<int64_t>(SeqLens[req]);
  int64_t const tiles32 = (seqlen_k64 > 0) ? ((seqlen_k64 + 31) / 32) : 0;
  int64_t const per_split = (tiles32 > 0) ? ((tiles32 + num_splits - 1) / num_splits) : 0;
  int64_t const tile32_lo = static_cast<int64_t>(split) * per_split;
  int64_t const tile32_hi = (tile32_lo + per_split < tiles32) ? (tile32_lo + per_split) : tiles32;

  if (tile32_lo >= tile32_hi) {
    device::PDLTriggerSecondary<true>();
    if (tid < kMRows) {
      int const tok = tid / kGroup;
      int const head = kv_head * kGroup + (tid - tok * kGroup);
      int64_t const flat = (static_cast<int64_t>(split) * num_tokens + (q_start + tok)) * kHeads + head;
      MPart[flat] = kNegInf;
      LPart[flat] = 0.0f;
    }
    constexpr int total_vec = kMRows * 32;
    float4 const zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int idx = tid; idx < total_vec; idx += kThreads) {
      int const m = idx >> 5, d4 = (idx & 31) << 2;
      int const tok = m / kGroup, head = kv_head * kGroup + (m - tok * kGroup);
      int64_t const part = (static_cast<int64_t>(split) * num_tokens + (q_start + tok)) * kHeads + head;
      *reinterpret_cast<float4*>(OPart + part * kHeadDim + d4) = zero4;
    }
    return;
  }

  __shared__ VerifyMxfp8Smem smem;
  int const seqlen_k = static_cast<int>(seqlen_k64);
  int const k_lo_abs = static_cast<int>(tile32_lo * 32);
  int const k_hi_abs = static_cast<int>((tile32_hi * 32 < seqlen_k64) ? (tile32_hi * 32) : seqlen_k64);
  int const t64_start = static_cast<int>(tile32_lo >> 1);
  int const t64_end = static_cast<int>((tile32_hi - 1) >> 1) + 1;
  int const past = seqlen_k - kQlen;
  int32_t const* bt = PageTable + static_cast<int64_t>(req) * sp_stride;
  int const n_bulk_tiles = (past >= kRelExtent - 1) ? ((past - (kRelExtent - 1)) >> 6) : 0;
  int const bulk_end = (t64_end < n_bulk_tiles) ? t64_end : n_bulk_tiles;
  int const bulk_count = (bulk_end > t64_start) ? (bulk_end - t64_start) : 0;
  int const tail_start = (t64_start > n_bulk_tiles) ? t64_start : n_bulk_tiles;
  int const tail_count = (t64_end > tail_start) ? (t64_end - tail_start) : 0;
  int cur_page = bt[t64_start >> 1];

  if (bulk_count > 0) {
    int const sb = (t64_start & 1) << 6;
    cp_async_mxfp8_tile(
        smem.k[0],
        smem.ks[0],
        K,
        KWords,
        kv_head,
        cur_page,
        sb,
        t64_start << 6,
        k_lo_abs,
        k_hi_abs,
        sk0,
        sk1,
        sk2,
        ss0,
        ss1,
        tid,
        true);
    cp_async_mxfp8_tile(
        smem.v,
        smem.vs,
        V,
        VWords,
        kv_head,
        cur_page,
        sb,
        t64_start << 6,
        k_lo_abs,
        k_hi_abs,
        sk0,
        sk1,
        sk2,
        ss0,
        ss1,
        tid,
        true);
    cp_async_commit();
  } else {
    int const sb = (tail_start & 1) << 6;
    cp_async_mxfp8_tile(
        smem.k[0],
        smem.ks[0],
        K,
        KWords,
        kv_head,
        cur_page,
        sb,
        tail_start << 6,
        k_lo_abs,
        k_hi_abs,
        sk0,
        sk1,
        sk2,
        ss0,
        ss1,
        tid,
        false);
    cp_async_rel_tile(smem.rel, Rel, q_start, kv_head, past, tail_start << 6, sb0, sb1, tid);
    cp_async_commit();
  }

  uint32_t q_b0[8][kMTiles], q_b1[8][kMTiles];
  load_q_direct_reg(q_b0, q_b1, Q, q_start, kv_head, sq0, sq1, gid, tig);

  float m_i[kMTiles][2], l_i[kMTiles][2], acc[2][kMTiles][4];
#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
    m_i[mt][0] = m_i[mt][1] = kNegInf;
    l_i[mt][0] = l_i[mt][1] = 0.0f;
#pragma unroll
    for (int dt = 0; dt < 2; ++dt) {
      acc[dt][mt][0] = acc[dt][mt][1] = acc[dt][mt][2] = acc[dt][mt][3] = 0.0f;
    }
  }

  int const k_local0 = w * 16 + gid;
  int const k_local1 = k_local0 + 8;

  for (int it = 0; it < bulk_count; ++it) {
    int const cur = it & 1;
    int const nxt = cur ^ 1;
    int const t_idx = t64_start + it;
    int const page = cur_page;
    int64_t const blk32_lo = static_cast<int64_t>(t_idx) * 2;
    bool const use_lo = (blk32_lo >= tile32_lo) && (blk32_lo < tile32_hi);
    bool const use_hi = (blk32_lo + 1 >= tile32_lo) && (blk32_lo + 1 < tile32_hi);

    if (it == 0) {
      cp_async_wait<0>();
    } else {
      cp_async_mxfp8_tile(
          smem.v,
          smem.vs,
          V,
          VWords,
          kv_head,
          page,
          (t_idx & 1) << 6,
          t_idx << 6,
          k_lo_abs,
          k_hi_abs,
          sk0,
          sk1,
          sk2,
          ss0,
          ss1,
          tid,
          true);
      cp_async_commit();
      cp_async_wait<1>();
    }
    __syncthreads();

    float s[kMTiles][4];
    compute_kq_mxfp8_ldm(s, smem.k[cur], smem.ks[cur], q_b0, q_b1, w, lane);

#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt) {
      s[mt][0] = __fmul_rn(s[mt][0], sm_scale);
      s[mt][1] = __fmul_rn(s[mt][1], sm_scale);
      s[mt][2] = __fmul_rn(s[mt][2], sm_scale);
      s[mt][3] = __fmul_rn(s[mt][3], sm_scale);
      int const m0 = mt * 8 + tig * 2, m1 = m0 + 1;
      float const wm0 = col_reduce_max8(fmaxf(s[mt][0], s[mt][2]));
      float const wm1 = col_reduce_max8(fmaxf(s[mt][1], s[mt][3]));
      if (gid == 0) {
        smem.m[w][m0] = wm0;
        smem.m[w][m1] = wm1;
      }
    }
    __syncthreads();

    float alpha_lo[kMTiles][2], alpha_hi[kMTiles][2];
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt) {
      int const m0 = mt * 8 + tig * 2, m1 = m0 + 1;
      float mn0_lo = m_i[mt][0], mn1_lo = m_i[mt][1];
      float ms0_lo = 0.0f, ms1_lo = 0.0f;
      if (use_lo) {
        float const tm0_lo = fmaxf(smem.m[0][m0], smem.m[1][m0]);
        float const tm1_lo = fmaxf(smem.m[0][m1], smem.m[1][m1]);
        mn0_lo = fmaxf(m_i[mt][0], tm0_lo);
        mn1_lo = fmaxf(m_i[mt][1], tm1_lo);
        ms0_lo = (mn0_lo == kNegInf) ? 0.0f : mn0_lo;
        ms1_lo = (mn1_lo == kNegInf) ? 0.0f : mn1_lo;
        alpha_lo[mt][0] = ex2_approx(__fmul_rn(__fsub_rn(m_i[mt][0], ms0_lo), kLog2E));
        alpha_lo[mt][1] = ex2_approx(__fmul_rn(__fsub_rn(m_i[mt][1], ms1_lo), kLog2E));
      } else {
        alpha_lo[mt][0] = 1.0f;
        alpha_lo[mt][1] = 1.0f;
      }

      float mn0_hi = mn0_lo, mn1_hi = mn1_lo;
      float ms0_hi = 0.0f, ms1_hi = 0.0f;
      if (use_hi) {
        float const tm0_hi = fmaxf(smem.m[2][m0], smem.m[3][m0]);
        float const tm1_hi = fmaxf(smem.m[2][m1], smem.m[3][m1]);
        mn0_hi = fmaxf(mn0_lo, tm0_hi);
        mn1_hi = fmaxf(mn1_lo, tm1_hi);
        ms0_hi = (mn0_hi == kNegInf) ? 0.0f : mn0_hi;
        ms1_hi = (mn1_hi == kNegInf) ? 0.0f : mn1_hi;
        alpha_hi[mt][0] = ex2_approx(__fmul_rn(__fsub_rn(mn0_lo, ms0_hi), kLog2E));
        alpha_hi[mt][1] = ex2_approx(__fmul_rn(__fsub_rn(mn1_lo, ms1_hi), kLog2E));
      } else {
        alpha_hi[mt][0] = 1.0f;
        alpha_hi[mt][1] = 1.0f;
      }
      m_i[mt][0] = mn0_hi;
      m_i[mt][1] = mn1_hi;

      bool const active_w = (w < 2) ? use_lo : use_hi;
      if (active_w) {
        float const ms0 = (w < 2) ? ms0_lo : ms0_hi;
        float const ms1 = (w < 2) ? ms1_lo : ms1_hi;
        float const p00 = ex2_approx(__fmul_rn(__fsub_rn(s[mt][0], ms0), kLog2E));
        float const p01 = ex2_approx(__fmul_rn(__fsub_rn(s[mt][1], ms1), kLog2E));
        float const p10 = ex2_approx(__fmul_rn(__fsub_rn(s[mt][2], ms0), kLog2E));
        float const p11 = ex2_approx(__fmul_rn(__fsub_rn(s[mt][3], ms1), kLog2E));
        smem.p[m0][k_local0] = __float2bfloat16_rn(p00);
        smem.p[m1][k_local0] = __float2bfloat16_rn(p01);
        smem.p[m0][k_local1] = __float2bfloat16_rn(p10);
        smem.p[m1][k_local1] = __float2bfloat16_rn(p11);
        float const wl0 = col_reduce_sum8(p00 + p10), wl1 = col_reduce_sum8(p01 + p11);
        if (gid == 0) {
          smem.l[w][m0] = wl0;
          smem.l[w][m1] = wl1;
        }
      }
    }

    if (it + 1 < bulk_count) {
      int const nt = t_idx + 1;
      cur_page = bt[nt >> 1];
      cp_async_mxfp8_tile(
          smem.k[nxt],
          smem.ks[nxt],
          K,
          KWords,
          kv_head,
          cur_page,
          (nt & 1) << 6,
          nt << 6,
          k_lo_abs,
          k_hi_abs,
          sk0,
          sk1,
          sk2,
          ss0,
          ss1,
          tid,
          true);
      cp_async_commit();
      if (it > 0) cp_async_wait<1>();
    } else if (tail_count > 0) {
      cur_page = bt[tail_start >> 1];
      cp_async_mxfp8_tile(
          smem.k[0],
          smem.ks[0],
          K,
          KWords,
          kv_head,
          cur_page,
          (tail_start & 1) << 6,
          tail_start << 6,
          k_lo_abs,
          k_hi_abs,
          sk0,
          sk1,
          sk2,
          ss0,
          ss1,
          tid,
          false);
      cp_async_rel_tile(smem.rel, Rel, q_start, kv_head, past, tail_start << 6, sb0, sb1, tid);
      cp_async_commit();
      if (it > 0) cp_async_wait<1>();
    } else if (it > 0) {
      cp_async_wait<0>();
    }
    __syncthreads();

    if (use_lo) {
#pragma unroll
      for (int mt = 0; mt < kMTiles; ++mt) {
        int const m0 = mt * 8 + tig * 2, m1 = m0 + 1;
        float const sum0_lo = smem.l[0][m0] + smem.l[1][m0];
        float const sum1_lo = smem.l[0][m1] + smem.l[1][m1];
        l_i[mt][0] = __fmaf_rn(l_i[mt][0], alpha_lo[mt][0], sum0_lo);
        l_i[mt][1] = __fmaf_rn(l_i[mt][1], alpha_lo[mt][1], sum1_lo);
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) {
          acc[dt][mt][0] *= alpha_lo[mt][0];
          acc[dt][mt][1] *= alpha_lo[mt][1];
          acc[dt][mt][2] *= alpha_lo[mt][0];
          acc[dt][mt][3] *= alpha_lo[mt][1];
        }
      }
      compute_vp_mxfp8_ldm_half<0>(acc, smem.v, smem.vs, smem.p, w, lane, gid, tig);
    }
    if (use_hi) {
#pragma unroll
      for (int mt = 0; mt < kMTiles; ++mt) {
        int const m0 = mt * 8 + tig * 2, m1 = m0 + 1;
        float const sum0_hi = smem.l[2][m0] + smem.l[3][m0];
        float const sum1_hi = smem.l[2][m1] + smem.l[3][m1];
        l_i[mt][0] = __fmaf_rn(l_i[mt][0], alpha_hi[mt][0], sum0_hi);
        l_i[mt][1] = __fmaf_rn(l_i[mt][1], alpha_hi[mt][1], sum1_hi);
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) {
          acc[dt][mt][0] *= alpha_hi[mt][0];
          acc[dt][mt][1] *= alpha_hi[mt][1];
          acc[dt][mt][2] *= alpha_hi[mt][0];
          acc[dt][mt][3] *= alpha_hi[mt][1];
        }
      }
      compute_vp_mxfp8_ldm_half<2>(acc, smem.v, smem.vs, smem.p, w, lane, gid, tig);
    }
    if (it + 1 < bulk_count || tail_count > 0) __syncthreads();
  }

  int const rev0 = (kBN - 1) - k_local0;
  int const rev1 = (kBN - 1) - k_local1;
  for (int it = 0; it < tail_count; ++it) {
    int const cur = it & 1, nxt = cur ^ 1;
    int const t_idx = tail_start + it, k0 = t_idx << 6, page = cur_page;
    int64_t const blk32_lo = static_cast<int64_t>(t_idx) * 2;
    bool const use_lo = (blk32_lo >= tile32_lo) && (blk32_lo < tile32_hi);
    bool const use_hi = (blk32_lo + 1 >= tile32_lo) && (blk32_lo + 1 < tile32_hi);

    cp_async_mxfp8_tile(
        smem.v,
        smem.vs,
        V,
        VWords,
        kv_head,
        page,
        (t_idx & 1) << 6,
        k0,
        k_lo_abs,
        k_hi_abs,
        sk0,
        sk1,
        sk2,
        ss0,
        ss1,
        tid,
        false);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();

    float s[kMTiles][4];
    compute_kq_mxfp8_ldm(s, smem.k[cur], smem.ks[cur], q_b0, q_b1, w, lane);

    int const k_abs0 = k0 + k_local0, k_abs1 = k0 + k_local1;
    bool const kv_ok0 = (k_abs0 >= k_lo_abs) && (k_abs0 < k_hi_abs);
    bool const kv_ok1 = (k_abs1 >= k_lo_abs) && (k_abs1 < k_hi_abs);
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt) {
      int const m0 = mt * 8 + tig * 2, m1 = m0 + 1;
      int const tok0 = m0 / kGroup, tok1 = m1 / kGroup;
      int const sh0 = ((past + tok0) - (k0 + kBN - 1)) & 7;
      int const sh1 = ((past + tok1) - (k0 + kBN - 1)) & 7;
      bool const v00 = kv_ok0 && ((past + tok0) >= k_abs0);
      bool const v01 = kv_ok0 && ((past + tok1) >= k_abs0);
      bool const v10 = kv_ok1 && ((past + tok0) >= k_abs1);
      bool const v11 = kv_ok1 && ((past + tok1) >= k_abs1);
      s[mt][0] = v00 ? __fmaf_rn(s[mt][0], sm_scale, __bfloat162float(smem.rel[m0][rev0 + sh0])) : kNegInf;
      s[mt][1] = v01 ? __fmaf_rn(s[mt][1], sm_scale, __bfloat162float(smem.rel[m1][rev0 + sh1])) : kNegInf;
      s[mt][2] = v10 ? __fmaf_rn(s[mt][2], sm_scale, __bfloat162float(smem.rel[m0][rev1 + sh0])) : kNegInf;
      s[mt][3] = v11 ? __fmaf_rn(s[mt][3], sm_scale, __bfloat162float(smem.rel[m1][rev1 + sh1])) : kNegInf;
      float const wm0 = col_reduce_max8(fmaxf(s[mt][0], s[mt][2]));
      float const wm1 = col_reduce_max8(fmaxf(s[mt][1], s[mt][3]));
      if (gid == 0) {
        smem.m[w][m0] = wm0;
        smem.m[w][m1] = wm1;
      }
    }
    __syncthreads();

    float alpha_lo[kMTiles][2], alpha_hi[kMTiles][2];
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt) {
      int const m0 = mt * 8 + tig * 2, m1 = m0 + 1;
      float mn0_lo = m_i[mt][0], mn1_lo = m_i[mt][1];
      float ms0_lo = 0.0f, ms1_lo = 0.0f;
      if (use_lo) {
        float const tm0_lo = fmaxf(smem.m[0][m0], smem.m[1][m0]);
        float const tm1_lo = fmaxf(smem.m[0][m1], smem.m[1][m1]);
        mn0_lo = fmaxf(m_i[mt][0], tm0_lo);
        mn1_lo = fmaxf(m_i[mt][1], tm1_lo);
        ms0_lo = (mn0_lo == kNegInf) ? 0.0f : mn0_lo;
        ms1_lo = (mn1_lo == kNegInf) ? 0.0f : mn1_lo;
        alpha_lo[mt][0] = ex2_approx(__fmul_rn(__fsub_rn(m_i[mt][0], ms0_lo), kLog2E));
        alpha_lo[mt][1] = ex2_approx(__fmul_rn(__fsub_rn(m_i[mt][1], ms1_lo), kLog2E));
      } else {
        alpha_lo[mt][0] = 1.0f;
        alpha_lo[mt][1] = 1.0f;
      }

      float mn0_hi = mn0_lo, mn1_hi = mn1_lo;
      float ms0_hi = 0.0f, ms1_hi = 0.0f;
      if (use_hi) {
        float const tm0_hi = fmaxf(smem.m[2][m0], smem.m[3][m0]);
        float const tm1_hi = fmaxf(smem.m[2][m1], smem.m[3][m1]);
        mn0_hi = fmaxf(mn0_lo, tm0_hi);
        mn1_hi = fmaxf(mn1_lo, tm1_hi);
        ms0_hi = (mn0_hi == kNegInf) ? 0.0f : mn0_hi;
        ms1_hi = (mn1_hi == kNegInf) ? 0.0f : mn1_hi;
        alpha_hi[mt][0] = ex2_approx(__fmul_rn(__fsub_rn(mn0_lo, ms0_hi), kLog2E));
        alpha_hi[mt][1] = ex2_approx(__fmul_rn(__fsub_rn(mn1_lo, ms1_hi), kLog2E));
      } else {
        alpha_hi[mt][0] = 1.0f;
        alpha_hi[mt][1] = 1.0f;
      }
      m_i[mt][0] = mn0_hi;
      m_i[mt][1] = mn1_hi;

      bool const active_w = (w < 2) ? use_lo : use_hi;
      if (active_w) {
        float const ms0 = (w < 2) ? ms0_lo : ms0_hi;
        float const ms1 = (w < 2) ? ms1_lo : ms1_hi;
        float const p00 = ex2_approx(__fmul_rn(__fsub_rn(s[mt][0], ms0), kLog2E));
        float const p01 = ex2_approx(__fmul_rn(__fsub_rn(s[mt][1], ms1), kLog2E));
        float const p10 = ex2_approx(__fmul_rn(__fsub_rn(s[mt][2], ms0), kLog2E));
        float const p11 = ex2_approx(__fmul_rn(__fsub_rn(s[mt][3], ms1), kLog2E));
        smem.p[m0][k_local0] = __float2bfloat16_rn(p00);
        smem.p[m1][k_local0] = __float2bfloat16_rn(p01);
        smem.p[m0][k_local1] = __float2bfloat16_rn(p10);
        smem.p[m1][k_local1] = __float2bfloat16_rn(p11);
        float const wl0 = col_reduce_sum8(p00 + p10), wl1 = col_reduce_sum8(p01 + p11);
        if (gid == 0) {
          smem.l[w][m0] = wl0;
          smem.l[w][m1] = wl1;
        }
      }
    }

    if (it + 1 < tail_count) {
      int const nt = t_idx + 1;
      cur_page = bt[nt >> 1];
      cp_async_mxfp8_tile(
          smem.k[nxt],
          smem.ks[nxt],
          K,
          KWords,
          kv_head,
          cur_page,
          (nt & 1) << 6,
          nt << 6,
          k_lo_abs,
          k_hi_abs,
          sk0,
          sk1,
          sk2,
          ss0,
          ss1,
          tid,
          false);
      cp_async_rel_tile(smem.rel, Rel, q_start, kv_head, past, nt << 6, sb0, sb1, tid);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    if (use_lo) {
#pragma unroll
      for (int mt = 0; mt < kMTiles; ++mt) {
        int const m0 = mt * 8 + tig * 2, m1 = m0 + 1;
        float const sum0_lo = smem.l[0][m0] + smem.l[1][m0];
        float const sum1_lo = smem.l[0][m1] + smem.l[1][m1];
        l_i[mt][0] = __fmaf_rn(l_i[mt][0], alpha_lo[mt][0], sum0_lo);
        l_i[mt][1] = __fmaf_rn(l_i[mt][1], alpha_lo[mt][1], sum1_lo);
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) {
          acc[dt][mt][0] *= alpha_lo[mt][0];
          acc[dt][mt][1] *= alpha_lo[mt][1];
          acc[dt][mt][2] *= alpha_lo[mt][0];
          acc[dt][mt][3] *= alpha_lo[mt][1];
        }
      }
      compute_vp_mxfp8_ldm_half<0>(acc, smem.v, smem.vs, smem.p, w, lane, gid, tig);
    }
    if (use_hi) {
#pragma unroll
      for (int mt = 0; mt < kMTiles; ++mt) {
        int const m0 = mt * 8 + tig * 2, m1 = m0 + 1;
        float const sum0_hi = smem.l[2][m0] + smem.l[3][m0];
        float const sum1_hi = smem.l[2][m1] + smem.l[3][m1];
        l_i[mt][0] = __fmaf_rn(l_i[mt][0], alpha_hi[mt][0], sum0_hi);
        l_i[mt][1] = __fmaf_rn(l_i[mt][1], alpha_hi[mt][1], sum1_hi);
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) {
          acc[dt][mt][0] *= alpha_hi[mt][0];
          acc[dt][mt][1] *= alpha_hi[mt][1];
          acc[dt][mt][2] *= alpha_hi[mt][0];
          acc[dt][mt][3] *= alpha_hi[mt][1];
        }
      }
      compute_vp_mxfp8_ldm_half<2>(acc, smem.v, smem.vs, smem.p, w, lane, gid, tig);
    }
    if (it + 1 < tail_count) __syncthreads();
  }

  device::PDLTriggerSecondary<true>();
  __syncthreads();
  float (*smem_o)[kHeadDim] = reinterpret_cast<float (*)[kHeadDim]>(&smem.k[0][0]);

#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
    int const m0 = mt * 8 + tig * 2, m1 = m0 + 1;
#pragma unroll
    for (int dt = 0; dt < 2; ++dt) {
      int const d0 = w * 32 + dt * 16 + gid, d1 = d0 + 8;
      smem_o[m0][d0] = acc[dt][mt][0];
      smem_o[m0][d1] = acc[dt][mt][2];
      smem_o[m1][d0] = acc[dt][mt][1];
      smem_o[m1][d1] = acc[dt][mt][3];
    }
    if (w == 0 && gid == 0) {
      int const tok0 = m0 / kGroup, h0 = kv_head * kGroup + (m0 - tok0 * kGroup);
      int const tok1 = m1 / kGroup, h1 = kv_head * kGroup + (m1 - tok1 * kGroup);
      int64_t const flat0 = (static_cast<int64_t>(split) * num_tokens + (q_start + tok0)) * kHeads + h0;
      int64_t const flat1 = (static_cast<int64_t>(split) * num_tokens + (q_start + tok1)) * kHeads + h1;
      bool const ok0 = !(m_i[mt][0] == kNegInf);
      bool const ok1 = !(m_i[mt][1] == kNegInf);
      MPart[flat0] = ok0 ? m_i[mt][0] : kNegInf;
      LPart[flat0] = ok0 ? l_i[mt][0] : 0.0f;
      MPart[flat1] = ok1 ? m_i[mt][1] : kNegInf;
      LPart[flat1] = ok1 ? l_i[mt][1] : 0.0f;
    }
  }
  __syncthreads();
  constexpr int total_vec = kMRows * 32;
  for (int idx = tid; idx < total_vec; idx += kThreads) {
    int const m = idx >> 5, d4 = (idx & 31) << 2;
    int const tok = m / kGroup, head = kv_head * kGroup + (m - tok * kGroup);
    int64_t const part = (static_cast<int64_t>(split) * num_tokens + (q_start + tok)) * kHeads + head;
    *reinterpret_cast<float4*>(OPart + part * kHeadDim + d4) = *reinterpret_cast<float4 const*>(&smem_o[m][d4]);
  }
}

inline void validate_verify_inputs(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k_cache,
    tvm::ffi::TensorView v_cache,
    tvm::ffi::TensorView k_words,
    tvm::ffi::TensorView v_words,
    tvm::ffi::TensorView page_table,
    tvm::ffi::TensorView seqused_k,
    tvm::ffi::TensorView rel_bias,
    tvm::ffi::TensorView o_part,
    tvm::ffi::TensorView m_part,
    tvm::ffi::TensorView l_part,
    int64_t num_splits) {
  using namespace host;
  auto B = SymbolicSize{"batch"};
  auto T = SymbolicSize{"num_tokens"};
  auto P = SymbolicSize{"num_pages"};
  auto MaxP = SymbolicSize{"max_pages"};
  auto S = SymbolicSize{"num_splits"};
  auto sq0 = SymbolicSize{"sq0"};
  auto sq1 = SymbolicSize{"sq1"};
  auto sk0 = SymbolicSize{"sk0"};
  auto sk1 = SymbolicSize{"sk1"};
  auto sk2 = SymbolicSize{"sk2"};
  auto sv0 = SymbolicSize{"sv0"};
  auto sv1 = SymbolicSize{"sv1"};
  auto sv2 = SymbolicSize{"sv2"};
  auto sw0 = SymbolicSize{"sw0"};
  auto sw1 = SymbolicSize{"sw1"};
  auto svw0 = SymbolicSize{"svw0"};
  auto svw1 = SymbolicSize{"svw1"};
  auto sp0 = SymbolicSize{"sp0"};
  auto sb0 = SymbolicSize{"sb0"};
  auto sb1 = SymbolicSize{"sb1"};
  auto dev = SymbolicDevice{};

  S.set_value(num_splits);
  TensorMatcher({B}).with_dtype<int32_t>().with_device<kDLCUDA>(dev).verify(seqused_k);
  int64_t const batch = B.unwrap();
  CHECK_HOST(batch > 0 && batch <= kMaxBatch) << "batch must be in [1, " << kMaxBatch << "], got " << batch;
  CHECK_HOST(num_splits > 0 && num_splits <= kMaxSplits)
      << "num_splits must be in [1, " << kMaxSplits << "], got " << num_splits;
  T.set_value(batch * kQlen);

  TensorMatcher({T, kHeads, kHeadDim})
      .with_strides({sq0, sq1, 1})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(dev)
      .ensure_alignment(8)
      .verify(q);
  TensorMatcher({P, kPageSize, kKvHeads, kHeadDim})
      .with_strides({sk0, sk1, sk2, 1})
      .with_dtype<fp8_e4m3_t>()
      .with_device<kDLCUDA>(dev)
      .ensure_alignment(16)
      .verify(k_cache);
  TensorMatcher({P, kPageSize, kKvHeads, kHeadDim})
      .with_strides({sv0, sv1, sv2, 1})
      .with_dtype<fp8_e4m3_t>()
      .with_device<kDLCUDA>(dev)
      .ensure_alignment(16)
      .verify(v_cache);
  CHECK_HOST(
      v_cache.stride(0) == k_cache.stride(0) && v_cache.stride(1) == k_cache.stride(1) &&
      v_cache.stride(2) == k_cache.stride(2))
      << "k_cache and v_cache must have identical strides";

  TensorMatcher({P, kKvHeads, kPageSize})
      .with_strides({sw0, sw1, 1})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(dev)
      .ensure_alignment(4)
      .verify(k_words);
  TensorMatcher({P, kKvHeads, kPageSize})
      .with_strides({svw0, svw1, 1})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(dev)
      .ensure_alignment(4)
      .verify(v_words);
  CHECK_HOST(v_words.stride(0) == k_words.stride(0) && v_words.stride(1) == k_words.stride(1))
      << "k_words and v_words must have identical strides";

  auto PT_B = SymbolicSize{"pt_batch"};
  TensorMatcher({PT_B, MaxP}).with_strides({sp0, 1}).with_dtype<int32_t>().with_device<kDLCUDA>(dev).verify(page_table);
  CHECK_HOST(PT_B.unwrap() >= batch) << "page_table rows (" << PT_B.unwrap() << ") must be >= batch (" << batch << ")";
  CHECK_HOST(MaxP.unwrap() > 0 && MaxP.unwrap() <= std::numeric_limits<int>::max() / kPageSize)
      << "page_table max_pages (" << MaxP.unwrap() << ") out of valid range";
  TensorMatcher({T, kHeads, kRelExtent})
      .with_strides({sb0, sb1, 1})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(dev)
      .ensure_alignment(16)
      .verify(rel_bias);
  TensorMatcher({S, T, 1, kHeads, kHeadDim})
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(dev)
      .ensure_alignment(16)
      .verify(o_part);
  TensorMatcher({S, T, 1, kHeads}).with_dtype<fp32_t>().with_device<kDLCUDA>(dev).verify(m_part);
  TensorMatcher({S, T, 1, kHeads}).with_dtype<fp32_t>().with_device<kDLCUDA>(dev).verify(l_part);
}

}  // namespace

/// \brief Run SM12x full-attention MXFP8 paged verify, writing (o_part, m_part, l_part).
/// \param q Query tensor [batch * 8, 24, 128] in BF16.
/// \param k_cache Paged MXFP8 key cache [pages, 128, 8, 128] in E4M3.
/// \param v_cache Paged MXFP8 value cache [pages, 128, 8, 128] in E4M3.
/// \param k_words Packed UE8M0 key scales [pages, 8, 128] as int32 words.
/// \param v_words Packed UE8M0 value scales [pages, 8, 128] as int32 words.
/// \param page_table Block table [batch, max_pages] in int32.
/// \param seqused_k Per-request KV lengths [batch] in int32.
/// \param rel_bias Compact relative bias [batch * 8, 24, 1024] in BF16.
/// \param o_part Unnormalized FP32 partial outputs [num_splits, batch * 8, 1, 24, 128].
/// \param m_part Natural-log partial maxima [num_splits, batch * 8, 1, 24] in FP32.
/// \param l_part Partial softmax denominators [num_splits, batch * 8, 1, 24] in FP32.
/// \param sm_scale Softmax attention scale (1 / sqrt(head_dim)).
/// \param num_splits Number of per-request KV splits (1 <= num_splits <= 64).
inline void rel_verify_full_mxfp8(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k_cache,
    tvm::ffi::TensorView v_cache,
    tvm::ffi::TensorView k_words,
    tvm::ffi::TensorView v_words,
    tvm::ffi::TensorView page_table,
    tvm::ffi::TensorView seqused_k,
    tvm::ffi::TensorView rel_bias,
    tvm::ffi::TensorView o_part,
    tvm::ffi::TensorView m_part,
    tvm::ffi::TensorView l_part,
    double sm_scale,
    int64_t num_splits) {
  int64_t const batch = seqused_k.size(0);
  if (batch == 0) return;
  validate_verify_inputs(
      q, k_cache, v_cache, k_words, v_words, page_table, seqused_k, rel_bias, o_part, m_part, l_part, num_splits);
  dim3 grid(static_cast<unsigned>(batch * kKvHeads), static_cast<unsigned>(num_splits), 1);
  host::LaunchKernel(grid, kThreads, q.device())
      .enable_pdl(true)(
          rel_verify_full_mxfp8_kernel,
          static_cast<bf16_t const*>(q.data_ptr()),
          static_cast<uint8_t const*>(k_cache.data_ptr()),
          static_cast<uint8_t const*>(v_cache.data_ptr()),
          static_cast<uint32_t const*>(k_words.data_ptr()),
          static_cast<uint32_t const*>(v_words.data_ptr()),
          static_cast<int32_t const*>(page_table.data_ptr()),
          static_cast<int32_t const*>(seqused_k.data_ptr()),
          static_cast<bf16_t const*>(rel_bias.data_ptr()),
          static_cast<float*>(o_part.data_ptr()),
          static_cast<float*>(m_part.data_ptr()),
          static_cast<float*>(l_part.data_ptr()),
          static_cast<int>(batch),
          static_cast<float>(sm_scale),
          static_cast<int>(num_splits),
          q.stride(0),
          q.stride(1),
          k_cache.stride(0),
          k_cache.stride(1),
          k_cache.stride(2),
          k_words.stride(0),
          k_words.stride(1),
          page_table.stride(0),
          rel_bias.stride(0),
          rel_bias.stride(1));
}

}  // namespace sglang::inkling2_rel_verify_mxfp8_sm120
