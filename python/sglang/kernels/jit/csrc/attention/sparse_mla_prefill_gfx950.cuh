// DSA sparse MLA prefill on gfx950 for fp8_ds_mla KV rows (656 B: 512 fp8 e4m3 NoPE,
// 4 fp32 per-128 scales, 64 bf16 RoPE), H <= 16 heads, BF16 MFMA with FP32 accumulation.
//
// One CTA per query token, 4 waves. Wave w owns NoPE scale group w (dims [128w, 128w+128)):
// it computes that group's QK partial and that group's 128 output dims, so no wave repeats
// another's dot or fp8 convert. The 4 QK partials of a 32-key tile are summed through LDS.
// QK runs as S^T = K Q^T and PV as O^T = V^T P^T so that every lane owns one head
// (lane % 16): softmax stats, P and the output accumulator never change lanes.
// fp8 converts are exact (unit scale); group scales go onto the QK partial and onto P.
//
// kSplit (decode / verify batches too small to fill the GPU one row per CTA): grid.y CTAs share
// a row, each over a contiguous range of its key tiles, and leave the unnormalized output, the
// running max and the row sum in fp32 for sparse_mla_split_combine_kernel.
#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {
namespace sparse_mla_prefill_gfx950 {

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef float floatx4 __attribute__((ext_vector_type(4)));
typedef int intx2 __attribute__((ext_vector_type(2)));
typedef __attribute__((address_space(3))) intx2 lds_intx2;
typedef unsigned uintx2 __attribute__((ext_vector_type(2)));

constexpr int kDv = 512;
constexpr int kRow = 656;
constexpr int kScaleOff = 512;
constexpr int kRopeOff = 528;
constexpr int kKeys = 32;  // keys per tile
constexpr int kWaves = 4;
constexpr int kThreads = 64 * kWaves;
constexpr int kTileBytes = kKeys * kRow;
constexpr int kRowChunks = kRow / 16;  // 41 16-B chunks per row
constexpr int kRowsPerWave = kKeys / kWaves;

struct Params {
  const __bf16* q_nope;
  const __bf16* q_rope;
  const uint8_t* kv;
  const int32_t* idx;
  uint32_t kv_bytes;  // KV pool size; rows are addressed by a 32-bit buffer offset
  __bf16* out;
  float* lse;
  int64_t sqn_t, sqn_h, sqr_t, sqr_h;  // element strides
  int H;
  int topk;
  float qk_scale;  // sm_scale * log2(e)
  int store_lse;
  // kSplit: [T, splits, H, 512] fp32 partial outputs and [T, splits, H] fp32 (max, sum), log2
  float* part_acc;
  float* part_m;
  float* part_l;
  int tiles_per_split;
};

SGL_DEVICE bf16x8 cvt_fp8x8(intx2 v) {
  const bf16x2 a = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(v[0], 1.0f, false);
  const bf16x2 b = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(v[0], 1.0f, true);
  const bf16x2 c = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(v[1], 1.0f, false);
  const bf16x2 d = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(v[1], 1.0f, true);
  return bf16x8{a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]};
}

// __syncthreads() fences all memory, i.e. also waits for the next tile's LDS DMA; the loop
// only needs this wave's LDS stores done (DMA completion is waited for explicitly).
SGL_DEVICE void lds_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __builtin_amdgcn_s_barrier();
}

SGL_DEVICE floatx4 mfma(bf16x8 a, bf16x8 b, floatx4 c) {
  return __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
}

constexpr int kBufferConfig = 0x00020000;
constexpr int kXcds = 8;  // gfx950 raw buffer descriptor word 3

// Max / sum with the lanes 32 and 16 apart, without going through LDS.
SGL_DEVICE float xmax32(float x) {
  const uintx2 r = __builtin_amdgcn_permlane32_swap(__float_as_uint(x), __float_as_uint(x), false, true);
  return fmaxf(__uint_as_float(r[0]), __uint_as_float(r[1]));
}
SGL_DEVICE float xmax16(float x) {
  const uintx2 r = __builtin_amdgcn_permlane16_swap(__float_as_uint(x), __float_as_uint(x), false, true);
  return fmaxf(__uint_as_float(r[0]), __uint_as_float(r[1]));
}
SGL_DEVICE float xsum32(float x) {
  const uintx2 r = __builtin_amdgcn_permlane32_swap(__float_as_uint(x), __float_as_uint(x), false, true);
  return __uint_as_float(r[0]) + __uint_as_float(r[1]);
}
SGL_DEVICE float xsum16(float x) {
  const uintx2 r = __builtin_amdgcn_permlane16_swap(__float_as_uint(x), __float_as_uint(x), false, true);
  return __uint_as_float(r[0]) + __uint_as_float(r[1]);
}

// kStages tiles of KV rows live in LDS; wave w fetches rows [8w, 8w + 8) of each tile with
// LDS DMA (lanes 0..40 move one 16-B chunk each), addressed from wave-uniform indices.
template <int kStages, bool kSplit>
__global__ __launch_bounds__(kThreads) void sparse_mla_prefill_kernel(const Params p) {
  static_assert(kStages >= 2 && kStages <= 4, "vmcnt waits are written for 2..4 stages");
  __shared__ __attribute__((aligned(16))) uint8_t s_kv[kStages][kTileBytes];
  __shared__ __attribute__((aligned(16))) float s_mask[kStages][kKeys];  // 0 or -inf per key
  __shared__ floatx4 s_red[kWaves][2][64];

  // Blocks go round-robin over the 8 XCDs; give each XCD a contiguous token range so that
  // neighbouring tokens, whose top-k sets overlap, share that XCD's L2.
  int t;
  {
    const int T = static_cast<int>(gridDim.x);
    const int q = T / kXcds;
    const int b = static_cast<int>(blockIdx.x);
    t = b < q * kXcds ? (b % kXcds) * q + b / kXcds : b;
  }
  const int tid = threadIdx.x;
  // = NoPE scale group. Wave-uniform in an SGPR: MFMA ignores EXEC, so a branch on a value
  // the compiler thinks divergent would run both sides' MFMAs.
  const int w = __builtin_amdgcn_readfirstlane(tid >> 6);
  const int l = tid & 63;
  const int n = l & 15;  // head
  const int g = l >> 4;
  // this CTA's key tiles: [j0, j0 + ntiles)
  int j0 = 0;
  int ntiles = (p.topk + kKeys - 1) / kKeys;
  if constexpr (kSplit) {
    j0 = static_cast<int>(blockIdx.y) * p.tiles_per_split;
    ntiles = max(0, min(ntiles - j0, p.tiles_per_split));
  }
  const int32_t* idx_row = p.idx + static_cast<int64_t>(t) * p.topk;

  // Q^T as the QK B operand: col = head n, k = dims (g * 8 + j) of each 32-dim step.
  const bf16x8 zero8 = {};
  const bool head_ok = n < p.H;
  bf16x8 qn[4];
  bf16x8 qr = zero8;
  const int rks = w & 1;   // this wave's RoPE k-step
  const int rnb = w >> 1;  // ... and key half
  {
    const __bf16* qp = p.q_nope + t * p.sqn_t + n * p.sqn_h + w * 128 + g * 8;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks)
      qn[ks] = head_ok ? *reinterpret_cast<const bf16x8*>(qp + ks * 32) : zero8;
    if (head_ok) qr = *reinterpret_cast<const bf16x8*>(p.q_rope + t * p.sqr_t + n * p.sqr_h + rks * 32 + g * 8);
  }

  // Lanes l hold the index of row 8w + (l & 7) of tile j.
  // Branch-free (clamped address, then masked) so no load result merges at a join, which
  // would make the compiler wait for every outstanding load.
  auto load_idx = [&](int j) -> int {
    const int k = (j0 + j) * kKeys + kRowsPerWave * w + (l & 7);
    const int v = idx_row[min(k, p.topk - 1)];
    return k < p.topk ? v : -1;
  };
  // Invalid rows read through an empty descriptor: out-of-range buffer loads return zeros.
  const __amdgpu_buffer_rsrc_t rsrc_kv =
      __builtin_amdgcn_make_buffer_rsrc(const_cast<uint8_t*>(p.kv), 0, static_cast<int>(p.kv_bytes), kBufferConfig);
  const __amdgpu_buffer_rsrc_t rsrc_none =
      __builtin_amdgcn_make_buffer_rsrc(const_cast<uint8_t*>(p.kv), 0, 0, kBufferConfig);
  auto issue_kv = [&](int stage, int vidx) {
    if (l < kRowsPerWave) s_mask[stage][kRowsPerWave * w + l] = vidx >= 0 ? 0.0f : -INFINITY;
    if (l < kRowChunks) {
#pragma unroll
      for (int r = 0; r < kRowsPerWave; ++r) {
        const int id = __builtin_amdgcn_readlane(vidx, r);
        auto* dst = (__attribute__((address_space(3))) void*)&s_kv[stage][(kRowsPerWave * w + r) * kRow];
        if (id >= 0)
          __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc_kv, dst, 16, l * 16, id * kRow, 0, 0);
        else
          __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc_none, dst, 16, l * 16, 0, 0, 0);
      }
    }
  };

  // VMEM order per iteration i: 8 DMAs of tile i+S-1, then the index load of tile i+2S-1.
  // vq[r] = indices of tile j+S-1+r at the top of iteration j, so the indices a DMA needs
  // were loaded before the DMAs of the tile we wait for.
  int vq[kStages];
#pragma unroll
  for (int t = 0; t < kStages - 1; ++t)
    if (t < ntiles) issue_kv(t, load_idx(t));
#pragma unroll
  for (int r = 0; r < kStages; ++r)
    vq[r] = load_idx(kStages - 1 + r);

  // Online softmax state for head n (identical in all 4 waves), log2 domain.
  float m_i = -INFINITY;
  float l_i = 0.0f;  // this lane's partial row sum; summed over the 4 lane groups at the end
  floatx4 acc[8];
#pragma unroll
  for (int nb = 0; nb < 8; ++nb)
    acc[nb] = floatx4{0, 0, 0, 0};

  // PV A operand via ds_read_b64_tr_b8: within a 16-lane group, result lane i element j is
  // byte (i & 7) of source lane (2j + (i >> 3)). Source lane s therefore points at key
  // K_g[s >> 1], byte (s & 1) * 8 of the 16-dim block; K_g = {4g..4g+3, 16+4g..16+4g+3} are
  // the keys this lane holds in the QK C layout.
  const int v_j = n >> 1;
  const int v_key = v_j < 4 ? g * 4 + v_j : 16 + g * 4 + (v_j - 4);
  const int v_off = v_key * kRow + w * 128 + (n & 1) * 8;

  for (int j = 0; j < ntiles; ++j) {
    const int stage = j % kStages;
    // Tile j's DMAs must have landed. Issued after them: one index load in their iteration
    // and 9 VMEM in each of the S-2 later ones, unless the tail skipped DMAs.
    if (j + kStages - 2 < ntiles) {
      if constexpr (kStages == 2)
        asm volatile("s_waitcnt vmcnt(1)" ::: "memory");
      else if constexpr (kStages == 3)
        asm volatile("s_waitcnt vmcnt(10)" ::: "memory");
      else
        asm volatile("s_waitcnt vmcnt(19)" ::: "memory");
    } else {
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
    lds_barrier();  // tile j visible to all waves; everyone is done with tile j-1
    if (j + kStages - 1 < ntiles) issue_kv((j + kStages - 1) % kStages, vq[0]);
#pragma unroll
    for (int r = 0; r + 1 < kStages; ++r)
      vq[r] = vq[r + 1];
    vq[kStages - 1] = load_idx(j + 2 * kStages - 1);

    const uint8_t* kvs = s_kv[stage];

    // ---- QK partial over this wave's 128 NoPE dims: S^T[key, head] ----
    floatx4 c0 = {0, 0, 0, 0}, c1 = {0, 0, 0, 0};
    {
      const uint8_t* r0 = kvs + n * kRow + w * 128 + g * 8;
      const uint8_t* r1 = r0 + 16 * kRow;
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        c0 = mfma(cvt_fp8x8(*reinterpret_cast<const intx2*>(r0 + ks * 32)), qn[ks], c0);
        c1 = mfma(cvt_fp8x8(*reinterpret_cast<const intx2*>(r1 + ks * 32)), qn[ks], c1);
      }
    }
    float sc0[4], sc1[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sc0[i] = *reinterpret_cast<const float*>(kvs + (g * 4 + i) * kRow + kScaleOff + 4 * w);
      sc1[i] = *reinterpret_cast<const float*>(kvs + (16 + g * 4 + i) * kRow + kScaleOff + 4 * w);
      c0[i] *= sc0[i];
      c1[i] *= sc1[i];
    }
    {
      const bf16x8 kr =
          *reinterpret_cast<const bf16x8*>(kvs + (rnb * 16 + n) * kRow + kRopeOff + (rks * 32 + g * 8) * 2);
      if (rnb == 0)
        c0 = mfma(kr, qr, c0);
      else
        c1 = mfma(kr, qr, c1);
    }
    s_red[w][0][l] = c0;
    s_red[w][1][l] = c1;
    lds_barrier();
    floatx4 s0 = s_red[0][0][l], s1 = s_red[0][1][l];
#pragma unroll
    for (int ww = 1; ww < kWaves; ++ww) {
      s0 += s_red[ww][0][l];
      s1 += s_red[ww][1][l];
    }

    // ---- online softmax over the tile's 32 keys for head n ----
    const floatx4 mk0 = *reinterpret_cast<const floatx4*>(&s_mask[stage][g * 4]);
    const floatx4 mk1 = *reinterpret_cast<const floatx4*>(&s_mask[stage][16 + g * 4]);
    float sv[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sv[i] = fmaf(s0[i], p.qk_scale, mk0[i]);
      sv[4 + i] = fmaf(s1[i], p.qk_scale, mk1[i]);
    }
    float mloc =
        fmaxf(fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3])), fmaxf(fmaxf(sv[4], sv[5]), fmaxf(sv[6], sv[7])));
    mloc = xmax16(xmax32(mloc));
    // Lazy rescale: keep the running max unless some head's max grew by more than 2^8, so
    // p <= 256 and the accumulator is rescaled only on the rare tiles that move it.
    if (__builtin_amdgcn_ballot_w64(mloc > m_i + 8.0f) != 0) {
      const float m_new = fmaxf(m_i, mloc);
      const float alpha = __builtin_amdgcn_exp2f(m_i - (m_new == -INFINITY ? 0.0f : m_new));
      l_i *= alpha;
#pragma unroll
      for (int nb = 0; nb < 8; ++nb)
        acc[nb] *= alpha;
      m_i = m_new;
    }
    const float m_use = m_i == -INFINITY ? 0.0f : m_i;
    float psum = 0.0f;
    bf16x8 pb;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float p0 = __builtin_amdgcn_exp2f(sv[i] - m_use);
      const float p1 = __builtin_amdgcn_exp2f(sv[4 + i] - m_use);
      psum += p0 + p1;
      pb[i] = static_cast<__bf16>(p0 * sc0[i]);
      pb[4 + i] = static_cast<__bf16>(p1 * sc1[i]);
    }
    l_i += psum;

    // ---- PV over this wave's 128 output dims: O^T[dim, head] ----
#pragma unroll
    for (int nb = 0; nb < 8; ++nb) {
      const intx2 v = __builtin_amdgcn_ds_read_tr8_b64_v2i32((lds_intx2*)(kvs + v_off + nb * 16));
      acc[nb] = mfma(cvt_fp8x8(v), pb, acc[nb]);
    }
  }

  const float l_tot = xsum16(xsum32(l_i));
  if (!head_ok) return;
  if constexpr (kSplit) {
    const int64_t ps = (static_cast<int64_t>(t) * gridDim.y + blockIdx.y) * p.H + n;
    float* ap = p.part_acc + ps * kDv + w * 128 + g * 4;
#pragma unroll
    for (int nb = 0; nb < 8; ++nb)
      *reinterpret_cast<floatx4*>(ap + nb * 16) = acc[nb];
    if (w == 0 && g == 0) {
      p.part_m[ps] = m_i;
      p.part_l[ps] = l_tot;
    }
    return;
  }
  const float inv = l_tot > 0.0f ? 1.0f / l_tot : 0.0f;
  __bf16* op = p.out + (static_cast<int64_t>(t) * p.H + n) * kDv + w * 128 + g * 4;
#pragma unroll
  for (int nb = 0; nb < 8; ++nb) {
    typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
    const bf16x4 o = {
        static_cast<__bf16>(acc[nb][0] * inv),
        static_cast<__bf16>(acc[nb][1] * inv),
        static_cast<__bf16>(acc[nb][2] * inv),
        static_cast<__bf16>(acc[nb][3] * inv)};
    *reinterpret_cast<bf16x4*>(op + nb * 16) = o;
  }
  if (p.store_lse && w == 0 && g == 0)
    p.lse[static_cast<int64_t>(t) * p.H + n] = l_tot > 0.0f ? (m_i + log2f(l_tot)) * 0.6931471805599453f : -INFINITY;
}

// out[t, h] = bf16(sum_s acc_s 2^(m_s - M) / sum_s l_s 2^(m_s - M)), M = max_s m_s: one block
// per (row, head), thread i owning output dims [4i, 4i + 4), so the S partial loads of a head
// are two dependent round trips rather than 2H.
__global__ __launch_bounds__(128) void sparse_mla_split_combine_kernel(const Params p, int splits) {
  const int t = blockIdx.x / p.H;
  const int n = blockIdx.x - t * p.H;
  const int d = threadIdx.x * 4;
  const int64_t base = static_cast<int64_t>(t) * splits * p.H + n;
  float m = -INFINITY;
  for (int s = 0; s < splits; ++s)
    m = fmaxf(m, p.part_m[base + s * p.H]);
  const float m_use = m == -INFINITY ? 0.0f : m;
  float l = 0.0f;
  floatx4 o = {0, 0, 0, 0};
  for (int s = 0; s < splits; ++s) {
    const int64_t ps = base + s * p.H;
    const float scale = __builtin_amdgcn_exp2f(p.part_m[ps] - m_use);
    l += p.part_l[ps] * scale;
    o += *reinterpret_cast<const floatx4*>(p.part_acc + ps * kDv + d) * scale;
  }
  const float inv = l > 0.0f ? 1.0f / l : 0.0f;
  typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
  const bf16x4 r = {
      static_cast<__bf16>(o[0] * inv),
      static_cast<__bf16>(o[1] * inv),
      static_cast<__bf16>(o[2] * inv),
      static_cast<__bf16>(o[3] * inv)};
  *reinterpret_cast<bf16x4*>(p.out + (static_cast<int64_t>(t) * p.H + n) * kDv + d) = r;
  if (p.store_lse && threadIdx.x == 0)
    p.lse[static_cast<int64_t>(t) * p.H + n] = l > 0.0f ? (m + log2f(l)) * 0.6931471805599453f : -INFINITY;
}

}  // namespace sparse_mla_prefill_gfx950

template <int kStages>
struct SparseMlaPrefillGfx950Kernel {
  static void
  run(const tvm::ffi::TensorView q_nope,   // [T, H, 512] bf16, unit inner stride
      const tvm::ffi::TensorView q_rope,   // [T, H, 64] bf16, unit inner stride
      const tvm::ffi::TensorView kv,       // [N, 656] uint8 fp8_ds_mla rows
      const tvm::ffi::TensorView indices,  // [T, topk] int32, -1 = no key
      const tvm::ffi::TensorView out,      // [T, H, 512] bf16
      const tvm::ffi::TensorView lse,      // [T, H] fp32, natural log
      const double sm_scale,
      const int64_t store_lse) {
    using namespace host;
    namespace k = sparse_mla_prefill_gfx950;
    const int64_t T = q_nope.size(0);
    const int64_t H = q_nope.size(1);
    RuntimeCheck(H >= 1 && H <= 16, "1..16 heads");
    RuntimeCheck(q_nope.size(2) == 512 && q_nope.stride(2) == 1, "q_nope [T, H, 512]");
    RuntimeCheck(q_rope.size(2) == 64 && q_rope.stride(2) == 1, "q_rope [T, H, 64]");
    RuntimeCheck(
        q_nope.stride(0) % 8 == 0 && q_nope.stride(1) % 8 == 0 && q_rope.stride(0) % 8 == 0 &&
            q_rope.stride(1) % 8 == 0,
        "q strides must keep 16-B alignment");
    RuntimeCheck(kv.size(-1) == k::kRow && kv.stride(0) == k::kRow, "kv [N, 656] bytes");
    RuntimeCheck(indices.size(0) == T && indices.stride(1) == 1 && indices.stride(0) == indices.size(1), "indices");
    // the index prefetch reads idx_row[topk - 1] before the tile loop
    RuntimeCheck(indices.size(1) >= 1, "topk must be at least 1");
    RuntimeCheck(out.stride(1) == 512 && out.stride(0) == H * 512 && lse.stride(0) == H, "out / lse dense");
    if (T == 0) return;
    k::Params p{};
    p.q_nope = static_cast<const __bf16*>(q_nope.data_ptr());
    p.q_rope = static_cast<const __bf16*>(q_rope.data_ptr());
    p.kv = static_cast<const uint8_t*>(kv.data_ptr());
    p.idx = static_cast<const int32_t*>(indices.data_ptr());
    RuntimeCheck(kv.size(0) * k::kRow <= 0x7fffffffLL, "KV pool must fit a 31-bit buffer offset");
    p.kv_bytes = static_cast<uint32_t>(kv.size(0) * k::kRow);
    p.out = static_cast<__bf16*>(out.data_ptr());
    p.lse = static_cast<float*>(lse.data_ptr());
    p.sqn_t = q_nope.stride(0);
    p.sqn_h = q_nope.stride(1);
    p.sqr_t = q_rope.stride(0);
    p.sqr_h = q_rope.stride(1);
    p.H = static_cast<int>(H);
    p.topk = static_cast<int>(indices.size(1));
    p.qk_scale = static_cast<float>(sm_scale * 1.4426950408889634);
    p.store_lse = static_cast<int>(store_lse);
    LaunchKernel(static_cast<uint32_t>(T), k::kThreads, q_nope.device())(
        k::sparse_mla_prefill_kernel<kStages, false>, p);
  }

  // The same attention with each row's key tiles split over `splits` CTAs, then combined
  // through part_acc [T, splits, H, 512] / part_ml [2, T, splits, H] fp32 workspaces.
  static void run_split(
      const tvm::ffi::TensorView q_nope,
      const tvm::ffi::TensorView q_rope,
      const tvm::ffi::TensorView kv,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView lse,
      const tvm::ffi::TensorView part_acc,
      const tvm::ffi::TensorView part_ml,
      const double sm_scale,
      const int64_t store_lse,
      const int64_t splits) {
    using namespace host;
    namespace k = sparse_mla_prefill_gfx950;
    const int64_t T = q_nope.size(0);
    const int64_t H = q_nope.size(1);
    RuntimeCheck(H >= 1 && H <= 16, "1..16 heads");
    RuntimeCheck(q_nope.size(2) == 512 && q_nope.stride(2) == 1, "q_nope [T, H, 512]");
    RuntimeCheck(q_rope.size(2) == 64 && q_rope.stride(2) == 1, "q_rope [T, H, 64]");
    RuntimeCheck(
        q_nope.stride(0) % 8 == 0 && q_nope.stride(1) % 8 == 0 && q_rope.stride(0) % 8 == 0 &&
            q_rope.stride(1) % 8 == 0,
        "q strides must keep 16-B alignment");
    RuntimeCheck(kv.size(-1) == k::kRow && kv.stride(0) == k::kRow, "kv [N, 656] bytes");
    RuntimeCheck(indices.size(0) == T && indices.stride(1) == 1 && indices.stride(0) == indices.size(1), "indices");
    RuntimeCheck(indices.size(1) >= 1, "topk must be at least 1");
    RuntimeCheck(out.stride(1) == 512 && out.stride(0) == H * 512 && lse.stride(0) == H, "out / lse dense");
    RuntimeCheck(splits >= 1, "at least one split");
    RuntimeCheck(part_acc.numel() >= T * splits * H * 512 && part_ml.numel() >= 2 * T * splits * H, "workspace");
    if (T == 0) return;
    k::Params p{};
    p.q_nope = static_cast<const __bf16*>(q_nope.data_ptr());
    p.q_rope = static_cast<const __bf16*>(q_rope.data_ptr());
    p.kv = static_cast<const uint8_t*>(kv.data_ptr());
    p.idx = static_cast<const int32_t*>(indices.data_ptr());
    RuntimeCheck(kv.size(0) * k::kRow <= 0x7fffffffLL, "KV pool must fit a 31-bit buffer offset");
    p.kv_bytes = static_cast<uint32_t>(kv.size(0) * k::kRow);
    p.out = static_cast<__bf16*>(out.data_ptr());
    p.lse = static_cast<float*>(lse.data_ptr());
    p.sqn_t = q_nope.stride(0);
    p.sqn_h = q_nope.stride(1);
    p.sqr_t = q_rope.stride(0);
    p.sqr_h = q_rope.stride(1);
    p.H = static_cast<int>(H);
    p.topk = static_cast<int>(indices.size(1));
    p.qk_scale = static_cast<float>(sm_scale * 1.4426950408889634);
    p.store_lse = static_cast<int>(store_lse);
    const int ntiles = (p.topk + k::kKeys - 1) / k::kKeys;
    p.tiles_per_split = (ntiles + static_cast<int>(splits) - 1) / static_cast<int>(splits);
    p.part_acc = static_cast<float*>(part_acc.data_ptr());
    p.part_m = static_cast<float*>(part_ml.data_ptr());
    p.part_l = p.part_m + T * splits * H;
    const DLDevice dev = q_nope.device();
    LaunchKernel(dim3(static_cast<uint32_t>(T), static_cast<uint32_t>(splits)), k::kThreads, dev)(
        k::sparse_mla_prefill_kernel<kStages, true>, p);
    LaunchKernel(static_cast<uint32_t>(T * H), 128, dev)(
        k::sparse_mla_split_combine_kernel, p, static_cast<int>(splits));
  }
};

}  // namespace sglang
