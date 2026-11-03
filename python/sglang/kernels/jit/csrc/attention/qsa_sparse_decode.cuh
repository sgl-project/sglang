// QSA (Qwen4-Exp sparse attention) decode / target-verify attention over an FP8 E4M3 KV pool.
//
// Each query row attends to the token positions its top-k row selects (logical positions, -1 = hole). The kernels
// read the selected K / V rows straight from the KV pool through req_to_token, so the selected rows are never
// gathered into a scratch buffer, and run a split-K online softmax (flash decoding):
//   split kernel:   one CTA per (row, kv head, split) of kNSplit splits; 4 warps; 32-column blocks loaded with a
//                   cp.async pipeline of kStages blocks; s = q k^T and acc += p v on bf16 m16n8k16 tensor cores.
//                   It writes unnormalized partials (acc, running max m, running sum l) for the kGP q-head rows.
//   combine kernel: one CTA per (row, kv head); merges the kNSplit partials of its G q heads into bf16 output.
// A column is used when 0 <= pos < seq_len[row] and req_to_token[req, pos] >= 0. A row with no valid column
// outputs zeros. The work of a row never depends on the other rows of the batch, so outputs are batch invariant.
//
// FP8 E4M3 -> bf16 is exact. The fast conversion (byte permute, shift / mask, one bf16 multiply by 2^120) maps the
// NaN codes 0x7f / 0xff to +-480, which no finite E4M3 value reaches; a fragment that contains one is converted
// again with cvt so NaN propagates.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace qsa_sparse_decode {

constexpr int kD = 256;
constexpr int kGP = 16;  // q-head rows per (row, kv head) tile: G <= 16, rows past G are zero
constexpr int kBN = 32;
constexpr int kNSplit = 8;
constexpr int kThreads = 128;
constexpr int kTile = kBN * kD;  // fp8 bytes per K or V block tile
constexpr int kMaxBlocks = 64;   // columns per split <= 2048
constexpr unsigned kFull = 0xffffffffu;

struct Params {
  const uint16_t* q;  // bf16 [R, HQ, D]
  const uint8_t* k;   // fp8 e4m3 [slots, HKV, D]
  const uint8_t* v;
  const int32_t* req_to_token;
  const void* row_req;  // int32 or int64 [R]
  const int32_t* idx;   // [R, TOPK] logical positions, -1 = hole
  const int32_t* seqlen;
  float* part_acc;  // [R * HKV * kNSplit * kGP, D]
  float* part_m;    // [R * HKV * kNSplit * kGP]
  float* part_l;
  uint16_t* out;  // bf16 [R, HQ, D]
  int64_t stride_qr, stride_qh, stride_kslot, stride_kh, stride_r2t, stride_idx, stride_or, stride_oh;
  float qk_scale;
  int32_t topk, chunk, G, HKV;
};

SGL_DEVICE float ex2_approx(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

SGL_DEVICE uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

SGL_DEVICE void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}

SGL_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int kPending>
SGL_DEVICE void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(kPending) : "memory");
}

SGL_DEVICE void ldsm_x4(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

SGL_DEVICE void ldsm_x4_trans(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

// D = A * B + D, m16n8k16, bf16 x bf16 -> f32
SGL_DEVICE void mma_bf16(
    float (&d)[4],
    const uint32_t a0,
    const uint32_t a1,
    const uint32_t a2,
    const uint32_t a3,
    const uint32_t b0,
    const uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// bytes (bi, bj) of raw (fp8 e4m3) -> bf16x2 {lo: bi, hi: bj}. kSel = prmt selector [bi, bi | 8, bj, bj | 8] puts
// each byte in the high byte of a 16-bit lane over its replicated sign; (x << 4) & 0x87f0 is then the e4m3 code in
// bf16 position, and the multiply by 2^120 rebiases the exponent (zero and subnormals included).
template <uint32_t kSel>
SGL_DEVICE uint32_t e4m3x2_bf16x2(const uint32_t raw) {
  uint32_t x, r;
  asm("prmt.b32 %0, %1, 0, %2;" : "=r"(x) : "r"(raw), "n"(kSel));
  const uint32_t y = (x << 4) & 0x87F087F0u;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(y), "r"(0x7B807B80u));
  return r;
}

// the same pair through cvt (NaN-preserving); kSel puts bytes bi, bj in the low 16 bits
template <uint32_t kSel>
SGL_DEVICE uint32_t e4m3x2_bf16x2_cvt(const uint32_t raw) {
  uint32_t two, r;
  asm("prmt.b32 %0, %1, 0, %2;" : "=r"(two) : "r"(raw), "n"(kSel));
  asm("{\n"
      ".reg .b16 c;\n"
      ".reg .b32 a;\n"
      ".reg .f16 a<2>;\n"
      ".reg .b16 b<2>;\n"
      "cvt.u16.u32 c, %1;\n"
      "cvt.rn.f16x2.e4m3x2 a, c;\n"
      "mov.b32 {a0, a1}, a;\n"
      "cvt.bf16.f16 b0, a0;\n"
      "cvt.bf16.f16 b1, a1;\n"
      "mov.b32 %0, {b0, b1};\n"
      "}"
      : "=r"(r)
      : "r"(two));
  return r;
}

SGL_DEVICE void nan_guard(uint32_t& guard, const uint32_t v) {
  asm("max.xorsign.abs.bf16x2 %0, %0, %1;" : "+r"(guard) : "r"(v));
}

// |lane| >= 480 (bf16 0x43f0) in either half: a NaN code went through the fast conversion
SGL_DEVICE bool nan_seen(const uint32_t guard) {
  return (guard & 0x7FFFu) >= 0x43F0u || ((guard >> 16) & 0x7FFFu) >= 0x43F0u;
}

// V tile row swizzle: the 8 rows read by one ldmatrix.trans matrix land on 8 different 16-byte bank groups
SGL_DEVICE int v_swz(const int r) {
  return ((r >> 1) & 6) | (r & 1);
}

struct Smem {
  int k_off, v_off, p_off, max_off, sum_off, mask_off, q_off, slot_off, bytes;
};

__host__ __device__ constexpr Smem smem_layout(const int stages, const int chunk, const bool q_smem) {
  Smem s{};
  s.k_off = 0;
  s.v_off = s.k_off + stages * kTile;
  s.p_off = s.v_off + stages * kTile;   // p tile [16 rows][32] bf16, 4 x 16-byte chunks per row, swizzled
  s.max_off = s.p_off + kGP * kBN * 2;  // [16 rows][4 warps] f32
  s.sum_off = s.max_off + kGP * 4 * 4;
  s.mask_off = s.sum_off + kGP * 4 * 4;   // per-block valid-column bit masks
  s.q_off = s.mask_off + kMaxBlocks * 4;  // q A fragments [chunk c][half h][lane] x 4 regs (q_smem only)
  s.slot_off = s.q_off + (q_smem ? 8 * 2 * 32 * 16 : 0);
  s.bytes = s.slot_off + chunk * 4;
  return s;
}

template <bool kRowReq64>
SGL_DEVICE int64_t load_row_req(const void* p, const int row) {
  if constexpr (kRowReq64) {
    return static_cast<const int64_t*>(p)[row];
  } else {
    return static_cast<const int32_t*>(p)[row];
  }
}

// cp.async the K and V rows of block b (32 columns) into stage buffers: thread -> 16-byte chunk j of rows
// tid / 16 + 8 i; invalid columns are zero-filled (src size 0)
SGL_DEVICE void issue_block(
    const Params& p, const int kvh, const int b, const uint8_t* k_st, const uint8_t* v_st, const int32_t* s_slot) {
  const int j = threadIdx.x & 15;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int r = (threadIdx.x >> 4) + 8 * i;
    const int slot = s_slot[b * kBN + r];
    const bool ok = slot >= 0;
    const int64_t off = static_cast<int64_t>(ok ? slot : 0) * p.stride_kslot + kvh * p.stride_kh + 16 * j;
    cp_async16(smem_u32(k_st + r * kD + ((j ^ (r & 7)) << 4)), p.k + off, ok ? 16 : 0);
    cp_async16(smem_u32(v_st + r * kD + ((j ^ v_swz(r)) << 4)), p.v + off, ok ? 16 : 0);
  }
}

template <int kStages, bool kQSmem, bool kRowReq64, bool kUsePDL>
__global__ void __launch_bounds__(kThreads) split_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int rh = blockIdx.x;
  const int split = blockIdx.y;
  const int row = rh / p.HKV;
  const int kvh = rh % p.HKV;
  const Smem L = smem_layout(kStages, p.chunk, kQSmem);
  uint8_t* const k_buf = smem + L.k_off;
  uint8_t* const v_buf = smem + L.v_off;
  uint8_t* const p_buf = smem + L.p_off;
  float* const red_max = reinterpret_cast<float*>(smem + L.max_off);
  float* const red_sum = reinterpret_cast<float*>(smem + L.sum_off);
  uint32_t* const s_mask = reinterpret_cast<uint32_t*>(smem + L.mask_off);
  int32_t* const s_slot = reinterpret_cast<int32_t*>(smem + L.slot_off);
  const int nblk = p.chunk / kBN;

  device::PDLWaitPrimary<kUsePDL>();

  // KV slot of each column of this split (-1 where invalid) and per-block valid-column masks
  {
    const int length = p.seqlen[row];
    const int64_t req = load_row_req<kRowReq64>(p.row_req, row);
    for (int i0 = 0; i0 < p.chunk; i0 += kThreads) {
      const int i = i0 + static_cast<int>(threadIdx.x);
      const int col = split * p.chunk + i;
      int slot = -1;
      if (i < p.chunk && col < p.topk) {
        const int pos = p.idx[row * p.stride_idx + col];
        if (pos >= 0 && pos < length) slot = p.req_to_token[req * p.stride_r2t + pos];
      }
      const bool ok = slot >= 0;
      if (i < p.chunk) s_slot[i] = ok ? slot : -1;
      const uint32_t bits = __ballot_sync(kFull, ok && i < p.chunk);
      if (lane == 0 && i0 + 32 * warp < p.chunk) s_mask[(i0 + 32 * warp) / kBN] = bits;
    }
  }

  // q A fragments (rows = q heads kvh * G + 0 .. 15, zero past G) for 32-d chunk c, half h:
  // {q[g][32c+4t+2h ..+1], q[g+8][same], q[g][32c+16+4t+2h ..+1], q[g+8][same]}. kQSmem keeps them in shared
  // memory (warp w loads chunks 2w, 2w + 1) to free registers.
  constexpr int kQRegChunks = kQSmem ? 1 : 8;
  uint32_t qa[kQRegChunks][2][4];
  uint4* const s_q = reinterpret_cast<uint4*>(smem + L.q_off);
  {
    const uint16_t* q0 = p.q + row * p.stride_qr + (kvh * p.G + g) * p.stride_qh;
    const uint16_t* q1 = q0 + 8 * p.stride_qh;
    const bool ok0 = g < p.G;
    const bool ok1 = g + 8 < p.G;
#pragma unroll
    for (int cc = 0; cc < (kQSmem ? 2 : 8); ++cc) {
      const int c = kQSmem ? 2 * warp + cc : cc;
      uint32_t frag[2][4];
#pragma unroll
      for (int part = 0; part < 2; ++part) {
        const int d = 32 * c + 16 * part + 4 * t;
        const uint2 v0 = ok0 ? *reinterpret_cast<const uint2*>(q0 + d) : make_uint2(0, 0);
        const uint2 v1 = ok1 ? *reinterpret_cast<const uint2*>(q1 + d) : make_uint2(0, 0);
        frag[0][2 * part + 0] = v0.x;
        frag[0][2 * part + 1] = v1.x;
        frag[1][2 * part + 0] = v0.y;
        frag[1][2 * part + 1] = v1.y;
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        if constexpr (kQSmem) {
          s_q[(c * 2 + h) * 32 + lane] = make_uint4(frag[h][0], frag[h][1], frag[h][2], frag[h][3]);
        } else {
#pragma unroll
          for (int r = 0; r < 4; ++r)
            qa[kQSmem ? 0 : c][h][r] = frag[h][r];
        }
      }
    }
  }
  __syncthreads();

#pragma unroll
  for (int b = 0; b < kStages - 1; ++b) {
    if (b < nblk && s_mask[b] != 0) issue_block(p, kvh, b, k_buf + b * kTile, v_buf + b * kTile, s_slot);
    cp_async_commit();
  }

  const float log2e = __int_as_float(0x3FB8AA3B);
  const float ninf = __int_as_float(0xFF800000);
  float m_i[2] = {ninf, ninf};  // rows g, g + 8
  float l_i[2] = {0.f, 0.f};
  float acc[4][2][4];  // [u: d 64w+16u ..][parity][c0..c3]
#pragma unroll
  for (int u = 0; u < 4; ++u)
#pragma unroll
    for (int e = 0; e < 2; ++e)
#pragma unroll
      for (int r = 0; r < 4; ++r)
        acc[u][e][r] = 0.f;

  const uint32_t p_base = smem_u32(p_buf);
  for (int b = 0; b < nblk; ++b) {
    cp_async_wait<kStages - 2>();
    __syncthreads();
    {
      const int nb = b + kStages - 1;
      const int st = nb % kStages;
      if (nb < nblk && s_mask[nb] != 0) issue_block(p, kvh, nb, k_buf + st * kTile, v_buf + st * kTile, s_slot);
      cp_async_commit();
    }
    const uint32_t bmask = s_mask[b];
    if (bmask == 0) continue;  // no valid column: m, l and acc are unchanged
    const int st = b % kStages;
    const uint8_t* k_st = k_buf + st * kTile;
    const uint8_t* v_st = v_buf + st * kTile;

    // ---- s = q k^T; warp w owns columns 8w .. 8w+7 (B fragments by ldmatrix: matrix = 8 columns x 16 d bytes)
    float s[4] = {0.f, 0.f, 0.f, 0.f};
    {
      const int kr = 8 * warp + (lane & 7);
      const uint32_t k_row = smem_u32(k_st + kr * kD);
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        uint32_t raw[4];
        ldsm_x4(k_row + (((4 * q4 + (lane >> 3)) ^ (kr & 7)) << 4), raw);
        uint32_t kb[2][2][2];  // [chunk 2q4 + cc][half][part]
        uint32_t guard = 0;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          kb[m >> 1][0][m & 1] = e4m3x2_bf16x2<0x9180>(raw[m]);
          kb[m >> 1][1][m & 1] = e4m3x2_bf16x2<0xB3A2>(raw[m]);
          nan_guard(guard, kb[m >> 1][0][m & 1]);
          nan_guard(guard, kb[m >> 1][1][m & 1]);
        }
        if (nan_seen(guard)) {
#pragma unroll
          for (int m = 0; m < 4; ++m) {
            kb[m >> 1][0][m & 1] = e4m3x2_bf16x2_cvt<0x0010>(raw[m]);
            kb[m >> 1][1][m & 1] = e4m3x2_bf16x2_cvt<0x0032>(raw[m]);
          }
        }
#pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          const int c = 2 * q4 + cc;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            if constexpr (kQSmem) {
              const uint4 a = s_q[(c * 2 + h) * 32 + lane];
              mma_bf16(s, a.x, a.y, a.z, a.w, kb[cc][h][0], kb[cc][h][1]);
            } else {
              mma_bf16(
                  s,
                  qa[kQSmem ? 0 : c][h][0],
                  qa[kQSmem ? 0 : c][h][1],
                  qa[kQSmem ? 0 : c][h][2],
                  qa[kQSmem ? 0 : c][h][3],
                  kb[cc][h][0],
                  kb[cc][h][1]);
            }
          }
        }
      }
    }

    // ---- online softmax over the block (rows g: s[0], s[1]; rows g + 8: s[2], s[3]; columns 8w + 2t, + 1);
    // the row max / sum go across warps through shared memory
    const int col0 = 8 * warp + 2 * t;
    const bool v0 = (bmask >> col0) & 1u;
    const bool v1 = (bmask >> (col0 + 1)) & 1u;
    float x[4];
    x[0] = v0 ? p.qk_scale * s[0] : ninf;
    x[1] = v1 ? p.qk_scale * s[1] : ninf;
    x[2] = v0 ? p.qk_scale * s[2] : ninf;
    x[3] = v1 ? p.qk_scale * s[3] : ninf;
    float wm[2] = {fmaxf(x[0], x[1]), fmaxf(x[2], x[3])};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      wm[r] = fmaxf(wm[r], __shfl_xor_sync(kFull, wm[r], 2));
      wm[r] = fmaxf(wm[r], __shfl_xor_sync(kFull, wm[r], 1));
    }
    if (t == 0) {
      red_max[g * 4 + warp] = wm[0];
      red_max[(g + 8) * 4 + warp] = wm[1];
    }
    __syncthreads();
    float m_safe[2], alpha[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float4 w4 = *reinterpret_cast<const float4*>(red_max + (g + 8 * r) * 4);
      const float mb = fmaxf(fmaxf(w4.x, w4.z), fmaxf(w4.y, w4.w));
      const float m_new = fmaxf(m_i[r], mb);
      m_safe[r] = m_new == ninf ? 0.f : m_new;
      alpha[r] = ex2_approx((m_i[r] - m_safe[r]) * log2e);
      m_i[r] = m_new;
    }
    float pr[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      pr[i] = ex2_approx((x[i] - m_safe[i >> 1]) * log2e);
    float ws[2] = {pr[0] + pr[1], pr[2] + pr[3]};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      ws[r] += __shfl_xor_sync(kFull, ws[r], 2);
      ws[r] += __shfl_xor_sync(kFull, ws[r], 1);
    }
    if (t == 0) {
      red_sum[g * 4 + warp] = ws[0];
      red_sum[(g + 8) * 4 + warp] = ws[1];
    }
    // p (bf16) into the A-operand tile: column c = 4a + e -> matrix (e >> 1) * 2 + (a >= 4), element 2 (a & 3) + (e &
    // 1)
    {
      const int a = col0 >> 2;
      const int mi = ((col0 & 3) >> 1) * 2 + (a >= 4 ? 1 : 0);
      const int el = 2 * (a & 3);
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int prow = g + 8 * r;
        uint32_t pk;
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(pk) : "f"(pr[2 * r + 1]), "f"(pr[2 * r]));
        const int chunk16 = mi ^ ((prow >> 1) & 3);
        *reinterpret_cast<uint32_t*>(p_buf + prow * 64 + chunk16 * 16 + el * 2) = pk;
      }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float4 w4 = *reinterpret_cast<const float4*>(red_sum + (g + 8 * r) * 4);
      const float S = (w4.x + w4.z) + (w4.y + w4.w);
      l_i[r] = fmaf(l_i[r], alpha[r], S);
    }
#pragma unroll
    for (int u = 0; u < 4; ++u)
#pragma unroll
      for (int e = 0; e < 2; ++e)
#pragma unroll
        for (int r = 0; r < 4; ++r)
          acc[u][e][r] *= alpha[r >> 1];

    // ---- p A fragments: x4 = {rows 0-7, rows 8-15} x {lo, hi} of MMA 1 (matrices 0, 1) and MMA 2 (2, 3)
    uint32_t pa[2][4];
    {
      const int prow = (lane & 7) + 8 * ((lane >> 3) & 1);
#pragma unroll
      for (int mm = 0; mm < 2; ++mm) {
        const int mi = 2 * mm + (lane >> 4);
        ldsm_x4(p_base + prow * 64 + ((mi ^ ((prow >> 1) & 3)) << 4), pa[mm]);
      }
    }
    // ---- acc += p v; warp w owns d 64w .. 64w+63. V B fragments come from ldmatrix.trans on byte pairs: a lane
    // gets two columns x two d, so one load feeds an even-d and an odd-d n8 tile. Matrix mi rows = columns
    // 16 (mi & 1) + 2 (mi >> 1) + 4 (i >> 1) + (i & 1)
    {
      const int mi = lane >> 3;
      const int i = lane & 7;
      const int vc = 16 * (mi & 1) + 2 * (mi >> 1) + 4 * (i >> 1) + (i & 1);
      const uint32_t v_row = smem_u32(v_st + vc * kD);
      const int sw = v_swz(vc);
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        uint32_t raw[4];
        ldsm_x4_trans(v_row + (((4 * warp + u) ^ sw) << 4), raw);
        uint32_t vb[2][4];  // [parity][matrix]
        uint32_t guard = 0;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          vb[0][m] = e4m3x2_bf16x2<0xA280>(raw[m]);
          vb[1][m] = e4m3x2_bf16x2<0xB391>(raw[m]);
          nan_guard(guard, vb[0][m]);
          nan_guard(guard, vb[1][m]);
        }
        if (nan_seen(guard)) {
#pragma unroll
          for (int m = 0; m < 4; ++m) {
            vb[0][m] = e4m3x2_bf16x2_cvt<0x0020>(raw[m]);
            vb[1][m] = e4m3x2_bf16x2_cvt<0x0031>(raw[m]);
          }
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          mma_bf16(acc[u][e], pa[0][0], pa[0][1], pa[0][2], pa[0][3], vb[e][0], vb[e][1]);
          mma_bf16(acc[u][e], pa[1][0], pa[1][1], pa[1][2], pa[1][3], vb[e][2], vb[e][3]);
        }
      }
    }
  }
  cp_async_wait<0>();

  // ---- partials: acc rows g, g + 8, d 64w + 16u + 4t + {0, 1, 2, 3}
  const int64_t base = (static_cast<int64_t>(rh) * kNSplit + split) * kGP;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    float* dst = p.part_acc + (base + g + 8 * r) * kD + 64 * warp + 4 * t;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      *reinterpret_cast<float4*>(dst + 16 * u) =
          make_float4(acc[u][0][2 * r], acc[u][1][2 * r], acc[u][0][2 * r + 1], acc[u][1][2 * r + 1]);
    }
  }
  if (warp == 0 && t == 0) {
    p.part_m[base + g] = m_i[0];
    p.part_m[base + g + 8] = m_i[1];
    p.part_l[base + g] = l_i[0];
    p.part_l[base + g + 8] = l_i[1];
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

// Merge the kNSplit partials of the G q heads of one (row, kv head): w_s = exp(m_s - max_s m_s),
// out = sum_s w_s acc_s / sum_s w_s l_s (0 when the row has no valid column)
template <bool kUsePDL>
__global__ void __launch_bounds__(kThreads) combine_kernel(const __grid_constant__ Params p) {
  __shared__ float s_w[kGP * kNSplit];
  __shared__ float s_den[kGP];
  const int rh = blockIdx.x;
  const int row = rh / p.HKV;
  const int kvh = rh % p.HKV;
  const float log2e = __int_as_float(0x3FB8AA3B);
  const float ninf = __int_as_float(0xFF800000);
  const int64_t base = static_cast<int64_t>(rh) * kNSplit * kGP;  // split s, head gi: base + s * kGP + gi

  device::PDLWaitPrimary<kUsePDL>();

  if (static_cast<int>(threadIdx.x) < p.G) {
    const int gi = threadIdx.x;
    float m[kNSplit], w[kNSplit];
    float mx = ninf;
#pragma unroll
    for (int s = 0; s < kNSplit; ++s) {
      m[s] = p.part_m[base + s * kGP + gi];
      mx = fmaxf(mx, m[s]);
    }
    const float m_safe = mx == ninf ? 0.f : mx;
    float den = 0.f;
#pragma unroll
    for (int s = 0; s < kNSplit; ++s) {
      w[s] = ex2_approx((m[s] - m_safe) * log2e);
      den = fmaf(w[s], p.part_l[base + s * kGP + gi], den);
      s_w[gi * kNSplit + s] = w[s];
    }
    s_den[gi] = den;
  }
  __syncthreads();
  // thread -> d pair (2 tid, 2 tid + 1), every head
  const int d = 2 * static_cast<int>(threadIdx.x);
  for (int gi = 0; gi < p.G; ++gi) {
    const float* w = s_w + gi * kNSplit;
    float2 o = make_float2(0.f, 0.f);
#pragma unroll
    for (int s = 0; s < kNSplit; ++s) {
      const float2 a = *reinterpret_cast<const float2*>(p.part_acc + (base + s * kGP + gi) * kD + d);
      o.x = fmaf(w[s], a.x, o.x);
      o.y = fmaf(w[s], a.y, o.y);
    }
    const float den = s_den[gi];
    const float inv = den > 0.f ? 1.f / den : 0.f;
    uint32_t pk;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(pk) : "f"(o.y * inv), "f"(o.x * inv));
    *reinterpret_cast<uint32_t*>(p.out + row * p.stride_or + (kvh * p.G + gi) * p.stride_oh + d) = pk;
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace qsa_sparse_decode

template <int kStages, bool kQSmem, bool kUsePDL>
struct QsaSparseDecode {
  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView req_to_token,
      const tvm::ffi::TensorView row_req,
      const tvm::ffi::TensorView idx,
      const tvm::ffi::TensorView seqlen,
      const tvm::ffi::TensorView part_acc,
      const tvm::ffi::TensorView part_m,
      const tvm::ffi::TensorView part_l,
      const tvm::ffi::TensorView out,
      double qk_scale) {
    using namespace host;
    using namespace qsa_sparse_decode;
    auto R_ = SymbolicSize{"rows"};
    auto HQ_ = SymbolicSize{"q_heads"};
    auto S_ = SymbolicSize{"slots"};
    auto HKV_ = SymbolicSize{"kv_heads"};
    auto NR_ = SymbolicSize{"reqs"};
    auto CTX_ = SymbolicSize{"ctx"};
    auto TOPK_ = SymbolicSize{"topk"};
    auto N_ = SymbolicSize{"partials"};
    auto NA_ = SymbolicSize{"partial_acc"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R_, HQ_, kD}).with_dtype<bf16_t>().with_device(device).with_strides({-1, -1, 1}).verify(q);
    // fp8 e4m3 pools passed as uint8 views
    TensorMatcher({S_, HKV_, kD}).with_dtype<uint8_t>().with_device(device).with_strides({-1, -1, 1}).verify(k);
    TensorMatcher({S_, HKV_, kD}).with_dtype<uint8_t>().with_device(device).with_strides({-1, -1, 1}).verify(v);
    TensorMatcher({NR_, CTX_}).with_dtype<int32_t>().with_device(device).with_strides({-1, 1}).verify(req_to_token);
    TensorMatcher({R_}).with_dtype<int32_t, int64_t>().with_device(device).with_strides({1}).verify(row_req);
    TensorMatcher({R_, TOPK_}).with_dtype<int32_t>().with_device(device).with_strides({-1, 1}).verify(idx);
    TensorMatcher({R_}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(seqlen);
    TensorMatcher({N_}).with_dtype<fp32_t>().with_device(device).with_strides({1}).verify(part_m).verify(part_l);
    TensorMatcher({NA_}).with_dtype<fp32_t>().with_device(device).with_strides({1}).verify(part_acc);
    TensorMatcher({R_, HQ_, kD}).with_dtype<bf16_t>().with_device(device).with_strides({-1, -1, 1}).verify(out);
    const int64_t rows = R_.unwrap();
    const int64_t hq = HQ_.unwrap();
    const int64_t hkv = HKV_.unwrap();
    const int64_t topk = TOPK_.unwrap();
    RuntimeCheck(hkv > 0 && hq % hkv == 0 && hq / hkv <= kGP, "q heads per kv head must be <= 16");
    RuntimeCheck(k.stride(0) == v.stride(0) && k.stride(1) == v.stride(1), "K and V pools must share strides");
    const int64_t chunk = (((topk + kBN - 1) / kBN + kNSplit - 1) / kNSplit) * kBN;
    RuntimeCheck(topk > 0 && chunk <= kMaxBlocks * kBN, "top-k width must be in [1, 16384]");
    RuntimeCheck(N_.unwrap() >= rows * hkv * kNSplit * kGP, "part_m / part_l too small");
    RuntimeCheck(NA_.unwrap() >= rows * hkv * kNSplit * kGP * kD, "part_acc too small");
    const auto a16 = [](const void* ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; };
    RuntimeCheck(
        a16(k.data_ptr()) && a16(v.data_ptr()) && k.stride(0) % 16 == 0 && k.stride(1) % 16 == 0,
        "K / V rows must be 16-byte aligned");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(q.data_ptr()) % 8 == 0 && q.stride(0) % 4 == 0 && q.stride(1) % 4 == 0,
        "q rows must be 8-byte aligned");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(out.data_ptr()) % 4 == 0 && out.stride(0) % 2 == 0 && out.stride(1) % 2 == 0,
        "out rows must be 4-byte aligned");
    RuntimeCheck(a16(part_acc.data_ptr()), "part_acc must be 16-byte aligned");
    if (rows == 0) return;
    const auto params = Params{
        .q = static_cast<const uint16_t*>(q.data_ptr()),
        .k = static_cast<const uint8_t*>(k.data_ptr()),
        .v = static_cast<const uint8_t*>(v.data_ptr()),
        .req_to_token = static_cast<const int32_t*>(req_to_token.data_ptr()),
        .row_req = row_req.data_ptr(),
        .idx = static_cast<const int32_t*>(idx.data_ptr()),
        .seqlen = static_cast<const int32_t*>(seqlen.data_ptr()),
        .part_acc = static_cast<float*>(part_acc.data_ptr()),
        .part_m = static_cast<float*>(part_m.data_ptr()),
        .part_l = static_cast<float*>(part_l.data_ptr()),
        .out = static_cast<uint16_t*>(out.data_ptr()),
        .stride_qr = q.stride(0),
        .stride_qh = q.stride(1),
        .stride_kslot = k.stride(0),
        .stride_kh = k.stride(1),
        .stride_r2t = req_to_token.stride(0),
        .stride_idx = idx.stride(0),
        .stride_or = out.stride(0),
        .stride_oh = out.stride(1),
        .qk_scale = static_cast<float>(qk_scale),
        .topk = static_cast<int32_t>(topk),
        .chunk = static_cast<int32_t>(chunk),
        .G = static_cast<int32_t>(hq / hkv),
        .HKV = static_cast<int32_t>(hkv),
    };
    const int smem = smem_layout(kStages, static_cast<int>(chunk), kQSmem).bytes;
    const DLDevice dev = device.unwrap();
    if (is_type<int64_t>(row_req.dtype())) {
      launch_split<true>(rows * hkv, smem, params, dev);
    } else {
      launch_split<false>(rows * hkv, smem, params, dev);
    }
    LaunchKernel(static_cast<uint32_t>(rows * hkv), kThreads, dev).enable_pdl(kUsePDL)(combine_kernel<kUsePDL>, params);
  }

 private:
  template <bool kRowReq64>
  static void
  launch_split(const int64_t row_heads, const int smem, const qsa_sparse_decode::Params& params, const DLDevice dev) {
    using namespace host;
    const auto kernel = qsa_sparse_decode::split_kernel<kStages, kQSmem, kRowReq64, kUsePDL>;
    static int configured[64] = {};  // per device: the opt-in dynamic shared memory size set so far
    int& done = configured[dev.device_id & 63];
    if (smem > done) {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      done = smem;
    }
    const dim3 grid(static_cast<uint32_t>(row_heads), static_cast<uint32_t>(qsa_sparse_decode::kNSplit));
    LaunchKernel(grid, qsa_sparse_decode::kThreads, dev, static_cast<std::size_t>(smem))
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
