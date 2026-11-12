// SPDX-License-Identifier: Apache-2.0
// Decode / MTP verify attention prologue, equal to the Triton _attention_prologue
// in prologue.py with COMPACT: K/V short convolutions (decode: conv cache shifted,
// tracked rows copied; verify: over the verify tokens, conv windows saved), K L2
// clip and V RMSNorm, Q L2 clip (head groups optionally padded), and the rel L2
// clip with the rel-logits projection. Verify can write the rel bias in the layout
// FA4's ShearingBias builds (seqlens_k set) instead of compact.

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>

namespace sglang::inkling2_prologue {

using bf16 = __nv_bfloat16;

constexpr int kThreads = 128;
constexpr int kD = 128;  // head dim
constexpr int kRelWorkers = 4;

__device__ __forceinline__ float bf(float v) {
  return __bfloat162float(__float2bfloat16_rn(v));
}

__device__ __forceinline__ float rsqrt_approx(float x) {
  float y;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// Butterfly over 8 lanes (xor 4, 2, 1), as _lane_sum8.
__device__ __forceinline__ float lane_sum8(float v) {
  v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFF, v, 4));
  v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFF, v, 2));
  return __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFF, v, 1));
}

__device__ __forceinline__ float clip_coeff(float total, float threshold, float eps) {
  float const norm = total > 0.0f ? __fmul_rn(total, rsqrt_approx(total)) : 0.0f;
  float const coeff = __fdiv_rn(threshold, __fadd_rn(eps, norm));
  return coeff >= 1.0f ? 1.0f : coeff;
}

struct Params {
  bf16 const *q, *k, *v, *r;  // projected slices, row stride sp
  int64_t sp;
  void const *k_weight, *v_weight;
  int64_t sw0, sw1;
  void const *k_dynamic, *v_dynamic;  // bf16 (tokens, n) or FP32 split-K partials
  int dyn_groups, dyn_rows;
  bf16 *k_cache, *v_cache;
  int64_t sc0, sc1;
  int64_t const* indices;
  bool const* mask;
  bool const* track_mask;
  int64_t const* track_indices;
  int64_t const* cu;
  int32_t const* seq;
  int64_t const* positions;
  void const *q_scale, *r_scale;
  bf16 const* rel_proj;
  bf16 *out_q, *out_k, *out_v, *out_rel;
  bf16 *k_window, *v_window;
  int64_t swb, swt, sww;
  float qk_threshold, rel_threshold, clip_eps, norm_eps;
  float tau_floor_rcp, tau_alpha;
  int32_t const* seqlens_k;  // FA4 cache_seqlens; non-null: sheared rel rows
  int window_left;           // < 0: none
  float left_pad;
};

template <typename T>
__device__ __forceinline__ float to_f(T v) {
  if constexpr (sizeof(T) == 2)
    return __bfloat162float(v);
  else
    return v;
}

__device__ __forceinline__ float2 load2(bf16 const* p) {
  return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(p));
}

__device__ __forceinline__ void store2(bf16* p, float a, float b) {
  *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a, b);
}

// 8 bf16 at p (16-byte aligned) as floats.
__device__ __forceinline__ void load8(bf16 const* p, float (&v)[8]) {
  uint4 const raw = *reinterpret_cast<uint4 const*>(p);
  bf16 const* h = reinterpret_cast<bf16 const*>(&raw);
#pragma unroll
  for (int i = 0; i < 8; ++i)
    v[i] = __bfloat162float(h[i]);
}

__device__ __forceinline__ void store8(bf16* p, float const (&v)[8]) {
  uint4 raw;
  bf16* h = reinterpret_cast<bf16*>(&raw);
#pragma unroll
  for (int i = 0; i < 8; ++i)
    h[i] = __float2bfloat16_rn(v[i]);
  *reinterpret_cast<uint4*>(p) = raw;
}

template <typename T>
__device__ __forceinline__ void load8_scale(void const* base, int offset, float (&v)[8]) {
  if constexpr (sizeof(T) == 2) {
    load8(static_cast<bf16 const*>(base) + offset, v);
  } else {
    float4 const* p4 = reinterpret_cast<float4 const*>(static_cast<float const*>(base) + offset);
    float4 const a = p4[0], b = p4[1];
    v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w;
    v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
  }
}

// mma.sync m16n8k16, bf16 inputs, FP32 accumulate from zero.
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t const (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
}

__device__ __forceinline__ uint32_t pack(bf16 lo, bf16 hi) {
  return uint32_t(__bfloat16_as_ushort(lo)) | (uint32_t(__bfloat16_as_ushort(hi)) << 16);
}

// HEADS query heads and KV_HEADS kv heads per rank; query heads padded from
// GROUP to PADDED_GROUP per kv head; D_REL = 16; REL_EXTENT rel columns.
// Grid (tokens, 2 + kRelWorkers): part 0 K/V, part 1 Q, then rel workers.
// DECODE rows are one token per sequence (row = sequence); otherwise MTP verify.
template <
    int HEADS,
    int KV_HEADS,
    int PADDED_GROUP,
    int REL_EXTENT,
    int W,
    bool DYNAMIC,
    bool TAU,
    bool DECODE,
    typename TW,
    typename TS>
__global__ void __launch_bounds__(kThreads) prologue_kernel(Params p) {
  constexpr int kGroup = HEADS / KV_HEADS;
  constexpr int kOutHeads = KV_HEADS * PADDED_GROUP;
  constexpr int kKvWidth = KV_HEADS * kD;
  constexpr int kDRel = 16;
  constexpr int kDyn = KV_HEADS * W;  // dynamic weights per token and tensor
  constexpr int kChunks = REL_EXTENT / 128;
  constexpr int kSteps = (kChunks + kRelWorkers - 1) / kRelWorkers;
  static_assert(kOutHeads <= 16 && (KV_HEADS == 1 || KV_HEADS == 2));
  static_assert(W == 4 && HEADS * 8 <= kThreads && REL_EXTENT % 128 == 0);
  // (W * (2 if DYNAMIC else 1) + 1) ** -0.5 rounded to FP32
  constexpr float kScale = DYNAMIC ? 0.3333333432674408f : 0.4472135901451111f;
  __shared__ float conv[2][kKvWidth];
  __shared__ float coeff[2 * KV_HEADS];
  __shared__ float dyn[2][kDyn];
  __shared__ __align__(16) bf16 r_tile[16][kDRel];
  __shared__ __align__(16) bf16 proj[kDRel][kSteps * 128 + 8];
  __shared__ __align__(16) bf16 out_tile[16][kSteps * 128 + 8];
  int const row = blockIdx.x, part = blockIdx.y, t = threadIdx.x;
  int const lane = t % 32;

  // Before the wait: metadata, weights, scales, the rel projection, and the conv
  // cache, none of which the preceding kernel writes.
  int const seq = DECODE ? row : p.seq[row];
  int const bos = DECODE ? row : int(p.cu[seq]);
  int64_t const slot = p.indices[seq];
  bool const in_pool = p.mask[seq];
  bool const has_cache = in_pool && slot > 0;
  float tau = 1.0f;
  if constexpr (TAU) {
    float const ratio = fmaxf(__fmul_rn(float(p.positions[row] + 1), p.tau_floor_rcp), 1.0f);
    tau = __fadd_rn(1.0f, __fmul_rn(p.tau_alpha, logf(ratio)));
  }

  if (part == 0) {
    int const col = 2 * t;
    bool const active = col < kKvWidth;  // 1 KV head: the upper half of the block idles
    bf16 const* x[2] = {p.k, p.v};
    bf16* cache[2] = {p.k_cache, p.v_cache};
    bf16* window[2] = {p.k_window, p.v_window};
    TW const* w[2] = {static_cast<TW const*>(p.k_weight), static_cast<TW const*>(p.v_weight)};
    float weight[2][W][2] = {};
    float2 prefix[2][W] = {};
    if (active)
#pragma unroll
      for (int s = 0; s < 2; ++s)
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int tap = 0; tap < W; ++tap) {
            int const lag = DYNAMIC ? tap : W - 1 - tap;
            weight[s][tap][i] =
                DYNAMIC ? to_f(w[s][lag * p.sw0 + (col + i) * p.sw1]) : bf(to_f(w[s][(col + i) * p.sw0 + tap * p.sw1]));
          }
    if (active)
#pragma unroll
      for (int s = 0; s < 2; ++s)
#pragma unroll
        for (int lag = 1; lag < W; ++lag) {
          int const src = row - lag;
          if constexpr (DECODE)
            prefix[s][lag] = has_cache ? load2(cache[s] + slot * p.sc0 + int64_t(W - 1 - lag) * p.sc1 + col)
                                       : make_float2(0.0f, 0.0f);
          else
            prefix[s][lag] = src < bos && in_pool
                                 ? load2(cache[s] + slot * p.sc0 + int64_t(src - bos + W - 1) * p.sc1 + col)
                                 : make_float2(0.0f, 0.0f);
        }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
    if constexpr (DYNAMIC) {
      if (t < 2 * kDyn) {
        int const s = t / kDyn, n = t % kDyn;
        void const* d = s ? p.v_dynamic : p.k_dynamic;
        float value;
        if (p.dyn_groups) {
          float const* parts = static_cast<float const*>(d);
          float total = parts[int64_t(row) * kDyn + n];
          for (int g = 1; g < p.dyn_groups; ++g)
            total = __fadd_rn(total, parts[(int64_t(g) * p.dyn_rows + row) * kDyn + n]);
          value = bf(total);
        } else {
          value = __bfloat162float(static_cast<bf16 const*>(d)[int64_t(row) * kDyn + n]);
        }
        dyn[s][n] = value;
      }
    }
    float2 xs[2][W] = {};
    if (active)
#pragma unroll
      for (int s = 0; s < 2; ++s)
#pragma unroll
        for (int lag = 0; lag < W; ++lag)
          xs[s][lag] = load2(x[s] + int64_t(DECODE ? row : max(row - lag, 0)) * p.sp + col);
    if constexpr (DYNAMIC) __syncthreads();
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      if (!active) break;
      float acc[2] = {0.0f, 0.0f};
#pragma unroll
      for (int tap = 0; tap < W; ++tap) {
        int const lag = DYNAMIC ? tap : W - 1 - tap;
        bool const from_x = row - lag >= bos;
        float2 const raw = from_x ? xs[s][lag] : prefix[s][lag];
        float const value[2] = {from_x || has_cache ? raw.x : 0.0f, from_x || has_cache ? raw.y : 0.0f};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          float wt = weight[s][tap][i];
          if constexpr (DYNAMIC) wt = __fadd_rn(wt, dyn[s][(col / kD) * W + lag]);
          acc[i] = (DYNAMIC && tap == 0) ? __fmul_rn(value[i], wt) : __fmaf_rn(value[i], wt, acc[i]);
        }
        if (!DECODE && lag < W - 1 && in_pool)
          store2(window[s] + seq * p.swb + int64_t(row - bos) * p.swt + (W - 2 - lag) * p.sww + col, raw.x, raw.y);
      }
      if constexpr (DECODE) {
        if (has_cache) {
          bool const track = p.track_mask != nullptr && p.track_mask[row];
          int64_t const dest = track ? p.track_indices[row] : 0;
#pragma unroll
          for (int tap = 0; tap < W - 1; ++tap) {
            float2 const value = tap == W - 2 ? xs[s][0] : prefix[s][W - 2 - tap];
            store2(cache[s] + slot * p.sc0 + int64_t(tap) * p.sc1 + col, value.x, value.y);
            if (track) store2(cache[s] + dest * p.sc0 + int64_t(tap) * p.sc1 + col, value.x, value.y);
          }
        }
      }
      conv[s][col] = bf(__fmul_rn(__fadd_rn(acc[0], xs[s][0].x), kScale));
      conv[s][col + 1] = bf(__fmul_rn(__fadd_rn(acc[1], xs[s][0].y), kScale));
    }
    __syncthreads();
    if (t < 32) {
      // K L2 clip: 8 lanes per head, lane l owning [8l, 8l + 8) and [64 + 8l, ...).
      int const head = t / 8, l = t % 8;
      float acc = 0.0f;
      if (head < KV_HEADS) {
#pragma unroll
        for (int half = 0; half < 2; ++half)
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            float const v = conv[0][head * kD + half * 64 + l * 8 + j];
            acc = __fmaf_rn(v, v, acc);
          }
      }
      float const total = lane_sum8(acc);
      if (l == 0 && head < KV_HEADS) coeff[head] = clip_coeff(total, p.qk_threshold, p.clip_eps);
    } else if (t < 64) {
      // V RMSNorm: 16 lanes per head, lane l owning [4l, 4l + 4) and [64 + 4l, ...).
      int const head = (t - 32) / 16, l = (t - 32) % 16;
      float acc = 0.0f;
      if (head < KV_HEADS) {
#pragma unroll
        for (int half = 0; half < 2; ++half)
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            float const v = conv[1][head * kD + half * 64 + l * 4 + j];
            acc = __fmaf_rn(v, v, acc);
          }
      }
      acc = __fadd_rn(acc, __shfl_xor_sync(0xFFFFFFFF, acc, 8));
      float const total = lane_sum8(acc);
      if (l == 0 && head < KV_HEADS) coeff[KV_HEADS + head] = rsqrt_approx(__fmaf_rn(total, 0.0078125f, p.norm_eps));
    }
    __syncthreads();
    if (!active) return;
    int const head = col / kD;
    store2(
        p.out_k + int64_t(row) * kKvWidth + col,
        __fmul_rn(conv[0][col], coeff[head]),
        __fmul_rn(conv[0][col + 1], coeff[head]));
    store2(
        p.out_v + int64_t(row) * kKvWidth + col,
        __fmul_rn(conv[1][col], coeff[KV_HEADS + head]),
        __fmul_rn(conv[1][col + 1], coeff[KV_HEADS + head]));
  } else if (part == 1) {
    // Q L2 clip (scaled, tau), written with padded head groups; 8 lanes per
    // head, lane l owning [8l, 8l + 8) and [64 + 8l, 64 + 8l + 8).
    int const head = t / 8, l = t % 8;
    bf16* out = p.out_q + int64_t(row) * kOutHeads * kD;
    float scale[2][8];
    if (t < HEADS * 8) {
#pragma unroll
      for (int half = 0; half < 2; ++half)
        load8_scale<TS>(p.q_scale, head * kD + half * 64 + l * 8, scale[half]);
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
    if (t < HEADS * 8) {
      bf16 const* x = p.q + int64_t(row) * p.sp + head * kD;
      float v[2][8];
      float acc = 0.0f;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        load8(x + half * 64 + l * 8, v[half]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          v[half][j] = __fmul_rn(scale[half][j], v[half][j]);
          acc = __fmaf_rn(v[half][j], v[half][j], acc);
        }
      }
      float const c = clip_coeff(lane_sum8(acc), p.qk_threshold, p.clip_eps);
      int const out_head = (head / kGroup) * PADDED_GROUP + head % kGroup;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float y[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          y[j] = __fmul_rn(v[half][j], c);
          if constexpr (TAU) y[j] = __fmul_rn(bf(y[j]), tau);
        }
        store8(out + out_head * kD + half * 64 + l * 8, y);
      }
    } else {
      constexpr int kPad = PADDED_GROUP - kGroup;
      constexpr int kPadVecs = KV_HEADS * kPad * kD / 8;
      float const zero[8] = {};
      for (int i = t - HEADS * 8; i < kPadVecs; i += kThreads - HEADS * 8) {
        int const pad = i / (kD / 8), c = i % (kD / 8) * 8;
        int const out_head = (pad / kPad) * PADDED_GROUP + kGroup + pad % kPad;
        store8(out + out_head * kD + c, zero);
      }
    }
  } else {
    int const worker = part - 2;
    int const col0 = worker * kSteps * 128;
    constexpr int kVecs = kDRel * kSteps * 128 / 8;
    for (int i = t; i < kVecs; i += kThreads) {
      int const k = i / (kSteps * 16), c = i % (kSteps * 16) * 8;
      if (col0 + c < REL_EXTENT)
        *reinterpret_cast<uint4*>(&proj[k][c]) =
            *reinterpret_cast<uint4 const*>(p.rel_proj + k * REL_EXTENT + col0 + c);
    }
    // Rel L2 clip on padded heads (8 lanes per head; D_REL = 16: lanes 0-1).
    int const out_head = t / 8, l = t % 8;
    int const slot_in_group = out_head % PADDED_GROUP;
    bool const real = out_head < kOutHeads && slot_in_group < kGroup && l < 2;
    int const head = (out_head / PADDED_GROUP) * kGroup + slot_in_group;
    float scale[8] = {};
    if (real) load8_scale<TS>(p.r_scale, head * kDRel + l * 8, scale);
    // Sheared rel rows (verify with seqlens_k): ShearingBias's BlockInfo bounds,
    // one packed-Q attention block per sequence. out[j] = rel[right - 1 - j] on
    // [left, right), -inf right of it, left_pad left of it. This worker's columns
    // c < cnt land at j = top - c, staged in out_tile at j - base8.
    constexpr int kPadded = REL_EXTENT + 256;
    constexpr int kBlocks = kPadded / 128;  // FA4 N tiles per sheared row
    bool const shear = !DECODE && p.seqlens_k != nullptr;
    int right = 0, left = 0, top = 0, cnt = 0, lo = 0, base8 = 0;
    if (shear) {
      int const qlen = int(p.cu[seq + 1]) - bos, sk = p.seqlens_k[seq];
      int const n_right = row - bos + 1 + sk - qlen;
      int const n_left = p.window_left < 0 ? 0 : max(n_right - 1 - p.window_left, 0);
      int const vals = n_right - max(n_left, n_right - REL_EXTENT);
      if (vals > 0) {
        int const n_block_max = min((sk + 127) >> 7, ((128 + PADDED_GROUP - 1) / PADDED_GROUP + sk - qlen + 127) >> 7);
        right = max(kBlocks - (n_block_max - ((n_right - 1) >> 7)), 0) * 128 + ((n_right - 1) & 127) + 1;
        left = max(right - vals, 0);
      }
      top = right - 1 - col0;
      cnt = max(0, min(min(min(REL_EXTENT, vals), right) - col0, kSteps * 128));
      lo = top - cnt + 1;
      base8 = cnt > 0 ? lo & ~7 : 0;
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
    float v[8] = {};
    if (real) {
      load8(p.r + int64_t(row) * p.sp + head * kDRel + l * 8, v);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        v[j] = __fmul_rn(scale[j], v[j]);
    }
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      acc = __fmaf_rn(v[j], v[j], acc);
    float const c = clip_coeff(lane_sum8(acc), p.rel_threshold, p.clip_eps);
    if (l < 2) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float y = __fmul_rn(v[j], c);
        if constexpr (TAU) y = __fmul_rn(bf(y), tau);
        r_tile[out_head][l * 8 + j] = __float2bfloat16_rn(y);
      }
    }
    __syncthreads();
    // rel = r_tile (16 x 16) @ rel_proj (16 x REL_EXTENT), this worker's chunks.
    int const warp = t / 32, gid = lane / 4, tig = lane % 4;
    uint32_t a[4];
    a[0] = pack(r_tile[gid][tig * 2], r_tile[gid][tig * 2 + 1]);
    a[1] = pack(r_tile[gid + 8][tig * 2], r_tile[gid + 8][tig * 2 + 1]);
    a[2] = pack(r_tile[gid][tig * 2 + 8], r_tile[gid][tig * 2 + 9]);
    a[3] = pack(r_tile[gid + 8][tig * 2 + 8], r_tile[gid + 8][tig * 2 + 9]);
#pragma unroll
    for (int step = 0; step < kSteps; ++step) {
      if (col0 + step * 128 >= REL_EXTENT) break;
#pragma unroll
      for (int tile = 0; tile < 4; ++tile) {
        int const n = step * 128 + (warp * 4 + tile) * 8;
        uint32_t const b0 = pack(proj[tig * 2][n + gid], proj[tig * 2 + 1][n + gid]);
        uint32_t const b1 = pack(proj[tig * 2 + 8][n + gid], proj[tig * 2 + 9][n + gid]);
        float c4[4];
        mma16816(c4, a, b0, b1);
        if (shear) {
          int const c = n + tig * 2, slot = top - base8 - c;
          if (slot & 1) {
            if (c + 1 < cnt) {
              store2(&out_tile[gid][slot - 1], c4[1], c4[0]);
              store2(&out_tile[gid + 8][slot - 1], c4[3], c4[2]);
            } else if (c < cnt) {
              out_tile[gid][slot] = __float2bfloat16_rn(c4[0]);
              out_tile[gid + 8][slot] = __float2bfloat16_rn(c4[2]);
            }
          } else {
#pragma unroll
            for (int e = 0; e < 4; ++e)
              if (c + (e & 1) < cnt) out_tile[gid + (e >> 1) * 8][slot - (e & 1)] = __float2bfloat16_rn(c4[e]);
          }
        } else {
          store2(&out_tile[gid][n + tig * 2], c4[0], c4[1]);
          store2(&out_tile[gid + 8][n + tig * 2], c4[2], c4[3]);
        }
      }
    }
    __syncthreads();
    if (!shear) {
      bf16* rel = p.out_rel + int64_t(row) * kOutHeads * REL_EXTENT;
      for (int i = t; i < kOutHeads * kSteps * 16; i += kThreads) {
        int const k = i / (kSteps * 16), c = i % (kSteps * 16) * 8;
        if (col0 + c < REL_EXTENT)
          *reinterpret_cast<uint4*>(rel + k * REL_EXTENT + col0 + c) = *reinterpret_cast<uint4 const*>(&out_tile[k][c]);
      }
      return;
    }
    bf16* rel = p.out_rel + int64_t(row) * kOutHeads * kPadded;
    // Whole 8-column groups inside [lo, top] as uint4, then the <= 7 + 7 edge columns,
    // then the 8-column pad groups touching [0, left) or [right, kPadded).
    if (cnt > 0) {
      int const gs = (lo - base8 + 7) >> 3, ge = (top + 1 - base8) >> 3;
      for (int i = t; i < kOutHeads * kSteps * 16; i += kThreads) {
        int const k = i / (kSteps * 16), g = gs + i % (kSteps * 16);
        if (g < ge)
          *reinterpret_cast<uint4*>(rel + k * kPadded + base8 + g * 8) =
              *reinterpret_cast<uint4 const*>(&out_tile[k][g * 8]);
      }
      int const left_end = ge > gs ? base8 + gs * 8 : top + 1, right_start = ge > gs ? base8 + ge * 8 : top + 1;
      for (int i = t; i < kOutHeads * 16; i += kThreads) {
        int const k = i >> 4, e = i & 15;
        int const j = ge <= gs ? lo + e : e < 7 ? lo + e : right_start + e - 7;
        bool const in = ge <= gs ? e < 14 && j <= top : e < 7 ? j < left_end && j <= top : e < 14 && j <= top;
        if (in) rel[k * kPadded + j] = out_tile[k][j - base8];
      }
    }
    bf16 const neg_inf = __float2bfloat16_rn(-INFINITY), lpad = __float2bfloat16_rn(p.left_pad);
    int const left_groups = (left + 7) >> 3, right_g0 = right >> 3;
    int const pad_groups = left_groups + kPadded / 8 - right_g0;
    for (int i = worker * kThreads + t; i < kOutHeads * pad_groups; i += kRelWorkers * kThreads) {
      int const k = i % kOutHeads, u = i / kOutHeads;
      int const j = (u < left_groups ? u : right_g0 + u - left_groups) * 8;
      bf16* dst = rel + k * kPadded + j;
      if (j >= right || j + 8 <= left) {
        bf16 const fill = j >= right ? neg_inf : lpad;
        uint4 raw;
        bf16* h = reinterpret_cast<bf16*>(&raw);
#pragma unroll
        for (int e = 0; e < 8; ++e)
          h[e] = fill;
        *reinterpret_cast<uint4*>(dst) = raw;
      } else {
#pragma unroll
        for (int e = 0; e < 8; ++e)
          if (j + e >= right)
            dst[e] = neg_inf;
          else if (j + e < left)
            dst[e] = lpad;
      }
    }
  }
}

template <
    int HEADS,
    int KV_HEADS,
    int PADDED_GROUP,
    int REL_EXTENT,
    int W,
    bool DYNAMIC,
    bool TAU,
    bool DECODE,
    typename TW,
    typename TS>
void run(
    tvm::ffi::TensorView q,
    int64_t k,
    int64_t v,
    int64_t r,
    int64_t sp,
    int64_t k_weight,
    int64_t v_weight,
    int64_t sw0,
    int64_t sw1,
    int64_t k_dynamic,
    int64_t v_dynamic,
    int64_t dyn_groups,
    int64_t k_cache,
    int64_t v_cache,
    int64_t sc0,
    int64_t sc1,
    int64_t indices,
    int64_t mask,
    int64_t track_mask,
    int64_t track_indices,
    int64_t cu,
    int64_t seq,
    int64_t positions,
    int64_t q_scale,
    int64_t r_scale,
    int64_t rel_proj,
    int64_t out_q,
    int64_t out_k,
    int64_t out_v,
    int64_t out_rel,
    int64_t k_window,
    int64_t v_window,
    int64_t swb,
    int64_t swt,
    int64_t sww,
    int64_t tokens,
    double qk_threshold,
    double rel_threshold,
    double clip_eps,
    double norm_eps,
    double tau_floor_rcp,
    double tau_alpha,
    int64_t seqlens_k,
    int64_t window_left,
    double left_pad) {
  auto ptr = [](int64_t a) { return reinterpret_cast<void*>(a); };
  Params p{
      static_cast<bf16 const*>(q.data_ptr()),
      static_cast<bf16 const*>(ptr(k)),
      static_cast<bf16 const*>(ptr(v)),
      static_cast<bf16 const*>(ptr(r)),
      sp,
      ptr(k_weight),
      ptr(v_weight),
      sw0,
      sw1,
      ptr(k_dynamic),
      ptr(v_dynamic),
      int(dyn_groups),
      int(tokens),
      static_cast<bf16*>(ptr(k_cache)),
      static_cast<bf16*>(ptr(v_cache)),
      sc0,
      sc1,
      static_cast<int64_t const*>(ptr(indices)),
      static_cast<bool const*>(ptr(mask)),
      static_cast<bool const*>(ptr(track_mask)),
      static_cast<int64_t const*>(ptr(track_indices)),
      static_cast<int64_t const*>(ptr(cu)),
      static_cast<int32_t const*>(ptr(seq)),
      static_cast<int64_t const*>(ptr(positions)),
      ptr(q_scale),
      ptr(r_scale),
      static_cast<bf16 const*>(ptr(rel_proj)),
      static_cast<bf16*>(ptr(out_q)),
      static_cast<bf16*>(ptr(out_k)),
      static_cast<bf16*>(ptr(out_v)),
      static_cast<bf16*>(ptr(out_rel)),
      static_cast<bf16*>(ptr(k_window)),
      static_cast<bf16*>(ptr(v_window)),
      swb,
      swt,
      sww,
      float(qk_threshold),
      float(rel_threshold),
      float(clip_eps),
      float(norm_eps),
      float(tau_floor_rcp),
      float(tau_alpha),
      static_cast<int32_t const*>(ptr(seqlens_k)),
      int(window_left),
      float(left_pad)};
  host::LaunchKernel(dim3(tokens, 2 + kRelWorkers), kThreads, q.device())
      .enable_pdl(true)(prologue_kernel<HEADS, KV_HEADS, PADDED_GROUP, REL_EXTENT, W, DYNAMIC, TAU, DECODE, TW, TS>, p);
}

}  // namespace sglang::inkling2_prologue
