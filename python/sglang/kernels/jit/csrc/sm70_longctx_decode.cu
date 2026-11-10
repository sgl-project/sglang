// SPDX-License-Identifier: Apache-2.0
// Long-context grouped decode for Volta (SM70), D256 G6 (H6/Hkv1), E5M2 KV.
//
// Read-once split-KV design: grid = (kv_heads, splits, batch). Each CTA owns
// one kv head and one context partition, streaming that partition's K/V once
// with 16-byte vectorized loads from the dense token-major layout. This keeps
// DRAM traffic at the theoretical minimum (unlike per-query-head partition
// schemes that re-read the same KV once per GQA head). A split-merge kernel
// (the existing TileLang combine) rescales the FP32 softmax states, with the
// same partial-output ABI. Numerics mirror _kernels_paged_decode.py: scores
// are scaled by softmax_scale*k_scale, probabilities are exp2((s-m)*scale*log2e),
// and the partial output is normalized by the partition sum and v_scale.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace sm70_longctx {

constexpr int kThreads = 512;
constexpr int kBlockN = 32;          // tokens per tile (smem-bounded)
constexpr int kDim = 256;
constexpr int kPairsPerLane = (kDim / 2) / 32;  // half2 pairs per lane (4)
constexpr int kAccPerLane = 2 * kPairsPerLane;  // fp32 accumulators per lane
constexpr int kGroup = 6;            // heads / kv_heads (TP4 GQA layout)
constexpr int kPageSize = 16;        // KV cache tokens per page
constexpr int kKVStride = 264;       // padded smem row stride (half units)
constexpr int kSmemKV = kBlockN * kKVStride;  // halves per K or V tile
constexpr float kLog2E = 1.4426950408889634f;

constexpr int kIndexerThreads = 256;
constexpr int kIndexerWarps = kIndexerThreads / 32;
constexpr int kIndexerHeads = 4;
constexpr int kIndexerDim = 128;
constexpr int kIndexerPageSize = 4;
constexpr int kIndexerKeysPerWarp = 8;
constexpr int kIndexerKeysPerBlock = kIndexerWarps * kIndexerKeysPerWarp;

// smem (bytes): ks + vs (fp16 tiles) + qs + scores (fp32) + probs (fp16)
constexpr int kSmemBytes = 2 * kSmemKV * sizeof(__half) +
                           kGroup * kDim * sizeof(__half) +
                           kGroup * kBlockN * sizeof(float) +
                           kGroup * kBlockN * sizeof(__half);

__device__ __forceinline__ void e5m2_to_fp16_16(const uint4 raw, half* dst) {
  // E5M2 shares sign/exponent bit positions with FP16: each byte shifted left
  // by 8 is an exact value conversion, and the 16 output halves must keep the
  // same dim order as the 16 input bytes. Each 32-bit input word b = [b0..b3]
  // produces one uint32 pair [fp16(b0), fp16(b1)] and one [fp16(b2), fp16(b3)];
  // the pairs go to consecutive uint32 slots so dim order is preserved.
  const uint32_t* r = reinterpret_cast<const uint32_t*>(&raw);
  uint32_t out[8];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const uint32_t b = r[w];
    out[2 * w] = (b & 0x000000ffu) << 8 | ((b >> 8) & 0x000000ffu) << 24;
    out[2 * w + 1] = ((b >> 16) & 0x000000ffu) << 8 |
                     ((b >> 24) & 0x000000ffu) << 24;
  }
  *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(out);
  *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(out + 4);
}

// Load one K/V token tile into shared memory. The cache element type selects
// the read path: the E5M2 byte cache is read as one 16-byte vector per thread
// and dequantized to FP16; an FP16 cache is read as two 16-byte vectors per
// thread and copied directly. Both write 16 FP16 values per thread to the same
// smem row layout, so the QK/softmax/PV math that reads ks/vs is dtype-agnostic.
template <typename KV>
__device__ __forceinline__ void load_kv_tile(const KV* __restrict__ k_cache,
                                             const KV* __restrict__ v_cache,
                                             const int* __restrict__ slots,
                                             const int tile_tokens, const int tid,
                                             __half* __restrict__ ks,
                                             __half* __restrict__ vs) {
  constexpr int kLoadIters = (kBlockN * (kDim / 16) + kThreads - 1) / kThreads;
#pragma unroll
  for (int it = 0; it < kLoadIters; ++it) {
    const int vector = tid + it * kThreads;
    const int off = vector << 4;       // 16 elements per thread
    const int tok_local = off >> 8;    // 256 elements per token row
    const int d_off = off & 255;       // dim offset, 16-aligned
    const int dst = tok_local * kKVStride + d_off;
    const int slot = slots[tok_local];
    uint4* dk = reinterpret_cast<uint4*>(ks + dst);
    uint4* dv = reinterpret_cast<uint4*>(vs + dst);
    if (tok_local < tile_tokens && slot >= 0) {
      const int64_t base = (int64_t)slot * kDim + d_off;
      if constexpr (sizeof(KV) == 1) {
        // E5M2 byte cache: base is in bytes, one 16-byte vector per thread.
        e5m2_to_fp16_16(*reinterpret_cast<const uint4*>(k_cache + base),
                        ks + dst);
        e5m2_to_fp16_16(*reinterpret_cast<const uint4*>(v_cache + base),
                        vs + dst);
      } else {
        // FP16 cache: base is in halves, two 16-byte vectors per thread.
        const uint4* sk = reinterpret_cast<const uint4*>(k_cache + base);
        const uint4* sv = reinterpret_cast<const uint4*>(v_cache + base);
        dk[0] = sk[0];
        dk[1] = sk[1];
        dv[0] = sv[0];
        dv[1] = sv[1];
      }
    } else {
      // Out-of-range token or invalid slot: zero the 16 FP16 smem slots,
      // matching the E5M2 path's zero dequant.
      const uint4 zero = make_uint4(0, 0, 0, 0);
      dk[0] = zero;
      dk[1] = zero;
      dv[0] = zero;
      dv[1] = zero;
    }
  }
}

__global__ void __launch_bounds__(kThreads, 1)
decode_partial_kernel(const __half* __restrict__ q,
                      const uint8_t* __restrict__ k_cache,
                      const uint8_t* __restrict__ v_cache,
                      const int* __restrict__ page_table,
                      const int* __restrict__ seq_lens, const int max_splits,
                      const int min_tokens_per_split, const int max_blocks,
                      const float score_scale, const float kv_scale,
                      __half* __restrict__ partial_o,
                      float* __restrict__ partial_lse) {
  const int kv_head = blockIdx.x;
  const int kv_heads = gridDim.x;
  const int kv_stride = kDim * kv_heads;   // per-token bytes across KV heads
  const int heads_total = kGroup * kv_heads;
  const int split_id = blockIdx.y;
  const int seq_id = blockIdx.z;

  __shared__ __half ks[kSmemKV];
  __shared__ __half vs[kSmemKV];
  __shared__ __half qs[kGroup * kDim];
  __shared__ float scores[kGroup * kBlockN];
  __shared__ __half probs[kGroup * kBlockN];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const bool is_compute_warp = warp < kGroup;

  const int context = seq_lens[seq_id];
  const int active_splits =
      min(max_splits, max(1, (context + min_tokens_per_split - 1) /
                                 min_tokens_per_split));
  if (split_id >= active_splits || context <= 0) {
    return;
  }
  const int split_len = (context + active_splits - 1) / active_splits;
  const int split_begin = split_id * split_len;
  const int split_end = min(context, split_begin + split_len);
  if (split_begin >= split_end) {
    return;
  }
  const float scale_log2 = score_scale * kLog2E;

  // Load the six query rows for this kv head into shared memory.
  for (int i = tid; i < kGroup * kDim; i += kThreads) {
    const int r = i / kDim;
    const int d = i % kDim;
    qs[r * kDim + d] = q[(kv_head * kGroup + r) * kDim + d];
  }

  float m_row[kGroup];
  float l_row[kGroup];
  float o_acc[kGroup][kAccPerLane];  // fp32 accumulators per lane (dim pair)
#pragma unroll
  for (int r = 0; r < kGroup; ++r) {
    m_row[r] = -1.0e30f;
    l_row[r] = 0.f;
  }
  if (is_compute_warp) {
#pragma unroll
    for (int j = 0; j < kAccPerLane; ++j) {
      o_acc[warp][j] = 0.f;
    }
  }
  __syncthreads();

  const int num_tiles = (split_end - split_begin + kBlockN - 1) / kBlockN;
  constexpr int kLoadIters = (kBlockN * (kDim / 16) + kThreads - 1) / kThreads;
  for (int tile = 0; tile < num_tiles; ++tile) {
    const int tile_begin = split_begin + tile * kBlockN;
    const int tile_tokens = min(kBlockN, split_end - tile_begin);

    // ---- Cooperative vectorized load + E5M2 -> FP16 convert (K and V) ----
    // kBlockN tokens * 256 bytes = 8KB/tensor. Each thread: 16B (one uint4)
    // per tensor. slot = tid + it*512; byte_off = slot*16; token = byte>>8;
    // d_off = byte & 255. Consecutive threads read consecutive 16B chunks of
    // a 256B token row -> fully coalesced 128B transactions.
#pragma unroll
    for (int it = 0; it < kLoadIters; ++it) {
      const int slot = tid + it * kThreads;   // 16-byte units [0, kBlockN*16)
      const int byte_off = slot << 4;
      const int tok_local = byte_off >> 8;
      const int d_off = byte_off & 255;
      const int token = tile_begin + tok_local;
      const bool valid = (token < split_end) && (tok_local < kBlockN);
      uint4 raw_k = make_uint4(0, 0, 0, 0);
      uint4 raw_v = make_uint4(0, 0, 0, 0);
      if (valid) {
        const int64_t base =
            page_table != nullptr
                ? (int64_t)page_table[seq_id * max_blocks + (token >> 4)] *
                          (kPageSize * kv_stride) +
                      (int64_t)(token & (kPageSize - 1)) * kv_stride +
                      kv_head * kDim + d_off
                : (int64_t)token * kv_stride + kv_head * kDim + d_off;
        raw_k = *reinterpret_cast<const uint4*>(k_cache + base);
        raw_v = *reinterpret_cast<const uint4*>(v_cache + base);
      }
      const int dst = tok_local * kKVStride + d_off;
      e5m2_to_fp16_16(raw_k, ks + dst);
      e5m2_to_fp16_16(raw_v, vs + dst);
    }
    __syncthreads();

    // ---- QK: each compute warp owns one query row ----
    if (is_compute_warp) {
      const int r = warp;
      const __half2* qr = reinterpret_cast<const __half2*>(qs + r * kDim);
      {
        const int n = lane;  // kBlockN == 32 == warp width
        const __half2* krow =
            reinterpret_cast<const __half2*>(ks + n * kKVStride);
        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
#pragma unroll
        for (int h = 0; h < kDim / 8; ++h) {
          const __half2 qh0 = qr[4 * h];
          const __half2 kh0 = krow[4 * h];
          const __half2 qh1 = qr[4 * h + 1];
          const __half2 kh1 = krow[4 * h + 1];
          const __half2 qh2 = qr[4 * h + 2];
          const __half2 kh2 = krow[4 * h + 2];
          const __half2 qh3 = qr[4 * h + 3];
          const __half2 kh3 = krow[4 * h + 3];
          acc0 = fmaf(__half2float(qh0.x), __half2float(kh0.x), acc0);
          acc0 = fmaf(__half2float(qh0.y), __half2float(kh0.y), acc0);
          acc1 = fmaf(__half2float(qh1.x), __half2float(kh1.x), acc1);
          acc1 = fmaf(__half2float(qh1.y), __half2float(kh1.y), acc1);
          acc2 = fmaf(__half2float(qh2.x), __half2float(kh2.x), acc2);
          acc2 = fmaf(__half2float(qh2.y), __half2float(kh2.y), acc2);
          acc3 = fmaf(__half2float(qh3.x), __half2float(kh3.x), acc3);
          acc3 = fmaf(__half2float(qh3.y), __half2float(kh3.y), acc3);
        }
        const float acc = acc0 + acc1 + acc2 + acc3;
        if (n < tile_tokens) {
          scores[r * kBlockN + n] = acc;
        }
      }
      __syncwarp();
      float loc_max = -1.0e30f;
      if (lane < tile_tokens) {
        loc_max = scores[r * kBlockN + lane];
      }
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        loc_max = fmaxf(loc_max, __shfl_xor_sync(0xffffffffu, loc_max, off));
      }
      const float m_new = fmaxf(m_row[r], loc_max);
      const float alpha = exp2f((m_row[r] - m_new) * scale_log2);
      m_row[r] = m_new;
      l_row[r] *= alpha;
      if (alpha != 1.f) {
#pragma unroll
        for (int j = 0; j < kAccPerLane; ++j) {
          o_acc[r][j] *= alpha;
        }
      }
      float loc_sum = 0.f;
      {
        const int n = lane;
        float p = 0.f;
        if (n < tile_tokens) {
          p = exp2f((scores[r * kBlockN + n] - m_new) * scale_log2);
          probs[r * kBlockN + n] = __float2half(p);
        }
        loc_sum = p;
      }
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        loc_sum += __shfl_xor_sync(0xffffffffu, loc_sum, off);
      }
      l_row[r] += loc_sum;
    }
    __syncthreads();

    // ---- PV: o_acc += p * V ----
    if (is_compute_warp) {
      const int r = warp;
#pragma unroll
      for (int n = 0; n < kBlockN; ++n) {
        const float p = __half2float(probs[r * kBlockN + n]);
        if (p == 0.f) {
          continue;
        }
        const __half2* vrow =
            reinterpret_cast<const __half2*>(vs + n * kKVStride);
#pragma unroll
        for (int j = 0; j < kPairsPerLane; ++j) {
          const __half2 v = vrow[lane + j * 32];
          o_acc[r][2 * j] = fmaf(p, __half2float(v.x), o_acc[r][2 * j]);
          o_acc[r][2 * j + 1] =
              fmaf(p, __half2float(v.y), o_acc[r][2 * j + 1]);
        }
      }
    }
    __syncthreads();
  }

  // ---- Finalize partial output for this partition ----
  if (is_compute_warp) {
    const int r = warp;
    const float inv_l = l_row[r] > 0.f ? 1.f / l_row[r] : 0.f;
    const int oh = kv_head * kGroup + r;
    __half* o_row = partial_o +
                    (((int64_t)seq_id * max_splits + split_id) * heads_total +
                     oh) *
                        kDim;
#pragma unroll
    for (int j = 0; j < kPairsPerLane; ++j) {
      o_row[2 * (lane + j * 32)] =
          __float2half_rn(o_acc[r][2 * j] * inv_l * kv_scale);
      o_row[2 * (lane + j * 32) + 1] =
          __float2half_rn(o_acc[r][2 * j + 1] * inv_l * kv_scale);
    }
  }
  if (is_compute_warp && lane == 0) {
    const int r = warp;
    const float lse =
        l_row[r] > 0.f ? __log2f(l_row[r]) + m_row[r] * scale_log2 : -1.0e30f;
    partial_lse[((int64_t)seq_id * max_splits + split_id) * heads_total +
                kv_head * kGroup + r] = lse;
  }
}

// Split-KV QSA decode for the TP4 H6/Hkv1/D256 layout.  Unlike the generic
// fallback, this resolves the selected logical positions while loading the
// cache and keeps K/V shared by all six query heads.  No FP16 compact-KV
// scratch is written or read.
template <typename KV>
__global__ void __launch_bounds__(kThreads, 1)
qsa_decode_partial_kernel(
    const __half* __restrict__ q, const KV* __restrict__ k_cache,
    const KV* __restrict__ v_cache,
    const int* __restrict__ req_to_token,
    const int* __restrict__ req_indices, const int* __restrict__ indices,
    const int* __restrict__ seq_lens, const int req_stride, const int topk,
    const int max_splits, const int min_tokens_per_split,
    const float score_scale, __half* __restrict__ partial_o,
    float* __restrict__ partial_lse) {
  const int split_id = blockIdx.y;
  const int seq_id = blockIdx.z;
  const int seq_len = seq_lens[seq_id];
  const int context = min(seq_len, topk);
  const int active_splits =
      min(max_splits, max(1, (context + min_tokens_per_split - 1) /
                                 min_tokens_per_split));
  if (split_id >= active_splits || context <= 0) {
    return;
  }
  const int split_len = (context + active_splits - 1) / active_splits;
  const int split_begin = split_id * split_len;
  const int split_end = min(context, split_begin + split_len);
  if (split_begin >= split_end) {
    return;
  }

  __shared__ __half ks[kSmemKV];
  __shared__ __half vs[kSmemKV];
  __shared__ __half qs[kGroup * kDim];
  __shared__ float scores[kGroup * kBlockN];
  __shared__ __half probs[kGroup * kBlockN];
  __shared__ int slots[kBlockN];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const bool is_compute_warp = warp < kGroup;
  const int req_index = req_indices[seq_id];
  const float scale_log2 = score_scale * kLog2E;

  for (int i = tid; i < kGroup * kDim; i += kThreads) {
    qs[i] = q[(int64_t)seq_id * kGroup * kDim + i];
  }

  // Each compute warp owns one query head. Thread-local arrays indexed by
  // the runtime warp ID force these accumulators into local memory on SM70.
  // Keep only this thread's head state so ptxas can retain it in registers.
  float m_row = -1.0e30f;
  float l_row = 0.f;
  float o_acc[kAccPerLane];
  if (is_compute_warp) {
#pragma unroll
    for (int j = 0; j < kAccPerLane; ++j) {
      o_acc[j] = 0.f;
    }
  }
  __syncthreads();

  const int num_tiles = (split_end - split_begin + kBlockN - 1) / kBlockN;
  for (int tile = 0; tile < num_tiles; ++tile) {
    const int tile_begin = split_begin + tile * kBlockN;
    const int tile_tokens = min(kBlockN, split_end - tile_begin);
    if (tid < kBlockN) {
      const int selected = tile_begin + tid;
      int slot = -1;
      if (selected < split_end) {
        const int logical = indices[(int64_t)seq_id * topk + selected];
        if (logical >= 0 && logical < seq_len) {
          slot = req_to_token[(int64_t)req_index * req_stride + logical];
        }
      }
      slots[tid] = slot;
    }
    __syncthreads();

    load_kv_tile<KV>(k_cache, v_cache, slots, tile_tokens, tid, ks, vs);
    __syncthreads();

    if (is_compute_warp) {
      const int r = warp;
      const __half2* qr = reinterpret_cast<const __half2*>(qs + r * kDim);
      const int n = lane;
      const __half2* krow =
          reinterpret_cast<const __half2*>(ks + n * kKVStride);
      float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
#pragma unroll
      for (int h = 0; h < kDim / 8; ++h) {
        const __half2 qh0 = qr[4 * h];
        const __half2 kh0 = krow[4 * h];
        const __half2 qh1 = qr[4 * h + 1];
        const __half2 kh1 = krow[4 * h + 1];
        const __half2 qh2 = qr[4 * h + 2];
        const __half2 kh2 = krow[4 * h + 2];
        const __half2 qh3 = qr[4 * h + 3];
        const __half2 kh3 = krow[4 * h + 3];
        acc0 = fmaf(__half2float(qh0.x), __half2float(kh0.x), acc0);
        acc0 = fmaf(__half2float(qh0.y), __half2float(kh0.y), acc0);
        acc1 = fmaf(__half2float(qh1.x), __half2float(kh1.x), acc1);
        acc1 = fmaf(__half2float(qh1.y), __half2float(kh1.y), acc1);
        acc2 = fmaf(__half2float(qh2.x), __half2float(kh2.x), acc2);
        acc2 = fmaf(__half2float(qh2.y), __half2float(kh2.y), acc2);
        acc3 = fmaf(__half2float(qh3.x), __half2float(kh3.x), acc3);
        acc3 = fmaf(__half2float(qh3.y), __half2float(kh3.y), acc3);
      }
      scores[r * kBlockN + n] =
          (n < tile_tokens && slots[n] >= 0) ? acc0 + acc1 + acc2 + acc3
                                             : -1.0e30f;
      __syncwarp();

      float loc_max = scores[r * kBlockN + lane];
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        loc_max = fmaxf(loc_max, __shfl_xor_sync(0xffffffffu, loc_max, off));
      }
      const float m_new = fmaxf(m_row, loc_max);
      const float alpha = exp2f((m_row - m_new) * scale_log2);
      m_row = m_new;
      l_row *= alpha;
      if (alpha != 1.f) {
#pragma unroll
        for (int j = 0; j < kAccPerLane; ++j) {
          o_acc[j] *= alpha;
        }
      }
      float p = 0.f;
      if (lane < tile_tokens && slots[lane] >= 0) {
        p = exp2f((scores[r * kBlockN + lane] - m_new) * scale_log2);
      }
      probs[r * kBlockN + lane] = __float2half(p);
      float loc_sum = p;
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        loc_sum += __shfl_xor_sync(0xffffffffu, loc_sum, off);
      }
      l_row += loc_sum;
    }
    __syncthreads();

    if (is_compute_warp) {
      const int r = warp;
#pragma unroll
      for (int n = 0; n < kBlockN; ++n) {
        const float p = __half2float(probs[r * kBlockN + n]);
        if (p == 0.f) {
          continue;
        }
        const __half2* vrow =
            reinterpret_cast<const __half2*>(vs + n * kKVStride);
#pragma unroll
        for (int j = 0; j < kPairsPerLane; ++j) {
          const __half2 value = vrow[lane + j * 32];
          o_acc[2 * j] =
              fmaf(p, __half2float(value.x), o_acc[2 * j]);
          o_acc[2 * j + 1] =
              fmaf(p, __half2float(value.y), o_acc[2 * j + 1]);
        }
      }
    }
    __syncthreads();
  }

  if (is_compute_warp) {
    const int r = warp;
    const float inv_l = l_row > 0.f ? 1.f / l_row : 0.f;
    __half* out_row =
        partial_o +
        (((int64_t)seq_id * max_splits + split_id) * kGroup + r) * kDim;
#pragma unroll
    for (int j = 0; j < kPairsPerLane; ++j) {
      out_row[2 * (lane + j * 32)] =
          __float2half_rn(o_acc[2 * j] * inv_l);
      out_row[2 * (lane + j * 32) + 1] =
          __float2half_rn(o_acc[2 * j + 1] * inv_l);
    }
    if (lane == 0) {
      partial_lse[((int64_t)seq_id * max_splits + split_id) * kGroup + r] =
          l_row > 0.f
              ? __log2f(l_row) + m_row * scale_log2
              : -1.0e30f;
    }
  }
}

// QSA indexer decode scoring for Qwen3.8.  A warp owns one compressed key,
// reads its 128 FP16 values once, and accumulates the four query-head scores
// in registers.  This avoids padding four real heads to Volta MMA's 16-head
// tile and emits the exact FP32 logits consumed by fast_topk.
__global__ void __launch_bounds__(kIndexerThreads, 2)
qsa_indexer_decode_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k_cache,
    const int* __restrict__ page_table,
    const int* __restrict__ context_lens, const int max_pages,
    const int max_model_len, const float score_scale,
    float* __restrict__ logits) {
  const int seq_id = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int context = context_lens[seq_id];
  const int block_begin = blockIdx.x * kIndexerKeysPerBlock;
  if (block_begin >= context) {
    return;
  }

  __shared__ __half qs[kIndexerHeads * kIndexerDim];
  for (int i = tid; i < kIndexerHeads * kIndexerDim; i += kIndexerThreads) {
    qs[i] = q[(int64_t)seq_id * kIndexerHeads * kIndexerDim + i];
  }
  __syncthreads();

#pragma unroll
  for (int it = 0; it < kIndexerKeysPerWarp; ++it) {
    const int position = block_begin + it * kIndexerWarps + warp;
    if (position >= context || position >= max_model_len) {
      continue;
    }
    const int logical_page = position / kIndexerPageSize;
    const int page_offset = position % kIndexerPageSize;
    const int physical_page =
        page_table[(int64_t)seq_id * max_pages + logical_page];
    const __half2* key = reinterpret_cast<const __half2*>(
        k_cache +
        ((int64_t)physical_page * kIndexerPageSize + page_offset) *
            kIndexerDim);
    const __half2 k0 = key[lane * 2];
    const __half2 k1 = key[lane * 2 + 1];
    float scores[kIndexerHeads];
#pragma unroll
    for (int h = 0; h < kIndexerHeads; ++h) {
      const __half2* qh =
          reinterpret_cast<const __half2*>(qs + h * kIndexerDim);
      const __half2 q0 = qh[lane * 2];
      const __half2 q1 = qh[lane * 2 + 1];
      float acc = 0.f;
      acc = fmaf(__half2float(q0.x), __half2float(k0.x), acc);
      acc = fmaf(__half2float(q0.y), __half2float(k0.y), acc);
      acc = fmaf(__half2float(q1.x), __half2float(k1.x), acc);
      acc = fmaf(__half2float(q1.y), __half2float(k1.y), acc);
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
      }
      scores[h] = acc;
    }
    if (lane == 0) {
      float score = 0.f;
#pragma unroll
      for (int h = 0; h < kIndexerHeads; ++h) {
        score += fmaxf(scores[h], 0.f);
      }
      logits[(int64_t)seq_id * max_model_len + position] = score / score_scale;
    }
  }
}

// QSA chunk-prefill for the exact TP4 full-attention shape.  One CTA owns one
// query token and evaluates all six GQA heads together, so every selected FP8
// K/V row is fetched and decoded once instead of once per query head.  The
// logical QSA indices are resolved through req_to_token in the load phase;
// this avoids materializing and converting the entire accumulated context on
// every 8K serving chunk.
template <typename KV>
__global__ void __launch_bounds__(kThreads, 1)
qsa_prefill_kernel(const __half* __restrict__ q,
                   const KV* __restrict__ k_cache,
                   const KV* __restrict__ v_cache,
                   const int* __restrict__ req_to_token,
                   const int* __restrict__ req_indices,
                   const int* __restrict__ indices,
                   const int* __restrict__ seq_lens, const int req_stride,
                   const int topk,
                   const float score_scale, __half* __restrict__ output) {
  const int query = blockIdx.x;
  const int req_index = req_indices[0];
  const int seq_len = seq_lens[0];
  // seq_len includes the complete serving chunk.  QSA stores each row's
  // causal candidates first, so mirror _sparse_gqa_chunk_prefill and consume
  // only the prefix visible to this query row.
  const int visible = min(seq_len, query + seq_len - gridDim.x + 1);
  const int row_topk = min(topk, max(0, visible));

  __shared__ __half ks[kSmemKV];
  __shared__ __half vs[kSmemKV];
  __shared__ __half qs[kGroup * kDim];
  __shared__ float scores[kGroup * kBlockN];
  __shared__ __half probs[kGroup * kBlockN];
  __shared__ int slots[kBlockN];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const bool is_compute_warp = warp < kGroup;
  const float scale_log2 = score_scale * kLog2E;

  for (int i = tid; i < kGroup * kDim; i += kThreads) {
    qs[i] = q[(int64_t)query * kGroup * kDim + i];
  }

  float m_row[kGroup];
  float l_row[kGroup];
  float o_acc[kGroup][kAccPerLane];
#pragma unroll
  for (int r = 0; r < kGroup; ++r) {
    m_row[r] = -1.0e30f;
    l_row[r] = 0.f;
  }
  if (is_compute_warp) {
#pragma unroll
    for (int j = 0; j < kAccPerLane; ++j) {
      o_acc[warp][j] = 0.f;
    }
  }
  __syncthreads();

  const int num_tiles = (row_topk + kBlockN - 1) / kBlockN;
  for (int tile = 0; tile < num_tiles; ++tile) {
    const int tile_begin = tile * kBlockN;
    const int tile_tokens = min(kBlockN, row_topk - tile_begin);
    if (tid < kBlockN) {
      const int selected = tile_begin + tid;
      int slot = -1;
      if (selected < row_topk) {
        const int logical = indices[(int64_t)query * topk + selected];
        if (logical >= 0 && logical < visible) {
          slot = req_to_token[(int64_t)req_index * req_stride + logical];
        }
      }
      slots[tid] = slot;
    }
    __syncthreads();

    load_kv_tile<KV>(k_cache, v_cache, slots, tile_tokens, tid, ks, vs);
    __syncthreads();

    if (is_compute_warp) {
      const int r = warp;
      const __half2* qr = reinterpret_cast<const __half2*>(qs + r * kDim);
      const int n = lane;
      const __half2* krow =
          reinterpret_cast<const __half2*>(ks + n * kKVStride);
      float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
#pragma unroll
      for (int h = 0; h < kDim / 8; ++h) {
        const __half2 qh0 = qr[4 * h];
        const __half2 kh0 = krow[4 * h];
        const __half2 qh1 = qr[4 * h + 1];
        const __half2 kh1 = krow[4 * h + 1];
        const __half2 qh2 = qr[4 * h + 2];
        const __half2 kh2 = krow[4 * h + 2];
        const __half2 qh3 = qr[4 * h + 3];
        const __half2 kh3 = krow[4 * h + 3];
        acc0 = fmaf(__half2float(qh0.x), __half2float(kh0.x), acc0);
        acc0 = fmaf(__half2float(qh0.y), __half2float(kh0.y), acc0);
        acc1 = fmaf(__half2float(qh1.x), __half2float(kh1.x), acc1);
        acc1 = fmaf(__half2float(qh1.y), __half2float(kh1.y), acc1);
        acc2 = fmaf(__half2float(qh2.x), __half2float(kh2.x), acc2);
        acc2 = fmaf(__half2float(qh2.y), __half2float(kh2.y), acc2);
        acc3 = fmaf(__half2float(qh3.x), __half2float(kh3.x), acc3);
        acc3 = fmaf(__half2float(qh3.y), __half2float(kh3.y), acc3);
      }
      scores[r * kBlockN + n] =
          (n < tile_tokens && slots[n] >= 0) ? acc0 + acc1 + acc2 + acc3
                                             : -1.0e30f;
      __syncwarp();

      float loc_max = scores[r * kBlockN + lane];
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        loc_max = fmaxf(loc_max, __shfl_xor_sync(0xffffffffu, loc_max, off));
      }
      const float m_new = fmaxf(m_row[r], loc_max);
      const float alpha = exp2f((m_row[r] - m_new) * scale_log2);
      m_row[r] = m_new;
      l_row[r] *= alpha;
      if (alpha != 1.f) {
#pragma unroll
        for (int j = 0; j < kAccPerLane; ++j) {
          o_acc[r][j] *= alpha;
        }
      }
      float p = 0.f;
      if (lane < tile_tokens && slots[lane] >= 0) {
        p = exp2f((scores[r * kBlockN + lane] - m_new) * scale_log2);
      }
      probs[r * kBlockN + lane] = __float2half(p);
      float loc_sum = p;
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        loc_sum += __shfl_xor_sync(0xffffffffu, loc_sum, off);
      }
      l_row[r] += loc_sum;
    }
    __syncthreads();

    if (is_compute_warp) {
      const int r = warp;
#pragma unroll
      for (int n = 0; n < kBlockN; ++n) {
        const float p = __half2float(probs[r * kBlockN + n]);
        if (p == 0.f) {
          continue;
        }
        const __half2* vrow =
            reinterpret_cast<const __half2*>(vs + n * kKVStride);
#pragma unroll
        for (int j = 0; j < kPairsPerLane; ++j) {
          const __half2 value = vrow[lane + j * 32];
          o_acc[r][2 * j] =
              fmaf(p, __half2float(value.x), o_acc[r][2 * j]);
          o_acc[r][2 * j + 1] =
              fmaf(p, __half2float(value.y), o_acc[r][2 * j + 1]);
        }
      }
    }
    __syncthreads();
  }

  if (is_compute_warp) {
    const int r = warp;
    const float inv_l = l_row[r] > 0.f ? 1.f / l_row[r] : 0.f;
    __half* out_row =
        output + ((int64_t)query * kGroup + r) * kDim;
#pragma unroll
    for (int j = 0; j < kPairsPerLane; ++j) {
      out_row[2 * (lane + j * 32)] =
          __float2half_rn(o_acc[r][2 * j] * inv_l);
      out_row[2 * (lane + j * 32) + 1] =
          __float2half_rn(o_acc[r][2 * j + 1] * inv_l);
    }
  }
}

}  // namespace sm70_longctx

void sm70_longctx_decode(torch::Tensor q, torch::Tensor k_cache,
                         torch::Tensor v_cache, torch::Tensor page_table,
                         torch::Tensor seq_lens, int64_t max_splits,
                         int64_t min_tokens_per_split, double softmax_scale,
                         double k_scale, double v_scale,
                         torch::Tensor partial_o,
                         torch::Tensor partial_lse) {
  using namespace sm70_longctx;
  c10::cuda::CUDAGuard guard(q.device());
  const int heads = q.size(1);
  const int heads_kv = k_cache.size(2);
  const int dim = q.size(2);
  const int batch = q.size(0);
  TORCH_CHECK(heads_kv >= 1 && heads == heads_kv * kGroup && dim == kDim,
              "sm70_longctx expects H6/D256 GQA with a full KV-head split "
              "(TP1 Hkv4, TP2 Hkv2, TP4 Hkv1)");
  TORCH_CHECK(k_cache.scalar_type() == torch::kUInt8 &&
                  v_cache.scalar_type() == torch::kUInt8,
              "sm70_longctx expects E5M2 byte KV cache");
  TORCH_CHECK(q.dtype() == torch::kHalf, "sm70_longctx expects fp16 queries");
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)heads_kv, (unsigned)max_splits, (unsigned)batch);
  const int* page_ptr = page_table.numel() > 0 ? page_table.data_ptr<int>()
                                               : nullptr;
  const int max_blocks = page_table.numel() > 0 ? (int)page_table.size(1) : 0;
  decode_partial_kernel<<<grid, sm70_longctx::kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr()),
      reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
      reinterpret_cast<const uint8_t*>(v_cache.data_ptr()), page_ptr,
      seq_lens.data_ptr<int>(), (int)max_splits, (int)min_tokens_per_split,
      max_blocks, (float)(softmax_scale * k_scale), (float)v_scale,
      reinterpret_cast<__half*>(partial_o.data_ptr()),
      partial_lse.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_qsa_decode(torch::Tensor q, torch::Tensor k_cache,
                     torch::Tensor v_cache, torch::Tensor req_to_token,
                     torch::Tensor req_indices, torch::Tensor indices,
                     torch::Tensor seq_lens, int64_t max_splits,
                     int64_t min_tokens_per_split, double softmax_scale,
                     torch::Tensor partial_o, torch::Tensor partial_lse) {
  using namespace sm70_longctx;
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.scalar_type() == torch::kHalf && q.dim() == 3 &&
                  q.size(1) == kGroup && q.size(2) == kDim,
              "sm70_qsa_decode expects FP16 Q [batch,6,256]");
  const bool is_e5m2 = k_cache.scalar_type() == torch::kUInt8;
  const bool is_fp16 = k_cache.scalar_type() == torch::kHalf;
  TORCH_CHECK((is_e5m2 || is_fp16) &&
                  v_cache.scalar_type() == k_cache.scalar_type() &&
                  k_cache.dim() == 3 && k_cache.size(1) == 1 &&
                  k_cache.size(2) == kDim && v_cache.sizes() == k_cache.sizes(),
              "sm70_qsa_decode expects E5M2 byte or FP16 KV [pool,1,256]");
  TORCH_CHECK(indices.scalar_type() == torch::kInt && indices.dim() == 2 &&
                  indices.size(0) == q.size(0),
              "sm70_qsa_decode expects int32 indices [batch,topk]");
  TORCH_CHECK(req_to_token.scalar_type() == torch::kInt &&
                  req_to_token.dim() == 2 &&
                  req_indices.scalar_type() == torch::kInt &&
                  req_indices.numel() == q.size(0) &&
                  seq_lens.scalar_type() == torch::kInt &&
                  seq_lens.numel() == q.size(0),
              "sm70_qsa_decode expects one int32 request and length per row");
  TORCH_CHECK(max_splits > 0 && min_tokens_per_split > 0,
              "sm70_qsa_decode split parameters must be positive");
  TORCH_CHECK(partial_o.scalar_type() == torch::kHalf &&
                  partial_o.dim() == 4 && partial_o.size(0) == q.size(0) &&
                  partial_o.size(1) == max_splits &&
                  partial_o.size(2) == kGroup && partial_o.size(3) == kDim,
              "sm70_qsa_decode partial output has the wrong shape");
  TORCH_CHECK(partial_lse.scalar_type() == torch::kFloat &&
                  partial_lse.dim() == 3 &&
                  partial_lse.size(0) == q.size(0) &&
                  partial_lse.size(1) == max_splits &&
                  partial_lse.size(2) == kGroup,
              "sm70_qsa_decode partial LSE has the wrong shape");
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(1, (unsigned int)max_splits, (unsigned int)q.size(0));
  if (is_e5m2) {
    qsa_decode_partial_kernel<uint8_t><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr()),
        k_cache.data_ptr<uint8_t>(), v_cache.data_ptr<uint8_t>(),
        req_to_token.data_ptr<int>(), req_indices.data_ptr<int>(),
        indices.data_ptr<int>(), seq_lens.data_ptr<int>(),
        (int)req_to_token.size(1), (int)indices.size(1), (int)max_splits,
        (int)min_tokens_per_split, (float)softmax_scale,
        reinterpret_cast<__half*>(partial_o.data_ptr()),
        partial_lse.data_ptr<float>());
  } else {
    qsa_decode_partial_kernel<__half><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr()),
        reinterpret_cast<const __half*>(k_cache.data_ptr()),
        reinterpret_cast<const __half*>(v_cache.data_ptr()),
        req_to_token.data_ptr<int>(), req_indices.data_ptr<int>(),
        indices.data_ptr<int>(), seq_lens.data_ptr<int>(),
        (int)req_to_token.size(1), (int)indices.size(1), (int)max_splits,
        (int)min_tokens_per_split, (float)softmax_scale,
        reinterpret_cast<__half*>(partial_o.data_ptr()),
        partial_lse.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_qsa_indexer_decode(torch::Tensor q, torch::Tensor k_cache,
                             torch::Tensor page_table,
                             torch::Tensor context_lens,
                             int64_t max_model_len, double score_scale,
                             torch::Tensor logits) {
  using namespace sm70_longctx;
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.scalar_type() == torch::kHalf && q.dim() == 3 &&
                  q.size(1) == kIndexerHeads && q.size(2) == kIndexerDim,
              "sm70_qsa_indexer_decode expects FP16 Q [batch,4,128]");
  TORCH_CHECK(k_cache.scalar_type() == torch::kHalf && k_cache.dim() == 4 &&
                  k_cache.size(1) == kIndexerPageSize &&
                  k_cache.size(2) == 1 && k_cache.size(3) == kIndexerDim,
              "sm70_qsa_indexer_decode expects FP16 K [pages,4,1,128]");
  TORCH_CHECK(page_table.scalar_type() == torch::kInt &&
                  page_table.dim() == 2 && page_table.size(0) == q.size(0) &&
                  context_lens.scalar_type() == torch::kInt &&
                  context_lens.numel() == q.size(0),
              "sm70_qsa_indexer_decode expects int32 page metadata");
  TORCH_CHECK(max_model_len > 0 && score_scale > 0,
              "sm70_qsa_indexer_decode dimensions and scale must be positive");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat && logits.dim() == 2 &&
                  logits.size(0) == q.size(0) &&
                  logits.size(1) == max_model_len,
              "sm70_qsa_indexer_decode logits have the wrong shape");
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(
      (unsigned int)((max_model_len + kIndexerKeysPerBlock - 1) /
                     kIndexerKeysPerBlock),
      (unsigned int)q.size(0));
  qsa_indexer_decode_kernel<<<grid, kIndexerThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr()),
      reinterpret_cast<const __half*>(k_cache.data_ptr()),
      page_table.data_ptr<int>(), context_lens.data_ptr<int>(),
      (int)page_table.size(1), (int)max_model_len, (float)score_scale,
      logits.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_qsa_prefill(torch::Tensor q, torch::Tensor k_cache,
                      torch::Tensor v_cache, torch::Tensor req_to_token,
                      torch::Tensor req_indices, torch::Tensor indices,
                      torch::Tensor seq_lens, double softmax_scale,
                      torch::Tensor output) {
  using namespace sm70_longctx;
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.scalar_type() == torch::kHalf && q.dim() == 3 &&
                  q.size(1) == kGroup && q.size(2) == kDim,
              "sm70_qsa_prefill expects FP16 Q [tokens,6,256]");
  const bool is_e5m2 = k_cache.scalar_type() == torch::kUInt8;
  TORCH_CHECK((is_e5m2 || k_cache.scalar_type() == torch::kHalf) &&
                  v_cache.scalar_type() == k_cache.scalar_type() &&
                  k_cache.dim() == 3 && k_cache.size(1) == 1 &&
                  k_cache.size(2) == kDim && v_cache.sizes() == k_cache.sizes(),
              "sm70_qsa_prefill expects E5M2 byte or FP16 KV [pool,1,256]");
  TORCH_CHECK(indices.scalar_type() == torch::kInt && indices.dim() == 2 &&
                  indices.size(0) == q.size(0),
              "sm70_qsa_prefill expects int32 indices [tokens,topk]");
  TORCH_CHECK(req_to_token.scalar_type() == torch::kInt &&
                  req_to_token.dim() == 2 &&
                  req_indices.scalar_type() == torch::kInt &&
                  req_indices.numel() == 1 && seq_lens.scalar_type() == torch::kInt &&
                  seq_lens.numel() == 1,
              "sm70_qsa_prefill currently supports one int32 request row");
  TORCH_CHECK(output.scalar_type() == torch::kHalf &&
                  output.sizes() == q.sizes(),
              "sm70_qsa_prefill output must match Q");
  auto stream = at::cuda::getCurrentCUDAStream();
  if (is_e5m2) {
    qsa_prefill_kernel<uint8_t><<<q.size(0), kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr()),
        k_cache.data_ptr<uint8_t>(), v_cache.data_ptr<uint8_t>(),
        req_to_token.data_ptr<int>(), req_indices.data_ptr<int>(),
        indices.data_ptr<int>(), seq_lens.data_ptr<int>(),
        (int)req_to_token.size(1), (int)indices.size(1),
        (float)softmax_scale, reinterpret_cast<__half*>(output.data_ptr()));
  } else {
    qsa_prefill_kernel<__half><<<q.size(0), kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr()),
        reinterpret_cast<const __half*>(k_cache.data_ptr()),
        reinterpret_cast<const __half*>(v_cache.data_ptr()),
        req_to_token.data_ptr<int>(), req_indices.data_ptr<int>(),
        indices.data_ptr<int>(), seq_lens.data_ptr<int>(),
        (int)req_to_token.size(1), (int)indices.size(1),
        (float)softmax_scale, reinterpret_cast<__half*>(output.data_ptr()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sm70_longctx_decode", &sm70_longctx_decode,
        "Long-context grouped decode partial (D256 G6 E5M2)");
  m.def("sm70_qsa_decode", &sm70_qsa_decode,
        "QSA grouped split-KV decode (D256 G6 E5M2)");
  m.def("sm70_qsa_indexer_decode", &sm70_qsa_indexer_decode,
        "QSA indexer decode scoring (H4 D128 FP16)");
  m.def("sm70_qsa_prefill", &sm70_qsa_prefill,
        "QSA chunk-prefill (D256 G6 E5M2)");
}
