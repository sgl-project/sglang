/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// SM100 sparse MLA with e4m3 Q and KV on all 576 dims (512 NoPE + 64 RoPE) and per-tensor scales.
// Inputs follow TRT-LLM-gen's FP8 sparse MLA: merged e4m3 Q, 576 B e4m3 KV rows, token-level top-k
// slots, and fused bmm1 / bmm2 scales. The bmm1 scale is folded into the softmax log2 constant.
//
// Persistent CTAs walk work items (query token, 64-head block, split). Per 64-token block, QK^T is
// split between two issuers into two partial-logit tiles: TS MMAs over NoPE (Q in TMEM, duplicated in
// both lane halves so each lane holds a full dot product) and SS MMAs over RoPE. P is e4m3 with a
// lazy O rescale, and PV reads V as the K NoPE tile in MN-major order. O stays in TMEM. Issuer A
// issues the next item's Q copy and first QK^T before the last PV of the current item.
//
// Warp roles (512 threads):
//   warps 0-3    softmax, O rescale
//   warps 4-7    TMA gather4 of K NoPE and K RoPE
//   warp  8      MMA issuer A: TMEM alloc, Q UTCCP, QK^T NoPE dims 0-255, PV
//   warp  9      top-k validity masks
//   warp  10     Q NoPE / RoPE TMA for the next item
//   warp  11     MMA issuer B: QK^T NoPE dims 256-511 and RoPE
//   warps 12-15  epilogue, overlapping the softmax of the next item
#pragma once

#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cutlass/arch/arch.h>
#include <cutlass/cuda_host_adapter.hpp>

#include "helpers.h"
#include "params.h"
#include <algorithm>
#include <cuda_fp8.h>
#include <math_constants.h>

namespace sglang::q8kv8_sm100 {

constexpr int D_NOPE = 512;
constexpr int D_ROPE = 64;
constexpr int D_QK = D_NOPE + D_ROPE;
constexpr int D_V = 512;
constexpr int B_H = 64;
constexpr int B_TOPK = 64;
constexpr int NUM_STAGES = 4;
// NoPE K-steps (of 16) issued by issuer A; issuer B takes the rest plus RoPE.
constexpr int KA = 8;
constexpr int NUM_THREADS = 512;
constexpr int MMA_A_WARP = 8;
constexpr int MASK_WARP = 9;
constexpr int Q_LOAD_WARP = 10;
constexpr int MMA_B_WARP = 11;
constexpr int EPI_WARP0 = 12;
constexpr float MAX_INIT_VAL = -1e30f;
// e4m3 tops out at 448: P * 2^P_SHIFT never exceeds 2^(P_SHIFT + RESCALE_THRESHOLD) = 256.
constexpr float P_SHIFT = 4.0f;
constexpr float RESCALE_THRESHOLD = 4.0f;

constexpr int K_NOPE_STAGE_BYTES = B_TOPK * D_NOPE;  // 4 SW128 column tiles of 64 rows x 128B
constexpr int K_ROPE_STAGE_BYTES = B_TOPK * D_ROPE;  // 1 SW64 tile of 64 rows x 64B
constexpr int Q_ROPE_BYTES = B_H * D_ROPE;           // same layout as a K RoPE stage
constexpr int S_BYTES = B_H * B_TOPK;                // 64 rows x 64B, SW64
constexpr int KV_BYTES_PER_ROW_GROUP = 4 * D_QK;

namespace tmem_cols {
constexpr uint32_t O = 0;         // 256 columns: two .ws N=256 halves of the 64 x 512 output
constexpr uint32_t P_A = 256;     // 2 x 32 columns: logits over NoPE dims 0-255
constexpr uint32_t P_B = 320;     // 2 x 32 columns: logits over NoPE dims 256-511 plus RoPE
constexpr uint32_t Q_NOPE = 384;  // 128 columns: 512 e4m3 per lane, rows duplicated in both halves
}  // namespace tmem_cols

// QK^T NoPE, QK^T RoPE and PV are all e4m3 x e4m3 -> f32 (kind::f8f6f4).
constexpr uint32_t IDESC_QK = make_idesc(0, 0, false, B_H, B_TOPK);
constexpr uint32_t IDESC_PV = make_idesc(0, 0, true, B_H, 256);

enum NamedBarriers : uint32_t {
  // Warps w and w^2 own the two token halves of the same 32 rows.
  pair_sync_base = 1,
};

struct alignas(1024) SharedStorage {
  uint8_t k_nope[NUM_STAGES][K_NOPE_STAGE_BYTES];
  uint8_t q_nope[B_H * D_NOPE];  // K NoPE stage layout, UTCCP source
  uint8_t k_rope[NUM_STAGES][K_ROPE_STAGE_BYTES];
  uint8_t q_rope[2][Q_ROPE_BYTES];  // SS operand for the whole item, so double-buffered by item
  uint8_t s[2][S_BYTES];
  float max_buf[2][128];
  float sum_buf[128];
  uint8_t is_k_valid[NUM_STAGES][B_TOPK / 8];
  // Per-lane output scale of item it for the epilogue. 3 deep: softmax of item it+2 waits only on PV
  // of item it, so it can run two items ahead of the epilogue.
  float epi_scale[3][128];
  // q_full alternates by item: issuer B may still be waiting on item i when item i+1 lands.
  transac_bar_t bar_q_full[2], bar_q_free, bar_o_free, bar_epi_ready[3];
  transac_bar_t bar_kv_full[NUM_STAGES], bar_sv_done[NUM_STAGES];
  transac_bar_t bar_k_valid_ready[NUM_STAGES], bar_k_valid_free[NUM_STAGES];
  transac_bar_t bar_qk_done[2], bar_p_free[2], bar_s_ready[2];
  uint32_t tmem_base;
};

struct TmaMaps {
  CUtensorMap q;        // u8 {576, h_q, s_q}, box {128, 64, 1}, SW128 (NoPE column tiles)
  CUtensorMap q_rope;   // u8 {576, h_q, s_q}, box {64, 64, 1},  SW64  (column 512)
  CUtensorMap kv;       // u8 {576, s_kv},     box {128, 1},     SW128, gather4
  CUtensorMap kv_rope;  // u8 {576, s_kv},     box {64, 1},      SW64,  gather4
};

// SW64 K-major tile with 64B rows: physical offset of 16B chunk `c` in row `r`.
__device__ __forceinline__ uint32_t sw64_offset(int r, int c) {
  return r * 64 + ((c ^ ((r >> 1) & 3)) << 4);
}

struct WorkItem {
  int s_q_idx, head_block, split, num_blocks, topk_length;
};

// Items run (query token, 64-head block, split) with the split fastest.
template <bool HAVE_TOPK_LENGTH>
__device__ __forceinline__ WorkItem get_item(const SparseMlaQ8Kv8Sm100Params& params, int item) {
  const int items_per_token = (params.h_q / B_H) * params.num_splits;
  WorkItem w;
  w.s_q_idx = item / items_per_token;
  const int r = item % items_per_token;
  w.head_block = r / params.num_splits;
  w.split = r % params.num_splits;
  const int total = HAVE_TOPK_LENGTH ? max(0, min(__ldg(params.topk_length + w.s_q_idx), params.topk)) : params.topk;
  w.topk_length = max(0, min(total - w.split * params.split_len, params.split_len));
  // At least one block keeps the pipeline shape fixed; an empty row is fully masked.
  w.num_blocks = max((w.topk_length + B_TOPK - 1) / B_TOPK, 1);
  return w;
}

__device__ __forceinline__ const int* item_indices(const SparseMlaQ8Kv8Sm100Params& params, const WorkItem& w) {
  return params.indices + int64_t(w.s_q_idx) * params.stride_indices_s_q + w.split * params.split_len;
}

// QK^T of one block from one issuer: NoPE K-steps [0, KA) for issuer A, [KA, 16) plus RoPE for B.
template <bool WITH_ROPE>
__device__ __forceinline__ void issue_qk(SharedStorage& ss, uint32_t tmem_base, int st, int pb, const uint8_t* q_rope) {
  const uint32_t tmem_p = tmem_base + (WITH_ROPE ? tmem_cols::P_B : tmem_cols::P_A) + pb * 32;
  constexpr int s0 = WITH_ROPE ? KA : 0;
  constexpr int s1 = WITH_ROPE ? D_NOPE / 32 : KA;
  CUTE_UNROLL
  for (int s = s0; s < s1; ++s) {
    mma_ws_ts_f8(
        tmem_p,
        tmem_base + tmem_cols::Q_NOPE + s * 8,
        make_smem_desc(
            ss.k_nope[st] + (s / 4) * (B_TOPK * 128) + (s % 4) * 32, cute::UMMA::LayoutType::SWIZZLE_128B, 16, 1024),
        IDESC_QK,
        s > s0 ? 1u : 0u);
  }
  if constexpr (WITH_ROPE) {
    CUTE_UNROLL
    for (int s = 0; s < D_ROPE / 32; ++s) {
      mma_ws_ss_f8(
          tmem_p,
          make_smem_desc(q_rope + s * 32, cute::UMMA::LayoutType::SWIZZLE_64B, 16, 512),
          make_smem_desc(ss.k_rope[st] + s * 32, cute::UMMA::LayoutType::SWIZZLE_64B, 16, 512),
          IDESC_QK,
          1u);
    }
  }
}

// Split merge of one (query token, head) row: lane l combines dims [16 l, 16 l + 16) of the partials
// by their natural-log lse; the sink enters the denominator once, here.
template <int NUM_SPLITS, bool HAVE_ATTN_SINK>
__device__ __forceinline__ void merge_row(const SparseMlaQ8Kv8Sm100Params& params, int64_t row, int lane) {
  const int64_t rows = int64_t(params.s_q) * params.h_q;
  // The partial rows do not depend on the lse: load both in one round trip (+1-5% on split decode).
  uint4 pre[NUM_SPLITS][2];
  CUTE_UNROLL
  for (int s = 0; s < NUM_SPLITS; ++s) {
    const uint4* src = reinterpret_cast<const uint4*>(params.out + (s * rows + row) * D_V + lane * 16);
    pre[s][0] = src[0];
    pre[s][1] = src[1];
  }
  float l[NUM_SPLITS];
  float m = -CUDART_INF_F, ml = -CUDART_INF_F;
  CUTE_UNROLL
  for (int s = 0; s < NUM_SPLITS; ++s) {
    l[s] = params.lse[s * rows + row];
    m = fmaxf(m, l[s]);
    ml = fmaxf(ml, params.max_logits[s * rows + row]);
  }
  uint4* dst = reinterpret_cast<uint4*>(params.merged_out + row * D_V + lane * 16);
  if (m == -CUDART_INF_F) {
    dst[0] = dst[1] = make_uint4(0, 0, 0, 0);
    if (lane == 0) {
      params.merged_lse[row] = CUDART_INF_F;
      params.merged_max_logits[row] = -CUDART_INF_F;
    }
    return;
  }
  float w[NUM_SPLITS];
  float sum = 0.0f;
  CUTE_UNROLL
  for (int s = 0; s < NUM_SPLITS; ++s) {
    w[s] = __expf(l[s] - m);
    sum += w[s];
  }
  float denom = sum;
  if constexpr (HAVE_ATTN_SINK) {
    denom += __expf(__ldg(params.attn_sink + row % params.h_q) - m);
  }
  float acc[16];
  CUTE_UNROLL
  for (int i = 0; i < 16; ++i)
    acc[i] = 0.0f;
  CUTE_UNROLL
  for (int s = 0; s < NUM_SPLITS; ++s) {
    const float ws = w[s] / denom;
    CUTE_UNROLL
    for (int j = 0; j < 2; ++j) {
      const uint4 raw = pre[s][j];
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
      CUTE_UNROLL
      for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(h[i]);
        acc[j * 8 + 2 * i] = fmaf(ws, f.x, acc[j * 8 + 2 * i]);
        acc[j * 8 + 2 * i + 1] = fmaf(ws, f.y, acc[j * 8 + 2 * i + 1]);
      }
    }
  }
  CUTE_UNROLL
  for (int j = 0; j < 2; ++j) {
    uint4 raw;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&raw);
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
      h[i] = __floats2bfloat162_rn(acc[j * 8 + 2 * i], acc[j * 8 + 2 * i + 1]);
    }
    dst[j] = raw;
  }
  if (lane == 0) {
    params.merged_lse[row] = m + logf(sum);
    params.merged_max_logits[row] = ml;
  }
}

// Split merge as its own kernel: one warp per row.
template <int NUM_SPLITS, bool HAVE_ATTN_SINK>
__global__ void __launch_bounds__(256) merge_splits_kernel(__grid_constant__ const SparseMlaQ8Kv8Sm100Params params) {
  const int64_t row = int64_t(blockIdx.x) * 8 + threadIdx.x / 32;
  if (row < int64_t(params.s_q) * params.h_q) {
    merge_row<NUM_SPLITS, HAVE_ATTN_SINK>(params, row, threadIdx.x % 32);
  }
}

// Item epilogue for TMEM lane quadrant `tmem_lane`: O * out_scale as BF16; out_scale 0 writes zeros.
__device__ __forceinline__ void store_o(uint32_t tmem_lane, bf16* g_out, int half, float out_scale) {
  CUTE_UNROLL
  for (int nh = 0; nh < 2; ++nh) {
    CUTE_UNROLL
    for (int c = 0; c < 4; ++c) {
      float o[32];
      tmem_ld_32x32b_x32(tmem_lane + tmem_cols::O + nh * 128 + c * 32, o);
      tmem_wait_ld();
      uint32_t packed[16];
      CUTE_UNROLL
      for (int i = 0; i < 16; ++i) {
        const float x = out_scale != 0.0f ? o[2 * i] * out_scale : 0.0f;
        const float y = out_scale != 0.0f ? o[2 * i + 1] * out_scale : 0.0f;
        const __nv_bfloat162 h2 = __floats2bfloat162_rn(x, y);
        packed[i] = *reinterpret_cast<const uint32_t*>(&h2);
      }
      // .ws N=256 half nh: lanes 0-63 hold dims [256 nh, +128), lanes 64-127 the next 128.
      bf16* dst = g_out + nh * 256 + half * 128 + c * 32;
      stg_256(dst, packed);
      stg_256(dst + 16, packed + 8);
    }
  }
}

template <bool HAVE_TOPK_LENGTH, bool HAVE_ATTN_SINK>
__global__ void __launch_bounds__(NUM_THREADS, 1) sparse_mla_q8kv8_sm100_kernel(
    __grid_constant__ const SparseMlaQ8Kv8Sm100Params params, __grid_constant__ const TmaMaps maps) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100)) || \
    (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
  const int num_items = params.s_q * (params.h_q / B_H) * params.num_splits;
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int lane_idx = threadIdx.x % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_raw);

  if (threadIdx.x == 0) {
    // Swizzled tiles need a 1024B-aligned base.
    if ((cute::cast_smem_ptr_to_uint(smem_raw) & 1023u) != 0) {
      asm volatile("trap;");
    }
    ss.bar_q_full[0].init(1);
    ss.bar_q_full[1].init(1);
    ss.bar_q_free.init(1);
    ss.bar_o_free.init(4);
    CUTE_UNROLL
    for (int i = 0; i < NUM_STAGES; ++i) {
      ss.bar_kv_full[i].init(4);
      ss.bar_sv_done[i].init(1);
      ss.bar_k_valid_ready[i].init(B_TOPK / 8);
      ss.bar_k_valid_free[i].init(4);
    }
    CUTE_UNROLL
    for (int i = 0; i < 2; ++i) {
      ss.bar_qk_done[i].init(2);  // one commit per issuer
      ss.bar_p_free[i].init(4);
      ss.bar_s_ready[i].init(4);
    }
    CUTE_UNROLL
    for (int i = 0; i < 3; ++i) {
      ss.bar_epi_ready[i].init(4);
    }
    cutlass::arch::fence_barrier_init();
  }
  if (warp_idx == MMA_A_WARP) {
    cute::TMEM::Allocator1Sm().allocate(512, &ss.tmem_base);
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  if (warp_idx == Q_LOAD_WARP && lane_idx == 0) {
    cute::prefetch_tma_descriptor(&maps.q);
    cute::prefetch_tma_descriptor(&maps.q_rope);
    cute::prefetch_tma_descriptor(&maps.kv);
    cute::prefetch_tma_descriptor(&maps.kv_rope);
  }
  __syncthreads();
  tcgen05_after_thread_sync();
  const uint32_t tmem_base = ss.tmem_base;
  // Everything above only touched this CTA; inputs written by the previous kernel are read below.
  device::PDLWaitPrimary<true>();

  // Every role walks the same item sequence; g counts 64-token blocks across items.
  if (warp_idx < 4) {
    // Thread t owns row t % 64 and token half t / 64, i.e. exactly TMEM lane t of P and O.
    const int t = threadIdx.x;
    const int row = t & 63;
    const int half = t >> 6;
    const uint32_t tmem_lane = tmem_base + (uint32_t(warp_idx * 32) << 16);
    // TMEM logits are raw e4m3 sums; this maps them straight to the log2 domain.
    const float scale_log2 = params.scale_log2;
    const float2 scale2 = make_float2(scale_log2, scale_log2);
    const uint32_t pair_bar = NamedBarriers::pair_sync_base + (warp_idx & 1);
    const bool partial = params.num_splits > 1;

    int g = 0, it = 0;
    for (int item = blockIdx.x; item < num_items; item += gridDim.x, ++it) {
      const WorkItem w = get_item<HAVE_TOPK_LENGTH>(params, item);
      const int head = w.head_block * B_H + row;
      const int64_t o_row = (int64_t(w.split) * params.s_q + w.s_q_idx) * params.h_q + head;
      float mi = MAX_INIT_VAL;  // reference max of the stored P (log2 domain), updated lazily
      float li = 0.0f;          // sum of 2^(x - mi + P_SHIFT) over this thread's tokens
      float real_mi = -CUDART_INF_F;

      CUTE_NO_UNROLL
      for (int k = 0; k < w.num_blocks; ++k, ++g) {
        const int pb = g & 1;
        const int st = g % NUM_STAGES;
        ss.bar_qk_done[pb].wait((g >> 1) & 1);
        ss.bar_k_valid_ready[st].wait((g / NUM_STAGES) & 1);
        tcgen05_after_thread_sync();

        float p[32], pr[32];
        tmem_ld_32x32b_x32(tmem_lane + tmem_cols::P_A + pb * 32, p);
        tmem_ld_32x32b_x32(tmem_lane + tmem_cols::P_B + pb * 32, pr);
        // Shared-memory read overlaps the TMEM loads.
        const uint32_t valid = *reinterpret_cast<const uint32_t*>(ss.is_k_valid[st] + half * 4);
        tmem_wait_ld();
        tcgen05_before_thread_sync();
        // One arrival per warp: wait::ld is warp-collective and __syncwarp orders the lanes.
        __syncwarp();
        if (lane_idx == 0) {
          ss.bar_p_free[pb].arrive();
          ss.bar_k_valid_free[st].arrive();
        }

        CUTE_UNROLL
        for (int i = 0; i < 16; ++i) {
          const float2 v = float2_add(make_float2(p[2 * i], p[2 * i + 1]), make_float2(pr[2 * i], pr[2 * i + 1]));
          p[2 * i] = v.x;
          p[2 * i + 1] = v.y;
        }
        // Fully valid blocks (the common case) skip the per-element selects.
        if (!__all_sync(0xffffffff, valid == 0xffffffffu)) {
          CUTE_UNROLL
          for (int i = 0; i < 32; ++i) {
            p[i] = (valid >> i & 1u) ? p[i] : -CUDART_INF_F;
          }
        }
        float cur_max = -CUDART_INF_F;
        CUTE_UNROLL
        for (int i = 0; i < 32; i += 4) {
          cur_max = fmaxf(cur_max, fmaxf(fmaxf(p[i], p[i + 1]), fmaxf(p[i + 2], p[i + 3])));
        }
        cur_max *= scale_log2;
        ss.max_buf[g & 1][t] = cur_max;
        NamedBarrier::arrive_and_wait(64, pair_bar);
        cur_max = fmaxf(cur_max, ss.max_buf[g & 1][t ^ 64]);
        real_mi = fmaxf(real_mi, cur_max);
        // Warps w and w^2 see the same rows and maxima, so both O lanes of a row agree.
        const bool need_rescale = __any_sync(0xffffffff, cur_max - mi > RESCALE_THRESHOLD);
        float scale_old = 1.0f;
        if (need_rescale) {
          const float new_max = fmaxf(cur_max, mi);
          scale_old = exp2f(mi - new_max);
          mi = new_max;
        }

        // P = 2^(x - mi + P_SHIFT).
        const float bias = P_SHIFT - mi;
        const float2 neg_bias = make_float2(bias, bias);
        float2 sum2 = make_float2(0.0f, 0.0f);
        uint32_t packed[8];
        CUTE_UNROLL
        for (int i = 0; i < 8; ++i) {
          float2 d0 = float2_fma(make_float2(p[4 * i], p[4 * i + 1]), scale2, neg_bias);
          float2 d1 = float2_fma(make_float2(p[4 * i + 2], p[4 * i + 3]), scale2, neg_bias);
          d0.x = exp2f(d0.x);
          d0.y = exp2f(d0.y);
          d1.x = exp2f(d1.x);
          d1.y = exp2f(d1.y);
          sum2 = float2_add(sum2, float2_add(d0, d1));
          const uint32_t lo = __nv_cvt_float2_to_fp8x2(d0, __NV_SATFINITE, __NV_E4M3);
          const uint32_t hi = __nv_cvt_float2_to_fp8x2(d1, __NV_SATFINITE, __NV_E4M3);
          packed[i] = lo | (hi << 16);
        }
        li = fmaf(li, scale_old, sum2.x + sum2.y);

        // S[pb] was last read by PV(g - 2).
        if (g >= 2) {
          ss.bar_sv_done[(g - 2) % NUM_STAGES].wait(((g - 2) / NUM_STAGES) & 1);
        }
        *reinterpret_cast<uint4*>(ss.s[pb] + sw64_offset(row, half * 2)) =
            make_uint4(packed[0], packed[1], packed[2], packed[3]);
        *reinterpret_cast<uint4*>(ss.s[pb] + sw64_offset(row, half * 2 + 1)) =
            make_uint4(packed[4], packed[5], packed[6], packed[7]);

        if (k > 0 && need_rescale) {
          // PV(g - 1) must land before its accumulator is rescaled; PV(g) waits for s_ready.
          ss.bar_sv_done[(g - 1) % NUM_STAGES].wait(((g - 1) / NUM_STAGES) & 1);
          tcgen05_after_thread_sync();
          const float2 s2 = make_float2(scale_old, scale_old);
          CUTE_UNROLL
          for (int c = 0; c < (D_V / 2) / 32; ++c) {
            float o[32];
            tmem_ld_32x32b_x32(tmem_lane + tmem_cols::O + c * 32, o);
            tmem_wait_ld();
            CUTE_UNROLL
            for (int i = 0; i < 16; ++i) {
              const float2 v = float2_mul(make_float2(o[2 * i], o[2 * i + 1]), s2);
              o[2 * i] = v.x;
              o[2 * i + 1] = v.y;
            }
            tmem_st_32x32b_x32(tmem_lane + tmem_cols::O + c * 32, o);
          }
          tmem_wait_st();
          tcgen05_before_thread_sync();
        }
        fence_view_async_shared();
        __syncwarp();
        if (lane_idx == 0) {
          ss.bar_s_ready[pb].arrive();
        }
      }

      ss.sum_buf[t] = li;
      NamedBarrier::arrive_and_wait(64, pair_bar);
      const float li_total = li + ss.sum_buf[t ^ 64];
      // real_mi - mi <= RESCALE_THRESHOLD, so any valid token makes li_total >= 1.
      const bool has_valid = real_mi != -CUDART_INF_F;
      if (half == 0) {
        params.max_logits[o_row] = has_valid ? real_mi * CUDART_LN2_F : -CUDART_INF_F;
        // Empty partials get weight 0 in the split merge; an empty final row reports +inf.
        params.lse[o_row] =
            has_valid ? logf(li_total) + (mi - P_SHIFT) * CUDART_LN2_F : (partial ? -CUDART_INF_F : CUDART_INF_F);
      }
      float out_scale = 0.0f;
      if (has_valid) {
        float denom = li_total;
        if (HAVE_ATTN_SINK && !partial) {
          // +inf sink zeroes the row, -inf sink is a no-op; split partials get it in the merge.
          denom += exp2f(__ldg(params.attn_sink + head) * CUDART_L2E_F - (mi - P_SHIFT));
        }
        out_scale = params.bmm2_scale / denom;
      }
      ss.epi_scale[it % 3][t] = out_scale;
      __syncwarp();
      if (lane_idx == 0) {
        ss.bar_epi_ready[it % 3].arrive();
      }
    }
  } else if (warp_idx >= EPI_WARP0) {
    // Epilogue warpgroup: thread e owns TMEM lane e, like softmax thread e.
    const int e = threadIdx.x - EPI_WARP0 * 32;
    const uint32_t tmem_lane = tmem_base + (uint32_t((warp_idx - EPI_WARP0) * 32) << 16);
    int g = 0, it = 0;
    for (int item = blockIdx.x; item < num_items; item += gridDim.x, ++it) {
      const WorkItem w = get_item<HAVE_TOPK_LENGTH>(params, item);
      g += w.num_blocks;
      const int64_t o_row = (int64_t(w.split) * params.s_q + w.s_q_idx) * params.h_q + w.head_block * B_H + (e & 63);
      ss.bar_epi_ready[it % 3].wait((it / 3) & 1);
      const int g_last = g - 1;
      ss.bar_sv_done[g_last % NUM_STAGES].wait((g_last / NUM_STAGES) & 1);
      tcgen05_after_thread_sync();
      store_o(tmem_lane, params.out + o_row * D_V, e >> 6, ss.epi_scale[it % 3][e]);
      tcgen05_before_thread_sync();
      __syncwarp();
      if (lane_idx == 0) {
        ss.bar_o_free.arrive();
      }
    }
  } else if (warp_idx < 8) {
    // Producer warp pw gathers row groups {pw, pw + 4, pw + 8, pw + 12} (4 tokens each).
    const int pw = warp_idx - 4;
    if (cute::elect_one_sync()) {
      const auto in_range = [&](int idx) { return uint32_t(idx) < uint32_t(params.s_kv); };
      int g = 0;
      for (int item = blockIdx.x; item < num_items; item += gridDim.x) {
        const WorkItem w = get_item<HAVE_TOPK_LENGTH>(params, item);
        const int* g_indices = item_indices(params, w);
        int4 rows[4];
        CUTE_UNROLL
        for (int r = 0; r < 4; ++r) {
          rows[r] = __ldg(reinterpret_cast<const int4*>(g_indices) + pw + 4 * r);
        }
        CUTE_NO_UNROLL
        for (int k = 0; k < w.num_blocks; ++k, ++g) {
          const int st = g % NUM_STAGES;
          bool issue[4];
          int num_issue = 0;
          CUTE_UNROLL
          for (int r = 0; r < 4; ++r) {
            // After its first fill a stage only holds finite data, so masked groups may keep stale rows.
            issue[r] = g < NUM_STAGES || in_range(rows[r].x) || in_range(rows[r].y) || in_range(rows[r].z) ||
                       in_range(rows[r].w);
            num_issue += issue[r] ? 1 : 0;
          }
          ss.bar_sv_done[st].wait(((g / NUM_STAGES) & 1) ^ 1);
          ss.bar_kv_full[st].arrive_and_expect_tx(num_issue * KV_BYTES_PER_ROW_GROUP);
          CUTE_UNROLL
          for (int r = 0; r < 4; ++r) {
            if (!issue[r]) continue;
            const int rg = pw + 4 * r;
            CUTE_UNROLL
            for (int c = 0; c < D_NOPE / 128; ++c) {
              tma_gather4(
                  &maps.kv,
                  ss.bar_kv_full[st],
                  ss.k_nope[st] + c * (B_TOPK * 128) + rg * 512,
                  c * 128,
                  rows[r],
                  int64_t(cute::TMA::CacheHintSm90::EVICT_LAST));
            }
            tma_gather4(
                &maps.kv_rope,
                ss.bar_kv_full[st],
                ss.k_rope[st] + rg * 256,
                D_NOPE,
                rows[r],
                int64_t(cute::TMA::CacheHintSm90::EVICT_LAST));
          }
          if (k + 1 < w.num_blocks) {
            CUTE_UNROLL
            for (int r = 0; r < 4; ++r) {
              rows[r] = __ldg(reinterpret_cast<const int4*>(g_indices + (k + 1) * B_TOPK) + pw + 4 * r);
            }
          }
        }
      }
    }
  } else if (warp_idx == MMA_A_WARP) {
    if (cute::elect_one_sync()) {
      // PV(gp) of block gp; the first block of item it_p overwrites O once item it_p - 1 has left it.
      const auto issue_pv = [&](int gp, bool first_of_item, int it_p) {
        const int st = gp % NUM_STAGES;
        const int sb = gp & 1;
        ss.bar_s_ready[sb].wait((gp >> 1) & 1);
        if (first_of_item && it_p > 0) {
          ss.bar_o_free.wait((it_p - 1) & 1);
        }
        tcgen05_after_thread_sync();
        CUTE_UNROLL
        for (int nh = 0; nh < 2; ++nh) {
          CUTE_UNROLL
          for (int s = 0; s < B_TOPK / 32; ++s) {
            // V = the K NoPE tile read MN-major: 128-dim atoms 8KB apart, 8-token atoms 1KB apart.
            mma_ws_ss_f8(
                tmem_base + tmem_cols::O + nh * 128,
                make_smem_desc(ss.s[sb] + s * 32, cute::UMMA::LayoutType::SWIZZLE_64B, 16, 512),
                make_smem_desc(
                    ss.k_nope[st] + nh * (2 * B_TOPK * 128) + s * (32 * 128),
                    cute::UMMA::LayoutType::SWIZZLE_128B,
                    B_TOPK * 128,
                    1024),
                IDESC_PV,
                (!first_of_item || s > 0) ? 1u : 0u);
          }
        }
        umma_commit(ss.bar_sv_done[st]);
      };
      // One flat block sequence across items: QK^T(g) is issued before PV(g - 1), also when g starts
      // a new item, so the next item's UTCCP and first QK^T do not wait for the last softmax.
      int g = 0;
      int it = 0;
      int pend_g = -1, pend_it = 0;
      bool pend_first = false;
      for (int item = blockIdx.x; item < num_items; item += gridDim.x, ++it) {
        const WorkItem w = get_item<HAVE_TOPK_LENGTH>(params, item);
        if (g > 0) {
          // Both issuers' QK^T of the previous block (the previous item's last) must be done before the
          // TMEM copy of Q is replaced.
          ss.bar_qk_done[(g - 1) & 1].wait(((g - 1) >> 1) & 1);
        }
        ss.bar_q_full[it & 1].wait((it >> 1) & 1);
        tcgen05_after_thread_sync();
        // 16B per row per copy; warpx2::02_13 duplicates the 64 rows into both lane halves.
        CUTE_UNROLL
        for (int i = 0; i < D_NOPE / 16; ++i) {
          utccp_64x128b_warpx2_0213(
              tmem_base + tmem_cols::Q_NOPE + i * 4,
              make_smem_desc(
                  ss.q_nope + (i / 8) * (B_H * 128) + (i % 8) * 16, cute::UMMA::LayoutType::SWIZZLE_128B, 16, 1024));
        }
        // Releases the Q NoPE staging buffer and tells issuer B that TMEM Q is ready.
        umma_commit(ss.bar_q_free);

        CUTE_NO_UNROLL
        for (int k = 0; k < w.num_blocks; ++k, ++g) {
          const int st = g % NUM_STAGES;
          const int pb = g & 1;
          ss.bar_kv_full[st].wait((g / NUM_STAGES) & 1);
          ss.bar_p_free[pb].wait(((g >> 1) & 1) ^ 1);
          tcgen05_after_thread_sync();
          issue_qk<false>(ss, tmem_base, st, pb, nullptr);
          umma_commit(ss.bar_qk_done[pb]);
          if (pend_g >= 0) {
            issue_pv(pend_g, pend_first, pend_it);
          }
          pend_g = g;
          pend_first = k == 0;
          pend_it = it;
        }
      }
      if (pend_g >= 0) {
        issue_pv(pend_g, pend_first, pend_it);
      }
    }
  } else if (warp_idx == MMA_B_WARP) {
    if (cute::elect_one_sync()) {
      int g = 0;
      int it = 0;
      for (int item = blockIdx.x; item < num_items; item += gridDim.x, ++it) {
        const WorkItem w = get_item<HAVE_TOPK_LENGTH>(params, item);
        ss.bar_q_full[it & 1].wait((it >> 1) & 1);  // Q RoPE landed
        ss.bar_q_free.wait(it & 1);                 // issuer A's UTCCP of Q NoPE landed
        const uint8_t* q_rope = ss.q_rope[it & 1];
        CUTE_NO_UNROLL
        for (int k = 0; k < w.num_blocks; ++k) {
          const int gq = g + k;
          const int st = gq % NUM_STAGES;
          const int pb = gq & 1;
          ss.bar_kv_full[st].wait((gq / NUM_STAGES) & 1);
          ss.bar_p_free[pb].wait(((gq >> 1) & 1) ^ 1);
          tcgen05_after_thread_sync();
          issue_qk<true>(ss, tmem_base, st, pb, q_rope);
          umma_commit(ss.bar_qk_done[pb]);
        }
        g += w.num_blocks;
      }
    }
  } else if (warp_idx == MASK_WARP) {
    if (lane_idx < B_TOPK / 8) {
      int g = 0;
      for (int item = blockIdx.x; item < num_items; item += gridDim.x) {
        const WorkItem w = get_item<HAVE_TOPK_LENGTH>(params, item);
        const int* g_indices = item_indices(params, w);
        CUTE_NO_UNROLL
        for (int k = 0; k < w.num_blocks; ++k, ++g) {
          const int st = g % NUM_STAGES;
          const int pos = k * B_TOPK + lane_idx * 8;
          const int4 a = __ldg(reinterpret_cast<const int4*>(g_indices + pos));
          const int4 b = __ldg(reinterpret_cast<const int4*>(g_indices + pos) + 1);
          const int idx[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
          uint32_t mask = 0;
          CUTE_UNROLL
          for (int i = 0; i < 8; ++i) {
            const bool ok = uint32_t(idx[i]) < uint32_t(params.s_kv) && pos + i < w.topk_length;
            mask |= (ok ? 1u : 0u) << i;
          }
          ss.bar_k_valid_free[st].wait(((g / NUM_STAGES) & 1) ^ 1);
          ss.is_k_valid[st][lane_idx] = uint8_t(mask);
          ss.bar_k_valid_ready[st].arrive();
        }
      }
    }
  } else if (warp_idx == Q_LOAD_WARP) {
    // Q of item i+1 is staged while item i runs. UTCCP(i) frees the NoPE staging; the RoPE staging
    // is an SS operand for the whole item, so it alternates by item, and UTCCP(i) also implies that
    // item i-1 (the previous user of buffer (i+1) % 2) has finished its QK^T.
    if (lane_idx == 0) {
      int it = 0;
      for (int item = blockIdx.x; item < num_items; item += gridDim.x, ++it) {
        const WorkItem w = get_item<HAVE_TOPK_LENGTH>(params, item);
        if (it > 0) {
          ss.bar_q_free.wait((it - 1) & 1);
        }
        transac_bar_t& bar = ss.bar_q_full[it & 1];
        bar.arrive_and_expect_tx(B_H * D_QK);
        CUTE_UNROLL
        for (int j = 0; j < D_NOPE / 128; ++j) {
          cute::SM90_TMA_LOAD_3D::copy(
              &maps.q,
              reinterpret_cast<uint64_t*>(&bar),
              uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST),
              ss.q_nope + j * (B_H * 128),
              j * 128,
              w.head_block * B_H,
              w.s_q_idx);
        }
        cute::SM90_TMA_LOAD_3D::copy(
            &maps.q_rope,
            reinterpret_cast<uint64_t*>(&bar),
            uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST),
            ss.q_rope[it & 1],
            int(D_NOPE),
            w.head_block * B_H,
            w.s_q_idx);
      }
    }
  }

  // This CTA has issued all its work, so the next kernel may launch; its griddepcontrol.wait still
  // waits for this whole grid to finish.
  device::PDLTriggerSecondary<true>();
  // The epilogue fenced its last tcgen05.ld before this barrier.
  __syncthreads();
  if (warp_idx == MMA_A_WARP) {
    tcgen05_after_thread_sync();
    cute::TMEM::Allocator1Sm().free(tmem_base, 512);
  }
#else
  if (cute::thread0()) {
    CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
  }
#endif
}

inline void encode_tma(
    CUtensorMap* map,
    uint32_t rank,
    const void* gaddr,
    const uint64_t* dims,
    const uint64_t* strides_bytes,
    const uint32_t* box,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion promotion,
    const char* name) {
  const uint32_t elem_strides[3] = {1, 1, 1};
  const CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
      map,
      CU_TENSOR_MAP_DATA_TYPE_UINT8,
      rank,
      const_cast<void*>(gaddr),
      dims,
      strides_bytes,
      box,
      elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      promotion,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CHECK_HOST(res == CUDA_SUCCESS) << "sparse_mla_q8kv8_sm100: tensor map encoding failed for " << name << ": "
                                  << int(res);
}

/**
 * \brief Encode the TMA descriptors and launch the main kernel, plus the split merge kernel when
 *        `p.num_splits > 1`.
 *
 * \tparam HAVE_TOPK_LENGTH Whether `p.topk_length` is set.
 * \tparam HAVE_ATTN_SINK   Whether `p.attn_sink` is set.
 * \param p       Launch parameters; `p.stream` is the stream to launch on.
 * \param num_sms SM count of the device, which caps the persistent grid.
 */
template <bool HAVE_TOPK_LENGTH, bool HAVE_ATTN_SINK>
void run_sparse_mla_q8kv8_sm100(const SparseMlaQ8Kv8Sm100Params& p, int32_t num_sms) {
  CHECK_HOST(p.h_q > 0 && p.h_q % B_H == 0) << "sparse_mla_q8kv8_sm100: h_q must be a multiple of 64, got " << p.h_q;
  CHECK_HOST(p.topk > 0 && p.topk % B_TOPK == 0)
      << "sparse_mla_q8kv8_sm100: topk must be a multiple of 64, got " << p.topk;
  CHECK_HOST(p.s_q > 0 && p.s_kv > 0) << "sparse_mla_q8kv8_sm100: empty problem";
  CHECK_HOST(p.num_splits >= 1 && p.split_len % B_TOPK == 0 && p.split_len * p.num_splits == p.topk)
      << "sparse_mla_q8kv8_sm100: topk must split into num_splits chunks of a multiple of 64";
  CHECK_HOST(
      p.stride_kv_s_kv % 16 == 0 && p.stride_q_h_q % 16 == 0 && p.stride_q_s_q % 16 == 0 &&
      p.stride_indices_s_q % 4 == 0)
      << "sparse_mla_q8kv8_sm100: TMA and vector loads need 16B-aligned row strides";
  // gather4 index loads are int4 and the epilogue stores are 32 B per lane.
  CHECK_HOST(
      reinterpret_cast<uintptr_t>(p.q) % 16 == 0 && reinterpret_cast<uintptr_t>(p.kv) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(p.indices) % 16 == 0)
      << "sparse_mla_q8kv8_sm100: q, kv and indices must be 16B-aligned";
  CHECK_HOST(
      reinterpret_cast<uintptr_t>(p.out) % 32 == 0 &&
      (p.num_splits == 1 || reinterpret_cast<uintptr_t>(p.merged_out) % 16 == 0))
      << "sparse_mla_q8kv8_sm100: out must be 32B-aligned";

  TmaMaps maps;
  {
    const uint64_t dims[3] = {uint64_t(D_QK), uint64_t(p.h_q), uint64_t(p.s_q)};
    const uint64_t strides[2] = {uint64_t(p.stride_q_h_q), uint64_t(p.stride_q_s_q)};
    const uint32_t box[3] = {128, B_H, 1};
    encode_tma(&maps.q, 3, p.q, dims, strides, box, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, "q");
    const uint32_t box_rope[3] = {D_ROPE, B_H, 1};
    encode_tma(
        &maps.q_rope,
        3,
        p.q,
        dims,
        strides,
        box_rope,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        "q_rope");
  }
  {
    const uint64_t dims[2] = {uint64_t(D_QK), uint64_t(p.s_kv)};
    const uint64_t strides[1] = {uint64_t(p.stride_kv_s_kv)};
    const uint32_t box[2] = {128, 1};
    encode_tma(
        &maps.kv, 2, p.kv, dims, strides, box, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, "kv");
    const uint32_t box_rope[2] = {D_ROPE, 1};
    encode_tma(
        &maps.kv_rope,
        2,
        p.kv,
        dims,
        strides,
        box_rope,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
        "kv_rope");
  }

  auto kernel = &sparse_mla_q8kv8_sm100_kernel<HAVE_TOPK_LENGTH, HAVE_ATTN_SINK>;
  constexpr size_t smem_size = sizeof(SharedStorage);
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  const int32_t num_items = p.s_q * (p.h_q / B_H) * p.num_splits;
  // PDL only without splits: with the split merge kernel in the chain it measured 5-8% slower at 32-64
  // query tokens.
  host::LaunchKernel(uint32_t(std::min(num_items, num_sms)), NUM_THREADS, p.stream, smem_size)
      .enable_pdl(p.num_splits == 1)(kernel, p, maps);

  if (p.num_splits > 1) {
    const int64_t rows = int64_t(p.s_q) * p.h_q;
    const uint32_t blocks = uint32_t((rows + 7) / 8);
    if (p.num_splits == 2) {
      host::LaunchKernel(blocks, 256, p.stream)(&merge_splits_kernel<2, HAVE_ATTN_SINK>, p);
    } else if (p.num_splits == 4) {
      host::LaunchKernel(blocks, 256, p.stream)(&merge_splits_kernel<4, HAVE_ATTN_SINK>, p);
    } else if (p.num_splits == 8) {
      host::LaunchKernel(blocks, 256, p.stream)(&merge_splits_kernel<8, HAVE_ATTN_SINK>, p);
    } else {
      SGL_PANIC << "sparse_mla_q8kv8_sm100: num_splits must be 1, 2, 4 or 8, got " << p.num_splits;
    }
  }
}

}  // namespace sglang::q8kv8_sm100
