// Router top-k and trtllm-gen MoE routing tables for up to 64 (or kMaxTokens) tokens in one launch, written into the
// precomputed routing metadata FlashInfer's trtllm-gen MoE body reads (trtllm_fp4_block_scale_moe with
// da_routing_metadata), so FlashInfer's routing kernels are skipped. Top-k key, softmax and table layout
// follow FlashInfer's routingCustom (TopKExpertSelect<NoOpPreprocess, SoftmaxPostprocess>,
// routingPermutation); FlashInfer builds its routing with -use_fast_math and so must this file.
//
// Portions adapted from FlashInfer 0.7.1 trtllm-gen MoE routing (include/flashinfer/trtllm/fused_moe/):
// calcSoftmax, initRoutingTilePadding and the CTA tables of routingPermutation (RoutingKernel.cuh), the
// TopKRedType top-k comparison key (RoutingKernelTopK.cuh) and getMaxNumCtasInBatchDim (runner.h).
// Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
// Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Modified by NVIDIA: the top-10 over these keys is a bound-and-compact warp selection, and the last CTA to
// arrive builds the routing tables in the same launch.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cooperative_groups/reduce.h>
#include <cub/block/block_scan.cuh>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cooperative_groups.h>
#include <cstdint>

namespace sglang {

namespace qwen_router_tables {

namespace cg = cooperative_groups;

inline constexpr uint32_t kExperts = 512;
inline constexpr uint32_t kTopK = 10;
inline constexpr uint32_t kThreads = kExperts;
inline constexpr uint32_t kPerLane = kExperts / 32;

struct Params {
  const bf16_t* logits;                   // [M, kExperts]
  bf16_t* expert_weights;                 // [M, kTopK]
  int32_t* topk_ids;                      // [M, kTopK]
  int32_t* expanded_idx_to_permuted_idx;  // [M * kTopK]
  int32_t* permuted_idx_to_token_idx;     // [max padded + 1]
  int32_t* cta_idx_xy_to_batch_idx;       // [max CTAs]
  int32_t* cta_idx_xy_to_mn_limit;        // [max CTAs]
  int32_t* num_non_exiting_ctas;          // [1]
  int32_t* total_num_padded_tokens;       // [1]
  int32_t* num_tokens_per_expert;         // [kExperts]
  uint32_t* arrival;                      // monotonic arrival counter, one per call site
  uint32_t num_tokens;
  uint32_t tile_log2;
};

// Order-preserving key of a bf16 score and its expert: larger score first, the smaller expert on ties
// (FlashInfer TopKRedType<bf16>::makeCmpVal: cub's float twiddle above 65535 - expert).
SGL_DEVICE uint32_t topk_key(uint32_t bits, uint32_t expert) {
  const uint32_t twiddled = bits ^ ((bits & 0x8000u) ? 0xFFFFu : 0x8000u);
  return (twiddled << 16) | (65535u - expert);
}

SGL_DEVICE void topk_unkey(uint32_t key, uint32_t& bits, uint32_t& expert) {
  expert = 65535u - (key & 0xFFFFu);
  const uint32_t twiddled = key >> 16;
  bits = twiddled ^ ((twiddled & 0x8000u) ? 0x8000u : 0xFFFFu);
}

// FlashInfer initRoutingTilePadding: -1 in a tile's unused rows (TMA gathers may read them; the GEMM masks
// their outputs), scalar stores up to 16 B alignment and int4 stores after.
SGL_DEVICE void pad_rows(int32_t* token_idx, int32_t begin, int32_t end) {
  for (; begin < end && reinterpret_cast<uintptr_t>(token_idx + begin) % alignof(int4) != 0; ++begin) {
    token_idx[begin] = -1;
  }
  for (; end - begin >= 4; begin += 4)
    *reinterpret_cast<int4*>(token_idx + begin) = make_int4(-1, -1, -1, -1);
  for (; begin < end; ++begin)
    token_idx[begin] = -1;
}

SGL_DEVICE uint32_t max16(const uint32_t (&k)[kPerLane]) {
  const uint32_t a = max(max(max(k[0], k[1]), max(k[2], k[3])), max(max(k[4], k[5]), max(k[6], k[7])));
  const uint32_t b = max(max(max(k[8], k[9]), max(k[10], k[11])), max(max(k[12], k[13]), max(k[14], k[15])));
  return max(a, b);
}

SGL_DEVICE uint32_t warp_max_u32(uint32_t v) {
  uint32_t r;
  asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
  return r;
}

SGL_DEVICE uint32_t warp_sum_u32(uint32_t v) {
  uint32_t r;
  asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
  return r;
}

// The k-th largest key of the warp (lane k holds it, k < kTopK) from up to one candidate per lane.
SGL_DEVICE uint32_t warp_topk_of_lanes(uint32_t v, uint32_t lane) {
  uint32_t mine = 0;
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    const uint32_t top = warp_max_u32(v);
    v = v == top ? 0u : v;
    mine = lane == k ? top : mine;
  }
  return mine;
}

// Top-10 keys of 512 (16 per lane), lane k holding the k-th largest. Keys are unique and nonzero.
// The 10th largest lane maximum bounds the 10th largest key from below, so the top-10 are among the keys
// at or above it; usually few enough to give one per lane, otherwise ten full rounds over all keys.
SGL_DEVICE uint32_t warp_top10(uint32_t (&keys)[kPerLane], uint32_t lane, uint32_t* s_cand) {
  uint32_t lane_max = max16(keys);
  uint32_t v = lane_max;
  uint32_t bound = 0;
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    bound = warp_max_u32(v);
    v = v == bound ? 0u : v;
  }
  uint32_t count = 0;
#pragma unroll
  for (uint32_t i = 0; i < kPerLane; ++i)
    count += keys[i] >= bound;
  const uint32_t total = warp_sum_u32(count);
  if (total <= 32) {
    uint32_t offset = count;
#pragma unroll
    for (uint32_t d = 1; d < 32; d <<= 1) {
      const uint32_t up = __shfl_up_sync(0xffffffffu, offset, d);
      offset += lane >= d ? up : 0u;
    }
    offset -= count;
#pragma unroll
    for (uint32_t i = 0; i < kPerLane; ++i) {
      if (keys[i] >= bound) s_cand[offset++] = keys[i];
    }
    __syncwarp();
    const uint32_t cand = lane < total ? s_cand[lane] : 0u;
    __syncwarp();
    return warp_topk_of_lanes(cand, lane);
  }
  uint32_t mine = 0;
#pragma unroll 1
  for (uint32_t k = 0; k < kTopK; ++k) {
    const uint32_t top = warp_max_u32(max16(keys));
#pragma unroll
    for (uint32_t i = 0; i < kPerLane; ++i)
      keys[i] = keys[i] == top ? 0u : keys[i];
    mine = lane == k ? top : mine;
  }
  return mine;
}

// FlashInfer calcSoftmax(warp, score, laneIdx, NumTopExperts) with DataType = bf16.
SGL_DEVICE bf16_t softmax_topk(cg::thread_block_tile<32> const& warp, bf16_t score, uint32_t lane) {
  float max_score = -INFINITY;
  if (lane < kTopK) {
    const float si = static_cast<float>(score);
    max_score = si >= max_score ? si : max_score;
  }
  max_score = cg::reduce(warp, max_score, cg::greater<float>());
  float sum_score = 0.f;
  float new_score = 0.f;
  if (lane < kTopK) {
    new_score = static_cast<float>(score) - max_score;
    new_score = expf(new_score);
    sum_score += new_score;
  }
  sum_score = cg::reduce(warp, sum_score, cg::plus<float>());
  if (lane < kTopK) {
    score = static_cast<bf16_t>(new_score / sum_score);
  }
  return score;
}

/// \brief CTA b computes token b's top-10 experts and softmax weights with its first warp; the last CTA to
/// arrive (acq_rel counter, no spinning) builds the expert histogram, the tile-padded expert offsets, the
/// CTA -> expert / mn-limit tables and the expanded -> permuted and permuted -> token maps in FlashInfer's
/// routingPermutation layout.
template <bool kUsePDL, uint32_t kMaxTokens>
__global__ __launch_bounds__(kThreads, 1) void router_topk_tables_kernel(const __grid_constant__ Params p) {
  constexpr uint32_t kMaxExpanded = kMaxTokens * kTopK;
  constexpr uint32_t kExpandedPerThread = (kMaxExpanded + kThreads - 1) / kThreads;
  using Scan = cub::BlockScan<int32_t, kThreads, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename Scan::TempStorage scan_storage;
  __shared__ int32_t s_ids[kMaxExpanded];
  __shared__ int32_t s_count[kExperts];
  __shared__ int32_t s_offset[kExperts];
  __shared__ uint32_t s_cand[32];
  __shared__ uint32_t s_last;

  const uint32_t lane = threadIdx.x % 32;
  const uint32_t M = p.num_tokens;
  const uint32_t t = blockIdx.x;
  s_count[threadIdx.x] = 0;
  device::PDLWaitPrimary<kUsePDL>();

  // Lane l holds experts 16 l .. 16 l + 15 (the keys are exact, so the layout does not change the result).
  if (threadIdx.x < 32) {
    const auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    const uint4* src = reinterpret_cast<const uint4*>(p.logits + static_cast<size_t>(t) * kExperts + lane * kPerLane);
    const uint4 v[2] = {src[0], src[1]};
    uint32_t keys[kPerLane];
#pragma unroll
    for (uint32_t i = 0; i < kPerLane / 2; ++i) {
      const uint32_t w = (&v[i / 4].x)[i % 4];
      keys[2 * i] = topk_key(w & 0xFFFFu, lane * kPerLane + 2 * i);
      keys[2 * i + 1] = topk_key(w >> 16, lane * kPerLane + 2 * i + 1);
    }
    uint32_t bits, expert;
    topk_unkey(warp_top10(keys, lane, s_cand), bits, expert);
    const bf16_t weight = softmax_topk(warp, __ushort_as_bfloat16(static_cast<uint16_t>(bits)), lane);
    if (lane < kTopK) {
      p.expert_weights[t * kTopK + lane] = weight;
      p.topk_ids[t * kTopK + lane] = static_cast<int32_t>(expert);
    }
  }
  // bar.sync orders this CTA's id stores before thread 0's gpu-scope release; the last arrival acquires.
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t prev;
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;" : "=r"(prev) : "l"(p.arrival) : "memory");
    s_last = ((prev + 1u) % M) == 0u ? 1u : 0u;
  }
  __syncthreads();
  if (!s_last) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }

  const uint32_t expanded = M * kTopK;
#pragma unroll
  for (uint32_t j = 0; j < kExpandedPerThread; ++j) {
    const uint32_t i = threadIdx.x + j * kThreads;
    if (i < expanded) s_ids[i] = __ldcg(p.topk_ids + i);
  }
  __syncthreads();
  // Rank of each (token, slot) inside its expert from shared-memory atomics (FlashInfer's order is
  // atomic too; only the tables' per-expert row sets are fixed).
  int32_t rank[kExpandedPerThread];
#pragma unroll
  for (uint32_t j = 0; j < kExpandedPerThread; ++j) {
    const uint32_t i = threadIdx.x + j * kThreads;
    rank[j] = i < expanded ? atomicAdd(&s_count[s_ids[i]], 1) : 0;
  }
  __syncthreads();

  // Thread e owns expert e: CTA count, tile-padded offset, CTA tables and the padding rows of its tiles.
  const uint32_t e = threadIdx.x;
  const int32_t count = s_count[e];
  const int32_t tile = 1 << p.tile_log2;
  const int32_t num_cta = (count + tile - 1) >> p.tile_log2;
  int32_t cta_offset, total_cta;
  Scan(scan_storage).ExclusiveSum(num_cta, cta_offset, total_cta);
  for (int32_t c = 0; c < num_cta; ++c) {
    const int32_t end = (cta_offset + c + 1) << p.tile_log2;
    const int32_t limit = min(end, (cta_offset << p.tile_log2) + count);
    p.cta_idx_xy_to_batch_idx[cta_offset + c] = static_cast<int32_t>(e);
    p.cta_idx_xy_to_mn_limit[cta_offset + c] = limit;
    pad_rows(p.permuted_idx_to_token_idx, limit, end);
  }
  s_offset[e] = cta_offset << p.tile_log2;
  p.num_tokens_per_expert[e] = count;
  if (e == 0) {
    p.num_non_exiting_ctas[0] = total_cta;
    p.total_num_padded_tokens[0] = total_cta << p.tile_log2;
  }
  __syncthreads();

#pragma unroll
  for (uint32_t j = 0; j < kExpandedPerThread; ++j) {
    const uint32_t i = threadIdx.x + j * kThreads;
    if (i < expanded) {
      const int32_t permuted = s_offset[s_ids[i]] + rank[j];
      p.expanded_idx_to_permuted_idx[i] = permuted;
      p.permuted_idx_to_token_idx[permuted] = static_cast<int32_t>(i / kTopK);
    }
  }
  // The FC1 launch reads every table written above.
  __syncthreads();
  device::PDLTriggerSecondary<kUsePDL>();
}

// kMaxTokens = 64 is the stock instance; a wider one serves 64 < M <= kMaxTokens.
template <bool kUsePDL, uint32_t kMaxTokens = 64>
struct RouterTopkTables {
  static void
  run(const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView expert_weights,
      const tvm::ffi::TensorView topk_ids,
      const tvm::ffi::TensorView expanded_idx_to_permuted_idx,
      const tvm::ffi::TensorView permuted_idx_to_token_idx,
      const tvm::ffi::TensorView cta_idx_xy_to_batch_idx,
      const tvm::ffi::TensorView cta_idx_xy_to_mn_limit,
      const tvm::ffi::TensorView num_non_exiting_ctas,
      const tvm::ffi::TensorView total_num_padded_tokens,
      const tvm::ffi::TensorView num_tokens_per_expert,
      const tvm::ffi::TensorView arrival,
      int64_t tile_log2) {
    using namespace host;
    auto M_ = SymbolicSize{"num_tokens"};
    auto X_ = SymbolicSize{"num_expanded"};
    auto P_ = SymbolicSize{"max_padded_plus_one"};
    auto C_ = SymbolicSize{"max_ctas"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M_, kExperts}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(logits);
    TensorMatcher({M_, kTopK}).with_dtype<bf16_t>().with_device(device).verify(expert_weights);
    TensorMatcher({M_, kTopK}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({X_}).with_dtype<int32_t>().with_device(device).verify(expanded_idx_to_permuted_idx);
    TensorMatcher({P_}).with_dtype<int32_t>().with_device(device).verify(permuted_idx_to_token_idx);
    TensorMatcher({C_}).with_dtype<int32_t>().with_device(device).verify(cta_idx_xy_to_batch_idx);
    TensorMatcher({C_}).with_dtype<int32_t>().with_device(device).verify(cta_idx_xy_to_mn_limit);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(num_non_exiting_ctas);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(total_num_padded_tokens);
    TensorMatcher({kExperts}).with_dtype<int32_t>().with_device(device).verify(num_tokens_per_expert);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(arrival);
    const auto M = static_cast<uint32_t>(M_.unwrap());
    CHECK_HOST(M > 0 && M <= kMaxTokens) << "router_topk_tables: num_tokens " << M << " outside 1.." << kMaxTokens;
    CHECK_HOST(static_cast<uint64_t>(X_.unwrap()) >= M * kTopK) << "router_topk_tables: expanded map too small";
    CHECK_HOST(tile_log2 >= 3 && tile_log2 <= 8) << "router_topk_tables: tile_log2 " << tile_log2;
    // FlashInfer getMaxNumCtasInBatchDim: one CTA per touched expert, plus full tiles of the remainder.
    const int64_t filled = std::min<int64_t>(kExperts, int64_t{M} * kTopK);
    const int64_t max_ctas = filled + (int64_t{M} * kTopK - filled) / (int64_t{1} << tile_log2);
    CHECK_HOST(C_.unwrap() >= max_ctas) << "router_topk_tables: CTA tables too small";
    CHECK_HOST(P_.unwrap() >= (max_ctas << tile_log2) + 1) << "router_topk_tables: permuted map too small";
    Params params{};
    params.logits = static_cast<const bf16_t*>(logits.data_ptr());
    params.expert_weights = static_cast<bf16_t*>(expert_weights.data_ptr());
    params.topk_ids = static_cast<int32_t*>(topk_ids.data_ptr());
    params.expanded_idx_to_permuted_idx = static_cast<int32_t*>(expanded_idx_to_permuted_idx.data_ptr());
    params.permuted_idx_to_token_idx = static_cast<int32_t*>(permuted_idx_to_token_idx.data_ptr());
    params.cta_idx_xy_to_batch_idx = static_cast<int32_t*>(cta_idx_xy_to_batch_idx.data_ptr());
    params.cta_idx_xy_to_mn_limit = static_cast<int32_t*>(cta_idx_xy_to_mn_limit.data_ptr());
    params.num_non_exiting_ctas = static_cast<int32_t*>(num_non_exiting_ctas.data_ptr());
    params.total_num_padded_tokens = static_cast<int32_t*>(total_num_padded_tokens.data_ptr());
    params.num_tokens_per_expert = static_cast<int32_t*>(num_tokens_per_expert.data_ptr());
    params.arrival = static_cast<uint32_t*>(arrival.data_ptr());
    params.num_tokens = M;
    params.tile_log2 = static_cast<uint32_t>(tile_log2);
    LaunchKernel(M, kThreads, device.unwrap())
        .enable_pdl(kUsePDL)(router_topk_tables_kernel<kUsePDL, kMaxTokens>, params);
  }
};

}  // namespace qwen_router_tables

using qwen_router_tables::RouterTopkTables;

}  // namespace sglang
