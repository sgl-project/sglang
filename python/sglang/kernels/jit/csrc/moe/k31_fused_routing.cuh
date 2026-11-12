// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
// Portions adapted from sgl-project/sglang #46475 / #46484 (route_quant_metadata.cuh's metadata method)
// and from SGLang's route_radix.cuh / route_quant_fused.cuh.
// K3.1 decode MoE routing in one launch: route_quant_fused (radix top-16 of
// 896 on CTAs [0, M), the mxfp8 group quant of the routed activations on CTAs
// [M, 2M)), plus the padded-row mask and the trtllm-gen routing metadata the
// routed MoE GEMMs consume (expert-sorted, tile-padded permutation and the
// per-CTA expert / row-limit tables). The routing CTA that finishes last builds
// the metadata from every row's ids, so the routed MoE runs its GEMMs straight
// from it: the separate mask kernel and the trtllm-gen routingIndices kernel
// (DynBlock or Cluster, per layer at decode M) drop out.
//
// The routing and quant bodies are route_quant_fused's (route_radix_block and
// route_quant_metadata.cuh's Mxfp8RowQuant, same TU flags), so ids, weights and
// the quantized activations are bit-identical to it. The metadata follows the
// trtllm-gen routingPermutation contract (RoutingKernel.cuh): local experts in
// id order, each padded to whole tiles of tile_tokens rows, tile padding rows
// of the token map -1. Rows inside an expert are in token order up to
// kK31SmallMaxTokens rows and in arrival order above (the GEMM rows are
// independent, as with trtllm-gen's own atomics).
//
// Up to kK31SmallMaxTokens rows the last CTA lays the metadata out from
// per-row expert bitmaps, the method of route_quant_metadata.cuh's
// write_multi_token_metadata (sgl-project/sglang #46475 / #46484), extended to
// an expert range (EP), padded rows, any tile, and experts that span several
// tiles (M > tile). Above it, every routing CTA counts its own routes into a
// per-expert arrival counter (global atomics, so the histogram is built in
// parallel and each route gets its row in its expert) and parks the entry in
// expanded_idx_to_permuted_idx; the last CTA only scans the counts, writes the
// tile tables and turns each entry into its permuted row. Its shared memory
// and its serial tail no longer grow with M.

#include "route_quant_metadata.cuh"

namespace sglang {

// An entry packs (local expert << 16) | its row in the expert, so the row (at
// most M - 1: a row picks an expert once) must fit 16 bits.
inline constexpr uint32_t kK31RoutingMaxTokens = 4096;
inline constexpr uint32_t kK31RoutingExpertsPerThread = kNumExperts_ / LargeRouterRadixTrait::kBlockSize;  // 4

struct K31RoutingMetaParams {
  // trtllm-gen expert weights [M, 16], fp32 or bf16 (the routed MoE's finalize dtype)
  void* __restrict__ expert_weights;
  int32_t* __restrict__ expanded_idx_to_permuted_idx;  // [M * 16]
  int32_t* __restrict__ permuted_idx_to_token_idx;     // [max padded rows + 1]
  int32_t* __restrict__ cta_idx_xy_to_batch_idx;       // [max ctas]
  int32_t* __restrict__ cta_idx_xy_to_mn_limit;        // [max ctas]
  int32_t* __restrict__ num_non_exiting_ctas;          // [1]
  int32_t* __restrict__ total_num_padded_tokens;       // [1]
  int32_t* __restrict__ num_tokens_per_expert;         // [896], indexed by global expert id
  // Arrival counter of the routing CTAs; zero between launches (the last CTA
  // resets it), so concurrent launches must not share one.
  uint32_t* __restrict__ counter;
  // [896] routes per local expert (above kK31SmallMaxTokens rows); zero
  // between launches like `counter` (the last CTA reads and resets it).
  int32_t* __restrict__ expert_counter;
  // Rows at or past the bound are padding: their ids become -1 and they are
  // left out of the permutation. nullptr means num_valid_value.
  const int32_t* __restrict__ num_valid_ptr;
  int32_t num_valid_value;
  int32_t local_expert_offset;
  int32_t num_local_experts;
  int32_t tile_tokens;
  int32_t weights_fp32;
};

struct K31FusedRoutingParams {
  RouteRadixParams route;
  QuantKernelParams quant;
  K31RoutingMetaParams meta;
};

inline constexpr uint32_t kK31SmallMaxTokens = 16;
inline constexpr uint32_t kK31SmallWords = (kNumExperts_ + 31) / 32;  // 28

struct K31SmallRoutingSmem {
  // bit e of row t: row t routes to local expert e
  uint32_t token_bits[kK31SmallMaxTokens][kK31SmallWords];
  // union of the rows' bitmaps: the present local experts
  uint32_t any_bits[kK31SmallWords];
  // tiles of the local experts below each one within its 32-expert word
  int32_t tiles_in_word[kK31SmallWords * 32];
  // exclusive prefix of the tiles over the words
  int32_t word_tiles[kK31SmallWords + 1];
};

struct K31RoutingSmem {
  uint32_t warp_sum[LargeRouterRadixTrait::kNumWarps];
  int32_t num_ctas;
  int32_t expert_offset[kNumExperts_];  // first padded row of each local expert
};

// A routing CTA's share of the histogram (rows above kK31SmallMaxTokens): its
// route in slot `slot` takes the next row of its local expert, and the entry
// (local expert << 16) | row, -1 off the local range or on a padded row, is
// parked in expanded_idx_to_permuted_idx until the last CTA rewrites it.
SGL_DEVICE void k31_count_route(const K31RoutingMetaParams& meta, uint32_t row, uint32_t slot, int32_t id) {
  const int32_t local = id - meta.local_expert_offset;
  int32_t entry = -1;
  if (id >= 0 && local >= 0 && local < meta.num_local_experts) {
    entry = (local << 16) | atomicAdd(meta.expert_counter + local, 1);
  }
  meta.expanded_idx_to_permuted_idx[row * kTopK_ + slot] = entry;
}

// Build the trtllm-gen routing metadata from the routing CTAs' per-expert
// counts and parked entries (run by the last routing CTA, after every row's
// entries and counts are visible).
SGL_DEVICE void
k31_build_routing_metadata(const RouteRadixParams& route, const K31RoutingMetaParams& meta, K31RoutingSmem& smem) {
  using T = LargeRouterRadixTrait;
  constexpr uint32_t kThreads = T::kBlockSize;
  constexpr uint32_t kEPT = kK31RoutingExpertsPerThread;
  const uint32_t tx = threadIdx.x;
  const uint32_t lane_id = tx % device::kWarpThreads;
  const uint32_t warp_id = tx / device::kWarpThreads;
  const int32_t num_local = meta.num_local_experts;
  const int32_t first = meta.local_expert_offset;
  const int32_t tile = meta.tile_tokens;
  const uint32_t num_expanded = static_cast<uint32_t>(route.M) * T::kTopK;

  // Tiles per expert, exclusive scan over the experts in id order.
  // The counts are read (and reset for the next launch) by their owner thread.
  int32_t count[kEPT];
  int32_t num_ctas[kEPT];
  int32_t thread_ctas = 0;
#pragma unroll
  for (uint32_t j = 0; j < kEPT; ++j) {
    const int32_t e = static_cast<int32_t>(tx * kEPT + j);
    count[j] = e < num_local ? __ldcg(meta.expert_counter + e) : 0;
    num_ctas[j] = (count[j] + tile - 1) / tile;
    thread_ctas += num_ctas[j];
  }
#pragma unroll
  for (uint32_t j = 0; j < kEPT; ++j) {
    const int32_t e = static_cast<int32_t>(tx * kEPT + j);
    if (e < num_local) meta.expert_counter[e] = 0;
  }
  int32_t cta_offset = static_cast<int32_t>(
      moe::radix::block_exclusive_sum(static_cast<uint32_t>(thread_ctas), lane_id, warp_id, smem.warp_sum));
  if (tx == kThreads - 1) smem.num_ctas = cta_offset + thread_ctas;

#pragma unroll
  for (uint32_t j = 0; j < kEPT; ++j) {
    const int32_t e = static_cast<int32_t>(tx * kEPT + j);
    if (e < num_local) {
      smem.expert_offset[e] = cta_offset * tile;
      const int32_t limit = cta_offset * tile + count[j];
      for (int32_t c = 0; c < num_ctas[j]; ++c) {
        const int32_t cta = cta_offset + c;
        const int32_t tile_end = (cta + 1) * tile;
        const int32_t mn_limit = min(tile_end, limit);
        meta.cta_idx_xy_to_batch_idx[cta] = e;
        meta.cta_idx_xy_to_mn_limit[cta] = mn_limit;
        // TMA gathers may read a tile's unused rows; -1 suppresses those loads.
        for (int32_t p = mn_limit; p < tile_end; ++p) {
          meta.permuted_idx_to_token_idx[p] = -1;
        }
      }
    }
    cta_offset += num_ctas[j];
  }
  // Global expert ids off the local range count zero; the local ones are this
  // thread's own counts.
#pragma unroll
  for (uint32_t j = 0; j < kEPT; ++j) {
    const int32_t e = static_cast<int32_t>(tx * kEPT + j);
    if (e < num_local) meta.num_tokens_per_expert[first + e] = count[j];
  }
  for (int32_t g = static_cast<int32_t>(tx); g < static_cast<int32_t>(kNumExperts_); g += kThreads) {
    const int32_t local = g - first;
    if (local < 0 || local >= num_local) meta.num_tokens_per_expert[g] = 0;
  }
  __syncthreads();
  if (tx == 0) {
    meta.num_non_exiting_ctas[0] = smem.num_ctas;
    meta.total_num_padded_tokens[0] = smem.num_ctas * tile;
  }

  // Parked entry -> permuted row. Eight entries per thread are loaded before
  // any is used, so the L2 reads overlap (faster than 4 on GB300 at 240-512
  // rows; 16 or 32 gain nothing more).
  constexpr uint32_t kUnroll = 8;
  for (uint32_t base = 0; base < num_expanded; base += kThreads * kUnroll) {
    int32_t entry[kUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      const uint32_t i = base + u * kThreads + tx;
      entry[u] = i < num_expanded ? __ldcg(meta.expanded_idx_to_permuted_idx + i) : -1;
    }
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      const uint32_t i = base + u * kThreads + tx;
      if (i >= num_expanded) continue;
      int32_t permuted = -1;
      if (entry[u] >= 0) {
        permuted = smem.expert_offset[entry[u] >> 16] + (entry[u] & 0xffff);
        meta.permuted_idx_to_token_idx[permuted] = static_cast<int32_t>(i / T::kTopK);
      }
      meta.expanded_idx_to_permuted_idx[i] = permuted;
    }
  }
}

// The metadata of up to kK31SmallMaxTokens rows (DFLASH verify bs 1-2) from
// per-row local-expert bitmaps: a row selects an expert at most once, so an
// expert's rows are the rows whose bitmap has its bit, in row order. When no
// expert can outgrow one tile (M <= tile) its tile index is its rank among the
// present experts, a popcount over the union bitmap (route_quant_metadata.cuh's
// layout); otherwise a per-expert count pass sizes each expert's tiles first.
SGL_DEVICE void k31_build_small_routing_metadata(
    const RouteRadixParams& route, const K31RoutingMetaParams& meta, K31SmallRoutingSmem& smem) {
  using T = LargeRouterRadixTrait;
  constexpr uint32_t kThreads = T::kBlockSize;
  constexpr uint32_t kRoutesPerThread = (kK31SmallMaxTokens * kTopK_ + kThreads - 1) / kThreads;  // 2
  const uint32_t tx = threadIdx.x;
  const uint32_t lane_id = tx % device::kWarpThreads;
  const uint32_t warp_id = tx / device::kWarpThreads;
  const uint32_t M = static_cast<uint32_t>(route.M);
  const uint32_t routes = M * kTopK_;
  const int32_t first = meta.local_expert_offset;
  const int32_t num_local = meta.num_local_experts;
  const int32_t tile = meta.tile_tokens;
  const bool one_tile = static_cast<int32_t>(M) <= tile;

  // Local expert of each of this thread's routes, -1 off the range or padded.
  int32_t local[kRoutesPerThread];
#pragma unroll
  for (uint32_t j = 0; j < kRoutesPerThread; ++j) {
    const uint32_t i = tx + j * kThreads;
    const int32_t id = i < routes ? __ldcg(route.out_i + (i / kTopK_) * route.out_i_stride + i % kTopK_) : -1;
    const int32_t e = id - first;
    local[j] = (id >= 0 && e >= 0 && e < num_local) ? e : -1;
  }
  for (uint32_t i = tx; i < M * kK31SmallWords; i += kThreads) {
    (&smem.token_bits[0][0])[i] = 0;
  }
  if (tx < kK31SmallWords) smem.any_bits[tx] = 0;
  __syncthreads();
#pragma unroll
  for (uint32_t j = 0; j < kRoutesPerThread; ++j) {
    if (local[j] >= 0) {
      const uint32_t i = tx + j * kThreads;
      const uint32_t bit = 1u << (local[j] % 32);
      atomicOr(&smem.token_bits[i / kTopK_][local[j] / 32], bit);
      atomicOr(&smem.any_bits[local[j] / 32], bit);
    }
  }
  __syncthreads();

  if (one_tile) {
    // One tile per present expert: tiles per word = present experts per word.
    if (warp_id == 0) {
      int32_t total = lane_id < kK31SmallWords ? __popc(smem.any_bits[lane_id]) : 0;
#pragma unroll
      for (uint32_t offset = 1; offset < device::kWarpThreads; offset *= 2) {
        const int32_t up = __shfl_up_sync(0xffffffffu, total, offset);
        if (lane_id >= offset) total += up;
      }
      if (lane_id < kK31SmallWords) smem.word_tiles[lane_id + 1] = total;
      if (lane_id == 0) smem.word_tiles[0] = 0;
    }
    // Experts nobody routes to count zero; the present ones are written below.
    for (int32_t g = static_cast<int32_t>(tx); g < static_cast<int32_t>(kNumExperts_); g += kThreads) {
      const int32_t e = g - first;
      if (e < 0 || e >= num_local || !((smem.any_bits[e / 32] >> (e % 32)) & 1u)) {
        meta.num_tokens_per_expert[g] = 0;
      }
    }
  } else {
    // One 32-expert word per warp pass, one expert per lane: its row count, its
    // tiles, and the tiles of the experts below it in the word.
    for (uint32_t w = warp_id; w < kK31SmallWords; w += T::kNumWarps) {
      int32_t count = 0;
      for (uint32_t t = 0; t < M; ++t) {
        count += (smem.token_bits[t][w] >> lane_id) & 1u;
      }
      const int32_t tiles = (count + tile - 1) / tile;
      int32_t below = tiles;
#pragma unroll
      for (uint32_t offset = 1; offset < device::kWarpThreads; offset *= 2) {
        const int32_t up = __shfl_up_sync(0xffffffffu, below, offset);
        if (lane_id >= offset) below += up;
      }
      smem.tiles_in_word[w * 32 + lane_id] = below - tiles;
      if (lane_id == device::kWarpThreads - 1) smem.word_tiles[w + 1] = below;
    }
    for (int32_t g = static_cast<int32_t>(tx); g < static_cast<int32_t>(kNumExperts_); g += kThreads) {
      const int32_t e = g - first;
      if (e < 0 || e >= num_local || !((smem.any_bits[e / 32] >> (e % 32)) & 1u)) {
        meta.num_tokens_per_expert[g] = 0;
      }
    }
    __syncthreads();
    if (warp_id == 0) {
      int32_t total = lane_id < kK31SmallWords ? smem.word_tiles[lane_id + 1] : 0;
#pragma unroll
      for (uint32_t offset = 1; offset < device::kWarpThreads; offset *= 2) {
        const int32_t up = __shfl_up_sync(0xffffffffu, total, offset);
        if (lane_id >= offset) total += up;
      }
      if (lane_id < kK31SmallWords) smem.word_tiles[lane_id + 1] = total;
      if (lane_id == 0) smem.word_tiles[0] = 0;
    }
  }
  __syncthreads();
  if (tx == 0) {
    const int32_t num_ctas = smem.word_tiles[kK31SmallWords];
    meta.num_non_exiting_ctas[0] = num_ctas;
    meta.total_num_padded_tokens[0] = num_ctas * tile;
  }

#pragma unroll
  for (uint32_t j = 0; j < kRoutesPerThread; ++j) {
    const uint32_t i = tx + j * kThreads;
    if (i >= routes) continue;
    const int32_t e = local[j];
    if (e < 0) {
      meta.expanded_idx_to_permuted_idx[i] = -1;
      continue;
    }
    const uint32_t token = i / kTopK_;
    const uint32_t word = e / 32, bit = e % 32;
    int32_t within = 0, count = 0;
    for (uint32_t t = 0; t < M; ++t) {
      const int32_t has = (smem.token_bits[t][word] >> bit) & 1u;
      within += t < token ? has : 0;
      count += has;
    }
    const int32_t first_cta =
        smem.word_tiles[word] + (one_tile ? __popc(smem.any_bits[word] & ((1u << bit) - 1u)) : smem.tiles_in_word[e]);
    const int32_t start = first_cta * tile;
    const int32_t permuted = start + within;
    meta.expanded_idx_to_permuted_idx[i] = permuted;
    meta.permuted_idx_to_token_idx[permuted] = static_cast<int32_t>(token);
    if (within == 0) meta.num_tokens_per_expert[first + e] = count;
    if (within % tile == 0) {
      // The first row of each of the expert's tiles writes that tile's entries.
      const int32_t cta = first_cta + within / tile;
      const int32_t tile_end = (cta + 1) * tile;
      const int32_t mn_limit = min(tile_end, start + count);
      meta.cta_idx_xy_to_batch_idx[cta] = e;
      meta.cta_idx_xy_to_mn_limit[cta] = mn_limit;
      // TMA gathers may read a tile's unused rows; -1 suppresses those loads.
      for (int32_t p = mn_limit; p < tile_end; ++p) {
        meta.permuted_idx_to_token_idx[p] = -1;
      }
    }
  }
}

template <bool kUsePDL, typename TScore, typename TX>
__global__ __launch_bounds__(LargeRouterRadixTrait::kBlockSize)  //
    void k31_fused_routing_kernel(const __grid_constant__ K31FusedRoutingParams params) {
  using T = LargeRouterRadixTrait;
  const auto M = static_cast<uint32_t>(params.route.M);
  const uint32_t tx = threadIdx.x;
  if (blockIdx.x < M) {
    __shared__ union {
      typename T::Smem route;
      K31RoutingSmem meta;
      K31SmallRoutingSmem small;
    } smem;
    __shared__ bool is_last;
    const auto& meta = params.meta;
    device::PDLWaitPrimary<kUsePDL>();
    // The padded-row bound, loaded under the radix select.
    const int32_t num_valid = meta.num_valid_ptr != nullptr ? meta.num_valid_ptr[0] : meta.num_valid_value;
    route_radix_block</*kUsePDL=*/false, TScore>(params.route, smem.route);
    // Let the routed MoE launch under the mask and the metadata tail, as
    // route_quant_fused and route_quant_metadata.cuh trigger from their
    // routing bodies: a dependent's griddepcontrol.wait still waits for this
    // whole grid, metadata included, before it reads anything.
    device::PDLTriggerSecondary<kUsePDL>();

    const uint32_t row = blockIdx.x;
    if (tx < T::kTopK) {
      // Each slot is read back by the thread that wrote it (route_radix_block
      // with sorted == 0 writes slot tx from thread tx).
      const float w = params.route.out_w[row * params.route.out_w_stride + tx];
      int32_t id = params.route.out_i[row * params.route.out_i_stride + tx];
      if (static_cast<int32_t>(row) >= num_valid) {
        params.route.out_i[row * params.route.out_i_stride + tx] = -1;
        id = -1;
      }
      if (meta.weights_fp32) {
        static_cast<fp32_t*>(meta.expert_weights)[row * T::kTopK + tx] = w;
      } else {
        static_cast<bf16_t*>(meta.expert_weights)[row * T::kTopK + tx] = __float2bfloat16_rn(w);
      }
      if (M > kK31SmallMaxTokens) k31_count_route(meta, row, tx, id);
    }
    // Publish this row's ids (and counts) and take a ticket; the last row
    // builds the metadata (the cooperative-groups grid-sync pattern: block
    // barrier, then one thread's fence + atomic, and a fence on the acquiring
    // side).
    __syncthreads();
    if (tx == 0) {
      __threadfence();
      is_last = atomicAdd(meta.counter, 1u) == M - 1;
      if (is_last) __threadfence();
    }
    __syncthreads();
    if (!is_last) return;
    if (M <= kK31SmallMaxTokens) {
      k31_build_small_routing_metadata(params.route, meta, smem.small);
    } else {
      k31_build_routing_metadata(params.route, meta, smem.meta);
    }
    if (tx == 0) *meta.counter = 0u;
  } else {
    device::PDLWaitPrimary<kUsePDL>();
    Mxfp8RowQuant::run<TX>(params.quant, blockIdx.x - M);
    device::PDLTriggerSecondary<kUsePDL>();
  }
}

template <bool kUsePDL>
struct K31FusedRoutingKernel {
  static void
  run(const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView out_w,
      const tvm::ffi::TensorView out_i,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView out_q,
      const tvm::ffi::TensorView out_s,
      const tvm::ffi::TensorView expert_weights,
      const tvm::ffi::TensorView expanded_idx_to_permuted_idx,
      const tvm::ffi::TensorView permuted_idx_to_token_idx,
      const tvm::ffi::TensorView cta_idx_xy_to_batch_idx,
      const tvm::ffi::TensorView cta_idx_xy_to_mn_limit,
      const tvm::ffi::TensorView num_non_exiting_ctas,
      const tvm::ffi::TensorView total_num_padded_tokens,
      const tvm::ffi::TensorView num_tokens_per_expert,
      const tvm::ffi::TensorView counter,
      const tvm::ffi::TensorView expert_counter,
      const tvm::ffi::Optional<tvm::ffi::TensorView> num_valid,
      int64_t topk,
      double routed_scaling_factor,
      bool renormalize,
      bool apply_scale,
      int64_t local_expert_offset,
      int64_t num_local_experts,
      int64_t tile_tokens) {
    using namespace host;

    auto M_ = SymbolicSize{"num_tokens"};
    auto N_ = SymbolicSize{"num_experts"};
    auto K_ = SymbolicSize{"topk"};
    auto P_ = SymbolicSize{"max_padded_rows"};
    auto C_ = SymbolicSize{"max_ctas"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    auto score_dtype = SymbolicDType{};
    TensorMatcher({M_, N_})
        .with_dtype<bf16_t, fp32_t>(score_dtype)
        .with_device(device)
        .with_strides({-1, 1})
        .verify(scores);
    TensorMatcher({N_}).with_dtype<fp32_t>().with_device(device).verify(bias);
    TensorMatcher({M_, K_}).with_dtype<fp32_t>().with_device(device).verify(out_w);
    TensorMatcher({M_, K_}).with_dtype<int32_t>().with_device(device).verify(out_i);
    auto w_dtype = SymbolicDType{};
    TensorMatcher({M_, K_}).with_dtype<bf16_t, fp32_t>(w_dtype).with_strides({K_, 1}).with_device(device).verify(
        expert_weights);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(expanded_idx_to_permuted_idx);
    TensorMatcher({P_}).with_dtype<int32_t>().with_device(device).verify(permuted_idx_to_token_idx);
    TensorMatcher({C_}).with_dtype<int32_t>().with_device(device).verify(cta_idx_xy_to_batch_idx);
    TensorMatcher({C_}).with_dtype<int32_t>().with_device(device).verify(cta_idx_xy_to_mn_limit);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(num_non_exiting_ctas);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(total_num_padded_tokens);
    TensorMatcher({N_}).with_dtype<int32_t>().with_device(device).verify(num_tokens_per_expert);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(counter);
    TensorMatcher({N_}).with_dtype<int32_t>().with_device(device).verify(expert_counter);

    RuntimeCheck(
        N_.unwrap() == kNumExperts_ && K_.unwrap() == kTopK_ && topk == kTopK_,
        "k31_fused_routing is specialized for N=896, K=16");
    RuntimeCheck(scores.stride(0) % 4 == 0, "k31_fused_routing: scores row stride must be a multiple of 4");
    const auto M = static_cast<uint32_t>(M_.unwrap());
    RuntimeCheck(M <= kK31RoutingMaxTokens, "k31_fused_routing: at most 4096 rows");
    RuntimeCheck(
        expanded_idx_to_permuted_idx.size(0) == static_cast<int64_t>(M) * kTopK_,
        "k31_fused_routing: expanded_idx_to_permuted_idx must hold M * 16 entries");
    RuntimeCheck(
        0 <= local_expert_offset && 0 < num_local_experts &&
            local_expert_offset + num_local_experts <= static_cast<int64_t>(kNumExperts_),
        "k31_fused_routing: the local expert range must lie within the 896 experts");
    RuntimeCheck(0 < tile_tokens && tile_tokens <= 256, "k31_fused_routing: tile_tokens must be in [1, 256]");

    const int32_t* num_valid_ptr = nullptr;
    int32_t num_valid_value = static_cast<int32_t>(M);
    if (num_valid.has_value()) {
      const auto& nv = num_valid.value();
      RuntimeCheck(nv.numel() == 1, "k31_fused_routing: num_valid must be a scalar");
      if (nv.device().device_type == kDLCUDA) {
        RuntimeCheck(is_type<int32_t>(nv.dtype()), "k31_fused_routing: num_valid on CUDA must be int32");
        num_valid_ptr = static_cast<const int32_t*>(nv.data_ptr());
      } else if (is_type<int32_t>(nv.dtype())) {
        num_valid_value = *static_cast<const int32_t*>(nv.data_ptr());
      } else {
        RuntimeCheck(is_type<int64_t>(nv.dtype()), "k31_fused_routing: num_valid on CPU must be int32 or int64");
        num_valid_value = static_cast<int32_t>(*static_cast<const int64_t*>(nv.data_ptr()));
      }
    }

    auto x_dtype = SymbolicDType{};
    const auto quant_params = Mxfp8RowQuant::make(x, out_q, out_s, /*global_scale: unused*/ out_s, M_, device, x_dtype);
    if (M == 0) return;

    const auto params = K31FusedRoutingParams{
        .route =
            {scores.data_ptr(),
             static_cast<const fp32_t*>(bias.data_ptr()),
             static_cast<fp32_t*>(out_w.data_ptr()),
             static_cast<int32_t*>(out_i.data_ptr()),
             /*out_packed=*/nullptr,
             static_cast<int>(M),
             static_cast<long long>(scores.stride(0)),
             static_cast<long long>(out_w.stride(0)),
             static_cast<long long>(out_i.stride(0)),
             /*out_packed_stride=*/0,
             static_cast<float>(routed_scaling_factor),
             renormalize ? 1 : 0,
             apply_scale ? 1 : 0,
             /*sorted=*/0},
        .quant = quant_params,
        .meta =
            {expert_weights.data_ptr(),
             static_cast<int32_t*>(expanded_idx_to_permuted_idx.data_ptr()),
             static_cast<int32_t*>(permuted_idx_to_token_idx.data_ptr()),
             static_cast<int32_t*>(cta_idx_xy_to_batch_idx.data_ptr()),
             static_cast<int32_t*>(cta_idx_xy_to_mn_limit.data_ptr()),
             static_cast<int32_t*>(num_non_exiting_ctas.data_ptr()),
             static_cast<int32_t*>(total_num_padded_tokens.data_ptr()),
             static_cast<int32_t*>(num_tokens_per_expert.data_ptr()),
             static_cast<uint32_t*>(counter.data_ptr()),
             static_cast<int32_t*>(expert_counter.data_ptr()),
             num_valid_ptr,
             num_valid_value,
             static_cast<int32_t>(local_expert_offset),
             static_cast<int32_t>(num_local_experts),
             static_cast<int32_t>(tile_tokens),
             w_dtype.is_type<fp32_t>() ? 1 : 0},
    };

#define SGL_K31_ROUTING_LAUNCH(TS, TX)                                    \
  LaunchKernel(2 * M, LargeRouterRadixTrait::kBlockSize, device.unwrap()) \
      .enable_pdl(kUsePDL)(k31_fused_routing_kernel<kUsePDL, TS, TX>, params)

    if (score_dtype.is_type<fp32_t>()) {
      if (x_dtype.is_type<fp32_t>()) {
        SGL_K31_ROUTING_LAUNCH(fp32_t, fp32_t);
      } else {
        SGL_K31_ROUTING_LAUNCH(fp32_t, bf16_t);
      }
    } else {
      if (x_dtype.is_type<fp32_t>()) {
        SGL_K31_ROUTING_LAUNCH(bf16_t, fp32_t);
      } else {
        SGL_K31_ROUTING_LAUNCH(bf16_t, bf16_t);
      }
    }
#undef SGL_K31_ROUTING_LAUNCH
  }
};

}  // namespace sglang
