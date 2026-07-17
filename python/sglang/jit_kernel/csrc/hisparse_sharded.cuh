#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <stddef.h>
#include <stdint.h>

namespace {

constexpr int kShardedWarpSize = 32;
constexpr unsigned kShardedFullWarpMask = 0xffffffffu;
constexpr uint16_t kShardedQueueHit = 0xffffu;

__device__ __forceinline__ uint32_t sharded_mix32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

__device__ __forceinline__ void
sharded_transfer_item_warp(int lane, const void* src_addr, void* dst_addr, int64_t item_size_bytes) {
  const int total_pairs = item_size_bytes / 16;
  const auto* src = static_cast<const uint64_t*>(src_addr);
  auto* dst = static_cast<uint64_t*>(dst_addr);
  for (int pair = lane; pair < total_pairs; pair += kShardedWarpSize) {
    uint64_t lo;
    uint64_t hi;
    const uint64_t* src_pair = src + pair * 2;
    asm volatile("ld.global.nc.v2.b64 {%0,%1},[%2];" : "=l"(lo), "=l"(hi) : "l"(src_pair) : "memory");
    uint64_t* dst_pair = dst + pair * 2;
    asm volatile("st.global.cg.v2.b64 [%0],{%1,%2};" : : "l"(dst_pair), "l"(lo), "l"(hi) : "memory");
  }

  const int tail_words = (item_size_bytes - total_pairs * 16) / 8;
  if (lane < tail_words) {
    const auto* src_tail = reinterpret_cast<const uint64_t*>(static_cast<const char*>(src_addr) + total_pairs * 16);
    auto* dst_tail = reinterpret_cast<uint64_t*>(static_cast<char*>(dst_addr) + total_pairs * 16);
    uint64_t value;
    asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(value) : "l"(src_tail + lane) : "memory");
    asm volatile("st.global.cg.b64 [%0],%1;" : : "l"(dst_tail + lane), "l"(value) : "memory");
  }
}

template <int BLOCK_SIZE, int HOT_BUFFER_SIZE, int NUM_CTAS>
struct ShardedSmemLayout {
  static_assert(HOT_BUFFER_SIZE % kShardedWarpSize == 0, "HOT_BUFFER_SIZE must be divisible by warp size");
  static constexpr int kLogicalShards = HOT_BUFFER_SIZE / kShardedWarpSize;
  static_assert(kLogicalShards % NUM_CTAS == 0, "NUM_CTAS must divide logical shards");
  static constexpr int kWays = kShardedWarpSize;
  static constexpr int kLocalShards = kLogicalShards / NUM_CTAS;
  static constexpr int kWarps = BLOCK_SIZE / kShardedWarpSize;
  static constexpr int kWorkers = kLocalShards < kWarps ? kLocalShards : kWarps;
  static constexpr size_t kQueueBytes = kLocalShards * kWays * sizeof(uint16_t);
  static constexpr size_t kCountsBytes = kLocalShards * sizeof(int32_t);
  static constexpr size_t kWorkerU8Bytes = 2 * kWorkers * kWays * sizeof(uint8_t);
  static constexpr size_t kWorkerU16Bytes = kWorkers * kWays * sizeof(uint16_t);
  static constexpr size_t kBytes = kQueueBytes + 2 * kCountsBytes + kWorkerU8Bytes + kWorkerU16Bytes;
};

// Process one logical shard after the CTA has routed its top-k entries.
//
// Required invariants:
// - request_lru[shard_base:shard_base + warp_size] is a permutation of [0, warp_size).
// - Cache tags are unique within the shard.
// - Queued top-k tokens are unique and non-negative.
// - The first miss_count evictable ways become victims.
// - The resulting LRU order is stale ways, newly loaded ways, then cache hits.
__device__ __forceinline__ void process_shard(
    int lane,
    unsigned lanes_before,
    int logical_shard,
    int shard_base,
    uint16_t* queue,
    int queue_count,
    const int32_t* request_top_k,
    int32_t* request_output,
    int32_t* request_tags,
    const int32_t* request_locations,
    const int64_t* request_host,
    uint8_t* request_lru,
    const void* host_cache,
    void* device_buffer,
    int32_t* request_counts,
    int32_t* request_overflows,
    int local_overflow,
    uint8_t* worker_evictable,
    uint8_t* worker_hits,
    uint16_t* worker_misses,
    int64_t item_size_bytes) {
  const int32_t tag = request_tags[shard_base + lane];
  bool lane_hit = false;
  for (int i = 0; i < queue_count; ++i) {
    const int selected = queue[i];
    const int32_t token = request_top_k[selected];
    if (tag == token) {
      lane_hit = true;
      request_output[selected] = request_locations[shard_base + lane];
      queue[i] = kShardedQueueHit;
    }
    __syncwarp(kShardedFullWarpMask);
  }
  const unsigned protected_mask = __ballot_sync(kShardedFullWarpMask, lane_hit);

  const uint8_t old_way = request_lru[shard_base + lane];
  const bool is_hit = (protected_mask >> old_way) & 1u;
  const unsigned evict_positions = __ballot_sync(kShardedFullWarpMask, !is_hit);
  const unsigned hit_positions = __ballot_sync(kShardedFullWarpMask, is_hit);
  const int evictable_count = __popc(evict_positions);
  const int hit_count = __popc(hit_positions);
  if (!is_hit) {
    worker_evictable[__popc(evict_positions & lanes_before)] = old_way;
  } else {
    worker_hits[__popc(hit_positions & lanes_before)] = old_way;
  }
  __syncwarp(kShardedFullWarpMask);

  const bool is_miss = lane < queue_count && queue[lane] != kShardedQueueHit;
  const unsigned miss_positions = __ballot_sync(kShardedFullWarpMask, is_miss);
  const int miss_count = __popc(miss_positions);
  if (is_miss) {
    worker_misses[__popc(miss_positions & lanes_before)] = queue[lane];
  }
  __syncwarp(kShardedFullWarpMask);

  // Shared atomics do not preserve top-k order. Only misses need a stable
  // order because that order determines their exact-LRU victims; hits only
  // contribute to the protected-way set above.
  if (lane == 0) {
    for (int i = 1; i < miss_count; ++i) {
      const uint16_t value = worker_misses[i];
      int j = i - 1;
      while (j >= 0 && worker_misses[j] > value) {
        worker_misses[j + 1] = worker_misses[j];
        --j;
      }
      worker_misses[j + 1] = value;
    }
  }
  __syncwarp(kShardedFullWarpMask);

  for (int miss = lane; miss < miss_count; miss += kShardedWarpSize) {
    const int selected = worker_misses[miss];
    const int32_t token = request_top_k[selected];
    const int victim = worker_evictable[miss];
    const int32_t destination = request_locations[shard_base + victim];
    request_tags[shard_base + victim] = token;
    request_output[selected] = destination;
  }
  __syncwarp(kShardedFullWarpMask);

  for (int miss = 0; miss < miss_count; ++miss) {
    const int selected = worker_misses[miss];
    const int32_t token = request_top_k[selected];
    const int64_t src_loc = request_host[token];
    const int32_t dst_loc = request_output[selected];
    const auto* src = static_cast<const char*>(host_cache) + src_loc * item_size_bytes;
    auto* dst = static_cast<char*>(device_buffer) + static_cast<int64_t>(dst_loc) * item_size_bytes;
    sharded_transfer_item_warp(lane, src, dst, item_size_bytes);
  }

  uint8_t lru_value;
  if (lane < evictable_count - miss_count) {
    lru_value = worker_evictable[lane + miss_count];
  } else if (lane < evictable_count) {
    lru_value = worker_evictable[lane - (evictable_count - miss_count)];
  } else {
    lru_value = worker_hits[lane - evictable_count];
  }
  request_lru[shard_base + lane] = lru_value;
  if (lane == 0) {
    request_counts[logical_shard] = miss_count;
    request_overflows[logical_shard] = local_overflow;
  }
}

template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE, int NUM_CTAS, int MIN_BLOCKS_PER_SM>
__global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_SM) void sharded_kernel(
    const int32_t* __restrict__ top_k_tokens,
    int32_t* __restrict__ device_buffer_tokens,
    const int64_t* __restrict__ host_cache_locs,
    const int32_t* __restrict__ device_buffer_locs,
    int32_t* __restrict__ top_k_device_locs,
    const int64_t* __restrict__ req_pool_indices,
    const int32_t* __restrict__ seq_lens,
    uint8_t* __restrict__ lru_slots,
    const void* __restrict__ host_cache,
    void* __restrict__ device_buffer,
    int32_t* __restrict__ split_miss_counts,
    int32_t* __restrict__ shard_overflows,
    const int32_t* __restrict__ num_real_reqs,
    int64_t device_buffer_tokens_stride,
    int64_t device_buffer_locs_stride,
    int64_t host_stride,
    int64_t item_size_bytes) {
  static_assert(BLOCK_SIZE % kShardedWarpSize == 0, "whole warps required");
  static_assert(BLOCK_SIZE >= kShardedWarpSize && BLOCK_SIZE <= 1024, "unsupported CTA size");
  static_assert(HOT_BUFFER_SIZE % kShardedWarpSize == 0, "HOT_BUFFER_SIZE must be divisible by warp size");
  constexpr int kLogicalShards = HOT_BUFFER_SIZE / kShardedWarpSize;
  static_assert(NUM_CTAS > 0 && (NUM_CTAS & (NUM_CTAS - 1)) == 0, "NUM_CTAS must be a power of two");
  static_assert(kLogicalShards % NUM_CTAS == 0, "NUM_CTAS must divide shards");
  using Layout = ShardedSmemLayout<BLOCK_SIZE, HOT_BUFFER_SIZE, NUM_CTAS>;
  constexpr int kWays = Layout::kWays;
  constexpr int kLocalShards = Layout::kLocalShards;
  constexpr int kWorkers = Layout::kWorkers;
  static_assert(kWays == kShardedWarpSize, "one warp must own exactly one shard");

  const int request = blockIdx.x / NUM_CTAS;
  if (request >= num_real_reqs[0]) return;
  const int cta_rank = blockIdx.x & (NUM_CTAS - 1);
  const int tid = threadIdx.x;
  const int warp = tid / kShardedWarpSize;
  const int lane = tid & (kShardedWarpSize - 1);
  const unsigned lanes_before = (1u << lane) - 1u;
  const int64_t pool_index = req_pool_indices[request];
  const int32_t seq_len = seq_lens[request];
  const int32_t newest_token = seq_len - 1;
  const int32_t num_selected = seq_len < NUM_TOP_K ? seq_len : NUM_TOP_K;

  const int32_t* request_top_k = top_k_tokens + request * NUM_TOP_K;
  int32_t* request_output = top_k_device_locs + request * NUM_TOP_K;
  int32_t* request_tags = device_buffer_tokens + pool_index * device_buffer_tokens_stride;
  const int32_t* request_locations = device_buffer_locs + pool_index * device_buffer_locs_stride;
  const int64_t* request_host = host_cache_locs + pool_index * host_stride;
  uint8_t* request_lru = lru_slots + pool_index * HOT_BUFFER_SIZE;
  int32_t* request_counts = split_miss_counts + request * kLogicalShards;
  int32_t* request_overflows = shard_overflows + request * kLogicalShards;

  extern __shared__ char shared_raw[];
  uint16_t* queues = reinterpret_cast<uint16_t*>(shared_raw);
  int32_t* queue_counts = reinterpret_cast<int32_t*>(shared_raw + Layout::kQueueBytes);
  int32_t* local_overflows = reinterpret_cast<int32_t*>(shared_raw + Layout::kQueueBytes + Layout::kCountsBytes);
  uint8_t* evictable = reinterpret_cast<uint8_t*>(shared_raw + Layout::kQueueBytes + 2 * Layout::kCountsBytes);
  uint8_t* hit_order = evictable + kWorkers * kWays;
  uint16_t* miss_indices = reinterpret_cast<uint16_t*>(hit_order + kWorkers * kWays);

  // Phase 1: initialize the per-CTA shard queues.
  for (int local = tid; local < kLocalShards; local += BLOCK_SIZE) {
    queue_counts[local] = 0;
    local_overflows[local] = 0;
  }
  __syncthreads();

  // Phase 2: route top-k entries to this CTA's logical shards.
  for (int selected = tid; selected < num_selected; selected += BLOCK_SIZE) {
    const int32_t token = request_top_k[selected];
    if (token < 0) continue;
    const int logical_shard = sharded_mix32(static_cast<uint32_t>(token)) % kLogicalShards;
    if ((logical_shard & (NUM_CTAS - 1)) != cta_rank) continue;
    if (token == newest_token) {
      request_output[selected] = request_locations[HOT_BUFFER_SIZE];
      continue;
    }
    const int local_shard = logical_shard / NUM_CTAS;
    const int position = atomicAdd(&queue_counts[local_shard], 1);
    if (position < kWays) {
      queues[local_shard * kWays + position] = static_cast<uint16_t>(selected);
    } else {
      // TODO: Add a slower fallback path instead of dropping entries when a shard queue exceeds kWays.
      atomicAdd(&local_overflows[local_shard], 1);
    }
  }
  __syncthreads();

  // Phase 3: process each queued shard and apply its fused metadata/copy
  // transition. Warps only reuse their own scratch, so waves can progress
  // independently after routing completes.
  constexpr int kNumWaves = (kLocalShards + kWorkers - 1) / kWorkers;
  for (int wave = 0; wave < kNumWaves; ++wave) {
    const int local_shard = wave * kWorkers + warp;
    if (warp < kWorkers && local_shard < kLocalShards) {
      const int logical_shard = cta_rank + local_shard * NUM_CTAS;
      const int shard_base = logical_shard * kWays;
      uint16_t* queue = queues + local_shard * kWays;
      const int queue_count = queue_counts[local_shard] < kWays ? queue_counts[local_shard] : kWays;
      uint8_t* worker_evictable = evictable + warp * kWays;
      uint8_t* worker_hits = hit_order + warp * kWays;
      uint16_t* worker_misses = miss_indices + warp * kWays;
      process_shard(
          lane,
          lanes_before,
          logical_shard,
          shard_base,
          queue,
          queue_count,
          request_top_k,
          request_output,
          request_tags,
          request_locations,
          request_host,
          request_lru,
          host_cache,
          device_buffer,
          request_counts,
          request_overflows,
          local_overflows[local_shard],
          worker_evictable,
          worker_hits,
          worker_misses,
          item_size_bytes);
    }
  }
}

template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE, int NUM_CTAS, int MIN_BLOCKS_PER_SM>
void load_cache_to_device_buffer_mla_sharded(
    tvm::ffi::TensorView top_k_tokens,
    tvm::ffi::TensorView device_buffer_tokens,
    tvm::ffi::TensorView host_cache_locs,
    tvm::ffi::TensorView device_buffer_locs,
    tvm::ffi::TensorView host_cache,
    tvm::ffi::TensorView device_buffer,
    tvm::ffi::TensorView top_k_device_locs,
    tvm::ffi::TensorView req_pool_indices,
    tvm::ffi::TensorView seq_lens,
    tvm::ffi::TensorView lru_slots,
    tvm::ffi::TensorView split_miss_counts,
    tvm::ffi::TensorView shard_overflows,
    tvm::ffi::TensorView num_real_reqs,
    int64_t item_size_bytes) {
  using namespace host;
  const int batch_size = static_cast<int>(top_k_tokens.shape()[0]);
  const auto device = LaunchKernel::resolve_device(top_k_tokens.device());
  constexpr size_t shared_bytes = ShardedSmemLayout<BLOCK_SIZE, HOT_BUFFER_SIZE, NUM_CTAS>::kBytes;
  LaunchKernel(batch_size * NUM_CTAS, BLOCK_SIZE, device, shared_bytes)(
      sharded_kernel<BLOCK_SIZE, NUM_TOP_K, HOT_BUFFER_SIZE, NUM_CTAS, MIN_BLOCKS_PER_SM>,
      static_cast<const int32_t*>(top_k_tokens.data_ptr()),
      static_cast<int32_t*>(device_buffer_tokens.data_ptr()),
      static_cast<const int64_t*>(host_cache_locs.data_ptr()),
      static_cast<const int32_t*>(device_buffer_locs.data_ptr()),
      static_cast<int32_t*>(top_k_device_locs.data_ptr()),
      static_cast<const int64_t*>(req_pool_indices.data_ptr()),
      static_cast<const int32_t*>(seq_lens.data_ptr()),
      static_cast<uint8_t*>(lru_slots.data_ptr()),
      host_cache.data_ptr(),
      device_buffer.data_ptr(),
      static_cast<int32_t*>(split_miss_counts.data_ptr()),
      static_cast<int32_t*>(shard_overflows.data_ptr()),
      static_cast<const int32_t*>(num_real_reqs.data_ptr()),
      device_buffer_tokens.strides()[0],
      device_buffer_locs.strides()[0],
      host_cache_locs.strides()[0],
      item_size_bytes);
}

}  // namespace
