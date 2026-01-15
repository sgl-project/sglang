#include <torch/extension.h>

#include <cuda_runtime.h>
#include <stdint.h>

constexpr int WARP_SIZE = 32;
constexpr uint16_t NOT_PRESENT = 0xFFFF;

__device__ __forceinline__ void
transfer_item_warp(int32_t lane_id, const void* src_addr, void* dst_addr, int64_t item_size_bytes) {
  const uint64_t* __restrict__ src = static_cast<const uint64_t*>(src_addr);
  uint64_t* __restrict__ dst = static_cast<uint64_t*>(dst_addr);
  const int total_chunks = item_size_bytes / sizeof(uint64_t);

#pragma unroll
  for (int j = lane_id; j < total_chunks; j += WARP_SIZE) {
    uint64_t tmp;
    asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src + j) : "memory");
    asm volatile("st.global.cg.b64 [%0],%1;" ::"l"(dst + j), "l"(tmp) : "memory");
  }
}

__device__ __forceinline__ int warp_inclusive_scan(int* s_data, int lane_id, int offset, int count, int accumulator) {
  int idx = lane_id + offset;
  int val = (idx < count) ? s_data[idx] : 0;

#pragma unroll
  for (int i = 1; i < 32; i *= 2) {
    int n = __shfl_up_sync(0xffffffff, val, i);
    if (lane_id >= i) val += n;
  }
  val += accumulator;
  if (idx < count) {
    s_data[idx] = val;
  }
  accumulator = __shfl_sync(0xffffffff, val, 31);
  return accumulator;
}

// todo, each block for a request
template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE>
__global__ void load_cache_to_device_buffer(
    const uint32_t* __restrict__ top_k_tokens,
    uint32_t* __restrict__ device_buffer_tokens,
    uint16_t* __restrict__ token_residence_mapping,
    const int64_t* __restrict__ host_cache_locs,
    const int64_t* __restrict__ device_buffer_locs,
    const void* __restrict__ host_cache,
    void* __restrict__ device_buffer,
    int64_t* __restrict__ top_k_device_locs,
    uint32_t max_tokens,
    int64_t item_size_bytes) {
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  constexpr int NUM_CHUNKS = (NUM_TOP_K + WARP_SIZE - 1) / WARP_SIZE;

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const unsigned int lanes_before = (1u << lane_id) - 1;

  // Protected slots bitmap
  // todo: avoid bank conflicts
  __shared__ bool s_protected_bitmap[HOT_BUFFER_SIZE];
  // Warp-level miss tracking
  __shared__ int32_t s_chunk_miss_offset[NUM_CHUNKS + 1];
  __shared__ int32_t s_missed_tokens[NUM_TOP_K];
  __shared__ int16_t s_missed_tokens_idx[NUM_TOP_K];
  // Evictable slots (only need num_misses worth)
  __shared__ int32_t s_evictable_slots[NUM_TOP_K];
  __shared__ int32_t s_total_misses;

  // Initialize shared memory
  if (tid == 0) {
    s_total_misses = 0;
  }
  // Clear bitmaps (warp-efficient: each thread clears one word)
  for (int i = tid; i < HOT_BUFFER_SIZE; i += BLOCK_SIZE) {
    s_protected_bitmap[i] = false;
  }
  for (int i = tid; i < NUM_CHUNKS + 1; i += BLOCK_SIZE) {
    s_chunk_miss_offset[i] = 0;
  }
  __syncthreads();

  // ===== Phase 1: Hit and Miss Detection ====
  constexpr int ITERATIONS_PER_WARP = (NUM_CHUNKS + NUM_WARPS - 1) / NUM_WARPS;
  for (int iter = 0; iter < ITERATIONS_PER_WARP; iter++) {
    int chunk_idx = warp_id + iter * NUM_WARPS;
    bool has_valid_chunk = chunk_idx < NUM_CHUNKS;

    const int chunk_token_start = chunk_idx * WARP_SIZE;
    const int my_token_idx = chunk_token_start + lane_id;
    const bool has_valid_token = has_valid_chunk && (my_token_idx < NUM_TOP_K);

    int32_t my_token = 0;
    bool is_hit = false;
    bool is_miss = false;
    int32_t my_device_slot = -1;
    int local_miss_offset = 0;

    if (has_valid_token) {
      my_token = top_k_tokens[my_token_idx];
      uint16_t slot = token_residence_mapping[my_token];
      is_hit = (slot != NOT_PRESENT);

      if (is_hit) {
        my_device_slot = static_cast<int32_t>(slot);
        s_protected_bitmap[my_device_slot] = true;
        top_k_device_locs[my_token_idx] = device_buffer_locs[my_device_slot];
      } else {
        is_miss = true;
      }
    }

    // Intra-warp communication for miss counting
    const unsigned int miss_mask = __ballot_sync(0xFFFFFFFF, is_miss);
    if (has_valid_chunk) {
      local_miss_offset = __popc(miss_mask & lanes_before);
      const int warp_miss_count = __popc(miss_mask);
      if (lane_id == 0) {
        s_chunk_miss_offset[chunk_idx + 1] = warp_miss_count;
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      s_total_misses = warp_inclusive_scan(s_chunk_miss_offset, lane_id, chunk_idx + 1, NUM_CHUNKS + 1, s_total_misses);
    }
    __syncthreads();

    if (is_miss) {
      int miss_offset = s_chunk_miss_offset[chunk_idx] + local_miss_offset;
      s_missed_tokens[miss_offset] = my_token;
      s_missed_tokens_idx[miss_offset] = static_cast<int16_t>(my_token_idx);
    }
    __syncthreads();
  }

  // ===== Phase 2: Find Evictable Slots =====
  // todo: further parallelize this if needed
  if (warp_id == 0) {
    int found = 0;
    int base_slot = 0;

    while (found < s_total_misses && base_slot < HOT_BUFFER_SIZE) {
      const int my_slot = base_slot + lane_id;
      bool is_evictable = false;

      if (my_slot < HOT_BUFFER_SIZE) {
        is_evictable = !s_protected_bitmap[my_slot];
      }
      // Warp ballot to find evictable slots
      const unsigned int evict_mask = __ballot_sync(0xFFFFFFFF, is_evictable);
      const int num_evictable = __popc(evict_mask);
      const int need = s_total_misses - found;
      const int use_count = min(num_evictable, need);

      // Each evictable lane stores its slot
      if (is_evictable) {
        const int my_pos = __popc(evict_mask & lanes_before);
        if (my_pos < use_count) {
          s_evictable_slots[found + my_pos] = my_slot;
        }
      }

      found += use_count;
      base_slot += WARP_SIZE;
    }
  }
  __syncthreads();

  // ===== Phase 3a: Metadata Update (all threads in parallel) =====
  // Each thread handles one or more misses
  for (int miss_idx = tid; miss_idx < s_total_misses; miss_idx += BLOCK_SIZE) {
    const int32_t miss_token = s_missed_tokens[miss_idx];
    const int evict_slot = s_evictable_slots[miss_idx];
    const int top_k_idx = s_missed_tokens_idx[miss_idx];
    const int32_t old_token = device_buffer_tokens[evict_slot];

    // Clear old token's residence mapping
    if (old_token >= 0 && old_token < max_tokens) {
      token_residence_mapping[old_token] = NOT_PRESENT;
    }
    // Set new token's mapping
    token_residence_mapping[miss_token] = static_cast<uint16_t>(evict_slot);
    device_buffer_tokens[evict_slot] = miss_token;
    top_k_device_locs[top_k_idx] = device_buffer_locs[evict_slot];
  }
  __syncthreads();

  for (int miss_idx = warp_id; miss_idx < s_total_misses; miss_idx += NUM_WARPS) {
    const int32_t miss_token = s_missed_tokens[miss_idx];
    const int evict_slot = s_evictable_slots[miss_idx];

    const int64_t src_offset = host_cache_locs[miss_token] * item_size_bytes;
    const int64_t dst_offset = device_buffer_locs[evict_slot] * item_size_bytes;

    transfer_item_warp(
        lane_id,
        static_cast<const char*>(host_cache) + src_offset,
        static_cast<char*>(device_buffer) + dst_offset,
        item_size_bytes);
  }
}
