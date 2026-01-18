#include <torch/extension.h>

#include <cuda_runtime.h>
#include <stdint.h>

constexpr int WARP_SIZE = 32;
constexpr int32_t TOKEN_HIT = 0xFFFFFFFF;

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
    const int32_t* __restrict__ top_k_tokens,
    int32_t* __restrict__ device_buffer_tokens,
    const int64_t* __restrict__ host_cache_locs,
    const int64_t* __restrict__ device_buffer_locs,
    const void* __restrict__ host_cache,
    void* __restrict__ device_buffer,
    int64_t* __restrict__ top_k_device_locs,
    int64_t item_size_bytes) {
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  constexpr int NUM_TOKEN_CHUNKS = (NUM_TOP_K + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int NUM_BUFFER_CHUNKS = (HOT_BUFFER_SIZE + WARP_SIZE - 1) / WARP_SIZE;

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const unsigned int lanes_before = (1u << lane_id) - 1;

  // todo, use hashing for hit detection if top_k is large
  __shared__ int32_t s_top_k_tokens[NUM_TOP_K];
  // Warp-level miss tracking
  // HOT_BUFFER_SIZE is guaranteed to be larger than NUM_TOP_K
  __shared__ int32_t s_chunk_offset[NUM_BUFFER_CHUNKS + 1];

  __shared__ int32_t s_missed_tokens[NUM_TOP_K];
  __shared__ int32_t s_evictable_slots[NUM_TOP_K];
  __shared__ int32_t s_total_misses;

  // Initialize shared memory
  if (tid == 0) {
    s_total_misses = 0;
  }
  for (int i = tid; i < NUM_TOP_K; i += BLOCK_SIZE) {
    s_top_k_tokens[i] = top_k_tokens[i];
  }
  for (int i = tid; i < NUM_BUFFER_CHUNKS + 1; i += BLOCK_SIZE) {
    s_chunk_offset[i] = 0;
  }
  __syncthreads();

  constexpr int ITERATIONS_PER_WARP_BUFFER = (NUM_BUFFER_CHUNKS + NUM_WARPS - 1) / NUM_WARPS;
  int total_evictable = 0;
  for (int iter = 0; iter < ITERATIONS_PER_WARP_BUFFER; iter++) {
    int chunk_idx = warp_id + iter * NUM_WARPS;
    bool has_valid_chunk = chunk_idx < NUM_BUFFER_CHUNKS;

    const int chunk_slot_start = chunk_idx * WARP_SIZE;
    const int buf_slot = chunk_slot_start + lane_id;
    const bool has_valid_slot = has_valid_chunk && (buf_slot < HOT_BUFFER_SIZE);

    int32_t my_buffer_token = has_valid_slot ? device_buffer_tokens[buf_slot] : -1;
    int my_found_top_k_idx = -1;
    int local_evictable_offset = 0;

    if (has_valid_chunk) {
      for (int top_k_base = 0; top_k_base < NUM_TOP_K; top_k_base += WARP_SIZE) {
        // Each lane loads one top_k token into register
        int top_k_idx = top_k_base + lane_id;
        int32_t my_top_k_token = (top_k_idx < NUM_TOP_K) ? s_top_k_tokens[top_k_idx] : -1;

// 32 shuffle rotations for all-to-all comparison
#pragma unroll
        for (int rot = 0; rot < WARP_SIZE; rot++) {
          // Get top_k token from lane (lane_id + rot) % 32
          int src_lane = (lane_id + rot) & 31;  // Fast modulo for power of 2
          int32_t rotated_top_k = __shfl_sync(0xFFFFFFFF, my_top_k_token, src_lane);
          int rotated_top_k_idx = top_k_base + src_lane;

          // Compare my buffer token against rotated top_k token
          if (my_buffer_token >= 0 && my_buffer_token == rotated_top_k && rotated_top_k_idx < NUM_TOP_K) {
            my_found_top_k_idx = rotated_top_k_idx;
          }
        }
      }
    }

    // Record hits: if my buffer token matched a top_k token
    if (my_found_top_k_idx >= 0) {
      // reuse s_top_k_tokens to mark hits
      s_top_k_tokens[my_found_top_k_idx] = TOKEN_HIT;
      top_k_device_locs[my_found_top_k_idx] = device_buffer_locs[buf_slot];
    }
    __syncthreads();

    bool is_evictable = has_valid_slot && (my_found_top_k_idx == -1);
    if (has_valid_chunk) {
      const unsigned int evictable_mask = __ballot_sync(0xFFFFFFFF, is_evictable);
      local_evictable_offset = __popc(evictable_mask & lanes_before);
      const int warp_evictable_count = __popc(evictable_mask);
      if (lane_id == 0) {
        s_chunk_offset[chunk_idx + 1] = warp_evictable_count;
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      total_evictable =
          warp_inclusive_scan(s_chunk_offset, lane_id, chunk_idx + 1, NUM_BUFFER_CHUNKS + 1, total_evictable);
    }
    __syncthreads();

    if (is_evictable) {
      int evictable_offset = s_chunk_offset[chunk_idx] + local_evictable_offset;
      if (evictable_offset < NUM_TOP_K) {
        s_evictable_slots[evictable_offset] = buf_slot;
      }
    }
    __syncthreads();
  }

  // Reset offsets for next phase
  for (int i = tid; i < NUM_BUFFER_CHUNKS + 1; i += BLOCK_SIZE) {
    s_chunk_offset[i] = 0;
  }
  __syncthreads();

  constexpr int ITERATIONS_PER_WARP_TOKEN = (NUM_TOKEN_CHUNKS + NUM_WARPS - 1) / NUM_WARPS;
  for (int iter = 0; iter < ITERATIONS_PER_WARP_TOKEN; iter++) {
    int chunk_idx = warp_id + iter * NUM_WARPS;
    bool has_valid_chunk = chunk_idx < NUM_TOKEN_CHUNKS;

    const int chunk_token_start = chunk_idx * WARP_SIZE;
    const int my_token_idx = chunk_token_start + lane_id;
    const bool has_valid_token = has_valid_chunk && (my_token_idx < NUM_TOP_K);

    int32_t my_token = 0;
    bool is_miss = false;
    int local_miss_offset = 0;

    if (has_valid_token) {
      is_miss = s_top_k_tokens[my_token_idx] != TOKEN_HIT;
      if (is_miss) {
        my_token = s_top_k_tokens[my_token_idx];
      }
    }

    // Intra-warp communication for miss counting
    const unsigned int miss_mask = __ballot_sync(0xFFFFFFFF, is_miss);
    if (has_valid_chunk) {
      local_miss_offset = __popc(miss_mask & lanes_before);
      const int warp_miss_count = __popc(miss_mask);
      if (lane_id == 0) {
        s_chunk_offset[chunk_idx + 1] = warp_miss_count;
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      s_total_misses =
          warp_inclusive_scan(s_chunk_offset, lane_id, chunk_idx + 1, NUM_TOKEN_CHUNKS + 1, s_total_misses);
    }
    __syncthreads();

    if (is_miss) {
      int miss_offset = s_chunk_offset[chunk_idx] + local_miss_offset;
      int evict_slot = s_evictable_slots[miss_offset];
      s_missed_tokens[miss_offset] = my_token;
      top_k_device_locs[my_token_idx] = device_buffer_locs[evict_slot];
      device_buffer_tokens[evict_slot] = my_token;
    }
    __syncthreads();
  }

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
