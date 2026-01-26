#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int WARP_SIZE = 32;
constexpr int32_t TOKEN_HIT = 0xFFFFFFFF;

__device__ __forceinline__ int warpReduceMaxInt(int value) {
  value = max(value, __shfl_xor_sync(0xFFFFFFFF, value, 16));
  value = max(value, __shfl_xor_sync(0xFFFFFFFF, value, 8));
  value = max(value, __shfl_xor_sync(0xFFFFFFFF, value, 4));
  value = max(value, __shfl_xor_sync(0xFFFFFFFF, value, 2));
  value = max(value, __shfl_xor_sync(0xFFFFFFFF, value, 1));
  return value;
}

__device__ __forceinline__ int blockReduceMaxInt(int value) {
  static __shared__ int warpLevelMaxs[WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;

  value = warpReduceMaxInt(value);

  if (laneId == 0) warpLevelMaxs[warpId] = value;
  __syncthreads();

  value = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelMaxs[laneId] : 0;
  if (warpId == 0) value = warpReduceMaxInt(value);

  return value;
}

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

__device__ __forceinline__ int
warp_inclusive_scan(int* s_data, int lane_id, int offset, int count, int accumulator) {
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

// Each block processes one request
template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE, bool IsMLA>
__global__ void load_cache_to_device_buffer_kernel(
    const int32_t* __restrict__ top_k_tokens,
    int32_t* __restrict__ device_buffer_tokens,
    const int64_t* __restrict__ host_cache_locs,
    const int32_t* __restrict__ device_buffer_locs,
    const void* __restrict__ host_cache_k,
    const void* __restrict__ host_cache_v,
    void* __restrict__ device_buffer_k,
    void* __restrict__ device_buffer_v,
    int32_t* __restrict__ top_k_device_locs,
    const int32_t* __restrict__ page_table,
    int16_t* __restrict__ diff_map,
    const int64_t* __restrict__ req_pool_indices,
    const bool* __restrict__ sparse_mask,
    const int64_t* __restrict__ seq_lens,
    int64_t buffer_stride_0,
    int64_t buffer_stride_1,
    int64_t host_stride,
    int64_t page_table_stride,
    int64_t diff_map_stride,
    int64_t page_size,
    int64_t layer_id,
    int64_t item_size_bytes) {
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  constexpr int NUM_TOKEN_CHUNKS = (NUM_TOP_K + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int NUM_BUFFER_CHUNKS = (HOT_BUFFER_SIZE + WARP_SIZE - 1) / WARP_SIZE;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int64_t rid = req_pool_indices[bid];
  const bool sparse_mask_val = sparse_mask[bid];
  const int64_t seq_len = seq_lens[bid] - 1;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const unsigned int lanes_before = (1u << lane_id) - 1;

  // Calculate offsets for this request
  const int top_k_offset = bid * NUM_TOP_K;
  const int buffer_offset = rid * buffer_stride_0 + layer_id * buffer_stride_1;
  const int host_offset = rid * host_stride;
  const int page_table_offset = rid * page_table_stride;
  const int diff_map_offset = bid * diff_map_stride;

  // Get pointers for this request
  const int32_t* my_top_k_tokens = top_k_tokens + top_k_offset;
  int32_t* my_device_buffer_tokens = device_buffer_tokens + buffer_offset;
  const int64_t* my_host_cache_locs = host_cache_locs + host_offset;
  const int32_t* my_device_buffer_locs = device_buffer_locs + buffer_offset;
  int32_t* my_top_k_device_locs = top_k_device_locs + top_k_offset;
  const int32_t* my_page_table = page_table + page_table_offset;
  int16_t* my_diff_map = diff_map + diff_map_offset;

  // Fast path: if sparse is disabled
  if (!sparse_mask_val || (seq_len <= 0)) {
    for (int i = tid; i < NUM_TOP_K; i += BLOCK_SIZE) {
      int32_t top_k_val = my_top_k_tokens[i];
      if (top_k_val >= 0) {
        int32_t page_start = my_page_table[top_k_val * page_size];
        my_top_k_device_locs[i] = page_start / page_size;
      }
    }
    return;
  }

  __shared__ int32_t s_top_k_tokens[NUM_TOP_K];
  __shared__ int32_t s_chunk_offset[NUM_BUFFER_CHUNKS + 1];
  __shared__ int32_t s_missed_tokens[NUM_TOP_K];
  __shared__ int32_t s_evictable_slots[NUM_TOP_K];
  __shared__ int32_t s_total_misses;
  __shared__ int32_t s_top_k_max;
  __shared__ int32_t s_device_buffer_max;

  // Initialize shared memory
  if (tid == 0) {
    s_total_misses = 0;
    s_top_k_max = 0;
    s_device_buffer_max = 0;
  }

  int32_t top_k_max_value = 0;
  int32_t device_buffer_max_value = 0;

  for (int i = tid; i < NUM_TOP_K; i += BLOCK_SIZE) {
    int32_t top_k_val = my_top_k_tokens[i];
    int32_t device_buffer_val = my_device_buffer_tokens[i];
    my_diff_map[top_k_val] = i;

    top_k_max_value = max(top_k_max_value, top_k_val);
    device_buffer_max_value = max(device_buffer_max_value, device_buffer_val);

    s_top_k_tokens[i] = top_k_val;
  }

  top_k_max_value = blockReduceMaxInt(top_k_max_value);
  device_buffer_max_value = blockReduceMaxInt(device_buffer_max_value);

  if (tid == 0) {
    s_top_k_max = top_k_max_value;
    s_device_buffer_max = device_buffer_max_value;
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

    int32_t my_buffer_token = has_valid_slot ? my_device_buffer_tokens[buf_slot] : -1;
    int local_evictable_offset = 0;
    int32_t prev_max = s_device_buffer_max;
    int32_t curr_max = s_top_k_max;
    bool cross_page = curr_max != prev_max;
    int my_found_top_k_idx = my_diff_map[my_buffer_token];

    if (cross_page & my_buffer_token == prev_max) {
      my_found_top_k_idx = my_diff_map[curr_max];
      my_device_buffer_tokens[buf_slot] = curr_max;
    }

    // Record hits
    if (my_found_top_k_idx >= 0 && buf_slot < HOT_BUFFER_SIZE) {
      s_top_k_tokens[my_found_top_k_idx] = TOKEN_HIT;
      my_top_k_device_locs[my_found_top_k_idx] = my_device_buffer_locs[buf_slot];
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

  for (int i = tid; i < NUM_TOP_K; i += BLOCK_SIZE) {
    int32_t top_k_val = my_top_k_tokens[i];
    my_diff_map[top_k_val] = -1;
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
      my_top_k_device_locs[my_token_idx] = my_device_buffer_locs[evict_slot];
      my_device_buffer_tokens[evict_slot] = my_token;
    }
    __syncthreads();
  }

  // Transfer missed items from host cache to device buffer with paging support
  for (int miss_idx = warp_id; miss_idx < s_total_misses; miss_idx += NUM_WARPS) {
    const int32_t miss_token = s_missed_tokens[miss_idx];
    const int evict_slot = s_evictable_slots[miss_idx];

#pragma unroll
    for (int page_offset = 0; page_offset < page_size; page_offset++) {
      const int64_t src_offset = my_host_cache_locs[miss_token * page_size + page_offset] * item_size_bytes;
      const int64_t dst_offset = (my_device_buffer_locs[evict_slot] * page_size + page_offset) * item_size_bytes;

      transfer_item_warp(
          lane_id,
          static_cast<const char*>(host_cache_k) + src_offset,
          static_cast<char*>(device_buffer_k) + dst_offset,
          item_size_bytes);
      if constexpr (!IsMLA) {
        transfer_item_warp(
            lane_id,
            static_cast<const char*>(host_cache_v) + src_offset,
            static_cast<char*>(device_buffer_v) + dst_offset,
            item_size_bytes);
      }
    }
  }
}

template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE, bool IsMLA>
void load_cache_to_device_buffer(
    tvm::ffi::TensorView top_k_tokens,
    tvm::ffi::TensorView device_buffer_tokens,
    tvm::ffi::TensorView host_cache_locs,
    tvm::ffi::TensorView device_buffer_locs,
    tvm::ffi::TensorView host_cache_k,
    tvm::ffi::TensorView host_cache_v,
    tvm::ffi::TensorView device_buffer_k,
    tvm::ffi::TensorView device_buffer_v,
    tvm::ffi::TensorView top_k_device_locs,
    tvm::ffi::TensorView page_table,
    tvm::ffi::TensorView diff_map,
    tvm::ffi::TensorView req_pool_indices,
    tvm::ffi::TensorView sparse_mask,
    tvm::ffi::TensorView seq_lens,
    int64_t page_size,
    int64_t layer_id,
    int64_t item_size_bytes) {
  using namespace host;

  // Extract dimensions and strides
  const int64_t bs = top_k_tokens.shape()[0];
  const int64_t host_stride = host_cache_locs.shape()[1];
  // Get buffer strides from tensor (critical for correct memory access!)
  const int64_t buffer_stride_0 = device_buffer_tokens.strides()[0];
  const int64_t buffer_stride_1 = device_buffer_tokens.strides()[1];
  const int64_t page_table_stride = page_table.shape()[1];
  const int64_t diff_map_stride = diff_map.shape()[1];

  // Extract raw pointers
  const int32_t* top_k_tokens_ptr = static_cast<const int32_t*>(top_k_tokens.data_ptr());
  int32_t* device_buffer_tokens_ptr = static_cast<int32_t*>(device_buffer_tokens.data_ptr());
  const int64_t* host_cache_locs_ptr = static_cast<const int64_t*>(host_cache_locs.data_ptr());
  const int32_t* device_buffer_locs_ptr = static_cast<const int32_t*>(device_buffer_locs.data_ptr());
  const void* host_cache_k_ptr = host_cache_k.data_ptr();
  // For V cache: set to nullptr if MLA mode OR if tensor is empty (ndim == 0)
  const void* host_cache_v_ptr = (IsMLA || host_cache_v.ndim() == 0) ? nullptr : host_cache_v.data_ptr();
  void* device_buffer_k_ptr = device_buffer_k.data_ptr();
  void* device_buffer_v_ptr = (IsMLA || device_buffer_v.ndim() == 0) ? nullptr : device_buffer_v.data_ptr();
  int32_t* top_k_device_locs_ptr = static_cast<int32_t*>(top_k_device_locs.data_ptr());
  const int32_t* page_table_ptr = static_cast<const int32_t*>(page_table.data_ptr());
  int16_t* diff_map_ptr = static_cast<int16_t*>(diff_map.data_ptr());
  const int64_t* req_pool_indices_ptr = static_cast<const int64_t*>(req_pool_indices.data_ptr());
  const bool* sparse_mask_ptr = static_cast<const bool*>(sparse_mask.data_ptr());
  const int64_t* seq_lens_ptr = static_cast<const int64_t*>(seq_lens.data_ptr());

  const auto device = LaunchKernel::resolve_device(top_k_tokens.device());

  // Launch kernel with batch size as grid dimension
  LaunchKernel(bs, BLOCK_SIZE, device)(
      load_cache_to_device_buffer_kernel<BLOCK_SIZE, NUM_TOP_K, HOT_BUFFER_SIZE, IsMLA>,
      top_k_tokens_ptr,
      device_buffer_tokens_ptr,
      host_cache_locs_ptr,
      device_buffer_locs_ptr,
      host_cache_k_ptr,
      host_cache_v_ptr,
      device_buffer_k_ptr,
      device_buffer_v_ptr,
      top_k_device_locs_ptr,
      page_table_ptr,
      diff_map_ptr,
      req_pool_indices_ptr,
      sparse_mask_ptr,
      seq_lens_ptr,
      buffer_stride_0,
      buffer_stride_1,
      host_stride,
      page_table_stride,
      diff_map_stride,
      page_size,
      layer_id,
      item_size_bytes);
}

}  // namespace
