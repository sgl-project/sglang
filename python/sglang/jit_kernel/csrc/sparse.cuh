#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_runtime.h>
#include <stdexcept>
#include <stdint.h>
#include <string>

namespace device::warp {

template <typename T, std::size_t N>
struct device_vec {
  T data[N];
};

namespace details {

template <std::size_t kUnit>
inline constexpr auto get_mem_package() {
  if constexpr (kUnit == 16) {
    return uint4{};
  } else if constexpr (kUnit == 8) {
    return uint2{};
  } else if constexpr (kUnit == 4) {
    return uint1{};
  } else {
    static_assert(kUnit == 16 || kUnit == 8 || kUnit == 4, "Unsupported memory package size");
  }
}

template <std::size_t kBytes, std::size_t kUnit>
using mem_package_t = decltype(get_mem_package<kUnit>());

__always_inline __device__ auto load_nc(const uint1* __restrict__ src) -> uint1 {
  uint32_t tmp;
  asm volatile("ld.global.cs.b32 %0,[%1];" : "=r"(tmp) : "l"(src));
  return uint1{tmp};
}

__always_inline __device__ auto load_nc(const uint2* __restrict__ src) -> uint2 {
  uint32_t tmp0, tmp1;
  asm volatile("ld.global.cs.v2.b32 {%0,%1},[%2];" : "=r"(tmp0), "=r"(tmp1) : "l"(src));
  return uint2{tmp0, tmp1};
}

__always_inline __device__ auto load_nc(const uint4* __restrict__ src) -> uint4 {
  uint32_t tmp0, tmp1, tmp2, tmp3;
  asm volatile("ld.global.cs.v4.b32 {%0,%1,%2,%3},[%4];" : "=r"(tmp0), "=r"(tmp1), "=r"(tmp2), "=r"(tmp3) : "l"(src));
  return uint4{tmp0, tmp1, tmp2, tmp3};
}

__always_inline __device__ void store_nc(uint1* __restrict__ dst, const uint1& value) {
  uint32_t tmp = value.x;
  asm volatile("st.global.cs.b32 [%0],%1;" ::"l"(dst), "r"(tmp));
}

__always_inline __device__ void store_nc(uint2* __restrict__ dst, const uint2& value) {
  uint32_t tmp0 = value.x;
  uint32_t tmp1 = value.y;
  asm volatile("st.global.cs.v2.b32 [%0],{%1,%2};" ::"l"(dst), "r"(tmp0), "r"(tmp1));
}

__always_inline __device__ void store_nc(uint4* __restrict__ dst, const uint4& value) {
  uint32_t tmp0 = value.x;
  uint32_t tmp1 = value.y;
  uint32_t tmp2 = value.z;
  uint32_t tmp3 = value.w;
  asm volatile("st.global.cs.v4.b32 [%0],{%1,%2,%3,%4};" ::"l"(dst), "r"(tmp0), "r"(tmp1), "r"(tmp2), "r"(tmp3));
}

}  // namespace details

template <std::size_t kBytes, std::size_t kUnit, std::size_t kThreads>
__always_inline __device__ auto load_vec(const void* __restrict__ src) {
  using Package = details::mem_package_t<kBytes, kUnit>;
  constexpr auto kBytesPerLoop = sizeof(Package) * kThreads;
  constexpr auto kLoopCount = kBytes / kBytesPerLoop;
  static_assert(kBytes % kBytesPerLoop == 0, "kBytes must be multiple of 128 bytes");

  const auto src_packed = static_cast<const Package*>(src);
  const auto lane_id = threadIdx.x % kThreads;
  device_vec<Package, kLoopCount> vec;

#pragma unroll kLoopCount
  for (std::size_t i = 0; i < kLoopCount; ++i) {
    const auto j = i * kThreads + lane_id;
    vec.data[i] = details::load_nc(src_packed + j);
  }

  return vec;
}

template <std::size_t kBytes, std::size_t kUnit, std::size_t kThreads, typename Tp>
__always_inline __device__ void store_vec(void* __restrict__ dst, const Tp& vec) {
  using Package = details::mem_package_t<kBytes, kUnit>;
  constexpr auto kBytesPerLoop = sizeof(Package) * kThreads;
  constexpr auto kLoopCount = kBytes / kBytesPerLoop;
  static_assert(kBytes % kBytesPerLoop == 0, "kBytes must be multiple of 128 bytes");
  static_assert(std::is_same_v<Tp, device_vec<Package, kLoopCount>>);

  const auto dst_packed = static_cast<Package*>(dst);
  const auto lane_id = threadIdx.x % kThreads;

#pragma unroll kLoopCount
  for (std::size_t i = 0; i < kLoopCount; ++i) {
    const auto j = i * kThreads + lane_id;
    details::store_nc(dst_packed + j, vec.data[i]);
  }
}

}  // namespace device::warp

namespace {

#define DEBUG_SPARSE_IMA 0

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

template <std::size_t kItemSize, int BLOCK_SIZE, int NUM_TOP_K>
__global__ void parallel_transfer_kernel(
    const int64_t* __restrict__ transfer_tasks_src,
    const int64_t* __restrict__ transfer_tasks_dst,
    const void* __restrict__ host_cache_k,
    const void* __restrict__ host_cache_v,
    void* __restrict__ device_buffer_k,
    void* __restrict__ device_buffer_v,
    int64_t total_tasks,
    int64_t page_size,
    bool is_mla) {
  constexpr int WARP_THREADS = WARP_SIZE;
  constexpr int kGranularity = 128 / WARP_THREADS;

  const int64_t tasks_per_block = NUM_TOP_K * page_size;
  const int64_t stride_per_block = tasks_per_block + 1;

  const int global_task_id = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + threadIdx.x / WARP_SIZE;
  if (global_task_id >= total_tasks) return;

  const int block_id = global_task_id / tasks_per_block;
  const int task_in_block = global_task_id % tasks_per_block;

  // Layout: [task_0...task_N | count] per block
  const int64_t block_base = block_id * stride_per_block;
  const int64_t count_idx = block_base + tasks_per_block;
  const int64_t valid_count = transfer_tasks_src[count_idx];

  if (task_in_block >= valid_count) return;

  const int64_t task_idx = block_base + task_in_block;
  const int64_t src_loc = transfer_tasks_src[task_idx];
  const int64_t dst_loc = transfer_tasks_dst[task_idx];

  const int64_t src_offset = src_loc * kItemSize;
  const int64_t dst_offset = dst_loc * kItemSize;

  const auto src_k = static_cast<const char*>(host_cache_k) + src_offset;
  const auto dst_k = static_cast<char*>(device_buffer_k) + dst_offset;

  const auto vec_k = device::warp::load_vec<kItemSize, kGranularity, WARP_THREADS>(src_k);
  device::warp::store_vec<kItemSize, kGranularity, WARP_THREADS>(dst_k, vec_k);

  if (!is_mla) {
    const auto src_v = static_cast<const char*>(host_cache_v) + src_offset;
    const auto dst_v = static_cast<char*>(device_buffer_v) + dst_offset;
    const auto vec_v = device::warp::load_vec<kItemSize, kGranularity, WARP_THREADS>(src_v);
    device::warp::store_vec<kItemSize, kGranularity, WARP_THREADS>(dst_v, vec_v);
  }
}

// Each block processes one request
// IndexT: type for req_pool_indices and seq_lens (int32_t or int64_t), The cuda graph mode requires int32_t
template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE, bool IsMLA, typename IndexT>
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
    const IndexT* __restrict__ req_pool_indices,
    const bool* __restrict__ sparse_mask,
    const IndexT* __restrict__ seq_lens,
    int16_t* __restrict__ lru_slots,
    int64_t* __restrict__ transfer_tasks_src,
    int64_t* __restrict__ transfer_tasks_dst,
    int64_t buffer_stride_0,
    int64_t buffer_stride_1,
    int64_t host_stride,
    int64_t page_table_stride,
    int64_t diff_map_stride,
    int64_t lru_slot_stride_0,
    int64_t lru_slot_stride_1,
    int64_t top_k_tokens_stride,
    int64_t top_k_device_locs_stride,
    int64_t page_size,
    int64_t layer_id,
    int64_t item_size_bytes) {
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  constexpr int NUM_TOKEN_CHUNKS = (NUM_TOP_K + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int LRU_SIZE = HOT_BUFFER_SIZE - 1;
  constexpr int NUM_BUFFER_CHUNKS = (LRU_SIZE + WARP_SIZE - 1) / WARP_SIZE;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int64_t rid = req_pool_indices[bid];
  const bool sparse_mask_val = sparse_mask[bid];
  const int64_t seq_len = seq_lens[bid] - 1;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const unsigned int lanes_before = ((unsigned int)1 << lane_id) - 1;

  // Calculate offsets for this request
  const int top_k_tokens_offset = bid * top_k_tokens_stride;
  const int top_k_device_locs_offset = bid * top_k_device_locs_stride;
  const int buffer_offset = rid * buffer_stride_0 + layer_id * buffer_stride_1;
  const int host_offset = rid * host_stride;
  const int page_table_offset = rid * page_table_stride;
  const int diff_map_offset = bid * diff_map_stride;
  const int lru_slot_offset = rid * lru_slot_stride_0 + layer_id * lru_slot_stride_1;

  const int32_t* my_top_k_tokens = top_k_tokens + top_k_tokens_offset;
  int32_t* my_device_buffer_tokens = device_buffer_tokens + buffer_offset;
  const int64_t* my_host_cache_locs = host_cache_locs + host_offset;
  const int32_t* my_device_buffer_locs = device_buffer_locs + buffer_offset;
  int32_t* my_top_k_device_locs = top_k_device_locs + top_k_device_locs_offset;
  const int32_t* my_page_table = page_table + page_table_offset;
  int16_t* my_diff_map = diff_map + diff_map_offset;
  int16_t* my_lru_slots = lru_slots + lru_slot_offset;

  // Fast path: if sparse is disabled; Building page table directly
  if (!sparse_mask_val || (seq_len <= 0)) {
    for (int i = tid; i < NUM_TOP_K; i += BLOCK_SIZE) {
      int32_t top_k_val = my_top_k_tokens[i];
      if (top_k_val >= 0) {
        int32_t page_start = my_page_table[top_k_val * page_size];
        my_top_k_device_locs[i] = page_start / page_size;
      }
    }
    if (tid == 0) {
      const int64_t tasks_per_block = NUM_TOP_K * page_size;
      const int64_t stride_per_block = tasks_per_block + 1;
      const int64_t block_base = bid * stride_per_block;
      const int64_t count_idx = block_base + tasks_per_block;
      transfer_tasks_src[count_idx] = 0;
    }
    return;
  }

  __shared__ int32_t s_top_k_tokens[NUM_TOP_K];
  __shared__ int32_t s_chunk_offset[NUM_BUFFER_CHUNKS + 1];
  __shared__ int32_t s_missed_tokens[NUM_TOP_K];
  __shared__ int32_t s_evictable_slots[NUM_TOP_K];
  __shared__ int32_t s_total_misses;
  __shared__ int32_t s_total_hits;
  __shared__ int32_t s_total_evictable;
  __shared__ int32_t s_newest_hit;
  __shared__ int32_t s_top_k_max;
  __shared__ bool s_lru_bitmap[HOT_BUFFER_SIZE];
  __shared__ int16_t s_lru_slots_out[LRU_SIZE];
#if DEBUG_SPARSE_IMA
  __shared__ int s_debug_once;
  __shared__ int s_debug_count;
#endif

  // Initialize shared memory
  if (tid == 0) {
    s_total_misses = 0;
    s_total_hits = 0;
    s_total_evictable = 0;
    s_newest_hit = 0;
    s_top_k_max = 0;
#if DEBUG_SPARSE_IMA
    s_debug_once = 0;
    s_debug_count = 0;
#endif
  }

  const int newest_slot = HOT_BUFFER_SIZE - 1;
  // For page-wise topk, use page id.
  const int32_t newest_token =
      (seq_len >= 0)
          ? static_cast<int32_t>((page_size > 1) ? (seq_len / page_size) : seq_len)
          : -1;
  int32_t top_k_max_value = 0;

  for (int i = tid; i < HOT_BUFFER_SIZE; i += BLOCK_SIZE) {
    if (i < NUM_TOP_K) {
      int32_t top_k_val = my_top_k_tokens[i];
#if DEBUG_SPARSE_IMA
      if ((top_k_val < 0 || top_k_val >= diff_map_stride) &&
          atomicAdd(&s_debug_count, 1) < 8) {
        printf("[ima bad topk] bid=%d rid=%lld idx=%d val=%d diff_map=%lld\n",
               bid, (long long)rid, i, (int)top_k_val, (long long)diff_map_stride);
      }
#endif
      my_diff_map[top_k_val] = i;
      s_top_k_tokens[i] = top_k_val;
      top_k_max_value = max(top_k_max_value, top_k_val);
    }
    s_lru_bitmap[i] = false;
#if DEBUG_SPARSE_IMA
    // no-op: keep shared memory usage low for debug builds
#endif
  }
  for (int i = tid; i < NUM_TOP_K; i += BLOCK_SIZE) {
    s_evictable_slots[i] = -1;
  }

  top_k_max_value = blockReduceMaxInt(top_k_max_value);
  if (tid == 0) {
    s_top_k_max = top_k_max_value;
#if DEBUG_SPARSE_IMA
    if (atomicAdd(&s_debug_count, 1) < 8) {
      const int newest_topk_idx =
          (newest_token >= 0 && newest_token < diff_map_stride) ? my_diff_map[newest_token] : -1;
      printf("[ima newest state] bid=%d rid=%lld seq_len=%lld newest_token=%d newest_idx=%d newest_slot=%d token_at_slot=%d\n",
             bid, (long long)rid, (long long)seq_len, (int)newest_token, newest_topk_idx,
             newest_slot, (int)my_device_buffer_tokens[newest_slot]);
    }
#endif
  }
  __syncthreads();

  // If topk includes seq_len - 1, bind it to newest_slot and mark as hit.
  if (tid == 0 && newest_token >= 0 && newest_token < diff_map_stride) {
    const int newest_topk_idx = my_diff_map[newest_token];
    if (newest_topk_idx >= 0) {
      s_top_k_tokens[newest_topk_idx] = TOKEN_HIT;
      my_top_k_device_locs[newest_topk_idx] = my_device_buffer_locs[newest_slot];
      my_device_buffer_tokens[newest_slot] = newest_token;
      s_newest_hit = 1;
    }
  }
  __syncthreads();

  for (int i = tid; i < NUM_BUFFER_CHUNKS + 1; i += BLOCK_SIZE) {
    s_chunk_offset[i] = 0;
  }
  __syncthreads();

  constexpr int ITERATIONS_PER_WARP_BUFFER = (NUM_BUFFER_CHUNKS + NUM_WARPS - 1) / NUM_WARPS;
  int total_hit_count = 0;
  for (int iter = 0; iter < ITERATIONS_PER_WARP_BUFFER; iter++) {
    int chunk_idx = warp_id + iter * NUM_WARPS;
    bool has_valid_chunk = chunk_idx < NUM_BUFFER_CHUNKS;

    const int slot_idx = chunk_idx * WARP_SIZE + lane_id;
    const bool has_valid_slot = has_valid_chunk && (slot_idx < LRU_SIZE);
    const int32_t buf_slot = has_valid_slot ? static_cast<int32_t>(my_lru_slots[slot_idx]) : -1;
    const bool has_valid_buf_slot = has_valid_slot && (buf_slot >= 0) && (buf_slot < HOT_BUFFER_SIZE);
#if DEBUG_SPARSE_IMA
    if (has_valid_slot && !has_valid_buf_slot) {
      if (atomicCAS(&s_debug_once, 0, 1) == 0) {
        printf("[ima bad buf_slot] bid=%d rid=%lld slot_idx=%d buf_slot=%d hot=%d\n",
               bid, (long long)rid, slot_idx, (int)buf_slot, HOT_BUFFER_SIZE);
      }
    }
#endif


    int32_t my_buffer_token = has_valid_buf_slot ? my_device_buffer_tokens[buf_slot] : -1;
    int my_found_top_k_idx = my_buffer_token >= 0 ? my_diff_map[my_buffer_token] : -1;
#if DEBUG_SPARSE_IMA
    if ((my_buffer_token < 0 || my_buffer_token >= diff_map_stride) &&
        atomicAdd(&s_debug_count, 1) < 8) {
      printf("[ima bad buffer token] bid=%d rid=%lld slot=%d token=%d diff_map=%lld\n",
             bid, (long long)rid, (int)buf_slot, (int)my_buffer_token, (long long)diff_map_stride);
    }
#endif

    // Record hits
    if (my_found_top_k_idx >= 0 && has_valid_buf_slot) {
      s_top_k_tokens[my_found_top_k_idx] = TOKEN_HIT;
      my_top_k_device_locs[my_found_top_k_idx] = my_device_buffer_locs[buf_slot];
    }
    __syncthreads();

    bool is_hit = my_found_top_k_idx != -1;
    int local_hit_offset = 0;
    if (warp_id == 0) {
      const int base_chunk = iter * NUM_WARPS;
      const int idx = base_chunk + lane_id + 1;
      if (idx < NUM_BUFFER_CHUNKS + 1) {
        s_chunk_offset[idx] = 0;
      }
    }
    __syncthreads();
    if (has_valid_chunk) {
      const unsigned int hit_mask = __ballot_sync(0xFFFFFFFF, is_hit);
      local_hit_offset = __popc(hit_mask & lanes_before);
      int warp_hit_count = __popc(hit_mask);
      if (lane_id == 0) {
        s_chunk_offset[chunk_idx + 1] = warp_hit_count;
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      total_hit_count =
          warp_inclusive_scan(s_chunk_offset, lane_id, chunk_idx + 1, NUM_BUFFER_CHUNKS + 1, total_hit_count);
      if (tid == 0) {
        s_total_hits = total_hit_count;
      }
    }
    __syncthreads();

    if (is_hit && has_valid_buf_slot) {
      int hit_offset = s_chunk_offset[chunk_idx] + local_hit_offset;
      s_lru_slots_out[hit_offset] = buf_slot;
      s_lru_bitmap[buf_slot] = true;
    }
  }
  __syncthreads();

  // Move staged hits to the tail so hits are most recent.
  for (int i = s_total_hits - 1 - tid; i >= 0; i -= BLOCK_SIZE) {
    const int dst = LRU_SIZE - s_total_hits + i;
    s_lru_slots_out[dst] = s_lru_slots_out[i];
  }
  __syncthreads();

  // Second pass to collect evictable slots
  for (int i = tid; i < NUM_BUFFER_CHUNKS + 1; i += BLOCK_SIZE) {
    s_chunk_offset[i] = 0;
  }
  __syncthreads();

  int total_evictable = 0;
  for (int iter = 0; iter < ITERATIONS_PER_WARP_BUFFER; iter++) {
    const int chunk_idx = warp_id + iter * NUM_WARPS;
    const bool has_valid_chunk = chunk_idx < NUM_BUFFER_CHUNKS;

    const int slot_idx = chunk_idx * WARP_SIZE + lane_id;
    const bool has_valid_slot = has_valid_chunk && (slot_idx < LRU_SIZE);
    const int32_t buf_slot = has_valid_slot ? static_cast<int32_t>(my_lru_slots[slot_idx]) : -1;
    const bool has_valid_buf_slot = has_valid_slot && (buf_slot >= 0) && (buf_slot < HOT_BUFFER_SIZE);
#if DEBUG_SPARSE_IMA
    if (has_valid_slot && !has_valid_buf_slot) {
      if (atomicCAS(&s_debug_once, 0, 1) == 0) {
        printf("[ima bad buf_slot evict] bid=%d rid=%lld slot_idx=%d buf_slot=%d hot=%d\n",
               bid, (long long)rid, slot_idx, (int)buf_slot, HOT_BUFFER_SIZE);
      }
    }
#endif

    bool is_evictable = has_valid_buf_slot && (buf_slot != newest_slot) && !s_lru_bitmap[buf_slot];
    int local_evictable_offset = 0;
    if (warp_id == 0) {
      const int base_chunk = iter * NUM_WARPS;
      const int idx = base_chunk + lane_id + 1;
      if (idx < NUM_BUFFER_CHUNKS + 1) {
        s_chunk_offset[idx] = 0;
      }
    }
    __syncthreads();

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

    if (is_evictable && has_valid_buf_slot) {
      const int evictable_offset = s_chunk_offset[chunk_idx] + local_evictable_offset;
      int num_misses = NUM_TOP_K - s_total_hits - s_newest_hit;
      if (num_misses < 0) {
        num_misses = 0;
      }
      if (evictable_offset < num_misses) {
        s_evictable_slots[evictable_offset] = buf_slot;
        s_lru_slots_out[LRU_SIZE - s_total_hits - 1 - evictable_offset] = buf_slot;
      } else {
        s_lru_slots_out[evictable_offset - num_misses] = buf_slot;
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
    s_total_evictable = total_evictable;
#if DEBUG_SPARSE_IMA
    if (atomicAdd(&s_debug_count, 1) < 8) {
      const int num_misses = NUM_TOP_K - s_total_hits;
      printf("[ima summary] bid=%d rid=%lld hits=%d num_misses=%d evictable=%d\n",
             bid, (long long)rid, s_total_hits, num_misses, s_total_evictable);
      printf("[ima evictable head] %d %d %d %d\n",
             (int)s_evictable_slots[0],
             (int)s_evictable_slots[1],
             (int)s_evictable_slots[2],
             (int)s_evictable_slots[3]);
    }
#endif
  }

  for (int i = tid; i < HOT_BUFFER_SIZE; i += BLOCK_SIZE) {
    if (i < NUM_TOP_K) {
      int32_t top_k_val = my_top_k_tokens[i];
      my_diff_map[top_k_val] = -1;
    }
    if (i < LRU_SIZE) {
      my_lru_slots[i] = s_lru_slots_out[i];
    }
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
    if (warp_id == 0) {
      const int base_chunk = iter * NUM_WARPS;
      const int idx = base_chunk + lane_id + 1;
      if (idx < NUM_TOKEN_CHUNKS + 1) {
        s_chunk_offset[idx] = 0;
      }
    }
    __syncthreads();
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

    if (tid == 0 && s_total_misses > s_total_evictable) {
#if DEBUG_SPARSE_IMA
      if (atomicAdd(&s_debug_count, 1) < 8) {
        printf("[ima miss clamp] bid=%d rid=%lld misses=%d evictable=%d\n",
               bid, (long long)rid, s_total_misses, s_total_evictable);
      }
#endif
      s_total_misses = s_total_evictable;
    }
    __syncthreads();

    if (is_miss) {
      int miss_offset = s_chunk_offset[chunk_idx] + local_miss_offset;
      if (miss_offset >= s_total_evictable) {
#if DEBUG_SPARSE_IMA
        if (atomicAdd(&s_debug_count, 1) < 8) {
          printf("[ima miss overflow] bid=%d rid=%lld miss_offset=%d evictable=%d\n",
                 bid, (long long)rid, miss_offset, s_total_evictable);
        }
#endif
        continue;
      }
      int evict_slot = s_evictable_slots[miss_offset];
      s_missed_tokens[miss_offset] = my_token;
      if (evict_slot >= 0 && evict_slot < HOT_BUFFER_SIZE) {
        my_top_k_device_locs[my_token_idx] = my_device_buffer_locs[evict_slot];
        my_device_buffer_tokens[evict_slot] = my_token;
#if DEBUG_SPARSE_IMA
        if (atomicAdd(&s_debug_count, 1) < 8) {
          for (int i = 0; i < HOT_BUFFER_SIZE; ++i) {
            if (i != evict_slot && my_device_buffer_tokens[i] == my_token) {
              printf("[ima dup token post] bid=%d rid=%lld token=%d slot=%d other=%d\n",
                     bid, (long long)rid, (int)my_token, (int)evict_slot, i);
              break;
            }
          }
        }
#endif
      } else {
        my_top_k_device_locs[my_token_idx] = -1;
#if DEBUG_SPARSE_IMA
        if (atomicAdd(&s_debug_count, 1) < 8) {
          const int num_misses = NUM_TOP_K - s_total_hits;
          printf("[ima bad evict_slot] bid=%d rid=%lld miss_offset=%d evict_slot=%d hot=%d\n",
                 bid, (long long)rid, miss_offset, evict_slot, HOT_BUFFER_SIZE);
          printf("[ima state] hits=%d misses=%d num_misses=%d total_evictable=%d\n",
                 s_total_hits, s_total_misses, num_misses, s_total_evictable);
          printf("[ima evictable head] %d %d %d %d\n",
                 (int)s_evictable_slots[0],
                 (int)s_evictable_slots[1],
                 (int)s_evictable_slots[2],
                 (int)s_evictable_slots[3]);
          printf("[ima lru head] %d %d %d %d\n",
                 (int)s_lru_slots_out[0],
                 (int)s_lru_slots_out[1],
                 (int)s_lru_slots_out[2],
                 (int)s_lru_slots_out[3]);
        }
#endif
      }
    }
    __syncthreads();
  }

  const int64_t tasks_per_block = NUM_TOP_K * page_size;
  const int64_t stride_per_block = tasks_per_block + 1;
  const int64_t block_base = bid * stride_per_block;

  for (int miss_idx = tid; miss_idx < s_total_misses; miss_idx += BLOCK_SIZE) {
    const int32_t miss_token = s_missed_tokens[miss_idx];
    const int evict_slot = s_evictable_slots[miss_idx];

    if (evict_slot >= 0 && evict_slot < HOT_BUFFER_SIZE && miss_token >= 0) {
      for (int page_offset = 0; page_offset < page_size; page_offset++) {
        const int64_t src_loc = my_host_cache_locs[miss_token * page_size + page_offset];
        const int64_t dst_loc = my_device_buffer_locs[evict_slot] * page_size + page_offset;

        const int task_idx = block_base + miss_idx * page_size + page_offset;
        transfer_tasks_src[task_idx] = src_loc;
        transfer_tasks_dst[task_idx] = dst_loc;
      }
    } else {
#if DEBUG_SPARSE_IMA
      if (atomicAdd(&s_debug_count, 1) < 8) {
        const int num_misses = NUM_TOP_K - s_total_hits;
        printf("[ima bad miss] bid=%d rid=%lld miss_idx=%d miss_token=%d evict_slot=%d\n",
               bid, (long long)rid, miss_idx, (int)miss_token, (int)evict_slot);
        printf("[ima state] hits=%d misses=%d num_misses=%d total_evictable=%d\n",
               s_total_hits, s_total_misses, num_misses, s_total_evictable);
        printf("[ima evictable head] %d %d %d %d\n",
               (int)s_evictable_slots[0],
               (int)s_evictable_slots[1],
               (int)s_evictable_slots[2],
               (int)s_evictable_slots[3]);
        printf("[ima lru head] %d %d %d %d\n",
               (int)s_lru_slots_out[0],
               (int)s_lru_slots_out[1],
               (int)s_lru_slots_out[2],
               (int)s_lru_slots_out[3]);
      }
#endif
    }
  }

  if (tid == 0) {
    const int64_t count_idx = block_base + tasks_per_block;
    transfer_tasks_src[count_idx] = s_total_misses * page_size;
  }
}

template <std::size_t kItemSize, int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE, bool IsMLA>
struct SparseCacheKernel {
  template <typename IndexT>
  static void
  run(tvm::ffi::TensorView top_k_tokens,
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
      tvm::ffi::TensorView lru_slots,
      tvm::ffi::TensorView transfer_tasks_src,
      tvm::ffi::TensorView transfer_tasks_dst,
      int64_t page_size,
      int64_t layer_id,
      int64_t item_size_bytes) {
    using namespace host;

    RuntimeCheck(kItemSize == item_size_bytes, "SparseCacheKernel: item_size_bytes mismatch");

    const int64_t bs = top_k_tokens.shape()[0];
    const int64_t host_stride = host_cache_locs.shape()[1];
    const int64_t buffer_stride_0 = device_buffer_tokens.strides()[0];
    const int64_t buffer_stride_1 = device_buffer_tokens.strides()[1];
    const int64_t page_table_stride = page_table.shape()[1];
    const int64_t diff_map_stride = diff_map.shape()[1];
    const int64_t lru_slot_stride_0 = lru_slots.strides()[0];
    const int64_t lru_slot_stride_1 = lru_slots.strides()[1];
    const int64_t top_k_tokens_stride = top_k_tokens.strides()[0];
    const int64_t top_k_device_locs_stride = top_k_device_locs.strides()[0];

    const int32_t* top_k_tokens_ptr = static_cast<const int32_t*>(top_k_tokens.data_ptr());
    int32_t* device_buffer_tokens_ptr = static_cast<int32_t*>(device_buffer_tokens.data_ptr());
    const int64_t* host_cache_locs_ptr = static_cast<const int64_t*>(host_cache_locs.data_ptr());
    const int32_t* device_buffer_locs_ptr = static_cast<const int32_t*>(device_buffer_locs.data_ptr());
    const void* host_cache_k_ptr = host_cache_k.data_ptr();
    const void* host_cache_v_ptr = (IsMLA || host_cache_v.ndim() == 0) ? nullptr : host_cache_v.data_ptr();
    void* device_buffer_k_ptr = device_buffer_k.data_ptr();
    void* device_buffer_v_ptr = (IsMLA || device_buffer_v.ndim() == 0) ? nullptr : device_buffer_v.data_ptr();
    int32_t* top_k_device_locs_ptr = static_cast<int32_t*>(top_k_device_locs.data_ptr());
    const int32_t* page_table_ptr = static_cast<const int32_t*>(page_table.data_ptr());
    int16_t* diff_map_ptr = static_cast<int16_t*>(diff_map.data_ptr());
    const IndexT* req_pool_indices_ptr = static_cast<const IndexT*>(req_pool_indices.data_ptr());
    const bool* sparse_mask_ptr = static_cast<const bool*>(sparse_mask.data_ptr());
    const IndexT* seq_lens_ptr = static_cast<const IndexT*>(seq_lens.data_ptr());
    int16_t* lru_slots_ptr = static_cast<int16_t*>(lru_slots.data_ptr());

    const auto device = LaunchKernel::resolve_device(top_k_tokens.device());

    int64_t* transfer_tasks_src_ptr = static_cast<int64_t*>(transfer_tasks_src.data_ptr());
    int64_t* transfer_tasks_dst_ptr = static_cast<int64_t*>(transfer_tasks_dst.data_ptr());
    const int64_t max_transfer_tasks = bs * NUM_TOP_K * page_size;

    // Kernel 1: Determine hits/misses and collect transfer tasks
    LaunchKernel(bs, BLOCK_SIZE, device)(
        load_cache_to_device_buffer_kernel<BLOCK_SIZE, NUM_TOP_K, HOT_BUFFER_SIZE, IsMLA, IndexT>,
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
        lru_slots_ptr,
        transfer_tasks_src_ptr,
        transfer_tasks_dst_ptr,
        buffer_stride_0,
        buffer_stride_1,
        host_stride,
        page_table_stride,
        diff_map_stride,
        lru_slot_stride_0,
        lru_slot_stride_1,
        top_k_tokens_stride,
        top_k_device_locs_stride,
        page_size,
        layer_id,
        item_size_bytes);

    // Kernel 2: parallel transfer
    constexpr int TRANSFER_BLOCK_SIZE = 256;
    constexpr int WARPS_PER_BLOCK = TRANSFER_BLOCK_SIZE / WARP_SIZE;
    const int64_t num_transfer_blocks = (max_transfer_tasks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    LaunchKernel(num_transfer_blocks, TRANSFER_BLOCK_SIZE, device)(
        parallel_transfer_kernel<kItemSize, TRANSFER_BLOCK_SIZE, NUM_TOP_K>,
        transfer_tasks_src_ptr,
        transfer_tasks_dst_ptr,
        host_cache_k_ptr,
        host_cache_v_ptr,
        device_buffer_k_ptr,
        device_buffer_v_ptr,
        max_transfer_tasks,
        page_size,
        IsMLA);
  }
};

template <std::size_t kItemSize, int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE, bool IsMLA>
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
    tvm::ffi::TensorView lru_slots,
    tvm::ffi::TensorView transfer_tasks_src,
    tvm::ffi::TensorView transfer_tasks_dst,
    int64_t page_size,
    int64_t layer_id,
    int64_t item_size_bytes) {
  const auto& dtype = req_pool_indices.dtype();
  const bool is_int64 = (dtype.bits == 64);

  if (is_int64) {
    SparseCacheKernel<kItemSize, BLOCK_SIZE, NUM_TOP_K, HOT_BUFFER_SIZE, IsMLA>::template run<int64_t>(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache_k,
        host_cache_v,
        device_buffer_k,
        device_buffer_v,
        top_k_device_locs,
        page_table,
        diff_map,
        req_pool_indices,
        sparse_mask,
        seq_lens,
        lru_slots,
        transfer_tasks_src,
        transfer_tasks_dst,
        page_size,
        layer_id,
        item_size_bytes);
  } else {
    SparseCacheKernel<kItemSize, BLOCK_SIZE, NUM_TOP_K, HOT_BUFFER_SIZE, IsMLA>::template run<int32_t>(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache_k,
        host_cache_v,
        device_buffer_k,
        device_buffer_v,
        top_k_device_locs,
        page_table,
        diff_map,
        req_pool_indices,
        sparse_mask,
        seq_lens,
        lru_slots,
        transfer_tasks_src,
        transfer_tasks_dst,
        page_size,
        layer_id,
        item_size_bytes);
  }
}

}  // namespace
