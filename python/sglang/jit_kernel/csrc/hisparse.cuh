#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/deepseek_v4/kvcacheio.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <stdexcept>
#include <stdint.h>
#include <string>

namespace {

#ifdef USE_ROCM
constexpr int WARP_SIZE = 64;
using BallotMask = uint64_t;
constexpr BallotMask FULL_WARP_MASK = 0xFFFFFFFFFFFFFFFFull;
#else
constexpr int WARP_SIZE = 32;
using BallotMask = unsigned int;
constexpr BallotMask FULL_WARP_MASK = 0xFFFFFFFFu;
#endif
constexpr int32_t TOKEN_HIT = 0xFFFFFFFF;
constexpr int32_t HASH_EMPTY = -1;

// Knuth multiplicative hash for open-addressing table of size hash_size.
__device__ __forceinline__ int hash_slot(int32_t key, int hash_size) {
  return ((uint32_t)key * 2654435761u) % (uint32_t)hash_size;
}

#ifdef USE_ROCM
__device__ __forceinline__ void transfer_item_warp(
    int32_t lane_id, const void* __restrict__ src_addr, void* __restrict__ dst_addr, int64_t item_size_bytes) {
  const auto src = static_cast<const char*>(src_addr);
  auto dst = static_cast<char*>(dst_addr);

  const int64_t word_count = item_size_bytes / static_cast<int64_t>(sizeof(uint64_t));
  const auto src_words = reinterpret_cast<const uint64_t*>(src);
  auto dst_words = reinterpret_cast<uint64_t*>(dst);
  for (int64_t i = lane_id; i < word_count; i += WARP_SIZE) {
    dst_words[i] = src_words[i];
  }

  const int64_t tail_start = word_count * static_cast<int64_t>(sizeof(uint64_t));
  for (int64_t i = tail_start + lane_id; i < item_size_bytes; i += WARP_SIZE) {
    dst[i] = src[i];
  }
}
#else
__device__ __forceinline__ void
transfer_item_warp(int32_t lane_id, const void* src_addr, void* dst_addr, int64_t item_size_bytes) {
  // 128-bit bulk transfer via paired 64-bit loads (avoids alignment issues with uint4)
  const int total_pairs = item_size_bytes / 16;  // number of 16-byte chunks
  {
    const uint64_t* __restrict__ src = static_cast<const uint64_t*>(src_addr);
    uint64_t* __restrict__ dst = static_cast<uint64_t*>(dst_addr);
    for (int j = lane_id; j < total_pairs; j += WARP_SIZE) {
      uint64_t lo, hi;
      const uint64_t* s = src + j * 2;
      asm volatile("ld.global.nc.v2.b64 {%0,%1},[%2];" : "=l"(lo), "=l"(hi) : "l"(s) : "memory");
      uint64_t* d = dst + j * 2;
      asm volatile("st.global.cg.v2.b64 [%0],{%1,%2};" ::"l"(d), "l"(lo), "l"(hi) : "memory");
    }
  }

  // Tail: 64-bit for remaining 8-byte chunk (if item_size not multiple of 16)
  const int tail_8B = (item_size_bytes - total_pairs * 16) / 8;
  if (tail_8B > 0 && lane_id < tail_8B) {
    const uint64_t* __restrict__ src8 =
        reinterpret_cast<const uint64_t*>(static_cast<const char*>(src_addr) + total_pairs * 16);
    uint64_t* __restrict__ dst8 = reinterpret_cast<uint64_t*>(static_cast<char*>(dst_addr) + total_pairs * 16);
    uint64_t tmp;
    asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src8 + lane_id) : "memory");
    asm volatile("st.global.cg.b64 [%0],%1;" ::"l"(dst8 + lane_id), "l"(tmp) : "memory");
  }
}
#endif

__device__ __forceinline__ int popc_mask(BallotMask mask) {
#ifdef USE_ROCM
  return __popcll(mask);
#else
  return __popc(mask);
#endif
}

template <int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 1) void transfer_cache_dsv4_mla_kernel(
    void** src_caches,
    void** dst_caches,
    const int64_t* src_indices,
    const int64_t* dst_indices,
    uint32_t num_items,
    uint32_t num_layers) {
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  const int total_warps = gridDim.x * NUM_WARPS;

  for (uint32_t i = global_tid / WARP_SIZE; i < num_items; i += total_warps) {
    const int32_t src_index = static_cast<int32_t>(src_indices[i]);
    const int32_t dst_index = static_cast<int32_t>(dst_indices[i]);
    for (uint32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
      device::hisparse::transfer_item(
          /*dst_cache=*/dst_caches[layer_id],
          /*src_cache=*/src_caches[layer_id],
          /*dst_index=*/dst_index,
          /*src_index=*/src_index);
    }
  }
}

template <int BLOCK_SIZE>
void transfer_cache_dsv4_mla(
    tvm::ffi::TensorView src_ptrs,
    tvm::ffi::TensorView dst_ptrs,
    tvm::ffi::TensorView src_indices,
    tvm::ffi::TensorView dst_indices) {
  using namespace host;
  auto N = SymbolicSize{"num_items"};
  auto L = SymbolicSize{"num_layers"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({L}).with_dtype<uint64_t>().with_device(device).verify(src_ptrs).verify(dst_ptrs);
  TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(src_indices).verify(dst_indices);

  const auto num_items = static_cast<uint32_t>(N.unwrap());
  if (num_items == 0) {
    return;
  }
  const auto num_layers = static_cast<uint32_t>(L.unwrap());
  const int num_warps = BLOCK_SIZE / WARP_SIZE;
  const int grid = (num_items + num_warps - 1) / num_warps;
  LaunchKernel(grid, BLOCK_SIZE, device.unwrap())(
      transfer_cache_dsv4_mla_kernel<BLOCK_SIZE>,
      static_cast<void**>(src_ptrs.data_ptr()),
      static_cast<void**>(dst_ptrs.data_ptr()),
      static_cast<const int64_t*>(src_indices.data_ptr()),
      static_cast<const int64_t*>(dst_indices.data_ptr()),
      num_items,
      num_layers);
}

__device__ __forceinline__ int warp_inclusive_scan(int* s_data, int lane_id, int offset, int count, int accumulator) {
  int idx = lane_id + offset;
  int val = (idx < count) ? s_data[idx] : 0;

#pragma unroll
  for (int i = 1; i < WARP_SIZE; i *= 2) {
    int n = __shfl_up_sync(FULL_WARP_MASK, val, i);
    if (lane_id >= i) val += n;
  }
  val += accumulator;
  if (idx < count) {
    s_data[idx] = val;
  }
  accumulator = __shfl_sync(FULL_WARP_MASK, val, WARP_SIZE - 1);
  return accumulator;
}

// Shared memory size calculation for dynamic allocation.
// Layout: int32_t region (4-byte aligned) followed by int16_t region (2-byte aligned).
template <int NUM_TOP_K, int HOT_BUFFER_SIZE>
struct SmemLayout {
  static constexpr int HASH_SIZE = NUM_TOP_K * 2;
  static constexpr int NUM_BUFFER_CHUNKS = (HOT_BUFFER_SIZE + WARP_SIZE - 1) / WARP_SIZE;
  // int32_t region: top_k_tokens + chunk_offset + evict_chunk_offset + hash_keys + total_hits + newest_hit
  static constexpr int TOTAL_INT32 = NUM_TOP_K + (NUM_BUFFER_CHUNKS + 1) + (NUM_BUFFER_CHUNKS + 1) + HASH_SIZE + 2;
  // int16_t region: lru_slots_out + hash_vals
  static constexpr int TOTAL_INT16 = HOT_BUFFER_SIZE + HASH_SIZE;
  static constexpr size_t BYTES = TOTAL_INT32 * sizeof(int32_t) + TOTAL_INT16 * sizeof(int16_t);
};

// Each block processes one request
// req_pool_indices and seq_lens can each be int32_t or int64_t
// Layout: [HOT_BUFFER_SIZE slots for LRU] + [page_size slots for newest token]
// newest_slot is at HOT_BUFFER_SIZE (first position of extra page)
//
// IsDsv4Layout selects the miss-copy addressing:
//   false -> generic byte-stride: device + host both linear, stride = item_size_bytes
//   true  -> DSv4 page-padded device + page-padded host (kvcacheio.cuh constants)
template <
    int BLOCK_SIZE,
    int NUM_TOP_K,
    int HOT_BUFFER_SIZE,
    bool IsMLA,
    bool IsDsv4Layout,
    typename SeqLensT,
    typename ReqPoolIndicesT>
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
    const ReqPoolIndicesT* __restrict__ req_pool_indices,
    const SeqLensT* __restrict__ seq_lens,
    int16_t* __restrict__ lru_slots,
    const int32_t* __restrict__ num_real_reqs,
    int64_t buffer_stride_0,
    int64_t host_stride,
    int64_t lru_slot_stride_0,
    int64_t top_k_tokens_stride,
    int64_t top_k_device_locs_stride,
    int64_t page_size,
    int64_t item_size_bytes,
    int64_t* __restrict__ miss_src_out,
    int64_t* __restrict__ miss_dst_out,
    int32_t* __restrict__ miss_count_out,
    int64_t plan_stride) {
  // When miss_*_out != nullptr, additionally record this step's miss plan:
  //   miss_src_out[bid, k] / miss_dst_out[bid, k] = (host_loc, device_loc) of the
  //   k-th miss, miss_count_out[bid] = number of misses. A shared-index skip layer
  //   later replays exactly these copies into its own buffers (copy_cache_planned)
  //   instead of re-running the full plan+IO kernel.
  static_assert(!IsDsv4Layout || IsMLA, "DSv4 page-padded layout is K-only (MLA).");
  // todo hisparse: support page wise sparsity
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  constexpr int NUM_TOKEN_CHUNKS = (NUM_TOP_K + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int NUM_BUFFER_CHUNKS = (HOT_BUFFER_SIZE + WARP_SIZE - 1) / WARP_SIZE;

  const int bid = blockIdx.x;
  // Early exit for padded blocks (CUDA graph pads batch to a captured size)
  if (bid >= num_real_reqs[0]) return;

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const BallotMask lanes_before = (BallotMask(1) << lane_id) - BallotMask(1);

  const int64_t rid = req_pool_indices[bid];
  const int64_t seq_len = seq_lens[bid];

  // Calculate offsets for this request
  const int32_t* req_top_k_tokens = top_k_tokens + bid * top_k_tokens_stride;
  int32_t* req_top_k_device_locs = top_k_device_locs + bid * top_k_device_locs_stride;

  const int64_t buffer_offset = rid * buffer_stride_0;
  int32_t* req_device_buffer_tokens = device_buffer_tokens + buffer_offset;
  const int32_t* req_device_buffer_locs = device_buffer_locs + buffer_offset;
  const int64_t* req_host_cache_locs = host_cache_locs + rid * host_stride;
  int16_t* req_lru_slots = lru_slots + rid * lru_slot_stride_0;

  // Fast path: short sequences have all tokens in the device buffer in order.
  if (seq_len <= HOT_BUFFER_SIZE) {
    const int count = (seq_len < NUM_TOP_K) ? static_cast<int>(seq_len) : NUM_TOP_K;
    for (int i = tid; i < count; i += BLOCK_SIZE) {
      int32_t token_pos = req_top_k_tokens[i];
      if (token_pos >= 0) {
        req_top_k_device_locs[i] = req_device_buffer_locs[token_pos];
      }
    }
    // Short sequences load nothing from host: an empty miss plan for this request.
    if (miss_count_out != nullptr && tid == 0) {
      miss_count_out[bid] = 0;
    }
    return;
  }

  // Dynamic shared memory layout: int32_t arrays first, then int16_t arrays.
  extern __shared__ char smem_raw[];
  using Layout = SmemLayout<NUM_TOP_K, HOT_BUFFER_SIZE>;
  constexpr int HASH_SIZE = Layout::HASH_SIZE;

  int32_t* smem_i32 = reinterpret_cast<int32_t*>(smem_raw);
  // Top-k token positions; reused as miss-token scratch in the copy phase
  int32_t* s_top_k_tokens = smem_i32;
  // Prefix-sum offsets for hit counting and miss counting
  int32_t* s_chunk_offset = s_top_k_tokens + NUM_TOP_K;
  // Prefix-sum offsets for evictable counting
  int32_t* s_evict_chunk_offset = s_chunk_offset + (NUM_BUFFER_CHUNKS + 1);
  // Open-addressing hash table: top-k token_id -> top-k index (keys)
  int32_t* s_hash_keys = s_evict_chunk_offset + (NUM_BUFFER_CHUNKS + 1);
  // Scalar counters
  int32_t& s_total_hits = s_hash_keys[HASH_SIZE];
  int32_t& s_newest_hit = s_hash_keys[HASH_SIZE + 1];

  int16_t* smem_i16 = reinterpret_cast<int16_t*>(smem_i32 + Layout::TOTAL_INT32);
  // Compacted slot ordering: [hits fwd->  ...  <-evictables bwd]
  int16_t* s_lru_slots_out = smem_i16;
  // Open-addressing hash table: top-k token_id -> top-k index (values)
  int16_t* s_hash_vals = s_lru_slots_out + HOT_BUFFER_SIZE;

  // Initialize shared memory: counters, hash table, prefix-sum offsets.
  if (tid == 0) {
    s_total_hits = 0;
    s_newest_hit = 0;
  }
  for (int i = tid; i < HASH_SIZE; i += BLOCK_SIZE) {
    s_hash_keys[i] = HASH_EMPTY;
  }
  for (int i = tid; i < NUM_BUFFER_CHUNKS + 1; i += BLOCK_SIZE) {
    s_chunk_offset[i] = 0;
    s_evict_chunk_offset[i] = 0;
  }
  __syncthreads();

  const int newest_slot = HOT_BUFFER_SIZE;
  const int32_t newest_token = seq_len - 1;

  // Insert top-k tokens into shared-memory hash table.
  for (int i = tid; i < NUM_TOP_K; i += BLOCK_SIZE) {
    int32_t token_idx = req_top_k_tokens[i];
    if (token_idx == newest_token) {
      // If topk includes the latest token, bind its canonical occurrence to newest_slot (at HOT_BUFFER_SIZE) and mark
      // it as a hit. newest_slot is at the first position of the extra page, excluded from LRU tracking.
      s_top_k_tokens[i] = TOKEN_HIT;
      req_top_k_device_locs[i] = req_device_buffer_locs[newest_slot];
      s_newest_hit = 1;
    } else {
      int slot = hash_slot(token_idx, HASH_SIZE);
      while (true) {
        int32_t old = atomicCAS(&s_hash_keys[slot], HASH_EMPTY, token_idx);
        if (old == HASH_EMPTY || old == token_idx) {
          s_hash_vals[slot] = static_cast<int16_t>(i);
          break;
        }
        slot = (slot + 1) % HASH_SIZE;
      }
      s_top_k_tokens[i] = token_idx;
    }
  }
  __syncthreads();

  constexpr int ITERATIONS_PER_WARP_BUFFER = (NUM_BUFFER_CHUNKS + NUM_WARPS - 1) / NUM_WARPS;
  int total_hit_count = 0;
  int total_evict_count = 0;
  for (int iter = 0; iter < ITERATIONS_PER_WARP_BUFFER; iter++) {
    int chunk_idx = warp_id + iter * NUM_WARPS;
    bool has_valid_chunk = chunk_idx < NUM_BUFFER_CHUNKS;

    const int slot_idx = chunk_idx * WARP_SIZE + lane_id;
    const bool has_valid_slot = has_valid_chunk && (slot_idx < HOT_BUFFER_SIZE);
    const int16_t buf_slot = has_valid_slot ? req_lru_slots[slot_idx] : -1;
    int32_t my_buffer_token = (buf_slot >= 0) ? req_device_buffer_tokens[buf_slot] : -1;
    int my_found_top_k_idx = -1;
    if (my_buffer_token >= 0) {
      int h = hash_slot(my_buffer_token, HASH_SIZE);
      while (true) {
        int32_t k = s_hash_keys[h];
        if (k == my_buffer_token) {
          my_found_top_k_idx = static_cast<int32_t>(s_hash_vals[h]);
          break;
        }
        if (k == HASH_EMPTY) break;
        h = (h + 1) % HASH_SIZE;
      }
    }
    bool is_hit = my_found_top_k_idx >= 0;
    bool is_evictable = has_valid_slot && !is_hit;

    // Record hits
    if (is_hit) {
      s_top_k_tokens[my_found_top_k_idx] = TOKEN_HIT;
      req_top_k_device_locs[my_found_top_k_idx] = req_device_buffer_locs[buf_slot];
    }

    int local_hit_offset = 0;
    int local_evict_offset = 0;
    if (has_valid_chunk) {
      const BallotMask hit_mask = __ballot_sync(FULL_WARP_MASK, is_hit);
      const BallotMask evict_mask = __ballot_sync(FULL_WARP_MASK, is_evictable);
      local_hit_offset = popc_mask(hit_mask & lanes_before);
      local_evict_offset = popc_mask(evict_mask & lanes_before);
      if (lane_id == 0) {
        s_chunk_offset[chunk_idx + 1] = popc_mask(hit_mask);
        s_evict_chunk_offset[chunk_idx + 1] = popc_mask(evict_mask);
      }
    }
    __syncthreads();

    if (warp_id == 0) {
#ifdef USE_ROCM
      // ROCm wavefront64: WARP_SIZE (64) > NUM_WARPS (16 at block_size=1024),
      // so the wide-count form below would let lanes beyond this iteration's
      // NUM_WARPS-wide window write the accumulator into s_chunk_offset
      // positions belonging to future iterations, corrupting their reads.
      // Bound the scan window to NUM_WARPS lanes.
      const int scan_offset = iter * NUM_WARPS + 1;
      const int scan_count = min(scan_offset + NUM_WARPS, NUM_BUFFER_CHUNKS + 1);
      total_hit_count = warp_inclusive_scan(s_chunk_offset, lane_id, scan_offset, scan_count, total_hit_count);
      total_evict_count =
          warp_inclusive_scan(s_evict_chunk_offset, lane_id, scan_offset, scan_count, total_evict_count);
#else
      total_hit_count =
          warp_inclusive_scan(s_chunk_offset, lane_id, chunk_idx + 1, NUM_BUFFER_CHUNKS + 1, total_hit_count);
      total_evict_count =
          warp_inclusive_scan(s_evict_chunk_offset, lane_id, chunk_idx + 1, NUM_BUFFER_CHUNKS + 1, total_evict_count);
#endif
      if (tid == 0) {
        s_total_hits = total_hit_count;
      }
    }
    __syncthreads();

    // Hits grow forward from index 0
    if (is_hit) {
      int hit_offset = s_chunk_offset[chunk_idx] + local_hit_offset;
      s_lru_slots_out[hit_offset] = buf_slot;
    }
    // Evictables grow backward from HOT_BUFFER_SIZE - 1
    if (is_evictable) {
      int evict_offset = s_evict_chunk_offset[chunk_idx] + local_evict_offset;
      s_lru_slots_out[HOT_BUFFER_SIZE - 1 - evict_offset] = buf_slot;
    }
  }
  __syncthreads();

  // Reset offsets for the miss counting phase (only NUM_TOKEN_CHUNKS + 1 entries needed).
  for (int i = tid; i < NUM_TOKEN_CHUNKS + 1; i += BLOCK_SIZE) {
    s_chunk_offset[i] = 0;
  }
  __syncthreads();

  // Third pass to identify misses and their evictable slots
  int total_misses = 0;
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

    if (has_valid_chunk) {
      const BallotMask miss_mask = __ballot_sync(FULL_WARP_MASK, is_miss);
      local_miss_offset = popc_mask(miss_mask & lanes_before);
      const int warp_miss_count = popc_mask(miss_mask);
      if (lane_id == 0) {
        s_chunk_offset[chunk_idx + 1] = warp_miss_count;
      }
    }
    __syncthreads();

    if (warp_id == 0) {
#ifdef USE_ROCM
      const int scan_offset = iter * NUM_WARPS + 1;
      const int scan_count = min(scan_offset + NUM_WARPS, NUM_TOKEN_CHUNKS + 1);
      total_misses = warp_inclusive_scan(s_chunk_offset, lane_id, scan_offset, scan_count, total_misses);
#else
      total_misses = warp_inclusive_scan(s_chunk_offset, lane_id, chunk_idx + 1, NUM_TOKEN_CHUNKS + 1, total_misses);
#endif
    }
    __syncthreads();

    if (is_miss) {
      int miss_offset = s_chunk_offset[chunk_idx] + local_miss_offset;
      int16_t evict_slot = s_lru_slots_out[HOT_BUFFER_SIZE - 1 - miss_offset];
      // Reuse s_top_k_tokens as miss scratch: miss_offset < my_token_idx always
      // holds (hits are skipped), so compacted writes never overrun pending reads.
      s_top_k_tokens[miss_offset] = my_token;
      req_top_k_device_locs[my_token_idx] = req_device_buffer_locs[evict_slot];
      req_device_buffer_tokens[evict_slot] = my_token;
    }
  }
  __syncthreads();

  total_misses = NUM_TOP_K - s_total_hits - s_newest_hit;
  if (miss_count_out != nullptr && tid == 0) {
    miss_count_out[bid] = total_misses;
  }
  // Write back LRU order: evictables at front (LRU), hits at back (MRU).
  {
    const int total_evictable = HOT_BUFFER_SIZE - s_total_hits;
#ifdef USE_ROCM
    // ROCm: cap writeback threads at 512 for large kernels.
    constexpr int LRU_WRITEBACK_THREADS = (BLOCK_SIZE > 512) ? 512 : BLOCK_SIZE;
    if (tid < LRU_WRITEBACK_THREADS) {
      for (int i = tid; i < HOT_BUFFER_SIZE; i += LRU_WRITEBACK_THREADS) {
        if (i < total_misses) {
          // Misses: just loaded from host, place right before hits
          req_lru_slots[total_evictable - total_misses + i] = s_lru_slots_out[HOT_BUFFER_SIZE - 1 - i];
        } else if (i < total_evictable) {
          // Remaining evictables: truly stale, dest at LRU front
          req_lru_slots[i - total_misses] = s_lru_slots_out[HOT_BUFFER_SIZE - 1 - i];
        } else {
          // Hits: source at forward end, dest at MRU back
          req_lru_slots[i] = s_lru_slots_out[i - total_evictable];
        }
      }
    }
#else
    for (int i = tid; i < HOT_BUFFER_SIZE; i += BLOCK_SIZE) {
      if (i < total_misses) {
        // Misses: just loaded from host, place right before hits
        req_lru_slots[total_evictable - total_misses + i] = s_lru_slots_out[HOT_BUFFER_SIZE - 1 - i];
      } else if (i < total_evictable) {
        // Remaining evictables: truly stale, dest at LRU front
        req_lru_slots[i - total_misses] = s_lru_slots_out[HOT_BUFFER_SIZE - 1 - i];
      } else {
        // Hits: source at forward end, dest at MRU back
        req_lru_slots[i] = s_lru_slots_out[i - total_evictable];
      }
    }
#endif
  }

  // each warp copies one miss directly, can be separated into a new kernel if parallelism is a concern
  for (int miss_idx = warp_id; miss_idx < total_misses; miss_idx += NUM_WARPS) {
    const int32_t miss_token = s_top_k_tokens[miss_idx];
    const int16_t evict_slot = s_lru_slots_out[HOT_BUFFER_SIZE - 1 - miss_idx];

    const int64_t src_loc = req_host_cache_locs[miss_token];
    const int64_t dst_loc = static_cast<int64_t>(req_device_buffer_locs[evict_slot]);

    // Record the (host, device) locations so shared-index skip layers can replay
    // this exact copy without re-planning (device/host locs are layer-independent).
    if (miss_src_out != nullptr && lane_id == 0) {
      miss_src_out[bid * plan_stride + miss_idx] = src_loc;
      miss_dst_out[bid * plan_stride + miss_idx] = dst_loc;
    }

    if constexpr (IsDsv4Layout) {
#ifdef USE_ROCM
      // ROCm path: host cache and device buffer both use the page-padded C4
      // layout (same as the write path and the CUDA branch). We can't reuse
      // device::hisparse::transfer_item here because its warp logic is hardcoded
      // to a 32-lane warp; on wavefront64 we use the warp-width-agnostic
      // transfer_item_warp with paged source and destination addressing.
      using namespace device::hisparse;
      const auto [dst_value_ptr, dst_scale_ptr] = get_pointer_paged(device_buffer_k, static_cast<int32_t>(dst_loc));
      const auto [src_value_ptr, src_scale_ptr] =
          get_pointer_paged(const_cast<void*>(host_cache_k), static_cast<int32_t>(src_loc));
      transfer_item_warp(lane_id, src_value_ptr, dst_value_ptr, kValueBytes);
      transfer_item_warp(lane_id, src_scale_ptr, dst_scale_ptr, kScaleBytes);
#else
      // CUDA path: page-padded device layout + page-padded host layout, K-only.
      // The host cache is pinned DRAM but uses the same row layout as the GPU C4
      // cache, so use the page-padded address calculation for both ends.
      device::hisparse::transfer_item(
          /*dst_cache=*/device_buffer_k,
          /*src_cache=*/const_cast<void*>(host_cache_k),
          /*dst_index=*/static_cast<int32_t>(dst_loc),
          /*src_index=*/static_cast<int32_t>(src_loc));
#endif
    } else {
      // Generic path: device + host both linear, stride = item_size_bytes.
      const auto src_k = static_cast<const char*>(host_cache_k) + src_loc * item_size_bytes;
      auto dst_k = static_cast<char*>(device_buffer_k) + dst_loc * item_size_bytes;
      transfer_item_warp(lane_id, src_k, dst_k, item_size_bytes);

      if constexpr (!IsMLA) {
        const auto src_v = static_cast<const char*>(host_cache_v) + src_loc * item_size_bytes;
        auto dst_v = static_cast<char*>(device_buffer_v) + dst_loc * item_size_bytes;
        transfer_item_warp(lane_id, src_v, dst_v, item_size_bytes);
      }
    }
  }
}

template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE, bool IsMLA, bool IsDsv4Layout>
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
    tvm::ffi::TensorView req_pool_indices,
    tvm::ffi::TensorView seq_lens,
    tvm::ffi::TensorView lru_slots,
    tvm::ffi::TensorView num_real_reqs,
    int64_t page_size,
    int64_t item_size_bytes,
    tvm::ffi::TensorView miss_src_out,
    tvm::ffi::TensorView miss_dst_out,
    tvm::ffi::TensorView miss_count_out,
    int64_t plan_stride) {
  using namespace host;

  const int64_t bs = top_k_tokens.shape()[0];
  // Optional miss-plan outputs: 0-dim tensors mean "don't record".
  int64_t* const miss_src_ptr = (miss_src_out.ndim() == 0) ? nullptr : static_cast<int64_t*>(miss_src_out.data_ptr());
  int64_t* const miss_dst_ptr = (miss_dst_out.ndim() == 0) ? nullptr : static_cast<int64_t*>(miss_dst_out.data_ptr());
  int32_t* const miss_count_ptr =
      (miss_count_out.ndim() == 0) ? nullptr : static_cast<int32_t*>(miss_count_out.data_ptr());
  const int64_t host_stride = host_cache_locs.shape()[1];
  const int64_t buffer_stride_0 = device_buffer_tokens.strides()[0];
  const int64_t lru_slot_stride_0 = lru_slots.strides()[0];
  const int64_t top_k_tokens_stride = top_k_tokens.strides()[0];
  const int64_t top_k_device_locs_stride = top_k_device_locs.strides()[0];
  const auto device = LaunchKernel::resolve_device(top_k_tokens.device());

  // Generic lambda: int32/int64 kernel variants are compiled for both
  // seq_lens and req_pool_indices; the correct combo is selected at runtime.
  auto launch = [&](auto kernel_fn, const auto* seq_lens_ptr, const auto* req_pool_indices_ptr) {
    constexpr size_t smem_bytes = SmemLayout<NUM_TOP_K, HOT_BUFFER_SIZE>::BYTES;
#ifndef USE_ROCM
    if constexpr (smem_bytes > 48u * 1024u) {
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
#endif
    LaunchKernel(bs, BLOCK_SIZE, device, smem_bytes)(
        kernel_fn,
        static_cast<const int32_t*>(top_k_tokens.data_ptr()),
        static_cast<int32_t*>(device_buffer_tokens.data_ptr()),
        static_cast<const int64_t*>(host_cache_locs.data_ptr()),
        static_cast<const int32_t*>(device_buffer_locs.data_ptr()),
        host_cache_k.data_ptr(),
        (IsMLA || host_cache_v.ndim() == 0) ? (const void*)nullptr : host_cache_v.data_ptr(),
        device_buffer_k.data_ptr(),
        (IsMLA || device_buffer_v.ndim() == 0) ? (void*)nullptr : device_buffer_v.data_ptr(),
        static_cast<int32_t*>(top_k_device_locs.data_ptr()),
        req_pool_indices_ptr,
        seq_lens_ptr,
        static_cast<int16_t*>(lru_slots.data_ptr()),
        static_cast<const int32_t*>(num_real_reqs.data_ptr()),
        buffer_stride_0,
        host_stride,
        lru_slot_stride_0,
        top_k_tokens_stride,
        top_k_device_locs_stride,
        page_size,
        item_size_bytes,
        miss_src_ptr,
        miss_dst_ptr,
        miss_count_ptr,
        plan_stride);
  };

  const auto seq_dtype = seq_lens.dtype();
  const auto rpi_dtype = req_pool_indices.dtype();
  const bool seq_is_i64 = (seq_dtype.code == kDLInt && seq_dtype.bits == 64);
  const bool rpi_is_i64 = (rpi_dtype.code == kDLInt && rpi_dtype.bits == 64);

  if (seq_is_i64 && rpi_is_i64) {
    launch(
        load_cache_to_device_buffer_kernel<
            BLOCK_SIZE,
            NUM_TOP_K,
            HOT_BUFFER_SIZE,
            IsMLA,
            IsDsv4Layout,
            int64_t,
            int64_t>,
        static_cast<const int64_t*>(seq_lens.data_ptr()),
        static_cast<const int64_t*>(req_pool_indices.data_ptr()));
  } else if (seq_is_i64 && !rpi_is_i64) {
    launch(
        load_cache_to_device_buffer_kernel<
            BLOCK_SIZE,
            NUM_TOP_K,
            HOT_BUFFER_SIZE,
            IsMLA,
            IsDsv4Layout,
            int64_t,
            int32_t>,
        static_cast<const int64_t*>(seq_lens.data_ptr()),
        static_cast<const int32_t*>(req_pool_indices.data_ptr()));
  } else if (!seq_is_i64 && rpi_is_i64) {
    launch(
        load_cache_to_device_buffer_kernel<
            BLOCK_SIZE,
            NUM_TOP_K,
            HOT_BUFFER_SIZE,
            IsMLA,
            IsDsv4Layout,
            int32_t,
            int64_t>,
        static_cast<const int32_t*>(seq_lens.data_ptr()),
        static_cast<const int64_t*>(req_pool_indices.data_ptr()));
  } else {
    launch(
        load_cache_to_device_buffer_kernel<
            BLOCK_SIZE,
            NUM_TOP_K,
            HOT_BUFFER_SIZE,
            IsMLA,
            IsDsv4Layout,
            int32_t,
            int32_t>,
        static_cast<const int32_t*>(seq_lens.data_ptr()),
        static_cast<const int32_t*>(req_pool_indices.data_ptr()));
  }
}

// Copy-only ("IO-only") swap-in for shared-index skip layers.
//
// Replays a miss plan recorded by the anchor's load_cache_to_device_buffer:
// for each real request r, copy its miss_counts[r] entries host->device using the
// anchor's (miss_src_locs, miss_dst_locs). No hit detection / LRU / hashing -- the
// slot layout is shared with the anchor (lockstep), so the anchor's slot table is
// reused verbatim. Launched with a small fixed grid (NUM_BLOCKS) so the copies keep
// a low SM/copy-engine footprint while overlapping compute on a side stream.
template <int BLOCK_SIZE, bool IsMLA, bool IsDsv4Layout>
__global__ __launch_bounds__(BLOCK_SIZE, 1) void copy_cache_planned_kernel(
    const int64_t* __restrict__ miss_src_locs,
    const int64_t* __restrict__ miss_dst_locs,
    const int32_t* __restrict__ miss_counts,
    const int32_t* __restrict__ num_real_reqs,
    const void* __restrict__ host_cache_k,
    const void* __restrict__ host_cache_v,
    void* __restrict__ device_buffer_k,
    void* __restrict__ device_buffer_v,
    int64_t plan_stride,
    int64_t item_size_bytes) {
  static_assert(!IsDsv4Layout || IsMLA, "DSv4 page-padded layout is K-only (MLA).");
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int warp_global = blockIdx.x * NUM_WARPS + threadIdx.x / WARP_SIZE;
  const int total_warps = gridDim.x * NUM_WARPS;
  const int real = num_real_reqs[0];

  for (int r = 0; r < real; ++r) {
    const int cnt = miss_counts[r];
    const int64_t* src_row = miss_src_locs + static_cast<int64_t>(r) * plan_stride;
    const int64_t* dst_row = miss_dst_locs + static_cast<int64_t>(r) * plan_stride;
    for (int m = warp_global; m < cnt; m += total_warps) {
      const int64_t src_loc = src_row[m];
      const int64_t dst_loc = dst_row[m];
      if constexpr (IsDsv4Layout) {
#ifdef USE_ROCM
        using namespace device::hisparse;
        const auto [dst_value_ptr, dst_scale_ptr] = get_pointer_paged(device_buffer_k, static_cast<int32_t>(dst_loc));
        const auto [src_value_ptr, src_scale_ptr] =
            get_pointer_paged(const_cast<void*>(host_cache_k), static_cast<int32_t>(src_loc));
        transfer_item_warp(lane_id, src_value_ptr, dst_value_ptr, kValueBytes);
        transfer_item_warp(lane_id, src_scale_ptr, dst_scale_ptr, kScaleBytes);
#else
        device::hisparse::transfer_item(
            /*dst_cache=*/device_buffer_k,
            /*src_cache=*/const_cast<void*>(host_cache_k),
            /*dst_index=*/static_cast<int32_t>(dst_loc),
            /*src_index=*/static_cast<int32_t>(src_loc));
#endif
      } else {
        const auto src_k = static_cast<const char*>(host_cache_k) + src_loc * item_size_bytes;
        auto dst_k = static_cast<char*>(device_buffer_k) + dst_loc * item_size_bytes;
        transfer_item_warp(lane_id, src_k, dst_k, item_size_bytes);
        if constexpr (!IsMLA) {
          const auto src_v = static_cast<const char*>(host_cache_v) + src_loc * item_size_bytes;
          auto dst_v = static_cast<char*>(device_buffer_v) + dst_loc * item_size_bytes;
          transfer_item_warp(lane_id, src_v, dst_v, item_size_bytes);
        }
      }
    }
  }
}

template <int BLOCK_SIZE, bool IsMLA, bool IsDsv4Layout>
void copy_cache_planned(
    tvm::ffi::TensorView miss_src_locs,
    tvm::ffi::TensorView miss_dst_locs,
    tvm::ffi::TensorView miss_counts,
    tvm::ffi::TensorView num_real_reqs,
    tvm::ffi::TensorView host_cache_k,
    tvm::ffi::TensorView host_cache_v,
    tvm::ffi::TensorView device_buffer_k,
    tvm::ffi::TensorView device_buffer_v,
    int64_t num_blocks,
    int64_t item_size_bytes) {
  using namespace host;
  const int64_t plan_stride = miss_src_locs.strides()[0];
  const auto device = LaunchKernel::resolve_device(miss_src_locs.device());
  LaunchKernel(num_blocks, BLOCK_SIZE, device)(
      copy_cache_planned_kernel<BLOCK_SIZE, IsMLA, IsDsv4Layout>,
      static_cast<const int64_t*>(miss_src_locs.data_ptr()),
      static_cast<const int64_t*>(miss_dst_locs.data_ptr()),
      static_cast<const int32_t*>(miss_counts.data_ptr()),
      static_cast<const int32_t*>(num_real_reqs.data_ptr()),
      host_cache_k.data_ptr(),
      (IsMLA || host_cache_v.ndim() == 0) ? (const void*)nullptr : host_cache_v.data_ptr(),
      device_buffer_k.data_ptr(),
      (IsMLA || device_buffer_v.ndim() == 0) ? (void*)nullptr : device_buffer_v.data_ptr(),
      plan_stride,
      item_size_bytes);
}

}  // namespace
