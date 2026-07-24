#pragma once

// Multi-step HiSparse swap kernels. DSv4 cache transfer remains in hisparse.cuh.

#include <sgl_kernel/tensor.h>  // TensorMatcher and symbolic tensor validation
#include <sgl_kernel/utils.h>   // RuntimeCheck and host utilities

#include <sgl_kernel/utils.cuh>  // LaunchKernel and PDL helpers

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <stdint.h>

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
constexpr int64_t HASH_DELETED = -2;
constexpr int32_t COMPACT_HASH_BITS = 13;
constexpr int32_t COMPACT_HASH_MASK = (1 << COMPACT_HASH_BITS) - 1;
constexpr int32_t COMPACT_APPROX_CLAIM_FLAG = int32_t{1} << 29;
constexpr int32_t COMPACT_APPROX_ADMIT_FLAG = int32_t{1} << 30;
constexpr int32_t APPROX_ADMISSION_STEPS = 1;
constexpr int32_t CLOCK_VICTIM_SAMPLES = 6;
constexpr uint32_t HASH_DEGRADED_FLAG = uint32_t{1} << 31;
constexpr uint32_t SCRATCH_EPOCH_MASK = HASH_DEGRADED_FLAG - 1;

struct MtpCacheState {
  int64_t* __restrict__ hash_primary;
  int64_t* __restrict__ hash_secondary;
  int32_t* __restrict__ ring_state;
  int32_t* __restrict__ ref_epochs;
  int64_t hash_stride;
  int64_t hash_size;
  int64_t ref_epoch_stride;
};

struct MtpMissWorkspace {
  int32_t* __restrict__ locs;
  int32_t* __restrict__ metadata;
  int32_t* __restrict__ counters;
  int64_t loc_stride;
  int64_t metadata_stride;
  int64_t counter_capacity;
};

constexpr int32_t ceil_log2_constexpr(int32_t value) {
  int32_t bits = 0;
  int32_t capacity = 1;
  while (capacity < value) {
    capacity <<= 1;
    ++bits;
  }
  return bits;
}

template <int32_t HOT_BUFFER_SIZE>
struct PackedRingState {
  static constexpr int32_t CURSOR_BITS = ceil_log2_constexpr(HOT_BUFFER_SIZE);
  static constexpr int32_t EPOCH_BITS = 32 - CURSOR_BITS < 23 ? 32 - CURSOR_BITS : 23;
  static constexpr uint32_t CURSOR_MASK = (uint32_t{1} << CURSOR_BITS) - 1;
  static constexpr uint32_t EPOCH_MASK = (uint32_t{1} << EPOCH_BITS) - 1;

  static_assert(HOT_BUFFER_SIZE > 1, "MTP hot buffer must contain at least two slots.");
  static_assert(CURSOR_BITS <= 15, "MTP CLOCK cursor requires hot_buffer_size <= 32768.");

  __device__ static int32_t next_epoch(int32_t packed_state) {
    int32_t epoch = static_cast<int32_t>((static_cast<uint32_t>(packed_state) >> CURSOR_BITS) + 1) & EPOCH_MASK;
    return epoch == 0 ? 1 : epoch;
  }

  __device__ static int32_t cursor(int32_t packed_state) {
    return static_cast<int32_t>(static_cast<uint32_t>(packed_state) & CURSOR_MASK) % HOT_BUFFER_SIZE;
  }

  __device__ static int32_t pack(int32_t epoch, int32_t cursor) {
    return static_cast<int32_t>(
        (static_cast<uint32_t>(epoch) << CURSOR_BITS) | (static_cast<uint32_t>(cursor) & CURSOR_MASK));
  }
};

template <int32_t HOT_BUFFER_SIZE>
struct PackedRingEntry {
  static constexpr int32_t SLOT_BITS = ceil_log2_constexpr(HOT_BUFFER_SIZE);
  static constexpr uint64_t SLOT_MASK = (uint64_t{1} << SLOT_BITS) - 1;
  static constexpr int64_t TOKEN_CAPACITY = int64_t{1} << 31;

  __device__ static int64_t pack(int32_t token, int32_t slot) {
    return static_cast<int64_t>(
        (static_cast<uint64_t>(static_cast<uint32_t>(token)) << SLOT_BITS) | static_cast<uint32_t>(slot));
  }

  __device__ static int32_t token(int64_t packed) {
    return static_cast<int32_t>(static_cast<uint64_t>(packed) >> SLOT_BITS);
  }

  __device__ static int32_t slot(int64_t packed) {
    return static_cast<int32_t>(static_cast<uint64_t>(packed) & SLOT_MASK);
  }
};

__device__ __forceinline__ int64_t atomic_cas_i64(int64_t* address, int64_t expected, int64_t desired) {
  return static_cast<int64_t>(atomicCAS(
      reinterpret_cast<unsigned long long*>(address),
      static_cast<unsigned long long>(expected),
      static_cast<unsigned long long>(desired)));
}

__device__ __forceinline__ int64_t atomic_exch_i64(int64_t* address, int64_t value) {
  return static_cast<int64_t>(
      atomicExch(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(value)));
}

// Knuth multiplicative hash for open-addressing table of size hash_size.
__device__ __forceinline__ int hash_slot(int32_t key, int hash_size) {
  const uint32_t size = static_cast<uint32_t>(hash_size);
  const uint32_t hash = static_cast<uint32_t>(key) * 2654435761u;
  return static_cast<int>((size & (size - 1)) == 0 ? hash & (size - 1) : hash % size);
}

__device__ __forceinline__ int next_hash_slot(int slot, int hash_size) {
  const uint32_t size = static_cast<uint32_t>(hash_size);
  const uint32_t next = static_cast<uint32_t>(slot + 1);
  return static_cast<int>((size & (size - 1)) == 0 ? next & (size - 1) : next % size);
}

__device__ __forceinline__ int ring_hash_slot(int32_t key, int64_t hash_size) {
  return static_cast<int>((static_cast<uint32_t>(key) * 2654435761u) & static_cast<uint32_t>(hash_size - 1));
}

__device__ __forceinline__ int ring_hash_slot_secondary(int32_t key, int64_t hash_size) {
  uint32_t hash = static_cast<uint32_t>(key);
  hash ^= hash >> 16;
  hash *= 0x7FEB352Du;
  hash ^= hash >> 15;
  hash *= 0x846CA68Bu;
  hash ^= hash >> 16;
  return static_cast<int>(hash & static_cast<uint32_t>(hash_size - 1));
}

__device__ __forceinline__ bool mark_cache_epoch(int32_t* __restrict__ cache_ref, int32_t cache_epoch) {
  return atomicExch(cache_ref, cache_epoch) != cache_epoch;
}

__device__ __forceinline__ int32_t scratch_union_lookup(
    const unsigned long long* __restrict__ table,
    const unsigned long long* __restrict__ indices,
    int32_t table_size,
    uint32_t scratch_epoch,
    int32_t token) {
  int32_t hash_pos = hash_slot(token, table_size);
  for (int32_t attempt = 0; attempt < table_size; ++attempt) {
    const auto packed = table[hash_pos];
    if (static_cast<uint32_t>(packed >> 32) != scratch_epoch) {
      return -1;
    }
    if (static_cast<int32_t>(static_cast<uint32_t>(packed)) == token) {
      const auto index_entry = indices[hash_pos];
      return static_cast<uint32_t>(index_entry >> 32) == scratch_epoch
                 ? static_cast<int32_t>(static_cast<uint32_t>(index_entry))
                 : -1;
    }
    hash_pos = next_hash_slot(hash_pos, table_size);
  }
  return -1;
}

template <int HOT_BUFFER_SIZE>
__device__ __forceinline__ int32_t ring_hash_lookup(
    const int64_t* __restrict__ keys,
    const int64_t* __restrict__ vals,
    int64_t hash_size,
    int32_t token,
    const int32_t* __restrict__ req_device_buffer_tokens,
    bool hash_degraded) {
  if (token < 0) {
    return -1;
  }
  using Entry = PackedRingEntry<HOT_BUFFER_SIZE>;
  const int32_t primary = keys[ring_hash_slot(token, hash_size)];
  if (primary >= 0 && Entry::token(primary) == token) {
    const int32_t slot = Entry::slot(primary);
    if (slot < HOT_BUFFER_SIZE && req_device_buffer_tokens[slot] == token) {
      return slot;
    }
  }
  const int32_t secondary = vals[ring_hash_slot_secondary(token, hash_size)];
  if (secondary >= 0 && Entry::token(secondary) == token) {
    const int32_t slot = Entry::slot(secondary);
    if (slot < HOT_BUFFER_SIZE && req_device_buffer_tokens[slot] == token) {
      return slot;
    }
  }
  if (hash_degraded) {
    for (int32_t slot = 0; slot < HOT_BUFFER_SIZE; ++slot) {
      if (req_device_buffer_tokens[slot] == token) {
        return slot;
      }
    }
  }
  return -1;
}

template <int HOT_BUFFER_SIZE>
__device__ __forceinline__ int32_t hot_cache_lookup(
    const int64_t* __restrict__ keys,
    const int64_t* __restrict__ vals,
    int64_t hash_size,
    int32_t token,
    const int32_t* __restrict__ req_device_buffer_tokens,
    bool hash_degraded) {
  if (token >= 0 && token < HOT_BUFFER_SIZE && req_device_buffer_tokens[token] == token) {
    return token;
  }
  return ring_hash_lookup<HOT_BUFFER_SIZE>(keys, vals, hash_size, token, req_device_buffer_tokens, hash_degraded);
}

template <int HOT_BUFFER_SIZE>
__device__ __forceinline__ int32_t ring_hash_insert_atomic(
    int64_t* __restrict__ keys, int64_t* __restrict__ vals, int64_t hash_size, int32_t token, int32_t buf_slot) {
  if (token < 0) {
    return -1;
  }
  using Entry = PackedRingEntry<HOT_BUFFER_SIZE>;
  const int64_t packed = Entry::pack(token, buf_slot);
  const int32_t primary_slot = ring_hash_slot(token, hash_size);
  const int32_t secondary_slot = ring_hash_slot_secondary(token, hash_size);
  int64_t old = keys[primary_slot];
  if (old >= 0 && Entry::token(old) == token) {
    atomic_exch_i64(keys + primary_slot, packed);
    return primary_slot;
  }
  if (old < 0 && atomic_cas_i64(keys + primary_slot, old, packed) == old) {
    return primary_slot;
  }
  old = vals[secondary_slot];
  if (old >= 0 && Entry::token(old) == token) {
    atomic_exch_i64(vals + secondary_slot, packed);
    return secondary_slot;
  }
  if (old < 0 && atomic_cas_i64(vals + secondary_slot, old, packed) == old) {
    return secondary_slot;
  }

  int64_t current = packed;
  bool use_secondary = false;
  for (int32_t kick = 0; kick < 64; ++kick) {
    const int32_t current_token = Entry::token(current);
    int64_t* table = use_secondary ? vals : keys;
    const int32_t slot =
        use_secondary ? ring_hash_slot_secondary(current_token, hash_size) : ring_hash_slot(current_token, hash_size);
    const int64_t displaced = atomic_exch_i64(table + slot, current);
    if (displaced < 0 || Entry::token(displaced) == current_token) {
      return slot;
    }
    current = displaced;
    use_secondary = !use_secondary;
  }
  return -1;
}

template <int HOT_BUFFER_SIZE>
__device__ __forceinline__ void ring_hash_erase_atomic(
    int64_t* __restrict__ keys, int64_t* __restrict__ vals, int64_t hash_size, int32_t token, int32_t buf_slot) {
  if (token < 0) {
    return;
  }
  using Entry = PackedRingEntry<HOT_BUFFER_SIZE>;
  const int64_t expected = Entry::pack(token, buf_slot);
  const int32_t primary_slot = ring_hash_slot(token, hash_size);
  if (atomic_cas_i64(keys + primary_slot, expected, HASH_DELETED) == expected) {
    return;
  }
  const int32_t secondary_slot = ring_hash_slot_secondary(token, hash_size);
  atomic_cas_i64(vals + secondary_slot, expected, HASH_DELETED);
}

#ifdef USE_ROCM
template <int ITEM_SIZE_BYTES>
__device__ __forceinline__ void
transfer_item_warp(int32_t lane_id, const void* __restrict__ src_addr, void* __restrict__ dst_addr) {
  const auto src = static_cast<const char*>(src_addr);
  auto dst = static_cast<char*>(dst_addr);

  constexpr int64_t word_count = ITEM_SIZE_BYTES / static_cast<int64_t>(sizeof(uint64_t));
  const auto src_words = reinterpret_cast<const uint64_t*>(src);
  auto dst_words = reinterpret_cast<uint64_t*>(dst);
  for (int64_t i = lane_id; i < word_count; i += WARP_SIZE) {
    dst_words[i] = src_words[i];
  }

  constexpr int64_t tail_start = word_count * static_cast<int64_t>(sizeof(uint64_t));
  for (int64_t i = tail_start + lane_id; i < ITEM_SIZE_BYTES; i += WARP_SIZE) {
    dst[i] = src[i];
  }
}

#else
template <int ITEM_SIZE_BYTES>
__device__ __forceinline__ void transfer_item_warp(int32_t lane_id, const void* src_addr, void* dst_addr) {
  // Issue the 512B body and 64B edge loads before either store so both host
  // reads can remain in flight. Rows alternate between 0B and 64B offsets in
  // a 128B transaction, so place the body on the aligned side of each row.
  if constexpr (ITEM_SIZE_BYTES == 576) {
    const auto src = static_cast<const char*>(src_addr);
    auto dst = static_cast<char*>(dst_addr);
    const bool edge_first = (reinterpret_cast<uintptr_t>(src_addr) & 127u) == 64u;
    const int32_t body_offset = edge_first ? 64 : 0;
    const int32_t edge_offset = edge_first ? 0 : 512;
    uint64_t body_lo, body_hi;
    uint64_t edge_lo, edge_hi;
    const auto body_src = reinterpret_cast<const uint64_t*>(src + body_offset + lane_id * 16);
    auto body_dst = reinterpret_cast<uint64_t*>(dst + body_offset + lane_id * 16);
    asm volatile("ld.global.nc.v2.b64 {%0,%1},[%2];" : "=l"(body_lo), "=l"(body_hi) : "l"(body_src) : "memory");
    if (lane_id < 4) {
      const auto edge_src = reinterpret_cast<const uint64_t*>(src + edge_offset + lane_id * 16);
      asm volatile("ld.global.nc.v2.b64 {%0,%1},[%2];" : "=l"(edge_lo), "=l"(edge_hi) : "l"(edge_src) : "memory");
    }
    asm volatile("st.global.cg.v2.b64 [%0],{%1,%2};" ::"l"(body_dst), "l"(body_lo), "l"(body_hi) : "memory");
    if (lane_id < 4) {
      auto edge_dst = reinterpret_cast<uint64_t*>(dst + edge_offset + lane_id * 16);
      asm volatile("st.global.cg.v2.b64 [%0],{%1,%2};" ::"l"(edge_dst), "l"(edge_lo), "l"(edge_hi) : "memory");
    }
    return;
  }

  // 128-bit bulk transfer via paired 64-bit loads (avoids alignment issues with uint4)
  constexpr int total_pairs = ITEM_SIZE_BYTES / 16;  // number of 16-byte chunks
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
  constexpr int tail_8B = (ITEM_SIZE_BYTES - total_pairs * 16) / 8;
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

__device__ __forceinline__ int first_set_lane(BallotMask mask) {
#ifdef USE_ROCM
  return __ffsll(mask) - 1;
#else
  return __ffs(mask) - 1;
#endif
}

template <int HOT_BUFFER_SIZE>
__device__ __forceinline__ bool try_get_extra_page_device_loc(
    int32_t token_idx,
    int64_t seq_len,
    const int32_t* __restrict__ req_device_buffer_tokens,
    const int32_t* __restrict__ req_device_buffer_locs,
    int64_t page_size,
    int32_t* __restrict__ out_loc) {
  int64_t slot = -1;
  if (static_cast<int64_t>(token_idx) >= seq_len - page_size) {
    if (page_size == 4) {
      const int4 page_tokens = *reinterpret_cast<const int4*>(req_device_buffer_tokens + HOT_BUFFER_SIZE);
      if (page_tokens.x == token_idx) {
        slot = HOT_BUFFER_SIZE;
      } else if (page_tokens.y == token_idx) {
        slot = HOT_BUFFER_SIZE + 1;
      } else if (page_tokens.z == token_idx) {
        slot = HOT_BUFFER_SIZE + 2;
      } else if (page_tokens.w == token_idx) {
        slot = HOT_BUFFER_SIZE + 3;
      }
    } else {
      for (int64_t candidate_slot = HOT_BUFFER_SIZE; candidate_slot < HOT_BUFFER_SIZE + page_size; candidate_slot++) {
        if (req_device_buffer_tokens[candidate_slot] == token_idx) {
          slot = candidate_slot;
          break;
        }
      }
    }
  }

  if (slot < HOT_BUFFER_SIZE || slot >= HOT_BUFFER_SIZE + page_size) {
    return false;
  }
  const int32_t loc = req_device_buffer_locs[slot];
  if (loc < 0) {
    return false;
  }
  *out_loc = loc;
  return true;
}

// Flatten all speculative steps. Each lane resolves one occurrence; the warp
// cooperatively copies only lanes that won a unique-miss claim.
template <
    int BLOCK_SIZE,
    int NUM_TOP_K,
    int HOT_BUFFER_SIZE,
    int ITEM_SIZE_BYTES,
    int NUM_STEPS,
    typename SeqLensT,
    typename ReqPoolIndicesT>
__global__ void load_cache_to_device_buffer_mtp_gather_kernel(
    const int32_t* __restrict__ top_k_tokens,
    int32_t* __restrict__ device_buffer_tokens,
    const int64_t* __restrict__ host_cache_locs,
    int32_t* __restrict__ device_buffer_locs,
    const void* __restrict__ host_cache_k,
    void* __restrict__ device_buffer_k,
    int32_t* __restrict__ top_k_device_locs,
    const ReqPoolIndicesT* __restrict__ req_pool_indices,
    const SeqLensT* __restrict__ seq_lens,
    MtpCacheState cache_state,
    MtpMissWorkspace miss_workspace,
    const int32_t* __restrict__ num_real_reqs,
    int64_t buffer_stride_0,
    int64_t host_stride,
    int64_t top_k_tokens_stride,
    int64_t top_k_device_locs_stride,
    int64_t page_size) {
  const int bid = blockIdx.x;
  if (bid >= num_real_reqs[0]) return;

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  constexpr int64_t total_occurrences = NUM_STEPS * NUM_TOP_K;
  const int64_t total_warps = static_cast<int64_t>(gridDim.y) * NUM_WARPS;
  const int64_t global_warp = static_cast<int64_t>(blockIdx.y) * NUM_WARPS + warp_id;
  const int64_t occ = static_cast<int64_t>(lane_id) * total_warps + global_warp;
  const int64_t rid = req_pool_indices[bid];
  const int64_t buffer_offset = rid * buffer_stride_0;

  int32_t* req_device_buffer_tokens = device_buffer_tokens + buffer_offset;
  int32_t* req_device_buffer_locs = device_buffer_locs + buffer_offset;
  const int64_t* req_host_cache_locs = host_cache_locs + rid * host_stride;
  int64_t* req_ring_hash_keys = cache_state.hash_primary + rid * cache_state.hash_stride;
  int64_t* req_ring_hash_vals = cache_state.hash_secondary + rid * cache_state.hash_stride;
  int32_t* req_cache_ref_bits = cache_state.ref_epochs + rid * cache_state.ref_epoch_stride;
  int32_t* req_scratch_locs = miss_workspace.locs + rid * miss_workspace.loc_stride;
  int32_t* req_scratch_tokens = miss_workspace.metadata + rid * miss_workspace.metadata_stride;
  auto* req_scratch_table = reinterpret_cast<unsigned long long*>(req_scratch_tokens);
  auto* req_scratch_indices = req_scratch_table + total_occurrences;
  int32_t* req_compact_hash_positions = req_scratch_tokens + 4 * total_occurrences;
  int32_t* req_work_count = miss_workspace.counters + miss_workspace.counter_capacity + rid;
  int32_t* req_union_hit_count = miss_workspace.counters + 2 * miss_workspace.counter_capacity + rid;
  int32_t* req_scratch_generation = miss_workspace.counters + 3 * miss_workspace.counter_capacity + rid;
  using RingState = PackedRingState<HOT_BUFFER_SIZE>;
  const int32_t cache_epoch = RingState::next_epoch(cache_state.ring_state[rid]);
  const uint32_t packed_scratch_generation = static_cast<uint32_t>(*req_scratch_generation);
  const bool hash_degraded = (packed_scratch_generation & HASH_DEGRADED_FLAG) != 0;
  uint32_t next_scratch_epoch = (packed_scratch_generation & SCRATCH_EPOCH_MASK) + 1;
  next_scratch_epoch = next_scratch_epoch > SCRATCH_EPOCH_MASK ? 1 : next_scratch_epoch;
  const int32_t scratch_epoch = static_cast<int32_t>(next_scratch_epoch);
  const int32_t* req_top_k_tokens = top_k_tokens + bid * top_k_tokens_stride;
  int32_t* req_top_k_device_locs = top_k_device_locs + bid * top_k_device_locs_stride;

  int32_t token = -1;
  int32_t loc = -1;
  int32_t cache_slot = -1;
  bool copy_owner = false;
  int64_t src_loc = -1;
  bool needs_cache_lookup = false;
  if (occ < total_occurrences) {
    const int64_t step = occ / NUM_TOP_K;
    token = req_top_k_tokens[occ];
    const int64_t seq_len = static_cast<int64_t>(seq_lens[bid * NUM_STEPS + step]);
    if (token >= 0 && token < seq_len) {
      int32_t direct_loc = -1;
      if (try_get_extra_page_device_loc<HOT_BUFFER_SIZE>(
              token, seq_len, req_device_buffer_tokens, req_device_buffer_locs, page_size, &direct_loc)) {
        loc = direct_loc;
      } else {
        needs_cache_lookup = true;
      }
    }
  }

  // The flattened occurrence mapping places the same top-k position from
  // different MTP steps in one warp. Stable hits only need one Ring lookup.
  int32_t lookup_owner_lane = lane_id;
  bool reuse_owner_loc = false;
  if (total_warps <= NUM_TOP_K && NUM_TOP_K % total_warps == 0) {
    const int32_t lanes_per_step = static_cast<int32_t>(NUM_TOP_K / total_warps);
    if (lanes_per_step > 0 && NUM_STEPS * lanes_per_step <= WARP_SIZE) {
      lookup_owner_lane = lane_id % lanes_per_step;
      const int32_t owner_token = __shfl_sync(FULL_WARP_MASK, token, lookup_owner_lane);
      const int32_t owner_needs_lookup = __shfl_sync(FULL_WARP_MASK, needs_cache_lookup ? 1 : 0, lookup_owner_lane);
      reuse_owner_loc =
          needs_cache_lookup && lane_id != lookup_owner_lane && owner_needs_lookup != 0 && owner_token == token;
    }
  }

  if (needs_cache_lookup && !reuse_owner_loc) {
    cache_slot = hot_cache_lookup<HOT_BUFFER_SIZE>(
        req_ring_hash_keys, req_ring_hash_vals, cache_state.hash_size, token, req_device_buffer_tokens, hash_degraded);
    if (cache_slot >= 0) {
      loc = req_device_buffer_locs[cache_slot];
    } else {
      src_loc = req_host_cache_locs[token];
      if (src_loc >= 0) {
        int32_t hash_pos = hash_slot(token, static_cast<int>(total_occurrences));
        const auto epoch_bits = static_cast<unsigned long long>(static_cast<uint32_t>(scratch_epoch)) << 32;
        const auto token_bits = static_cast<uint32_t>(token);
        const auto packed = epoch_bits | token_bits;
        for (int64_t attempt = 0; attempt < total_occurrences; ++attempt) {
          auto old = atomicCAS(req_scratch_table + hash_pos, 0ull, 0ull);
          if (static_cast<uint32_t>(old >> 32) != static_cast<uint32_t>(scratch_epoch)) {
            const auto claimed = atomicCAS(req_scratch_table + hash_pos, old, packed);
            if (claimed == old) {
              const int32_t unique_idx = atomicAdd(req_work_count, 1);
              req_compact_hash_positions[unique_idx] = hash_pos;
              __threadfence();
              atomicExch(req_scratch_indices + hash_pos, epoch_bits | static_cast<uint32_t>(unique_idx));
              if (unique_idx < miss_workspace.loc_stride) {
                loc = req_scratch_locs[unique_idx];
                copy_owner = loc >= 0;
              }
              break;
            }
            continue;
          }
          if (static_cast<uint32_t>(old) == token_bits) {
            auto index_entry = atomicCAS(req_scratch_indices + hash_pos, 0ull, 0ull);
            while (static_cast<uint32_t>(index_entry >> 32) != static_cast<uint32_t>(scratch_epoch)) {
              index_entry = atomicCAS(req_scratch_indices + hash_pos, 0ull, 0ull);
            }
            const int32_t unique_idx = static_cast<int32_t>(static_cast<uint32_t>(index_entry));
            if (unique_idx < miss_workspace.loc_stride) {
              loc = req_scratch_locs[unique_idx];
            }
            break;
          }
          hash_pos = next_hash_slot(hash_pos, static_cast<int>(total_occurrences));
        }
      }
    }
  }

  const int32_t owner_loc = __shfl_sync(FULL_WARP_MASK, loc, lookup_owner_lane);
  const int32_t owner_cache_slot = __shfl_sync(FULL_WARP_MASK, cache_slot, lookup_owner_lane);
  if (reuse_owner_loc) {
    loc = owner_loc;
    cache_slot = owner_cache_slot;
  }
  if (cache_slot >= 0) {
    if (mark_cache_epoch(req_cache_ref_bits + cache_slot, cache_epoch)) {
      atomicAdd(req_union_hit_count, 1);
    }
  }
  if (occ < total_occurrences) {
    req_top_k_device_locs[occ] = loc;
  }

  BallotMask copy_mask = __ballot_sync(FULL_WARP_MASK, copy_owner);
  while (copy_mask != 0) {
    const int owner_lane = first_set_lane(copy_mask);
    const int32_t copy_token = __shfl_sync(FULL_WARP_MASK, token, owner_lane);
    const int32_t copy_loc = __shfl_sync(FULL_WARP_MASK, loc, owner_lane);
    const int64_t copy_src_loc = __shfl_sync(FULL_WARP_MASK, src_loc, owner_lane);
    if (copy_token >= 0 && copy_loc >= 0 && copy_src_loc >= 0) {
      const auto src_k = static_cast<const char*>(host_cache_k) + copy_src_loc * ITEM_SIZE_BYTES;
      auto dst_k = static_cast<char*>(device_buffer_k) + static_cast<int64_t>(copy_loc) * ITEM_SIZE_BYTES;
      transfer_item_warp<ITEM_SIZE_BYTES>(lane_id, src_k, dst_k);
    }
    copy_mask &= ~(static_cast<BallotMask>(1) << owner_lane);
  }
}

template <int NUM_TOP_K, int HOT_BUFFER_SIZE, int ITEM_SIZE_BYTES, int NUM_STEPS, typename ReqPoolIndicesT>
__global__ void load_cache_to_device_buffer_mtp_commit_kernel(
    const int32_t* __restrict__ top_k_tokens,
    int32_t* __restrict__ top_k_device_locs,
    int32_t* __restrict__ device_buffer_tokens,
    int32_t* __restrict__ device_buffer_locs,
    const int64_t* __restrict__ host_cache_locs,
    const void* __restrict__ host_cache_k,
    void* __restrict__ device_buffer_k,
    const ReqPoolIndicesT* __restrict__ req_pool_indices,
    MtpCacheState cache_state,
    MtpMissWorkspace miss_workspace,
    const int32_t* __restrict__ num_real_reqs,
    int64_t top_k_tokens_stride,
    int64_t top_k_device_locs_stride,
    int64_t buffer_stride_0,
    int64_t host_stride) {
  device::PDLWaitPrimary<true>();
  const int bid = blockIdx.x;
  if (bid >= num_real_reqs[0]) return;

  const int lane_id = threadIdx.x % WARP_SIZE;
  const int64_t rid = req_pool_indices[bid];
  constexpr int64_t total_occurrences = NUM_STEPS * NUM_TOP_K;
  using RingState = PackedRingState<HOT_BUFFER_SIZE>;
  const int32_t packed_ring_state = cache_state.ring_state[rid];
  const int32_t cache_epoch = RingState::next_epoch(packed_ring_state);
  const int32_t clock_cursor = RingState::cursor(packed_ring_state);
  const int64_t buffer_offset = rid * buffer_stride_0;
  int32_t* req_device_buffer_tokens = device_buffer_tokens + buffer_offset;
  int32_t* req_device_buffer_locs = device_buffer_locs + buffer_offset;
  int64_t* req_ring_hash_keys = cache_state.hash_primary + rid * cache_state.hash_stride;
  int64_t* req_ring_hash_vals = cache_state.hash_secondary + rid * cache_state.hash_stride;
  int32_t* req_cache_ref_bits = cache_state.ref_epochs + rid * cache_state.ref_epoch_stride;
  int32_t* req_scratch_locs = miss_workspace.locs + rid * miss_workspace.loc_stride;
  int32_t* req_scratch_tokens = miss_workspace.metadata + rid * miss_workspace.metadata_stride;
  const auto* req_scratch_table = reinterpret_cast<const unsigned long long*>(req_scratch_tokens);
  auto* req_scratch_indices = reinterpret_cast<unsigned long long*>(req_scratch_tokens) + total_occurrences;
  int32_t* req_compact_hash_positions = req_scratch_tokens + 4 * total_occurrences;
  const int64_t* req_host_cache_locs = host_cache_locs + rid * host_stride;
  const int32_t* req_top_k_tokens = top_k_tokens + bid * top_k_tokens_stride;
  int32_t* req_top_k_device_locs = top_k_device_locs + bid * top_k_device_locs_stride;
  int32_t* req_work_count = miss_workspace.counters + miss_workspace.counter_capacity + rid;
  int32_t* req_union_hit_count = miss_workspace.counters + 2 * miss_workspace.counter_capacity + rid;
  int32_t* req_scratch_generation = miss_workspace.counters + 3 * miss_workspace.counter_capacity + rid;
  const uint32_t packed_scratch_generation = static_cast<uint32_t>(*req_scratch_generation);
  const bool hash_degraded = (packed_scratch_generation & HASH_DEGRADED_FLAG) != 0;
  uint32_t next_scratch_epoch = (packed_scratch_generation & SCRATCH_EPOCH_MASK) + 1;
  next_scratch_epoch = next_scratch_epoch > SCRATCH_EPOCH_MASK ? 1 : next_scratch_epoch;
  const int32_t scratch_epoch = static_cast<int32_t>(next_scratch_epoch);
  const int32_t miss_count = *req_work_count;
  const int32_t union_hit_count = *req_union_hit_count;
  const bool scratch_overflow = miss_count > miss_workspace.loc_stride;
  const int64_t compact_iterations = (miss_count + blockDim.x - 1) / blockDim.x;
  const bool lock_free_single_pass = compact_iterations == 1;

  // The union fast path can preserve every token needed by this MTP group.
  // Its victim selection below partitions the ring by miss ordinal, avoiding
  // a full hot-cache scan while keeping all current union hits protected.
  __shared__ int32_t s_use_union_clock;
  __shared__ int32_t s_approx_admission_budget;
  __shared__ int32_t s_hash_degraded;
  if (threadIdx.x == 0) {
    s_use_union_clock = miss_count + union_hit_count <= HOT_BUFFER_SIZE;
    const int32_t mandatory_direct_misses =
        scratch_overflow ? miss_count - static_cast<int32_t>(miss_workspace.loc_stride) : 0;
    s_approx_admission_budget =
        s_use_union_clock ? 0
                          : (scratch_overflow ? max(0, HOT_BUFFER_SIZE - union_hit_count - mandatory_direct_misses)
                                              : HOT_BUFFER_SIZE);
    s_hash_degraded = hash_degraded;
  }
  __syncthreads();
  if (!s_use_union_clock) {
    // The complete union is already available from hot + scratch for this
    // attention call. Only the cache update is approximate: keep the most
    // recent speculative working set and avoid replaying four LRU passes.
    if (!scratch_overflow) {
      for (int32_t slot = threadIdx.x; slot < HOT_BUFFER_SIZE; slot += blockDim.x) {
        req_cache_ref_bits[slot] = 0;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      *req_union_hit_count = 0;
    }
    __syncthreads();
    constexpr int64_t admission_step = NUM_STEPS > APPROX_ADMISSION_STEPS ? NUM_STEPS - APPROX_ADMISSION_STEPS : 0;
    const int64_t admission_start = admission_step * NUM_TOP_K;
    for (int64_t occ = admission_start + threadIdx.x; occ < total_occurrences; occ += blockDim.x) {
      const int32_t token = req_top_k_tokens[occ];
      if (token < 0) {
        continue;
      }
      const int32_t slot = hot_cache_lookup<HOT_BUFFER_SIZE>(
          req_ring_hash_keys,
          req_ring_hash_vals,
          cache_state.hash_size,
          token,
          req_device_buffer_tokens,
          s_hash_degraded != 0);
      if (slot >= 0) {
        mark_cache_epoch(req_cache_ref_bits + slot, cache_epoch);
        continue;
      }
      const int32_t unique_idx = scratch_union_lookup(
          req_scratch_table,
          req_scratch_indices,
          static_cast<int32_t>(total_occurrences),
          static_cast<uint32_t>(scratch_epoch),
          token);
      if (unique_idx >= 0 && unique_idx < miss_workspace.loc_stride) {
        const int32_t old_metadata = atomicOr(req_compact_hash_positions + unique_idx, COMPACT_APPROX_CLAIM_FLAG);
        if ((old_metadata & COMPACT_APPROX_CLAIM_FLAG) == 0) {
          const int32_t admission_ordinal = atomicAdd(req_union_hit_count, 1);
          if (admission_ordinal < s_approx_admission_budget) {
            atomicOr(req_compact_hash_positions + unique_idx, COMPACT_APPROX_ADMIT_FLAG);
          }
        }
      }
    }
    __syncthreads();
  }

  for (int64_t iteration = 0; iteration < compact_iterations; ++iteration) {
    const int64_t compact_idx = threadIdx.x + iteration * blockDim.x;
    bool copy_owner = false;
    int64_t copy_src_loc = -1;
    int32_t copy_dst_loc = -1;
    int64_t hash_pos = -1;
    int32_t token = -1;
    int32_t unique_idx = -1;
    int32_t victim = -1;
    bool direct_overflow = false;
    bool rotate_compact = false;

    if (compact_idx < miss_count) {
      const int32_t compact_entry = req_compact_hash_positions[compact_idx];
      hash_pos = compact_entry & COMPACT_HASH_MASK;
      const auto packed = req_scratch_table[hash_pos];
      const auto index_entry = req_scratch_indices[hash_pos];
      if (static_cast<uint32_t>(packed >> 32) == static_cast<uint32_t>(scratch_epoch) &&
          static_cast<uint32_t>(index_entry >> 32) == static_cast<uint32_t>(scratch_epoch)) {
        token = static_cast<int32_t>(static_cast<uint32_t>(packed));
        unique_idx = static_cast<int32_t>(static_cast<uint32_t>(index_entry));
        direct_overflow = scratch_overflow && unique_idx >= miss_workspace.loc_stride;
        const bool approximate_admission = !s_use_union_clock && (compact_entry & COMPACT_APPROX_ADMIT_FLAG) != 0;
        rotate_compact = unique_idx < miss_workspace.loc_stride && (s_use_union_clock || approximate_admission);

        if ((direct_overflow || rotate_compact) && s_use_union_clock) {
          uint32_t oldest_age = 0;
          const int32_t partition_size = (HOT_BUFFER_SIZE - 1 - compact_idx) / miss_count + 1;
          const int32_t sample_count = min(partition_size, CLOCK_VICTIM_SAMPLES);
          for (int32_t sample = 0; sample < sample_count; ++sample) {
            const int32_t partition_offset = sample * partition_size / sample_count;
            const int32_t linear = compact_idx + partition_offset * miss_count;
            const int32_t candidate = (clock_cursor + linear) % HOT_BUFFER_SIZE;
            const int32_t ref_epoch = req_cache_ref_bits[candidate];
            if (ref_epoch == cache_epoch) {
              continue;
            }
            const uint32_t age =
                (static_cast<uint32_t>(cache_epoch) - static_cast<uint32_t>(ref_epoch)) & RingState::EPOCH_MASK;
            if (victim < 0 || age > oldest_age) {
              victim = candidate;
              oldest_age = age;
            }
          }
          if (victim >= 0) {
            req_cache_ref_bits[victim] = cache_epoch;
          }
        } else if (direct_overflow || rotate_compact) {
          const int32_t scratch_capacity = static_cast<int32_t>(miss_workspace.loc_stride);
          const int32_t victim_count = direct_overflow ? miss_count - scratch_capacity : miss_count;
          const int32_t victim_ordinal = direct_overflow ? unique_idx - scratch_capacity : unique_idx;
          for (int32_t linear = victim_ordinal; linear < HOT_BUFFER_SIZE; linear += victim_count) {
            const int32_t candidate = static_cast<int32_t>(
                (static_cast<uint32_t>(linear) * 2654435761u + static_cast<uint32_t>(cache_epoch)) % HOT_BUFFER_SIZE);
            const int32_t observed = req_cache_ref_bits[candidate];
            if (observed == cache_epoch) {
              continue;
            }
            if (lock_free_single_pass) {
              req_cache_ref_bits[candidate] = cache_epoch;
              victim = candidate;
              break;
            }
            if (atomicCAS(req_cache_ref_bits + candidate, observed, cache_epoch) == observed) {
              victim = candidate;
              break;
            }
          }
        }
      }
    }

    // Ring partitions are disjoint. Mark their direct choices before the rare
    // fallback probes globally for a partition that had no available slot.
    __syncthreads();

    if ((direct_overflow || rotate_compact) && victim < 0) {
      const uint32_t start =
          (static_cast<uint32_t>(clock_cursor) + static_cast<uint32_t>(compact_idx) * 2654435761u) % HOT_BUFFER_SIZE;
      for (int32_t attempt = 0; attempt < HOT_BUFFER_SIZE; ++attempt) {
        const int32_t candidate =
            static_cast<int32_t>((start + static_cast<uint32_t>(attempt) * 2654435761u) % HOT_BUFFER_SIZE);
        const int32_t observed = req_cache_ref_bits[candidate];
        if (observed == cache_epoch) {
          continue;
        }
        if (atomicCAS(req_cache_ref_bits + candidate, observed, cache_epoch) == observed) {
          victim = candidate;
          break;
        }
      }
    }

    if (direct_overflow || rotate_compact) {
      if (victim >= 0) {
        const int32_t old_loc = req_device_buffer_locs[victim];
        const int32_t new_loc = direct_overflow ? old_loc : req_scratch_locs[unique_idx];
        if (new_loc >= 0 && old_loc >= 0) {
          const int32_t old_token = req_device_buffer_tokens[victim];
          req_device_buffer_tokens[victim] = token;
          req_cache_ref_bits[victim] = cache_epoch;
          if (rotate_compact) {
            req_device_buffer_locs[victim] = new_loc;
            req_scratch_locs[unique_idx] = old_loc;
          } else {
            const auto epoch_bits = static_cast<unsigned long long>(static_cast<uint32_t>(scratch_epoch)) << 32;
            atomicExch(req_scratch_indices + hash_pos, epoch_bits | static_cast<uint32_t>(old_loc));
            copy_src_loc = req_host_cache_locs[token];
            copy_dst_loc = old_loc;
            copy_owner = copy_src_loc >= 0;
          }
          ring_hash_erase_atomic<HOT_BUFFER_SIZE>(
              req_ring_hash_keys, req_ring_hash_vals, cache_state.hash_size, old_token, victim);
          const int32_t inserted_slot = ring_hash_insert_atomic<HOT_BUFFER_SIZE>(
              req_ring_hash_keys, req_ring_hash_vals, cache_state.hash_size, token, victim);
          if (inserted_slot < 0) {
            atomicExch(&s_hash_degraded, 1);
          }
        }
      } else if (direct_overflow) {
        const auto epoch_bits = static_cast<unsigned long long>(static_cast<uint32_t>(scratch_epoch)) << 32;
        atomicExch(req_scratch_indices + hash_pos, epoch_bits | UINT32_MAX);
      }
    }

    BallotMask copy_mask = __ballot_sync(FULL_WARP_MASK, copy_owner);
    while (copy_mask != 0) {
      const int owner_lane = first_set_lane(copy_mask);
      const int64_t src_loc = __shfl_sync(FULL_WARP_MASK, copy_src_loc, owner_lane);
      const int32_t dst_loc = __shfl_sync(FULL_WARP_MASK, copy_dst_loc, owner_lane);
      if (src_loc >= 0 && dst_loc >= 0) {
        const auto src_k = static_cast<const char*>(host_cache_k) + src_loc * ITEM_SIZE_BYTES;
        auto dst_k = static_cast<char*>(device_buffer_k) + static_cast<int64_t>(dst_loc) * ITEM_SIZE_BYTES;
        transfer_item_warp<ITEM_SIZE_BYTES>(lane_id, src_k, dst_k);
      }
      copy_mask &= ~(static_cast<BallotMask>(1) << owner_lane);
    }
  }

  __syncthreads();
  if (scratch_overflow) {
    for (int64_t occ = threadIdx.x; occ < total_occurrences; occ += blockDim.x) {
      if (req_top_k_device_locs[occ] >= 0) {
        continue;
      }
      const int32_t token = req_top_k_tokens[occ];
      if (token < 0) {
        continue;
      }

      int32_t hash_pos = hash_slot(token, static_cast<int>(total_occurrences));
      for (int64_t attempt = 0; attempt < total_occurrences; ++attempt) {
        const auto packed = req_scratch_table[hash_pos];
        if (static_cast<uint32_t>(packed >> 32) != static_cast<uint32_t>(scratch_epoch)) {
          break;
        }
        if (static_cast<uint32_t>(packed) == static_cast<uint32_t>(token)) {
          const auto index_entry = req_scratch_indices[hash_pos];
          if (static_cast<uint32_t>(index_entry >> 32) == static_cast<uint32_t>(scratch_epoch)) {
            const uint32_t resolved_loc = static_cast<uint32_t>(index_entry);
            if (resolved_loc != UINT32_MAX) {
              req_top_k_device_locs[occ] = static_cast<int32_t>(resolved_loc);
            }
          }
          break;
        }
        hash_pos = next_hash_slot(hash_pos, static_cast<int>(total_occurrences));
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    miss_workspace.counters[rid] = miss_count;
    *req_work_count = 0;
    *req_union_hit_count = 0;
    *req_scratch_generation =
        static_cast<int32_t>(static_cast<uint32_t>(scratch_epoch) | (s_hash_degraded != 0 ? HASH_DEGRADED_FLAG : 0));
    const int32_t next_cursor =
        miss_count > 0 ? (clock_cursor + min(miss_count, HOT_BUFFER_SIZE)) % HOT_BUFFER_SIZE : clock_cursor;
    cache_state.ring_state[rid] = RingState::pack(cache_epoch, next_cursor);
  }
}

template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE, int ITEM_SIZE_BYTES, int NUM_STEPS>
void load_cache_to_device_buffer_mtp(
    tvm::ffi::TensorView top_k_tokens,
    tvm::ffi::TensorView device_buffer_tokens,
    tvm::ffi::TensorView host_cache_locs,
    tvm::ffi::TensorView device_buffer_locs,
    tvm::ffi::TensorView host_cache_k,
    tvm::ffi::TensorView device_buffer_k,
    tvm::ffi::TensorView top_k_device_locs,
    tvm::ffi::TensorView req_pool_indices,
    tvm::ffi::TensorView seq_lens,
    tvm::ffi::TensorView ring_hash_keys,
    tvm::ffi::TensorView ring_hash_vals,
    tvm::ffi::TensorView ring_epoch,
    tvm::ffi::TensorView cache_ref_bits,
    tvm::ffi::TensorView scratch_locs,
    tvm::ffi::TensorView scratch_tokens,
    tvm::ffi::TensorView scratch_counts,
    tvm::ffi::TensorView num_real_reqs,
    int64_t page_size) {
  using namespace host;

  static_assert(NUM_STEPS > 1 && NUM_STEPS <= 4, "HiSparse MTP swap requires 2-4 steps.");
  static_assert(NUM_TOP_K >= 1024, "HiSparse MTP swap requires top_k >= 1024.");
  static_assert(NUM_STEPS * NUM_TOP_K <= 8192, "HiSparse MTP swap supports at most 8192 occurrences.");

  const int64_t bs = top_k_tokens.shape()[0];
  constexpr int64_t total_occurrences = NUM_STEPS * NUM_TOP_K;
  RuntimeCheck(top_k_tokens.ndim() == 3, "MTP top_k_tokens must have shape [batch, steps, top_k].");
  RuntimeCheck(top_k_device_locs.ndim() == 3, "MTP output must have shape [batch, steps, top_k].");
  RuntimeCheck(top_k_tokens.shape()[1] == NUM_STEPS, "top_k_tokens step dimension mismatch.");
  RuntimeCheck(top_k_tokens.shape()[2] == NUM_TOP_K, "top_k_tokens top-k dimension mismatch.");
  RuntimeCheck(ring_hash_keys.shape()[1] == ring_hash_vals.shape()[1], "ring hash key/value capacity mismatch.");
  RuntimeCheck(
      ring_hash_keys.shape()[1] > 0 && (ring_hash_keys.shape()[1] & (ring_hash_keys.shape()[1] - 1)) == 0,
      "ring hash capacity must be a power of two.");
  RuntimeCheck(cache_ref_bits.shape()[1] >= HOT_BUFFER_SIZE, "cache_ref_bits hot capacity is too small.");
  RuntimeCheck(scratch_locs.shape()[1] > 0, "MTP scratch_locs must not be empty.");
  RuntimeCheck(
      scratch_tokens.ndim() == 2 && scratch_tokens.shape()[1] >= 5 * total_occurrences,
      "MTP scratch metadata capacity is too small.");
  RuntimeCheck(scratch_tokens.strides()[0] % 2 == 0, "MTP scratch metadata stride must be 64-bit aligned.");
  RuntimeCheck(
      scratch_counts.ndim() == 1 && scratch_counts.shape()[0] >= 4 * scratch_locs.shape()[0],
      "MTP scratch counter capacity is too small.");

  const int64_t host_stride = host_cache_locs.shape()[1];
  RuntimeCheck(
      host_stride <= PackedRingEntry<HOT_BUFFER_SIZE>::TOKEN_CAPACITY,
      "MTP packed ring metadata supports sequence lengths up to ",
      PackedRingEntry<HOT_BUFFER_SIZE>::TOKEN_CAPACITY,
      ", got ",
      host_stride);
  const int64_t buffer_stride_0 = device_buffer_tokens.strides()[0];
  const int64_t top_k_tokens_stride = top_k_tokens.strides()[0];
  const int64_t top_k_device_locs_stride = top_k_device_locs.strides()[0];
  const int64_t ring_hash_stride_0 = ring_hash_keys.strides()[0];
  const int64_t ring_hash_size = ring_hash_keys.shape()[1];
  const int64_t cache_ref_stride_0 = cache_ref_bits.strides()[0];
  const int64_t scratch_stride_0 = scratch_locs.strides()[0];
  const int64_t scratch_tokens_stride_0 = scratch_tokens.strides()[0];
  const int64_t scratch_count_capacity = scratch_counts.shape()[0] / 4;
  const MtpCacheState cache_state{
      static_cast<int64_t*>(ring_hash_keys.data_ptr()),
      static_cast<int64_t*>(ring_hash_vals.data_ptr()),
      static_cast<int32_t*>(ring_epoch.data_ptr()),
      static_cast<int32_t*>(cache_ref_bits.data_ptr()),
      ring_hash_stride_0,
      ring_hash_size,
      cache_ref_stride_0};
  const MtpMissWorkspace miss_workspace{
      static_cast<int32_t*>(scratch_locs.data_ptr()),
      static_cast<int32_t*>(scratch_tokens.data_ptr()),
      static_cast<int32_t*>(scratch_counts.data_ptr()),
      scratch_stride_0,
      scratch_tokens_stride_0,
      scratch_count_capacity};
  const auto device = LaunchKernel::resolve_device(top_k_tokens.device());

  bool use_pdl = false;
#ifndef USE_ROCM
  static thread_local int cached_pdl_device = -1;
  static thread_local bool cached_use_pdl = false;
  const int current_device = top_k_tokens.device().device_id;
  if (cached_pdl_device != current_device) {
    int compute_capability_major = 0;
    RuntimeDeviceCheck(
        cudaDeviceGetAttribute(&compute_capability_major, cudaDevAttrComputeCapabilityMajor, current_device));
    cached_pdl_device = current_device;
    cached_use_pdl = compute_capability_major >= 9;
  }
  use_pdl = cached_use_pdl;
#endif
  auto launch =
      [&](auto gather_kernel_fn, auto commit_kernel_fn, const auto* seq_lens_ptr, const auto* req_pool_indices_ptr) {
        const uint32_t tiles = static_cast<uint32_t>((total_occurrences + BLOCK_SIZE - 1) / BLOCK_SIZE);
        const dim3 gather_grid(static_cast<uint32_t>(bs), tiles);
        LaunchKernel(gather_grid, BLOCK_SIZE, device)(
            gather_kernel_fn,
            static_cast<const int32_t*>(top_k_tokens.data_ptr()),
            static_cast<int32_t*>(device_buffer_tokens.data_ptr()),
            static_cast<const int64_t*>(host_cache_locs.data_ptr()),
            static_cast<int32_t*>(device_buffer_locs.data_ptr()),
            host_cache_k.data_ptr(),
            device_buffer_k.data_ptr(),
            static_cast<int32_t*>(top_k_device_locs.data_ptr()),
            req_pool_indices_ptr,
            seq_lens_ptr,
            cache_state,
            miss_workspace,
            static_cast<const int32_t*>(num_real_reqs.data_ptr()),
            buffer_stride_0,
            host_stride,
            top_k_tokens_stride,
            top_k_device_locs_stride,
            page_size);

        const dim3 commit_grid(static_cast<uint32_t>(bs));
        LaunchKernel(commit_grid, 512, device)
            .enable_pdl(use_pdl)(
                commit_kernel_fn,
                static_cast<const int32_t*>(top_k_tokens.data_ptr()),
                static_cast<int32_t*>(top_k_device_locs.data_ptr()),
                static_cast<int32_t*>(device_buffer_tokens.data_ptr()),
                static_cast<int32_t*>(device_buffer_locs.data_ptr()),
                static_cast<const int64_t*>(host_cache_locs.data_ptr()),
                host_cache_k.data_ptr(),
                device_buffer_k.data_ptr(),
                req_pool_indices_ptr,
                cache_state,
                miss_workspace,
                static_cast<const int32_t*>(num_real_reqs.data_ptr()),
                top_k_tokens_stride,
                top_k_device_locs_stride,
                buffer_stride_0,
                host_stride);
      };

  const auto seq_dtype = seq_lens.dtype();
  const auto rpi_dtype = req_pool_indices.dtype();
  const bool seq_is_i64 = seq_dtype.code == kDLInt && seq_dtype.bits == 64;
  const bool rpi_is_i64 = rpi_dtype.code == kDLInt && rpi_dtype.bits == 64;
  if (seq_is_i64 && rpi_is_i64) {
    launch(
        load_cache_to_device_buffer_mtp_gather_kernel<
            BLOCK_SIZE,
            NUM_TOP_K,
            HOT_BUFFER_SIZE,
            ITEM_SIZE_BYTES,
            NUM_STEPS,
            int64_t,
            int64_t>,
        load_cache_to_device_buffer_mtp_commit_kernel<NUM_TOP_K, HOT_BUFFER_SIZE, ITEM_SIZE_BYTES, NUM_STEPS, int64_t>,
        static_cast<const int64_t*>(seq_lens.data_ptr()),
        static_cast<const int64_t*>(req_pool_indices.data_ptr()));
  } else if (seq_is_i64) {
    launch(
        load_cache_to_device_buffer_mtp_gather_kernel<
            BLOCK_SIZE,
            NUM_TOP_K,
            HOT_BUFFER_SIZE,
            ITEM_SIZE_BYTES,
            NUM_STEPS,
            int64_t,
            int32_t>,
        load_cache_to_device_buffer_mtp_commit_kernel<NUM_TOP_K, HOT_BUFFER_SIZE, ITEM_SIZE_BYTES, NUM_STEPS, int32_t>,
        static_cast<const int64_t*>(seq_lens.data_ptr()),
        static_cast<const int32_t*>(req_pool_indices.data_ptr()));
  } else if (rpi_is_i64) {
    launch(
        load_cache_to_device_buffer_mtp_gather_kernel<
            BLOCK_SIZE,
            NUM_TOP_K,
            HOT_BUFFER_SIZE,
            ITEM_SIZE_BYTES,
            NUM_STEPS,
            int32_t,
            int64_t>,
        load_cache_to_device_buffer_mtp_commit_kernel<NUM_TOP_K, HOT_BUFFER_SIZE, ITEM_SIZE_BYTES, NUM_STEPS, int64_t>,
        static_cast<const int32_t*>(seq_lens.data_ptr()),
        static_cast<const int64_t*>(req_pool_indices.data_ptr()));
  } else {
    launch(
        load_cache_to_device_buffer_mtp_gather_kernel<
            BLOCK_SIZE,
            NUM_TOP_K,
            HOT_BUFFER_SIZE,
            ITEM_SIZE_BYTES,
            NUM_STEPS,
            int32_t,
            int32_t>,
        load_cache_to_device_buffer_mtp_commit_kernel<NUM_TOP_K, HOT_BUFFER_SIZE, ITEM_SIZE_BYTES, NUM_STEPS, int32_t>,
        static_cast<const int32_t*>(seq_lens.data_ptr()),
        static_cast<const int32_t*>(req_pool_indices.data_ptr()));
  }
}

}  // namespace
