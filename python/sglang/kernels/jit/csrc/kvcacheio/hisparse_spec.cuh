#pragma once

// Multi-step speculative HiSparse cache-management kernels.

#include <sgl_kernel/tensor.h>  // TensorMatcher and symbolic tensor validation
#include <sgl_kernel/utils.h>   // RuntimeCheck and host utilities

#include <sgl_kernel/utils.cuh>  // LaunchKernel and PDL helpers

#include "hisparse_transfer.cuh"

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <stdint.h>

namespace sglang {

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

struct SpecCacheState {
  int64_t* __restrict__ hash_primary;
  int64_t* __restrict__ hash_secondary;
  int32_t* __restrict__ ring_state;
  int32_t* __restrict__ ref_epochs;
  int64_t hash_stride;
  int64_t hash_size;
  int64_t ref_epoch_stride;
};

struct SpecMissWorkspace {
  int32_t* __restrict__ locs;
  int32_t* __restrict__ metadata;
  int32_t* __restrict__ counters;
  int64_t loc_stride;
  // A sliced tail view can have a row stride larger than its usable width.
  int64_t loc_capacity;
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

  static_assert(HOT_BUFFER_SIZE > 1, "speculative hot buffer must contain at least two slots.");
  static_assert(CURSOR_BITS <= 15, "speculative CLOCK cursor requires hot_buffer_size <= 32768.");

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
  static constexpr int64_t TOKEN_CAPACITY = int64_t{1} << 31;

  // Keep the token in the low word so the common lookup path can compare it
  // without a 64-bit shift; the slot is decoded only after a token match.
  __device__ static int64_t pack(int32_t token, int32_t slot) {
    return static_cast<int64_t>(
        (static_cast<uint64_t>(static_cast<uint32_t>(slot)) << 32) | static_cast<uint32_t>(token));
  }

  __device__ static int32_t token(int64_t packed) {
    return static_cast<int32_t>(static_cast<uint32_t>(packed));
  }

  __device__ static int32_t slot(int64_t packed) {
    return static_cast<int32_t>(static_cast<uint64_t>(packed) >> 32);
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
  const int64_t primary = keys[ring_hash_slot(token, hash_size)];
  if (primary >= 0 && Entry::token(primary) == token) {
    const int32_t slot = Entry::slot(primary);
    if (slot < HOT_BUFFER_SIZE && req_device_buffer_tokens[slot] == token) {
      return slot;
    }
  }
  const int64_t secondary = vals[ring_hash_slot_secondary(token, hash_size)];
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

__device__ __forceinline__ int first_set_lane(BallotMask mask) {
#ifdef USE_ROCM
  return __ffsll(mask) - 1;
#else
  return __ffs(mask) - 1;
#endif
}

template <int HOT_BUFFER_SIZE>
__device__ __forceinline__ bool try_get_active_tail_device_loc(
    int32_t token_idx,
    int64_t seq_len,
    const int32_t* __restrict__ req_device_buffer_tokens,
    const int32_t* __restrict__ req_device_buffer_locs,
    int64_t active_tail_slots,
    int32_t* __restrict__ out_loc) {
  int64_t slot = -1;
  if (static_cast<int64_t>(token_idx) >= seq_len - active_tail_slots) {
    if (active_tail_slots == 4) {
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
      for (int64_t candidate_slot = HOT_BUFFER_SIZE;
           candidate_slot < HOT_BUFFER_SIZE + active_tail_slots;
           candidate_slot++) {
        if (req_device_buffer_tokens[candidate_slot] == token_idx) {
          slot = candidate_slot;
          break;
        }
      }
    }
  }

  if (slot < HOT_BUFFER_SIZE || slot >= HOT_BUFFER_SIZE + active_tail_slots) {
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
    int NUM_STEPS,
    bool RecordMissPlan,
    typename KVTransferPolicy>
__global__ void load_cache_to_device_buffer_spec_gather_kernel(
    const int32_t* __restrict__ top_k_tokens,
    int32_t* __restrict__ device_buffer_tokens,
    const int64_t* __restrict__ host_cache_locs,
    int32_t* __restrict__ device_buffer_locs,
    const void* __restrict__ host_cache_k,
    void* __restrict__ device_buffer_k,
    int32_t* __restrict__ top_k_device_locs,
    const int64_t* __restrict__ req_pool_indices,
    const int32_t* __restrict__ seq_lens,
    SpecCacheState cache_state,
    SpecMissWorkspace miss_workspace,
    const int32_t* __restrict__ num_real_reqs,
    int64_t* __restrict__ miss_src_out,
    int32_t* __restrict__ miss_dst_out,
    int32_t* __restrict__ miss_count_out,
    int64_t buffer_stride_0,
    int64_t host_stride,
    int64_t top_k_tokens_stride,
    int64_t top_k_device_locs_stride,
    int64_t plan_stride,
    int64_t active_tail_slots) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  constexpr int64_t total_occurrences = NUM_STEPS * NUM_TOP_K;
  int32_t* req_top_k_device_locs = top_k_device_locs + bid * top_k_device_locs_stride;
  if (bid >= num_real_reqs[0]) {
    if constexpr (RecordMissPlan) {
      if (blockIdx.y == 0 && tid == 0) {
        miss_count_out[bid] = 0;
      }
    }
    for (int64_t i = tid; i < total_occurrences; i += BLOCK_SIZE)
      req_top_k_device_locs[i] = 0;
    return;
  }

  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
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

  int32_t token = -1;
  int32_t loc = -1;
  int32_t cache_slot = -1;
  bool copy_owner = false;
  int32_t miss_idx = -1;
  int64_t src_loc = -1;
  bool needs_cache_lookup = false;
  if (occ < total_occurrences) {
    const int64_t step = occ / NUM_TOP_K;
    token = req_top_k_tokens[occ];
    const int64_t seq_len = static_cast<int64_t>(seq_lens[bid * NUM_STEPS + step]);
    if (token >= 0 && token < seq_len) {
      int32_t direct_loc = -1;
      if (try_get_active_tail_device_loc<HOT_BUFFER_SIZE>(
              token,
              seq_len,
              req_device_buffer_tokens,
              req_device_buffer_locs,
              active_tail_slots,
              &direct_loc)) {
        loc = direct_loc;
      } else {
        needs_cache_lookup = true;
      }
    }
  }

  // The flattened occurrence mapping places the same top-k position from
  // different speculative steps in one warp. Stable hits only need one Ring lookup.
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
              if constexpr (RecordMissPlan) {
                miss_idx = unique_idx;
              }
              req_compact_hash_positions[unique_idx] = hash_pos;
              __threadfence();
              atomicExch(req_scratch_indices + hash_pos, epoch_bits | static_cast<uint32_t>(unique_idx));
              if (unique_idx < miss_workspace.loc_capacity) {
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
            if (unique_idx < miss_workspace.loc_capacity) {
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
      if constexpr (RecordMissPlan) {
        const int32_t copy_miss_idx = __shfl_sync(FULL_WARP_MASK, miss_idx, owner_lane);
        if (lane_id == owner_lane) {
          miss_src_out[bid * plan_stride + copy_miss_idx] = copy_src_loc;
          miss_dst_out[bid * plan_stride + copy_miss_idx] = copy_loc;
        }
      }
      KVTransferPolicy::copy_warp(
          lane_id, host_cache_k, device_buffer_k, copy_src_loc, static_cast<int64_t>(copy_loc));
    }
    copy_mask &= ~(static_cast<BallotMask>(1) << owner_lane);
  }
}

template <
    int NUM_TOP_K,
    int HOT_BUFFER_SIZE,
    int NUM_STEPS,
    bool RecordMissPlan,
    typename KVTransferPolicy,
    bool USE_PDL>
__global__ void load_cache_to_device_buffer_spec_commit_kernel(
    const int32_t* __restrict__ top_k_tokens,
    int32_t* __restrict__ top_k_device_locs,
    int32_t* __restrict__ device_buffer_tokens,
    int32_t* __restrict__ device_buffer_locs,
    const int64_t* __restrict__ host_cache_locs,
    const void* __restrict__ host_cache_k,
    void* __restrict__ device_buffer_k,
    const int64_t* __restrict__ req_pool_indices,
    SpecCacheState cache_state,
    SpecMissWorkspace miss_workspace,
    const int32_t* __restrict__ num_real_reqs,
    int64_t* __restrict__ miss_src_out,
    int32_t* __restrict__ miss_dst_out,
    int32_t* __restrict__ miss_count_out,
    int64_t top_k_tokens_stride,
    int64_t top_k_device_locs_stride,
    int64_t buffer_stride_0,
    int64_t host_stride,
    int64_t plan_stride) {
  device::PDLWaitPrimary<USE_PDL>();
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
  const bool scratch_overflow = miss_count > miss_workspace.loc_capacity;
  const int64_t compact_iterations = (miss_count + blockDim.x - 1) / blockDim.x;
  const bool lock_free_single_pass = compact_iterations == 1;

  // The union fast path can preserve every token needed by this speculative group.
  // Its victim selection below partitions the ring by miss ordinal, avoiding
  // a full hot-cache scan while keeping all current union hits protected.
  __shared__ int32_t s_use_union_clock;
  __shared__ int32_t s_approx_admission_budget;
  __shared__ int32_t s_hash_degraded;
  if (threadIdx.x == 0) {
    s_use_union_clock = miss_count + union_hit_count <= HOT_BUFFER_SIZE;
    const int32_t mandatory_direct_misses =
        scratch_overflow ? miss_count - static_cast<int32_t>(miss_workspace.loc_capacity) : 0;
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
      if (unique_idx >= 0 && unique_idx < miss_workspace.loc_capacity) {
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
        direct_overflow = scratch_overflow && unique_idx >= miss_workspace.loc_capacity;
        const bool approximate_admission = !s_use_union_clock && (compact_entry & COMPACT_APPROX_ADMIT_FLAG) != 0;
        rotate_compact = unique_idx < miss_workspace.loc_capacity && (s_use_union_clock || approximate_admission);

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
          const int32_t scratch_capacity = static_cast<int32_t>(miss_workspace.loc_capacity);
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
        if constexpr (RecordMissPlan) {
          const int32_t miss_idx = __shfl_sync(FULL_WARP_MASK, unique_idx, owner_lane);
          if (lane_id == owner_lane) {
            miss_src_out[bid * plan_stride + miss_idx] = src_loc;
            miss_dst_out[bid * plan_stride + miss_idx] = dst_loc;
          }
        }
        KVTransferPolicy::copy_warp(
            lane_id, host_cache_k, device_buffer_k, src_loc, static_cast<int64_t>(dst_loc));
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
    if constexpr (RecordMissPlan) {
      miss_count_out[bid] = miss_count;
    }
    *req_work_count = 0;
    *req_union_hit_count = 0;
    *req_scratch_generation =
        static_cast<int32_t>(static_cast<uint32_t>(scratch_epoch) | (s_hash_degraded != 0 ? HASH_DEGRADED_FLAG : 0));
    const int32_t next_cursor =
        miss_count > 0 ? (clock_cursor + min(miss_count, HOT_BUFFER_SIZE)) % HOT_BUFFER_SIZE : clock_cursor;
    cache_state.ring_state[rid] = RingState::pack(cache_epoch, next_cursor);
  }
}

template <
    int BLOCK_SIZE,
    int NUM_TOP_K,
    int HOT_BUFFER_SIZE,
    int NUM_STEPS,
    bool RecordMissPlan,
    typename KVTransferPolicy,
    bool USE_PDL>
void load_cache_to_device_buffer_spec(
    tvm::ffi::TensorView top_k_tokens,
    tvm::ffi::TensorView device_buffer_tokens,
    tvm::ffi::TensorView host_cache_locs,
    tvm::ffi::TensorView device_buffer_locs,
    tvm::ffi::TensorView host_cache_k,
    tvm::ffi::TensorView device_buffer_k,
    tvm::ffi::TensorView top_k_device_locs,
    tvm::ffi::TensorView req_pool_indices,
    tvm::ffi::TensorView seq_lens,
    tvm::ffi::TensorView cache_index,
    tvm::ffi::TensorView cache_policy,
    tvm::ffi::TensorView scratch_locs,
    tvm::ffi::TensorView scratch_state,
    tvm::ffi::TensorView num_real_reqs,
    int64_t active_tail_slots,
    tvm::ffi::TensorView miss_src_out,
    tvm::ffi::TensorView miss_dst_out,
    tvm::ffi::TensorView miss_count_out) {
  using namespace host;

  static_assert(NUM_STEPS > 0, "HiSparse speculative swap requires at least one step.");
  static_assert(NUM_TOP_K > 0, "HiSparse speculative swap requires a positive top_k.");
  static_assert(NUM_STEPS * NUM_TOP_K <= 8192, "HiSparse speculative swap supports at most 8192 occurrences.");

  const int64_t bs = top_k_tokens.shape()[0];
  constexpr int64_t total_occurrences = NUM_STEPS * NUM_TOP_K;
  RuntimeCheck(top_k_tokens.ndim() == 3, "speculative top_k_tokens must have shape [batch, steps, top_k].");
  RuntimeCheck(top_k_device_locs.ndim() == 3, "speculative output must have shape [batch, steps, top_k].");
  RuntimeCheck(top_k_tokens.shape()[1] == NUM_STEPS, "top_k_tokens step dimension mismatch.");
  RuntimeCheck(top_k_tokens.shape()[2] == NUM_TOP_K, "top_k_tokens top-k dimension mismatch.");
  RuntimeCheck(
      cache_index.ndim() == 3 && cache_index.shape()[1] == 2,
      "speculative cache_index must have shape [num_requests, 2, hash_size].");
  const int64_t ring_hash_size = cache_index.shape()[2];
  RuntimeCheck(
      ring_hash_size > 0 && (ring_hash_size & (ring_hash_size - 1)) == 0, "ring hash capacity must be a power of two.");
  RuntimeCheck(scratch_locs.ndim() == 2, "speculative scratch_locs must have shape [num_requests, capacity].");
  const int64_t num_request_slots = scratch_locs.shape()[0];
  const int64_t scratch_capacity = scratch_locs.shape()[1];
  RuntimeCheck(
      cache_index.shape()[0] >= num_request_slots,
      "speculative cache_index request capacity is smaller than scratch_locs.");
  RuntimeCheck(
      cache_policy.ndim() == 2 && cache_policy.shape()[0] >= num_request_slots + 1 &&
          cache_policy.shape()[1] >= HOT_BUFFER_SIZE,
      "speculative cache_policy must contain one CLOCK control row and one reference-epoch row per request.");
  RuntimeCheck(
      scratch_state.ndim() == 2 && scratch_state.shape()[0] >= num_request_slots + 1 &&
          scratch_state.shape()[1] >= 4 * num_request_slots && scratch_state.shape()[1] >= 5 * total_occurrences,
      "speculative scratch_state must contain one counter row and one miss-metadata row per request.");
  RuntimeCheck(scratch_state.strides()[0] % 2 == 0, "speculative scratch metadata stride must be 64-bit aligned.");

  const int64_t host_stride = host_cache_locs.shape()[1];
  RuntimeCheck(
      host_stride <= PackedRingEntry<HOT_BUFFER_SIZE>::TOKEN_CAPACITY,
      "speculative packed ring metadata supports sequence lengths up to ",
      PackedRingEntry<HOT_BUFFER_SIZE>::TOKEN_CAPACITY,
      ", got ",
      host_stride);
  const int64_t buffer_stride_0 = device_buffer_tokens.strides()[0];
  const int64_t top_k_tokens_stride = top_k_tokens.strides()[0];
  const int64_t top_k_device_locs_stride = top_k_device_locs.strides()[0];
  const int64_t cache_index_stride_0 = cache_index.strides()[0];
  const int64_t cache_index_stride_1 = cache_index.strides()[1];
  const int64_t cache_policy_stride_0 = cache_policy.strides()[0];
  const int64_t scratch_stride_0 = scratch_locs.strides()[0];
  const int64_t scratch_state_stride_0 = scratch_state.strides()[0];
  int64_t* const miss_src_ptr = RecordMissPlan ? static_cast<int64_t*>(miss_src_out.data_ptr()) : nullptr;
  int32_t* const miss_dst_ptr = RecordMissPlan ? static_cast<int32_t*>(miss_dst_out.data_ptr()) : nullptr;
  int32_t* const miss_count_ptr = RecordMissPlan ? static_cast<int32_t*>(miss_count_out.data_ptr()) : nullptr;
  const int64_t plan_stride = RecordMissPlan ? miss_src_out.strides()[0] : 0;
  if constexpr (RecordMissPlan) {
    RuntimeCheck(
        miss_src_out.ndim() == 2 && miss_src_out.shape()[0] >= bs && miss_src_out.shape()[1] >= total_occurrences,
        "speculative miss_src must have shape [batch, >= steps * top_k].");
    RuntimeCheck(
        miss_dst_out.ndim() == 2 && miss_dst_out.shape()[0] >= bs && miss_dst_out.shape()[1] >= total_occurrences,
        "speculative miss_dst must have shape [batch, >= steps * top_k].");
    RuntimeCheck(
        miss_count_out.ndim() == 1 && miss_count_out.shape()[0] >= bs,
        "speculative miss_count must have shape [batch].");
    RuntimeCheck(miss_dst_out.strides()[0] == plan_stride, "speculative miss_src/miss_dst row strides differ.");
  }

  // Preserve the device kernels' independent restrict-qualified views while
  // exposing only four packed tensors through FFI. Row 0 is the control
  // plane; request-owned rows begin at row 1.
  auto* cache_index_ptr = static_cast<int64_t*>(cache_index.data_ptr());
  auto* cache_policy_ptr = static_cast<int32_t*>(cache_policy.data_ptr());
  auto* scratch_state_ptr = static_cast<int32_t*>(scratch_state.data_ptr());
  const SpecCacheState cache_state{
      cache_index_ptr,
      cache_index_ptr + cache_index_stride_1,
      cache_policy_ptr,
      cache_policy_ptr + cache_policy_stride_0,
      cache_index_stride_0,
      ring_hash_size,
      cache_policy_stride_0};
  const SpecMissWorkspace miss_workspace{
      static_cast<int32_t*>(scratch_locs.data_ptr()),
      scratch_state_ptr + scratch_state_stride_0,
      scratch_state_ptr,
      scratch_stride_0,
      scratch_capacity,
      scratch_state_stride_0,
      num_request_slots};
  auto cuda_device = SymbolicDevice{};
  cuda_device.set_options<kDLCUDA>();
  TensorMatcher({bs}).with_dtype<int64_t>().with_device(cuda_device).verify(req_pool_indices);
  TensorMatcher({bs * NUM_STEPS}).with_dtype<int32_t>().with_device(cuda_device).verify(seq_lens);
  const auto device = cuda_device.unwrap();

  const uint32_t tiles = static_cast<uint32_t>((total_occurrences + BLOCK_SIZE - 1) / BLOCK_SIZE);
  LaunchKernel(dim3(static_cast<uint32_t>(bs), tiles), BLOCK_SIZE, device)(
      load_cache_to_device_buffer_spec_gather_kernel<
          BLOCK_SIZE,
          NUM_TOP_K,
          HOT_BUFFER_SIZE,
          NUM_STEPS,
          RecordMissPlan,
          KVTransferPolicy>,
      static_cast<const int32_t*>(top_k_tokens.data_ptr()),
      static_cast<int32_t*>(device_buffer_tokens.data_ptr()),
      static_cast<const int64_t*>(host_cache_locs.data_ptr()),
      static_cast<int32_t*>(device_buffer_locs.data_ptr()),
      host_cache_k.data_ptr(),
      device_buffer_k.data_ptr(),
      static_cast<int32_t*>(top_k_device_locs.data_ptr()),
      static_cast<const int64_t*>(req_pool_indices.data_ptr()),
      static_cast<const int32_t*>(seq_lens.data_ptr()),
      cache_state,
      miss_workspace,
      static_cast<const int32_t*>(num_real_reqs.data_ptr()),
      miss_src_ptr,
      miss_dst_ptr,
      miss_count_ptr,
      buffer_stride_0,
      host_stride,
      top_k_tokens_stride,
      top_k_device_locs_stride,
      plan_stride,
      active_tail_slots);

  LaunchKernel(dim3(static_cast<uint32_t>(bs)), 512, device)
      .enable_pdl(USE_PDL)(
          load_cache_to_device_buffer_spec_commit_kernel<
              NUM_TOP_K,
              HOT_BUFFER_SIZE,
              NUM_STEPS,
              RecordMissPlan,
              KVTransferPolicy,
              USE_PDL>,
          static_cast<const int32_t*>(top_k_tokens.data_ptr()),
          static_cast<int32_t*>(top_k_device_locs.data_ptr()),
          static_cast<int32_t*>(device_buffer_tokens.data_ptr()),
          static_cast<int32_t*>(device_buffer_locs.data_ptr()),
          static_cast<const int64_t*>(host_cache_locs.data_ptr()),
          host_cache_k.data_ptr(),
          device_buffer_k.data_ptr(),
          static_cast<const int64_t*>(req_pool_indices.data_ptr()),
          cache_state,
          miss_workspace,
          static_cast<const int32_t*>(num_real_reqs.data_ptr()),
          miss_src_ptr,
          miss_dst_ptr,
          miss_count_ptr,
          top_k_tokens_stride,
          top_k_device_locs_stride,
          buffer_stride_0,
          host_stride,
          plan_stride);
}

template <int HOT_BUFFER_SIZE>
__global__ void initialize_hisparse_spec_state_kernel(
    const int32_t* __restrict__ device_buffer_tokens,
    const int64_t* __restrict__ req_pool_indices,
    int64_t* __restrict__ cache_index,
    int32_t* __restrict__ cache_policy,
    int32_t* __restrict__ scratch_state,
    int64_t buffer_stride,
    int64_t hash_row_stride,
    int64_t hash_bank_stride,
    int64_t hash_size,
    int64_t policy_row_stride,
    int64_t state_row_stride,
    int64_t num_request_slots) {
  const int64_t rid = req_pool_indices[blockIdx.x];
  int64_t* primary = cache_index + rid * hash_row_stride;
  int64_t* secondary = primary + hash_bank_stride;
  int32_t* ring_state = cache_policy;
  int32_t* ref_epochs = cache_policy + (rid + 1) * policy_row_stride;
  int32_t* counters = scratch_state;
  int32_t* metadata = scratch_state + (rid + 1) * state_row_stride;

  for (int64_t i = threadIdx.x; i < hash_size; i += blockDim.x) {
    primary[i] = -1;
    secondary[i] = -1;
  }
  for (int64_t i = threadIdx.x; i < HOT_BUFFER_SIZE; i += blockDim.x) {
    ref_epochs[i] = 0;
  }
  for (int64_t i = threadIdx.x; i < state_row_stride; i += blockDim.x) {
    metadata[i] = 0;
  }
  if (threadIdx.x == 0) {
    ring_state[rid] = 0;
    counters[rid] = 0;
    counters[num_request_slots + rid] = 0;
    counters[2 * num_request_slots + rid] = 0;
    counters[3 * num_request_slots + rid] = 0;
  }
  __syncthreads();

  const int32_t* tokens = device_buffer_tokens + rid * buffer_stride;
  for (int32_t slot = threadIdx.x; slot < HOT_BUFFER_SIZE; slot += blockDim.x) {
    const int32_t token = tokens[slot];
    if (token >= 0 &&
        ring_hash_insert_atomic<HOT_BUFFER_SIZE>(primary, secondary, hash_size, token, slot) < 0) {
      atomicOr(counters + 3 * num_request_slots + rid, static_cast<int32_t>(HASH_DEGRADED_FLAG));
    }
  }
}

template <int HOT_BUFFER_SIZE>
void initialize_hisparse_spec_state(
    tvm::ffi::TensorView device_buffer_tokens,
    tvm::ffi::TensorView req_pool_indices,
    tvm::ffi::TensorView cache_index,
    tvm::ffi::TensorView cache_policy,
    tvm::ffi::TensorView scratch_state) {
  using namespace host;

  const int64_t batch_size = req_pool_indices.shape()[0];
  const int64_t num_request_slots = cache_index.shape()[0];
  const int64_t hash_size = cache_index.shape()[2];
  auto cuda_device = SymbolicDevice{};
  cuda_device.set_options<kDLCUDA>();
  TensorMatcher({batch_size}).with_dtype<int64_t>().with_device(cuda_device).verify(req_pool_indices);
  const auto device = cuda_device.unwrap();

  LaunchKernel(static_cast<uint32_t>(batch_size), 256, device)(
      initialize_hisparse_spec_state_kernel<HOT_BUFFER_SIZE>,
      static_cast<const int32_t*>(device_buffer_tokens.data_ptr()),
      static_cast<const int64_t*>(req_pool_indices.data_ptr()),
      static_cast<int64_t*>(cache_index.data_ptr()),
      static_cast<int32_t*>(cache_policy.data_ptr()),
      static_cast<int32_t*>(scratch_state.data_ptr()),
      device_buffer_tokens.strides()[0],
      cache_index.strides()[0],
      cache_index.strides()[1],
      hash_size,
      cache_policy.strides()[0],
      scratch_state.strides()[0],
      num_request_slots);
}

template <int VERIFY_WIDTH, int COMPRESS_RATIO, int HOT_BUFFER_SIZE, int SPECULATIVE_SLOTS>
__global__ void prepare_hisparse_spec_verify_kernel(
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ prefix_lens,
    const int64_t* __restrict__ hisparse_out_locs,
    const int64_t* __restrict__ req_to_device_buffer,
    int32_t* __restrict__ device_buffer_tokens,
    const int32_t* __restrict__ num_real_reqs,
    int64_t* __restrict__ logical_to_device_mapping,
    int64_t owner_stride,
    int64_t token_layer_stride,
    int64_t token_req_stride,
    int32_t num_layers) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  if (bid >= num_real_reqs[0]) {
    return;
  }
  const int64_t rid = req_pool_indices[bid];
  const int64_t prefix = prefix_lens[bid];
  const int32_t old_count = static_cast<int32_t>(prefix / COMPRESS_RATIO);
  const int32_t generated_count =
      static_cast<int32_t>((prefix + VERIFY_WIDTH) / COMPRESS_RATIO) - old_count;

  for (int i = tid; i < num_layers * SPECULATIVE_SLOTS; i += blockDim.x) {
    const int layer = i / SPECULATIVE_SLOTS;
    const int ordinal = i % SPECULATIVE_SLOTS;
    device_buffer_tokens[
        layer * token_layer_stride + rid * token_req_stride + HOT_BUFFER_SIZE + 1 + ordinal] = -1;
  }
  if (tid == 0) {
    for (int step = 0; step < VERIFY_WIDTH; ++step) {
      const int64_t seq_len = prefix + step + 1;
      if (seq_len % COMPRESS_RATIO != 0) {
        continue;
      }
      const int32_t position = static_cast<int32_t>(seq_len / COMPRESS_RATIO - 1);
      const int32_t ordinal = position - old_count;
      const int64_t logical_loc = hisparse_out_locs[bid * VERIFY_WIDTH + step];
      const int64_t device_loc =
          req_to_device_buffer[rid * owner_stride + HOT_BUFFER_SIZE + 1 + ordinal];
      logical_to_device_mapping[logical_loc] = device_loc;
    }
  }
  __syncthreads();

  for (int ordinal = 0; ordinal < generated_count; ++ordinal) {
    const int32_t position = old_count + ordinal;
    for (int layer = tid; layer < num_layers; layer += blockDim.x) {
      device_buffer_tokens[
          layer * token_layer_stride + rid * token_req_stride + HOT_BUFFER_SIZE + 1 + ordinal] = position;
    }
  }
}

template <int VERIFY_WIDTH, int COMPRESS_RATIO, int HOT_BUFFER_SIZE, int SPECULATIVE_SLOTS>
void prepare_hisparse_spec_verify(
    tvm::ffi::TensorView req_pool_indices,
    tvm::ffi::TensorView prefix_lens,
    tvm::ffi::TensorView hisparse_out_locs,
    tvm::ffi::TensorView req_to_device_buffer,
    tvm::ffi::TensorView device_buffer_tokens,
    tvm::ffi::TensorView num_real_reqs,
    tvm::ffi::TensorView logical_to_device_mapping) {
  using namespace host;

  auto batch_size = SymbolicSize{"batch_size"};
  auto num_out_locs = SymbolicSize{"num_out_locs"};
  auto num_layers = SymbolicSize{"num_layers"};
  auto num_req_slots = SymbolicSize{"num_req_slots"};
  auto buffer_slots = SymbolicSize{"buffer_slots"};
  auto mapping_size = SymbolicSize{"mapping_size"};
  auto cuda_device = SymbolicDevice{};
  cuda_device.set_options<kDLCUDA>();
  TensorMatcher({batch_size}).with_dtype<int64_t>().with_device(cuda_device).verify(req_pool_indices).verify(
      prefix_lens);
  TensorMatcher({num_out_locs})
      .with_dtype<int64_t>()
      .with_device(cuda_device)
      .verify(hisparse_out_locs);
  RuntimeCheck(
      num_out_locs.unwrap() == batch_size.unwrap() * VERIFY_WIDTH,
      "HiSparse speculative output locations must contain batch_size * verify_width entries.");
  TensorMatcher({num_req_slots, buffer_slots})
      .with_dtype<int64_t>()
      .with_device(cuda_device)
      .verify(req_to_device_buffer);
  TensorMatcher({num_layers, num_req_slots, buffer_slots})
      .with_dtype<int32_t>()
      .with_device(cuda_device)
      .verify(device_buffer_tokens);
  TensorMatcher({1}).with_dtype<int32_t>().with_device(cuda_device).verify(num_real_reqs);
  TensorMatcher({mapping_size})
      .with_dtype<int64_t>()
      .with_device(cuda_device)
      .verify(logical_to_device_mapping);

  LaunchKernel(batch_size.unwrap(), 256, cuda_device.unwrap())(
      prepare_hisparse_spec_verify_kernel<
          VERIFY_WIDTH, COMPRESS_RATIO, HOT_BUFFER_SIZE, SPECULATIVE_SLOTS>,
      static_cast<const int64_t*>(req_pool_indices.data_ptr()),
      static_cast<const int64_t*>(prefix_lens.data_ptr()),
      static_cast<const int64_t*>(hisparse_out_locs.data_ptr()),
      static_cast<const int64_t*>(req_to_device_buffer.data_ptr()),
      static_cast<int32_t*>(device_buffer_tokens.data_ptr()),
      static_cast<const int32_t*>(num_real_reqs.data_ptr()),
      static_cast<int64_t*>(logical_to_device_mapping.data_ptr()),
      req_to_device_buffer.strides()[0],
      device_buffer_tokens.strides()[0],
      device_buffer_tokens.strides()[1],
      static_cast<int32_t>(num_layers.unwrap()));
}

constexpr int HISPARSE_SPEC_TRANSFER_BLOCK_SIZE = 256;
constexpr int HISPARSE_SPEC_TRANSFER_WARPS = HISPARSE_SPEC_TRANSFER_BLOCK_SIZE / WARP_SIZE;

template <int COMPRESS_RATIO, int HOT_BUFFER_SIZE, int SPECULATIVE_SLOTS, typename KVTransferPolicy>
__global__ __launch_bounds__(HISPARSE_SPEC_TRANSFER_BLOCK_SIZE, 1) void backup_accepted_hisparse_spec_kernel(
    void** device_caches,
    void** host_caches,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ prefix_lens,
    const int32_t* __restrict__ commit_lens,
    const int64_t* __restrict__ req_to_host_pool,
    const int32_t* __restrict__ device_buffer_locs,
    int64_t host_stride,
    int64_t loc_layer_stride,
    int64_t loc_req_stride,
    int32_t batch_size,
    int32_t num_layers) {
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int total_warps = gridDim.x * HISPARSE_SPEC_TRANSFER_WARPS;
  const int num_items = batch_size * SPECULATIVE_SLOTS;

  for (int item = global_tid / WARP_SIZE; item < num_items; item += total_warps) {
    const int bid = item / SPECULATIVE_SLOTS;
    const int ordinal = item % SPECULATIVE_SLOTS;
    const int64_t rid = req_pool_indices[bid];
    const int64_t prefix = prefix_lens[bid];
    const int32_t old_count = static_cast<int32_t>(prefix / COMPRESS_RATIO);
    const int32_t accepted =
        static_cast<int32_t>((prefix + commit_lens[bid]) / COMPRESS_RATIO) - old_count;
    if (ordinal >= accepted) {
      continue;
    }

    const int64_t dst = req_to_host_pool[rid * host_stride + old_count + ordinal];
    for (int layer = 0; layer < num_layers; ++layer) {
      const int64_t src = device_buffer_locs[
          layer * loc_layer_stride + rid * loc_req_stride + HOT_BUFFER_SIZE + 1 + ordinal];
      KVTransferPolicy::copy_warp(lane_id, device_caches[layer], host_caches[layer], src, dst);
    }
  }
}

template <int COMPRESS_RATIO, int HOT_BUFFER_SIZE, int SPECULATIVE_SLOTS, typename KVTransferPolicy>
__global__ __launch_bounds__(HISPARSE_SPEC_TRANSFER_BLOCK_SIZE, 1) void promote_accepted_hisparse_spec_kernel(
    void** device_caches,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ prefix_lens,
    const int32_t* __restrict__ commit_lens,
    const int32_t* __restrict__ device_buffer_locs,
    int64_t loc_layer_stride,
    int64_t loc_req_stride,
    int32_t batch_size,
    int32_t num_layers) {
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int total_warps = gridDim.x * HISPARSE_SPEC_TRANSFER_WARPS;
  const int num_items = batch_size * SPECULATIVE_SLOTS;

  for (int item = global_tid / WARP_SIZE; item < num_items; item += total_warps) {
    const int bid = item / SPECULATIVE_SLOTS;
    const int ordinal = item % SPECULATIVE_SLOTS;
    const int64_t rid = req_pool_indices[bid];
    const int64_t prefix = prefix_lens[bid];
    const int32_t old_count = static_cast<int32_t>(prefix / COMPRESS_RATIO);
    const int32_t accepted =
        static_cast<int32_t>((prefix + commit_lens[bid]) / COMPRESS_RATIO) - old_count;
    if (ordinal >= accepted) {
      continue;
    }

    const int32_t position = old_count + ordinal;
    const bool promote_to_hot = position < HOT_BUFFER_SIZE;
    const bool promote_to_canonical = !promote_to_hot && ordinal == accepted - 1;
    if (!promote_to_hot && !promote_to_canonical) {
      continue;
    }

    // Before the sequence fills H, compressed position p owns H[p]. Once H is
    // full, only the newest accepted position stays device-resident in C; the
    // other accepted positions have already been persisted to host above.
    const int32_t dst_slot = promote_to_hot ? position : HOT_BUFFER_SIZE;
    for (int layer = 0; layer < num_layers; ++layer) {
      const int64_t loc_offset = layer * loc_layer_stride + rid * loc_req_stride;
      const int64_t src = device_buffer_locs[loc_offset + HOT_BUFFER_SIZE + 1 + ordinal];
      const int64_t dst = device_buffer_locs[loc_offset + dst_slot];
      KVTransferPolicy::copy_warp(lane_id, device_caches[layer], device_caches[layer], src, dst);
    }
  }
}

template <int COMPRESS_RATIO, int HOT_BUFFER_SIZE, int SPECULATIVE_SLOTS, typename KVTransferPolicy>
void transfer_hisparse_spec_finalize(
    tvm::ffi::TensorView device_ptrs,
    tvm::ffi::TensorView host_ptrs,
    tvm::ffi::TensorView req_pool_indices,
    tvm::ffi::TensorView prefix_lens,
    tvm::ffi::TensorView commit_lens,
    tvm::ffi::TensorView req_to_host_pool,
    tvm::ffi::TensorView device_buffer_locs) {
  using namespace host;

  auto num_layers = SymbolicSize{"num_layers"};
  auto batch_size = SymbolicSize{"batch_size"};
  auto num_req_slots = SymbolicSize{"num_req_slots"};
  auto host_slots = SymbolicSize{"host_slots"};
  auto device_slots = SymbolicSize{"device_slots"};
  auto cuda_device = SymbolicDevice{};
  cuda_device.set_options<kDLCUDA>();
  TensorMatcher({num_layers})
      .with_dtype<uint64_t>()
      .with_device(cuda_device)
      .verify(device_ptrs)
      .verify(host_ptrs);
  TensorMatcher({batch_size})
      .with_dtype<int64_t>()
      .with_device(cuda_device)
      .verify(req_pool_indices)
      .verify(prefix_lens);
  TensorMatcher({batch_size}).with_dtype<int32_t>().with_device(cuda_device).verify(commit_lens);
  TensorMatcher({num_req_slots, host_slots})
      .with_dtype<int64_t>()
      .with_device(cuda_device)
      .verify(req_to_host_pool);
  TensorMatcher({num_layers, num_req_slots, device_slots})
      .with_dtype<int32_t>()
      .with_device(cuda_device)
      .verify(device_buffer_locs);

  const int32_t batch = static_cast<int32_t>(batch_size.unwrap());
  const int32_t layers = static_cast<int32_t>(num_layers.unwrap());
  const int backup_items = batch * SPECULATIVE_SLOTS;
  LaunchKernel(
      (backup_items + HISPARSE_SPEC_TRANSFER_WARPS - 1) / HISPARSE_SPEC_TRANSFER_WARPS,
      HISPARSE_SPEC_TRANSFER_BLOCK_SIZE,
      cuda_device.unwrap())(
      backup_accepted_hisparse_spec_kernel<
          COMPRESS_RATIO, HOT_BUFFER_SIZE, SPECULATIVE_SLOTS, KVTransferPolicy>,
      static_cast<void**>(device_ptrs.data_ptr()),
      static_cast<void**>(host_ptrs.data_ptr()),
      static_cast<const int64_t*>(req_pool_indices.data_ptr()),
      static_cast<const int64_t*>(prefix_lens.data_ptr()),
      static_cast<const int32_t*>(commit_lens.data_ptr()),
      static_cast<const int64_t*>(req_to_host_pool.data_ptr()),
      static_cast<const int32_t*>(device_buffer_locs.data_ptr()),
      req_to_host_pool.strides()[0],
      device_buffer_locs.strides()[0],
      device_buffer_locs.strides()[1],
      batch,
      layers);
  LaunchKernel(
      (backup_items + HISPARSE_SPEC_TRANSFER_WARPS - 1) / HISPARSE_SPEC_TRANSFER_WARPS,
      HISPARSE_SPEC_TRANSFER_BLOCK_SIZE,
      cuda_device.unwrap())(
      promote_accepted_hisparse_spec_kernel<
          COMPRESS_RATIO, HOT_BUFFER_SIZE, SPECULATIVE_SLOTS, KVTransferPolicy>,
      static_cast<void**>(device_ptrs.data_ptr()),
      static_cast<const int64_t*>(req_pool_indices.data_ptr()),
      static_cast<const int64_t*>(prefix_lens.data_ptr()),
      static_cast<const int32_t*>(commit_lens.data_ptr()),
      static_cast<const int32_t*>(device_buffer_locs.data_ptr()),
      device_buffer_locs.strides()[0],
      device_buffer_locs.strides()[1],
      batch,
      layers);
}

template <int VERIFY_WIDTH, int COMPRESS_RATIO, int HOT_BUFFER_SIZE, int SPECULATIVE_SLOTS>
__global__ void complete_hisparse_spec_finalize_kernel(
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ prefix_lens,
    const int32_t* __restrict__ commit_lens,
    const int32_t* __restrict__ req_to_token,
    int32_t* __restrict__ device_buffer_tokens,
    int64_t* __restrict__ logical_to_device_mapping,
    int64_t req_to_token_stride,
    int64_t token_layer_stride,
    int64_t token_req_stride,
    int32_t num_layers) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t rid = req_pool_indices[bid];
  const int64_t prefix = prefix_lens[bid];
  const int32_t old_count = static_cast<int32_t>(prefix / COMPRESS_RATIO);
  const int32_t generated_count =
      static_cast<int32_t>((prefix + VERIFY_WIDTH) / COMPRESS_RATIO) - old_count;
  const int32_t accepted =
      static_cast<int32_t>((prefix + commit_lens[bid]) / COMPRESS_RATIO) - old_count;

  for (int i = tid; i < num_layers * SPECULATIVE_SLOTS; i += blockDim.x) {
    const int layer = i / SPECULATIVE_SLOTS;
    const int ordinal = i % SPECULATIVE_SLOTS;
    device_buffer_tokens[
        layer * token_layer_stride + rid * token_req_stride + HOT_BUFFER_SIZE + 1 + ordinal] = -1;
    const int32_t position = old_count + ordinal;
    if (ordinal < accepted && position < HOT_BUFFER_SIZE) {
      // hot_cache_lookup resolves this positional H entry through
      // device_buffer_tokens[position] == position, without consulting the
      // speculative hash state.
      device_buffer_tokens[layer * token_layer_stride + rid * token_req_stride + position] = position;
    }
  }
  if (accepted > 0) {
    const int32_t newest_position = old_count + accepted - 1;
    for (int layer = tid; layer < num_layers; layer += blockDim.x) {
      device_buffer_tokens[layer * token_layer_stride + rid * token_req_stride + HOT_BUFFER_SIZE] =
          newest_position >= HOT_BUFFER_SIZE ? newest_position : -1;
    }
  }
  __syncthreads();

  if (tid == 0) {
    for (int ordinal = 0; ordinal < generated_count; ++ordinal) {
      const int64_t position = old_count + ordinal;
      const int64_t boundary = (position + 1) * COMPRESS_RATIO - 1;
      const int64_t full_logical = req_to_token[rid * req_to_token_stride + boundary];
      // The mapping is a per-forward write route into E, not persistent
      // residency metadata. Invalidate every generated route before E reuse.
      logical_to_device_mapping[full_logical / COMPRESS_RATIO] = 0;
    }
  }
}

template <int VERIFY_WIDTH, int COMPRESS_RATIO, int HOT_BUFFER_SIZE, int SPECULATIVE_SLOTS>
void complete_hisparse_spec_finalize(
    tvm::ffi::TensorView req_pool_indices,
    tvm::ffi::TensorView prefix_lens,
    tvm::ffi::TensorView commit_lens,
    tvm::ffi::TensorView req_to_token,
    tvm::ffi::TensorView device_buffer_tokens,
    tvm::ffi::TensorView logical_to_device_mapping) {
  using namespace host;

  auto batch_size = SymbolicSize{"batch_size"};
  auto num_layers = SymbolicSize{"num_layers"};
  auto num_req_slots = SymbolicSize{"num_req_slots"};
  auto context_size = SymbolicSize{"context_size"};
  auto buffer_slots = SymbolicSize{"buffer_slots"};
  auto mapping_size = SymbolicSize{"mapping_size"};
  auto cuda_device = SymbolicDevice{};
  cuda_device.set_options<kDLCUDA>();
  TensorMatcher({batch_size})
      .with_dtype<int64_t>()
      .with_device(cuda_device)
      .verify(req_pool_indices)
      .verify(prefix_lens);
  TensorMatcher({batch_size}).with_dtype<int32_t>().with_device(cuda_device).verify(commit_lens);
  TensorMatcher({num_req_slots, context_size})
      .with_dtype<int32_t>()
      .with_device(cuda_device)
      .verify(req_to_token);
  TensorMatcher({num_layers, num_req_slots, buffer_slots})
      .with_dtype<int32_t>()
      .with_device(cuda_device)
      .verify(device_buffer_tokens);
  TensorMatcher({mapping_size})
      .with_dtype<int64_t>()
      .with_device(cuda_device)
      .verify(logical_to_device_mapping);

  LaunchKernel(batch_size.unwrap(), 256, cuda_device.unwrap())(
      complete_hisparse_spec_finalize_kernel<
          VERIFY_WIDTH, COMPRESS_RATIO, HOT_BUFFER_SIZE, SPECULATIVE_SLOTS>,
      static_cast<const int64_t*>(req_pool_indices.data_ptr()),
      static_cast<const int64_t*>(prefix_lens.data_ptr()),
      static_cast<const int32_t*>(commit_lens.data_ptr()),
      static_cast<const int32_t*>(req_to_token.data_ptr()),
      static_cast<int32_t*>(device_buffer_tokens.data_ptr()),
      static_cast<int64_t*>(logical_to_device_mapping.data_ptr()),
      req_to_token.strides()[0],
      device_buffer_tokens.strides()[0],
      device_buffer_tokens.strides()[1],
      static_cast<int32_t>(num_layers.unwrap()));
}

}  // namespace sglang
