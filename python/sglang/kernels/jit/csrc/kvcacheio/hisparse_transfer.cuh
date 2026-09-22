#pragma once

#include <sgl_kernel/deepseek_v4/kvcacheio.cuh>

#include <stdint.h>

namespace sglang {

#ifdef USE_ROCM
constexpr int HISPARSE_TRANSFER_WARP_SIZE = 64;
#else
constexpr int HISPARSE_TRANSFER_WARP_SIZE = 32;
#endif

template <int ITEM_SIZE_BYTES>
struct LinearKVTransferPolicy {
  __device__ static __forceinline__ void copy_warp(
      int32_t lane_id,
      const void* __restrict__ src_cache,
      void* __restrict__ dst_cache,
      int64_t src_loc,
      int64_t dst_loc) {
    const auto* src = static_cast<const char*>(src_cache) + src_loc * ITEM_SIZE_BYTES;
    auto* dst = static_cast<char*>(dst_cache) + dst_loc * ITEM_SIZE_BYTES;

#ifdef USE_ROCM
    constexpr int64_t word_count = ITEM_SIZE_BYTES / static_cast<int64_t>(sizeof(uint64_t));
    const auto* src_words = reinterpret_cast<const uint64_t*>(src);
    auto* dst_words = reinterpret_cast<uint64_t*>(dst);
    for (int64_t i = lane_id; i < word_count; i += HISPARSE_TRANSFER_WARP_SIZE) {
      dst_words[i] = src_words[i];
    }

    constexpr int64_t tail_start = word_count * static_cast<int64_t>(sizeof(uint64_t));
    for (int64_t i = tail_start + lane_id; i < ITEM_SIZE_BYTES; i += HISPARSE_TRANSFER_WARP_SIZE) {
      dst[i] = src[i];
    }
#else
    // Issue the 512B body and 64B edge loads before either store so both host
    // reads can remain in flight. Rows alternate between 0B and 64B offsets in
    // a 128B transaction, so place the body on the aligned side of each row.
    if constexpr (ITEM_SIZE_BYTES == 576) {
      const bool edge_first = (reinterpret_cast<uintptr_t>(src) & 127u) == 64u;
      const int32_t body_offset = edge_first ? 64 : 0;
      const int32_t edge_offset = edge_first ? 0 : 512;
      uint64_t body_lo, body_hi;
      uint64_t edge_lo, edge_hi;
      const auto* body_src = reinterpret_cast<const uint64_t*>(src + body_offset + lane_id * 16);
      auto* body_dst = reinterpret_cast<uint64_t*>(dst + body_offset + lane_id * 16);
      asm volatile("ld.global.nc.v2.b64 {%0,%1},[%2];" : "=l"(body_lo), "=l"(body_hi) : "l"(body_src) : "memory");
      if (lane_id < 4) {
        const auto* edge_src = reinterpret_cast<const uint64_t*>(src + edge_offset + lane_id * 16);
        asm volatile("ld.global.nc.v2.b64 {%0,%1},[%2];" : "=l"(edge_lo), "=l"(edge_hi) : "l"(edge_src) : "memory");
      }
      asm volatile("st.global.cg.v2.b64 [%0],{%1,%2};" : : "l"(body_dst), "l"(body_lo), "l"(body_hi) : "memory");
      if (lane_id < 4) {
        auto* edge_dst = reinterpret_cast<uint64_t*>(dst + edge_offset + lane_id * 16);
        asm volatile("st.global.cg.v2.b64 [%0],{%1,%2};" : : "l"(edge_dst), "l"(edge_lo), "l"(edge_hi) : "memory");
      }
      return;
    }

    constexpr int total_pairs = ITEM_SIZE_BYTES / 16;
    const auto* src_words = reinterpret_cast<const uint64_t*>(src);
    auto* dst_words = reinterpret_cast<uint64_t*>(dst);
    for (int j = lane_id; j < total_pairs; j += HISPARSE_TRANSFER_WARP_SIZE) {
      uint64_t lo, hi;
      const auto* word_src = src_words + j * 2;
      asm volatile("ld.global.nc.v2.b64 {%0,%1},[%2];" : "=l"(lo), "=l"(hi) : "l"(word_src) : "memory");
      auto* word_dst = dst_words + j * 2;
      asm volatile("st.global.cg.v2.b64 [%0],{%1,%2};" : : "l"(word_dst), "l"(lo), "l"(hi) : "memory");
    }

    constexpr int tail_words = (ITEM_SIZE_BYTES - total_pairs * 16) / 8;
    if (tail_words > 0 && lane_id < tail_words) {
      const auto* tail_src = reinterpret_cast<const uint64_t*>(src + total_pairs * 16);
      auto* tail_dst = reinterpret_cast<uint64_t*>(dst + total_pairs * 16);
      uint64_t value;
      asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(value) : "l"(tail_src + lane_id) : "memory");
      asm volatile("st.global.cg.b64 [%0],%1;" : : "l"(tail_dst + lane_id), "l"(value) : "memory");
    }
#endif
  }
};

struct Dsv4PagedKVTransferPolicy {
  __device__ static __forceinline__ void copy_warp(
      int32_t lane_id,
      const void* __restrict__ src_cache,
      void* __restrict__ dst_cache,
      int64_t src_loc,
      int64_t dst_loc) {
    (void)lane_id;
    device::hisparse::transfer_item(
        /*dst_cache=*/dst_cache,
        /*src_cache=*/const_cast<void*>(src_cache),
        /*dst_index=*/static_cast<int32_t>(dst_loc),
        /*src_index=*/static_cast<int32_t>(src_loc));
  }
};

}  // namespace sglang
