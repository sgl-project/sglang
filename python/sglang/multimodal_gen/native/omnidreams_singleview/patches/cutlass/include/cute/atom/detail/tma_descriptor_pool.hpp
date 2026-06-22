/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/***************************************************************************************************
 * Host-side TMA descriptor pool with content-based caching.
 *
 * Problem: on SM120 (Blackwell), nvcc 13.0/13.2 materializes `__grid_constant__`
 * kernel parameters that embed a `TmaDescriptor` (128-byte, alignas(64)) into
 * thread-local memory before the address resolves to a state space. The PTX
 * instructions `prefetch.tensormap` and `cp.async.bulk.tensor` reject any
 * address that resolves to `.shared` or `.local` state space, aborting the
 * kernel with "Misaligned shared or local address". This is a compiler bug
 * affecting CUTLASS 4.2.x on SM120 + CUDA 13.x.
 *
 * Workaround: keep the actual TMA descriptor bytes in `cudaMalloc`'d global
 * memory (which is unambiguously `.global` state space), and store only a
 * `TmaDescriptor const*` pointer inside `Copy_Traits<SM90_TMA_*>`. Kernel
 * params then carry an 8-byte pointer (which can live in .local or registers
 * without issue) and the PTX TMA instructions dereference through the pointer
 * to the global descriptor.
 *
 * Cache: `make_tma_copy_atom` is called inside `Gemm::initialize()`, which may
 * execute under CUDA Graph capture. cudaMemcpy/cudaMemcpyAsync from default
 * stream during capture is illegal. To avoid this, the pool deduplicates by
 * descriptor content (linear search through a small list - typically <100
 * entries across a process lifetime). The first call for a unique descriptor
 * (in warmup, outside capture) allocates + copies; subsequent calls (during
 * capture for the same descriptor) hit the cache and do no CUDA work.
 *
 * Pool design:
 *   - One process-wide pool with persistent (leaked-at-exit) cudaMalloc slots.
 *   - Each slot is a 128-byte 64-aligned region.
 *   - Backing storage: 4 MB blocks (32k slots each), grown on demand.
 *   - Cache: linear list of (host_desc bytes, device ptr) for content dedup.
 *
 * This file is included from `cute/atom/copy_traits_sm90_tma.hpp` and friends.
 * It only needs to be visible in host-side translation units; device code never
 * touches the pool, only reads through pointers returned by `pin_tma_descriptor`.
 **************************************************************************************************/

#pragma once

#include <cute/arch/copy_sm90_desc.hpp>

#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>
#endif

namespace cute {
namespace detail {

#if !defined(__CUDACC_RTC__)

// Single TMA descriptor slot is exactly the size of CUtensorMap on this build.
// Note: alignof(TmaDescriptor) is 8 (it's just `CUtensorMap = struct { uint64_t opaque[16]; }`
// in CUDA 12+ driver headers), but the `prefetch.tensormap` / `cp.async.bulk.tensor`
// PTX instructions require the descriptor *address* to be 64-byte aligned. Since
// cudaMalloc returns at least 256-byte-aligned pointers and our slot stride is
// 128 bytes (a multiple of 64), every slot ptr is naturally 64-aligned.
static constexpr size_t kTmaDescriptorSlotBytes = sizeof(TmaDescriptor);
static_assert(kTmaDescriptorSlotBytes == 128,
              "Expected TmaDescriptor to be 128 bytes for SM90+ TMA");
static_assert(kTmaDescriptorSlotBytes % 64 == 0,
              "Slot stride must be a multiple of 64 for TMA descriptor alignment");

// Default block size: 32k slots (~4 MB). Most workloads use <100 descriptors
// across their lifetime, so this is comfortably oversized.
static constexpr size_t kTmaDescriptorBlockSlots = 32 * 1024;
// Hard cap on total pool size: 16 MB = 4 blocks.
static constexpr size_t kTmaDescriptorMaxBlocks = 4;

struct TmaDescriptorPool {
  static TmaDescriptorPool& instance() {
    static TmaDescriptorPool pool;
    return pool;
  }

  // Look up or pin `host_desc` and return its device pointer.
  // - Cache hit: returns the cached device pointer with NO CUDA calls
  //   (safe to call under CUDA Graph capture).
  // - Cache miss: allocates a fresh slot, cudaMemcpy's `host_desc` bytes into
  //   it, stores the mapping, returns the new device pointer.
  // Returns nullptr on cudaMalloc / cudaMemcpy failure.
  TmaDescriptor const* pin(TmaDescriptor const& host_desc) {
    std::lock_guard<std::mutex> guard(mutex_);

    // Cache hit: linear search through the entries. With <100 unique
    // descriptors typical for inference workloads, this is fast.
    for (auto const& entry : entries_) {
      if (std::memcmp(&entry.first, &host_desc, kTmaDescriptorSlotBytes) == 0) {
        return entry.second;
      }
    }

    // Cache miss: allocate + copy.
    void* slot = acquire_slot_locked();
    if (slot == nullptr) {
      return nullptr;
    }
    // Synchronous cudaMemcpy: the pool guarantees the device-side bytes are
    // ready before this returns. This MUST be called outside CUDA Graph
    // capture; the cache above ensures repeated calls during capture are
    // satisfied without touching the runtime.
    cudaError_t e = cudaMemcpy(slot, &host_desc, kTmaDescriptorSlotBytes,
                               cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
      std::fprintf(stderr,
                   "[cute::TmaDescriptorPool] cudaMemcpy failed: %s\n",
                   cudaGetErrorString(e));
      return nullptr;
    }
    auto const* device_ptr = static_cast<TmaDescriptor const*>(slot);
    entries_.emplace_back(host_desc, device_ptr);
    return device_ptr;
  }

  // For tests/diagnostics: number of unique descriptors pinned.
  size_t entry_count() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return entries_.size();
  }

 private:
  TmaDescriptorPool() {
    entries_.reserve(256);  // Typical workload upper bound.
  }
  ~TmaDescriptorPool() {
    // Leak: process exit. Avoids ordering issues with cuda runtime teardown.
  }

  TmaDescriptorPool(TmaDescriptorPool const&) = delete;
  TmaDescriptorPool& operator=(TmaDescriptorPool const&) = delete;

  // Returns a 128-byte slot from the current block, expanding on demand.
  // Must be called with `mutex_` held.
  void* acquire_slot_locked() {
    if (next_slot_index_ >= blocks_allocated_ * kTmaDescriptorBlockSlots) {
      if (!expand_locked()) {
        return nullptr;
      }
    }
    size_t idx = next_slot_index_++;
    size_t block = idx / kTmaDescriptorBlockSlots;
    size_t slot_in_block = idx % kTmaDescriptorBlockSlots;
    char* base = static_cast<char*>(blocks_[block]);
    return base + slot_in_block * kTmaDescriptorSlotBytes;
  }

  bool expand_locked() {
    if (blocks_allocated_ >= kTmaDescriptorMaxBlocks) {
      std::fprintf(stderr,
                   "[cute::TmaDescriptorPool] hit %zu-block cap (%zu slots); "
                   "TMA descriptor allocation will fail.\n",
                   kTmaDescriptorMaxBlocks,
                   kTmaDescriptorMaxBlocks * kTmaDescriptorBlockSlots);
      return false;
    }
    void* new_block = nullptr;
    cudaError_t e = cudaMalloc(&new_block, kTmaDescriptorBlockSlots * kTmaDescriptorSlotBytes);
    if (e != cudaSuccess || new_block == nullptr) {
      std::fprintf(stderr,
                   "[cute::TmaDescriptorPool] cudaMalloc(%zu MB) failed: %s\n",
                   (kTmaDescriptorBlockSlots * kTmaDescriptorSlotBytes) >> 20,
                   cudaGetErrorString(e));
      return false;
    }
    blocks_[blocks_allocated_++] = new_block;
    return true;
  }

  // Persistent storage. Each block is one cudaMalloc; never freed.
  void* blocks_[kTmaDescriptorMaxBlocks] = {nullptr};
  size_t blocks_allocated_ = 0;
  size_t next_slot_index_ = 0;

  // Cache for content-based dedup. Vector of (host_desc bytes, device ptr).
  std::vector<std::pair<TmaDescriptor, TmaDescriptor const*>> entries_;

  mutable std::mutex mutex_;
};

// Convenience wrapper used from `make_tma_copy_atom`. Pins the host-built
// `TmaDescriptor` into a pool slot and returns the device pointer. Returns
// nullptr on failure (caller should assert).
inline TmaDescriptor const* pin_tma_descriptor(TmaDescriptor const& host_desc) {
  return TmaDescriptorPool::instance().pin(host_desc);
}

#endif  // !defined(__CUDACC_RTC__)

}  // namespace detail
}  // namespace cute
