/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Synchronization event logging for race condition debugging.
*/

#pragma once

#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/cutlass.h"
#if defined(__CUDACC_RTC__)
#include CUDA_STD_HEADER(cstdint)
#else
#include <cstdint>
#endif

#if !defined(__CUDACC_RTC__)
#include <mutex>
#include <vector>
#endif

namespace cutlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ENABLE_SYNCLOG)

constexpr uint32_t synclog_cap = 1 << 26;

inline std::mutex synclog_mutex;
inline std::vector<uint32_t*> synclog_buf_list;
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
CUTLASS_DEVICE uint32_t* synclog_buf;
#endif

CUTLASS_DEVICE
uint32_t* synclog_alloc(uint32_t n) {
  #if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
  uint32_t* buf = synclog_buf;
  if (buf == nullptr) return nullptr;
  uint32_t last = atomicAdd(&buf[0], n);
  if (last + n < synclog_cap) return buf + last + 1;
  if (last >= synclog_cap) atomicAdd(&buf[0], -n);
  #endif
  return nullptr;
}

CUTLASS_DEVICE
void synclog_emit_prefix(uint32_t* to, uint32_t header, uint32_t line) {
  #if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
  uint64_t time64;
  asm volatile (
    "mov.u64 %0, %%globaltimer;\n"
    : "=l"(time64) :
  );
  to[0] = header;
  to[1] = line;
  to[2] = time64;
  to[3] = time64 >> 32;
  to[4] = threadIdx.x;
  to[5] = threadIdx.y;
  to[6] = threadIdx.z;
  to[7] = blockIdx.x;
  to[8] = blockIdx.y;
  to[9] = blockIdx.z;
  #endif
}

constexpr uint32_t synclog_header_none = 0;
constexpr uint32_t synclog_length_prefix = 1 + 1 + 2 + 3 + 3;

constexpr bool     synclog_enable_syncthreads = true;
constexpr uint32_t synclog_header_syncthreads = 1;
constexpr uint32_t synclog_length_syncthreads = synclog_length_prefix + 0;

constexpr bool     synclog_enable_syncwarp = true;
constexpr uint32_t synclog_header_syncwarp = 2;
constexpr uint32_t synclog_length_syncwarp = synclog_length_prefix + 0;

constexpr bool     synclog_enable_named_barrier_arrive_and_wait = true;
constexpr uint32_t synclog_header_named_barrier_arrive_and_wait = 3;
constexpr uint32_t synclog_length_named_barrier_arrive_and_wait = synclog_length_prefix + 2;

constexpr bool     synclog_enable_named_barrier_arrive = true;
constexpr uint32_t synclog_header_named_barrier_arrive = 4;
constexpr uint32_t synclog_length_named_barrier_arrive = synclog_length_prefix + 2;

constexpr bool     synclog_enable_cluster_barrier_init = true;
constexpr uint32_t synclog_header_cluster_barrier_init = 5;
constexpr uint32_t synclog_length_cluster_barrier_init = synclog_length_prefix + 2;

constexpr bool     synclog_enable_cluster_barrier_wait = true;
constexpr uint32_t synclog_header_cluster_barrier_wait = 6;
constexpr uint32_t synclog_length_cluster_barrier_wait = synclog_length_prefix + 2;
constexpr bool     synclog_enable_cluster_barrier_test_wait = true;
constexpr uint32_t synclog_header_cluster_barrier_test_wait = 7;
constexpr uint32_t synclog_length_cluster_barrier_test_wait = synclog_length_prefix + 3;
constexpr bool     synclog_enable_cluster_barrier_try_wait = true;
constexpr uint32_t synclog_header_cluster_barrier_try_wait = 8;
constexpr uint32_t synclog_length_cluster_barrier_try_wait = synclog_length_prefix + 2;
constexpr bool     synclog_enable_cluster_barrier_arrive_cluster = true;
constexpr uint32_t synclog_header_cluster_barrier_arrive_cluster = 9;
constexpr uint32_t synclog_length_cluster_barrier_arrive_cluster = synclog_length_prefix + 3;
constexpr bool     synclog_enable_cluster_barrier_arrive = true;
constexpr uint32_t synclog_header_cluster_barrier_arrive = 10;
constexpr uint32_t synclog_length_cluster_barrier_arrive = synclog_length_prefix + 1;
constexpr bool     synclog_enable_cluster_barrier_invalidate = true;
constexpr uint32_t synclog_header_cluster_barrier_invalidate = 11;
constexpr uint32_t synclog_length_cluster_barrier_invalidate = synclog_length_prefix + 1;
constexpr bool     synclog_enable_cluster_transaction_barrier_arrive_and_expect_tx = true;
constexpr uint32_t synclog_header_cluster_transaction_barrier_arrive_and_expect_tx = 12;
constexpr uint32_t synclog_length_cluster_transaction_barrier_arrive_and_expect_tx = synclog_length_prefix + 2;
constexpr bool     synclog_enable_cluster_transaction_barrier_arrive_and_expect_tx_cluster = true;
constexpr uint32_t synclog_header_cluster_transaction_barrier_arrive_and_expect_tx_cluster = 13;
constexpr uint32_t synclog_length_cluster_transaction_barrier_arrive_and_expect_tx_cluster = synclog_length_prefix + 4;
constexpr bool     synclog_enable_cluster_transaction_barrier_expect_transaction = true;
constexpr uint32_t synclog_header_cluster_transaction_barrier_expect_transaction = 14;
constexpr uint32_t synclog_length_cluster_transaction_barrier_expect_transaction = synclog_length_prefix + 2;
constexpr bool     synclog_enable_cluster_transaction_barrier_complete_transaction = true;
constexpr uint32_t synclog_header_cluster_transaction_barrier_complete_transaction = 15;
constexpr uint32_t synclog_length_cluster_transaction_barrier_complete_transaction = synclog_length_prefix + 4;
constexpr bool     synclog_enable_fence_barrier_init = true;
constexpr uint32_t synclog_header_fence_barrier_init = 16;
constexpr uint32_t synclog_length_fence_barrier_init = synclog_length_prefix + 0;

constexpr bool     synclog_enable_fence_view_async_shared = true;
constexpr uint32_t synclog_header_fence_view_async_shared = 17;
constexpr uint32_t synclog_length_fence_view_async_shared = synclog_length_prefix + 0;

constexpr bool     synclog_enable_cp_async_wait = true;
constexpr uint32_t synclog_header_cp_async_wait = 18;
constexpr uint32_t synclog_length_cp_async_wait = synclog_length_prefix + 1;

constexpr bool     synclog_enable_cp_async_wait_all = true;
constexpr uint32_t synclog_header_cp_async_wait_all = 19;
constexpr uint32_t synclog_length_cp_async_wait_all = synclog_length_prefix + 0;

constexpr bool     synclog_enable_cp_async_fence = true;
constexpr uint32_t synclog_header_cp_async_fence = 20;
constexpr uint32_t synclog_length_cp_async_fence = synclog_length_prefix + 0;

constexpr bool     synclog_enable_cp_async_nan = true;
constexpr uint32_t synclog_header_cp_async_nan = 21;
constexpr uint32_t synclog_length_cp_async_nan = synclog_length_prefix + 4;

constexpr bool     synclog_enable_cp_async_zfill = true;
constexpr uint32_t synclog_header_cp_async_zfill = 22;
constexpr uint32_t synclog_length_cp_async_zfill = synclog_length_prefix + 5;

constexpr bool     synclog_enable_cp_async = true;
constexpr uint32_t synclog_header_cp_async = 23;
constexpr uint32_t synclog_length_cp_async = synclog_length_prefix + 5;

constexpr bool     synclog_enable_tma_load = true;
constexpr uint32_t synclog_header_tma_load = 24;
constexpr uint32_t synclog_length_tma_load = synclog_length_prefix + 4;

constexpr bool     synclog_enable_tma_store = true;
constexpr uint32_t synclog_header_tma_store = 25;
constexpr uint32_t synclog_length_tma_store = synclog_length_prefix + 3;

constexpr bool     synclog_enable_tma_store_arrive = true;
constexpr uint32_t synclog_header_tma_store_arrive = 26;
constexpr uint32_t synclog_length_tma_store_arrive = synclog_length_prefix + 0;

constexpr bool     synclog_enable_tma_store_wait = true;
constexpr uint32_t synclog_header_tma_store_wait = 27;
constexpr uint32_t synclog_length_tma_store_wait = synclog_length_prefix + 1;

constexpr bool     synclog_enable_warpgroup_arrive = true;
constexpr uint32_t synclog_header_warpgroup_arrive = 28;
constexpr uint32_t synclog_length_warpgroup_arrive = synclog_length_prefix + 0;

constexpr bool     synclog_enable_warpgroup_wait = true;
constexpr uint32_t synclog_header_warpgroup_wait = 29;
constexpr uint32_t synclog_length_warpgroup_wait = synclog_length_prefix + 1;

constexpr bool     synclog_enable_warpgroup_commit_batch = true;
constexpr uint32_t synclog_header_warpgroup_commit_batch = 30;
constexpr uint32_t synclog_length_warpgroup_commit_batch = synclog_length_prefix + 0;

constexpr bool     synclog_enable_wgmma_reg_smem = true;
constexpr uint32_t synclog_header_wgmma_reg_smem = 31;
constexpr uint32_t synclog_length_wgmma_reg_smem = synclog_length_prefix + 2;

constexpr bool     synclog_enable_wgmma_smem_smem = true;
constexpr uint32_t synclog_header_wgmma_smem_smem = 32;
constexpr uint32_t synclog_length_wgmma_smem_smem = synclog_length_prefix + 4;

constexpr bool     synclog_enable_cpasync_barrier_arrive = true;
constexpr uint32_t synclog_header_cpasync_barrier_arrive = 33;
constexpr uint32_t synclog_length_cpasync_barrier_arrive = synclog_length_prefix + 1;
CUTLASS_DEVICE
bool synclog_condition_emit() {
  #if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
  return threadIdx.x % NumThreadsPerWarp == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
    blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
  #else
  return 0;
  #endif
}

CUTLASS_DEVICE
bool synclog_condition_print() {
  #if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
  return threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
    blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
  #else
  return false;
  #endif
}

CUTLASS_DEVICE
void synclog_print_prefix(char const* header, uint32_t at) {
  #if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
  uint32_t line = synclog_buf[at + 1];
  uint32_t timeLo = synclog_buf[at + 2];
  uint32_t timeHi = synclog_buf[at + 3];
  uint32_t threadIdxX = synclog_buf[at + 4];
  uint32_t threadIdxY = synclog_buf[at + 5];
  uint32_t threadIdxZ = synclog_buf[at + 6];
  uint32_t blockIdxX = synclog_buf[at + 7];
  uint32_t blockIdxY = synclog_buf[at + 8];
  uint32_t blockIdxZ = synclog_buf[at + 9];
  printf(
    "%s line=%u time=%lu thread=%u,%u,%u block=%u,%u,%u ",
    header, line,
    (uint64_t)timeHi << 32 | timeLo,
    threadIdxX, threadIdxY, threadIdxZ,
    blockIdxX, blockIdxY, blockIdxZ
  );
  #endif
}

CUTLASS_DEVICE
void synclog_print_wgmma_desc(char const* str, uint32_t lo, uint32_t hi, char const* sep) {
  CUTLASS_UNUSED(hi);
  uint32_t smem_int_ptr = (lo & ((1 << 14) - 1)) << 4;
  printf("%s_smem_int_ptr=%u%s", str, smem_int_ptr, sep);
}

#endif // defined(CUTLASS_ENABLE_SYNCLOG)

////////////////////////////////////////////////////////////////////////////////////////////////////

inline void synclog_setup() {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  #if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
  std::scoped_lock lock(synclog_mutex);
  auto fail = [] () {
    fprintf(stderr, "synclog_setup() failed\n");
    std::terminate();
  };
  int orig_device = 0;
  if (cudaGetDevice(&orig_device) != cudaSuccess) {
    fail();
  }
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
    fail();
  }
  if (synclog_buf_list.size() == 0) {
    for (int device = 0; device < device_count; device++) {
      uint32_t* buf = 0;
      if (cudaSetDevice(device) != cudaSuccess ||
        cudaMalloc(&buf, synclog_cap * sizeof(uint32_t)) != cudaSuccess) {
        fail();
      }
      synclog_buf_list.push_back(buf);
    }
  }
  for (int device = 0; device < device_count; device++) {
    uint32_t* buf = synclog_buf_list.at(device);
    if (cudaSetDevice(device) != cudaSuccess ||
      cudaMemset(buf, 0, synclog_cap * sizeof(uint32_t)) != cudaSuccess ||
      cudaMemcpyToSymbol(synclog_buf, &buf, sizeof(buf)) != cudaSuccess) {
      fail();
    }
  }
  if (cudaSetDevice(orig_device) != cudaSuccess) {
    fail();
  }
  #endif
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_syncthreads(uint32_t line) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_syncthreads) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_syncthreads);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_syncthreads, line);
  #else
  CUTLASS_UNUSED(line);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_syncwarp(uint32_t line) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_syncwarp) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_syncwarp);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_syncwarp, line);
  #else
  CUTLASS_UNUSED(line);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_named_barrier_arrive_and_wait(
  uint32_t line,
  uint32_t num_threads,
  uint32_t barrier_id) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_named_barrier_arrive_and_wait) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_named_barrier_arrive_and_wait);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_named_barrier_arrive_and_wait, line);
  to[synclog_length_prefix + 0] = num_threads;
  to[synclog_length_prefix + 1] = barrier_id;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(num_threads);
  CUTLASS_UNUSED(barrier_id);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_named_barrier_arrive(
  uint32_t line,
  uint32_t num_threads,
  uint32_t barrier_id) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_named_barrier_arrive) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_named_barrier_arrive);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_named_barrier_arrive, line);
  to[synclog_length_prefix + 0] = num_threads;
  to[synclog_length_prefix + 1] = barrier_id;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(num_threads);
  CUTLASS_UNUSED(barrier_id);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_barrier_init(
  uint32_t line,
  uint32_t smem_addr,
  uint32_t arrive_count) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_barrier_init) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_cluster_barrier_init);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_cluster_barrier_init, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = arrive_count;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(arrive_count);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_barrier_wait(
  uint32_t line,
  uint32_t smem_addr,
  uint32_t phase) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_barrier_wait) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_cluster_barrier_wait);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_cluster_barrier_wait, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = phase;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(phase);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_barrier_test_wait(
  uint32_t line,
  uint32_t smem_addr,
  uint32_t phase,
  uint32_t pred) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_barrier_test_wait) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_cluster_barrier_test_wait);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_cluster_barrier_test_wait, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = phase;
  to[synclog_length_prefix + 2] = pred;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(phase);
  CUTLASS_UNUSED(pred);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_barrier_try_wait(
  uint32_t line,
  uint32_t smem_addr,
  uint32_t phase) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_barrier_try_wait) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_cluster_barrier_try_wait);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_cluster_barrier_try_wait, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = phase;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(phase);  
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_barrier_arrive_cluster(
  uint32_t line,
  uint32_t smem_addr,
  uint32_t cta_id,
  uint32_t pred) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_barrier_arrive_cluster) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_cluster_barrier_arrive_cluster);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_cluster_barrier_arrive_cluster, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = cta_id;
  to[synclog_length_prefix + 2] = pred;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(cta_id);
  CUTLASS_UNUSED(pred);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_barrier_arrive(
  uint32_t line,
  uint32_t smem_addr) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_barrier_arrive) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_cluster_barrier_arrive);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_cluster_barrier_arrive, line);
  to[synclog_length_prefix + 0] = smem_addr;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_barrier_invalidate(
  uint32_t line,
  uint32_t smem_addr) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_barrier_invalidate) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_cluster_barrier_invalidate);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_cluster_barrier_invalidate, line);
  to[synclog_length_prefix + 0] = smem_addr;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_transaction_barrier_arrive_and_expect_tx(
  uint32_t line,
  uint32_t smem_addr,
  uint32_t transaction_bytes) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_transaction_barrier_arrive_and_expect_tx) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_cluster_transaction_barrier_arrive_and_expect_tx);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_cluster_transaction_barrier_arrive_and_expect_tx, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = transaction_bytes;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(transaction_bytes);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_transaction_barrier_arrive_and_expect_tx_cluster(
  uint32_t line,
  uint32_t smem_addr,
  uint32_t transaction_bytes,
  uint32_t cta_id,
  uint32_t pred) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_transaction_barrier_arrive_and_expect_tx_cluster) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_cluster_transaction_barrier_arrive_and_expect_tx_cluster);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_cluster_transaction_barrier_arrive_and_expect_tx_cluster, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = transaction_bytes;
  to[synclog_length_prefix + 2] = cta_id;
  to[synclog_length_prefix + 3] = pred;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(transaction_bytes);
  CUTLASS_UNUSED(cta_id);
  CUTLASS_UNUSED(pred);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_transaction_barrier_expect_transaction(
  uint32_t line,
  uint32_t smem_addr,
  uint32_t transaction_bytes) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_transaction_barrier_expect_transaction) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_cluster_transaction_barrier_expect_transaction);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_cluster_transaction_barrier_expect_transaction, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = transaction_bytes;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(transaction_bytes);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_transaction_barrier_complete_transaction(
  uint32_t line,
  uint32_t smem_addr,
  uint32_t dst_cta_id,
  uint32_t transaction_bytes,
  uint32_t pred) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_transaction_barrier_complete_transaction) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_cluster_transaction_barrier_complete_transaction);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_cluster_transaction_barrier_complete_transaction, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = dst_cta_id;
  to[synclog_length_prefix + 2] = transaction_bytes;
  to[synclog_length_prefix + 3] = pred;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(dst_cta_id);
  CUTLASS_UNUSED(transaction_bytes);
  CUTLASS_UNUSED(pred);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_fence_barrier_init(uint32_t line) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_fence_barrier_init) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_fence_barrier_init);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_fence_barrier_init, line);
  #else
  CUTLASS_UNUSED(line);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_fence_view_async_shared(uint32_t line) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_fence_view_async_shared) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_fence_view_async_shared);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_fence_view_async_shared, line);
  #else
  CUTLASS_UNUSED(line);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cp_async_wait(
  uint32_t line,
  uint32_t n) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cp_async_wait) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_cp_async_wait);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_cp_async_wait, line);
  to[synclog_length_prefix + 0] = n;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(n);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cp_async_wait_all(uint32_t line) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cp_async_wait_all) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_cp_async_wait_all);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_cp_async_wait_all, line);
  #else
  CUTLASS_UNUSED(line);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cp_async_fence(uint32_t line) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cp_async_fence) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_cp_async_fence);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_cp_async_fence, line);
  #else
  CUTLASS_UNUSED(line);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cp_async_nan(
  uint32_t line,
  uint32_t smem_addr,
  const void* gmem_ptr,
  uint32_t pred) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cp_async_nan) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_cp_async_nan);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_cp_async_nan, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = (uint32_t)((uint64_t)gmem_ptr);
  to[synclog_length_prefix + 2] = (uint32_t)((uint64_t)gmem_ptr >> 32);
  to[synclog_length_prefix + 3] = pred;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(gmem_ptr);
  CUTLASS_UNUSED(pred);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cp_async_zfill(
  uint32_t line,
  uint32_t smem_addr,
  const void* gmem_ptr,
  uint32_t pred,
  uint32_t size) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cp_async_zfill) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_cp_async_zfill);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_cp_async_zfill, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = (uint32_t)((uint64_t)gmem_ptr);
  to[synclog_length_prefix + 2] = (uint32_t)((uint64_t)gmem_ptr >> 32);
  to[synclog_length_prefix + 3] = pred;
  to[synclog_length_prefix + 4] = size;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(gmem_ptr);
  CUTLASS_UNUSED(pred);
  CUTLASS_UNUSED(size);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cp_async(
  uint32_t line,
  uint32_t smem_addr,
  const void* gmem_ptr,
  uint32_t pred,
  uint32_t size) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cp_async) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_cp_async);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_cp_async, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = (uint32_t)((uint64_t)gmem_ptr);
  to[synclog_length_prefix + 2] = (uint32_t)((uint64_t)gmem_ptr >> 32);
  to[synclog_length_prefix + 3] = pred;
  to[synclog_length_prefix + 4] = size;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(gmem_ptr);
  CUTLASS_UNUSED(pred);
  CUTLASS_UNUSED(size);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_tma_load(
  uint32_t line,
  uint64_t gmem_int_desc,
  uint32_t smem_int_mbar,
  uint32_t smem_int_ptr) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_tma_load) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_tma_load);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_tma_load, line);
  to[synclog_length_prefix + 0] = (uint32_t)((uint64_t)gmem_int_desc);
  to[synclog_length_prefix + 1] = (uint32_t)((uint64_t)gmem_int_desc >> 32);
  to[synclog_length_prefix + 2] = smem_int_mbar;
  to[synclog_length_prefix + 3] = smem_int_ptr;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(gmem_int_desc);
  CUTLASS_UNUSED(smem_int_mbar);
  CUTLASS_UNUSED(smem_int_ptr);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_tma_store(
  uint32_t line,
  uint64_t gmem_int_desc,
  uint32_t smem_int_ptr) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_tma_store) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_tma_store);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_tma_store, line);
  to[synclog_length_prefix + 0] = (uint32_t)((uint64_t)gmem_int_desc);
  to[synclog_length_prefix + 1] = (uint32_t)((uint64_t)gmem_int_desc >> 32);
  to[synclog_length_prefix + 2] = smem_int_ptr;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(gmem_int_desc);
  CUTLASS_UNUSED(smem_int_ptr);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_tma_store_arrive(uint32_t line) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_tma_store_arrive) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_tma_store_arrive);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_tma_store_arrive, line);
  #else
  CUTLASS_UNUSED(line);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_tma_store_wait(
  uint32_t line,
  uint32_t count) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_tma_store_wait) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_tma_store_wait);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_tma_store_wait, line);
  to[synclog_length_prefix + 0] = count;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(count);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_warpgroup_arrive(
  uint32_t line) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_warpgroup_arrive) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_warpgroup_arrive);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_warpgroup_arrive, line);
  #else
  CUTLASS_UNUSED(line);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_warpgroup_wait(
  uint32_t line,
  uint32_t n) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_warpgroup_wait) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_warpgroup_wait);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_warpgroup_wait, line);
  to[synclog_length_prefix + 0] = n;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(n);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_warpgroup_commit_batch(
  uint32_t line) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_warpgroup_commit_batch) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_warpgroup_commit_batch);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_warpgroup_commit_batch, line);
  #else
  CUTLASS_UNUSED(line);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_wgmma_reg_smem(
  uint32_t line,
  uint64_t desc_b) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_wgmma_reg_smem) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_wgmma_reg_smem);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_wgmma_reg_smem, line);
  to[synclog_length_prefix + 0] = desc_b;
  to[synclog_length_prefix + 1] = desc_b >> 32;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(desc_b);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_wgmma_smem_smem(
  uint32_t line,
  uint64_t desc_a,
  uint64_t desc_b) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_wgmma_smem_smem) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_wgmma_smem_smem);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_wgmma_smem_smem, line);
  to[synclog_length_prefix + 0] = desc_a;
  to[synclog_length_prefix + 1] = desc_a >> 32;
  to[synclog_length_prefix + 2] = desc_b;
  to[synclog_length_prefix + 3] = desc_b >> 32;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(desc_a);
  CUTLASS_UNUSED(desc_b);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cpasync_barrier_arrive(
  uint32_t line,
  uint32_t smem_addr) {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cpasync_barrier_arrive) return;
  if (!synclog_condition_emit()) return;
  uint32_t* to = synclog_alloc(synclog_length_cpasync_barrier_arrive);
  if (to == nullptr) return;
  synclog_emit_prefix(to, synclog_header_cpasync_barrier_arrive, line);
  to[synclog_length_prefix + 0] = smem_addr;
  #else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

#if !defined(CUTLASS_ENABLE_SYNCLOG)
CUTLASS_DEVICE
#elif defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
static __attribute__((__noinline__)) __device__
#else
static __attribute__((__noinline__))
#endif
void synclog_print() {
  #if defined(CUTLASS_ENABLE_SYNCLOG)
  #if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
  if (synclog_buf == nullptr || !synclog_condition_print()) {
    return;
  }
  printf("synclog start\n");
  for (uint32_t at = 1; at < synclog_cap; ) {
    uint32_t header = synclog_buf[at];
    if (header == synclog_header_none) {
      break;
    }
    printf("synclog at %u: ", at);
    if constexpr (synclog_enable_syncthreads) {
      if (header == synclog_header_syncthreads) {
        synclog_print_prefix("syncthreads", at);
        at += synclog_length_syncthreads;
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_syncwarp) {
      if (header == synclog_header_syncwarp) {
        synclog_print_prefix("syncwarp", at);
        at += synclog_length_syncwarp;
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_named_barrier_arrive_and_wait) {
      if (header == synclog_header_named_barrier_arrive_and_wait) {
        synclog_print_prefix("named_barrier_arrive_and_wait", at);
        at += synclog_length_named_barrier_arrive_and_wait;
        printf("num_threads=%u barrier_id=%u\n", synclog_buf[at-2], synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_named_barrier_arrive) {
      if (header == synclog_header_named_barrier_arrive) {
        synclog_print_prefix("named_barrier_arrive", at);
        at += synclog_length_named_barrier_arrive;
        printf("num_threads=%u barrier_id=%u\n", synclog_buf[at-2], synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_barrier_init) {
      if (header == synclog_header_cluster_barrier_init) {
        synclog_print_prefix("cluster_barrier_init", at);
        at += synclog_length_cluster_barrier_init;
        printf("smem_addr=%u arrive_count=%u\n", synclog_buf[at-2], synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_barrier_wait) {
      if (header == synclog_header_cluster_barrier_wait) {
        synclog_print_prefix("cluster_barrier_wait", at);
        at += synclog_length_cluster_barrier_wait;
        printf("smem_addr=%u phase=%u\n", synclog_buf[at-2], synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_barrier_test_wait) {
      if (header == synclog_header_cluster_barrier_test_wait) {
        synclog_print_prefix("cluster_barrier_test_wait", at);
        at += synclog_length_cluster_barrier_test_wait;
        printf("smem_addr=%u phase=%u pred=%u\n", synclog_buf[at-3], synclog_buf[at-2], synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_barrier_try_wait) {
      if (header == synclog_header_cluster_barrier_try_wait) {
        synclog_print_prefix("cluster_barrier_try_wait", at);
        at += synclog_length_cluster_barrier_try_wait;
        printf("smem_addr=%u phase=%u\n", synclog_buf[at-2], synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_barrier_arrive_cluster) {
      if (header == synclog_header_cluster_barrier_arrive_cluster) {
        synclog_print_prefix("cluster_barrier_arrive_cluster", at);
        at += synclog_length_cluster_barrier_arrive_cluster;
        printf("smem_addr=%u cta_id=%u pred=%u\n", synclog_buf[at-3], synclog_buf[at-2], synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_barrier_arrive) {
      if (header == synclog_header_cluster_barrier_arrive) {
        synclog_print_prefix("cluster_barrier_arrive", at);
        at += synclog_length_cluster_barrier_arrive;
        printf("smem_addr=%u\n", synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_barrier_invalidate) {
      if (header == synclog_header_cluster_barrier_invalidate) {
        synclog_print_prefix("cluster_barrier_invalidate", at);
        at += synclog_length_cluster_barrier_invalidate;
        printf("smem_addr=%u\n", synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_transaction_barrier_arrive_and_expect_tx) {
      if (header == synclog_header_cluster_transaction_barrier_arrive_and_expect_tx) {
        synclog_print_prefix("cluster_transaction_barrier_arrive_and_expect_tx", at);
        at += synclog_length_cluster_transaction_barrier_arrive_and_expect_tx;
        printf("smem_addr=%u transaction_bytes=%u\n", synclog_buf[at-2], synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_transaction_barrier_arrive_and_expect_tx_cluster) {
      if (header == synclog_header_cluster_transaction_barrier_arrive_and_expect_tx_cluster) {
        synclog_print_prefix("cluster_transaction_barrier_arrive_and_expect_tx_cluster", at);
        at += synclog_length_cluster_transaction_barrier_arrive_and_expect_tx_cluster;
        printf("smem_addr=%u transaction_bytes=%u cta_id=%u pred=%u\n", synclog_buf[at-4], synclog_buf[at-3], synclog_buf[at-2], synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_transaction_barrier_expect_transaction) {
      if (header == synclog_header_cluster_transaction_barrier_expect_transaction) {
        synclog_print_prefix("cluster_transaction_barrier_expect_transaction", at);
        at += synclog_length_cluster_transaction_barrier_expect_transaction;
        printf("smem_addr=%u transaction_bytes=%u\n", synclog_buf[at-2], synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_transaction_barrier_complete_transaction) {
      if (header == synclog_header_cluster_transaction_barrier_complete_transaction) {
        synclog_print_prefix("cluster_transaction_barrier_complete_transaction", at);
        at += synclog_length_cluster_transaction_barrier_complete_transaction;
        printf("smem_addr=%u dst_cta_id=%u transaction_bytes=%u pred=%u\n", synclog_buf[at-4], synclog_buf[at-3], synclog_buf[at-2], synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_fence_barrier_init) {
      if (header == synclog_header_fence_barrier_init) {
        synclog_print_prefix("fence_barrier_init", at);
        at += synclog_length_fence_barrier_init;
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_fence_view_async_shared) {
      if (header == synclog_header_fence_view_async_shared) {
        synclog_print_prefix("fence_view_async_shared", at);
        at += synclog_length_fence_view_async_shared;
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_cp_async_wait) {
      if (header == synclog_header_cp_async_wait) {
        synclog_print_prefix("cp_async_wait", at);
        at += synclog_length_cp_async_wait;
        printf("n=%u\n", synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_cp_async_wait_all) {
      if (header == synclog_header_cp_async_wait_all) {
        synclog_print_prefix("cp_async_wait_all", at);
        at += synclog_length_cp_async_wait_all;
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_cp_async_fence) {
      if (header == synclog_header_cp_async_fence) {
        synclog_print_prefix("cp_async_fence", at);
        at += synclog_length_cp_async_fence;
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_cp_async_nan) {
      if (header == synclog_header_cp_async_nan) {
        synclog_print_prefix("cp_async_nan", at);
        at += synclog_length_cp_async_nan;
        uint64_t gmem_addr = synclog_buf[at-3];
        gmem_addr += (uint64_t)synclog_buf[at-2] << 32;
        printf("smem_addr=%u gmem_addr=%llu pred=%u\n", synclog_buf[at-4], gmem_addr, synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_cp_async_zfill) {
      if (header == synclog_header_cp_async_zfill) {
        synclog_print_prefix("cp_async_zfill", at);
        at += synclog_length_cp_async_zfill;
        uint64_t gmem_addr = synclog_buf[at-4];
        gmem_addr += (uint64_t)synclog_buf[at-3] << 32;
        printf("smem_addr=%u gmem_addr=%llu pred=%u size=%u\n", synclog_buf[at-5], gmem_addr, synclog_buf[at-2], synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_cp_async) {
      if (header == synclog_header_cp_async) {
        synclog_print_prefix("cp_async", at);
        at += synclog_length_cp_async;
        uint64_t gmem_addr = synclog_buf[at-4];
        gmem_addr += (uint64_t)synclog_buf[at-3] << 32;
        printf("smem_addr=%u gmem_addr=%llu pred=%u size=%u\n", synclog_buf[at-5], gmem_addr, synclog_buf[at-2], synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_tma_load) {
      if (header == synclog_header_tma_load) {
        synclog_print_prefix("tma_load", at);
        at += synclog_length_tma_load;
        uint64_t gmem_int_desc = synclog_buf[at-4];
        gmem_int_desc += (uint64_t)synclog_buf[at-3] << 32;
        printf("gmem_int_desc=%llu smem_int_mbar=%u smem_int_ptr=%u\n", gmem_int_desc, synclog_buf[at-2], synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_tma_store) {
      if (header == synclog_header_tma_store) {
        synclog_print_prefix("tma_store", at);
        at += synclog_length_tma_store;
        uint64_t gmem_int_desc = synclog_buf[at-3];
        gmem_int_desc += (uint64_t)synclog_buf[at-2] << 32;
        printf("gmem_int_desc=%llu smem_int_ptr=%u\n", gmem_int_desc, synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_tma_store_arrive) {
      if (header == synclog_header_tma_store_arrive) {
        synclog_print_prefix("tma_store_arrive", at);
        at += synclog_length_tma_store_arrive;
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_tma_store_wait) {
      if (header == synclog_header_tma_store_wait) {
        synclog_print_prefix("tma_store_wait", at);
        at += synclog_length_tma_store_wait;
        printf("count=%u\n", synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_warpgroup_arrive) {
      if (header == synclog_header_warpgroup_arrive) {
        synclog_print_prefix("warpgroup_arrive", at);
        at += synclog_length_warpgroup_arrive;
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_warpgroup_wait) {
      if (header == synclog_header_warpgroup_wait) {
        synclog_print_prefix("warpgroup_wait", at);
        at += synclog_length_warpgroup_wait;
        printf("n=%u\n", synclog_buf[at-1]);
        continue;
      }
    }
    if constexpr (synclog_enable_warpgroup_commit_batch) {
      if (header == synclog_header_warpgroup_commit_batch) {
        synclog_print_prefix("warpgroup_commit_batch", at);
        at += synclog_length_warpgroup_commit_batch;
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_wgmma_reg_smem) {
      if (header == synclog_header_wgmma_reg_smem) {
        synclog_print_prefix("wgmma_reg_smem", at);
        at += synclog_length_wgmma_reg_smem;
        synclog_print_wgmma_desc("desc_b", synclog_buf[at-2], synclog_buf[at-1], "");
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_wgmma_smem_smem) {
      if (header == synclog_header_wgmma_smem_smem) {
        synclog_print_prefix("wgmma_smem_smem", at);
        at += synclog_length_wgmma_smem_smem;
        synclog_print_wgmma_desc("desc_a", synclog_buf[at-4], synclog_buf[at-3], " ");
        synclog_print_wgmma_desc("desc_b", synclog_buf[at-2], synclog_buf[at-1], "");
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_cpasync_barrier_arrive) {
      if (header == synclog_header_cpasync_barrier_arrive) {
        synclog_print_prefix("cpasync_barrier_arrive", at);
        at += synclog_length_cpasync_barrier_arrive;
        printf("smem_addr=%u\n", synclog_buf[at-1]);
        continue;
      }
    }
    asm volatile ("brkpt;\n" ::);
  }
  if (synclog_buf[0] >= synclog_cap) {
    printf(
      "synclog was truncated (exceeded capacity of %lu bytes)\n",
      (synclog_cap - 1) * sizeof(uint32_t)
    );
  }
  printf("synclog end\n");
  #endif
  #endif // defined(CUTLASS_ENABLE_SYNCLOG)
}

////////////////////////////////////////////////////////////////////////////////////////////////////


#if defined(CUTLASS_ENABLE_SYNCLOG)
#undef __syncthreads
#define __syncthreads() do {\
  cutlass::arch::synclog_emit_syncthreads(__LINE__);\
  __syncthreads();\
} while (0)
#endif // defined(CUTLASS_ENABLE_SYNCLOG)

#if defined(CUTLASS_ENABLE_SYNCLOG)
#undef __syncwarp
#define __syncwarp(...) do {\
  cutlass::arch::synclog_emit_syncwarp(__LINE__);\
  __syncwarp(__VA_ARGS__);\
} while (0)
#endif // defined(CUTLASS_ENABLE_SYNCLOG)


////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass
