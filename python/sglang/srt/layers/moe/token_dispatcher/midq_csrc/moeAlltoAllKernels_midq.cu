/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Modified from FlashInfer 0.7.1 csrc/nv_internal/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu
// Modified by NVIDIA: the combine kernel splits each token row over MIDQ_COMBINE_SPLIT CTAs. Every output
// element is computed by the same instruction sequence as upstream (same 16-byte vectors, same top-k tree, same
// cast), so the output is bit-identical; only the CTA that computes a vector changes.
#ifndef MIDQ_COMBINE_SPLIT
#define MIDQ_COMBINE_SPLIT 1
#endif
#include <cooperative_groups.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <type_traits>

#include "flashinfer/exception.h"
#include "flashinfer/utils.cuh"
#include "flashinfer/vec_dtypes.cuh"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h"
#include "tensorrt_llm/kernels/quantization_utils.cuh"

namespace tensorrt_llm::kernels::moe_alltoall {

using tensorrt_llm::common::launchWithPdlWhenEnabled;

#define ENABLE_DEBUG_PRINT 0
#define DISABLE_SYNC_FOR_PROFILING 0

constexpr int kEp4Size = 4;
// Two 32-thread warps can cover at most 64 16-byte activation vectors.
constexpr int kCompactDispatchMaxPayloadBytes = 1024;
constexpr int kCompactDispatchBlockSize = 128;
constexpr int kDispatchWarpSize = 32;
constexpr int kDispatchActivationWarps = 2;
constexpr int kDispatchWideVectorBytes = 16;
constexpr int kDispatchMetadataVectorBytes = 8;
constexpr int kDispatchHalfWarpSize = 16;
constexpr int kNvfp4ActivationToScaleRatio = 8;
constexpr int kNvfp4H2048ActivationBytes = 2048 / 2;
constexpr int kNvfp4H2048ScaleBytes = 2048 / 16;
constexpr int kTopK22MetadataBytes = 22 * sizeof(int32_t);
constexpr int kNvfp4H2048ActivationWorkers = kNvfp4H2048ActivationBytes / kDispatchWideVectorBytes;
constexpr int kNvfp4H2048ScaleWorkers = kNvfp4H2048ScaleBytes / kDispatchWideVectorBytes;
constexpr int kTopK22MetadataWorkers = kTopK22MetadataBytes / kDispatchMetadataVectorBytes;
static_assert(kCompactDispatchMaxPayloadBytes % kDispatchWideVectorBytes == 0);
static_assert(kCompactDispatchMaxPayloadBytes / kDispatchWideVectorBytes <=
              kDispatchActivationWarps * kDispatchWarpSize);
static_assert(kCompactDispatchBlockSize == (kDispatchActivationWarps + 2) * kDispatchWarpSize);

#ifndef DISABLE_TIMEOUT
#define DISABLE_TIMEOUT 0
#endif

// Helper function for ceiling division
template <typename T>
__host__ __device__ inline T ceilDiv(T m, T n) {
  return (m + n - 1) / n;
}

// The phased schedule reserves two warps for packed activations, one for scales,
// and splits the last warp between the two routing-metadata payloads. The
// activation cap intentionally limits this path to hidden sizes up to 2048.
static bool isPhasedNvfp4PayloadLayout(DispatchKernelPointers const& ptrs, int num_payloads,
                                       int top_k) {
  if (num_payloads != 4 || top_k <= kEp4Size) {
    return false;
  }

  int const activation_bytes = ptrs.payload_bytes_per_token[0];
  int const scale_bytes = ptrs.payload_bytes_per_token[1];
  int const metadata_bytes = top_k * static_cast<int>(sizeof(int32_t));
  // Smaller top-k layouts support runtime activation widths (including H1024).
  // Top-k 22 is restricted to the measured H2048 configuration.
  bool const validated_top_k_layout = top_k < 22 || activation_bytes == kNvfp4H2048ActivationBytes;
  return validated_top_k_layout && activation_bytes > 0 &&
         activation_bytes <= kCompactDispatchMaxPayloadBytes &&
         activation_bytes % kDispatchWideVectorBytes == 0 && scale_bytes > 0 &&
         scale_bytes % kDispatchWideVectorBytes == 0 &&
         activation_bytes == kNvfp4ActivationToScaleRatio * scale_bytes &&
         metadata_bytes % kDispatchMetadataVectorBytes == 0 &&
         metadata_bytes / kDispatchMetadataVectorBytes <= kDispatchHalfWarpSize &&
         ptrs.payload_bytes_per_token[2] == metadata_bytes &&
         ptrs.payload_bytes_per_token[3] == metadata_bytes;
}

// Macros for concise launch-time specialization
#define SWITCH_BOOL(flag, NAME, ...) \
  if (flag) {                        \
    constexpr bool NAME = true;      \
    __VA_ARGS__                      \
  } else {                           \
    constexpr bool NAME = false;     \
    __VA_ARGS__                      \
  }

#define SWITCH_TOP_K(top_k, TOP_K, ...)             \
  switch (top_k) {                                  \
    case 22: {                                      \
      constexpr int TOP_K = 22;                     \
      __VA_ARGS__;                                  \
      break;                                        \
    }                                               \
    case 18: {                                      \
      constexpr int TOP_K = 18;                     \
      __VA_ARGS__;                                  \
      break;                                        \
    }                                               \
    case 16: {                                      \
      constexpr int TOP_K = 16;                     \
      __VA_ARGS__;                                  \
      break;                                        \
    }                                               \
    case 14: {                                      \
      constexpr int TOP_K = 14;                     \
      __VA_ARGS__;                                  \
      break;                                        \
    }                                               \
    case 12: {                                      \
      constexpr int TOP_K = 12;                     \
      __VA_ARGS__;                                  \
      break;                                        \
    }                                               \
    case 10: {                                      \
      constexpr int TOP_K = 10;                     \
      __VA_ARGS__;                                  \
      break;                                        \
    }                                               \
    case 8: {                                       \
      constexpr int TOP_K = 8;                      \
      __VA_ARGS__;                                  \
      break;                                        \
    }                                               \
    case 6: {                                       \
      constexpr int TOP_K = 6;                      \
      __VA_ARGS__;                                  \
      break;                                        \
    }                                               \
    case 4: {                                       \
      constexpr int TOP_K = 4;                      \
      __VA_ARGS__;                                  \
      break;                                        \
    }                                               \
    case 2: {                                       \
      constexpr int TOP_K = 2;                      \
      __VA_ARGS__;                                  \
      break;                                        \
    }                                               \
    case 1: {                                       \
      constexpr int TOP_K = 1;                      \
      __VA_ARGS__;                                  \
      break;                                        \
    }                                               \
    default: {                                      \
      FLASHINFER_CHECK(false, "Unsupported top_k"); \
    }                                               \
  }

#define SWITCH_DTYPE(dtype, TYPE, ...)                                  \
  switch (dtype) {                                                      \
    case nvinfer1::DataType::kHALF: {                                   \
      using TYPE = half;                                                \
      __VA_ARGS__;                                                      \
      break;                                                            \
    }                                                                   \
    case nvinfer1::DataType::kBF16: {                                   \
      using TYPE = __nv_bfloat16;                                       \
      __VA_ARGS__;                                                      \
      break;                                                            \
    }                                                                   \
    case nvinfer1::DataType::kFLOAT: {                                  \
      using TYPE = float;                                               \
      __VA_ARGS__;                                                      \
      break;                                                            \
    }                                                                   \
    case nvinfer1::DataType::kFP8: {                                    \
      using TYPE = __nv_fp8_e4m3;                                       \
      __VA_ARGS__;                                                      \
      break;                                                            \
    }                                                                   \
    default: {                                                          \
      FLASHINFER_CHECK(false, "Unsupported dtype for moe_a2a_combine"); \
    }                                                                   \
  }

// Quantized combine currently only supports FP16/BF16 inputs.
#define SWITCH_QUANT_MODE(dtype, quant_mode, QUANT_MODE, ...)                          \
  if constexpr (std::is_same_v<dtype, half> || std::is_same_v<dtype, __nv_bfloat16>) { \
    switch (quant_mode) {                                                              \
      case MoeA2ACombineQuantMode::NONE: {                                             \
        constexpr auto QUANT_MODE = MoeA2ACombineQuantMode::NONE;                      \
        __VA_ARGS__;                                                                   \
        break;                                                                         \
      }                                                                                \
      case MoeA2ACombineQuantMode::MXFP8: {                                            \
        constexpr auto QUANT_MODE = MoeA2ACombineQuantMode::MXFP8;                     \
        __VA_ARGS__;                                                                   \
        break;                                                                         \
      }                                                                                \
      case MoeA2ACombineQuantMode::NVFP4: {                                            \
        constexpr auto QUANT_MODE = MoeA2ACombineQuantMode::NVFP4;                     \
        __VA_ARGS__;                                                                   \
        break;                                                                         \
      }                                                                                \
      case MoeA2ACombineQuantMode::MXFP4: {                                            \
        constexpr auto QUANT_MODE = MoeA2ACombineQuantMode::MXFP4;                     \
        __VA_ARGS__;                                                                   \
        break;                                                                         \
      }                                                                                \
      default: {                                                                       \
        FLASHINFER_CHECK(false, "Unsupported quant_mode for moe_a2a_combine");         \
      }                                                                                \
    }                                                                                  \
  } else {                                                                             \
    FLASHINFER_CHECK((quant_mode) == MoeA2ACombineQuantMode::NONE,                     \
                     "All to All Combine Quantization currently only supports "        \
                     "FP16/BF16");                                                     \
    constexpr auto QUANT_MODE = MoeA2ACombineQuantMode::NONE;                          \
    __VA_ARGS__;                                                                       \
  }

#define SWITCH_SWIZZLE_MODE(swizzle_mode, SWIZZLE_MODE, ...)                   \
  switch (swizzle_mode) {                                                      \
    case MoeA2ACombineSwizzleSFMode::LINEAR: {                                 \
      constexpr auto SWIZZLE_MODE = MoeA2ACombineSwizzleSFMode::LINEAR;        \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case MoeA2ACombineSwizzleSFMode::SWIZZLE_128x4: {                          \
      constexpr auto SWIZZLE_MODE = MoeA2ACombineSwizzleSFMode::SWIZZLE_128x4; \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case MoeA2ACombineSwizzleSFMode::SWIZZLE_8x4: {                            \
      constexpr auto SWIZZLE_MODE = MoeA2ACombineSwizzleSFMode::SWIZZLE_8x4;   \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    default: {                                                                 \
      FLASHINFER_CHECK(false, "Unsupported swizzle_mode for moe_a2a_combine"); \
    }                                                                          \
  }

#if DISABLE_TIMEOUT
#define check_timeout(s) false
#else
// 300 * 2000 MHz - should be high enough on any GPU but will prevent a hang
#define check_timeout(s) ((clock64() - (s)) > (300ll * 2000ll * 1000ll * 1000ll))
#endif

// ============================================================================
// Helper Functions for Expert-to-Rank Mapping
// ============================================================================

// Compute which rank owns a given expert using contiguous ceil/floor partitioning.
// Supports non-divisible distribution when num_experts % ep_size != 0:
//   base      = num_experts / ep_size
//   remainder = num_experts % ep_size
//   - Ranks [0, remainder) each own (base + 1) experts.
//   - Ranks [remainder, ep_size) each own base experts.
//
// Example A (uniform): 32 experts, 4 ranks -> base=8, remainder=0
//   - Rank 0: experts 0-7, Rank 1: 8-15, Rank 2: 16-23, Rank 3: 24-31
// Example B (non-divisible): 384 experts, 5 ranks -> base=76, remainder=4
//   - Ranks 0-3: 77 experts each, Rank 4: 76 experts
//
// base and remainder are precomputed by the caller once outside the per-token TOP_K loop
// so the hot path performs at most one integer divide.
__device__ __forceinline__ int compute_target_rank_id(int expert_id, int base, int remainder) {
  // Fast path for the uniform (num_experts % ep_size == 0) case: identical to the
  // pre-ceil/floor implementation, so existing divisible deployments incur no overhead.
  if (remainder == 0) {
    return expert_id / base;
  }
  int const split = remainder * (base + 1);  // boundary expert id
  if (expert_id < split) {
    // Falls inside the (base + 1)-sized prefix block.
    return expert_id / (base + 1);
  }
  // Falls inside the base-sized suffix block.
  return remainder + (expert_id - split) / base;
}

// Test bit `rank` in a kRankMaskWords-wide little-endian uint64 bitmask.
// Word 0 covers ranks 0..63, word 1 covers ranks 64..127, etc.
// `rank >> 6` and `rank & 63` divide / modulo by 64.
__device__ __forceinline__ bool is_rank_active(uint64_t const* mask, int rank) {
  return (mask[rank >> 6] >> (rank & 63)) & 1ULL;
}

// ============================================================================
// Helper Functions for Vectorized Memory Operations
// ============================================================================

template <int VEC_SIZE>
__device__ void vectorized_copy_impl(void* dst, void const* src, int size) {
  using flashinfer::vec_t;

  uint8_t* dst_ptr = static_cast<uint8_t*>(dst);
  uint8_t const* src_ptr = static_cast<uint8_t const*>(src);

  int const stride = blockDim.x * VEC_SIZE;

  for (int offset = threadIdx.x * VEC_SIZE; offset < size; offset += stride) {
    vec_t<uint8_t, VEC_SIZE> v;
    v.load(src_ptr + offset);
    v.store(dst_ptr + offset);
  }
}

__device__ void vectorized_copy(void* dst, void const* src, int size) {
  if (size % 16 == 0) {
    vectorized_copy_impl<16>(dst, src, size);
  } else if (size % 8 == 0) {
    vectorized_copy_impl<8>(dst, src, size);
  } else if (size % 4 == 0) {
    vectorized_copy_impl<4>(dst, src, size);
  } else if (size % 2 == 0) {
    vectorized_copy_impl<2>(dst, src, size);
  } else {
    vectorized_copy_impl<1>(dst, src, size);
  }
}

// Vectorized dispatch: load one vec from source and write to up to TOP_K destinations
template <int VEC_SIZE, int TOP_K>
__device__ void vectorized_dispatch_impl(uint8_t const* src_ptr, int bytes_per_token, int rank_id,
                                         int max_tokens_per_rank, int payload_idx,
                                         DispatchKernelPointers const& ptrs,
                                         int const* topk_target_ranks, int const* topk_send_indices,
                                         int worker_idx, int num_workers) {
  using flashinfer::vec_t;

  // Precompute destination base pointers per k
  uint8_t* dst_base_k[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    int dst_idx_k = topk_send_indices[k];
    int target_rank_k = topk_target_ranks[k];
    if (dst_idx_k < 0) {
      dst_base_k[k] = nullptr;
      continue;
    }
    uint8_t* dst_data = static_cast<uint8_t*>(ptrs.recv_buffers[target_rank_k][payload_idx]);
    size_t base_source_rank =
        static_cast<size_t>(rank_id) * static_cast<size_t>(max_tokens_per_rank) +
        static_cast<size_t>(dst_idx_k);
    size_t base_token = base_source_rank * static_cast<size_t>(bytes_per_token);
    dst_base_k[k] = dst_data + base_token;
  }

  // The caller assigns a worker subset to this payload. Only those workers pay
  // the per-destination pointer setup and copy cost.
  int const stride = num_workers * VEC_SIZE;
  for (int offset = worker_idx * VEC_SIZE; offset < bytes_per_token; offset += stride) {
    vec_t<uint8_t, VEC_SIZE> v;
    v.load(src_ptr + offset);

#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      uint8_t* dst_base = dst_base_k[k];
      if (dst_base == nullptr) {
        continue;
      }
      v.store(dst_base + offset);
    }
  }
}

template <int TOP_K>
__device__ void vectorized_dispatch(uint8_t const* src_ptr, int bytes_per_token, int rank_id,
                                    int max_tokens_per_rank, int payload_idx,
                                    DispatchKernelPointers const& ptrs,
                                    int const* topk_target_ranks, int const* topk_send_indices) {
  // Receive planes are packed without vector-width padding. A small preceding
  // payload can misalign the next plane even when its row size is divisible by
  // 16. Token offsets are multiples of bytes_per_token, so checking every base
  // together with the row size preserves wide copies exactly where safe.
  uintptr_t copy_alignment =
      reinterpret_cast<uintptr_t>(src_ptr) | static_cast<uintptr_t>(bytes_per_token);
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    if (topk_send_indices[k] >= 0) {
      copy_alignment |=
          reinterpret_cast<uintptr_t>(ptrs.recv_buffers[topk_target_ranks[k]][payload_idx]);
    }
  }
  if (copy_alignment % 16 == 0) {
    vectorized_dispatch_impl<16, TOP_K>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
                                        payload_idx, ptrs, topk_target_ranks, topk_send_indices,
                                        threadIdx.x, blockDim.x);
  } else if (copy_alignment % 8 == 0) {
    vectorized_dispatch_impl<8, TOP_K>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
                                       payload_idx, ptrs, topk_target_ranks, topk_send_indices,
                                       threadIdx.x, blockDim.x);
  } else if (copy_alignment % 4 == 0) {
    vectorized_dispatch_impl<4, TOP_K>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
                                       payload_idx, ptrs, topk_target_ranks, topk_send_indices,
                                       threadIdx.x, blockDim.x);
  } else if (copy_alignment % 2 == 0) {
    vectorized_dispatch_impl<2, TOP_K>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
                                       payload_idx, ptrs, topk_target_ranks, topk_send_indices,
                                       threadIdx.x, blockDim.x);
  } else {
    vectorized_dispatch_impl<1, TOP_K>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
                                       payload_idx, ptrs, topk_target_ranks, topk_send_indices,
                                       threadIdx.x, blockDim.x);
  }
}

__global__ void moeA2APrepareDispatchKernel(int* send_counters, int* local_token_counter,
                                            int ep_size, uint32_t* flag_val_ptr, bool enable_pdl) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (enable_pdl) cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
#endif
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Zero send_counters
  if (idx < ep_size) {
    send_counters[idx] = 0;
  }
  // Zero local_token_counter and increment flag_val
  if (idx == 0) {
    *local_token_counter = 0;
    // Increment flag_val for this dispatch round
    *flag_val_ptr = *flag_val_ptr + 1;
  }
}

// ============================================================================
// Generic Dispatch Kernel Implementation
// One CTA processes one token and all its payloads.
// ============================================================================

template <int TOP_K, bool ENABLE_EPLB, bool ENABLE_RANK_MASK, bool COMPACT_EP4,
          bool PHASE_H2048_NVFP4 = false, bool PHASE_NVFP4_PAYLOADS = false>
__global__ void moeA2ADispatchKernel(
    int32_t const* token_selected_experts,  // [local_num_tokens, TOP_K]
    const DispatchKernelPointers ptrs,      // Struct containing all kernel pointers
    int num_payloads,                       // Number of payloads
    int max_tokens_per_rank,                // Maximum tokens per rank
    int local_num_tokens, int rank_id, int ep_size, int num_experts, int eplb_stats_num_experts,
    bool enable_pdl) {
  static_assert(!COMPACT_EP4 || TOP_K > kEp4Size);
  static_assert(!PHASE_H2048_NVFP4 || TOP_K == 22);
  static_assert(!PHASE_NVFP4_PAYLOADS ||
                (TOP_K > kEp4Size && TOP_K % 2 == 0 &&
                 TOP_K * static_cast<int>(sizeof(int32_t)) / kDispatchMetadataVectorBytes <=
                     kDispatchHalfWarpSize));
  static_assert(!PHASE_H2048_NVFP4 || !PHASE_NVFP4_PAYLOADS);
  int thread_idx = threadIdx.x;
  int local_token_idx = blockIdx.x;

  if (local_num_tokens == 0) {
    // Special case: If local_num_tokens == 0,
    // we need to keep the threads where local_token_idx == 0 alive to participate in the
    // synchronization. Other threads should return.
    if (local_token_idx > 0) return;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if (enable_pdl) cudaGridDependencySynchronize();
#endif
  } else {
    // Threads that do not have a token to process should return.
    if (local_token_idx >= local_num_tokens) return;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if (enable_pdl) cudaGridDependencySynchronize();
#endif

    // Prepare per-policy shared-memory tiles for this token
    extern __shared__ int smem[];
    int* smem_topk_target_ranks = smem;
    int* smem_topk_send_indices = nullptr;
    int* smem_compact_send_indices = smem;
    if constexpr (!COMPACT_EP4) {
      smem_topk_send_indices = smem + TOP_K;
    }

    // Precompute the ceil/floor partition parameters once per thread, outside the
    // per-token TOP_K loop.  The fast path (remainder == 0) then collapses to a single
    // integer divide per call, matching the pre-PR uniform-partition cost exactly.
    int const ep_base = num_experts / ep_size;
    int const ep_remainder = num_experts - ep_base * ep_size;  // == num_experts % ep_size

    if (thread_idx < warpSize) {
      int lane_id = thread_idx;
      if constexpr (COMPACT_EP4) {
        if (lane_id < kEp4Size) {
          smem_compact_send_indices[lane_id] = -1;
        }
        __syncwarp();
      }

      unsigned topk_mask = __ballot_sync(0xffffffff, lane_id < TOP_K);
      if (lane_id < TOP_K) {
        int k = lane_id;
        int expert_id = token_selected_experts[local_token_idx * TOP_K + k];
        // Use contiguous ceil/floor partitioning (supports non-divisible num_experts % ep_size).
        int target_rank = compute_target_rank_id(expert_id, ep_base, ep_remainder);

        // Elect the first top-k lane for each destination rank; duplicate targets within a
        // token collapse to a single send (replaces the old serial already_copied bitmask).
        unsigned matching_lanes = __match_any_sync(topk_mask, target_rank);
        bool is_first = lane_id == __ffs(matching_lanes) - 1;

        // Dead ranks (ENABLE_RANK_MASK) are dropped the same way as duplicates: no send is
        // issued and topk_send_indices[k] stays < 0, so combine's existing skip check on that
        // sentinel handles both cases uniformly. When ENABLE_RANK_MASK is false, the dead-rank
        // check is compiled out entirely (no active_rank_mask load/branch on this hot path).
        bool is_valid = is_first;
        if constexpr (ENABLE_RANK_MASK) {
          is_valid = is_valid && is_rank_active(ptrs.active_rank_mask, target_rank);
        }

        int dst_token_idx = -1;
        if (is_valid) {
          dst_token_idx = atomicAdd(&ptrs.send_counters[target_rank], 1);
          if constexpr (COMPACT_EP4) {
            smem_compact_send_indices[target_rank] = dst_token_idx;
          }
        } else {
          target_rank = -1;
        }

        ptrs.topk_target_ranks[local_token_idx * TOP_K + k] = target_rank;
        ptrs.topk_send_indices[local_token_idx * TOP_K + k] = dst_token_idx;
        if constexpr (!COMPACT_EP4) {
          smem_topk_target_ranks[k] = target_rank;
          smem_topk_send_indices[k] = dst_token_idx;
        }
      }
    }
    // Sync before dispatching data
    __syncthreads();

    // EP4 has at most four unique destinations, regardless of TOP_K. The compact
    // specialization avoids expanding those destinations back to TOP_K entries.
    constexpr int NUM_DESTINATIONS = COMPACT_EP4 ? kEp4Size : TOP_K;
    int target_ranks[NUM_DESTINATIONS];
    int send_indices[NUM_DESTINATIONS];
    if constexpr (COMPACT_EP4) {
#pragma unroll
      for (int target_rank = 0; target_rank < NUM_DESTINATIONS; ++target_rank) {
        target_ranks[target_rank] = target_rank;
        send_indices[target_rank] = smem_compact_send_indices[target_rank];
      }
    } else {
#pragma unroll
      for (int k = 0; k < TOP_K; ++k) {
        target_ranks[k] = smem_topk_target_ranks[k];
        send_indices[k] = smem_topk_send_indices[k];
      }
    }

    if constexpr (PHASE_H2048_NVFP4) {
      // Keep compile-time bounds for the measured Nemotron H2048 target.
      int warp_id = thread_idx / warpSize;
      int lane_id = thread_idx % warpSize;
      if (warp_id < kDispatchActivationWarps) {
        auto const* src = static_cast<uint8_t const*>(ptrs.src_data_ptrs[0]) +
                          static_cast<size_t>(local_token_idx) * kNvfp4H2048ActivationBytes;
        vectorized_dispatch_impl<16, NUM_DESTINATIONS>(
            src, kNvfp4H2048ActivationBytes, rank_id, max_tokens_per_rank, 0, ptrs, target_ranks,
            send_indices, thread_idx, kNvfp4H2048ActivationWorkers);
      } else if (warp_id == kDispatchActivationWarps && lane_id < kNvfp4H2048ScaleWorkers) {
        auto const* src = static_cast<uint8_t const*>(ptrs.src_data_ptrs[1]) +
                          static_cast<size_t>(local_token_idx) * kNvfp4H2048ScaleBytes;
        vectorized_dispatch_impl<16, NUM_DESTINATIONS>(
            src, kNvfp4H2048ScaleBytes, rank_id, max_tokens_per_rank, 1, ptrs, target_ranks,
            send_indices, lane_id, kNvfp4H2048ScaleWorkers);
      } else if (warp_id == kDispatchActivationWarps + 1 && lane_id < 2 * kTopK22MetadataWorkers) {
        bool second_tail = lane_id >= kTopK22MetadataWorkers;
        int payload_idx = second_tail ? 3 : 2;
        int worker_idx = second_tail ? lane_id - kTopK22MetadataWorkers : lane_id;
        auto const* src = static_cast<uint8_t const*>(ptrs.src_data_ptrs[payload_idx]) +
                          static_cast<size_t>(local_token_idx) * kTopK22MetadataBytes;
        vectorized_dispatch_impl<8, NUM_DESTINATIONS>(
            src, kTopK22MetadataBytes, rank_id, max_tokens_per_rank, payload_idx, ptrs,
            target_ranks, send_indices, worker_idx, kTopK22MetadataWorkers);
      }
    } else if constexpr (PHASE_NVFP4_PAYLOADS) {
      // Match warps to the NVFP4 payload widths instead of making all 128
      // threads materialize destination pointers for every payload.
      int const activation_bytes = ptrs.payload_bytes_per_token[0];
      int const scale_bytes = ptrs.payload_bytes_per_token[1];
      constexpr int METADATA_BYTES = TOP_K * sizeof(int32_t);
      constexpr int METADATA_WORKERS = METADATA_BYTES / kDispatchMetadataVectorBytes;
      int const activation_workers = activation_bytes / kDispatchWideVectorBytes;
      int const scale_workers = scale_bytes / kDispatchWideVectorBytes;
      int warp_id = thread_idx / warpSize;
      int lane_id = thread_idx % warpSize;
      if (warp_id < kDispatchActivationWarps && thread_idx < activation_workers) {
        auto const* src = static_cast<uint8_t const*>(ptrs.src_data_ptrs[0]) +
                          static_cast<size_t>(local_token_idx) * activation_bytes;
        vectorized_dispatch_impl<kDispatchWideVectorBytes, NUM_DESTINATIONS>(
            src, activation_bytes, rank_id, max_tokens_per_rank, 0, ptrs, target_ranks,
            send_indices, thread_idx, activation_workers);
      } else if (warp_id == kDispatchActivationWarps && lane_id < scale_workers) {
        auto const* src = static_cast<uint8_t const*>(ptrs.src_data_ptrs[1]) +
                          static_cast<size_t>(local_token_idx) * scale_bytes;
        vectorized_dispatch_impl<kDispatchWideVectorBytes, NUM_DESTINATIONS>(
            src, scale_bytes, rank_id, max_tokens_per_rank, 1, ptrs, target_ranks, send_indices,
            lane_id, scale_workers);
      } else if (warp_id == kDispatchActivationWarps + 1 && lane_id < 2 * METADATA_WORKERS) {
        bool second_tail = lane_id >= METADATA_WORKERS;
        int payload_idx = second_tail ? 3 : 2;
        int worker_idx = second_tail ? lane_id - METADATA_WORKERS : lane_id;
        auto const* src = static_cast<uint8_t const*>(ptrs.src_data_ptrs[payload_idx]) +
                          static_cast<size_t>(local_token_idx) * METADATA_BYTES;
        vectorized_dispatch_impl<kDispatchMetadataVectorBytes, NUM_DESTINATIONS>(
            src, METADATA_BYTES, rank_id, max_tokens_per_rank, payload_idx, ptrs, target_ranks,
            send_indices, worker_idx, METADATA_WORKERS);
      }
    } else {
      // Perform a single source load and fan out to each unique destination.
      for (int payload_idx = 0; payload_idx < num_payloads; payload_idx++) {
        uint8_t const* src_data = static_cast<uint8_t const*>(ptrs.src_data_ptrs[payload_idx]);
        int bytes_per_token = ptrs.payload_bytes_per_token[payload_idx];
        uint8_t const* src_ptr = src_data + local_token_idx * bytes_per_token;

        vectorized_dispatch<NUM_DESTINATIONS>(src_ptr, bytes_per_token, rank_id,
                                              max_tokens_per_rank, payload_idx, ptrs, target_ranks,
                                              send_indices);
      }
    }

    __syncthreads();
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif

  bool is_first_warp = threadIdx.x / warpSize == 0;
  if (is_first_warp) {
    int lane_id = threadIdx.x % warpSize;

    bool is_last_token = false;
    if (lane_id == 0) {
      if (local_num_tokens != 0) {
        int cnt = atomicAdd(ptrs.local_token_counter, 1);
        is_last_token = cnt + 1 == local_num_tokens;
      } else {
        is_last_token = true;
      }
    }
    is_last_token = __shfl_sync(0xffffffff, is_last_token, 0);

    if (is_last_token) {
// Store send_counters to recv_counters.
// Skip masked target ranks: their symmetric memory may be inaccessible.
#pragma unroll 1  // No unroll as one iter is typically enough
      for (int target_rank = lane_id; target_rank < ep_size; target_rank += warpSize) {
        if constexpr (ENABLE_RANK_MASK) {
          if (!is_rank_active(ptrs.active_rank_mask, target_rank)) continue;
        }
        int send_count = ptrs.send_counters[target_rank];
        ptrs.recv_counters[target_rank][rank_id] = send_count;
      }

      if constexpr (ENABLE_EPLB) {
        // Write local stats into peer buffers before the release fence below.
        // Skip masked target ranks for the same reason as above.
#pragma unroll 1
        for (int target_rank = 0; target_rank < ep_size; ++target_rank) {
          if constexpr (ENABLE_RANK_MASK) {
            if (!is_rank_active(ptrs.active_rank_mask, target_rank)) continue;
          }
          int* target_stats = ptrs.eplb_gathered_stats[target_rank];
          for (int expert_id = lane_id; expert_id < eplb_stats_num_experts; expert_id += warpSize) {
            int stat_val = ptrs.eplb_local_stats[expert_id];
            target_stats[rank_id * eplb_stats_num_experts + expert_id] = stat_val;
          }
        }
      }

#if !DISABLE_SYNC_FOR_PROFILING
      uint32_t expected_value = *ptrs.flag_val;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
      asm volatile("fence.release.sys;");
#else
      __threadfence_system();
#endif
// Signal completion to all active peers; skip dead ranks (their symmetric memory is
// unreachable).
#pragma unroll 1  // No unroll as one iter is typically enough
      for (int target_rank = lane_id; target_rank < ep_size; target_rank += warpSize) {
        if constexpr (ENABLE_RANK_MASK) {
          if (!is_rank_active(ptrs.active_rank_mask, target_rank)) continue;
        }
        uint32_t* flag_addr = &ptrs.completion_flags[target_rank][rank_id];
        asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(flag_addr), "r"(expected_value));

#if ENABLE_DEBUG_PRINT
        printf("dispatch: +++Rank %d setting completion flag to %d for rank %d\n", rank_id,
               expected_value, target_rank);
#endif
      }

// Wait for all active peers to signal; skip dead ranks (otherwise we would spin forever —
// this is the bug the rank-mask is here to prevent).
#pragma unroll 1  // No unroll
      for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize) {
        if constexpr (ENABLE_RANK_MASK) {
          if (!is_rank_active(ptrs.active_rank_mask, peer_rank)) continue;
        }
        bool flag_set = false;
        [[maybe_unused]] auto s = clock64();
        do {
          uint32_t* flag_ptr = &ptrs.completion_flags[rank_id][peer_rank];
          uint32_t flag_value;
          // Acquire load to ensure visibility of peer's release-store
          asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(flag_value) : "l"(flag_ptr));
#if ENABLE_DEBUG_PRINT
          printf(
              "dispatch: ---Rank %d received completion flag from rank %d, flag_value: %d, "
              "expected_value: "
              "%d, address: %p\n",
              rank_id, peer_rank, flag_value, expected_value, flag_ptr);
#endif
          flag_set = flag_value == expected_value;
        } while (!flag_set && !check_timeout(s));

        if (__builtin_expect(!flag_set, 0)) {
          printf("dispatch: ---Rank %d timed out waiting for completion flag from rank %d\n",
                 rank_id, peer_rank);
          asm volatile("trap;");
          return;
        }
      }
      // asm volatile("fence.acquire.sys;");
#endif
    }
  }
}

void moe_a2a_prepare_dispatch_launch(MoeA2ADispatchParams const& params) {
  launchWithPdlWhenEnabled("moeA2APrepareDispatchKernel", params.enable_pdl,
                           moeA2APrepareDispatchKernel, 1, params.ep_size, 0, params.stream,
                           params.send_counters, params.local_token_counter, params.ep_size,
                           params.flag_val, params.enable_pdl);
}

// ============================================================================
// Launch Functions
// ============================================================================

void moe_a2a_dispatch_launch(MoeA2ADispatchParams const& params) {
  // Validate parameters
  TLLM_CHECK(params.top_k > 0 && params.top_k <= kMaxTopK);
  TLLM_CHECK(params.ep_size > 0 && params.ep_size <= kMaxRanks);
  TLLM_CHECK(params.ep_rank >= 0 && params.ep_rank < params.ep_size);
  TLLM_CHECK(params.local_num_tokens >= 0);
  TLLM_CHECK(params.num_payloads > 0 && params.num_payloads <= kMaxPayloads);
  if (params.enable_rank_mask) {
    // The local rank must always be marked active in its own view of the mask; otherwise the
    // kernel itself would be running on a "dead" rank.
    TLLM_CHECK_WITH_INFO(
        (params.active_rank_mask[params.ep_rank >> 6] >> (params.ep_rank & 63)) & 1ULL,
        "active_rank_mask must mark the local ep_rank (%d) as active", params.ep_rank);
  }

  // Prepare kernel pointers struct
  DispatchKernelPointers kernel_ptrs = {};

  // Fill source data pointers and payload sizes
  for (int i = 0; i < params.num_payloads; i++) {
    kernel_ptrs.src_data_ptrs[i] = params.payloads[i].src_data;
    kernel_ptrs.payload_bytes_per_token[i] =
        params.payloads[i].element_size * params.payloads[i].elements_per_token;
  }

  // Fill receive buffer pointers
  for (int target_rank = 0; target_rank < params.ep_size; target_rank++) {
    kernel_ptrs.recv_counters[target_rank] = params.recv_counters[target_rank];
    kernel_ptrs.eplb_gathered_stats[target_rank] = params.eplb_gathered_stats[target_rank];
    for (int payload = 0; payload < params.num_payloads; payload++) {
      kernel_ptrs.recv_buffers[target_rank][payload] = params.recv_buffers[target_rank][payload];
    }
  }

  // Copy completion flag pointers
  for (int i = 0; i < params.ep_size; i++) {
    kernel_ptrs.completion_flags[i] = params.completion_flags[i];
  }
  kernel_ptrs.flag_val = params.flag_val;

  // Copy communication tracking pointers
  kernel_ptrs.send_counters = params.send_counters;
  kernel_ptrs.local_token_counter = params.local_token_counter;
  kernel_ptrs.topk_target_ranks = params.topk_target_ranks;
  kernel_ptrs.topk_send_indices = params.topk_send_indices;
  kernel_ptrs.eplb_local_stats = params.eplb_local_stats;

  // Copy active-rank bitmask into the kernel pointers struct
  for (int w = 0; w < kRankMaskWords; ++w) {
    kernel_ptrs.active_rank_mask[w] = params.active_rank_mask[w];
  }

  int block_size = tensorrt_llm::common::getEnvMoeA2ADispatchBlockSize();

  bool const use_compact_ep4 = params.ep_size == kEp4Size && params.top_k > kEp4Size;

  int max_payload_bytes_per_token = 0;
  for (int i = 0; i < params.num_payloads; ++i) {
    if (kernel_ptrs.payload_bytes_per_token[i] > max_payload_bytes_per_token) {
      max_payload_bytes_per_token = kernel_ptrs.payload_bytes_per_token[i];
    }
  }
  bool const supports_phased_nvfp4 =
      params.ep_size == kEp4Size || params.ep_size == 8 || params.ep_size == 16;
  bool const phased_nvfp4_layout =
      supports_phased_nvfp4 &&
      isPhasedNvfp4PayloadLayout(kernel_ptrs, params.num_payloads, params.top_k);

  // Small payloads do not have enough 16-byte vectors to use larger CTAs. A GB200
  // sweep found 128 threads faster than both 64 and 256 at prefill token counts.
  if ((use_compact_ep4 || phased_nvfp4_layout) &&
      max_payload_bytes_per_token <= kCompactDispatchMaxPayloadBytes &&
      block_size > kCompactDispatchBlockSize) {
    block_size = kCompactDispatchBlockSize;
  }
  bool const phase_nvfp4_payloads = phased_nvfp4_layout && block_size == kCompactDispatchBlockSize;

  // Configure kernel launch: one block per token
  int grid_size = params.local_num_tokens;
  // If local_num_tokens is 0, we still need to launch a minimal kernel to participate in the
  // synchronization.
  if (grid_size == 0) {
    grid_size = 1;
  }
  SWITCH_BOOL(params.enable_rank_mask, ENABLE_RANK_MASK, {
    SWITCH_BOOL(params.enable_eplb, EPLB_STATS, {
      if (phase_nvfp4_payloads && params.top_k == 22 &&
          kernel_ptrs.payload_bytes_per_token[0] == kNvfp4H2048ActivationBytes) {
        FLASHINFER_CHECK(kernel_ptrs.payload_bytes_per_token[1] == kNvfp4H2048ScaleBytes &&
                             kernel_ptrs.payload_bytes_per_token[2] == kTopK22MetadataBytes &&
                             kernel_ptrs.payload_bytes_per_token[3] == kTopK22MetadataBytes,
                         "Invalid H2048/top-k-22 NVFP4 dispatch payload layout");
        if (use_compact_ep4) {
          int shared_bytes = kEp4Size * (int)sizeof(int);
          auto kernel_fn = moeA2ADispatchKernel<22, EPLB_STATS, ENABLE_RANK_MASK, true, true>;
          launchWithPdlWhenEnabled("moeA2ADispatchKernel", params.enable_pdl, kernel_fn, grid_size,
                                   block_size, shared_bytes, params.stream,
                                   params.token_selected_experts, kernel_ptrs, params.num_payloads,
                                   params.max_tokens_per_rank, params.local_num_tokens,
                                   params.ep_rank, params.ep_size, params.num_experts,
                                   params.eplb_stats_num_experts, params.enable_pdl);
        } else {
          int shared_bytes = 2 * params.top_k * (int)sizeof(int);
          auto kernel_fn = moeA2ADispatchKernel<22, EPLB_STATS, ENABLE_RANK_MASK, false, true>;
          launchWithPdlWhenEnabled("moeA2ADispatchKernel", params.enable_pdl, kernel_fn, grid_size,
                                   block_size, shared_bytes, params.stream,
                                   params.token_selected_experts, kernel_ptrs, params.num_payloads,
                                   params.max_tokens_per_rank, params.local_num_tokens,
                                   params.ep_rank, params.ep_size, params.num_experts,
                                   params.eplb_stats_num_experts, params.enable_pdl);
        }
      } else if (phase_nvfp4_payloads) {
        if (use_compact_ep4) {
          int shared_bytes = kEp4Size * (int)sizeof(int);
          SWITCH_TOP_K(
              params.top_k, TOP_K, if constexpr (TOP_K > kEp4Size && TOP_K % 2 == 0) {
                auto kernel_fn =
                    moeA2ADispatchKernel<TOP_K, EPLB_STATS, ENABLE_RANK_MASK, true, false, true>;
                launchWithPdlWhenEnabled(
                    "moeA2ADispatchKernel", params.enable_pdl, kernel_fn, grid_size, block_size,
                    shared_bytes, params.stream, params.token_selected_experts, kernel_ptrs,
                    params.num_payloads, params.max_tokens_per_rank, params.local_num_tokens,
                    params.ep_rank, params.ep_size, params.num_experts,
                    params.eplb_stats_num_experts, params.enable_pdl);
              })
        } else {
          int shared_bytes = 2 * params.top_k * (int)sizeof(int);
          SWITCH_TOP_K(
              params.top_k, TOP_K, if constexpr (TOP_K > kEp4Size && TOP_K % 2 == 0) {
                auto kernel_fn =
                    moeA2ADispatchKernel<TOP_K, EPLB_STATS, ENABLE_RANK_MASK, false, false, true>;
                launchWithPdlWhenEnabled(
                    "moeA2ADispatchKernel", params.enable_pdl, kernel_fn, grid_size, block_size,
                    shared_bytes, params.stream, params.token_selected_experts, kernel_ptrs,
                    params.num_payloads, params.max_tokens_per_rank, params.local_num_tokens,
                    params.ep_rank, params.ep_size, params.num_experts,
                    params.eplb_stats_num_experts, params.enable_pdl);
              })
        }
      } else if (use_compact_ep4) {
        int shared_bytes = kEp4Size * (int)sizeof(int);
        SWITCH_TOP_K(
            params.top_k, TOP_K, if constexpr (TOP_K > kEp4Size) {
              auto kernel_fn = moeA2ADispatchKernel<TOP_K, EPLB_STATS, ENABLE_RANK_MASK, true>;
              launchWithPdlWhenEnabled(
                  "moeA2ADispatchKernel", params.enable_pdl, kernel_fn, grid_size, block_size,
                  shared_bytes, params.stream, params.token_selected_experts, kernel_ptrs,
                  params.num_payloads, params.max_tokens_per_rank, params.local_num_tokens,
                  params.ep_rank, params.ep_size, params.num_experts, params.eplb_stats_num_experts,
                  params.enable_pdl);
            })
      } else {
        int shared_bytes = 2 * params.top_k * (int)sizeof(int);
        SWITCH_TOP_K(params.top_k, TOP_K, {
          auto kernel_fn = moeA2ADispatchKernel<TOP_K, EPLB_STATS, ENABLE_RANK_MASK, false>;
          launchWithPdlWhenEnabled("moeA2ADispatchKernel", params.enable_pdl, kernel_fn, grid_size,
                                   block_size, shared_bytes, params.stream,
                                   params.token_selected_experts, kernel_ptrs, params.num_payloads,
                                   params.max_tokens_per_rank, params.local_num_tokens,
                                   params.ep_rank, params.ep_size, params.num_experts,
                                   params.eplb_stats_num_experts, params.enable_pdl);
        });
      }
    });
  })
}

// ============================================================================
// Combine kernels
// ============================================================================

// Accumulate across all valid ranks into float32 registers, then store as T (QuantMode==NONE)
// or quantize the result to MXFP8 with a per-block scale factor (QuantMode==MXFP8).
//
// InT: input element type in the recv buffer (defaults to T for same-type accumulation;
//      __nv_fp8_e4m3 for the low-precision combine path).  sizeof(InT) must divide VEC_SIZE_BYTES.
// T:   output / accumulation-cast element type (and the input type when InT == T).
//
// Unified path: load VEC_SIZE_BYTES bytes, reinterpret as InT[elems_per_vec], accumulate as
// float32, then either cast_store as T (NONE) or convert to MXFP8 + emit the block scale factor
// (MXFP8, which always uses InT == T).  Block-per-token indexing
// (threadIdx.x/blockDim.x/blockIdx.x).
template <int VEC_SIZE_BYTES, int TOP_K, typename T, typename InT = T,
          MoeA2ACombineQuantMode QuantMode = MoeA2ACombineQuantMode::NONE,
          MoeA2ACombineSwizzleSFMode SwizzleMode = MoeA2ACombineSwizzleSFMode::LINEAR>
__device__ void vectorized_combine_impl(void* output_buffer, void* sf_output, int row_idx,
                                        int row_size, int stride_per_token, int rank_id,
                                        int max_tokens_per_rank, CombineKernelPointers const& ptrs,
                                        float OutputScalarScale = 1.0f) {
  using flashinfer::vec_t;

  // elems_per_vec: number of InT elements per VEC_SIZE_BYTES-byte load (constexpr).
  constexpr int elems_per_vec = VEC_SIZE_BYTES / static_cast<int>(sizeof(InT));
  // size_per_token: byte span of one token in the recv buffer (InT-typed).
  const int size_per_token = row_size * static_cast<int>(sizeof(InT));

  // Output base pointers.  NONE writes T elements; MXFP8 writes packed fp8 bytes.
  T* dst_typed_base = nullptr;
  uint8_t* dst_bytes = nullptr;
  if constexpr (QuantMode == MoeA2ACombineQuantMode::NONE) {
    dst_typed_base = static_cast<T*>(output_buffer) + static_cast<size_t>(row_idx) * row_size;
  } else {
    // MXFP8 stores one byte per logical element; FP4 packs two e2m1 values per byte, so its
    // rows are half as wide. The accumulation still reads InT-sized inputs either way.
    size_t const bytes_per_row = QuantMode == MoeA2ACombineQuantMode::MXFP8
                                     ? static_cast<size_t>(row_size)
                                     : static_cast<size_t>(row_size) / 2;
    dst_bytes = static_cast<uint8_t*>(output_buffer) + static_cast<size_t>(row_idx) * bytes_per_row;
  }

  int const stride = blockDim.x * VEC_SIZE_BYTES;
  int const local_token_idx = blockIdx.x / MIDQ_COMBINE_SPLIT;
  // Part p of MIDQ_COMBINE_SPLIT covers a contiguous, vector-aligned byte range of the row.
  int const vecs_per_part =
      (size_per_token / VEC_SIZE_BYTES + MIDQ_COMBINE_SPLIT - 1) / MIDQ_COMBINE_SPLIT;
  int const part_begin = (blockIdx.x % MIDQ_COMBINE_SPLIT) * vecs_per_part * VEC_SIZE_BYTES;
  int const part_end = min(size_per_token, part_begin + vecs_per_part * VEC_SIZE_BYTES);

  // offset is a byte offset into the recv buffer, stepping by VEC_SIZE_BYTES bytes.
  for (int offset = part_begin + threadIdx.x * VEC_SIZE_BYTES; offset < part_end; offset += stride) {
    int logical_offset = offset / static_cast<int>(sizeof(InT));
    // Per-k vec_t<float, elems_per_vec> accumulators, zero-initialised via fill().
    vec_t<float, elems_per_vec> acc[TOP_K];

    // Pass 1: issue all TOP_K loads back-to-back without any type conversion.
    // Raw InT bytes are loaded directly into acc[k]'s register storage, reinterpreted as
    // vec_t<InT, elems_per_vec> (VEC_SIZE_BYTES bytes, fitting in the low end of acc[k]'s
    // sizeof(float)*elems_per_vec allocation).  Separating load from cast lets the compiler
    // schedule all VEC_SIZE_BYTES-byte global loads consecutively, hiding memory latency across k.
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      int target_rank = ptrs.topk_target_ranks[local_token_idx * TOP_K + k];
      int dst_idx = ptrs.topk_send_indices[local_token_idx * TOP_K + k];
      // dst_idx < 0: duplicate/dead-target sentinel already set by dispatch (which consults
      // active_rank_mask once per token). Rechecking is_rank_active here is redundant and,
      // being on the per-k vectorized hot path, measurably regresses combine throughput.
      if (dst_idx < 0) {
        acc[k].fill(0.0f);
        continue;
      }

      uint8_t const* recv_buffer = static_cast<uint8_t const*>(ptrs.recv_buffers[target_rank][0]);
      size_t base_source_rank =
          static_cast<size_t>(rank_id) * static_cast<size_t>(max_tokens_per_rank) +
          static_cast<size_t>(dst_idx);
      // stride_per_token: byte distance between tokens in the recv buffer.
      // Equals size_per_token for normal cases; may differ for FP8 in-place
      // (BF16-stride workspace but FP8-sized payload).
      size_t base_token = base_source_rank * static_cast<size_t>(stride_per_token);

      reinterpret_cast<vec_t<InT, elems_per_vec>&>(acc[k]).load(
          reinterpret_cast<InT const*>(recv_buffer + base_token + offset));
    }

    // Pass 2: in-place cast InT → float, iterating j in descending order.
    // float[j] occupies bytes [j*4, j*4+3]; InT[j] occupies [j*sizeof(InT), ...).
    // For sizeof(InT) < sizeof(float), high-j float writes land above all remaining
    // InT bytes, so descending order is always write-after-read safe.
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      if (ptrs.topk_send_indices[local_token_idx * TOP_K + k] < 0)
        continue;  // acc[k] already holds 0.0f from fill() above
#pragma unroll
      for (int j = elems_per_vec - 1; j >= 0; --j)
        acc[k][j] = static_cast<float>(reinterpret_cast<InT const*>(&acc[k])[j]);
    }
    // Reduce acc[TOP_K] into acc[0] via unrolled tree-reduction.
    // acc[k][j] uses vec_t::operator[] which returns float& — no indirection overhead.
    if constexpr (TOP_K == 22) {
#pragma unroll
      for (int j = 0; j < elems_per_vec; ++j) {
        acc[0][j] += acc[1][j];
        acc[2][j] += acc[3][j];
        acc[4][j] += acc[5][j];
        acc[6][j] += acc[7][j];
        acc[8][j] += acc[9][j];
        acc[10][j] += acc[11][j];
        acc[12][j] += acc[13][j];
        acc[14][j] += acc[15][j];
        acc[16][j] += acc[17][j];
        acc[18][j] += acc[19][j];
        acc[20][j] += acc[21][j];
      }
#pragma unroll
      for (int j = 0; j < elems_per_vec; ++j) {
        acc[0][j] += acc[2][j];
        acc[4][j] += acc[6][j];
        acc[8][j] += acc[10][j];
        acc[12][j] += acc[14][j];
        acc[16][j] += acc[18][j];
      }
#pragma unroll
      for (int j = 0; j < elems_per_vec; ++j) {
        acc[0][j] += acc[4][j];
        acc[8][j] += acc[12][j];
        acc[16][j] += acc[20][j];
      }
#pragma unroll
      for (int j = 0; j < elems_per_vec; ++j) {
        acc[0][j] += acc[8][j];
        acc[0][j] += acc[16][j];
      }
    } else if constexpr (TOP_K == 16) {
#pragma unroll
      for (int j = 0; j < elems_per_vec; ++j) {
        acc[0][j] += acc[1][j];
        acc[2][j] += acc[3][j];
        acc[4][j] += acc[5][j];
        acc[6][j] += acc[7][j];
        acc[8][j] += acc[9][j];
        acc[10][j] += acc[11][j];
        acc[12][j] += acc[13][j];
        acc[14][j] += acc[15][j];
      }
#pragma unroll
      for (int j = 0; j < elems_per_vec; ++j) {
        acc[0][j] += acc[2][j];
        acc[4][j] += acc[6][j];
        acc[8][j] += acc[10][j];
        acc[12][j] += acc[14][j];
      }
#pragma unroll
      for (int j = 0; j < elems_per_vec; ++j) {
        acc[0][j] += acc[4][j];
        acc[8][j] += acc[12][j];
      }
#pragma unroll
      for (int j = 0; j < elems_per_vec; ++j) {
        acc[0][j] += acc[8][j];
      }
    } else if constexpr (TOP_K == 10) {
#pragma unroll
      for (int j = 0; j < elems_per_vec; ++j) {
        acc[0][j] += acc[1][j];
        acc[2][j] += acc[3][j];
        acc[4][j] += acc[5][j];
        acc[6][j] += acc[7][j];
        acc[8][j] += acc[9][j];
      }
#pragma unroll
      for (int j = 0; j < elems_per_vec; ++j) {
        acc[0][j] += acc[2][j];
        acc[4][j] += acc[6][j];
      }
#pragma unroll
      for (int j = 0; j < elems_per_vec; ++j) {
        acc[0][j] += acc[4][j];
        acc[0][j] += acc[8][j];
      }
    } else if constexpr (TOP_K == 8) {
#pragma unroll
      for (int j = 0; j < elems_per_vec; ++j) {
        acc[0][j] += acc[1][j];
        acc[2][j] += acc[3][j];
        acc[4][j] += acc[5][j];
        acc[6][j] += acc[7][j];
      }
#pragma unroll
      for (int j = 0; j < elems_per_vec; ++j) {
        acc[0][j] += acc[2][j];
        acc[4][j] += acc[6][j];
      }
#pragma unroll
      for (int j = 0; j < elems_per_vec; ++j) {
        acc[0][j] += acc[4][j];
      }
    } else if constexpr (TOP_K == 6) {
#pragma unroll
      for (int j = 0; j < elems_per_vec; ++j) {
        acc[0][j] += acc[1][j];
        acc[2][j] += acc[3][j];
        acc[4][j] += acc[5][j];
      }
#pragma unroll
      for (int j = 0; j < elems_per_vec; ++j) {
        acc[0][j] += acc[2][j];
        acc[0][j] += acc[4][j];
      }
    } else if constexpr (TOP_K == 4) {
#pragma unroll
      for (int j = 0; j < elems_per_vec; ++j) {
        acc[0][j] += acc[1][j];
        acc[2][j] += acc[3][j];
      }
#pragma unroll
      for (int j = 0; j < elems_per_vec; ++j) {
        acc[0][j] += acc[2][j];
      }
    } else if constexpr (TOP_K == 2) {
#pragma unroll
      for (int j = 0; j < elems_per_vec; ++j) {
        acc[0][j] += acc[1][j];
      }
    } else if constexpr (TOP_K == 1) {
      // nothing to do
    } else {
      // Generic fallback: accumulate all into acc[0]
#pragma unroll
      for (int k = 1; k < TOP_K; ++k) {
#pragma unroll
        for (int j = 0; j < elems_per_vec; ++j) {
          acc[0][j] += acc[k][j];
        }
      }
    }
    // Epilogue: store the float accumulator as T (NONE) or quantize to MXFP8 + scale factor.
    if constexpr (QuantMode == MoeA2ACombineQuantMode::NONE) {
      // cast_store: converts float->T element-by-element then writes via vectorized int4 store.
      acc[0].cast_store(dst_typed_base + offset / static_cast<int>(sizeof(InT)));
    } else {
      constexpr uint32_t sf_vec_size = QuantMode == MoeA2ACombineQuantMode::NVFP4 ? 16 : 32;
      constexpr uint32_t threads_per_sf = sf_vec_size / elems_per_vec;
      uint8_t scale;
      auto store_sf = [&]() {
        int64_t sf_offset;
        const int64_t num_vecs_per_row = (row_size + sf_vec_size - 1) / sf_vec_size;
        const int64_t sf_idx = logical_offset / sf_vec_size;
        if constexpr (SwizzleMode == MoeA2ACombineSwizzleSFMode::LINEAR) {
          sf_offset = row_idx * num_vecs_per_row + sf_idx;
        } else if constexpr (SwizzleMode == MoeA2ACombineSwizzleSFMode::SWIZZLE_128x4) {
          sf_offset = get_sf_out_offset_128x4(row_idx, sf_idx, num_vecs_per_row);
        } else {  // SWIZZLE_8x4
          sf_offset = get_sf_out_offset_8x4(row_idx, sf_idx, num_vecs_per_row);
        }
        if (threadIdx.x % threads_per_sf == 0) {
          reinterpret_cast<uint8_t*>(sf_output)[sf_offset] = scale;
        }
      };
      // Cast the float32 accumulator down to T to feed the MXFP8/FP4 converter's PackedVec input.
      tensorrt_llm::kernels::PackedVec<T, elems_per_vec> packed_vec;
      T* packed_elems = reinterpret_cast<T*>(&packed_vec);
#pragma unroll
      for (int j = 0; j < elems_per_vec; ++j) {
        packed_elems[j] = static_cast<T>(acc[0][j]);
      }
      if constexpr (QuantMode == MoeA2ACombineQuantMode::MXFP8) {
        static_assert(elems_per_vec == 8, "MXFP8 quantization requires 8 elements per vector");
        uint64_t fp8x8 =
            tensorrt_llm::kernels::cvt_warp_fp16_to_mxfp8<T, 32, elems_per_vec>(packed_vec, &scale);
        reinterpret_cast<uint64_t*>(dst_bytes)[logical_offset / elems_per_vec] = fp8x8;
      } else if constexpr (QuantMode == MoeA2ACombineQuantMode::NVFP4 ||
                           QuantMode == MoeA2ACombineQuantMode::MXFP4) {
        static_assert(elems_per_vec == 8 || elems_per_vec == 16,
                      "FP4 quantization requires 8 or 16 elements per vector");
        constexpr int SF_VEC_SIZE = QuantMode == MoeA2ACombineQuantMode::MXFP4 ? 32 : 16;
        auto fp4_packed = tensorrt_llm::kernels::cvt_warp_fp16_to_fp4 < T, SF_VEC_SIZE,
             elems_per_vec,
             QuantMode == MoeA2ACombineQuantMode::MXFP4 > (packed_vec, OutputScalarScale, &scale);
        // cvt_warp_fp16_to_fp4 returns uint32_t for 8 elems (4 packed bytes) and uint64_t for 16
        // (8 packed bytes); store at the matching width so packed rows stay contiguous and fit the
        // dim/2-byte output buffer (a wider store would overflow into the next token's row).
        reinterpret_cast<decltype(fp4_packed)*>(dst_bytes)[logical_offset / elems_per_vec] =
            fp4_packed;
      }
      store_sf();
    }
  }
}

// Wrapper that selects vector width based on size_per_token alignment.
// QuantMode==NONE writes T (optionally upcast from an fp8 InT); QuantMode==MXFP8 writes packed
// fp8 + scale factors (always InT == T, fixed 16-byte / 8-element vectors).
// stride_per_token: byte distance between tokens in the recv buffer (may differ from
// size_per_token when FP8 in-place uses a BF16-stride workspace with an FP8-sized payload).
// InT: input element type in recv buffer (defaults to T for same-type accumulation).
template <int TOP_K, typename T, typename InT = T,
          MoeA2ACombineQuantMode QuantMode = MoeA2ACombineQuantMode::NONE,
          MoeA2ACombineSwizzleSFMode SwizzleMode = MoeA2ACombineSwizzleSFMode::LINEAR>
__device__ void vectorized_combine(void* output_buffer, void* sf_output, int row_idx, int row_size,
                                   int stride_per_token, int rank_id, int max_tokens_per_rank,
                                   CombineKernelPointers const& ptrs,
                                   float OutputScalarScale = 1.0f) {
  const int size_per_token = row_size * static_cast<int>(sizeof(InT));
  if constexpr (QuantMode != MoeA2ACombineQuantMode::NONE) {
    // MXFP8/FP4 require 16-byte vectors (8 or 16 elements per vec), always InT == T.
    vectorized_combine_impl<16, TOP_K, T, InT, QuantMode, SwizzleMode>(
        output_buffer, sf_output, row_idx, row_size, stride_per_token, rank_id, max_tokens_per_rank,
        ptrs, OutputScalarScale);
  } else {
    // Each branch is guarded by if constexpr (sizeof(InT) <= VEC_SIZE_BYTES) so the compiler
    // never instantiates vectorized_combine_impl with elems_per_vec == 0.
    if (size_per_token % 16 == 0) {
      if constexpr (static_cast<int>(sizeof(InT)) <= 16)
        vectorized_combine_impl<16, TOP_K, T, InT>(output_buffer, nullptr, row_idx, row_size,
                                                   stride_per_token, rank_id, max_tokens_per_rank,
                                                   ptrs);
    } else if (size_per_token % 8 == 0) {
      if constexpr (static_cast<int>(sizeof(InT)) <= 8)
        vectorized_combine_impl<8, TOP_K, T, InT>(output_buffer, nullptr, row_idx, row_size,
                                                  stride_per_token, rank_id, max_tokens_per_rank,
                                                  ptrs);
    } else if (size_per_token % 4 == 0) {
      if constexpr (static_cast<int>(sizeof(InT)) <= 4)
        vectorized_combine_impl<4, TOP_K, T, InT>(output_buffer, nullptr, row_idx, row_size,
                                                  stride_per_token, rank_id, max_tokens_per_rank,
                                                  ptrs);
    } else if (size_per_token % 2 == 0) {
      if constexpr (static_cast<int>(sizeof(InT)) <= 2)
        vectorized_combine_impl<2, TOP_K, T, InT>(output_buffer, nullptr, row_idx, row_size,
                                                  stride_per_token, rank_id, max_tokens_per_rank,
                                                  ptrs);
    } else {
      if constexpr (static_cast<int>(sizeof(InT)) <= 1)
        vectorized_combine_impl<1, TOP_K, T, InT>(output_buffer, nullptr, row_idx, row_size,
                                                  stride_per_token, rank_id, max_tokens_per_rank,
                                                  ptrs);
    }
  }
}

// ---- vec_convert: per-vector type conversion, specialized by PTX where available ----
// Generic: SrcT -> float -> DstT (all architectures, all type combinations).
template <size_t VEC_SIZE, typename SrcT, typename DstT>
__device__ __forceinline__ void vec_convert(flashinfer::vec_t<DstT, VEC_SIZE>& out,
                                            flashinfer::vec_t<SrcT, VEC_SIZE> const& in) {
#pragma unroll
  for (int j = 0; j < VEC_SIZE; ++j) out[j] = DstT(static_cast<float>(in[j]));
}

// BF16 -> FP8 e4m3: use CUDA intrinsic (SM100+, Blackwell).
// Inline PTX "cvt.rn.satfinite.e4m3x2.bf16x2 %h, %r" is rejected by SM100a ptxas
// ("Unexpected instruction types for cvt") because SM100a requires a 32-bit output
// register for this instruction.  __nv_fp8x2_e4m3(bfloat162) emits the correct form.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
template <size_t VEC_SIZE, std::enable_if_t<(VEC_SIZE % 2 == 0), int> = 0>
__device__ __forceinline__ void vec_convert(flashinfer::vec_t<__nv_fp8_e4m3, VEC_SIZE>& out,
                                            flashinfer::vec_t<__nv_bfloat16, VEC_SIZE> const& in) {
  __nv_fp8x2_e4m3* out_fp8x2 = reinterpret_cast<__nv_fp8x2_e4m3*>(&out);
  __nv_bfloat162 const* in_bf16x2 = reinterpret_cast<__nv_bfloat162 const*>(&in);
#pragma unroll
  for (int p = 0; p < static_cast<int>(VEC_SIZE) / 2; ++p)
    out_fp8x2[p] = __nv_fp8x2_e4m3(in_bf16x2[p]);
}
#endif

// FP16 -> FP8 e4m3: paired PTX cvt.rn.satfinite.e4m3x2.f16x2 (SM89+, Hopper).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
template <size_t VEC_SIZE, std::enable_if_t<(VEC_SIZE % 2 == 0), int> = 0>
__device__ __forceinline__ void vec_convert(flashinfer::vec_t<__nv_fp8_e4m3, VEC_SIZE>& out,
                                            flashinfer::vec_t<half, VEC_SIZE> const& in) {
  uint32_t const* src_u32 = reinterpret_cast<uint32_t const*>(&in);
  uint16_t* dst_u16 = reinterpret_cast<uint16_t*>(&out);
#pragma unroll
  for (int p = 0; p < VEC_SIZE / 2; ++p) {
    uint16_t d;
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(d) : "r"(src_u32[p]));
    dst_u16[p] = d;
  }
}
#endif

// ---- vectorized_quant_impl: load -> sync -> convert -> store ----
// VEC_SIZE is in elements (not bytes), so both SrcT and DstT vectors hold VEC_SIZE values.
// Block-per-token: threadIdx.x indexes vectors within the token, blockDim.x is the stride.
template <int VEC_SIZE, typename SrcT, typename DstT>
__device__ void vectorized_quant_impl(DstT* dst, SrcT const* src, int num_elements) {
  using flashinfer::vec_t;

  // num_elements is a multiple of VEC_SIZE here (selected by the wrapper).
  int const num_vecs = num_elements / VEC_SIZE;
  // Round the iteration bound up to a multiple of blockDim.x so every thread reaches the
  // __syncthreads() the same number of times (a divergent barrier would deadlock when
  // num_vecs is not a multiple of blockDim.x, e.g. hidden=7168 with 256 threads).
  int const vec_iters = ((num_vecs + blockDim.x - 1) / blockDim.x) * blockDim.x;

  for (int iter = threadIdx.x; iter < vec_iters; iter += blockDim.x) {
    bool const active = iter < num_vecs;
    int const e = iter * VEC_SIZE;
    vec_t<SrcT, VEC_SIZE> in_vec;
    if (active) in_vec.load(src + e);

    // Sync so all threads finish loading inputs before any thread writes output.  This avoids
    // write-after-read hazards in the FP8 in-place case where this kernel's output is read as
    // input on the next iteration.  All threads participate (active guards the work, not the sync).
    __syncthreads();

    if (active) {
      vec_t<DstT, VEC_SIZE> out_vec;
      vec_convert(out_vec, in_vec);
      out_vec.store(dst + e);
    }
  }
}

template <typename SrcT, typename DstT>
__device__ void vectorized_quant(DstT* dst, SrcT const* src, int num_elements) {
  if (num_elements % 16 == 0)
    vectorized_quant_impl<16, SrcT, DstT>(dst, src, num_elements);
  else if (num_elements % 8 == 0)
    vectorized_quant_impl<8, SrcT, DstT>(dst, src, num_elements);
  else if (num_elements % 4 == 0)
    vectorized_quant_impl<4, SrcT, DstT>(dst, src, num_elements);
  else if (num_elements % 2 == 0)
    vectorized_quant_impl<2, SrcT, DstT>(dst, src, num_elements);
  else
    vectorized_quant_impl<1, SrcT, DstT>(dst, src, num_elements);
}

// LOW_PRECISION=false: vectorized byte-copy (SrcT = payload dtype).
// LOW_PRECISION=true:  vectorized SrcT->FP8 quantization via vectorized_quant<SrcT, fp8_e4m3>.
// stride_per_token: byte distance between tokens in recv_buffer_bytes (host-computed, avoids
//   per-thread recomputation):
//   - FP8 external payload: elements_per_token x 1  (compact FP8 layout)
//   - FP8 in-place / byte-copy: elements_per_token x sizeof(SrcT)  (payload-dtype stride)
// Copy payload to recv buffer; one block per token.
template <bool LOW_PRECISION, typename SrcT>
__global__ void moeA2APrepareCombineKernel(uint8_t* recv_buffer_bytes, void const* payload,
                                           int elements_per_token, int ep_size,
                                           int max_tokens_per_rank, uint32_t* flag_val_ptr,
                                           int const* recv_counters, int stride_per_token,
                                           bool enable_pdl) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (enable_pdl) cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
#endif

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    // Increment flag_val for this combine round
    *flag_val_ptr = *flag_val_ptr + 1;
  }

  // Copy path: null payload means data is already in workspace — nothing to do.
  if (!LOW_PRECISION && payload == nullptr) return;

  int global_token_idx = blockIdx.x;

  int global_token_num = ep_size * max_tokens_per_rank;
  if (global_token_idx >= global_token_num) return;

  // Map global_token_idx to (rank_idx, local_token_idx)
  int rank_idx = global_token_idx / max_tokens_per_rank;
  int local_token_idx = global_token_idx % max_tokens_per_rank;

  // Skip invalid tokens beyond per-rank recv count
  if (local_token_idx >= recv_counters[rank_idx]) return;

  size_t const token_offset = static_cast<size_t>(global_token_idx) * stride_per_token;

  if constexpr (LOW_PRECISION) {
    // Source pointer: external payload or in-place from workspace.
    SrcT const* src_ptr = (payload != nullptr)
                              ? static_cast<SrcT const*>(payload) +
                                    static_cast<size_t>(global_token_idx) * elements_per_token
                              : reinterpret_cast<SrcT const*>(recv_buffer_bytes + token_offset);

    // Destination: stride_per_token encodes the correct layout for both paths
    // (compact FP8 for external, payload-dtype stride for in-place).
    __nv_fp8_e4m3* dst_ptr = reinterpret_cast<__nv_fp8_e4m3*>(recv_buffer_bytes + token_offset);

    vectorized_quant<SrcT, __nv_fp8_e4m3>(dst_ptr, src_ptr, elements_per_token);
  } else {
    // Generic byte copy (payload guaranteed non-null by early return above).
    vectorized_copy(recv_buffer_bytes + token_offset,
                    static_cast<uint8_t const*>(payload) + token_offset, stride_per_token);
  }
}

// ============================================================================
// Generic Combine Kernel Implementation (Templated by data type)
// ============================================================================

template <typename T, int TOP_K, bool ENABLE_RANK_MASK,
          MoeA2ACombineQuantMode QuantMode = MoeA2ACombineQuantMode::NONE,
          MoeA2ACombineSwizzleSFMode SwizzleMode = MoeA2ACombineSwizzleSFMode::LINEAR>
__global__ void moeA2ACombineKernel(
    const CombineKernelPointers ptrs,  // Combine-specific struct, src_data_ptrs[0] is output
    int max_tokens_per_rank, int elements_per_token, int stride_per_token, int local_num_tokens,
    int rank_id, int ep_size, bool enable_pdl, float OutputScalarScale) {
  int local_token_idx = blockIdx.x / MIDQ_COMBINE_SPLIT;

  if (local_num_tokens == 0) {
    // Special case: If local_num_tokens == 0,
    // we need to keep the threads where local_token_idx == 0 alive to participate in the
    // synchronization. Other threads should return.
    if (local_token_idx > 0) return;
  } else {
    // Threads that do not have a token to process should return.
    if (local_token_idx >= local_num_tokens) return;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (enable_pdl) cudaGridDependencySynchronize();
#endif

#if !DISABLE_SYNC_FOR_PROFILING
  // In-kernel readiness synchronization at start of combine:
  // - One warp signals readiness to all peers with current flag_val.
  // - The first warp of each block waits for all peers' readiness (equality), then __syncthreads.
  bool is_first_warp = threadIdx.x / warpSize == 0;
  if (is_first_warp) {
    int lane_id = threadIdx.x % warpSize;
    uint32_t expected_value = *ptrs.flag_val;

    if (blockIdx.x == 0) {
      // asm volatile("fence.release.sys;");
      // Signal readiness to all active peers; skip dead ranks (their symmetric memory is
      // unreachable).
#pragma unroll 1  // No unroll
      for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize) {
        if constexpr (ENABLE_RANK_MASK) {
          if (!is_rank_active(ptrs.active_rank_mask, peer_rank)) continue;
        }
        uint32_t* flag_addr = &ptrs.completion_flags[peer_rank][rank_id];
        asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(flag_addr), "r"(expected_value));
#if ENABLE_DEBUG_PRINT
        printf("combine: +++Rank %d setting completion flag to %d for rank %d\n", rank_id,
               expected_value, peer_rank);
#endif
      }
    }

// Wait for all active peers to signal; skip dead ranks (otherwise we would spin forever —
// this is the bug the rank-mask is here to prevent).
#pragma unroll 1  // No unroll
    for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize) {
      if constexpr (ENABLE_RANK_MASK) {
        if (!is_rank_active(ptrs.active_rank_mask, peer_rank)) continue;
      }
      bool flag_set = false;
      [[maybe_unused]] auto s = clock64();
      do {
        uint32_t* flag_ptr = &ptrs.completion_flags[rank_id][peer_rank];
        uint32_t flag_value;
        // Acquire load to ensure visibility of peer's release-store
        asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(flag_value) : "l"(flag_ptr));
#if ENABLE_DEBUG_PRINT
        printf(
            "combine: ---Rank %d received completion flag from rank %d, flag_value: %d, "
            "expected_value: %d, "
            "address: %p\n",
            rank_id, peer_rank, flag_value, expected_value, flag_ptr);
#endif
        flag_set = flag_value == expected_value;
      } while (!flag_set && !check_timeout(s));

      if (__builtin_expect(!flag_set, 0)) {
        printf("combine: ---Rank %d timed out waiting for completion flag from rank %d\n", rank_id,
               peer_rank);
        asm volatile("trap;");
        return;
      }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("fence.acquire.sys;");
#else
    __threadfence_system();
#endif
  }
  __syncthreads();
#endif

  if (local_num_tokens == 0) return;

  // Dispatch the combine path:
  //   - T == fp8_e4m3: low-precision combine, FP8 recv buffer -> BF16 output (InT=fp8, NONE quant).
  //   - otherwise: same-type accumulate, then store as T (NONE) or quantize to MXFP8/FP4
  //   (QuantMode).
  // vectorized_combine derives the per-token output pointer from row_idx (= local_token_idx).
  if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
    // src_data_ptrs[0] points to a BF16 output buffer (set by moeA2ACombineOp for low precision).
    vectorized_combine<TOP_K, __nv_bfloat16, __nv_fp8_e4m3>(
        ptrs.src_data_ptrs[0], nullptr, local_token_idx, elements_per_token, stride_per_token,
        rank_id, max_tokens_per_rank, ptrs);
  } else {
    vectorized_combine<TOP_K, T, T, QuantMode, SwizzleMode>(
        ptrs.src_data_ptrs[0], ptrs.output_scales, local_token_idx, elements_per_token,
        stride_per_token, rank_id, max_tokens_per_rank, ptrs, OutputScalarScale);
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (enable_pdl) cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void moe_a2a_prepare_combine_launch(MoeA2ACombineParams const& params) {
  constexpr int kBlockSize = 256;

  // FP8 in-place (use_low_precision, prepare_payload==nullptr): each CTA writes FP8 at the
  // BF16-stride position, so CTAs never race — all tokens must be processed.  Copy path with a
  // null payload is a no-op; 1 block suffices for the flag increment only.
  int grid = (params.use_low_precision || params.prepare_payload != nullptr)
                 ? params.ep_size * params.max_tokens_per_rank
                 : 1;

  uint8_t* recv_buffer_bytes =
      static_cast<uint8_t*>(const_cast<void*>(params.recv_buffers[params.ep_rank]));
  void const* payload = params.prepare_payload;

  // stride_per_token (host-computed, avoids per-thread recompute):
  //   FP8 external: EPT x 1            (compact FP8, dst packed tightly)
  //   FP8 in-place / byte-copy: EPT x sizeof(SrcT)  (payload-dtype stride)
  SWITCH_BOOL(params.use_low_precision, LOW_PRECISION, {
    SWITCH_DTYPE(params.dtype, SrcT, {
      bool const low_precision_staged = LOW_PRECISION && (params.prepare_payload != nullptr);
      int const stride_per_token = low_precision_staged
                                       ? params.elements_per_token
                                       : params.elements_per_token * static_cast<int>(sizeof(SrcT));
      auto kernel_fn = moeA2APrepareCombineKernel<LOW_PRECISION, SrcT>;
      launchWithPdlWhenEnabled("moeA2APrepareCombineKernel", params.enable_pdl, kernel_fn, grid,
                               kBlockSize, 0, params.stream, recv_buffer_bytes, payload,
                               params.elements_per_token, params.ep_size,
                               params.max_tokens_per_rank, params.flag_val, params.recv_counters,
                               stride_per_token, params.enable_pdl);
    });
  });
}

// ============================================================================
// Combine Launch Function
// ============================================================================

void moe_a2a_combine_launch(MoeA2ACombineParams const& params) {
  // Validate parameters
  TLLM_CHECK(params.top_k > 0 && params.top_k <= kMaxTopK);
  TLLM_CHECK(params.ep_size > 0 && params.ep_size <= kMaxRanks);
  TLLM_CHECK(params.ep_rank >= 0 && params.ep_rank < params.ep_size);
  TLLM_CHECK(params.local_num_tokens >= 0);
  TLLM_CHECK(params.elements_per_token > 0);
  if (params.enable_rank_mask) {
    // The local rank must always be marked active in its own view of the mask; otherwise the
    // kernel itself would be running on a "dead" rank.
    TLLM_CHECK_WITH_INFO(
        (params.active_rank_mask[params.ep_rank >> 6] >> (params.ep_rank & 63)) & 1ULL,
        "active_rank_mask must mark the local ep_rank (%d) as active", params.ep_rank);
  }

  // Configure kernel launch: one block per token
  int const kBlockSize = tensorrt_llm::common::getEnvMoeA2ACombineBlockSize();
  int grid_size_block = params.local_num_tokens * MIDQ_COMBINE_SPLIT;
  // If local_num_tokens is 0, we still need to launch a minimal kernel to participate in the
  // synchronization.
  if (grid_size_block == 0) {
    grid_size_block = MIDQ_COMBINE_SPLIT;
  }

  // Prepare kernel pointers struct for combine
  CombineKernelPointers kernel_ptrs = {};  // Zero-initialize

  // Set output data pointer in src_data_ptrs[0]
  kernel_ptrs.src_data_ptrs[0] = params.output_data;
  kernel_ptrs.output_scales = params.output_scales;

  // Fill recv buffer pointers
  for (int rank = 0; rank < params.ep_size; rank++) {
    kernel_ptrs.recv_buffers[rank][0] = params.recv_buffers[rank];
  }

  // Copy completion flag pointers
  for (int i = 0; i < params.ep_size; i++) {
    kernel_ptrs.completion_flags[i] = params.completion_flags[i];
  }
  kernel_ptrs.flag_val = params.flag_val;

  // Copy communication tracking pointers
  kernel_ptrs.topk_target_ranks = params.topk_target_ranks;
  kernel_ptrs.topk_send_indices = params.topk_send_indices;

  // Copy active-rank bitmask into the kernel pointers struct
  for (int w = 0; w < kRankMaskWords; ++w) {
    kernel_ptrs.active_rank_mask[w] = params.active_rank_mask[w];
  }

  int grid = grid_size_block;  // one block per token

  // stride_per_token: byte distance between tokens in the recv buffer.
  //   FP8 external payload: EPT × 1            (compact FP8 layout)
  //   FP8 in-place / non-FP8: EPT × sizeof(PayloadT)  (payload-dtype stride)
  bool const low_precision_staged = params.use_low_precision && (params.prepare_payload != nullptr);
  int stride_per_token;
  SWITCH_DTYPE(params.dtype, PayloadT, {
    stride_per_token = low_precision_staged
                           ? params.elements_per_token
                           : params.elements_per_token * static_cast<int>(sizeof(PayloadT));
  });

  // When use_low_precision is set the recv buffers contain FP8 data regardless of params.dtype,
  // so dispatch the FP8 accumulation kernel in that case.
  auto const effective_dtype = params.use_low_precision ? nvinfer1::DataType::kFP8 : params.dtype;

  // Launch appropriate kernel with compact macros
  SWITCH_BOOL(params.enable_rank_mask, ENABLE_RANK_MASK, {
    SWITCH_DTYPE(effective_dtype, TKernelType, {
      SWITCH_TOP_K(params.top_k, TOP_K, {
        SWITCH_QUANT_MODE(TKernelType, params.quant_mode, QUANT_MODE, {
          SWITCH_SWIZZLE_MODE(params.swizzle_mode, SWIZZLE_MODE, {
            auto kernel_fn =
                moeA2ACombineKernel<TKernelType, TOP_K, ENABLE_RANK_MASK, QUANT_MODE, SWIZZLE_MODE>;
            launchWithPdlWhenEnabled("moeA2ACombineKernel", params.enable_pdl, kernel_fn, grid,
                                     kBlockSize, 0, params.stream, kernel_ptrs,
                                     params.max_tokens_per_rank, params.elements_per_token,
                                     stride_per_token, params.local_num_tokens, params.ep_rank,
                                     params.ep_size, params.enable_pdl, params.output_scalar_scale);
          });
        });
      });
    });
  })
}

// Kernel to sanitize expert ids for invalid tokens
__global__ void moeA2ASanitizeExpertIdsKernel(int32_t* expert_ids_ptr,
                                              int32_t const* recv_counters_ptr, int ep_size,
                                              int max_tokens_per_rank, int top_k,
                                              int32_t invalid_id, bool enable_pdl) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (enable_pdl) cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
#endif
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_tokens = ep_size * max_tokens_per_rank;
  if (tid >= total_tokens) return;

  int source_rank = tid / max_tokens_per_rank;
  int token_idx = tid % max_tokens_per_rank;

  if (token_idx >= recv_counters_ptr[source_rank]) {
    int32_t* token_expert_ids = expert_ids_ptr + tid * top_k;
    for (int k = 0; k < top_k; ++k) {
      token_expert_ids[k] = invalid_id;
    }
  }
}

void moe_a2a_sanitize_expert_ids_launch(int32_t* expert_ids, int32_t const* recv_counters,
                                        int32_t invalid_id, int ep_size, int max_tokens_per_rank,
                                        int top_k, cudaStream_t stream, bool enable_pdl) {
  constexpr int kBlockSize = 256;
  int total_tokens = ep_size * max_tokens_per_rank;
  int grid = ceilDiv(total_tokens, kBlockSize);
  launchWithPdlWhenEnabled("moeA2ASanitizeExpertIdsKernel", enable_pdl,
                           moeA2ASanitizeExpertIdsKernel, grid, kBlockSize, 0, stream, expert_ids,
                           recv_counters, ep_size, max_tokens_per_rank, top_k, invalid_id,
                           enable_pdl);
}

}  // namespace tensorrt_llm::kernels::moe_alltoall
