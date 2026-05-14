/*
 * Fused metadata copy kernel for NSA backend CUDA graph replay.
 * JIT-compiled version for python/sglang/jit_kernel.
 *
 * OVERVIEW:
 * This kernel fuses multiple tensor copy operations (cache_seqlens, cu_seqlens_k,
 * page_table, nsa metadata, and optional FlashMLA metadata) into single kernel
 * launches, significantly reducing kernel launch overhead and improving CUDA
 * graph replay performance during inference.
 *
 * PERFORMANCE BENEFITS:
 * - Single kernel launch vs. multiple separate copies (3-10x faster)
 * - Optimized memory coalescing and SM utilization
 * - __grid_constant__ parameter passing via constant memory
 * - Especially beneficial in CUDA graph replay scenarios
 *
 * DESIGN:
 * - Unified kernel supporting all forward modes (DECODE, TARGET_VERIFY, DRAFT_EXTEND)
 * - Structured parameter passing (SourcePointers/DestinationPointers) for clarity
 * - Template parameters (HAS_REAL_PAGE_TABLE, HAS_FLASHMLA) for compile-time optimization
 * - Multi-backend variant copies to 3 destinations in one kernel (for speculative decoding)
 *
 * USAGE:
 * This header is included by JIT compilation system. The FusedMetadataCopyKernel
 * and FusedMetadataCopyMultiKernel wrapper structs provide the Python-accessible interface.
 */

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>  // for std::min
#include <cuda_runtime.h>

// Forward mode enum (must match Python ForwardMode in sglang/srt/layers/attention/nsa_backend.py)
enum ForwardModeEnum { DECODE = 0, TARGET_VERIFY = 1, DRAFT_EXTEND = 2 };

/**
 * Source pointers for metadata copy operations.
 * Groups all source tensor pointers for cleaner parameter passing.
 * Some pointers may be nullptr depending on forward mode and feature flags.
 */
struct SourcePointers {
  const int32_t* __restrict__ cache_seqlens;        // [bs] sequence lengths in cache
  const int32_t* __restrict__ cu_seqlens_k;         // [bs+1] cumulative sequence lengths
  const int32_t* __restrict__ page_indices;         // page table indices
  const int32_t* __restrict__ nsa_cache_seqlens;    // NSA-specific cache lengths
  const int32_t* __restrict__ seqlens_expanded;     // expanded sequence lengths (TARGET_VERIFY/DRAFT_EXTEND only)
  const int32_t* __restrict__ nsa_cu_seqlens_k;     // NSA cumulative sequence lengths
  const int32_t* __restrict__ real_page_table;      // optional real page table
  const int32_t* __restrict__ flashmla_num_splits;  // optional FlashMLA split counts
  const int32_t* __restrict__ flashmla_metadata;    // optional FlashMLA metadata
};

/**
 * Destination pointers for metadata copy operations.
 * Groups all destination tensor pointers for cleaner parameter passing.
 * Layout matches SourcePointers for consistency.
 */
struct DestinationPointers {
  int32_t* __restrict__ cache_seqlens;        // [bs] sequence lengths in cache
  int32_t* __restrict__ cu_seqlens_k;         // [bs+1] cumulative sequence lengths
  int32_t* __restrict__ page_table_1;         // page table (note: different name from source)
  int32_t* __restrict__ nsa_cache_seqlens;    // NSA-specific cache lengths
  int32_t* __restrict__ seqlens_expanded;     // expanded sequence lengths (TARGET_VERIFY/DRAFT_EXTEND only)
  int32_t* __restrict__ nsa_cu_seqlens_k;     // NSA cumulative sequence lengths
  int32_t* __restrict__ real_page_table;      // optional real page table
  int32_t* __restrict__ flashmla_num_splits;  // optional FlashMLA split counts
  int32_t* __restrict__ flashmla_metadata;    // optional FlashMLA metadata
};

/**
 * Parameter structure for single-backend fused metadata copy kernel.
 * Passed via __grid_constant__ for efficient constant memory access.
 */
struct FusedMetadataCopyParams {
  SourcePointers src;       // Source tensor pointers
  DestinationPointers dst;  // Destination tensor pointers

  // Kernel parameters
  int forward_mode;                // 0=DECODE, 1=TARGET_VERIFY, 2=DRAFT_EXTEND
  int bs;                          // Batch size
  int max_len;                     // Max length for DECODE mode
  int max_seqlen_k;                // Max sequence length for TARGET_VERIFY/DRAFT_EXTEND
  int seqlens_expanded_size;       // Size of expanded sequence lengths
  int page_indices_rows;           // Number of rows in page_indices
  int page_table_1_stride;         // Stride for page_table_1
  int real_page_table_cols;        // Columns in real_page_table
  int real_page_table_dst_stride;  // Stride for destination real_page_table
  int flashmla_metadata_size;      // Size of FlashMLA metadata
};

/**
 * Parameter structure for multi-backend fused metadata copy kernel.
 * Enables copying from one source to three destinations in a single kernel launch.
 * Used for speculative decoding with multiple draft backends.
 */
struct FusedMetadataCopyMultiParams {
  SourcePointers src;        // Source pointers (shared across all backends)
  DestinationPointers dst0;  // Backend 0 destination pointers
  DestinationPointers dst1;  // Backend 1 destination pointers
  DestinationPointers dst2;  // Backend 2 destination pointers

  // Kernel parameters
  int bs;                          // Batch size
  int max_len;                     // Max length (DECODE mode only)
  int seqlens_expanded_size;       // Size of expanded sequence lengths
  int page_table_1_stride;         // Stride for page_table_1
  int real_page_table_cols;        // Columns in real_page_table
  int real_page_table_dst_stride;  // Stride for destination real_page_table
  int flashmla_metadata_size;      // Size of FlashMLA metadata
};

/**
 * Unified kernel for all forward modes (DECODE, TARGET_VERIFY, DRAFT_EXTEND).
 * Uses runtime branches for mode selection, with template parameters for
 * compile-time optimization of optional features.
 *
 * DESIGN:
 * - Runtime branches (forward_mode) handle mode-specific logic
 * - Template parameters (HAS_*) eliminate unused feature code at compile time
 * - Structured parameters (SourcePointers/DestinationPointers) passed via constant memory
 *
 * Used by FusedMetadataCopyKernel for single-backend metadata copy.
 *
 * @tparam HAS_REAL_PAGE_TABLE Compile-time flag for real_page_table support
 * @tparam HAS_FLASHMLA Compile-time flag for FlashMLA metadata support
 */
template <bool HAS_REAL_PAGE_TABLE, bool HAS_FLASHMLA>
__global__ void fused_metadata_copy_kernel(const FusedMetadataCopyParams __grid_constant__ params) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;

  // Unpack parameters for readability
  const auto& src = params.src;
  const auto& dst = params.dst;
  const int forward_mode = params.forward_mode;
  const int bs = params.bs;
  const int max_len = params.max_len;
  const int max_seqlen_k = params.max_seqlen_k;
  const int seqlens_expanded_size = params.seqlens_expanded_size;
  const int page_indices_rows = params.page_indices_rows;
  const int page_table_1_stride = params.page_table_1_stride;
  const int real_page_table_cols = params.real_page_table_cols;
  const int real_page_table_dst_stride = params.real_page_table_dst_stride;
  const int flashmla_metadata_size = params.flashmla_metadata_size;

  // Copy cache_seqlens (bs elements) - common to all modes
#pragma unroll 8
  for (int i = tid; i < bs; i += total_threads) {
    dst.cache_seqlens[i] = src.cache_seqlens[i];
  }

  // Copy cu_seqlens_k (skip first element) - common to all modes
#pragma unroll 8
  for (int i = tid; i < bs; i += total_threads) {
    dst.cu_seqlens_k[i + 1] = src.cu_seqlens_k[i + 1];
  }

  // Branch 1: page_table copy (different dimensions per mode)
  if (forward_mode == 0) {  // DECODE
    int page_table_elements = bs * max_len;
#pragma unroll 4
    for (int i = tid; i < page_table_elements; i += total_threads) {
      int row = i / max_len;
      int col = i % max_len;
      dst.page_table_1[row * page_table_1_stride + col] = src.page_indices[i];
    }
  } else {  // TARGET_VERIFY or DRAFT_EXTEND
    int page_table_elements = page_indices_rows * max_seqlen_k;
#pragma unroll 4
    for (int i = tid; i < page_table_elements; i += total_threads) {
      int row = i / max_seqlen_k;
      int col = i % max_seqlen_k;
      dst.page_table_1[row * page_table_1_stride + col] = src.page_indices[i];
    }
  }

  // Branch 2: seqlens_expanded copy (only for TARGET_VERIFY/DRAFT_EXTEND)
  if (forward_mode != 0) {  // TARGET_VERIFY or DRAFT_EXTEND
#pragma unroll 4
    for (int i = tid; i < seqlens_expanded_size; i += total_threads) {
      dst.seqlens_expanded[i] = src.seqlens_expanded[i];
    }
  }

  // Branch 3: NSA metadata copy (different loop sizes per mode)
  if (forward_mode == 0) {  // DECODE
#pragma unroll 8
    for (int i = tid; i < bs; i += total_threads) {
      dst.nsa_cache_seqlens[i] = src.nsa_cache_seqlens[i];
    }

#pragma unroll 8
    for (int i = tid; i < bs; i += total_threads) {
      dst.nsa_cu_seqlens_k[i + 1] = src.nsa_cu_seqlens_k[i + 1];
    }
  } else {  // TARGET_VERIFY or DRAFT_EXTEND
#pragma unroll 4
    for (int i = tid; i < seqlens_expanded_size; i += total_threads) {
      dst.nsa_cache_seqlens[i] = src.nsa_cache_seqlens[i];
    }

#pragma unroll 4
    for (int i = tid; i < seqlens_expanded_size; i += total_threads) {
      dst.nsa_cu_seqlens_k[i + 1] = src.nsa_cu_seqlens_k[i + 1];
    }
  }

  // Copy real page table - compile-time branch
  if constexpr (HAS_REAL_PAGE_TABLE) {
    int real_table_elements = (forward_mode == 0 ? bs : page_indices_rows) * real_page_table_cols;
#pragma unroll 2
    for (int i = tid; i < real_table_elements; i += total_threads) {
      int row = i / real_page_table_cols;
      int col = i % real_page_table_cols;
      dst.real_page_table[row * real_page_table_dst_stride + col] =
          src.real_page_table[row * real_page_table_cols + col];
    }
  }

  // Branch 4: FlashMLA metadata copy (different sizes per mode)
  if constexpr (HAS_FLASHMLA) {
    int flashmla_size = (forward_mode == 0) ? (bs + 1) : (seqlens_expanded_size + 1);

    if (forward_mode == 0) {
#pragma unroll 8
      for (int i = tid; i < flashmla_size; i += total_threads) {
        dst.flashmla_num_splits[i] = src.flashmla_num_splits[i];
      }
    } else {
#pragma unroll 4
      for (int i = tid; i < flashmla_size; i += total_threads) {
        dst.flashmla_num_splits[i] = src.flashmla_num_splits[i];
      }
    }

#pragma unroll 2
    for (int i = tid; i < flashmla_metadata_size; i += total_threads) {
      dst.flashmla_metadata[i] = src.flashmla_metadata[i];
    }
  }
}

/**
 * Multi-backend kernel for DECODE mode.
 * Copies from one source to THREE destinations in a single kernel launch.
 *
 * PERFORMANCE: 3x faster than three separate kernel launches due to:
 * - Reduced kernel launch overhead (1 launch instead of 3)
 * - Improved memory coalescing (source read once, written to 3 destinations)
 * - Better instruction-level parallelism
 *
 * Used by FusedMetadataCopyMultiKernel for speculative decoding scenarios.
 *
 * @tparam HAS_REAL_PAGE_TABLE Compile-time flag for real_page_table support
 * @tparam HAS_FLASHMLA Compile-time flag for FlashMLA metadata support
 */
template <bool HAS_REAL_PAGE_TABLE, bool HAS_FLASHMLA>
__global__ void fused_metadata_copy_multi_kernel(const FusedMetadataCopyMultiParams __grid_constant__ params) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;

  // Unpack parameters for readability
  const auto& src = params.src;
  const auto& dst0 = params.dst0;
  const auto& dst1 = params.dst1;
  const auto& dst2 = params.dst2;
  const int bs = params.bs;
  const int max_len = params.max_len;
  const int seqlens_expanded_size = params.seqlens_expanded_size;
  const int page_table_1_stride = params.page_table_1_stride;
  const int real_page_table_cols = params.real_page_table_cols;
  const int real_page_table_dst_stride = params.real_page_table_dst_stride;
  const int flashmla_metadata_size = params.flashmla_metadata_size;

  // Copy cache_seqlens to all 3 backends
#pragma unroll 8
  for (int i = tid; i < bs; i += total_threads) {
    int32_t val = src.cache_seqlens[i];
    dst0.cache_seqlens[i] = val;
    dst1.cache_seqlens[i] = val;
    dst2.cache_seqlens[i] = val;
  }

  // Copy cu_seqlens_k to all 3 backends (skip first element)
#pragma unroll 8
  for (int i = tid; i < bs; i += total_threads) {
    int32_t val = src.cu_seqlens_k[i + 1];
    dst0.cu_seqlens_k[i + 1] = val;
    dst1.cu_seqlens_k[i + 1] = val;
    dst2.cu_seqlens_k[i + 1] = val;
  }

  // DECODE mode: copy page_table_1 to all 3 backends
  int page_table_elements = bs * max_len;
#pragma unroll 4
  for (int i = tid; i < page_table_elements; i += total_threads) {
    int row = i / max_len;
    int col = i % max_len;
    int32_t val = src.page_indices[i];
    dst0.page_table_1[row * page_table_1_stride + col] = val;
    dst1.page_table_1[row * page_table_1_stride + col] = val;
    dst2.page_table_1[row * page_table_1_stride + col] = val;
  }

  // Copy nsa_cache_seqlens to all 3 backends
#pragma unroll 8
  for (int i = tid; i < bs; i += total_threads) {
    int32_t val = src.nsa_cache_seqlens[i];
    dst0.nsa_cache_seqlens[i] = val;
    dst1.nsa_cache_seqlens[i] = val;
    dst2.nsa_cache_seqlens[i] = val;
  }

  // Copy NSA cu_seqlens to all 3 backends
#pragma unroll 8
  for (int i = tid; i < bs; i += total_threads) {
    int32_t val = src.nsa_cu_seqlens_k[i + 1];
    dst0.nsa_cu_seqlens_k[i + 1] = val;
    dst1.nsa_cu_seqlens_k[i + 1] = val;
    dst2.nsa_cu_seqlens_k[i + 1] = val;
  }

  // Copy real page table to all 3 backends
  if (src.real_page_table != nullptr && dst0.real_page_table != nullptr) {
    int real_table_elements = bs * real_page_table_cols;
#pragma unroll 2
    for (int i = tid; i < real_table_elements; i += total_threads) {
      int row = i / real_page_table_cols;
      int col = i % real_page_table_cols;
      int src_idx = row * real_page_table_cols + col;
      int dst_idx = row * real_page_table_dst_stride + col;
      int32_t val = src.real_page_table[src_idx];
      dst0.real_page_table[dst_idx] = val;
      dst1.real_page_table[dst_idx] = val;
      dst2.real_page_table[dst_idx] = val;
    }
  }

  // Copy FlashMLA metadata to all 3 backends
  if constexpr (HAS_FLASHMLA) {
    int flashmla_size = bs + 1;
#pragma unroll 8
    for (int i = tid; i < flashmla_size; i += total_threads) {
      int32_t val = src.flashmla_num_splits[i];
      dst0.flashmla_num_splits[i] = val;
      dst1.flashmla_num_splits[i] = val;
      dst2.flashmla_num_splits[i] = val;
    }

#pragma unroll 2
    for (int i = tid; i < flashmla_metadata_size; i += total_threads) {
      int32_t val = src.flashmla_metadata[i];
      dst0.flashmla_metadata[i] = val;
      dst1.flashmla_metadata[i] = val;
      dst2.flashmla_metadata[i] = val;
    }
  }
}

// ============================================================================
// Host-side launcher wrappers for JIT compilation
// ============================================================================

namespace {

// Launch configuration constants
constexpr int THREADS_PER_BLOCK = 256;
constexpr int MAX_GRID_SIZE = 1024;  // Limit to prevent excessive resource usage

/**
 * Helper function to extract a typed data pointer from a TensorView.
 * Performs runtime type checking and returns the properly cast pointer.
 *
 * @tparam T The expected element type (e.g., int32_t)
 * @param tensor The TensorView to extract the pointer from
 * @param name The name of the tensor (for error reporting)
 * @return Typed pointer to the tensor data
 */
template <typename T>
inline const T* unwrap_data_ptr(const tvm::ffi::TensorView& tensor, const char* name) {
  using namespace host;
  if (tensor.data_ptr()) {
    RuntimeCheck(is_type<T>(tensor.dtype()), "Tensor ", name, " must have dtype int32");
  }
  return static_cast<const T*>(tensor.data_ptr());
}

/**
 * Helper function to extract a typed mutable data pointer from a TensorView.
 * Performs runtime type checking and returns the properly cast pointer.
 *
 * @tparam T The expected element type (e.g., int32_t)
 * @param tensor The TensorView to extract the pointer from
 * @param name The name of the tensor (for error reporting)
 * @return Typed mutable pointer to the tensor data
 */
template <typename T>
inline T* unwrap_data_ptr_mut(const tvm::ffi::TensorView& tensor, const char* name) {
  using namespace host;
  if (tensor.data_ptr()) {
    RuntimeCheck(is_type<T>(tensor.dtype()), "Tensor ", name, " must have dtype int32");
  }
  return static_cast<T*>(tensor.data_ptr());
}

/**
 * Helper function to extract a typed data pointer from an Optional TensorView.
 * Returns nullptr if the optional has no value, otherwise performs type checking.
 *
 * @tparam T The expected element type (e.g., int32_t)
 * @param optional_tensor The Optional TensorView to extract the pointer from
 * @param name The name of the tensor (for error reporting)
 * @return Typed pointer to the tensor data, or nullptr if optional has no value
 */
template <typename T>
inline const T*
unwrap_optional_data_ptr(const tvm::ffi::Optional<tvm::ffi::TensorView>& optional_tensor, const char* name) {
  using namespace host;
  if (!optional_tensor.has_value()) {
    return nullptr;
  }
  const auto& tensor = optional_tensor.value();
  RuntimeCheck(is_type<T>(tensor.dtype()), "Tensor ", name, " must have dtype int32");
  return static_cast<const T*>(tensor.data_ptr());
}

/**
 * Helper function to extract a typed mutable data pointer from an Optional TensorView.
 * Returns nullptr if the optional has no value, otherwise performs type checking.
 *
 * @tparam T The expected element type (e.g., int32_t)
 * @param optional_tensor The Optional TensorView to extract the pointer from
 * @param name The name of the tensor (for error reporting)
 * @return Typed mutable pointer to the tensor data, or nullptr if optional has no value
 */
template <typename T>
inline T*
unwrap_optional_data_ptr_mut(const tvm::ffi::Optional<tvm::ffi::TensorView>& optional_tensor, const char* name) {
  using namespace host;
  if (!optional_tensor.has_value()) {
    return nullptr;
  }
  const auto& tensor = optional_tensor.value();
  RuntimeCheck(is_type<T>(tensor.dtype()), "Tensor ", name, " must have dtype int32");
  return static_cast<T*>(tensor.data_ptr());
}

/**
 * Calculate kernel launch configuration.
 *
 * @param total_work Total number of work items
 * @param threads_per_block Threads per block (default: THREADS_PER_BLOCK)
 * @return Grid dimension for kernel launch
 */
inline dim3 get_launch_config(int total_work, int threads_per_block = THREADS_PER_BLOCK) {
  int num_blocks = (total_work + threads_per_block - 1) / threads_per_block;
  // Limit grid size to prevent excessive resource usage while ensuring coverage
  num_blocks = std::min(num_blocks, MAX_GRID_SIZE);
  return dim3(num_blocks);
}

/**
 * JIT wrapper for single-backend fused metadata copy kernel.
 *
 * This struct provides a unified interface for launching the fused metadata copy
 * kernel with different forward modes. It constructs the parameter struct and
 * launches the unified kernel.
 *
 * IMPLEMENTATION:
 * - Extracts raw pointers from TensorView objects
 * - Constructs FusedMetadataCopyParams with nested SourcePointers/DestinationPointers
 * - Calculates grid configuration based on maximum work size
 * - Launches fused_metadata_copy_kernel with __grid_constant__ parameters
 *
 * @tparam FORWARD_MODE Forward mode: 0=DECODE, 1=TARGET_VERIFY, 2=DRAFT_EXTEND
 * @tparam HAS_REAL_PAGE_TABLE Whether real_page_table tensors are present
 * @tparam HAS_FLASHMLA Whether FlashMLA metadata tensors are present
 */
template <int FORWARD_MODE, bool HAS_REAL_PAGE_TABLE, bool HAS_FLASHMLA>
struct FusedMetadataCopyKernel {
  static_assert(
      FORWARD_MODE >= 0 && FORWARD_MODE <= 2,
      "FORWARD_MODE must be 0 (DECODE), 1 (TARGET_VERIFY), or 2 (DRAFT_EXTEND)");

  static void
  run(const tvm::ffi::TensorView cache_seqlens_src,
      const tvm::ffi::TensorView cu_seqlens_k_src,
      const tvm::ffi::TensorView page_indices_src,
      const tvm::ffi::TensorView nsa_cache_seqlens_src,
      const tvm::ffi::Optional<tvm::ffi::TensorView> seqlens_expanded_src,
      const tvm::ffi::TensorView nsa_cu_seqlens_k_src,
      const tvm::ffi::Optional<tvm::ffi::TensorView> real_page_table_src,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_num_splits_src,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_metadata_src,
      const tvm::ffi::TensorView cache_seqlens_dst,
      const tvm::ffi::TensorView cu_seqlens_k_dst,
      const tvm::ffi::TensorView page_table_1_dst,
      const tvm::ffi::TensorView nsa_cache_seqlens_dst,
      const tvm::ffi::Optional<tvm::ffi::TensorView> seqlens_expanded_dst,
      const tvm::ffi::TensorView nsa_cu_seqlens_k_dst,
      const tvm::ffi::Optional<tvm::ffi::TensorView> real_page_table_dst,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_num_splits_dst,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_metadata_dst,
      int bs,
      int max_len,
      int max_seqlen_k,
      int seqlens_expanded_size) {
    using namespace host;

    // Build parameter struct with nested source/destination pointers
    // unwrap_data_ptr and unwrap_optional_data_ptr perform dtype validation
    const auto params = FusedMetadataCopyParams{
        .src =
            {
                .cache_seqlens = unwrap_data_ptr<int32_t>(cache_seqlens_src, "cache_seqlens_src"),
                .cu_seqlens_k = unwrap_data_ptr<int32_t>(cu_seqlens_k_src, "cu_seqlens_k_src"),
                .page_indices = unwrap_data_ptr<int32_t>(page_indices_src, "page_indices_src"),
                .nsa_cache_seqlens = unwrap_data_ptr<int32_t>(nsa_cache_seqlens_src, "nsa_cache_seqlens_src"),
                .seqlens_expanded = unwrap_optional_data_ptr<int32_t>(seqlens_expanded_src, "seqlens_expanded_src"),
                .nsa_cu_seqlens_k = unwrap_data_ptr<int32_t>(nsa_cu_seqlens_k_src, "nsa_cu_seqlens_k_src"),
                .real_page_table = unwrap_optional_data_ptr<int32_t>(real_page_table_src, "real_page_table_src"),
                .flashmla_num_splits =
                    unwrap_optional_data_ptr<int32_t>(flashmla_num_splits_src, "flashmla_num_splits_src"),
                .flashmla_metadata = unwrap_optional_data_ptr<int32_t>(flashmla_metadata_src, "flashmla_metadata_src"),
            },
        .dst =
            {
                .cache_seqlens = unwrap_data_ptr_mut<int32_t>(cache_seqlens_dst, "cache_seqlens_dst"),
                .cu_seqlens_k = unwrap_data_ptr_mut<int32_t>(cu_seqlens_k_dst, "cu_seqlens_k_dst"),
                .page_table_1 = unwrap_data_ptr_mut<int32_t>(page_table_1_dst, "page_table_1_dst"),
                .nsa_cache_seqlens = unwrap_data_ptr_mut<int32_t>(nsa_cache_seqlens_dst, "nsa_cache_seqlens_dst"),
                .seqlens_expanded = unwrap_optional_data_ptr_mut<int32_t>(seqlens_expanded_dst, "seqlens_expanded_dst"),
                .nsa_cu_seqlens_k = unwrap_data_ptr_mut<int32_t>(nsa_cu_seqlens_k_dst, "nsa_cu_seqlens_k_dst"),
                .real_page_table = unwrap_optional_data_ptr_mut<int32_t>(real_page_table_dst, "real_page_table_dst"),
                .flashmla_num_splits =
                    unwrap_optional_data_ptr_mut<int32_t>(flashmla_num_splits_dst, "flashmla_num_splits_dst"),
                .flashmla_metadata =
                    unwrap_optional_data_ptr_mut<int32_t>(flashmla_metadata_dst, "flashmla_metadata_dst"),
            },
        .forward_mode = FORWARD_MODE,
        .bs = bs,
        .max_len = max_len,
        .max_seqlen_k = max_seqlen_k,
        .seqlens_expanded_size = seqlens_expanded_size,
        .page_indices_rows = static_cast<int>(page_indices_src.shape()[0]),
        .page_table_1_stride = static_cast<int>(page_table_1_dst.shape()[1]),
        .real_page_table_cols =
            real_page_table_src.has_value() ? static_cast<int>(real_page_table_src.value().shape()[1]) : 0,
        .real_page_table_dst_stride =
            real_page_table_dst.has_value() ? static_cast<int>(real_page_table_dst.value().stride(0)) : 0,
        .flashmla_metadata_size =
            flashmla_metadata_src.has_value() ? static_cast<int>(flashmla_metadata_src.value().numel()) : 0,
    };

    // Calculate grid configuration
    int max_elements = std::max(
        {bs,
         params.page_indices_rows * max_seqlen_k,
         seqlens_expanded_size,
         HAS_FLASHMLA ? (seqlens_expanded_size + 1) : 0,
         HAS_FLASHMLA ? params.flashmla_metadata_size : 0});

    dim3 grid = get_launch_config(max_elements);
    dim3 block(THREADS_PER_BLOCK);
    DLDevice device = cache_seqlens_src.device();

    // Launch unified kernel with params struct
    host::LaunchKernel(grid, block, device)(fused_metadata_copy_kernel<HAS_REAL_PAGE_TABLE, HAS_FLASHMLA>, params);
  }
};

/**
 * JIT wrapper for multi-backend fused metadata copy kernel.
 *
 * This kernel optimizes the common case where metadata needs to be copied from
 * one source to THREE destination backends in a single kernel launch. This is
 * 3x faster than launching three separate kernels due to:
 * - Reduced kernel launch overhead (1 launch instead of 3)
 * - Improved memory coalescing (source read once, written to 3 destinations)
 * - Better GPU occupancy and instruction-level parallelism
 *
 * USAGE: Primarily for speculative decoding with multiple draft models, where
 * the same source metadata needs to be replicated to multiple backend contexts.
 *
 * LIMITATION: Currently only supports DECODE mode, which is the most frequently
 * used mode in speculative decoding scenarios.
 *
 * IMPLEMENTATION:
 * - Constructs FusedMetadataCopyMultiParams with 1 SourcePointers + 3 DestinationPointers
 * - Launches fused_metadata_copy_multi_kernel with __grid_constant__ parameters
 *
 * @tparam HAS_REAL_PAGE_TABLE Whether real_page_table tensors are present
 * @tparam HAS_FLASHMLA Whether FlashMLA metadata tensors are present
 */
template <bool HAS_REAL_PAGE_TABLE, bool HAS_FLASHMLA>
struct FusedMetadataCopyMultiKernel {
  static void
  run(const tvm::ffi::TensorView cache_seqlens_src,
      const tvm::ffi::TensorView cu_seqlens_k_src,
      const tvm::ffi::TensorView page_indices_src,
      const tvm::ffi::TensorView nsa_cache_seqlens_src,
      const tvm::ffi::TensorView nsa_cu_seqlens_k_src,
      const tvm::ffi::Optional<tvm::ffi::TensorView> real_page_table_src,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_num_splits_src,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_metadata_src,
      const tvm::ffi::TensorView cache_seqlens_dst0,
      const tvm::ffi::TensorView cu_seqlens_k_dst0,
      const tvm::ffi::TensorView page_table_1_dst0,
      const tvm::ffi::TensorView nsa_cache_seqlens_dst0,
      const tvm::ffi::TensorView nsa_cu_seqlens_k_dst0,
      const tvm::ffi::Optional<tvm::ffi::TensorView> real_page_table_dst0,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_num_splits_dst0,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_metadata_dst0,
      const tvm::ffi::TensorView cache_seqlens_dst1,
      const tvm::ffi::TensorView cu_seqlens_k_dst1,
      const tvm::ffi::TensorView page_table_1_dst1,
      const tvm::ffi::TensorView nsa_cache_seqlens_dst1,
      const tvm::ffi::TensorView nsa_cu_seqlens_k_dst1,
      const tvm::ffi::Optional<tvm::ffi::TensorView> real_page_table_dst1,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_num_splits_dst1,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_metadata_dst1,
      const tvm::ffi::TensorView cache_seqlens_dst2,
      const tvm::ffi::TensorView cu_seqlens_k_dst2,
      const tvm::ffi::TensorView page_table_1_dst2,
      const tvm::ffi::TensorView nsa_cache_seqlens_dst2,
      const tvm::ffi::TensorView nsa_cu_seqlens_k_dst2,
      const tvm::ffi::Optional<tvm::ffi::TensorView> real_page_table_dst2,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_num_splits_dst2,
      const tvm::ffi::Optional<tvm::ffi::TensorView> flashmla_metadata_dst2,
      int bs,
      int max_len,
      int seqlens_expanded_size) {
    using namespace host;

    // Build parameter struct with nested source/destination pointers
    // unwrap_data_ptr and unwrap_optional_data_ptr perform dtype validation
    const auto params = FusedMetadataCopyMultiParams{
        .src =
            {
                .cache_seqlens = unwrap_data_ptr<int32_t>(cache_seqlens_src, "cache_seqlens_src"),
                .cu_seqlens_k = unwrap_data_ptr<int32_t>(cu_seqlens_k_src, "cu_seqlens_k_src"),
                .page_indices = unwrap_data_ptr<int32_t>(page_indices_src, "page_indices_src"),
                .nsa_cache_seqlens = unwrap_data_ptr<int32_t>(nsa_cache_seqlens_src, "nsa_cache_seqlens_src"),
                .seqlens_expanded = nullptr,  // Not used in multi-backend DECODE mode
                .nsa_cu_seqlens_k = unwrap_data_ptr<int32_t>(nsa_cu_seqlens_k_src, "nsa_cu_seqlens_k_src"),
                .real_page_table = unwrap_optional_data_ptr<int32_t>(real_page_table_src, "real_page_table_src"),
                .flashmla_num_splits =
                    unwrap_optional_data_ptr<int32_t>(flashmla_num_splits_src, "flashmla_num_splits_src"),
                .flashmla_metadata = unwrap_optional_data_ptr<int32_t>(flashmla_metadata_src, "flashmla_metadata_src"),
            },
        .dst0 =
            {
                .cache_seqlens = unwrap_data_ptr_mut<int32_t>(cache_seqlens_dst0, "cache_seqlens_dst0"),
                .cu_seqlens_k = unwrap_data_ptr_mut<int32_t>(cu_seqlens_k_dst0, "cu_seqlens_k_dst0"),
                .page_table_1 = unwrap_data_ptr_mut<int32_t>(page_table_1_dst0, "page_table_1_dst0"),
                .nsa_cache_seqlens = unwrap_data_ptr_mut<int32_t>(nsa_cache_seqlens_dst0, "nsa_cache_seqlens_dst0"),
                .seqlens_expanded = nullptr,
                .nsa_cu_seqlens_k = unwrap_data_ptr_mut<int32_t>(nsa_cu_seqlens_k_dst0, "nsa_cu_seqlens_k_dst0"),
                .real_page_table = unwrap_optional_data_ptr_mut<int32_t>(real_page_table_dst0, "real_page_table_dst0"),
                .flashmla_num_splits =
                    unwrap_optional_data_ptr_mut<int32_t>(flashmla_num_splits_dst0, "flashmla_num_splits_dst0"),
                .flashmla_metadata =
                    unwrap_optional_data_ptr_mut<int32_t>(flashmla_metadata_dst0, "flashmla_metadata_dst0"),
            },
        .dst1 =
            {
                .cache_seqlens = unwrap_data_ptr_mut<int32_t>(cache_seqlens_dst1, "cache_seqlens_dst1"),
                .cu_seqlens_k = unwrap_data_ptr_mut<int32_t>(cu_seqlens_k_dst1, "cu_seqlens_k_dst1"),
                .page_table_1 = unwrap_data_ptr_mut<int32_t>(page_table_1_dst1, "page_table_1_dst1"),
                .nsa_cache_seqlens = unwrap_data_ptr_mut<int32_t>(nsa_cache_seqlens_dst1, "nsa_cache_seqlens_dst1"),
                .seqlens_expanded = nullptr,
                .nsa_cu_seqlens_k = unwrap_data_ptr_mut<int32_t>(nsa_cu_seqlens_k_dst1, "nsa_cu_seqlens_k_dst1"),
                .real_page_table = unwrap_optional_data_ptr_mut<int32_t>(real_page_table_dst1, "real_page_table_dst1"),
                .flashmla_num_splits =
                    unwrap_optional_data_ptr_mut<int32_t>(flashmla_num_splits_dst1, "flashmla_num_splits_dst1"),
                .flashmla_metadata =
                    unwrap_optional_data_ptr_mut<int32_t>(flashmla_metadata_dst1, "flashmla_metadata_dst1"),
            },
        .dst2 =
            {
                .cache_seqlens = unwrap_data_ptr_mut<int32_t>(cache_seqlens_dst2, "cache_seqlens_dst2"),
                .cu_seqlens_k = unwrap_data_ptr_mut<int32_t>(cu_seqlens_k_dst2, "cu_seqlens_k_dst2"),
                .page_table_1 = unwrap_data_ptr_mut<int32_t>(page_table_1_dst2, "page_table_1_dst2"),
                .nsa_cache_seqlens = unwrap_data_ptr_mut<int32_t>(nsa_cache_seqlens_dst2, "nsa_cache_seqlens_dst2"),
                .seqlens_expanded = nullptr,
                .nsa_cu_seqlens_k = unwrap_data_ptr_mut<int32_t>(nsa_cu_seqlens_k_dst2, "nsa_cu_seqlens_k_dst2"),
                .real_page_table = unwrap_optional_data_ptr_mut<int32_t>(real_page_table_dst2, "real_page_table_dst2"),
                .flashmla_num_splits =
                    unwrap_optional_data_ptr_mut<int32_t>(flashmla_num_splits_dst2, "flashmla_num_splits_dst2"),
                .flashmla_metadata =
                    unwrap_optional_data_ptr_mut<int32_t>(flashmla_metadata_dst2, "flashmla_metadata_dst2"),
            },
        .bs = bs,
        .max_len = max_len,
        .seqlens_expanded_size = seqlens_expanded_size,
        .page_table_1_stride = static_cast<int>(page_table_1_dst0.shape()[1]),
        .real_page_table_cols =
            real_page_table_src.has_value() ? static_cast<int>(real_page_table_src.value().shape()[1]) : 0,
        .real_page_table_dst_stride =
            real_page_table_dst0.has_value() ? static_cast<int>(real_page_table_dst0.value().stride(0)) : 0,
        .flashmla_metadata_size =
            flashmla_metadata_src.has_value() ? static_cast<int>(flashmla_metadata_src.value().numel()) : 0,
    };

    dim3 grid = get_launch_config(bs * max_len);
    dim3 block(THREADS_PER_BLOCK);
    DLDevice device = cache_seqlens_src.device();

    // Launch multi-backend kernel with params struct
    host::LaunchKernel(grid, block, device)(
        fused_metadata_copy_multi_kernel<HAS_REAL_PAGE_TABLE, HAS_FLASHMLA>, params);
  }
};

}  // namespace
