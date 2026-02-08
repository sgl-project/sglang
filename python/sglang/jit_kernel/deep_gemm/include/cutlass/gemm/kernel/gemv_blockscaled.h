/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief 
*/

#pragma once

#include "cutlass/arch/cache_operation.h"  /// cutlass::arch::CacheOperation
#include "cutlass/arch/memory.h"           // cutlass::arch::global_load
#include "cutlass/arch/memory_sm80.h"      // cp.async helpers, ldsm, cp_async_wait
#include "cutlass/complex.h"               // cutlass::ComplexTransform:
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"             // cutlass::fast_max
#include "cutlass/layout/matrix.h"         // cutlass::layout::RowMajor
#include "cutlass/matrix_coord.h"          // cutlass::MatrixCoord
#include "cutlass/numeric_conversion.h"    // cutlass::FloatRoundStyle, cutlass::NumericConverter
#include "cutlass/numeric_types.h"         // cutlass::float_e4m3_t
#include "cutlass/platform/platform.h"     // cutlass::is_same_v
#include "cutlass/tensor_ref.h"            // cutlass::TensorRef
#include "cutlass/semaphore.h"             // split-k

#include "cute/algorithm/functional.hpp"   // cute::for_each
#include "cute/numeric/arithmetic_tuple.hpp" // cute::make_int_sequence

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA_,
  typename LayoutA_,
  typename ElementB_,
  typename ElementC_,
  typename ElementAccumulator_,
  typename EpilogueOutputOp_,
  int kElementsPerAccess_ = 1,            ///< Number of elements involved in a global access.
  int kThreadCount_ = 0,                  ///< Number of threads in the thread block.
                                          ///  It will be calculated automatically if set to 0.
  int kThreadsPerRow_ = 0,                ///< Number of threads in the k dimension.
                                          ///  It will be calculated automatically if set to 0.
  typename ElementSFA_ = cutlass::float_e4m3_t,
  typename ElementSFB_ = cutlass::float_e4m3_t,
  int kSFVecSize_ = 16
>
struct GemvBlockScaled;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////

// GEMV for row-major A matrix
template <typename ElementA_,
          typename ElementB_,
          typename ElementC_,
          typename ElementAccumulator_,
          typename EpilogueOutputOp_,
          int kElementsPerAccess_,
          int kThreadCount_,
          int kThreadsPerRow_,
          typename ElementSFA_,
          typename ElementSFB_,
          int kSFVecSize_>
struct GemvBlockScaled<ElementA_,
            cutlass::layout::RowMajor,
            ElementB_,
            ElementC_,
            ElementAccumulator_,
            EpilogueOutputOp_,
            kElementsPerAccess_,
            kThreadCount_,
            kThreadsPerRow_,
            ElementSFA_,
            ElementSFB_,
            kSFVecSize_>
{
public:
  using ElementA = ElementA_;
  using ElementSFA = ElementSFA_;
  using LayoutA = cutlass::layout::RowMajor;
  using TensorRefA = cutlass::TensorRef<ElementA, LayoutA>;
  static_assert(cutlass::sizeof_bits<ElementSFA>::value == 8, "ElementSFA should be FP8 type");

  using ElementB = ElementB_;
  using ElementSFB = ElementSFB_;
  using LayoutB = cutlass::layout::ColumnMajor;
  static_assert(cutlass::sizeof_bits<ElementSFB>::value == 8, "ElementSFB should be FP8 type");

  using ElementC = ElementC_;
  using LayoutC = cutlass::layout::ColumnMajor;

  using ElementAccumulator = ElementAccumulator_;

  static constexpr cutlass::ComplexTransform kTransformA = cutlass::ComplexTransform::kNone;
  static constexpr cutlass::ComplexTransform kTransformB = cutlass::ComplexTransform::kNone;

  static constexpr FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest;

  // number of return elements in a global access
  static constexpr int kElementsPerAccess = kElementsPerAccess_;
  static constexpr int kSFVecSize = kSFVecSize_;
  static constexpr int kSFPerAccess = cutlass::const_max(1, kElementsPerAccess / kSFVecSize);

  static_assert(kSFVecSize == 16, "Only SFVecSize = 16 is supported");
  // Hardcode some check for easier debug
  static_assert(kElementsPerAccess == 32, "for fp4 kernel, 32 elt per access");
  static_assert(kSFPerAccess == 2, "fpr fp4 kernel, 2 sf read per thread");

  static constexpr bool kDequantizeA = cutlass::sizeof_bits<ElementA>::value == 4;
  static constexpr bool kDequantizeB = cutlass::sizeof_bits<ElementB>::value == 4;
  static constexpr int kPackedElementsA = cutlass::sizeof_bits<ElementA>::value == 4 ? 2 : 1;
  static constexpr int kPackedElementsB = cutlass::sizeof_bits<ElementB>::value == 4 ? 2 : 1;
  static constexpr int kPackedElements = cutlass::const_max(kPackedElementsA, kPackedElementsB);

  static_assert(kDequantizeA == true, "kDequantizeA should be true");
  static_assert(kDequantizeB == true, "kDequantizeB should be true");

  using FragmentA = cutlass::Array<ElementA, kElementsPerAccess>;
  using FragmentB = cutlass::Array<ElementB, kElementsPerAccess>;
  using FragmentCompute = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentSFA = cutlass::Array<ElementSFA, kSFPerAccess>;
  using FragmentSFB = cutlass::Array<ElementSFB, kSFPerAccess>;
  using FragmentPackedA = cutlass::Array<ElementA, kPackedElements>;
  using FragmentPackedB = cutlass::Array<ElementB, kPackedElements>;

  static_assert(sizeof_bits<FragmentA>::value == 128, "FragmentA should be 128 bits");
  static_assert(sizeof_bits<FragmentB>::value == 128, "FragmentB should be 128 bits");

  // // thread block shape (kThreadsPerRow, kThreadCount / kThreadsPerRow, 1)
  static constexpr int kThreadCount = (kThreadCount_ <= 0) ? 128 : kThreadCount_;
  static constexpr int kThreadsPerRow = (kThreadsPerRow_ <= 0) ? 
                                        cutlass::const_min(static_cast<int>(kThreadCount / cutlass::bits_to_bytes(kElementsPerAccess * cutlass::sizeof_bits<ElementA>::value)), 16) :
                                        kThreadsPerRow_;
  static constexpr int kThreadsPerCol = kThreadCount / kThreadsPerRow;

  static constexpr int kStageCount = 4;
  static constexpr int kBufferCount = 2;

  // Number of elements stored in shared memory per stage for operands A and B.
  // Each thread contributes `kElementsPerAccess / kPackedElements{A,B}` packed
  // values.
  static constexpr int kSmemPerStageA = kThreadCount * kElementsPerAccess / kPackedElementsA;
  // B is uniform across all threads in the same k-column, so only store it once per k-thread
  static constexpr int kSmemPerStageB = kThreadsPerRow * kElementsPerAccess / kPackedElementsB;

  using EpilogueOutputOp = EpilogueOutputOp_;

  // Ensure epilogue and mainloop have same thread layout
  static_assert(kThreadCount == EpilogueOutputOp::kThreadCount, "mainloop, epilogue thread count mismatch");
  static_assert(kThreadsPerRow == EpilogueOutputOp::kThreadsPerRow, "mainloop, epilogue thread per row mismatch");
  static_assert(kThreadsPerCol == EpilogueOutputOp::kThreadsPerCol, "mainloop, epilogue thread per col mismatch");

  //
  // Structures
  //

  /// Argument structure
  struct Arguments
  {
    MatrixCoord problem_size;
    int32_t batch_count{0};
    typename EpilogueOutputOp::Params epilogue;

    TensorRefA ref_A;

    ElementB const *ptr_B{nullptr};
    ElementC const *ptr_C{nullptr};
    ElementC *ptr_D{nullptr};

    ElementSFA const *ptr_SFA{nullptr};
    ElementSFB const *ptr_SFB{nullptr};

    int64_t stride_A{0};
    int64_t batch_stride_A{0};
    int64_t batch_stride_B{0};
    int64_t batch_stride_C{0};
    int64_t batch_stride_D{0};

    int64_t batch_stride_SFA{0};
    int64_t batch_stride_SFB{0};
    int64_t batch_stride_SFD{0};
  };

  using Params = Arguments;

  /// Shared memory storage structure
  struct SharedStorage
  {
    using EpilogueStorage = typename EpilogueOutputOp::SharedStorage;
    EpilogueStorage epilogue;

    alignas(16) ElementA  smem_A[kBufferCount][kStageCount][kSmemPerStageA];
    alignas(16) ElementB  smem_B[kBufferCount][kStageCount][kSmemPerStageB];
    alignas(16) ElementSFA smem_SFA[kBufferCount][kStageCount][kThreadCount * kSFPerAccess];
    alignas(16) ElementSFB smem_SFB[kBufferCount][kStageCount][kThreadsPerRow * kSFPerAccess];
  };

public:
  //
  // Methods
  //
  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::MatrixCoord const &problem_size)
  {
    if (problem_size.column() % kElementsPerAccess != 0) {
      return Status::kErrorMisalignedOperand;
    }
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args)
  {
    return can_implement(args.problem_size);
  }

  /// Executes one GEMV
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage)
  {
    EpilogueOutputOp epilogue(params.epilogue, shared_storage.epilogue);

    // Converters only needed for regular GEMV fallback case
    NumericConverter<ElementAccumulator, ElementA, Round> A_converter;
    NumericConverter<ElementAccumulator, ElementB, Round> B_converter;
    NumericConverter<ElementAccumulator, ElementSFA, Round> SFA_converter;
    NumericConverter<ElementAccumulator, ElementSFB, Round> SFB_converter;

    const int32_t gemm_m = params.problem_size.row();
    [[maybe_unused]] static constexpr int32_t gemm_n = 1;
    const int32_t gemm_k = params.problem_size.column();
    const int32_t gemm_batch = params.batch_count;

    // Loop over batch indices
    for (int batch_idx = blockIdx.z; batch_idx < gemm_batch; batch_idx += gridDim.z) {
      
      int idx_col_k = threadIdx.x;
      int idx_row_m = blockIdx.x * blockDim.y + threadIdx.y;

      if (idx_row_m < gemm_m) {
        // problem_size (row = m, column = k)
        // matrix A (batch, m, k)
        // vector B (batch, k, 1)
        // vector C (batch, m, 1)
        // vector D (batch, m, 1)
        // move in the batch dimension
        ElementA const *ptr_A = params.ref_A.data() + batch_idx * params.batch_stride_A / kPackedElementsA;
        ElementB const *ptr_B = params.ptr_B + batch_idx * params.batch_stride_B / kPackedElementsB;
        ElementC const *ptr_C = params.ptr_C + batch_idx * params.batch_stride_C;
        ElementC *ptr_D = params.ptr_D + batch_idx * params.batch_stride_D;

        // move in the k dimension
        ptr_A += idx_col_k * kElementsPerAccess / kPackedElementsA;
        ptr_B += idx_col_k * kElementsPerAccess / kPackedElementsB;

        // move in the m dimension
        ptr_A += idx_row_m * params.stride_A / kPackedElementsA;
        ptr_C += idx_row_m;
        ptr_D += idx_row_m;

        ElementSFA const *ptr_SF_A{nullptr};
        ElementSFB const *ptr_SF_B{nullptr};
        int global_k{0};

        int SF_blocks_by_M = (gemm_m + 127) >> 7;
        int SF_blocks_by_K = (gemm_k / kSFVecSize + 3) >> 2;

        // move in the batch dimension
        ptr_SF_A = params.ptr_SFA + batch_idx * SF_blocks_by_M * SF_blocks_by_K * 512;
        ptr_SF_B = params.ptr_SFB + batch_idx * SF_blocks_by_K * 512;

         // move in the m dimension
        ptr_SF_A += (((idx_row_m >> 7) * SF_blocks_by_K) << 9) + ((idx_row_m & 0x1f) << 4) + ((idx_row_m & 0x7f) >> 5 << 2);

        global_k = idx_col_k * kElementsPerAccess;

        ElementAccumulator accum = ElementAccumulator(0);

        // Local aliases
        const int tileA_k_local = kThreadsPerRow * kElementsPerAccess;
        const int total_tiles   = gemm_k / tileA_k_local;

        int unroll_col_k = 0; // total K elements consumed so far by this thread
        const int thread_id = threadIdx.y * kThreadsPerRow + threadIdx.x;
        const bool is_even_thread = (threadIdx.x % 2 == 0);
        const bool load_b = (threadIdx.y == 0);
        const int smem_sf_write_offset = (thread_id / 2) * 4;  // 4 FP8 per even thread
        const int smem_sf_offset = thread_id * kSFPerAccess;
        
        // Fast path: if the problem fits entirely in the tail path, skip SMEM
        if (total_tiles == 0) {
          accum += process_tail_elements(0, idx_col_k, gemm_k,
                                         ptr_A, ptr_B,
                                         ptr_SF_A, ptr_SF_B,
                                         A_converter, B_converter,
                                         SFA_converter, SFB_converter);
        } else {

          // Scaling factors are now loaded from shared memory, no register pipeline needed

          // Thread-local SMEM line offset
          const int thread_linear = threadIdx.y * kThreadsPerRow + threadIdx.x;
          const int smem_offset_A = thread_linear * (kElementsPerAccess / kPackedElementsA);
          // Only one row of threads (threadIdx.y == 0) loads B
          const int smem_offset_B = threadIdx.x * (kElementsPerAccess / kPackedElementsB);

          // PROLOGUE – prime first kStageCount-1 stages into buffer 0
          CUTLASS_PRAGMA_UNROLL
          for (int b = 0; b < kBufferCount - 1; ++b) {
            // Load all stages using the helper function
            load_stages_gmem_to_smem(
                b,                    // buffer_idx
                kStageCount,          // num_stages
                unroll_col_k,         // passed by reference
                global_k,             // passed by reference
                tileA_k_local,
                smem_offset_A,
                smem_offset_B,
                smem_sf_write_offset,
                is_even_thread,
                load_b,
                true,                 // valid_tile = true for prologue
                ptr_A,
                ptr_B,
                ptr_SF_A,
                ptr_SF_B,
                shared_storage);
          }
          cutlass::arch::cp_async_fence();

          // Ensure first stage committed
          cutlass::arch::cp_async_wait<kBufferCount - 2>();
          __syncthreads();

          // Register double buffering for A/B fragments and SFA/SFB like SM80
          FragmentA fragA_reg[2];
          FragmentB fragB_reg[2];
          FragmentSFA fragSFA_reg[2];
          FragmentSFB fragSFB_reg[2];
          
          // Current pipe index in smem to read from
          int smem_pipe_read  = 0;
          // Current pipe index in smem to write to  
          int smem_pipe_write = kBufferCount - 1;

          // PREFETCH register pipeline - load first kblock (stage 0) into register bank 0
          if constexpr (kStageCount > 1) 
          {
            int frag_idx = 0;
            
            // Load fragments using the helper function
            load_smem_fragments(
                fragA_reg[frag_idx], 
                fragB_reg[frag_idx],
                fragSFA_reg[frag_idx],
                fragSFB_reg[frag_idx],
                smem_pipe_read,
                0,  // k_block = 0
                smem_offset_A,
                smem_offset_B,
                smem_sf_offset,
                shared_storage);
            
          }

          // Mainloop
          int tile_idx = 0;
          while (tile_idx < total_tiles) {
            int smem_pipe_read_curr = smem_pipe_read;

            for_each(make_int_sequence<kStageCount>{}, [&] (auto k_block)
            {
              if (k_block == kStageCount - 1)
              {
                cutlass::arch::cp_async_wait<kBufferCount - 2>();
                __syncthreads();
                
                smem_pipe_read_curr = smem_pipe_read;
              }

              // Load A/B/SFA/SFB smem->regs for k_block_next
              auto k_block_next = (k_block + Int<1>{}) % kStageCount;
              int frag_idx_next = (k_block + 1) & 1;

              // Prefetch next kblock data using saved pipe index
              load_smem_fragments(
                  fragA_reg[frag_idx_next],
                  fragB_reg[frag_idx_next],
                  fragSFA_reg[frag_idx_next],
                  fragSFB_reg[frag_idx_next],
                  smem_pipe_read_curr,
                  k_block_next,
                  smem_offset_A,
                  smem_offset_B,
                  smem_sf_offset,
                  shared_storage);
              // Copy gmem to smem before computing gemm on each k-pipe
              if (k_block == 0)
              {
                // Use predicate instead of branch for cp_async
                bool valid_tile = (global_k < gemm_k);
                
                // Load all stages using the helper function
                load_stages_gmem_to_smem(
                    smem_pipe_write,      // buffer_idx
                    kStageCount,          // num_stages
                    unroll_col_k,         // passed by reference
                    global_k,             // passed by reference
                    tileA_k_local,
                    smem_offset_A,
                    smem_offset_B,
                    smem_sf_write_offset,
                    is_even_thread,
                    load_b,
                    valid_tile,
                    ptr_A,
                    ptr_B,
                    ptr_SF_A,
                    ptr_SF_B,
                    shared_storage);
                
                cutlass::arch::cp_async_fence();
                
                // Advance the pipe indices
                smem_pipe_write = smem_pipe_read;
                ++smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == kBufferCount) ? 0 : smem_pipe_read;
              }

              {
                int frag_idx = k_block & 1;
                
                // Compute using current fragments
                accum += blockscaled_multiply_add(
                    fragA_reg[frag_idx], fragB_reg[frag_idx],
                    fragSFA_reg[frag_idx],
                    fragSFB_reg[frag_idx]);
              }
            });

            tile_idx += kStageCount;
          }

          // Drain outstanding async copies
          cutlass::arch::cp_async_wait<0>();
          __syncthreads();

          // Tail elements that don't fill a full tile
          if (unroll_col_k + idx_col_k * kPackedElementsA < gemm_k) {
            accum += process_tail_elements(unroll_col_k, idx_col_k, gemm_k,
                                           ptr_A, ptr_B,
                                           ptr_SF_A, ptr_SF_B,
                                           A_converter, B_converter,
                                           SFA_converter, SFB_converter);
          }
        }

        CUTLASS_PRAGMA_UNROLL
        for (int mask = (kThreadsPerRow >> 1); mask > 0; mask >>= 1) {
          accum += ElementAccumulator(__shfl_xor_sync(0xFFFFFFFF, static_cast<float>(accum), mask, 32));
        }

        auto frag_acc = static_cast<typename EpilogueOutputOp::ElementAccumulator>(accum);
        auto frag_c = static_cast<typename EpilogueOutputOp::ElementC>(*(ptr_C));
        
        // Applying blockscaled epilogue
        epilogue(frag_acc, frag_c, batch_idx);
      }
    }
  } //end of operator()

private:
  // Load multiple stages from global to shared memory
  CUTLASS_DEVICE
  void load_stages_gmem_to_smem(
      int buffer_idx,
      int num_stages,
      int& unroll_col_k,
      int& global_k,
      int tileA_k_local,
      int smem_offset_A,
      int smem_offset_B,
      int smem_sf_write_offset,
      bool is_even_thread,
      bool load_b,
      bool valid_tile,
      ElementA const* ptr_A,
      ElementB const* ptr_B,
      ElementSFA const* ptr_SF_A,
      ElementSFB const* ptr_SF_B,
      SharedStorage& shared_storage) {
    
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < num_stages; ++s) {
      // Load scaling factors using cp.async - only even threads participate
      // Calculate SF indices for this thread
      int SF_idx = global_k / kSFVecSize;
      int SF_offset_by_k = ((SF_idx >> 2) << 9) + (SF_idx & 0x3);
        
      void *smem_ptr_SFA = &shared_storage.smem_SFA[buffer_idx][s][smem_sf_write_offset];
      const void *gmem_ptr_SFA = ptr_SF_A + SF_offset_by_k;
      // Load 4 FP8 values (32 bits) - for this thread and next thread
      cutlass::arch::cp_async<sizeof(uint32_t)>(smem_ptr_SFA, gmem_ptr_SFA, valid_tile && is_even_thread);
        
      void *smem_ptr_SFB = &shared_storage.smem_SFB[buffer_idx][s][(threadIdx.x / 2) * 4];
      const void *gmem_ptr_SFB = ptr_SF_B + SF_offset_by_k;
      // Load 4 FP8 values (32 bits) - for this thread and next thread, only if threadIdx.y == 0
      cutlass::arch::cp_async<sizeof(uint32_t)>(smem_ptr_SFB, gmem_ptr_SFB, valid_tile && load_b && is_even_thread);

      void *smem_ptr_A = &shared_storage.smem_A[buffer_idx][s][smem_offset_A];
      const void *gmem_ptr_A = ptr_A + unroll_col_k / kPackedElementsA;
      cutlass::arch::cp_async<sizeof(FragmentA)>(smem_ptr_A, gmem_ptr_A, valid_tile);

      void *smem_ptr_B = &shared_storage.smem_B[buffer_idx][s][smem_offset_B];
      const void *gmem_ptr_B = ptr_B + unroll_col_k / kPackedElementsB;
      cutlass::arch::cp_async<sizeof(FragmentB)>(smem_ptr_B, gmem_ptr_B, valid_tile && load_b);

      unroll_col_k += tileA_k_local;
      global_k     += tileA_k_local;
    }
  }

  /// Fused blockscaled GEMV computation using PTX
  CUTLASS_DEVICE
  ElementAccumulator blockscaled_multiply_add(
      FragmentA const& fragA,
      FragmentB const& fragB, 
      FragmentSFA const& fragSFA,
      FragmentSFB const& fragSFB) {

      #if defined(CUDA_PTX_FP4FP6_CVT_ENABLED)
        uint16_t const& src_fragSFA_packed = reinterpret_cast<uint16_t const&>(fragSFA);
        uint16_t const& src_fragSFB_packed = reinterpret_cast<uint16_t const&>(fragSFB);

        uint32_t const* src_fragA_packed = reinterpret_cast<uint32_t const*>(&fragA);
        uint32_t const* src_fragB_packed = reinterpret_cast<uint32_t const*>(&fragB);

        ElementAccumulator out;
        uint16_t* out_fp16 = reinterpret_cast<uint16_t*>(&out);

        asm volatile( \
            "{\n" \
            // declare registers for A / B tensors
            ".reg .b8 byte0_0, byte0_1, byte0_2, byte0_3;\n" \
            ".reg .b8 byte0_4, byte0_5, byte0_6, byte0_7;\n" \
            ".reg .b8 byte1_0, byte1_1, byte1_2, byte1_3;\n" \
            ".reg .b8 byte1_4, byte1_5, byte1_6, byte1_7;\n" \
            ".reg .b8 byte2_0, byte2_1, byte2_2, byte2_3;\n" \
            ".reg .b8 byte2_4, byte2_5, byte2_6, byte2_7;\n" \
            ".reg .b8 byte3_0, byte3_1, byte3_2, byte3_3;\n" \
            ".reg .b8 byte3_4, byte3_5, byte3_6, byte3_7;\n" \

            // declare registers for accumulators
            ".reg .f16x2 accum_0_0, accum_0_1, accum_0_2, accum_0_3;\n" \
            ".reg .f16x2 accum_1_0, accum_1_1, accum_1_2, accum_1_3;\n" \
            ".reg .f16x2 accum_2_0, accum_2_1, accum_2_2, accum_2_3;\n" \
            ".reg .f16x2 accum_3_0, accum_3_1, accum_3_2, accum_3_3;\n" \

            // declare registers for scaling factors
            ".reg .f16x2 sfa_f16x2;\n" \
            ".reg .f16x2 sfb_f16x2;\n" \
            ".reg .f16x2 sf_f16x2;\n" \
            
            // declare registers for conversion
            ".reg .f16x2 cvt_0_0, cvt_0_1, cvt_0_2, cvt_0_3;\n" \
            ".reg .f16x2 cvt_0_4, cvt_0_5, cvt_0_6, cvt_0_7;\n" \
            ".reg .f16x2 cvt_1_0, cvt_1_1, cvt_1_2, cvt_1_3;\n" \
            ".reg .f16x2 cvt_1_4, cvt_1_5, cvt_1_6, cvt_1_7;\n" \
            ".reg .f16x2 cvt_2_0, cvt_2_1, cvt_2_2, cvt_2_3;\n" \
            ".reg .f16x2 cvt_2_4, cvt_2_5, cvt_2_6, cvt_2_7;\n" \
            ".reg .f16x2 cvt_3_0, cvt_3_1, cvt_3_2, cvt_3_3;\n" \
            ".reg .f16x2 cvt_3_4, cvt_3_5, cvt_3_6, cvt_3_7;\n" \
            ".reg .f16 result_f16, lane0, lane1;\n" \
            ".reg .f16x2 mul_f16x2_0, mul_f16x2_1;\n" \

            // convert scaling factors from fp8 to f16x2
            "cvt.rn.f16x2.e4m3x2 sfa_f16x2, %1;\n" \
            "cvt.rn.f16x2.e4m3x2 sfb_f16x2, %2;\n" \
            
            // clear accumulators
            "mov.b32 accum_0_0, 0;\n" \
            "mov.b32 accum_0_1, 0;\n" \
            "mov.b32 accum_0_2, 0;\n" \
            "mov.b32 accum_0_3, 0;\n" \
            "mov.b32 accum_1_0, 0;\n" \
            "mov.b32 accum_1_1, 0;\n" \
            "mov.b32 accum_1_2, 0;\n" \
            "mov.b32 accum_1_3, 0;\n" \
            "mov.b32 accum_2_0, 0;\n" \
            "mov.b32 accum_2_1, 0;\n" \
            "mov.b32 accum_2_2, 0;\n" \
            "mov.b32 accum_2_3, 0;\n" \
            "mov.b32 accum_3_0, 0;\n" \
            "mov.b32 accum_3_1, 0;\n" \
            "mov.b32 accum_3_2, 0;\n" \
            "mov.b32 accum_3_3, 0;\n" \
            
            // multiply, unpacking and permuting scale factors
            "mul.rn.f16x2 sf_f16x2, sfa_f16x2, sfb_f16x2;\n" \
            "mov.b32 {lane0, lane1}, sf_f16x2;\n" \
            "mov.b32 mul_f16x2_0, {lane0, lane0};\n" \
            "mov.b32 mul_f16x2_1, {lane1, lane1};\n" \

            // unpacking A and B tensors
            "mov.b32 {byte0_0, byte0_1, byte0_2, byte0_3}, %3;\n" \
            "mov.b32 {byte0_4, byte0_5, byte0_6, byte0_7}, %4;\n" \
            "mov.b32 {byte1_0, byte1_1, byte1_2, byte1_3}, %5;\n" \
            "mov.b32 {byte1_4, byte1_5, byte1_6, byte1_7}, %6;\n" \
            "mov.b32 {byte2_0, byte2_1, byte2_2, byte2_3}, %7;\n" \
            "mov.b32 {byte2_4, byte2_5, byte2_6, byte2_7}, %8;\n" \
            "mov.b32 {byte3_0, byte3_1, byte3_2, byte3_3}, %9;\n" \
            "mov.b32 {byte3_4, byte3_5, byte3_6, byte3_7}, %10;\n" \

            // convert A and B tensors from fp4 to f16x2

            // A[0 - 7] and B[0 - 7]
            "cvt.rn.f16x2.e2m1x2 cvt_0_0, byte0_0;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_0_1, byte0_1;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_0_2, byte0_2;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_0_3, byte0_3;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_0_4, byte0_4;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_0_5, byte0_5;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_0_6, byte0_6;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_0_7, byte0_7;\n" \

            // A[8 - 15] and B[8 - 15]
            "cvt.rn.f16x2.e2m1x2 cvt_1_0, byte1_0;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_1_1, byte1_1;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_1_2, byte1_2;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_1_3, byte1_3;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_1_4, byte1_4;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_1_5, byte1_5;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_1_6, byte1_6;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_1_7, byte1_7;\n" \

            // A[16 - 23] and B[16 - 23]
            "cvt.rn.f16x2.e2m1x2 cvt_2_0, byte2_0;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_2_1, byte2_1;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_2_2, byte2_2;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_2_3, byte2_3;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_2_4, byte2_4;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_2_5, byte2_5;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_2_6, byte2_6;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_2_7, byte2_7;\n" \

            // A[24 - 31] and B[24 - 31]
            "cvt.rn.f16x2.e2m1x2 cvt_3_0, byte3_0;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_3_1, byte3_1;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_3_2, byte3_2;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_3_3, byte3_3;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_3_4, byte3_4;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_3_5, byte3_5;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_3_6, byte3_6;\n" \
            "cvt.rn.f16x2.e2m1x2 cvt_3_7, byte3_7;\n" \

            // fma for A[0 - 7] and B[0 - 7]
            "fma.rn.f16x2 accum_0_0, cvt_0_0, cvt_0_4, accum_0_0;\n" \
            "fma.rn.f16x2 accum_0_1, cvt_0_1, cvt_0_5, accum_0_1;\n" \
            "fma.rn.f16x2 accum_0_2, cvt_0_2, cvt_0_6, accum_0_2;\n" \
            "fma.rn.f16x2 accum_0_3, cvt_0_3, cvt_0_7, accum_0_3;\n" \

            // fma for A[8 - 15] and B[8 - 15]
            "fma.rn.f16x2 accum_1_0, cvt_1_0, cvt_1_4, accum_1_0;\n" \
            "fma.rn.f16x2 accum_1_1, cvt_1_1, cvt_1_5, accum_1_1;\n" \
            "fma.rn.f16x2 accum_1_2, cvt_1_2, cvt_1_6, accum_1_2;\n" \
            "fma.rn.f16x2 accum_1_3, cvt_1_3, cvt_1_7, accum_1_3;\n" \

            // fma for A[16 - 23] and B[16 - 23]
            "fma.rn.f16x2 accum_2_0, cvt_2_0, cvt_2_4, accum_2_0;\n" \
            "fma.rn.f16x2 accum_2_1, cvt_2_1, cvt_2_5, accum_2_1;\n" \
            "fma.rn.f16x2 accum_2_2, cvt_2_2, cvt_2_6, accum_2_2;\n" \
            "fma.rn.f16x2 accum_2_3, cvt_2_3, cvt_2_7, accum_2_3;\n" \

            // fma for A[24 - 31] and B[24 - 31]
            "fma.rn.f16x2 accum_3_0, cvt_3_0, cvt_3_4, accum_3_0;\n" \
            "fma.rn.f16x2 accum_3_1, cvt_3_1, cvt_3_5, accum_3_1;\n" \
            "fma.rn.f16x2 accum_3_2, cvt_3_2, cvt_3_6, accum_3_2;\n" \
            "fma.rn.f16x2 accum_3_3, cvt_3_3, cvt_3_7, accum_3_3;\n" \

            // tree reduction for accumulators
            "add.rn.f16x2 accum_0_0, accum_0_0, accum_0_1;\n" \
            "add.rn.f16x2 accum_0_2, accum_0_2, accum_0_3;\n" \
            "add.rn.f16x2 accum_1_0, accum_1_0, accum_1_1;\n" \
            "add.rn.f16x2 accum_1_2, accum_1_2, accum_1_3;\n" \
            "add.rn.f16x2 accum_2_0, accum_2_0, accum_2_1;\n" \
            "add.rn.f16x2 accum_2_2, accum_2_2, accum_2_3;\n" \
            "add.rn.f16x2 accum_3_0, accum_3_0, accum_3_1;\n" \
            "add.rn.f16x2 accum_3_2, accum_3_2, accum_3_3;\n" \

            "add.rn.f16x2 accum_0_0, accum_0_0, accum_0_2;\n" \
            "add.rn.f16x2 accum_1_0, accum_1_0, accum_1_2;\n" \
            "add.rn.f16x2 accum_2_0, accum_2_0, accum_2_2;\n" \
            "add.rn.f16x2 accum_3_0, accum_3_0, accum_3_2;\n" \

            "add.rn.f16x2 accum_0_0, accum_0_0, accum_1_0;\n" \
            "add.rn.f16x2 accum_2_0, accum_2_0, accum_3_0;\n" \

            // apply scaling factors and final reduction
            "mul.rn.f16x2 accum_0_0, mul_f16x2_0, accum_0_0;\n" \
            "mul.rn.f16x2 accum_2_0, mul_f16x2_1, accum_2_0;\n" \

            "add.rn.f16x2 accum_0_0, accum_0_0, accum_2_0;\n" \
            
            "mov.b32 {lane0, lane1}, accum_0_0;\n" \
            "add.rn.f16 result_f16, lane0, lane1;\n" \

            "mov.b16 %0, result_f16;\n" \

            "}\n"
            : "=h"(out_fp16[0])                                     // 0
            : "h"(src_fragSFA_packed), "h"(src_fragSFB_packed),     // 1, 2
              "r"(src_fragA_packed[0]), "r"(src_fragB_packed[0]),   // 3, 4
              "r"(src_fragA_packed[1]), "r"(src_fragB_packed[1]),   // 5, 6
              "r"(src_fragA_packed[2]), "r"(src_fragB_packed[2]),   // 7, 8
              "r"(src_fragA_packed[3]), "r"(src_fragB_packed[3])    // 9, 10
            : "memory"
        );

        return out;

      #else
        NumericArrayConverter<ElementAccumulator, ElementA, kElementsPerAccess, Round> srcA_converter;
        NumericArrayConverter<ElementAccumulator, ElementB, kElementsPerAccess, Round> srcB_converter;
        NumericConverter<ElementAccumulator, ElementSFA, Round> SFA_converter;
        NumericConverter<ElementAccumulator, ElementSFB, Round> SFB_converter;

        FragmentCompute fragA_Compute = srcA_converter(fragA);
        FragmentCompute fragB_Compute = srcB_converter(fragB);
        ElementAccumulator accum = ElementAccumulator(0);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kSFPerAccess; i++) {
          ElementAccumulator accum_SF_block = ElementAccumulator(0);

          int local_k_offset = i * kSFVecSize;
          ElementAccumulator multiplier{1};
                  
          multiplier = SFA_converter(fragSFA.at(i)) * SFB_converter(fragSFB.at(i));


          CUTLASS_PRAGMA_UNROLL
          for (int e = 0; e < kSFVecSize; e++) {
            accum_SF_block += fragA_Compute.at(e + local_k_offset) * fragB_Compute.at(e + local_k_offset);
          }

          accum_SF_block *= multiplier;
          accum += accum_SF_block;
        }

        return accum;

      #endif
  }

  CUTLASS_DEVICE
  ElementAccumulator process_tail_elements(
      int unroll_col_k,
      int idx_col_k,
      int gemm_k,
      ElementA const *ptr_A,
      ElementB const *ptr_B,
      ElementSFA const *ptr_SF_A,
      ElementSFB const *ptr_SF_B,
      NumericConverter<ElementAccumulator, ElementA, Round> const &A_converter,
      NumericConverter<ElementAccumulator, ElementB, Round> const &B_converter,
      NumericConverter<ElementAccumulator, ElementSFA, Round> const &SFA_converter,
      NumericConverter<ElementAccumulator, ElementSFB, Round> const &SFB_converter) {

      ElementAccumulator accum = ElementAccumulator(0);

      // calculate the rest of K elements
      // each thread fetch 1 element each time
      for (int k = unroll_col_k + idx_col_k * kPackedElementsA; k < gemm_k; k += kThreadsPerRow * kPackedElementsA) {
        // blockscaled GEMV
        int SF_idx = k / kSFVecSize;
        int SF_offset_by_k = ((SF_idx >> 2) << 9) + (SF_idx & 0x3);

        ElementSFA sfa = *(ptr_SF_A + SF_offset_by_k);
        ElementSFB sfb = *(ptr_SF_B + SF_offset_by_k);

        FragmentPackedA fragA;
        FragmentPackedB fragB;

        // fetch from matrix A
        arch::global_load<FragmentPackedA, sizeof(FragmentPackedA), arch::CacheOperation::Always>(
          fragA,
          ptr_A - (idx_col_k * kElementsPerAccess - k) / kPackedElementsA,
          true);

        // fetch from vector B
        arch::global_load<FragmentPackedB, sizeof(FragmentPackedB), arch::CacheOperation::Always>(
          fragB,
          ptr_B - (idx_col_k * kElementsPerAccess - k) / kPackedElementsB,
          true);

        ElementAccumulator accum_SF_packed = ElementAccumulator(0);

        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < kPackedElements; e++) {
          accum_SF_packed += A_converter(fragA.at(e)) * B_converter(fragB.at(e));
        }

        accum_SF_packed *= SFA_converter(sfa) * SFB_converter(sfb);

        accum += accum_SF_packed;

      }

      return accum;
  }

  // Load fragments from shared memory
  template<typename FragmentA, typename FragmentB, typename FragmentSFA, typename FragmentSFB>
  CUTLASS_DEVICE 
  void load_smem_fragments(
      FragmentA& fragA,
      FragmentB& fragB,
      FragmentSFA& fragSFA,
      FragmentSFB& fragSFB,
      int smem_pipe_idx,
      int k_block,
      int smem_offset_A,
      int smem_offset_B,
      int smem_sf_offset,
      SharedStorage& shared_storage) const {
    
    // Load A/B fragments
    arch::shared_load(fragA, &shared_storage.smem_A[smem_pipe_idx][k_block][smem_offset_A]);
    arch::shared_load(fragB, &shared_storage.smem_B[smem_pipe_idx][k_block][smem_offset_B]);
    
    // Load SF fragments
    uint32_t smem_ptr = cutlass::arch::cutlass_get_smem_pointer(&shared_storage.smem_SFA[smem_pipe_idx][k_block][smem_sf_offset]);
    arch::shared_load<2>(&fragSFA, smem_ptr);
    smem_ptr = cutlass::arch::cutlass_get_smem_pointer(&shared_storage.smem_SFB[smem_pipe_idx][k_block][threadIdx.x * kSFPerAccess]);
    arch::shared_load<2>(&fragSFB, smem_ptr);

  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
