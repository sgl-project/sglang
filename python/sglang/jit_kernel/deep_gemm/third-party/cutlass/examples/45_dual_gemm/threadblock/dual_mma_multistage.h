/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/threadblock/mma_base.h"
#include "dual_mma_base.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B0 operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB0_,
    /// Iterates over tiles of B0 operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB0_,
    /// Cache operation for operand B
    cutlass::arch::CacheOperation::Kind CacheOpB,
    /// Iterates over tiles of B1 operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB1_,
    /// Iterates over tiles of B1 operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB1_,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy0_,
    /// B1-specific version of the policy (concept: MmaPolicy)
    typename Policy1_,
    /// Number of stages,
    int Stages,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Used for partial specialization
    typename Enable = bool>
class DualMmaMultistage : 
  public DualMmaBase<Shape_, Policy0_, Policy1_, Stages> {
public:
  ///< Base class
  using Base = DualMmaBase<Shape_, Policy0_, Policy1_, Stages>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B0 operand in global memory
  using IteratorB0 = IteratorB0_;
  ///< Iterates over tiles of B1 operand in global memory
  using IteratorB1 = IteratorB1_;
  ///< Data type of accumulator matrix
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;
  ///< Policy describing tuning details
  using Policy0 = Policy0_;
  using Policy1 = Policy1_;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB0 = SmemIteratorB0_;
  using SmemIteratorB1 = SmemIteratorB1_;

  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC = typename Policy0::Operator::FragmentC;

  /// Warp-level Mma
  using Operator0 = typename Policy0::Operator;
  using Operator1 = typename Policy1::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = arch::Sm80;
  
  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator0::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB0 = Operator0::kTransformB;
  static ComplexTransform const kTransformB1 = Operator1::kTransformB;

  /// Internal structure exposed for introspection.
  struct Detail {

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB =
        IteratorB0::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = Stages;

    /// Number of cp.async instructions to load on group of operand A
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;
  };

 private:

  using WarpLoadedFragmentA = typename Operator0::FragmentA;
  using WarpLoadedFragmentB0 = typename Operator0::FragmentB;
  using WarpLoadedFragmentB1 = typename Operator1::FragmentB;
  using WarpTransformedFragmentA = typename Operator0::TransformedFragmentA;
  using WarpTransformedFragmentB0 = typename Operator0::TransformedFragmentB;
  using WarpTransformedFragmentB1 = typename Operator1::TransformedFragmentB;

 private:

  //
  // Data members
  //

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB0 smem_iterator_B0_;
  SmemIteratorB1 smem_iterator_B1_;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  DualMmaMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B0_(shared_storage.operand_B0_ref(), thread_idx),
      smem_iterator_B1_(shared_storage.operand_B1_ref(), thread_idx)
  {
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B0_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
    this->warp_tile_iterator_B1_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }

  CUTLASS_DEVICE
  void copy_tiles_and_advance(IteratorA &iterator_A, IteratorB0 &iterator_B0, IteratorB1 &iterator_B1,
                              int group_start_A = 0, int group_start_B = 0) {
    iterator_A.set_iteration_index(group_start_A *
                                   IteratorA::kAccessesPerVector);
    this->smem_iterator_A_.set_iteration_index(group_start_A);

    // Async Copy for operand A
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {
      if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                this->smem_iterator_A_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value *
                              IteratorA::ThreadMap::kElementsPerAccess /
                              IteratorA::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_A.get();

          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
                dst_ptr + v, gmem_ptr, iterator_A.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpA>(
                dst_ptr + v, gmem_ptr, iterator_A.valid());
          }

          ++iterator_A;
        }

        ++this->smem_iterator_A_;
      }
    }

    iterator_B0.set_iteration_index(group_start_B *
                                   IteratorB0::kAccessesPerVector);
    iterator_B1.set_iteration_index(group_start_B *
                                   IteratorB1::kAccessesPerVector);
    this->smem_iterator_B0_.set_iteration_index(group_start_B);
    this->smem_iterator_B1_.set_iteration_index(group_start_B);

    // Async Copy for operand B0
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
        typename IteratorB0::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB0::AccessType *>(
                this->smem_iterator_B0_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorB0::Element>::value *
                              IteratorB0::ThreadMap::kElementsPerAccess /
                              IteratorB0::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB0::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_B0.get();

          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
                dst_ptr + v, gmem_ptr, iterator_B0.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpB>(
                dst_ptr + v, gmem_ptr, iterator_B0.valid());
          }

          ++iterator_B0;
        }
        ++this->smem_iterator_B0_;
      }
    }
    // Async Copy for operand B1
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
        typename IteratorB1::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB1::AccessType *>(
                this->smem_iterator_B1_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorB1::Element>::value *
                              IteratorB1::ThreadMap::kElementsPerAccess /
                              IteratorB1::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB1::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_B1.get();

          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
                dst_ptr + v, gmem_ptr, iterator_B1.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpB>(
                dst_ptr + v, gmem_ptr, iterator_B1.valid());
          }

          ++iterator_B1;
        }
        ++this->smem_iterator_B1_;
      }
    }
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum0,
      FragmentC &accum1,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB0 iterator_B0,
      IteratorB1 iterator_B1,
      ///< initial value of accumulator
      FragmentC const &src_accum0,
      FragmentC const &src_accum1
    ) {

    //
    // Prologue
    //

    // Issue several complete stages
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1;
         ++stage, --gemm_k_iterations) {

      iterator_A.clear_mask(gemm_k_iterations == 0);
      iterator_B0.clear_mask(gemm_k_iterations == 0);
      iterator_B1.clear_mask(gemm_k_iterations == 0);

      iterator_A.set_iteration_index(0);
      this->smem_iterator_A_.set_iteration_index(0);

      // Async Copy for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                this->smem_iterator_A_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorA::Element>::value *
              IteratorA::ThreadMap::kElementsPerAccess /
              IteratorA::kAccessesPerVector / 8;

          int src_bytes = (iterator_A.valid() ? kSrcBytes : 0);

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
              dst_ptr + v, iterator_A.get(), iterator_A.valid());

          ++iterator_A;
        }

        ++this->smem_iterator_A_;
      }

      iterator_B0.set_iteration_index(0);
      iterator_B1.set_iteration_index(0);
      this->smem_iterator_B0_.set_iteration_index(0);
      this->smem_iterator_B1_.set_iteration_index(0);

      // Async Copy for operand B0
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB0::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB0::AccessType *>(
                this->smem_iterator_B0_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB0::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorB0::Element>::value *
              IteratorB0::ThreadMap::kElementsPerAccess /
              IteratorB0::kAccessesPerVector / 8;

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
              dst_ptr + v, iterator_B0.get(), iterator_B0.valid());

          ++iterator_B0;
        }

        ++this->smem_iterator_B0_;
      }
      // Async Copy for operand B1
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB1::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB1::AccessType *>(
                this->smem_iterator_B1_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB1::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorB1::Element>::value *
              IteratorB1::ThreadMap::kElementsPerAccess /
              IteratorB1::kAccessesPerVector / 8;

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
              dst_ptr + v, iterator_B1.get(), iterator_B1.valid());

          ++iterator_B1;
        }

        ++this->smem_iterator_B1_;
      }

      // Move to the next stage
      iterator_A.add_tile_offset({0, 1});
      iterator_B0.add_tile_offset({1, 0});
      iterator_B1.add_tile_offset({1, 0});

      this->smem_iterator_A_.add_tile_offset({0, 1});
      this->smem_iterator_B0_.add_tile_offset({1, 0});
      this->smem_iterator_B1_.add_tile_offset({1, 0});

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();
    }

    // Perform accumulation in the 'd' output operand
    accum0 = src_accum0;
    accum1 = src_accum1;

    //
    // Clear the remaining tiles of SMEM. This is a functional requirement for some kernels
    // so that all accumulator elements outside the GEMM footprint are zero.
    //

    if (SharedMemoryClear == SharedMemoryClearOption::kClearLastStage) {

      /// Iterator to write threadblock-scoped tile of A operand to shared memory
      SmemIteratorA last_smem_iterator_A(this->smem_iterator_A_);

      typename IteratorA::AccessType zero_A;
      zero_A.clear();

      last_smem_iterator_A.set_iteration_index(0);

      // Async Copy for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {

        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                last_smem_iterator_A.get());

        *dst_ptr = zero_A;

        ++last_smem_iterator_A;
      }

      typename IteratorB0::AccessType zero_B;
      zero_B.clear();

      /// Iterator to write threadblock-scoped tile of B0 operand to shared memory
      SmemIteratorB0 last_smem_iterator_B0(this->smem_iterator_B0_);
      last_smem_iterator_B0.set_iteration_index(0);

      // Async Copy for operand B0
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB0::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB0::AccessType *>(
                last_smem_iterator_B0.get());

        *dst_ptr = zero_B;

        ++last_smem_iterator_B0;
      }

      /// Iterator to write threadblock-scoped tile of B1 operand to shared memory
      SmemIteratorB1 last_smem_iterator_B1(this->smem_iterator_B1_);
      last_smem_iterator_B1.set_iteration_index(0);

      // Async Copy for operand B1
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {

        typename IteratorB1::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB1::AccessType *>(
                last_smem_iterator_B1.get());

        *dst_ptr = zero_B;

        ++last_smem_iterator_B1;
      }
    }

    // Waits until stages up to the previous (kStages-2)th stage have committed.
    cutlass::arch::cp_async_wait<Base::kStages - 2>();
    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpLoadedFragmentA warp_loaded_frag_A[2];
    WarpLoadedFragmentB0 warp_loaded_frag_B0[2];
    WarpLoadedFragmentB1 warp_loaded_frag_B1[2];
    WarpTransformedFragmentA warp_transformed_frag_A[2];
    WarpTransformedFragmentB0 warp_transformed_frag_B0[2];
    WarpTransformedFragmentB1 warp_transformed_frag_B1[2];

    Operator0 warp_mma0;
    Operator1 warp_mma1;

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_B0_.set_kgroup_index(0);
    this->warp_tile_iterator_B1_.set_kgroup_index(0);

    this->warp_tile_iterator_A_.load(warp_loaded_frag_A[0]);
    this->warp_tile_iterator_B0_.load(warp_loaded_frag_B0[0]);
    this->warp_tile_iterator_B1_.load(warp_loaded_frag_B1[0]);

    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B0_;
    ++this->warp_tile_iterator_B1_;

    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B0.clear_mask(gemm_k_iterations == 0);
    iterator_B1.clear_mask(gemm_k_iterations == 0);

    int smem_write_stage_idx = Base::kStages - 1;
    int smem_read_stage_idx = 0;

    warp_mma0.transform(warp_transformed_frag_A[0], warp_transformed_frag_B0[0],
                        warp_loaded_frag_A[0], warp_loaded_frag_B0[0]);
    warp_mma1.transform(warp_transformed_frag_A[0], warp_transformed_frag_B1[0],
                        warp_loaded_frag_A[0], warp_loaded_frag_B1[0]);

    // tf32x3 kernels use staging accumulation. warp_mma uses a temporary
    // accumulator and this temporary accumulator is added to the final
    // accumulator once in every mainloop iteration.
    plus<FragmentC> plus_accum;

    FragmentC tmp_accum0, tmp_accum1;

    if (platform::is_same<typename Operator0::MathOperator,
                          arch::OpMultiplyAddFastF32>::value
      || platform::is_same<typename Operator0::MathOperator,
                           arch::OpMultiplyAddComplexFastF32>::value) {

      tmp_accum0.clear();
      tmp_accum1.clear();
    }

    //
    // Mainloop
    //

    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > (-Base::kStages + 1);) {
      //
      // Loop over GEMM K dimension
      //

      // Computes a warp-level GEMM on data held in shared memory
      // Each "warp_mma_k" refers to a warp-level matrix multiply-accumulate
      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
           ++warp_mma_k) {

        // Load warp-level tiles from shared memory, wrapping to k offset if
        // this is the last group as the case may be.

        this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B0_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B1_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        
        this->warp_tile_iterator_A_.load(warp_loaded_frag_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B0_.load(warp_loaded_frag_B0[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B1_.load(warp_loaded_frag_B1[(warp_mma_k + 1) % 2]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B0_;
        ++this->warp_tile_iterator_B1_;

        if (warp_mma_k > 0) {
          warp_mma0.transform(warp_transformed_frag_A[warp_mma_k % 2],
                              warp_transformed_frag_B0[warp_mma_k % 2],
                              warp_loaded_frag_A[warp_mma_k % 2],
                              warp_loaded_frag_B0[warp_mma_k % 2]);
          warp_mma1.transform(warp_transformed_frag_A[warp_mma_k % 2],
                              warp_transformed_frag_B1[warp_mma_k % 2],
                              warp_loaded_frag_A[warp_mma_k % 2],
                              warp_loaded_frag_B1[warp_mma_k % 2]);
        }

        if (platform::is_same<typename Operator0::MathOperator,
                              arch::OpMultiplyAddFastF32>::value
          || platform::is_same<typename Operator0::MathOperator,
                               arch::OpMultiplyAddComplexFastF32>::value) {

          warp_mma0(
            tmp_accum0,
            warp_transformed_frag_A[warp_mma_k % 2],
            warp_transformed_frag_B0[warp_mma_k % 2], 
            tmp_accum0
          );
          warp_mma1(
            tmp_accum1,
            warp_transformed_frag_A[warp_mma_k % 2],
            warp_transformed_frag_B1[warp_mma_k % 2], 
            tmp_accum1
          );

          if (warp_mma_k == 0) {
            accum0 = plus_accum(accum0, tmp_accum0);
            accum1 = plus_accum(accum1, tmp_accum1);
            tmp_accum0.clear();
            tmp_accum1.clear();
          }
        } else {
          warp_mma0(
            accum0,
            warp_transformed_frag_A[warp_mma_k % 2],
            warp_transformed_frag_B0[warp_mma_k % 2],
            accum0
          );
          warp_mma1(
            accum1,
            warp_transformed_frag_A[warp_mma_k % 2],
            warp_transformed_frag_B1[warp_mma_k % 2],
            accum1
          );
        }

        // Issue global->shared copies for the this stage
        if (warp_mma_k < Base::kWarpGemmIterations - 1) {
          int group_start_iteration_A, group_start_iteration_B;

          group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
          group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;

          copy_tiles_and_advance(iterator_A, iterator_B0, iterator_B1, group_start_iteration_A, 
                               group_start_iteration_B);
        }

        if (warp_mma_k + 2 == Base::kWarpGemmIterations) {
          int group_start_iteration_A, group_start_iteration_B;
          group_start_iteration_A =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
          group_start_iteration_B =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupB;

          copy_tiles_and_advance(iterator_A, iterator_B0, iterator_B1, group_start_iteration_A, 
                               group_start_iteration_B);

          // Inserts a memory fence between stages of cp.async instructions.
          cutlass::arch::cp_async_fence();

          // Waits until stages up to the previous (kStages-2)th stage have committed.
          arch::cp_async_wait<Base::kStages - 2>();
          __syncthreads();

          // Move to the next stage
          iterator_A.add_tile_offset({0, 1});
          iterator_B0.add_tile_offset({1, 0});
          iterator_B1.add_tile_offset({1, 0});

          this->smem_iterator_A_.add_tile_offset({0, 1});
          this->smem_iterator_B0_.add_tile_offset({1, 0});
          this->smem_iterator_B1_.add_tile_offset({1, 0});

          // Add negative offsets to return iterators to the 'start' of the
          // circular buffer in shared memory
          if (smem_write_stage_idx == (Base::kStages - 1)) {
            this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B0_.add_tile_offset({-Base::kStages, 0});
            this->smem_iterator_B1_.add_tile_offset({-Base::kStages, 0});
            smem_write_stage_idx = 0;
          } else {
            ++smem_write_stage_idx;
          }

          if (smem_read_stage_idx == (Base::kStages - 1)) {
            this->warp_tile_iterator_A_.add_tile_offset(
                {0, -Base::kStages * Policy0::kPartitionsK *
                        Base::kWarpGemmIterations});
            this->warp_tile_iterator_B0_.add_tile_offset(
                {-Base::kStages * Policy0::kPartitionsK *
                     Base::kWarpGemmIterations,
                 0});
            this->warp_tile_iterator_B1_.add_tile_offset(
                {-Base::kStages * Policy1::kPartitionsK *
                     Base::kWarpGemmIterations,
                 0});
            smem_read_stage_idx = 0;
          } else {
            ++smem_read_stage_idx;
          }

          --gemm_k_iterations;
          iterator_A.clear_mask(gemm_k_iterations == 0);
          iterator_B0.clear_mask(gemm_k_iterations == 0);
          iterator_B1.clear_mask(gemm_k_iterations == 0);
        }

        // Do any conversions feeding the first stage at the end of the loop so
        // we can start right away on mma instructions
        if (warp_mma_k + 1 == Base::kWarpGemmIterations) {
          warp_mma0.transform(warp_transformed_frag_A[(warp_mma_k + 1) % 2],
                              warp_transformed_frag_B0[(warp_mma_k + 1) % 2],
                              warp_loaded_frag_A[(warp_mma_k + 1) % 2],
                              warp_loaded_frag_B0[(warp_mma_k + 1) % 2]);
          warp_mma1.transform(warp_transformed_frag_A[(warp_mma_k + 1) % 2],
                              warp_transformed_frag_B1[(warp_mma_k + 1) % 2],
                              warp_loaded_frag_A[(warp_mma_k + 1) % 2],
                              warp_loaded_frag_B1[(warp_mma_k + 1) % 2]);
        }
      }

    }

    if (platform::is_same<typename Operator0::MathOperator,
                          arch::OpMultiplyAddFastF32>::value
      || platform::is_same<typename Operator0::MathOperator,
                           arch::OpMultiplyAddComplexFastF32>::value) {
      accum0 = plus_accum(accum0, tmp_accum0); 
      accum1 = plus_accum(accum1, tmp_accum1); 
    }
 
    // commit and drain all pending and predicated cp.async pnz from the GEMM mainloop
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
