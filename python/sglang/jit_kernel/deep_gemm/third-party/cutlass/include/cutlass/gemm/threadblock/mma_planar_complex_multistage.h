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

#include "cutlass/cutlass.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/array_planar_complex.h"
#include "cutlass/functional.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/mma_planar_complex_base.h"

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
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    cutlass::arch::CacheOperation::Kind CacheOpB,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Transformation applied to A
    ComplexTransform TransformA = ComplexTransform::kNone,
    /// Transformation applied to B
    ComplexTransform TransformB = ComplexTransform::kNone
>
class MmaPlanarComplexMultistage : 
  public MmaPlanarComplexBase<Shape_, Policy_, Stages> {
public:
  ///< Base class
  using Base = MmaPlanarComplexBase<Shape_, Policy_, Stages>;

  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;

  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;

  ///< Data type of accumulator matrix
  using ElementC = ElementC_;

  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;

  ///< Policy describing tuning details
  using Policy = Policy_;

  ///< Architecture tag
  using ArchTag = arch::Sm80;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  /// Transformation applied to A
  static ComplexTransform const kTransformA = TransformA;

  /// Transformation applied to B
  static ComplexTransform const kTransformB = TransformB;

  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC = ArrayPlanarComplex<
    typename Policy::Operator::FragmentC::Element,
    Policy::Operator::FragmentC::kElements
  >;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Internal structure exposed for introspection.
  struct Detail {

    static_assert(Base::kWarpGemmIterations > 1,
                  "The pipelined structure requires at least two warp-level "
                  "GEMM operations.");

    /// Number of cp.async instructions to load one stage of operand A
    static int const TBLoadIterationsA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const TBLoadIterationsB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = Stages;

    static int const kAccessesPerGroupA =
        (TBLoadIterationsA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    static int const kAccessesPerGroupB =
        (TBLoadIterationsB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;
  };

 private:

  using WarpFragmentA = typename Operator::FragmentA;
  using WarpFragmentB = typename Operator::FragmentB;

 private:

  //
  // Data members
  //

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  MmaPlanarComplexMultistage(
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
      smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx)
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
    this->warp_tile_iterator_A_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset({Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }

private:

  CUTLASS_DEVICE
  void copy_tiles_and_advance(
    IteratorA &iterator_A_real,
    IteratorA &iterator_A_imag,
    
    IteratorB &iterator_B_real, 
    IteratorB &iterator_B_imag, 
    
    int group_start_A = 0, 
    int group_start_B = 0) {

    iterator_A_real.set_iteration_index(group_start_A * IteratorA::kAccessesPerVector);
    iterator_A_imag.set_iteration_index(group_start_A * IteratorA::kAccessesPerVector);
    this->smem_iterator_A_.set_iteration_index(group_start_A);

    // Load for operand A
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {
        
      typename IteratorA::AccessType *dst_ptr = 
        reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_A_.get());
          
      int const kSrcBytes = 
        sizeof_bits<typename IteratorA::Element>::value * 
        IteratorA::ThreadMap::kElementsPerAccess / IteratorA::kAccessesPerVector / 8;

      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {

        auto gmem_ptr_real = iterator_A_real.get();
        auto gmem_ptr_imag = iterator_A_imag.get();

        bool pred_guard = iterator_A_real.valid();
        cutlass::arch::cp_async<kSrcBytes, kCacheOpA>(
            dst_ptr + v,
            gmem_ptr_real,
            pred_guard);
        cutlass::arch::cp_async<kSrcBytes, kCacheOpA>(
            dst_ptr + v + (Base::SharedStorage::kImaginaryStrideA / IteratorA::ThreadMap::kElementsPerAccess),
            reinterpret_cast<char const *>(gmem_ptr_imag),
            pred_guard);

        ++iterator_A_real;
        ++iterator_A_imag;
      }

      ++this->smem_iterator_A_;
    }

    iterator_B_real.set_iteration_index(group_start_B * IteratorB::kAccessesPerVector);
    iterator_B_imag.set_iteration_index(group_start_B * IteratorB::kAccessesPerVector);
    this->smem_iterator_B_.set_iteration_index(group_start_B);

    // Load for operand B
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      typename IteratorB::AccessType *dst_ptr = 
        reinterpret_cast<typename IteratorB::AccessType *>(this->smem_iterator_B_.get());
      
      int const kSrcBytes = 
        sizeof_bits<typename IteratorB::Element>::value * 
        IteratorB::ThreadMap::kElementsPerAccess / IteratorB::kAccessesPerVector / 8;

      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
        auto gmem_ptr_real = iterator_B_real.get();
        auto gmem_ptr_imag = iterator_B_imag.get();

        bool pred_guard = iterator_B_real.valid();
        cutlass::arch::cp_async<kSrcBytes, kCacheOpB>(
            dst_ptr + v,
            gmem_ptr_real,
            pred_guard);
        cutlass::arch::cp_async<kSrcBytes, kCacheOpB>(
            dst_ptr + v + (Base::SharedStorage::kImaginaryStrideB / IteratorB::ThreadMap::kElementsPerAccess),
            reinterpret_cast<char const *>(gmem_ptr_imag),
            pred_guard);

        ++iterator_B_real;
        ++iterator_B_imag;
      }
      ++this->smem_iterator_B_;
    }
  }

  CUTLASS_DEVICE
  void warp_mma_planar_complex(
    Operator & warp_mma, 
    FragmentC &accum,
    WarpFragmentA const & real_A, 
    WarpFragmentA const & imag_A, 
    WarpFragmentB const & real_B, 
    WarpFragmentB const & imag_B) {

    cutlass::negate<Array<typename WarpFragmentB::Element, WarpFragmentB::kElements>> neg_op_B;

    WarpFragmentB neg_real_B = neg_op_B(real_B);
    WarpFragmentB neg_imag_B = neg_op_B(imag_B);

    warp_mma(accum.real, real_A, real_B, accum.real);  

    if (kTransformB == ComplexTransform::kNone) {
      warp_mma(accum.imag, real_A, imag_B, accum.imag);
    }
    else {
      warp_mma(accum.imag, real_A, neg_imag_B, accum.imag);
    }

    if (kTransformA == ComplexTransform::kNone) {
      warp_mma(accum.imag, imag_A, real_B, accum.imag);
    }
    else {
      warp_mma(accum.imag, imag_A, neg_real_B, accum.imag);
    }

    if (kTransformA == ComplexTransform::kNone ^ kTransformB == ComplexTransform::kNone) {
      warp_mma(accum.real, imag_A, imag_B, accum.real);
    }
    else {
      warp_mma(accum.real, imag_A, neg_imag_B, accum.real);
    }
  }

public:
  
  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA iterator_A_real,
      ///< iterator over A operand in global memory
      IteratorA iterator_A_imag,
      ///< iterator over B operand in global memory
      IteratorB iterator_B_real,
      ///< iterator over B operand in global memory
      IteratorB iterator_B_imag,
      ///< initial value of accumulator
      FragmentC const &src_accum) {

    //
    // Prologue
    //

    // Issue several complete stages
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1;
         ++stage, --gemm_k_iterations) {

      iterator_A_real.clear_mask(gemm_k_iterations == 0);
      iterator_A_imag.clear_mask(gemm_k_iterations == 0);
      iterator_B_real.clear_mask(gemm_k_iterations == 0);
      iterator_B_imag.clear_mask(gemm_k_iterations == 0);

      iterator_A_real.set_iteration_index(0);
      iterator_A_imag.set_iteration_index(0);

      this->smem_iterator_A_.set_iteration_index(0);

      // Load for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::TBLoadIterationsA; ++j) {

        typename IteratorA::AccessType *dst_ptr = 
          reinterpret_cast<typename IteratorA::AccessType *>(this->smem_iterator_A_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {

          int const kSrcBytes = 
            sizeof_bits<typename IteratorA::Element>::value * 
            IteratorA::ThreadMap::kElementsPerAccess / IteratorA::kAccessesPerVector / 8;

          bool pred_guard = iterator_A_real.valid();

          auto src_ptr_real = iterator_A_real.get();
          auto src_ptr_imag = iterator_A_imag.get();

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
              dst_ptr + v, src_ptr_real, pred_guard);

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
              dst_ptr + v +
                  Base::SharedStorage::kImaginaryStrideA /
                      IteratorA::ThreadMap::kElementsPerAccess,
              reinterpret_cast<char const *>(src_ptr_imag),
              pred_guard);

          ++iterator_A_real;
          ++iterator_A_imag;
        }

        ++this->smem_iterator_A_;
      }

      iterator_B_real.set_iteration_index(0);
      iterator_B_imag.set_iteration_index(0);

      this->smem_iterator_B_.set_iteration_index(0);

      // Load for operand B
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::TBLoadIterationsB; ++j) {

        typename IteratorB::AccessType *dst_ptr = 
          reinterpret_cast<typename IteratorB::AccessType *>(this->smem_iterator_B_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {

          int const kSrcBytes = 
            sizeof_bits<typename IteratorB::Element>::value * 
            IteratorB::ThreadMap::kElementsPerAccess / IteratorB::kAccessesPerVector / 8;

          bool pred_guard = iterator_B_real.valid();

          auto src_ptr_real = iterator_B_real.get();
          auto src_ptr_imag = iterator_B_imag.get();

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
            dst_ptr + v, src_ptr_real, pred_guard);

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
              dst_ptr + v +
                  Base::SharedStorage::kImaginaryStrideB /
                      IteratorB::ThreadMap::kElementsPerAccess,
              reinterpret_cast<char const *>(src_ptr_imag),
              pred_guard);

          ++iterator_B_real;
          ++iterator_B_imag;
        }

        ++this->smem_iterator_B_;
      }

      // Move to the next stage
      iterator_A_real.add_tile_offset({0, 1});
      iterator_A_imag.add_tile_offset({0, 1});

      iterator_B_real.add_tile_offset({1, 0});
      iterator_B_imag.add_tile_offset({1, 0});

      this->smem_iterator_A_.add_tile_offset({0, 1});
      this->smem_iterator_B_.add_tile_offset({1, 0});

      // Inserts a memory fence between stages of cp.async instructions
      cutlass::arch::cp_async_fence();
    }

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    // Blocks until all but kStages-2 cp.async stages have committed.
    cutlass::arch::cp_async_wait<Base::kStages - 2>();
    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions

    WarpFragmentA warp_frag_real_A[2];
    WarpFragmentA warp_frag_imag_A[2];

    WarpFragmentB warp_frag_real_B[2];
    WarpFragmentB warp_frag_imag_B[2];

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.set_kgroup_index(0);

    this->warp_tile_iterator_A_.load(warp_frag_real_A[0]);
    this->warp_tile_iterator_A_.load_with_pointer_offset(warp_frag_imag_A[0], Base::SharedStorage::kImaginaryStrideA);

    this->warp_tile_iterator_B_.load(warp_frag_real_B[0]);
    this->warp_tile_iterator_B_.load_with_pointer_offset(warp_frag_imag_B[0], Base::SharedStorage::kImaginaryStrideB);

    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B_;

    iterator_A_real.clear_mask(gemm_k_iterations == 0);
    iterator_A_imag.clear_mask(gemm_k_iterations == 0);
    iterator_B_real.clear_mask(gemm_k_iterations == 0);
    iterator_B_imag.clear_mask(gemm_k_iterations == 0);

    // Start issuing the first group of the next stage outside of the mainloop
    copy_tiles_and_advance(iterator_A_real, iterator_A_imag, iterator_B_real, iterator_B_imag);

    Operator warp_mma;

    int smem_write_stage_idx = Base::kStages - 1;
    int smem_read_stage_idx = 0;

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
        this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        
        this->warp_tile_iterator_A_.load(warp_frag_real_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_A_.load_with_pointer_offset(warp_frag_imag_A[(warp_mma_k + 1) % 2], Base::SharedStorage::kImaginaryStrideA);
        
        this->warp_tile_iterator_B_.load(warp_frag_real_B[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load_with_pointer_offset(warp_frag_imag_B[(warp_mma_k + 1) % 2], Base::SharedStorage::kImaginaryStrideB);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        // Issue global->shared copies for the next stage
        int group_start_iteration_A, group_start_iteration_B;

        if (warp_mma_k + 1 == Base::kWarpGemmIterations) {
          group_start_iteration_A = 0;
          group_start_iteration_B = 0;
        }
        else {
          group_start_iteration_A = (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
          group_start_iteration_B = (warp_mma_k + 1) * Detail::kAccessesPerGroupB;
        }
    
        copy_tiles_and_advance(
          iterator_A_real, 
          iterator_A_imag,
          iterator_B_real, 
          iterator_B_imag,
          group_start_iteration_A, 
          group_start_iteration_B);

        if (warp_mma_k + 2 == Base::kWarpGemmIterations) {
          // Inserts a memory fence between stages of cp.async instructions
          cutlass::arch::cp_async_fence();

          // Blocks until all but kStages-2 cp.async stages have committed.
          arch::cp_async_wait<Base::kStages - 2>();
          __syncthreads();

          // Move to the next stage
          iterator_A_real.add_tile_offset({0, 1});
          iterator_A_imag.add_tile_offset({0, 1});
          
          iterator_B_real.add_tile_offset({1, 0});
          iterator_B_imag.add_tile_offset({1, 0});

          this->smem_iterator_A_.add_tile_offset({0, 1});
          this->smem_iterator_B_.add_tile_offset({1, 0});

          // Add negative offsets to return iterators to the 'start' of the
          // circular buffer in shared memory
          if (smem_write_stage_idx == (Base::kStages - 1)) {
            this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
            smem_write_stage_idx = 0;
          } else {
            ++smem_write_stage_idx;
          }

          if (smem_read_stage_idx == (Base::kStages - 1)) {

            this->warp_tile_iterator_A_.add_tile_offset(
                {0, -Base::kStages * Policy::kPartitionsK *
                        Base::kWarpGemmIterations});

            this->warp_tile_iterator_B_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK *
                     Base::kWarpGemmIterations,
                 0});
            smem_read_stage_idx = 0;
          } else {
            ++smem_read_stage_idx;
          }

          --gemm_k_iterations;
          iterator_A_real.clear_mask(gemm_k_iterations == 0);
          iterator_A_imag.clear_mask(gemm_k_iterations == 0);
          iterator_B_real.clear_mask(gemm_k_iterations == 0);
          iterator_B_imag.clear_mask(gemm_k_iterations == 0);
        }

        warp_mma_planar_complex(
          warp_mma, 
          accum, 
          warp_frag_real_A[warp_mma_k % 2], 
          warp_frag_imag_A[warp_mma_k % 2],
          warp_frag_real_B[warp_mma_k % 2], 
          warp_frag_imag_B[warp_mma_k % 2]);
      }

    }


    // Commit and drain all pending and predicated cp.async pnz from the GEMM mainloop
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
