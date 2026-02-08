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
#include "cutlass/array.h"
#include "cutlass/aligned_buffer.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

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
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
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
class MmaPlanarComplexPipelined : 
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

  using ArchTag = typename Policy::Operator::ArchTag;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

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

 private:

  using FragmentA = typename IteratorA::Fragment;
  using FragmentB = typename IteratorB::Fragment;
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
  MmaPlanarComplexPipelined(
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

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    FragmentA tb_frag_A_real;
    FragmentA tb_frag_A_imag;

    FragmentB tb_frag_B_real;
    FragmentB tb_frag_B_imag;

    tb_frag_A_real.clear();
    tb_frag_A_imag.clear();

    tb_frag_B_real.clear();
    tb_frag_B_imag.clear();

    // The last kblock is loaded in the prolog
    iterator_A_real.load(tb_frag_A_real);
    iterator_A_imag.load(tb_frag_A_imag);

    iterator_B_real.load(tb_frag_B_real);
    iterator_B_imag.load(tb_frag_B_imag);

    ++iterator_A_real;
    ++iterator_A_imag;

    ++iterator_B_real;
    ++iterator_B_imag;

    this->smem_iterator_A_.store(tb_frag_A_real);
    this->smem_iterator_A_.store_with_pointer_offset(tb_frag_A_imag, Base::SharedStorage::kImaginaryStrideA);

    this->smem_iterator_B_.store(tb_frag_B_real);
    this->smem_iterator_B_.store_with_pointer_offset(tb_frag_B_imag, Base::SharedStorage::kImaginaryStrideB);

    ++this->smem_iterator_A_;
    ++this->smem_iterator_B_;

    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math instructions
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

    Operator warp_mma;

    int smem_write_stage_idx = 1;

    // Avoid reading out of bounds
    iterator_A_real.clear_mask(gemm_k_iterations <= 1);
    iterator_A_imag.clear_mask(gemm_k_iterations <= 1);
    
    iterator_B_real.clear_mask(gemm_k_iterations <= 1);
    iterator_B_imag.clear_mask(gemm_k_iterations <= 1);

    // Issue loads during the first warp-level matrix multiply-add *AFTER* issuing 
    // shared memory loads (which have the tightest latency requirement).

    //
    // Mainloop
    //

    // Note: The main loop does not support Base::kWarpGemmIterations == 2.
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      //
      // Loop over GEMM K dimension
      //

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {

        // Load warp-level tiles from shared memory, wrapping to k offset if this is the last group
        // as the case may be.

        if (warp_mma_k == Base::kWarpGemmIterations - 1) {

          // Write fragments to shared memory
          this->smem_iterator_A_.store(tb_frag_A_real);
          this->smem_iterator_A_.store_with_pointer_offset(tb_frag_A_imag, Base::SharedStorage::kImaginaryStrideA);

          this->smem_iterator_B_.store(tb_frag_B_real);
          this->smem_iterator_B_.store_with_pointer_offset(tb_frag_B_imag, Base::SharedStorage::kImaginaryStrideB);

          __syncthreads();
          
          ++this->smem_iterator_B_;
          ++this->smem_iterator_A_;

          // Add negative offsets to return iterators to the 'start' of the circular buffer in shared memory
          if (smem_write_stage_idx == 1) {
            this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
          }
          else {
            this->warp_tile_iterator_A_.add_tile_offset(
                {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
            this->warp_tile_iterator_B_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations,
                 0});
          }

          smem_write_stage_idx ^= 1;
        }

        this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        
        this->warp_tile_iterator_A_.load(warp_frag_real_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_A_.load_with_pointer_offset(warp_frag_imag_A[(warp_mma_k + 1) % 2], Base::SharedStorage::kImaginaryStrideA);
        
        this->warp_tile_iterator_B_.load(warp_frag_real_B[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load_with_pointer_offset(warp_frag_imag_B[(warp_mma_k + 1) % 2], Base::SharedStorage::kImaginaryStrideB);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        if (warp_mma_k == 0) {

          iterator_A_real.load(tb_frag_A_real);
          iterator_A_imag.load(tb_frag_A_imag);

          iterator_B_real.load(tb_frag_B_real);
          iterator_B_imag.load(tb_frag_B_imag);

          ++iterator_A_real;
          ++iterator_A_imag;
          ++iterator_B_real;
          ++iterator_B_imag;

          // Avoid reading out of bounds if this was the last loop iteration
          iterator_A_real.clear_mask(gemm_k_iterations <= 2);
          iterator_A_imag.clear_mask(gemm_k_iterations <= 2);
          iterator_B_real.clear_mask(gemm_k_iterations <= 2);
          iterator_B_imag.clear_mask(gemm_k_iterations <= 2);
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

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
