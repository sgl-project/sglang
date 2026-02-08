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

#include "cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h"

#include "threadblock/b2b_mma_base.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape0_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA0_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA0_,
    /// Cache operation for operand A
    cutlass::arch::CacheOperation::Kind CacheOpA0,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB0_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB0_,
    /// Cache operation for operand B
    cutlass::arch::CacheOperation::Kind CacheOpB0,
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape1_,
    /// Iterates over the intermediate accumulator tile
    //  (concept::MmaTensorOpFragmentIterator) 
    typename FragmentIteratorA1_,
    /// Iterates over vectors of scale and bias vector in global memory
    //  (concept: VectorIterator)
    typename IteratorAccumulatorScaleBias_,
    /// WarpIterator to load Scale or Bias vector from threadblock fragment
    typename FragmentIteratorA1ScaleBias_,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB1_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB1_,
    /// Cache operation for operand B
    cutlass::arch::CacheOperation::Kind CacheOpB1,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Output operator for 1st Gemm(concept: epilogue::thread::LinearCombinationClamp, etc...) 
    typename OutputOp_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy0_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy1_,
    /// Number of stages,
    int Stages,
    /// Used for partial specialization
    typename Enable = bool>
class B2bMmaMultistage : 
  public B2bMmaBase<Shape0_, Shape1_, Policy0_, Policy1_, Stages> {
public:
  ///< Base class
  using Base = B2bMmaBase<Shape0_, Shape1_, Policy0_, Policy1_, Stages>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape0 = Shape0_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA0 = IteratorA0_;
  using IteratorA = IteratorA0;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB0 = IteratorB0_;
  using IteratorB = IteratorB0;
  ///< Policy describing tuning details
  using Policy0 = Policy0_;

  using SmemIteratorA0 = SmemIteratorA0_;
  using SmemIteratorB0 = SmemIteratorB0_;

  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape1 = Shape1_;
  ///< Iterates over intermediate accumulator tile
  using FragmentIteratorA1 = FragmentIteratorA1_;
  ///< Iterates over tiles of the scale and bias vectors in global memory
  using IteratorAccumulatorScaleBias = IteratorAccumulatorScaleBias_;
  ///< WarpIterator to load Scale or Bias vector from threadblock fragment
  using FragmentIteratorA1ScaleBias = FragmentIteratorA1ScaleBias_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB1 = IteratorB1_;
  ///< Policy describing tuning details
  using Policy1 = Policy1_;

  ///< Export Policy0 as the threadblock-level Mma's policy
  using Policy = Policy0;
  using Shape = Shape0;
  
  using SmemIteratorB1 = SmemIteratorB1_;

  ///< Data type of accumulator matrix
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;

  ///< Epilogue after 1st Gemm
  using OutputOp = OutputOp_;
  
  static const bool PerChannelScale = (OutputOp::kScale ==
      epilogue::thread::ScaleType::OnlyAlphaPerChannelScaling);
 
  static cutlass::arch::CacheOperation::Kind const kCacheOpA0 = CacheOpA0;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB0 = CacheOpB0;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB1 = CacheOpB1;

  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC0 = typename Policy0::Operator::FragmentC;

  /// Warp-level Mma
  using Operator0 = typename Policy0::Operator;
  
  /// Fragment of Scale and Bias loaded from global memory
  using FragmentA1ScaleBias = typename IteratorAccumulatorScaleBias::Fragment;

  /// Fragment of accumulator tile
  using FragmentC1 = typename Policy1::Operator::FragmentC;

  /// Warp-level Mma
  using Operator1 = typename Policy1::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = arch::Sm80;
  
  /// Complex transform on A operand
  static ComplexTransform const kTransformA0 = Operator0::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB0 = Operator0::kTransformB;
  
  /// Complex transform on B operand
  static ComplexTransform const kTransformB1 = Operator1::kTransformB;

  /// Complex transform exports needed by higher-level kernels
  static ComplexTransform const kTransformA = kTransformA0;
  static ComplexTransform const kTransformB = kTransformB0;

  /// Internal structure exposed for introspection.
  struct Detail {

    static_assert(Base::kWarpGemmIterations0 > 1,
                  "The pipelined structure requires at least two warp-level "
                  "GEMM operations.");
    static_assert(Base::kWarpGemmIterations1 > 1,
                  "The pipelined structure requires at least two warp-level "
                  "GEMM operations.");

    /// Number of cp.async instructions to load one stage of operand A
    static int const TBLoadIterationsA0 =
        IteratorA0::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const TBLoadIterationsB0 =
        IteratorB0::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const TBLoadIterationsB1 =
        IteratorB1::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = Stages;

    /// Number of cp.async instructions to load on group of operand A
    static int const kAccessesPerGroupA0 =
        (TBLoadIterationsA0 + Base::kWarpGemmIterations0 - 1) / Base::kWarpGemmIterations0;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB0 =
        (TBLoadIterationsB0 + Base::kWarpGemmIterations0 - 1) / Base::kWarpGemmIterations0;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB1 =
        (TBLoadIterationsB1 + Base::kWarpGemmIterations1 - 1) / Base::kWarpGemmIterations1;
  };

 private:

  using WarpLoadedFragmentA0 = typename Operator0::FragmentA;
  using WarpLoadedFragmentB0 = typename Operator0::FragmentB;
  /// Warp Fragment of operand A1 loaded from accmulator tile
  using WarpLoadedFragmentA1 = typename FragmentIteratorA1::Fragment;
  using WarpLoadedFragmentA1ScaleBias =
      typename FragmentIteratorA1ScaleBias::Fragment;
  using WarpLoadedFragmentB1 = typename Operator1::FragmentB;
  using WarpTransformedFragmentA0 = typename Operator0::TransformedFragmentA;
  using WarpTransformedFragmentB0 = typename Operator0::TransformedFragmentB;
  using WarpTransformedFragmentA1 = typename Operator1::TransformedFragmentA;
  using WarpTransformedFragmentB1 = typename Operator1::TransformedFragmentB;

 private:

  //
  // Data members
  //

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA0 smem_iterator_A0_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB0 smem_iterator_B0_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB1 smem_iterator_B1_;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  B2bMmaMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::B2bMmaSharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx,
      ///< GEMM0 N is used for accumulator extent
      int problem_size_0_n
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      smem_iterator_A0_(shared_storage.shared_storage0.operand_A_ref(), thread_idx),
      smem_iterator_B0_(shared_storage.shared_storage0.operand_B_ref(), thread_idx),
      smem_iterator_B1_(shared_storage.shared_storage1.operand_B_ref(), thread_idx)
  {
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount0::kM * Base::WarpCount0::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount0::kM * Base::WarpCount0::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount0::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount0::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A0_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations0 * warp_idx_k});
    this->warp_tile_iterator_B0_.add_tile_offset(
        {Base::kWarpGemmIterations0 * warp_idx_k, warp_idx_n});
    this->warp_tile_iterator_B1_.add_tile_offset(
        {Base::kWarpGemmIterations1 * warp_idx_k, warp_idx_n});
  }

  CUTLASS_DEVICE
  void copy_tiles_and_advance_0(IteratorA0 &iterator_A0, IteratorB0 &iterator_B0,
                              int group_start_A0 = 0, int group_start_B0 = 0) {
    iterator_A0.set_iteration_index(group_start_A0 *
                                   IteratorA0::kAccessesPerVector);
    this->smem_iterator_A0_.set_iteration_index(group_start_A0);

    // Load for operand A
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupA0; ++j) {
      if (group_start_A0 + j < Detail::TBLoadIterationsA0) {
        typename IteratorA0::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA0::AccessType *>(
                this->smem_iterator_A0_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorA0::Element>::value *
                              IteratorA0::ThreadMap::kElementsPerAccess /
                              IteratorA0::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA0::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_A0.get();

          cutlass::arch::cp_async<kSrcBytes, kCacheOpA0>(
              dst_ptr + v, gmem_ptr, iterator_A0.valid());

          ++iterator_A0;
        }

        ++this->smem_iterator_A0_;
      }
    }

    iterator_B0.set_iteration_index(group_start_B0 *
                                   IteratorB0::kAccessesPerVector);
    this->smem_iterator_B0_.set_iteration_index(group_start_B0);

    // Load for operand B
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB0; ++j) {
      if (group_start_B0 + j < Detail::TBLoadIterationsB0) {
        typename IteratorB0::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB0::AccessType *>(
                this->smem_iterator_B0_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorB0::Element>::value *
                              IteratorB0::ThreadMap::kElementsPerAccess /
                              IteratorB0::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB0::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_B0.get();

          cutlass::arch::cp_async<kSrcBytes, kCacheOpB0>(
              dst_ptr + v, gmem_ptr, iterator_B0.valid());

          ++iterator_B0;
        }
        ++this->smem_iterator_B0_;
      }
    }
  }

  CUTLASS_DEVICE
  void copy_tiles_and_advance_1(IteratorB1 &iterator_B1,
                              int group_start_B1 = 0) {
    iterator_B1.set_iteration_index(group_start_B1 *
                                   IteratorB1::kAccessesPerVector);
    this->smem_iterator_B1_.set_iteration_index(group_start_B1);

    // Load for operand B
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB1; ++j) {
      if (group_start_B1 + j < Detail::TBLoadIterationsB1) {
        typename IteratorB1::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB1::AccessType *>(
                this->smem_iterator_B1_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorB1::Element>::value *
                              IteratorB1::ThreadMap::kElementsPerAccess /
                              IteratorB1::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB1::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_B1.get();

          cutlass::arch::cp_async<kSrcBytes, kCacheOpB1>(
              dst_ptr + v, gmem_ptr, iterator_B1.valid());

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
      int gemm_k_iterations_0,
      ///< destination accumulator tile
      FragmentC1 &accum,
      ///< iterator over A0 operand in global memory
      IteratorA0 iterator_A0,
      ///< iterator over B0 operand in global memory
      IteratorB0 iterator_B0,
      ///< iterator over A1 operand scale vector in global memory
      IteratorAccumulatorScaleBias iterator_A1_scale,
      ///< iterator over A1 operand bias vector in global memory
      IteratorAccumulatorScaleBias iterator_A1_bias,
      ///< iterator over B1 operand in global memory
      IteratorB1 iterator_B1,
      ///< initial value of accumulator
      FragmentC0 const &src_accum,
      ///< epilogue operation after 1st Gemm
      OutputOp output_op_0)
    {
    //
    // Prologue
    //

    // Issue several complete stages
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1;
         ++stage, --gemm_k_iterations_0) {

      iterator_A0.clear_mask(gemm_k_iterations_0 == 0);
      iterator_B0.clear_mask(gemm_k_iterations_0 == 0);

      iterator_A0.set_iteration_index(0);
      this->smem_iterator_A0_.set_iteration_index(0);

      // Load for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::TBLoadIterationsA0; ++j) {
        typename IteratorA0::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA0::AccessType *>(
                this->smem_iterator_A0_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA0::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorA0::Element>::value *
              IteratorA0::ThreadMap::kElementsPerAccess /
              IteratorA0::kAccessesPerVector / 8;

          int src_bytes = (iterator_A0.valid() ? kSrcBytes : 0);

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA0>(
              dst_ptr + v, iterator_A0.get(), iterator_A0.valid());

          ++iterator_A0;
        }

        ++this->smem_iterator_A0_;
      }

      iterator_B0.set_iteration_index(0);
      this->smem_iterator_B0_.set_iteration_index(0);

      // Load for operand B
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::TBLoadIterationsB0; ++j) {
        typename IteratorB0::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB0::AccessType *>(
                this->smem_iterator_B0_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB0::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorB0::Element>::value *
              IteratorB0::ThreadMap::kElementsPerAccess /
              IteratorB0::kAccessesPerVector / 8;

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB0>(
              dst_ptr + v, iterator_B0.get(), iterator_B0.valid());

          ++iterator_B0;
        }

        ++this->smem_iterator_B0_;
      }

      // Move to the next stage
      iterator_A0.add_tile_offset({0, 1});
      iterator_B0.add_tile_offset({1, 0});

      this->smem_iterator_A0_.add_tile_offset({0, 1});
      this->smem_iterator_B0_.add_tile_offset({1, 0});

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();
    }

    // Perform accumulation in the 'd' output operand
    FragmentC0 accum0 = src_accum;

    // DEPBAR+SYNC
    cutlass::arch::cp_async_wait<Base::kStages - 2>();
    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpLoadedFragmentA0 warp_loaded_frag_A0[2];
    WarpLoadedFragmentB0 warp_loaded_frag_B0[2];
    WarpTransformedFragmentA0 warp_transformed_frag_A0[2];
    WarpTransformedFragmentB0 warp_transformed_frag_B0[2];

    Operator0 warp_mma0;

    this->warp_tile_iterator_A0_.set_kgroup_index(0);
    this->warp_tile_iterator_B0_.set_kgroup_index(0);

    this->warp_tile_iterator_A0_.load(warp_loaded_frag_A0[0]);
    this->warp_tile_iterator_B0_.load(warp_loaded_frag_B0[0]);

    ++this->warp_tile_iterator_A0_;
    ++this->warp_tile_iterator_B0_;

    iterator_A0.clear_mask(gemm_k_iterations_0 == 0);
    iterator_B0.clear_mask(gemm_k_iterations_0 == 0);

    int smem_write_stage_idx = Base::kStages - 1;
    int smem_read_stage_idx = 0;

    warp_mma0.transform(warp_transformed_frag_A0[0], warp_transformed_frag_B0[0],
                       warp_loaded_frag_A0[0], warp_loaded_frag_B0[0]);

    //
    // Mainloop
    //

    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations_0 > (-Base::kStages + 1);) {
      //
      // Loop over GEMM K dimension
      //

      // Computes a warp-level GEMM on data held in shared memory
      // Each "warp_mma_k" refers to a warp-level matrix multiply-accumulate
      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations0;
           ++warp_mma_k) {

        // Load warp-level tiles from shared memory, wrapping to k offset if
        // this is the last group as the case may be.

        this->warp_tile_iterator_A0_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations0);
        this->warp_tile_iterator_B0_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations0);
        
        this->warp_tile_iterator_A0_.load(warp_loaded_frag_A0[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B0_.load(warp_loaded_frag_B0[(warp_mma_k + 1) % 2]);

        ++this->warp_tile_iterator_A0_;
        ++this->warp_tile_iterator_B0_;

        if (warp_mma_k > 0)
          warp_mma0.transform(warp_transformed_frag_A0[warp_mma_k % 2],
                             warp_transformed_frag_B0[warp_mma_k % 2],
                             warp_loaded_frag_A0[warp_mma_k % 2],
                             warp_loaded_frag_B0[warp_mma_k % 2]);

        warp_mma0(
          accum0, 
          warp_transformed_frag_A0[warp_mma_k % 2],
          warp_transformed_frag_B0[warp_mma_k % 2], 
          accum0
        );

        // Issue global->shared copies for the this stage
        if (warp_mma_k < Base::kWarpGemmIterations0 - 1) {
          int group_start_iteration_A0, group_start_iteration_B0;

          group_start_iteration_A0 = warp_mma_k * Detail::kAccessesPerGroupA0;
          group_start_iteration_B0 = warp_mma_k * Detail::kAccessesPerGroupB0;

          copy_tiles_and_advance_0(iterator_A0, iterator_B0, group_start_iteration_A0, 
                               group_start_iteration_B0);
        }

        if (warp_mma_k + 2 == Base::kWarpGemmIterations0) {
          int group_start_iteration_A0, group_start_iteration_B0;
          group_start_iteration_A0 =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupA0;
          group_start_iteration_B0 =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupB0;

          copy_tiles_and_advance_0(iterator_A0, iterator_B0, group_start_iteration_A0, 
                               group_start_iteration_B0);

          // Inserts a memory fence between stages of cp.async instructions.
          cutlass::arch::cp_async_fence();

          // Waits until kStages-2 stages have committed.
          arch::cp_async_wait<Base::kStages - 2>();
          __syncthreads();

          // Move to the next stage
          iterator_A0.add_tile_offset({0, 1});
          iterator_B0.add_tile_offset({1, 0});

          this->smem_iterator_A0_.add_tile_offset({0, 1});
          this->smem_iterator_B0_.add_tile_offset({1, 0});

          // Add negative offsets to return iterators to the 'start' of the
          // circular buffer in shared memory
          if (smem_write_stage_idx == (Base::kStages - 1)) {
            this->smem_iterator_A0_.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B0_.add_tile_offset({-Base::kStages, 0});
            smem_write_stage_idx = 0;
          } else {
            ++smem_write_stage_idx;
          }

          if (smem_read_stage_idx == (Base::kStages - 1)) {
            this->warp_tile_iterator_A0_.add_tile_offset(
                {0, -Base::kStages * Policy0::kPartitionsK *
                        Base::kWarpGemmIterations0});
            this->warp_tile_iterator_B0_.add_tile_offset(
                {-Base::kStages * Policy0::kPartitionsK *
                     Base::kWarpGemmIterations0,
                 0});
            smem_read_stage_idx = 0;
          } else {
            ++smem_read_stage_idx;
          }

          --gemm_k_iterations_0;
          iterator_A0.clear_mask(gemm_k_iterations_0 == 0);
          iterator_B0.clear_mask(gemm_k_iterations_0 == 0);
        }

        // Do any conversions feeding the first stage at the end of the loop so
        // we can start right away on mma instructions
        if (warp_mma_k + 1 == Base::kWarpGemmIterations0)
          warp_mma0.transform(warp_transformed_frag_A0[(warp_mma_k + 1) % 2],
                             warp_transformed_frag_B0[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_A0[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_B0[(warp_mma_k + 1) % 2]);
      }

    }

    // Commit and drain all pending and predicated cp.async pnz from the GEMM mainloop
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    // 2nd Gemm

    /// Iterator to load a warp-scoped tile of A1 operand from intermediate accumulator tile
    FragmentIteratorA1 warp_tile_iterator_A1_(accum0);
    FragmentA1ScaleBias tb_frag_A1_scale;
    FragmentA1ScaleBias tb_frag_A1_bias;
    FragmentIteratorA1ScaleBias warp_tile_iterator_A1_scale_(tb_frag_A1_scale);
    FragmentIteratorA1ScaleBias warp_tile_iterator_A1_bias_(tb_frag_A1_bias);

    if(PerChannelScale) {
        tb_frag_A1_scale.clear();
        iterator_A1_scale.load(tb_frag_A1_scale);
        ++iterator_A1_scale;
    }
    tb_frag_A1_bias.clear();
    iterator_A1_bias.load(tb_frag_A1_bias);
    ++iterator_A1_bias;
 
    //
    // Prologue
    //
    int gemm_k_iterations_1 = (FragmentIteratorA1::Policy::kIterations + Base::kWarpGemmIterations1 - 1) / Base::kWarpGemmIterations1;

    // Issue several complete stages
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1;
         ++stage, --gemm_k_iterations_1) {

      iterator_B1.clear_mask(gemm_k_iterations_1 == 0);

      iterator_B1.set_iteration_index(0);
      this->smem_iterator_B1_.set_iteration_index(0);

      // Load for operand B
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::TBLoadIterationsB1; ++j) {
        typename IteratorB1::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB1::AccessType *>(
                this->smem_iterator_B1_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB1::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorB1::Element>::value *
              IteratorB1::ThreadMap::kElementsPerAccess /
              IteratorB1::kAccessesPerVector / 8;

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB1>(
              dst_ptr + v, iterator_B1.get(), iterator_B1.valid());

          ++iterator_B1;
        }

        ++this->smem_iterator_B1_;
      }

      // Move to the next stage
      iterator_B1.add_tile_offset({1, 0});

      this->smem_iterator_B1_.add_tile_offset({1, 0});

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();
    }

    // DEPBAR+SYNC
    cutlass::arch::cp_async_wait<Base::kStages - 2>();
    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpLoadedFragmentA1 warp_loaded_frag_A1[2];
    WarpLoadedFragmentA1ScaleBias warp_loaded_frag_A1_scale[2];
    WarpLoadedFragmentA1ScaleBias warp_loaded_frag_A1_bias[2];
    WarpLoadedFragmentB1 warp_loaded_frag_B1[2];
    WarpTransformedFragmentA1 warp_transformed_frag_A1[2];
    WarpTransformedFragmentB1 warp_transformed_frag_B1[2];

    Operator1 warp_mma1;

    if(PerChannelScale) {
        warp_tile_iterator_A1_scale_.load(warp_loaded_frag_A1_scale[0]);
        ++warp_tile_iterator_A1_scale_;
    }
    warp_tile_iterator_A1_bias_.load(warp_loaded_frag_A1_bias[0]);
    ++warp_tile_iterator_A1_bias_;

    warp_tile_iterator_A1_.load(warp_loaded_frag_A1[0], 
        warp_loaded_frag_A1_scale[0],
        warp_loaded_frag_A1_bias[0], 
        output_op_0);
    ++warp_tile_iterator_A1_;

    this->warp_tile_iterator_B1_.set_kgroup_index(0);
    this->warp_tile_iterator_B1_.load(warp_loaded_frag_B1[0]);
    ++this->warp_tile_iterator_B1_;

    iterator_B1.clear_mask(gemm_k_iterations_1 == 0);

    smem_write_stage_idx = Base::kStages - 1;
    smem_read_stage_idx = 0;

    warp_mma1.transform(warp_transformed_frag_A1[0], warp_transformed_frag_B1[0],
                       warp_loaded_frag_A1[0], warp_loaded_frag_B1[0]);

    //
    // Mainloop
    //

    gemm_k_iterations_1 = (FragmentIteratorA1::Policy::kIterations + Base::kWarpGemmIterations1 - 1) / Base::kWarpGemmIterations1 - (Base::kStages - 1);
    CUTLASS_PRAGMA_UNROLL
    for (; gemm_k_iterations_1 > (-Base::kStages + 1); gemm_k_iterations_1--) {
      //
      // Loop over GEMM K dimension
      //

      // Computes a warp-level GEMM on data held in shared memory
      // Each "warp_mma_k" refers to a warp-level matrix multiply-accumulate
      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations1;
           ++warp_mma_k) {

        // Load threadblock-level scale/bias vector from global memory
        if (warp_mma_k + 1 == Base::kWarpGemmIterations1) {
          if(PerChannelScale) {
              tb_frag_A1_scale.clear();
              iterator_A1_scale.load(tb_frag_A1_scale);
              ++iterator_A1_scale;
          }
          tb_frag_A1_bias.clear();
          iterator_A1_bias.load(tb_frag_A1_bias);
          ++iterator_A1_bias;
        }

        // Load warp-level scale bias fragment from threadblock scale/bias vector
        if(PerChannelScale) {
          warp_tile_iterator_A1_scale_.load(warp_loaded_frag_A1_scale[(warp_mma_k + 1) % 2]);
          ++warp_tile_iterator_A1_scale_;
        }
        warp_tile_iterator_A1_bias_.load(warp_loaded_frag_A1_bias[(warp_mma_k + 1) % 2]);
        ++warp_tile_iterator_A1_bias_;

        // Load warp-level tile from accumulator fragment
        warp_tile_iterator_A1_.load(warp_loaded_frag_A1[(warp_mma_k + 1) % 2],
            warp_loaded_frag_A1_scale[(warp_mma_k + 1) % 2], 
            warp_loaded_frag_A1_bias[(warp_mma_k + 1) % 2], 
            output_op_0);
        ++warp_tile_iterator_A1_;

        // Load warp-level tiles from shared memory, wrapping to k offset if
        // this is the last group as the case may be.
        this->warp_tile_iterator_B1_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations1);
        this->warp_tile_iterator_B1_.load(warp_loaded_frag_B1[(warp_mma_k + 1) % 2]);
        ++this->warp_tile_iterator_B1_;

        if (warp_mma_k > 0)
          warp_mma1.transform(warp_transformed_frag_A1[warp_mma_k % 2],
                             warp_transformed_frag_B1[warp_mma_k % 2],
                             warp_loaded_frag_A1[warp_mma_k % 2],
                             warp_loaded_frag_B1[warp_mma_k % 2]);


        warp_mma1(
          accum, 
          warp_transformed_frag_A1[warp_mma_k % 2],
          warp_transformed_frag_B1[warp_mma_k % 2], 
          accum
        );

        // Issue global->shared copies for the this stage
        if (warp_mma_k < Base::kWarpGemmIterations1 - 1) {
          int group_start_iteration_B1;

          group_start_iteration_B1 = warp_mma_k * Detail::kAccessesPerGroupB1;

          copy_tiles_and_advance_1(iterator_B1, group_start_iteration_B1);
        }

        if (warp_mma_k + 2 == Base::kWarpGemmIterations1) {
          int group_start_iteration_B1;
          group_start_iteration_B1 =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupB1;

          copy_tiles_and_advance_1(iterator_B1, group_start_iteration_B1);

          // Inserts a memory fence between stages of cp.async instructions.
          cutlass::arch::cp_async_fence();

          // Waits until kStages-2 stages have committed.
          arch::cp_async_wait<Base::kStages - 2>();
          __syncthreads();

          // Move to the next stage
          iterator_B1.add_tile_offset({1, 0});

          this->smem_iterator_B1_.add_tile_offset({1, 0});

          // Add negative offsets to return iterators to the 'start' of the
          // circular buffer in shared memory
          if (smem_write_stage_idx == (Base::kStages - 1)) {
            this->smem_iterator_B1_.add_tile_offset({-Base::kStages, 0});
            smem_write_stage_idx = 0;
          } else {
            ++smem_write_stage_idx;
          }

          if (smem_read_stage_idx == (Base::kStages - 1)) {
            this->warp_tile_iterator_B1_.add_tile_offset(
                {-Base::kStages * Policy1::kPartitionsK *
                     Base::kWarpGemmIterations1,
                 0});
            smem_read_stage_idx = 0;
          } else {
            ++smem_read_stage_idx;
          }

          iterator_B1.clear_mask(gemm_k_iterations_1 == 1);
        }

        // Do any conversions feeding the first stage at the end of the loop so
        // we can start right away on mma instructions
        if (warp_mma_k + 1 == Base::kWarpGemmIterations1)
          warp_mma1.transform(warp_transformed_frag_A1[(warp_mma_k + 1) % 2],
                             warp_transformed_frag_B1[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_A1[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_B1[(warp_mma_k + 1) % 2]);
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
