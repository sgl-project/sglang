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
    \brief Template for a multistage threadblock-scoped fused activation's 
   scale+bias+relu and Implicit GEMM Convolution kernel.

   The original implicit gemm will store out-of-bound data as zeroes in the
   shared memory because zeros into the tensor core, zeroes out of the tensor
   cores.  The result is remained the same.   When fusing scale+bias+relu
   into the mainloop, it is no longer true because

     0 x scale + bias = bias

   which is no longer always 0.  So, instead of storing zeroes, this fused
   kernel stores the out-of-bound data as a special NaN (0x7eff), when applying
   scale+bias+relu, the code is like

     if (data == 0x7eff)
       data = 0;
     else
       data = scale+bias+relu(data, scale, bias);

  See include/cutlass/conv/warp/scale_bias_relu_transformation.h for the 
  elementwise computation.  See include/cutlass/arch/memory_sm80.h for nan fill.
*/

#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/cache_operation.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/gemm/warp/scale_bias_tile_iterator.h"
#include "cutlass/conv/warp/scale_bias_relu_transform.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Element type of scale and bias vectors 
    typename ElementScaleBias_,
    /// Layout of scale and bias vectors
    typename LayoutScaleBias_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// WarpIterator to load Scale or Bias vector from the shared memory
    typename WarpIteratorScaleBias_,
    /// Number of stages,
    int Stages,
    /// Used for partial specialization
    typename Enable = bool>
class MmaFpropFusionBase {
 public:
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  ///< Element type of scale and bias vectors 
  using ElementScaleBias = ElementScaleBias_;

  /// Layout of scale and bias vectors
  using LayoutScaleBias = LayoutScaleBias_;

  ///< Policy describing tuning details
  using Policy = Policy_;

  ///< WarpIterator to load Scale or Bias vector from the shared memory
  using WarpIteratorScaleBias = WarpIteratorScaleBias_;

  //
  // Dependent types
  //

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Shape describing the overall GEMM computed from shared memory
  /// by each warp.
  using WarpGemm = typename Policy::Operator::Shape;

  /// Shape describing the number of warps filling the CTA
  using WarpCount = cutlass::gemm::GemmShape<Shape::kM / WarpGemm::kM,
                                             Shape::kN / WarpGemm::kN,
                                             Shape::kK / WarpGemm::kK>;

  /// Number of warp-level GEMM oeprations
  static int const kWarpGemmIterations =
      (WarpGemm::kK / Operator::Policy::MmaShape::kK);

  /// Number of stages
  static int const kStages = Stages;

  /// Tensor reference to the A operand
  using TensorRefA = TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;

  /// Tensor reference to the scale and bias vectors
  using TensorRefScaleBias = TensorRef<ElementScaleBias, LayoutScaleBias>;

  /// Tensor reference to the B operand
  using TensorRefB = TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;

  static_assert(kWarpGemmIterations > 1,
                "The pipelined structure requires at least two warp-level "
                "GEMM operations.");

  static_assert((kWarpGemmIterations % 2) == 0,
                "Inner loop iteration must be an even number.");

  //
  // Nested structs
  //

  /// Shared storage object needed by threadblock-scoped GEMM
  class SharedStorage {
   public:
    //
    // Type definitions
    //

    /// Shape of the A matrix operand in shared memory
    using ShapeA = MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow,
                               Shape::kK * kStages +
                                   Policy::SmemPaddingA::kColumn>;

    /// Shape of the A scale and bias vectors in shared memory
    using ShapeScaleBias =
        MatrixShape<1 + Policy::SmemPaddingA::kRow,
                    2 * Shape::kK * kStages + Policy::SmemPaddingA::kColumn>;

    /// Shape of the B matrix operand in shared memory
    using ShapeB =
        MatrixShape<Shape::kK * kStages + Policy::SmemPaddingB::kRow,
                    Shape::kN + Policy::SmemPaddingB::kColumn>;

   public:
    //
    // Data members
    //

    /// Buffer for A operand
    AlignedBuffer<typename Operator::ElementA, ShapeA::kCount> operand_A;

    /// Buffer for B operand
    AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_B;

    /// Buffer for A operand Scale and Bias
    AlignedBuffer<ElementScaleBias, ShapeScaleBias::kCount> operand_A_scale_bias;

   public:

    //
    // Methods
    //

    /// Returns a layout object for the A matrix
    CUTLASS_DEVICE
    static typename Operator::LayoutA LayoutA() {
      return Operator::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});
    }

    /// Returns a layout object for the B matrix
    CUTLASS_HOST_DEVICE
    static typename Operator::LayoutB LayoutB() {
      return Operator::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});
    }

    /// Returns a layout object for the A scale and bias vectors
    CUTLASS_DEVICE
    static LayoutScaleBias LayoutScaleBias() {
      return LayoutScaleBias::packed(
          {ShapeScaleBias::kRow, ShapeScaleBias::kColumn});
    }

    /// Returns a TensorRef to the A operand
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A_ref() {
      return TensorRefA{operand_A.data(), LayoutA()};
    }

    /// Returns a TensorRef to the B operand
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B_ref() {
      return TensorRefB{operand_B.data(), LayoutB()};
    }

    /// Returns a TensorRef to the A operand Scale vector
    CUTLASS_HOST_DEVICE
    TensorRefScaleBias operand_A_scale_bias_ref() {
      return TensorRefScaleBias{operand_A_scale_bias.data(), LayoutScaleBias()};
    }
  };

 protected:

  //
  // Data members
  //

  /// Iterator to load a warp-scoped tile of A operand from shared memory
  typename Operator::IteratorA warp_tile_iterator_A_;

  /// Iterator to load a warp-scoped tile of A operand scale and bias vector
  /// from shared memory
  WarpIteratorScaleBias warp_tile_iterator_A_scale_bias_;

  /// Iterator to load a warp-scoped tile of B operand from shared memory
  typename Operator::IteratorB warp_tile_iterator_B_;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  MmaFpropFusionBase(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx)
      : warp_tile_iterator_A_(shared_storage.operand_A_ref(), lane_idx),
        warp_tile_iterator_A_scale_bias_(
            shared_storage.operand_A_scale_bias_ref(), lane_idx),
        warp_tile_iterator_B_(shared_storage.operand_B_ref(), lane_idx) {}
};

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
    /// Iterates over vectors of scale and bias vector in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorScaleBias_,
    /// Iterates over vectors of scale and bias vector in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorScaleBias_,
    /// Cache operation for scale/bias operand 
    cutlass::arch::CacheOperation::Kind CacheOpScaleBias,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// WarpIterator to load Scale or Bias vector from the shared memory
    typename WarpIteratorScaleBias_,
    /// Number of stages,
    int Stages,
    /// Used for partial specialization
    typename Enable = bool>
class ImplicitGemmFpropFusionMultistage
    : public MmaFpropFusionBase<Shape_, typename IteratorScaleBias_::Element,
                       typename IteratorScaleBias_::Layout, Policy_,
                       WarpIteratorScaleBias_, Stages> {
 public:
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Iterates over tiles of the scale and bias vectors in global memory
  using IteratorScaleBias = IteratorScaleBias_;
  ///< WarpIterator to load Scale or Bias vector from the shared memory
  using WarpIteratorScaleBias = WarpIteratorScaleBias_;
  ///< Policy describing tuning details
  using Policy = Policy_;
  ///< Base class
  using Base = MmaFpropFusionBase<Shape_, typename IteratorScaleBias::Element,
                         typename IteratorScaleBias::Layout, Policy,
                         WarpIteratorScaleBias, Stages>;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;
  using SmemIteratorScaleBias = SmemIteratorScaleBias_;

  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;
  static cutlass::arch::CacheOperation::Kind const kCacheOpScaleBias =
      CacheOpScaleBias;

  //
  // Dependent types
  //

  /// Fragment of accumulator tile

  using ElementC = typename Policy::Operator::ElementC;
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;
  
  /// Internal structure exposed for introspection.
  struct Detail {

    static_assert(Base::kWarpGemmIterations > 1,
                  "The pipelined structure requires at least two warp-level "
                  "GEMM operations.");

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

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

  using WarpLoadedFragmentA = typename Operator::FragmentA;
  using WarpLoadedFragmentB = typename Operator::FragmentB;
  using WarpLoadedFragmentScaleBias =
      typename WarpIteratorScaleBias::Fragment;

  using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
  using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;

 private:

  //
  // Data members
  //

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of A operand scale vector to shared memory
  SmemIteratorScaleBias smem_iterator_A_scale_bias_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;
  
public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  ImplicitGemmFpropFusionMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx)
      : Base(shared_storage, thread_idx, warp_idx, lane_idx),
        smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
        smem_iterator_A_scale_bias_(shared_storage.operand_A_scale_bias_ref(),
                                    thread_idx),
        smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx) {
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
    this->warp_tile_iterator_A_scale_bias_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }

  CUTLASS_DEVICE
  void copy_tiles_and_advance(IteratorA &iterator_A,
                              IteratorScaleBias &iterator_A_scale_bias,
                              IteratorB &iterator_B, int group_start_A = 0,
                              int group_start_B = 0) {
    iterator_A.set_iteration_index(group_start_A);
    this->smem_iterator_A_.set_iteration_index(group_start_A);
      
    // Async Copy for operand A
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {

      if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                this->smem_iterator_A_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value *
                              IteratorA::ThreadMap::kElementsPerAccess / 8;

        // Uses nan fill for out of bound data
        cutlass::arch::cp_async_nan<kSrcBytes, kCacheOpA>(
            dst_ptr, iterator_A.get(), iterator_A.valid());

        ++iterator_A;

        ++this->smem_iterator_A_;
      }
    }

    // Async Copy for operand A scale and bias vector.  Scale and bias vectors
    // are small.  One iteration is enough.
    if (group_start_A == 0) {
      typename IteratorScaleBias::AccessType *dst_ptr =
          reinterpret_cast<typename IteratorScaleBias::AccessType *>(
              this->smem_iterator_A_scale_bias_.get());

      int const kSrcBytes =
          sizeof_bits<typename IteratorScaleBias::Element>::value *
          IteratorScaleBias::kElementsPerAccess / 8;

      cutlass::arch::cp_async<kSrcBytes, kCacheOpScaleBias>(
          dst_ptr, iterator_A_scale_bias.get(), iterator_A_scale_bias.valid());
    }

    iterator_B.set_iteration_index(group_start_B);

    this->smem_iterator_B_.set_iteration_index(group_start_B);
    
    // Async Copy for operand B
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                this->smem_iterator_B_.get());
        
        int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value *
                              IteratorB::ThreadMap::kElementsPerAccess / 8;

        cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
                dst_ptr, iterator_B.get(), iterator_B.valid());

        ++iterator_B;
        ++this->smem_iterator_B_;
      }
    }
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B,
      ///< iterator over scale and bias vectors in global memory
      IteratorScaleBias iterator_A_scale_bias,
      ///< initial value of accumulator
      FragmentC const &src_accum,
      ///< number of iterations per channel
      int gemm_k_iterations_per_channel = 0,  
      ///< Imaginary strides used for planar-complex only - ignored here
      int64_t imag_stride_A = 0,
      int64_t imag_stride_B = 0) {

    //
    // Prologue
    //

    // Issue several complete stages
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1;
         ++stage, --gemm_k_iterations) {

      iterator_A.set_iteration_index(0);
      this->smem_iterator_A_.set_iteration_index(0);

      // Async Copy for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType *dst_ptr =
          reinterpret_cast<typename IteratorA::AccessType *>(
            this->smem_iterator_A_.get());

        int const kSrcBytes =
            sizeof_bits<typename IteratorA::Element>::value *
            IteratorA::ThreadMap::kElementsPerAccess / 8;
        
        // Uses Nan fill for out of bound data
        cutlass::arch::cp_async_nan<kSrcBytes, kCacheOpA>(
            dst_ptr, iterator_A.get(), iterator_A.valid());

        ++iterator_A;
        ++this->smem_iterator_A_;
      }

      // Async Copy for operand A scale and bias vectors.  Scale and bias
      // vectors are small.  One iteration is enough.
      {
        typename IteratorScaleBias::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorScaleBias::AccessType *>(
                this->smem_iterator_A_scale_bias_.get());

        int const kSrcBytes =
            sizeof_bits<typename IteratorScaleBias::Element>::value *
            IteratorScaleBias::kElementsPerAccess / 8;

        cutlass::arch::cp_async<kSrcBytes, kCacheOpScaleBias>(
            dst_ptr, iterator_A_scale_bias.get(), iterator_A_scale_bias.valid());
      }

      iterator_B.set_iteration_index(0);
      this->smem_iterator_B_.set_iteration_index(0);

      // Async Copy for operand B
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType *dst_ptr =
          reinterpret_cast<typename IteratorB::AccessType *>(
              this->smem_iterator_B_.get());

        int const kSrcBytes =
            sizeof_bits<typename IteratorB::Element>::value *
            IteratorB::ThreadMap::kElementsPerAccess / 8;

        cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
            dst_ptr, iterator_B.get(), iterator_B.valid());

        ++iterator_B;
        ++this->smem_iterator_B_;
      }

      // Move to the next stage
      iterator_A.advance();
      iterator_A_scale_bias.advance();
      iterator_B.advance();

      this->smem_iterator_A_.add_tile_offset({0, 1});
      this->smem_iterator_A_scale_bias_.add_tile_offset({0, 1});
      this->smem_iterator_B_.add_tile_offset({1, 0});

      // Inserts a fence to group cp.async instructions into stages.
      cutlass::arch::cp_async_fence();
    }

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    // Waits until kStages-2 stages have committed. 
    cutlass::arch::cp_async_wait<Base::kStages - 2>();
    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpLoadedFragmentA warp_loaded_frag_A[2];
    WarpLoadedFragmentB warp_loaded_frag_B[2];
    WarpLoadedFragmentScaleBias warp_loaded_frag_A_scale_bias[2];
    WarpTransformedFragmentA warp_transformed_frag_A[2];
    WarpTransformedFragmentB warp_transformed_frag_B[2];

    Operator warp_mma;
    cutlass::conv::warp::FpropScaleBiasReluTransform<WarpTransformedFragmentA,
                                            WarpLoadedFragmentScaleBias>
        elementwise_transform;

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_scale_bias_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.set_kgroup_index(0);

    this->warp_tile_iterator_A_.load(warp_loaded_frag_A[0]);
    this->warp_tile_iterator_A_scale_bias_.load(
        warp_loaded_frag_A_scale_bias[0]);
    this->warp_tile_iterator_B_.load(warp_loaded_frag_B[0]);

    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_A_scale_bias_;
    ++this->warp_tile_iterator_B_;

    // Start issuing the first group of the next stage outside of the mainloop
    copy_tiles_and_advance(iterator_A, iterator_A_scale_bias, iterator_B);

    int smem_write_stage_idx = Base::kStages - 1;
    int smem_read_stage_idx = 0;

    warp_mma.transform(warp_transformed_frag_A[0], warp_transformed_frag_B[0],
                       warp_loaded_frag_A[0], warp_loaded_frag_B[0]);

    elementwise_transform(warp_transformed_frag_A[0],
                         warp_loaded_frag_A_scale_bias[0]);

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
        this->warp_tile_iterator_A_scale_bias_.set_kgroup_index(
            (warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);

        this->warp_tile_iterator_A_.load(warp_loaded_frag_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_A_scale_bias_.load(
            warp_loaded_frag_A_scale_bias[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load(warp_loaded_frag_B[(warp_mma_k + 1) % 2]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_A_scale_bias_;
        ++this->warp_tile_iterator_B_;

        if (warp_mma_k > 0) {
          warp_mma.transform(warp_transformed_frag_A[warp_mma_k % 2],
                             warp_transformed_frag_B[warp_mma_k % 2],
                             warp_loaded_frag_A[warp_mma_k % 2],
                             warp_loaded_frag_B[warp_mma_k % 2]);

          elementwise_transform(warp_transformed_frag_A[warp_mma_k % 2],
                               warp_loaded_frag_A_scale_bias[warp_mma_k % 2]);
        }

        warp_mma(
                 accum, 
                 warp_transformed_frag_A[warp_mma_k % 2],
                 warp_transformed_frag_B[warp_mma_k % 2],
                 accum
                );

        // Issue global->shared copies for the next stage
        int group_start_iteration_A, group_start_iteration_B;

        if (warp_mma_k + 1 == Base::kWarpGemmIterations) {
          group_start_iteration_A = 0;
          group_start_iteration_B = 0;
        } else {
          group_start_iteration_A =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
          group_start_iteration_B =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupB;
        }

        copy_tiles_and_advance(iterator_A, iterator_A_scale_bias, iterator_B,
                               group_start_iteration_A,
                               group_start_iteration_B);


        if (warp_mma_k + 1 == Base::kWarpGemmIterations) {
          warp_mma.transform(warp_transformed_frag_A[(warp_mma_k + 1) % 2],
                             warp_transformed_frag_B[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_A[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_B[(warp_mma_k + 1) % 2]);

          elementwise_transform(
              warp_transformed_frag_A[(warp_mma_k + 1) % 2],
              warp_loaded_frag_A_scale_bias[(warp_mma_k + 1) % 2]);
        }

        if (warp_mma_k + 2 == Base::kWarpGemmIterations) {
          // Inserts a fence to group cp.async instructions into stages.
          cutlass::arch::cp_async_fence();

          // Waits until kStages-2 stages of cp.async have committed
          arch::cp_async_wait<Base::kStages - 2>();
          __syncthreads();

          // Move to the next stage
          iterator_A.advance();
          iterator_A_scale_bias.advance();
          iterator_B.advance();

          this->smem_iterator_A_.add_tile_offset({0, 1});
          this->smem_iterator_A_scale_bias_.add_tile_offset({0, 1});
          this->smem_iterator_B_.add_tile_offset({1, 0});

          // Add negative offsets to return iterators to the 'start' of the
          // circular buffer in shared memory
          if (smem_write_stage_idx == (Base::kStages - 1)) {
            this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_A_scale_bias_.add_tile_offset(
                {0, -Base::kStages});
            this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
            smem_write_stage_idx = 0;
          } else {
            ++smem_write_stage_idx;
          }

          if (smem_read_stage_idx == (Base::kStages - 1)) {
            this->warp_tile_iterator_A_.add_tile_offset(
                {0, -Base::kStages * Policy::kPartitionsK *
                        Base::kWarpGemmIterations});
            this->warp_tile_iterator_A_scale_bias_.add_tile_offset(
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
        }
      }

    }

    // Insert fence and wait for all outstanding cp.async operations to commit.
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
