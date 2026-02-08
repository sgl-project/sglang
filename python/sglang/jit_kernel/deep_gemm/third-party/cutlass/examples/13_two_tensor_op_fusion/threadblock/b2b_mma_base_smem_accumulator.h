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
#include "threadblock/b2b_mma_base.h"
////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape0_,
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape1_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy0_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy1_,
    /// Shared Memory Accumulator Iterator
    typename SmemAccumulatorIterator0_,
    /// Number of stages,
    int Stages,
    /// Used for partial specialization
    typename Enable = bool>
class B2bMmaBaseSmemAccumulator :
  public B2bMmaBase<Shape0_, Shape1_, Policy0_, Policy1_, Stages> {

 public:
  ///< Base class
  using Base = B2bMmaBase<Shape0_, Shape1_, Policy0_, Policy1_, Stages>;

  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape0 = Shape0_;
  using Shape1 = Shape1_;

  ///< Policy describing tuning details
  using Policy0 = Policy0_;
  using Policy1 = Policy1_;


  using SmemAccumulatorIterator0 = SmemAccumulatorIterator0_;

  //
  // Nested structs
  //
  /// Shared storage object needed by accumulator
  template<
    typename Shape_,
    typename Element_,
    typename Layout_,
    typename Padding_
  >
  class AccumulatorSharedStorage {
   public:
    //
    // Type definitions
    //
    using Shape = Shape_;
    using Element = Element_;
    using Layout = Layout_;
    using Padding = Padding_;

    /// Tensor reference to the accumulator
    using TensorRefAccum = TensorRef<Element, Layout>;

    /// Shape of the accumulator matrix in shared memory
    using ShapeAccum = MatrixShape<Shape::kM + Padding::kRow, 
                                    Shape::kN + Padding::kColumn>;

   public:
    //
    // Data members
    //

    /// Buffer for accumulator
    AlignedBuffer<Element, ShapeAccum::kCount> accum;

   public:

    //
    // Methods
    //

    /// Returns a layout object for the Accum matrix
    CUTLASS_DEVICE
    static Layout LayoutAccum() {
      return Layout::packed({ShapeAccum::kRow, ShapeAccum::kColumn});
    }

    /// Returns a TensorRef to the Accumulator
    CUTLASS_HOST_DEVICE
    TensorRefAccum accum_ref() {
      return TensorRefAccum{accum.data(), LayoutAccum()};
    }

  };

  using AccumulatorSharedStorage0 = AccumulatorSharedStorage<
                                    Shape0, typename SmemAccumulatorIterator0::Element, 
                                    typename SmemAccumulatorIterator0::TensorLayout,
                                    typename SmemAccumulatorIterator0::Padding>;

  struct B2bMmaSharedStorage {
    typename Base::B2bMmaSharedStorage b2b_mma_shared_storage;
    AccumulatorSharedStorage0 accumulator_shared_storage0;
  }; 

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  B2bMmaBaseSmemAccumulator(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      B2bMmaSharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      Base(shared_storage.b2b_mma_shared_storage, thread_idx, warp_idx, lane_idx) {
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
