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
    /// Number of stages,
    int Stages,
    /// Used for partial specialization
    typename Enable = bool>
class B2bMmaBase {
 public:
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape0 = Shape0_;
  using Shape1 = Shape1_;

  ///< Policy describing tuning details
  using Policy0 = Policy0_;
  using Policy1 = Policy1_;

  //
  // Dependent types
  //

  /// Warp-level Mma
  using Operator0 = typename Policy0::Operator;
  using Operator1 = typename Policy1::Operator;

  /// Shape describing the overall GEMM computed from shared memory
  /// by each warp.
  using WarpGemm0 = typename Policy0::Operator::Shape;
  using WarpGemm1 = typename Policy1::Operator::Shape;

  /// Shape describing the number of warps filling the CTA
  using WarpCount0 = GemmShape<Shape0::kM / WarpGemm0::kM,
                               Shape0::kN / WarpGemm0::kN,
                               Shape0::kK / WarpGemm0::kK>;
  using WarpCount1 = GemmShape<Shape1::kM / WarpGemm1::kM,
                               Shape1::kN / WarpGemm1::kN,
                               Shape1::kK / WarpGemm1::kK>;

  /// Number of warp-level GEMM oeprations
  static int const kWarpGemmIterations0 =
      (WarpGemm0::kK / Operator0::Policy::MmaShape::kK);
  static int const kWarpGemmIterations1 =
      (WarpGemm1::kK / Operator1::Policy::MmaShape::kK);

  /// Number of stages
  static int const kStages = Stages;

  //
  // Nested structs
  //

  /// Shared storage object needed by threadblock-scoped GEMM
  template<
    typename Shape_,
    typename Policy_
  >
  class SharedStorage {
   public:
    //
    // Type definitions
    //
    using Shape = Shape_;
    using Policy = Policy_;
    using Operator = typename Policy::Operator;

    /// Tensor reference to the A operand
    using TensorRefA = TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;
  
    /// Tensor reference to the B operand
    using TensorRefB = TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;


    /// Shape of the A matrix operand in shared memory
    using ShapeA = MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow,
                               Shape::kK * kStages +
                                   Policy::SmemPaddingA::kColumn>;

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
  };

  using SharedStorage0 = SharedStorage<Shape0, Policy0>;
  using SharedStorage1 = SharedStorage<Shape1, Policy1>;
  union B2bMmaSharedStorage {
    SharedStorage0 shared_storage0;
    SharedStorage1 shared_storage1;
  };


 protected:

  //
  // Data members
  //

  /// Iterator to load a warp-scoped tile of A0 operand from shared memory
  typename Operator0::IteratorA warp_tile_iterator_A0_;

  /// Iterator to load a warp-scoped tile of B0 operand from shared memory
  typename Operator0::IteratorB warp_tile_iterator_B0_;

  /// Iterator to load a warp-scoped tile of B1 operand from shared memory
  typename Operator1::IteratorB warp_tile_iterator_B1_;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  B2bMmaBase(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      B2bMmaSharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      warp_tile_iterator_A0_(shared_storage.shared_storage0.operand_A_ref(), lane_idx),
      warp_tile_iterator_B0_(shared_storage.shared_storage0.operand_B_ref(), lane_idx),
      warp_tile_iterator_B1_(shared_storage.shared_storage1.operand_B_ref(), lane_idx) {

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
