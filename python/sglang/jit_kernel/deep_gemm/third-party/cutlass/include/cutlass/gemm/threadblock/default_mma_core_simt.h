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
    \brief Defines basic properties needed by CTA-level GEMMs assuming expectations about data
      layout of the global memory fragments, data types, and internal tile sizes.

      Partial specializations for threadblock::Mma operations targeting simt instructions.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/fast_math.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"


#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear_2dthreadtile.h"

#include "cutlass/gemm/warp/mma_simt_policy.h"
#include "cutlass/gemm/warp/mma_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

namespace detail {

// convert a WarpShape which is the whole tile of elements into warp num threads.
// The goal is for each thread's tile of elements to be as square as possible
// for performance (4x4 will be faster than 2x8).
template<typename WarpShape>
constexpr int simt_get_warp_threads_m() {
    return (WarpShape::kM > WarpShape::kN) ? 8 : 4;
}

/// Computes padding in shared memory to perform efficient transpose without bank conflicts.
constexpr int simt_transpose_padding(int threads, int crosswise, int size_in_bits) {
  return (size_in_bits >= 32 ?
      threads / crosswise / (size_in_bits / 32) :
      threads / crosswise * (32 / size_in_bits)
  );
}

}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: column-major
///   B: row-major
///   Operator: simt class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Operation performed by GEMM
    typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 1>, ElementA_,
                      layout::ColumnMajor, ElementB_, layout::RowMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, 2, Operator_
                     > {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<1, 1, 1>;
  using ElementA = ElementA_;
  using LayoutA = layout::ColumnMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassSimt;
  static int const PartitionsK = Shape::kK / WarpShape::kK;

  /// Default Operator
  using Operator = Operator_;

  /// Number of warps present
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    PartitionsK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajor;
  using SmemLayoutB = layout::RowMajor;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kM, Shape::kK>,
    kThreads,
    kElementsPerAccess
  >;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kM, Shape::kK>, 
    ElementA, 
    SmemLayoutA,
    1,
    IteratorThreadMapA
  >;

  /// Policy of iterator B
  using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kN, Shape::kK>,
    kThreads,
    kElementsPerAccess
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kK, Shape::kN>, 
    ElementB, 
    SmemLayoutB,
    0,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
  static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
  static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);
  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      1>;
  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,    /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
    ElementA,     /// Data type of A elements
    SmemLayoutA,  /// Layout of A matrix (concept: MatrixLayout)
    ElementB,     /// Data type of B elements
    SmemLayoutB,  /// Layout of B matrix (concept: MatrixLayout)
    ElementC,     /// Element type of C matrix
    LayoutC,      /// Layout of C matrix (concept: MatrixLayout)
    Policy        /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
    >;            /// Used for partial specialization

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<
    MmaWarpSimt,
    MatrixShape<0, 0>,
    MatrixShape<0, 0>,
    WarpCount::kK
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: row-major
///   B: column-major
///   Operator: simt class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Operation performed by GEMM
    typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 1>, ElementA_,
                      layout::RowMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, 2, Operator_
                     > {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<1, 1, 1>;
  using ElementA = ElementA_;
  using LayoutA = layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassSimt;
  static int const PartitionsK = Shape::kK / WarpShape::kK;

  /// Default Operator
  using Operator = Operator_;

  /// Number of warps present
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    PartitionsK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;
  
  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajor;
  using SmemLayoutB = layout::RowMajor;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kK, Shape::kM>,
    kThreads,
    kElementsPerAccess
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapA = transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kM, Shape::kK>, 
    ElementA, 
    SmemLayoutA,
    1,
    SmemThreadMapA // was IteratorThreadMapA
  >;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kK, Shape::kN>,
    kThreads,
    kElementsPerAccess
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapB = transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapB>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kK, Shape::kN>, 
    ElementB, 
    SmemLayoutB,
    0,
    SmemThreadMapB // was IteratorThreadMapA
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
  static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
  static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

  static int const kPaddingM = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementA>::value);
  static int const kPaddingN = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementB>::value);

  static_assert(!(kPaddingM % LaneM) && !(kPaddingN % LaneN),
                "Padding must be divisible by Lane");

  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      1>;
  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
      WarpShape,      /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
      ElementA,       /// Data type of A elements
      SmemLayoutA,    /// Layout of A matrix (concept: MatrixLayout)
      ElementB,       /// Data type of B elements
      SmemLayoutB,    /// Layout of B matrix (concept: MatrixLayout)
      ElementC,       /// Element type of C matrix
      LayoutC,        /// Layout of C matrix (concept: MatrixLayout)
      Policy          /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
  >;

  /// Policy used to define MmaPipelined 
  using MmaPolicy = MmaPolicy<
    MmaWarpSimt,
    MatrixShape<kPaddingM, 0>,    // skew for A matrix to avoid SMEM bank conflicts
    MatrixShape<0, kPaddingN>,    // skew for B matrix to avoid SMEM bank conflicts
    WarpCount::kK
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: row-major
///   B: row-major
///   Operator: simt class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Operation performed by GEMM
    typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 1>, ElementA_,
                      layout::RowMajor, ElementB_, layout::RowMajor, ElementC_,
                      LayoutC_, arch::OpClassSimt, 2, Operator_
                     > {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<1, 1, 1>;
  using ElementA = ElementA_;
  using LayoutA = layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassSimt;
  static int const PartitionsK = Shape::kK / WarpShape::kK;

  /// Default Operator
  using Operator = Operator_;

  /// Number of warps present
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    PartitionsK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajor;
  using SmemLayoutB = layout::RowMajor;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kK, Shape::kM>,
    kThreads,
    kElementsPerAccess
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapA = transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kM, Shape::kK>, 
    ElementA, 
    SmemLayoutA,
    1,
    SmemThreadMapA
  >;

  /// Policy of iterator B
  using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kN, Shape::kK>,
    kThreads,
    kElementsPerAccess
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kK, Shape::kN>, 
    ElementB, 
    SmemLayoutB,
    0,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
  static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
  static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

  static int const kPaddingM = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementA>::value);

  static_assert(!(kPaddingM % LaneM),
                "Padding must be divisible by Lane");

  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      1>;
  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
      WarpShape,    /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
      ElementA,     /// Data type of A elements
      SmemLayoutA,  /// Layout of A matrix (concept: MatrixLayout)
      ElementB,     /// Data type of B elements
      SmemLayoutB,  /// Layout of B matrix (concept: MatrixLayout)
      ElementC,     /// Element type of C matrix
      LayoutC,      /// Layout of C matrix (concept: MatrixLayout)
      Policy        /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
  >;

  /// Policy used to define MmaPipelined 
  using MmaPolicy = MmaPolicy<
    MmaWarpSimt,
    MatrixShape<kPaddingM, 0>,    // skew for A matrix to avoid SMEM bank conflicts
    MatrixShape<0, 0>,
    WarpCount::kK
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: column-major
///   B: column-major
///   Operator: simt class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Operation performed by GEMM
    typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 1>, ElementA_,
                      layout::ColumnMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, 2, Operator_
                     > {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<1, 1, 1>;
  using ElementA = ElementA_;
  using LayoutA = layout::ColumnMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassSimt;
  static int const PartitionsK = Shape::kK / WarpShape::kK;

  /// Default Operator
  using Operator = Operator_;

  /// Number of warps present
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    PartitionsK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajor;
  using SmemLayoutB = layout::RowMajor;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kM, Shape::kK>,
    kThreads,
    kElementsPerAccess
  >;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kM, Shape::kK>, 
    ElementA,
    SmemLayoutA,
    1,
    IteratorThreadMapA
  >;

  /// ThreadMap of iterator B
  using IteratorThreadMapB =  transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kK, Shape::kN>,
    kThreads,
    kElementsPerAccess
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapB = transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapB>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kK, Shape::kN>, 
    ElementB,
    SmemLayoutB,
    0,
    SmemThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
  static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
  static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

  static int const kPaddingN = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementB>::value);

  static_assert(!(kPaddingN % LaneN),
                "Padding must be divisible by Lane");

  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      1>;
  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
      WarpShape,    /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
      ElementA,     /// Data type of A elements
      SmemLayoutA,  /// Layout of A matrix (concept: MatrixLayout)
      ElementB,     /// Data type of B elements
      SmemLayoutB,  /// Layout of B matrix (concept: MatrixLayout)
      ElementC,     /// Element type of C matrix
      LayoutC,      /// Layout of C matrix (concept: MatrixLayout)
      Policy        /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
  >;

  /// Policy used to define MmaPipelined 
  using MmaPolicy = MmaPolicy<
    MmaWarpSimt,
    MatrixShape<0, 0>,
    MatrixShape<0, kPaddingN>, // skew for B matrix to avoid SMEM bank conflicts
    WarpCount::kK
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: column-major
///   B: row-major
///   Operator: simt class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Operation performed by GEMM
    typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 1>, ElementA_,
                      layout::AffineRank2ColumnMajor, ElementB_, layout::AffineRank2RowMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, 2, Operator_
                     > {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<1, 1, 1>;
  using ElementA = ElementA_;
  using LayoutA = layout::AffineRank2ColumnMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::AffineRank2RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassSimt;

  /// Default Operator
  using Operator = Operator_;

  using Base = DefaultMmaCore<Shape,
                              WarpShape,
                              InstructionShape,
                              ElementA,
                              layout::ColumnMajor,
                              ElementB,
                              layout::RowMajor,
                              ElementC,
                              LayoutC,
                              OperatorClass,
                              2,
                              Operator>;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutB = typename Base::SmemLayoutB;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = typename Base::IteratorThreadMapA;

  /// Shared memory iterator to A operand
  using SmemIteratorA = typename Base::SmemIteratorA;

  /// Policy of iterator B
  using IteratorThreadMapB = typename Base::IteratorThreadMapB;

  /// Shared memory iterator to B operand
  using SmemIteratorB = typename Base::SmemIteratorB;

  //
  // Warp-level matrix multiply operator
  //

  /// Policy used to define MmaPipelined
  using MmaPolicy = typename Base::MmaPolicy;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: row-major
///   B: column-major
///   Operator: simt class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Operation performed by GEMM
    typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 1>, ElementA_,
                      layout::AffineRank2RowMajor, ElementB_, layout::AffineRank2ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, 2, Operator_
                     > {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<1, 1, 1>;
  using ElementA = ElementA_;
  using LayoutA = layout::AffineRank2RowMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::AffineRank2ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassSimt;

  /// Default Operator
  using Operator = Operator_;

  using Base = DefaultMmaCore<Shape,
                              WarpShape,
                              InstructionShape,
                              ElementA,
                              layout::RowMajor,
                              ElementB,
                              layout::ColumnMajor,
                              ElementC,
                              LayoutC,
                              OperatorClass,
                              2,
                              Operator>;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutB = typename Base::SmemLayoutB;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = typename Base::IteratorThreadMapA;

  /// Shared memory iterator to A operand
  using SmemIteratorA = typename Base::SmemIteratorA;

  /// Policy of iterator B
  using IteratorThreadMapB = typename Base::IteratorThreadMapB;

  /// Shared memory iterator to B operand
  using SmemIteratorB = typename Base::SmemIteratorB;

  //
  // Warp-level matrix multiply operator
  //

  /// Policy used to define MmaPipelined
  using MmaPolicy = typename Base::MmaPolicy;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: row-major
///   B: row-major
///   Operator: simt class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Operation performed by GEMM
    typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 1>, ElementA_,
                      layout::AffineRank2RowMajor, ElementB_, layout::AffineRank2RowMajor, ElementC_,
                      LayoutC_, arch::OpClassSimt, 2, Operator_
                     > {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<1, 1, 1>;
  using ElementA = ElementA_;
  using LayoutA = layout::AffineRank2RowMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::AffineRank2RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassSimt;

  /// Default Operator
  using Operator = Operator_;

  using Base = DefaultMmaCore<Shape,
                              WarpShape,
                              InstructionShape,
                              ElementA,
                              layout::RowMajor,
                              ElementB,
                              layout::RowMajor,
                              ElementC,
                              LayoutC,
                              OperatorClass,
                              2,
                              Operator>;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutB = typename Base::SmemLayoutB;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = typename Base::IteratorThreadMapA;

  /// Shared memory iterator to A operand
  using SmemIteratorA = typename Base::SmemIteratorA;

  /// Policy of iterator B
  using IteratorThreadMapB = typename Base::IteratorThreadMapB;

  /// Shared memory iterator to B operand
  using SmemIteratorB = typename Base::SmemIteratorB;

  //
  // Warp-level matrix multiply operator
  //

  /// Policy used to define MmaPipelined
  using MmaPolicy = typename Base::MmaPolicy;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: column-major
///   B: column-major
///   Operator: simt class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Operation performed by GEMM
    typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 1>, ElementA_,
                      layout::AffineRank2ColumnMajor, ElementB_, layout::AffineRank2ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, 2, Operator_
                     > {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<1, 1, 1>;
  using ElementA = ElementA_;
  using LayoutA = layout::AffineRank2ColumnMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::AffineRank2ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassSimt;

  /// Default Operator
  using Operator = Operator_;

  using Base = DefaultMmaCore<Shape,
                              WarpShape,
                              InstructionShape,
                              ElementA,
                              layout::ColumnMajor,
                              ElementB,
                              layout::ColumnMajor,
                              ElementC,
                              LayoutC,
                              OperatorClass,
                              2,
                              Operator>;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutB = typename Base::SmemLayoutB;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = typename Base::IteratorThreadMapA;

  /// Shared memory iterator to A operand
  using SmemIteratorA = typename Base::SmemIteratorA;

  /// Policy of iterator B
  using IteratorThreadMapB = typename Base::IteratorThreadMapB;

  /// Shared memory iterator to B operand
  using SmemIteratorB = typename Base::SmemIteratorB;

  //
  // Warp-level matrix multiply operator
  //

  /// Policy used to define MmaPipelined
  using MmaPolicy = typename Base::MmaPolicy;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: column-major
///   B: row-major
///   Operator: simt class, for dp4a
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Operation performed by GEMM
    typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 4>, int8_t,
                      layout::ColumnMajor, int8_t, layout::RowMajor, ElementC_,
                      LayoutC_, arch::OpClassSimt, 2, Operator_
                    > {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<1, 1, 4>;
  using ElementA = int8_t;
  using LayoutA = layout::ColumnMajor;
  using ElementB = int8_t;
  using LayoutB = layout::RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassSimt;
  static int const PartitionsK = Shape::kK / WarpShape::kK;

  /// Default Operator
  using Operator = Operator_;

  /// Number of warps present
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    PartitionsK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajorInterleaved<4>;
  using SmemLayoutB = layout::RowMajorInterleaved<4>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinear2DThreadTileStripminedThreadMap<
    layout::PitchLinearShape<Shape::kM, Shape::kK>,
    kThreads,
    layout::PitchLinearShape<4, 4>
  >;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileIterator2dThreadTile<
    MatrixShape<Shape::kM, Shape::kK>, 
    ElementA, 
    SmemLayoutA,
    1,
    IteratorThreadMapA
  >;
  

  /// Policy of iterator B
  using IteratorThreadMapB = transform::PitchLinear2DThreadTileStripminedThreadMap<
    layout::PitchLinearShape<Shape::kN, Shape::kK>,
    kThreads,
    layout::PitchLinearShape<4, 4>
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator2dThreadTile<
    MatrixShape<Shape::kK, Shape::kN>, 
    ElementB, 
    SmemLayoutB,
    0,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
  static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(4, ThreadTileM);
  static const int LaneN = cutlass::const_min(4, ThreadTileN);
  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      4>;

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::ColumnMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,    /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
    ElementA,     /// Data type of A elements
    SmemLayoutA,  /// Layout of A matrix (concept: MatrixLayout)
    ElementB,     /// Data type of B elements
    SmemLayoutB,  /// Layout of B matrix (concept: MatrixLayout)
    ElementC,     /// Element type of C matrix
    LayoutC,      /// Layout of C matrix (concept: MatrixLayout)
    Policy,       /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
    PartitionsK   /// Number of partitions along K dimension
    >;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<
    MmaWarpSimt,
    MatrixShape<0, 0>,
    MatrixShape<0, 0>,
    WarpCount::kK
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Partial specialization:
//
///
///   A: Row-major
///   B: Column-major
///   Operator: simt class, for dp4a
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Operation performed by GEMM
    typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 4>, int8_t,
                      layout::RowMajor, int8_t, layout::ColumnMajor, ElementC_,
                      LayoutC_, arch::OpClassSimt, 2, Operator_
                      > {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<1, 1, 4>;
  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassSimt;
  static int const PartitionsK = Shape::kK / WarpShape::kK;

  /// Default Operator
  using Operator = Operator_;

  /// Number of warps present
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    PartitionsK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajorInterleaved<4>;
  using SmemLayoutB = layout::RowMajorInterleaved<4>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinear2DThreadTileStripminedThreadMap<
    layout::PitchLinearShape<Shape::kK, Shape::kM>,
    kThreads,
    layout::PitchLinearShape<4, 4>
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapA = transform::TransposePitchLinearThreadMap2DThreadTile<IteratorThreadMapA>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileIterator2dThreadTile<
    MatrixShape<Shape::kM, Shape::kK>, 
    ElementA, 
    SmemLayoutA,
    1,
    SmemThreadMapA
  >;
  

  /// Policy of iterator B
  using IteratorThreadMapB = transform::PitchLinear2DThreadTileStripminedThreadMap<
    layout::PitchLinearShape<Shape::kK, Shape::kN>,
    kThreads,
    layout::PitchLinearShape<4, 4>
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapB = transform::TransposePitchLinearThreadMap2DThreadTile<IteratorThreadMapB>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator2dThreadTile<
    MatrixShape<Shape::kK, Shape::kN>, 
    ElementB, 
    SmemLayoutB,
    0,
    SmemThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
  static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(4, ThreadTileM);
  static const int LaneN = cutlass::const_min(4, ThreadTileN);
  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      4>;

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::ColumnMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,    /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
    ElementA,     /// Data type of A elements
    SmemLayoutA,  /// Layout of A matrix (concept: MatrixLayout)
    ElementB,     /// Data type of B elements
    SmemLayoutB,  /// Layout of B matrix (concept: MatrixLayout)
    ElementC,     /// Element type of C matrix
    LayoutC,      /// Layout of C matrix (concept: MatrixLayout)
    Policy,       /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
    PartitionsK   /// Number of partitions along K dimension
    >;

  static int const kPaddingM = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementA>::value);
  static int const kPaddingN = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementB>::value);

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<
    MmaWarpSimt,
    MatrixShape<kPaddingM, 0>,
    MatrixShape<0, kPaddingN>,
    WarpCount::kK
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Partial specialization:
//
///
///   A: Row-major
///   B: Row-major
///   Operator: simt class, for dp4a
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Operation performed by GEMM
    typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 4>, int8_t,
                      layout::RowMajor, int8_t, layout::RowMajor, ElementC_,
                      LayoutC_, arch::OpClassSimt, 2, Operator_
                      > {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<1, 1, 4>;
  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = layout::RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassSimt;
  static int const PartitionsK = Shape::kK / WarpShape::kK;

  /// Default Operator
  using Operator = Operator_;

  /// Number of warps present
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    PartitionsK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajorInterleaved<4>;
  using SmemLayoutB = layout::RowMajorInterleaved<4>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinear2DThreadTileStripminedThreadMap<
    layout::PitchLinearShape<Shape::kK, Shape::kM>,
    kThreads,
    layout::PitchLinearShape<4, 4>
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapA = transform::TransposePitchLinearThreadMap2DThreadTile<IteratorThreadMapA>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileIterator2dThreadTile<
    MatrixShape<Shape::kM, Shape::kK>, 
    ElementA, 
    SmemLayoutA,
    1,
    SmemThreadMapA
  >;
  
  /// Policy of iterator B
  using IteratorThreadMapB = transform::PitchLinear2DThreadTileStripminedThreadMap<
    layout::PitchLinearShape<Shape::kN, Shape::kK>,
    kThreads,
    layout::PitchLinearShape<4, 4>
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator2dThreadTile<
    MatrixShape<Shape::kK, Shape::kN>, 
    ElementB, 
    SmemLayoutB,
    0,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
  static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(4, ThreadTileM);
  static const int LaneN = cutlass::const_min(4, ThreadTileN);
  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      4>;

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::ColumnMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,    /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
    ElementA,     /// Data type of A elements
    SmemLayoutA,  /// Layout of A matrix (concept: MatrixLayout)
    ElementB,     /// Data type of B elements
    SmemLayoutB,  /// Layout of B matrix (concept: MatrixLayout)
    ElementC,     /// Element type of C matrix
    LayoutC,      /// Layout of C matrix (concept: MatrixLayout)
    Policy,       /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
    PartitionsK   /// Number of partitions along K dimension
    >;

  static int const kPaddingM = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementA>::value);
  static int const kPaddingN = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementB>::value);

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<
    MmaWarpSimt,
    MatrixShape<kPaddingM, 0>,
    MatrixShape<0, 0>,
    WarpCount::kK
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Partial specialization:
//
///
///   A: Column-major
///   B: Column-major
///   Operator: simt class, for dp4a
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Operation performed by GEMM
    typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 4>, int8_t,
                      layout::ColumnMajor, int8_t, layout::ColumnMajor, ElementC_,
                      LayoutC_, arch::OpClassSimt, 2, Operator_
                      > {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<1, 1, 4>;
  using ElementA = int8_t;
  using LayoutA = layout::ColumnMajor;
  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassSimt;
  static int const PartitionsK = Shape::kK / WarpShape::kK;

  /// Default Operator
  using Operator = Operator_;

  /// Number of warps present
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    PartitionsK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajorInterleaved<4>;
  using SmemLayoutB = layout::RowMajorInterleaved<4>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinear2DThreadTileStripminedThreadMap<
    layout::PitchLinearShape<Shape::kM, Shape::kK>,
    kThreads,
    layout::PitchLinearShape<4, 4>
  >;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileIterator2dThreadTile<
    MatrixShape<Shape::kM, Shape::kK>, 
    ElementA, 
    SmemLayoutA,
    1,
    IteratorThreadMapA
  >;
  

  /// Policy of iterator B
  using IteratorThreadMapB = transform::PitchLinear2DThreadTileStripminedThreadMap<
    layout::PitchLinearShape<Shape::kK, Shape::kN>,
    kThreads,
    layout::PitchLinearShape<4, 4>
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapB = transform::TransposePitchLinearThreadMap2DThreadTile<IteratorThreadMapB>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator2dThreadTile<
    MatrixShape<Shape::kK, Shape::kN>, 
    ElementB, 
    SmemLayoutB,
    0,
    SmemThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
  static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(4, ThreadTileM);
  static const int LaneN = cutlass::const_min(4, ThreadTileN);
  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      4>;

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::ColumnMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape,    /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
    ElementA,     /// Data type of A elements
    SmemLayoutA,  /// Layout of A matrix (concept: MatrixLayout)
    ElementB,     /// Data type of B elements
    SmemLayoutB,  /// Layout of B matrix (concept: MatrixLayout)
    ElementC,     /// Element type of C matrix
    LayoutC,      /// Layout of C matrix (concept: MatrixLayout)
    Policy,       /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
    PartitionsK   /// Number of partitions along K dimension
    >;

  static int const kPaddingM = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementA>::value);
  static int const kPaddingN = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementB>::value);

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<
    MmaWarpSimt,
    MatrixShape<0, 0>,
    MatrixShape<0, kPaddingN>,
    WarpCount::kK
  >;
};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass
