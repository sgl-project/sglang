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
  \brief Epilogue for threadblock scoped complex GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

  
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"
#include "cutlass/epilogue/thread/linear_combination_sigmoid.h"
#include "cutlass/epilogue/thread/linear_combination_planar_complex.h"

#include "cutlass/epilogue/thread/conversion_op.h"
#include "cutlass/epilogue/thread/reduction_op.h"

#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"

#include "cutlass/epilogue/warp/fragment_iterator_complex_tensor_op.h"
#include "cutlass/epilogue/warp/fragment_iterator_gaussian_complex_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_thread_map_tensor_op.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_blas3.h"
#include "cutlass/epilogue/threadblock/shared_load_iterator.h"

#include "cutlass/epilogue/threadblock/epilogue.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Specialization and defines sensible defaults for epilogues for complex*complex case
//  4 real-valued mma operations (Complex)
//  A = (ar + j ai), B (br +j bi), D = AB
//  D = dr + j di = (ar*br - ai*bi) + j (ar*bi + ai*br) 
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  /// Epilogue Shape
  typename Shape_,
  /// Warp-level mma operator
  typename WarpMmaTensorOp_,
  /// Number of k partitions
  int PartitionsK,
  /// Epilogue output operator
  typename OutputOp_,
  /// Elements accessed by inner-most loop of AccumulatorFragmentIterator::load()
  int ElementsPerAccess,
  /// Multiply-add operator 
  /// Selects between (arch::OpMultiplyAddComplex, arch::OpMultiplyGaussianComplex) 
  typename Operator_ = arch::OpMultiplyAddComplex,
  /// Is for a symmetric kernel
  BlasMode BlasMode_ = BlasMode::kGemm
> 
struct DefaultEpilogueComplexTensorOpBlas3 {

  using Shape = Shape_;
  using WarpMmaTensorOp = WarpMmaTensorOp_;
  static int const kPartitionsK = PartitionsK;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = ElementsPerAccess;
  using Operator = Operator_;
  static BlasMode const kBlasMode = BlasMode_;

  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaTensorOp::LayoutC;
  using ElementAccumulator = typename WarpMmaTensorOp::ElementC;

  //
  // Thread map
  //

  using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
    Shape,
    typename WarpMmaTensorOp::Shape,
    kPartitionsK,
    ElementOutput,
    kElementsPerAccess
  >::Type;

  using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIteratorBlas3<
    OutputTileThreadMap,
    ElementOutput
    , kBlasMode
  >;

  using AccumulatorFragmentIterator = cutlass::epilogue::warp::FragmentIteratorComplexTensorOp<
    typename WarpMmaTensorOp::Shape,
    typename WarpMmaTensorOp::Policy::Operator::Shape,
    typename WarpMmaTensorOp::Policy::Operator::ElementC,
    typename WarpMmaTensorOp::Policy::Operator::FragmentC,
    LayoutC
  >;

  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorTensorOp<
    typename WarpMmaTensorOp::Shape,
    typename WarpMmaTensorOp::Policy::Operator::Shape,
    ElementAccumulator,
    LayoutC
  >;

  using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIterator<
    typename OutputTileThreadMap::CompactedThreadMap,
    ElementAccumulator
  >;

  /// Hard-coded padding elements added 
  using Padding = cutlass::MatrixShape<0, 0>;

  //
  // Define the epilogue
  //
  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    OutputTileIterator,
    AccumulatorFragmentIterator,
    WarpTileIterator,
    SharedLoadIterator,
    OutputOp,
    Padding
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Partial specialization and defines sensible defaults for epilogues for complex*complex case
//  3 real-valued mma operations (Gaussian Complex)
//  A  = (ar + j ai), B = (br +j bi), D = AB
//  P1 = (ar + ai) * br, P2 = - ar * (br - bi), P3 = ai * (br + bi) 
//  D  = dr + j di = (P1 - P3) + j (P1 + P2)
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  typename Shape_,
  typename WarpMmaTensorOp_,
  int PartitionsK,
  typename OutputOp_,
  int ElementsPerAccess, 
  BlasMode BlasMode_
>
struct DefaultEpilogueComplexTensorOpBlas3 <Shape_, WarpMmaTensorOp_, PartitionsK, 
                                      OutputOp_, ElementsPerAccess, 
                                      arch::OpMultiplyAddGaussianComplex
                                      , BlasMode_
> {

  using Shape = Shape_;
  using WarpMmaTensorOp = WarpMmaTensorOp_;
  static int const kPartitionsK = PartitionsK;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = ElementsPerAccess;
  using Operator = arch::OpMultiplyAddGaussianComplex;
  static BlasMode const kBlasMode = BlasMode_;

  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaTensorOp::LayoutC;
  using ElementAccumulator = typename WarpMmaTensorOp::ElementC;

  //
  // Thread map
  //

  using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
    Shape,
    typename WarpMmaTensorOp::Shape,
    kPartitionsK,
    ElementOutput,
    kElementsPerAccess
  >::Type;

  using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIteratorBlas3<
    OutputTileThreadMap,
    ElementOutput,
    kBlasMode
  >;

  using AccumulatorFragmentIterator = cutlass::epilogue::warp::FragmentIteratorGaussianComplexTensorOp<
    typename WarpMmaTensorOp::Shape,
    typename WarpMmaTensorOp::Policy::Operator::Shape,
    typename WarpMmaTensorOp::Policy::Operator::ElementC,
    typename WarpMmaTensorOp::Policy::Operator::FragmentC,
    LayoutC
  >;

  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorTensorOp<
    typename WarpMmaTensorOp::Shape,
    typename WarpMmaTensorOp::Policy::Operator::Shape,
    ElementAccumulator,
    LayoutC
  >;

  using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIterator<
    typename OutputTileThreadMap::CompactedThreadMap,
    ElementAccumulator
  >;

  /// Hard-coded padding elements added 
  using Padding = cutlass::MatrixShape<0, 0>;

  //
  // Define the epilogue
  //
  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    OutputTileIterator,
    AccumulatorFragmentIterator,
    WarpTileIterator,
    SharedLoadIterator,
    OutputOp,
    Padding
  >;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
