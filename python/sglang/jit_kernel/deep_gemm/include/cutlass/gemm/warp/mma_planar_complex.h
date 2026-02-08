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
    \brief Templates implementing warp-level matrix multiply-accumulate operations.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/complex.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/array_planar_complex.h"
#include "cutlass/gemm/warp/tile_iterator_planar_complex.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  /// Underlying real-valued warp-level matrix multiply
  typename Operator_,
  /// Transformation applied to A operand (typically folded into math instruction)
  ComplexTransform TransformA = ComplexTransform::kNone,
  /// Transformation applied to B operand (typically folded into math instruction)
  ComplexTransform TransformB = ComplexTransform::kNone
>
class MmaPlanarComplex {
public:

  /// Underlying real-valued warp-level matrix multiply
  using Operator = Operator_;

  /// Shape of warp-level matrix multipy
  using Shape = typename Operator::Shape;

  /// Transformation applied to A operand (typically folded into math instruction)
  static ComplexTransform const kTransformA = TransformA;

  /// Transformation applied to B operand (typically folded into math instruction)
  static ComplexTransform const kTransformB = TransformB;

  /// Fragment of elements
  using FragmentA = ArrayPlanarComplex<typename Operator::ElementA, Operator::FragmentA::kElements>;

  /// Iterator into planar complex
  using IteratorA = TileIteratorPlanarComplex<typename Operator::IteratorA>;

  /// Layout in memory of the A operand
  using LayoutA = typename Operator::LayoutA;

  using FragmentB = ArrayPlanarComplex<typename Operator::ElementB, Operator::FragmentB::kElements>;

  /// Iterator into planar complex
  using IteratorB = TileIteratorPlanarComplex<typename Operator::IteratorB>;

  /// Layout in memory of the B operand
  using LayoutB = typename Operator::LayoutB;

  /// Tile iterator for accumulator
  using IteratorC = TileIteratorPlanarComplex<typename Operator::IteratorC>;

  /// Accumulator fragment
  using FragmentC = ArrayPlanarComplex<typename Operator::ElementC, Operator::FragmentC::kElements>;

  /// Layout of accumulator fragment in memory
  using LayoutC = typename Operator::LayoutC;

private:

    /// Number of mma operations performed
  using MmaIterations = MatrixShape<
    Operator::Shape::kM / Operator::Policy::Operator::Shape::kM,
    Operator::Shape::kN / Operator::Policy::Operator::Shape::kN
  >;

public:
  /// Ctor
  CUTLASS_DEVICE
  MmaPlanarComplex() {}

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D, 
    FragmentA const &A_in, 
    FragmentB const &B_in, 
    FragmentC const &C) const {

    D.real = C.real;
    D.imag = C.imag;

    //
    // Transform fragments based on conjugate operations.
    //

    negate<typename FragmentA::ArrayReal> neg_A;

    FragmentA frag_A;
    frag_A.real = A_in.real;

    if (kTransformA == ComplexTransform::kConjugate) {
      frag_A.imag = neg_A(frag_A.imag);
    }
    else {
      frag_A.imag = frag_A.imag;
    }

    FragmentB frag_B;
    frag_B.real = B_in.real;

    if (kTransformB == ComplexTransform::kConjugate) {
      negate<typename FragmentB::ArrayReal> neg;
      frag_B.imag = neg(frag_B.imag);
    }
    else {
      frag_B.imag = frag_B.imag;
    }

    //
    // Accumulated real-valued matrix multiplies
    //

    Operator real_mma;

    // D.i += A.i * B.r
    real_mma(D.imag, frag_A.imag, frag_B.real, D.imag);

    // D.r += A.r * B.r
    real_mma(D.real, frag_A.real, frag_B.real, D.real);

    // D.i += A.r * B.i
    real_mma(D.imag, frag_A.real, frag_B.imag, D.imag);

    // D.r += -A.i * B.i
    frag_A.imag = neg_A(frag_A.imag);
    real_mma(D.real, frag_A.imag, frag_B.imag, D.real);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
