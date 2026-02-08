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
    \brief Template for a threadblock-scoped GEMV kernel.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/gemm/gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix-vector product using SIMT math instructions.
template <
  class Core_ //< GemvCore
>
class Gemv {
public:
  using Shape = typename Core_::Shape;

  /// The MMA operator that computes GEMV 
  using Operator = typename Core_::Operator;

  /// Iterates over A in global memory
  using IteratorA = typename Core_::IteratorA;

  /// Iterates over B in global memory
  using IteratorB = typename Core_::IteratorB;

  /// Fragment of operand C loaded from global memory
  using IteratorC = typename Core_::IteratorC;

  /// Fragment of operand A loaded from global memory
  using FragmentA = typename IteratorA::Fragment;

  /// Fragment of operand B loaded from global memory
  using FragmentB = typename IteratorB::Fragment;

  /// Fragment of operand accumulator loaded/stored to global memory
  using FragmentC = typename Operator::FragmentC;

  /// Shape of the per-thread GEMV operation
  using ThreadShape = typename Core_::ThreadShape;

public:
  CUTLASS_DEVICE
  Gemv() { }

  CUTLASS_DEVICE
  void operator()(
    GemmCoord const &problem_size,    ///< problem size of batched GEMV
    FragmentC &accum,                 ///< destination accumulator tile
    IteratorA iterator_A,             ///< iterator over A operand in global memory
    IteratorB iterator_B,             ///< iterator over B operand in global memory
    FragmentC const &src_accum) {     ///< source accumulator tile

    //
    // Prologue
    //

    FragmentA frag_A;
    FragmentB frag_B;
    frag_A.clear();
    frag_B.clear();

    iterator_A.load(frag_A);
    iterator_B.load(frag_B);
    ++iterator_A;
    ++iterator_B;

    //
    // Mainloop
    //
    Operator thread_mma;
    int gemm_k = problem_size.k();

    if (gemm_k < Shape::kK)
    {
      iterator_A.clear_mask();
      iterator_B.clear_mask();
    }

    // iterate over K to accumulate result
    CUTLASS_GEMM_LOOP
    for (; gemm_k > 0; gemm_k -= Shape::kK) {
      thread_mma(accum, frag_A, frag_B, accum);

      iterator_A.load(frag_A);
      iterator_B.load(frag_B);
      ++iterator_A;
      ++iterator_B;

      if (gemm_k < Shape::kK)
      {
        iterator_A.clear_mask();
        iterator_B.clear_mask();
      }
    }

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass
