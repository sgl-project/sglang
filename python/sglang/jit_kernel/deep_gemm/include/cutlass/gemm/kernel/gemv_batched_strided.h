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

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/gemm/gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

namespace detail
{
  template<typename ElementAlphaBeta, bool BetaIsZero>
  struct GemvBatchedStridedEpilogueScaling
  {
    ElementAlphaBeta const & alpha;
    ElementAlphaBeta const & beta;

    CUTLASS_DEVICE
    GemvBatchedStridedEpilogueScaling(ElementAlphaBeta& alpha_, ElementAlphaBeta& beta_) :
      alpha(alpha_), beta(beta_)
    { }

    template<typename FragmentCD, typename FragmentAccumulator>
    CUTLASS_DEVICE
    void operator()(FragmentAccumulator& accumulators,
                    FragmentCD const& fragment_C,
                    FragmentCD& fragment_D) const
    {
      using AccType = typename FragmentAccumulator::value_type;
      using CDType = typename FragmentCD::value_type;

      static_assert(FragmentCD::kElements == FragmentAccumulator::kElements,
                    "Mismatch in fragment sizes.");

      for (int i = 0; i < FragmentCD::kElements; ++i)
      {
        if (BetaIsZero)
        {
          fragment_D[i] = CDType(accumulators[i] * AccType(alpha));
        }
        else
        {
          fragment_D[i] = CDType(accumulators[i] * AccType(alpha)
                                 + AccType(fragment_C[i]) * AccType(beta));
        } 
      } 
    }
  };
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemvKernel, typename ElementAlphaBeta, bool BetaIsZero=false>
CUTLASS_DEVICE void GemvBatchedStridedDevice(
  cutlass::gemm::BatchedGemmCoord problem_size,
  ElementAlphaBeta alpha,
  ElementAlphaBeta beta,
  typename GemvKernel::IteratorA::TensorRef ref_A,
  typename GemvKernel::IteratorA::TensorRef::LongIndex lda, 
  typename GemvKernel::IteratorB::TensorRef ref_B,
  typename GemvKernel::IteratorB::TensorRef::LongIndex ldb, 
  typename GemvKernel::IteratorCD::TensorRef ref_C,
  typename GemvKernel::IteratorCD::TensorRef::LongIndex ldc,
  typename GemvKernel::IteratorCD::TensorRef ref_D,
  typename GemvKernel::IteratorCD::TensorRef::LongIndex ldd)
{
  using ThreadBlockGemv = typename GemvKernel::ThreadBlockGemv;
  using ThreadBlockSwizzle = typename GemvKernel::ThreadBlockSwizzle;
  using EpilogueScale = detail::GemvBatchedStridedEpilogueScaling<ElementAlphaBeta, BetaIsZero>;

  ThreadBlockSwizzle swizzler;

  // Compute initial location in logical coordinates
  BatchedGemmCoord tb_offset = swizzler.get_tile_offset();
  int const batch_idx = swizzler.get_batch_idx();

  // Offset to the batch
  ref_A.add_pointer_offset(batch_idx*lda);
  ref_B.add_pointer_offset(batch_idx*ldb);

  // Construct iterators to A and B operands
  typename GemvKernel::IteratorA::Params params_A(ref_A.layout());
  typename GemvKernel::IteratorA iterator_A(
      params_A,
      ref_A.data(),
      { 1, problem_size.k() },
      0,
      { 0, 0 });

  typename GemvKernel::IteratorB::Params params_B(ref_B.layout());
  typename GemvKernel::IteratorB iterator_B(
      params_B,
      ref_B.data(),
      { problem_size.k(), problem_size.n() },
      threadIdx.x,
      { 0, tb_offset.n()*ThreadBlockGemv::Shape::kN });

  //
  // Main loop
  //

  // Construct thread-scoped matrix multiply
  ThreadBlockGemv mma;

  typename ThreadBlockGemv::FragmentC accumulators;
  accumulators.clear();

  // Compute threadblock-scoped gemv
  mma(problem_size.mnk(), accumulators, iterator_A, iterator_B, accumulators);

  //
  // Epilogue
  //
  typename GemvKernel::FragmentCD fragment_CD;

  // Load C (skip if beta is zero)
  if (!BetaIsZero)
  {
    tb_offset = swizzler.get_tile_offset();
    ref_C.add_pointer_offset(batch_idx*ldc);
    typename GemvKernel::IteratorCD::Params params_C(ref_C.layout());
    typename GemvKernel::IteratorCD iterator_C(
        params_C,
        ref_C.data(),
        { 1, problem_size.n() },
        threadIdx.x,
        { 0, tb_offset.n()*ThreadBlockGemv::Shape::kN });
    iterator_C.load(fragment_CD);
  }

  // Apply alpha/beta scaling
  EpilogueScale epilogue_scale(alpha, beta);
  epilogue_scale(accumulators, fragment_CD, fragment_CD);

  // Store D
  tb_offset = swizzler.get_tile_offset();
  ref_D.add_pointer_offset(batch_idx*ldd);
  typename GemvKernel::IteratorCD::Params params_D(ref_D.layout());
  typename GemvKernel::IteratorCD iterator_D(
      params_D,
      ref_D.data(),
      { 1, problem_size.n() },
      threadIdx.x,
      { 0, tb_offset.n()*ThreadBlockGemv::Shape::kN });
  iterator_D.store(fragment_CD);
}

template <typename GemvKernel, typename ElementAlphaBeta, bool BetaIsZero>
CUTLASS_GLOBAL void GemvBatchedStrided(
  cutlass::gemm::BatchedGemmCoord problem_size,
  ElementAlphaBeta alpha,
  ElementAlphaBeta beta,
  typename GemvKernel::IteratorA::TensorRef ref_A,
  typename GemvKernel::IteratorA::TensorRef::LongIndex lda, 
  typename GemvKernel::IteratorB::TensorRef ref_B,
  typename GemvKernel::IteratorB::TensorRef::LongIndex ldb, 
  typename GemvKernel::IteratorCD::TensorRef ref_C,
  typename GemvKernel::IteratorCD::TensorRef::LongIndex ldc,
  typename GemvKernel::IteratorCD::TensorRef ref_D,
  typename GemvKernel::IteratorCD::TensorRef::LongIndex ldd)
{
  GemvBatchedStridedDevice<GemvKernel, ElementAlphaBeta, BetaIsZero>(
    problem_size, alpha, beta, ref_A, lda, ref_B, ldb, ref_C, ldc, ref_D, ldd
  );
}

template <typename GemvKernel, typename ElementAlphaBeta>
CUTLASS_GLOBAL void GemvBatchedStrided(
  cutlass::gemm::BatchedGemmCoord problem_size,
  ElementAlphaBeta alpha,
  typename GemvKernel::IteratorA::TensorRef ref_A,
  typename GemvKernel::IteratorA::TensorRef::LongIndex lda, 
  typename GemvKernel::IteratorB::TensorRef ref_B,
  typename GemvKernel::IteratorB::TensorRef::LongIndex ldb, 
  typename GemvKernel::IteratorCD::TensorRef ref_D,
  typename GemvKernel::IteratorCD::TensorRef::LongIndex ldd)
{
  GemvBatchedStridedDevice<GemvKernel, ElementAlphaBeta, true>(
    problem_size, alpha, ElementAlphaBeta(0), ref_A, lda, ref_B, ldb, ref_D, ldd, ref_D, ldd
  );
}

template <typename GemvKernel>
CUTLASS_GLOBAL void GemvBatchedStrided(
  cutlass::gemm::BatchedGemmCoord problem_size,
  typename GemvKernel::IteratorA::TensorRef ref_A,
  typename GemvKernel::IteratorA::TensorRef::LongIndex lda, 
  typename GemvKernel::IteratorB::TensorRef ref_B,
  typename GemvKernel::IteratorB::TensorRef::LongIndex ldb, 
  typename GemvKernel::IteratorCD::TensorRef ref_D,
  typename GemvKernel::IteratorCD::TensorRef::LongIndex ldd)
{
  using ElementAlphaBeta = typename GemvKernel::IteratorCD::Element;
  GemvBatchedStridedDevice<GemvKernel, ElementAlphaBeta, true>(
    problem_size, ElementAlphaBeta(1), ElementAlphaBeta(0), ref_A, lda, ref_B, ldb, ref_D, ldd, ref_D, ldd
  );
}


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass
