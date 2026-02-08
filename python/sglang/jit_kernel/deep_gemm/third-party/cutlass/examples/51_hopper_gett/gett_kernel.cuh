/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cute/tensor.hpp"

#include "cutlass/arch/arch.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace example {

//
// GETT entry point
//
template <
  class ProblemShapeMNKL,
  class ElementA,
  class StrideA,
  class ElementB,
  class StrideB,
  class ElementAccumulator,
  class ElementC,
  class StrideC,
  class ElementD,
  class StrideD,
  class ElementEpilogue>
cutlass::Status
gett_kernel(
    ProblemShapeMNKL problem_shape_mnkl,
    ElementA const* ptr_A, StrideA stride_a_mkl,
    ElementB const* ptr_B, StrideB stride_b_nkl,
    ElementAccumulator _,
    ElementC const* ptr_C, StrideC stride_c_mnl,
    ElementD      * ptr_D, StrideD stride_d_mnl,
    ElementEpilogue alpha, ElementEpilogue beta,
    cudaStream_t stream = 0) {
  using namespace cute;

  // TileShape -- GETT configuration
  // Specify the number of elements to take from each mode 
  // BLK_M = (M0,M1,...)  BLK_N = (M0,M1,...)  BLK_K = (K0,K1,...)

  // Take 128 from m0, 128 from n0, 64 from k0
  using TileShape = Shape<Shape<_128>, Shape<_128>, Shape<_64>>;

  /* Other examples:
   * Take 32 elements from m0 and 4 elements from m1
   * Take 64 elements from n0 and 2 elements from n1
   * Take  8 elements from k0 and 8 elements from k1
  **/
  // using TileShape = Shape<Shape<_32,_4>, Shape<_64,_2>, Shape<_8,_8>>;
  
  using EpilogueThreadOp = cutlass::epilogue::thread::LinearCombination<
      ElementD, 1, ElementAccumulator, ElementEpilogue, cutlass::epilogue::thread::ScaleType::Default,
      cutlass::FloatRoundStyle::round_to_nearest, ElementC>;

  // No changes are required to the default epilogue
  using CollectiveEpilogue = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::DefaultEpilogue<
      ElementC,
      StrideC,
      StrideD,
      EpilogueThreadOp,
      cutlass::gemm::EpilogueDefault>>;

  // CollectiveMma for GETTs can be built using the CollectiveBuilders
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, StrideA, 128 / cutlass::sizeof_bits<ElementA>::value,
      ElementB, StrideB, 128 / cutlass::sizeof_bits<ElementB>::value,
      ElementAccumulator,
      TileShape, Shape<_1,_2,_1>,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  // The GETT kernel is a composition of a collective mainloop and epilogue, just like any 3.x GEMM
  using GettKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShapeMNKL,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using GettOperator = cutlass::gemm::device::GemmUniversalAdapter<GettKernel>;

  typename GettOperator::Arguments args {
    cutlass::gemm::GemmUniversalMode::kBatched,
    problem_shape_mnkl,
    { ptr_A, stride_a_mkl, ptr_B, stride_b_nkl }, 
    { {alpha, beta}, ptr_C, stride_c_mnl, ptr_D, stride_d_mnl }
  };

#if CUTLASS_DEBUG_TRACE_LEVEL > 0
  print("Problem shape:");
  print("\tM: "); print(cute::get<0>(problem_shape_mnkl)); print("\n");
  print("\tN: "); print(cute::get<1>(problem_shape_mnkl)); print("\n");
  print("\tK: "); print(cute::get<2>(problem_shape_mnkl)); print("\n");
  print("\tL: "); print(cute::get<3>(problem_shape_mnkl)); print("\n");
  print("TileSape:"); print(TileShape{}); print("\n");
#endif

  GettOperator op;
  return op(args, stream);
}

} // namespace example
