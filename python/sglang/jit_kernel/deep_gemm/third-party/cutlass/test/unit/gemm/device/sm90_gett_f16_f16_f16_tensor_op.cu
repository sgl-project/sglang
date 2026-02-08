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
    \brief Tests for device-wide GEMM interface
*/

#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/sm70_epilogue_vectorized.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/reference/device/gett.hpp"
#include "cutlass/util/reference/device/tensor_compare.h"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gett_f16t_f16n_f16n_tensor_op_gmma_f16, 8x8x8x8x8x8) {

  using BatModeStrides = int;

  using RowModeStridesA = cute::Stride<int, int>;
  using RedModeStrides = cute::Stride<cute::_1, int>;

  using ColModeStridesB = cute::Stride<int, int>;

  using RowModeStridesC = cute::Stride<cute::_1, int>;
  using ColModeStridesC = cute::Stride<int, int>;

  using StrideA = cute::Stride<RowModeStridesA, RedModeStrides,  BatModeStrides>;
  using StrideB = cute::Stride<ColModeStridesB, RedModeStrides,  BatModeStrides>;
  using StrideC = cute::Stride<RowModeStridesC, ColModeStridesC, BatModeStrides>;
  using StrideD = StrideC;

  using TileShape = Shape<Shape<_8, _8>, Shape<_8, _8>, Shape<_8, _8>>;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, StrideA, 8,
      cutlass::half_t, StrideB, 8,
      cutlass::half_t,
      TileShape, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      cutlass::half_t, cutlass::half_t,
      cutlass::half_t, StrideC, 8,
      cutlass::half_t, StrideC, 8,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GettKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<Shape<int,int>,
            Shape<int,int>,
            Shape<int,int>,
            int>,
      CollectiveOp,
      CollectiveEpilogue
  >;

  using Gett = cutlass::gemm::device::GemmUniversalAdapter<GettKernel>;

  auto problem_shape = make_shape(
    make_shape(32,8),
    make_shape(32,4),
    make_shape(32,2),
    1 
  );

  auto [M, N, K, L] = problem_shape;

  StrideA dA = make_stride(make_stride(64, 2048), make_stride(_1{}, 32), size(M) * size(K));
  StrideB dB = make_stride(make_stride(64, 2048), make_stride(_1{}, 32), size(N) * size(K));
  StrideC dC = make_stride(make_stride(_1{}, 32), make_stride(256, 8192), size(M) * size(N));
  StrideD dD = dC;

  cutlass::half_t alpha = cutlass::half_t(1.0f);
  cutlass::half_t beta  = cutlass::half_t(1.0f);

  thrust::host_vector<cutlass::half_t> A_h(size(M) * size(K) * size(L));
  thrust::host_vector<cutlass::half_t> B_h(size(N) * size(K) * size(L));
  thrust::host_vector<cutlass::half_t> C_h(size(M) * size(N) * size(L));
  thrust::host_vector<cutlass::half_t> D_h(size(M) * size(N) * size(L));
  thrust::host_vector<cutlass::half_t> D_h_ref(size(M) * size(N) * size(L));

  for (auto& a : A_h) a = cutlass::half_t(static_cast<int>(4 * (rand() / double(RAND_MAX) - 1)));
  for (auto& b : B_h) b = cutlass::half_t(static_cast<int>(4 * (rand() / double(RAND_MAX) - 1)));
  for (auto& c : C_h) c = cutlass::half_t(static_cast<int>(4 * (rand() / double(RAND_MAX) - 1)));
  for (auto& d : D_h) d = cutlass::half_t(-1);
  for (auto& d : D_h_ref) d = cutlass::half_t(-1);

  thrust::device_vector<cutlass::half_t> A = A_h;
  thrust::device_vector<cutlass::half_t> B = B_h;
  thrust::device_vector<cutlass::half_t> C = C_h;
  thrust::device_vector<cutlass::half_t> D = D_h;
  thrust::device_vector<cutlass::half_t> D_ref = D_h_ref;

  typename Gett::Arguments args {
    cutlass::gemm::GemmUniversalMode::kBatched,
    problem_shape,
    {A.data().get(), dA, B.data().get(), dB},
    { {alpha, beta}, C.data().get(), dC, D.data().get(), dD}
  };

  Gett gett;
  auto status = gett(args);
  EXPECT_TRUE(status == cutlass::Status::kSuccess);
  auto cuda_err = cudaDeviceSynchronize();

  EXPECT_TRUE(cuda_err == cudaSuccess);

  cutlass::reference::device::gett(
    problem_shape,
    A.data().get(), dA,
    B.data().get(), dB,
    cutlass::half_t(0.0f),
    C.data().get(), dC,
    D_ref.data().get(), dD,
    alpha, beta);
  
  cuda_err = cudaDeviceSynchronize();
  EXPECT_TRUE(cuda_err == cudaSuccess);

  bool passed = cutlass::reference::device::BlockCompareEqual(
      D.data().get(), D_ref.data().get(), D_ref.size());
  EXPECT_TRUE(passed);
}


/////////////////////////////////////////////////////////////////////////////////////////////////

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
