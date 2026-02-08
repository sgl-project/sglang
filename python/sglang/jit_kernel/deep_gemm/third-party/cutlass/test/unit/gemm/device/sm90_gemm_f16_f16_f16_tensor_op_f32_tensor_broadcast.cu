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
/*! \file
    \brief Tests for device-wide GEMM interface with an elementwise tensor-tensor broadcast epilogue
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/epilogue_tensor_broadcast.hpp"
#include "cutlass/epilogue/thread/linear_combination_tensor_broadcast.hpp"

#include "../../common/cutlass_unit_test.h"

#include "gemm_testbed_3x_tensor_broadcast.hpp"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16t_f16n_tensor_op_gmma_f32_tensor_broadcast, 64x128x64_ActIdentity_Bin0Plus_Bin1NoOp_UnaryIdentity) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using ElementOutput = float;
  using ElementAccumulator = ElementOutput;
  using ElementCompute = ElementOutput;
  using ElementBias = ElementOutput;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      ElementOutput,
      Shape<_64,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::EpilogueTensorBroadcast<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombinationTensorBroadcast<ElementOutput>,
      cutlass::gemm::EpilogueDefault>>;

  EXPECT_TRUE(EpilogueOp::IsBinaryOp0Enabled);
  EXPECT_TRUE(!EpilogueOp::IsBinaryOp1Enabled);
  EXPECT_TRUE(!EpilogueOp::IsUnaryOpEnabled);

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllTensorBroadcast<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16t_f16n_tensor_op_gmma_f32_tensor_broadcast, 64x128x64_ActReLu_Bin0Plus_Bin1Plus_UnaryNegate) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using ElementOutput = float;
  using ElementAccumulator = ElementOutput;
  using ElementCompute = ElementOutput;
  using ElementBias = ElementOutput;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      ElementOutput,
      Shape<_64,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::EpilogueTensorBroadcast<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombinationTensorBroadcast<
        ElementOutput, ElementAccumulator, ElementCompute, ElementBias,
        cutlass::epilogue::thread::ReLu,
        cutlass::plus,
        cutlass::plus,
        cutlass::negate
        >,
      cutlass::gemm::EpilogueDefault>>;

  EXPECT_TRUE(EpilogueOp::IsBinaryOp0Enabled);
  EXPECT_TRUE(EpilogueOp::IsBinaryOp1Enabled);
  EXPECT_TRUE(EpilogueOp::IsUnaryOpEnabled);

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllTensorBroadcast<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16n_f16t_f16t_tensor_op_gmma_f32_tensor_broadcast, 64x128x64_ActReLu_Bin0Mul_Bin1Plus_UnaryNegate) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using ElementOutput = float;
  using ElementAccumulator = ElementOutput;
  using ElementCompute = ElementOutput;
  using ElementBias = ElementOutput;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      ElementOutput,
      Shape<_64,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::EpilogueTensorBroadcast<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombinationTensorBroadcast<
        ElementOutput, ElementAccumulator, ElementCompute, ElementBias,
        cutlass::epilogue::thread::ReLu,
        cutlass::multiplies,
        cutlass::plus,
        cutlass::negate
        >,
      cutlass::gemm::EpilogueDefault>>;

  EXPECT_TRUE(EpilogueOp::IsBinaryOp0Enabled);
  EXPECT_TRUE(EpilogueOp::IsBinaryOp1Enabled);
  EXPECT_TRUE(EpilogueOp::IsUnaryOpEnabled);

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllTensorBroadcast<Gemm>());
}
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16t_f16n_tensor_op_gmma_f32_tensor_broadcast, 128x128x64_ActReLu_Bin0NoOp_Bin1Plus_UnaryNegate) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using ElementOutput = float;
  using ElementAccumulator = ElementOutput;
  using ElementCompute = ElementOutput;
  using ElementBias = ElementOutput;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      ElementOutput,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::EpilogueTensorBroadcast<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombinationTensorBroadcast<
        ElementOutput, ElementAccumulator, ElementCompute, ElementBias,
        cutlass::epilogue::thread::ReLu,
        cutlass::epilogue::thread::detail::NoOp,
        cutlass::plus,
        cutlass::negate
        >,
      cutlass::gemm::EpilogueDefault>>;

  EXPECT_TRUE(!EpilogueOp::IsBinaryOp0Enabled);
  EXPECT_TRUE(EpilogueOp::IsBinaryOp1Enabled);
  EXPECT_TRUE(EpilogueOp::IsUnaryOpEnabled);

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllTensorBroadcast<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_warpspecialized_tensor_broadcast, 64x128x64_2x2x1_ActReLu_Bin0Mul_Bin1Plus_UnaryNegate) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using ElementOutput = float;
  using ElementAccumulator = ElementOutput;
  using ElementCompute = ElementOutput;
  using ElementBias = ElementOutput;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_2,_2,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::EpilogueTensorBroadcast<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombinationTensorBroadcast<
        ElementOutput, ElementAccumulator, ElementCompute, ElementBias,
        cutlass::epilogue::thread::ReLu,
        cutlass::multiplies,
        cutlass::plus,
        cutlass::negate
        >,
      cutlass::gemm::EpilogueDefault>>;

  EXPECT_TRUE(EpilogueOp::IsBinaryOp0Enabled);
  EXPECT_TRUE(EpilogueOp::IsBinaryOp1Enabled);
  EXPECT_TRUE(EpilogueOp::IsUnaryOpEnabled);

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllTensorBroadcast<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
