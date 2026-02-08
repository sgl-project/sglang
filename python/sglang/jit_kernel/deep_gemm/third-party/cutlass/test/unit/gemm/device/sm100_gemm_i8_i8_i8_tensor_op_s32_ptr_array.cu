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
    \brief Tests for device-wide GEMM interface
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "../../common/cutlass_unit_test.h"

#include "gemm_testbed_3x_ptr_array.hpp"

using namespace cute;

#if (defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) && !defined(CUTLASS_SM100_FAMILY_ARCHS_ENABLED))

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// 128x64x128 Cluster1x1x1 TMEM 4x1 ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM100_Device_Gemm_s8t_s8n_s8n_tensorop_1cta_s32_ptr_array, 128x64x128_1x1x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementC = int8_t;
  using ElementD = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using ElementBias = int8_t;
  using MmaTileShape = cute::Shape<_128,_64,Int<128 / sizeof(ElementA)>>;
  using ClusterShape = Shape<_1,_1,_1>;

  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementD, LayoutC, 16 / sizeof(ElementD),
      EpilogueSchedule
    >::CollectiveOp;

  using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16 / sizeof(ElementA),
      ElementB, LayoutB, 16 / sizeof(ElementB),
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cutlass::gemm::ArrayProblemShape<Shape<int,int,int,int>>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = TestSmall<Gemm>(2, 0.5, CheckEquality::EXACT);
  EXPECT_TRUE(pass);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// 128x64x128 Cluster4x2x1 TMEM 4x1 ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM100_Device_Gemm_s8t_s8n_s8n_tensorop_1cta_s32_ptr_array, 512x128x128_4x2x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementC = int8_t;
  using ElementD = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using ElementBias = int8_t;
  using MmaTileShape = Shape<_128,_64,Int<128 / sizeof(ElementA)>>;
  using ClusterShape = Shape<_4,_2,_1>;

  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementD, LayoutC, 16 / sizeof(ElementD),
      EpilogueSchedule
    >::CollectiveOp;

  using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16 / sizeof(ElementA),
      ElementB, LayoutB, 16 / sizeof(ElementB),
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cutlass::gemm::ArrayProblemShape<Shape<int,int,int,int>>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = TestSmall<Gemm>(2, 0.5, CheckEquality::EXACT);
  EXPECT_TRUE(pass);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// 64x256x128 Cluster1x1x1 TMEM 4x1 ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM100_Device_Gemm_s8t_s8n_s32n_tensorop_1cta_s32_ptr_array, 64x256x128_1x1x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementC = int32_t;
  using ElementD = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;
  using ElementBias = int32_t;
  using MmaTileShape = cute::Shape<_64,_256,Int<128 / sizeof(ElementA)>>;
  using ClusterShape = Shape<_1,_1,_1>;

  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementD, LayoutC, 16 / sizeof(ElementD),
      EpilogueSchedule
    >::CollectiveOp;

  using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16 / sizeof(ElementA),
      ElementB, LayoutB, 16 / sizeof(ElementB),
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cutlass::gemm::ArrayProblemShape<Shape<int,int,int,int>>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = TestSmall<Gemm>(2, 0.5, CheckEquality::EXACT);
  EXPECT_TRUE(pass);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// 64x256x128 Cluster2x4x1 TMEM 2x2 ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM100_Device_Gemm_s8t_s8n_s8n_tensorop_2cta_s32_ptr_array, 128x1024x128_2x4x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementC = int8_t;
  using ElementD = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using ElementBias = int8_t;
  using MmaTileShape = Shape<_128,_256,Int<128 / sizeof(ElementA)>>;
  using ClusterShape = Shape<_2,_4,_1>;

  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementD, LayoutC, 16 / sizeof(ElementD),
      EpilogueSchedule
    >::CollectiveOp;

  using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16 / sizeof(ElementA),
      ElementB, LayoutB, 16 / sizeof(ElementB),
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cutlass::gemm::ArrayProblemShape<Shape<int,int,int,int>>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = TestSmall<Gemm>(2, 0.5, CheckEquality::EXACT);
  EXPECT_TRUE(pass);
}

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) && !defined(CUTLASS_SM100_FAMILY_ARCHS_ENABLED)

