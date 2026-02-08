/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cutlass/epilogue/thread/activation.h"
#include "../../common/cutlass_unit_test.h"

#include "gemm_testbed_3x_ptr_array.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

TEST(SM100_Device_Gemm_e2m3t_e2m3n_f32t_tensorop_1sm_f32_ptr_array, 128x128x256_1x1x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementA = cutlass::float_e2m3_t;
  using ElementB = cutlass::float_e2m3_t;
  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementSF = cutlass::float_ue8m0_t;
  using MmaTypePairA = cute::tuple<ElementA, ElementSF>;
  using MmaTypePairB = cute::tuple<ElementB, ElementSF>;

  using MmaTileShape = cute::Shape<_128,_128,_256>;
  using ClusterShape = Shape<_1,_1,_1>;

  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, 4,
      ElementD, LayoutC, 4,
      EpilogueSchedule
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      MmaTypePairA, LayoutA, 128,
      MmaTypePairB, LayoutB, 128,
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

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = test::gemm::device::TestSmall<Gemm, true>(1.0, 0.5);
  EXPECT_TRUE(pass);
}

TEST(SM100_Device_Gemm_e2m3t_e2m3n_f32n_tensorop_1sm_f32_ptr_array, 256x512x256_2x4x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using ElementA = cutlass::float_e2m3_t;
  using ElementB = cutlass::float_e2m3_t;
  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementSF = cutlass::float_ue8m0_t;
  using MmaTypePairA = cute::tuple<ElementA, ElementSF>;
  using MmaTypePairB = cute::tuple<ElementB, ElementSF>;

  using MmaTileShape = cute::Shape<_128,_128,_256>;
  using ClusterShape = Shape<_2,_4,_1>;

  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, 4,
      ElementD, LayoutC, 4,
      EpilogueSchedule
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      MmaTypePairA, LayoutA, 128,
      MmaTypePairB, LayoutB, 128,
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

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = test::gemm::device::TestSmall<Gemm>(1.0, 0.5);
  EXPECT_TRUE(pass);
}

TEST(SM100_Device_Gemm_e2m3t_e2m3n_f32t_tensorop_1sm_f32_ptr_array, 512x768x256_4x4x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementA = cutlass::float_e2m3_t;
  using ElementB = cutlass::float_e2m3_t;
  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementSF = cutlass::float_ue8m0_t;
  using MmaTypePairA = cute::tuple<ElementA, ElementSF>;
  using MmaTypePairB = cute::tuple<ElementB, ElementSF>;

  using MmaTileShape = cute::Shape<_128,_192,_256>;
  using ClusterShape = Shape<_4,_4,_1>;

  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, 4,
      ElementD, LayoutC, 4,
      EpilogueSchedule
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      MmaTypePairA, LayoutA, 128,
      MmaTypePairB, LayoutB, 128,
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

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = test::gemm::device::TestSmall<Gemm, true>(1.0, 0.5);
  EXPECT_TRUE(pass);
}

TEST(SM100_Device_Gemm_e2m3t_e2m3n_f32n_tensorop_1sm_f32_ptr_array, 512x1024x256_4x4x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using ElementA = cutlass::float_e2m3_t;
  using ElementB = cutlass::float_e2m3_t;
  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementSF = cutlass::float_ue8m0_t;
  using MmaTypePairA = cute::tuple<ElementA, ElementSF>;
  using MmaTypePairB = cute::tuple<ElementB, ElementSF>;

  using MmaTileShape = cute::Shape<_128,_256,_256>;
  using ClusterShape = Shape<_4,_4,_1>;

  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, 4,
      ElementD, LayoutC, 4,
      EpilogueSchedule
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      MmaTypePairA, LayoutA, 128,
      MmaTypePairB, LayoutB, 128,
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

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = test::gemm::device::TestSmall<Gemm>(1.0, 0.5);
  EXPECT_TRUE(pass);
}

TEST(SM100_Device_Gemm_e2m3t_e2m3n_f32n_tensorop_2sm_f32_ptr_array, 256x256x256_2x2x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using ElementA = cutlass::float_e2m3_t;
  using ElementB = cutlass::float_e2m3_t;
  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementSF = cutlass::float_ue8m0_t;
  using MmaTypePairA = cute::tuple<ElementA, ElementSF>;
  using MmaTypePairB = cute::tuple<ElementB, ElementSF>;

  using MmaTileShape = cute::Shape<_256,_128,_256>;
  using ClusterShape = Shape<_2,_2,_1>;

  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
  using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, 4,
      ElementD, LayoutC, 4,
      EpilogueSchedule
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      MmaTypePairA, LayoutA, 128,
      MmaTypePairB, LayoutB, 128,
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

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = test::gemm::device::TestSmall<Gemm, true>(1.0, 0.5);
  EXPECT_TRUE(pass);
}

TEST(SM100_Device_Gemm_e2m3t_e2m3n_f32t_tensorop_2sm_f32_ptr_array, 512x512x256_4x4x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementA = cutlass::float_e2m3_t;
  using ElementB = cutlass::float_e2m3_t;
  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementSF = cutlass::float_ue8m0_t;
  using MmaTypePairA = cute::tuple<ElementA, ElementSF>;
  using MmaTypePairB = cute::tuple<ElementB, ElementSF>;

  using MmaTileShape = cute::Shape<_256,_128,_256>;
  using ClusterShape = Shape<_4,_4,_1>;

  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
  using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, 4,
      ElementD, LayoutC, 4,
      EpilogueSchedule
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      MmaTypePairA, LayoutA, 128,
      MmaTypePairB, LayoutB, 128,
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

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = test::gemm::device::TestSmall<Gemm>(1.0, 0.5);
  EXPECT_TRUE(pass);
}

TEST(SM100_Device_Gemm_e2m3t_e2m3n_f32n_tensorop_2sm_f32_ptr_array, 512x768x256_4x4x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using ElementA = cutlass::float_e2m3_t;
  using ElementB = cutlass::float_e2m3_t;
  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementSF = cutlass::float_ue8m0_t;
  using MmaTypePairA = cute::tuple<ElementA, ElementSF>;
  using MmaTypePairB = cute::tuple<ElementB, ElementSF>;

  using MmaTileShape = cute::Shape<_256,_192,_256>;
  using ClusterShape = Shape<_4,_4,_1>;

  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
  using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, 4,
      ElementD, LayoutC, 4,
      EpilogueSchedule
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      MmaTypePairA, LayoutA, 128,
      MmaTypePairB, LayoutB, 128,
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

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = test::gemm::device::TestSmall<Gemm>(1.0, 0.5);
  EXPECT_TRUE(pass);
}

TEST(SM100_Device_Gemm_e2m3t_e2m3n_f32t_tensorop_2sm_f32_ptr_array, 512x1024x256_4x4x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementA = cutlass::float_e2m3_t;
  using ElementB = cutlass::float_e2m3_t;
  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementSF = cutlass::float_ue8m0_t;
  using MmaTypePairA = cute::tuple<ElementA, ElementSF>;
  using MmaTypePairB = cute::tuple<ElementB, ElementSF>;

  using MmaTileShape = cute::Shape<_256,_256,_256>;
  using ClusterShape = Shape<_4,_4,_1>;

  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
  using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, 4,
      ElementD, LayoutC, 4,
      EpilogueSchedule
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      MmaTypePairA, LayoutA, 128,
      MmaTypePairB, LayoutB, 128,
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

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = test::gemm::device::TestSmall<Gemm, true>(1.0, 0.5);
  EXPECT_TRUE(pass);
}

#endif // #if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
