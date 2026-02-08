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
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/epilogue/thread/activation.h"
#include "../../common/cutlass_unit_test.h"

#include "gemm_testbed_3x.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/// A Row B Col
TEST(SM100_Device_Gemm_e4m3t_e4m3n_f32t_tensorop_2sm_f32_auto, 512x512x128_4x4x1) {
  using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementC = void;
  using ElementD = float;
  using ElementCompute = float;
  using ElementAccumulator = float;
  using ElementAccumulator = float;
  using GmemLayoutA = cutlass::layout::RowMajor;
  using GmemLayoutB = cutlass::layout::ColumnMajor;
  using GmemLayoutC = cutlass::layout::RowMajor;
  using MmaTileShape_MNK = Shape<_256,_128,_128>;
  using ClusterShape_MNK = Shape<_4,_4,_1>;
  
  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, GmemLayoutC, 4,
      ElementD, GmemLayoutC, 4,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  //
  // Construct CollectiveMainloop
  //
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementA, GmemLayoutA, 16,
      ElementB, GmemLayoutB, 16,
      ElementAccumulator,
      MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
    >;
  
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = test::gemm::device::TestSmallFusion<Gemm>(1.0, 0);
  EXPECT_TRUE(pass);
}

/// A Row B Col
TEST(SM100_Device_Gemm_e4m3t_e4m3n_f32t_tensorop_2sm_f32_auto, 512x256x128_4x4x1) {
  using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementC = void;
  using ElementD = float;
  using ElementCompute = float;
  using ElementAccumulator = float;
  using ElementAccumulator = float;
  using GmemLayoutA = cutlass::layout::RowMajor;
  using GmemLayoutB = cutlass::layout::ColumnMajor;
  using GmemLayoutC = cutlass::layout::RowMajor;
  using MmaTileShape_MNK = Shape<_256,_64,_128>;
  using ClusterShape_MNK = Shape<_4,_4,_1>;
  
  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, GmemLayoutC, 4,
      ElementD, GmemLayoutC, 4,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  //
  // Construct CollectiveMainloop
  //
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementA, GmemLayoutA, 16,
      ElementB, GmemLayoutB, 16,
      ElementAccumulator,
      MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
    >;
  
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = test::gemm::device::TestSmallFusion<Gemm>(1.0, 0);
  EXPECT_TRUE(pass);
}


/// A Col B Row
TEST(SM100_Device_Gemm_e4m3n_e4m3t_f32t_tensorop_2sm_f32_auto, 512x512x128_4x4x1) {
  using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementC = void;
  using ElementD = float;
  using ElementCompute = float;
  using ElementAccumulator = float;
  using ElementAccumulator = float;
  using GmemLayoutA = cutlass::layout::ColumnMajor;
  using GmemLayoutB = cutlass::layout::RowMajor;
  using GmemLayoutC = cutlass::layout::RowMajor;
  using MmaTileShape_MNK = Shape<_256,_128,_128>;
  using ClusterShape_MNK = Shape<_4,_4,_1>;
  
  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, GmemLayoutC, 4,
      ElementD, GmemLayoutC, 4,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  //
  // Construct CollectiveMainloop
  //
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementA, GmemLayoutA, 16,
      ElementB, GmemLayoutB, 16,
      ElementAccumulator,
      MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
    >;
  
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = test::gemm::device::TestSmallFusion<Gemm>(1.0, 0);
  EXPECT_TRUE(pass);
}

/// A Row B Row
TEST(SM100_Device_Gemm_e4m3t_e4m3t_f32t_tensorop_2sm_f32_auto, 512x512x128_4x4x1) {
  using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementC = void;
  using ElementD = float;
  using ElementCompute = float;
  using ElementAccumulator = float;
  using ElementAccumulator = float;
  using GmemLayoutA = cutlass::layout::RowMajor;
  using GmemLayoutB = cutlass::layout::RowMajor;
  using GmemLayoutC = cutlass::layout::RowMajor;
  using MmaTileShape_MNK = Shape<_256,_128,_128>;
  using ClusterShape_MNK = Shape<_4,_4,_1>;
  
  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, GmemLayoutC, 4,
      ElementD, GmemLayoutC, 4,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  //
  // Construct CollectiveMainloop
  //
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementA, GmemLayoutA, 16,
      ElementB, GmemLayoutB, 16,
      ElementAccumulator,
      MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
    >;
  
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = test::gemm::device::TestSmallFusion<Gemm>(1.0, 0);
  EXPECT_TRUE(pass);
}

/// A Col B Col
TEST(SM100_Device_Gemm_e4m3n_e4m3n_f32t_tensorop_2sm_f32_auto, 512x512x128_4x4x1) {
  using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementC = void;
  using ElementD = float;
  using ElementCompute = float;
  using ElementAccumulator = float;
  using ElementAccumulator = float;
  using GmemLayoutA = cutlass::layout::ColumnMajor;
  using GmemLayoutB = cutlass::layout::ColumnMajor;
  using GmemLayoutC = cutlass::layout::RowMajor;
  using MmaTileShape_MNK = Shape<_256,_128,_128>;
  using ClusterShape_MNK = Shape<_4,_4,_1>;
  
  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, GmemLayoutC, 4,
      ElementD, GmemLayoutC, 4,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  //
  // Construct CollectiveMainloop
  //
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementA, GmemLayoutA, 16,
      ElementB, GmemLayoutB, 16,
      ElementAccumulator,
      MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
    >;
  
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = test::gemm::device::TestSmallFusion<Gemm>(1.0, 0);
  EXPECT_TRUE(pass);
}
#endif // #if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
