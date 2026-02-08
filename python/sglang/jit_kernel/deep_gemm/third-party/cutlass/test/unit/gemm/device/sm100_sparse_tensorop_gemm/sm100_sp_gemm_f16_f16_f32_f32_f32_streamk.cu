/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>

#include "../../../common/cutlass_unit_test.h"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "../gemm_testbed_3x.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////// 128x128x64 //////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM100_Device_Sparse_Gemm_f16t_f16n_f32t_tensorop_1cta_f32_streamk, 128x128x64_1x1x1) {
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using MmaTileShape = Shape<_128,_128,_64>;
  using ClusterShape = Shape<_1,_1,_1>;

  constexpr int ALIGNMENT_C = 4;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, ALIGNMENT_C,
      float, LayoutC, ALIGNMENT_C,
      cutlass::epilogue::TmaWarpSpecialized1Sm
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      cutlass::half_t, LayoutATag, 16,
      cutlass::half_t, LayoutBTag, 8,
      float,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
      cutlass::gemm::KernelSparseTmaWarpSpecialized1SmSm100
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler
  >;

  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(1.0, 0.0, CheckEquality::EXACT, ScalarLoc::ON_DEVICE, VectorScale::ENABLED, {64, 1536});
  EXPECT_TRUE(result);
}

TEST(SM100_Device_Sparse_Gemm_f16n_f16t_f32n_tensorop_1cta_f32_streamk, 256x256x64_2x2x1) {
  using LayoutATag = cutlass::layout::ColumnMajor;
  using LayoutBTag = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using MmaTileShape = Shape<_128,_128,_64>;
  using ClusterShape = Shape<_2,_2,_1>;

  constexpr int ALIGNMENT_C = 4;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, ALIGNMENT_C,
      float, LayoutC, ALIGNMENT_C,
      cutlass::epilogue::TmaWarpSpecialized1Sm
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      cutlass::half_t, LayoutATag, 16,
      cutlass::half_t, LayoutBTag, 8,
      float,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
      cutlass::gemm::KernelSparseTmaWarpSpecialized1SmSm100
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler
  >;

  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(1.0, 0.0, CheckEquality::EXACT, ScalarLoc::ON_DEVICE, VectorScale::ENABLED, {64, 1536});
  EXPECT_TRUE(result);
}


TEST(SM100_Device_Sparse_Gemm_f16t_f16n_f32t_tensorop_2cta_f32_streamk, 256x256x64_2x2x1) {
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using MmaTileShape = Shape<_256,_128,_64>;
  using ClusterShape = Shape<_2,_2,_1>;

  constexpr int ALIGNMENT_C = 4;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, ALIGNMENT_C,
      float, LayoutC, ALIGNMENT_C,
      cutlass::epilogue::TmaWarpSpecialized2Sm
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      cutlass::half_t, LayoutATag, 16,
      cutlass::half_t, LayoutBTag, 8,
      float,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
      cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler
  >;

  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(1.0, 0.0, CheckEquality::EXACT, ScalarLoc::ON_DEVICE, VectorScale::ENABLED, {64, 1536});
  EXPECT_TRUE(result);
}

TEST(SM100_Device_Sparse_Gemm_f16n_f16t_f32n_tensorop_2cta_f32_streamk, 512x512x64_4x4x1) {
  using LayoutATag = cutlass::layout::ColumnMajor;
  using LayoutBTag = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using MmaTileShape = Shape<_256,_128,_64>;
  using ClusterShape = Shape<_4,_4,_1>;

  constexpr int ALIGNMENT_C = 4;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, ALIGNMENT_C,
      float, LayoutC, ALIGNMENT_C,
      cutlass::epilogue::TmaWarpSpecialized2Sm
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      cutlass::half_t, LayoutATag, 16,
      cutlass::half_t, LayoutBTag, 8,
      float,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
      cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler
  >;

  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(1.0, 0.0, CheckEquality::EXACT, ScalarLoc::ON_DEVICE, VectorScale::ENABLED, {64, 1536});
  EXPECT_TRUE(result);
}

TEST(SM100_Device_Sparse_Gemm_f16t_f16n_f32n_tensorop_2cta_f32_streamk, 256x512x128_2x4x1) {
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using MmaTileShape = Shape<_256,_128,_128>;
  using ClusterShape = Shape<_2,_4,_1>;

  constexpr int ALIGNMENT_C = 4;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, ALIGNMENT_C,
      float, LayoutC, ALIGNMENT_C,
      cutlass::epilogue::TmaWarpSpecialized2Sm
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      cutlass::half_t, LayoutATag, 16,
      cutlass::half_t, LayoutBTag, 8,
      float,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
      cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler
  >;

  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(1.0, 0.0, CheckEquality::EXACT, ScalarLoc::ON_DEVICE, VectorScale::ENABLED, {64, 1536});
  EXPECT_TRUE(result);
}

TEST(SM100_Device_Sparse_Gemm_f16t_f16n_f32t_tensorop_2cta_f32_streamk, 256x256x64_2x1x1) {
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using MmaTileShape = Shape<_256,_256,_64>;
  using ClusterShape = Shape<_2,_1,_1>;

  constexpr int ALIGNMENT_C = 4;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, ALIGNMENT_C,
      float, LayoutC, ALIGNMENT_C,
      cutlass::epilogue::TmaWarpSpecialized2Sm
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      cutlass::half_t, LayoutATag, 16,
      cutlass::half_t, LayoutBTag, 8,
      float,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
      cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler
  >;

  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(1.0, 0.0, CheckEquality::EXACT, ScalarLoc::ON_DEVICE, VectorScale::ENABLED, {64, 1536});
  EXPECT_TRUE(result);
}

// Enable this after linearized scheduler is functional again.
#if 0
TEST(SM100_Device_Sparse_Gemm_f16t_f16n_f32t_tensorop_1cta_f32_linearized, 128x128x64_1x1x1) {
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using MmaTileShape = Shape<_128,_128,_64>;
  using ClusterShape = Shape<_1,_1,_1>;

  constexpr int ALIGNMENT_C = 4;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, ALIGNMENT_C,
      float, LayoutC, ALIGNMENT_C,
      cutlass::epilogue::TmaWarpSpecialized1Sm
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      cutlass::half_t, LayoutATag, 16,
      cutlass::half_t, LayoutBTag, 8,
      float,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
      cutlass::gemm::KernelSparseTmaWarpSpecialized1SmSm100
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::LinearizedScheduler
  >;

  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(1.0, 0.0, CheckEquality::EXACT, ScalarLoc::ON_DEVICE, VectorScale::ENABLED, {64, 1024, 2048});
  EXPECT_TRUE(result);
}

TEST(SM100_Device_Sparse_Gemm_f16n_f16t_f32n_tensorop_1cta_f32_linearized, 256x256x64_2x2x1) {
  using LayoutATag = cutlass::layout::ColumnMajor;
  using LayoutBTag = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using MmaTileShape = Shape<_128,_128,_64>;
  using ClusterShape = Shape<_2,_2,_1>;

  constexpr int ALIGNMENT_C = 4;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, ALIGNMENT_C,
      float, LayoutC, ALIGNMENT_C,
      cutlass::epilogue::TmaWarpSpecialized1Sm
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      cutlass::half_t, LayoutATag, 16,
      cutlass::half_t, LayoutBTag, 8,
      float,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
      cutlass::gemm::KernelSparseTmaWarpSpecialized1SmSm100
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::LinearizedScheduler
  >;

  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(1.0, 0.0, CheckEquality::EXACT, ScalarLoc::ON_DEVICE, VectorScale::ENABLED, {64, 1024, 2048});
  EXPECT_TRUE(result);
}


TEST(SM100_Device_Sparse_Gemm_f16t_f16n_f32t_tensorop_2cta_f32_linearized, 256x256x64_2x2x1) {
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using MmaTileShape = Shape<_256,_128,_64>;
  using ClusterShape = Shape<_2,_2,_1>;

  constexpr int ALIGNMENT_C = 4;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, ALIGNMENT_C,
      float, LayoutC, ALIGNMENT_C,
      cutlass::epilogue::TmaWarpSpecialized2Sm
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      cutlass::half_t, LayoutATag, 16,
      cutlass::half_t, LayoutBTag, 8,
      float,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
      cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::LinearizedScheduler
  >;

  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(1.0, 0.0, CheckEquality::EXACT, ScalarLoc::ON_DEVICE, VectorScale::ENABLED, {64, 1024, 2048});
  EXPECT_TRUE(result);
}

TEST(SM100_Device_Sparse_Gemm_f16n_f16t_f32n_tensorop_2cta_f32_linearized, 512x512x64_4x4x1) {
  using LayoutATag = cutlass::layout::ColumnMajor;
  using LayoutBTag = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using MmaTileShape = Shape<_256,_128,_64>;
  using ClusterShape = Shape<_4,_4,_1>;

  constexpr int ALIGNMENT_C = 4;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, ALIGNMENT_C,
      float, LayoutC, ALIGNMENT_C,
      cutlass::epilogue::TmaWarpSpecialized2Sm
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      cutlass::half_t, LayoutATag, 16,
      cutlass::half_t, LayoutBTag, 8,
      float,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
      cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::LinearizedScheduler
  >;

  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(1.0, 0.0, CheckEquality::EXACT, ScalarLoc::ON_DEVICE, VectorScale::ENABLED, {64, 1024, 2048});
  EXPECT_TRUE(result);
}

TEST(SM100_Device_Sparse_Gemm_f16t_f16n_f32n_tensorop_2cta_f32_linearized, 256x512x128_2x4x1) {
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using MmaTileShape = Shape<_256,_128,_128>;
  using ClusterShape = Shape<_2,_4,_1>;

  constexpr int ALIGNMENT_C = 4;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, ALIGNMENT_C,
      float, LayoutC, ALIGNMENT_C,
      cutlass::epilogue::TmaWarpSpecialized2Sm
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      cutlass::half_t, LayoutATag, 16,
      cutlass::half_t, LayoutBTag, 8,
      float,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
      cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::LinearizedScheduler
  >;

  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(1.0, 0.0, CheckEquality::EXACT, ScalarLoc::ON_DEVICE, VectorScale::ENABLED, {64, 1024, 2048});
  EXPECT_TRUE(result);
}

TEST(SM100_Device_Sparse_Gemm_f16t_f16n_f32t_tensorop_2cta_f32_linearized, 256x256x64_2x1x1) {
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using MmaTileShape = Shape<_256,_256,_64>;
  using ClusterShape = Shape<_2,_1,_1>;

  constexpr int ALIGNMENT_C = 4;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, ALIGNMENT_C,
      float, LayoutC, ALIGNMENT_C,
      cutlass::epilogue::TmaWarpSpecialized2Sm
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
      cutlass::half_t, LayoutATag, 16,
      cutlass::half_t, LayoutBTag, 8,
      float,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
      cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::LinearizedScheduler
  >;

  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(1.0, 0.0, CheckEquality::EXACT, ScalarLoc::ON_DEVICE, VectorScale::ENABLED, {64, 1024, 2048});
  EXPECT_TRUE(result);
}

#endif

#endif // #if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
