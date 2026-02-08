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

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "../../common/cutlass_unit_test.h"

#include "gemm_testbed_3x.hpp"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_tf32t_tf32n_f32n_tensor_op_gmma_rs_ws_f32, 64x128x32) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_32>;
  using ClusterShape_MNK = Shape<_1,_1,_1>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 4,
      ElementB, LayoutB, 4,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, 4,
      float, LayoutC, 4,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_tf32n_tf32n_f32n_tensor_op_gmma_rs_ws_f32, 64x128x32) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_32>;
  using ClusterShape_MNK = Shape<_1,_1,_1>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 4,
      ElementB, LayoutB, 4,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, 4,
      float, LayoutC, 4,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_tf32t_tf32t_f32n_tensor_op_gmma_rs_ws_f32, 64x128x32) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_32>;
  using ClusterShape_MNK = Shape<_1,_1,_1>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 4,
      ElementB, LayoutB, 4,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, 4,
      float, LayoutC, 4,
      cutlass::gemm::EpilogueTransposed
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_tf32n_tf32t_f32n_tensor_op_gmma_rs_ws_f32, 64x128x32) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_32>;
  using ClusterShape_MNK = Shape<_1,_1,_1>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 4,
      ElementB, LayoutB, 4,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, 4,
      float, LayoutC, 4,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_tf32t_tf32n_f32n_tensor_op_gmma_rs_ws_f32, 64x128x32_4x2x1) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_32>;
  using ClusterShape_MNK = Shape<_4,_2,_1>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 4,
      ElementB, LayoutB, 4,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, 4,
      float, LayoutC, 4,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_tf32n_tf32n_f32n_tensor_op_gmma_rs_ws_f32, 64x128x32_4x2x1) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_32>;
  using ClusterShape_MNK = Shape<_4,_2,_1>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 4,
      ElementB, LayoutB, 4,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, 4,
      float, LayoutC, 4,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_tf32t_tf32t_f32n_tensor_op_gmma_rs_ws_f32, 64x128x32_4x2x1) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_32>;
  using ClusterShape_MNK = Shape<_4,_2,_1>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 4,
      ElementB, LayoutB, 4,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, 4,
      float, LayoutC, 4,
      cutlass::gemm::EpilogueTransposed
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_tf32n_tf32t_f32n_tensor_op_gmma_rs_ws_f32, 64x128x32_4x2x1) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_32>;
  using ClusterShape_MNK = Shape<_4,_2,_1>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 4,
      ElementB, LayoutB, 4,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, 4,
      float, LayoutC, 4,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////
////////////  CollectiveBuilder with KernelScheduleAuto  //////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_tf32t_tf32n_f32n_tensor_op_gmma_rs_ws_f32, 64x128x32_4x2x1_auto_schedule) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_32>;
  using ClusterShape_MNK = Shape<_4,_2,_1>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 4,
      ElementB, LayoutB, 4,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, 4,
      float, LayoutC, 4,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_tf32n_tf32n_f32n_tensor_op_gmma_rs_ws_f32, 64x128x32_4x2x1_auto_schedule) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_32>;
  using ClusterShape_MNK = Shape<_4,_2,_1>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 4,
      ElementB, LayoutB, 4,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, 4,
      float, LayoutC, 4,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_tf32t_tf32t_f32n_tensor_op_gmma_rs_ws_f32, 64x128x32_4x2x1_auto_schedule) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_32>;
  using ClusterShape_MNK = Shape<_4,_2,_1>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 4,
      ElementB, LayoutB, 4,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, 4,
      float, LayoutC, 4,
      cutlass::gemm::EpilogueTransposed
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_tf32n_tf32t_f32n_tensor_op_gmma_rs_ws_f32, 64x128x32_4x2x1_auto_schedule) {
  using ElementA = cutlass::tfloat32_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::tfloat32_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_32>;
  using ClusterShape_MNK = Shape<_4,_2,_1>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 4,
      ElementB, LayoutB, 4,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, 4,
      float, LayoutC, 4,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
