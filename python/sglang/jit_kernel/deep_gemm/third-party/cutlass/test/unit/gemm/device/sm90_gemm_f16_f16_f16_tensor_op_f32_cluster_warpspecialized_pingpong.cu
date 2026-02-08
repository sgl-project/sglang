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
#include "cutlass/epilogue/collective/sm70_epilogue_vectorized.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "../../common/cutlass_unit_test.h"

#include "gemm_testbed_3x.hpp"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;

TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_persistent, 64x128x64_1x1x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_64>;
  using ClusterShape_MNK = Shape<_1,_1,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_persistent, 64x128x64_2x1x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_1,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_persistent, 64x128x64_1x2x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_64>;
  using ClusterShape_MNK = Shape<_1,_2,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_persistent, 64x128x64_2x2x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}


TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_persistent, 64x128x64_4x1x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_64>;
  using ClusterShape_MNK = Shape<_4,_1,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_persistent, 64x128x64_1x4x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_64>;
  using ClusterShape_MNK = Shape<_1,_4,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_persistent, 64x128x64_2x4x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_4,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_persistent, 64x128x64_4x4x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_64>;
  using ClusterShape_MNK = Shape<_4,_4,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_persistent, 128x128x64_1x1x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_128,_128,_64>;
  using ClusterShape_MNK = Shape<_1,_1,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_persistent, 128x128x64_2x1x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_128,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_1,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_persistent, 128x128x64_1x2x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_128,_128,_64>;
  using ClusterShape_MNK = Shape<_1,_2,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_persistent, 128x128x64_2x2x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_128,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}


TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_persistent, 128x128x64_4x1x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_128,_128,_64>;
  using ClusterShape_MNK = Shape<_4,_1,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_persistent, 128x128x64_1x4x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_128,_128,_64>;
  using ClusterShape_MNK = Shape<_1,_4,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_persistent, 128x128x64_2x4x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_128,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_4,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_persistent, 128x128x64_4x4x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_128,_128,_64>;
  using ClusterShape_MNK = Shape<_4,_4,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16n_f16n_tensor_op_gmma_f16_persistent_Epilogue, 64x128x64_2x2x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementAccumulator = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using PreSwizzleLayout = Layout<Shape<_64,_128>,Stride<_1,_64>>;
  using TileShapeS2R = Shape<_64,_16>;

  using CollectiveEpilogue = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::Epilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
      ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag_bits<sizeof_bits_v<ElementAccumulator>>, PreSwizzleLayout>,
      Copy_Atom<SM90_U16x8_STSM_T, ElementAccumulator>,
      TiledCopy<Copy_Atom<DefaultCopy, ElementAccumulator>,Layout<Shape<_128,_8>,Stride<_8,_1>>,TileShapeS2R>,
      Copy_Atom<DefaultCopy, ElementC>>>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16n_f16n_tensor_op_gmma_f16_persistent_Epilogue, 128x64x64_2x2x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementAccumulator = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_128,_64,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using PreSwizzleLayout = Layout<Shape<Shape<_64,_2>,_64>,Stride<Stride<_1,_4096>,_64>>;
  using TileShapeS2R = Shape<_128,_8>;

  using CollectiveEpilogue = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::Epilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
      ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag_bits<sizeof_bits_v<ElementAccumulator>>, PreSwizzleLayout>,
      Copy_Atom<SM90_U16x8_STSM_T, ElementAccumulator>,
      TiledCopy<Copy_Atom<DefaultCopy, ElementAccumulator>,Layout<Shape<_128,_8>,Stride<_8,_1>>,TileShapeS2R>,
      Copy_Atom<DefaultCopy, ElementC>>>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16n_f16t_tensor_op_gmma_f16_persistent_Epilogue, 64x128x64_2x2x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementAccumulator = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::RowMajor;
  using TileShape_MNK = Shape<_64,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using PreSwizzleLayout = Layout<Shape<_64,Shape<_64,_2>>,Stride<_64,Stride<_1,_4096>>>;
  using TileShapeS2R = Shape<_8,_128>;

  using CollectiveEpilogue = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::Epilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
      ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag_bits<sizeof_bits_v<ElementAccumulator>>, PreSwizzleLayout>,
      Copy_Atom<SM90_U32x4_STSM_N, ElementAccumulator>,
      TiledCopy<Copy_Atom<DefaultCopy, ElementAccumulator>,Layout<Shape<_128,_8>,Stride<_8,_1>>,TileShapeS2R>,
      Copy_Atom<DefaultCopy, ElementC>>>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16n_f16t_tensor_op_gmma_f16_persistent_Epilogue, 128x64x64_2x2x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementAccumulator = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::RowMajor;
  using TileShape_MNK = Shape<_128,_64,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using PreSwizzleLayout = Layout<Shape<_128,_64>,Stride<_64,_1>>;
  using TileShapeS2R = Shape<_16,_64>;

  using CollectiveEpilogue = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::Epilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
      ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag_bits<sizeof_bits_v<ElementAccumulator>>, PreSwizzleLayout>,
      Copy_Atom<SM90_U32x4_STSM_N, ElementAccumulator>,
      TiledCopy<Copy_Atom<DefaultCopy, ElementAccumulator>,Layout<Shape<_128,_8>,Stride<_8,_1>>,TileShapeS2R>,
      Copy_Atom<DefaultCopy, ElementC>>>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16n_f16n_tensor_op_gmma_f32_persistent_Epilogue, 64x128x64_2x2x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using PreSwizzleLayout = Layout<Shape<_64,_128>,Stride<_1,_64>>;
  using TileShapeS2R = Shape<_64,_16>;

  using CollectiveEpilogue = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::Epilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
      ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag_bits<sizeof_bits_v<ElementAccumulator>>, PreSwizzleLayout>,
      Copy_Atom<DefaultCopy, ElementAccumulator>,
      TiledCopy<Copy_Atom<DefaultCopy, ElementAccumulator>,Layout<Shape<_128,_8>,Stride<_8,_1>>,TileShapeS2R>,
      Copy_Atom<DefaultCopy, ElementC>>>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16n_f16n_tensor_op_gmma_f32_persistent_Epilogue, 128x64x64_2x2x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_128,_64,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using PreSwizzleLayout = Layout<Shape<Shape<_64,_2>,_64>,Stride<Stride<_1,_4096>,_64>>;
  using TileShapeS2R = Shape<_128,_8>;

  using CollectiveEpilogue = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::Epilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
      ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag_bits<sizeof_bits_v<ElementAccumulator>>, PreSwizzleLayout>,
      Copy_Atom<DefaultCopy, ElementAccumulator>,
      TiledCopy<Copy_Atom<DefaultCopy, ElementAccumulator>,Layout<Shape<_128,_8>,Stride<_8,_1>>,TileShapeS2R>,
      Copy_Atom<DefaultCopy, ElementC>>>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16n_f16t_tensor_op_gmma_f32_persistent_Epilogue, 64x128x64_2x2x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::RowMajor;
  using TileShape_MNK = Shape<_64,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using PreSwizzleLayout = Layout<Shape<_64,Shape<_64,_2>>,Stride<_64,Stride<_1,_4096>>>;
  using TileShapeS2R = Shape<_8,_128>;

  using CollectiveEpilogue = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::Epilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
      ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag_bits<sizeof_bits_v<ElementAccumulator>>, PreSwizzleLayout>,
      Copy_Atom<DefaultCopy, ElementAccumulator>,
      TiledCopy<Copy_Atom<DefaultCopy, ElementAccumulator>,Layout<Shape<_128,_8>,Stride<_8,_1>>,TileShapeS2R>,
      Copy_Atom<DefaultCopy, ElementC>>>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16n_f16t_tensor_op_gmma_f32_persistent_Epilogue, 128x64x64_2x2x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::RowMajor;
  using TileShape_MNK = Shape<_128,_64,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using PreSwizzleLayout = Layout<Shape<_128,_64>,Stride<_64,_1>>;
  using TileShapeS2R = Shape<_16,_64>;

  using CollectiveEpilogue = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::Epilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
      ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag_bits<sizeof_bits_v<ElementAccumulator>>, PreSwizzleLayout>,
      Copy_Atom<DefaultCopy, ElementAccumulator>,
      TiledCopy<Copy_Atom<DefaultCopy, ElementAccumulator>,Layout<Shape<_128,_8>,Stride<_8,_1>>,TileShapeS2R>,
      Copy_Atom<DefaultCopy, ElementC>>>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16n_f16n_tensor_op_gmma_f32_persistent_epilogue, 128x128x64_2x2x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_128,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementC, LayoutC, 16 / sizeof(ElementC),
      cutlass::epilogue::TmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16 / sizeof(ElementA),
      ElementB, LayoutB, 16 / sizeof(ElementB),
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>(1, 1));
}

TEST(SM90_Device_Gemm_f16t_f16n_f16t_tensor_op_gmma_f32_persistent_epilogue, 128x128x64_2x2x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::RowMajor;
  using TileShape_MNK = Shape<_128,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementC, LayoutC, 16 / sizeof(ElementC),
      cutlass::epilogue::TmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16 / sizeof(ElementA),
      ElementB, LayoutB, 16 / sizeof(ElementB),
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>(1, 1));
}

TEST(SM90_Device_Gemm_f16t_f16n_f32n_tensor_op_gmma_f32_persistent_epilogue, 128x128x64_2x2x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;
  using ElementC = float;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_128,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementC, LayoutC, 16 / sizeof(ElementC),
      cutlass::epilogue::TmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16 / sizeof(ElementA),
      ElementB, LayoutB, 16 / sizeof(ElementB),
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>(1, 1));
}

TEST(SM90_Device_Gemm_f16t_f16n_f32t_tensor_op_gmma_f32_persistent_epilogue, 128x128x64_2x2x1) {
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;
  using TileShape_MNK = Shape<_128,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementC, LayoutC, 16 / sizeof(ElementC),
      cutlass::epilogue::TmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16 / sizeof(ElementA),
      ElementB, LayoutB, 16 / sizeof(ElementB),
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>(1, 1));
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
