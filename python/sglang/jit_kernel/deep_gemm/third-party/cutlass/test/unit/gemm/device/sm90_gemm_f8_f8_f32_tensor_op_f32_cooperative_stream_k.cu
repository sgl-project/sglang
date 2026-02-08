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
    \brief Tests for device-wide GEMM interface with stream-K interface
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "../../common/cutlass_unit_test.h"

#include "gemm_testbed_3x.hpp"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;

TEST(SM90_Device_Gemm_e4m3t_e4m3n_f32n_tensor_op_gmma_f32_cooperative_stream_k, 128x128x128_1x1x1) {
  using ElementA = cutlass::float_e4m3_t;
  using LayoutA  = cutlass::layout::RowMajor;
  using ElementB = cutlass::float_e4m3_t;
  using LayoutB  = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC  = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  using TileShape_MNK = Shape<_128,_128,_128>;
  using ClusterShape_MNK = Shape<_1,_1,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::Identity, ElementC, ElementAccumulator, ElementAccumulator>;
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementC, LayoutC, 16 / sizeof(ElementC),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16,
      ElementB, LayoutB, 16,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp,
      cutlass::gemm::StreamKScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>(1.0, 0.0));
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>(1.0, 1.0));
}

TEST(SM90_Device_Gemm_e4m3t_e4m3n_f32n_tensor_op_gmma_f32_cooperative_stream_k, 256x128x128_1x1x1) {
  using ElementA = cutlass::float_e4m3_t;
  using LayoutA  = cutlass::layout::RowMajor;
  using ElementB = cutlass::float_e4m3_t;
  using LayoutB  = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC  = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  using TileShape_MNK = Shape<_256,_128,_128>;
  using ClusterShape_MNK = Shape<_1,_1,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::Identity, ElementC, ElementAccumulator, ElementAccumulator>;
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementC, LayoutC, 16 / sizeof(ElementC),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16,
      ElementB, LayoutB, 16,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp,
      cutlass::gemm::StreamKScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>(1.0, 0.0));
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>(1.0, 1.0));
}

///////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Cluster 2x1x1  ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_e4m3t_e4m3n_f32n_tensor_op_gmma_f32_cooperative_stream_k, 128x128x128_1x2x1) {
  using ElementA = cutlass::float_e4m3_t;
  using LayoutA  = cutlass::layout::RowMajor;
  using ElementB = cutlass::float_e4m3_t;
  using LayoutB  = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC  = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  using TileShape_MNK = Shape<_128,_128,_128>;
  using ClusterShape_MNK = Shape<_1,_2,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::Identity, ElementC, ElementAccumulator, ElementAccumulator>;
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementC, LayoutC, 16 / sizeof(ElementC),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16,
      ElementB, LayoutB, 16,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp,
      cutlass::gemm::StreamKScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>(1.0, 0.0));
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>(1.0, 1.0));
}

TEST(SM90_Device_Gemm_e4m3t_e4m3n_f32n_tensor_op_gmma_f32_cooperative_stream_k, 256x128x128_1x2x1) {
  using ElementA = cutlass::float_e4m3_t;
  using LayoutA  = cutlass::layout::RowMajor;
  using ElementB = cutlass::float_e4m3_t;
  using LayoutB  = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC  = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  using TileShape_MNK = Shape<_256,_128,_128>;
  using ClusterShape_MNK = Shape<_1,_2,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::Identity, ElementC, ElementAccumulator, ElementAccumulator>;
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementC, LayoutC, 16 / sizeof(ElementC),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16,
      ElementB, LayoutB, 16,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp,
      cutlass::gemm::StreamKScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>(1.0, 0.0));
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>(1.0, 1.0));
}

///////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Cluster 1x4x1  ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_e4m3t_e4m3n_f32n_tensor_op_gmma_f32_cooperative_stream_k, 128x128x128_1x4x1) {
  using ElementA = cutlass::float_e4m3_t;
  using LayoutA  = cutlass::layout::RowMajor;
  using ElementB = cutlass::float_e4m3_t;
  using LayoutB  = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC  = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  using TileShape_MNK = Shape<_128,_128,_128>;
  using ClusterShape_MNK = Shape<_1,_4,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::Identity, ElementC, ElementAccumulator, ElementAccumulator>;
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementC, LayoutC, 16 / sizeof(ElementC),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16,
      ElementB, LayoutB, 16,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp,
      cutlass::gemm::StreamKScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>(1.0, 0.0));
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>(1.0, 1.0));
}

TEST(SM90_Device_Gemm_e4m3t_e4m3n_f32n_tensor_op_gmma_f32_cooperative_stream_k, 256x128x128_1x4x1) {
  using ElementA = cutlass::float_e4m3_t;
  using LayoutA  = cutlass::layout::RowMajor;
  using ElementB = cutlass::float_e4m3_t;
  using LayoutB  = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC  = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  using TileShape_MNK = Shape<_256,_128,_128>;
  using ClusterShape_MNK = Shape<_1,_4,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::Identity, ElementC, ElementAccumulator, ElementAccumulator>;
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementC, LayoutC, 16 / sizeof(ElementC),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16,
      ElementB, LayoutB, 16,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp,
      cutlass::gemm::StreamKScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>(1.0, 0.0));
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>(1.0, 1.0));
}

///////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Cluster 4x1x1  ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_e4m3t_e4m3n_f32n_tensor_op_gmma_f32_cooperative_stream_k, 128x128x128_4x1x1) {
  using ElementA = cutlass::float_e4m3_t;
  using LayoutA  = cutlass::layout::RowMajor;
  using ElementB = cutlass::float_e4m3_t;
  using LayoutB  = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC  = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  using TileShape_MNK = Shape<_128,_128,_128>;
  using ClusterShape_MNK = Shape<_4,_1,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::Identity, ElementC, ElementAccumulator, ElementAccumulator>;
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementC, LayoutC, 16 / sizeof(ElementC),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16,
      ElementB, LayoutB, 16,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp,
      cutlass::gemm::StreamKScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>(1.0, 0.0));
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>(1.0, 1.0));
}

TEST(SM90_Device_Gemm_e4m3t_e4m3n_f32n_tensor_op_gmma_f32_cooperative_stream_k, 256x128x128_4x1x1) {
  using ElementA = cutlass::float_e4m3_t;
  using LayoutA  = cutlass::layout::RowMajor;
  using ElementB = cutlass::float_e4m3_t;
  using LayoutB  = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC  = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  using TileShape_MNK = Shape<_256,_128,_128>;
  using ClusterShape_MNK = Shape<_4,_1,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::Identity, ElementC, ElementAccumulator, ElementAccumulator>;
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementC, LayoutC, 16 / sizeof(ElementC),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16,
      ElementB, LayoutB, 16,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp,
      cutlass::gemm::StreamKScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>(1.0, 0.0));
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>(1.0, 1.0));
}

///////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Cluster 2x4x1  ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_e4m3t_e4m3n_f32n_tensor_op_gmma_f32_cooperative_stream_k, 128x128x128_2x4x1) {
  using ElementA = cutlass::float_e4m3_t;
  using LayoutA  = cutlass::layout::RowMajor;
  using ElementB = cutlass::float_e4m3_t;
  using LayoutB  = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC  = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  using TileShape_MNK = Shape<_128,_128,_128>;
  using ClusterShape_MNK = Shape<_2,_4,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::Identity, ElementC, ElementAccumulator, ElementAccumulator>;
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementC, LayoutC, 16 / sizeof(ElementC),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16,
      ElementB, LayoutB, 16,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp,
      cutlass::gemm::StreamKScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>(1.0, 0.0));
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>(1.0, 1.0));
}

TEST(SM90_Device_Gemm_e4m3t_e4m3n_f32n_tensor_op_gmma_f32_cooperative_stream_k, 256x128x128_2x4x1_fp8_fast_accum) {
  using ElementA = cutlass::float_e4m3_t;
  using LayoutA  = cutlass::layout::RowMajor;
  using ElementB = cutlass::float_e4m3_t;
  using LayoutB  = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC  = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  using TileShape_MNK = Shape<_256,_128,_128>;
  using ClusterShape_MNK = Shape<_2,_4,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::Identity, ElementC, ElementAccumulator, ElementAccumulator>;
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementC, LayoutC, 16 / sizeof(ElementC),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16,
      ElementB, LayoutB, 16,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp,
      cutlass::gemm::StreamKScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>(1.0, 0.0));
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>(1.0, 1.0));
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
