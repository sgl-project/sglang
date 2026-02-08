
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

#include "../../common/cutlass_unit_test.h"

#include "gemm_testbed_3x.hpp"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;
///////////////////////////////////////////////////////////////////////////////
///////////////////////////// bf16 = e4m3 * e4m3 (TN) /////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_e4m3t_e4m3n_bf16n_tensor_op_gmma_f32, 64x128x128) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::Identity, cutlass::bfloat16_t, float, float>;
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_64,_128,_128>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::bfloat16_t, LayoutC, 16 / sizeof(cutlass::bfloat16_t),
      cutlass::bfloat16_t, LayoutC, 16 / sizeof(cutlass::bfloat16_t),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::float_e4m3_t, LayoutA, 16,
      cutlass::float_e4m3_t, LayoutB, 16,
      float,
      Shape<_64,_128,_128>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////// bf16 = e4m3 * e4m3 (TN) /////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_e4m3t_e4m3n_bf16n_tensor_op_gmma_f32, 64x128x128_relu) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::ReLu, cutlass::bfloat16_t, float, float>;
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_64,_128,_128>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::bfloat16_t, LayoutC, 16 / sizeof(cutlass::bfloat16_t),
      cutlass::bfloat16_t, LayoutC, 16 / sizeof(cutlass::bfloat16_t),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::float_e4m3_t, LayoutA, 16,
      cutlass::float_e4m3_t, LayoutB, 16,
      float,
      Shape<_64,_128,_128>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>());
}


///////////////////////////////////////////////////////////////////////////////
///////////////////////////// bf16 = e4m3 * e4m3 (TN) /////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_e4m3t_e4m3n_bf16n_tensor_op_gmma_f32, 64x128x128_bias_bf16_relu) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::ReLu, cutlass::bfloat16_t, float, cutlass::bfloat16_t>;
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_64,_128,_128>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::bfloat16_t, LayoutC, 16 / sizeof(cutlass::bfloat16_t),
      cutlass::bfloat16_t, LayoutC, 16 / sizeof(cutlass::bfloat16_t),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::float_e4m3_t, LayoutA, 16,
      cutlass::float_e4m3_t, LayoutB, 16,
      float,
      Shape<_64,_128,_128>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////// bf16 = e5m2 * e4m3 (TN) /////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_e5m2t_e4m3n_bf16n_tensor_op_gmma_f32, 64x128x128) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::Identity, cutlass::bfloat16_t, float, float>;
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_64,_128,_128>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::bfloat16_t, LayoutC, 16 / sizeof(cutlass::bfloat16_t),
      cutlass::bfloat16_t, LayoutC, 16 / sizeof(cutlass::bfloat16_t),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::float_e5m2_t, LayoutA, 16,
      cutlass::float_e4m3_t, LayoutB, 16,
      float,
      Shape<_64,_128,_128>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////// bf16 = e4m3 * e5m2 (TN) /////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_e4m3t_e5m2n_bf16n_tensor_op_gmma_f32, 64x128x128) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::Identity, cutlass::bfloat16_t, float, float>;
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_64,_128,_128>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::bfloat16_t, LayoutC, 16 / sizeof(cutlass::bfloat16_t),
      cutlass::bfloat16_t, LayoutC, 16 / sizeof(cutlass::bfloat16_t),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::float_e4m3_t, LayoutA, 16,
      cutlass::float_e5m2_t, LayoutB, 16,
      float,
      Shape<_64,_128,_128>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>());
}


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Cluster 2x2x1  //////////////////////////////////
///////////////////////////// bf16 = e4m3 * e4m3 (TN) /////////////////////////
///////////////////////////////////////////////////////////////////////////////
TEST(SM90_Device_Gemm_e4m3t_e4m3n_bf16n_tensor_op_gmma_f32, 64x128x128_2x2x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::Identity, cutlass::bfloat16_t, float, float>;
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_64,_128,_128>, Shape<_2,_2,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::bfloat16_t, LayoutC, 16 / sizeof(cutlass::bfloat16_t),
      cutlass::bfloat16_t, LayoutC, 16 / sizeof(cutlass::bfloat16_t),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::float_e4m3_t, LayoutA, 16,
      cutlass::float_e4m3_t, LayoutB, 16,
      float,
      Shape<_64,_128,_128>, Shape<_2,_2,_1>,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Cluster 1x4x1  //////////////////////////////////
///////////////////////////// bf16 = e4m3 * e4m3 (TN) /////////////////////////
///////////////////////////////////////////////////////////////////////////////
TEST(SM90_Device_Gemm_e4m3t_e4m3n_bf16n_tensor_op_gmma_f32, 64x128x128_1x4x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::Identity, cutlass::bfloat16_t, float, float>;
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_64,_128,_128>, Shape<_1,_4,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::bfloat16_t, LayoutC, 16 / sizeof(cutlass::bfloat16_t),
      cutlass::bfloat16_t, LayoutC, 16 / sizeof(cutlass::bfloat16_t),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::float_e4m3_t, LayoutA, 16,
      cutlass::float_e4m3_t, LayoutB, 16,
      float,
      Shape<_64,_128,_128>, Shape<_1,_4,_1>,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>());
}


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Cluster 4x1x1  //////////////////////////////////
///////////////////////////// bf16 = e4m3 * e4m3 (TN) /////////////////////////
///////////////////////////////////////////////////////////////////////////////
TEST(SM90_Device_Gemm_e4m3t_e4m3n_bf16n_tensor_op_gmma_f32, 64x128x128_4x1x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::Identity, cutlass::bfloat16_t, float, float>;
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_64,_128,_128>, Shape<_4,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::bfloat16_t, LayoutC, 16 / sizeof(cutlass::bfloat16_t),
      cutlass::bfloat16_t, LayoutC, 16 / sizeof(cutlass::bfloat16_t),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::float_e4m3_t, LayoutA, 16,
      cutlass::float_e4m3_t, LayoutB, 16,
      float,
      Shape<_64,_128,_128>, Shape<_4,_1,_1>,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Cluster 2x4x1  //////////////////////////////////
///////////////////////////// bf16 = e4m3 * e4m3 (TN) /////////////////////////
///////////////////////////////////////////////////////////////////////////////
TEST(SM90_Device_Gemm_e4m3t_e4m3n_bf16n_tensor_op_gmma_f32, 64x128x128_2x4x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::Identity, cutlass::bfloat16_t, float, float>;
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_64,_128,_128>, Shape<_2,_4,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::bfloat16_t, LayoutC, 16 / sizeof(cutlass::bfloat16_t),
      cutlass::bfloat16_t, LayoutC, 16 / sizeof(cutlass::bfloat16_t),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::float_e4m3_t, LayoutA, 16,
      cutlass::float_e4m3_t, LayoutB, 16,
      float,
      Shape<_64,_128,_128>, Shape<_2,_4,_1>,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////
//////////////////////////////// TMA epilogue /////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_e4m3t_e4m3n_bf16n_tensor_op_gmma_f32, 64x128x128_tma_epilogue) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_64,_128,_128>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::bfloat16_t, LayoutC, 8,
      cutlass::bfloat16_t, LayoutC, 8,
      cutlass::epilogue::TmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::float_e4m3_t, LayoutA, 16,
      cutlass::float_e4m3_t, LayoutB, 16,
      float,
      Shape<_64,_128,_128>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;


  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_e4m3t_e4m3n_bf16t_tensor_op_gmma_f32, 64x128x128_tma_epilogue_fp8_fast_accum) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_64,_128,_128>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::bfloat16_t, LayoutC, 8,
      cutlass::bfloat16_t, LayoutC, 8,
      cutlass::epilogue::TmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::float_e4m3_t, LayoutA, 16,
      cutlass::float_e4m3_t, LayoutB, 16,
      float,
      Shape<_64,_128,_128>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum
    >::CollectiveOp;


  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
