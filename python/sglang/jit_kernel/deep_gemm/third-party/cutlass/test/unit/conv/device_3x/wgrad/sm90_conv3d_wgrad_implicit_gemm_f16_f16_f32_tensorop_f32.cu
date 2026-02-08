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

#include "cutlass_unit_test.h"

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/conv/device/conv_universal_adapter.hpp"
#include "cutlass/conv/kernel/conv_universal.hpp"
#include "cutlass/conv/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "../testbed_conv.hpp"
using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

//////////////////////////////////////////////////////////////////////////////////////////////////
// Tile shape 64x64x64
//////////////////////////////////////////////////////////////////////////////////////////////////

//
// Cluster 1x1x1
//

TEST(SM90_device_conv3d_wgrad_implicitgemm_f16ndhwc_f16ndhwc_f32ndhwc_tensor_op_f32, 64x64x64_1x1x1) {
  using ElementAct     = cutlass::half_t;
  using ElementFlt     = cutlass::half_t;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using TileShapeMNK = Shape<_64, Shape<_64>, Shape<_64>>;
  using ClusterShapeMNK = Shape<_1,_1,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      cutlass::half_t, cutlass::layout::TensorKCSRT, 8,
      cutlass::half_t, cutlass::layout::TensorKCSRT, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kWgrad,
      ElementAct, cutlass::layout::TensorNDHWC, 8,
      ElementFlt, cutlass::layout::TensorNDHWC, 8,
      ElementAcc,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::conv::collective::KernelScheduleAuto
    >::CollectiveOp;

  using ProblemShape=cutlass::conv::ConvProblemShape<CollectiveMainloop::DispatchPolicy::ConvOp, CollectiveMainloop::DispatchPolicy::NumSpatialDimensions>;
  using ConvKernel = cutlass::conv::kernel::ConvUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue
    >;

  using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;

  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>());
}

//
// Cluster 2x1x1
//

TEST(SM90_device_conv3d_wgrad_implicitgemm_f16ndhwc_f16ndhwc_f32ndhwc_tensor_op_f32, 64x64x64_2x1x1) {
  using ElementAct     = cutlass::half_t;
  using ElementFlt     = cutlass::half_t;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using TileShapeMNK = Shape<_64, Shape<_64>, Shape<_64>>;
  using ClusterShapeMNK = Shape<_2,_1,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      cutlass::half_t, cutlass::layout::TensorKCSRT, 8,
      cutlass::half_t, cutlass::layout::TensorKCSRT, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kWgrad,
      ElementAct, cutlass::layout::TensorNDHWC, 8,
      ElementFlt, cutlass::layout::TensorNDHWC, 8,
      ElementAcc,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::conv::collective::KernelScheduleAuto
    >::CollectiveOp;

  using ProblemShape=cutlass::conv::ConvProblemShape<CollectiveMainloop::DispatchPolicy::ConvOp, CollectiveMainloop::DispatchPolicy::NumSpatialDimensions>;
  using ConvKernel = cutlass::conv::kernel::ConvUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue
    >;

  using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;

  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>());
}

//
// Cluster 1x2x1
//

TEST(SM90_device_conv3d_wgrad_implicitgemm_f16ndhwc_f16ndhwc_f32ndhwc_tensor_op_f32, 64x64x64_1x2x1) {
  using ElementAct     = cutlass::half_t;
  using ElementFlt     = cutlass::half_t;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using TileShapeMNK = Shape<_64, Shape<_64>, Shape<_64>>;
  using ClusterShapeMNK = Shape<_1,_2,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      cutlass::half_t, cutlass::layout::TensorKCSRT, 8,
      cutlass::half_t, cutlass::layout::TensorKCSRT, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kWgrad,
      ElementAct, cutlass::layout::TensorNDHWC, 8,
      ElementFlt, cutlass::layout::TensorNDHWC, 8,
      ElementAcc,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::conv::collective::KernelScheduleAuto
    >::CollectiveOp;

  using ProblemShape=cutlass::conv::ConvProblemShape<CollectiveMainloop::DispatchPolicy::ConvOp, CollectiveMainloop::DispatchPolicy::NumSpatialDimensions>;
  using ConvKernel = cutlass::conv::kernel::ConvUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue
    >;

  using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;

  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>());
}

//
// Cluster 2x2x1
//

TEST(SM90_device_conv3d_wgrad_implicitgemm_f16ndhwc_f16ndhwc_f32ndhwc_tensor_op_f32, 64x64x64_2x2x1) {
  using ElementAct     = cutlass::half_t;
  using ElementFlt     = cutlass::half_t;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using TileShapeMNK = Shape<_64, Shape<_64>, Shape<_64>>;
  using ClusterShapeMNK = Shape<_2,_2,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      cutlass::half_t, cutlass::layout::TensorKCSRT, 8,
      cutlass::half_t, cutlass::layout::TensorKCSRT, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kWgrad,
      ElementAct, cutlass::layout::TensorNDHWC, 8,
      ElementFlt, cutlass::layout::TensorNDHWC, 8,
      ElementAcc,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::conv::collective::KernelScheduleAuto
    >::CollectiveOp;

  using ProblemShape=cutlass::conv::ConvProblemShape<CollectiveMainloop::DispatchPolicy::ConvOp, CollectiveMainloop::DispatchPolicy::NumSpatialDimensions>;
  using ConvKernel = cutlass::conv::kernel::ConvUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue
    >;

  using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;

  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>());
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Tile shape 128x64x64
//////////////////////////////////////////////////////////////////////////////////////////////////

//
// Cluster 1x1x1
//

TEST(SM90_device_conv3d_wgrad_implicitgemm_f16ndhwc_f16ndhwc_f32ndhwc_tensor_op_f32, 128x64x64_1x1x1) {
  using ElementAct     = cutlass::half_t;
  using ElementFlt     = cutlass::half_t;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using TileShapeMNK = Shape<_128, Shape<_64>, Shape<_64>>;
  using ClusterShapeMNK = Shape<_1,_1,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      cutlass::half_t, cutlass::layout::TensorKCSRT, 8,
      cutlass::half_t, cutlass::layout::TensorKCSRT, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kWgrad,
      ElementAct, cutlass::layout::TensorNDHWC, 8,
      ElementFlt, cutlass::layout::TensorNDHWC, 8,
      ElementAcc,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::conv::collective::KernelScheduleAuto
    >::CollectiveOp;

  using ProblemShape=cutlass::conv::ConvProblemShape<CollectiveMainloop::DispatchPolicy::ConvOp, CollectiveMainloop::DispatchPolicy::NumSpatialDimensions>;
  using ConvKernel = cutlass::conv::kernel::ConvUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue
    >;

  using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;

  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>());
}

//
// Cluster 2x1x1
//

TEST(SM90_device_conv3d_wgrad_implicitgemm_f16ndhwc_f16ndhwc_f32ndhwc_tensor_op_f32, 128x64x64_2x1x1) {
  using ElementAct     = cutlass::half_t;
  using ElementFlt     = cutlass::half_t;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using TileShapeMNK = Shape<_128, Shape<_64>, Shape<_64>>;
  using ClusterShapeMNK = Shape<_2,_1,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      cutlass::half_t, cutlass::layout::TensorKCSRT, 8,
      cutlass::half_t, cutlass::layout::TensorKCSRT, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kWgrad,
      ElementAct, cutlass::layout::TensorNDHWC, 8,
      ElementFlt, cutlass::layout::TensorNDHWC, 8,
      ElementAcc,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::conv::collective::KernelScheduleAuto
    >::CollectiveOp;

  using ProblemShape=cutlass::conv::ConvProblemShape<CollectiveMainloop::DispatchPolicy::ConvOp, CollectiveMainloop::DispatchPolicy::NumSpatialDimensions>;
  using ConvKernel = cutlass::conv::kernel::ConvUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue
    >;

  using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;

  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>());
}

//
// Cluster 1x2x1
//

TEST(SM90_device_conv3d_wgrad_implicitgemm_f16ndhwc_f16ndhwc_f32ndhwc_tensor_op_f32, 128x64x64_1x2x1) {
  using ElementAct     = cutlass::half_t;
  using ElementFlt     = cutlass::half_t;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using TileShapeMNK = Shape<_128, Shape<_64>, Shape<_64>>;
  using ClusterShapeMNK = Shape<_1,_2,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      cutlass::half_t, cutlass::layout::TensorKCSRT, 8,
      cutlass::half_t, cutlass::layout::TensorKCSRT, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kWgrad,
      ElementAct, cutlass::layout::TensorNDHWC, 8,
      ElementFlt, cutlass::layout::TensorNDHWC, 8,
      ElementAcc,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::conv::collective::KernelScheduleAuto
    >::CollectiveOp;

  using ProblemShape=cutlass::conv::ConvProblemShape<CollectiveMainloop::DispatchPolicy::ConvOp, CollectiveMainloop::DispatchPolicy::NumSpatialDimensions>;
  using ConvKernel = cutlass::conv::kernel::ConvUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue
    >;

  using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;

  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>());
}

//
// Cluster 2x2x1
//

TEST(SM90_device_conv3d_wgrad_implicitgemm_f16ndhwc_f16ndhwc_f32ndhwc_tensor_op_f32, 128x64x64_2x2x1) {
  using ElementAct     = cutlass::half_t;
  using ElementFlt     = cutlass::half_t;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using TileShapeMNK = Shape<_128, Shape<_64>, Shape<_64>>;
  using ClusterShapeMNK = Shape<_2,_2,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      cutlass::half_t, cutlass::layout::TensorKCSRT, 8,
      cutlass::half_t, cutlass::layout::TensorKCSRT, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kWgrad,
      ElementAct, cutlass::layout::TensorNDHWC, 8,
      ElementFlt, cutlass::layout::TensorNDHWC, 8,
      ElementAcc,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::conv::collective::KernelScheduleAuto
    >::CollectiveOp;

  using ProblemShape=cutlass::conv::ConvProblemShape<CollectiveMainloop::DispatchPolicy::ConvOp, CollectiveMainloop::DispatchPolicy::NumSpatialDimensions>;
  using ConvKernel = cutlass::conv::kernel::ConvUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue
    >;

  using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;

  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>());
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
