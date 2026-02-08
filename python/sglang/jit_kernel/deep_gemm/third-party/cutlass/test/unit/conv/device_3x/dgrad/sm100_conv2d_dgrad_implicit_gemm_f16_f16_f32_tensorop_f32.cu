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
    \brief Tests for device-wide CONV interface
*/

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

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

//////////////////////////////////////////////////////////////////////////////////////////////////
// Static cluster
//////////////////////////////////////////////////////////////////////////////////////////////////

//
// Cluster tile shape 64x64x64
// Cluster shape 1x1x1
//
TEST(SM100_device_conv2d_dgrad_implicitgemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32, 64x64x64_1x1x1) {
  using ElementAct     = cutlass::half_t;
  using ElementFlt     = cutlass::half_t;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using MmaTileShape = Shape<_64, _64, Shape<_64>>;
  using ClusterShape = Shape<_1,_1,_1>;
 
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementAct, cutlass::layout::TensorNHWC, 128 / cutlass::sizeof_bits<ElementAct>::value,
      ElementOut, cutlass::layout::TensorNHWC, 128 /  cutlass::sizeof_bits<ElementOut>::value,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kDgrad,
      ElementAct, cutlass::layout::TensorNHWC, 8,
      ElementFlt, cutlass::layout::TensorNHWC, 8,
      ElementAcc,
      MmaTileShape, ClusterShape,
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
// Cluster tile shape 128x64x64
// Cluster shape 1x1x1
//
TEST(SM100_device_conv2d_dgrad_implicitgemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32, 128x64x64_1x1x1) {
  using ElementAct     = cutlass::half_t;
  using ElementFlt     = cutlass::half_t;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using MmaTileShape = Shape<_128, _64, Shape<_64>>;
  using ClusterShape = Shape<_1,_1,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementAct, cutlass::layout::TensorNHWC, 128 / cutlass::sizeof_bits<ElementAct>::value,
      ElementOut, cutlass::layout::TensorNHWC, 128 /  cutlass::sizeof_bits<ElementOut>::value,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kDgrad,
      ElementAct, cutlass::layout::TensorNHWC, 8,
      ElementFlt, cutlass::layout::TensorNHWC, 8,
      ElementAcc,
      MmaTileShape, ClusterShape,
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
// Cluster tile shape 128x128x64
// Cluster shape 1x2x1
//
TEST(SM100_device_conv2d_dgrad_implicitgemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32, 128x128x64_1x2x1) {
  using ElementAct     = cutlass::half_t;
  using ElementFlt     = cutlass::half_t;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using MmaTileShape = Shape<_128, _64, Shape<_64>>;
  using ClusterShape = Shape<_1,_2,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementAct, cutlass::layout::TensorNHWC, 128 / cutlass::sizeof_bits<ElementAct>::value,
      ElementOut, cutlass::layout::TensorNHWC, 128 /  cutlass::sizeof_bits<ElementOut>::value,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kDgrad,
      ElementAct, cutlass::layout::TensorNHWC, 8,
      ElementFlt, cutlass::layout::TensorNHWC, 8,
      ElementAcc,
      MmaTileShape, ClusterShape,
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
// Cluster tile shape 256x64x64
// Cluster shape 2x1x1
//
TEST(SM100_device_conv2d_dgrad_implicitgemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32, 256x64x64_2x1x1) {
  using ElementAct     = cutlass::half_t;
  using ElementFlt     = cutlass::half_t;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using MmaTileShape = Shape<_256, _64, Shape<_64>>;
  using ClusterShape = Shape<_2,_1,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementAct, cutlass::layout::TensorNHWC, 128 / cutlass::sizeof_bits<ElementAct>::value,
      ElementOut, cutlass::layout::TensorNHWC, 128 /  cutlass::sizeof_bits<ElementOut>::value,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kDgrad,
      ElementAct, cutlass::layout::TensorNHWC, 8,
      ElementFlt, cutlass::layout::TensorNHWC, 8,
      ElementAcc,
      MmaTileShape, ClusterShape,
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
// Cluster tile shape 256x128x64
// Cluster shape 2x2x1
//
TEST(SM100_device_conv2d_dgrad_implicitgemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32, 256x128x64_2x2x1) {
  using ElementAct     = cutlass::half_t;
  using ElementFlt     = cutlass::half_t;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using MmaTileShape = Shape<_256, _64, Shape<_64>>;
  using ClusterShape = Shape<_2,_2,_1>;
 
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementAct, cutlass::layout::TensorNHWC, 128 / cutlass::sizeof_bits<ElementAct>::value,
      ElementOut, cutlass::layout::TensorNHWC, 128 /  cutlass::sizeof_bits<ElementOut>::value,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kDgrad,
      ElementAct, cutlass::layout::TensorNHWC, 8,
      ElementFlt, cutlass::layout::TensorNHWC, 8,
      ElementAcc,
      MmaTileShape, ClusterShape,
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
// Dynamic cluster
//////////////////////////////////////////////////////////////////////////////////////////////////

//
// CTA tile shape 64x64x64
// preferred cluster shape 2x4x1
// fallback cluster shape  2x2x1
//
TEST(SM100_device_conv2d_dgrad_implicitgemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32, 64x64x64_preferred_2x4x1_fallback_2x2x1) {
  using ElementAct     = cutlass::half_t;
  using ElementFlt     = cutlass::half_t;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using MmaTileShape = Shape<_64, _64, Shape<_64>>;
  using ClusterShape = decltype(make_shape(int(0), int(0), Int<1>{}));

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementAct, cutlass::layout::TensorNHWC, 128 / cutlass::sizeof_bits<ElementAct>::value,
      ElementOut, cutlass::layout::TensorNHWC, 128 /  cutlass::sizeof_bits<ElementOut>::value,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kDgrad,
      ElementAct, cutlass::layout::TensorNHWC, 8,
      ElementFlt, cutlass::layout::TensorNHWC, 8,
      ElementAcc,
      MmaTileShape, ClusterShape,
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

  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>(1.0, 0.0, 0.0f, dim3(2,4,1), dim3(2,2,1)));
}

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
