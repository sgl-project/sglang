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

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "../../../common/cutlass_unit_test.h"

#include "../gemm_testbed_3x.hpp"

#if (defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED))

using namespace cute;

///////////////////////////////////////////////////////////////////////////////

namespace kernel_1 {

  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  using LayoutSFDTag = cutlass::layout::RowMajor;

  using ElementA =  cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementB =  cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementC = cutlass::bfloat16_t;
  using ElementD = cutlass::float_e4m3_t;
  using ElementBias = cutlass::bfloat16_t;
  using ElementSF = cutlass::float_ue8m0_t;
  using ElementAccumulator = float;
  using ElementCompute = float;

  constexpr int kAlignmentA = 32;
  constexpr int kAlignmentB = 16;
  constexpr int kAlignmentC = 1;
  constexpr int kAlignmentD = 4;

  using ProblemShape = Shape<int,int,int,int>;
  using ClusterShape = Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
  using MainloopTileShape = Shape<cute::Int<128>, cute::Int<128>, cute::Int<256>>;
  using EpilogueTileShape = Shape<cute::Int<128>, cute::Int<128>, cute::Int<256>>;
  using ArchTag = cutlass::arch::Sm120;
  using OpClassEpilogue = cutlass::arch::OpClassTensorOp;
  using OpClassMainLoop = cutlass::arch::OpClassBlockScaledSparseTensorOp;
  using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
  using EpilogueScheduleType = cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120;
  using KernelScheduleType =  cutlass::gemm::KernelSparseTmaWarpSpecializedMxf8f6f4Acc2x4Sm120;
  using ElementAccumulator = float;
  using ElementEpilogueCompute = float;
  using ElementBias = cutlass::bfloat16_t;
  using TileScheduler = void;
  static constexpr int SFVectorSize = 64;

  using FusionOperation = cutlass::epilogue::fusion::LinCombPerColBiasEltActBlockScaleFactor<
      cutlass::epilogue::thread::Clamp,
      SFVectorSize,
      ElementD,
      ElementCompute,
      ElementSF,
      LayoutSFDTag,
      ElementBias,
      ElementC>;

  using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OpClassEpilogue,
      EpilogueTileShape,
      ClusterShape,
      EpilogueTile,
      ElementAccumulator,
      ElementEpilogueCompute,
      ElementC, LayoutCTag, kAlignmentC,
      ElementD, LayoutDTag, kAlignmentD,
      EpilogueScheduleType
      , FusionOperation
    >::CollectiveOp;

  using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

  using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OpClassMainLoop,
      ElementA, LayoutATag, kAlignmentA,
      ElementB, LayoutBTag, kAlignmentB,
      ElementAccumulator,
      MainloopTileShape,
      ClusterShape,
      StageCount,
      KernelScheduleType
    >::CollectiveOp;

  template <typename T>
  struct dummy {
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        TileScheduler
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  };
  using GemmKernel = typename dummy<void>::GemmKernel;
  using Gemm = typename dummy<void>::Gemm;

} // kernel_1

TEST(SM120_Device_Sparse_Gemm_fe4m3t_fe4m3n_fe4m3t_tensor_op_f32, 128x128x256) {
  bool result = test::gemm::device::TestSmall<kernel_1::Gemm, false /*force_legacy_epilogue*/, false /*apply_alignment_offset*/>(1.0, 0.5);
  EXPECT_TRUE(result);
}

#endif // (defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED))
