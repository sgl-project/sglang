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


/*! \file
    \brief Tests for device-wide grouped GEMM interface
*/

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
#include "cutlass/epilogue/thread/activation.h"

#include "../../../common/cutlass_unit_test.h"
#include "../gemm_testbed_3x_ptr_array.hpp"


using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)

// Pingpong kernel schedule
TEST(SM120_Device_Gemm_e5m2t_e2m1n_e2m1t_tensorop_f32_epilogue_VS32_group_pingpong, row_sf) {
  using ElementInputA = float_e5m2_t;
  using ElementInputB = float_e2m1_t;
  using ElementA = cutlass::mx_float8_t<ElementInputA>;
  using ElementB = cutlass::mx_float4_t<ElementInputB>;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::float_e2m1_t;
  using ElementCompute = float;
  using ElementAccumulator = float;
  using ElementSF = cutlass::float_ue8m0_t;
  using ElementSFD  = ElementSF;
  using ElementAccumulator = float;
  using GmemLayoutA = cutlass::layout::RowMajor;
  using GmemLayoutB = cutlass::layout::ColumnMajor;
  using GmemLayoutC = cutlass::layout::RowMajor;
  constexpr int SFVectorSize = 32;
  using TileShape_MNK = Shape<_128,_128,_128>;
  using ClusterShape_MNK = Shape<_1,_1,_1>;

  constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementInputA>::value;
  constexpr int AlignmentB  = 128;
  constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;
  constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;  

  //
  // Construct CollectiveEpilogue
  //

  constexpr int OutputSFVectorSize = SFVectorSize;
  // D = alpha * acc + beta * C
  // With Row-major BlockScaleFactor generation.
  using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
      OutputSFVectorSize,
      ElementD, 
      ElementCompute, 
      ElementSFD, GmemLayoutC,
      ElementC>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, GmemLayoutC *, AlignmentC,
      ElementD, GmemLayoutC *, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      FusionOperation
    >::CollectiveOp;

  //
  // Construct CollectiveMainloop
  //
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementA, GmemLayoutA *, AlignmentA,
      ElementB, GmemLayoutB *, AlignmentB,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cutlass::gemm::GroupProblemShape<Shape<int,int,int>>,
      CollectiveMainloop,
      CollectiveEpilogue
    >;
  
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = test::gemm::device::TestSmallFusion<Gemm>(1.0, 0.5);
  EXPECT_TRUE(pass);
}



TEST(SM120_Device_Gemm_e5m2t_e2m1n_e2m1t_tensorop_f32_epilogue_VS32_group_pingpong, silu_row_sf) {
  using ElementInputA = float_e5m2_t;
  using ElementInputB = float_e2m1_t;
  using ElementA = cutlass::mx_float8_t<ElementInputA>;
  using ElementB = cutlass::mx_float4_t<ElementInputB>;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::float_e2m1_t;
  using ElementCompute = float;
  using ElementAccumulator = float;
  using ElementSF = cutlass::float_ue4m3_t;
  using ElementSFD  = ElementSF;
  using ElementAccumulator = float;
  using GmemLayoutA = cutlass::layout::RowMajor;
  using GmemLayoutB = cutlass::layout::ColumnMajor;
  using GmemLayoutC = cutlass::layout::RowMajor;
  constexpr int SFVectorSize = 32;
  using TileShape_MNK = Shape<_128,_128,_128>;
  using ClusterShape_MNK = Shape<_1,_1,_1>;

  constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementInputA>::value;
  constexpr int AlignmentB  = 128;
  constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;
  constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;  

  //
  // Construct CollectiveEpilogue
  //

  constexpr int OutputSFVectorSize = SFVectorSize;
  // D = SiLu(alpha * acc + beta * C)
  // With Row-major BlockScaleFactor generation.
  using FusionOperation = cutlass::epilogue::fusion::LinCombEltActBlockScaleFactor<
      cutlass::epilogue::thread::SiLu,
      OutputSFVectorSize,
      ElementD, 
      ElementCompute, 
      ElementSFD, GmemLayoutC,
      ElementC>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, GmemLayoutC *, AlignmentC,
      ElementD, GmemLayoutC *, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      FusionOperation
    >::CollectiveOp;

  //
  // Construct CollectiveMainloop
  //
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementA, GmemLayoutA *, AlignmentA,
      ElementB, GmemLayoutB *, AlignmentB,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cutlass::gemm::GroupProblemShape<Shape<int,int,int>>,
      CollectiveMainloop,
      CollectiveEpilogue
    >;
  
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = test::gemm::device::TestSmallFusion<Gemm>(1.0, 0.5);
  EXPECT_TRUE(pass);
}


// Cooperative kenel schedule
TEST(SM120_Device_Gemm_e5m2t_e2m1n_e2m1t_tensorop_f32_epilogue_VS32_group_cooperative, row_sf) {
  using ElementInputA = float_e5m2_t;
  using ElementInputB = float_e2m1_t;
  using ElementA = cutlass::mx_float8_t<ElementInputA>;
  using ElementB = cutlass::mx_float4_t<ElementInputB>;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::float_e2m1_t;
  using ElementCompute = float;
  using ElementAccumulator = float;
  using ElementSF = cutlass::float_ue4m3_t;
  using ElementSFD  = ElementSF;
  using ElementAccumulator = float;
  using GmemLayoutA = cutlass::layout::RowMajor;
  using GmemLayoutB = cutlass::layout::ColumnMajor;
  using GmemLayoutC = cutlass::layout::RowMajor;
  constexpr int SFVectorSize = 32;
  using TileShape_MNK = Shape<_128,_128,_128>;
  using ClusterShape_MNK = Shape<_1,_1,_1>;

  constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementInputA>::value;
  constexpr int AlignmentB  = 128;
  constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;
  constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;  

  //
  // Construct CollectiveEpilogue
  //

  constexpr int OutputSFVectorSize = SFVectorSize;
  // D = alpha * acc + beta * C
  // With Row-major BlockScaleFactor generation.
  using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
      OutputSFVectorSize,
      ElementD, 
      ElementCompute, 
      ElementSFD, GmemLayoutC,
      ElementC>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, GmemLayoutC *, AlignmentC,
      ElementD, GmemLayoutC *, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      FusionOperation
    >::CollectiveOp;

  //
  // Construct CollectiveMainloop
  //
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementA, GmemLayoutA *, AlignmentA,
      ElementB, GmemLayoutB *, AlignmentB,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cutlass::gemm::GroupProblemShape<Shape<int,int,int>>,
      CollectiveMainloop,
      CollectiveEpilogue
    >;
  
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = test::gemm::device::TestSmallFusion<Gemm>(1.0, 0.5);
  EXPECT_TRUE(pass);
}



TEST(SM120_Device_Gemm_e5m2t_e2m1n_e2m1t_tensorop_f32_epilogue_VS32_group_cooperative, silu_row_sf) {
  using ElementInputA = float_e5m2_t;
  using ElementInputB = float_e2m1_t;
  using ElementA = cutlass::mx_float8_t<ElementInputA>;
  using ElementB = cutlass::mx_float4_t<ElementInputB>;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::float_e2m1_t;
  using ElementCompute = float;
  using ElementAccumulator = float;
  using ElementSF = cutlass::float_ue4m3_t;
  using ElementSFD  = ElementSF;
  using ElementAccumulator = float;
  using GmemLayoutA = cutlass::layout::RowMajor;
  using GmemLayoutB = cutlass::layout::ColumnMajor;
  using GmemLayoutC = cutlass::layout::RowMajor;
  constexpr int SFVectorSize = 32;
  using TileShape_MNK = Shape<_128,_128,_128>;
  using ClusterShape_MNK = Shape<_1,_1,_1>;

  constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementInputA>::value;
  constexpr int AlignmentB  = 128;
  constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;
  constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;  

  //
  // Construct CollectiveEpilogue
  //

  constexpr int OutputSFVectorSize = SFVectorSize;
  // D = SiLu(alpha * acc + beta * C)
  // With Row-major BlockScaleFactor generation.
  using FusionOperation = cutlass::epilogue::fusion::LinCombEltActBlockScaleFactor<
      cutlass::epilogue::thread::SiLu,
      OutputSFVectorSize,
      ElementD, 
      ElementCompute, 
      ElementSFD, GmemLayoutC,
      ElementC>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, GmemLayoutC *, AlignmentC,
      ElementD, GmemLayoutC *, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      FusionOperation
    >::CollectiveOp;

  //
  // Construct CollectiveMainloop
  //
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementA, GmemLayoutA *, AlignmentA,
      ElementB, GmemLayoutB *, AlignmentB,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cutlass::gemm::GroupProblemShape<Shape<int,int,int>>,
      CollectiveMainloop,
      CollectiveEpilogue
    >;
  
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = test::gemm::device::TestSmallFusion<Gemm>(1.0, 0.5);
  EXPECT_TRUE(pass);
}
#endif // #if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
