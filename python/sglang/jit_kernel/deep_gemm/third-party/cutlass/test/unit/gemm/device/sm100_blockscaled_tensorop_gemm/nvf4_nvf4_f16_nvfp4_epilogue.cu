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
    \brief Unit test for nvfp4 Block Scaled Gemm with nvfp4 output
      D tensor: 
        * Types: e2m1x{ue4m3}
        * Layout: Column Major (T)
        * Alignment: 32
      * Scale factors need to be generated with the fp4 output.  It is generated along the continuous dimensions of the D tensor.
      * Meanwhile, before scale factor generation, it could have other epilogue fusion operation.
        * alpha
        * beta
        * activation
        * bias
        This UT tests 
        - alpha + beta + scale-factor generation
        - alpha + beta + bias + scale-factor generation
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/epilogue/thread/activation.h"
#include "../../../common/cutlass_unit_test.h"

#include "../gemm_testbed_3x.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

//////////////////////////////////////////////////////////////////////////////
// FusionOperation: k-major output and datatype is float_e2m1_t with float_ue4m3_t scale-factor (vecsize 16)
//                  with alpha/beta fusion
//////////////////////////////////////////////////////////////////////////////
TEST(SM100Only_Device_Gemm_ue8m0xe2m1t_ue8m0xe2m1n_ue8m0xe2m1t_outputVs16_bstensorop_1sm_f32, 128x128x256_4x4x1) {
  // Describe A and B tensors
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  constexpr int AlignA = 32;
  using GmemLayoutA = cutlass::layout::RowMajor;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  constexpr int AlignB = 32;
  using GmemLayoutB = cutlass::layout::ColumnMajor;
  // Describe C and D tensors
  using ElementC = cutlass::half_t;
  constexpr int AlignC = 8;
  using GmemLayoutC = cutlass::layout::RowMajor;
  using ElementD = cutlass::float_e2m1_t;
  constexpr int AlignD = 32;
  using GmemLayoutD = cutlass::layout::RowMajor;
  // Describe SFD tensor
  using ElementSFD = cutlass::float_ue4m3_t;
  using GmemLayoutSFD = GmemLayoutD;
  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;

  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_128,_128,_256>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_4,_1>;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;

  //
  // Construct FusionOperation
  //
  constexpr int SFDVectorSize = 16;
  // Define the fusion operation applied during epilogue
  using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
      SFDVectorSize,
      ElementD, ElementCompute, 
      ElementSFD, GmemLayoutSFD,
      ElementC
    >;
  
  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,      // Arch and Tensorop spec
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      cutlass::epilogue::collective::EpilogueTileAuto,                      // Epilogue subtile shape. Auto will find a suitable tile shape
      ElementAccumulator, ElementCompute,                                   // Mma instr's accumulator type and compute precision for epilogue
      ElementC, GmemLayoutC, AlignC,                                        // C tensor description
      ElementD, GmemLayoutD, AlignD,                                        // D tensor description
      cutlass::epilogue::TmaWarpSpecialized1Sm                              // Epilogue schedule policy
    >::CollectiveOp;

  //
  // Construct CollectiveMainloop
  //

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,      // Arch and Tensorop spec
      ElementA, GmemLayoutA, AlignA,                                        // A tensor elem type, layout and alignment requirement
      ElementB, GmemLayoutB, AlignB,                                        // B tensor elem type, layout and alignment requirement
      ElementAccumulator,                                                   // Mma instruction accumulator type
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      // Epilogue's SMEM usage that needs to be subtracted from overall SMEM capacity 
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized1SmBlockScaledSm100
    >::CollectiveOp;

  // Create Gemm Kernel using CollectiveEpilogue and CollectiveMainloop created by the builders
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
    >;
  
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  // Run tests
  auto pass = test::gemm::device::TestAll<Gemm>();
  // Check results
  EXPECT_TRUE(pass); 
}

//////////////////////////////////////////////////////////////////////////////

TEST(SM100Only_Device_Gemm_ue4m3xe2m1t_ue4m3xe2m1n_ue4m3xe2m1t_outputVs16_bstensorop_2sm_f32, 256x128x256_4x4x1) {
  // Describe A and B tensors
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  constexpr int AlignA = 32;
  using GmemLayoutA = cutlass::layout::RowMajor;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  constexpr int AlignB = 32;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = cutlass::half_t;
  constexpr int AlignC = 8;
  using GmemLayoutC = cutlass::layout::RowMajor;
  using ElementD = cutlass::float_e2m1_t;
  constexpr int AlignD = 32;
  using GmemLayoutD = cutlass::layout::RowMajor;

  // Describe SFD tensor
  using ElementSFD = cutlass::float_ue4m3_t;
  using GmemLayoutSFD = GmemLayoutD;

  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_256,_128,_256>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_4,_1>;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;

  //
  // Construct FusionOperation
  //
  constexpr int SFDVectorSize = 16;
  // Define the fusion operation applied during epilogue
  using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
      SFDVectorSize,
      ElementD, ElementCompute, 
      ElementSFD, GmemLayoutSFD,
      ElementC
    >;
  
  //
  // Construct CollectiveEpilogue
  //
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,      // Arch and Tensorop spec
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      cutlass::epilogue::collective::EpilogueTileAuto,                      // Epilogue subtile shape. Auto will find a suitable tile shape
      ElementAccumulator, ElementCompute,                                   // Mma instr's accumulator type and compute precision for epilogue
      ElementC, GmemLayoutC, AlignC,                                        // C tensor description
      ElementD, GmemLayoutD, AlignD,                                        // D tensor description
      cutlass::epilogue::TmaWarpSpecialized2Sm                              // Epilogue schedule policy
    >::CollectiveOp;

  //
  // Construct CollectiveMainloop
  //
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,      // Arch and Tensorop spec
      ElementA, GmemLayoutA, AlignA,                                        // A tensor elem type, layout and alignment requirement
      ElementB, GmemLayoutB, AlignB,                                        // B tensor elem type, layout and alignment requirement
      ElementAccumulator,                                                   // Mma instruction accumulator type
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      // Epilogue's SMEM usage that needs to be subtracted from overall SMEM capacity 
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100
    >::CollectiveOp;

  // Create Gemm Kernel using CollectiveEpilogue and CollectiveMainloop created by the builders
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
    >;
  
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  // Run tests
  auto pass = test::gemm::device::TestAll<Gemm>();
  // Check results
  EXPECT_TRUE(pass); 
}

//////////////////////////////////////////////////////////////////////////////
// FusionOperation: k-major output and datatype is float_e2m1_t with float_ue4m3_t scale-factor (vecsize 32)
//                  with alpha/beta fusion
//////////////////////////////////////////////////////////////////////////////

TEST(SM100Only_Device_Gemm_ue8m0xe2m1t_ue8m0xe2m1n_ue8m0xe2m1t_outputVs32_bstensorop_1sm_f32, 128x128x256_4x4x1) {
  // Describe A and B tensors
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  constexpr int AlignA = 32;
  using GmemLayoutA = cutlass::layout::RowMajor;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  constexpr int AlignB = 32;
  using GmemLayoutB = cutlass::layout::ColumnMajor;
  // Describe C and D tensors
  using ElementC = cutlass::half_t;
  constexpr int AlignC = 8;
  using GmemLayoutC = cutlass::layout::RowMajor;
  using ElementD = cutlass::float_e2m1_t;
  constexpr int AlignD = 32;
  using GmemLayoutD = cutlass::layout::RowMajor;
  // Describe SFD tensor
  using ElementSFD = cutlass::float_ue4m3_t;
  using GmemLayoutSFD = GmemLayoutD;
  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;

  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_128,_128,_256>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_4,_1>;

  //
  // Construct FusionOperation
  //
  constexpr int SFDVectorSize = 32;
  // Define the fusion operation applied during epilogue
  using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
      SFDVectorSize,
      ElementD, ElementCompute, 
      ElementSFD, GmemLayoutSFD,
      ElementC
    >;
  
  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,      // Arch and Tensorop spec
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      cutlass::epilogue::collective::EpilogueTileAuto,                      // Epilogue subtile shape. Auto will find a suitable tile shape
      ElementAccumulator, ElementCompute,                                   // Mma instr's accumulator type and compute precision for epilogue
      ElementC, GmemLayoutC, AlignC,                                        // C tensor description
      ElementD, GmemLayoutD, AlignD,                                        // D tensor description
      cutlass::epilogue::collective::EpilogueScheduleAuto                   // Epilogue schedule policy
    >::CollectiveOp;

  //
  // Construct CollectiveMainloop
  //

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,      // Arch and Tensorop spec
      ElementA, GmemLayoutA, AlignA,                                        // A tensor elem type, layout and alignment requirement
      ElementB, GmemLayoutB, AlignB,                                        // B tensor elem type, layout and alignment requirement
      ElementAccumulator,                                                   // Mma instruction accumulator type
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      // Epilogue's SMEM usage that needs to be subtracted from overall SMEM capacity 
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto    // Kernel schedule policy. Auto or using targeted scheduling policy
    >::CollectiveOp;

  // Create Gemm Kernel using CollectiveEpilogue and CollectiveMainloop created by the builders
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
    >;
  
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  // Run tests
  auto pass = test::gemm::device::TestAll<Gemm>();
  // Check results
  EXPECT_TRUE(pass); 
}

//////////////////////////////////////////////////////////////////////////////
// FusionOperation: k-major output and datatype is float_e2m1_t with float_ue4m3_t scale-factor (vecsize 16)
//                  with alpha+beta+relu+bias fusion
//////////////////////////////////////////////////////////////////////////////

TEST(SM100Only_Device_Gemm_ue8m0xe2m1t_ue8m0xe2m1n_ue8m0xe2m1n_outputVs16_bstensorop_1sm_f32_bias_relu, 128x128x256_4x4x1) {
  // Describe A and B tensors
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  constexpr int AlignA = 32;
  using GmemLayoutA = cutlass::layout::RowMajor;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  constexpr int AlignB = 32;
  using GmemLayoutB = cutlass::layout::ColumnMajor;
  // Describe C and D tensors
  using ElementC = cutlass::half_t;
  constexpr int AlignC = 8;
  using GmemLayoutC = cutlass::layout::RowMajor;
  using ElementD = cutlass::float_e2m1_t;
  constexpr int AlignD = 32;
  using GmemLayoutD = cutlass::layout::RowMajor;
  // Describe SFD tensor
  using ElementSFD = cutlass::float_ue4m3_t;
  using GmemLayoutSFD = GmemLayoutD;
  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  // Bias type
  using ElementBias = float;

  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_128,_128,_256>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_4,_1>;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  constexpr int SFDVectorSize = 32;

  using FusionOperation = cutlass::epilogue::fusion::LinCombPerColBiasBlockScaleFactor<
      SFDVectorSize, ElementD, ElementCompute, 
      ElementSFD, GmemLayoutSFD,
      ElementBias, ElementC
    >;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, GmemLayoutC, AlignC,
      ElementD, GmemLayoutC, AlignD,
      cutlass::epilogue::TmaWarpSpecialized1Sm,
      FusionOperation
    >::CollectiveOp;

  //
  // Construct CollectiveMainloop
  //

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementA, GmemLayoutA, AlignA,
      ElementB, GmemLayoutB, AlignB,
      ElementAccumulator,
      MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
    >;
  
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  auto pass = test::gemm::device::TestAll<Gemm>();
  EXPECT_TRUE(pass);  
}

#endif // #if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
