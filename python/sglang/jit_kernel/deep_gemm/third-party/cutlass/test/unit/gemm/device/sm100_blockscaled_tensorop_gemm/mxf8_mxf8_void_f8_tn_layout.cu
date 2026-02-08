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
    \brief Unit tests for mxfp8xmxfp8 Block Scaled Gemm

    * A tensor: 
      * Types: {e5m2,e4m3}xue8m0
      * Layout: Row Major (T)
      * Alignment: 16 elements
    * B tensor: 
      * Types: {e5m2,e4m3}xue8m0
      * Layout: Column Major (N)
      * Alignment: 16 elements
    * Mma Tile Shapes supported:
      For the A tensor (mxfp6 type) the tile dimension with stride-1 should be divisible by 128, i.e., 128 element aligned.
      Support Matrix (Y: Yes, N: No)
      | 1/2 SM | Mma Tile Size | TN (*) | TT | NT | NN |
      |--------|---------------|--------|----|----|----|
      | 1SM    | 128x128x128   | Y      | Y  | Y  | Y  |
      | 1SM    | 128x192x128   | Y      | Y  | Y  | Y  |
      | 1SM    | 128x256x128   | Y      | Y  | Y  | Y  |
      | 2SM    | 256x128x128   | Y      | Y  | Y  | Y  |
      | 2SM    | 256x192x128   | Y      | Y  | Y  | Y  |
      | 2SM    | 256x256x128   | Y      | Y  | Y  | Y  |

      (*) Unit tests in this file
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

TEST(SM100Only_Device_Gemm_ue8m0xe5m2t_ue8m0xe5m2n_void_f8t_bstensorop_f32, 128x128x128_4x4x1_1sm_auto) {
  // Describe A and B tensors
  using ElementA = cutlass::mx_float8_t<cutlass::float_e5m2_t>;
  constexpr int AlignA = 16;
  using GmemLayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::mx_float8_t<cutlass::float_e5m2_t>;
  constexpr int AlignB = 16;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 16;
  using GmemLayoutC = cutlass::layout::RowMajor;
  using ElementD = cutlass::float_e5m2_t;
  constexpr int AlignD = 16;
  using GmemLayoutD = cutlass::layout::RowMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_128,_128,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_4,_1>;

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
      cutlass::gemm::KernelTmaWarpSpecialized1SmMxf8f6f4Sm100
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


TEST(SM100Only_Device_Gemm_ue8m0xe5m2t_ue8m0xe4m3n_void_f8t_bstensorop_f32, 128x192x128_2x2x1_1sm_auto) {
  // Describe A and B tensors
  using ElementA = cutlass::mx_float8_t<cutlass::float_e5m2_t>;
  constexpr int AlignA = 16;
  using GmemLayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  constexpr int AlignB = 16;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 16;
  using GmemLayoutC = cutlass::layout::RowMajor;
  using ElementD = cutlass::float_e5m2_t;
  constexpr int AlignD = 16;
  using GmemLayoutD = cutlass::layout::RowMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_128,_192,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_2,_2,_1>;

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

TEST(SM100Only_Device_Gemm_ue8m0xe4m3t_ue8m0xe5m2n_void_f8t_bstensorop_f32, 128x256x128_4x2x1_1sm_auto) {
  // Describe A and B tensors
  using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  constexpr int AlignA = 16;
  using GmemLayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::mx_float8_t<cutlass::float_e5m2_t>;
  constexpr int AlignB = 16;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  // For N=256 using f32 and f16 consumes too much SMEM space for Epilogue.
  using ElementC = void;
  constexpr int AlignC = 16;
  using GmemLayoutC = cutlass::layout::RowMajor;
  using ElementD = cutlass::float_e5m2_t;
  constexpr int AlignD = 16;
  using GmemLayoutD = cutlass::layout::RowMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_128,_256,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_2,_1>;

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

TEST(SM100Only_Device_Gemm_ue8m0xe4m3t_ue8m0xe4m3n_void_f8t_bstensorop_f32, 256x128x128_4x4x1_2sm_auto) {
  // Describe A and B tensors
  using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  constexpr int AlignA = 16;
  using GmemLayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  constexpr int AlignB = 16;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 16;
  using GmemLayoutC = cutlass::layout::RowMajor;
  using ElementD = cutlass::float_e5m2_t;
  constexpr int AlignD = 16;
  using GmemLayoutD = cutlass::layout::RowMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_256,_128,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_4,_1>;

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
      cutlass::gemm::KernelTmaWarpSpecialized2SmMxf8f6f4Sm100
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

TEST(SM100Only_Device_Gemm_ue8m0xe5m2t_ue8m0xe4m3n_void_f8t_bstensorop_f32, 256x192x128_2x2x1_2sm_auto) {
  // Describe A and B tensors
  using ElementA = cutlass::mx_float8_t<cutlass::float_e5m2_t>;
  constexpr int AlignA = 16;
  using GmemLayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  constexpr int AlignB = 16;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 16;
  using GmemLayoutC = cutlass::layout::RowMajor;
  using ElementD = cutlass::float_e5m2_t;
  constexpr int AlignD = 16;
  using GmemLayoutD = cutlass::layout::RowMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_256,_192,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_2,_2,_1>;

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
      cutlass::gemm::KernelTmaWarpSpecialized2SmBlockScaledSm100
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

TEST(SM100Only_Device_Gemm_ue8m0xe4m3t_ue8m0xe5m2n_void_f8t_bstensorop_f32, 256x256x128_2x1x1_2sm_auto) {
  // Describe A and B tensors
  using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  constexpr int AlignA = 16;
  using GmemLayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::mx_float8_t<cutlass::float_e5m2_t>;
  constexpr int AlignB = 16;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 16;
  using GmemLayoutC = cutlass::layout::RowMajor;
  using ElementD = cutlass::float_e5m2_t;
  constexpr int AlignD = 16;
  using GmemLayoutD = cutlass::layout::RowMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_256,_256,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_2,_1,_1>;

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

#endif // #if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
