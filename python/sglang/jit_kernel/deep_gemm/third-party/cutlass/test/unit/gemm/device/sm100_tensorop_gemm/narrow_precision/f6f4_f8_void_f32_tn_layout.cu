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
    \brief Unit tests for {f6f4}xf8 Gemm

    * A tensor: 
      * Types: {e2m1,e2m3,e3m2}
      * Alignment: 128 elements
    * B tensor: 
      * Types: {e5m2,e4m3}
      * Alignment: 16 elements
    * Mma Tile Shapes supported:
      Support Matrix (Y: Yes, N: No)
      | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy                    |
      |--------|----------------|----|----|----|----|------------------------------------|
      | 1SM    | 64x64x128      | Y  | Y  | N  | N  | `KernelTmaWarpSpecialized1SmSm100` |
      | 1SM    | 64x128x128     | Y  | Y  | N  | N  | `KernelTmaWarpSpecialized1SmSm100` |
      | 1SM    | 64x192x128     | Y  | Y  | N  | N  | `KernelTmaWarpSpecialized1SmSm100` |
      | 1SM    | 64x256x128     | Y  | Y  | N  | N  | `KernelTmaWarpSpecialized1SmSm100` |
      | 1SM    | 128x64x128     | Y  | Y  | Y  | Y  | `KernelTmaWarpSpecialized1SmSm100` |
      | 1SM    | 128x128x128    | Y  | Y  | Y  | Y  | `KernelTmaWarpSpecialized1SmSm100` |
      | 1SM    | 128x192x128    | Y  | Y  | Y  | Y  | `KernelTmaWarpSpecialized1SmSm100` |
      | 1SM    | 128x256x128    | Y  | Y  | Y  | Y  | `KernelTmaWarpSpecialized1SmSm100` |
      | 2SM    | 128x64x128     | Y  | Y  | Y  | Y  | `KernelTmaWarpSpecialized2SmSm100` |
      | 2SM    | 128x128x128    | Y  | Y  | N  | N  | `KernelTmaWarpSpecialized2SmSm100` |
      | 2SM    | 128x192x128    | Y  | Y  | N  | N  | `KernelTmaWarpSpecialized2SmSm100` |
      | 2SM    | 128x256x128    | Y  | Y  | N  | N  | `KernelTmaWarpSpecialized2SmSm100` |
      | 2SM    | 256x64x128     | Y  | Y  | Y  | Y  | `KernelTmaWarpSpecialized2SmSm100` |
      | 2SM    | 256x128x128    | Y  | Y  | Y  | Y  | `KernelTmaWarpSpecialized2SmSm100` |
      | 2SM    | 256x192x128    | Y  | Y  | Y  | Y  | `KernelTmaWarpSpecialized2SmSm100` |
      | 2SM    | 256x256x128    | Y  | Y  | Y  | Y  | `KernelTmaWarpSpecialized2SmSm100` |
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
#include "../../../../common/cutlass_unit_test.h"
#include "../../gemm_testbed_3x.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

TEST(SM100Only_Device_Gemm_e2m1t_e4m3n_void_f32n_tensor_op_f32, 64x64x128_4x1x1_1sm) {
  // Describe A and B tensors
  using ElementA = cutlass::float_e2m1_t;
  constexpr int AlignA = 128;
  using GmemLayoutA = cutlass::layout::RowMajor;
  constexpr int AlignB = 16;
  using ElementB = cutlass::float_e4m3_t;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 4;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = float;
  constexpr int AlignD = 4;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_64,_64,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_1,_1>;

  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
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
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
      ElementA, GmemLayoutA, AlignA,                                        // A tensor elem type, layout and alignment requirement
      ElementB, GmemLayoutB, AlignB,                                        // B tensor elem type, layout and alignment requirement
      ElementAccumulator,                                                   // Mma instruction accumulator type
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      // Epilogue's SMEM usage that needs to be subtracted from overall SMEM capacity 
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100                       // Kernel schedule policy. Auto or using targeted scheduling policy
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

TEST(SM100Only_Device_Gemm_e2m3t_e5m2n_void_f32n_tensor_op_f32, 64x128x128_2x1x1_1sm_streamK) {
  // Describe A and B tensors
  using ElementA = cutlass::float_e2m3_t;
  constexpr int AlignA = 128;
  using GmemLayoutA = cutlass::layout::RowMajor;
  constexpr int AlignB = 16;
  using ElementB = cutlass::float_e5m2_t;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 4;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = float;
  constexpr int AlignD = 4;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_64,_128,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_2,_1,_1>;

  // Tile Scheduler
  using TileScheduler = cutlass::gemm::StreamKScheduler;

  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
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
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
      ElementA, GmemLayoutA, AlignA,                                        // A tensor elem type, layout and alignment requirement
      ElementB, GmemLayoutB, AlignB,                                        // B tensor elem type, layout and alignment requirement
      ElementAccumulator,                                                   // Mma instruction accumulator type
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      // Epilogue's SMEM usage that needs to be subtracted from overall SMEM capacity 
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100                       // Kernel schedule policy. Auto or using targeted scheduling policy
    >::CollectiveOp;

  // Create Gemm Kernel using CollectiveEpilogue and CollectiveMainloop created by the builders
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileScheduler
    >;
  
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  // Run tests
  auto pass = test::gemm::device::TestAll<Gemm>();
  // Check results
  EXPECT_TRUE(pass);
}


TEST(SM100Only_Device_Gemm_e3m2t_e4m3n_void_f32n_tensor_op_f32, 64x192x128_2x4x1_1sm) {
  // Describe A and B tensors
  using ElementA = cutlass::float_e3m2_t;
  constexpr int AlignA = 128;
  using GmemLayoutA = cutlass::layout::RowMajor;
  constexpr int AlignB = 16;
  using ElementB = cutlass::float_e4m3_t;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 4;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = float;
  constexpr int AlignD = 4;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_64,_192,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_2,_4,_1>;

  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
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
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
      ElementA, GmemLayoutA, AlignA,                                        // A tensor elem type, layout and alignment requirement
      ElementB, GmemLayoutB, AlignB,                                        // B tensor elem type, layout and alignment requirement
      ElementAccumulator,                                                   // Mma instruction accumulator type
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      // Epilogue's SMEM usage that needs to be subtracted from overall SMEM capacity 
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100                       // Kernel schedule policy. Auto or using targeted scheduling policy
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

TEST(SM100Only_Device_Gemm_e2m1t_e5m2n_void_f32n_tensor_op_f32, 64x256x128_2x2x1_1sm) {
  // Describe A and B tensors
  using ElementA = cutlass::float_e2m1_t;
  constexpr int AlignA = 128;
  using GmemLayoutA = cutlass::layout::RowMajor;
  constexpr int AlignB = 16;
  using ElementB = cutlass::float_e5m2_t;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 4;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = float;
  constexpr int AlignD = 4;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_64,_256,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_2,_2,_1>;

  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
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
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
      ElementA, GmemLayoutA, AlignA,                                        // A tensor elem type, layout and alignment requirement
      ElementB, GmemLayoutB, AlignB,                                        // B tensor elem type, layout and alignment requirement
      ElementAccumulator,                                                   // Mma instruction accumulator type
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      // Epilogue's SMEM usage that needs to be subtracted from overall SMEM capacity 
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100                       // Kernel schedule policy. Auto or using targeted scheduling policy
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

TEST(SM100Only_Device_Gemm_e3m2t_e4m3n_void_f32n_tensor_op_f32, 128x64x128_4x1x1_1sm_streamK) {
  // Describe A and B tensors
  using ElementA = cutlass::float_e3m2_t;
  constexpr int AlignA = 128;
  using GmemLayoutA = cutlass::layout::RowMajor;
  constexpr int AlignB = 16;
  using ElementB = cutlass::float_e4m3_t;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 4;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = float;
  constexpr int AlignD = 4;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_128,_64,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_1,_1>;

  // Tile Scheduler
  using TileScheduler = cutlass::gemm::StreamKScheduler;

  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
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
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
      ElementA, GmemLayoutA, AlignA,                                        // A tensor elem type, layout and alignment requirement
      ElementB, GmemLayoutB, AlignB,                                        // B tensor elem type, layout and alignment requirement
      ElementAccumulator,                                                   // Mma instruction accumulator type
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      // Epilogue's SMEM usage that needs to be subtracted from overall SMEM capacity 
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100                       // Kernel schedule policy. Auto or using targeted scheduling policy
    >::CollectiveOp;

  // Create Gemm Kernel using CollectiveEpilogue and CollectiveMainloop created by the builders
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileScheduler
    >;
  
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  // Run tests
  auto pass = test::gemm::device::TestAll<Gemm>();
  // Check results
  EXPECT_TRUE(pass);
}

TEST(SM100Only_Device_Gemm_e2m3t_e5m2n_void_f32n_tensor_op_f32, 128x128x128_2x1x1_1sm) {
  // Describe A and B tensors
  using ElementA = cutlass::float_e2m3_t;
  constexpr int AlignA = 128;
  using GmemLayoutA = cutlass::layout::RowMajor;
  constexpr int AlignB = 16;
  using ElementB = cutlass::float_e4m3_t;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 4;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = float;
  constexpr int AlignD = 4;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_128,_128,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_2,_1,_1>;

  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
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
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
      ElementA, GmemLayoutA, AlignA,                                        // A tensor elem type, layout and alignment requirement
      ElementB, GmemLayoutB, AlignB,                                        // B tensor elem type, layout and alignment requirement
      ElementAccumulator,                                                   // Mma instruction accumulator type
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      // Epilogue's SMEM usage that needs to be subtracted from overall SMEM capacity 
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100                       // Kernel schedule policy. Auto or using targeted scheduling policy
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

TEST(SM100Only_Device_Gemm_e2m3t_e4m3n_void_f32n_tensor_op_f32, 128x192x128_2x4x1_1sm_streamK) {
  // Describe A and B tensors
  using ElementA = cutlass::float_e2m3_t;
  constexpr int AlignA = 128;
  using GmemLayoutA = cutlass::layout::RowMajor;
  constexpr int AlignB = 16;
  using ElementB = cutlass::float_e4m3_t;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 4;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = float;
  constexpr int AlignD = 4;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_128,_192,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_2,_4,_1>;

  // Tile Scheduler
  using TileScheduler = cutlass::gemm::StreamKScheduler;

  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
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
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
      ElementA, GmemLayoutA, AlignA,                                        // A tensor elem type, layout and alignment requirement
      ElementB, GmemLayoutB, AlignB,                                        // B tensor elem type, layout and alignment requirement
      ElementAccumulator,                                                   // Mma instruction accumulator type
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      // Epilogue's SMEM usage that needs to be subtracted from overall SMEM capacity 
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100                       // Kernel schedule policy. Auto or using targeted scheduling policy
    >::CollectiveOp;

  // Create Gemm Kernel using CollectiveEpilogue and CollectiveMainloop created by the builders
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileScheduler
    >;
  
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  // Run tests
  auto pass = test::gemm::device::TestAll<Gemm>();
  // Check results
  EXPECT_TRUE(pass);
}

TEST(SM100Only_Device_Gemm_e2m1t_e5m2n_void_f32n_tensor_op_f32, 128x256x128_2x2x1_1sm) {
  // Describe A and B tensors
  using ElementA = cutlass::float_e2m1_t;
  constexpr int AlignA = 128;
  using GmemLayoutA = cutlass::layout::RowMajor;
  constexpr int AlignB = 16;
  using ElementB = cutlass::float_e4m3_t;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 4;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = float;
  constexpr int AlignD = 4;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_128,_256,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_2,_2,_1>;

  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
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
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
      ElementA, GmemLayoutA, AlignA,                                        // A tensor elem type, layout and alignment requirement
      ElementB, GmemLayoutB, AlignB,                                        // B tensor elem type, layout and alignment requirement
      ElementAccumulator,                                                   // Mma instruction accumulator type
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      // Epilogue's SMEM usage that needs to be subtracted from overall SMEM capacity 
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100                       // Kernel schedule policy. Auto or using targeted scheduling policy
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

TEST(SM100Only_Device_Gemm_e2m3t_e4m3n_void_f32n_tensor_op_f32, 128x64x128_4x1x1_2sm_streamK) {
  // Describe A and B tensors
  using ElementA = cutlass::float_e2m3_t;
  constexpr int AlignA = 128;
  using GmemLayoutA = cutlass::layout::RowMajor;
  constexpr int AlignB = 16;
  using ElementB = cutlass::float_e4m3_t;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 4;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = float;
  constexpr int AlignD = 4;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_128,_64,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_1,_1>;

  // Tile Scheduler
  using TileScheduler = cutlass::gemm::StreamKScheduler;

  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
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
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
      ElementA, GmemLayoutA, AlignA,                                        // A tensor elem type, layout and alignment requirement
      ElementB, GmemLayoutB, AlignB,                                        // B tensor elem type, layout and alignment requirement
      ElementAccumulator,                                                   // Mma instruction accumulator type
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      // Epilogue's SMEM usage that needs to be subtracted from overall SMEM capacity 
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized2SmSm100                       // Kernel schedule policy. Auto or using targeted scheduling policy
    >::CollectiveOp;

  // Create Gemm Kernel using CollectiveEpilogue and CollectiveMainloop created by the builders
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileScheduler
    >;
  
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  // Run tests
  auto pass = test::gemm::device::TestAll<Gemm>();
  // Check results
  EXPECT_TRUE(pass);
}

TEST(SM100Only_Device_Gemm_e2m1t_e5m2n_void_f32n_tensor_op_f32, 128x128x128_2x1x1_2sm) {
  // Describe A and B tensors
  using ElementA = cutlass::float_e2m1_t;
  constexpr int AlignA = 128;
  using GmemLayoutA = cutlass::layout::RowMajor;
  constexpr int AlignB = 16;
  using ElementB = cutlass::float_e5m2_t;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 4;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = float;
  constexpr int AlignD = 4;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_128,_128,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_2,_1,_1>;

  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
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
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
      ElementA, GmemLayoutA, AlignA,                                        // A tensor elem type, layout and alignment requirement
      ElementB, GmemLayoutB, AlignB,                                        // B tensor elem type, layout and alignment requirement
      ElementAccumulator,                                                   // Mma instruction accumulator type
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      // Epilogue's SMEM usage that needs to be subtracted from overall SMEM capacity 
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized2SmSm100                       // Kernel schedule policy. Auto or using targeted scheduling policy
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

TEST(SM100Only_Device_Gemm_e2m1t_e5m2n_void_f32n_tensor_op_f32, 128x192x128_2x4x1_2sm) {
  // Describe A and B tensors
  using ElementA = cutlass::float_e2m1_t;
  constexpr int AlignA = 128;
  using GmemLayoutA = cutlass::layout::RowMajor;
  constexpr int AlignB = 16;
  using ElementB = cutlass::float_e5m2_t;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 4;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = float;
  constexpr int AlignD = 4;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_128,_192,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_2,_4,_1>;

  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
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
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
      ElementA, GmemLayoutA, AlignA,                                        // A tensor elem type, layout and alignment requirement
      ElementB, GmemLayoutB, AlignB,                                        // B tensor elem type, layout and alignment requirement
      ElementAccumulator,                                                   // Mma instruction accumulator type
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      // Epilogue's SMEM usage that needs to be subtracted from overall SMEM capacity 
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized2SmSm100                       // Kernel schedule policy. Auto or using targeted scheduling policy
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

TEST(SM100Only_Device_Gemm_e3m2t_e5m2n_void_f32n_tensor_op_f32, 128x256x128_2x2x1_2sm) {
  // Describe A and B tensors
  using ElementA = cutlass::float_e3m2_t;
  constexpr int AlignA = 128;
  using GmemLayoutA = cutlass::layout::RowMajor;
  constexpr int AlignB = 16;
  using ElementB = cutlass::float_e5m2_t;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 4;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = float;
  constexpr int AlignD = 4;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_128,_256,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_2,_2,_1>;

  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
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
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
      ElementA, GmemLayoutA, AlignA,                                        // A tensor elem type, layout and alignment requirement
      ElementB, GmemLayoutB, AlignB,                                        // B tensor elem type, layout and alignment requirement
      ElementAccumulator,                                                   // Mma instruction accumulator type
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      // Epilogue's SMEM usage that needs to be subtracted from overall SMEM capacity 
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized2SmSm100                       // Kernel schedule policy. Auto or using targeted scheduling policy
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

TEST(SM100Only_Device_Gemm_e2m1t_e4m3n_void_f32n_tensor_op_f32, 256x64x128_4x1x1_2sm_streamK) {
  // Describe A and B tensors
  using ElementA = cutlass::float_e2m1_t;
  constexpr int AlignA = 128;
  using GmemLayoutA = cutlass::layout::RowMajor;
  constexpr int AlignB = 16;
  using ElementB = cutlass::float_e4m3_t;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 4;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = float;
  constexpr int AlignD = 4;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_256,_64,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_1,_1>;

  // Tile Scheduler
  using TileScheduler = cutlass::gemm::StreamKScheduler;

  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
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
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
      ElementA, GmemLayoutA, AlignA,                                        // A tensor elem type, layout and alignment requirement
      ElementB, GmemLayoutB, AlignB,                                        // B tensor elem type, layout and alignment requirement
      ElementAccumulator,                                                   // Mma instruction accumulator type
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      // Epilogue's SMEM usage that needs to be subtracted from overall SMEM capacity 
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized2SmSm100                       // Kernel schedule policy. Auto or using targeted scheduling policy
    >::CollectiveOp;

  // Create Gemm Kernel using CollectiveEpilogue and CollectiveMainloop created by the builders
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileScheduler
    >;
  
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  // Run tests
  auto pass = test::gemm::device::TestAll<Gemm>();
  // Check results
  EXPECT_TRUE(pass);
}

TEST(SM100Only_Device_Gemm_e3m2t_e5m2n_void_f32n_tensor_op_f32, 256x128x128_2x1x1_2sm) {
  // Describe A and B tensors
  using ElementA = cutlass::float_e3m2_t;
  constexpr int AlignA = 128;
  using GmemLayoutA = cutlass::layout::RowMajor;
  constexpr int AlignB = 16;
  using ElementB = cutlass::float_e5m2_t;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 4;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = float;
  constexpr int AlignD = 4;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_256,_128,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_2,_1,_1>;

  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
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
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
      ElementA, GmemLayoutA, AlignA,                                        // A tensor elem type, layout and alignment requirement
      ElementB, GmemLayoutB, AlignB,                                        // B tensor elem type, layout and alignment requirement
      ElementAccumulator,                                                   // Mma instruction accumulator type
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      // Epilogue's SMEM usage that needs to be subtracted from overall SMEM capacity 
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized2SmSm100                       // Kernel schedule policy. Auto or using targeted scheduling policy
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

TEST(SM100Only_Device_Gemm_e2m1t_e4m3n_void_f32n_tensor_op_f32, 256x192x128_2x4x1_2sm_streamK) {
  // Describe A and B tensors
  using ElementA = cutlass::float_e2m1_t;
  constexpr int AlignA = 128;
  using GmemLayoutA = cutlass::layout::RowMajor;
  constexpr int AlignB = 16;
  using ElementB = cutlass::float_e4m3_t;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 4;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = float;
  constexpr int AlignD = 4;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_256,_192,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_2,_4,_1>;

  // Tile Scheduler
  using TileScheduler = cutlass::gemm::StreamKScheduler;

  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
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
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
      ElementA, GmemLayoutA, AlignA,                                        // A tensor elem type, layout and alignment requirement
      ElementB, GmemLayoutB, AlignB,                                        // B tensor elem type, layout and alignment requirement
      ElementAccumulator,                                                   // Mma instruction accumulator type
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      // Epilogue's SMEM usage that needs to be subtracted from overall SMEM capacity 
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized2SmSm100                       // Kernel schedule policy. Auto or using targeted scheduling policy
    >::CollectiveOp;

  // Create Gemm Kernel using CollectiveEpilogue and CollectiveMainloop created by the builders
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileScheduler
    >;
  
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  // Run tests
  auto pass = test::gemm::device::TestAll<Gemm>();
  // Check results
  EXPECT_TRUE(pass);
}

TEST(SM100Only_Device_Gemm_e2m1t_e5m2n_void_f32n_tensor_op_f32, 256x256x128_2x2x1_2sm) {
  // Describe A and B tensors
  using ElementA = cutlass::float_e2m1_t;
  constexpr int AlignA = 128;
  using GmemLayoutA = cutlass::layout::RowMajor;
  constexpr int AlignB = 16;
  using ElementB = cutlass::float_e5m2_t;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 4;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = float;
  constexpr int AlignD = 4;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;
  
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_256,_256,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_2,_2,_1>;

  //
  // Construct CollectiveEpilogue
  //

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
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
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,                 // Arch and Tensorop spec
      ElementA, GmemLayoutA, AlignA,                                        // A tensor elem type, layout and alignment requirement
      ElementB, GmemLayoutB, AlignB,                                        // B tensor elem type, layout and alignment requirement
      ElementAccumulator,                                                   // Mma instruction accumulator type
      MmaTileShape_MNK, ClusterShape_MNK,                                   // Mma instruction tile shape, cluster shape
      // Epilogue's SMEM usage that needs to be subtracted from overall SMEM capacity 
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized2SmSm100                       // Kernel schedule policy. Auto or using targeted scheduling policy
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
#endif
