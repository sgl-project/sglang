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
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/epilogue/thread/activation.h"
#include "../../../common/cutlass_unit_test.h"

#include "../gemm_testbed_3x.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// FP8T x FP8N -> FP32 with 64x8x32 MMA-1CTA
TEST(SM100Only_Device_Gemm_e4m3t_e4m3n_void_bf16n_tensor_op_f32, 64x8x128_4x1x1_1sm) {
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_64,_8,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_1,_1>;

  // Describe A and B tensors
  using ElementA = cutlass::float_e4m3_t;
  constexpr int AlignA = 128 / sizeof_bits<ElementA>::value;
  using GmemLayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::float_e4m3_t;
  constexpr int AlignB = 128 / sizeof_bits<ElementB>::value;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 0;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = cutlass::bfloat16_t;
  constexpr int AlignD = 128 / sizeof_bits<ElementD>::value;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;

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

// FP8N x FP8N -> FP32 with 64x8x32 MMA-1CTA
TEST(SM100Only_Device_Gemm_e4m3n_e4m3n_void_bf16n_tensor_op_f32, 64x8x128_4x1x1_1sm) {
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_64,_8,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_1,_1>;

  // Describe A and B tensors
  using ElementA = cutlass::float_e4m3_t;
  constexpr int AlignA = 128 / sizeof_bits<ElementA>::value;
  using GmemLayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::float_e4m3_t;
  constexpr int AlignB = 128 / sizeof_bits<ElementB>::value;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 0;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = cutlass::bfloat16_t;
  constexpr int AlignD = 128 / sizeof_bits<ElementD>::value;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;

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

// FP8N x FP8T -> FP32 with 64x16x32 MMA-1CTA
TEST(SM100Only_Device_Gemm_e4m3n_e4m3t_void_bf16n_tensor_op_f32, 64x16x128_4x1x1_1sm) {
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_64,_16,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_1,_1>;

  // Describe A and B tensors
  using ElementA = cutlass::float_e4m3_t;
  constexpr int AlignA = 128 / sizeof_bits<ElementA>::value;
  using GmemLayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::float_e4m3_t;
  constexpr int AlignB = 128 / sizeof_bits<ElementB>::value;
  using GmemLayoutB = cutlass::layout::RowMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 0;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = cutlass::bfloat16_t;
  constexpr int AlignD = 128 / sizeof_bits<ElementD>::value;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;

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

// FP8T x FP8N -> FP32 with 128x8x32 MMA-1CTA
TEST(SM100Only_Device_Gemm_e4m3t_e4m3n_void_bf16n_tensor_op_f32, 128x8x128_4x1x1_1sm) {
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_128,_8,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_1,_1>;

  // Describe A and B tensors
  using ElementA = cutlass::float_e4m3_t;
  constexpr int AlignA = 128 / sizeof_bits<ElementA>::value;
  using GmemLayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::float_e4m3_t;
  constexpr int AlignB = 128 / sizeof_bits<ElementB>::value;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 0;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = cutlass::bfloat16_t;
  constexpr int AlignD = 128 / sizeof_bits<ElementD>::value;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;

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

// FP8N x FP8N -> FP32 with 128x8x32 MMA-1CTA
TEST(SM100Only_Device_Gemm_e4m3n_e4m3n_void_bf16n_tensor_op_f32, 128x8x128_4x1x1_1sm) {
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_128,_8,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_1,_1>;

  // Describe A and B tensors
  using ElementA = cutlass::float_e4m3_t;
  constexpr int AlignA = 128 / sizeof_bits<ElementA>::value;
  using GmemLayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::float_e4m3_t;
  constexpr int AlignB = 128 / sizeof_bits<ElementB>::value;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 0;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = cutlass::bfloat16_t;
  constexpr int AlignD = 128 / sizeof_bits<ElementD>::value;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;

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

// FP8N x FP8T -> FP32 with 128x16x32 MMA-1CTA
TEST(SM100Only_Device_Gemm_e4m3n_e4m3t_void_bf16n_tensor_op_f32, 128x16x128_4x1x1_1sm) {
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_128,_16,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_1,_1>;

  // Describe A and B tensors
  using ElementA = cutlass::float_e4m3_t;
  constexpr int AlignA = 128 / sizeof_bits<ElementA>::value;
  using GmemLayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::float_e4m3_t;
  constexpr int AlignB = 128 / sizeof_bits<ElementB>::value;
  using GmemLayoutB = cutlass::layout::RowMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 0;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = cutlass::bfloat16_t;
  constexpr int AlignD = 128 / sizeof_bits<ElementD>::value;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;

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

// FP8T x FP8N -> FP32 with 128x16x32 MMA-2CTA
TEST(SM100Only_Device_Gemm_e4m3t_e4m3n_void_bf16n_tensor_op_f32, 128x16x128_4x1x1_2sm) {
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_128,_16,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_1,_1>;

  // Describe A and B tensors
  using ElementA = cutlass::float_e4m3_t;
  constexpr int AlignA = 128 / sizeof_bits<ElementA>::value;
  using GmemLayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::float_e4m3_t;
  constexpr int AlignB = 128 / sizeof_bits<ElementB>::value;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 0;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = cutlass::bfloat16_t;
  constexpr int AlignD = 128 / sizeof_bits<ElementD>::value;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;

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

// FP8N x FP8N -> FP32 with 128x16x32 MMA-2CTA
TEST(SM100Only_Device_Gemm_e4m3n_e4m3n_void_bf16n_tensor_op_f32, 128x16x128_4x1x1_2sm) {
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_128,_16,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_1,_1>;

  // Describe A and B tensors
  using ElementA = cutlass::float_e4m3_t;
  constexpr int AlignA = 128 / sizeof_bits<ElementA>::value;
  using GmemLayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::float_e4m3_t;
  constexpr int AlignB = 128 / sizeof_bits<ElementB>::value;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 0;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = cutlass::bfloat16_t;
  constexpr int AlignD = 128 / sizeof_bits<ElementD>::value;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;

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

// FP8N x FP8T -> FP32 with 128x32x32 MMA-2CTA
TEST(SM100Only_Device_Gemm_e4m3n_e4m3t_void_bf16n_tensor_op_f32, 128x32x128_4x1x1_2sm) {
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_128,_32,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_1,_1>;

  // Describe A and B tensors
  using ElementA = cutlass::float_e4m3_t;
  constexpr int AlignA = 128 / sizeof_bits<ElementA>::value;
  using GmemLayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::float_e4m3_t;
  constexpr int AlignB = 128 / sizeof_bits<ElementB>::value;
  using GmemLayoutB = cutlass::layout::RowMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 0;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = cutlass::bfloat16_t;
  constexpr int AlignD = 128 / sizeof_bits<ElementD>::value;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;

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

// FP8T x FP8N -> FP32 with 256x16x32 MMA-2CTA
TEST(SM100Only_Device_Gemm_e4m3t_e4m3n_void_bf16n_tensor_op_f32, 256x16x128_4x1x1_2sm) {
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_256,_16,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_1,_1>;

  // Describe A and B tensors
  using ElementA = cutlass::float_e4m3_t;
  constexpr int AlignA = 128 / sizeof_bits<ElementA>::value;
  using GmemLayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::float_e4m3_t;
  constexpr int AlignB = 128 / sizeof_bits<ElementB>::value;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 0;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = cutlass::bfloat16_t;
  constexpr int AlignD = 128 / sizeof_bits<ElementD>::value;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;

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

// FP8N x FP8N -> FP32 with 256x16x32 MMA-2CTA
TEST(SM100Only_Device_Gemm_e4m3n_e4m3n_void_bf16n_tensor_op_f32, 256x16x128_4x1x1_2sm) {
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_256,_16,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_1,_1>;

  // Describe A and B tensors
  using ElementA = cutlass::float_e4m3_t;
  constexpr int AlignA = 128 / sizeof_bits<ElementA>::value;
  using GmemLayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::float_e4m3_t;
  constexpr int AlignB = 128 / sizeof_bits<ElementB>::value;
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 0;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = cutlass::bfloat16_t;
  constexpr int AlignD = 128 / sizeof_bits<ElementD>::value;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;

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

// FP8N x FP8T -> FP32 with 256x32x32 MMA-2CTA
TEST(SM100Only_Device_Gemm_e4m3n_e4m3t_void_bf16n_tensor_op_f32, 256x32x128_4x1x1_2sm) {
  // Tile and cluster shapes
  // Collective MMA takes tile shape of the MMA operation as input
  using MmaTileShape_MNK = Shape<_256,_32,_128>;
  // Cluster size for multicast
  using ClusterShape_MNK = Shape<_4,_1,_1>;

  // Describe A and B tensors
  using ElementA = cutlass::float_e4m3_t;
  constexpr int AlignA = 128 / sizeof_bits<ElementA>::value;
  using GmemLayoutA = cutlass::layout::ColumnMajor;
  using ElementB = cutlass::float_e4m3_t;
  constexpr int AlignB = 128 / sizeof_bits<ElementB>::value;
  using GmemLayoutB = cutlass::layout::RowMajor;

  // Describe C and D tensors
  using ElementC = void;
  constexpr int AlignC = 0;
  using GmemLayoutC = cutlass::layout::ColumnMajor;
  using ElementD = cutlass::bfloat16_t;
  constexpr int AlignD = 128 / sizeof_bits<ElementD>::value;
  using GmemLayoutD = cutlass::layout::ColumnMajor;

  // Mma's accumulator type
  using ElementAccumulator = float;
  // Epilogue computation's precision type
  using ElementCompute = float;

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
