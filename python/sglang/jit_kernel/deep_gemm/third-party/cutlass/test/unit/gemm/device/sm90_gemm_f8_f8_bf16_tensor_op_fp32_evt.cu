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
    \brief Tests for Sm90 f8_f8_bf16 with EVT epilogue 
    ScaledLinCombPerRowBiasEltAct and ScaledLinCombPerRowBiasEltActAmaxAux
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
#include "cutlass/epilogue/collective/sm70_epilogue_vectorized.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"

#include "../../common/cutlass_unit_test.h"

#include "gemm_testbed_3x_evt.hpp"
#include "sm90_evt_operations.hpp"


#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;

// Z = scale_a * scale_b * alpha * acc + beta * scale_c * C + per-row bias
// if D is fp8 
//   D = scale_d * activation(Z)
// else
//   D = activation(Z)
TEST(SM90_Device_Gemm_e4m3t_e4m3n_bf16t_tensor_op_gmma_f32_epilogue, 64x128x128_ScaledLinCombPerRowBiasEltAct) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using TileShape_MNK = Shape<_64,_128,_128>;
  using ClusterShape_MNK = Shape<_1,_1,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using FusionCallbacks = cutlass::epilogue::fusion::Sm90ScaledLinCombPerRowBiasEltAct<
    TileShape_MNK,                      // CtaTileShapeMNK
    cutlass::epilogue::thread::ReLu,    // ActivationFn
    cutlass::bfloat16_t,                // ElementOutput
    float,                              // ElementCompute
    cutlass::bfloat16_t                 // ElementBias
  >;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::bfloat16_t, LayoutC, 8,
      cutlass::bfloat16_t, LayoutC, 8,
      EpilogueSchedule,
      FusionCallbacks
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::float_e4m3_t, LayoutA, 16,
      cutlass::float_e4m3_t, LayoutB, 16,
      float,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Host reference
  using HostReference = test::gemm::device::HostScaledLinCombPerRowBiasEltAct<
    Gemm, cutlass::epilogue::thread::ReLu, cutlass::bfloat16_t
  >;
  bool passed = test::gemm::device::TestAllEVT<Gemm, HostReference>(true);
  EXPECT_TRUE(passed);
}

// Z = scale_a * scale_b * alpha * acc + scale_c * beta * C + per-row bias
// if D is fp8 
//   amax_d = max(abs(elements in activation(Z)))
//   D = scale_d * activation(Z)
// else
//   D = activation(Z)
// if Aux is fp8 
//   amax_aux = max(abs(elements in Z))
//   Aux = scale_aux * Z
// else
//   Aux = Z
TEST(SM90_Device_Gemm_e4m3t_e4m3n_bf16n_tensor_op_gmma_f32_epilogue, 64x128x128_4x1x1_ScaledLinCombPerRowBiasEltActAmaxAux) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_64,_128,_128>;
  using ClusterShape_MNK = Shape<_2,_4,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<
    TileShape_MNK, EpilogueTileType, cutlass::bfloat16_t, cutlass::bfloat16_t, EpilogueSchedule>;
  using AuxStoreDescriptor = cutlass::epilogue::collective::detail::AuxStoreDescriptor<
    EpilogueDescriptor, cutlass::layout::RowMajor, cutlass::bfloat16_t>;
    
  using FusionCallbacks = cutlass::epilogue::fusion::Sm90ScaledLinCombPerRowBiasEltActAmaxAux<
    TileShape_MNK,                               // CtaTileShapeMNK
    typename EpilogueDescriptor::EpilogueTile,   // EpilogueTile
    EpilogueDescriptor::StagesD,                 // StagesD
    typename AuxStoreDescriptor::Stride,         // StrideAux
    typename AuxStoreDescriptor::SmemLayoutAtom, // SmemLayoutAtom
    typename AuxStoreDescriptor::CopyOpR2S,      // CopyOpR2S
    cutlass::epilogue::thread::ReLu,             // ActivationFn
    cutlass::bfloat16_t,                         // ElementOutput
    float,                                       // ElementCompute
    cutlass::bfloat16_t,                         // ElementBias
    float                                        // ElementScalar
  >;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      EpilogueTileType,
      float, float,
      cutlass::bfloat16_t, LayoutC, 16,
      cutlass::bfloat16_t, LayoutC, 16,
      EpilogueSchedule,
      FusionCallbacks
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::float_e4m3_t, LayoutA, 16,
      cutlass::float_e4m3_t, LayoutB, 16,
      float,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Host reference
  using HostReference = test::gemm::device::HostScaledLinCombPerRowBiasEltActAmaxAux<
    Gemm, cutlass::epilogue::thread::ReLu, cutlass::bfloat16_t
  >;
  bool passed = test::gemm::device::TestAllEVT<Gemm, HostReference>(true);
  EXPECT_TRUE(passed);
}
#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
