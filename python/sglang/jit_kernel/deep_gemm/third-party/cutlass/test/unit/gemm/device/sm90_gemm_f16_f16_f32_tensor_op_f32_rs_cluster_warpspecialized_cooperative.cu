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

TEST(SM90_Device_Gemm_f16t_f16n_f32n_tensor_op_gmma_f32_cooperative, 128x192x64_1x1x1) {
  using ElementA = cutlass::half_t;
  using LayoutA  = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB  = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC  = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  using TileShape_MNK = Shape<_128,_192,_64>;
  using ClusterShape_MNK = Shape<_1,_1,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::Identity, ElementC, ElementAccumulator, ElementAccumulator>;

  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementC, LayoutC, 16 / sizeof(ElementC),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using KernelScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using AtomLayoutMNK = Layout<Shape<_2,_1,_1>>;
  using TiledMma = decltype(cute::make_tiled_mma(GMMA::rs_op_selector<
      ElementA, ElementB, ElementAccumulator, TileShape_MNK, GMMA::Major::K, GMMA::Major::K>(), AtomLayoutMNK{}));
  using GmemTiledCopyA = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));
  static constexpr GMMA::Major GmmaMajorA = cutlass::gemm::collective::detail::gmma_rs_tag_to_major_A<LayoutA>();
  static constexpr GMMA::Major GmmaMajorB = cutlass::gemm::collective::detail::gmma_rs_tag_to_major_B<LayoutB>();
  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::rs_smem_selector<GmmaMajorA, ElementA,
      decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{})), false>());
  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::rs_smem_selector<GmmaMajorB, ElementB,
      decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{})), false>());
  
  using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>;

  static constexpr int PipelineStages = cutlass::gemm::collective::detail::compute_stage_count_or_override<
      cutlass::gemm::collective::detail::sm90_smem_capacity_bytes,
      ElementA, ElementB, TileShape_MNK>(StageCountType{});
  using DispatchPolicy = cutlass::gemm::MainloopSm90TmaGmmaRmemAWarpSpecialized<
      PipelineStages, ClusterShape_MNK, KernelScheduleType>;

  using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementB,
      cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,ElementA>,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      void,
      cute::identity
    >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16n_f32n_tensor_op_gmma_f32_cooperative, 128x192x64_2x1x1) {
  using ElementA = cutlass::half_t;
  using LayoutA  = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB  = cutlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC  = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  using TileShape_MNK = Shape<_128,_192,_64>;
  using ClusterShape_MNK = Shape<_2,_1,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::Identity, ElementC, ElementAccumulator, ElementAccumulator>;

  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, 16 / sizeof(ElementC),
      ElementC, LayoutC, 16 / sizeof(ElementC),
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using KernelScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using AtomLayoutMNK = Layout<Shape<_2,_1,_1>>;
  using TiledMma = decltype(cute::make_tiled_mma(GMMA::rs_op_selector<
      ElementA, ElementB, ElementAccumulator, TileShape_MNK, GMMA::Major::K, GMMA::Major::K>(), AtomLayoutMNK{}));
  using GmemTiledCopyA = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));
  static constexpr GMMA::Major GmmaMajorA = cutlass::gemm::collective::detail::gmma_rs_tag_to_major_A<LayoutA>();
  static constexpr GMMA::Major GmmaMajorB = cutlass::gemm::collective::detail::gmma_rs_tag_to_major_B<LayoutB>();
  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::rs_smem_selector<GmmaMajorA, ElementA,
      decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{})), false>());
  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::rs_smem_selector<GmmaMajorB, ElementB,
      decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{})), false>());
  
  using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename EpilogueOp::SharedStorage)>;

  static constexpr int PipelineStages = cutlass::gemm::collective::detail::compute_stage_count_or_override<
      cutlass::gemm::collective::detail::sm90_smem_capacity_bytes,
      ElementA, ElementB, TileShape_MNK>(StageCountType{});
  using DispatchPolicy = cutlass::gemm::MainloopSm90TmaGmmaRmemAWarpSpecialized<
      PipelineStages, ClusterShape_MNK, KernelScheduleType>;

  using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementB,
      cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,ElementA>,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      void,
      cute::identity
    >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAllBiasElementwise<Gemm>());
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
