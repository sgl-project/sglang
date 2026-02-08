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
    \brief Tests for Sm90 f16_f16_f16 with cooperative EVT epilogue
    D = alpha * acc + beta * c + aux_load
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
#include "cutlass/util/reference/device/tensor_compare.h"

#include "../../common/cutlass_unit_test.h"

#include "gemm_testbed_3x_evt.hpp"
#include "sm90_evt_operations.hpp"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;

namespace test::gemm::device {
template <class ElementCompute, class ElementAccumulator, bool IsCNeed>
static constexpr auto select_evt_d() {
  using namespace cutlass::epilogue::fusion;
  constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  using BinaryCompute0 = Sm90EVT<Sm90Compute<
                                   cutlass::multiplies,
                                   ElementCompute,
                                   ElementCompute,
                                   RoundStyle>,                          // alpha * acc
                            Sm90ScalarBroadcast<ElementAccumulator>,  // alpha
                            Sm90AccFetch                              // acc
                         >;
  if constexpr (IsCNeed) {
    using EVT_D = Sm90EVT<Sm90Compute<cutlass::homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle>,
                    Sm90ScalarBroadcast<ElementAccumulator>,  // beta
                    Sm90SrcFetch<ElementCompute>,                             // C
                    BinaryCompute0>;
    return EVT_D{};
  } else {
    return BinaryCompute0{};
  }
}

template <class Gemm, class GemmWithoutD>
bool testEVTAuxStoreWithoutD() {
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  int max_alignment = std::max(Gemm::kAlignmentA, Gemm::kAlignmentB);
  std::vector<int> problem_size_m = {max_alignment, 512 - 3 * max_alignment};
  std::vector<int> problem_size_n = {max_alignment, 512 - 2 * max_alignment};

  if constexpr (std::is_same_v<typename Gemm::GemmKernel::DispatchPolicy::Schedule,
        cutlass::gemm::KernelTmaWarpSpecializedPingpong>) {
    problem_size_m.push_back(768);
    problem_size_n.push_back(768);
  }

  using GemmKernel = typename Gemm::GemmKernel;
  constexpr int Stages = Gemm::GemmKernel::DispatchPolicy::Stages;
  constexpr int TileShapeK = cute::size<2>(typename Gemm::GemmKernel::TileShape{});

  std::vector<int> problem_size_k = {max_alignment, TileShapeK * (Stages + 1) - max_alignment};
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  using ElementD = typename Gemm::ElementD;
  constexpr bool has_c = not cute::is_void_v<ElementC>;
  cutlass::DeviceAllocation<ElementA> A_block;
  cutlass::DeviceAllocation<ElementB> B_block;
  cutlass::DeviceAllocation<cute::conditional_t<has_c, ElementC, ElementD>> C_block;
  cutlass::DeviceAllocation<ElementD> D_block;
  cutlass::DeviceAllocation<ElementD> aux_store_D_block;
  cutlass::DeviceAllocation<uint8_t> workspace;

  for (int m : problem_size_m) {
  for (int n : problem_size_n) {
    for (int k : problem_size_k) {
    ProblemShapeType problem_size;
    int l = 1;
    problem_size = ProblemShapeType{m, n, k, l};

    // Run Base Gemm to get reference D
    A_block.reset(m * k);
    B_block.reset(k * n);
    C_block.reset(m * n);
    D_block.reset(m * n);
    aux_store_D_block.reset(m * n);
    Gemm gemm_op_base;

    auto stride_A = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {m, k, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {n, k, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename GemmKernel::StrideC{}, {m, n, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {m, n, 1});

    auto arguments_base = typename Gemm::Arguments {
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {
        A_block.get(), stride_A,
        B_block.get(), stride_B
      },
      {   // Epilogue arguments
        {}, // thread
        has_c ? C_block.get() : nullptr, stride_C,
        D_block.get(), stride_D,
      },  // Epilogue arguments end
      /*hw_info=*/{},
      /*scheduler_args=*/{}
    };

    // check without D aux store
    // set D to be void and use Sm90AuxStore to write to D
    // and then the D is the same
    GemmWithoutD gemm_op;

    auto arguments = typename GemmWithoutD::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {
        A_block.get(), stride_A,
        B_block.get(), stride_B
      },
      {   // Epilogue arguments
        {}, // thread
        has_c ? C_block.get() : nullptr, stride_C,
        nullptr, stride_D,
      },  // Epilogue arguments end
      /*hw_info=*/{},
      /*scheduler_args=*/{}
    };

    constexpr float beta [[maybe_unused]] = 1.0;
    constexpr float alpha [[maybe_unused]] = 1.0;

    using ElementC = typename GemmWithoutD::ElementC;

    if constexpr (not has_c) {
      arguments_base.epilogue.thread = {
        // binary op : alpha * acc
        {{alpha}},  // leaf op+args : alpha
        {},         // leaf op+args : acc
        {}          // binary args : multiplies
      };
      arguments.epilogue.thread = {
        // unary op: aux store D
        {
          // binary op : alpha * acc
          {{alpha}},  // leaf op+args : alpha
          {},         // leaf op+args : acc
          {}          // binary args : multiplies
        },
        {aux_store_D_block.get(), stride_D}
      };

    } else {
      arguments_base.epilogue.thread = {
        // ternary op : beta * C + (alpha * acc)
        {{beta}}, // leaf op+args : beta
        {},  // op+args : C
        {
            // binary op : alpha * acc
            {{alpha}},  // leaf op+args : alpha
            {},         // leaf op+args : acc
            {}          // binary args : multiplies
        },              // end binary op
        {}              // ternary args : multiply_add
      };
      arguments.epilogue.thread = {
        // unary op: aux store D
        {
          // ternary op : beta * C + (alpha * acc)
          {{beta}}, // leaf op+args : beta
          {},  // op+args : C
          {
              // binary op : alpha * acc
              {{alpha}},  // leaf op+args : alpha
              {},         // leaf op+args : acc
              {}          // binary args : multiplies
          },              // end binary op
          {}              // ternary args : multiply_add
        },
        {aux_store_D_block.get(), stride_D}
      };
    }


    cutlass::Status status;
    cudaError_t result;

    status = gemm_op_base.can_implement(arguments_base);
    EXPECT_EQ(status, cutlass::Status::kSuccess) << "Error gemm base not supported";
    size_t workspace_size_base = Gemm::get_workspace_size(arguments_base);
    workspace.reset(workspace_size_base);
    status = gemm_op_base.initialize(arguments_base, workspace.get());
    status = gemm_op_base.run();
    result = cudaDeviceSynchronize();
    EXPECT_EQ(result, cudaSuccess) << "Error at Base Kernel Sync.";

    size_t workspace_size = GemmWithoutD::get_workspace_size(arguments);
    workspace.reset(workspace_size);
    status = gemm_op.can_implement(arguments);
    EXPECT_EQ(status, cutlass::Status::kSuccess);
    status = gemm_op.initialize(arguments, workspace.get());
    status = gemm_op.run();
    result = cudaDeviceSynchronize();
    EXPECT_EQ(result, cudaSuccess) << "Error at Kernel Sync.";

    bool passed = cutlass::reference::device::BlockCompareEqual(aux_store_D_block.get(), D_block.get(), m * n);
    if (!passed) {
      return false;
    }
    }
  }
  }
  return true;
}
}

TEST(SM90_Device_Gemm_f16t_f16n_f32t_tensor_op_gmma_f32_cooperative_epilogue, 256x128x64_2x2x1_VoidC_VoidD_AuxStoreF16_RowMajor) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using TileShape_MNK = Shape<_256,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

  using EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<
    TileShape_MNK, EpilogueTileType, cutlass::half_t, cutlass::half_t, EpilogueSchedule
  >;
  using AuxStoreDescriptor = cutlass::epilogue::collective::detail::AuxStoreDescriptor<
    EpilogueDescriptor, cutlass::layout::RowMajor, cutlass::half_t
  >;

  using namespace cutlass::epilogue::fusion;

  constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  constexpr bool has_c = false;

  using EVT_D = decltype(test::gemm::device::select_evt_d<cutlass::half_t, float, has_c>());
  using AuxStore = Sm90AuxStore<AuxStoreDescriptor::Stages, typename EpilogueDescriptor::EpilogueTile,
                     typename AuxStoreDescriptor::Element, RoundStyle,
                     typename AuxStoreDescriptor::Stride, typename AuxStoreDescriptor::SmemLayoutAtom,
                     typename AuxStoreDescriptor::CopyOpR2S>;

  constexpr auto select_kernel = [](auto has_c, auto has_d) {
    using FusionCallbacks =
        cute::conditional_t<decltype(has_d){}, EVT_D, Sm90EVT<AuxStore, EVT_D>>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_MNK, ClusterShape_MNK,
        EpilogueTileType,
        float, float,
        cute::conditional_t<decltype(has_c){}, cutlass::half_t, void>, LayoutC, 8,
        cute::conditional_t<decltype(has_d){}, cutlass::half_t, void>, LayoutC, 8,
        EpilogueSchedule,
        FusionCallbacks
      >::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        cutlass::half_t, LayoutA, 8,
        cutlass::half_t, LayoutB, 8,
        float,
        TileShape_MNK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative
      >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue>;

    return GemmKernel{};
  };

  using GemmKernel = decltype(select_kernel(cute::C<has_c>{}, cute::C<true>{}));
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using GemmKernelWithoutD = decltype(select_kernel(cute::C<has_c>{}, cute::C<false>{}));
  using GemmWithoutD = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelWithoutD>;

  bool passed = test::gemm::device::testEVTAuxStoreWithoutD<Gemm, GemmWithoutD>();

  EXPECT_TRUE(passed);
}

TEST(SM90_Device_Gemm_f16t_f16n_f32t_tensor_op_gmma_f32_cooperative_epilogue, 256x128x64_2x2x1_VoidC_VoidD_AuxStoreNoSmemF16_RowMajor) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using TileShape_MNK = Shape<_256,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

  using EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<
    TileShape_MNK, EpilogueTileType, cutlass::half_t, cutlass::half_t, EpilogueSchedule
  >;

  using namespace cutlass::epilogue::fusion;

  constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  constexpr bool has_c = false;

  using EVT_D = decltype(test::gemm::device::select_evt_d<cutlass::half_t, float, has_c>());
  using AuxStore = Sm90AuxStore<0, void, cutlass::half_t, RoundStyle, cutlass::layout::RowMajor, void, void>;

  constexpr auto select_kernel = [](auto has_c, auto has_d) {
    using FusionCallbacks =
        cute::conditional_t<decltype(has_d){}, EVT_D, Sm90EVT<AuxStore, EVT_D>>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_MNK, ClusterShape_MNK,
        EpilogueTileType,
        float, float,
        cute::conditional_t<decltype(has_c){}, cutlass::half_t, void>, LayoutC, 8,
        cute::conditional_t<decltype(has_d){}, cutlass::half_t, void>, LayoutC, 8,
        EpilogueSchedule,
        FusionCallbacks
      >::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        cutlass::half_t, LayoutA, 8,
        cutlass::half_t, LayoutB, 8,
        float,
        TileShape_MNK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative
      >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue>;

    return GemmKernel{};
  };

  using GemmKernel = decltype(select_kernel(cute::C<has_c>{}, cute::C<true>{}));
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using GemmKernelWithoutD = decltype(select_kernel(cute::C<has_c>{}, cute::C<false>{}));
  using GemmWithoutD = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelWithoutD>;

  bool passed = test::gemm::device::testEVTAuxStoreWithoutD<Gemm, GemmWithoutD>();

  EXPECT_TRUE(passed);
}

TEST(SM90_Device_Gemm_f16t_f16n_f32n_tensor_op_gmma_f32_cooperative_epilogue, 256x128x64_2x2x1_VoidC_VoidD_AuxStoreF16_ColumnMajor) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_256,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

  using EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<
    TileShape_MNK, EpilogueTileType, cutlass::half_t, cutlass::half_t, EpilogueSchedule
  >;
  using AuxStoreDescriptor = cutlass::epilogue::collective::detail::AuxStoreDescriptor<
    EpilogueDescriptor, cutlass::layout::ColumnMajor, cutlass::half_t
  >;

  using namespace cutlass::epilogue::fusion;

  constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  constexpr bool has_c = false;

  using EVT_D = decltype(test::gemm::device::select_evt_d<cutlass::half_t, float, has_c>());
  using AuxStore = Sm90AuxStore<AuxStoreDescriptor::Stages, typename EpilogueDescriptor::EpilogueTile,
                     typename AuxStoreDescriptor::Element, RoundStyle,
                     typename AuxStoreDescriptor::Stride, typename AuxStoreDescriptor::SmemLayoutAtom,
                     typename AuxStoreDescriptor::CopyOpR2S>;

  constexpr auto select_kernel = [](auto has_c, auto has_d) {
    using FusionCallbacks =
        cute::conditional_t<decltype(has_d){}, EVT_D, Sm90EVT<AuxStore, EVT_D>>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_MNK, ClusterShape_MNK,
        EpilogueTileType,
        float, float,
        cute::conditional_t<decltype(has_c){}, cutlass::half_t, void>, LayoutC, 8,
        cute::conditional_t<decltype(has_d){}, cutlass::half_t, void>, LayoutC, 8,
        EpilogueSchedule,
        FusionCallbacks
      >::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        cutlass::half_t, LayoutA, 8,
        cutlass::half_t, LayoutB, 8,
        float,
        TileShape_MNK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative
      >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue>;

    return GemmKernel{};
  };

  using GemmKernel = decltype(select_kernel(cute::C<has_c>{}, cute::C<true>{}));
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using GemmKernelWithoutD = decltype(select_kernel(cute::C<has_c>{}, cute::C<false>{}));
  using GemmWithoutD = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelWithoutD>;

  bool passed = test::gemm::device::testEVTAuxStoreWithoutD<Gemm, GemmWithoutD>();

  EXPECT_TRUE(passed);
}

TEST(SM90_Device_Gemm_f16t_f16n_f32t_tensor_op_gmma_f32_cooperative_epilogue, 128x128x64_2x2x1_VoidC_VoidD_AuxStoreF32_RowMajor) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using TileShape_MNK = Shape<_128,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

  using EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<
    TileShape_MNK, EpilogueTileType, cutlass::half_t, cutlass::half_t, EpilogueSchedule
  >;
  using AuxStoreDescriptor = cutlass::epilogue::collective::detail::AuxStoreDescriptor<
    EpilogueDescriptor, cutlass::layout::RowMajor, cutlass::half_t
  >;

  using namespace cutlass::epilogue::fusion;

  constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  constexpr bool has_c = false;

  using EVT_D = decltype(test::gemm::device::select_evt_d<cutlass::half_t, float, has_c>());
  using AuxStore = Sm90AuxStore<AuxStoreDescriptor::Stages, typename EpilogueDescriptor::EpilogueTile,
                     typename AuxStoreDescriptor::Element, RoundStyle,
                     typename AuxStoreDescriptor::Stride, typename AuxStoreDescriptor::SmemLayoutAtom,
                     typename AuxStoreDescriptor::CopyOpR2S>;

  constexpr auto select_kernel = [](auto has_c, auto has_d) {
    using FusionCallbacks =
        cute::conditional_t<decltype(has_d){}, EVT_D, Sm90EVT<AuxStore, EVT_D>>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_MNK, ClusterShape_MNK,
        EpilogueTileType,
        float, float,
        cute::conditional_t<decltype(has_c){}, cutlass::half_t, void>, LayoutC, 8,
        cute::conditional_t<decltype(has_d){}, cutlass::half_t, void>, LayoutC, 8,
        EpilogueSchedule,
        FusionCallbacks
      >::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        cutlass::half_t, LayoutA, 8,
        cutlass::half_t, LayoutB, 8,
        float,
        TileShape_MNK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative
      >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue>;

    return GemmKernel{};
  };

  using GemmKernel = decltype(select_kernel(cute::C<has_c>{}, cute::C<true>{}));
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using GemmKernelWithoutD = decltype(select_kernel(cute::C<has_c>{}, cute::C<false>{}));
  using GemmWithoutD = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelWithoutD>;

  bool passed = test::gemm::device::testEVTAuxStoreWithoutD<Gemm, GemmWithoutD>();

  EXPECT_TRUE(passed);
}

TEST(SM90_Device_Gemm_f16t_f16n_f32t_tensor_op_gmma_f32_cooperative_epilogue, 256x128x64_2x2x1_WithC_VoidD_AuxStoreF16_RowMajor) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using TileShape_MNK = Shape<_256,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

  using EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<
    TileShape_MNK, EpilogueTileType, cutlass::half_t, cutlass::half_t, EpilogueSchedule
  >;
  using AuxStoreDescriptor = cutlass::epilogue::collective::detail::AuxStoreDescriptor<
    EpilogueDescriptor, cutlass::layout::RowMajor, cutlass::half_t
  >;

  using namespace cutlass::epilogue::fusion;

  constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  constexpr bool has_c = true;

  using EVT_D = decltype(test::gemm::device::select_evt_d<cutlass::half_t, float, has_c>());
  using AuxStore = Sm90AuxStore<AuxStoreDescriptor::Stages, typename EpilogueDescriptor::EpilogueTile,
                     typename AuxStoreDescriptor::Element, RoundStyle,
                     typename AuxStoreDescriptor::Stride, typename AuxStoreDescriptor::SmemLayoutAtom,
                     typename AuxStoreDescriptor::CopyOpR2S>;

  constexpr auto select_kernel = [](auto has_c, auto has_d) {
    using FusionCallbacks =
        cute::conditional_t<decltype(has_d){}, EVT_D, Sm90EVT<AuxStore, EVT_D>>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_MNK, ClusterShape_MNK,
        EpilogueTileType,
        float, float,
        cute::conditional_t<decltype(has_c){}, cutlass::half_t, void>, LayoutC, 8,
        cute::conditional_t<decltype(has_d){}, cutlass::half_t, void>, LayoutC, 8,
        EpilogueSchedule,
        FusionCallbacks
      >::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        cutlass::half_t, LayoutA, 8,
        cutlass::half_t, LayoutB, 8,
        float,
        TileShape_MNK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative
      >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue>;

    return GemmKernel{};
  };

  using GemmKernel = decltype(select_kernel(cute::C<has_c>{}, cute::C<true>{}));
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using GemmKernelWithoutD = decltype(select_kernel(cute::C<has_c>{}, cute::C<false>{}));
  using GemmWithoutD = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelWithoutD>;

  bool passed = test::gemm::device::testEVTAuxStoreWithoutD<Gemm, GemmWithoutD>();

  EXPECT_TRUE(passed);
}

TEST(SM90_Device_Gemm_f16t_f16n_f32n_tensor_op_gmma_f32_cooperative_epilogue, 256x128x64_2x2x1_WithC_VoidD_AuxStoreF16_ColumnMajor) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_256,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

  using EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<
    TileShape_MNK, EpilogueTileType, cutlass::half_t, cutlass::half_t, EpilogueSchedule
  >;
  using AuxStoreDescriptor = cutlass::epilogue::collective::detail::AuxStoreDescriptor<
    EpilogueDescriptor, cutlass::layout::ColumnMajor, cutlass::half_t
  >;

  using namespace cutlass::epilogue::fusion;

  constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  constexpr bool has_c = true;

  using EVT_D = decltype(test::gemm::device::select_evt_d<cutlass::half_t, float, has_c>());
  using AuxStore = Sm90AuxStore<AuxStoreDescriptor::Stages, typename EpilogueDescriptor::EpilogueTile,
                     typename AuxStoreDescriptor::Element, RoundStyle,
                     typename AuxStoreDescriptor::Stride, typename AuxStoreDescriptor::SmemLayoutAtom,
                     typename AuxStoreDescriptor::CopyOpR2S>;

  constexpr auto select_kernel = [](auto has_c, auto has_d) {
    using FusionCallbacks =
        cute::conditional_t<decltype(has_d){}, EVT_D, Sm90EVT<AuxStore, EVT_D>>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_MNK, ClusterShape_MNK,
        EpilogueTileType,
        float, float,
        cute::conditional_t<decltype(has_c){}, cutlass::half_t, void>, LayoutC, 8,
        cute::conditional_t<decltype(has_d){}, cutlass::half_t, void>, LayoutC, 8,
        EpilogueSchedule,
        FusionCallbacks
      >::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        cutlass::half_t, LayoutA, 8,
        cutlass::half_t, LayoutB, 8,
        float,
        TileShape_MNK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative
      >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue>;

    return GemmKernel{};
  };

  using GemmKernel = decltype(select_kernel(cute::C<has_c>{}, cute::C<true>{}));
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using GemmKernelWithoutD = decltype(select_kernel(cute::C<has_c>{}, cute::C<false>{}));
  using GemmWithoutD = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelWithoutD>;

  bool passed = test::gemm::device::testEVTAuxStoreWithoutD<Gemm, GemmWithoutD>();

  EXPECT_TRUE(passed);
}

TEST(SM90_Device_Gemm_f16t_f16n_f32t_tensor_op_gmma_f32_cooperative_epilogue, 128x128x64_2x2x1_WithC_VoidD_AuxStoreF32_RowMajor) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using TileShape_MNK = Shape<_128,_128,_64>;
  using ClusterShape_MNK = Shape<_2,_2,_1>;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

  using EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<
    TileShape_MNK, EpilogueTileType, cutlass::half_t, cutlass::half_t, EpilogueSchedule
  >;
  using AuxStoreDescriptor = cutlass::epilogue::collective::detail::AuxStoreDescriptor<
    EpilogueDescriptor, cutlass::layout::RowMajor, cutlass::half_t
  >;

  using namespace cutlass::epilogue::fusion;

  constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  constexpr bool has_c = true;

  using EVT_D = decltype(test::gemm::device::select_evt_d<cutlass::half_t, float, has_c>());
  using AuxStore = Sm90AuxStore<AuxStoreDescriptor::Stages, typename EpilogueDescriptor::EpilogueTile,
                     typename AuxStoreDescriptor::Element, RoundStyle,
                     typename AuxStoreDescriptor::Stride, typename AuxStoreDescriptor::SmemLayoutAtom,
                     typename AuxStoreDescriptor::CopyOpR2S>;

  constexpr auto select_kernel = [](auto has_c, auto has_d) {
    using FusionCallbacks =
        cute::conditional_t<decltype(has_d){}, EVT_D, Sm90EVT<AuxStore, EVT_D>>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_MNK, ClusterShape_MNK,
        EpilogueTileType,
        float, float,
        cute::conditional_t<decltype(has_c){}, cutlass::half_t, void>, LayoutC, 8,
        cute::conditional_t<decltype(has_d){}, cutlass::half_t, void>, LayoutC, 8,
        EpilogueSchedule,
        FusionCallbacks
      >::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        cutlass::half_t, LayoutA, 8,
        cutlass::half_t, LayoutB, 8,
        float,
        TileShape_MNK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative
      >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue>;

    return GemmKernel{};
  };

  using GemmKernel = decltype(select_kernel(cute::C<has_c>{}, cute::C<true>{}));
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using GemmKernelWithoutD = decltype(select_kernel(cute::C<has_c>{}, cute::C<false>{}));
  using GemmWithoutD = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelWithoutD>;

  bool passed = test::gemm::device::testEVTAuxStoreWithoutD<Gemm, GemmWithoutD>();

  EXPECT_TRUE(passed);
}
#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
