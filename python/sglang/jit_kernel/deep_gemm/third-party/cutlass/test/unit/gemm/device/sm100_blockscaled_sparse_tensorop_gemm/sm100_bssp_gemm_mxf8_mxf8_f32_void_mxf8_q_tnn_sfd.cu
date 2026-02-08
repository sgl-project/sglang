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

// * Test Config
// 1. layout {tnn_InputVs64_OutputVs64}
// 2. tilesize {128x128, 128x192, 128x256, 256x128, 256x192, 256x256}

// * Test list
// 1. 128x128_tnn_vs64in_vs64out
// 2. 128x192_tnn_vs64in_vs64out
// 3. 128x256_tnn_vs64in_vs64out
// 4. 256x128_tnn_vs64in_vs64out
// 5. 256x192_tnn_vs64in_vs64out
// 6. 256x256_tnn_vs64in_vs64out

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// 1. 
namespace cutlass3x_sm100_bssptensorop_bsspgemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_ue8m0xe4m3_128x128x256_0_vs64_tnn_align32_q_1sm_epiVs64n {

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::ColumnMajor;

    using ElementPairA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementPairB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementC = void;
    using ElementD = cutlass::float_e4m3_t;
    using ElementSF = cutlass::float_ue8m0_t;

    constexpr int kAlignmentA = 32;
    constexpr int kAlignmentB = 16;
    constexpr int kAlignmentD = 16;
    constexpr int kAlignmentC = 16;
    constexpr int SFDVectorSize = 64;

    using ProblemShape = Shape<int,int,int,int>;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using MmaTileShape = Shape<_128, _128, _256>;
    using ArchTag = cutlass::arch::Sm100;
    using OpClassTag = cutlass::arch::OpClassBlockScaledSparseTensorOp;
    using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
    using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecialized1SmMxf8f6f4;
    using KernelScheduleType = cutlass::gemm::KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100;
    using ElementAccumulator = float;
    using ElementEpilogueCompute = float;
    using ElementBias = cutlass::half_t;
    using TileScheduler = cutlass::gemm::PersistentScheduler;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            ArchTag,
            OpClassTag,
            MmaTileShape,
            ClusterShape,
            EpilogueTile,
            ElementAccumulator,
            ElementEpilogueCompute,
            ElementC, LayoutC, kAlignmentC,
            ElementD, LayoutD, kAlignmentD,
            EpilogueScheduleType,
            cutlass::epilogue::fusion::LinCombPerRowBiasEltActBlockScaleFactor<
                cutlass::epilogue::thread::Clamp,
                SFDVectorSize,
                ElementD,
                ElementEpilogueCompute,
                ElementSF,
                LayoutD,
                ElementBias,
                ElementC,
                ElementEpilogueCompute>
        >::CollectiveOp;

    using StageCount = cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            ArchTag,
            OpClassTag,
            ElementPairA, LayoutA, kAlignmentA,
            ElementPairB, LayoutB, kAlignmentB,
            ElementAccumulator,
            MmaTileShape,
            ClusterShape,
            StageCount,
            KernelScheduleType
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 2.
namespace cutlass3x_sm100_bssptensorop_bsspgemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_ue8m0xe4m3_128x192x256_0_vs64_tnn_align32_q_1sm_epiVs64n {

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::ColumnMajor;

    using ElementPairA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementPairB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementC = void;
    using ElementD = cutlass::float_e4m3_t;
    using ElementSF = cutlass::float_ue8m0_t;

    constexpr int kAlignmentA = 32;
    constexpr int kAlignmentB = 16;
    constexpr int kAlignmentD = 16;
    constexpr int kAlignmentC = 16;
    constexpr int SFDVectorSize = 64;

    using ProblemShape = Shape<int,int,int,int>;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using MmaTileShape = Shape<_128, _192, _256>;
    using ArchTag = cutlass::arch::Sm100;
    using OpClassTag = cutlass::arch::OpClassBlockScaledSparseTensorOp;
    using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
    using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecialized1SmMxf8f6f4;
    using KernelScheduleType = cutlass::gemm::KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100;
    using ElementAccumulator = float;
    using ElementEpilogueCompute = float;
    using ElementBias = cutlass::half_t;
    using TileScheduler = cutlass::gemm::PersistentScheduler;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            ArchTag,
            OpClassTag,
            MmaTileShape,
            ClusterShape,
            EpilogueTile,
            ElementAccumulator,
            ElementEpilogueCompute,
            ElementC, LayoutC, kAlignmentC,
            ElementD, LayoutD, kAlignmentD,
            EpilogueScheduleType,
            cutlass::epilogue::fusion::LinCombPerRowBiasEltActBlockScaleFactor<
                cutlass::epilogue::thread::Clamp,
                SFDVectorSize,
                ElementD,
                ElementEpilogueCompute,
                ElementSF,
                LayoutD,
                ElementBias,
                ElementC,
                ElementEpilogueCompute>
        >::CollectiveOp;

    using StageCount = cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            ArchTag,
            OpClassTag,
            ElementPairA, LayoutA, kAlignmentA,
            ElementPairB, LayoutB, kAlignmentB,
            ElementAccumulator,
            MmaTileShape,
            ClusterShape,
            StageCount,
            KernelScheduleType
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 3.
namespace cutlass3x_sm100_bssptensorop_bsspgemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_ue8m0xe4m3_128x256x256_0_vs64_tnn_align32_q_1sm_epiVs64n {

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::ColumnMajor;

    using ElementPairA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementPairB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementC = void;
    using ElementD = cutlass::float_e4m3_t;
    using ElementSF = cutlass::float_ue8m0_t;

    constexpr int kAlignmentA = 32;
    constexpr int kAlignmentB = 16;
    constexpr int kAlignmentD = 16;
    constexpr int kAlignmentC = 16;
    constexpr int SFDVectorSize = 64;

    using ProblemShape = Shape<int,int,int,int>;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using MmaTileShape = Shape<_128, _256, _256>;
    using ArchTag = cutlass::arch::Sm100;
    using OpClassTag = cutlass::arch::OpClassBlockScaledSparseTensorOp;
    using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
    using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecialized1SmMxf8f6f4;
    using KernelScheduleType = cutlass::gemm::KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100;
    using ElementAccumulator = float;
    using ElementEpilogueCompute = float;
    using ElementBias = cutlass::half_t;
    using TileScheduler = cutlass::gemm::PersistentScheduler;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            ArchTag,
            OpClassTag,
            MmaTileShape,
            ClusterShape,
            EpilogueTile,
            ElementAccumulator,
            ElementEpilogueCompute,
            ElementC, LayoutC, kAlignmentC,
            ElementD, LayoutD, kAlignmentD,
            EpilogueScheduleType,
            cutlass::epilogue::fusion::LinCombPerRowBiasEltActBlockScaleFactor<
                cutlass::epilogue::thread::Clamp,
                SFDVectorSize,
                ElementD,
                ElementEpilogueCompute,
                ElementSF,
                LayoutD,
                ElementBias,
                ElementC,
                ElementEpilogueCompute>
        >::CollectiveOp;

    using StageCount = cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            ArchTag,
            OpClassTag,
            ElementPairA, LayoutA, kAlignmentA,
            ElementPairB, LayoutB, kAlignmentB,
            ElementAccumulator,
            MmaTileShape,
            ClusterShape,
            StageCount,
            KernelScheduleType
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 4.
namespace cutlass3x_sm100_bssptensorop_bsspgemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_ue8m0xe4m3_256x128x256_0_vs64_tnn_align32_q_2sm_epiVs64n {

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::ColumnMajor;

    using ElementPairA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementPairB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementC = void;
    using ElementD = cutlass::float_e4m3_t;
    using ElementSF = cutlass::float_ue8m0_t;

    constexpr int kAlignmentA = 32;
    constexpr int kAlignmentB = 16;
    constexpr int kAlignmentD = 16;
    constexpr int kAlignmentC = 16;
    constexpr int SFDVectorSize = 64;

    using ProblemShape = Shape<int,int,int,int>;
    using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
    using MmaTileShape = Shape<_256, _128, _256>;
    using ArchTag = cutlass::arch::Sm100;
    using OpClassTag = cutlass::arch::OpClassBlockScaledSparseTensorOp;
    using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
    using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecialized2SmMxf8f6f4;
    using KernelScheduleType = cutlass::gemm::KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100;
    using ElementAccumulator = float;
    using ElementEpilogueCompute = float;
    using ElementBias = cutlass::half_t;
    using TileScheduler = cutlass::gemm::PersistentScheduler;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            ArchTag,
            OpClassTag,
            MmaTileShape,
            ClusterShape,
            EpilogueTile,
            ElementAccumulator,
            ElementEpilogueCompute,
            ElementC, LayoutC, kAlignmentC,
            ElementD, LayoutD, kAlignmentD,
            EpilogueScheduleType,
            cutlass::epilogue::fusion::LinCombPerRowBiasEltActBlockScaleFactor<
                cutlass::epilogue::thread::Clamp,
                SFDVectorSize,
                ElementD,
                ElementEpilogueCompute,
                ElementSF,
                LayoutD,
                ElementBias,
                ElementC,
                ElementEpilogueCompute>
        >::CollectiveOp;

    using StageCount = cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            ArchTag,
            OpClassTag,
            ElementPairA, LayoutA, kAlignmentA,
            ElementPairB, LayoutB, kAlignmentB,
            ElementAccumulator,
            MmaTileShape,
            ClusterShape,
            StageCount,
            KernelScheduleType
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 5.
namespace cutlass3x_sm100_bssptensorop_bsspgemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_ue8m0xe4m3_256x192x256_0_vs64_tnn_align32_q_2sm_epiVs64n {

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::ColumnMajor;

    using ElementPairA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementPairB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementC = void;
    using ElementD = cutlass::float_e4m3_t;
    using ElementSF = cutlass::float_ue8m0_t;

    constexpr int kAlignmentA = 32;
    constexpr int kAlignmentB = 16;
    constexpr int kAlignmentD = 16;
    constexpr int kAlignmentC = 16;
    constexpr int SFDVectorSize = 64;

    using ProblemShape = Shape<int,int,int,int>;
    using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
    using MmaTileShape = Shape<_256, _192, _256>;
    using ArchTag = cutlass::arch::Sm100;
    using OpClassTag = cutlass::arch::OpClassBlockScaledSparseTensorOp;
    using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
    using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecialized2SmMxf8f6f4;
    using KernelScheduleType = cutlass::gemm::KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100;
    using ElementAccumulator = float;
    using ElementEpilogueCompute = float;
    using ElementBias = cutlass::half_t;
    using TileScheduler = cutlass::gemm::PersistentScheduler;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            ArchTag,
            OpClassTag,
            MmaTileShape,
            ClusterShape,
            EpilogueTile,
            ElementAccumulator,
            ElementEpilogueCompute,
            ElementC, LayoutC, kAlignmentC,
            ElementD, LayoutD, kAlignmentD,
            EpilogueScheduleType,
            cutlass::epilogue::fusion::LinCombPerRowBiasEltActBlockScaleFactor<
                cutlass::epilogue::thread::Clamp,
                SFDVectorSize,
                ElementD,
                ElementEpilogueCompute,
                ElementSF,
                LayoutD,
                ElementBias,
                ElementC,
                ElementEpilogueCompute>
        >::CollectiveOp;

    using StageCount = cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            ArchTag,
            OpClassTag,
            ElementPairA, LayoutA, kAlignmentA,
            ElementPairB, LayoutB, kAlignmentB,
            ElementAccumulator,
            MmaTileShape,
            ClusterShape,
            StageCount,
            KernelScheduleType
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 6.
namespace cutlass3x_sm100_bssptensorop_bsspgemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_ue8m0xe4m3_256x256x256_0_vs64_tnn_align32_q_2sm_epiVs64n {

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::ColumnMajor;

    using ElementPairA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementPairB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementC = void;
    using ElementD = cutlass::float_e4m3_t;
    using ElementSF = cutlass::float_ue8m0_t;

    constexpr int kAlignmentA = 32;
    constexpr int kAlignmentB = 16;
    constexpr int kAlignmentD = 16;
    constexpr int kAlignmentC = 16;
    constexpr int SFDVectorSize = 64;

    using ProblemShape = Shape<int,int,int,int>;
    using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
    using MmaTileShape = Shape<_256, _256, _256>;
    using ArchTag = cutlass::arch::Sm100;
    using OpClassTag = cutlass::arch::OpClassBlockScaledSparseTensorOp;
    using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
    using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecialized2SmMxf8f6f4;
    using KernelScheduleType = cutlass::gemm::KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100;
    using ElementAccumulator = float;
    using ElementEpilogueCompute = float;
    using ElementBias = cutlass::half_t;
    using TileScheduler = cutlass::gemm::PersistentScheduler;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            ArchTag,
            OpClassTag,
            MmaTileShape,
            ClusterShape,
            EpilogueTile,
            ElementAccumulator,
            ElementEpilogueCompute,
            ElementC, LayoutC, kAlignmentC,
            ElementD, LayoutD, kAlignmentD,
            EpilogueScheduleType,
            cutlass::epilogue::fusion::LinCombPerRowBiasEltActBlockScaleFactor<
                cutlass::epilogue::thread::Clamp,
                SFDVectorSize,
                ElementD,
                ElementEpilogueCompute,
                ElementSF,
                LayoutD,
                ElementBias,
                ElementC,
                ElementEpilogueCompute>
        >::CollectiveOp;

    using StageCount = cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            ArchTag,
            OpClassTag,
            ElementPairA, LayoutA, kAlignmentA,
            ElementPairB, LayoutB, kAlignmentB,
            ElementAccumulator,
            MmaTileShape,
            ClusterShape,
            StageCount,
            KernelScheduleType
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 1.
TEST(cutlass3x_sm100_bssptensorop_bsspgemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_ue8m0xe4m3_128x128x256_0_vs64_tnn_align32_q_1sm_epiVs64n, sfd_fusion) {
  namespace gemm = cutlass3x_sm100_bssptensorop_bsspgemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_ue8m0xe4m3_128x128x256_0_vs64_tnn_align32_q_1sm_epiVs64n;
  EXPECT_TRUE(test::gemm::device::TestSmallFusion<gemm::Gemm>(
    1, 0,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {256, 2560}));
}

// 2.
TEST(cutlass3x_sm100_bssptensorop_bsspgemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_ue8m0xe4m3_128x192x256_0_vs64_tnn_align32_q_1sm_epiVs64n, sfd_fusion) {
  namespace gemm = cutlass3x_sm100_bssptensorop_bsspgemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_ue8m0xe4m3_128x192x256_0_vs64_tnn_align32_q_1sm_epiVs64n;
  EXPECT_TRUE(test::gemm::device::TestSmallFusion<gemm::Gemm>(
    1, 0,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {256, 2560}));
}

// 3.
TEST(cutlass3x_sm100_bssptensorop_bsspgemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_ue8m0xe4m3_128x256x256_0_vs64_tnn_align32_q_1sm_epiVs64n, sfd_fusion) {
  namespace gemm = cutlass3x_sm100_bssptensorop_bsspgemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_ue8m0xe4m3_128x256x256_0_vs64_tnn_align32_q_1sm_epiVs64n;
  EXPECT_TRUE(test::gemm::device::TestSmallFusion<gemm::Gemm>(
    1, 0,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {256, 2560}));
}

// 4.
TEST(cutlass3x_sm100_bssptensorop_bsspgemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_ue8m0xe4m3_256x128x256_0_vs64_tnn_align32_q_2sm_epiVs64n, sfd_fusion) {
  namespace gemm = cutlass3x_sm100_bssptensorop_bsspgemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_ue8m0xe4m3_256x128x256_0_vs64_tnn_align32_q_2sm_epiVs64n;
  EXPECT_TRUE(test::gemm::device::TestSmallFusion<gemm::Gemm>(
    1, 0,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {256, 2560}));
}

// 5.
TEST(cutlass3x_sm100_bssptensorop_bsspgemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_ue8m0xe4m3_256x192x256_0_vs64_tnn_align32_q_2sm_epiVs64n, sfd_fusion) {
  namespace gemm = cutlass3x_sm100_bssptensorop_bsspgemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_ue8m0xe4m3_256x192x256_0_vs64_tnn_align32_q_2sm_epiVs64n;
  EXPECT_TRUE(test::gemm::device::TestSmallFusion<gemm::Gemm>(
    1, 0,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {256, 2560}));
}

// 6.
TEST(cutlass3x_sm100_bssptensorop_bsspgemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_ue8m0xe4m3_256x256x256_0_vs64_tnn_align32_q_2sm_epiVs64n, sfd_fusion) {
  namespace gemm = cutlass3x_sm100_bssptensorop_bsspgemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_ue8m0xe4m3_256x256x256_0_vs64_tnn_align32_q_2sm_epiVs64n;
  EXPECT_TRUE(test::gemm::device::TestSmallFusion<gemm::Gemm>(
    1, 0,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {256, 2560}));
}

#endif
