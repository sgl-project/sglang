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


#include "../../../common/cutlass_unit_test.h"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "../gemm_testbed_3x.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// 1.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_128x64x128_1x1x1_0_tnn_align32_1sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_128, cute::_64, cute::_128>,
            cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized1Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            cutlass::half_t,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_128, cute::_64, cute::_128>,
            cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized1SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 2.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_128x128x128_1x1x1_0_tnn_align32_1sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_128, cute::_128, cute::_128>,
            cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized1Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            cutlass::half_t,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_128, cute::_128, cute::_128>,
            cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized1SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 3.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_128x192x128_1x1x1_0_tnn_align32_1sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_128, cute::_192, cute::_128>,
            cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized1Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            cutlass::half_t,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_128, cute::_192, cute::_128>,
            cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized1SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 4.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_128x256x128_1x1x1_0_tnn_align32_1sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_128, cute::_256, cute::_128>,
            cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized1Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            cutlass::half_t,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_128, cute::_256, cute::_128>,
            cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized1SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 5.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x64x128_2x1x1_0_tnn_align32_2sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_256, cute::_64, cute::_128>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            cutlass::half_t,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_256, cute::_64, cute::_128>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 6.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x64x256_2x1x1_0_tnn_align32_2sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_256, cute::_64, cute::_256>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            cutlass::half_t,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_256, cute::_64, cute::_256>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 7.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x128x128_2x1x1_0_tnn_align32_2sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_256, cute::_128, cute::_128>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            cutlass::half_t,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_256, cute::_128, cute::_128>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 8.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x128x256_2x1x1_0_tnn_align32_2sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_256, cute::_128, cute::_256>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            cutlass::half_t,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_256, cute::_128, cute::_256>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 9.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x192x128_2x1x1_0_tnn_align32_2sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_256, cute::_192, cute::_128>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            cutlass::half_t,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_256, cute::_192, cute::_128>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 10.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x256x128_2x1x1_0_tnn_align32_2sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_256, cute::_256, cute::_128>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            cutlass::half_t,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_256, cute::_256, cute::_128>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 11.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x256x256_2x1x1_0_tnn_align32_2sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_256, cute::_256, cute::_256>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            cutlass::half_t,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_256, cute::_256, cute::_256>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 1.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_128x64x128_1x1x1_0_tnn_align32_1sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_128x64x128_1x1x1_0_tnn_align32_1sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 1,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

// 2.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_128x128x128_1x1x1_0_tnn_align32_1sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_128x128x128_1x1x1_0_tnn_align32_1sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 1,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

// 3.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_128x192x128_1x1x1_0_tnn_align32_1sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_128x192x128_1x1x1_0_tnn_align32_1sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 1,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

// 4.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_128x256x128_1x1x1_0_tnn_align32_1sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_128x256x128_1x1x1_0_tnn_align32_1sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 1,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

// 5.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x64x128_2x1x1_0_tnn_align32_2sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x64x128_2x1x1_0_tnn_align32_2sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 1,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

//6.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x64x256_2x1x1_0_tnn_align32_2sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x64x256_2x1x1_0_tnn_align32_2sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 1,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

// 7.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x128x128_2x1x1_0_tnn_align32_2sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x128x128_2x1x1_0_tnn_align32_2sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 1,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

// 8.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x128x256_2x1x1_0_tnn_align32_2sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x128x256_2x1x1_0_tnn_align32_2sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 1,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

// 9.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x192x128_2x1x1_0_tnn_align32_2sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x192x128_2x1x1_0_tnn_align32_2sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 1,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

// 10.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x256x128_2x1x1_0_tnn_align32_2sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x256x128_2x1x1_0_tnn_align32_2sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 1,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

// 11.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x256x256_2x1x1_0_tnn_align32_2sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_f16_f16_256x256x256_2x1x1_0_tnn_align32_2sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 1,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

// 1.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_128x64x128_1x1x1_0_tnn_align32_1sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_128, cute::_64, cute::_128>,
            cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            void, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized1Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            void,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_128, cute::_64, cute::_128>,
            cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized1SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 2.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_128x128x128_1x1x1_0_tnn_align32_1sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_128, cute::_128, cute::_128>,
            cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            void, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized1Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            void,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_128, cute::_128, cute::_128>,
            cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized1SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 3.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_128x192x128_1x1x1_0_tnn_align32_1sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_128, cute::_192, cute::_128>,
            cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            void, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized1Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            void,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_128, cute::_192, cute::_128>,
            cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized1SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 4.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_128x256x128_1x1x1_0_tnn_align32_1sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_128, cute::_256, cute::_128>,
            cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            void, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized1Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            void,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_128, cute::_256, cute::_128>,
            cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized1SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 5.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x64x128_2x1x1_0_tnn_align32_2sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_256, cute::_64, cute::_128>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            void, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            void,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_256, cute::_64, cute::_128>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 6.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x64x256_2x1x1_0_tnn_align32_2sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_256, cute::_64, cute::_256>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            void, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            void,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_256, cute::_64, cute::_256>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 7.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x128x128_2x1x1_0_tnn_align32_2sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_256, cute::_128, cute::_128>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            void, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            void,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_256, cute::_128, cute::_128>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 8.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x128x256_2x1x1_0_tnn_align32_2sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_256, cute::_128, cute::_256>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            void, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            void,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_256, cute::_128, cute::_256>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 9.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x192x128_2x1x1_0_tnn_align32_2sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_256, cute::_192, cute::_128>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            void, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            void,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_256, cute::_192, cute::_128>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 10.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x256x256_2x1x1_0_tnn_align32_2sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_256, cute::_256, cute::_256>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            void, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            void,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_256, cute::_256, cute::_256>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 11.
namespace cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x256x128_2x1x1_0_tnn_align32_2sm {

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cute::Shape<cute::_256, cute::_256, cute::_128>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            float, float,
            void, cutlass::layout::ColumnMajor, 16,
            cutlass::half_t, cutlass::layout::ColumnMajor, 16,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            cutlass::epilogue::fusion::LinearCombination<
            cutlass::half_t,
            float,
            void,
            float
            >
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassSparseTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 32,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cute::Shape<cute::_256, cute::_256, cute::_128>,
            cute::Shape<cute::_2, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
            cutlass::gemm::KernelSparseTmaWarpSpecialized2SmSm100
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}

// 1.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_128x64x128_1x1x1_0_tnn_align32_1sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_128x64x128_1x1x1_0_tnn_align32_1sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 0,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

// 2.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_128x128x128_1x1x1_0_tnn_align32_1sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_128x128x128_1x1x1_0_tnn_align32_1sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 0,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

// 3.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_128x192x128_1x1x1_0_tnn_align32_1sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_128x192x128_1x1x1_0_tnn_align32_1sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 0,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

// 4.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_128x256x128_1x1x1_0_tnn_align32_1sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_128x256x128_1x1x1_0_tnn_align32_1sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 0,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

// 5.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x64x128_2x1x1_0_tnn_align32_2sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x64x128_2x1x1_0_tnn_align32_2sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 0,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

// 6.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x64x256_2x1x1_0_tnn_align32_2sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x64x256_2x1x1_0_tnn_align32_2sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 0,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

// 7.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x128x128_2x1x1_0_tnn_align32_2sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x128x128_2x1x1_0_tnn_align32_2sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 0,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

// 8.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x128x256_2x1x1_0_tnn_align32_2sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x128x256_2x1x1_0_tnn_align32_2sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 0,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

// 9.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x192x128_2x1x1_0_tnn_align32_2sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x192x128_2x1x1_0_tnn_align32_2sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 0,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

// 10.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x256x128_2x1x1_0_tnn_align32_2sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x256x128_2x1x1_0_tnn_align32_2sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 0,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

// 11.
TEST(cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x256x256_2x1x1_0_tnn_align32_2sm, func_check)
{
  namespace gemm = cutlass3x_sm100_sptensorop_spgemm_e4m3_e4m3_f32_void_f16_256x256x256_2x1x1_0_tnn_align32_2sm;

  EXPECT_TRUE(test::gemm::device::TestSmall<gemm::Gemm>(
    1, 0,
    test::gemm::device::CheckEquality::RELATIVE,
    test::gemm::device::ScalarLoc::ON_DEVICE,
    test::gemm::device::VectorScale::ENABLED,
    {512}));
}

#endif
