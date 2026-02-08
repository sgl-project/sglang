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
    \brief Tests for device-wide GEMM interface
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"
#include "cutlass/arch/mma_sm100.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/epilogue/thread/activation.h"
#include "../../common/cutlass_unit_test.h"

#include "gemm_testbed_3x.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM103_SUPPORTED)

TEST(SM103_Device_Gemm_e2m1t_e2m1n_f32t_tensorop_1sm_f32_vs32, 512x256x768_4x2x1) {
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm103, cutlass::arch::OpClassBlockScaledTensorOp,
    cute::Shape<cute::_128, cute::_128, Int<768>>,
    cute::Shape<cute::_4, cute::_2, cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    void,  cutlass::layout::RowMajor, 4,
    float, cutlass::layout::RowMajor, 4,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm
  >::CollectiveOp;
  
  using CollectiveMainloop =  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm103, cutlass::arch::OpClassBlockScaledTensorOp,
    cute::tuple<cutlass::float_e2m1_t,cutlass::float_ue8m0_t>, cutlass::layout::RowMajor, 32,
    cute::tuple<cutlass::float_e2m1_t,cutlass::float_ue8m0_t>, cutlass::layout::ColumnMajor, 32,
    float,
    cute::Shape<cute::_128, cute::_128, Int<768>>,
    cute::Shape<cute::_4, cute::_2, cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs32Sm103
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
    >;

  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  auto pass = test::gemm::device::TestSmall<Gemm, false /*force_legacy_epilogue*/>(1.0, 0.0);
  EXPECT_TRUE(pass);
}


TEST(SM103_Device_Gemm_e2m1t_e2m1n_f32t_tensorop_1sm_f32_vs16, 512x256x768_4x2x1) {
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm103, cutlass::arch::OpClassBlockScaledTensorOp,
    cute::Shape<cute::_128, cute::_128, Int<768>>,
    cute::Shape<cute::_4, cute::_2, cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    void,  cutlass::layout::RowMajor, 4,
    float, cutlass::layout::RowMajor, 4,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm
  >::CollectiveOp;
  
  using CollectiveMainloop =  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm103, cutlass::arch::OpClassBlockScaledTensorOp,
    cute::tuple<cutlass::float_e2m1_t,cutlass::float_ue8m0_t>, cutlass::layout::RowMajor, 32,
    cute::tuple<cutlass::float_e2m1_t,cutlass::float_ue8m0_t>, cutlass::layout::ColumnMajor, 32,
    float,
    cute::Shape<cute::_128, cute::_128, Int<768>>,
    cute::Shape<cute::_4, cute::_2, cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
    >;

  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  auto pass = test::gemm::device::TestSmall<Gemm, false /*force_legacy_epilogue*/>(1.0, 0.0);
  EXPECT_TRUE(pass);
}
#endif // defined(CUTLASS_ARCH_MMA_SM103_SUPPORTED)
