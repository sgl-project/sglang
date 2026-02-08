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
    \brief Tests for device-wide GEMM interface (SGEMM)
*/

#include "cutlass/cutlass.h"

#include "cutlass/numeric_types.h"
#include "cutlass/arch/mma_sm100.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "../../common/cutlass_unit_test.h"

#include "gemm_testbed_3x.hpp"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////

// CTA tile shape: 128x128x16

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM100Only_Device_Gemm_f32n_f32t_f32n_simt_f32_align1, 128x128x16) {
  // NT layout
  using namespace cute;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = LayoutC;
  using ElementA = float;
  using ElementB = float;
  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using TileShape = Shape<_128, _128, _16>;
  using ClusterShape = Shape<_1, _1, _1>;
  static constexpr int kAlignmentA = 1;
  static constexpr int kAlignmentB = 1;
  static constexpr int kAlignmentC = 1;
  static constexpr int kAlignmentD = 1;

  // Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassSimt,
    ElementA, LayoutA, kAlignmentA,
    ElementB, LayoutB, kAlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCount<3>,
    cutlass::gemm::KernelMultistage
  >::CollectiveOp;

  // Epilogue
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassSimt,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    LayoutC,
    kAlignmentC,
    ElementD,
    LayoutD,
    kAlignmentD,
    cutlass::epilogue::EpilogueSimtVectorized
  >::CollectiveOp;

  // Kernel
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = test::gemm::device::TestSmall<Gemm, true>();
  EXPECT_TRUE(result);
}

TEST(SM100Only_Device_Gemm_f32t_f32n_f32n_simt_f32_align1, 128x128x16) {
  // TN layout
  using namespace cute;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = LayoutC;
  using ElementA = float;
  using ElementB = float;
  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using TileShape = Shape<_128, _128, _16>;
  using ClusterShape = Shape<_1, _1, _1>;
  static constexpr int kAlignmentA = 1;
  static constexpr int kAlignmentB = 1;
  static constexpr int kAlignmentC = 1;
  static constexpr int kAlignmentD = 1;

  // Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassSimt,
    ElementA, LayoutA, kAlignmentA,
    ElementB, LayoutB, kAlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCount<3>,
    cutlass::gemm::KernelMultistage
  >::CollectiveOp;

  // Epilogue
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassSimt,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    LayoutC,
    kAlignmentC,
    ElementD,
    LayoutD,
    kAlignmentD,
    cutlass::epilogue::EpilogueSimtVectorized
  >::CollectiveOp;

  // Kernel
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = test::gemm::device::TestSmall<Gemm, true>();
  EXPECT_TRUE(result);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// CTA tile shape: 64x32x16

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM100Only_Device_Gemm_f32n_f32n_f32n_simt_f32_align1, 64x32x16) {
  // NN layout
  using namespace cute;
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = LayoutC;
  using ElementA = float;
  using ElementB = float;
  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using TileShape = Shape<_64, _32, _16>;
  using ClusterShape = Shape<_1, _1, _1>;
  static constexpr int kAlignmentA = 1;
  static constexpr int kAlignmentB = 1;
  static constexpr int kAlignmentC = 1;
  static constexpr int kAlignmentD = 1;

  // Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassSimt,
    ElementA, LayoutA, kAlignmentA,
    ElementB, LayoutB, kAlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCount<3>,
    cutlass::gemm::KernelMultistage
  >::CollectiveOp;

  // Epilogue
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassSimt,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    LayoutC,
    kAlignmentC,
    ElementD,
    LayoutD,
    kAlignmentD,
    cutlass::epilogue::EpilogueSimtVectorized
  >::CollectiveOp;

  // Kernel
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = test::gemm::device::TestSmall<Gemm, true>();
  EXPECT_TRUE(result);
}

TEST(SM100Only_Device_Gemm_f32t_f32t_f32n_simt_f32_align1, 64x32x16) {
  // TT layout
  using namespace cute;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = LayoutC;
  using ElementA = float;
  using ElementB = float;
  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using TileShape = Shape<_64, _32, _16>;
  using ClusterShape = Shape<_1, _1, _1>;
  static constexpr int kAlignmentA = 1;
  static constexpr int kAlignmentB = 1;
  static constexpr int kAlignmentC = 1;
  static constexpr int kAlignmentD = 1;

  // Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassSimt,
    ElementA, LayoutA, kAlignmentA,
    ElementB, LayoutB, kAlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCount<3>,
    cutlass::gemm::KernelMultistage
  >::CollectiveOp;

  // Epilogue
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassSimt,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    LayoutC,
    kAlignmentC,
    ElementD,
    LayoutD,
    kAlignmentD,
    cutlass::epilogue::EpilogueSimtVectorized
  >::CollectiveOp;

  // Kernel
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = test::gemm::device::TestSmall<Gemm, true>();
  EXPECT_TRUE(result);
}

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
