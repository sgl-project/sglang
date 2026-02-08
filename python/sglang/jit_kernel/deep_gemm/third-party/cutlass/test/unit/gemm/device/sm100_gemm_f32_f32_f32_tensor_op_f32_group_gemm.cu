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
    \brief Tests for device-wide Grouped GEMM interface
*/



#include <iostream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/sm70_epilogue_vectorized.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "../../common/cutlass_unit_test.h"

#include "gemm_testbed_3x_ptr_array.hpp"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

using namespace cute;

TEST(SM100_Device_Gemm_f32t_f32n_f32n_tensor_op_1sm_f32_group, 128x128x64_1x2x1) {
// A matrix configuration
using         ElementA    = float;                                // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)
// B matrix configuration
using         ElementB    = float;                                // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)
// C matrix configuration
using         ElementC    = float;                                // Element type for C matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// D matrix configuration
using         ElementD    = float;                                // Element type for D matrix operands
using         LayoutD     = cutlass::layout::ColumnMajor;                   // Layout type for D matrix operands
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of D matrix in units of elements (up to 16 bytes)
// Core kernel configurations
using ElementAccumulator  = float;                                           // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm100;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                  // Operator class tag
using MmaTileShape = Shape<_128,_64,_64>;
using ClusterShape = Shape<_1,_2,_1>;
using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;   // Kernel to launch
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;          // Epilogue to launch

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC *, AlignmentC,
    ElementD, LayoutD *, AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA *, AlignmentA,
    ElementB, LayoutB *, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cutlass::gemm::GroupProblemShape<Shape<int,int,int>>,
    CollectiveMainloop,
    CollectiveEpilogue
>;
  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(1.0, 2.0);
  EXPECT_TRUE(result);
}

TEST(SM100Only_Device_Gemm_f32t_f32n_f32n_tensor_op_1sm_f32_group, 128x64x64_1x2x1) {
// A matrix configuration
using         ElementA    = float;                                // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)
// B matrix configuration
using         ElementB    = float;                                // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)
// C matrix configuration
using         ElementC    = float;                                // Element type for C matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// D matrix configuration
using         ElementD    = float;                                // Element type for D matrix operands
using         LayoutD     = cutlass::layout::ColumnMajor;                   // Layout type for D matrix operands
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of D matrix in units of elements (up to 16 bytes)
// Core kernel configurations
using ElementAccumulator  = float;                                           // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm100;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                  // Operator class tag
using MmaTileShape = Shape<_128,_32,_64>;
using ClusterShape = Shape<_1,_2,_1>;
using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;   // Kernel to launch
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;          // Epilogue to launch

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC *, AlignmentC,
    ElementD, LayoutD *, AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA *, AlignmentA,
    ElementB, LayoutB *, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cutlass::gemm::GroupProblemShape<Shape<int,int,int>>,
    CollectiveMainloop,
    CollectiveEpilogue
>;
  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(3.0, 2.0);
  EXPECT_TRUE(result);
}

TEST(SM100Only_Device_Gemm_f32t_f32n_f32n_tensor_op_2sm_f32_group, 256x128x64_2x1x1) {
// A matrix configuration
using         ElementA    = float;                                // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)
// B matrix configuration
using         ElementB    = float;                                // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)
// C matrix configuration
using         ElementC    = float;                                // Element type for C matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// D matrix configuration
using         ElementD    = float;                                // Element type for D matrix operands
using         LayoutD     = cutlass::layout::ColumnMajor;                   // Layout type for D matrix operands
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of D matrix in units of elements (up to 16 bytes)
// Core kernel configurations
using ElementAccumulator  = float;                                           // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm100;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                  // Operator class tag
using MmaTileShape = Shape<_256,_128,_64>;
using ClusterShape = Shape<_2,_1,_1>;
using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;   // Kernel to launch
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;          // Epilogue to launch

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC *, AlignmentC,
    ElementD, LayoutD *, AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA *, AlignmentA,
    ElementB, LayoutB *, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cutlass::gemm::GroupProblemShape<Shape<int,int,int>>,
    CollectiveMainloop,
    CollectiveEpilogue
>;
  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(1.0, 0.0);
  EXPECT_TRUE(result);
}

TEST(SM100Only_Device_Gemm_f32t_f32n_f32t_tensor_op_2sm_f32_group, 256x256x64_2x2x1) {
// A matrix configuration
using         ElementA    = float;                                // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)
// B matrix configuration
using         ElementB    = float;                                // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)
// C matrix configuration
using         ElementC    = float;                                // Element type for C matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                   // Layout type for C matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// D matrix configuration
using         ElementD    = float;                                // Element type for D matrix operands
using         LayoutD     = cutlass::layout::RowMajor;                   // Layout type for D matrix operands
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of D matrix in units of elements (up to 16 bytes)
// Core kernel configurations
using ElementAccumulator  = float;                                           // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm100;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                  // Operator class tag
using MmaTileShape = Shape<_256,_128,_64>;
using ClusterShape = Shape<_2,_2,_1>;
using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;   // Kernel to launch
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;          // Epilogue to launch
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC *, AlignmentC,
    ElementD, LayoutD *, AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA *, AlignmentA,
    ElementB, LayoutB *, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cutlass::gemm::GroupProblemShape<Shape<int,int,int>>,
    CollectiveMainloop,
    CollectiveEpilogue
>;
  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(2.0, 2.0);
  EXPECT_TRUE(result);
}


TEST(SM100Only_Device_Gemm_f32t_f32t_f32n_tensor_op_1sm_f32_group, 64x128x64_1x1x1) {
// A matrix configuration
using         ElementA    = float;                                // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)
// B matrix configuration
using         ElementB    = float;                                // Element type for B matrix operand
using         LayoutB     = cutlass::layout::RowMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)
// C matrix configuration
using         ElementC    = float;                                // Element type for C matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// D matrix configuration
using         ElementD    = float;                                // Element type for D matrix operands
using         LayoutD     = cutlass::layout::ColumnMajor;                   // Layout type for D matrix operands
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of D matrix in units of elements (up to 16 bytes)
// Core kernel configurations
using ElementAccumulator  = float;                                           // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm100;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                  // Operator class tag
using MmaTileShape = Shape<_64,_128,_64>;
using ClusterShape = Shape<_1,_1,_1>;
using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;   // Kernel to launch
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;          // Epilogue to launch

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC *, AlignmentC,
    ElementD, LayoutD *, AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA *, AlignmentA,
    ElementB, LayoutB *, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cutlass::gemm::GroupProblemShape<Shape<int,int,int>>,
    CollectiveMainloop,
    CollectiveEpilogue
>;
  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(1.0, 2.0);
  EXPECT_TRUE(result);
}

TEST(SM100Only_Device_Gemm_f32n_f32n_f32n_tensor_op_1sm_f32_group, 128x128x64_1x2x1) {
// A matrix configuration
using         ElementA    = float;                                // Element type for A matrix operand
using         LayoutA     = cutlass::layout::ColumnMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)
// B matrix configuration
using         ElementB    = float;                                // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)
// C matrix configuration
using         ElementC    = float;                                // Element type for C matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// D matrix configuration
using         ElementD    = float;                                // Element type for D matrix operands
using         LayoutD     = cutlass::layout::ColumnMajor;                   // Layout type for D matrix operands
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of D matrix in units of elements (up to 16 bytes)
// Core kernel configurations
using ElementAccumulator  = float;                                           // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm100;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                  // Operator class tag
using MmaTileShape = Shape<_128,_64,_64>;
using ClusterShape = Shape<_1,_2,_1>;
using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;   // Kernel to launch
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;          // Epilogue to launch

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC *, AlignmentC,
    ElementD, LayoutD *, AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA *, AlignmentA,
    ElementB, LayoutB *, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cutlass::gemm::GroupProblemShape<Shape<int,int,int>>,
    CollectiveMainloop,
    CollectiveEpilogue
>;
  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(1.0, 2.0);
  EXPECT_TRUE(result);
}

TEST(SM100Only_Device_Gemm_f32n_f32t_f32n_tensor_op_1sm_f32_group, 128x64x64_1x2x1) {
// A matrix configuration
using         ElementA    = float;                                // Element type for A matrix operand
using         LayoutA     = cutlass::layout::ColumnMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)
// B matrix configuration
using         ElementB    = float;                                // Element type for B matrix operand
using         LayoutB     = cutlass::layout::RowMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)
// C matrix configuration
using         ElementC    = float;                                // Element type for C matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// D matrix configuration
using         ElementD    = float;                                // Element type for D matrix operands
using         LayoutD     = cutlass::layout::ColumnMajor;                   // Layout type for D matrix operands
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of D matrix in units of elements (up to 16 bytes)
// Core kernel configurations
using ElementAccumulator  = float;                                           // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm100;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                  // Operator class tag
using MmaTileShape = Shape<_128,_32,_64>;
using ClusterShape = Shape<_1,_2,_1>;
using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;   // Kernel to launch
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;          // Epilogue to launch

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC *, AlignmentC,
    ElementD, LayoutD *, AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA *, AlignmentA,
    ElementB, LayoutB *, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cutlass::gemm::GroupProblemShape<Shape<int,int,int>>,
    CollectiveMainloop,
    CollectiveEpilogue
>;
  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(3.0, 2.0);
  EXPECT_TRUE(result);
}

TEST(SM100Only_Device_Gemm_f32t_f32t_f32n_tensor_op_2sm_f32_group, 256x128x64_2x1x1) {
// A matrix configuration
using         ElementA    = float;                                // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)
// B matrix configuration
using         ElementB    = float;                                // Element type for B matrix operand
using         LayoutB     = cutlass::layout::RowMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)
// C matrix configuration
using         ElementC    = float;                                // Element type for C matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// D matrix configuration
using         ElementD    = float;                                // Element type for D matrix operands
using         LayoutD     = cutlass::layout::ColumnMajor;                   // Layout type for D matrix operands
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of D matrix in units of elements (up to 16 bytes)
// Core kernel configurations
using ElementAccumulator  = float;                                           // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm100;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                  // Operator class tag
using MmaTileShape = Shape<_256,_128,_64>;
using ClusterShape = Shape<_2,_1,_1>;
using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;   // Kernel to launch
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;          // Epilogue to launch

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC *, AlignmentC,
    ElementD, LayoutD *, AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA *, AlignmentA,
    ElementB, LayoutB *, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cutlass::gemm::GroupProblemShape<Shape<int,int,int>>,
    CollectiveMainloop,
    CollectiveEpilogue
>;
  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(1.0, 0.0);
  EXPECT_TRUE(result);
}

TEST(SM100Only_Device_Gemm_f32n_f32n_f32n_tensor_op_2sm_f32_group, 256x256x64_2x2x1) {
// A matrix configuration
using         ElementA    = float;                                // Element type for A matrix operand
using         LayoutA     = cutlass::layout::ColumnMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)
// B matrix configuration
using         ElementB    = float;                                // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)
// C matrix configuration
using         ElementC    = float;                                // Element type for C matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// D matrix configuration
using         ElementD    = float;                                // Element type for D matrix operands
using         LayoutD     = cutlass::layout::ColumnMajor;                   // Layout type for D matrix operands
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of D matrix in units of elements (up to 16 bytes)
// Core kernel configurations
using ElementAccumulator  = float;                                           // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm100;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                  // Operator class tag
using MmaTileShape = Shape<_256,_128,_64>;
using ClusterShape = Shape<_2,_2,_1>;
using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;   // Kernel to launch
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;          // Epilogue to launch
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC *, AlignmentC,
    ElementD, LayoutD *, AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA *, AlignmentA,
    ElementB, LayoutB *, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cutlass::gemm::GroupProblemShape<Shape<int,int,int>>,
    CollectiveMainloop,
    CollectiveEpilogue
>;
  using namespace test::gemm::device;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  bool result = TestSmall<Gemm>(2.0, 2.0);
  EXPECT_TRUE(result);
}

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
