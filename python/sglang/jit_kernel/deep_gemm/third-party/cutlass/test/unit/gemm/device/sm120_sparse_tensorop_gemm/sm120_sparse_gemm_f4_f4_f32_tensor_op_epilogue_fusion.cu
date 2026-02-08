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
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "../../../common/cutlass_unit_test.h"

#include "../gemm_testbed_3x.hpp"

#if (defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED))

using namespace cute;

///////////////////////////////////////////////////////////////////////////////

// D = relu(alpha * accum + beta * C + per-row bias)
// C: fp32
// Bias: bf16
// Acc: fp32
// Scale (alpha, beta): fp32
// D: fp32
namespace kernel_1 {
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutCTag = cutlass::layout::ColumnMajor;
  using LayoutDTag = cutlass::layout::ColumnMajor;
  
  using TileShape = Shape<_128,_64,_256>;  // M, N, K
  using ClusterShape = Shape<_1,_1,_1>;

  using ElementA = cutlass::float_e2m1_t;
  using ElementB = cutlass::float_e2m1_t;
  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementBias = cutlass::bfloat16_t;

  static constexpr int AlignmentA = 64 * 8 * 2 / cutlass::sizeof_bits<ElementA>::value; // Align to 64 bytes with sparse ratio 4:2.
  static constexpr int AlignmentB = 64 * 8 / cutlass::sizeof_bits<ElementB>::value; // Align to 64 bytes.
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using FusionOperation = cutlass::epilogue::fusion::LinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::ReLu, ElementD, ElementCompute, ElementBias, ElementAccumulator, ElementCompute>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutCTag, AlignmentC,
      float, LayoutDTag, AlignmentD,
      cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120,
      FusionOperation
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassSparseTensorOp,
      ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  template <typename T>
  struct dummy {
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  };
  using GemmKernel = typename dummy<void>::GemmKernel;
  using Gemm = typename dummy<void>::Gemm;

} // kernel_1

// D = alpha * accum + beta * C
// C: fp32
// Bias: bf16
// Acc: fp32
// Scale (alpha, beta): fp32
// Scale factor: fp8
// D: fp4
namespace kernel_2 {
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutCTag = cutlass::layout::ColumnMajor;
  using LayoutDTag = cutlass::layout::ColumnMajor;
  using LayoutSFDTag = cutlass::layout::ColumnMajor;

  using ElementA = cutlass::float_e2m1_t;
  using ElementB = cutlass::float_e2m1_t;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementBias = cutlass::bfloat16_t;
  using ElementC = cutlass::bfloat16_t;
  using ElementD = cutlass::float_e2m1_t;
  using ElementSF = cutlass::float_ue8m0_t;

  static constexpr int AlignmentA = 64 * 8 * 2 / cutlass::sizeof_bits<ElementA>::value; // Align to 64 bytes with sparse ratio 4:2.
  static constexpr int AlignmentB = 64 * 8 / cutlass::sizeof_bits<ElementB>::value;     // Align to 64 bytes.
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  constexpr int SFVectorSize = 64;

  using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
      SFVectorSize,
      ElementD, 
      ElementCompute, 
      ElementSF,
      LayoutSFDTag,
      ElementC
  >;

  using TileShape = Shape<_128,_128,_256>;  // M, N, K
  using ClusterShape = Shape<_1,_1,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120,
      FusionOperation
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassSparseTensorOp,
      ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelScheduleSparseF8f6f4Sm120
    >::CollectiveOp;

  template <typename T>
  struct dummy {
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  };
  using GemmKernel = typename dummy<void>::GemmKernel;
  using Gemm = typename dummy<void>::Gemm;
} // kernel_2

// D = alpha * accum + beta * C + per-column bias
// C: fp32
// Bias: bf16
// Acc: fp32
// Scale (alpha, beta): fp32
// Scale factor: fp8
// D: fp4
namespace kernel_3 {
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutCTag = cutlass::layout::ColumnMajor;
  using LayoutDTag = cutlass::layout::ColumnMajor;
  using LayoutSFDTag = cutlass::layout::ColumnMajor;

  using ElementA = cutlass::float_e2m1_t;
  using ElementB = cutlass::float_e2m1_t;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementBias = cutlass::bfloat16_t;
  using ElementC = cutlass::bfloat16_t;
  using ElementD = cutlass::float_e2m1_t;
  using ElementSF = cutlass::float_ue8m0_t;

  static constexpr int AlignmentA = 64 * 8 * 2 / cutlass::sizeof_bits<ElementA>::value; // Align to 64 bytes with sparse ratio 4:2.
  static constexpr int AlignmentB = 64 * 8 / cutlass::sizeof_bits<ElementB>::value;     // Align to 64 bytes.
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  constexpr int SFVectorSize = 32;

  using FusionOperation = cutlass::epilogue::fusion::LinCombPerColBiasBlockScaleFactor<
    SFVectorSize,
    ElementD,
    ElementCompute,
    ElementSF,
    LayoutSFDTag,
    ElementBias,
    ElementC
  >;

  using TileShape = Shape<_128,_128,_256>;  // M, N, K
  using ClusterShape = Shape<_1,_1,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120,
      FusionOperation
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassSparseTensorOp,
      ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelScheduleSparseF8f6f4Sm120
    >::CollectiveOp;

  template <typename T>
  struct dummy {
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  };
  using GemmKernel = typename dummy<void>::GemmKernel;
  using Gemm = typename dummy<void>::Gemm;
} // kernel_3

// D = relu(alpha * accum + beta * C + per-column bias)
// C: fp32
// Bias: bf16
// Acc: fp32
// Scale (alpha, beta): fp32
// Scale factor: fp8
// D: fp4
namespace kernel_4 {
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutCTag = cutlass::layout::ColumnMajor;
  using LayoutDTag = cutlass::layout::ColumnMajor;
  using LayoutSFDTag = cutlass::layout::ColumnMajor;

  using ElementA = cutlass::float_e2m1_t;
  using ElementB = cutlass::float_e2m1_t;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementBias = cutlass::bfloat16_t;
  using ElementC = cutlass::bfloat16_t;
  using ElementD = cutlass::float_e2m1_t;
  using ElementSF = cutlass::float_ue8m0_t;

  static constexpr int AlignmentA = 64 * 8 * 2 / cutlass::sizeof_bits<ElementA>::value; // Align to 64 bytes with sparse ratio 4:2.
  static constexpr int AlignmentB = 64 * 8 / cutlass::sizeof_bits<ElementB>::value;     // Align to 64 bytes.
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  constexpr int SFVectorSize = 64;

  using FusionOperation = cutlass::epilogue::fusion::LinCombPerColBiasEltActBlockScaleFactor<
    cutlass::epilogue::thread::ReLU,
    SFVectorSize,
    ElementD,
    ElementCompute,
    ElementSF,
    LayoutSFDTag,
    ElementBias,
    ElementC
  >;

  using TileShape = Shape<_128,_128,_256>;  // M, N, K
  using ClusterShape = Shape<_1,_1,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120,
      FusionOperation
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassSparseTensorOp,
      ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelScheduleSparseF8f6f4Sm120
    >::CollectiveOp;

  template <typename T>
  struct dummy {
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  };
  using GemmKernel = typename dummy<void>::GemmKernel;
  using Gemm = typename dummy<void>::Gemm;
} // kernel_4

// D = gelu(alpha * accum + beta * C + per-column bias)
// C: fp32
// Bias: bf16
// Acc: fp32
// Scale (alpha, beta): fp32
// Scale factor: fp8
// D: fp4
namespace kernel_5 {
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutCTag = cutlass::layout::ColumnMajor;
  using LayoutDTag = cutlass::layout::ColumnMajor;
  using LayoutSFDTag = cutlass::layout::ColumnMajor;

  using ElementA = cutlass::float_e2m1_t;
  using ElementB = cutlass::float_e2m1_t;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementBias = cutlass::bfloat16_t;
  using ElementC = cutlass::bfloat16_t;
  using ElementD = cutlass::float_e2m1_t;
  using ElementSF = cutlass::float_ue8m0_t;

  static constexpr int AlignmentA = 64 * 8 * 2 / cutlass::sizeof_bits<ElementA>::value; // Align to 64 bytes with sparse ratio 4:2.
  static constexpr int AlignmentB = 64 * 8 / cutlass::sizeof_bits<ElementB>::value;     // Align to 64 bytes.
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  constexpr int SFVectorSize = 32;

  using FusionOperation = cutlass::epilogue::fusion::LinCombPerColBiasEltActBlockScaleFactor<
    cutlass::epilogue::thread::GELU,
    SFVectorSize,
    ElementD,
    ElementCompute,
    ElementSF,
    LayoutSFDTag,
    ElementBias,
    ElementC
  >;

  using TileShape = Shape<_128,_128,_256>;  // M, N, K
  using ClusterShape = Shape<_1,_1,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120,
      FusionOperation
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassSparseTensorOp,
      ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelScheduleSparseF8f6f4Sm120
    >::CollectiveOp;

  template <class T>
  struct dummy {
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  };
  using GemmKernel = typename dummy<void>::GemmKernel;
  using Gemm = typename dummy<void>::Gemm;
} // kernel_5

// D = gelu(alpha * accum + beta * C + per-column bias)
// C: fp32
// Bias: bf16
// Acc: fp32
// Scale (alpha, beta): fp32
// Scale factor: fp8
// D: fp4 with SF VEC16
namespace kernel_6 {
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutCTag = cutlass::layout::ColumnMajor;
  using LayoutDTag = cutlass::layout::ColumnMajor;
  using LayoutSFDTag = cutlass::layout::ColumnMajor;

  using ElementA = cutlass::float_e2m1_t;
  using ElementB = cutlass::float_e2m1_t;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementBias = cutlass::bfloat16_t;
  using ElementC = cutlass::bfloat16_t;
  using ElementD = cutlass::float_e2m1_t;
  using ElementSF = cutlass::float_ue8m0_t;

  static constexpr int AlignmentA = 64 * 8 * 2 / cutlass::sizeof_bits<ElementA>::value; // Align to 64 bytes with sparse ratio 4:2.
  static constexpr int AlignmentB = 64 * 8 / cutlass::sizeof_bits<ElementB>::value;     // Align to 64 bytes.
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  constexpr int SFVectorSize = 16;

  using FusionOperation = cutlass::epilogue::fusion::LinCombPerColBiasEltActBlockScaleFactor<
    cutlass::epilogue::thread::GELU,
    SFVectorSize,
    ElementD,
    ElementCompute,
    ElementSF,
    LayoutSFDTag,
    ElementBias,
    ElementC
  >;

  using TileShape = Shape<_128,_128,_256>;  // M, N, K
  using ClusterShape = Shape<_1,_1,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120,
      FusionOperation
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassSparseTensorOp,
      ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelScheduleSparseF8f6f4Sm120
    >::CollectiveOp;

  template <class T>
  struct dummy {
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  };
  using GemmKernel = typename dummy<void>::GemmKernel;
  using Gemm = typename dummy<void>::Gemm;
} // kernel_6

// D = relu(alpha * accum + beta * C + per-row bias)
// C: fp32
// Bias: bf16
// Acc: fp32
// Scale (alpha, beta): fp32
// D: fp32
TEST(SM120_Device_Sparse_Gemm_fe2m1t_fe2m1n_f32n_tensor_op_f32, 128x64x256_per_row_bias_relu) {
  bool result = test::gemm::device::TestSmallFusion<kernel_1::Gemm, false /*force_legacy_epilogue*/, false /*apply_alignment_offset*/>(1.0, 0.5);
  EXPECT_TRUE(result);
}

// D = alpha * accum + beta * C
// C: fp32
// Bias: bf16
// Acc: fp32
// Scale (alpha, beta): fp32
// Scale factor: fp8
// D: fp4
TEST(SM120_Device_Sparse_Gemm_fe2m1t_fe2m1n_f32n_tensor_op_f4, 128x128x256_column_major) {
  bool result = test::gemm::device::TestSmallFusion<kernel_2::Gemm, false /*force_legacy_epilogue*/, false /*apply_alignment_offset*/>(1.0, 0.5);
  EXPECT_TRUE(result);
}

// D = alpha * accum + beta * C + per-column bias
// C: fp32
// Bias: bf16
// Acc: fp32
// Scale (alpha, beta): fp32
// Scale factor: fp8
// D: fp4
TEST(SM120_Device_Sparse_Gemm_fe2m1t_fe2m1n_f32n_tensor_op_f4, 128x128x256_column_major_bias) {
  bool result = test::gemm::device::TestSmallFusion<kernel_3::Gemm, false /*force_legacy_epilogue*/, false /*apply_alignment_offset*/>(1.0, 0.5);
  EXPECT_TRUE(result);
}

// D = relu(alpha * accum + beta * C + per-column bias)
// C: fp32
// Bias: bf16
// Acc: fp32
// Scale (alpha, beta): fp32
// Scale factor: fp8
// D: fp4
TEST(SM120_Device_Sparse_Gemm_fe2m1t_fe2m1n_f32n_tensor_op_f4, 128x128x256_column_major_bias_relu) {
  bool result = test::gemm::device::TestSmallFusion<kernel_4::Gemm, false /*force_legacy_epilogue*/, false /*apply_alignment_offset*/>(1.0, 0.5);
  EXPECT_TRUE(result);
}

// D = gelu(alpha * accum + beta * C + per-column bias)
// C: fp32
// Bias: bf16
// Acc: fp32
// Scale (alpha, beta): fp32
// Scale factor: fp8
// D: fp4
TEST(SM120_Device_Sparse_Gemm_fe2m1t_fe2m1n_f32n_tensor_op_f4, 128x128x256_column_major_bias_gelu) {
  bool result = test::gemm::device::TestSmallFusion<kernel_5::Gemm, false /*force_legacy_epilogue*/, false /*apply_alignment_offset*/>(1.0, 0.5);
  EXPECT_TRUE(result);
}

// D = gelu(alpha * accum + beta * C + per-column bias)
// C: fp32
// Bias: bf16
// Acc: fp32
// Scale (alpha, beta): fp32
// Scale factor: fp8
// D: fp4 SF VEC16
TEST(SM120_Device_Sparse_Gemm_fe2m1t_fe2m1n_f32n_tensor_op_f4, 128x128x256_column_major_sf_vec16_bias_gelu) {
  bool result = test::gemm::device::TestSmallFusion<kernel_6::Gemm, false /*force_legacy_epilogue*/, false /*apply_alignment_offset*/>(1.0, 0.5);
  EXPECT_TRUE(result);
}

#endif // (defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED))
