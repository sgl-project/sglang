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
#include "cutlass/functional.h"

#include "cutlass/gemm/kernel/default_gemm_with_reduction.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/epilogue/thread/linear_combination_drelu.h"
#include "cutlass/epilogue/thread/linear_combination_dgelu.h"

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "testbed_gemm_with_reduction.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////

struct dReluLambda {
  float operator()(float d_y, float t) {
    if (t <= 0) {
      d_y = 0;
    }
    return d_y;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_GemmWithReduction_dReLU_bGrad_f16n_f16n_f16n_tensor_op_f32, 128x128x32_64x64x8) {
  
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationDRelu<
    float,
    float,
    cutlass::half_t,
    cutlass::half_t,
    8
  >;

  using GemmKernel = 
    typename cutlass::gemm::kernel::DefaultGemmWithReduction<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
      cutlass::half_t, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      EpilogueOutputOp,
      cutlass::plus<float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ReferenceOp = test::gemm::device::GemmWithReductionReference<
    Gemm, 
    dReluLambda
  >;

  test::gemm::device::TestGemmWithReduction<Gemm, ReferenceOp>(
    {520, 264, 96},
    cutlass::gemm::GemmUniversalMode::kGemm,
    2,
    float(1.25),
    float(2.25)
  );
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_GemmWithReduction_dReLU_bGrad_f16n_f16n_f16n_tensor_op_f32, 256x128x32_64x64x8) {
  
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationDRelu<
    float,
    float,
    cutlass::half_t,
    cutlass::half_t,
    8
  >;

  using GemmKernel = 
    typename cutlass::gemm::kernel::DefaultGemmWithReduction<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
      cutlass::half_t, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75,
      cutlass::gemm::GemmShape<256, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      EpilogueOutputOp,
      cutlass::plus<float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ReferenceOp = test::gemm::device::GemmWithReductionReference<
    Gemm, 
    dReluLambda
  >;

  test::gemm::device::TestGemmWithReduction<Gemm, ReferenceOp>(
    {520, 264, 96},
    cutlass::gemm::GemmUniversalMode::kGemm,
    1,
    float(1.25),
    float(2.25)
  );
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM70_Device_GemmWithReduction_dReLU_bGrad_f16n_f16n_f16n_tensor_op_f32, 128x128x32_64x64x8) {
  
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationDRelu<
    float,
    float,
    cutlass::half_t,
    cutlass::half_t,
    8
  >;

  using GemmKernel = 
    typename cutlass::gemm::kernel::DefaultGemmWithReduction<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
      cutlass::half_t, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm70,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<8, 8, 4>,
      EpilogueOutputOp,
      cutlass::plus<float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ReferenceOp = test::gemm::device::GemmWithReductionReference<
    Gemm, 
    dReluLambda
  >;
  
  test::gemm::device::TestGemmWithReduction<Gemm, ReferenceOp>(
    {520, 264, 96},
    cutlass::gemm::GemmUniversalMode::kGemm,
    2,
    float(1.25),
    float(2.25)
  );
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM70_Device_GemmWithReduction_dReLU_bGrad_f16n_f16n_f16n_tensor_op_f32, 256x128x32_64x64x8) {
  
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationDRelu<
    float,
    float,
    cutlass::half_t,
    cutlass::half_t,
    8
  >;

  using GemmKernel = 
    typename cutlass::gemm::kernel::DefaultGemmWithReduction<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
      cutlass::half_t, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm70,
      cutlass::gemm::GemmShape<256, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<8, 8, 4>,
      EpilogueOutputOp,
      cutlass::plus<float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ReferenceOp = test::gemm::device::GemmWithReductionReference<
    Gemm, 
    dReluLambda
  >;
  
  test::gemm::device::TestGemmWithReduction<Gemm, ReferenceOp>(
    {520, 264, 96},
    cutlass::gemm::GemmUniversalMode::kGemm,
    1,
    float(1.25),
    float(2.25)
  );
}

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace gemm {
namespace device {

template <typename Gemm>
struct Gemm_dReLU_packed_bits_reference_op {
  using ElementAccumulator = typename Gemm::ElementAccumulator;
  using ElementCompute = typename Gemm::GemmKernel::Epilogue::ElementCompute;
  using ElementC = typename Gemm::ElementC;
  using ElementT = typename Gemm::GemmKernel::Epilogue::ElementTensor;

  //
  // Methods
  //

  Gemm_dReLU_packed_bits_reference_op() { }

  ElementCompute operator()(
    ElementAccumulator d_y, 
    ElementT t) const {
    
    ElementCompute result = ElementCompute(d_y);

    bool cond = bool(t);
    if (!cond) {
      result = ElementCompute();
    }

    return result;
  }
};

} // namespace device
} // namespace gemm
} // namespace test


/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_GemmWithReduction_dReLU_conditional_bits_bGrad_f16n_f16n_f16n_tensor_op_f32, 128x128x32_64x64x8) {
  
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationDReluConditionalBits<
    float,
    float,
    cutlass::half_t,
    8
  >;

  using GemmKernel = 
    typename cutlass::gemm::kernel::DefaultGemmWithReduction<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
      cutlass::half_t, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      EpilogueOutputOp,
      cutlass::plus<float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ReferenceOp = test::gemm::device::Gemm_dReLU_packed_bits_reference_op<Gemm>;

  test::gemm::device::TestGemmWithReduction<Gemm, ReferenceOp>(
    {520, 264, 96},
    cutlass::gemm::GemmUniversalMode::kGemm,
    2,
    float(1.25),
    float(2.25)
  );
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM70_Device_GemmWithReduction_dReLU_conditional_bits_bGrad_f16n_f16n_f16n_tensor_op_f32, 128x128x32_64x64x8) {
  
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationDReluConditionalBits<
    float,
    float,
    cutlass::half_t,
    8
  >;

  using GemmKernel = 
    typename cutlass::gemm::kernel::DefaultGemmWithReduction<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
      cutlass::half_t, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm70,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<8, 8, 4>,
      EpilogueOutputOp,
      cutlass::plus<float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ReferenceOp = test::gemm::device::Gemm_dReLU_packed_bits_reference_op<Gemm>;

  test::gemm::device::TestGemmWithReduction<Gemm, ReferenceOp>(
    {520, 264, 96},
    cutlass::gemm::GemmUniversalMode::kGemm,
    2,
    float(1.25),
    float(2.25)
  );
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif  // if defiend(CUTLASS_ARCH_MMA_SM75_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
