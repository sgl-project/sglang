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

#include "cutlass/gemm/kernel/default_gemm_with_broadcast.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"
#include "cutlass/epilogue/thread/linear_combination_bias_relu.h"

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "testbed_gemm_with_broadcast.h"


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes:
///
///  Z = GEMM+Bias+ReLu
///  T = Relu conditional
///
template <typename Gemm>
struct GemmWithBiasReluReferenceOp {

  using OutputOp = typename Gemm::GemmKernel::Epilogue::OutputOp;

  using ElementCompute = typename OutputOp::ElementCompute;
  using ElementZ = typename OutputOp::ElementZ;
  using ElementT = typename OutputOp::ElementT;

  typename OutputOp::BinaryOp binary_op;
  typename OutputOp::ElementwiseOp elementwise_op;

  GemmWithBiasReluReferenceOp() { }

  void operator()(ElementZ &Z, ElementT &T, ElementCompute gemm, ElementCompute bias) {
    
    ElementCompute kThreshold = ElementCompute();

    ElementCompute z_full = binary_op(gemm, bias);
    
    bool conditional = (z_full >= kThreshold);

    if (!conditional) {
      z_full = kThreshold;
    }

    Z = ElementZ(z_full);
    T = ElementT(conditional);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_GemmWithBroadcast_GELU_f16n_f16n_f16n_tensor_op_f32, 128x128x32_64x64x8) {
  
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    cutlass::half_t,
    float,
    float,
    cutlass::half_t,
    cutlass::half_t,
    8,
    cutlass::epilogue::thread::GELU_taylor<float>
  >;

  using GemmKernel = 
    typename cutlass::gemm::kernel::DefaultGemmWithBroadcast<
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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  test::gemm::device::TestAllGemmWithBroadcast<Gemm>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM70_Device_GemmWithBroadcast_GELU_f16n_f16n_f16n_tensor_op_f32, 128x128x32_64x64x8) {
  
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    cutlass::half_t,
    float,
    float,
    cutlass::half_t,
    cutlass::half_t,
    8,
    cutlass::epilogue::thread::GELU_taylor<float>
  >;

  using GemmKernel = 
    typename cutlass::gemm::kernel::DefaultGemmWithBroadcast<
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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  test::gemm::device::TestAllGemmWithBroadcast<Gemm>();
}


/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_GemmWithBroadcast_RELU_f16n_f16n_f16n_tensor_op_f32, 128x128x32_64x64x8) {
  
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationBiasRelu<
    cutlass::half_t,
    float,
    float,
    cutlass::half_t,
    8,
    true
  >;

  using GemmKernel = 
    typename cutlass::gemm::kernel::DefaultGemmWithBroadcast<
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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  test::gemm::device::TestAllGemmWithBroadcast<Gemm, GemmWithBiasReluReferenceOp<Gemm> >();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM70_Device_GemmWithBroadcast_RELU_f16n_f16n_f16n_tensor_op_f32, 128x128x32_64x64x8) {
  
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationBiasRelu<
    cutlass::half_t,
    float,
    float,
    cutlass::half_t,
    8,
    true
  >;

  using GemmKernel = 
    typename cutlass::gemm::kernel::DefaultGemmWithBroadcast<
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
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  test::gemm::device::TestAllGemmWithBroadcast<Gemm, GemmWithBiasReluReferenceOp<Gemm> >();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif  // if defiend(CUTLASS_ARCH_MMA_SM75_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmWithBroadcast_GELU_f16n_f16n_f16n_tensor_op_f32, 128x128_32x5_64x64x32_16x8x16) {
  
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    cutlass::half_t,
    float,
    float,
    cutlass::half_t,
    cutlass::half_t,
    8,
    cutlass::epilogue::thread::GELU_taylor<float>
  >;

  using GemmKernel = 
    typename cutlass::gemm::kernel::DefaultGemmWithBroadcast<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
      cutlass::half_t, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      5,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  test::gemm::device::TestAllGemmWithBroadcast<Gemm>();
}

TEST(SM80_Device_GemmWithBroadcast_RELU_f16n_f16n_f16n_tensor_op_f32, 128x128_32x5_64x64x32_16x8x16) {
  
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationBiasRelu<
    cutlass::half_t,
    float,
    float,
    cutlass::half_t,
    8,
    true
  >;

  using GemmKernel = 
    typename cutlass::gemm::kernel::DefaultGemmWithBroadcast<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
      cutlass::half_t, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      5,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  test::gemm::device::TestAllGemmWithBroadcast<Gemm, GemmWithBiasReluReferenceOp<Gemm>>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmWithBroadcast_GELU_f16n_f16n_f16n_tensor_op_f32, 128x128_32x4_64x64x32_16x8x16) {
  
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    cutlass::half_t,
    float,
    float,
    cutlass::half_t,
    cutlass::half_t,
    8,
    cutlass::epilogue::thread::GELU_taylor<float>
  >;

  using GemmKernel = 
    typename cutlass::gemm::kernel::DefaultGemmWithBroadcast<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
      cutlass::half_t, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      4,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  test::gemm::device::TestAllGemmWithBroadcast<Gemm>();
}

TEST(SM80_Device_GemmWithBroadcast_RELU_f16n_f16n_f16n_tensor_op_f32, 128x128_32x4_64x64x32_16x8x16) {
  
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationBiasRelu<
    cutlass::half_t,
    float,
    float,
    cutlass::half_t,
    8,
    true
  >;

  using GemmKernel = 
    typename cutlass::gemm::kernel::DefaultGemmWithBroadcast<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
      cutlass::half_t, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      4,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  test::gemm::device::TestAllGemmWithBroadcast<Gemm, GemmWithBiasReluReferenceOp<Gemm>>();
}


/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmWithBroadcast_GELU_f16n_f16n_f16n_tensor_op_f32, 128x128_32x3_64x64x32_16x8x16) {
  
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    cutlass::half_t,
    float,
    float,
    cutlass::half_t,
    cutlass::half_t,
    8,
    cutlass::epilogue::thread::GELU_taylor<float>
  >;

  using GemmKernel = 
    typename cutlass::gemm::kernel::DefaultGemmWithBroadcast<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
      cutlass::half_t, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      3,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  test::gemm::device::TestAllGemmWithBroadcast<Gemm>();
}

TEST(SM80_Device_GemmWithBroadcast_RELU_f16n_f16n_f16n_tensor_op_f32, 128x128_32x3_64x64x32_16x8x16) {
  
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationBiasRelu<
    cutlass::half_t,
    float,
    float,
    cutlass::half_t,
    8,
    true
  >;

  using GemmKernel = 
    typename cutlass::gemm::kernel::DefaultGemmWithBroadcast<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
      cutlass::half_t, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      3,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  test::gemm::device::TestAllGemmWithBroadcast<Gemm, GemmWithBiasReluReferenceOp<Gemm> >();
}

TEST(SM80_Device_GemmWithBroadcast_RELU_f32n_f32n_f32n_tensor_op_f32, 64x64_16x10_32x32x16_16x8x8) {

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationBiasRelu<
    float,
    float,
    float,
    float,
    4,
    false
  >;

  using GemmKernel =
    typename cutlass::gemm::kernel::DefaultGemmWithBroadcast<
      float, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 4,    // transposed B operand
      float, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 4,    // transposed A operand
      float, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 16>,
      cutlass::gemm::GemmShape<32, 32, 16>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      10,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  test::gemm::device::TestAllGemmWithBroadcast<Gemm, GemmWithBiasReluReferenceOp<Gemm> >();
}
/////////////////////////////////////////////////////////////////////////////////////////////////

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
