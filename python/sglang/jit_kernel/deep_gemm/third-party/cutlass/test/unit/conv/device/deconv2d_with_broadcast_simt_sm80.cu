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
    \brief Tests for device-wide Implicit GEMM interface
*/

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"
#include "cutlass/epilogue/thread/linear_combination_residual_block.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/conv/kernel/default_deconv2d_with_broadcast.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "conv2d_with_broadcast_testbed.h"

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)


TEST(SM80_Device_Deconv2d_With_Broadcast_Analytic_ImplicitGemm_f32nhwc_f32nhwc_f32nhwc_simt_f32,
  128x128_32x2_64x64x32) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA = float;
  using ElementB = float;
  using ElementC = float;
  using ElementCompute = float;
  using ElementAccumulator = float;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    ElementC,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    ElementC,
    1,
    cutlass::epilogue::thread::ReLu<float>
  >;

  /// Device-level Conv2d instance
  using Deconv2dKernel = typename cutlass::conv::kernel::DefaultDeconv2dWithBroadcast<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementC, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueOutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic,
    cutlass::conv::StrideSupport::kUnity
  >::Kernel;

  using Deconv2d = cutlass::conv::device::ImplicitGemmConvolution<Deconv2dKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2dWithBroadcast<Deconv2d>());
}

// Test residual block fusion: UnaryOp(BinaryOp(ActivationOp(Conv2d(X) + bias), residual))
// LinearCombinationResidualBlock does not support the split-k mode unless ActivationOp is Identity.
// This is because the activation needs to be applied to the fully accumulated output of the Conv2d op,
// which only the last thread block would have an access to, before applying BinaryOp.
// The epilogue functor in the last thread block would have to be given three inputs, namely
// partial outputs, bias, and residual, but this is not supported in the current interface.
// Set TestSplitK = false to skip split-k tests with non-trivial ActivationOp.
template <
 template<typename T> class ActivationOp,
 template<typename T> class BinaryOp,
 template<typename T> class UnaryOp,
 bool TestSplitK = true
>
static void Deconv2dSM80TestResidualBlock() {
  using ElementA = float;
  using ElementB = float;
  using ElementC = float;
  using ElementD = ElementC;
  using ElementCompute = float;
  using ElementAccumulator = float;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationResidualBlock<
    ElementD,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    1,
    ActivationOp,
    BinaryOp,
    UnaryOp
  >;

  using Deconv2dKernel = typename cutlass::conv::kernel::DefaultDeconv2dWithBroadcast<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementC, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueOutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic,
    cutlass::conv::StrideSupport::kUnity
  >::Kernel;

  using Deconv2d = cutlass::conv::device::ImplicitGemmConvolution<Deconv2dKernel>;

  struct ReferenceOp {
    using OutputOp = typename Deconv2d::EpilogueOutputOp;
    using ElementZ = typename OutputOp::ElementZ;

    ActivationOp<ElementCompute> activation;
    BinaryOp<ElementCompute> binary_op;
    UnaryOp<ElementCompute> unary_op;

    void operator()(ElementZ &Z, ElementZ&, ElementCompute conv2d, ElementCompute residual) {
      Z = ElementZ(unary_op(binary_op(activation(conv2d), residual)));
    }
  };

  bool passed = test::conv::device::TestAllConv2dWithBroadcast<Deconv2d, ReferenceOp, true, TestSplitK>();
  EXPECT_TRUE(passed);
}

TEST(SM80_Device_Deconv2d_With_Residual_Block_Plus_Analytic_ImplicitGemm_f32nhwc_f32nhwc_f32nhwc_simt_f32,
     128x128_8x4_32x64x8) {
  // Resnet
  Deconv2dSM80TestResidualBlock<cutlass::epilogue::thread::Identity, cutlass::plus, cutlass::epilogue::thread::ReLu>();
}

////////////////////////////////////////////////////////////////////////////////

#endif  // CUTLASS_ARCH_MMA_SM80_SUPPORTED

////////////////////////////////////////////////////////////////////////////////
