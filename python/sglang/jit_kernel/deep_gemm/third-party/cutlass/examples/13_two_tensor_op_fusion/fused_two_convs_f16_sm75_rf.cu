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

#include <iostream>

#include "cutlass/cutlass.h"

#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "device/b2b_implicit_gemm_convolution.h"
#include "b2b_conv2d_run.h"
#include "test_run.h"

////////////////////////////////////////////////////////////////////////////////

cutlass::conv::Conv2dProblemSize conv2d_f16_sm75_problem_size_0 (
    {32, 56, 56, 64},    // input size (NHWC)
    {64, 3, 3, 64},   // filter size (KRSC)
    {1, 1, 1, 1},     // padding (pad_h, _, pad_w, _)
    {1, 1},           // stride (stride_h, stride_w)
    {1, 1},           // dilation (dilation_h, dilation_w)
    {32, 56, 56, 64}     // output size (NPQK)
  );
cutlass::conv::Conv2dProblemSize conv2d_f16_sm75_problem_size_1 (
    {32, 56, 56, 64},    // input size (NHWC)
    {128, 1, 1, 64},   // filter size (KRSC)
    {0, 0, 0, 0},     // padding (pad_h, _, pad_w, _)
    {1, 1},           // stride (stride_h, stride_w)
    {1, 1},           // dilation (dilation_h, dilation_w)
    {32, 56, 56, 128}     // output size (NPQK)
  );

bool run_nonfused_conv2d_fprop_optimized_f16_sm75() {

  using ElementA           = cutlass::half_t;
  using ElementB           = cutlass::half_t;
  using ElementC           = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute     = cutlass::half_t;

  ElementCompute alpha0 = ElementCompute(1);
  ElementCompute beta0 = ElementCompute(1); //beta=1 for bias
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(1); //beta=1 for bias

  using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape0 = cutlass::gemm::GemmShape<32, 32, 32>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape1 = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  using Conv2dFpropKernel0 = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementC, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ThreadblockShape0,
    WarpShape0,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using Conv2dFprop0 = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel0>;

  using Conv2dFpropKernel1 = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementC, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ThreadblockShape1,
    WarpShape1,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using Conv2dFprop1 = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel1>;

  B2bNonFusedConv2dRun<Conv2dFprop0, Conv2dFprop1> nonFusedConv2d;

  std::cout << "Running Non-fused back-to-back FP16 Optimized Convolution Fprops...\n";
  bool pass = nonFusedConv2d.run(conv2d_f16_sm75_problem_size_0, conv2d_f16_sm75_problem_size_1, cutlass::conv::SplitKMode::kSerial,
      alpha0, beta0, alpha1, beta1);

  if(pass)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return pass;
}

bool run_fused_conv2d_fprop_optimized_f16_sm75_rf_res() {

  using ElementA           = cutlass::half_t; 
  using ElementB           = cutlass::half_t; 
  using ElementC           = cutlass::half_t; 
  using ElementAccumulator = cutlass::half_t; 
  using ElementCompute     = cutlass::half_t; 

  ElementCompute alpha0 = ElementCompute(1);
  //Fused kernel has built-in bias, setting beta=0
  ElementCompute beta0 = ElementCompute(0); 
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(1); //use beta for bias

  using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape0 = cutlass::gemm::GemmShape<16, 64, 32>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<64, 128, 32>;
  using WarpShape1 = cutlass::gemm::GemmShape<16, 128, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  using EpilogueOutputOp0 = 
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementC,
      InstructionShape::kM * InstructionShape::kN / 32,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >;

  using EpilogueOutputOp1 = 
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;


  const bool SmemAccumulator = false;

  using B2bConv2dFpropKernel = typename cutlass::conv::kernel::DefaultB2bConv2dFprop<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementC, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ThreadblockShape0,
    ThreadblockShape1,
    WarpShape0,
    WarpShape1,
    InstructionShape,
    EpilogueOutputOp0,
    EpilogueOutputOp1,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    SmemAccumulator
  >::Kernel;
  
  using B2bConv2dFprop = cutlass::conv::device::B2bImplicitGemmConvolution<B2bConv2dFpropKernel>;

  B2bFusedConv2dRun<B2bConv2dFprop> fusedConv2d;

  std::cout << "Running Fused back-to-back FP16 Optimized Convolution Fprops with RF Residency...\n";
  bool pass = fusedConv2d.run(conv2d_f16_sm75_problem_size_0, conv2d_f16_sm75_problem_size_1, cutlass::conv::SplitKMode::kSerial,
      alpha0, beta0, alpha1, beta1);

  if(pass)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return pass;
}

int main() {

  std::vector<bool (*)()>funcs = {
    &run_nonfused_conv2d_fprop_optimized_f16_sm75,
    &run_fused_conv2d_fprop_optimized_f16_sm75_rf_res
  };

  return testRun(75, funcs, "conv f16 RF residency");

}

////////////////////////////////////////////////////////////////////////////////
