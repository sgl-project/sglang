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


#include "cutlass/conv/kernel/default_conv3d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "conv3d_testbed.h"

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////
TEST(SM80_Device_Conv3d_Fprop_Analytic_ImplicitGemm_f32ndhwc_f32ndhwc_f32ndhwc_simt_f32,
  128x128_8x4_32x64x8) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = float;
  using ElementB           = float;
  using ElementC           = float;
  using ElementAccumulator = float;
  using ElementCompute     = float;


  /// Device-level Conv3d instance
  using Conv3dFpropKernel = typename cutlass::conv::kernel::DefaultConv3dFprop<
    ElementA, 
    cutlass::layout::TensorNDHWC,
    ElementB, 
    cutlass::layout::TensorNDHWC,
    ElementC, 
    cutlass::layout::TensorNDHWC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>, 
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv3dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv3dFpropKernel>;

  /// Run all unit test sizes with device-level Conv3d instance
  EXPECT_TRUE(test::conv::device::TestAllConv3d<Conv3dFprop>());

}

////////////////////////////////////////////////////////////////////////////////
TEST(SM80_Device_Conv3d_Fprop_Optimized_ImplicitGemm_f32ndhwc_f32ndhwc_f32ndhwc_simt_f32,
  128x128_8x4_64x32x8) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = float;
  using ElementB           = float;
  using ElementC           = float;
  using ElementAccumulator = float;
  using ElementCompute     = float;


  /// Device-level Conv3d instance
  using Conv3dFpropKernel = typename cutlass::conv::kernel::DefaultConv3dFprop<
    ElementA, 
    cutlass::layout::TensorNDHWC,
    ElementB, 
    cutlass::layout::TensorNDHWC,
    ElementC, 
    cutlass::layout::TensorNDHWC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<64, 32, 8>, 
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using Conv3dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv3dFpropKernel>;

  /// Run all unit test sizes with device-level Conv3d instance
  EXPECT_TRUE(test::conv::device::TestAllConv3d<Conv3dFprop>());

}

////////////////////////////////////////////////////////////////////////////////
#endif  // CUTLASS_ARCH_MMA_SM80_SUPPORTED
