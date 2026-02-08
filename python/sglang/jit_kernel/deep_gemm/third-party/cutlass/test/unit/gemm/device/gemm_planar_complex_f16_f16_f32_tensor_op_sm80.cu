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
    \brief Tests for device-level GEMM API for Planar Complex.
*/

#include <iostream>

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"

#include "cutlass/gemm/kernel/default_gemm_planar_complex_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "testbed_planar_complex.h"

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

using gemm_planar_complex_s16816_tn_base = typename cutlass::gemm::kernel::DefaultGemmPlanarComplexUniversal<
  cutlass::half_t,
  cutlass::layout::RowMajor,
  cutlass::ComplexTransform::kNone,
  8,
  cutlass::half_t,
  cutlass::layout::ColumnMajor,
  cutlass::ComplexTransform::kNone,
  8,
  float,
  cutlass::layout::RowMajor,
  float,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<64, 64, 32>,
  cutlass::gemm::GemmShape<32, 32, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>,
  cutlass::epilogue::thread::LinearCombinationPlanarComplex<
    float,
    4,
    float,
    float
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3,
  cutlass::arch::OpMultiplyAdd
>::GemmKernel;

struct gemm_planar_complex_s16816_tn : gemm_planar_complex_s16816_tn_base {

};

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmPlanarComplex_f16t_f16n_f32n_tensor_op_f32_16816, 64x64x32_32x32x32) {

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<gemm_planar_complex_s16816_tn>;

  EXPECT_TRUE(test::gemm::device::TestAllGemmPlanarComplex<Gemm>());
}

////////////////////////////////////////////////////////////////////////////////

using gemm_planar_complex_f16_s16816_tn_base = typename cutlass::gemm::kernel::DefaultGemmPlanarComplexUniversal<
  cutlass::half_t,
  cutlass::layout::RowMajor,
  cutlass::ComplexTransform::kNone,
  8,
  cutlass::half_t,
  cutlass::layout::ColumnMajor,
  cutlass::ComplexTransform::kNone,
  8,
  cutlass::half_t,
  cutlass::layout::RowMajor,
  float,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<64, 64, 32>,
  cutlass::gemm::GemmShape<32, 32, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>,
  cutlass::epilogue::thread::LinearCombinationPlanarComplex<
    float,
    4,
    float,
    float
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3,
  cutlass::arch::OpMultiplyAdd
>::GemmKernel;

struct gemm_planar_complex_f16_s16816_tn : gemm_planar_complex_f16_s16816_tn_base {

};

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmPlanarComplex_f16t_f16n_f16n_tensor_op_f32_16816, 64x64x32_32x32x32) {

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<gemm_planar_complex_f16_s16816_tn>;

  EXPECT_TRUE(test::gemm::device::TestAllGemmPlanarComplex<Gemm>());
}

////////////////////////////////////////////////////////////////////////////////

using gemm_planar_complex_s16816_hc_base = typename cutlass::gemm::kernel::DefaultGemmPlanarComplexUniversal<
  cutlass::half_t,
  cutlass::layout::RowMajor,
  cutlass::ComplexTransform::kConjugate,
  8,
  cutlass::half_t,
  cutlass::layout::ColumnMajor,
  cutlass::ComplexTransform::kConjugate,
  8,
  float,
  cutlass::layout::RowMajor,
  float,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<64, 64, 32>,
  cutlass::gemm::GemmShape<32, 32, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>,
  cutlass::epilogue::thread::LinearCombinationPlanarComplex<
    float,
    4,
    float,
    float
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3,
  cutlass::arch::OpMultiplyAdd
>::GemmKernel;

struct gemm_planar_complex_s16816_hc : gemm_planar_complex_s16816_hc_base {

};

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmPlanarComplex_f16h_f16c_f32n_tensor_op_f32_16816, 64x64x32_32x32x32) {

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<gemm_planar_complex_s16816_hc>;

  EXPECT_TRUE(test::gemm::device::TestAllGemmPlanarComplex<Gemm>());
}

////////////////////////////////////////////////////////////////////////////////

using gemm_planar_complex_f16_s16816_hc_base = typename cutlass::gemm::kernel::DefaultGemmPlanarComplexUniversal<
  cutlass::half_t,
  cutlass::layout::RowMajor,
  cutlass::ComplexTransform::kConjugate,
  8,
  cutlass::half_t,
  cutlass::layout::ColumnMajor,
  cutlass::ComplexTransform::kConjugate,
  8,
  cutlass::half_t,
  cutlass::layout::RowMajor,
  float,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<64, 64, 32>,
  cutlass::gemm::GemmShape<32, 32, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>,
  cutlass::epilogue::thread::LinearCombinationPlanarComplex<
    float,
    4,
    float,
    float
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3,
  cutlass::arch::OpMultiplyAdd
>::GemmKernel;

struct gemm_planar_complex_f16_s16816_hc : gemm_planar_complex_f16_s16816_hc_base {

};

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmPlanarComplex_f16h_f16c_f16n_tensor_op_f32_16816, 64x64x32_32x32x32) {

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<gemm_planar_complex_f16_s16816_hc>;

  EXPECT_TRUE(test::gemm::device::TestAllGemmPlanarComplex<Gemm>());
}

////////////////////////////////////////////////////////////////////////////////

using gemm_planar_complex_s16816_nt_base = typename cutlass::gemm::kernel::DefaultGemmPlanarComplexUniversal<
  cutlass::half_t,
  cutlass::layout::ColumnMajor,
  cutlass::ComplexTransform::kNone,
  8,
  cutlass::half_t,
  cutlass::layout::RowMajor,
  cutlass::ComplexTransform::kNone,
  8,
  float,
  cutlass::layout::RowMajor,
  float,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<64, 64, 32>,
  cutlass::gemm::GemmShape<32, 32, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>,
  cutlass::epilogue::thread::LinearCombinationPlanarComplex<
    float,
    4,
    float,
    float
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3,
  cutlass::arch::OpMultiplyAdd
>::GemmKernel;

struct gemm_planar_complex_s16816_nt : gemm_planar_complex_s16816_nt_base {

};

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmPlanarComplex_f16n_f16t_f32n_tensor_op_f32_16816, 64x64x32_32x32x32) {

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<gemm_planar_complex_s16816_nt>;

  EXPECT_TRUE(test::gemm::device::TestAllGemmPlanarComplex<Gemm>());
}


////////////////////////////////////////////////////////////////////////////////

using gemm_planar_complex_f16_s16816_nt_base = typename cutlass::gemm::kernel::DefaultGemmPlanarComplexUniversal<
  cutlass::half_t,
  cutlass::layout::ColumnMajor,
  cutlass::ComplexTransform::kNone,
  8,
  cutlass::half_t,
  cutlass::layout::RowMajor,
  cutlass::ComplexTransform::kNone,
  8,
  cutlass::half_t,
  cutlass::layout::RowMajor,
  float,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<64, 64, 32>,
  cutlass::gemm::GemmShape<32, 32, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>,
  cutlass::epilogue::thread::LinearCombinationPlanarComplex<
    float,
    4,
    float,
    float
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3,
  cutlass::arch::OpMultiplyAdd
>::GemmKernel;

struct gemm_planar_complex_f16_s16816_nt : gemm_planar_complex_f16_s16816_nt_base {

};

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmPlanarComplex_f16n_f16t_f16n_tensor_op_f32_16816, 64x64x32_32x32x32) {

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<gemm_planar_complex_f16_s16816_nt_base>;

  EXPECT_TRUE(test::gemm::device::TestAllGemmPlanarComplex<Gemm>());
}

////////////////////////////////////////////////////////////////////////////////

using gemm_planar_complex_s16816_ch_base = typename cutlass::gemm::kernel::DefaultGemmPlanarComplexUniversal<
  cutlass::half_t,
  cutlass::layout::ColumnMajor,
  cutlass::ComplexTransform::kConjugate,
  8,
  cutlass::half_t,
  cutlass::layout::RowMajor,
  cutlass::ComplexTransform::kConjugate,
  8,
  float,
  cutlass::layout::RowMajor,
  float,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<64, 64, 32>,
  cutlass::gemm::GemmShape<32, 32, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>,
  cutlass::epilogue::thread::LinearCombinationPlanarComplex<
    float,
    4,
    float,
    float
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3,
  cutlass::arch::OpMultiplyAdd
>::GemmKernel;

struct gemm_planar_complex_s16816_ch : gemm_planar_complex_s16816_ch_base {

};

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmPlanarComplex_f16c_f16h_f32n_tensor_op_f32_16816, 64x64x32_32x32x32) {

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<gemm_planar_complex_s16816_ch>;

  EXPECT_TRUE(test::gemm::device::TestAllGemmPlanarComplex<Gemm>());
}

////////////////////////////////////////////////////////////////////////////////

using gemm_planar_complex_cf16_s16816_ch_base = typename cutlass::gemm::kernel::DefaultGemmPlanarComplexUniversal<
  cutlass::half_t,
  cutlass::layout::ColumnMajor,
  cutlass::ComplexTransform::kConjugate,
  8,
  cutlass::half_t,
  cutlass::layout::RowMajor,
  cutlass::ComplexTransform::kConjugate,
  8,
  cutlass::half_t,
  cutlass::layout::RowMajor,
  float,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<64, 64, 32>,
  cutlass::gemm::GemmShape<32, 32, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>,
  cutlass::epilogue::thread::LinearCombinationPlanarComplex<
    float,
    4,
    float,
    float
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3,
  cutlass::arch::OpMultiplyAdd
>::GemmKernel;

struct gemm_planar_complex_cf16_s16816_ch : gemm_planar_complex_cf16_s16816_ch_base {

};

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmPlanarComplex_f16c_f16h_f16n_tensor_op_f32_16816, 64x64x32_32x32x32) {

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<gemm_planar_complex_cf16_s16816_ch>;

  EXPECT_TRUE(test::gemm::device::TestAllGemmPlanarComplex<Gemm>());
}
////////////////////////////////////////////////////////////////////////////////

#endif // #if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
