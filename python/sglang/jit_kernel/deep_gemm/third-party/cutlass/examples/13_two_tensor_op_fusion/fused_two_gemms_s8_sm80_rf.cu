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
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "device/b2b_gemm.h"
#include "b2b_interleaved_gemm_run.h"
#include "test_run.h"

////////////////////////////////////////////////////////////////////////////////

cutlass::gemm::GemmCoord gemm_s8_sm80_problem_size_0(128*640, 64, 576);
cutlass::gemm::GemmCoord gemm_s8_sm80_problem_size_1(128*640, 128, 64);

bool run_nonfused_gemm_s8_sm80() {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  ElementCompute alpha0 = ElementCompute(1);
  ElementCompute beta0 = ElementCompute(1); //beta=1 for bias
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(1); //beta=1 for bias

  using ThreadblockShape0 = cutlass::gemm::GemmShape<128, 64, 64>;
  using WarpShape0 = cutlass::gemm::GemmShape<64, 64, 64>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape1 = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  using Gemm0 = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::ColumnMajorInterleaved<32>,
    int8_t,
    cutlass::layout::RowMajorInterleaved<32>,
    ElementOutput,
    cutlass::layout::ColumnMajorInterleaved<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape0,
    WarpShape0,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      64 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    16,
    16,
    false,
    cutlass::arch::OpMultiplyAddSaturate
  >;
  using Gemm1 = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::ColumnMajorInterleaved<32>,
    int8_t,
    cutlass::layout::RowMajorInterleaved<32>,
    ElementOutput,
    cutlass::layout::ColumnMajorInterleaved<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape1,
    WarpShape1,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      64 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    16,
    16,
    false,
    cutlass::arch::OpMultiplyAddSaturate
  >;

  B2bInterleavedNonFusedGemmRun<Gemm0, Gemm1, 32> nonFusedGemm;

  std::cout << "Running Non-fused back-to-back INT8 NT interleaved GEMMs...\n";
  bool pass = nonFusedGemm.run(gemm_s8_sm80_problem_size_0, gemm_s8_sm80_problem_size_1, alpha0, beta0, alpha1, beta1);
  if(pass)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return pass;
}


bool run_fused_gemm_s8_sm80_rf_res() {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  ElementCompute alpha0 = ElementCompute(1);
  //Fused kernel has built-in bias, setting beta=0
  ElementCompute beta0 = ElementCompute(0);
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(1); //beta=1 for bias

  using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape0 = cutlass::gemm::GemmShape<16, 64, 64>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<64, 128, 64>;
  using WarpShape1 = cutlass::gemm::GemmShape<16, 128, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  using EpilogueOutputOp0 =
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      8 * InstructionShape::kN / 32,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >;

  using EpilogueOutputOp1 =
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      64 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

  const bool SmemAccumulator = false;

  using B2bGemm = cutlass::gemm::device::B2bGemm<
    int8_t,
    cutlass::layout::ColumnMajorInterleaved<32>,
    int8_t,
    cutlass::layout::RowMajorInterleaved<32>,
    ElementOutput,
    cutlass::layout::ColumnMajorInterleaved<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape0,
    ThreadblockShape1,
    WarpShape0,
    WarpShape1,
    InstructionShape,
    EpilogueOutputOp0,
    EpilogueOutputOp1,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    SmemAccumulator,
    16,
    16,
    cutlass::arch::OpMultiplyAddSaturate
  >;

  B2bInterleavedFusedGemmRun<B2bGemm, 32> fusedGemm;

  std::cout << "Running Fused back-to-back INT8 NT interleaved GEMMs with RF residency...\n";
  bool passed = fusedGemm.run(
    gemm_s8_sm80_problem_size_0,
    gemm_s8_sm80_problem_size_1,
    alpha0,
    beta0,
    alpha1,
    beta1
  );

  if(passed)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return passed;
}

bool run_fused_gemm_s8_sm80_rf_res_batch() {


  cutlass::gemm::GemmCoord gemm_s8_sm80_problem_size_0(256, 64, 128);
  cutlass::gemm::GemmCoord gemm_s8_sm80_problem_size_1(256, 128, 64);

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  ElementCompute alpha0 = ElementCompute(1);
  //Fused kernel has built-in bias, setting beta=0
  ElementCompute beta0 = ElementCompute(0);
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(1); //beta=1 for bias

  using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape0 = cutlass::gemm::GemmShape<16, 64, 64>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<64, 128, 64>;

  using WarpShape1 = cutlass::gemm::GemmShape<16, 128, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  using EpilogueOutputOp0 =
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      8 * InstructionShape::kN / 32,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >;

  using EpilogueOutputOp1 =
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      64 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

  const bool SmemAccumulator = false;

  using B2bGemm = cutlass::gemm::device::B2bGemm<
    int8_t,
    cutlass::layout::ColumnMajorInterleaved<32>,
    int8_t,
    cutlass::layout::RowMajorInterleaved<32>,
    ElementOutput,
    cutlass::layout::ColumnMajorInterleaved<32>,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape0,
    ThreadblockShape1,
    WarpShape0,
    WarpShape1,
    InstructionShape,
    EpilogueOutputOp0,
    EpilogueOutputOp1,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    SmemAccumulator,
    16,
    16,
    cutlass::arch::OpMultiplyAddSaturate
  >;

  B2bInterleavedFusedGemmRun<B2bGemm, 32> fusedGemm;

  int batch_count = 2;
  int64_t batch_stride_A0 = gemm_s8_sm80_problem_size_0.m() * gemm_s8_sm80_problem_size_0.k();
  int64_t batch_stride_B0 = gemm_s8_sm80_problem_size_1.k() * gemm_s8_sm80_problem_size_1.n();
  int64_t batch_stride_C0 = gemm_s8_sm80_problem_size_0.m() * gemm_s8_sm80_problem_size_0.n();
  int64_t batch_stride_B1 = gemm_s8_sm80_problem_size_1.k() * gemm_s8_sm80_problem_size_1.n();
  int64_t batch_stride_C1 = gemm_s8_sm80_problem_size_1.n();
  int64_t batch_stride_D1 = gemm_s8_sm80_problem_size_1.m() * gemm_s8_sm80_problem_size_1.n();
  int64_t batch_stride_Bias0 = gemm_s8_sm80_problem_size_0.n();
  int64_t batch_stride_Scale0 = 0;

  std::cout << "Running Fused back-to-back INT8 NT interleaved Batched GEMMs with RF residency...\n";
  bool passed = fusedGemm.run(
    gemm_s8_sm80_problem_size_0,
    gemm_s8_sm80_problem_size_1,
    alpha0,
    beta0,
    alpha1,
    beta1,
    cutlass::gemm::GemmUniversalMode::kBatched,
    batch_count,
    batch_stride_A0,
    batch_stride_B0,
    batch_stride_C0,
    batch_stride_B1,
    batch_stride_C1,
    batch_stride_D1,
    batch_stride_Bias0,
    batch_stride_Scale0
  );

  if(passed)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return passed;
}

int main() {

  std::vector<bool (*)()>funcs = {
    &run_nonfused_gemm_s8_sm80,
    &run_fused_gemm_s8_sm80_rf_res,
    &run_fused_gemm_s8_sm80_rf_res_batch
  };

  return testRun(80, funcs, "gemm int8 RF residency");
}

////////////////////////////////////////////////////////////////////////////////
