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
    \brief CUTLASS Dual-GEMM Example.

    Fused kernel that outputs `D0` and `D1`.
    We assume that B0/B1 have the same shape/layout

```
D0 = epilogue0(X @ B0, C0)
D1 = epilogue1(X @ B1, C1)
D2 = element_wise(D0, D1)
```
    D0 and D1 will be optionally stored in gmem (`kStoreD0` / `kStoreD1`)
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "device/dual_gemm.h"
#include "thread/left_silu_and_mul.h"
#include "dual_gemm_run.h"
#include "test_run.h"


////////////////////////////////////////////////////////////////////////////////

cutlass::gemm::GemmCoord problem_size(4096, 4096, 8192);
cutlass::gemm::GemmCoord batch_problem_size(321, 256, 512);

constexpr int kStages = 3;
constexpr bool kSplitKSerial = false;
constexpr bool kUseBias = true;
constexpr int kBatchCount = 37;


#if 0
using ElementOperandA = cutlass::bfloat16_t;
using ElementOperandB = cutlass::bfloat16_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
#else
using ElementOperandA = cutlass::half_t;
using ElementOperandB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = cutlass::half_t;
using ElementCompute = cutlass::half_t;
#endif

constexpr auto kScaleType = kUseBias ? cutlass::epilogue::thread::ScaleType::NoBetaScaling : (
  // No bias
  kSplitKSerial ? cutlass::epilogue::thread::ScaleType::Default : cutlass::epilogue::thread::ScaleType::Nothing
);
using EpilogueOutputOp0 = cutlass::epilogue::thread::LinearCombination<
  ElementOutput,
  128 / cutlass::sizeof_bits<ElementOutput>::value,
  ElementAccumulator,
  ElementCompute,
  kScaleType
>;
using EpilogueOutputOp1 = cutlass::epilogue::thread::LinearCombination<
  ElementOutput,
  128 / cutlass::sizeof_bits<ElementOutput>::value,
  ElementAccumulator,
  ElementCompute,
  kScaleType
>;
using EpilogueOutputOp2 = cutlass::epilogue::thread::LeftSiLUAndMul<
  ElementOutput,
  128 / cutlass::sizeof_bits<ElementOutput>::value,
  ElementOutput,
  ElementCompute
>;

const ElementCompute alpha0 = ElementCompute(1);
const ElementCompute beta0 = ElementCompute(kUseBias ? 1 : 0);
const ElementCompute alpha1 = ElementCompute(1);
const ElementCompute beta1 = ElementCompute(kUseBias ? 1 : 0);

bool run_nonfused_gemm_f16_sm80() {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using Gemm0 = cutlass::gemm::device::Gemm<
    ElementOperandA,
    cutlass::layout::RowMajor,
    ElementOperandB,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp0,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    kStages,
    8,
    8,
    kSplitKSerial
  >;
  using Gemm1 = cutlass::gemm::device::Gemm<
    ElementOperandA,
    cutlass::layout::RowMajor,
    ElementOperandB,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp1,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    kStages,
    8,
    8,
    kSplitKSerial
  >;

  NonFusedDualGemmRun<Gemm0, Gemm1> nonFusedGemm;

  std::cout << "Running Non-fused GEMMs FP16 TN GEMMs...\n";

  bool pass = nonFusedGemm.run(
    problem_size,
    alpha0,
    beta0,
    alpha1,
    beta1,
    true  /* is_profiling */
  );

  if(pass)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return pass;
}

template <typename T>
struct LeftSiLUAndMul {
  struct Params{};
  CUTLASS_HOST_DEVICE LeftSiLUAndMul(Params p) {}

  CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}

  CUTLASS_HOST_DEVICE T operator() (
    T const &lhs, 
    T const &rhs) const {
    cutlass::epilogue::thread::SiLu<T> silu;
    cutlass::multiplies<T> mul;
    auto silu_lhs = silu(lhs);
    return mul(silu_lhs, rhs);
  }

  template <int kCount>
  CUTLASS_HOST_DEVICE cutlass::Array<T, kCount> operator() (
    cutlass::Array<T, kCount> const &lhs, 
    cutlass::Array<T, kCount> const &rhs) const {
    cutlass::epilogue::thread::SiLu<T> silu;
    cutlass::multiplies<T> mul;
    auto silu_lhs = silu(lhs);
    return mul(silu_lhs, rhs);
  }
};

bool run_fused_gemm_f16_sm80_shmem() {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  // Optionally, we might not need intermediate GEMM outputs
  constexpr bool kStoreD0 = true;
  constexpr bool kStoreD1 = true;

  using DualGemm = cutlass::gemm::device::DualGemm<
    ElementOperandA,
    cutlass::layout::RowMajor,
    ElementOperandB,
    cutlass::layout::ColumnMajor,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp0,
    EpilogueOutputOp1,
    EpilogueOutputOp2,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    kStages,
    kStoreD0,
    kStoreD1,
    kSplitKSerial
  >;

  DualFusedGemmRun<DualGemm> fusedGemm;

  std::cout << "Running Fused FP16 TN GEMMs + Epilogue2...\n";

  bool passed = fusedGemm.run(
    problem_size,
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

bool run_batched_fused_gemm_f16_sm80_shmem() {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  // Optionally, we might not need intermediate GEMM outputs
  constexpr bool kStoreD0 = true;
  constexpr bool kStoreD1 = true;

  using DualGemm = cutlass::gemm::device::DualGemm<
    ElementOperandA,
    cutlass::layout::RowMajor,
    ElementOperandB,
    cutlass::layout::ColumnMajor,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp0,
    EpilogueOutputOp1,
    EpilogueOutputOp2,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    kStages,
    kStoreD0,
    kStoreD1,
    kSplitKSerial
  >;

  DualFusedGemmRun<DualGemm> fusedGemm;

  std::cout << "Running Batched Fused FP16 TN GEMMs + Epilogue2...\n";

  bool passed = fusedGemm.run(
    batch_problem_size,
    alpha0,
    beta0,
    alpha1,
    beta1,
    kBatchCount,
    false,  /* broadcast_b1 */
    false   /* is_profiling */
  );

  if(passed)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return passed;
}

bool run_broadcast_fused_gemm_f16_sm80_shmem() {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  // Optionally, we might not need intermediate GEMM outputs
  constexpr bool kStoreD0 = true;
  constexpr bool kStoreD1 = true;

  using DualGemm = cutlass::gemm::device::DualGemm<
    ElementOperandA,
    cutlass::layout::RowMajor,
    ElementOperandB,
    // different LayoutB0 and B1
    cutlass::layout::RowMajor,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp0,
    EpilogueOutputOp1,
    EpilogueOutputOp2,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    kStages,
    kStoreD0,
    kStoreD1,
    kSplitKSerial
  >;

  DualFusedGemmRun<DualGemm> fusedGemm;

  std::cout << "Running Broadcast Fused FP16 TN GEMMs + Epilogue2...\n";

  bool passed = fusedGemm.run(
    problem_size,
    alpha0,
    beta0,
    alpha1,
    beta1,
    1,     /* batch_count */
    true,  /* broadcast_b1 */
    true   /* is_profiling */
  );

  if(passed)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return passed;
}

bool run_batched_broadcast_fused_gemm_f16_sm80_shmem() {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  // Optionally, we might not need intermediate GEMM outputs
  constexpr bool kStoreD0 = true;
  constexpr bool kStoreD1 = true;

  using DualGemm = cutlass::gemm::device::DualGemm<
    ElementOperandA,
    cutlass::layout::RowMajor,
    ElementOperandB,
    // different LayoutB0 and B1
    cutlass::layout::RowMajor,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp0,
    EpilogueOutputOp1,
    EpilogueOutputOp2,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    kStages,
    kStoreD0,
    kStoreD1,
    kSplitKSerial
  >;

  DualFusedGemmRun<DualGemm> fusedGemm;

  std::cout << "Running Batch Broadcast Fused FP16 TN GEMMs + Epilogue2...\n";

  bool passed = fusedGemm.run(
    batch_problem_size,
    alpha0,
    beta0,
    alpha1,
    beta1,
    kBatchCount,
    true,  /* broadcast_b1 */
    false  /* is_profiling */
  );

  if(passed)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return passed;
}

int main() {

  std::vector<bool (*)()>funcs = {
    &run_nonfused_gemm_f16_sm80,
    &run_fused_gemm_f16_sm80_shmem,
    &run_batched_fused_gemm_f16_sm80_shmem,
    &run_broadcast_fused_gemm_f16_sm80_shmem,
    &run_batched_broadcast_fused_gemm_f16_sm80_shmem
  };

  std::string test_name = (
    "dual-gemm f16 bias=" +
    std::to_string(kUseBias) +
    " split_k_serial=" +
    std::to_string(kSplitKSerial) +
    " batch_count=" +
    std::to_string(kBatchCount)
  );

  return testRun(80, funcs, test_name);
}



////////////////////////////////////////////////////////////////////////////////
