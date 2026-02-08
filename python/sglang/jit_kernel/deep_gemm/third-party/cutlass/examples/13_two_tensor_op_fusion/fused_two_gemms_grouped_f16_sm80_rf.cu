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
    \brief Example of running grouped back-to-back GEMMs when intermediate results are RF resident
*/

#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/base_grouped.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "device/b2b_gemm.h"
#include "kernel/default_b2b_gemm.h"
#include "threadblock/grouped_threadblock_swizzle.h"
#include "b2b_grouped_gemm_run.h"
#include "test_run.h"

////////////////////////////////////////////////////////////////////////////////

std::vector<cutlass::gemm::GemmCoord> gemm_f16_sm80_problem_sizes_0;
std::vector<cutlass::gemm::GemmCoord> gemm_f16_sm80_problem_sizes_1;

// Constraints:
//   1. Warp shape N must equal thread block shape N
//   2. Problem size N must equal thread block shape N
using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 64, 32>;
using WarpShape0 = cutlass::gemm::GemmShape<16, 64, 32>;
using ThreadblockShape1 = cutlass::gemm::GemmShape<64, 128, 32>;
using WarpShape1 = cutlass::gemm::GemmShape<16, 128, 32>;

// Command line options parsing
struct Options {

  bool help;
  bool error;
  bool reference_check;
  int alignment = 8;

  std::vector<cutlass::gemm::GemmCoord> problem_sizes0;
  std::vector<cutlass::gemm::GemmCoord> problem_sizes1;

  int problem_count;
  bool verbose;

  //
  // Methods
  //

  Options():
    help(false),
    error(false),
    reference_check(true),
    problem_count(15),
    verbose(false)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("problems", problem_count, 15);
    cmd.get_cmd_line_argument("reference-check", reference_check, true);
    cmd.get_cmd_line_argument("verbose", verbose, false);

    randomize_problems(cmd);
  }

  void randomize_problems(cutlass::CommandLine &cmd) {

    //
    // For now, randomly choose the problem sizes.
    //

    int cmd_line_m = -1;
    int cmd_line_k = -1;

    cmd.get_cmd_line_argument("m", cmd_line_m);
    cmd.get_cmd_line_argument("k", cmd_line_k);

    problem_sizes0.reserve(problem_count);
    problem_sizes1.reserve(problem_count);

    for (int i = 0; i < problem_count; ++i) {

      int m = cmd_line_m;
      int k = cmd_line_k;

      if (m < 1) {
        m = alignment * ((rand() % 256) + 1);
      }

      if (k < 1) {
        k = alignment * ((rand() % 256) + 1);
      }

      cutlass::gemm::GemmCoord problem0(m, ThreadblockShape0::kN, k);
      cutlass::gemm::GemmCoord problem1(m, ThreadblockShape1::kN, ThreadblockShape0::kN);

      problem_sizes0.push_back(problem0);
      problem_sizes1.push_back(problem1);
    }

    if (verbose) {
      print_problem_sizes();
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "13_fused_two_gemms_grouped_f16_sm80_rf\n\n"
      << "  This example runs a grouped back-to-back GEMM kernel. A group of independent back-to-back GEMMs are\n"
      << "  run in a single kernel. Each individual problem in the group is subject to the same constraints that non-grouped\n"
      << "  back-to-back GEMMs are subject to.s"
      << "Options:\n\n"
      << "  --help                           If specified, displays this usage statement.\n\n"
      << "  --problems=<int>                 Number of individual GEMM problems (default: --problems=15)\n"
      << "  --m=<int>                        Sets the M dimension of both GEMMs for all groups. Otherwise, it is selected randomly\n"
      << "  --k=<int>                        Sets the K dimension of the first GEMM for all groups. Otherwise, it is selected randomly\n"
      << "  --verbose=<bool>                 If true, prints problem sizes.\n";

    out << "\n\nExamples:\n\n"

      << "# Runs a grouped B2b GEMM with 10 random problem sizes\n"
      << "$ ./examples/13_two_tensor_op_fusion/13_fused_two_gemms_grouped_f16_sm80_rf --groups=10\n\n";

    return out;
  }

  void print_problem_sizes() {
    std::cout << std::endl;
    std::cout << "Executing " << problem_count << " independent back-to-back GEMMs in a group" << std::endl;
    for (int i = 0; i < problem_count; ++i) {
      cutlass::gemm::GemmCoord problem0 = problem_sizes0.at(i);
      cutlass::gemm::GemmCoord problem1 = problem_sizes1.at(i);
      std::cout << "Problem " << i
                << "\t\tGEMM0: " << problem0.m() << 'x' << problem0.n() << 'x' << problem0.k()
                << "\t\tGEMM1: " << problem1.m() << 'x' << problem1.n() << 'x' << problem1.k()
                << std::endl;
    }
  }
};

bool run_fused_grouped_gemm_f16_sm80_rf_res() {

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  ElementCompute alpha0 = ElementCompute(1);
  //Fused kernel has built-in bias, setting beta=0
  ElementCompute beta0 = ElementCompute(0); 
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(1); //beta=1 for bias

  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using EpilogueOutputOp0 = 
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      InstructionShape::kM * InstructionShape::kN / 32,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >;

  using EpilogueOutputOp1 = 
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;

  using GroupedThreadblockSwizzle = cutlass::gemm::threadblock::B2bGemmGroupedThreadblockSwizzle<
                                                                    ThreadblockShape0,
                                                                    cutlass::layout::RowMajor // LayoutC
                                                                    >;

  const int kAlignment = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  const int kStages = 3;
  using B2bGemmKernel = cutlass::gemm::kernel::DefaultB2bGemm<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        kAlignment,
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        kAlignment,
        cutlass::half_t,
        cutlass::layout::RowMajor,
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
        GroupedThreadblockSwizzle,
        kStages,
        cutlass::arch::OpMultiplyAdd
    >::B2bGemmKernel;

  using B2bGemm = cutlass::gemm::device::BaseGrouped<B2bGemmKernel>;

  B2bFusedGroupedGemmRun<B2bGemm> fusedGemm;

  std::cout << "Running Fused back-to-back FP16 TN Grouped GEMMs with RF residency...\n";
  bool passed = fusedGemm.run(gemm_f16_sm80_problem_sizes_0, gemm_f16_sm80_problem_sizes_1, alpha0, beta0, alpha1, beta1);
  if(passed)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return passed;
}

int main(int argc, char const **args) {

  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  gemm_f16_sm80_problem_sizes_0 = options.problem_sizes0;
  gemm_f16_sm80_problem_sizes_1 = options.problem_sizes1;

  std::vector<bool (*)()>funcs = {
    &run_fused_grouped_gemm_f16_sm80_rf_res
  };

  return testRun(80, funcs, "grouped gemm f16 RF residency");
}




////////////////////////////////////////////////////////////////////////////////
