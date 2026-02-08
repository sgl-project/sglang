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

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/gemm_complex.h"

#include "testbed_utils.h"

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm, bool Relu = false>
struct TestbedUniversal {

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  using ElementAccumulator = typename Gemm::ElementAccumulator;
  using ElementCompute = typename Gemm::GemmKernel::Epilogue::OutputOp::ElementCompute;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;

  cutlass::HostTensor<typename Gemm::ElementA, typename Gemm::LayoutA> tensor_A;
  cutlass::HostTensor<typename Gemm::ElementB, typename Gemm::LayoutB> tensor_B;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_C;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_D;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> reference_D;

  //
  // Methods
  //

  TestbedUniversal(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool initialize_tensor(
    cutlass::TensorView<Element, Layout> view,
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      double scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<typename Gemm::ElementC>::value;
      bool is_unsigned_int = std::numeric_limits<Element>::is_integer && !std::numeric_limits<Element>::is_signed;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = is_unsigned_int ? 2 : 1;
        scope_min = is_unsigned_int ? 0 : -1;
      } else if (bits_output == 16) {
        constexpr auto u8_bf16 =
          (cutlass::platform::is_same<ElementA, uint8_t>::value &&
           cutlass::platform::is_same<ElementB, cutlass::bfloat16_t>::value) ||
          (cutlass::platform::is_same<ElementA, cutlass::bfloat16_t>::value &&
           cutlass::platform::is_same<ElementB, uint8_t>::value);
        scope_max = is_unsigned_int ? 10 : (u8_bf16 ? 3 : 5);
        scope_min = is_unsigned_int ? 0 : (u8_bf16 ? -3 : -5);
      } else {
        scope_max = 8;
        scope_min = -8;
      }

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, scope_max, scope_min, 0);
    }
    else if (dist_kind == cutlass::Distribution::Identity) {

      cutlass::reference::host::TensorFillIdentity(view);
    }
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {

      cutlass::reference::host::BlockFillSequential(
        view.data(), view.capacity());
    }
    else {
      EXPECT_TRUE(false) << "Not implemented";
      return false;
    }

    return true;
  }

  /// Initializes data structures
  void initialize(cutlass::gemm::GemmCoord problem_size) {
    //
    // Allocate the GEMM workspace
    //

    tensor_A.resize(problem_size.mk());
    tensor_B.resize(problem_size.kn());
    tensor_C.resize(problem_size.mn());
    tensor_D.resize(problem_size.mn());
    reference_D.resize(problem_size.mn(), false);

    EXPECT_TRUE(initialize_tensor(tensor_A.host_view(), init_A, seed + 2019));
    EXPECT_TRUE(initialize_tensor(tensor_B.host_view(), init_B, seed + 2018));
    EXPECT_TRUE(initialize_tensor(tensor_C.host_view(), init_C, seed + 2017));

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    cutlass::Coord<2> origin(0);
    tensor_A.host_view().at(origin) = typename Gemm::ElementA(1);
    tensor_B.host_view().at(origin) = typename Gemm::ElementB(1);
    tensor_C.host_view().at(origin) = typename Gemm::ElementC(1);

    cutlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(
    cutlass::gemm::GemmCoord problem_size,
    ElementCompute alpha,
    ElementCompute beta) {

    tensor_D.sync_host();

    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_A.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_B.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_C.host_view()), 0);

    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_D.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(reference_D.host_view()), 0);

    bool passed = cutlass::reference::host::TensorEquals(reference_D.host_view(), tensor_D.host_view());

    EXPECT_TRUE(passed) << " mismatched reference";

    if (!passed) {

      /*

      std::stringstream fname;

      fname << "error_Gemm_device_"
        << problem_size.m() << "x"
        << problem_size.n() << "x"
        << problem_size.k() << "_"
        << Gemm::ThreadblockShape::kM << "x"
        << Gemm::ThreadblockShape::kN << "x"
        << Gemm::ThreadblockShape::kK << "_"
        << Gemm::WarpShape::kM << "x"
        << Gemm::WarpShape::kN << "x"
        << Gemm::WarpShape::kK << ".txt";

      std::ofstream file(fname.str());
      */

      std::ofstream file("testbed_universal_errors.txt");

      file
        << "problem: " << problem_size
        << ", alpha: " << alpha << ", beta: " << beta << "\n\n";

      file
        << "A =\n" << tensor_A.host_view()
        << "\nB =\n" << tensor_B.host_view()
        << "\nC =\n" << tensor_C.host_view()
        << "\n\nReference =\n" << reference_D.host_view()
        << "\nComputed =\n" << tensor_D.host_view();
    }

    return passed;
  }

  /// Verifies the result is a GEMM
  bool verify(
    cutlass::gemm::GemmCoord problem_size,
    ElementCompute alpha,
    ElementCompute beta) {

    //
    // Verify
    //

    cutlass::reference::host::GemmComplex<
        typename Gemm::ElementA, typename Gemm::LayoutA,
        typename Gemm::ElementB, typename Gemm::LayoutB,
        typename Gemm::ElementC, typename Gemm::LayoutC,
        ElementCompute, ElementAccumulator
    >(
      problem_size,
      alpha,
      tensor_A.host_ref(),
      Gemm::kTransformA,
      tensor_B.host_ref(),
      Gemm::kTransformB,
      beta,
      tensor_C.host_ref(),
      reference_D.host_ref(),
      ElementAccumulator(0)
    );

    if (Relu) {
      for (int i = 0; i < problem_size.m(); ++i) {
        for (int j = 0; j < problem_size.n(); ++j) {
           reference_D.at(cutlass::MatrixCoord(i, j)) =
                  ((ElementCompute)reference_D.at(cutlass::MatrixCoord(i, j)) < (ElementCompute)0)
                  ? (typename Gemm::ElementC)0
                  : reference_D.at(cutlass::MatrixCoord(i, j));
        }
      }
    }

    return compare_reference(problem_size, alpha, beta);
  }

  /// Returns true if the CUDA device is sufficient to execute the kernel.
  bool sufficient() const {
    //
    // Determine SMEM requirements and waive if not satisfied
    //

    size_t smem_size = sizeof(typename Gemm::GemmKernel::SharedStorage);

    cudaDeviceProp properties;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice() API call failed.");
    }

    result = cudaGetDeviceProperties(&properties, device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties() failed");
    }

    if (properties.sharedMemPerBlockOptin < smem_size) {
      return false;
    }

    return true;
  }

  /// Executes one test
  bool run(
    cutlass::gemm::GemmUniversalMode mode,
    cutlass::gemm::GemmCoord problem_size,
    int batch_count = 1,
    ElementCompute alpha = ElementCompute(1),
    ElementCompute beta = ElementCompute(0))
  {
/*
    std::cout << "\n-----------------------\n";
    std::cout << "mode: " << (int) mode << "\n";
    std::cout << "problem size: " << problem_size << "\n";
    std::cout << "batch_count: " << batch_count << "\n";
    std::cout << "alpha: " << alpha << "\n";
    std::cout << "beta: " << beta << "\n";
    std::cout << "-----------------------\n\n";
*/

    // Waive test if insufficient CUDA device
    if (!sufficient()) {
      if (CUTLASS_TEST_UNIT_ENABLE_WARNINGS) {
        std::cerr << "Test waived due to insufficient CUDA device." << std::endl;
      }
      return true;
    }

    this->initialize(problem_size);

    //
    // Initialize the GEMM operator
    //

    typename Gemm::Arguments arguments{
      mode,
      problem_size,
      batch_count,
      {alpha, beta},
      tensor_A.device_data(),
      tensor_B.device_data(),
      tensor_C.device_data(),
      tensor_D.device_data(),
      problem_size.m() * problem_size.k(),
      problem_size.n() * problem_size.k(),
      problem_size.m() * problem_size.n(),
      problem_size.m() * problem_size.n(),
      tensor_A.layout().stride(0),
      tensor_B.layout().stride(0),
      tensor_C.layout().stride(0),
      tensor_D.layout().stride(0)
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);

    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());

    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    //
    // Run the GEMM
    //

    status = gemm_op();

    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    //
    // Verify
    //

    bool passed = this->verify(problem_size, alpha, beta);

    if (!passed) {
      std::cout << "Failed with batch_count/split_k_slices = " << batch_count << std::endl;
    }

    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Gemm, bool Relu = false>
bool TestGemmUniversal(
  cutlass::gemm::GemmCoord const & problem_size,
  cutlass::gemm::GemmUniversalMode mode,
  int batch_count,
  double alpha = 1.0,
  double beta = 2.0) {

  bool passed = true;

  TestbedUniversal<Gemm, Relu> testbed;

  using ElementCompute = typename Gemm::EpilogueOutputOp::ElementCompute;

  passed = testbed.run(
    mode,
    problem_size,
    batch_count,
    cutlass::from_real<ElementCompute>(alpha),
    cutlass::from_real<ElementCompute>(beta)
  );

  return passed;
}

template <typename Gemm, bool Relu = false>
bool TestAllGemmUniversal() {
  bool passed = true;


  int const kMinimumOperandElementSize = 
    std::min(
      int(cutlass::sizeof_bits<typename Gemm::ElementA>::value), 
      int(cutlass::sizeof_bits<typename Gemm::ElementB>::value));

  int const kAlignment = cutlass::platform::is_same<
                              typename Gemm::OperatorClass, 
                              cutlass::arch::OpClassSimt>::value ? 1 : 128 / kMinimumOperandElementSize;

  // int8_t gemm alignment constraints
  int const kAlignmentM = cutlass::platform::is_same<typename Gemm::OperatorClass, cutlass::arch::OpClassSimt>::value &&
                          cutlass::platform::is_same<typename Gemm::ElementA, int8_t>::value &&
                          cutlass::platform::is_same<typename Gemm::LayoutA, cutlass::layout::ColumnMajor>::value ? 4 : kAlignment;

  int const kAlignmentN = cutlass::platform::is_same<typename Gemm::OperatorClass, cutlass::arch::OpClassSimt>::value &&
                          cutlass::platform::is_same<typename Gemm::ElementB, int8_t>::value &&
                          cutlass::platform::is_same<typename Gemm::LayoutB, cutlass::layout::RowMajor>::value ? 4 : kAlignment;

  int const kAlignmentK = cutlass::platform::is_same<typename Gemm::OperatorClass, cutlass::arch::OpClassSimt>::value &&
                          cutlass::platform::is_same<typename Gemm::ElementA, int8_t>::value &&
                          cutlass::platform::is_same<typename Gemm::ElementB, int8_t>::value &&
                          (cutlass::platform::is_same<typename Gemm::LayoutA, cutlass::layout::RowMajor>::value ||
                          cutlass::platform::is_same<typename Gemm::LayoutB, cutlass::layout::ColumnMajor>::value) ? 4 : kAlignment;



  cutlass::gemm::GemmUniversalMode modes[] = {
    cutlass::gemm::GemmUniversalMode::kGemm,
  };

  int problem_size_m[] = {
    kAlignmentM, 512 - 3*kAlignmentM
  };

  int problem_size_n[] = {
    kAlignmentN, 512 - 2*kAlignmentN
  };

  int problem_size_k[] = {
    kAlignmentK,
    Gemm::ThreadblockShape::kK * Gemm::kStages - kAlignmentK,
    Gemm::ThreadblockShape::kK * Gemm::kStages * 3 - kAlignmentK
  };

  int batch_counts[] = {      // may be interpretted as batch count or split-K slices
    1, 2, 3, 5, 7
  };

  double problem_alpha[] = {
    1
  };

  double problem_beta[] = {
    2.0
  };


  using ElementCompute = typename Gemm::EpilogueOutputOp::ElementCompute;

  for (cutlass::gemm::GemmUniversalMode mode : modes) {
    for (int m : problem_size_m) {
      for (int n : problem_size_n) {
        for (int k : problem_size_k) {
          for (int batch_count : batch_counts) {

            for (auto alpha : problem_alpha) {
              for (auto beta : problem_beta) {

                if (mode == cutlass::gemm::GemmUniversalMode::kGemm ||
                  mode == cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel) {

                  // skip very small K problems
                  if (k / batch_count < 2 * Gemm::ThreadblockShape::kK) {
                    continue;
                  }
                }

                cutlass::gemm::GemmCoord problem_size(m, n, k);

                TestbedUniversal<Gemm, Relu> testbed;

                passed = testbed.run(
                  mode,
                  problem_size,
                  batch_count,
                  cutlass::from_real<ElementCompute>(alpha),
                  cutlass::from_real<ElementCompute>(beta)
                );

                if (!passed) {
                  return false;
                }
              }
            }
          }
        }
      }
    }
  }

  /*
  // large problem with high coverage
  for (int split_k_slices = 1; split_k_slices <= 3; ++split_k_slices) {
    TestbedUniversal<Gemm> testbed;

    cutlass::gemm::GemmCoord problem_size(72, 56, 8192);

    passed = testbed.run(
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      split_k_slices,
      cutlass::from_real<ElementCompute>(1.0),
      cutlass::from_real<ElementCompute>(2.0)
    );

    if (!passed) {
      break;
    }
  }
  */

  return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////
