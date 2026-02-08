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

#include "testbed_utils.h"
#include "testbed_universal.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm, bool Relu = false>
struct Testbed {

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  using ElementAccumulator = typename Gemm::ElementAccumulator;
  using ElementCompute = typename Gemm::GemmKernel::Epilogue::OutputOp::ElementCompute;

  /// Initialization
  typename Gemm::LayoutA::Stride stride_factor_A;
  typename Gemm::LayoutB::Stride stride_factor_B;
  typename Gemm::LayoutC::Stride stride_factor_C;
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

  Testbed(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    stride_factor_A(typename Gemm::LayoutA::Stride()),
    stride_factor_B(typename Gemm::LayoutB::Stride()),
    stride_factor_C(typename Gemm::LayoutC::Stride()),
    init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

  Testbed(
    typename Gemm::LayoutA::Stride stride_factor_A_,
    typename Gemm::LayoutB::Stride stride_factor_B_,
    typename Gemm::LayoutC::Stride stride_factor_C_,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    stride_factor_A(stride_factor_A_),
    stride_factor_B(stride_factor_B_),
    stride_factor_C(stride_factor_C_),
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

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = 1;
        scope_min = -1;
      } else if (bits_output == 16) {
        scope_max = 5;
        scope_min = -5;
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

    tensor_A.resize(problem_size.mk(), cutlass::layout::Affine2Layout_Factory<typename Gemm::LayoutA>::layout_factory(problem_size.mk(), stride_factor_A));
    tensor_B.resize(problem_size.kn(), cutlass::layout::Affine2Layout_Factory<typename Gemm::LayoutB>::layout_factory(problem_size.kn(), stride_factor_B));
    tensor_C.resize(problem_size.mn(), cutlass::layout::Affine2Layout_Factory<typename Gemm::LayoutC>::layout_factory(problem_size.mn(), stride_factor_C));
    tensor_D.resize(problem_size.mn(), cutlass::layout::Affine2Layout_Factory<typename Gemm::LayoutC>::layout_factory(problem_size.mn(), stride_factor_C));
    reference_D.resize(problem_size.mn(), cutlass::layout::Affine2Layout_Factory<typename Gemm::LayoutC>::layout_factory(problem_size.mn(), stride_factor_C), false);

    EXPECT_TRUE(initialize_tensor(tensor_A.host_view(), init_A, seed + 2019));
    EXPECT_TRUE(initialize_tensor(tensor_B.host_view(), init_B, seed + 2018));
    EXPECT_TRUE(initialize_tensor(tensor_C.host_view(), init_C, seed + 2017));

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_A.host_view().at({0, 0}) = typename Gemm::ElementA(1);
    tensor_B.host_view().at({0, 0}) = typename Gemm::ElementB(1);
    tensor_C.host_view().at(cutlass::make_Coord(0, 0)) = typename Gemm::ElementC(1);

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

    if (tensor_D.size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_D.host_view()), 0)
        << "tensor_D (size " << tensor_D.size() << ") has nonpositive norm";
    }
    if (reference_D.size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(reference_D.host_view()), 0)
        << "reference_D (size " << reference_D.size() << ") has nonpositive norm";
    }
    bool passed = cutlass::reference::host::TensorEquals(reference_D.host_view(), tensor_D.host_view());

    EXPECT_TRUE(passed) << "reference_D does not equal tensor_D";

    if (!passed) {

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
    
    cutlass::reference::host::Gemm<
        typename Gemm::ElementA, typename Gemm::LayoutA,
        typename Gemm::ElementB, typename Gemm::LayoutB,
        typename Gemm::ElementC, typename Gemm::LayoutC, ElementCompute,
        ElementAccumulator, typename Gemm::Operator>
        reference_gemm;

    reference_gemm(
      problem_size,
      alpha, 
      tensor_A.host_ref(), 
      tensor_B.host_ref(), 
      beta, 
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

	/// Determine if the CUDA device is sufficient to run the kernel
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
    cutlass::gemm::GemmCoord problem_size,
    int split_k_slices = 1,
    ElementCompute alpha = ElementCompute(1),
    ElementCompute beta = ElementCompute(0))
  {
/*
    std::cout << "\n-----------------------\n";
    std::cout << "problem size: " << problem_size << "\n";
    std::cout << "split_k_slices: " << split_k_slices << "\n";
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
      problem_size,
      tensor_A.device_ref(),
      tensor_B.device_ref(),
      tensor_C.device_ref(),
      tensor_D.device_ref(),
      {alpha, beta},
      split_k_slices
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);

    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());

    EXPECT_TRUE(status == cutlass::Status::kSuccess)
      << "gemm_op.initialize returned with error " << to_string(status)
      << ", indicating that this test is not supported.  Last CUDA error: "
      << cudaGetErrorString(cudaGetLastError());
    if (status != cutlass::Status::kSuccess) {
      return true;
    }

    //
    // Run the GEMM
    //

    try {
      status = gemm_op();
    }
    catch (std::exception const& e) {
      EXPECT_TRUE(false) << "gemm_op() threw a std::exception: " << e.what();
      throw;
    }
    catch (...) {
      EXPECT_TRUE(false) << "gemm_op() threw an exception of unknown type";
      throw;
    }
    EXPECT_TRUE(status == cutlass::Status::kSuccess)
      << "gemm_op failed with error " << to_string(status);

    //
    // Verify
    //

    bool passed = this->verify(problem_size, alpha, beta);
    EXPECT_TRUE(passed) << "Error: split_k_slices = " << split_k_slices
      << ", alpha: " << alpha;

    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm, bool Relu=false>
bool TestAllGemmBasic(
    const typename Gemm::LayoutA::Stride& stride_factor_A = typename Gemm::LayoutA::Stride(),
    const typename Gemm::LayoutB::Stride& stride_factor_B = typename Gemm::LayoutB::Stride(),
    const typename Gemm::LayoutC::Stride& stride_factor_C = typename Gemm::LayoutC::Stride()) {
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

  int problem_size_m[] = {kAlignmentM, 512 - 3 * kAlignmentM};

  int problem_size_n[] = {kAlignmentN, 512 - 2 * kAlignmentN};

  int problem_size_k[] = {
      kAlignmentK, Gemm::ThreadblockShape::kK * (Gemm::kStages + 1) - kAlignmentK};

  int split_k_slices[] = {
    1, 2, 3
  };

  double problem_alpha[] = {
    1
  };

  double problem_beta[] = {
    2.0
  };

  Testbed<Gemm, Relu> testbed(stride_factor_A, stride_factor_B, stride_factor_C);

  using ElementCompute = typename Gemm::EpilogueOutputOp::ElementCompute;

  for (int m : problem_size_m) {
    for (int n : problem_size_n) {
      for (int k : problem_size_k) {
        for (int split_k : split_k_slices) {

          if (!Gemm::kSplitKSerial && split_k > 1) {
            continue;
          }

          if (split_k > 1 && k / Gemm::ThreadblockShape::kK < split_k) {
            continue;
          }

          for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {

              cutlass::gemm::GemmCoord problem_size(m, n, k);
              try {
                passed = testbed.run(
                  problem_size, 
                  split_k,
                  cutlass::from_real<ElementCompute>(alpha), 
                  cutlass::from_real<ElementCompute>(beta)
                );
              }
              catch (std::exception const& e) {
                EXPECT_TRUE(false) << "TestAllGemmBasic: testbed.run threw an "
                  "exception {alpha: " << alpha << ", beta: " << beta << ", m: "
                  << m << ", n: " << n << ", k: " << k << "}: " << e.what();
                throw;
              }
              catch (...) {
                EXPECT_TRUE(false) << "TestAllGemmBasic: testbed.run threw an "
                  "exception {alpha: " << alpha << ", beta: " << beta << ", m: "
                  << m << ", n: " << n << ", k: " << k << "}: (unknown)";
                throw;
              }

              if (!passed) {
                return false;
              }
            }
          }
        }
      }
    }
  }

  return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm, bool Relu=false>
bool TestAllGemm(
    const typename Gemm::LayoutA::Stride& stride_factor_A,
    const typename Gemm::LayoutB::Stride& stride_factor_B = typename Gemm::LayoutB::Stride(),
    const typename Gemm::LayoutC::Stride& stride_factor_C = typename Gemm::LayoutC::Stride())
{
  // Test basic GEMM with non-default stride factors
  return TestAllGemmBasic<Gemm, Relu>(stride_factor_A, stride_factor_B, stride_factor_C);
}

template <typename Gemm, bool Relu=false>
bool TestAllGemm()
{
#ifdef NDEBUG
  // Non-debug builds also test basic GEMM with default stride factors
  if (!TestAllGemmBasic<Gemm, Relu>()) {
    return false;
  }
#endif // NDEBUG

  // Test universal GEMM
#if 0
  // Define the universal kernel
  using UniversalKernel = cutlass::gemm::kernel::GemmUniversal<
    typename Gemm::GemmKernel::Mma,                                 // Mma
    typename Gemm::GemmKernel::Epilogue,                            // Epilogue
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>    // ThreadblockSwizzle
  >;
#else
  // Define the streamk universal kernel
  using UniversalKernel = cutlass::gemm::kernel::GemmUniversalStreamk<
    typename Gemm::GemmKernel::Mma,                                 // Mma
    typename Gemm::GemmKernel::Epilogue,                            // Epilogue
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK           // ThreadblockSwizzle
  >;
#endif

  // Define the universal adaptor
  using UniversalGemm = cutlass::gemm::device::GemmUniversalAdapter<UniversalKernel>;

  // Test universal GEMM
  return TestAllGemmUniversal<UniversalGemm, Relu>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Gemm>
bool TestGemmPerf(int iterations = 1) {
  bool passed = true;

  int problem_size_m[] = { 2048 };

  int problem_size_n[] = { 4352 };

  int problem_size_k[] = { 4096  };

  int split_k_slices[] = { 1 };
  double problem_alpha[] = { 1 };
  double problem_beta[] = { 0.0 };

  Testbed<Gemm> testbed;

  using ElementCompute = typename Gemm::EpilogueOutputOp::ElementCompute;

  for (int m : problem_size_m) {
    for (int n : problem_size_n) {
      for (int k : problem_size_k) {
        for (int split_k : split_k_slices) {

          if (!Gemm::kSplitKSerial && split_k > 1) {
            continue;
          }

          for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {

              cutlass::gemm::GemmCoord problem_size(m, n, k);

              for (int i = 0; i < iterations; i++){
                try {
                  passed = testbed.run(
                    problem_size, 
                    split_k,
                    cutlass::from_real<ElementCompute>(alpha), 
                    cutlass::from_real<ElementCompute>(beta)
                  );
                }
                catch (std::exception const& e) {
                  EXPECT_TRUE(false) << "TestGemmPerf: testbed.run threw an "
                    "exception {alpha: " << alpha << ", beta: " << beta << ", m: "
                    << m << ", n: " << n << ", k: " << k << "}: " << e.what();
                  throw;
                }
                catch (...) {
                  EXPECT_TRUE(false) << "TestGemmPerf: testbed.run threw an "
                    "exception {alpha: " << alpha << ", beta: " << beta << ", m: "
                    << m << ", n: " << n << ", k: " << k << "}: (unknown)";
                  throw;
                }
              }

              if (!passed) {
                return false;
              }
            }
          }
        }
      }
    }
  }

  return passed;
}

} // namespace device
} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

