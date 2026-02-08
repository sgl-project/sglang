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
    \brief Tests for device-wide Symm update interface
  
*/

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>

#include "../../common/cutlass_unit_test.h"
#include "cutlass/blas3.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/reference/host/symm.h"
#include "cutlass/util/reference/host/symm_complex.h"

#include "testbed_utils.h"

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Symm>
struct TestbedSymmUniversal {

  using ElementA = typename Symm::ElementA;
  using ElementB = typename Symm::ElementB;
  using ElementC = typename Symm::ElementC;
  using ElementAccumulator = typename Symm::ElementAccumulator;
  using ElementCompute = typename Symm::SymmKernel::Epilogue::OutputOp::ElementCompute;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;

  cutlass::HostTensor<typename Symm::ElementA, typename Symm::LayoutA> tensor_A;
  cutlass::HostTensor<typename Symm::ElementB, typename Symm::LayoutB> tensor_B;
  cutlass::HostTensor<typename Symm::ElementC, typename Symm::LayoutC> tensor_C;
  cutlass::HostTensor<typename Symm::ElementC, typename Symm::LayoutC> tensor_D;
  cutlass::HostTensor<typename Symm::ElementC, typename Symm::LayoutC> reference_D;

  //
  // Methods
  //

  TestbedSymmUniversal(
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
    uint64_t seed,
    int mantissa_in_bits) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      double scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<typename Symm::ElementC>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      } else if (bits_output == 16) {
        scope_max = 5;
        scope_min = -5;
      } else {
        scope_max = 8;
        scope_min = -8;
      }

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, scope_max, scope_min, mantissa_in_bits);
    } 
    else if (dist_kind == cutlass::Distribution::Identity) {

      cutlass::reference::host::TensorFillIdentity(view);
    } 
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5, mantissa_in_bits);
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {

      cutlass::reference::host::BlockFillSequential(
        view.data(), view.capacity());
    } 
    else {

      EXPECT_TRUE(false) << "Input distribution not implemented";
      return false;
    }

    return true;
  }


  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool initialize_symmetric_tensor(
    cutlass::TensorView<Element, Layout> view, 
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed,
    int mantissa_in_bits) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      double scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<typename Symm::ElementC>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      } else if (bits_output == 16) {
        scope_max = 5;
        scope_min = -5;
      } else {
        scope_max = 8;
        scope_min = -8;
      }

      cutlass::reference::host::TensorFillSymmetricRandomUniform(
        view, seed, Symm::kFillModeA, scope_max, scope_min, mantissa_in_bits);
    } 
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::host::TensorFillSymmetricRandomGaussian(
        view, seed, Symm::kFillModeA, 0, 0.5, mantissa_in_bits);
    }
    else {

      EXPECT_TRUE(false) << "Input distribution (symmetric tensor) not implemented";
      return false;
    }

    return true;
  }
  /// Initializes data structures
  void initialize(cutlass::gemm::GemmCoord problem_size) {
    //
    // Allocate the Symm workspace
    //

    if (Symm::kSideModeA == cutlass::SideMode::kLeft) {
      tensor_A.resize(cutlass::make_Coord(problem_size.m(),problem_size.m()));
    }
    else if (Symm::kSideModeA == cutlass::SideMode::kRight) {
      tensor_A.resize(cutlass::make_Coord(problem_size.n(),problem_size.n()));
    }

    tensor_B.resize(problem_size.mn());
    tensor_C.resize(problem_size.mn());
    tensor_D.resize(problem_size.mn());
    reference_D.resize(problem_size.mn(), false);

    EXPECT_TRUE(initialize_symmetric_tensor(tensor_A.host_view(), init_A, seed + 2019, cutlass::MantissaInBits<typename Symm::ElementA>::bits));
    EXPECT_TRUE(initialize_tensor(tensor_B.host_view(), init_B, seed + 2018, cutlass::MantissaInBits<typename Symm::ElementB>::bits));
    EXPECT_TRUE(initialize_tensor(tensor_C.host_view(), init_C, seed + 2017, cutlass::MantissaInBits<typename Symm::ElementC>::bits));

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_A.host_view().at({0, 0}) = typename Symm::ElementA(1);
    tensor_B.host_view().at({0, 0}) = typename Symm::ElementB(1);
    tensor_C.host_view().at({0, 0}) = typename Symm::ElementC(1);

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

    if (tensor_D.size() > 1)
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_D.host_view()), 0);

    if (reference_D.size() > 1)
      EXPECT_GT(cutlass::reference::host::TensorNorm(reference_D.host_view()), 0);

    double l2_norm = cutlass::reference::host::TensorRelativeErrorMetric(reference_D.host_view(), tensor_D.host_view());

    bool passed = l2_norm < cutlass::MantissaInBits<typename Symm::ElementA>::error;

    return passed;
  }

  /// Verifies the result is a Symm
  bool verify(
    cutlass::gemm::GemmCoord problem_size, 
    ElementCompute alpha, 
    ElementCompute beta) {

    //
    // Verify
    //

    using HostReference = typename cutlass::platform::conditional<
                              (cutlass::platform::is_same<typename Symm::ElementC,
                                                          cutlass::complex<double>
                                                         >::value ||
                              cutlass::platform::is_same<typename Symm::ElementC,
                                                          cutlass::complex<float>
                                                         >::value
                              ), 
                              cutlass::reference::host::SymmComplex<
                                  typename Symm::ElementA, typename Symm::LayoutA,
                                  Symm::kSideModeA, Symm::kFillModeA,
                                  typename Symm::ElementB, typename Symm::LayoutB,
                                  typename Symm::ElementC, typename Symm::LayoutC, 
                                  ElementCompute,
                                  ElementAccumulator,
                                  Symm::kBlasMode>,
                              cutlass::reference::host::Symm<
                                  typename Symm::ElementA, typename Symm::LayoutA,
                                  Symm::kSideModeA, Symm::kFillModeA, 
                                  typename Symm::ElementB, typename Symm::LayoutB,
                                  typename Symm::ElementC, typename Symm::LayoutC, 
                                  ElementCompute,
                                  ElementAccumulator>
                           >::type;


    HostReference reference_symm;

    reference_symm(
      problem_size,
      alpha, 
      tensor_A.host_ref(),
      tensor_B.host_ref(),
      beta, 
      tensor_C.host_ref(), 
      reference_D.host_ref(),
      ElementAccumulator(0)
    );

    return compare_reference(problem_size, alpha, beta);
  }

  /// Returns true if the CUDA device is sufficient to execute the kernel.
  bool sufficient() const {
    //
    // Determine SMEM requirements and waive if not satisfied
    //

    size_t smem_size = sizeof(typename Symm::SymmKernel::SharedStorage);

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
    ElementCompute beta = ElementCompute(0)) {

    // Waive test if insufficient CUDA device
    if (!sufficient()) {
      if (CUTLASS_TEST_UNIT_ENABLE_WARNINGS) {
        std::cerr << "Test waived due to insufficient CUDA device." << std::endl;
      }
      return true;
    }

#if 0
    std::cout << "[TestbedSymmUniversal::run()] problem(m, n, k): " << problem_size
              << " alpha: " << ElementCompute(alpha)
              << " beta: " << ElementCompute(beta) << std::endl;
#endif

    this->initialize(problem_size);

    //
    // Initialize the Symm operator
    //

    int batch_stride_A;
    if (Symm::kSideModeA == cutlass::SideMode::kLeft)
      batch_stride_A = problem_size.m()*problem_size.m();
    if (Symm::kSideModeA == cutlass::SideMode::kRight)
      batch_stride_A = problem_size.n()*problem_size.n();

    typename Symm::Arguments arguments{
      mode,
      problem_size,
      batch_count,
      {alpha, beta},
      tensor_A.device_data(),
      tensor_B.device_data(),
      tensor_C.device_data(),
      tensor_D.device_data(),
      batch_stride_A,
      problem_size.m() * problem_size.n(),
      problem_size.m() * problem_size.n(),
      problem_size.m() * problem_size.n(),
      tensor_A.layout().stride(0),
      tensor_B.layout().stride(0),
      tensor_C.layout().stride(0),
      tensor_D.layout().stride(0)
    };

    Symm symm_op;

    size_t workspace_size = Symm::get_workspace_size(arguments);

    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = symm_op.initialize(arguments, workspace.get());

    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    //
    // Run the Symm
    //

    status = symm_op();

    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    //
    // Verify
    //

    bool passed = this->verify(problem_size, alpha, beta);

    //if (true) {
    if (!passed) {
      std::stringstream fname;

      fname << "error_"
            << (Symm::kBlasMode == cutlass::BlasMode::kSymmetric ? "symm_" : "hemm_" )
            << "device_"
            << "fill_mode_a_"
            << (Symm::kSideModeA == cutlass::SideMode::kLeft ? "leftside_" :
                (Symm::kSideModeA == cutlass::SideMode::kRight ? "rightside_" : "invalid_"))            
            << (Symm::kFillModeA == cutlass::FillMode::kLower ? "lower_" :
                (Symm::kFillModeA == cutlass::FillMode::kUpper ? "upper_" : "invalid_"))
            << "mnk_"
            << problem_size.m() << "x"
            << problem_size.n() << "x"
            << problem_size.k() << "_"
            << Symm::ThreadblockShape::kM << "x"  
            << Symm::ThreadblockShape::kN << "x"  
            << Symm::ThreadblockShape::kK << "_"
            << Symm::WarpShape::kM << "x"  
            << Symm::WarpShape::kN << "x"  
            << Symm::WarpShape::kK << ".txt";

      std::cout << fname.str() << std::endl;

      std::ofstream results(fname.str());

      results << problem_size << std::endl;

      results
        << "alpha: " << ElementCompute(alpha) << "\n"
        << "beta: "  << ElementCompute(beta) << "\n"
        << "\nA:\n" << tensor_A.host_view() << "\n"
        << "\nB:\n" << tensor_B.host_view() << "\n"
        << "\nC:\n" << tensor_C.host_view() << "\n"
        << "\nD reference:\n" << reference_D.host_view() << "\n"
        << "\nD computed:\n" << tensor_D.host_view() << "\n";

    }

    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Symm>
bool TestsymmUniversal(
  cutlass::gemm::GemmCoord const & problem_size,
  cutlass::gemm::GemmUniversalMode mode,
  int batch_count,
  double alpha = 1.0, 
  double beta = 2.0) {

  bool passed = true;

  TestbedSymmUniversal<Symm> testbed;
  
  using ElementCompute = typename Symm::EpilogueOutputOp::ElementCompute;

  passed = testbed.run(
    mode,
    problem_size,
    batch_count,
    cutlass::from_real<ElementCompute>(alpha), 
    cutlass::from_real<ElementCompute>(beta)
  );

  return passed;
}

template <typename Symm>
bool TestAllSymmUniversal() {
  bool passed = true;


  int const kMinimumOperandElementSize = int(cutlass::sizeof_bits<typename Symm::ElementA>::value);

  int const kAlignment = cutlass::platform::is_same<
                              typename Symm::OperatorClass, 
                              cutlass::arch::OpClassSimt>::value ? 1 : 128 / kMinimumOperandElementSize;

  // int8_t gemm alignment constraints
  int const kAlignmentM = cutlass::platform::is_same<typename Symm::OperatorClass, cutlass::arch::OpClassSimt>::value &&
                          cutlass::platform::is_same<typename Symm::ElementA, int8_t>::value &&
                          cutlass::platform::is_same<typename Symm::LayoutA, cutlass::layout::ColumnMajor>::value ? 4 : kAlignment;

  int const kAlignmentN = kAlignmentM;

  int const kAlignmentK = cutlass::platform::is_same<typename Symm::OperatorClass, cutlass::arch::OpClassSimt>::value &&
                          cutlass::platform::is_same<typename Symm::ElementA, int8_t>::value &&
                          cutlass::platform::is_same<typename Symm::LayoutA, cutlass::layout::RowMajor>::value
                           ? 4 : kAlignment;

  cutlass::gemm::GemmUniversalMode modes[] = {
    cutlass::gemm::GemmUniversalMode::kGemm,
  };

  int problem_size_m[] = {
    kAlignmentK, 
    Symm::ThreadblockShape::kK * Symm::kStages - kAlignmentK, 
    Symm::ThreadblockShape::kK * Symm::kStages * 3 - kAlignmentK
  };

  int problem_size_n[] = {
    kAlignmentN, 512 - 2*kAlignmentN
  };

  int batch_counts[] = {      // may be interpretted as batch count or split-K slices
    1                         // Just running one batch for now (removing 2, 3, 5, 7)
  };

  double problem_alpha[] = {
    1.0, 3.0
  };

  double problem_beta[] = {
    0, 2.0
  };


  using ElementCompute = typename Symm::EpilogueOutputOp::ElementCompute;

  for (cutlass::gemm::GemmUniversalMode mode : modes) {
    for (int m : problem_size_m) {
      for (int n : problem_size_n) {
        for (int batch_count : batch_counts) {

          for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {
              
              int k = 0;
              if (Symm::kSideModeA == cutlass::SideMode::kLeft)
                k = m;
              else if (Symm::kSideModeA == cutlass::SideMode::kRight)
                k = n;

              if (mode == cutlass::gemm::GemmUniversalMode::kGemm ||
                mode == cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel) {

  #if 0
                // skip very small K problems
                if (k / batch_count < 2 * Symm::ThreadblockShape::kK) {
                  continue;
                }
  #endif
              }

              cutlass::gemm::GemmCoord problem_size(m, n, k);

              TestbedSymmUniversal<Symm> testbed;

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

  return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////
