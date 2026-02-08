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
    \brief Tests for device-wide TRMM interface

  
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
#include "cutlass/util/reference/host/trmm.h"
#include "cutlass/util/reference/host/trmm_complex.h"
#include "cutlass/core_io.h"

#include "testbed_utils.h"

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Trmm>
struct TestbedTrmmUniversal {

  using ElementA = typename Trmm::ElementA;
  using ElementB = typename Trmm::ElementB;
  using ElementC = typename Trmm::ElementC;
  using ElementAccumulator = typename Trmm::ElementAccumulator;
  using ElementCompute = typename Trmm::TrmmKernel::Epilogue::OutputOp::ElementCompute;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_D;
  uint64_t seed;

  cutlass::HostTensor<typename Trmm::ElementA, typename Trmm::LayoutA> tensor_A;
  cutlass::HostTensor<typename Trmm::ElementB, typename Trmm::LayoutB> tensor_B;
  cutlass::HostTensor<typename Trmm::ElementC, typename Trmm::LayoutC> tensor_D;
  cutlass::HostTensor<typename Trmm::ElementC, typename Trmm::LayoutC> reference_D;

  //
  // Methods
  //

  TestbedTrmmUniversal(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_D_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    init_A(init_A_), init_B(init_B_), init_D(init_D_), seed(seed_) { }

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
      int bits_output = cutlass::sizeof_bits<typename Trmm::ElementC>::value;

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
      EXPECT_TRUE(false) << "Not implemented";
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
      int bits_output = cutlass::sizeof_bits<typename Trmm::ElementC>::value;

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
        view, seed, Trmm::kFillMode, scope_max, scope_min, mantissa_in_bits);
    } 
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::host::TensorFillSymmetricRandomGaussian(
        view, seed, Trmm::kFillMode, 0, 0.5, mantissa_in_bits);
    }
    else {
      EXPECT_TRUE(false) << "Not implemented";
      return false;
    }

    return true;
  }

  /// Helper to initialize a tensor view (pad diagonal fill with zeros for up to alignment on wrong side of diagonal)
  template <typename Element, typename Layout>
  bool initialize_pad_diagonal_tensor(
    cutlass::TensorView<Element, Layout> view, 
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed,
    int alignment) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      double scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<typename Trmm::ElementC>::value;

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

      cutlass::reference::host::TensorFillPadDiagonalRandomUniform(
        view, seed, Trmm::kFillMode, scope_max, scope_min, 0, alignment);
    } 
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      EXPECT_TRUE(false) << "Gaussian distribution for pad diagonal not implemented";
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
    // Allocate the TRMM workspace
    //

    if (Trmm::kSideMode == cutlass::SideMode::kLeft) {
      tensor_A.resize(cutlass::make_Coord(problem_size.m(),problem_size.m()));
    }
    else if (Trmm::kSideMode == cutlass::SideMode::kRight) {
      tensor_A.resize(cutlass::make_Coord(problem_size.n(),problem_size.n()));
    }

    tensor_B.resize(problem_size.mn());
    tensor_D.resize(problem_size.mn());
    reference_D.resize(problem_size.mn(), false);

    //EXPECT_TRUE(initialize_symmetric_tensor(tensor_A.host_view(), init_A, seed + 2017));
    //EXPECT_TRUE(initialize_pad_diagonal_tensor(tensor_A.host_view(), init_A, seed + 2017, Trmm::kAlignmentA));
    EXPECT_TRUE(initialize_tensor(tensor_A.host_view(), init_A, seed + 2017, cutlass::MantissaInBits<typename Trmm::ElementA>::bits));
    EXPECT_TRUE(initialize_tensor(tensor_B.host_view(), init_B, seed + 2019, cutlass::MantissaInBits<typename Trmm::ElementB>::bits));

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_A.host_view().at({0, 0}) = typename Trmm::ElementA(1);
    tensor_B.host_view().at({0, 0}) = typename Trmm::ElementB(1);

    cutlass::reference::host::TensorCopy(reference_D.host_view(), tensor_D.host_view());

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_D.sync_device();
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(
    cutlass::gemm::GemmCoord problem_size,
    ElementCompute alpha) {

    tensor_D.sync_host();

    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_A.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_B.host_view()), 0);

    if (tensor_D.size() > 1)
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_D.host_view()), 0);

    if (reference_D.size() > 1)
      EXPECT_GT(cutlass::reference::host::TensorNorm(reference_D.host_view()), 0);

    double l2_norm = cutlass::reference::host::TensorRelativeErrorMetric(reference_D.host_view(), tensor_D.host_view());

    bool passed = l2_norm < cutlass::MantissaInBits<typename Trmm::ElementA>::error;

    return passed;
  }

  /// Verifies the result is a TRMM
  bool verify(
    cutlass::gemm::GemmCoord problem_size, 
    ElementCompute alpha) {

    //
    // Verify
    //

    using HostReference = typename cutlass::platform::conditional<
                              (cutlass::platform::is_same<typename Trmm::ElementC,
                                                          cutlass::complex<double>
                                                         >::value ||
                              cutlass::platform::is_same<typename Trmm::ElementC,
                                                          cutlass::complex<float>
                                                         >::value
                              ), 
                              cutlass::reference::host::TrmmComplex<
                                  typename Trmm::ElementA, typename Trmm::LayoutA,
                                  Trmm::kTransformA,
                                  Trmm::kSideMode, Trmm::kFillMode, Trmm::kDiagType,
                                  typename Trmm::ElementB, typename Trmm::LayoutB,
                                  Trmm::kTransformB,
                                  typename Trmm::ElementC, typename Trmm::LayoutC, 
                                  ElementCompute,
                                  ElementAccumulator>,
                              cutlass::reference::host::Trmm<
                                  typename Trmm::ElementA, typename Trmm::LayoutA,
                                  Trmm::kSideMode, Trmm::kFillMode, Trmm::kDiagType,
                                  typename Trmm::ElementB, typename Trmm::LayoutB,
                                  typename Trmm::ElementC, typename Trmm::LayoutC, 
                                  ElementCompute,
                                  ElementAccumulator>
                           >::type;


    HostReference reference_trmm;

    reference_trmm(
      problem_size,
      alpha, 
      tensor_A.host_ref(),
      tensor_B.host_ref(),
      reference_D.host_ref(), 
      ElementAccumulator(0)
    );

    return compare_reference(problem_size, alpha);
  }
  
  /// Returns true if the CUDA device is sufficient to execute the kernel.
  bool sufficient() const {
    //
    // Determine SMEM requirements and waive if not satisfied
    //

    size_t smem_size = sizeof(typename Trmm::TrmmKernel::SharedStorage);

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
    ElementCompute alpha = ElementCompute(1)) {

    // Waive test if insufficient CUDA device
    if (!sufficient()) {
      if (CUTLASS_TEST_UNIT_ENABLE_WARNINGS) {
        std::cerr << "Test waived due to insufficient CUDA device." << std::endl;
      }
      return true;
    }

#if 0
    std::cout << "[TestbedTrmmUniversal::run()] problem(m, n, k): " << problem_size
              << " alpha: " << ElementCompute(alpha) << std::endl;
#endif

    this->initialize(problem_size);

    //
    // Initialize the TRMM operator
    //

    int batch_stride_A;
    if (Trmm::kSideMode == cutlass::SideMode::kLeft)
      batch_stride_A = problem_size.m()*problem_size.m();
    if (Trmm::kSideMode == cutlass::SideMode::kRight)
      batch_stride_A = problem_size.n()*problem_size.n();

    typename Trmm::Arguments arguments{
      mode,
      problem_size,
      batch_count,
      {alpha},
      tensor_A.device_data(),
      tensor_B.device_data(),
      tensor_D.device_data(),
      batch_stride_A,
      problem_size.m() * problem_size.n(),
      problem_size.m() * problem_size.n(),
      tensor_A.layout().stride(0),
      tensor_B.layout().stride(0),
      tensor_D.layout().stride(0)
    };

    Trmm trmm_op;

    size_t workspace_size = Trmm::get_workspace_size(arguments);

    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = trmm_op.initialize(arguments, workspace.get());

    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    //
    // Run the TRMM
    //

    status = trmm_op();

    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    //
    // Verify
    //
    bool passed = this->verify(problem_size, alpha);

    if (!passed) {
      std::stringstream fname;

      fname << "error_Trmm_device_"
            << "fill_mode_"
            << (Trmm::kFillMode == cutlass::FillMode::kLower ? "lower_" :
                (Trmm::kFillMode == cutlass::FillMode::kUpper ? "upper_" : "invalid_"))
            << "side_mode_"
            << (Trmm::kSideMode == cutlass::SideMode::kLeft ? "left_" :
                (Trmm::kSideMode == cutlass::SideMode::kRight ? "right_" : "invalid_")) 
            << "mnk_"
            << problem_size.m() << "x"
            << problem_size.n() << "x"
            << problem_size.k() << "_"
            << Trmm::ThreadblockShape::kM << "x"  
            << Trmm::ThreadblockShape::kN << "x"  
            << Trmm::ThreadblockShape::kK << "_"
            << Trmm::WarpShape::kM << "x"  
            << Trmm::WarpShape::kN << "x"  
            << Trmm::WarpShape::kK << ".txt";

      std::cout << fname.str() << std::endl;

      std::ofstream results(fname.str());

      results << problem_size << std::endl;

      results
        << "\nA:\n" << tensor_A.host_view() << "\n"
        << "\nB:\n" << tensor_B.host_view() << "\n"
        << "\nD reference:\n" << reference_D.host_view() << "\n"
        << "\nD computed:\n" << tensor_D.host_view() << "\n";
    }

    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Trmm>
bool TestTrmmUniversal(
  cutlass::gemm::GemmCoord const & problem_size,
  cutlass::gemm::GemmUniversalMode mode,
  int batch_count,
  double alpha = 1.0) {

  bool passed = true;

  TestbedTrmmUniversal<Trmm> testbed;
  
  using ElementCompute = typename Trmm::EpilogueOutputOp::ElementCompute;

  passed = testbed.run(
    mode,
    problem_size,
    batch_count,
    cutlass::from_real<ElementCompute>(alpha) 
  );

  return passed;
}

template <typename Trmm>
bool TestAllTrmmUniversal() {
  bool passed = true;

  int const kMinimumOperandElementSize = int(cutlass::sizeof_bits<typename Trmm::ElementA>::value);

  int const kAlignment = cutlass::platform::is_same<
                              typename Trmm::OperatorClass, 
                              cutlass::arch::OpClassSimt>::value ? 1 : 128 / kMinimumOperandElementSize;

  // int8_t gemm alignment constraints
  int const kAlignmentM = cutlass::platform::is_same<typename Trmm::OperatorClass, cutlass::arch::OpClassSimt>::value &&
                          cutlass::platform::is_same<typename Trmm::ElementA, int8_t>::value &&
                          cutlass::platform::is_same<typename Trmm::LayoutA, cutlass::layout::ColumnMajor>::value ? 4 : kAlignment;

  int const kAlignmentN = kAlignmentM;

  int const kAlignmentK = cutlass::platform::is_same<typename Trmm::OperatorClass, cutlass::arch::OpClassSimt>::value &&
                          cutlass::platform::is_same<typename Trmm::ElementA, int8_t>::value &&
                          cutlass::platform::is_same<typename Trmm::LayoutA, cutlass::layout::RowMajor>::value
                           ? 4 : kAlignment;

  cutlass::gemm::GemmUniversalMode modes[] = {
    cutlass::gemm::GemmUniversalMode::kGemm,
  };

  int problem_size_m[] = {
    kAlignmentK, 
    Trmm::ThreadblockShape::kK * Trmm::kStages - kAlignmentK, 
    Trmm::ThreadblockShape::kK * Trmm::kStages * 3 - kAlignmentK
  };

  int problem_size_n[] = {
    kAlignmentN, 512 - 2*kAlignmentN
  };

  int batch_counts[] = {      // may be interpretted as batch count or split-K slices
    1                         // Just running one batch for now (removing 2, 3, 5, 7)
  };

  double problem_alpha[] = {
    1.0, 2.0
  };

  using ElementCompute = typename Trmm::EpilogueOutputOp::ElementCompute;

  for (cutlass::gemm::GemmUniversalMode mode : modes) {
    for (int m : problem_size_m) {
      for (int n : problem_size_n) {
        for (int batch_count : batch_counts) {
          for (auto alpha : problem_alpha) {
            
            int k = 0;
            if (Trmm::kSideMode == cutlass::SideMode::kLeft)
              k = m;
            else if (Trmm::kSideMode == cutlass::SideMode::kRight)
              k = n;

            if (mode == cutlass::gemm::GemmUniversalMode::kGemm ||
              mode == cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel) {

#if 0
              // skip very small K problems
              if (k / batch_count < 2 * Trmm::ThreadblockShape::kK) {
                continue;
              }
#endif
            }
            
            cutlass::gemm::GemmCoord problem_size(m, n, k);

            TestbedTrmmUniversal<Trmm> testbed;

            passed = testbed.run(
              mode,
              problem_size,
              batch_count,
              cutlass::from_real<ElementCompute>(alpha) 
            );

            if (!passed) {
              return false;
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

