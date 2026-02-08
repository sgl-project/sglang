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

  Testbed for sparse operations not to be released for CUDA 11.0 GA. Expected release is 11.1.
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
#include "cutlass/util/host_reorder.h"
#include "cutlass/util/host_uncompress.h"

#include "testbed_utils.h"

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
struct SparseTestbed {

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  using ElementAccumulator = typename Gemm::ElementAccumulator;
  using ElementCompute = typename Gemm::GemmKernel::Epilogue::OutputOp::ElementCompute;

  static int const kSparse = Gemm::GemmKernel::kSparse;
  static int const kMetaSizeInBits = Gemm::GemmKernel::kMetaSizeInBits;
  static int const kMaxID2 = Gemm::GemmKernel::kMaxID2;
  static int const kElementsPerElementE = Gemm::GemmKernel::kElementsPerElementE;

  using ElementE = typename Gemm::GemmKernel::ElementE;
  using LayoutE = cutlass::layout::RowMajor;
  using ReorderedLayoutE = typename Gemm::GemmKernel::LayoutE;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  cutlass::Distribution::Kind init_E;
  uint64_t seed;

  cutlass::HostTensor<typename Gemm::ElementA, typename Gemm::LayoutA> tensor_A;
  cutlass::HostTensor<typename Gemm::ElementA, typename Gemm::LayoutA> tensor_A_uncompressed;
  cutlass::HostTensor<typename Gemm::ElementB, typename Gemm::LayoutB> tensor_B;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_C;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_D;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> reference_D;
  cutlass::HostTensor<ElementE, LayoutE> tensor_E;
  cutlass::HostTensor<ElementE, ReorderedLayoutE> tensor_E_reordered;

  //
  // Methods
  //

  SparseTestbed(
      cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_E_ = cutlass::Distribution::Uniform,
      uint64_t seed_ = 2080)
      : init_A(init_A_),
        init_B(init_B_),
        init_C(init_C_),
        init_E(init_E_),
        seed(seed_) {}

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
    tensor_A.resize(cutlass::make_Coord(problem_size.m(), problem_size.k() / kSparse));
    tensor_A_uncompressed.resize(problem_size.mk());
    tensor_B.resize(problem_size.kn());
    tensor_C.resize(problem_size.mn());
    tensor_D.resize(problem_size.mn());
    reference_D.resize(problem_size.mn(), false);
    tensor_E.resize(cutlass::make_Coord(
        problem_size.m(), problem_size.k() / kSparse / kElementsPerElementE));
    tensor_E_reordered.resize(cutlass::make_Coord(
        problem_size.m(), problem_size.k() / kSparse / kElementsPerElementE));

    EXPECT_TRUE(initialize_tensor(tensor_A.host_view(), init_A, seed + 2019));
    EXPECT_TRUE(initialize_tensor(tensor_B.host_view(), init_B, seed + 2018));
    EXPECT_TRUE(initialize_tensor(tensor_C.host_view(), init_C, seed + 2017));

    if (init_E == cutlass::Distribution::Uniform) {
      uint64_t seed = 7;
      cutlass::reference::host::TensorFillRandomSparseMeta(
          tensor_E.host_view(), seed, kMetaSizeInBits);
    } else if (init_E == cutlass::Distribution::Identity) {
      uint32_t content = (kMaxID2 == 1) ? 0x44444444 : 0x4444;
      cutlass::reference::host::TensorFill(tensor_E.host_view(),
                                           (ElementE)(content));
    } else {
      EXPECT_TRUE(false);
    }

    cutlass::reorder_meta(tensor_E_reordered.host_ref(), tensor_E.host_ref(),
                          {problem_size.m(), problem_size.n(),
                           problem_size.k() / kSparse / kElementsPerElementE});

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_A.host_view().at({0, 0}) = typename Gemm::ElementA(1);
    tensor_B.host_view().at({0, 0}) = typename Gemm::ElementB(1);
    tensor_C.host_view().at({0, 0}) = typename Gemm::ElementC(1);

    cutlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
    tensor_E_reordered.sync_device();
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

    bool passed = cutlass::reference::host::TensorEquals(reference_D.host_view(), tensor_D.host_view());

    EXPECT_TRUE(passed);

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
        << "\nE =\n" << tensor_E.host_view()
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

    cutlass::uncompress(tensor_A_uncompressed.host_ref(), tensor_A.host_ref(),
                        tensor_E.host_ref(), problem_size.m(), problem_size.k());

    cutlass::reference::host::Gemm<
        typename Gemm::ElementA, typename Gemm::LayoutA,
        typename Gemm::ElementB, typename Gemm::LayoutB,
        typename Gemm::ElementC, typename Gemm::LayoutC, 
        ElementCompute,
        ElementAccumulator, typename Gemm::Operator>
        reference_gemm;

    reference_gemm(
      problem_size,
      alpha, 
      tensor_A_uncompressed.host_ref(), 
      tensor_B.host_ref(), 
      beta, 
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
    ElementCompute beta = ElementCompute(0)) {

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
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      split_k_slices,
      {alpha, beta},
      tensor_A.device_data(),
      tensor_B.device_data(),
      tensor_C.device_data(),
      tensor_D.device_data(),
      tensor_E_reordered.device_data(),
      int64_t(),
      int64_t(),
      int64_t(),
      int64_t(),
      int64_t(),
      tensor_A.layout().stride(0),                                     
      tensor_B.layout().stride(0),
      tensor_C.layout().stride(0),
      tensor_D.layout().stride(0),                                     
      tensor_E_reordered.layout().stride(0)
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);

    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());

		// This failure is likely due to insufficient device capabilities. Waive the test.
    if (status != cutlass::Status::kSuccess) {
      return true;
    }

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
      std::cout << "Error with split_k_slices = " << split_k_slices << ", alpha: " << alpha << ", beta: " << beta << ", m: " << problem_size.m() << ", n: " << problem_size.n() << ", k:" <<problem_size.k() << std::endl;
    }

    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
bool TestAllSparseGemm() {
  bool passed = true;

  int const kMinimumOperandElementSize = 
    std::min(
      int(cutlass::sizeof_bits<typename Gemm::ElementA>::value), 
      int(cutlass::sizeof_bits<typename Gemm::ElementB>::value));

  // M dimension has to be multiple of 32 (sparse float) or 16 (sparse int)
  // because of the reordering of operand E
  int const kAlignmentM = std::max(((sizeof(typename Gemm::ElementE) == 2) ? 32 : 16),
                                   kMinimumOperandElementSize);

  int const kAlignmentN = 128 / kMinimumOperandElementSize;

  int problem_size_m[] = {kAlignmentM, 512 - 3 * kAlignmentM};

  int problem_size_n[] = {kAlignmentN, 512 - 2 * kAlignmentN};

  int problem_size_k[] = {Gemm::ThreadblockShape::kK * 8};

  int split_k_slices[] = {
    1, 2
  };

  double problem_alpha[] = {
    1
  };

  double problem_beta[] = {
    2.0
  };

  SparseTestbed<Gemm> testbed;

  using ElementCompute = typename Gemm::EpilogueOutputOp::ElementCompute;

  for (int m : problem_size_m) {
    for (int n : problem_size_n) {
      for (int k : problem_size_k) {
        for (int split_k : split_k_slices) {

          for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {
              cutlass::gemm::GemmCoord problem_size(m, n, k);

              passed = testbed.run(
                problem_size, 
                split_k,
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
