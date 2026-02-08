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
    \brief Depthwise Direct Conv testbed
*/
#pragma once

#include <fstream>

#include "../../common/cutlass_unit_test.h"
#include "../cache_testbed_output.h"
#include "conv2d_problems.h"
#include "cutlass/conv/device/direct_convolution.h"

#include "cutlass/core_io.h"
#include "cutlass/cutlass.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/convolution.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

namespace test {
namespace conv {
namespace device {

template <typename Conv2d>
class TestbedDepthwiseDirectConv2d {
 public:
 
  using ElementA = typename Conv2d::ElementA;
  using LayoutA = typename Conv2d::LayoutA;
  using ElementB = typename Conv2d::ElementB;
  using LayoutB = typename Conv2d::LayoutB;
  using ElementC = typename Conv2d::ElementC;
  using LayoutC = typename Conv2d::LayoutC;
  using ElementAccumulator = typename Conv2d::ElementAccumulator;
  using ElementCompute = typename Conv2d::ElementCompute;
  using EpilogueOutputOp = typename Conv2d::EpilogueOutputOp;

  static cutlass::conv::Operator const kConvolutionalOperator = Conv2d::kConvolutionalOperator;

 public:
  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;

  cutlass::HostTensor<ElementA, LayoutA> tensor_A;
  cutlass::HostTensor<ElementB, LayoutB> tensor_B;
  cutlass::HostTensor<ElementB, LayoutB> tensor_reordered_B;
  cutlass::HostTensor<ElementC, LayoutC> tensor_C;
  cutlass::HostTensor<ElementC, LayoutC> tensor_D_computed;
  cutlass::HostTensor<ElementC, LayoutC> tensor_D_reference;

  int tested_problem_count;

 public:
  TestbedDepthwiseDirectConv2d(cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
                               cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
                               cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
                               uint64_t seed_ = 2080)
      : init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_), tested_problem_count(0) {}

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  void initialize_tensor(cutlass::TensorView<Element, Layout> view,
                         cutlass::Distribution::Kind dist_kind,
                         uint64_t seed) {
    if (dist_kind == cutlass::Distribution::Uniform) {
      int scope;
      int bits = cutlass::sizeof_bits<Element>::value;

      if (bits <= 8) {
        scope = 2;
      } else if (bits == 16) {
        if (cutlass::sizeof_bits<ElementAccumulator>::value <= 16) {
          scope = 3;
        } else {
          scope = 5;
        }
      } else {
        scope = 8;
      }
      cutlass::reference::host::TensorFillRandomUniform(view, seed, scope, -scope, 0);
    } else if (dist_kind == cutlass::Distribution::Identity) {
      cutlass::reference::host::TensorFillIdentity(view);

    } else if (dist_kind == cutlass::Distribution::Gaussian) {
      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    } else if (dist_kind == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
    } else {
    }
  }

  void initialize(cutlass::conv::Conv2dProblemSize const &problem_size, uint64_t seed = 2019) {
    tensor_A.resize(implicit_gemm_tensor_a_extent(kConvolutionalOperator, problem_size));
    tensor_B.resize(implicit_gemm_tensor_b_extent(kConvolutionalOperator, problem_size));
    tensor_reordered_B.resize(implicit_gemm_tensor_b_extent(kConvolutionalOperator, problem_size));
    tensor_C.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size));
    tensor_D_computed.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size));
    tensor_D_reference.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size));

    initialize_tensor(tensor_A.host_view(), init_A, seed);
    initialize_tensor(tensor_B.host_view(), init_B, seed * 17);
    initialize_tensor(tensor_reordered_B.host_view(), init_B, seed * 17);
    initialize_tensor(tensor_C.host_view(), init_C, seed * 39);

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_reordered_B.sync_device();
    tensor_C.sync_device();
    tensor_D_computed.sync_device();
    tensor_D_reference.sync_device();
  }

  bool sufficient(int smem_size) const {
    //
    // Determine SMEM requirements and waive if not satisfied
    //

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

    if (properties.sharedMemPerBlockOptin < static_cast<size_t>(smem_size)) {
      return false;
    }

    return true;
  }

  /// Executes one test
  bool run(cutlass::conv::Conv2dProblemSize const &problem_size,
           cutlass::conv::SplitKMode const &split_k_mode = cutlass::conv::SplitKMode::kSerial,
           ElementCompute alpha = ElementCompute(1.5),
           ElementCompute beta = ElementCompute(1)) {
    // increment tested problem count run by the testbed
    tested_problem_count++;

#if 0 // display conv2d problem size for debugging
    std::cout << problem_size << std::endl
              << "alpha, beta: (" << alpha << ", " << beta << ")" << std::endl
              << "split_k_mode: "
              << ((split_k_mode == cutlass::conv::SplitKMode::kSerial) ? "(serial)" : "(parallel)")
              << std::endl
              << std::endl;
#endif

    initialize(problem_size);

    // configure the operator
    Conv2d conv2d_op;

    typename Conv2d::Arguments conv2d_args(problem_size,
                                           tensor_A.device_ref(),
                                           tensor_B.device_ref(),
                                           tensor_C.device_ref(),
                                           tensor_D_computed.device_ref(),
                                           {alpha, beta},
                                           tensor_reordered_B.device_ref(),
                                           split_k_mode);

    // find workspace requirement for parallel split-k reduction
    size_t workspace_size = Conv2d::get_workspace_size(conv2d_args);

    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = conv2d_op.can_implement(problem_size);

    if (status != cutlass::Status::kSuccess) {
      cudaError_t error = cudaGetLastError();
      std::cerr << "This test is not supported: " << cudaGetErrorString(error) << "\n";
      return true;
    }

    status = conv2d_op.initialize(conv2d_args, workspace.get());

    if (status != cutlass::Status::kSuccess) {
      cudaError_t error = cudaGetLastError();
      std::cerr << "This test is not supported: " << cudaGetErrorString(error) << "\n";
      return true;
    }

    if (!sufficient(conv2d_op.get_smem_size())) {
      if (CUTLASS_TEST_UNIT_ENABLE_WARNINGS) {
        std::cerr << "Test waived due to insufficient CUDA device." << std::endl;
      }
      return true;
    }

    // run conv2d operator
    status = conv2d_op();

    EXPECT_TRUE(status == cutlass::Status::kSuccess);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to run." << std::endl;
      return false;
    }

    bool passed = false;

    cudaError_t result = cudaDeviceSynchronize();
    EXPECT_EQ(result, cudaSuccess) << " device reference error: " << cudaGetErrorString(result);

    tensor_D_computed.sync_host();

    //
    // Reference check - support caching results
    //

    CachedTestKey cached_test_key =
        CreateCachedConv2dTestKey<ElementA,
                                  LayoutA,
                                  ElementB,
                                  LayoutB,
                                  ElementC,
                                  LayoutC,
                                  ElementAccumulator,
                                  ElementCompute>(kConvolutionalOperator,
                                                  problem_size,
                                                  alpha,
                                                  beta,
                                                  tensor_A.host_view(),
                                                  tensor_B.host_view(),
                                                  tensor_C.host_view());

    //
    // Look for the cached key
    //

    bool cached_result_loaded = false;
    CachedTestResult cached_test_result;

    std::string conv2d_result_cache_name =
        std::string("cached_results_") + CUTLASS_TARGET_NAME + ".txt";

    if (CUTLASS_TEST_ENABLE_CACHED_RESULTS) {

      CachedTestResultListing cached_results(conv2d_result_cache_name);

      auto cached = cached_results.find(cached_test_key);

      cached_result_loaded = cached.first;
      if (cached_result_loaded) {
        cached_test_result = cached.second;
      }
    }

    if (!cached_result_loaded) {
#if CUTLASS_CONV_TEST_UNIT_REFERENCE_DEVICE_ENABLED

      cutlass::reference::device::Conv2d<ElementA,
                                         LayoutA,
                                         ElementB,
                                         LayoutB,
                                         ElementC,
                                         LayoutC,
                                         ElementCompute,
                                         ElementAccumulator>(kConvolutionalOperator,
                                                             problem_size,
                                                             tensor_A.device_ref(),
                                                             tensor_B.device_ref(),
                                                             tensor_C.device_ref(),
                                                             tensor_D_reference.device_ref(),
                                                             alpha,
                                                             beta);

      // sync host (copy device data to host) for dumping error output in case of mismatches
      tensor_D_reference.sync_host();

#else

      cutlass::reference::host::Conv2d<ElementA,
                                       LayoutA,
                                       ElementB,
                                       LayoutB,
                                       ElementC,
                                       LayoutC,
                                       ElementCompute,
                                       ElementAccumulator>(kConvolutionalOperator,
                                                           problem_size,
                                                           tensor_A.host_ref(),
                                                           tensor_B.host_ref(),
                                                           tensor_C.host_ref(),
                                                           tensor_D_reference.host_ref(),
                                                           alpha,
                                                           beta);

#endif

      if (CUTLASS_TEST_ENABLE_CACHED_RESULTS) {

        cached_test_result.D = TensorHash(tensor_D_reference.host_view());

        CachedTestResultListing cached_results(conv2d_result_cache_name);

        cached_results.append(cached_test_key, cached_test_result);
        cached_results.write(conv2d_result_cache_name);
      }
    } // if (!cached_result_loaded)

    uint32_t tensor_D_hash = TensorHash(tensor_D_computed.host_view());

    if (CUTLASS_TEST_ENABLE_CACHED_RESULTS) {
      passed = (tensor_D_hash == cached_test_result.D);

      EXPECT_EQ(tensor_D_hash, cached_test_result.D) 
        << "Hash-based comparison failed for key:" << "\n" << cached_test_key << "\n";
    }
    else {

      passed = cutlass::reference::host::TensorEquals(
        tensor_D_computed.host_view(), 
                                                      tensor_D_reference.host_view());
    }

    EXPECT_TRUE(passed);

    std::stringstream ss_problem_size_text;
    ss_problem_size_text         << "nhwc_"
        << problem_size.N << "x"
        << problem_size.H << "x"
        << problem_size.W << "x"
        << problem_size.C
        << "_krsc_"
        << problem_size.K << "x"
        << problem_size.R << "x"
        << problem_size.S << "x"
        << problem_size.C
        << "_padding_"
        << problem_size.pad_h << "x"
        << problem_size.pad_w
        << "_stride_"
        << problem_size.stride_h << "x"
        << problem_size.stride_w
        << "_dilation_"
        << problem_size.dilation_h << "x"
                         << problem_size.dilation_w << "_"
        << (problem_size.mode == cutlass::conv::Mode::kCrossCorrelation ? "xcorr_" : "conv_");

    if (!passed) {
      std::stringstream fname;

      fname << "error_Conv2d_DirectConv_device_"
        << (split_k_mode == cutlass::conv::SplitKMode::kSerial ? "serial_reduction_" : "parallel_reduction_")
        << (Conv2d::kConvolutionalOperator == cutlass::conv::Operator::kFprop ? "fprop_" :
            (Conv2d::kConvolutionalOperator == cutlass::conv::Operator::kDgrad ? "dgrad_" : "wgrad_"))
        << ss_problem_size_text.str()
        << Conv2d::ThreadblockShape::kM << "x"  
        << Conv2d::ThreadblockShape::kN << "x"  
        << Conv2d::ThreadblockShape::kK << "_"
        << Conv2d::WarpShape::kM << "x"  
        << Conv2d::WarpShape::kN << "x"  
        << Conv2d::WarpShape::kK << ".txt";

      std::cout << fname.str() << std::endl;

      std::ofstream results(fname.str());

      results << problem_size << std::endl;

      results
        << "\nA:\n" << tensor_A.host_view() << "\n"
        << "\nB:\n" << tensor_B.host_view() << "\n"
        << "\nC:\n" << tensor_C.host_view() << "\n";

      results << "\nD reference (hash: " << cached_test_result.D << ")\n";

      if (!cached_result_loaded) {
        results
          << tensor_D_reference.host_view() << "\n";  
      }

      results
        << "\nD computed (hash: " << tensor_D_hash << ")\n" 
              << tensor_D_computed.host_view() << "\n";

    }

    return passed;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DirectConv>
bool TestSpecificDepthwiseDirectConv2d(const Conv2dProblemVector &problem_sizes) {
  bool passed = true;

  //
  // Testbed object
  //
  TestbedDepthwiseDirectConv2d<DirectConv> testbed;

  // Sweep conv2d problem sizes (split-k-mode=kSerial, split-k-slice=1, alpha=1.0, beta=0.0)
  for (auto conv_problem : problem_sizes) {
    //
    // Test
    //

    // test mode = xcross
    passed = testbed.run(
      conv_problem,
      cutlass::conv::SplitKMode::kSerial);

    if (!passed) {
      return false;
    }

    // test mode = convolution
    passed = testbed.run(
      conv_problem.reset_mode(cutlass::conv::Mode::kConvolution),
      cutlass::conv::SplitKMode::kSerial);

    if (!passed) {
      return false;
    }
  }

  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace conv
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////
