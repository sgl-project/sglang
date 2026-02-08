/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Testbed for running device-level Conv2Ds with absolute maximum calculation and scaling
*/

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>

#include "conv2d_problems.h"
#include "../../common/cutlass_unit_test.h"
#include "../../gemm/device/testbed_utils.h"

#include "cutlass/matrix_coord.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_reduce.h"

namespace test {
namespace conv {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Conv,
  template<typename T> class ActivationFunctor
>
struct TestbedConv2dWithAbsMax {

  using ElementAccumulator = typename Conv::ElementAccumulator;
  using ElementCompute = typename Conv::UnderlyingKernel::Epilogue::OutputOp::ElementCompute;
  using ElementScalingFactor = typename Conv::EpilogueOutputOp::ElementScalingFactor;
  using ElementAbsmax = typename Conv::EpilogueOutputOp::ElementAbsmax;
  static cutlass::conv::Operator const kConvolutionalOperator = Conv::kConvolutionalOperator;

  static bool const kScaleAux = Conv::EpilogueOutputOp::kIsScalingAndAmaxAuxOutputNeeded;
  static bool const kScaleOutput = Conv::EpilogueOutputOp::kIsScalingAndAmaxOutputNeeded;
  bool doScaleA;
  bool doScaleB;
  bool doScaleC;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;

  cutlass::HostTensor<typename Conv::ElementA, typename Conv::LayoutA> tensor_A;
  cutlass::HostTensor<typename Conv::ElementB, typename Conv::LayoutB> tensor_B;
  cutlass::HostTensor<typename Conv::ElementC, typename Conv::LayoutC> tensor_C;
  cutlass::HostTensor<typename Conv::EpilogueOutputOp::ElementAuxOutput, typename Conv::LayoutC> tensor_Aux;
  cutlass::HostTensor<typename Conv::EpilogueOutputOp::ElementOutput, typename Conv::LayoutC> tensor_D;
  cutlass::HostTensor<typename Conv::ElementC, typename Conv::LayoutC> tensor_Vector;
  cutlass::HostTensor<ElementAccumulator, typename Conv::LayoutC> tmp_D;
  cutlass::HostTensor<typename Conv::EpilogueOutputOp::ElementOutput, typename Conv::LayoutC> reference_D;
  cutlass::HostTensor<typename Conv::EpilogueOutputOp::ElementAuxOutput, typename Conv::LayoutC> reference_Aux;
  cutlass::HostTensor<ElementScalingFactor, typename Conv::LayoutC> scale_A;
  cutlass::HostTensor<ElementScalingFactor, typename Conv::LayoutC> scale_B;
  cutlass::HostTensor<ElementScalingFactor, typename Conv::LayoutC> scale_C;
  cutlass::HostTensor<ElementScalingFactor, typename Conv::LayoutC> scale_D;
  cutlass::HostTensor<ElementScalingFactor, typename Conv::LayoutC> scale_Aux;
  cutlass::HostTensor<ElementAbsmax, typename Conv::LayoutC> abs_max_Aux;
  cutlass::HostTensor<ElementAbsmax, typename Conv::LayoutC> abs_max_D;
  cutlass::HostTensor<ElementAbsmax, typename Conv::LayoutC> reference_abs_max_Aux;
  cutlass::HostTensor<ElementAbsmax, typename Conv::LayoutC> reference_abs_max_D;

  //
  // Methods
  //

  TestbedConv2dWithAbsMax(
    bool scaleA = true,
    bool scaleB = true,
    bool scaleC = true,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    doScaleA(scaleA), doScaleB(scaleB), doScaleC(scaleC),
    init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

  /// Helper to initialize scaling factors
  template <typename Element, typename Layout>
  bool initialize_scale_factor(cutlass::TensorView<Element, Layout> view, uint64_t seed, int bits=0) {
    cutlass::reference::host::TensorFillRandomUniform(view, seed, double(1.), double(0.), bits);
    return true;
  }

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool initialize_tensor(
    cutlass::TensorView<Element, Layout> view,
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      double scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<typename Conv::ElementC>::value;

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
  void initialize(cutlass::conv::Conv2dProblemSize const &problem_size) {
    //
    // Allocate the GEMM workspace
    //

    tensor_A.resize(implicit_gemm_tensor_a_extent(kConvolutionalOperator, problem_size));
    tensor_B.resize(implicit_gemm_tensor_b_extent(kConvolutionalOperator, problem_size));
    tensor_C.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size));
    tensor_D.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size));
    tensor_Vector.resize({1, 1, 1, implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size).c()});
    reference_D.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size), false);
    tmp_D.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size), false);

    EXPECT_TRUE(initialize_tensor(tensor_A.host_view(), init_A, seed + 2019));
    EXPECT_TRUE(initialize_tensor(tensor_B.host_view(), init_B, seed + 2018));
    EXPECT_TRUE(initialize_tensor(tensor_C.host_view(), init_C, seed + 2017));
    EXPECT_TRUE(initialize_tensor(tensor_Vector.host_view(), init_C, seed + 2020));

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    cutlass::Coord<4> origin(0);
    tensor_A.host_view().at(origin) = typename Conv::ElementA(1);
    tensor_B.host_view().at(origin) = typename Conv::ElementB(1);
    tensor_C.host_view().at(origin) = typename Conv::ElementC(1);
    tensor_Vector.host_view().at(origin) = typename Conv::ElementC(1);

    cutlass::reference::host::TensorFill(tensor_D.host_view());
    cutlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
    tensor_Vector.sync_device();

    int scale_bits = 2;
    if (doScaleA) {
      scale_A.resize({1, 1, 1, 1});
      EXPECT_TRUE(initialize_scale_factor(scale_A.host_view(), seed + 2021, scale_bits));
      scale_A.sync_device();
    }

    if (doScaleB) {
      scale_B.resize({1, 1, 1, 1});
      EXPECT_TRUE(initialize_scale_factor(scale_B.host_view(), seed + 2022, scale_bits));
      scale_B.sync_device();
    }

    if (doScaleC) {
      scale_C.resize({1, 1, 1, 1});
      EXPECT_TRUE(initialize_scale_factor(scale_C.host_view(), seed + 2023, scale_bits));
      scale_C.sync_device();
    }

    if (kScaleOutput) {
      scale_D.resize({1, 1, 1, 1});
      EXPECT_TRUE(initialize_scale_factor(scale_D.host_view(), seed + 2024, scale_bits));
      scale_D.sync_device();

      abs_max_D.resize({1, 1, 1, 1});
      cutlass::reference::host::TensorFill(abs_max_D.host_view());
      abs_max_D.sync_device();

      reference_abs_max_D.resize({1, 1, 1, 1});
    }

    if (kScaleAux) {
      tensor_Aux.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size));
      cutlass::reference::host::TensorFill(tensor_Aux.host_view());
      tensor_Aux.sync_device();

      scale_Aux.resize({1, 1, 1, 1});
      EXPECT_TRUE(initialize_scale_factor(scale_Aux.host_view(), seed + 2025, scale_bits));
      scale_Aux.sync_device();

      abs_max_Aux.resize({1, 1, 1, 1});
      cutlass::reference::host::TensorFill(abs_max_Aux.host_view());
      abs_max_Aux.sync_device();

      reference_Aux.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size), false);
      reference_abs_max_Aux.resize({1, 1, 1, 1});
    }
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(
    cutlass::conv::Conv2dProblemSize const &problem_size,
    ElementCompute alpha,
    ElementCompute beta) {

    tensor_D.sync_host();

    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_A.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_B.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_C.host_view()), 0);

    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_D.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(reference_D.host_view()), 0);
    bool passed = cutlass::reference::host::TensorEquals(reference_D.host_view(), tensor_D.host_view());

    if (kScaleAux) {
      tensor_Aux.sync_host();
      abs_max_Aux.sync_host();
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_Aux.host_view()), 0);
      EXPECT_GT(cutlass::reference::host::TensorNorm(abs_max_Aux.host_view()), 0);
      EXPECT_GT(cutlass::reference::host::TensorNorm(reference_Aux.host_view()), 0);
      passed &= cutlass::reference::host::TensorEquals(reference_Aux.host_view(), tensor_Aux.host_view());
      passed &= cutlass::reference::host::TensorEquals(abs_max_Aux.host_view(), reference_abs_max_Aux.host_view());
    }

    if (kScaleOutput) {
      abs_max_D.sync_host();
      EXPECT_GT(cutlass::reference::host::TensorNorm(abs_max_D.host_view()), 0);
      passed &= cutlass::reference::host::TensorEquals(abs_max_D.host_view(), reference_abs_max_D.host_view());
    }

    EXPECT_TRUE(passed) << " mismatched reference";

    if (!passed) {

      std::ofstream file0("conv_testbed_with_amax_errors_reference.txt");
      std::ofstream file1("conv_testbed_with_amax_errors_computed.txt");

      std::ofstream file("conv_testbed_with_amax_errors.txt");

      file
        << "problem: " << problem_size
        << ", alpha: " << alpha << ", beta: " << beta << "\n\n";

      file
        << "A =\n" << tensor_A.host_view()
        << "\nB =\n" << tensor_B.host_view()
        << "\nC =\n" << tensor_C.host_view()
        << "\nVector =\n" << tensor_Vector.host_view()
        << "\nScaleA = " << scale_A.host_view()
        << "\nScaleB = " << scale_B.host_view()
        << "\nScaleC = " << scale_C.host_view()
        << "\nScaleD = " << scale_D.host_view()
        << "\nScaleAux = " << scale_Aux.host_view()
        << std::endl;

      file0 << "\n\nReference D =\n" << reference_D.host_view() << std::endl;
      file1 << "\n\nComputed D =\n" << tensor_D.host_view() << std::endl;
      if (kScaleAux) {
        file0 << "\n\nReference Aux =\n" << reference_Aux.host_view() << std::endl;
        file1 << "\n\nComputed Aux =\n" << tensor_Aux.host_view() << std::endl;
        file0 << "\n\nReference Absmax Aux = " << reference_abs_max_Aux.host_view() << std::endl;
        file1 << "\n\nComputed Absmax Aux = " << abs_max_Aux.host_view() << std::endl;
      }
      if (kScaleOutput) {
        file0 << "\n\nReference Absmax D = " << reference_abs_max_D.host_view() << std::endl;
        file1 << "\n\nComputed Absmax D = " << abs_max_D.host_view() << std::endl;
      }
    }

    return passed;
  }

  /// Verifies the result is a GEMM
  bool verify(
    cutlass::conv::Conv2dProblemSize const &problem_size,
    ElementCompute alpha,
    ElementCompute beta) {

    cutlass::Coord<4> origin(0);
    ElementCompute scaled_alpha = alpha;
    if (doScaleA) {
      scaled_alpha *= scale_A.host_view().at(origin);
    }
    if (doScaleB) {
      scaled_alpha *= scale_B.host_view().at(origin);
    }

    ElementCompute scaled_beta = beta;
    if (doScaleC) {
      scaled_beta *= scale_C.host_view().at(origin);
    }

    //
    // Verify
    //

    cutlass::reference::host::Conv2d<
        typename Conv::ElementA, typename Conv::LayoutA,
        typename Conv::ElementB, typename Conv::LayoutB,
        typename Conv::ElementC, typename Conv::LayoutC,
        ElementCompute, ElementAccumulator, ElementAccumulator
    >(
      kConvolutionalOperator,
      problem_size,
      tensor_A.host_ref(),
      tensor_B.host_ref(),
      tensor_C.host_ref(),
      tmp_D.host_ref(),
      scaled_alpha,
      scaled_beta
    );

    ElementCompute tmp_abs_max_Aux(0.);
    ElementCompute tmp_abs_max_D(0.);

    cutlass::NumericConverter<ElementCompute, typename Conv::ElementC> cvt_c_to_compute;
    cutlass::NumericConverter<ElementCompute, ElementAccumulator> cvt_accum_to_compute;
    cutlass::NumericConverter<ElementAbsmax, ElementCompute> cvt_compute_to_absmax;
    cutlass::NumericConverter<typename Conv::EpilogueOutputOp::ElementOutput, ElementCompute> cvt_compute_to_d;
    cutlass::NumericConverter<typename Conv::EpilogueOutputOp::ElementAuxOutput, ElementCompute> cvt_compute_to_aux;

    cutlass::absolute_value_op<ElementCompute> abs;
    cutlass::maximum_with_nan_propogation<ElementCompute> max;
    ActivationFunctor<ElementCompute> act;

    ElementScalingFactor d_scale = kScaleOutput ? scale_D.host_view().at(origin) : ElementScalingFactor(1.);

    for (int n = 0; n < problem_size.N; ++n) {
      for (int p = 0; p < problem_size.P; ++p) {
        for (int q = 0; q < problem_size.Q; ++q) {
          for (int k = 0; k < problem_size.K; ++k) {
            ElementCompute intermediate = cvt_accum_to_compute(tmp_D.host_view().at({n, p, q, k}));
            ElementCompute bias = cvt_c_to_compute(tensor_Vector.host_view().at({0, 0, 0, k}));
            ElementCompute aux = intermediate + bias;
            ElementCompute d = act(aux);
            tmp_abs_max_Aux = max(abs(aux), tmp_abs_max_Aux);
            tmp_abs_max_D = max(abs(d), tmp_abs_max_D);
            reference_D.host_view().at({n, p, q, k}) = cvt_compute_to_d(d * d_scale);

            if (kScaleAux) {
              reference_Aux.host_view().at({n, p, q, k}) = cvt_compute_to_aux(aux * scale_Aux.host_view().at(origin));
            }
          }
        }
      }
    }
    if (kScaleAux) {
      reference_abs_max_Aux.host_view().at(origin) = cvt_compute_to_absmax(tmp_abs_max_Aux);
    }

    if (kScaleOutput) {
      reference_abs_max_D.host_view().at(origin) = cvt_compute_to_absmax(tmp_abs_max_D);
    }

    return compare_reference(problem_size, alpha, beta);
  }

  /// Returns true if the CUDA device is sufficient to execute the kernel.
  bool sufficient() const {
    //
    // Determine SMEM requirements and waive if not satisfied
    //

    size_t smem_size = sizeof(typename Conv::UnderlyingKernel::SharedStorage);

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
    cutlass::conv::Conv2dProblemSize const &problem_size,
    ElementCompute alpha = ElementCompute(1),
    ElementCompute beta = ElementCompute(0))
  {

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

    typename Conv::EpilogueOutputOp::Params::ActivationParams activation_params{alpha, beta};
    typename Conv::EpilogueOutputOp::Params epilogue_params{
      activation_params,
      scale_A.device_data(),
      scale_B.device_data(),
      scale_C.device_data(),
      scale_D.device_data(),
      scale_Aux.device_data(),
      abs_max_Aux.device_data(),
      abs_max_D.device_data()
    };

    typename Conv::Arguments arguments{
      problem_size,
      tensor_A.device_ref(),
      tensor_B.device_ref(),
      tensor_C.device_ref(),
      tensor_D.device_ref(),
      tensor_Aux.device_ref(),
      epilogue_params,
      cutlass::conv::SplitKMode::kSerial,
      tensor_Vector.device_data(),
      0
    };

    Conv conv2d_op;

    cutlass::Status status = conv2d_op.can_implement(arguments);
    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    size_t workspace_size = Conv::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = conv2d_op.initialize(arguments, workspace.get());
    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    //
    // Run the GEMM
    //

    status = conv2d_op();

    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    cudaError_t cuda_error = cudaDeviceSynchronize();
    EXPECT_TRUE(cuda_error == cudaSuccess) << cudaGetErrorString(cuda_error);

    //
    // Verify
    //

    bool passed = this->verify(problem_size, alpha, beta);

    if (!passed) {
      std::cout << "Failed" << std::endl;
    }

    return passed;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ImplicitGemm,
  template<typename T> class ActivationFunctor = cutlass::epilogue::thread::Identity
>
bool TestAllConv2dWithAbsmax(bool scaleA=true, bool scaleB=true, bool scaleC=true) {
  const Conv2dProblemVector &conv_test_sizes = Conv2dProblemVector();
  const Conv2dProblemVector &conv_blacklist_sizes = Conv2dProblemVector();

  //
  // Testbed object
  //

  TestbedConv2dWithAbsMax<ImplicitGemm, ActivationFunctor> testbed(scaleA, scaleB, scaleC);

  //
  // Get conv problem sizes to run conv operator 
  //
  TestbedConv2dProblemSizes conv_problems(128/cutlass::sizeof_bits<typename ImplicitGemm::ElementA>::value);

  // Vector of conv2d problem sizes to avoid duplicate runs
  Conv2dProblemVector conv_tested_sizes;

  Conv2dProblemVector const *problem_vectors[] = {
    &conv_test_sizes,                               // run user specified sizes
    &conv_problems.conv2d_default_sizes,            // run default and cudnn bug sizes
    &conv_problems.conv2d_resnet50_sizes,           // run resnet50 sizes
#if CUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED 
    &conv_problems.conv2d_rigorous_sizes,           // run large and rigorous sizes if enabled
#endif
  };

  bool passed = true;

  // Sweep conv2d problem sizes (split-k-mode=kSerial, split-k-slice=1, alpha=1.0, beta=0.0)
  for (Conv2dProblemVector const * problem_vector : problem_vectors) {

    // Prune all problems with channels that aren't divisible by the number of elements accessed per
    // load for operands A and B. This is meant to align with the requirements of iterators used for
    // fprop kernels.
    ChannelDivisibilitySpecification channel_spec(128 / cutlass::sizeof_bits<typename ImplicitGemm::ElementA>::value);
    auto pruned_problem_vector = prune(*problem_vector, channel_spec);

    //  Run conv testbed on default convolution sizes
    for(auto conv_problem : pruned_problem_vector) {

      // Skip blacklist and avoid duplicate problem sizes
      if (std::find(conv_blacklist_sizes.begin(), conv_blacklist_sizes.end(), conv_problem) != conv_blacklist_sizes.end() ||
          std::find(conv_tested_sizes.begin(), conv_tested_sizes.end(), conv_problem) != conv_tested_sizes.end()) {
        continue;
      }

      //
      // Test
      //
      // push back tested problem size to avoid re-running duplicates
      conv_tested_sizes.push_back(conv_problem);

      // test mode = xcross
      passed &= testbed.run(conv_problem);

      if (!passed) {
        return false;
      }

      // test mode = convolution
      passed &= testbed.run(conv_problem.reset_mode(cutlass::conv::Mode::kConvolution));

      if (!passed) {
        return false;
      }
    }
  }

  return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace conv
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////
