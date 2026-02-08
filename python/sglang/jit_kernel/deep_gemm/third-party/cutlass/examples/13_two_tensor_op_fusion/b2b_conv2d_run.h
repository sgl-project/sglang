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

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"

#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/reduction/device/reduce_split_k.h"
#include "cutlass/reduction/thread/reduction_operators.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"

#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/reference/device/convolution.h"
#include "cutlass/util/reference/device/tensor_relu.h"

#include "cutlass/core_io.h"
#include "cutlass/util/tensor_view_io.h"

#include "reference/device/tensor_scale_bias.h"
#include "helper.h"

#define CHECK_GT(val1, val2) \
    if((val1) <= (val2)) \
        std::cerr << __FILE__ << " " << __LINE__ << ": CHECK_GT failed\n";
#define CHECK_TRUE(val) \
    if(!(val)) \
        std::cerr << __FILE__ << " " << __LINE__ << ": CHECK_TRUE failed\n";


template <typename Conv2d0_, typename Conv2d1_>
class B2bNonFusedConv2dRun {
public:

  using Conv2d0 = Conv2d0_;
  using Conv2d1 = Conv2d1_;
  using ElementAccumulator = typename Conv2d0::ElementAccumulator;
  using ElementCompute = typename Conv2d0::ElementCompute;

  static cutlass::conv::Operator const kConvolutionalOperator = Conv2d0::kConvolutionalOperator;
  static_assert(kConvolutionalOperator == Conv2d1::kConvolutionalOperator, 
        "Fused convolution operators must be the same");

public:

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  cutlass::Distribution::Kind init_Bias;
  uint64_t seed;

  cutlass::HostTensor<typename Conv2d0::ElementA, typename Conv2d0::LayoutA> tensor_A0;
  cutlass::HostTensor<typename Conv2d0::ElementB, typename Conv2d0::LayoutB> tensor_B0;
  cutlass::HostTensor<typename Conv2d0::ElementC, typename Conv2d0::LayoutC> tensor_C0;
  cutlass::HostTensor<typename Conv2d0::ElementCompute, typename Conv2d0::LayoutC> tensor_Bias0;
  cutlass::HostTensor<typename Conv2d0::ElementC, typename Conv2d0::LayoutC> tensor_D0_computed;
  cutlass::HostTensor<typename Conv2d0::ElementC, typename Conv2d0::LayoutC> tensor_D0_reference;

  cutlass::HostTensor<typename Conv2d1::ElementB, typename Conv2d1::LayoutB> tensor_B1;
  cutlass::HostTensor<typename Conv2d1::ElementC, typename Conv2d1::LayoutC> tensor_C1;
  cutlass::HostTensor<typename Conv2d1::ElementCompute, typename Conv2d0::LayoutC> tensor_Bias1;
  cutlass::HostTensor<typename Conv2d1::ElementC, typename Conv2d1::LayoutC> tensor_D1_computed;
  cutlass::HostTensor<typename Conv2d1::ElementC, typename Conv2d1::LayoutC> tensor_D1_reference;


public:

  B2bNonFusedConv2dRun(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_Bias_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    init_A(init_A_), init_B(init_B_), init_C(init_C_), init_Bias(init_Bias_), seed(seed_) {

  }

    /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  void initialize_tensor(
    cutlass::TensorView<Element, Layout> view, 
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      int scope;
      int bits = cutlass::sizeof_bits<Element>::value;

      if (bits <= 16) {
        scope = 2;
      }
      else {
        scope = 8;
      }
      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, scope, -scope, 0);
    } 
    else if (dist_kind == cutlass::Distribution::Identity) {

      cutlass::reference::host::TensorFillIdentity(view);
    } 
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {

      cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
    } 
    else if (dist_kind == cutlass::Distribution::AllZeros) {
      cutlass::reference::host::TensorFill(view, Element(0));
    }
    else if (dist_kind == cutlass::Distribution::AllOnes) {
      cutlass::reference::host::TensorFill(view, Element(1));
    }
    else {
      std::cerr << "Not implemented\n";
    }
  }

  void initialize(
    cutlass::conv::Conv2dProblemSize const &problem_size_0,
    cutlass::conv::Conv2dProblemSize const &problem_size_1,
    uint64_t seed = 2019) {
        
    tensor_A0.resize(implicit_gemm_tensor_a_extent(kConvolutionalOperator, problem_size_0));
    tensor_B0.resize(implicit_gemm_tensor_b_extent(kConvolutionalOperator, problem_size_0));
    tensor_C0.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size_0));
    tensor_Bias0.resize({1, 1, 1, problem_size_0.K});
    tensor_D0_computed.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size_0));
    tensor_D0_reference.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size_0));
    tensor_B1.resize(implicit_gemm_tensor_b_extent(kConvolutionalOperator, problem_size_1));
    tensor_C1.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size_1));
    tensor_Bias1.resize({1, 1, 1, problem_size_1.K});
    tensor_D1_computed.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size_1));
    tensor_D1_reference.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size_1));

    initialize_tensor(tensor_A0.host_view(), init_A, seed); 
    initialize_tensor(tensor_B0.host_view(), init_B, seed * 17); 
    initialize_tensor(tensor_C0.host_view(), init_C, seed * 39);
    initialize_tensor(tensor_Bias0.host_view(), init_Bias, seed * 83);
    initialize_tensor(tensor_B1.host_view(), init_B, seed * 18); 
    initialize_tensor(tensor_C1.host_view(), init_C, seed * 40);
    initialize_tensor(tensor_Bias1.host_view(), init_Bias, seed * 84);

    tensor_A0.sync_device();
    tensor_B0.sync_device();
    tensor_C0.sync_device();
    tensor_Bias0.sync_device();
    tensor_D0_computed.sync_device();
    tensor_D0_reference.sync_device();
    tensor_B1.sync_device();
    tensor_C1.sync_device();
    tensor_Bias1.sync_device();
    tensor_D1_computed.sync_device();
    tensor_D1_reference.sync_device();
  }

  /// Executes one test
  bool run(
    cutlass::conv::Conv2dProblemSize const &problem_size_0,
    cutlass::conv::Conv2dProblemSize const &problem_size_1,
    cutlass::conv::SplitKMode const &split_k_mode = cutlass::conv::SplitKMode::kSerial,
    ElementCompute alpha0 = ElementCompute(1),
    ElementCompute beta0 = ElementCompute(0),
    ElementCompute alpha1 = ElementCompute(1),
    ElementCompute beta1 = ElementCompute(0),
    bool relu = true,
    int warm_ups = 1,
    int runs = 100) {

    initialize(problem_size_0, problem_size_1);

    // configure the operator
    Conv2d0 conv2d_op_0;
    Conv2d1 conv2d_op_1;

    typename Conv2d0::Arguments conv2d_args_0(
      problem_size_0,
      tensor_A0.device_ref(),
      tensor_B0.device_ref(),
      {tensor_Bias0.device_data(), typename Conv2d0::LayoutC::Stride(0)},
      tensor_D0_computed.device_ref(),
      {alpha0, beta0},
      split_k_mode
    );
    typename Conv2d1::Arguments conv2d_args_1(
      problem_size_1,
      tensor_D0_computed.device_ref(),
      tensor_B1.device_ref(),
      {tensor_Bias1.device_data(), typename Conv2d1::LayoutC::Stride(0)},
      tensor_D1_computed.device_ref(),
      {alpha1, beta1},
      split_k_mode
    );


    cutlass::Status status = conv2d_op_0.initialize(conv2d_args_0);

    CUTLASS_CHECK(status);

    status = conv2d_op_1.initialize(conv2d_args_1);

    CUTLASS_CHECK(status);

    for(int i = 0; i < warm_ups; i++) {
        status = conv2d_op_0();
        CUTLASS_CHECK(status);
        status = conv2d_op_1();
        CUTLASS_CHECK(status);
    }

    //
    // Run Conv2d
    //
    cudaEvent_t start, stop1, stop2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop1);
    cudaEventCreate(&stop2);

    cudaEventRecord(start);


    for(int i = 0; i < runs; i++) {
        // run conv2d operator
        status = conv2d_op_0();
        CUTLASS_CHECK(status);
    }
    cudaEventRecord(stop1);    
    
    for(int i = 0; i < runs; i++) {
        // run conv2d operator
        status = conv2d_op_1();
        CUTLASS_CHECK(status);
    }
    cudaEventRecord(stop2);
    cudaDeviceSynchronize();
    float conv2d0Time, conv2d1Time, totalTime;
    cudaEventElapsedTime(&conv2d0Time, start, stop1);
    cudaEventElapsedTime(&conv2d1Time, stop1, stop2);
    cudaEventElapsedTime(&totalTime, start, stop2);
    std::cout << "conv2d 0 time " << conv2d0Time / (float)runs << " ms\n";
    std::cout << "conv2d 1 time " << conv2d1Time / (float)runs << " ms\n";
    std::cout << "Non-fusion time " << totalTime / (float)runs << " ms\n";

    tensor_D0_computed.sync_host();
    tensor_D1_computed.sync_host();
    
    bool passed = false;

    cutlass::reference::device::Conv2d<
      typename Conv2d0::ElementA,
      typename Conv2d0::LayoutA,
      typename Conv2d0::ElementB,
      typename Conv2d0::LayoutB,
      typename Conv2d0::ElementC,
      typename Conv2d0::LayoutC,
      ElementCompute,
      ElementAccumulator
    >(
      kConvolutionalOperator,
      problem_size_0,
      tensor_A0.device_ref(),
      tensor_B0.device_ref(),
      {tensor_Bias0.device_data(), typename Conv2d0::LayoutC::Stride(0)},
      tensor_D0_reference.device_ref(),
      alpha0, 
      beta0);
    
    if(relu) {
       cutlass::reference::device::TensorReLu(tensor_D0_reference.device_view()); 
    }

    cutlass::reference::device::Conv2d<
      typename Conv2d1::ElementA,
      typename Conv2d1::LayoutA,
      typename Conv2d1::ElementB,
      typename Conv2d1::LayoutB,
      typename Conv2d1::ElementC,
      typename Conv2d1::LayoutC,
      ElementCompute,
      ElementAccumulator
    >(
      kConvolutionalOperator,
      problem_size_1,
      tensor_D0_reference.device_ref(),
      tensor_B1.device_ref(),
      {tensor_Bias1.device_data(), typename Conv2d1::LayoutC::Stride(0)},
      tensor_D1_reference.device_ref(),
      alpha1, 
      beta1);

    if(relu) {
       cutlass::reference::device::TensorReLu(tensor_D1_reference.device_view()); 
    }

    cudaError_t result = cudaDeviceSynchronize();
    CHECK_TRUE(result == cudaSuccess);

    // sync host (copy device data to host) for dumping error output in case of mismatches
    tensor_D0_reference.sync_host();
    tensor_D1_reference.sync_host();
    
    CHECK_GT(cutlass::reference::host::TensorNorm(tensor_D0_computed.host_view()), 0);
    CHECK_GT(cutlass::reference::host::TensorNorm(tensor_D0_reference.host_view()), 0);
    CHECK_GT(cutlass::reference::host::TensorNorm(tensor_D1_computed.host_view()), 0);
    CHECK_GT(cutlass::reference::host::TensorNorm(tensor_D1_reference.host_view()), 0);

    passed = cutlass::reference::host::TensorEquals(
      tensor_D1_computed.host_view(), 
      tensor_D1_reference.host_view());

    CHECK_TRUE(passed);

    if (!passed) {
      std::stringstream fname;

      fname << "error_B2bImplicitGemm_device_nonfused.txt";
      std::cerr << "Dumping results in " << fname.str() << "\n";

      std::ofstream results(fname.str());

      results << problem_size_0 << std::endl;
      results << problem_size_1 << std::endl;

      results
        << "\nA0:\n" << tensor_A0.host_view() << "\n"
        << "\nB0:\n" << tensor_B0.host_view() << "\n"
        << "\nC0:\n" << tensor_C0.host_view() << "\n"
        << "\nBias0:\n" << tensor_Bias0.host_view() << "\n"
        << "\nD0 reference:\n" << tensor_D0_reference.host_view() << "\n"
        << "\nD0 computed:\n" << tensor_D0_computed.host_view() << "\n"
        << "\nB1:\n" << tensor_B1.host_view() << "\n"
        << "\nC1:\n" << tensor_C1.host_view() << "\n"
        << "\nBias1:\n" << tensor_Bias1.host_view() << "\n"
        << "\nD1 reference:\n" << tensor_D1_reference.host_view() << "\n"
        << "\nD1 computed:\n" << tensor_D1_computed.host_view();


    }

    return passed;
  }

};

template <typename B2bConv2d_>
class B2bFusedConv2dRun {
public:

  using B2bConv2d = B2bConv2d_;
  using ElementAccumulator = typename B2bConv2d::ElementAccumulator;
  using ElementCompute = typename B2bConv2d::ElementCompute;

  static cutlass::conv::Operator const kConvolutionalOperator = B2bConv2d::kConvolutionalOperator;

public:

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  cutlass::Distribution::Kind init_Scale;
  cutlass::Distribution::Kind init_Bias;
  uint64_t seed;

  cutlass::HostTensor<typename B2bConv2d::ElementA, typename B2bConv2d::LayoutA> tensor_A0;
  cutlass::HostTensor<typename B2bConv2d::ElementB, typename B2bConv2d::LayoutB> tensor_B0;
  cutlass::HostTensor<typename B2bConv2d::ElementC, typename B2bConv2d::LayoutC> tensor_C0;
  cutlass::HostTensor<typename B2bConv2d::ElementScaleBias, typename B2bConv2d::LayoutScaleBias> tensor_Scale0;
  cutlass::HostTensor<typename B2bConv2d::ElementScaleBias, typename B2bConv2d::LayoutScaleBias> tensor_Bias0;
  cutlass::HostTensor<ElementAccumulator, typename B2bConv2d::LayoutC> tensor_Z0_reference;
  cutlass::HostTensor<typename B2bConv2d::ElementC, typename B2bConv2d::LayoutC> tensor_D0_reference;

  cutlass::HostTensor<typename B2bConv2d::ElementB, typename B2bConv2d::LayoutB> tensor_B1;
  cutlass::HostTensor<typename B2bConv2d::ElementC, typename B2bConv2d::LayoutC> tensor_C1;
  cutlass::HostTensor<typename B2bConv2d::ElementCompute, typename B2bConv2d::LayoutC> tensor_Bias1;
  cutlass::HostTensor<typename B2bConv2d::ElementC, typename B2bConv2d::LayoutC> tensor_D1_computed;
  cutlass::HostTensor<typename B2bConv2d::ElementC, typename B2bConv2d::LayoutC> tensor_D1_reference;


public:

  B2bFusedConv2dRun(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_Scale_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_Bias_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    init_A(init_A_), init_B(init_B_), init_C(init_C_),
    init_Scale(init_Scale_), init_Bias(init_Bias_), seed(seed_) {

  }

    /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  void initialize_tensor(
    cutlass::TensorView<Element, Layout> view, 
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      int scope;
      int bits = cutlass::sizeof_bits<Element>::value;

      if (bits <= 16) {
        scope = 2;
      }
      else {
        scope = 8;
      }
      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, scope, -scope, 0);
    } 
    else if (dist_kind == cutlass::Distribution::Identity) {

      cutlass::reference::host::TensorFillIdentity(view);
    } 
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {

      cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
    } 
    else if (dist_kind == cutlass::Distribution::AllZeros) {
      cutlass::reference::host::TensorFill(view, Element(0));
    }
    else if (dist_kind == cutlass::Distribution::AllOnes) {
      cutlass::reference::host::TensorFill(view, Element(1));
    }
    else {
    }
  }

  void initialize(
    cutlass::conv::Conv2dProblemSize const &problem_size_0,
    cutlass::conv::Conv2dProblemSize const &problem_size_1,
    ElementCompute alpha0,
    ElementCompute alpha1,
    uint64_t seed = 2019) {
        
    tensor_A0.resize(implicit_gemm_tensor_a_extent(kConvolutionalOperator, problem_size_0));
    tensor_B0.resize(implicit_gemm_tensor_b_extent(kConvolutionalOperator, problem_size_0));
    tensor_C0.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size_0));
    if(alpha0 == ElementCompute(0)) //per-channel scale
        tensor_Scale0.resize({1, problem_size_0.K});
    tensor_Bias0.resize({1, problem_size_0.K});
    tensor_Z0_reference.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size_0));
    tensor_D0_reference.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size_0));
    tensor_B1.resize(implicit_gemm_tensor_b_extent(kConvolutionalOperator, problem_size_1));
    tensor_C1.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size_1));
    tensor_Bias1.resize({1, 1, 1, problem_size_1.K});
    tensor_D1_computed.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size_1));
    tensor_D1_reference.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size_1));

    initialize_tensor(tensor_A0.host_view(), init_A, seed); 
    initialize_tensor(tensor_B0.host_view(), init_B, seed * 17); 
    initialize_tensor(tensor_C0.host_view(), init_C, seed * 39);
    if(alpha0 == ElementCompute(0)) //per-channel scale
        initialize_tensor(tensor_Scale0.host_view(), init_Scale, seed * 61);
    initialize_tensor(tensor_Bias0.host_view(), init_Bias, seed * 83);
    initialize_tensor(tensor_B1.host_view(), init_B, seed * 18); 
    initialize_tensor(tensor_C1.host_view(), init_C, seed * 40);
    initialize_tensor(tensor_Bias1.host_view(), init_Bias, seed * 84);

    tensor_A0.sync_device();
    tensor_B0.sync_device();
    tensor_C0.sync_device();
    if(alpha0 == ElementCompute(0)) //per-channel scale
        tensor_Scale0.sync_device();
    tensor_Bias0.sync_device();
    tensor_D0_reference.sync_device();
    tensor_B1.sync_device();
    tensor_C1.sync_device();
    tensor_Bias1.sync_device();
    tensor_D1_computed.sync_device();
    tensor_D1_reference.sync_device();
  }

  /// Executes one test
  bool run(
    cutlass::conv::Conv2dProblemSize const &problem_size_0,
    cutlass::conv::Conv2dProblemSize const &problem_size_1,
    cutlass::conv::SplitKMode const &split_k_mode = cutlass::conv::SplitKMode::kSerial,
    ElementCompute alpha0 = ElementCompute(1),
    ElementCompute beta0 = ElementCompute(0),
    ElementCompute alpha1 = ElementCompute(1),
    ElementCompute beta1 = ElementCompute(0),
    bool relu = true,
    int warm_ups = 1,
    int runs = 100) {

    initialize(problem_size_0, problem_size_1, alpha0, alpha1);

    // configure the operator
    B2bConv2d b2b_conv2d_op;

    typename B2bConv2d::Arguments b2b_conv2d_args(
      problem_size_0,
      problem_size_1,
      tensor_A0.device_ref(),
      tensor_B0.device_ref(),
      tensor_C0.device_ref(),
      tensor_Scale0.device_ref(),
      tensor_Bias0.device_ref(),
      tensor_B1.device_ref(),
      {tensor_Bias1.device_data(), typename B2bConv2d::LayoutC::Stride(0)},
      tensor_D1_computed.device_ref(),
      {alpha0, beta0},
      {alpha1, beta1},
      split_k_mode
    );

    cutlass::Status status = b2b_conv2d_op.can_implement(b2b_conv2d_args);
    
    if(status != cutlass::Status::kSuccess) {
        std::cout << "Problem sizes not supported.\n"
                << "Requirments:\n"
                << "    problem_size_0.N*P*Q = problem_size_1.N*P*Q\n"
                << "    problem_size_0.K = problem_size_1.C\n"
                << "    problem_size_1.R = problem_size_1.S = 1\n"
                << "    ThreadblockShape0::kN = problem_size_0.K\n"
                << "    ThreadblockShape1::kN = problem_size_1.K" << std::endl;
    }

    CUTLASS_CHECK(status);

    status = b2b_conv2d_op.initialize(b2b_conv2d_args);

    CUTLASS_CHECK(status);

    for(int i = 0; i < warm_ups; i++) {
        status = b2b_conv2d_op();
        CUTLASS_CHECK(status);
    }

    //
    // Run the Conv2d
    //

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for(int i = 0; i < runs; i++) {

        // run conv2d operator
        status = b2b_conv2d_op();
        CUTLASS_CHECK(status);
    }
    
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float conv2dTime;
    cudaEventElapsedTime(&conv2dTime, start, stop);
    std::cout << "Fusion time " << conv2dTime / (float)runs << " ms\n";

    tensor_D1_computed.sync_host();
    
    bool passed = false;

    cutlass::reference::device::Conv2d<
      typename B2bConv2d::ElementA,
      typename B2bConv2d::LayoutA,
      typename B2bConv2d::ElementB,
      typename B2bConv2d::LayoutB,
      ElementAccumulator,
      typename B2bConv2d::LayoutC,
      ElementAccumulator,
      ElementAccumulator
    >(
      kConvolutionalOperator,
      problem_size_0,
      tensor_A0.device_ref(),
      tensor_B0.device_ref(),
      tensor_Z0_reference.device_ref(),
      tensor_Z0_reference.device_ref(),
      ElementAccumulator(1), // intermediate alpha = 1
      ElementAccumulator(0)  // beta = 0
    );

    cutlass::reference::device::TensorScaleBiasConv2d<
      ElementAccumulator,
      typename B2bConv2d::ElementC,
      typename B2bConv2d::LayoutC,
      ElementCompute,
      typename B2bConv2d::LayoutScaleBias
    >(
      problem_size_0,
      tensor_Z0_reference.device_ref(),
      tensor_D0_reference.device_ref(),
      alpha0,
      tensor_Scale0.device_ref(),
      tensor_Bias0.device_ref()
    );

    if(relu) {
       cutlass::reference::device::TensorReLu(tensor_D0_reference.device_view()); 
    }

    cutlass::reference::device::Conv2d<
      typename B2bConv2d::ElementA,
      typename B2bConv2d::LayoutA,
      typename B2bConv2d::ElementB,
      typename B2bConv2d::LayoutB,
      typename B2bConv2d::ElementC,
      typename B2bConv2d::LayoutC,
      ElementCompute,
      ElementAccumulator
    >(
      kConvolutionalOperator,
      problem_size_1,
      tensor_D0_reference.device_ref(),
      tensor_B1.device_ref(),
      {tensor_Bias1.device_data(), typename B2bConv2d::LayoutC::Stride(0)},
      tensor_D1_reference.device_ref(),
      alpha1, 
      beta1);

    if(relu) {
       cutlass::reference::device::TensorReLu(tensor_D1_reference.device_view()); 
    }

    cudaError_t result = cudaDeviceSynchronize();
    CHECK_TRUE(result == cudaSuccess);

    // sync host (copy device data to host) for dumping error output in case of mismatches
    tensor_D0_reference.sync_host();
    tensor_D1_reference.sync_host();
    
    CHECK_GT(cutlass::reference::host::TensorNorm(tensor_D0_reference.host_view()), 0);
    CHECK_GT(cutlass::reference::host::TensorNorm(tensor_D1_computed.host_view()), 0);
    CHECK_GT(cutlass::reference::host::TensorNorm(tensor_D1_reference.host_view()), 0);

    passed = cutlass::reference::host::TensorEquals(
      tensor_D1_computed.host_view(), 
      tensor_D1_reference.host_view());

    CHECK_TRUE(passed);

    if (!passed) {
      std::stringstream fname;

      fname << "error_B2bImplicitGemm_device_fused.txt";
      std::cerr << "Dumping results in " << fname.str() << "\n";

      std::ofstream results(fname.str());

      results << problem_size_0 << std::endl;
      results << problem_size_1 << std::endl;

      results
        << "\nA0:\n" << tensor_A0.host_view() << "\n"
        << "\nB0:\n" << tensor_B0.host_view() << "\n"
        << "\nC0:\n" << tensor_C0.host_view() << "\n"
        << "\nScale0:\n" << tensor_Scale0.host_view() << "\n"
        << "\nBias0:\n" << tensor_Bias0.host_view() << "\n"
        << "\nB1:\n" << tensor_B1.host_view() << "\n"
        << "\nC1:\n" << tensor_C1.host_view() << "\n"
        << "\nBias1:\n" << tensor_Bias1.host_view() << "\n"
        << "\nD1 reference:\n" << tensor_D1_reference.host_view() << "\n"
        << "\nD1 computed:\n" << tensor_D1_computed.host_view();


    }

    return passed;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////
