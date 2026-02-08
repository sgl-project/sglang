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

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_relu.h"

#include "reference/device/tensor_scale_bias.h"
#include "helper.h"

#define CHECK_GT(val1, val2) \
    if((val1) <= (val2)) \
        std::cerr << __FILE__ << " " << __LINE__ << ": CHECK_GT failed\n";
#define CHECK_TRUE(val) \
    if(!(val)) \
        std::cerr << __FILE__ << " " << __LINE__ << ": CHECK_TRUE failed\n";

////////////////////////////////////////////////////////////////////////////////

template <typename Gemm0_, typename Gemm1_>
struct B2bNonFusedGemmRun
{

  using Gemm0 = Gemm0_;
  using Gemm1 = Gemm1_;
  using ElementAccumulator = typename Gemm0::ElementAccumulator;
  using ElementCompute = typename Gemm0::GemmKernel::Epilogue::OutputOp::ElementCompute;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  cutlass::Distribution::Kind init_Bias;
  uint64_t seed;

  //
  // Methods
  //

  B2bNonFusedGemmRun(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_Bias_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    init_A(init_A_), init_B(init_B_), init_C(init_C_), init_Bias(init_Bias_), seed(seed_) { }

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool initialize_tensor(
    cutlass::TensorView<Element, Layout> view,
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, 2, -2, 0);
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
    else if (dist_kind == cutlass::Distribution::AllZeros) {
      cutlass::reference::host::TensorFill(view, Element(0));
    }
    else if (dist_kind == cutlass::Distribution::AllOnes) {
      cutlass::reference::host::TensorFill(view, Element(1));
    }
    else {
      std::cerr << "Not implemented\n";
      return false;
    }

    return true;
  }




  /// Executes one test
  bool run(
    cutlass::gemm::GemmCoord problem_size_0,
    cutlass::gemm::GemmCoord problem_size_1,
    ElementCompute alpha0 = ElementCompute(1),
    ElementCompute beta0 = ElementCompute(0),
    ElementCompute alpha1 = ElementCompute(1),
    ElementCompute beta1 = ElementCompute(0),
    bool relu = true,
    int warm_ups = 1,
    int runs = 100) {

    //
    // Allocate the GEMM workspace
    //

    cutlass::HostTensor<
      typename Gemm0::ElementA,
      typename Gemm0::LayoutA> tensor_A0(problem_size_0.mk());

    cutlass::HostTensor<
      typename Gemm0::ElementB,
      typename Gemm0::LayoutB> tensor_B0(problem_size_0.kn());

    cutlass::HostTensor<
      typename Gemm0::ElementC,
      typename Gemm0::LayoutC> tensor_C0(problem_size_0.mn());

    cutlass::HostTensor<
      ElementCompute,
      typename Gemm0::LayoutC> tensor_Bias0({1, problem_size_0.n()});

    cutlass::HostTensor<
      typename Gemm0::ElementC,
      typename Gemm0::LayoutC> tensor_D0(problem_size_0.mn());

    cutlass::HostTensor<
      typename Gemm0::ElementC,
      typename Gemm0::LayoutC> reference_D0(problem_size_0.mn());

    cutlass::HostTensor<
      typename Gemm1::ElementB,
      typename Gemm1::LayoutB> tensor_B1(problem_size_1.kn());

    cutlass::HostTensor<
      typename Gemm1::ElementC,
      typename Gemm1::LayoutC> tensor_C1(problem_size_1.mn());

    cutlass::HostTensor<
      ElementCompute,
      typename Gemm1::LayoutC> tensor_Bias1({1, problem_size_1.n()});

    cutlass::HostTensor<
      typename Gemm1::ElementC,
      typename Gemm1::LayoutC> tensor_D1(problem_size_1.mn());

    cutlass::HostTensor<
      typename Gemm1::ElementC,
      typename Gemm1::LayoutC> reference_D1(problem_size_1.mn());


    CHECK_TRUE(initialize_tensor(tensor_A0.host_view(), init_A, seed + 2019));
    CHECK_TRUE(initialize_tensor(tensor_B0.host_view(), init_B, seed + 2018));
    CHECK_TRUE(initialize_tensor(tensor_C0.host_view(), init_C, seed + 2017));
    CHECK_TRUE(initialize_tensor(tensor_Bias0.host_view(), init_Bias, seed + 2014));
    CHECK_TRUE(initialize_tensor(tensor_B1.host_view(), init_B, seed + 2016));
    CHECK_TRUE(initialize_tensor(tensor_C1.host_view(), init_C, seed + 2015));
    CHECK_TRUE(initialize_tensor(tensor_Bias1.host_view(), init_Bias, seed + 2013));

    cutlass::reference::host::TensorFill(
      tensor_D0.host_view());
    cutlass::reference::host::TensorFill(
      tensor_D1.host_view());
    cutlass::reference::host::TensorFill(
      reference_D0.host_view());
    cutlass::reference::host::TensorFill(
      reference_D1.host_view());

    tensor_A0.sync_device();
    tensor_B0.sync_device();
    tensor_C0.sync_device();
    tensor_Bias0.sync_device();
    tensor_D0.sync_device();
    tensor_B1.sync_device();
    tensor_C1.sync_device();
    tensor_Bias1.sync_device();
    tensor_D1.sync_device();
    reference_D0.sync_device();
    reference_D1.sync_device();

    //
    // Initialize the GEMM operator
    //

    typename Gemm0::Arguments arguments_0{
      problem_size_0,
      tensor_A0.device_ref(),
      tensor_B0.device_ref(),
      {tensor_Bias0.device_data(), typename Gemm0::LayoutC::Stride(0)},
      tensor_D0.device_ref(),
      {alpha0, beta0}
    };

    typename Gemm1::Arguments arguments_1{
      problem_size_1,
      tensor_D0.device_ref(),
      tensor_B1.device_ref(),
      {tensor_Bias1.device_data(), typename Gemm1::LayoutC::Stride(0)},
      tensor_D1.device_ref(),
      {alpha1, beta1}
    };


    Gemm0 gemm_op_0;
    Gemm1 gemm_op_1;

    cutlass::Status status = gemm_op_0.initialize(arguments_0);

    CUTLASS_CHECK(status);

    status = gemm_op_1.initialize(arguments_1);

    CUTLASS_CHECK(status);

    for(int i = 0; i < warm_ups; i++) {
        status = gemm_op_0();
        CUTLASS_CHECK(status);
        status = gemm_op_1();
        CUTLASS_CHECK(status);
    }

    //
    // Run the GEMM
    //
    cudaEvent_t start, stop1, stop2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop1);
    cudaEventCreate(&stop2);

    cudaEventRecord(start);

    for(int i = 0; i < runs; i++) {
        status = gemm_op_0();

        CUTLASS_CHECK(status);
    }
    cudaEventRecord(stop1);
    for(int i = 0; i < runs; i++) {
        status = gemm_op_1();

        CUTLASS_CHECK(status);
    }

    cudaEventRecord(stop2);
    cudaDeviceSynchronize();
    float gemm0Time, gemm1Time, totalTime;
    cudaEventElapsedTime(&gemm0Time, start, stop1);
    cudaEventElapsedTime(&gemm1Time, stop1, stop2);
    cudaEventElapsedTime(&totalTime, start, stop2);
    std::cout << "gemm 0 time " << gemm0Time / (float)runs << " ms\n";
    std::cout << "gemm 1 time " << gemm1Time / (float)runs << " ms\n";
    std::cout << "Non-fusion time " << totalTime / (float)runs << " ms\n";

    tensor_D0.sync_host();
    tensor_D1.sync_host();

    //
    // Verify
    //
    cutlass::reference::device::Gemm<
        typename Gemm0::ElementA, typename Gemm0::LayoutA,
        typename Gemm0::ElementB, typename Gemm0::LayoutB,
        typename Gemm0::ElementC, typename Gemm0::LayoutC, ElementCompute,
        ElementAccumulator, typename Gemm0::Operator>
        reference_gemm_0;

    cutlass::reference::device::Gemm<
        typename Gemm1::ElementA, typename Gemm1::LayoutA,
        typename Gemm1::ElementB, typename Gemm1::LayoutB,
        typename Gemm1::ElementC, typename Gemm1::LayoutC, ElementCompute,
        ElementAccumulator, typename Gemm1::Operator>
        reference_gemm_1;

    reference_gemm_0(
      problem_size_0,
      alpha0,
      tensor_A0.device_ref(),
      tensor_B0.device_ref(),
      beta0,
      {tensor_Bias0.device_data(), typename Gemm0::LayoutC::Stride(0)},
      reference_D0.device_ref()
    );

    if(relu) {
       cutlass::reference::device::TensorReLu(reference_D0.device_view());
    }

    reference_gemm_1(
      problem_size_1,
      alpha1,
      reference_D0.device_ref(),
      tensor_B1.device_ref(),
      beta1,
      {tensor_Bias1.device_data(), typename Gemm1::LayoutC::Stride(0)},
      reference_D1.device_ref()
    );

    if(relu) {
       cutlass::reference::device::TensorReLu(reference_D1.device_view());
    }

    // Wait for kernels to finish
    cudaDeviceSynchronize();
    reference_D0.sync_host();
    reference_D1.sync_host();

    CHECK_GT(cutlass::reference::host::TensorNorm(tensor_D0.host_view()), 0);
    CHECK_GT(cutlass::reference::host::TensorNorm(reference_D0.host_view()), 0);
    CHECK_GT(cutlass::reference::host::TensorNorm(tensor_D1.host_view()), 0);
    CHECK_GT(cutlass::reference::host::TensorNorm(reference_D1.host_view()), 0);

    bool passed = cutlass::reference::host::TensorEquals(
      reference_D1.host_view(),
      tensor_D1.host_view());

    CHECK_TRUE(passed);
    if (!passed) {

      std::stringstream fname;

      fname << "error_B2bGemm_device_nonfused.txt";
      std::cerr << "Dumping results in " << fname.str() << "\n";

      std::ofstream file(fname.str());

      file
        << "A0 =\n" << tensor_A0.host_view()
        << "\nB0 =\n" << tensor_B0.host_view()
        << "\nC0 =\n" << tensor_C0.host_view()
        << "\nBias0:\n" << tensor_Bias0.host_view() << "\n"
        << "\nD0 =\n" << tensor_D0.host_view()
        << "\nB1 =\n" << tensor_B1.host_view()
        << "\nC1 =\n" << tensor_C1.host_view()
        << "\nBias1:\n" << tensor_Bias1.host_view() << "\n"
        << "\n\nReference =\n" << reference_D1.host_view()
        << "\nComputed =\n" << tensor_D1.host_view();
    }
    return passed;
  }
};

template <typename B2bGemm_>
struct B2bFusedGemmRun
{

  using B2bGemm = B2bGemm_;
  using ElementAccumulator = typename B2bGemm::ElementAccumulator;
  using ElementCompute = typename B2bGemm::B2bGemmKernel::Epilogue::OutputOp::ElementCompute;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  cutlass::Distribution::Kind init_Scale;
  cutlass::Distribution::Kind init_Bias;
  uint64_t seed;

  //
  // Methods
  //

  B2bFusedGemmRun(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_Scale_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_Bias_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    init_A(init_A_), init_B(init_B_), init_C(init_C_),
    init_Scale(init_Scale_), init_Bias(init_Bias_), seed(seed_) { }

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool initialize_tensor(
    cutlass::TensorView<Element, Layout> view,
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, 2, -2, 0);
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
    else if (dist_kind == cutlass::Distribution::AllZeros) {
      cutlass::reference::host::TensorFill(view, Element(0));
    }
    else if (dist_kind == cutlass::Distribution::AllOnes) {
      cutlass::reference::host::TensorFill(view, Element(1));
    }
    else {
      std::cerr << "Not implemented\n";
      return false;
    }

    return true;
  }




  /// Executes one test
  bool run(
    cutlass::gemm::GemmCoord problem_size_0,
    cutlass::gemm::GemmCoord problem_size_1,
    ElementCompute alpha0 = ElementCompute(1),
    ElementCompute beta0 = ElementCompute(0),
    ElementCompute alpha1 = ElementCompute(1),
    ElementCompute beta1 = ElementCompute(0),
    cutlass::gemm::GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemm,

    // batch_count is used as split-k when mode is kGemm according
    // to the GemmUniversal interface

    int batch_count = 1,
    int64_t batch_stride_A0 = 0,
    int64_t batch_stride_B0 = 0,
    int64_t batch_stride_C0 = 0,
    int64_t batch_stride_B1 = 0,
    int64_t batch_stride_C1 = 0,
    int64_t batch_stride_D1 = 0,
    int64_t batch_stride_Bias0 = 0,
    int64_t batch_stride_Scale0 = 0,
    bool relu = true,
    int warm_ups = 1,
    int runs = 100) {

    //
    // Allocate the GEMM workspace
    //

    cutlass::gemm::GemmCoord CoordA0(problem_size_0.m(), problem_size_0.n(), batch_count * problem_size_0.k());
    cutlass::gemm::GemmCoord CoordB0(problem_size_0.m(), problem_size_0.n(), batch_count * problem_size_0.k());
    cutlass::gemm::GemmCoord CoordC0(problem_size_0.m(), batch_count * problem_size_0.n(), problem_size_0.k());
    cutlass::gemm::GemmCoord CoordB1(problem_size_1.m(), problem_size_1.n(), batch_count * problem_size_1.k());
    cutlass::gemm::GemmCoord CoordC1(problem_size_1.m(), batch_count * problem_size_1.n(), problem_size_1.k());

    cutlass::HostTensor<
      typename B2bGemm::ElementA,
      typename B2bGemm::LayoutA> tensor_A0(CoordA0.mk());

    cutlass::HostTensor<
      typename B2bGemm::ElementB,
      typename B2bGemm::LayoutB> tensor_B0(CoordB0.kn());

    cutlass::HostTensor<
      typename B2bGemm::ElementC,
      typename B2bGemm::LayoutC> tensor_C0(CoordC0.mn());

    cutlass::HostTensor<
      typename B2bGemm::ElementScaleBias,
      typename B2bGemm::LayoutScaleBias> tensor_Scale0;

    if(alpha0 == ElementCompute(0)) //per-channel scale
        tensor_Scale0.resize({1, batch_count * problem_size_0.n()});

    cutlass::HostTensor<
      typename B2bGemm::ElementScaleBias,
      typename B2bGemm::LayoutScaleBias> tensor_Bias0({1, batch_count * problem_size_0.n()});

    cutlass::HostTensor<
      ElementAccumulator,
      typename B2bGemm::LayoutC> reference_Z0(CoordC0.mn());

    cutlass::HostTensor<
      typename B2bGemm::ElementC,
      typename B2bGemm::LayoutC> reference_D0(CoordC0.mn());

    cutlass::HostTensor<
      typename B2bGemm::ElementB,
      typename B2bGemm::LayoutB> tensor_B1(CoordB1.kn());

    cutlass::HostTensor<
      typename B2bGemm::ElementC,
      typename B2bGemm::LayoutC> tensor_C1(CoordC1.mn());

    cutlass::HostTensor<
      typename B2bGemm::ElementC,
      typename B2bGemm::LayoutScaleBias> tensor_Bias1({1, batch_count * problem_size_1.n()});

    cutlass::HostTensor<
      typename B2bGemm::ElementC,
      typename B2bGemm::LayoutC> tensor_D1(CoordC1.mn());

    cutlass::HostTensor<
      typename B2bGemm::ElementC,
      typename B2bGemm::LayoutC> reference_D1(CoordC1.mn());


    CHECK_TRUE(initialize_tensor(tensor_A0.host_view(), init_A, seed + 2019));
    CHECK_TRUE(initialize_tensor(tensor_B0.host_view(), init_B, seed + 2018));
    CHECK_TRUE(initialize_tensor(tensor_C0.host_view(), init_C, seed + 2017));
    if(alpha0 == ElementCompute(0)) //per-channel scale
      CHECK_TRUE(initialize_tensor(tensor_Scale0.host_view(), init_Scale, seed + 2014));
    CHECK_TRUE(initialize_tensor(tensor_Bias0.host_view(), init_Bias, seed + 2013));
    CHECK_TRUE(initialize_tensor(tensor_B1.host_view(), init_B, seed + 2016));
    CHECK_TRUE(initialize_tensor(tensor_C1.host_view(), init_C, seed + 2015));
    CHECK_TRUE(initialize_tensor(tensor_Bias1.host_view(), init_Bias, seed + 2012));

    cutlass::reference::host::TensorFill(
      tensor_D1.host_view());
    cutlass::reference::host::TensorFill(
      reference_D0.host_view());
    cutlass::reference::host::TensorFill(
      reference_D1.host_view());

    tensor_A0.sync_device();
    tensor_B0.sync_device();
    tensor_C0.sync_device();
    if(alpha0 == ElementCompute(0)) //per-channel scale
        tensor_Scale0.sync_device();
    tensor_Bias0.sync_device();
    tensor_B1.sync_device();
    tensor_C1.sync_device();
    tensor_Bias1.sync_device();
    tensor_D1.sync_device();
    reference_D0.sync_device();
    reference_D1.sync_device();

    //
    // Initialize the GEMM operator
    //

    typename B2bGemm::Arguments arguments{
      mode,
      problem_size_0,
      problem_size_1,
      tensor_A0.device_ref(),
      tensor_B0.device_ref(),
      tensor_C0.device_ref(),
      tensor_Scale0.device_ref(),
      tensor_Bias0.device_ref(),
      tensor_B1.device_ref(),
      {tensor_Bias1.device_data(), typename B2bGemm::LayoutC::Stride(0)},
      tensor_D1.device_ref(),
      batch_stride_A0,
      batch_stride_B0,
      batch_stride_B1,
      batch_stride_C1,
      batch_stride_D1,
      batch_stride_Bias0,
      batch_stride_Scale0,
      {alpha0, beta0},
      {alpha1, beta1},
      batch_count,
    };

    B2bGemm b2b_gemm_op;

    cutlass::Status status = b2b_gemm_op.can_implement(arguments);

    if(status != cutlass::Status::kSuccess) {
        std::cout << "Problem sizes not supported.\n"
                << "Requirments:\n"
                << "    problem_size_0.M = problem_size_1.M\n"
                << "    problem_size_0.N = problem_size_1.K\n"
                << "    ThreadblockShape0::kN = problem_size_0.N\n"
                << "    ThreadblockShape1::kN = problem_size_1.N" << std::endl;
    }

    status = b2b_gemm_op.initialize(arguments);

    CUTLASS_CHECK(status);

    for(int i = 0; i < warm_ups; i++) {
        status = b2b_gemm_op();
        CUTLASS_CHECK(status);
    }

    //
    // Run the GEMM
    //

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for(int i = 0; i < runs; i++) {
        status = b2b_gemm_op();

        CUTLASS_CHECK(status);
    }

    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float gemmTime;
    cudaEventElapsedTime(&gemmTime, start, stop);
    std::cout << "Fusion time " << gemmTime / (float)runs << " ms\n";

    tensor_D1.sync_host();

    //
    // Verify
    //

    cutlass::reference::device::GemmComplex<
      typename B2bGemm::ElementA, typename B2bGemm::LayoutA,
      typename B2bGemm::ElementB, typename B2bGemm::LayoutB,
      ElementAccumulator, typename B2bGemm::LayoutC,
      ElementAccumulator, ElementAccumulator
    >(

      problem_size_0,
      ElementAccumulator(1), //intermediate alpha=1
      tensor_A0.device_ref(),
      cutlass::ComplexTransform::kNone,
      tensor_B0.device_ref(),
      cutlass::ComplexTransform::kNone,
      ElementAccumulator(0), //beta = 0
      reference_Z0.device_ref(),
      reference_Z0.device_ref(),
      ElementAccumulator(0),
      int(batch_count),
      batch_stride_A0,
      batch_stride_B0,
      batch_stride_C0,
      batch_stride_C0
    );

    cutlass::reference::device::TensorScaleBiasGemmBatched<
      ElementAccumulator, typename B2bGemm::ElementC, typename B2bGemm::LayoutC,
      ElementCompute, typename B2bGemm::LayoutScaleBias
    > (
      problem_size_0,
      reference_Z0.device_ref(),
      reference_D0.device_ref(),
      alpha0,
      tensor_Scale0.device_ref(),
      tensor_Bias0.device_ref(),
      int(batch_count),
      batch_stride_C0,
      batch_stride_C0,
      batch_stride_Scale0,
      batch_stride_Bias0
    );

    if(relu) {
       cutlass::reference::device::TensorReLu(reference_D0.device_view());
    }

    cutlass::reference::device::GemmComplex<
      typename B2bGemm::ElementA, typename B2bGemm::LayoutA,
      typename B2bGemm::ElementB, typename B2bGemm::LayoutB,
      typename B2bGemm::ElementC, typename B2bGemm::LayoutC,
      ElementCompute, ElementAccumulator
    >(
      problem_size_1,
      alpha1, //intermediate alpha=1
      reference_D0.device_ref(),
      cutlass::ComplexTransform::kNone,
      tensor_B1.device_ref(),
      cutlass::ComplexTransform::kNone,
      beta1, //beta = 0
      {tensor_Bias1.device_data(), typename B2bGemm::LayoutC::Stride(0)},
      reference_D1.device_ref(),
      ElementAccumulator(0),
      int(batch_count),
      batch_stride_C0,
      batch_stride_B1,
      batch_stride_C1,
      batch_stride_D1
    );

    if(relu) {
       cutlass::reference::device::TensorReLu(reference_D1.device_view());
    }

    cudaDeviceSynchronize();
    reference_D0.sync_host();
    reference_D1.sync_host();

    CHECK_GT(cutlass::reference::host::TensorNorm(reference_D0.host_view()), 0);
    CHECK_GT(cutlass::reference::host::TensorNorm(tensor_D1.host_view()), 0);
    CHECK_GT(cutlass::reference::host::TensorNorm(reference_D1.host_view()), 0);

    bool passed = cutlass::reference::host::TensorEquals(
      reference_D1.host_view(),
      tensor_D1.host_view());

    CHECK_TRUE(passed);
    if (!passed)
    {

      std::stringstream fname;

      fname << "error_B2bGemm_device_fused.txt";
      std::cerr << "Dumping results in " << fname.str() << "\n";

      std::ofstream file(fname.str());

      file
        << "A0 =\n" << tensor_A0.host_view()
        << "\nB0 =\n" << tensor_B0.host_view()
        << "\nC0 =\n" << tensor_C0.host_view()
        << "\nScale0:\n" << tensor_Scale0.host_view() << "\n"
        << "\nBias0:\n" << tensor_Bias0.host_view() << "\n"
        << "\nB1 =\n" << tensor_B1.host_view()
        << "\nC1 =\n" << tensor_C1.host_view()
        << "\nBias1:\n" << tensor_Bias1.host_view() << "\n"
        << "\n\nReference =\n" << reference_D1.host_view()
        << "\nComputed =\n" << tensor_D1.host_view();
    }
    return passed;
  }

};

////////////////////////////////////////////////////////////////////////////////
