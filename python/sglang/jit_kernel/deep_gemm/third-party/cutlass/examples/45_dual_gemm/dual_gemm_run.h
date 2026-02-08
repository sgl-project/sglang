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
#include <type_traits>

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_relu.h"

#include "cutlass/platform/platform.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "dual_gemm_common.h"
#include "helper.h"

#define CHECK_GT(val1, val2) \
    if((val1) <= (val2)) \
        std::cerr << __FILE__ << " " << __LINE__ << ": CHECK_GT failed\n";
#define CHECK_TRUE(val) \
    if(!(val)) \
        std::cerr << __FILE__ << " " << __LINE__ << ": CHECK_TRUE failed\n";

template <
  typename OutputOp,
  typename Element,
  typename Layout>
struct TensorEpilogueForEachFunc {
  /// View type
  using TensorView = cutlass::TensorView<Element, Layout>;

  /// Coordinate in tensor's index space
  using TensorCoord = typename TensorView::TensorCoord;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    TensorView view_x0;
    TensorView view_x1;
    TensorView view_y;
    OutputOp output_op;


    //
    // Methods
    //

    Params(
      TensorView view_x0_ = TensorView(),
      TensorView view_x1_ = TensorView(),
      TensorView view_y_ = TensorView(),
      OutputOp output_op_ = OutputOp(typename OutputOp::Params{})
    ):
      view_x0(view_x0_), view_x1(view_x1_), view_y(view_y_), output_op(output_op_) {
    }
  };

  Params params;

  CUTLASS_DEVICE
  TensorEpilogueForEachFunc(Params const &params): params(params) {

  }

  CUTLASS_DEVICE
  void operator()(TensorCoord const &coord) {
    Element const & x0 = params.view_x0.at(coord);
    Element const & x1 = params.view_x1.at(coord);
    Element& y = params.view_y.at(coord);
    y = params.output_op(x0, x1);
  }
};

template <
  typename OutputOp,
  typename Element,
  typename Layout>
void TensorEpilogueForEach(
  cutlass::TensorView<Element, Layout> x0,
  cutlass::TensorView<Element, Layout> x1,
  cutlass::TensorView<Element, Layout> y) {
  
  using Func = TensorEpilogueForEachFunc<OutputOp, Element, Layout>;
  using Params = typename Func::Params;

  cutlass::reference::device::TensorForEach<Func, Layout::kRank, Params>(
    y.extent(),
    Params(x0, x1, y)
  );
}

////////////////////////////////////////////////////////////////////////////////

template <typename Gemm0_, typename Gemm1_>
struct NonFusedDualGemmRun
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

  NonFusedDualGemmRun(
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
    cutlass::gemm::GemmCoord problem_size,
    ElementCompute alpha0 = ElementCompute(1),
    ElementCompute beta0 = ElementCompute(0),
    ElementCompute alpha1 = ElementCompute(1),
    ElementCompute beta1 = ElementCompute(0),
    bool is_profiling = true,
    bool relu = false,
    int warm_ups = 1,
    int runs = 100) {
    
    //
    // Allocate the GEMM workspace
    //

    cutlass::HostTensor<
      typename Gemm0::ElementA, 
      typename Gemm0::LayoutA> tensor_A0(problem_size.mk());

    cutlass::HostTensor<
      typename Gemm0::ElementB, 
      typename Gemm0::LayoutB> tensor_B0(problem_size.kn());

    cutlass::HostTensor<
      typename Gemm0::ElementC, 
      typename Gemm0::LayoutC> tensor_C0(problem_size.mn());

    cutlass::HostTensor<
      typename Gemm1::ElementC,
      typename Gemm0::LayoutC> tensor_Bias0({1, problem_size.n()});

    cutlass::HostTensor<
      typename Gemm0::ElementC, 
      typename Gemm0::LayoutC> tensor_D0(problem_size.mn());

    cutlass::HostTensor<
      typename Gemm0::ElementC, 
      typename Gemm0::LayoutC> reference_D0(problem_size.mn());

    cutlass::HostTensor<
      typename Gemm1::ElementB, 
      typename Gemm1::LayoutB> tensor_B1(problem_size.kn());

    cutlass::HostTensor<
      typename Gemm1::ElementC, 
      typename Gemm1::LayoutC> tensor_C1(problem_size.mn());

    cutlass::HostTensor<
      typename Gemm1::ElementC,
      typename Gemm1::LayoutC> tensor_Bias1({1, problem_size.n()});

    cutlass::HostTensor<
      typename Gemm1::ElementC, 
      typename Gemm1::LayoutC> tensor_D1(problem_size.mn());

    cutlass::HostTensor<
      typename Gemm1::ElementC, 
      typename Gemm1::LayoutC> reference_D1(problem_size.mn());


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
    reference_D0.sync_device();
    tensor_B1.sync_device();
    tensor_C1.sync_device();
    tensor_Bias1.sync_device();
    tensor_D1.sync_device();
    reference_D1.sync_device();

    //
    // Initialize the GEMM operator
    //

    int split_k_slices = Gemm0::kSplitKSerial ? 2 : 1;
    typename Gemm0::Arguments arguments_0{
      problem_size,
      tensor_A0.device_ref(),
      tensor_B0.device_ref(),
      {tensor_Bias0.device_data(), typename Gemm0::LayoutC::Stride(0)},
      tensor_D0.device_ref(),
      {alpha0, beta0},
      split_k_slices
    };

    split_k_slices = Gemm1::kSplitKSerial ? 2 : 1;
    typename Gemm1::Arguments arguments_1{
      problem_size,
      tensor_A0.device_ref(),
      tensor_B1.device_ref(),
      {tensor_Bias1.device_data(), typename Gemm1::LayoutC::Stride(0)},
      tensor_D1.device_ref(),
      {alpha1, beta1},
      split_k_slices
    };


    Gemm0 gemm_op_0;
    Gemm1 gemm_op_1;

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace0(gemm_op_0.get_workspace_size(arguments_0));
    cutlass::device_memory::allocation<uint8_t> workspace1(gemm_op_1.get_workspace_size(arguments_1));

    cutlass::Status status = gemm_op_0.initialize(arguments_0, workspace0.get());

    CUTLASS_CHECK(status);

    status = gemm_op_1.initialize(arguments_1, workspace1.get());

    CUTLASS_CHECK(status);

    for(int i = 0; i < warm_ups; i++) {
        status = gemm_op_0();
        CUTLASS_CHECK(status);
        status = gemm_op_1();
        CUTLASS_CHECK(status);
    }

    if (is_profiling) {
      //
      // Profile the GEMM
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
      std::cout << "Non-fusion GEMM only time " << totalTime / (float)runs << " ms\n";
    }

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
      problem_size,
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
      problem_size,
      alpha1, 
      tensor_A0.device_ref(), 
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

    bool passed0 = cutlass::reference::host::TensorEquals(
      reference_D1.host_view(), 
      tensor_D1.host_view());
    CHECK_TRUE(passed0);

    bool passed1 = cutlass::reference::host::TensorEquals(
      reference_D1.host_view(), 
      tensor_D1.host_view());
    CHECK_TRUE(passed1);
    if (!passed0 || !passed1) {

      std::stringstream fname;

      fname << "error_DualGemm_device_nonfused.txt";
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
    return passed0 && passed1;
  }
};

template <typename DualGemm_>
struct DualFusedGemmRun
{

  using DualGemm = DualGemm_;
  using ElementAccumulator = typename DualGemm::ElementAccumulator;
  using ElementCompute = typename DualGemm::DualGemmKernel::Epilogue0::OutputOp::ElementCompute;
  using EpilogueOutputOp2 = typename DualGemm::EpilogueOutputOp2;

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

  DualFusedGemmRun(
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
    cutlass::gemm::GemmCoord problem_size,
    ElementCompute alpha0 = ElementCompute(1),
    ElementCompute beta0 = ElementCompute(1),
    ElementCompute alpha1 = ElementCompute(1),
    ElementCompute beta1 = ElementCompute(1),
    int batch_count = 1,
    bool broadcast_b1 = false,
    bool is_profiling = true,
    bool relu = false,
    int warm_ups = 1,
    int runs = 100) {
    
    //
    // Allocate the GEMM workspace
    //

    cutlass::HostTensor<
      typename DualGemm::ElementA,
      typename DualGemm::LayoutA> tensor_A0(
        cutlass::platform::is_same<typename DualGemm::LayoutA, cutlass::layout::RowMajor>::value ?
          cutlass::MatrixCoord(batch_count * problem_size.m(), problem_size.k()) :
          cutlass::MatrixCoord(problem_size.m(), batch_count * problem_size.k()));

    cutlass::HostTensor<
      typename DualGemm::ElementB,
      typename DualGemm::LayoutB0> tensor_B0(
        cutlass::platform::is_same<typename DualGemm::LayoutB0, cutlass::layout::RowMajor>::value ?
          cutlass::MatrixCoord(batch_count * problem_size.k(), problem_size.n()) :
          cutlass::MatrixCoord(problem_size.k(), batch_count * problem_size.n()));

    cutlass::HostTensor<
      typename DualGemm::ElementC,
      typename DualGemm::LayoutC> tensor_C0(
        cutlass::platform::is_same<typename DualGemm::LayoutC, cutlass::layout::RowMajor>::value ?
          cutlass::MatrixCoord(batch_count * problem_size.m(), problem_size.n()) :
          cutlass::MatrixCoord(problem_size.m(), batch_count * problem_size.n()));

    cutlass::HostTensor<
      typename DualGemm::ElementC,
      typename DualGemm::LayoutScaleBias> tensor_Bias0({batch_count, problem_size.n()});

    cutlass::HostTensor<
      typename DualGemm::ElementC,
      typename DualGemm::LayoutC> tensor_D0(
        cutlass::platform::is_same<typename DualGemm::LayoutC, cutlass::layout::RowMajor>::value ?
          cutlass::MatrixCoord(batch_count * problem_size.m(), problem_size.n()) :
          cutlass::MatrixCoord(problem_size.m(), batch_count * problem_size.n()));

    cutlass::HostTensor<
      typename DualGemm::ElementC,
      typename DualGemm::LayoutC> reference_D0(
        cutlass::platform::is_same<typename DualGemm::LayoutC, cutlass::layout::RowMajor>::value ?
          cutlass::MatrixCoord(batch_count * problem_size.m(), problem_size.n()) :
          cutlass::MatrixCoord(problem_size.m(), batch_count * problem_size.n()));

    cutlass::HostTensor<
      typename DualGemm::ElementB,
      typename DualGemm::LayoutB1> tensor_B1(
        cutlass::platform::is_same<typename DualGemm::LayoutB1, cutlass::layout::RowMajor>::value ?
          cutlass::MatrixCoord(batch_count * problem_size.k(), problem_size.n()) :
          cutlass::MatrixCoord(problem_size.k(), batch_count * problem_size.n()));
    if (broadcast_b1) {
      tensor_B1.resize({problem_size.k(), batch_count});
    }

    cutlass::HostTensor<
      typename DualGemm::ElementC,
      typename DualGemm::LayoutC> tensor_C1(
        cutlass::platform::is_same<typename DualGemm::LayoutC, cutlass::layout::RowMajor>::value ?
          cutlass::MatrixCoord(batch_count * problem_size.m(), problem_size.n()) :
          cutlass::MatrixCoord(problem_size.m(), batch_count * problem_size.n()));

    cutlass::HostTensor<
      typename DualGemm::ElementC,
      typename DualGemm::LayoutScaleBias> tensor_Bias1({batch_count, problem_size.n()});

    cutlass::HostTensor<
      typename DualGemm::ElementC,
      typename DualGemm::LayoutC> tensor_D1(
        cutlass::platform::is_same<typename DualGemm::LayoutC, cutlass::layout::RowMajor>::value ?
          cutlass::MatrixCoord(batch_count * problem_size.m(), problem_size.n()) :
          cutlass::MatrixCoord(problem_size.m(), batch_count * problem_size.n()));

    cutlass::HostTensor<
      typename DualGemm::ElementC,
      typename DualGemm::LayoutC> tensor_D2(
        cutlass::platform::is_same<typename DualGemm::LayoutC, cutlass::layout::RowMajor>::value ?
          cutlass::MatrixCoord(batch_count * problem_size.m(), problem_size.n()) :
          cutlass::MatrixCoord(problem_size.m(), batch_count * problem_size.n()));

    cutlass::HostTensor<
      typename DualGemm::ElementC,
      typename DualGemm::LayoutC> reference_D1(
        cutlass::platform::is_same<typename DualGemm::LayoutC, cutlass::layout::RowMajor>::value ?
          cutlass::MatrixCoord(batch_count * problem_size.m(), problem_size.n()) :
          cutlass::MatrixCoord(problem_size.m(), batch_count * problem_size.n()));

    cutlass::HostTensor<
      typename DualGemm::ElementC,
      typename DualGemm::LayoutC> reference_D2(
        cutlass::platform::is_same<typename DualGemm::LayoutC, cutlass::layout::RowMajor>::value ?
          cutlass::MatrixCoord(batch_count * problem_size.m(), problem_size.n()) :
          cutlass::MatrixCoord(problem_size.m(), batch_count * problem_size.n()));

    CHECK_TRUE(initialize_tensor(tensor_A0.host_view(), init_A, seed + 2019));
    CHECK_TRUE(initialize_tensor(tensor_B0.host_view(), init_B, seed + 2118));
    CHECK_TRUE(initialize_tensor(tensor_C0.host_view(), init_C, seed + 2017));
    CHECK_TRUE(initialize_tensor(tensor_Bias0.host_view(), init_Bias, seed + 2011));
    CHECK_TRUE(initialize_tensor(tensor_B1.host_view(), init_B, seed + 2113));
    CHECK_TRUE(initialize_tensor(tensor_C1.host_view(), init_C, seed + 2015));
    CHECK_TRUE(initialize_tensor(tensor_Bias1.host_view(), init_Bias, seed + 2012));

    cutlass::reference::host::TensorFill(
      tensor_D0.host_view());
    cutlass::reference::host::TensorFill(
      tensor_D1.host_view());
    cutlass::reference::host::TensorFill(
      tensor_D2.host_view());
    cutlass::reference::host::TensorFill(
      reference_D0.host_view());
    cutlass::reference::host::TensorFill(
      reference_D1.host_view());
    cutlass::reference::host::TensorFill(
      reference_D2.host_view());

    tensor_A0.sync_device();
    tensor_B0.sync_device();
    tensor_C0.sync_device();
    tensor_Bias0.sync_device();
    tensor_B1.sync_device();
    tensor_C1.sync_device();
    tensor_Bias1.sync_device();
    tensor_D0.sync_device();
    tensor_D1.sync_device();
    tensor_D2.sync_device();
    reference_D0.sync_device();
    reference_D1.sync_device();
    reference_D2.sync_device();

    //
    // Batch strides (irrelevant when batch_count == 1)
    //

    int64_t batch_stride_A = problem_size.m() * problem_size.k();
    int64_t batch_stride_B0 = problem_size.k() * problem_size.n();
    int64_t batch_stride_B1 = problem_size.k() * problem_size.n();
    if (broadcast_b1) {
      // B1 is a (column) vector
      batch_stride_B1 = problem_size.k();
    }
    int64_t batch_stride_Bias = problem_size.n();
    int64_t batch_stride_D = problem_size.m() * problem_size.n();

    //
    // Initialize the GEMM operator
    //

    int split_k_slices = DualGemm::kSplitKSerial ? 2 : 1;
    typename cutlass::TensorRef<typename DualGemm::ElementC, typename DualGemm::LayoutC> nullptr_ref{};
    decltype(nullptr_ref) ref_B0, ref_B1;
    if (beta0 != ElementCompute(0)) {
      ref_B0 = {tensor_Bias0.device_data(), typename DualGemm::LayoutC::Stride(0)};
    }
    if (beta1 != ElementCompute(0)) {
      ref_B1 = {tensor_Bias1.device_data(), typename DualGemm::LayoutC::Stride(0)};
    }
    typename DualGemm::Arguments arguments{
      (batch_count > 1 ?
        cutlass::gemm::DualGemmMode::kBatched :
        cutlass::gemm::DualGemmMode::kGemm),
      problem_size,
      tensor_A0.device_ref(),
      tensor_B0.device_ref(),
      ref_B0,
      DualGemm::kStoreD0 ? tensor_D0.device_ref() : nullptr_ref,
      (broadcast_b1 ?
        typename DualGemm::TensorRefB1(tensor_B1.device_data(), 0) :
        tensor_B1.device_ref()),
      ref_B1,
      DualGemm::kStoreD1 ? tensor_D1.device_ref() : nullptr_ref,
      tensor_D2.device_ref(),
      {alpha0, beta0},
      {alpha1, beta1},
      {},
      split_k_slices,
      batch_count,
      batch_stride_A,
      batch_stride_B0,
      batch_stride_B1,
      batch_stride_Bias,
      batch_stride_D,
    };

    //
    // Run the GEMM
    //

    DualGemm b2b_gemm_op;

    cutlass::device_memory::allocation<uint8_t> workspace(b2b_gemm_op.get_workspace_size(arguments));
  
    cutlass::Status status = b2b_gemm_op.can_implement(arguments);

    CUTLASS_CHECK(status);

    status = b2b_gemm_op.initialize(arguments, workspace.get());

    CUTLASS_CHECK(status);

    for(int i = 0; i < warm_ups; i++) {
        status = b2b_gemm_op();
        CUTLASS_CHECK(status);
    }

    if (is_profiling) {
      //
      // Profile the GEMM
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
    }

    tensor_D0.sync_host();
    tensor_D1.sync_host();
    tensor_D2.sync_host();

    //
    // Verify
    //

    using GemmUniversal0 = cutlass::gemm::device::GemmUniversal<
      typename DualGemm::ElementA, typename DualGemm::LayoutA,
      typename DualGemm::ElementB, typename DualGemm::LayoutB0,
      typename DualGemm::ElementC, typename DualGemm::LayoutC,
      ElementAccumulator
    >;

    GemmUniversal0 reference_gemm0;

    typename GemmUniversal0::Arguments args0 {
      (batch_count > 1 ?
        cutlass::gemm::GemmUniversalMode::kBatched :
        cutlass::gemm::GemmUniversalMode::kGemm),
      problem_size,
      batch_count,
      {alpha0, beta0},
      tensor_A0.device_data(),
      tensor_B0.device_data(),
      tensor_Bias0.device_data(),
      reference_D0.device_data(),
      batch_stride_A,
      batch_stride_B0,
      batch_stride_Bias,
      batch_stride_D,
      tensor_A0.stride(0),
      tensor_B0.stride(0),
      0,  // zero stride for the bias vector
      reference_D0.stride(0),
    };

    status = reference_gemm0.can_implement(args0);
    CUTLASS_CHECK(status);
    status = reference_gemm0(args0);
    CUTLASS_CHECK(status);

    using GemmUniversal1 = cutlass::gemm::device::GemmUniversal<
      typename DualGemm::ElementA, typename DualGemm::LayoutA,
      typename DualGemm::ElementB, typename DualGemm::LayoutB1,
      typename DualGemm::ElementC, typename DualGemm::LayoutC,
      ElementAccumulator
    >;

    GemmUniversal1 reference_gemm1;

    typename GemmUniversal1::Arguments args1 {
      (batch_count > 1 ?
        cutlass::gemm::GemmUniversalMode::kBatched :
        cutlass::gemm::GemmUniversalMode::kGemm),
      problem_size,
      batch_count,
      {alpha1, beta1},
      tensor_A0.device_data(),
      tensor_B1.device_data(),
      tensor_Bias1.device_data(),
      reference_D1.device_data(),
      batch_stride_A,
      batch_stride_B1,
      batch_stride_Bias,
      batch_stride_D,
      tensor_A0.stride(0),
      (broadcast_b1 ? 0 : tensor_B1.stride(0)),
      0,  // zero stride for the bias vector
      reference_D1.stride(0),
    };

    status = reference_gemm1.can_implement(args1);
    CUTLASS_CHECK(status);
    status = reference_gemm1(args1);
    CUTLASS_CHECK(status);

    if(relu) {
       cutlass::reference::device::TensorReLu(reference_D0.device_view());
       cutlass::reference::device::TensorReLu(reference_D1.device_view()); 
    }

    TensorEpilogueForEach<EpilogueOutputOp2>(reference_D0.device_view(), reference_D1.device_view(), reference_D2.device_view());
    cudaDeviceSynchronize();
    reference_D0.sync_host();
    reference_D1.sync_host();
    reference_D2.sync_host();

    CHECK_GT(cutlass::reference::host::TensorNorm(reference_D0.host_view()), 0);
    CHECK_GT(cutlass::reference::host::TensorNorm(reference_D1.host_view()), 0);
    CHECK_GT(cutlass::reference::host::TensorNorm(tensor_D2.host_view()), 0);
    CHECK_GT(cutlass::reference::host::TensorNorm(reference_D2.host_view()), 0);

    bool passed_out0 = true;
    if (DualGemm::kStoreD0) {
      CHECK_GT(cutlass::reference::host::TensorNorm(tensor_D0.host_view()), 0);
      passed_out0 = cutlass::reference::host::TensorEquals(
        reference_D0.host_view(), 
        tensor_D0.host_view());
    }
    CHECK_TRUE(passed_out0);

    bool passed_out1 = true;
    if (DualGemm::kStoreD1) {
      CHECK_GT(cutlass::reference::host::TensorNorm(tensor_D1.host_view()), 0);
      passed_out1 = cutlass::reference::host::TensorEquals(
        reference_D1.host_view(), 
        tensor_D1.host_view());
    }
    CHECK_TRUE(passed_out1);

    bool passed_out2 = cutlass::reference::host::TensorEquals(
      reference_D2.host_view(), 
      tensor_D2.host_view());
    CHECK_TRUE(passed_out2);

    bool passed = passed_out0 && passed_out1 && passed_out2;
    if (!passed)
    {
      std::stringstream fname;

      fname << "error_DualGemm_device_fused.txt";
      std::cerr << "Dumping results in " << fname.str() << "\n";

      std::ofstream file(fname.str());

      file 
        << "A0 =\n" << tensor_A0.host_view()
        << "\nB0 =\n" << tensor_B0.host_view()
        << "\nC0 =\n" << tensor_C0.host_view()
        << "\nBias0:\n" << tensor_Bias0.host_view() << "\n"
        << "\nB1 =\n" << tensor_B1.host_view()
        << "\nC1 =\n" << tensor_C1.host_view()
        << "\nBias1:\n" << tensor_Bias1.host_view() << "\n"
        << "\n\nReference0 =\n" << reference_D0.host_view()
        << "\nComputed0 =\n" << tensor_D0.host_view()
        << "\n\nReference1 =\n" << reference_D1.host_view()
        << "\nComputed1 =\n" << tensor_D1.host_view()
        << "\n\nReference2 =\n" << reference_D2.host_view()
        << "\nComputed2 =\n" << tensor_D2.host_view();
    }
    //std::cout << "A0 " << tensor_A0.host_view() << std::endl;
    // std::cout << "reference_D0 " << reference_D0.host_view() << std::endl;
    // std::cout << "reference_D1 " << reference_D1.host_view() << std::endl;
    // std::cout << "reference_D2 " << reference_D2.host_view() << std::endl;
    //std::cout << "reference_D0 " << reference_D0.host_view() << std::endl;
    return passed;
  }

};

////////////////////////////////////////////////////////////////////////////////
