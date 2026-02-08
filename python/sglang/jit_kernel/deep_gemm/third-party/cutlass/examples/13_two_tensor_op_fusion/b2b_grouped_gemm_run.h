/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Containers for running grouped back-to-back GEMMs
*/

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/util/device_memory.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/device/gemm.h"
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

template <typename B2bGemm_>
struct B2bFusedGroupedGemmRun
{

  using B2bGemm = B2bGemm_;
  using ElementAccumulator = typename B2bGemm::ElementAccumulator;
  using ElementCompute = typename B2bGemm::BaseKernel::Epilogue::OutputOp::ElementCompute;

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

  B2bFusedGroupedGemmRun(
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
        view, seed, 1, -1, 0);
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
    std::vector<cutlass::gemm::GemmCoord> problem_sizes_0,
    std::vector<cutlass::gemm::GemmCoord> problem_sizes_1,
    ElementCompute alpha0 = ElementCompute(1),
    ElementCompute beta0 = ElementCompute(0),
    ElementCompute alpha1 = ElementCompute(1),
    ElementCompute beta1 = ElementCompute(0),
    bool relu = true,
    int warm_ups = 1,
    int runs = 100) {

    using HostTensorA = cutlass::HostTensor<typename B2bGemm::ElementA, typename B2bGemm::LayoutA>;
    using HostTensorB = cutlass::HostTensor<typename B2bGemm::ElementB, typename B2bGemm::LayoutB>;
    using HostTensorC = cutlass::HostTensor<typename B2bGemm::ElementC, typename B2bGemm::LayoutC>;
    using HostTensorScale = cutlass::HostTensor<ElementCompute, typename B2bGemm::LayoutC>;
    using HostTensorZ = cutlass::HostTensor<ElementAccumulator, typename B2bGemm::LayoutC>;
    using HostTensorBias = cutlass::HostTensor<ElementCompute, typename B2bGemm::LayoutC>;

    int problem_count = (int)problem_sizes_0.size();

    std::vector<HostTensorA> host_tensor_A0(problem_count);
    std::vector<HostTensorB> host_tensor_B0(problem_count);
    std::vector<HostTensorC> host_tensor_C0(problem_count);
    std::vector<HostTensorScale> host_tensor_Scale0(problem_count);
    std::vector<HostTensorScale> host_tensor_Bias0(problem_count);
    std::vector<HostTensorB> host_tensor_B1(problem_count);
    std::vector<HostTensorC> host_tensor_C1(problem_count);
    std::vector<HostTensorBias> host_tensor_Bias1(problem_count);
    std::vector<HostTensorC> host_tensor_D1(problem_count);
    std::vector<HostTensorZ> host_tensor_Z(problem_count);
    std::vector<HostTensorC> host_tensor_ref_D0(problem_count);
    std::vector<HostTensorC> host_tensor_ref_D1(problem_count);

    std::vector<typename HostTensorA::TensorRef> ref_A0(problem_count);
    std::vector<typename HostTensorB::TensorRef> ref_B0(problem_count);
    std::vector<typename HostTensorC::TensorRef> ref_C0(problem_count);
    std::vector<typename HostTensorScale::TensorRef> ref_Scale0(problem_count);
    std::vector<typename HostTensorScale::TensorRef> ref_Bias0(problem_count);
    std::vector<typename HostTensorB::TensorRef> ref_B1(problem_count);
    std::vector<typename HostTensorC::TensorRef> ref_C1(problem_count);
    std::vector<typename HostTensorBias::TensorRef> ref_Bias1(problem_count);
    std::vector<typename HostTensorC::TensorRef> ref_D1(problem_count);
    std::vector<typename HostTensorZ::TensorRef> ref_Z(problem_count);
    std::vector<typename HostTensorC::TensorRef> ref_ref_D0(problem_count);
    std::vector<typename HostTensorC::TensorRef> ref_ref_D1(problem_count);

    for (int i = 0; i < problem_count; ++i) {
      //
      // Allocate the GEMM workspace
      //

      auto problem_size_0 = problem_sizes_0[i];
      auto problem_size_1 = problem_sizes_1[i];

      host_tensor_A0.at(i) = HostTensorA(problem_size_0.mk());
      host_tensor_B0.at(i) = HostTensorB(problem_size_0.kn());
      host_tensor_C0.at(i) = HostTensorC(problem_size_0.mn());
      if (alpha0 == ElementCompute(0)) //per-channel scale
        host_tensor_Scale0.at(i) = HostTensorScale(typename HostTensorZ::Layout::TensorCoord{1, problem_size_0.n()});
      host_tensor_Bias0.at(i) = HostTensorScale(typename HostTensorBias::Layout::TensorCoord{1, problem_size_0.n()});
      host_tensor_Z.at(i) = HostTensorZ(problem_size_0.mn());
      host_tensor_ref_D0.at(i) = HostTensorC(problem_size_0.mn());
      host_tensor_B1.at(i) = HostTensorB(problem_size_1.kn());
      host_tensor_C1.at(i) = HostTensorC(problem_size_1.mn());
      host_tensor_Bias1.at(i) = HostTensorScale(typename HostTensorBias::Layout::TensorCoord{1, problem_size_1.n()});
      host_tensor_D1.at(i) = HostTensorC(problem_size_1.mn());
      host_tensor_ref_D1.at(i) = HostTensorC(problem_size_1.mn());

      CHECK_TRUE(initialize_tensor(host_tensor_A0.at(i).host_view(), init_A, seed + 2019));
      CHECK_TRUE(initialize_tensor(host_tensor_B0.at(i).host_view(), init_B, seed + 2018));
      CHECK_TRUE(initialize_tensor(host_tensor_C0.at(i).host_view(), init_C, seed + 2017));
      if (alpha0 == ElementCompute(0)) //per-channel scale
        CHECK_TRUE(initialize_tensor(host_tensor_Scale0.at(i).host_view(), init_Scale, seed + 2014));
      CHECK_TRUE(initialize_tensor(host_tensor_Bias0.at(i).host_view(), init_Bias, seed + 2013));
      CHECK_TRUE(initialize_tensor(host_tensor_B1.at(i).host_view(), init_B, seed + 2016));
      CHECK_TRUE(initialize_tensor(host_tensor_C1.at(i).host_view(), init_C, seed + 2015));
      CHECK_TRUE(initialize_tensor(host_tensor_Bias1.at(i).host_view(), init_Bias, seed + 2012));

      cutlass::reference::host::TensorFill(
        host_tensor_D1.at(i).host_view());
      cutlass::reference::host::TensorFill(
        host_tensor_ref_D0.at(i).host_view());
      cutlass::reference::host::TensorFill(
        host_tensor_ref_D1.at(i).host_view());

      host_tensor_A0.at(i).sync_device();
      host_tensor_B0.at(i).sync_device();
      host_tensor_C0.at(i).sync_device();
      if (alpha0 == ElementCompute(0)) //per-channel scale
        host_tensor_Scale0.at(i).sync_device();
      host_tensor_Bias0.at(i).sync_device();
      host_tensor_B1.at(i).sync_device();
      host_tensor_C1.at(i).sync_device();
      host_tensor_Bias1.at(i).sync_device();
      host_tensor_D1.at(i).sync_device();
      host_tensor_ref_D0.at(i).sync_device();
      host_tensor_ref_D1.at(i).sync_device();

      ref_A0.at(i) = (host_tensor_A0.at(i).device_ref());
      ref_B0.at(i) = (host_tensor_B0.at(i).device_ref());
      ref_C0.at(i) = (host_tensor_C0.at(i).device_ref());
      if (alpha0 == ElementCompute(0)) //per-channel scale
        ref_Scale0.at(i) = (host_tensor_Scale0.at(i).device_ref());
      ref_Bias0.at(i) = (host_tensor_Bias0.at(i).device_ref());
      ref_B1.at(i) = (host_tensor_B1.at(i).device_ref());
      ref_C1.at(i) = {host_tensor_Bias1.at(i).device_data(), typename B2bGemm::LayoutC::Stride(0)};
      ref_Bias1.at(i) = (host_tensor_Bias1.at(i).device_ref());
      ref_D1.at(i) = (host_tensor_D1.at(i).device_ref());
      ref_Z.at(i) = (host_tensor_Z.at(i).device_ref());
      ref_ref_D0.at(i) = (host_tensor_ref_D0.at(i).device_ref());
      ref_ref_D1.at(i) = (host_tensor_ref_D1.at(i).device_ref());
    }

    //
    // Initialize the GEMM operator
    //

    cutlass::DeviceAllocation<typename HostTensorA::TensorRef> device_ref_A0(problem_count);
    device_ref_A0.copy_from_host(ref_A0.data());
    cutlass::DeviceAllocation<typename HostTensorB::TensorRef> device_ref_B0(problem_count);
    device_ref_B0.copy_from_host(ref_B0.data());
    cutlass::DeviceAllocation<typename HostTensorC::TensorRef> device_ref_C0(problem_count);
    device_ref_C0.copy_from_host(ref_C0.data());
    cutlass::DeviceAllocation<typename HostTensorScale::TensorRef> device_ref_Scale0(problem_count);
    device_ref_Scale0.copy_from_host(ref_Scale0.data());
    cutlass::DeviceAllocation<typename HostTensorScale::TensorRef> device_ref_Bias0(problem_count);
    device_ref_Bias0.copy_from_host(ref_Bias0.data());
    cutlass::DeviceAllocation<typename HostTensorB::TensorRef> device_ref_B1(problem_count);
    device_ref_B1.copy_from_host(ref_B1.data());
    cutlass::DeviceAllocation<typename HostTensorC::TensorRef> device_ref_C1(problem_count);
    device_ref_C1.copy_from_host(ref_C1.data());
    cutlass::DeviceAllocation<typename HostTensorBias::TensorRef> device_ref_Bias1(problem_count);
    device_ref_Bias1.copy_from_host(ref_Bias1.data());
    cutlass::DeviceAllocation<typename HostTensorC::TensorRef> device_ref_D1(problem_count);
    device_ref_D1.copy_from_host(ref_D1.data());

    cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> device_problem_sizes_0(problem_count);
    device_problem_sizes_0.copy_from_host(problem_sizes_0.data());
    cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> device_problem_sizes_1(problem_count);
    device_problem_sizes_1.copy_from_host(problem_sizes_1.data());

    B2bGemm b2b_gemm_op;

    int threadblock_count = B2bGemm::sufficient(problem_sizes_1.data(), problem_count);
    if (!threadblock_count) {
      std::cout << "Active CUDA device lacks hardware resources to run CUTLASS Grouped GEMM kernel." << std::endl;
      return false;
    }

    typename B2bGemm::Arguments arguments{
      problem_count,
      device_problem_sizes_0.get(),
      device_problem_sizes_1.get(),
      device_ref_A0.get(),
      device_ref_B0.get(),
      device_ref_C0.get(),
      device_ref_Scale0.get(),
      device_ref_Bias0.get(),
      device_ref_B1.get(),
      device_ref_C1.get(),
      device_ref_D1.get(),
      {alpha0, beta0},
      {alpha1, beta1},
      threadblock_count
    };

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

    for (int i = 0; i < problem_count; ++i) {
      host_tensor_D1.at(i).sync_host();

      //
      // Verify
      //

      cutlass::reference::device::Gemm<
          typename B2bGemm::ElementA, typename B2bGemm::LayoutA,
          typename B2bGemm::ElementB, typename B2bGemm::LayoutB,
          ElementAccumulator, typename B2bGemm::LayoutC, 
          ElementAccumulator, ElementAccumulator>
          reference_gemm_0;

      cutlass::reference::device::Gemm<
          typename B2bGemm::ElementA, typename B2bGemm::LayoutA,
          typename B2bGemm::ElementB, typename B2bGemm::LayoutB,
          typename B2bGemm::ElementC, typename B2bGemm::LayoutC, ElementCompute,
          ElementAccumulator>
          reference_gemm_1;

      auto problem_size_0 = problem_sizes_0[i];
      auto problem_size_1 = problem_sizes_1[i];

      reference_gemm_0(
        problem_size_0,
        ElementAccumulator(1), //intermediate alpha=1
        ref_A0.at(i), 
        ref_B0.at(i), 
        ElementAccumulator(0), //beta = 0
        ref_Z.at(i),
        ref_Z.at(i),
        ElementAccumulator(0)
      );

      cutlass::reference::device::TensorScaleBiasGemm<
        ElementAccumulator, typename B2bGemm::ElementC, typename B2bGemm::LayoutC,
        ElementCompute, typename B2bGemm::LayoutC
      > (
        problem_size_0,
        ref_Z.at(i),
        ref_ref_D0.at(i),
        alpha0,
        ref_Scale0.at(i),
        ref_Bias0.at(i)
      );

      if(relu) {
        cutlass::reference::device::TensorReLu(host_tensor_ref_D0.at(i).device_view()); 
      }

      reference_gemm_1(
        problem_size_1,
        alpha1, 
        ref_ref_D0.at(i), 
        ref_B1.at(i), 
        beta1, 
        {host_tensor_Bias1.at(i).device_data(), typename B2bGemm::LayoutC::Stride(0)},
        ref_ref_D1.at(i)
      );
      if(relu) {
        cutlass::reference::device::TensorReLu(host_tensor_ref_D1.at(i).device_view()); 
      }
      cudaDeviceSynchronize();
      host_tensor_ref_D0.at(i).sync_host();
      host_tensor_ref_D1.at(i).sync_host();

      CHECK_GT(cutlass::reference::host::TensorNorm(host_tensor_ref_D0.at(i).host_view()), 0);
      CHECK_GT(cutlass::reference::host::TensorNorm(host_tensor_D1.at(i).host_view()), 0);
      CHECK_GT(cutlass::reference::host::TensorNorm(host_tensor_ref_D1.at(i).host_view()), 0);

      bool passed = cutlass::reference::host::TensorEquals(
        host_tensor_ref_D1.at(i).host_view(), 
        host_tensor_D1.at(i).host_view());

      CHECK_TRUE(passed);
      if (!passed)
      {

        std::stringstream fname;

        fname << "error_B2bGemm_device_fused.txt";
        std::cerr << "Check failed for GEMM " << i << " in the group." << std::endl;
        std::cerr << "Dumping results in " << fname.str() << "\n";

        std::ofstream file(fname.str());

        file 
          << "GEMM " << i << " in group\n"
          << "A0 =\n" << host_tensor_A0.at(i).host_view()
          << "\nB0 =\n" << host_tensor_B0.at(i).host_view()
          << "\nC0 =\n" << host_tensor_C0.at(i).host_view()
          << "\nScale0:\n" << host_tensor_Scale0.at(i).host_view() << "\n"
          << "\nBias0:\n" << host_tensor_Bias0.at(i).host_view() << "\n"
          << "\nB1 =\n" << host_tensor_B1.at(i).host_view()
          << "\nC1 =\n" << host_tensor_C1.at(i).host_view()
          << "\nBias1:\n" << host_tensor_Bias1.at(i).host_view() << "\n"
          << "\n\nReference =\n" << host_tensor_ref_D1.at(i).host_view()
          << "\nComputed =\n" << host_tensor_D1.at(i).host_view();

        return false;
      }
    }
    return true;
  }

};

////////////////////////////////////////////////////////////////////////////////
