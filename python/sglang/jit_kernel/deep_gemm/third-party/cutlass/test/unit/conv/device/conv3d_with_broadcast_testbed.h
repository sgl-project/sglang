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
    \brief Implicit GEMM for fused epilogue broadcast testbed

    Parallel split-k is not tested because we can just use regular conv kernel
    when we need to use parallel-splitk.  Broadcast can happen in the reduction
    kernel.
*/
#pragma once

#include <fstream>

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"

#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/reduction/device/reduce_split_k.h"
#include "cutlass/reduction/thread/reduction_operators.h"

#include "conv3d_problems.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_compare.h"

#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/reference/device/convolution.h"

#include "cutlass/core_io.h"
#include "cutlass/util/tensor_view_io.h"

#include "../cache_testbed_output.h"

namespace test {
namespace conv {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Conv3d>
struct Conv3dWithBroadcastReferenceOp {

  using OutputOp = typename Conv3d::EpilogueOutputOp;

  using ElementCompute = typename OutputOp::ElementCompute;
  using ElementZ = typename OutputOp::ElementZ;
  using ElementT = typename OutputOp::ElementT;

  typename OutputOp::BinaryOp binary_op;
  typename OutputOp::ElementwiseOp elementwise_op;

  Conv3dWithBroadcastReferenceOp() { }

  void operator()(ElementZ &Z, ElementT &T, ElementCompute conv3d, ElementCompute bias) {
    ElementCompute t_full = binary_op(conv3d, bias);
    T = ElementT(t_full);

    ElementCompute z_full = elementwise_op(t_full);
    Z = ElementZ(z_full);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Fused testbed
//
//  Y = CONV(AB, C)
//
//  T[n, o, p, q, k] = ReductionOp(Y[n, o, p, q, k], Broadcast[k])
//
//  Z[n, o, p, q, k] = Elementwise(T[n, o, p, q, k])
//

template <
  typename Conv3d,
  typename ReferenceOp,
  bool AddBroadcastFirst = false
>
class TestbedConv3dWithBroadcast {
public:

  using ElementA = typename Conv3d::ElementA;
  using LayoutA = typename Conv3d::LayoutA;
  using ElementB = typename Conv3d::ElementB;
  using LayoutB = typename Conv3d::LayoutB;
  using ElementC = typename Conv3d::ElementC;
  using LayoutC = typename Conv3d::LayoutC;
  using ElementAccumulator = typename Conv3d::ElementAccumulator;
  using ElementCompute = typename Conv3d::ElementCompute;
  using EpilogueOutputOp = typename Conv3d::EpilogueOutputOp;
  using ElementZ = typename EpilogueOutputOp::ElementZ;
  using ElementT = typename EpilogueOutputOp::ElementT;
  using ElementVector = typename EpilogueOutputOp::ElementVector;

  static cutlass::conv::Operator const kConvolutionalOperator = Conv3d::kConvolutionalOperator;
  static const bool kAddBroadcastFirst = AddBroadcastFirst;
  static const bool kStoreT = EpilogueOutputOp::kStoreT;

public:

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;

  cutlass::HostTensor<ElementA, LayoutA> tensor_A;
  cutlass::HostTensor<ElementB, LayoutB> tensor_B;
  cutlass::HostTensor<ElementC, LayoutC> tensor_C;
  cutlass::HostTensor<ElementAccumulator, LayoutC> tensor_C_reference;
  cutlass::HostTensor<ElementZ, LayoutC> tensor_Z_computed;
  cutlass::HostTensor<ElementZ, LayoutC> tensor_Z_reference;
  cutlass::HostTensor<ElementT, LayoutC> tensor_T_computed;
  cutlass::HostTensor<ElementT, LayoutC> tensor_T_reference;
  cutlass::HostTensor<ElementAccumulator, LayoutC> tensor_Y_reference;
  cutlass::HostTensor<ElementVector, LayoutC> tensor_Broadcast;            // Input Broadcast

public:

  TestbedConv3dWithBroadcast(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) {

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

      if (bits <= 8) {
        scope = 2;
      }
      else if (bits == 16) {
        if (cutlass::sizeof_bits<ElementAccumulator>::value <= 16) {
          scope = 3;
        }
        else {
          scope = 5;
        }
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
    else {
    }
  }

  void initialize(
    cutlass::conv::Conv3dProblemSize const &problem_size, bool non_packed_test = false, uint64_t seed = 2019) {
        
    // to make the layout of tensors a little bit bigger than the problem size
    cutlass::Tensor5DCoord stride_increment = cutlass::Tensor5DCoord(8, 16, 32, 32, 64);

    cutlass::Tensor5DCoord tensor_A_extent = implicit_gemm_tensor_a_extent(kConvolutionalOperator, problem_size);
    cutlass::Tensor5DCoord tensor_B_extent = implicit_gemm_tensor_b_extent(kConvolutionalOperator, problem_size);
    cutlass::Tensor5DCoord tensor_C_extent = implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size);

    if (non_packed_test) {
      tensor_A_extent += stride_increment;
      tensor_C_extent += stride_increment;
    }

    tensor_A.resize(tensor_A_extent);
    tensor_B.resize(tensor_B_extent);
    tensor_C.resize(tensor_C_extent);
    tensor_C_reference.resize(tensor_C_extent);
    tensor_Z_computed.resize(tensor_C_extent);
    tensor_Z_reference.resize(tensor_C_extent);
    tensor_T_computed.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size));
    tensor_T_reference.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size));
    tensor_Y_reference.resize(tensor_C_extent);
    tensor_Broadcast.resize({
      1,
      1,
      1,
      1,
      implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size).c(),
    });

    initialize_tensor(tensor_A.host_view(), init_A, seed); 
    initialize_tensor(tensor_B.host_view(), init_B, seed * 17); 
    initialize_tensor(tensor_C.host_view(), init_C, seed * 39);
    initialize_tensor(tensor_Broadcast.host_view(), init_C, seed * 39);
    for (int n = 0; n < tensor_C_reference.extent().n(); ++n) {
      for (int o = 0; o < tensor_C_reference.extent().d(); ++o) {
        for (int p = 0; p < tensor_C_reference.extent().h(); ++p) {
          for (int q = 0; q < tensor_C_reference.extent().w(); ++q) {
            for (int k = 0; k < tensor_C_reference.extent().c(); ++k) {
              tensor_C_reference.at({n, o, p, q, k}) = ElementAccumulator(tensor_C.at({n, o, p, q, k}));
            }
          }
        }
      }
    }
    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_Broadcast.sync_device();
    tensor_C_reference.sync_device();
    tensor_Z_computed.sync_device();
    tensor_Z_reference.sync_device();
    tensor_T_computed.sync_device();
    tensor_T_reference.sync_device();
    tensor_Y_reference.sync_device();
  }

  bool sufficient() const {
    //
    // Determine SMEM requirements and waive if not satisfied
    //

    size_t smem_size = sizeof(typename Conv3d::UnderlyingKernel::SharedStorage);

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
    cutlass::conv::Conv3dProblemSize const &problem_size,
    cutlass::conv::SplitKMode const &split_k_mode = cutlass::conv::SplitKMode::kSerial,
    bool non_packed_test = false,
    ElementCompute alpha = ElementCompute(1),
    ElementCompute beta = ElementCompute(1)) {

    // Waive test if insufficient CUDA device
    if (!sufficient()) {
      if (CUTLASS_TEST_UNIT_ENABLE_WARNINGS) {
        std::cerr << "Test waived due to insufficient CUDA device." << std::endl;
      }
      return true;
    }

#if 0 //display conv3d problem size for debugging
    std::cout << problem_size << std::endl
              << "alpha, beta: (" << alpha << ", " << beta << ")" << std::endl
              << "split_k_mode: " << ((split_k_mode == cutlass::conv::SplitKMode::kSerial) ? "(serial)" : "(parallel)") << std::endl
              << std::endl;
#endif

    initialize(problem_size, non_packed_test);

    // configure the operator
    Conv3d conv3d_op;
    typename Conv3d::Arguments conv3d_args(
      problem_size,
      tensor_A.device_ref(),
      tensor_B.device_ref(),
      tensor_C.device_ref(),
      tensor_Z_computed.device_ref(),
      {alpha, beta},
      split_k_mode,
      tensor_Broadcast.device_data(),
      kStoreT ? tensor_T_computed.device_data() : nullptr,
      0,         // This must be zero
      implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size).c()
    );

    // initialize the kernel 
    size_t workspace_size = Conv3d::get_workspace_size(conv3d_args);

    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = conv3d_op.initialize(conv3d_args, workspace.get());

    if (status != cutlass::Status::kSuccess) {
      cudaError_t error = cudaGetLastError();
      std::cerr << "This test is not supported: " << cudaGetErrorString(error) << "\n";
      return true;
    }

    // run conv3d operator
    status = conv3d_op();
    
    EXPECT_TRUE(status == cutlass::Status::kSuccess);
    if (status != cutlass::Status::kSuccess) {
      return false;
    }

    bool passed = false;

    cudaError_t result = cudaDeviceSynchronize();
    EXPECT_EQ(result, cudaSuccess) << " device reference error: " 
                                   << cudaGetErrorString(result);

    tensor_T_computed.sync_host();
    tensor_Z_computed.sync_host();

    //
    // Reference check
    //

    // When kAddBroadcastFirst is true, add bias on the host
    ElementCompute beta_ref = kAddBroadcastFirst ? ElementCompute(0) : beta;

#if CUTLASS_CONV_TEST_UNIT_REFERENCE_DEVICE_ENABLED

    cutlass::reference::device::Conv3d<
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementAccumulator,
      LayoutC,
      ElementAccumulator,
      ElementAccumulator 
    >(
      kConvolutionalOperator,
      problem_size,
      tensor_A.device_ref(),
      tensor_B.device_ref(),
      tensor_C_reference.device_ref(),
      tensor_Y_reference.device_ref(),
      alpha, 
      beta_ref);

    // sync host (copy device data to host) for dumping error output in case of mismatches
    tensor_Y_reference.sync_host();
    
#else 

    cutlass::reference::host::Conv3d<
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementAccumulator,
      LayoutC,
      ElementAccumulator,
      ElementAccumulator
    >(
      kConvolutionalOperator,
      problem_size,
      tensor_A.host_ref(),
      tensor_B.host_ref(),
      tensor_C_reference.host_ref(),
      tensor_Y_reference.host_ref(),
      alpha, 
      beta_ref);

#endif
    ReferenceOp reference_op;

    // compute tensor Z and tensor T
    for (int n = 0; n < problem_size.N; ++n) {
      for (int o = 0; o < (kConvolutionalOperator == cutlass::conv::Operator::kFprop ? problem_size.Z : problem_size.D); ++o) {
        for (int p = 0; p < (kConvolutionalOperator == cutlass::conv::Operator::kFprop ? problem_size.P : problem_size.H); ++p) {
          for (int q = 0; q < (kConvolutionalOperator == cutlass::conv::Operator::kFprop ? problem_size.Q : problem_size.W); ++q) {
            for (int k = 0; k < (kConvolutionalOperator == cutlass::conv::Operator::kFprop ? problem_size.K : problem_size.C); ++k) {
    
              ElementZ z{};
              ElementT t{};
      
              ElementCompute accum = tensor_Y_reference.at({n, o, p, q, k});
              ElementCompute bias = ElementCompute(tensor_Broadcast.at({0, 0, 0, 0, k}));


              if (kAddBroadcastFirst) {
                reference_op(z, t, accum + bias,
                            beta * ElementCompute(tensor_C_reference.at({n, o, p, q, k})));
              } else {
                reference_op(z, t, accum, bias);
              }   
  
              tensor_Z_reference.at({n, o, p, q, k}) = z;
              tensor_T_reference.at({n, o, p, q, k}) = t;
            }
          }
        }
      }
    }

    if (kStoreT) {
      passed = cutlass::reference::host::TensorEquals(
        tensor_T_computed.host_view(), 
        tensor_T_reference.host_view());

      EXPECT_TRUE(passed);
    }

    passed = cutlass::reference::host::TensorEquals(
      tensor_Z_computed.host_view(), 
      tensor_Z_reference.host_view());

    EXPECT_TRUE(passed);

    if (!passed) {
      std::stringstream fname;

      fname << "error_Conv3d_ImplicitGemm_device_"
        << (split_k_mode == cutlass::conv::SplitKMode::kSerial ? "serial_reduction_" : "parallel_reduction_")
        << (Conv3d::kConvolutionalOperator == cutlass::conv::Operator::kFprop ? "fprop_" :
            (Conv3d::kConvolutionalOperator == cutlass::conv::Operator::kDgrad ? "dgrad_" :
              (Conv3d::kConvolutionalOperator == cutlass::conv::Operator::kDeconv ? "deconv_" : "wgrad_")))
        << "nnhwc_"
        << problem_size.N << "x"
        << problem_size.D << "x"
        << problem_size.H << "x"
        << problem_size.W << "x"
        << problem_size.C 
        << "_krsc_"
        << problem_size.K << "x"
        << problem_size.T << "x"
        << problem_size.R << "x"
        << problem_size.S << "x"
        << problem_size.C 
        << "_padding_"
        << problem_size.pad_d << "x"
        << problem_size.pad_h << "x"
        << problem_size.pad_w 
        << "_stride_"
        << problem_size.stride_d << "x"
        << problem_size.stride_h << "x"
        << problem_size.stride_w 
        << "_dilation_"
        << problem_size.dilation_d << "x"
        << problem_size.dilation_h << "x"
        << problem_size.dilation_w << "_"
        << (problem_size.mode == cutlass::conv::Mode::kCrossCorrelation ? "xcorr_" : "conv_")
        << (non_packed_test ? "non_packed_tensor_test_" : "packed_tensor_test_")
        << Conv3d::ThreadblockShape::kM << "x"  
        << Conv3d::ThreadblockShape::kN << "x"  
        << Conv3d::ThreadblockShape::kK << "_"
        << Conv3d::WarpShape::kM << "x"  
        << Conv3d::WarpShape::kN << "x"  
        << Conv3d::WarpShape::kK << ".txt";

      std::cout << fname.str() << std::endl;

      std::ofstream results(fname.str());

      results << problem_size << std::endl;

      results
        << "\nA:\n" << tensor_A.host_view() << "\n"
        << "\nB:\n" << tensor_B.host_view() << "\n"
        << "\nC:\n" << tensor_C.host_view() << "\n"
        << "\nBroadcast:\n" << tensor_Broadcast.host_view() << "\n"
        << "\nY reference:\n" << tensor_Y_reference.host_view() << "\n"
        << "\nT reference:\n" << tensor_T_reference.host_view() << "\n"
        << "\nT computed:\n" << tensor_T_computed.host_view() << "\n"
        << "\nZ reference:\n" << tensor_Z_reference.host_view() << "\n"
        << "\nZ computed:\n" << tensor_Z_computed.host_view() << "\n";
    }

    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// TestAllConv: Runs cutlass::conv::device::ImplicitGemmConvolution operator and compares it with reference
// TestAllConv runs conv operator on default conv problem sizes from test::conv::device::TestbedConv3dProblemSizes
// Additionally, each conv3d test can provide conv problem sizes (conv_test_sizes) and blacklist of sizes 
// (conv_blacklist_sizes)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename ImplicitGemm,
          typename ReferenceOp = Conv3dWithBroadcastReferenceOp<ImplicitGemm>,
          bool AddBroadcastFirst = false,
          bool TestSplitK = true 
>
bool TestAllConv3dWithBroadcast(
  const Conv3dProblemVector &conv_test_sizes = Conv3dProblemVector(),
  const Conv3dProblemVector &conv_blacklist_sizes = Conv3dProblemVector(),
  bool non_packed_test = false) {

  bool passed = true;

  //
  // Testbed object
  //

  TestbedConv3dWithBroadcast<ImplicitGemm, ReferenceOp, AddBroadcastFirst> testbed;

  //
  // Get conv problem sizes to run conv operator 
  //
  TestbedConv3dProblemSizes conv3d_problems(128/cutlass::sizeof_bits<typename ImplicitGemm::ElementA>::value);

  // Vector of conv3d problem sizes to avoid duplicate runs
  Conv3dProblemVector conv_tested_sizes;

  Conv3dProblemVector const *problem_vectors[] = {
    &conv3d_problems.conv3d_default_sizes,
    &conv3d_problems.conv3d_vnet_medical_sizes,
    &conv_test_sizes
  };

  // Sweep conv3d problem sizes (split-k-mode=kSerial, split-k-slice=1, alpha=1.0, beta=0.0)
  for (Conv3dProblemVector const * problem_vector : problem_vectors) {

    //  Run conv testbed on default convolution sizes
    for(auto conv_problem : *problem_vector) {

      // Skip blacklist and avoid duplicate problem sizes
      if (std::find(conv_blacklist_sizes.begin(), conv_blacklist_sizes.end(), conv_problem) != conv_blacklist_sizes.end() ||
          std::find(conv_tested_sizes.begin(), conv_tested_sizes.end(), conv_problem) != conv_tested_sizes.end()) {
        continue;
      }

      //
      // Procedurally disable certain cases
      //
  
      // CUTLASS DGRAD's *unity* stride specialization only support stride {1, 1} 
      if ((ImplicitGemm::kConvolutionalOperator == cutlass::conv::Operator::kDgrad ||
            ImplicitGemm::kConvolutionalOperator == cutlass::conv::Operator::kDeconv) && 
          (ImplicitGemm::UnderlyingKernel::Mma::IteratorA::kStrideSupport == 
            cutlass::conv::StrideSupport::kUnity)) {
        if (!((conv_problem.stride_d == 1) &&
              (conv_problem.stride_h == 1) && 
              (conv_problem.stride_w == 1))
          ) {
          continue;
        }
      }

#if 0 // relax restrictions on analytic strided dgrad
      // CUTLASS DGRAD's *strided* specialization only support stride >= {2, 2} 
      if ((ImplicitGemm::kConvolutionalOperator == cutlass::conv::Operator::kDgrad ||
            ImplicitGemm::kConvolutionalOperator == cutlass::conv::Operator::kDeconv) && 
          (ImplicitGemm::UnderlyingKernel::Mma::IteratorA::kStrideSupport == 
            cutlass::conv::StrideSupport::kStrided)) {
         if (((conv_problem.stride_d == 1) && (conv_problem.stride_h == 1) && (conv_problem.stride_w == 1))) {
           continue;
         }
      }
#endif
      
      //
      // Test
      //
      // push back tested problem size to avoid re-running duplicates
      conv_tested_sizes.push_back(conv_problem);

      // test mode = xcross
      passed = testbed.run(
        conv_problem,
        cutlass::conv::SplitKMode::kSerial, non_packed_test);

      if (!passed) {
        return false;
      }

      // test mode = convolution
      passed = testbed.run(
        conv_problem.reset_mode(cutlass::conv::Mode::kConvolution),
        cutlass::conv::SplitKMode::kSerial, non_packed_test);

      if (!passed) {
        return false;
      }
    }
  }

  if (!TestSplitK)
    return passed;

  // Sweep split-k-slice using serial and prallel reduction with non-unity alpha and non-zero beta for 
  // a single conv3d problem size. Convolution unit tests take a long time to run so only sweep parameters 
  // which are abolutely necessary to catch functional bugs. The below code does provide option to sweep
  // alpha and beta for local testing, but only runs one value for alpha and beta.
  cutlass::conv::Conv3dProblemSize conv3d_split_k_test_size (
    {1, 8, 8, 8, 32},               // input size  (NDHWC)
    {32, 3, 3, 3, 32},              // filter size (KTRSC)
    cutlass::Coord<3>({0, 0, 0}),   // padding (pad_d, pad_h, pad_w)
    cutlass::Coord<3>({1, 1, 1}),   // stride (stride_d, stride_h, stride_w)
    cutlass::Coord<3>({1, 1, 1})    // dilation (dilation_d, dilation_h, dilation_w) 
  );

  cutlass::conv::SplitKMode split_k_modes [] = {
    cutlass::conv::SplitKMode::kSerial
  };

  int split_k_slices[] = {
    1, 2, 3, 4, 201
  };

  double problem_alpha[] = {
    2.0
  };

  double problem_beta[] = {
    2.0
  };

  for (auto split_k_mode : split_k_modes) {
    for (auto split_k_slice : split_k_slices) {
      for (auto alpha : problem_alpha) {
        for (auto beta : problem_beta) {

          passed = testbed.run(
            conv3d_split_k_test_size.reset_split_k_slices(split_k_slice),
            split_k_mode,
            false,/*non_packed_test*/
            cutlass::from_real<typename ImplicitGemm::ElementCompute>(alpha), 
            cutlass::from_real<typename ImplicitGemm::ElementCompute>(beta));

          if (!passed) {
            return false;
          }
        }
      }
    }
  }

  return passed;
}

template <typename ImplicitGemm,
          typename ReferenceOp = Conv3dWithBroadcastReferenceOp<ImplicitGemm>,
          bool AddBroadcastFirst = false>
bool TestSpecificConv3dWithBroadcast(
  const Conv3dProblemVector & problem_sizes,
  bool non_packed_test = false) {

  bool passed = true;

  //
  // Testbed object
  //

  TestbedConv3dWithBroadcast<ImplicitGemm, ReferenceOp, AddBroadcastFirst> testbed;

  // Sweep conv3d problem sizes (split-k-mode=kSerial, split-k-slice=1, alpha=1.0, beta=0.0)
  for(auto conv_problem : problem_sizes) {

    //
    // Test
    //

    // test mode = xcross, non_packed_test = false
    passed = testbed.run(
      conv_problem,
      cutlass::conv::SplitKMode::kSerial, non_packed_test);

    if (!passed) {
      return false;
    }

    // test mode = convolution, non_packed_test = false
    passed = testbed.run(
      conv_problem.reset_mode(cutlass::conv::Mode::kConvolution),
      cutlass::conv::SplitKMode::kSerial, non_packed_test);

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
