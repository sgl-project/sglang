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
    \brief Implicit GEMM testbed for 3.x API
*/
#pragma once

#include "cutlass/cutlass.h"
#include "../../common/cutlass_unit_test.h"

#include "cute/tensor.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/convnd_problem_shape.hpp"
#include "../test/unit/gemm/device/gemm_testbed_3x.hpp"

#include "thrust/universal_vector.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/conv.hpp"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "conv_problem_sizes.hpp"
#include "../cache_testbed_output.h"

#include <iostream>

#include "cute/layout.hpp"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test::conv::device {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Initializes a flat device buffer
template <typename Element>
static void
initialize_values(
    thrust::universal_vector<Element>& dst_ptr,
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {
  if (cutlass::Distribution::Uniform == dist_kind) {
    int scope;
    int bits = cutlass::sizeof_bits<Element>::value;

    if (bits <= 8) {
      scope = 2;
    }
    else if (bits == 16) {
      scope = 4;
    }
    else {
      scope = 8;
    }
    cutlass::reference::host::BlockFillRandomUniform(
        dst_ptr.data().get(), dst_ptr.size(), seed, scope, -scope, 0);
  }
  else if (cutlass::Distribution::Identity == dist_kind) {
    cutlass::reference::host::BlockFillRandomUniform(
        dst_ptr.data().get(), dst_ptr.size(), seed, 0, 0, 0);
  }
  else if (cutlass::Distribution::Gaussian == dist_kind) {
    cutlass::reference::host::BlockFillRandomGaussian(dst_ptr.data().get(), dst_ptr.size(), seed, 0, 0.5);
  }
  else if (cutlass::Distribution::Sequential == dist_kind) {
    cutlass::reference::host::BlockFillSequential(dst_ptr.data().get(), dst_ptr.size());
  }
  else {
    std::cerr << "Invalid distribution kind!\n.";
    exit(1);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// utils for sparse or dense conv parameters

template <class Conv>
struct DenseConvParams {
  // Default Kernel data types
  using ElementA = typename Conv::ConvKernel::ElementA;
  using ElementB = typename Conv::ConvKernel::ElementB;

  static constexpr cutlass::conv::Operator ConvOp = Conv::DispatchPolicy::ConvOp;
  static constexpr int NumSpatialDimensions = Conv::NumSpatialDimensions;
  using ProblemShape = cutlass::conv::ConvProblemShape<ConvOp, NumSpatialDimensions>;

  // get the default arguments without sparse data
  auto get_mainloop_arguments(
    [[maybe_unused]] ProblemShape const& problem_shape,
    thrust::universal_vector<ElementA>& tensor_A,
    thrust::universal_vector<ElementB>& tensor_B
  ) {
    auto args = typename Conv::ConvKernel::MainloopArguments {
      tensor_A.data().get(),
      tensor_B.data().get(),
    };
    return args;
  }
};

template <class Conv>
struct SparseConvParams {
};

/////////////////////////////////////////////////////////////////////////////////////////////////
template <class Conv, bool isSparseEnabled_ = false>
struct ConvTestbed {
  // Kernel data types
  using ElementA = typename Conv::ConvKernel::ElementA;
  using ElementB = typename Conv::ConvKernel::ElementB;
  using ElementC = cute::conditional_t<cute::is_void_v<typename Conv::ConvKernel::ElementC>,
      typename Conv::ConvKernel::ElementD, typename Conv::ConvKernel::ElementC>;
  using ElementD = typename Conv::ConvKernel::ElementD;
  using ElementAccumulator = typename Conv::ConvKernel::ElementAccumulator;

  // ConvTest for sparse kernel
  static constexpr bool isSparseEnabled = isSparseEnabled_;
  using ConvParams = cute::conditional_t<isSparseEnabled, SparseConvParams<Conv>, DenseConvParams<Conv>>;
  ConvParams params;

  //
  // FusionOperation derived types/queries
  //
  using FusionOp = typename Conv::EpilogueOutputOp;

  // fusion types are potentially void if the fusion is not supported
  // helper so we don't try to construct HostTensor with void type
  template <typename T, typename U = uint8_t>
  using non_void_t               = cute::conditional_t<cute::is_void_v<T>, U, T>;
  using ElementScalar            = typename FusionOp::ElementScalar;
  using ElementCompute           = typename FusionOp::ElementCompute;
  using BiasType                 = typename cutlass::epilogue::collective::detail::IsThreadEpilogueOpWithBias<FusionOp>::type;
  using ElementBias              = non_void_t<BiasType>;
  using ActivationType           = non_void_t<typename cutlass::epilogue::collective::detail::IsThreadEpilogueOpWithActivation<FusionOp>::type,
                                   cutlass::epilogue::thread::Identity<ElementCompute>>;
  static constexpr bool IsActivationEnabled = cutlass::epilogue::collective::detail::IsThreadEpilogueOpWithActivation<FusionOp>::value;
  using ActivationFunctor        = cute::conditional_t<IsActivationEnabled, ActivationType, cutlass::epilogue::thread::Identity<ElementCompute>>;

  static constexpr bool IsBiasEnabled = cutlass::epilogue::collective::detail::IsThreadEpilogueOpWithBias<FusionOp>::value &&
                                        !cute::is_same_v<BiasType, void>;
  static constexpr bool IsPerChannelScaleEnabled = cutlass::epilogue::collective::detail::IsThreadEpilogueOpWithPerChannelScaled<FusionOp>::value;

  static constexpr bool DisableSource = cute::is_void_v<typename FusionOp::ElementSource>;

  static constexpr bool IsResidualEnabled = cutlass::epilogue::collective::detail::IsThreadEpilogueOpWithResidualAdd<FusionOp>::value;

  using StrideC  = typename Conv::ConvKernel::StrideC;
  using StrideD  = typename Conv::ConvKernel::StrideD;
  using ThreadEpilogueOp = typename Conv::ConvKernel::CollectiveEpilogue::ThreadEpilogueOp;

  static constexpr cutlass::conv::Operator ConvOp = Conv::DispatchPolicy::ConvOp;
  static constexpr int NumSpatialDimensions = Conv::NumSpatialDimensions;
  using ProblemShape = cutlass::conv::ConvProblemShape<ConvOp, NumSpatialDimensions>;
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::RasterOrderOptions;
  using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
  using MaxSwizzleSize = typename gemm::device::detail::MaxSwizzleSize;
  using Splits = typename gemm::device::detail::Splits;

  using Schedule = typename Conv::DispatchPolicy::Schedule;
  /// Initialization
  cutlass::Distribution::Kind init_A = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind init_B = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind init_C = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind init_bias = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind init_disable = cutlass::Distribution::Identity; // all zeros
  uint64_t seed = 6090;
  float epsilon = 0.0f;
  int split_p_slices = 1;
  thrust::universal_vector<ElementA> tensor_A;
  thrust::universal_vector<ElementB> tensor_B;
  thrust::universal_vector<ElementC> tensor_C;
  thrust::universal_vector<ElementD> tensor_D_computed;
  thrust::universal_vector<ElementD> tensor_D_reference;
  thrust::universal_vector<ElementBias> tensor_bias;
  thrust::universal_vector<ElementScalar> tensor_alpha;
  thrust::universal_vector<ElementScalar> tensor_beta;

  // Return true on success, else false
  bool initialize(ProblemShape const& problem_shape, uint64_t seed = 6090) {
    tensor_A.resize(sizeof(ElementA) * problem_shape.size_A());
    tensor_B.resize(sizeof(ElementB) * problem_shape.size_B());
    tensor_C.resize(sizeof(ElementC) * problem_shape.size_C());
    tensor_D_computed.resize(sizeof(ElementD) * problem_shape.size_C());
    tensor_D_reference.resize(sizeof(ElementD) * problem_shape.size_C());
    tensor_bias.resize(sizeof(ElementBias) * cute::size(cute::get<0>(problem_shape.get_shape_B())));
    if constexpr (IsPerChannelScaleEnabled) {
      tensor_alpha.resize(sizeof(ElementScalar) * cute::size(cute::get<0>(problem_shape.get_shape_B())));
      tensor_beta.resize(sizeof(ElementScalar) * cute::size(cute::get<0>(problem_shape.get_shape_B())));
    }
    initialize_values(tensor_A, init_A, seed);
    initialize_values(tensor_B, init_B, seed * 11);
    initialize_values(tensor_C, init_C, seed * 17);
    initialize_values(tensor_bias, init_bias, seed * 19);
    if constexpr (IsPerChannelScaleEnabled) {
      initialize_values(tensor_alpha, init_bias, seed * 23);
      if constexpr (DisableSource) {
        initialize_values(tensor_beta, init_disable, seed * 27);
      }
      else {
        initialize_values(tensor_beta, init_bias, seed * 27);
      }
    }

    bool flag = true;
    if constexpr (isSparseEnabled) {
      flag &= params.initialize(problem_shape, tensor_B, static_cast<int>(seed + 2023));
    }

    return flag;
  }

  // Determine SMEM requirements and waive if not satisfied
  bool sufficient() const {
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);
    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice() API call failed.");
    }

    int max_smem_size;
    result = cudaDeviceGetAttribute(&max_smem_size, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_idx);
    if (result != cudaSuccess) {
      throw std::runtime_error("cudaDeviceGetAttribute() failed");
    }

    return max_smem_size >= Conv::ConvKernel::SharedStorageSize;
  }

  auto transform_shape_and_stride_with_groups(ProblemShape const& problem_shape) {
    using TensorExtent = cute::array<int32_t, NumSpatialDimensions + 3>;
    using TensorStride = cute::array<int64_t, NumSpatialDimensions + 3>;

    TensorExtent shape_a_g{};
    TensorExtent shape_b_g{};
    TensorExtent shape_c_g{};
    TensorStride stride_a_g{};
    TensorStride stride_b_g{};
    TensorStride stride_c_g{};

    auto shape_a = cute::reverse(problem_shape.shape_A);
    auto shape_b = cute::reverse(problem_shape.shape_B);
    auto shape_c = cute::reverse(problem_shape.shape_C);
    auto stride_a = cute::reverse(problem_shape.stride_A);
    auto stride_b = cute::reverse(problem_shape.stride_B);
    auto stride_c = cute::reverse(problem_shape.stride_C);

    int32_t G = problem_shape.groups;

    if constexpr (ConvOp == cutlass::conv::Operator::kFprop ||
                  ConvOp == cutlass::conv::Operator::kDgrad) {
      // shape_a_g = (c,w,h,d,n,g) or (k,q,p,z,n,g)
      // shape_b_g = (c,s,r,k,t,g)
      // shape_c_g = (k,q,p,z,n,g) or (c,w,h,d,n,g)
      shape_a_g = cute::to_array<int32_t>(tuple_cat(
        cute::make_shape(cute::size<0>(shape_a) / G),
        cute::take<1,NumSpatialDimensions + 2>(shape_a),
        cute::make_shape(G)));
      shape_b_g = cute::to_array<int32_t>(tuple_cat(
        cute::take<0,NumSpatialDimensions + 1>(shape_b),
        cute::make_shape(cute::size<NumSpatialDimensions + 1>(shape_b) / G, G)));
      shape_c_g = cute::to_array<int32_t>(tuple_cat(
        cute::make_shape(cute::size<0>(shape_c) / G),
        cute::take<1,NumSpatialDimensions + 2>(shape_c),
        cute::make_shape(G)));

      stride_a_g = cute::to_array<int64_t>(append(stride_a, cute::size<0>(shape_a) / G));
      stride_b_g = cute::to_array<int64_t>(append(stride_b,
        cute::size<NumSpatialDimensions + 1>(stride_b) * cute::size<NumSpatialDimensions + 1>(shape_b) / G));
      stride_c_g = cute::to_array<int64_t>(append(stride_c, cute::size<0>(shape_c) / G));
    }
    else if constexpr (ConvOp == cutlass::conv::Operator::kWgrad) {
      // shape_a_g = (k,q,p,z,n,g)
      // shape_b_g = (c,w,h,d,n,g)
      // shape_c_g = (c,s,r,k,t,g)
      shape_a_g = cute::to_array<int32_t>(tuple_cat(
        cute::make_shape(cute::size<0>(shape_a) / G),
        cute::take<1,NumSpatialDimensions + 2>(shape_a),
        cute::make_shape(G)));
      shape_b_g = cute::to_array<int32_t>(tuple_cat(
        cute::make_shape(cute::size<0>(shape_b) / G),
        cute::take<1,NumSpatialDimensions + 2>(shape_b),
        cute::make_shape(G)));
      shape_c_g = cute::to_array<int32_t>(tuple_cat(
        cute::take<0,NumSpatialDimensions + 1>(shape_c),
        cute::make_shape(cute::size<NumSpatialDimensions + 1>(shape_c) / G, G)));

      stride_a_g = cute::to_array<int64_t>(append(stride_a, cute::size<0>(shape_a) / G));
      stride_b_g = cute::to_array<int64_t>(append(stride_b, cute::size<0>(shape_b) / G));
      stride_c_g = cute::to_array<int64_t>(append(stride_c,
        cute::size<NumSpatialDimensions + 1>(stride_c) * cute::size<NumSpatialDimensions + 1>(shape_c) / G));
    }

    return make_tuple(shape_a_g, shape_b_g, shape_c_g,
                      stride_a_g, stride_b_g, stride_c_g);
  }

  // Executes one test
  bool run(
    ProblemShape const& problem_shape,
    ElementScalar alpha = ElementScalar(1),
    ElementScalar beta = ElementScalar(0),
    dim3 cluster_shape = dim3(0, 0, 0),
    dim3 cluster_shape_fallback = dim3(0, 0, 0),
    RasterOrderOptions raster_order = RasterOrderOptions::Heuristic,
    MaxSwizzleSize max_swizzle = MaxSwizzleSize{},
    Splits splits = Splits{},
    DecompositionMode decomposition_mode = DecompositionMode::Heuristic
  ) {

    // Waive test if insufficient CUDA device
    if (!sufficient()) {
      if (CUTLASS_TEST_UNIT_ENABLE_WARNINGS) {
        std::cerr << "Test waived due to insufficient CUDA device.\n";
      }
      return true;
    }

    bool ret = initialize(problem_shape);

    if (!ret) {
      std::cerr << "initialize failed for the given problem_shape: \n";
      return false;
    }

    cutlass::KernelHardwareInfo hw_info;
    cudaGetDevice(&hw_info.device_id);
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    hw_info.cluster_shape = cluster_shape;
    hw_info.cluster_shape_fallback = cluster_shape_fallback;

    // configure the operator
    Conv conv_op;
    auto stride_C = StrideC{};
    auto stride_D = StrideD{};
    if constexpr (ConvOp == cutlass::conv::Operator::kWgrad) {
      stride_C = cutlass::make_cute_packed_stride(
        StrideC{}, problem_shape.shape_C, problem_shape.stride_C, ConvOp);
      stride_D = cutlass::make_cute_packed_stride(
        StrideD{}, problem_shape.shape_C, problem_shape.stride_C, ConvOp);
    }
    // Need to support non-packed output strides for fprop and dgrad kernel.
    else {
      cute::for_each(cute::make_seq<cute::rank<0>(StrideC{})>{}, [&](auto i) {
        cute::get<0, i>(stride_C) = problem_shape.stride_C[ProblemShape::RankT-2-i];
      });
      cute::for_each(cute::make_seq<cute::rank<0>(StrideD{})>{}, [&](auto i) {
        cute::get<0, i>(stride_D) = problem_shape.stride_C[ProblemShape::RankT-2-i];
      });
    }

    using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::RasterOrderOptions;
   using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;

    typename Conv::ConvKernel::TileScheduler::Arguments scheduler_args{};
    if constexpr (cute::is_same_v<typename Conv::ConvKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>) {
      scheduler_args = { static_cast<int>(splits), static_cast<int>(max_swizzle), raster_order, decomposition_mode };
    }

    auto mainloop_args = params.get_mainloop_arguments(problem_shape, tensor_A, tensor_B); 

    auto epilogue_args = typename Conv::ConvKernel::EpilogueArguments {
      {},
      tensor_C.data().get(),
      stride_C,
      tensor_D_computed.data().get(),
      stride_D,
    };

    auto args = typename Conv::Arguments {
      problem_shape,
      mainloop_args, // MainloopArguments
      epilogue_args, // EpilogueArguments
      hw_info,
      scheduler_args
    };

    auto &fusion_args = args.epilogue.thread;

    fusion_args.alpha = alpha;
    fusion_args.beta = beta;

    if constexpr (IsPerChannelScaleEnabled) {
      fusion_args.alpha_ptr = tensor_alpha.data().get();
      fusion_args.beta_ptr = tensor_beta.data().get();
    }

    if constexpr (IsBiasEnabled) {
      fusion_args.bias_ptr = tensor_bias.data().get();
    }

    // Clamp bound
    if constexpr (cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::Clamp<ElementCompute>>) {
      fusion_args.activation.lower_bound = CUTLASS_STL_NAMESPACE::numeric_limits<ElementCompute>::lowest();
      fusion_args.activation.upper_bound = CUTLASS_STL_NAMESPACE::numeric_limits<ElementCompute>::max();
    }

    // Scale
    if constexpr (cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::ScaledGELU_taylor<ElementCompute>> ||
                  cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::ScaledGELU<ElementCompute>> ||
                  cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::ScaledSiLu<ElementCompute>> ||
                  cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::ScaledHardSwish<ElementCompute>> ) {
      fusion_args.activation.scale = ElementCompute{1};
    }

    // LeakyRelu
    if constexpr (cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::LeakyReLU<ElementCompute>> ) {
      fusion_args.activation.leaky_alpha = ElementCompute{0};
    }

    cutlass::Status status = cutlass::Status::kInvalid;

    status = conv_op.can_implement(args);
    EXPECT_EQ(conv_op.can_implement(args), cutlass::Status::kSuccess);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "can_implement failed for the given problem_shape: \n";
      print(problem_shape);
      return false;
    }

    // find workspace requirement for parallel split-k reduction
    size_t workspace_size = Conv::get_workspace_size(args);
    thrust::universal_vector<uint8_t> workspace(workspace_size);

    status = conv_op.initialize(args, workspace.data().get());
    if (status != cutlass::Status::kSuccess) {
      cudaError_t error = cudaGetLastError();
      std::cerr << "This test is not supported: " << cudaGetErrorString(error) << "\n";
      return true;
    }

    // run conv3d operator
    status = conv_op();

    EXPECT_TRUE(status == cutlass::Status::kSuccess);
    if (status != cutlass::Status::kSuccess) {
      return false;
    }

    bool passed = false;
    cudaError_t result = cudaDeviceSynchronize();
    EXPECT_EQ(result, cudaSuccess) << " Kernel execution error: "
                                   << cudaGetErrorString(result);

    // Create cute::Tensors using the logical rank-3 MNK multi-mode shapes the mainloop gives us
    auto [shape_mA, shape_mB, shape_mC, stride_mA, stride_mB, stride_mC] =
      transform_shape_and_stride_with_groups(problem_shape);
    auto shape_mBias = cute::make_shape(cute::size(cute::get<0>(problem_shape.get_shape_B())));

    auto mA = make_tensor(tensor_A.data().get(), make_layout(shape_mA, stride_mA));
    auto mB = make_tensor(tensor_B.data().get(), make_layout(shape_mB, stride_mB));
    auto mC = make_tensor(tensor_C.data().get(), make_layout(shape_mC, stride_mC));
    auto mD_ref = make_tensor(tensor_D_reference.data().get(), make_layout(shape_mC, stride_mC));
    auto mD_computed = make_tensor(tensor_D_computed.data().get(), make_layout(shape_mC, stride_mC));
    auto mBias = make_tensor(tensor_bias.data().get(), make_layout(shape_mBias));
    auto mAlpha = make_tensor(tensor_alpha.data().get(), make_layout(shape_mBias));
    auto mBeta = make_tensor(tensor_beta.data().get(), make_layout(shape_mBias));

    cutlass::reference::host::ConvEpilogueFusionParams<
      ElementAccumulator,
      ElementScalar,
      ElementCompute,
      ElementC,
      ElementD,
      IsResidualEnabled,
      decltype(mAlpha),
      decltype(mBeta),
      decltype(mBias),
      ActivationFunctor>
        epilogue_fusion_params{};

    epilogue_fusion_params.alpha = alpha;
    epilogue_fusion_params.beta = beta;

    if constexpr (IsPerChannelScaleEnabled) {
      epilogue_fusion_params.tensor_alpha = mAlpha;
      epilogue_fusion_params.tensor_beta = mBeta;
    }

    if constexpr (IsBiasEnabled) {
      epilogue_fusion_params.tensor_bias = mBias;
    }

    auto padding = cute::reverse(problem_shape.lower_padding);
    auto tstride = cute::reverse(problem_shape.traversal_stride);
    auto dilation = cute::reverse(problem_shape.dilation);

    cutlass::reference::host::ConvReferenceImpl<
      ConvOp,
      NumSpatialDimensions,
      decltype(mA),
      decltype(mB),
      decltype(mC),
      decltype(mD_ref),
      decltype(padding),
      decltype(tstride),
      decltype(dilation),
      decltype(epilogue_fusion_params)>
        reference_impl(mA, mB, mC, mD_ref, padding, tstride, dilation, epilogue_fusion_params);

    //
    // Reference check - support caching results
    //

    CachedTestKey cached_test_key = CreateCachedConvNd3xTestKey<
        ProblemShape,
        ElementA,
        ElementB,
        ElementC,
        ElementD
    >(
        ConvOp,
        problem_shape,
        alpha,
        beta,
        tensor_A,
        tensor_B,
        tensor_C
      );

    //
    // Look for the cached key
    //

    bool cached_result_loaded = false;
    CachedTestResult cached_test_result;

    std::string convnd_result_cache_name =
      std::string("cached_results_") + CUTLASS_TARGET_NAME + ".txt";

    #if (CUTLASS_TEST_ENABLE_CACHED_RESULTS)
      CachedTestResultListing cached_results(convnd_result_cache_name);

      auto cached = cached_results.find(cached_test_key);

      cached_result_loaded = cached.first;
      if (cached_result_loaded) {
        cached_test_result = cached.second;
      }
    #endif

    if (!cached_result_loaded) {
      // Compute reference
      reference_impl.compute_reference();

      #if (CUTLASS_TEST_ENABLE_CACHED_RESULTS)
        cached_test_result.D = TensorHash(tensor_D_reference);
        CachedTestResultListing cached_results(convnd_result_cache_name);

        cached_results.append(cached_test_key, cached_test_result);
        cached_results.write(convnd_result_cache_name);
      #endif
    } // if (!cached_result_loaded)

    #if (CUTLASS_TEST_ENABLE_CACHED_RESULTS)
      uint32_t tensor_D_computed_hash = TensorHash(tensor_D_computed);
      passed = (tensor_D_computed_hash == cached_test_result.D);
      // If hash fails, double check against reference implementation.
      if(!passed) {
        std::cerr << "Hash-based comparison unsuccessful for key:" << "\n" << cached_test_key
            << ", comparing with reference implementation now.\n";
        if (cached_result_loaded) {
          // Compute reference
          reference_impl.compute_reference();
        }
        // Validate kernel against reference
        passed = compare_reference(mD_ref, mD_computed, mA, mB, mAlpha, mBeta, mBias, this->epsilon);
      }
    #else
      // Validate kernel against reference
      passed = compare_reference(mD_ref, mD_computed, mA, mB, mAlpha, mBeta, mBias, this->epsilon);
    #endif

    EXPECT_TRUE(passed);
    return passed;
  }

  template<
    class Engine, class Layout,
    class EngineA, class LayoutA,
    class EngineB, class LayoutB,
    class EngineAlpha, class LayoutAlpha,
    class EngineBeta, class LayoutBeta,
    class EngineBias, class LayoutBias>
  static constexpr bool
  compare_reference(
      cute::Tensor<Engine, Layout> const& reference,
      cute::Tensor<Engine, Layout> const& computed,
      cute::Tensor<EngineA, LayoutA> const& A,
      cute::Tensor<EngineB, LayoutB> const& B,
      cute::Tensor<EngineAlpha, LayoutAlpha> const& tensor_alpha,
      cute::Tensor<EngineBeta, LayoutBeta> const& tensor_beta,
      cute::Tensor<EngineBias, LayoutBias> const& tensor_bias,
      float epsilon = 0.0f) {
    if (size(reference) != size(computed)) {
      return false;
    }

    bool passed = true;
    if (epsilon == 0.0f) {
      // fast refcheck w/o epsilon
      for (size_t i = 0; i < size_t(size(reference)); ++i) {
        if (reference(i) != computed(i)) {
          passed = false;
          printf("[%llu] %f, %f\n", static_cast<unsigned long long>(i),
            float(reference(i)), float(computed(i)));
          break;
        }
      }
    } else {
      // refcheck with epsilon
      for (size_t i = 0; i < size_t(size(reference)); ++i) {
        auto ref = static_cast<float>(reference(i));
        auto act = static_cast<float>(computed(i));
        auto abs_error = std::abs(act - ref);
        auto rel_error = abs_error / (std::max(std::abs(act), std::abs(ref)) + 0.00001f);
        if (std::isnan(abs_error) || std::isnan(rel_error) ||
            std::min(abs_error, rel_error) > epsilon) {
          passed = false;
          printf("[%llu] %f, %f\n", static_cast<unsigned long long>(i),
            float(reference(i)), float(computed(i)));
          break;
        }
      }
    }
    #if CUTLASS_DEBUG_TRACE_LEVEL > 1
    if (not passed) {
      cute::print("Reference:");
      cute::print_tensor(reference);
      cute::print("\nComputed:");
      cute::print_tensor(computed);
      cute::print("\n");

      for (size_t i = 0; i < size_t(size(A)); ++i) {
        printf("[%llu]: A = %f\n", static_cast<unsigned long long>(i), float(A(i)));
      }
      for (size_t i = 0; i < size_t(size(B)); ++i) {
        printf("[%llu]: B = %f\n", static_cast<unsigned long long>(i), float(B(i)));
      }
      if constexpr (IsPerChannelScaleEnabled) {
        for (size_t i = 0; i < size_t(size(tensor_alpha)); ++i) {
          printf("[%llu]: alpha = %f\n", static_cast<unsigned long long>(i),
            float(tensor_alpha(i)));
        }
        for (size_t i = 0; i < size_t(size(tensor_beta)); ++i) {
          printf("[%llu]: beta = %f\n", static_cast<unsigned long long>(i),
            float(tensor_beta(i)));
        }
      }
      if constexpr (IsBiasEnabled) {
        for (size_t i = 0; i < size_t(size(tensor_bias)); ++i) {
          printf("[%llu]: bias = %f\n", static_cast<unsigned long long>(i),
            float(tensor_bias(i)));
        }
      }
      for (size_t i = 0; i < size_t(size(reference)); ++i) {
        printf("[%llu]: ref = %f, computed = %f\n", static_cast<unsigned long long>(i),
          float(reference(i)), float(computed(i)));
      }
    }
    #endif
    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Conv, bool SupportStrides = (Conv::DispatchPolicy::ConvOp != cutlass::conv::Operator::kDgrad)>
bool TestAllConv(double alpha = 1.0, double beta = 0.0, float epsilon = 0.0f,
                 dim3 cluster_shape = dim3(0, 0, 0),
                 dim3 cluster_shape_fallback = dim3(0, 0, 0)
                 ) {
  using ElementScalar = typename Conv::EpilogueOutputOp::ElementScalar;

  bool passed = true;
  ConvTestbed<Conv> testbed;
  testbed.epsilon = epsilon;
  auto problem_vector = get_conv_problem_vector<
      Conv::NumSpatialDimensions, Conv::DispatchPolicy::ConvOp, SupportStrides>();

  using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::RasterOrderOptions;
  using MaxSwizzleSize = typename gemm::device::detail::MaxSwizzleSize;
  using Splits = typename gemm::device::detail::Splits;

  std::vector<DecompositionMode> decomposition_modes = {DecompositionMode::Heuristic};
  static constexpr bool UsesStreamKScheduler = cute::is_same_v<typename Conv::ConvKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>;
  if constexpr (UsesStreamKScheduler) {
    decomposition_modes.push_back(DecompositionMode::DataParallel);
    decomposition_modes.push_back(DecompositionMode::SplitK);
    decomposition_modes.push_back(DecompositionMode::StreamK);
  }

  for (auto conv_problem : problem_vector) {
    #if CUTLASS_DEBUG_TRACE_LEVEL > 0
    print(conv_problem);
    #endif
    for (DecompositionMode decomp_mode : decomposition_modes) {
      std::vector problem_splits = {Splits{1}};
      if constexpr (UsesStreamKScheduler) {
        if (decomp_mode == DecompositionMode::SplitK) {
          problem_splits.push_back(Splits{2});
          problem_splits.push_back(Splits{4});
        }
      }
      for (auto splits : problem_splits) {

        passed = testbed.run(
          conv_problem,
          cutlass::from_real<ElementScalar>(alpha),
          cutlass::from_real<ElementScalar>(beta),
          cluster_shape,
          cluster_shape_fallback,
          RasterOrderOptions::Heuristic, // raster_order
          MaxSwizzleSize(1),
          splits,
          decomp_mode
          );
        if (!passed) {
          printf("Failed test for "); print(conv_problem);
          return false;
        }
      } // splits
    } // decomposition_mode
  }

  return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace test::conv::device

/////////////////////////////////////////////////////////////////////////////////////////////////
