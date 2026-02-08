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
/* \file
   \brief Defines operations for all GEMM operation kinds in CUTLASS Library.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/detail/collective.hpp"
#include "cutlass/array.h"
#include "cutlass/array_subbyte.h"
#include "cutlass/library/library.h"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp" // StructuredSparseCompressor
#include "cutlass/transform/device/transform_universal_adapter.hpp" // TransformUniversalAdapter
#include "cutlass/util/packed_stride.hpp"        // make_cute_packed_stride
#include "gemm_operation_3x.hpp"
#include "library_internal.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/mixed_dtype_utils.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cute/tensor.hpp"
#include <unordered_map>

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::library {

///////////////////////////////////////////////////////////////////////////////////////////////////

// Limitation & Assumptions:
// 1. The tensor must be densely packed.  That is, lda is k if the tensor is k-major,
//    and lda is m if the tensor is m-major.
// 2. Circular buffer for tensorA and tensorE may have a less count compared to tensorB and others.
//    This is because we can not get the problem_count information in the get_device_workspace_size().
//    But I can promise it will use at least 192MB memory if we enable circular buffer.
template <typename Operator_>
class SparseGemmUniversal3xOperation : public GemmOperation3xBase<Operator_> {
public:

  using Operator = Operator_;
  using OperatorArguments = typename Operator::Arguments;
  using ElementA = typename Operator::ElementA;
  using LayoutA = typename Operator::LayoutA;
  using ElementB = typename Operator::ElementB;
  using LayoutB = typename Operator::LayoutB;
  using ElementC = typename Operator::ElementC;
  using LayoutC = typename Operator::LayoutC;
  using ElementD = typename Operator::ElementD;
  using LayoutD = typename Operator::LayoutD;
  using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

  using CollectiveMainloop = typename Operator::CollectiveMainloop;
  using CollectiveEpilogue = typename Operator::CollectiveEpilogue;
  using ThreadEpilogueOp = typename CollectiveEpilogue::ThreadEpilogueOp;

  static constexpr bool IsRuntimeDataTypeA = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementA>();

  static constexpr bool IsRuntimeDataTypeB = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementB>();

  static_assert((IsRuntimeDataTypeA && IsRuntimeDataTypeB) ||
                (!IsRuntimeDataTypeA && !IsRuntimeDataTypeB),
                "ElementA and ElementB in a GEMM kernel should be both runtime or both static.");

  static constexpr bool IsRuntimeDataType = IsRuntimeDataTypeA && IsRuntimeDataTypeB;

  using ElementE = typename CollectiveMainloop::ElementE;
  using LayoutE = typename CollectiveMainloop::LayoutE;
  using SparseConfig = typename CollectiveMainloop::SparseConfig;
  using LayoutATag = decltype(SparseConfig::deduce_layoutA_tag(typename CollectiveMainloop::LayoutA{}));
  using CompressorUtility = cutlass::transform::kernel::StructuredSparseCompressorUtility<
                              cute::Shape<int, int, int, int>,
                              ElementA,
                              LayoutATag,
                              SparseConfig>;
  using CompressorKernel = cutlass::transform::kernel::StructuredSparseCompressor<
                              cute::Shape<int, int, int, int>,
                              ElementA,
                              LayoutATag,
                              SparseConfig,
                              typename Operator::ArchTag>;

  using Compressor = cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

public:

  /// Constructor
  SparseGemmUniversal3xOperation(char const *name = "unknown_gemm"):
    GemmOperation3xBase<Operator_>(name, GemmKind::kUniversal) {}

protected:

  /// Constructs the arguments structure given the configuration and arguments
  static Status construct_arguments_(
      OperatorArguments &operator_args, GemmUniversalConfiguration const *configuration) {
    // NOTE: GemmUniversalConfiguration does not contain problem shapes or batch strides
    // Do nothing here and construct kernel arguments in update_arguments_ instead
    // We also cannot construct TMA descriptors without all the arguments available

    operator_args.mode = configuration->mode;
    return Status::kSuccess;
  }

  template<class FusionArgs, class = void>
  struct UpdateFusionArgs {
    static Status update_(FusionArgs const& fusion_args, GemmUniversalArguments const &arguments) {
      // If a custom EVT is instantiated then it is the users's responsibility
      // to ensure alpha and beta are updated appropriately
      return Status::kSuccess;
    }
  };

  template<class FusionArgs>
  struct UpdateFusionArgs<FusionArgs, cute::void_t<decltype(FusionArgs{}.alpha)>> {
    static Status update_(FusionArgs& fusion_args, GemmUniversalArguments const &arguments) {
      if (arguments.pointer_mode == ScalarPointerMode::kHost) {
        fusion_args.alpha = *static_cast<ElementCompute const *>(arguments.alpha);
        fusion_args.beta = *static_cast<ElementCompute const *>(arguments.beta);
        fusion_args.alpha_ptr = nullptr;
        fusion_args.beta_ptr = nullptr;

        return Status::kSuccess;
      }
      else if (arguments.pointer_mode == ScalarPointerMode::kDevice) {
        fusion_args.alpha = 0;
        fusion_args.beta = 0;
        fusion_args.alpha_ptr = static_cast<ElementCompute const *>(arguments.alpha);
        fusion_args.beta_ptr = static_cast<ElementCompute const *>(arguments.beta);

        return Status::kSuccess;
      }
      else {
        return Status::kErrorInvalidProblem;
      }
    }
  };

  /// Constructs the arguments structure given the configuration and arguments
  static Status update_arguments_(
      OperatorArguments &operator_args,
      GemmUniversalArguments const *arguments,
      CompressorUtility const& compressor_utility,
      void* device_a_compressed_ptr = nullptr,
      void* device_e_ptr = nullptr) {
    Status status = Status::kSuccess;

    status = UpdateFusionArgs<decltype(operator_args.epilogue.thread)>::update_(
      operator_args.epilogue.thread, *arguments);
    if (status != Status::kSuccess) {
      return status;
    }

    operator_args.problem_shape = cute::make_shape(
      arguments->problem_size.m(),
      arguments->problem_size.n(),
      arguments->problem_size.k(),
      arguments->batch_count);

    // update arguments

    if constexpr (IsRuntimeDataType) {
      using ArrayElementA = typename Operator::GemmKernel::CollectiveMainloop::ArrayElementA;
      using ArrayElementB = typename Operator::GemmKernel::CollectiveMainloop::ArrayElementB;
      operator_args.mainloop.ptr_A = static_cast<ArrayElementA const *>(device_a_compressed_ptr);
      operator_args.mainloop.ptr_B = static_cast<ArrayElementB const *>(arguments->B);

      std::unordered_map<RuntimeDatatype, cute::UMMA::MXF8F6F4Format> mapping = {
          {RuntimeDatatype::kE4M3, cute::UMMA::MXF8F6F4Format::E4M3},
          {RuntimeDatatype::kE5M2, cute::UMMA::MXF8F6F4Format::E5M2},
          {RuntimeDatatype::kE3M2, cute::UMMA::MXF8F6F4Format::E3M2},
          {RuntimeDatatype::kE2M1, cute::UMMA::MXF8F6F4Format::E2M1}
      };

      auto iter_runtime_a = mapping.find(arguments->runtime_input_datatype_a);
      auto iter_runtime_b = mapping.find(arguments->runtime_input_datatype_b);

      if (iter_runtime_a != mapping.end()) {
          operator_args.mainloop.runtime_data_type_a = iter_runtime_a->second;
      } else {
        assert("invalid runtime argument for datatype A!");
      }

      if (iter_runtime_b != mapping.end()) {
          operator_args.mainloop.runtime_data_type_b = iter_runtime_b->second;
      } else {
        assert("invalid runtime argument for datatype B!");
      }

    }
    else {
      operator_args.mainloop.ptr_A = static_cast<ElementA const *>(device_a_compressed_ptr);
      operator_args.mainloop.ptr_B = static_cast<ElementB const *>(arguments->B);
    }
    operator_args.mainloop.ptr_E = static_cast<ElementE const *>(device_e_ptr);
    operator_args.epilogue.ptr_C = static_cast<ElementC const *>(arguments->C);
    operator_args.epilogue.ptr_D = static_cast<ElementD       *>(arguments->D);

    operator_args.mainloop.layout_a = compressor_utility.fill_layoutA_from_compressor();
    operator_args.mainloop.layout_e = compressor_utility.fill_layoutE_from_compressor();
    operator_args.mainloop.dB = cute::make_int_tuple_from<typename Operator::GemmKernel::StrideB>(
        arguments->ldb, arguments->batch_stride_B);
    operator_args.epilogue.dC = cute::make_int_tuple_from<typename Operator::GemmKernel::StrideC>(
        arguments->ldc, arguments->batch_stride_C);
    operator_args.epilogue.dD = operator_args.epilogue.dC;

    /* Query device SM count and max active clusters to pass onto the kernel as an argument, where needed */
    operator_args.hw_info.sm_count = arguments->sm_count;
    if constexpr (!std::is_const_v<decltype(operator_args.scheduler.max_swizzle_size)>) {
      operator_args.scheduler.max_swizzle_size = arguments->swizzle_size;
    }

    if constexpr (!std::is_const_v<decltype(operator_args.scheduler.raster_order)>) {
      using Enum_t = decltype(operator_args.scheduler.raster_order);
      switch (arguments->raster_order) {
        case RasterOrder::kAlongN:
          operator_args.scheduler.raster_order = Enum_t::AlongN;
          break;
        case RasterOrder::kAlongM:
          operator_args.scheduler.raster_order = Enum_t::AlongM;
          break;
        default:
          operator_args.scheduler.raster_order = Enum_t::Heuristic;
      }
    }

    if constexpr (std::is_same_v<typename Operator::GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>) {
      operator_args.scheduler.splits = arguments->split_k_slices;
    }

    if constexpr (Operator::ArchTag::kMinComputeCapability >= 100) {
      operator_args.hw_info.cluster_shape = dim3(
        arguments->cluster_shape.m(),
        arguments->cluster_shape.n(),
        arguments->cluster_shape.k());
      operator_args.hw_info.cluster_shape_fallback = dim3(
        arguments->cluster_shape_fallback.m(),
        arguments->cluster_shape_fallback.n(),
        arguments->cluster_shape_fallback.k());
    }
    return status;
  }

public:

  /// Returns success if the operation can proceed
  Status can_implement(
      void const *configuration_ptr, void const *arguments_ptr) const override {

    GemmUniversalConfiguration const *configuration =
      static_cast<GemmUniversalConfiguration const *>(configuration_ptr);
    GemmUniversalArguments const *arguments =
      static_cast<GemmUniversalArguments const *>(arguments_ptr);

    OperatorArguments args;
    auto problem_shape_MNKL = cute::make_shape(
      configuration->problem_size.m(),
      configuration->problem_size.n(),
      configuration->problem_size.k(),
      configuration->batch_count);

    const int M = configuration->problem_size.m();
    const int N = configuration->problem_size.n();
    const int K = configuration->problem_size.k();
    const int L = configuration->batch_count;
    using StrideA = typename CompressorUtility::StrideA;
    auto dA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    compressor_utility.set_problem_size(problem_shape_MNKL, dA);
    auto status = update_arguments_(args, arguments, compressor_utility);
    if (status != Status::kSuccess) {
      return status;
    }

    // can_implement rules may need access to problem shape
    args.problem_shape = problem_shape_MNKL;
    return Operator::can_implement(args);
  }

  /// Gets the host-side workspace
  uint64_t get_host_workspace_size(void const *) const override {
    // Memory to hold operator
    host_op_workspace_size = sizeof(Operator);

    // Memory to hold result of `.structure_sparse_zero_mask_fill()`
    tensor_a_size          = compressor_utility.get_raw_tensor_A_bytes();

    // NOTE: order here is the order of workspace partition
    const uint64_t size = host_op_workspace_size + tensor_a_size;

    return size;
  }

  /// Gets the device-side workspace
  uint64_t get_device_workspace_size(
    void const *configuration_ptr,void const *arguments_ptr) const override {

    OperatorArguments args;
    auto status = update_arguments_(
      args, static_cast<GemmUniversalArguments const *>(arguments_ptr), compressor_utility);
    if (status != Status::kSuccess) {
      return 0;
    }

    typename Compressor::Arguments compress_arguments {
      {compressor_utility.M, 0, compressor_utility.K, compressor_utility.L},
      {/*Empty Not Use*/},
      {/*Empty Not Use*/} };

    // Size for one iteration
    // For multi-iteration, will need to multiply result of this function w/ actual problem_count
    tensor_ac_size           = compressor_utility.get_compressed_tensor_A_bytes();
    tensor_e_size            = compressor_utility.get_tensor_E_bytes();
    device_op_workspace_size = Operator::get_workspace_size(args);
    device_compress_workspace_size = Compressor::get_workspace_size(compress_arguments);

    // NOTE: order here is the order of workspace partition
    device_per_iter_workspace_size = device_op_workspace_size + device_compress_workspace_size + tensor_ac_size + tensor_e_size;

    return device_per_iter_workspace_size;
  }

  /// Initializes the workspace
  Status initialize(
      void const *configuration_ptr,
      void *host_workspace,
      void *device_workspace,
      cudaStream_t stream = nullptr) const override {
    return Status::kErrorInternal;
  }

  Status initialize_with_profiler_workspace(
      void const *configuration,
      void *host_workspace,
      void *device_workspace,
      uint8_t **profiler_workspaces,
      int problem_count_from_profiler,
      cudaStream_t stream = nullptr) {

    iter_idx.resize(static_cast<GemmUniversalConfiguration const*>(configuration)->device_count, 0);

    // Set problem_count.
    problem_count = problem_count_from_profiler;

    // * Host Ptr
    auto* host_op_workspace_ptr       = reinterpret_cast<uint8_t*>(host_workspace);
    auto* host_a_raw_ptr              = host_op_workspace_ptr + host_op_workspace_size;

    // * Construct Op
    Operator *op = new (host_op_workspace_ptr) Operator;

    // * Device Ptr (1st iteration)
    // Device workspace : | iter1 | iter2 | iter3 | .. | iterx |
    //            iteri : op_workspace | tensor_ac | tensor_e
    auto* device_ptr_iter1                = static_cast<uint8_t*>(device_workspace);
    auto* device_op_workspace_ptr_iter1         = device_ptr_iter1;
    auto* device_compressor_workspace_ptr_iter1 = device_op_workspace_ptr_iter1 + device_op_workspace_size;
    auto* device_a_compressed_ptr_iter1         = device_compressor_workspace_ptr_iter1 + device_compress_workspace_size;
    auto* device_e_ptr_iter1                    = device_a_compressed_ptr_iter1 + tensor_ac_size;

    // * Device A Raw Ptr
    auto* device_a_raw_ptr = profiler_workspaces[0];

    // * Random fill 50% of TensorA w/ zero following the structured sparse requirement
    CUDA_CHECK(cudaMemcpyAsync(host_a_raw_ptr, device_a_raw_ptr, tensor_a_size, cudaMemcpyDeviceToHost, stream));
    compressor_utility.structure_sparse_zero_mask_fill(host_a_raw_ptr, 2000);
    CUDA_CHECK(cudaMemcpyAsync(device_a_raw_ptr, host_a_raw_ptr, tensor_a_size, cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaGetLastError());

    // * Compress DTensorA and get DTensorAC & DTensorE
    cutlass::KernelHardwareInfo hw_info;
    CUDA_CHECK(cudaGetDevice(&hw_info.device_id));
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    typename Compressor::Arguments arguments{
        {compressor_utility.M, 0, compressor_utility.K, compressor_utility.L},
        {device_a_raw_ptr,
         compressor_utility.dA,
         device_a_compressed_ptr_iter1,
         device_e_ptr_iter1},
        {hw_info}
    };

    cutlass::Status status {cutlass::Status::kSuccess };

    Compressor compressor_op;
    status = compressor_op.can_implement(arguments);
    if (status != Status::kSuccess) {
      return status;
    }

    status = compressor_op.initialize(arguments, device_compressor_workspace_ptr_iter1, stream);
    if (status != Status::kSuccess) {
       return status;
    }

    status = compressor_op.run(stream);
    if (status != Status::kSuccess) {
       return status;
    }

    // * Copy Iter1's DTensorAC DTensorE to each iteration's DTensorAC DTensorE
    for (int iter_i = 1; iter_i < problem_count; iter_i++) {
      // * Device AC E Ptr per iteration
      // Device workspace : | iter1 | iter2 | iter3 | .. | iterx |
      //            iteri : op_workspace | tensor_ac | tensor_e
      auto* device_ptr_iteri                = static_cast<uint8_t*>(device_workspace) + device_per_iter_workspace_size * iter_i;
      auto* device_op_workspace_ptr         = device_ptr_iteri;
      auto* device_compressor_workspace_ptr = device_op_workspace_ptr + device_op_workspace_size;
      auto* device_a_compressed_ptr         = device_compressor_workspace_ptr + device_compress_workspace_size;
      auto* device_e_ptr                    = device_a_compressed_ptr + tensor_ac_size;

      CUDA_CHECK(cudaMemcpyAsync(device_a_compressed_ptr, device_a_compressed_ptr_iter1, tensor_ac_size, cudaMemcpyDeviceToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(device_e_ptr, device_e_ptr_iter1, tensor_e_size, cudaMemcpyDeviceToDevice, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaGetLastError());

    return Status::kSuccess;
  }

  /// Runs the kernel
  Status run(
      void const *arguments_ptr,
      void *host_workspace,
      void *device_workspace,
      cudaStream_t stream = nullptr) const override {

    OperatorArguments operator_args;


    const auto device_index = static_cast<GemmUniversalArguments const *>(arguments_ptr)->device_index;

    auto* device_ptr_iteri                = static_cast<uint8_t*>(device_workspace) + device_per_iter_workspace_size * iter_idx[device_index];
    auto* device_op_workspace_ptr         = device_ptr_iteri;
    auto* device_compressor_workspace_ptr = device_op_workspace_ptr + device_op_workspace_size;
    auto* device_a_compressed_ptr         = device_compressor_workspace_ptr + device_compress_workspace_size;
    auto* device_e_ptr                    = device_a_compressed_ptr + tensor_ac_size;
    iter_idx[device_index] = (iter_idx[device_index] + 1) % problem_count;

    Status status = update_arguments_(operator_args, static_cast<GemmUniversalArguments const *>(arguments_ptr), compressor_utility, device_a_compressed_ptr, device_e_ptr );

    if (status != Status::kSuccess) {
      return status;
    }

    Operator *op = static_cast<Operator *>(host_workspace);
    // We need to call initialize() since we have to rebuild TMA desc for every new set of args
    status = op->run(operator_args, device_op_workspace_ptr, stream, nullptr, 
                     static_cast<GemmUniversalArguments const *>(arguments_ptr)->use_pdl);
    return status;
  }

private:
  // Variables that must change in the const functions.
  mutable CompressorUtility compressor_utility;
  mutable int problem_count = 1;
  mutable std::vector<int> iter_idx;

  mutable uint64_t tensor_ac_size = 0;
  mutable uint64_t tensor_e_size = 0;
  mutable uint64_t tensor_a_size = 0;
  mutable uint64_t host_op_workspace_size = 0;
  mutable uint64_t device_compress_workspace_size = 0;
  mutable uint64_t device_op_workspace_size = 0;
  mutable uint64_t device_per_iter_workspace_size = 0;
};
///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::library

///////////////////////////////////////////////////////////////////////////////////////////////////
