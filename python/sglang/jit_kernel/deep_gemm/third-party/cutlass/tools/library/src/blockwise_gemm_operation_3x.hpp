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
#include "cutlass/library/library.h"
#include "library_internal.h"
#include "gemm_operation_3x.hpp"
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::library {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class BlockwiseGemmUniversal3xOperation : public GemmOperation3xBase<Operator_> {
public:
  using Operator = Operator_;
  using OperatorArguments = typename Operator::Arguments;
  using ElementA = typename Operator::CollectiveMainloop::ElementA;
  using ElementSFA = typename Operator::ElementAccumulator;
  using LayoutA = typename Operator::LayoutA;
  using ElementB = typename Operator::CollectiveMainloop::ElementB;
  using ElementSFB = typename Operator::ElementAccumulator;
  using LayoutB = typename Operator::LayoutB;
  using ElementC = typename Operator::ElementC;
  using LayoutC = typename Operator::LayoutC;
  using ElementD = typename Operator::ElementD;
  using LayoutD = typename Operator::LayoutD;
  using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

  using TiledMma = typename Operator::CollectiveMainloop::TiledMma;

  using CollectiveMainloop = typename Operator::CollectiveMainloop;
  using CollectiveEpilogue = typename Operator::CollectiveEpilogue;
  using ThreadEpilogueOp = typename CollectiveEpilogue::ThreadEpilogueOp;
  
  static constexpr bool IsRuntimeDataTypeA = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementA>();

  static constexpr bool IsRuntimeDataTypeB = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementB>();

  static_assert((IsRuntimeDataTypeA && IsRuntimeDataTypeB) ||
                (!IsRuntimeDataTypeA && !IsRuntimeDataTypeB), 
                "ElementA and ElementB in a GEMM kernel should be both runtime or both static.");

  static constexpr bool IsRuntimeDataType = IsRuntimeDataTypeA && IsRuntimeDataTypeB;

private:
  BlockwiseGemmDescription description_;

public:

  /// Constructor
  BlockwiseGemmUniversal3xOperation(char const *name = "unknown_gemm"):
      GemmOperation3xBase<Operator_>(name, GemmKind::kUniversal) {
    description_.kind = OperationKind::kBlockwiseGemm;
    description_.SFA.element = NumericTypeMap<ElementSFA>::kId;
    description_.SFA.layout = size<0,1>(typename CollectiveMainloop::LayoutSFA{}.stride()) == 1 ? 
        LayoutTypeID::kColumnMajor : LayoutTypeID::kRowMajor;
    description_.SFA.alignment = CollectiveMainloop::AlignmentSFA;
    description_.SFA.log_extent_range = 32;
    description_.SFA.log_stride_range = 32;

    description_.SFB.element = NumericTypeMap<ElementSFB>::kId;
    description_.SFB.layout = size<0,1>(typename CollectiveMainloop::LayoutSFB{}.stride()) == 1 ? 
        LayoutTypeID::kRowMajor : LayoutTypeID::kColumnMajor;
    description_.SFB.alignment = CollectiveMainloop::AlignmentSFA;
    description_.SFB.log_extent_range = 32;
    description_.SFB.log_stride_range = 32;

    description_.SFMVecSize = Operator::CollectiveMainloop::ScaleGranularityM;
    description_.SFNVecSize = Operator::CollectiveMainloop::ScaleGranularityN;
    description_.SFKVecSize = Operator::CollectiveMainloop::ScaleGranularityK;

    description_.name = name;
    description_.provider = Provider::kCUTLASS;
    description_.gemm_kind = GemmKind::kUniversal;

    description_.tile_description.threadblock_shape = make_Coord(
      Operator::ThreadblockShape::kM,
      Operator::ThreadblockShape::kN,
      Operator::ThreadblockShape::kK);

    if constexpr (Operator::ArchTag::kMinComputeCapability >= 90) {
      description_.tile_description.cluster_shape = make_Coord(
        Operator::ClusterShape::kM,
        Operator::ClusterShape::kN,
        Operator::ClusterShape::kK);
    }

    description_.tile_description.threadblock_stages = Operator::kStages;

    description_.tile_description.warp_count = make_Coord(
      Operator::WarpCount::kM,
      Operator::WarpCount::kN,
      Operator::WarpCount::kK);

    description_.tile_description.math_instruction.instruction_shape = make_Coord(
      Operator::InstructionShape::kM,
      Operator::InstructionShape::kN,
      Operator::InstructionShape::kK);

    description_.tile_description.math_instruction.element_accumulator =
      NumericTypeMap<ElementAccumulator>::kId;

    description_.tile_description.math_instruction.opcode_class =
      OpcodeClassMap<typename Operator::OperatorClass>::kId;

    description_.tile_description.math_instruction.math_operation =
      MathOperationMap<typename Operator::MathOperator>::kId;

    description_.tile_description.minimum_compute_capability =
      ArchMap<typename Operator::ArchTag, typename Operator::OperatorClass>::kMin;

    description_.tile_description.maximum_compute_capability =
      ArchMap<typename Operator::ArchTag, typename Operator::OperatorClass>::kMax;

    description_.A = make_TensorDescription<ElementA, LayoutA>(Operator::kAlignmentA);
    description_.B = make_TensorDescription<ElementB, LayoutB>(Operator::kAlignmentB);
    description_.C = make_TensorDescription<ElementC, LayoutC>(Operator::kAlignmentC);
    description_.D = make_TensorDescription<ElementD, LayoutD>(Operator::kAlignmentD);
    description_.element_epilogue = NumericTypeMap<ElementCompute>::kId;

    description_.split_k_mode = SplitKMode::kNone;
  }

  /// Returns the description of the GEMM operation
  virtual OperationDescription const & description() const {
    return description_;
  }

  /// Returns the description of the GEMM operation
  BlockwiseGemmDescription const& get_gemm_description() const {
    return description_;
  }

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
    static Status update_(FusionArgs const& fusion_args, BlockwiseGemmArguments const &arguments) {
      // If a custom EVT is instantiated then it is the users's responsibility
      // to ensure alpha and beta are updated appropriately
      return Status::kSuccess;
    }
  };

  template<class FusionArgs>
  struct UpdateFusionArgs<FusionArgs, cute::void_t<decltype(FusionArgs{}.alpha)>> {
    static Status update_(FusionArgs& fusion_args, BlockwiseGemmArguments const &arguments) {
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
      BlockwiseGemmArguments const *arguments) {
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
      operator_args.mainloop.ptr_A = static_cast<ArrayElementA const *>(arguments->A);
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
    
    operator_args.mainloop.ptr_A = static_cast<ElementA const *>(arguments->A);
    operator_args.mainloop.ptr_B = static_cast<ElementB const *>(arguments->B);
    } 
    operator_args.mainloop.ptr_SFA = static_cast<ElementSFA const *>(arguments->SFA);
    operator_args.mainloop.ptr_SFB = static_cast<ElementSFB const *>(arguments->SFB);
    operator_args.epilogue.ptr_C = static_cast<ElementC const *>(arguments->C);
    operator_args.epilogue.ptr_D = static_cast<ElementD       *>(arguments->D);

    operator_args.mainloop.dA = cute::make_int_tuple_from<typename Operator::GemmKernel::StrideA>(
        arguments->lda, arguments->batch_stride_A);
    operator_args.mainloop.dB = cute::make_int_tuple_from<typename Operator::GemmKernel::StrideB>(
        arguments->ldb, arguments->batch_stride_B);
    operator_args.epilogue.dC = cute::make_int_tuple_from<typename Operator::GemmKernel::StrideC>(
        arguments->ldc, arguments->batch_stride_C);
    operator_args.epilogue.dD = operator_args.epilogue.dC;

    operator_args.mainloop.layout_SFA = Operator::CollectiveMainloop::ScaleConfig::tile_atom_to_shape_SFA(operator_args.problem_shape);
    operator_args.mainloop.layout_SFB = Operator::CollectiveMainloop::ScaleConfig::tile_atom_to_shape_SFB(operator_args.problem_shape);

    /* Query device SM count to pass onto the kernel as an argument, where needed */
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
    BlockwiseGemmArguments const *arguments =
      static_cast<BlockwiseGemmArguments const *>(arguments_ptr);

    if (arguments->sf_m_vec_size != description_.SFMVecSize && arguments->sf_m_vec_size != 0) {
      return Status::kErrorInvalidProblem;
    }
    if (arguments->sf_n_vec_size != description_.SFNVecSize && arguments->sf_n_vec_size != 0) {
      return Status::kErrorInvalidProblem;
    }
    if (arguments->sf_k_vec_size != description_.SFKVecSize && arguments->sf_k_vec_size != 0) {
      return Status::kErrorInvalidProblem;
    }

    OperatorArguments args;
    auto status = update_arguments_(args, arguments);
    if (status != Status::kSuccess) {
      return status;
    }

    // can_implement rules may need access to problem shape
    args.problem_shape = cute::make_shape(
      configuration->problem_size.m(),
      configuration->problem_size.n(),
      configuration->problem_size.k(),
      configuration->batch_count);

    return Operator::can_implement(args);
  }

  /// Gets the host-side workspace
  uint64_t get_host_workspace_size(void const *configuration) const override {
    return sizeof(Operator);
  }

  /// Gets the device-side workspace
  uint64_t get_device_workspace_size(
      void const *configuration_ptr,void const *arguments_ptr) const override {

    OperatorArguments args;
    auto status = update_arguments_(
      args, static_cast<BlockwiseGemmArguments const *>(arguments_ptr));
    if (status != Status::kSuccess) {
      return 0;
    }

    uint64_t size = Operator::get_workspace_size(args);
    return size;
  }

  /// Initializes the workspace
  Status initialize(
      void const *configuration_ptr,
      void *host_workspace,
      void *device_workspace,
      cudaStream_t stream = nullptr) const override {
    Operator *op = new (host_workspace) Operator;
    return Status::kSuccess;
  }

  Status initialize_with_profiler_workspace(
      void const *configuration, 
      void *host_workspace, 
      void *device_workspace, 
      uint8_t **profiler_workspaces,
      int problem_count_from_profiler,
      cudaStream_t stream = nullptr) {
    return Status::kSuccess;
  }

  /// Runs the kernel
  Status run(
      void const *arguments_ptr,
      void *host_workspace,
      void *device_workspace = nullptr,
      cudaStream_t stream = nullptr) const override {

    OperatorArguments args;
    Status status = update_arguments_(args, static_cast<BlockwiseGemmArguments const *>(arguments_ptr));
    if (status != Status::kSuccess) {
      return status;
    }

    Operator *op = static_cast<Operator *>(host_workspace);
    // We need to call initialize() since we have to rebuild TMA desc for every new set of args
    status = op->run(args, device_workspace, stream, nullptr, static_cast<BlockwiseGemmArguments const *>(arguments_ptr)->use_pdl);
    return status;
  }
};
///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::library

///////////////////////////////////////////////////////////////////////////////////////////////////
