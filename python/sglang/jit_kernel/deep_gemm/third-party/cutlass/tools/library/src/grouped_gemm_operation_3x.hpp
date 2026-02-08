/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
   \brief Defines operations for all grouped GEMM operations in CUTLASS Library.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/detail/collective.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/library/library.h"
#include "cutlass/library/util.h"
#include "gemm_operation_3x.hpp"
#include "library_internal.h"

namespace cutlass::library {

template <typename Operator_>
class GroupedGemmOperation3xBase : public GemmOperation3xBase<Operator_> {
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

  GroupedGemmOperation3xBase(char const* name = "unknown_gemm")
      : GemmOperation3xBase<Operator_>(name, GemmKind::kGrouped) {
    this->description_.kind = OperationKind::kGroupedGemm;
    this->description_.name = name;
    this->description_.provider = Provider::kCUTLASS;

    this->description_.gemm = GemmOperation3xBase<Operator_>::description_;
    this->description_.tile_description = this->description_.gemm.tile_description;
  };

public:
  mutable CudaBuffer strideA_device;
  mutable CudaBuffer strideB_device;
  mutable CudaBuffer strideC_device;
  mutable CudaBuffer strideD_device;

  /// Returns the description of the GEMM operation
  virtual OperationDescription const& description() const override final { return description_; }
  /// Gets the host-side workspace
  uint64_t get_host_workspace_size(void const* configuration) const override final {
    return sizeof(Operator);
  }

protected:
  library::GroupedGemmDescription description_;

  Status initialize_strides(GemmGroupedConfiguration const& config) const {
    auto const num_groups = config.problem_count;
    this->strideA_device =
      CudaBuffer(sizeof(typename Operator::GemmKernel::InternalStrideA) * num_groups);
    this->strideB_device =
      CudaBuffer(sizeof(typename Operator::GemmKernel::InternalStrideB) * num_groups);
    this->strideC_device =
      CudaBuffer(sizeof(typename Operator::GemmKernel::InternalStrideC) * num_groups);
    this->strideD_device =
      CudaBuffer(sizeof(typename Operator::GemmKernel::InternalStrideD) * num_groups);

    std::vector<typename Operator::GemmKernel::InternalStrideA> strideA_host(num_groups);
    std::vector<typename Operator::GemmKernel::InternalStrideB> strideB_host(num_groups);
    std::vector<typename Operator::GemmKernel::InternalStrideC> strideC_host(num_groups);
    std::vector<typename Operator::GemmKernel::InternalStrideD> strideD_host(num_groups);
    for (int group_idx = 0; group_idx < num_groups; group_idx++) {
      strideA_host[group_idx] =
        cute::make_int_tuple_from<typename Operator::GemmKernel::InternalStrideA>(
          config.lda[group_idx]);
      strideB_host[group_idx] =
        cute::make_int_tuple_from<typename Operator::GemmKernel::InternalStrideB>(
          config.ldb[group_idx]);
      strideC_host[group_idx] =
        cute::make_int_tuple_from<typename Operator::GemmKernel::InternalStrideC>(
          config.ldc[group_idx]);
      strideD_host[group_idx] =
        cute::make_int_tuple_from<typename Operator::GemmKernel::InternalStrideD>(
          config.ldc[group_idx]);
    }
    CUDA_CHECK(cudaMemcpy(
      this->strideA_device.data(),
      strideA_host.data(),
      sizeof(typename Operator::GemmKernel::InternalStrideA) * num_groups,
      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
      this->strideB_device.data(),
      strideB_host.data(),
      sizeof(typename Operator::GemmKernel::InternalStrideB) * num_groups,
      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
      this->strideC_device.data(),
      strideC_host.data(),
      sizeof(typename Operator::GemmKernel::InternalStrideC) * num_groups,
      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
      this->strideD_device.data(),
      strideD_host.data(),
      sizeof(typename Operator::GemmKernel::InternalStrideD) * num_groups,
      cudaMemcpyHostToDevice));
    return Status::kSuccess;
  }

  /// Constructs the arguments structure given the configuration and arguments
  Status update_arguments_base(
    OperatorArguments& operator_args,
    GemmGroupedArguments const& arguments) const {
    operator_args.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
    operator_args.problem_shape = {
      arguments.problem_count,
      arguments.problem_sizes_3x,
      arguments.pointer_mode == ScalarPointerMode::kHost ? arguments.problem_sizes_3x_host
                                                         : nullptr};

    if constexpr (IsRuntimeDataType) {
      using ArrayElementA = typename Operator::GemmKernel::CollectiveMainloop::ArrayElementA;
      using ArrayElementB = typename Operator::GemmKernel::CollectiveMainloop::ArrayElementB;
      operator_args.mainloop.ptr_A = static_cast<ArrayElementA const**>(arguments.ptr_A);
      operator_args.mainloop.ptr_B = static_cast<ArrayElementB const**>(arguments.ptr_B);

      using RuntimeDataTypeA = typename Operator::GemmKernel::CollectiveMainloop::RuntimeDataTypeA;
      using RuntimeDataTypeB = typename Operator::GemmKernel::CollectiveMainloop::RuntimeDataTypeB;

      static_assert(cute::is_same_v<RuntimeDataTypeA, RuntimeDataTypeB>, 
        "RuntimeDataTypeA/B should be identical, either MXF8F6F4Format or MXF4Format");
      using RuntimeDatatypeArg = RuntimeDataTypeA;

      auto mapping = [](RuntimeDatatype type) {
        if constexpr (cute::is_same_v<RuntimeDatatypeArg, cute::UMMA::MXF8F6F4Format>) {
          if (type == RuntimeDatatype::kE5M2) {
            return cute::UMMA::MXF8F6F4Format::E5M2;
          }
          else if (type == RuntimeDatatype::kE4M3) {
            return cute::UMMA::MXF8F6F4Format::E4M3;
          }
          else if (type == RuntimeDatatype::kE3M2) {
            return cute::UMMA::MXF8F6F4Format::E3M2;
          }
          else if (type == RuntimeDatatype::kE2M3) {
            return cute::UMMA::MXF8F6F4Format::E2M3;
          }
          else if (type == RuntimeDatatype::kE2M1) {
            return cute::UMMA::MXF8F6F4Format::E2M1;
          }
          else {
            #if defined(CUTLASS_DEBUG_TRACE_LEVEL) && CUTLASS_DEBUG_TRACE_LEVEL >= 1
            std::cerr << "Invalid input datatype specified. Running with e4m3." << std::endl;
            #endif
            return cute::UMMA::MXF8F6F4Format::E4M3;
          }
        }
        else if constexpr (cute::is_same_v<RuntimeDatatypeArg, cute::UMMA::MXF4Format>) {
          if (type == RuntimeDatatype::kE2M1) {
            return cute::UMMA::MXF4Format::E2M1;
          }
          else {
            #if defined(CUTLASS_DEBUG_TRACE_LEVEL) && CUTLASS_DEBUG_TRACE_LEVEL >= 1
            std::cerr << "Invalid input datatype specified. Running with e2m1." << std::endl;
            #endif
            return cute::UMMA::MXF4Format::E2M1;
          }
        }
        // BlockScaled kernels receive either MXF4Format or MXF8F6F4Format runtime datatype
        CUTE_GCC_UNREACHABLE;
      };
      operator_args.mainloop.runtime_data_type_a = mapping(arguments.runtime_input_datatype_a);
      operator_args.mainloop.runtime_data_type_b = mapping(arguments.runtime_input_datatype_b);
    }
    else {
      operator_args.mainloop.ptr_A = static_cast<ElementA const**>(arguments.ptr_A);
      operator_args.mainloop.ptr_B = static_cast<ElementB const**>(arguments.ptr_B);
    }
    operator_args.epilogue.ptr_C = static_cast<ElementC const**>(arguments.ptr_C);
    operator_args.epilogue.ptr_D = static_cast<ElementD**>(arguments.ptr_D);

    operator_args.mainloop.dA =
      static_cast<typename Operator::GemmKernel::InternalStrideA*>(this->strideA_device.data());
    operator_args.mainloop.dB =
      static_cast<typename Operator::GemmKernel::InternalStrideB*>(this->strideB_device.data());
    operator_args.epilogue.dC =
      static_cast<typename Operator::GemmKernel::InternalStrideC*>(this->strideC_device.data());
    operator_args.epilogue.dD =
      static_cast<typename Operator::GemmKernel::InternalStrideD*>(this->strideD_device.data());

    /* Query device SM count and max active clusters to pass onto the kernel as an argument, where needed */
    operator_args.hw_info.sm_count = arguments.sm_count;
    if constexpr (Operator::ArchTag::kMinComputeCapability >= 90) {
      operator_args.hw_info.max_active_clusters = arguments.max_active_clusters;
    }
    if constexpr (!std::is_const_v<decltype(operator_args.scheduler.max_swizzle_size)>) {
      operator_args.scheduler.max_swizzle_size = arguments.swizzle_size;
    }

    if constexpr (!std::is_const_v<decltype(operator_args.scheduler.raster_order)>) {
      using Enum_t = decltype(operator_args.scheduler.raster_order);
      switch (arguments.raster_order) {
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

    if constexpr (Operator::ArchTag::kMinComputeCapability >= 100) {
      operator_args.hw_info.cluster_shape =
        dim3(arguments.cluster_shape.m(), arguments.cluster_shape.n(), arguments.cluster_shape.k());
      operator_args.hw_info.cluster_shape_fallback = dim3(
        arguments.cluster_shape_fallback.m(),
        arguments.cluster_shape_fallback.n(),
        arguments.cluster_shape_fallback.k());
    }
    return Status::kSuccess;
  }

  template <typename FusionArgs>
  static Status update_fusion_args(FusionArgs& fusion_args, GemmGroupedArguments const& arguments) {
    if (arguments.pointer_mode == ScalarPointerMode::kHost) {
      fusion_args.alpha = *static_cast<ElementCompute const*>(arguments.alpha);
      fusion_args.beta = *static_cast<ElementCompute const*>(arguments.beta);
      fusion_args.alpha_ptr = nullptr;
      fusion_args.beta_ptr = nullptr;
      fusion_args.alpha_ptr_array = nullptr;
      fusion_args.beta_ptr_array = nullptr;

      return Status::kSuccess;
    }
    else if (arguments.pointer_mode == ScalarPointerMode::kDevice) {
      fusion_args.alpha = 0;
      fusion_args.beta = 0;
      fusion_args.alpha_ptr = static_cast<ElementCompute const*>(arguments.alpha);
      fusion_args.beta_ptr = static_cast<ElementCompute const*>(arguments.beta);
      fusion_args.alpha_ptr_array = nullptr;
      fusion_args.beta_ptr_array = nullptr;
      return Status::kSuccess;
    }
    else {
      return Status::kErrorInvalidProblem;
    }
  }
};

/// **** CAUTION ****
/// Unlike other operations, initialize() must be called when
/// certain arguments change. See initialize() for details.
template <typename Operator_>
class GroupedGemmUniversal3xOperation : public GroupedGemmOperation3xBase<Operator_> {
public:
  using Operator = Operator_;
  using OperatorArguments = typename Operator::Arguments;

public:
  GroupedGemmUniversal3xOperation(char const* name = "unknown_gemm")
      : GroupedGemmOperation3xBase<Operator_>(name) {}

  ~GroupedGemmUniversal3xOperation() override = default;

private:
  int max_active_clusters{};

protected:
  template <class FusionArgs, class = void> struct UpdateFusionArgs {
    static Status update_(FusionArgs const& fusion_args, GemmGroupedArguments const& arguments) {
      // If a custom EVT is instantiated then it is the users's responsibility
      // to ensure alpha and beta are updated appropriately
      return Status::kSuccess;
    }
  };

  template <class FusionArgs>
  struct UpdateFusionArgs<FusionArgs, cute::void_t<decltype(FusionArgs{}.alpha)>> {
    static Status update_(FusionArgs& fusion_args, GemmGroupedArguments const& arguments) {
      return GroupedGemmOperation3xBase<Operator>::update_fusion_args(fusion_args, arguments);
    }
  };

  /// Constructs the arguments structure given the configuration and arguments
  Status
  update_arguments_(OperatorArguments& operator_args, GemmGroupedArguments const* arguments) const {

    Status status = UpdateFusionArgs<decltype(operator_args.epilogue.thread)>::update_(
      operator_args.epilogue.thread,
      *arguments);
    if (status != Status::kSuccess) {
      return status;
    }

    status = this->update_arguments_base(operator_args, *arguments);
    return status;
  }

public:
  /// Returns success if the operation can proceed
  Status can_implement([[maybe_unused]] void const* configuration_ptr, void const* arguments_ptr)
    const override {
    GemmGroupedArguments const* arguments = static_cast<GemmGroupedArguments const*>(arguments_ptr);
    OperatorArguments args;
    auto status = update_arguments_(args, arguments);
    if (status != Status::kSuccess) {
      return status;
    }

    status = Operator::can_implement(args);
    return status;
  }

  /// Gets the device-side workspace
  uint64_t get_device_workspace_size(void const* configuration_ptr, void const* arguments_ptr)
    const override {

    OperatorArguments args;
    auto status = update_arguments_(args, static_cast<GemmGroupedArguments const*>(arguments_ptr));
    if (status != Status::kSuccess) {
      return 0;
    }

    uint64_t size = Operator::get_workspace_size(args);
    return size;
  }

  /// Initializes the workspace
  /// **** CAUTION ****
  /// Must be called when lda, ldb, ldc, or ldd change.
  /// The CUTLASS library stores the operations in a type-
  /// erased manifest. Therefore, only this class knows
  /// the type of strideA, strideB, strideC, and strideD.
  /// Since grouped GEMM needs to allocate storage for
  /// the strides on device, the concrete type of the stride
  /// must be known in order to copy in the correct memory
  /// layout on device.
  Status initialize(
    void const* configuration_ptr,
    void* host_workspace,
    void* device_workspace,
    cudaStream_t stream = nullptr) const override {

    Operator* op = new (host_workspace) Operator;

    auto const& config = *static_cast<GemmGroupedConfiguration const*>(configuration_ptr);
    return this->initialize_strides(config);
  }

  /// **** CAUTION ****
  /// initialize() must be called if lda, ldb, ldc, or ldd change.
  Status run(
    void const* arguments_ptr,
    void* host_workspace,
    void* device_workspace = nullptr,
    cudaStream_t stream = nullptr) const override {

    OperatorArguments operator_args;
    auto const& args = *static_cast<GemmGroupedArguments const*>(arguments_ptr);

    Status status = update_arguments_(operator_args, &args);
    if (status != Status::kSuccess) {
      return status;
    }

    Operator* op = static_cast<Operator*>(host_workspace);
    // We need to call initialize() since we have to rebuild TMA desc for every new set of args
    status = op->run(operator_args, device_workspace, stream, nullptr, args.use_pdl);
    return status;
  }

  // Set arguments that should only be set once before verifying or profiling the kernel.
  // This should encompass any expensive operations that don't vary from run to run
  // (e.g., max_active_clusters).
  Status initialize_with_arguments(void* arguments_ptr) const override {
    if constexpr (Operator::ArchTag::kMinComputeCapability < 90) {
      return Status::kSuccess;
    }

    GemmGroupedArguments* args = static_cast<GemmGroupedArguments*>(arguments_ptr);

    dim3 cluster_dims;
    if constexpr (cute::is_static_v<typename Operator::GemmKernel::ClusterShape>) {
      cluster_dims = dim3(
        cute::size<0>(typename Operator::GemmKernel::ClusterShape{}),
        cute::size<1>(typename Operator::GemmKernel::ClusterShape{}),
        cute::size<2>(typename Operator::GemmKernel::ClusterShape{})
      );
    }
    else {
      cluster_dims = dim3(
        args->cluster_shape.m(),
        args->cluster_shape.n(),
        args->cluster_shape.k()
      );      
    }

    uint32_t threads_per_block = Operator::GemmKernel::MaxThreadsPerBlock;
    void const* kernel_ptr = (void*)(device_kernel<typename Operator::GemmKernel>);
    args->max_active_clusters = cutlass::KernelHardwareInfo::query_device_max_active_clusters(
      cluster_dims,
      threads_per_block,
      kernel_ptr);

    if (args->max_active_clusters == 0) {
      std::cerr << "Max Active Clusters could not be queried. " 
                << "Falling back to heuristics mode (static cluster shape) or preferred cluster mode.\n";
    }

    return Status::kSuccess;
  }
};

template <typename Operator_>
class GroupedBlockScaledGemmUniversal3xOperation : public GroupedGemmOperation3xBase<Operator_> {
public:
  using Operator = Operator_;
  using OperatorArguments = typename Operator::Arguments;
  using ElementD = typename Operator::ElementD;
  using LayoutD = typename Operator::LayoutD;
  using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

  using CollectiveMainloop = typename Operator::CollectiveMainloop;
  using CollectiveEpilogue = typename Operator::CollectiveEpilogue;
  using ThreadEpilogueOp = typename CollectiveEpilogue::ThreadEpilogueOp;

  using ElementSFA = typename Operator::CollectiveMainloop::ElementSF;
  using ElementSFB = typename Operator::CollectiveMainloop::ElementSF;

  using TiledMma = typename Operator::CollectiveMainloop::TiledMma;
  constexpr static int SFVecSize = TiledMma::SFVecSize;


  static constexpr bool epilogue_scalefactor_generation = not cute::is_same_v<typename ThreadEpilogueOp::ElementBlockScaleFactor, void>;
  static constexpr int32_t SFD_VectorSize = epilogue_scalefactor_generation ? ThreadEpilogueOp::SFVecSize : SFVecSize;
  using ElementSFD = cute::conditional_t<epilogue_scalefactor_generation, typename ThreadEpilogueOp::ElementBlockScaleFactor, void>;
  using LayoutSFD = cute::conditional_t<epilogue_scalefactor_generation, typename ThreadEpilogueOp::GmemLayoutTagScalefactor, LayoutD>; 

  GroupedBlockScaledGemmUniversal3xOperation(char const* name = "unknown_gemm")
      : GroupedGemmOperation3xBase<Operator_>(name) {

    BlockScaleDescription block_scaled_desc{};
    block_scaled_desc.kind = OperationKind::kBlockScaledGemm;
    block_scaled_desc.SFA.element = NumericTypeMap<ElementSFA>::kId;
    block_scaled_desc.SFA.layout = LayoutTypeID::kRowMajor;
    block_scaled_desc.SFA.alignment = 128;
    block_scaled_desc.SFA.log_extent_range = 32;
    block_scaled_desc.SFA.log_stride_range = 32;

    block_scaled_desc.SFB.element = NumericTypeMap<ElementSFB>::kId;
    block_scaled_desc.SFB.layout = LayoutTypeID::kRowMajor;
    block_scaled_desc.SFB.alignment = 128;
    block_scaled_desc.SFB.log_extent_range = 32;
    block_scaled_desc.SFB.log_stride_range = 32;

    block_scaled_desc.SFMVecSize = 1;
    block_scaled_desc.SFNVecSize = 1;
    block_scaled_desc.SFKVecSize = SFVecSize;

    block_scaled_desc.SFD = make_TensorDescription<ElementSFD, LayoutSFD>(128);
    block_scaled_desc.EpilogueSFVecSize = SFD_VectorSize;

    this->description_.block_scales = block_scaled_desc;
  }

  ~GroupedBlockScaledGemmUniversal3xOperation() override = default;

  mutable CudaBuffer layout_SFA_device;
  mutable CudaBuffer layout_SFB_device;

protected:
  template <class FusionArgs, class = void> struct UpdateFusionArgs {
    static Status update_(FusionArgs const& fusion_args, GemmGroupedArguments const& arguments) {
      // If a custom EVT is instantiated then it is the users's responsibility
      // to ensure alpha and beta are updated appropriately
      return Status::kSuccess;
    }
  };

  template <class FusionArgs>
  struct UpdateFusionArgs<FusionArgs, cute::void_t<decltype(FusionArgs{}.alpha)>> {
    static Status
    update_(FusionArgs& fusion_args, GroupedGemmBlockScaledArguments const& arguments) {

      if constexpr (epilogue_scalefactor_generation) {
        fusion_args.block_scale_factor_ptr = static_cast<ElementSFD**>(arguments.SFD);
        fusion_args.norm_constant_ptr = static_cast<ElementCompute const*>(arguments.norm_constant);
      }

      return GroupedGemmOperation3xBase<Operator>::update_fusion_args(fusion_args, arguments);
    }
  };

public:
  /// Returns success if the operation can proceed
  Status can_implement([[maybe_unused]] void const* configuration_ptr, void const* arguments_ptr)
    const override {
    GroupedGemmBlockScaledArguments const* arguments =
      static_cast<GroupedGemmBlockScaledArguments const*>(arguments_ptr);
    OperatorArguments args;
    auto status = update_arguments_(args, arguments);
    if (status != Status::kSuccess) {
      return status;
    }

    status = Operator::can_implement(args);
    return status;
  }

  Status update_arguments_(
    OperatorArguments& operator_args,
    GroupedGemmBlockScaledArguments const* arguments) const {
    Status status = UpdateFusionArgs<decltype(operator_args.epilogue.thread)>::update_(
      operator_args.epilogue.thread,
      *arguments);
    if (status != Status::kSuccess) {
      return status;
    }

    operator_args.mainloop.ptr_SFA =
      static_cast<const typename Operator::GemmKernel::ElementSF**>(arguments->SFA);
    operator_args.mainloop.ptr_SFB =
      static_cast<const typename Operator::GemmKernel::ElementSF**>(arguments->SFB);

    operator_args.mainloop.layout_SFA =
      static_cast<typename CollectiveMainloop::InternalLayoutSFA*>(this->layout_SFA_device.data());
    operator_args.mainloop.layout_SFB =
      static_cast<typename CollectiveMainloop::InternalLayoutSFB*>(this->layout_SFB_device.data());

    return this->update_arguments_base(operator_args, *arguments);
  }

  uint64_t get_device_workspace_size(void const* configuration_ptr, void const* arguments_ptr)
    const override {

    OperatorArguments args;
    auto status =
      update_arguments_(args, static_cast<GroupedGemmBlockScaledArguments const*>(arguments_ptr));
    if (status != Status::kSuccess) {
      return 0;
    }

    uint64_t size = Operator::get_workspace_size(args);
    return size;
  }

  /// Initializes the workspace
  /// **** CAUTION ****
  /// Must be called when lda, ldb, ldc, or ldd change.
  /// The CUTLASS library stores the operations in a type-
  /// erased manifest. Therefore, only this class knows
  /// the type of strideA, strideB, strideC, and strideD.
  /// Since grouped GEMM needs to allocate storage for
  /// the strides on device, the concrete type of the stride
  /// must be known in order to copy in the correct memory
  /// layout on device.
  Status initialize(
    void const* configuration_ptr,
    void* host_workspace,
    void* device_workspace,
    cudaStream_t stream = nullptr) const override {

    auto const& config = *static_cast<GemmGroupedConfiguration const*>(configuration_ptr);
    auto status = this->initialize_strides(config);
    if (status != Status::kSuccess) {
      return status;
    }

    auto num_groups = config.problem_count;
    this->layout_SFA_device =
      CudaBuffer(sizeof(typename CollectiveMainloop::InternalLayoutSFA) * num_groups);
    this->layout_SFB_device =
      CudaBuffer(sizeof(typename CollectiveMainloop::InternalLayoutSFB) * num_groups);
    auto layout_SFA_host = std::vector<typename CollectiveMainloop::InternalLayoutSFA>(num_groups);
    auto layout_SFB_host = std::vector<typename CollectiveMainloop::InternalLayoutSFB>(num_groups);

    for (int group_idx = 0; group_idx < num_groups; group_idx++) {
      auto const& shape = config.problem_sizes_3x_host[group_idx];
      auto M = get<0>(shape);
      auto N = get<1>(shape);
      auto K = get<2>(shape);

      auto layout_SFA = CollectiveMainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
      auto layout_SFB = CollectiveMainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
      layout_SFA_host[group_idx] = layout_SFA;
      layout_SFB_host[group_idx] = layout_SFB;
    }

    CUDA_CHECK(cudaMemcpy(
      this->layout_SFA_device.data(),
      layout_SFA_host.data(),
      sizeof(typename CollectiveMainloop::InternalLayoutSFA) * num_groups,
      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
      this->layout_SFB_device.data(),
      layout_SFB_host.data(),
      sizeof(typename CollectiveMainloop::InternalLayoutSFB) * num_groups,
      cudaMemcpyHostToDevice));

    Operator* op = new (host_workspace) Operator;
    return status;
  }

  /// **** CAUTION ****
  /// initialize() must be called if lda, ldb, ldc, or ldd change.
  Status run(
    void const* arguments_ptr,
    void* host_workspace,
    void* device_workspace = nullptr,
    cudaStream_t stream = nullptr) const override {

    OperatorArguments operator_args;
    auto const& args = *static_cast<GroupedGemmBlockScaledArguments const*>(arguments_ptr);

    Status status = update_arguments_(operator_args, &args);
    if (status != Status::kSuccess) {
      return status;
    }

    Operator* op = static_cast<Operator*>(host_workspace);
    status = op->run(operator_args, device_workspace, stream, nullptr);
    return status;
  }
};

template <typename Operator_>
class GroupedBlockwiseGemmUniversal3xOperation : public GroupedGemmOperation3xBase<Operator_> {
public:
  using Operator = Operator_;
  using OperatorArguments = typename Operator::Arguments;
  using ElementD = typename Operator::ElementD;
  using LayoutD = typename Operator::LayoutD;
  using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

  using CollectiveMainloop = typename Operator::CollectiveMainloop;
  using CollectiveEpilogue = typename Operator::CollectiveEpilogue;
  using ThreadEpilogueOp = typename CollectiveEpilogue::ThreadEpilogueOp;

  using ElementSFA = typename Operator::ElementAccumulator;
  using ElementSFB = typename Operator::ElementAccumulator;

  using TiledMma = typename Operator::CollectiveMainloop::TiledMma;

  GroupedBlockwiseGemmUniversal3xOperation(char const* name = "unknown_gemm")
      : GroupedGemmOperation3xBase<Operator_>(name) {

    BlockScaleDescription blockwise_desc{};
    blockwise_desc.kind = OperationKind::kBlockwiseGemm;
    blockwise_desc.SFA.element = NumericTypeMap<ElementSFA>::kId;
    blockwise_desc.SFA.layout = size<0,1>(typename CollectiveMainloop::InternalLayoutSFA{}.stride()) == 1 ? 
        LayoutTypeID::kColumnMajor : LayoutTypeID::kRowMajor;
    blockwise_desc.SFA.alignment = CollectiveMainloop::AlignmentSFA;
    blockwise_desc.SFA.log_extent_range = 32;
    blockwise_desc.SFA.log_stride_range = 32;

    blockwise_desc.SFB.element = NumericTypeMap<ElementSFB>::kId;
    blockwise_desc.SFB.layout = size<0,1>(typename CollectiveMainloop::InternalLayoutSFB{}.stride()) == 1 ? 
        LayoutTypeID::kRowMajor : LayoutTypeID::kColumnMajor;
    blockwise_desc.SFB.alignment = CollectiveMainloop::AlignmentSFA;
    blockwise_desc.SFB.log_extent_range = 32;
    blockwise_desc.SFB.log_stride_range = 32;

    blockwise_desc.SFMVecSize = Operator::CollectiveMainloop::ScaleGranularityM;
    blockwise_desc.SFNVecSize = Operator::CollectiveMainloop::ScaleGranularityN;
    blockwise_desc.SFKVecSize = Operator::CollectiveMainloop::ScaleGranularityK;

    blockwise_desc.EpilogueSFVecSize = 0;

    this->description_.block_scales = blockwise_desc;
  }

  ~GroupedBlockwiseGemmUniversal3xOperation() override = default;

  mutable CudaBuffer layout_SFA_device;
  mutable CudaBuffer layout_SFB_device;

protected:
  template <class FusionArgs, class = void> struct UpdateFusionArgs {
    static Status update_(FusionArgs const& fusion_args, GemmGroupedArguments const& arguments) {
      // If a custom EVT is instantiated then it is the users's responsibility
      // to ensure alpha and beta are updated appropriately
      return Status::kSuccess;
    }
  };

  template <class FusionArgs>
  struct UpdateFusionArgs<FusionArgs, cute::void_t<decltype(FusionArgs{}.alpha)>> {
    static Status
    update_(FusionArgs& fusion_args, GroupedGemmBlockwiseArguments const& arguments) {
      return GroupedGemmOperation3xBase<Operator>::update_fusion_args(fusion_args, arguments);
    }
  };

public:
  /// Returns success if the operation can proceed
  Status can_implement([[maybe_unused]] void const* configuration_ptr, void const* arguments_ptr)
    const override {
    GroupedGemmBlockwiseArguments const* arguments =
      static_cast<GroupedGemmBlockwiseArguments const*>(arguments_ptr);
    OperatorArguments args;
    auto status = update_arguments_(args, arguments);
    if (status != Status::kSuccess) {
      return status;
    }

    status = Operator::can_implement(args);
    return status;
  }

  Status update_arguments_(
    OperatorArguments& operator_args,
    GroupedGemmBlockwiseArguments const* arguments) const {
    Status status = UpdateFusionArgs<decltype(operator_args.epilogue.thread)>::update_(
      operator_args.epilogue.thread,
      *arguments);
    if (status != Status::kSuccess) {
      return status;
    }

    operator_args.mainloop.ptr_SFA =
      static_cast<const typename Operator::GemmKernel::ElementAccumulator**>(arguments->SFA);
    operator_args.mainloop.ptr_SFB =
      static_cast<const typename Operator::GemmKernel::ElementAccumulator**>(arguments->SFB);

    operator_args.mainloop.layout_SFA =
      static_cast<typename CollectiveMainloop::InternalLayoutSFA*>(this->layout_SFA_device.data());
    operator_args.mainloop.layout_SFB =
      static_cast<typename CollectiveMainloop::InternalLayoutSFB*>(this->layout_SFB_device.data());

    return this->update_arguments_base(operator_args, *arguments);
  }

  uint64_t get_device_workspace_size(void const* configuration_ptr, void const* arguments_ptr)
    const override {

    OperatorArguments args;
    auto status =
      update_arguments_(args, static_cast<GroupedGemmBlockwiseArguments const*>(arguments_ptr));
    if (status != Status::kSuccess) {
      return 0;
    }

    uint64_t size = Operator::get_workspace_size(args);
    return size;
  }

  /// Initializes the workspace
  /// **** CAUTION ****
  /// Must be called when lda, ldb, ldc, or ldd change.
  /// The CUTLASS library stores the operations in a type-
  /// erased manifest. Therefore, only this class knows
  /// the type of strideA, strideB, strideC, and strideD.
  /// Since grouped GEMM needs to allocate storage for
  /// the strides on device, the concrete type of the stride
  /// must be known in order to copy in the correct memory
  /// layout on device.
  Status initialize(
    void const* configuration_ptr,
    void* host_workspace,
    void* device_workspace,
    cudaStream_t stream = nullptr) const override {

    auto const& config = *static_cast<GemmGroupedConfiguration const*>(configuration_ptr);
    auto status = this->initialize_strides(config);
    if (status != Status::kSuccess) {
      return status;
    }

    auto num_groups = config.problem_count;
    this->layout_SFA_device =
      CudaBuffer(sizeof(typename CollectiveMainloop::InternalLayoutSFA) * num_groups);
    this->layout_SFB_device =
      CudaBuffer(sizeof(typename CollectiveMainloop::InternalLayoutSFB) * num_groups);
    auto layout_SFA_host = std::vector<typename CollectiveMainloop::InternalLayoutSFA>(num_groups);
    auto layout_SFB_host = std::vector<typename CollectiveMainloop::InternalLayoutSFB>(num_groups);

    for (int group_idx = 0; group_idx < num_groups; group_idx++) {
      auto const& shape = config.problem_sizes_3x_host[group_idx];
      auto M = get<0>(shape);
      auto N = get<1>(shape);
      auto K = get<2>(shape);

      auto layout_SFA = CollectiveMainloop::ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
      auto layout_SFB = CollectiveMainloop::ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
      layout_SFA_host[group_idx] = layout_SFA;
      layout_SFB_host[group_idx] = layout_SFB;
    }

    CUDA_CHECK(cudaMemcpy(
      this->layout_SFA_device.data(),
      layout_SFA_host.data(),
      sizeof(typename CollectiveMainloop::InternalLayoutSFA) * num_groups,
      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
      this->layout_SFB_device.data(),
      layout_SFB_host.data(),
      sizeof(typename CollectiveMainloop::InternalLayoutSFB) * num_groups,
      cudaMemcpyHostToDevice));

    Operator* op = new (host_workspace) Operator;
    return status;
  }

  /// **** CAUTION ****
  /// initialize() must be called if lda, ldb, ldc, or ldd change.
  Status run(
    void const* arguments_ptr,
    void* host_workspace,
    void* device_workspace = nullptr,
    cudaStream_t stream = nullptr) const override {

    OperatorArguments operator_args;
    auto const& args = *static_cast<GroupedGemmBlockwiseArguments const*>(arguments_ptr);

    Status status = update_arguments_(operator_args, &args);
    if (status != Status::kSuccess) {
      return status;
    }

    Operator* op = static_cast<Operator*>(host_workspace);
    status = op->run(operator_args, device_workspace, stream, nullptr);
    return status;
  }
};


} // namespace cutlass::library
