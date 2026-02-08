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
/*!
  \file Distributed GEMM Device Adapter

  Sets up local GEMM stages, the cuda graph, manages buffer and barrier spaces,
  and maps arguments to per-stage arguments.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/experimental/distributed/device/full_barrier.hpp"
#include "cutlass/experimental/distributed/device/detail.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass::distributed::device {

template <class GemmKernel_>
class DistributedGemmUniversalAdapter {
public:
  using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_>;
  using GemmKernel = GemmKernel_;
  using TileShape = typename GemmKernel::TileShape;
  using ElementA = typename GemmKernel::ElementA;
  using ElementB = typename GemmKernel::ElementB;
  using ElementC = typename GemmKernel::ElementC;
  using ElementD = typename GemmKernel::ElementD;
  using ElementAccumulator = typename GemmKernel::ElementAccumulator;
  using DispatchPolicy = typename GemmKernel::DispatchPolicy;
  using CollectiveMainloop = typename GemmKernel::CollectiveMainloop;
  using CollectiveEpilogue = typename GemmKernel::CollectiveEpilogue;

  // "Inherit" type decls and static values from device GEMM
  using LayoutA = typename DeviceGemm::LayoutA;
  using LayoutB = typename DeviceGemm::LayoutB;
  using LayoutC = typename DeviceGemm::LayoutC;
  using LayoutD = typename DeviceGemm::LayoutD;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

  static bool const kEnableCudaHostAdapter = DeviceGemm::kEnableCudaHostAdapter;

  static ComplexTransform const kTransformA = DeviceGemm::kTransformA;
  static ComplexTransform const kTransformB = DeviceGemm::kTransformB;

  using MathOperator = typename DeviceGemm::MathOperator;
  using OperatorClass = typename DeviceGemm::OperatorClass;
  using ArchTag = typename DeviceGemm::ArchTag;

  using ThreadblockSwizzle = typename DeviceGemm::ThreadblockSwizzle;
  using ThreadblockShape = typename DeviceGemm::ThreadblockShape;
  using ClusterShape = typename DeviceGemm::ClusterShape;
  using InstructionShape = typename DeviceGemm::InstructionShape;

  static int const kThreadCount = DeviceGemm::kThreadCount;
  static constexpr int WarpsInMma = DeviceGemm::WarpsInMma;
  static constexpr int WarpsInMmaM = DeviceGemm::WarpsInMmaM;
  static constexpr int WarpsInMmaN = DeviceGemm::WarpsInMmaN;

  using WarpCount = typename DeviceGemm::WarpCount;
  using WarpShape = typename DeviceGemm::WarpShape;

  static int constexpr kStages = DeviceGemm::kStages;

  static int constexpr kAlignmentA = DeviceGemm::kAlignmentA;
  static int constexpr kAlignmentB = DeviceGemm::kAlignmentB;
  static int constexpr kAlignmentC = DeviceGemm::kAlignmentC;
  static int constexpr kAlignmentD = DeviceGemm::kAlignmentD;

  using EpilogueOutputOp = typename DeviceGemm::EpilogueOutputOp;

  static int constexpr kSplitKAlignment = DeviceGemm::kSplitKAlignment;

  // Distributed GEMM types and defs
  using DistSchedule = typename GemmKernel::DistSchedule;
  static constexpr bool HasMemcpy = DistSchedule::HasMemcpy;
  using TP = typename DistSchedule::TP;
  static constexpr int TP_ = TP{};
  using ElementFlag = typename GemmKernel::ElementFlag;
  using ElementBarrier = uint32_t;

  using BufferHelper = detail::DistGemmBufferHelper<
    DistSchedule,
    ElementA,
    ElementB,
    ElementC,
    ElementD>;

  /// Argument structure
  using Arguments = typename GemmKernel::BaseArguments;
  using DistributedArguments = typename GemmKernel::DistributedArguments;
  using PackedArguments = typename GemmKernel::PackedArguments;

  /// Argument structure: Kernel API
  using Params = typename GemmKernel::PackedParams;

  struct DistributedGemmState {
    int device_idx;

    Params params_array[TP_];

    cudaGraph_t graph;
    cudaGraphExec_t graph_executable;

    bool graph_created = false;
    bool graph_instantiated = false;

    void * memcpy_source_ptr_array[TP_];
    void const * memcpy_remote_ptr_array[TP_];
    size_t memcpy_bytes[TP_];

    cutlass::Array<ElementBarrier*, TP_> device_barrier_ptrs;

    bool is_initialized = false;
  };

private:

  DistributedGemmState state_;

public:

  bool is_initialized() {
    return state_.is_initialized && state_.graph_created && state_.graph_instantiated;
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status
  can_implement(Arguments const& args) {
    if (args.epilogue.thread.beta != 0.0 && DistSchedule::RemoteC) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Selected TP uses Remote C to communicate " <<
          "partial results, which do not support non-zero values for beta yet " <<
          "(epilogue must be sourceless.)\n");
      return Status::kInvalid;
    }

    if (not DistSchedule::can_implement_global(args.problem_shape)) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem shape not divisible by TP.\n");
      return Status::kInvalid;
    }

    Arguments args_copy = args;
    args_copy.problem_shape = DistSchedule::get_local_gemm_shape(args.problem_shape);
    for (int iteration = 0; iteration < TP_; ++iteration) {
      if (not GemmKernel::can_implement(args_copy)) {
        return Status::kInvalid;
      }
    }
    return Status::kSuccess;
  }

  /// Gets buffer space size
  static size_t
  get_buffer_space_size(Arguments const& args) {
    size_t buffer_bytes = 0;

    buffer_bytes = BufferHelper::get_buffer_size(args.problem_shape);
    buffer_bytes = round_nearest(buffer_bytes, MinWorkspaceAlignment);

    return buffer_bytes;
  }

  static auto
  get_tensor_A_for_iter(Arguments const* args_array, void** buffer_space, int device_idx, int iteration) {
    auto args = args_array[device_idx];
    auto tensor_A = make_tensor(args.mainloop.ptr_A, make_layout(
          DistSchedule::get_local_a_shape(args.problem_shape),
          args.mainloop.dA));

    uint8_t* tensor_buffer = reinterpret_cast<uint8_t*>(buffer_space[device_idx]) +
      BufferHelper::get_buffer_offset_A(args.problem_shape);

    return DistSchedule::get_tensor_A(tensor_A, tensor_buffer, device_idx, iteration);
  }

  static auto
  get_tensor_B_for_iter(Arguments const* args_array, void** buffer_space, int device_idx, int iteration) {
    auto args = args_array[device_idx];
    auto tensor_B = make_tensor(args.mainloop.ptr_B, make_layout(
          DistSchedule::get_local_b_shape(args.problem_shape),
          args.mainloop.dB));

    uint8_t* tensor_buffer = reinterpret_cast<uint8_t*>(buffer_space[device_idx]) +
      BufferHelper::get_buffer_offset_B(args.problem_shape);

    return DistSchedule::get_tensor_B(tensor_B, tensor_buffer, device_idx, iteration);
  }

  static auto
  get_tensor_C_for_iter(Arguments const* args_array, void** buffer_space, int device_idx, int iteration) {
    auto args = args_array[device_idx];
    auto tensor_C = make_tensor(args.epilogue.ptr_C, make_layout(
          DistSchedule::get_local_c_shape(args.problem_shape),
          args.epilogue.dC));

    auto peer_idx_iter = DistSchedule::get_remote_peer_id(device_idx, iteration);
    void* buffer_ptr = DistSchedule::RemoteC ? buffer_space[peer_idx_iter] : buffer_space[device_idx];

    uint8_t* tensor_buffer = reinterpret_cast<uint8_t*>(buffer_ptr) +
      BufferHelper::get_buffer_offset_C(args.problem_shape);

    return DistSchedule::get_tensor_C(tensor_C, tensor_buffer, device_idx, iteration);
  }

  static auto
  get_tensor_D_for_iter(Arguments const* args_array, void** buffer_space, int device_idx, int iteration) {
    auto args = args_array[device_idx];
    auto tensor_D = make_tensor(args.epilogue.ptr_D, make_layout(
          DistSchedule::get_local_d_shape(args.problem_shape),
          args.epilogue.dD));

    // support remoteD
    uint8_t* tensor_buffer = reinterpret_cast<uint8_t*>(buffer_space[device_idx]) +
      BufferHelper::get_buffer_offset_D(args.problem_shape);

    return DistSchedule::get_tensor_D(tensor_D, tensor_buffer, device_idx, iteration);
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_bytes = 0;

    workspace_bytes = get_buffer_space_size(args);

    for (int iteration = 0; iteration < TP_; ++iteration) {
      // NOTE: assumes underlying kernels align up to alignment requirements on their own,
      // and that the alignment requirements of the individual kernels match.
      workspace_bytes += GemmKernel::get_workspace_size(args);
    }

    return workspace_bytes;
  }

  static size_t
  get_barrier_bytes() {
    return round_nearest(sizeof(ElementBarrier), 32);
  }

  static size_t
  get_flag_bytes() {
    return round_nearest(sizeof(ElementFlag) * TP_, 32);
  }

  static void *
  exclusive_workspace_ptr_to_flag_ptr(void * exclusive_workspace_ptr, int iteration) {
    return static_cast<void*>(
        static_cast<uint8_t*>(exclusive_workspace_ptr) + 
        get_barrier_bytes() + 
        (sizeof(ElementFlag) * iteration));
  }

  static size_t
  get_exclusive_workspace_size() {
    return get_barrier_bytes() + get_flag_bytes();
  }

  /// Initializes GEMM state from arguments.
  Status
  initialize(
    Arguments const* args,
    void** workspace_ptrs,
    void** exclusive_workspace_ptrs,
    int device_idx,
    cudaStream_t stream = nullptr,
    bool launch_with_pdl = false) {

    CUTLASS_TRACE_HOST("DistributedGemm::initialize() - stream: " << (stream ? "non-null" : "null"));

    state_.device_idx = device_idx;

    for (int device = 0; device < TP_; ++device) {
      state_.device_barrier_ptrs[device] = reinterpret_cast<ElementBarrier*>(exclusive_workspace_ptrs[device]);
    }

    // Zero out exclusive workspace
    zero_workspace(exclusive_workspace_ptrs[device_idx], get_exclusive_workspace_size(), stream, nullptr);

    for (int iteration = 0; iteration < TP_; ++iteration) {

      size_t workspace_iteration_offset = GemmKernel::get_workspace_size(args[device_idx]);
      uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace_ptrs[device_idx]) + 
        get_buffer_space_size(args[device_idx]) + 
        (iteration * workspace_iteration_offset);

      void * workspace_iter = reinterpret_cast<void*>(workspace_ptr);
      void** buffer_space = workspace_ptrs;

      // Set up GEMM arguments for the current stage/iteration
      auto tensor_a_iter = get_tensor_A_for_iter(args, buffer_space, device_idx, iteration);
      auto tensor_b_iter = get_tensor_B_for_iter(args, buffer_space, device_idx, iteration);
      auto tensor_c_iter = get_tensor_C_for_iter(args, buffer_space, device_idx, iteration);
      auto tensor_d_iter = get_tensor_D_for_iter(args, buffer_space, device_idx, iteration);

      Arguments base_args = args[device_idx];
      base_args.problem_shape = DistSchedule::get_local_gemm_shape(args[device_idx].problem_shape);
      base_args.mainloop = {
        reinterpret_cast<const ElementA*>(tensor_a_iter.data()),
        tensor_a_iter.stride(),
        reinterpret_cast<const ElementB*>(tensor_b_iter.data()),
        tensor_b_iter.stride()
      };
      base_args.epilogue = {
        base_args.epilogue.thread,
        reinterpret_cast<const ElementC*>(tensor_c_iter.data()),
        tensor_c_iter.stride(),
        reinterpret_cast<ElementD*>(tensor_d_iter.data()),
        tensor_d_iter.stride()
      };

      if constexpr (DistSchedule::RemoteC) {
        if (iteration > 0) {
          base_args.epilogue.thread.beta = 1.0;
        }
        else if (iteration == 0){
          base_args.epilogue.thread.beta = 0.0;
        }
      }

      auto [left_peer_idx, right_peer_idx] = DistSchedule::get_peers_for_device(device_idx);
      auto flag_peer_idx = DistSchedule::KernelWritesArrivalFlag ? right_peer_idx : device_idx;

      void * self_flag_ptr = exclusive_workspace_ptr_to_flag_ptr(exclusive_workspace_ptrs[device_idx], iteration);
      void * peer_flag_ptr = exclusive_workspace_ptr_to_flag_ptr(exclusive_workspace_ptrs[flag_peer_idx], iteration);

      DistributedArguments distributed_args = {
        device_idx,
        iteration,
        self_flag_ptr,
        peer_flag_ptr
      };
      PackedArguments args_iter = {base_args, distributed_args};

      // Initialize the workspace
      Status status = GemmKernel::initialize_workspace(args_iter, workspace_iter, stream);
      if (status != Status::kSuccess) {
        return status;
      }

      // Initialize the Params structure
      state_.params_array[iteration] = GemmKernel::to_underlying_arguments(args_iter, workspace_iter);

      // Set up peer buffer ptrs
      if (iteration > 0 && HasMemcpy) {
        auto peer_idx_iter = DistSchedule::get_remote_peer_id(device_idx, iteration);

        void * local_ptr_itr = nullptr;
        void const * remote_ptr_itr = nullptr;
        size_t local_size = 0;
        size_t remote_size = 0;

        static_assert(not DistSchedule::HasMemcpy || (
              DistSchedule::MemcpyA || DistSchedule::MemcpyB),
            "Expected to either memcpy A or B when scheduler requires memcpy.");
        if constexpr (DistSchedule::MemcpyA) {
          local_size = cute::cosize(tensor_a_iter.layout()) * sizeof(ElementA);
          local_ptr_itr = reinterpret_cast<void*>(tensor_a_iter.data());

          // Copy peer's slice in the first iteration (direct access memcpy instead of logical ring)
          auto remote_tensor_iter = get_tensor_A_for_iter(args, buffer_space, peer_idx_iter, 0);
          remote_ptr_itr = reinterpret_cast<void const*>(remote_tensor_iter.data());
          remote_size = cute::cosize(remote_tensor_iter.layout()) * sizeof(ElementA);
        }
        else if constexpr (DistSchedule::MemcpyB) {
          local_size = cute::cosize(tensor_b_iter.layout()) * sizeof(ElementB);
          local_ptr_itr = reinterpret_cast<void*>(tensor_b_iter.data());

          // Copy peer's slice in the first iteration (direct access memcpy instead of logical ring)
          auto remote_tensor_iter = get_tensor_B_for_iter(args, buffer_space, peer_idx_iter, 0);
          remote_ptr_itr = reinterpret_cast<void const*>(remote_tensor_iter.data());
          remote_size = cute::cosize(remote_tensor_iter.layout()) * sizeof(ElementB);
        }

        assert(local_size == remote_size && local_size > 0);

        state_.memcpy_source_ptr_array[iteration] = local_ptr_itr;
        state_.memcpy_remote_ptr_array[iteration] = remote_ptr_itr;
        state_.memcpy_bytes[iteration] = local_size;
      }
    }

    //
    // Account for dynamic smem capacity if needed
    //
    int smem_size = GemmKernel::SharedStorageSize;

    if (smem_size >= (48 << 10)) {
      CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
      cudaError_t result = cudaFuncSetAttribute(
          device_kernel<GemmKernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);
      if (cudaSuccess != result) {
        result = cudaGetLastError(); // to clear the error bit
        CUTLASS_TRACE_HOST("  cudaFuncSetAttribute() returned error: " << cudaGetErrorString(result));
        return Status::kErrorInternal;
      }
    }

    state_.is_initialized = true;

    // Instantiate graph
    Status status = construct_graph(launch_with_pdl);
    if (status != Status::kSuccess) {
      return status;
    }

    return Status::kSuccess;
  }

  Status
  construct_graph(bool launch_with_pdl) {
#if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 6))
    Status status = Status::kSuccess;

    // Destroy existing graph, if created
    if (state_.graph_created) {
      status = detail::check_cuda_status(cudaGraphDestroy(state_.graph));
      if (status != Status::kSuccess) {
        return status;
      }
    }

    state_.graph_created = true;

    cudaGraphNode_t full_barrier_node;

    // Create dummy stream
    cudaStream_t stream;
    status = detail::check_cuda_status(cudaStreamCreate(&stream));
    if (status != Status::kSuccess) {
      return status;
    }

    // Create graph
    status = detail::check_cuda_status(cudaGraphCreate(&state_.graph, 0));
    if (status != Status::kSuccess) {
      return status;
    }

    // 1. Full barrier node
    status = detail::check_cuda_status(cudaStreamBeginCaptureToGraph(
          stream,
          state_.graph,
          nullptr, nullptr, 0,
          cudaStreamCaptureModeRelaxed));
    if (status != Status::kSuccess) {
      return status;
    }

    cutlass::Array<ElementFlag*, TP_> self_flag_ptrs;
    for (int iteration = 0; iteration < TP_; ++iteration) {
      self_flag_ptrs[iteration] = state_.params_array[iteration].distributed.self_flag_ptr_;
    }

    launch_full_barrier<TP_, ElementBarrier, TP_, ElementFlag>(
        state_.device_barrier_ptrs, self_flag_ptrs, state_.device_idx, stream, launch_with_pdl);

    status = detail::check_cuda_status(cudaStreamEndCapture(stream, &state_.graph));
    if (status != Status::kSuccess) {
      return status;
    }

    size_t num_nodes;
    status = detail::check_cuda_status(cudaGraphGetNodes(state_.graph, nullptr, &num_nodes));
    if (status != Status::kSuccess) {
      return status;
    }
    if (num_nodes != 1) {
      CUTLASS_TRACE_HOST("  construct_graph() failure: expected a single node in the graph, got " << num_nodes << ".");
      return Status::kErrorInternal;
    }
    if (status != Status::kSuccess) {
      return status;
    }
    status = detail::check_cuda_status(cudaGraphGetNodes(state_.graph, &full_barrier_node, &num_nodes));
    if (status != Status::kSuccess) {
      return status;
    }

    // 2. Optional mem copy branch
    if constexpr (HasMemcpy) {

      status = detail::check_cuda_status(cudaStreamBeginCaptureToGraph(
            stream,
            state_.graph,
            &full_barrier_node,
            /* dependencyData = */ nullptr,
            1,
            cudaStreamCaptureModeRelaxed));

      if (status != Status::kSuccess) {
        return status;
      }

      // No copies for first iter; we assume the data is already there.
      for (int iteration = 1; iteration < TP_; ++iteration) {

        status = detail::check_cuda_status(cudaMemcpyAsync(
              state_.memcpy_source_ptr_array[iteration],
              state_.memcpy_remote_ptr_array[iteration],
              state_.memcpy_bytes[iteration],
              cudaMemcpyDeviceToDevice, stream));

        if (status != Status::kSuccess) {
          return status;
        }

        // Set flag to non zero
        status = detail::check_cuda_status(cudaMemsetAsync(
              reinterpret_cast<void *>(state_.params_array[iteration].distributed.peer_flag_ptr_),
              0b11111111,
              sizeof(ElementFlag),
              stream));

        if (status != Status::kSuccess) {
          return status;
        }
      }

      status = detail::check_cuda_status(cudaStreamEndCapture(stream, &state_.graph));
      if (status != Status::kSuccess) {
        return status;
      }
    }

    // 3. Run local GEMMs
    // 3.1. Create edge between full barrier and the correct gemm stage/iteration
    cudaGraphEdgeData barrier_to_gemm_edge = {};
    barrier_to_gemm_edge.from_port = HasMemcpy ? cudaGraphKernelNodePortLaunchCompletion: cudaGraphKernelNodePortProgrammatic;
    barrier_to_gemm_edge.type = cudaGraphDependencyTypeProgrammatic;

    status = detail::check_cuda_status(cudaStreamBeginCaptureToGraph(
          stream,
          state_.graph,
          &full_barrier_node,
          /* dependencyData = */ &barrier_to_gemm_edge,
          1,
          cudaStreamCaptureModeRelaxed));
    if (status != Status::kSuccess) {
      return status;
    }

    for (int iteration = 0; iteration < TP_; ++iteration) {
      status = DeviceGemm::run(
            state_.params_array[iteration],
            stream,
            /* cuda_adapter = */ nullptr,
            /* launch_with_pdl = */ launch_with_pdl);

      if (status != Status::kSuccess) {
        return status;
      }
    }

    status = detail::check_cuda_status(cudaStreamEndCapture(stream, &state_.graph));
    if (status != Status::kSuccess) {
      return status;
    }

    // 4. Cleanup.
    //// Destroy dummy stream
    status = detail::check_cuda_status(cudaStreamDestroy(stream));
    if (status != Status::kSuccess) {
      return status;
    }

    // 5. Instantiate graph
    status = detail::check_cuda_status(cudaGraphInstantiate(
          &state_.graph_executable,
          state_.graph,
          /* flags = */ 0));
    if (status != Status::kSuccess) {
      return status;
    }
    state_.graph_instantiated = true;

    return Status::kSuccess;
#else
      CUTLASS_TRACE_HOST("  construct_graph() failure: target was compiled with an incompatible " <<
          "version of the CUDA toolkit. Please compile Distributed GEMM with CUDA toolkit 12.4 or later.");
      return Status::kErrorInternal;
#endif
  }

  Status
  update(Arguments const& args, void* workspace = nullptr) {
    CUTLASS_TRACE_HOST("  DistributedGemm does not support updating arguments yet.");
    return Status::kErrorInternal;
  }

  // NOTE: the interface for run() is different in Distributed Gemm:
  //   1. launch_with_pdl is specified in `initialize`, where the cuda graph is being constructed,
  //   2. the state of distributed gemm is an array of params for different iterations, and a
  //      cuda graph.
  //   3. Custom cuda adapters aren't supported for simplicity.
  static Status
  run(DistributedGemmState& state,
      cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("DistributedGemm::run()");

    if (not state.is_initialized) {
      CUTLASS_TRACE_HOST("  Distributed gemm was not initialized. Did you forget to call initialize()?");
      return Status::kErrorInternal;
    }

    if (not state.graph_instantiated) {
      CUTLASS_TRACE_HOST("  Distributed gemm graph was not instantiated. Did you forget to call initialize()/construct_graph()?");
      return Status::kErrorInternal;
    }

    cudaError_t result = cudaGraphLaunch(state.graph_executable, stream);
    if (cudaSuccess != result) {
      result = cudaGetLastError(); // to clear the error bit
      CUTLASS_TRACE_HOST("  cudaGraphLaunch() returned error: " << cudaGetErrorString(result));
      return Status::kErrorInternal;
    }

    return Status::kSuccess;
  }

  //
  // Non-static launch overloads that first create and set the internal params struct of this kernel handle.
  //

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status
  run(
    cudaStream_t stream = nullptr) {
    return run(state_, stream);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status
  operator()(cudaStream_t stream = nullptr) {
    return run(state_, stream);
  }

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  Status
  run(
    Arguments const* args,
    void** workspace_ptrs,
    void** exclusive_workspace_ptrs,
    int device_idx,
    cudaStream_t stream = nullptr) {
    Status status = initialize(
        args,
        workspace_ptrs,
        exclusive_workspace_ptrs,
        device_idx,
        stream);

    if (Status::kSuccess == status) {
      status = run(stream);
    }
    return status;
  }

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  Status
  operator()(
    Arguments const* args,
    void** workspace_ptrs,
    void** exclusive_workspace_ptrs,
    int device_idx,
    cudaStream_t stream = nullptr) {
    return run(
        args,
        workspace_ptrs,
        exclusive_workspace_ptrs,
        device_idx,
        stream);
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::distributed::device

////////////////////////////////////////////////////////////////////////////////
