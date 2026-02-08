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
  \file Distributed GEMM Kernel Wrapper

  Prepends CUTLASS 3 GEMM kernels with barriers and other necessary instructions to exectue
  a Distributed GEMM stage.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/grid_dependency_control.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/experimental/distributed/kernel/detail.hpp"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::distributed::kernel {

namespace detail {

// Allow all CUTLASS 3.X GEMM kernels
template <typename GemmKernel_>
struct SupportsDistributedGemm: cutlass::gemm::detail::IsCutlass3GemmKernel<GemmKernel_> {};

} // namespace detail

/*!
  DistributedGemmKernelWrapper is a wrapper around a GEMM kernel.

  Depending on the underlying distribution policy/schedule, it prepends the underlying local GEMM
  kernel with a few additional instructions that gate the execution of the GEMM on buffers being
  ready for stages/iterations > 0.
*/

template <class GemmKernel_, class DistSchedule_, class Enable = void>
struct DistributedGemmKernelWrapper;

template <class GemmKernel_, class DistSchedule_>
struct DistributedGemmKernelWrapper<
  GemmKernel_,
  DistSchedule_,
  cute::enable_if_t<detail::SupportsDistributedGemm<GemmKernel_>::value>
  >: GemmKernel_
{
  using DistSchedule = DistSchedule_;
  using TP = typename DistSchedule::TP;

  static constexpr bool KernelWritesArrivalFlag = DistSchedule::KernelWritesArrivalFlag;

  using BaseKernel = GemmKernel_;
  using BaseArguments = typename BaseKernel::Arguments;
  using BaseParams = typename BaseKernel::Params;

  //static_assert(BaseKernel::ArchTag::kMinComputeCapability == 90, "DistGEMM only supports Hopper GEMMs for now.");
  static_assert(not cute::is_same_v<typename BaseKernel::ElementC, void>, "DistributedGEMM epilogues must have a source.");

  using ElementFlag = uint32_t;

  // Device side arguments
  struct DistributedArguments {
    int device_idx = 0;
    int iteration = 0;

    void* self_flag_ptr{nullptr};
    void* peer_flag_ptr{nullptr};
  };

  struct PackedArguments {
    BaseArguments base{};
    DistributedArguments distributed{};
  };

  struct DistributedParams {
    int device_idx = 0;
    int iteration = 0;

    ElementFlag* self_flag_ptr_{nullptr};
    ElementFlag* peer_flag_ptr_{nullptr};
  };

  // Kernel entry point API
  struct PackedParams {
    BaseParams base{};
    DistributedParams distributed{};
  };

  using Params = PackedParams;

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  PackedParams
  to_underlying_arguments(PackedArguments const& args, void* workspace) {
    CUTLASS_TRACE_HOST("distributed::to_underlying_arguments():");

    auto kernel_params = BaseKernel::to_underlying_arguments(args.base, workspace);

    DistributedParams dist_params = {
        args.distributed.device_idx,
        args.distributed.iteration,
        reinterpret_cast<ElementFlag*>(args.distributed.self_flag_ptr),
        reinterpret_cast<ElementFlag*>(args.distributed.peer_flag_ptr)
    };

    return {kernel_params, dist_params};
  }

  static bool
  can_implement(BaseArguments const& args) {
    return BaseKernel::can_implement(args);
  }

  static bool
  can_implement(PackedArguments const& args) {
    return BaseKernel::can_implement(args.base);
  }

  static size_t
  get_workspace_size(BaseArguments const& args) {
    return BaseKernel::get_workspace_size(args);
  }

  static size_t
  get_workspace_size(PackedArguments const& args) {
    return BaseKernel::get_workspace_size(args.base);
  }

  static cutlass::Status
  initialize_workspace(BaseArguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return BaseKernel::initialize_workspace(args, workspace, stream, cuda_adapter);
  }

  static cutlass::Status
  initialize_workspace(PackedArguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return BaseKernel::initialize_workspace(args.base, workspace, stream, cuda_adapter);
  }

  /// Computes the grid shape
  static dim3
  get_grid_shape(PackedParams const& params) {
    return BaseKernel::get_grid_shape(params.base);
  }
  
  static dim3
  get_grid_shape(BaseParams const& params) {
    return BaseKernel::get_grid_shape(params);
  }

  CUTLASS_DEVICE
  void
  barrier_buffer(PackedParams const& params) {
    if (params.distributed.iteration > 0) {

      ElementFlag comm_iter = 0;
      detail::ld_without_cache(comm_iter, params.distributed.self_flag_ptr_);
      while (comm_iter == 0) {
        detail::ld_without_cache(comm_iter, params.distributed.self_flag_ptr_);
        __nanosleep(40);
      }

    }
  }

  CUTLASS_DEVICE
  void
  maybe_signal_arrival(PackedParams const& params) {
    if constexpr (KernelWritesArrivalFlag) {
      if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
          threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
          params.distributed.iteration > 0) {
        *reinterpret_cast<ElementFlag*>(params.distributed.peer_flag_ptr_) = 1;
      }
    }
  }

  CUTLASS_DEVICE
  void
  operator()(PackedParams const& params, char* smem_buf) {
    // Launch next grid as soon as possible
    arch::launch_dependent_grids();

    // Wait on previous kernels to flush their memory.
    arch::wait_on_dependent_grids();

    // Optionally write arrivals for the previous stage/iteration.
    maybe_signal_arrival(params);

    // Spin-wait on an arrival flag, make sure the respective buffers are ready.
    // If the buffered operand is memcpied into, it would wait on its local flag.
    // If it's a remote buffer that is accessed directly, it would wait on its remote flag.
    barrier_buffer(params);

    // Perform local gemm
    BaseKernel gemm;
    gemm(params.base, smem_buf);
  }

};

} // namespace cutlass::distributed::kernel

///////////////////////////////////////////////////////////////////////////////

