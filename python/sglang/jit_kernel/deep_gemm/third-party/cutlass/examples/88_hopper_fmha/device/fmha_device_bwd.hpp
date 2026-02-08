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

#pragma once

/*!
  \file
  \brief An universal device layer for cutlass 3.x-style kernels.
*/

// common
#include "cutlass/cutlass.h"

#include "../device/device_universal.hpp"
#include "../collective/fmha_collective_bwd_tma_warpspecialized.hpp"
#include "../collective/fmha_fusion.hpp"
#include "../collective/fmha_epilogue_bwd.hpp"
#include "../kernel/fmha_kernel_bwd_sum_OdO.hpp"
#include "../kernel/fmha_kernel_bwd_convert.hpp"
#include "../kernel/fmha_kernel_tma_warpspecialized.hpp"
#include "../kernel/fmha_tile_scheduler.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass::fmha::device {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// CUTLASS 3.x API /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Element, class ElementAccumulator, class TileShape, class Fusion, class... Options>
class FmhaBwd {
public:
  /// Argument structure: User API
  struct Arguments {
    cute::tuple<int, int, int, int, int> problem_size;

    const Element* ptr_Q;
    cute::tuple<int, int, int, cute::_1> stride_Q;
    const Element* ptr_K;
    cute::tuple<int, int, int, cute::_1> stride_K;
    const Element* ptr_V;
    cute::tuple<int, int, int, cute::_1> stride_V;

    const Element* ptr_O;
    cute::tuple<int, int, int, cute::_1> stride_O;
    const ElementAccumulator* ptr_LSE;
    cute::tuple<int, int, _1> stride_LSE;

    const Element* ptr_dO;
    cute::tuple<int, int, int, cute::_1> stride_dO;

    Element* ptr_dQ;
    cute::tuple<int, int, int, cute::_1> stride_dQ;
    Element* ptr_dK;
    cute::tuple<int, int, int, cute::_1> stride_dK;
    Element* ptr_dV;
    cute::tuple<int, int, int, cute::_1> stride_dV;

    cutlass::KernelHardwareInfo hw_info;
  };

  using OperationSumOdO = cutlass::device::Universal<cutlass::fmha::kernel::FmhaKernelBwdSumOdO<Element, ElementAccumulator>>;
  using OperationConvert = cutlass::device::Universal<cutlass::fmha::kernel::FmhaKernelBwdConvert<Element, ElementAccumulator>>;

  using Mainloop = cutlass::fmha::collective::FmhaBwdMainloopTmaWarpSpecialized<
    Element, ElementAccumulator, TileShape, 
    cutlass::fmha::collective::FusionBwdAdapter<Fusion>, Options...>;
    
  using Epilogue = cutlass::fmha::collective::FmhaBwdEpilogueKV<Element, ElementAccumulator, typename Mainloop::TileShapePV>;
  
  using Operation = cutlass::device::Universal<
    cutlass::fmha::kernel::FmhaKernelTmaWarpSpecialized<
      Mainloop,
      Epilogue,
      cutlass::fmha::kernel::TileSchedulerBwdAdapter<cutlass::fmha::kernel::IndividualTileScheduler>, Options...>>;

  struct Params {
    OperationSumOdO op_sum_OdO;
    Operation op;
    OperationConvert op_convert;
    ElementAccumulator* dQ_acc;
    size_t dQ_acc_size;
  };

private:
  Params params_;

  static typename OperationSumOdO::Arguments to_sum_OdO_arguments(Arguments const& args, ElementAccumulator* dest = nullptr) {
    auto [B, H, Q, K, D] = args.problem_size;
    D = cutlass::round_up(D, 8);  // Alignment
    Q = cutlass::round_up(Q, 8);  // Alignment
    auto stride_sum_OdO = make_stride(H*Q, Q, _1{});
    return typename OperationSumOdO::Arguments {
      args.problem_size,
      args.ptr_O, args.stride_O,
      args.ptr_dO, args.stride_dO,
      dest, stride_sum_OdO
    };
  }

  static typename OperationConvert::Arguments to_convert_arguments(Arguments const& args, ElementAccumulator* src = nullptr) {
    auto [B, H, Q, K, D] = args.problem_size;
    D = cutlass::round_up(D, 8);  // Alignment
    Q = cutlass::round_up(Q, 8);  // Alignment
    auto stride_src_dQ = make_stride(B == 1 ? 0 : (H*Q*D), Q*D, D, _1{});
    return typename OperationConvert::Arguments {
      args.problem_size,
      src, stride_src_dQ,
      nullptr, stride_src_dQ,
      nullptr, stride_src_dQ,
      args.ptr_dQ, args.stride_dQ,
      nullptr, args.stride_dK,
      nullptr, args.stride_dV
    };
  }

  static typename Operation::Arguments to_bwd_arguments(
      Arguments const& args,
      ElementAccumulator* sum_OdO = nullptr, cute::tuple<int, int, _1> const& stride_sum_OdO = {},
      ElementAccumulator* dQ_acc = nullptr, cute::tuple<int, int, int, _1> const& stride_dQ = {}
  ) {
    return typename Operation::Arguments{
      args.problem_size,
      { args.ptr_Q, args.stride_Q,
        args.ptr_K, args.stride_K,
        args.ptr_V, args.stride_V,
        args.ptr_dO, args.stride_dO,
        args.ptr_LSE, args.stride_LSE,
        sum_OdO, stride_sum_OdO,
        dQ_acc, stride_dQ },
      { args.ptr_dK, args.stride_dK,
        args.ptr_dV, args.stride_dV },
      args.hw_info
    };
  }

public:

  /// Determines whether the GEMM can execute the given problem.
  static Status
  can_implement(Arguments const& args) {
    Status status = Status::kSuccess;

    status = OperationSumOdO::can_implement(to_sum_OdO_arguments(args));
    if (status != Status::kSuccess) {
      return status;
    }

    status = OperationConvert::can_implement(to_convert_arguments(args));
    if (status != Status::kSuccess) {
      return status;
    }

    status = Operation::can_implement(to_bwd_arguments(args));
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  /// Gets the workspace size
  static size_t
  get_workspace_size(Arguments const& args) {
    auto [B, H, Q, K, D] = args.problem_size;
    D = cutlass::round_up(D, 8);  // Alignment
    Q = cutlass::round_up(Q, 8);  // Alignment
    size_t workspace_bytes = 0;
    // OdO vector
    workspace_bytes += B*H*Q * sizeof(ElementAccumulator);
    // FP32 versions of outputs that are churned (start off with Q only)
    workspace_bytes += B*H*Q*D * sizeof(ElementAccumulator);
    return workspace_bytes;
  }

  /// Initializes state from arguments.
  Status
  initialize_split(Arguments const& args, void* workspace_dQ, void* workspace_sum_OdO, cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("Universal::initialize_split() - workspace_dQ="
      << workspace_dQ << ", workspace_sum_OdO=" << workspace_sum_OdO << "stream: " << (stream ? "non-null" : "null"));

    auto [B, H, Q, K, D] = args.problem_size;
    D = cutlass::round_up(D, 8);  // Alignment
    Q = cutlass::round_up(Q, 8);  // Alignment
    ElementAccumulator* sum_OdO = reinterpret_cast<ElementAccumulator*>(workspace_sum_OdO);
    ElementAccumulator* dQ_acc = reinterpret_cast<ElementAccumulator*>(workspace_dQ);
    params_.dQ_acc = dQ_acc;
    params_.dQ_acc_size = B*H*Q*D * sizeof(ElementAccumulator);
    auto args_sum_OdO = to_sum_OdO_arguments(args, sum_OdO);
    auto args_convert = to_convert_arguments(args, dQ_acc);
    params_.op_sum_OdO.initialize(args_sum_OdO, nullptr, stream);
    params_.op_convert.initialize(args_convert, nullptr, stream);
    auto args_bwd = to_bwd_arguments(args, sum_OdO, args_sum_OdO.stride_sum_OdO, dQ_acc, args_convert.stride_src_dQ);
    params_.op.initialize(args_bwd, nullptr, stream);

    return Status::kSuccess;
  }

  /// Initializes state from arguments.
  Status
  initialize(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("Universal::initialize() - workspace "
      << workspace << ", stream: " << (stream ? "non-null" : "null"));

    auto [B, H, Q, K, D] = args.problem_size;
    D = cutlass::round_up(D, 8);  // Alignment
    Q = cutlass::round_up(Q, 8);  // Alignment
    char* workspace_chr = reinterpret_cast<char*>(workspace);
    ElementAccumulator* sum_OdO = reinterpret_cast<ElementAccumulator*>(workspace_chr);
    workspace_chr += B*H*Q * sizeof(ElementAccumulator);
    ElementAccumulator* dQ_acc = reinterpret_cast<ElementAccumulator*>(workspace_chr);
    return initialize_split(args, dQ_acc, sum_OdO, stream);
  }

  /// Primary run() entry point API that is static allowing users to create and manage their own params.
  /// Supplied params struct must be construct by calling Kernel::to_underling_arguments()
  static Status
  run(Params& params, cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("FmhaDeviceBwd::run()");

    Status result = Status::kSuccess;
    result = params.op_sum_OdO.run(stream);
    if (result != Status::kSuccess) {
      return result;
    }

    auto cuda_result = cudaMemsetAsync(params.dQ_acc, 0, params.dQ_acc_size, stream);
    if (cuda_result != cudaSuccess) {
       return Status::kErrorInternal;
    }
    result = params.op.run(stream);
    if (result != Status::kSuccess) {
      return result;
    }

    result = params.op_convert.run(stream);
    if (result != Status::kSuccess) {
      return result;
    }

    return Status::kSuccess;
  }

  //
  // Non-static launch overloads that first create and set the internal params struct of this kernel handle.
  //

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  Status
  run(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    Status status = initialize(args, workspace, stream);
    if (Status::kSuccess == status) {
      status = run(params_, stream);
    }
    return status;
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status
  run(cudaStream_t stream = nullptr) {
    return run(params_, stream);
  }

};

////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::fmha::device

////////////////////////////////////////////////////////////////////////////////
