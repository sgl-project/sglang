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


#pragma once

// common
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cute/tensor.hpp"

#include "../device/fmha.hpp"
#include "../kernel/sm100_fmha_bwd_kernel_tma_warpspecialized.hpp"
#include "../kernel/sm100_fmha_bwd_mla_kernel_tma_warpspecialized.hpp"
#include "../kernel/fmha_kernel_bwd_sum_OdO.hpp"
#include "../kernel/fmha_kernel_bwd_convert.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass::fmha::device {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// CUTLASS 3.x API /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<
    class ProblemShape,
    class Element,
    class ElementAccumulator,
    class TileShape,
    bool IsMla,
    class Mask
>
class Sm100FmhaBwd {
private:
  template <typename T>
  constexpr static auto to_bwd_shape(T shape) {
    if constexpr (IsMla) { // remove GQA mode
      constexpr int R = decltype(rank(shape))::value;
      auto HB = get<R-1>(shape);
      auto rest = take<0,R-1>(shape);
      return append(rest, make_shape(size<0>(HB), get<1>(HB)));
    }
    else {
      return shape;
    }
  }

  template <typename T>
  constexpr static auto to_bwd_stride(T stride) {
    if constexpr (IsMla) { // remove GQA mode
      constexpr int R = decltype(rank(stride))::value;
      auto HB = get<R-1>(stride);
      auto rest = take<0,R-1>(stride);
      if constexpr (is_same_v<remove_cv_t<decltype(get<0,0>(HB))>, _0>) {
        return append(rest, make_stride(get<0,1>(HB), get<1>(HB)));
      }
      else {
        return append(rest, make_stride(get<0,0>(HB), get<1>(HB)));
      }
    }
    else {
      return stride;
    }
  }

public:
  /// Argument structure: User API
  struct Arguments {
    // Q K D D_VO HB
    ProblemShape problem_shape;

    const Element* ptr_Q;
    cute::tuple<int, cute::_1, cute::tuple<cute::tuple<int,int>, int>> stride_Q;
    const Element* ptr_K;
    cute::tuple<int, cute::_1, cute::tuple<cute::tuple<cute::_0,int>, int>> stride_K;
    const Element* ptr_V;
    cute::tuple<int, cute::_1, cute::tuple<cute::tuple<cute::_0,int>, int>> stride_V;

    const Element* ptr_O;
    cute::tuple<int, cute::_1, cute::tuple<cute::tuple<int,int>, int>> stride_O;
    const ElementAccumulator* ptr_LSE;
    cute::tuple<cute::_1, cute::tuple<cute::tuple<int,int>, int>> stride_LSE;

    const Element* ptr_dO;
    cute::tuple<int, cute::_1, cute::tuple<cute::tuple<int,int>, int>> stride_dO;

    Element* ptr_dQ;
    cute::tuple<int, cute::_1, cute::tuple<cute::tuple<int,int>, int>> stride_dQ;
    Element* ptr_dK;
    cute::tuple<int, cute::_1, cute::tuple<cute::tuple<cute::_0,int>, int>> stride_dK;
    Element* ptr_dV;
    cute::tuple<int, cute::_1, cute::tuple<cute::tuple<cute::_0,int>, int>> stride_dV;

    ElementAccumulator softmax_scale;

    cutlass::KernelHardwareInfo hw_info;
  };

  using OperationSumOdO = cutlass::fmha::device::FMHA<
    cutlass::fmha::kernel::FmhaKernelBwdSumOdO<ProblemShape, Element, ElementAccumulator>
  >;
  using OperationConvert = cutlass::fmha::device::FMHA<
    cutlass::fmha::kernel::FmhaKernelBwdConvert<ProblemShape, Element, ElementAccumulator>
  >;

  using OperationNormal= cutlass::fmha::device::FMHA<
      cutlass::fmha::kernel::Sm100FmhaBwdKernelTmaWarpSpecialized<
          ProblemShape, Element, ElementAccumulator, TileShape, Mask
      >
  >;

  using ProblemShapeMLA = decltype(to_bwd_shape(ProblemShape{}));
  using OperationMla = cutlass::fmha::device::FMHA<
      cutlass::fmha::kernel::Sm100FmhaBwdMlaKernelTmaWarpSpecialized<
          ProblemShapeMLA, Element, ElementAccumulator, TileShape, Mask
      >
  >;

  using Operation = std::conditional_t<IsMla, OperationMla, OperationNormal>;

  using Kernel = typename Operation::Kernel;

  struct Params {
    OperationSumOdO op_sum_OdO;
    Operation op;
    OperationConvert op_convert;
    ElementAccumulator* dQ_acc;
    size_t dQ_acc_size;
  };

private:
  Params params_;

  static typename OperationSumOdO::Arguments to_sum_OdO_arguments(
        Arguments const& args,
        ElementAccumulator* sum_odo = nullptr,
        ElementAccumulator* scaled_lse = nullptr) {
    using namespace cute;
    auto [Q_, K, D, D_VO, HB] = args.problem_shape;
    auto [H, B] = HB;
    auto [H_R, H_K] = H;
    D = cutlass::round_up(D, 8);  // Alignment
    int Q = cutlass::round_up(static_cast<int>(Q_), 8);  // Alignment
    auto stride_sum_OdO = make_stride(_1{}, make_stride(make_stride(Q, Q*H_R), B == 1 ? 0 : Q*H_R*H_K));
    auto stride_scaled_lse = make_stride(_1{}, make_stride(make_stride(Q, Q*H_R), B == 1 ? 0 : Q*H_R*H_K));
    auto log2_e = log2f(expf(1.0f));
    return typename OperationSumOdO::Arguments {
      args.problem_shape,
      args.ptr_O, args.stride_O,
      args.ptr_dO, args.stride_dO,
      sum_odo, stride_sum_OdO,
      args.ptr_LSE, args.stride_LSE,
      scaled_lse, stride_scaled_lse,
      -1.0f, -log2_e
    };
  }

  static typename OperationConvert::Arguments to_convert_arguments(Arguments const& args, ElementAccumulator* src = nullptr) {
    using namespace cute;
    auto [Q_, K, D, D_VO, HB] = args.problem_shape;
    auto [H, B] = HB;
    auto [H_R, H_K] = H;
    D = cutlass::round_up(D, 8);  // Alignment
    int Q = cutlass::round_up(static_cast<int>(Q_), 8);  // Alignment
    auto stride_src_dQ = make_stride(D, _1{}, make_stride(make_stride(D*Q, D*Q*H_R), B == 1 ? 0 : D*Q*H_R*H_K));
    return typename OperationConvert::Arguments {
      args.problem_shape,
      src, stride_src_dQ,
      nullptr, args.stride_dK,
      nullptr, args.stride_dV,
      args.ptr_dQ, args.stride_dQ,
      nullptr, args.stride_dK,
      nullptr, args.stride_dV,
      args.softmax_scale
    };
  }

  static typename Operation::Arguments to_bwd_arguments(
      Arguments const& args,
      ElementAccumulator* sum_OdO = nullptr, cute::tuple<cute::_1, cute::tuple<cute::tuple<int, int>, int>> const& stride_sum_OdO = {},
      ElementAccumulator* scaled_lse = nullptr, cute::tuple<cute::_1, cute::tuple<cute::tuple<int, int>, int>> const& stride_scaled_lse = {},
      ElementAccumulator* dQ_acc = nullptr, cute::tuple<int, cute::_1, cute::tuple<cute::tuple<int, int>, int>> const& stride_dQ = {}) {

    return typename Operation::Arguments{
      to_bwd_shape(args.problem_shape),
      { args.ptr_Q,  to_bwd_stride(args.stride_Q),
        args.ptr_K,  to_bwd_stride(args.stride_K),
        args.ptr_V,  to_bwd_stride(args.stride_V),
        args.ptr_dO, to_bwd_stride(args.stride_dO),
        scaled_lse, to_bwd_stride(stride_scaled_lse),
        sum_OdO, to_bwd_stride(stride_sum_OdO),
        dQ_acc, to_bwd_stride(stride_dQ),
        args.softmax_scale },
      { args.ptr_dK, to_bwd_stride(args.stride_dK),
        args.ptr_dV, to_bwd_stride(args.stride_dV) },
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
    auto [Q_, K, D, D_VO, HB] = args.problem_shape;
    auto [H, B] = product_each(HB);
    D = cutlass::round_up(D, 8);  // Alignment
    int Q = cutlass::round_up(static_cast<int>(Q_), 8);  // Alignment
    size_t workspace_bytes = 0;
    // OdO vector
    workspace_bytes += B*H*Q * sizeof(ElementAccumulator);
    // scaled LSE vector
    workspace_bytes += B*H*Q * sizeof(ElementAccumulator);
    // FP32 versions of outputs that are churned (start off with Q only)
    workspace_bytes += B*H*Q*D * sizeof(ElementAccumulator);
    return workspace_bytes;
  }

  /// Initializes state from arguments.
  Status
  initialize_split(Arguments const& args, void* workspace_dQ, void* workspace_sum_OdO, void* workspace_scaled_lse, cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("Universal::initialize_split() - workspace_dQ="
      << workspace_dQ << ", workspace_sum_OdO=" << workspace_sum_OdO << "stream: " << (stream ? "non-null" : "null"));

    auto [Q_, K, D, D_VO, HB] = args.problem_shape;
    auto [H, B] = product_each(HB);
    D = cutlass::round_up(D, 8);  // Alignment
    int Q = cutlass::round_up(static_cast<int>(Q_), 8);  // Alignment
    ElementAccumulator* sum_OdO = reinterpret_cast<ElementAccumulator*>(workspace_sum_OdO);
    ElementAccumulator* scaled_lse = reinterpret_cast<ElementAccumulator*>(workspace_scaled_lse);
    ElementAccumulator* dQ_acc = reinterpret_cast<ElementAccumulator*>(workspace_dQ);
    params_.dQ_acc = dQ_acc;
    params_.dQ_acc_size = B*H*Q*D * sizeof(ElementAccumulator);
    auto args_sum_OdO = to_sum_OdO_arguments(args, sum_OdO, scaled_lse);
    auto args_convert = to_convert_arguments(args, dQ_acc);
    params_.op_sum_OdO.initialize(args_sum_OdO, nullptr, stream);
    params_.op_convert.initialize(args_convert, nullptr, stream);
    auto args_bwd = to_bwd_arguments(
        args, sum_OdO, args_sum_OdO.stride_sum_OdO,
        scaled_lse, args_sum_OdO.stride_scaled_lse,
        dQ_acc, args_convert.stride_src_dQ
    );
    params_.op.initialize(args_bwd, nullptr, stream);

    return Status::kSuccess;
  }

  /// Initializes state from arguments.
  Status
  initialize(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("Universal::initialize() - workspace "
      << workspace << ", stream: " << (stream ? "non-null" : "null"));

    auto [Q_, K, D, D_VO, HB] = args.problem_shape;
    auto [H, B] = product_each(HB);
    D = cutlass::round_up(D, 8);  // Alignment
    int Q = cutlass::round_up(static_cast<int>(Q_), 8);  // Alignment
    char* workspace_chr = reinterpret_cast<char*>(workspace);
    ElementAccumulator* sum_OdO = reinterpret_cast<ElementAccumulator*>(workspace_chr);
    workspace_chr += B*H*Q * sizeof(ElementAccumulator);
    ElementAccumulator* scaled_lse = reinterpret_cast<ElementAccumulator*>(workspace_chr);
    workspace_chr += B*H*Q * sizeof(ElementAccumulator);
    ElementAccumulator* dQ_acc = reinterpret_cast<ElementAccumulator*>(workspace_chr);
    return initialize_split(args, dQ_acc, sum_OdO, scaled_lse, stream);
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
