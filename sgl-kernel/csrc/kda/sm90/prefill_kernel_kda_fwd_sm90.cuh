// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/util/device_memory.h>

#include <cstdio>
#include <cute/tensor.hpp>

#include "kda/sm90/device/device_universal.hpp"
#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include "kda/sm90/utils/common.hpp"

namespace kda::sm90 {

using namespace cute;

template <
    bool NeedsBeta,
    bool NeedsAlpha,
    bool InitStateFromInput,
    bool SafeGate,
    typename ArchTag,
    typename TO,
    typename TQKV,
    typename TState>
void launch_kda_fwd_prefill_kernel_gbai(
    cudaStream_t stream,
    TO* output,
    TState* output_state,
    TQKV const* q,
    TQKV const* k,
    TQKV const* v,
    TState const* input_state,
    float const* alpha,
    float const* beta,
    int32_t const* cu_seqlens,
    uint8_t* workspace_buffer,
    int32_t num_seqs,
    int32_t num_heads,
    int32_t head_size,
    int64_t total_seqlen,
    float scale,
    int32_t sm_count) {
#if defined(SGL_KDA_SM90A_ENABLED)
  constexpr bool HopperSupported = true;
#else
  constexpr bool HopperSupported = false;
#endif

  if constexpr (HopperSupported) {
    static_assert(std::is_same_v<TQKV, TO>);

    using namespace kda::sm90::kernel;
    using T = map_to_cutlass_t<TQKV>;

    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = sm_count;

    using SafeGateType = std::conditional_t<SafeGate, cute::true_type, cute::false_type>;
    using NeedsBetaType = std::conditional_t<NeedsBeta, cute::true_type, cute::false_type>;
    using NeedsAlphaType = std::conditional_t<NeedsAlpha, cute::true_type, cute::false_type>;
    using InitStateType = std::conditional_t<InitStateFromInput, cute::true_type, cute::false_type>;
    using Options = decltype(add_option(
        Option<Tag::kSafeGate, SafeGateType>{},
        add_option(
            Option<Tag::kInitStateFromInput, InitStateType>{},
            add_option(
                Option<Tag::kNeedsAlpha, NeedsAlphaType>{},
                add_option(
                    Option<Tag::kNeedsBeta, NeedsBetaType>{},
                    add_option(Option<Tag::kIsDeltaRule, cute::true_type>{}, DefaultOptions{}))))));

    using TileShape = Shape<_64, _64, _128>;
    using Scheduler = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using Operation = cutlass::device::Universal<typename kda::sm90::kernel::FlatBuilderKdaFwd<
        T,
        float,
        float,
        TileShape,
        /*LayoutQ=*/cute::tuple<int64_t, _1, int32_t>,
        /*LayoutK=*/cute::tuple<int64_t, _1, int32_t>,
        /*LayoutV=*/cute::tuple<int64_t, _1, int32_t>,
        /*LayoutO=*/cute::tuple<int64_t, _1, int32_t>,
        Scheduler,
        Options>::Kernel>;
    using Arguments = typename Operation::Arguments;

    // NOTE: LayoutQ/K/V in (seq, head_size, (b,h)) coordinate semantics

    int32_t tok_stride = num_heads * head_size;
    int32_t head_stride = head_size;

    Operation op;
    Arguments arguments{
        .problem_size =
            {
                .cu_seqlens = cu_seqlens,
                .total_seqlen = total_seqlen,
                .num_seqs = num_seqs,
                .num_heads = num_heads,
                .head_size = head_size,
            },
        .mainloop =
            {
                // clang-format off
                .ptr_Q = (T*)q,      .dQ = {tok_stride, _1{}, head_stride},
                .ptr_K = (T*)k,      .dK = {tok_stride, _1{}, head_stride},
                .ptr_V = (T*)v,      .dV = {tok_stride, _1{}, head_stride},
                .ptr_O = (T*)output, .dO = {tok_stride, _1{}, head_stride},
                .ptr_Alpha = alpha,  .dAlpha = {tok_stride, _1{}, head_stride},
                .ptr_output_state = (float*)output_state,
                .ptr_input_state  = (float*)input_state,
                .scale = scale,
                .beta_ptr  = beta,  .beta_stride  = {num_heads, 1},
        },  // clang-format on
        .hw_info = hw_info};

    cutlass::Status status;
    status = op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("can_implement failed");
    }

    status = op.initialize(arguments, workspace_buffer, stream);
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("initialize failed");
    }

    status = op.run(stream);
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("run failed");
    }

  } else {
    throw std::runtime_error("hopper not supported");
  }
}

};  // namespace kda::sm90
