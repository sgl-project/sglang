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

#include "cutlass/cutlass.h"

namespace cutlass::fmha::kernel {

template<auto kTag, typename Default, typename... Options>
struct find_option;

template<auto kTag, typename Default>
struct find_option<kTag, Default> {
  using option_value = Default;
};

template<auto kTag, typename Default, typename Option, typename... Options>
struct find_option<kTag, Default, Option, Options...> :
  std::conditional_t<
    Option::tag == kTag,
    Option,
    find_option<kTag, Default, Options...>
  >
{};

template<auto kTag, typename Default, typename... Options>
using find_option_t = typename find_option<kTag, Default, Options...>::option_value;

enum class Tag {
  kIsPersistent,
  kNumMmaWarpGroups,
  kLoadsQSeparately,

  kIsMainloopLocked,
  kIsEpilogueLocked,

  kStagesQ,
  kStagesKV,

  kEpilogueKind,

  kBlocksPerSM,
  kClusterM,

  kAccQK
};

template<auto kTag, class Value>
struct Option {
  static constexpr auto tag = kTag;
  using option_value = Value;
};

}  // namespace cutlass::fmha::kernel
