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

#include <tuple>

namespace kda::sm90::kernel {

template <auto kTag, class Value>
struct Option {
  static constexpr auto tag = kTag;
  using option_value = Value;
};

using DefaultOptions = std::tuple<>;

namespace detail {

template <auto kTag, typename Default, typename... Options>
struct find_option_impl;

template <auto kTag, typename Default>
struct find_option_impl<kTag, Default> {
  using option_value = Default;
};

template <auto kTag, typename Default>
struct find_option_impl<kTag, Default, void> : find_option_impl<kTag, Default> {};

template <auto kTag, typename Default, typename Option, typename... Options>
struct find_option_impl<kTag, Default, Option, Options...>
    : std::conditional_t<Option::tag == kTag, Option, find_option_impl<kTag, Default, Options...>> {};

template <auto kTag, typename Default, typename... Options>
struct find_option_impl<kTag, Default, std::tuple<Options...>> : find_option_impl<kTag, Default, Options...> {};

template <typename NewOption, typename... Options>
struct add_option_impl;

template <typename NewOption, typename... Options>
struct add_option_impl<NewOption, std::tuple<Options...>> {
  using options = std::tuple<Options..., NewOption>;
};

}  // namespace detail

template <auto kTag, typename Default, typename... Options>
using find_option_t = typename detail::find_option_impl<kTag, Default, std::tuple<Options...>>::option_value;

template <auto kTag, typename Value, typename... Options>
using add_option_t = typename detail::add_option_impl<Option<kTag, Value>, std::tuple<Options...>>::options;

template <auto kTag, typename Value, typename... Options>
constexpr auto add_option(Option<kTag, Value> new_option, std::tuple<Options...> options_tuple) {
  return add_option_t<kTag, Value, Options...>();
}

enum class Tag {
  kIsDeltaRule,
  kIsPersistent,
  kNumMmaWarpGroups,
  kStagesQ,
  kStagesK,
  kStagesV,
  kNeedsAlpha,          // gated delta rule
  kNeedsBeta,           // delta rule
  kInitStateFromInput,  // if true, initialize state by reading global memory instead of zero initialization.
  kSafeGate,            // KDA
};

}  // namespace kda::sm90::kernel
