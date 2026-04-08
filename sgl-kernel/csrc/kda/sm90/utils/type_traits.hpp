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

/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cutlass/numeric_types.h>

#include <type_traits>

namespace kda::sm90 {

// clang-format off
template <typename T> struct map_to_cutlass;
template<> struct map_to_cutlass<cutlass::half_t>             { using type = cutlass::half_t;                    };
template<> struct map_to_cutlass<cutlass::bfloat16_t>         { using type = cutlass::bfloat16_t;                };
template<> struct map_to_cutlass<half>                        { using type = cutlass::half_t;                    };
template<> struct map_to_cutlass<nv_bfloat16>                 { using type = cutlass::bfloat16_t;                };

template <typename T> using map_to_cutlass_t = typename map_to_cutlass<T>::type;
// clang-format on

template <typename... Ts>
struct first_non_void {
  static_assert(sizeof...(Ts) > 0, "all voids is not allowed");
  using type = void;
};

template <typename T, typename... Ts>
struct first_non_void<T, Ts...> {
  using type = T;
};

template <typename... Ts>
struct first_non_void<void, Ts...> : first_non_void<Ts...> {};

template <typename... Ts>
using first_non_void_t = typename first_non_void<Ts...>::type;

}  // namespace kda::sm90
