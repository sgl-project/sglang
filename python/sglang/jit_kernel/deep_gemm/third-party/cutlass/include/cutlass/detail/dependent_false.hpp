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
#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::detail {

/// @brief A bool constant that depends on one or more template parameters.
///
/// For more detailed documentation and use cases,
/// please see `dependent_false` below.
template <bool Value, class... Args>
inline constexpr bool dependent_bool_value = Value;

/// @brief An always-false value that depends on one or more template parameters.
///
/// This exists because `static_assert(false);` always fails,
/// even if it occurs in the `else` branch of an `if constexpr`.
/// The following example shows how to use `dependent_false` in that case.
///
/// @code
/// template<class T>
/// void foo (T t)
/// {
///     if constexpr (std::is_integral_v<T>) {
///         do_integer_stuff(t);
///     }
///     else if constexpr (std::is_floating_point_v<T>) {
///         do_floating_point_stuff(t);
///     }
///     else {
///         static_assert(dependent_false<T>, "T must be "
///             "an integral or floating-point type.");
///     }
/// }
/// @endcode
///
/// This implements the C++ Standard Library proposal P1830R1.
///
/// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1830r1.pdf
///
/// That proposal is under review as of 2022/12/05.
/// The following link shows P1830's current review status.
///
/// https://github.com/cplusplus/papers/issues/572
///
/// P2593R0 proposes an alternate solution to this problem,
/// that would change the C++ language itself.
///
/// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2593r0.html
///
/// For headers in this library, however, we only consider library solutions
/// as work-arounds for future C++ features.
template <class... Args>
inline constexpr bool dependent_false = dependent_bool_value<false, Args...>;

}  // end namespace cutlass::detail
