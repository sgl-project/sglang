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

/*! \file
    \brief Mainloop Fusion configs specific for scale factors
*/

#pragma once

#include <cute/util/type_traits.hpp> // cute::void_t

namespace cutlass::detail {

/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename CollectiveMainloop, typename = void>
struct ElementSFType {
  using type = void;
};

template <typename CollectiveMainloop>
struct ElementSFType<CollectiveMainloop, cute::void_t<typename CollectiveMainloop::ElementSF>> {
  using type = typename CollectiveMainloop::ElementSF;
};

template <typename CollectiveMainloop, typename = void>
struct LayoutSFAType {
  using type = void;
};

template <typename CollectiveMainloop>
struct LayoutSFAType<CollectiveMainloop, cute::void_t<typename CollectiveMainloop::LayoutSFA>> {
  using type = typename CollectiveMainloop::LayoutSFA;
};

template <typename CollectiveMainloop, typename = void>
struct LayoutSFBType {
  using type = void;
};

template <typename CollectiveMainloop>
struct LayoutSFBType<CollectiveMainloop, cute::void_t<typename CollectiveMainloop::LayoutSFB>> {
  using type = typename CollectiveMainloop::LayoutSFB;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::detail
