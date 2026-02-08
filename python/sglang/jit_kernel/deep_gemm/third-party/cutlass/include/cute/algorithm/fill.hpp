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

#include <cute/config.hpp>

#include <cute/tensor_impl.hpp>
#include <cute/algorithm/prefer.hpp>

namespace cute
{

//
// Accept mutable temporaries
//
template <class Engine, class Layout, class T>
CUTE_HOST_DEVICE
void
fill(Tensor<Engine, Layout>&& tensor, T const& value)
{
  return fill(tensor, value);
}

namespace detail
{

// Prefer fill(tensor.data(), value), if possible
template <class Engine, class Layout, class T>
CUTE_HOST_DEVICE
auto
fill(Tensor<Engine, Layout>& tensor, T const& value, prefer<1>)
    -> decltype(fill(tensor.data(), value))
{
  fill(tensor.data(), value);
}

// Default implementation
template <class Engine, class Layout, class T>
CUTE_HOST_DEVICE
void
fill(Tensor<Engine, Layout>& tensor, T const& value, prefer<0>)
{
  CUTE_UNROLL
  for (int i = 0; i < size(tensor); ++i) {
    tensor(i) = value;
  }
}

} // end namespace detail

template <class Engine, class Layout, class T>
CUTE_HOST_DEVICE
void
fill(Tensor<Engine, Layout>& tensor, T const& value)
{
  return detail::fill(tensor, value, prefer<1>{});
}

} // end namespace cute
