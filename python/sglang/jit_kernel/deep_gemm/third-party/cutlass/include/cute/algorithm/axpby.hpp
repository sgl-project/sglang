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

namespace cute
{

//
// Accept mutable temporaries
//
template <class Alpha,
          class XEngine, class XLayout,
          class Beta,
          class YEngine, class YLayout,
          class PrdTensor = constant_fn<true_type>>
CUTE_HOST_DEVICE
void
axpby(Alpha                    const& alpha,
      Tensor<XEngine, XLayout> const& x,
      Beta                     const& beta,
      Tensor<YEngine, YLayout>     && y,
      PrdTensor                const& p = {})
{
  return axpby(alpha, x, beta, y, p);
}

//
// AXPBY
//
template <class Alpha,
          class XEngine, class XLayout,
          class Beta,
          class YEngine, class YLayout,
          class PrdTensor = constant_fn<true_type>>
CUTE_HOST_DEVICE
void
axpby(Alpha                    const& alpha,
      Tensor<XEngine, XLayout> const& x,
      Beta                     const& beta,
      Tensor<YEngine, YLayout>      & y,
      PrdTensor                const& p = {})
{
  auto isBetaZero = [&] () {
    if constexpr (is_complex<Beta>::value) {
      return beta.real() == Int<0>{} && beta.imag() == Int<0>{};
    }
    else {
      return beta == Int<0>{};
    }

    CUTE_GCC_UNREACHABLE;
  } ();

  CUTE_UNROLL
  for (int i = 0; i < size(x); ++i) {
    if (p(i)) {
      y(i) = (isBetaZero ? alpha * x(i) : alpha * x(i) + beta * y(i));
    }
  }
}

} // end namespace cute
