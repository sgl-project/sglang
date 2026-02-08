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

#include <iostream>

#include <cute/config.hpp>
#include <cute/tensor_impl.hpp>
#include <cute/algorithm/functional.hpp>
#include <cute/algorithm/fill.hpp>

namespace cute
{

// Reduce @src tensor using binary reduction operator @op and initial value @init and return a scalar.
template <class SrcEngine, class SrcLayout, class T, class BinaryOp = cute::plus>
CUTE_HOST_DEVICE constexpr
T
reduce(Tensor<SrcEngine,SrcLayout> const& src, T init, BinaryOp op = {})
{
  for (auto i = 0; i < size(src); ++i) {
    init = op(init, src(i));
  }
  return init;
}

// Reduce @src tensor RedMode using binary reduction operator @op and store the result in @dst tensor
// for each index in @dst/BatchMode.
// @pre @src tensor has rank 2
// @pre size of @src batch mode is equal to size of @dst batch mode
template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout,
          class BinaryOp = cute::plus>
CUTE_HOST_DEVICE constexpr
void
batch_reduce(Tensor<SrcEngine, SrcLayout> const& src,       // (RedMode, BatchMode)
             Tensor<DstEngine, DstLayout>      & dst,       // (BatchMode)
             BinaryOp op = {})
{
  // Precondition
  CUTE_STATIC_ASSERT_V(rank(src) == Int<2>{});
  assert(size<1>(src) == size(dst));

  for (int i = 0; i < size(dst); ++i) {
    dst(i) = reduce(src(_,i), dst(i), op);
  }
}


// Reduce @src tensor along selected modes specified in @target_profile using binary reduction operator @op
// and store the result in @dst tensor. @target_profile is a tuple where '_' indicates modes to keep and 
// integers indicates modes to reduce.
// @pre @target_profile is compatible with @src layout
template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout,
          class TargetProfile,
          class BinaryOp = cute::plus>
CUTE_HOST_DEVICE constexpr
void
logical_reduce(Tensor<SrcEngine, SrcLayout> const& src,
               Tensor<DstEngine, DstLayout>      & dst,
               TargetProfile                const& target_profile,
               BinaryOp op = {})
{
  // Precondition
  assert(compatible(target_profile, shape(src)));

  auto diced_layout = dice(target_profile, src.layout());
  auto sliced_layout = slice(target_profile, src.layout());

  auto red_mode = conditional_return<rank(diced_layout) == Int<0>{}>(Layout<_1,_0>{}, diced_layout);
  auto batch_mode = conditional_return<rank(sliced_layout) == Int<0>{}>(Layout<_1,_0>{}, sliced_layout);

  auto src_tensor = make_tensor(src.data(), make_layout(red_mode, batch_mode));

  batch_reduce(src_tensor, dst, op);
}

} // end namespace cute
