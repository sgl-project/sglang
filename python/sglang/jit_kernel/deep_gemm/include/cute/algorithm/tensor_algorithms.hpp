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
/** Common algorithms on (hierarchical) tensors */

#pragma once

#include <cute/config.hpp>
#include <cute/tensor_impl.hpp>

namespace cute
{

//
// for_each
//

template <class Engine, class Layout, class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
for_each(Tensor<Engine,Layout> const& tensor, UnaryOp&& op)
{
  CUTE_UNROLL
  for (int i = 0; i < size(tensor); ++i) {
    op(tensor(i));
  }
}

template <class Engine, class Layout, class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
for_each(Tensor<Engine,Layout>& tensor, UnaryOp&& op)
{
  CUTE_UNROLL
  for (int i = 0; i < size(tensor); ++i) {
    op(tensor(i));
  }
}

// Accept mutable temporaries
template <class Engine, class Layout, class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
for_each(Tensor<Engine,Layout>&& tensor, UnaryOp&& op)
{
  return for_each(tensor, op);
}

//
// transform
//

// Similar to std::transform but does not return number of elements affected
template <class Engine, class Layout, class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
transform(Tensor<Engine,Layout>& tensor, UnaryOp&& op)
{
  CUTE_UNROLL
  for (int i = 0; i < size(tensor); ++i) {
    tensor(i) = op(tensor(i));
  }
}

// Accept mutable temporaries
template <class Engine, class Layout, class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
transform(Tensor<Engine,Layout>&& tensor, UnaryOp&& op)
{
  return transform(tensor, op);
}

// Similar to std::transform transforms one tensors and assigns it to another
template <class EngineIn, class LayoutIn,
          class EngineOut, class LayoutOut,
          class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
transform(Tensor<EngineIn, LayoutIn > const& tensor_in,
          Tensor<EngineOut,LayoutOut>      & tensor_out,
          UnaryOp&& op)
{
  CUTE_UNROLL
  for (int i = 0; i < size(tensor_in); ++i) {
    tensor_out(i) = op(tensor_in(i));
  }
}

// Accept mutable temporaries
template <class EngineIn, class LayoutIn,
          class EngineOut, class LayoutOut,
          class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
transform(Tensor<EngineIn, LayoutIn > const& tensor_in,
          Tensor<EngineOut,LayoutOut>     && tensor_out,
          UnaryOp&& op)
{
  return transform(tensor_in, tensor_out, op);
}

// Similar to std::transform with a binary operation
// Takes two tensors as input and one tensor as output.
// Applies the binary_op to tensor_in1 and tensor_in2 and
// assigns it to tensor_out
template <class EngineIn1, class LayoutIn1,
          class EngineIn2, class LayoutIn2,
          class EngineOut, class LayoutOut,
          class BinaryOp>
CUTE_HOST_DEVICE constexpr
void
transform(Tensor<EngineIn1,LayoutIn1> const& tensor_in1,
          Tensor<EngineIn2,LayoutIn2> const& tensor_in2,
          Tensor<EngineOut,LayoutOut>      & tensor_out,
          BinaryOp&& op)
{
  CUTE_UNROLL
  for (int i = 0; i < size(tensor_in1); ++i) {
    tensor_out(i) = op(tensor_in1(i), tensor_in2(i));
  }
}

// Accept mutable temporaries
template <class EngineIn1, class LayoutIn1,
          class EngineIn2, class LayoutIn2,
          class EngineOut, class LayoutOut,
          class BinaryOp>
CUTE_HOST_DEVICE constexpr
void
transform(Tensor<EngineIn1,LayoutIn1> const& tensor_in1,
          Tensor<EngineIn2,LayoutIn2> const& tensor_in2,
          Tensor<EngineOut,LayoutOut>     && tensor_out,
          BinaryOp&& op)
{
  return transform(tensor_in1, tensor_in2, tensor_out, op);
}

namespace lazy {

template <class Engine, class Layout, class Fn>
CUTE_HOST_DEVICE constexpr
auto
transform(cute::Tensor<Engine,Layout> const& t, Fn const& fn)
{
  return cute::make_tensor(cute::make_transform_iter(fn, t.data()), t.layout());
}

} // end namespace lazy

} // end namespace cute
