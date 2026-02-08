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

#include <cute/arch/mma_sm90.hpp>
#include <cute/atom/mma_traits.hpp>

#include <cute/layout.hpp>

namespace cute {

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp64 = fp64 * fp64 + fp64 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

using SM90_16x8x4_F64F64F64F64_TN = SM90::MMA_16x8x4_F64F64F64F64_TN;

template <>
struct MMA_Traits<SM90_16x8x4_F64F64F64F64_TN>
{
  using ValTypeD = double;
  using ValTypeA = double;
  using ValTypeB = double;
  using ValTypeC = double;

  using Shape_MNK = Shape<_16,_8,_4>;
  using ThrID =  Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,_2>,
                         Stride<Stride<_16,_1>,_8>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,_1>,
                         Stride<Stride< _8,_1>,_0>>;
  using CLayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                         Stride<Stride<_32,_1>,Stride<_16,_8>>>;
};

using SM90_16x8x8_F64F64F64F64_TN = SM90::MMA_16x8x8_F64F64F64F64_TN;

template <>
struct MMA_Traits<SM90_16x8x8_F64F64F64F64_TN>
{
  using ValTypeD = double;
  using ValTypeA = double;
  using ValTypeB = double;
  using ValTypeC = double;

  using Shape_MNK = Shape<_16,_8,_8>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape <_2, _2>>,
                         Stride<Stride<_16,_1>,Stride<_8,_64>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>, _2>,
                         Stride<Stride< _8,_1>,_32>>;
  using CLayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                         Stride<Stride<_32,_1>,Stride<_16,_8>>>;
};

using SM90_16x8x16_F64F64F64F64_TN = SM90::MMA_16x8x16_F64F64F64F64_TN;

template <>
struct MMA_Traits<SM90_16x8x16_F64F64F64F64_TN>
{
  using ValTypeD = double;
  using ValTypeA = double;
  using ValTypeB = double;
  using ValTypeC = double;

  using Shape_MNK = Shape<_16,_8,_16>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape <_2, _4>>,
                         Stride<Stride<_16,_1>,Stride<_8,_64>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>, _4>,
                         Stride<Stride< _8,_1>,_32>>;
  using CLayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                         Stride<Stride<_32,_1>,Stride<_16,_8>>>;
};

///////////////////////////////////////////////////////////////////////////////////
//////////////////////// cfp64 = cfp64 * cfp64 + cfp64 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

using SM90_16x8x4_C64C64C64C64_TN  = SM90::MMA_16x8x4_C64C64C64C64_TN;

template <>
struct MMA_Traits<SM90_16x8x4_C64C64C64C64_TN>
     : MMA_Traits<SM90_16x8x4_F64F64F64F64_TN>
{
  using ValTypeD = complex<double>;
  using ValTypeA = complex<double>;
  using ValTypeB = complex<double>;
  using ValTypeC = complex<double>;
};

using SM90_16x8x8_C64C64C64C64_TN  = SM90::MMA_16x8x8_C64C64C64C64_TN;

template <>
struct MMA_Traits<SM90_16x8x8_C64C64C64C64_TN>
     : MMA_Traits<SM90_16x8x8_F64F64F64F64_TN>
{
  using ValTypeD = complex<double>;
  using ValTypeA = complex<double>;
  using ValTypeB = complex<double>;
  using ValTypeC = complex<double>;
};

using SM90_16x8x16_C64C64C64C64_TN = SM90::MMA_16x8x16_C64C64C64C64_TN;

template <>
struct MMA_Traits<SM90_16x8x16_C64C64C64C64_TN>
     : MMA_Traits<SM90_16x8x16_F64F64F64F64_TN>
{
  using ValTypeD = complex<double>;
  using ValTypeA = complex<double>;
  using ValTypeB = complex<double>;
  using ValTypeC = complex<double>;
};

} // end namespace cute
