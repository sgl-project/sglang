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

#include <cute/arch/mma_sm70.hpp>

#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>

namespace cute
{

namespace {

// Logical thread id to thread idx (quadpair)
using SM70_QuadPair = Layout<Shape <_4, _2>,
                             Stride<_1,_16>>;
// (T8,V4) -> (M8,K4)
using SM70_8x4_Row  = Layout<Shape <_8,_4>,
                             Stride<_1,_8>>;
// (T8,V4) -> (M8,K4)
using SM70_8x4_Col  = Layout<Shape <Shape <_4,_2>,_4>,
                             Stride<Stride<_8,_4>,_1>>;
// (T8,V8) -> (M8,N8)
using SM70_8x8_16b  = Layout<Shape <_8,_8>,
                             Stride<_1,_8>>;
// (T8,V8) -> (M8,N8)
using SM70_8x8_32b  = Layout<Shape <Shape <_2, _2,_2>,Shape <_2,_2, _2>>,
                             Stride<Stride<_1,_16,_4>,Stride<_8,_2,_32>>>;

} 

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM70_8x8x4_F16F16F16F16_TN>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = SM70_QuadPair;
  using ALayout = SM70_8x4_Row;
  using BLayout = SM70_8x4_Row;
  using CLayout = SM70_8x8_16b;
};

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM70_8x8x4_F16F16F16F16_NT>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = SM70_QuadPair;
  using ALayout = SM70_8x4_Col;
  using BLayout = SM70_8x4_Col;
  using CLayout = SM70_8x8_16b;
};

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM70_8x8x4_F16F16F16F16_NN>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = SM70_QuadPair;
  using ALayout = SM70_8x4_Col;
  using BLayout = SM70_8x4_Row;
  using CLayout = SM70_8x8_16b;
};

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM70_8x8x4_F16F16F16F16_TT>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = SM70_QuadPair;
  using ALayout = SM70_8x4_Row;
  using BLayout = SM70_8x4_Col;
  using CLayout = SM70_8x8_16b;
};

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM70_8x8x4_F32F16F16F32_TN>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = SM70_QuadPair;
  using ALayout = SM70_8x4_Row;
  using BLayout = SM70_8x4_Row;
  using CLayout = SM70_8x8_32b;
};

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM70_8x8x4_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = SM70_QuadPair;
  using ALayout = SM70_8x4_Col;
  using BLayout = SM70_8x4_Col;
  using CLayout = SM70_8x8_32b;
};

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM70_8x8x4_F32F16F16F32_NN>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = SM70_QuadPair;
  using ALayout = SM70_8x4_Col;
  using BLayout = SM70_8x4_Row;
  using CLayout = SM70_8x8_32b;
};

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM70_8x8x4_F32F16F16F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = SM70_QuadPair;
  using ALayout = SM70_8x4_Row;
  using BLayout = SM70_8x4_Col;
  using CLayout = SM70_8x8_32b;
};

///////////////////////////////////////////////////////////////////////////////
} // namespace cute
