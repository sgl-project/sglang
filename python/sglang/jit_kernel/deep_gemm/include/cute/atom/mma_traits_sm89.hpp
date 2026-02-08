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

//

//
#pragma once

#include <cute/arch/mma_sm89.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace cute 
{

namespace {

// (T32,V4) -> (M16,N8)
using SM80_16x8_Row = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                             Stride<Stride<_32,_1>,Stride<_16,_8>>>;

}

template <>
struct MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN> {
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16,_8,_32>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _4,_2,  _2>>,
                         Stride<Stride<_64,_1>,Stride<_16,_8,_256>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,Shape <_4,  _2>>,
                         Stride<Stride<_32,_1>,Stride<_8,_128>>>;
  using CLayout = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM89_16x8x32_F32E4M3E5M2F32_TN> 
     : MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN> {
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;
};

template <>
struct MMA_Traits<SM89_16x8x32_F32E5M2E5M2F32_TN>
     : MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN> {
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;
};

template <>
struct MMA_Traits<SM89_16x8x32_F32E5M2E4M3F32_TN>
     : MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN> {
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;
};

template <>
struct MMA_Traits<SM89_16x8x32_F16E4M3E4M3F16_TN>
     : MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN> {
  using ValTypeD = cutlass::half_t;
  using ValTypeA = cutlass::float_e4m3_t;
  using ValTypeB = cutlass::float_e4m3_t;
  using ValTypeC = cutlass::half_t;
};

template <>
struct MMA_Traits<SM89_16x8x32_F16E4M3E5M2F16_TN>
     : MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN> {
  using ValTypeD = cutlass::half_t;
  using ValTypeA = cutlass::float_e4m3_t;
  using ValTypeB = cutlass::float_e5m2_t;
  using ValTypeC = cutlass::half_t;
};

template <>
struct MMA_Traits<SM89_16x8x32_F16E5M2E5M2F16_TN>
     : MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN> {
  using ValTypeD = cutlass::half_t;
  using ValTypeA = cutlass::float_e5m2_t;
  using ValTypeB = cutlass::float_e5m2_t;
  using ValTypeC = cutlass::half_t;
};

template <>
struct MMA_Traits<SM89_16x8x32_F16E5M2E4M3F16_TN> 
     : MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN> {
  using ValTypeD = cutlass::half_t;
  using ValTypeA = cutlass::float_e5m2_t;
  using ValTypeB = cutlass::float_e4m3_t;
  using ValTypeC = cutlass::half_t;
};

} // end namespace cute
