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

#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace cute
{

namespace {

// (T32,V1) -> (M8,N8)
using SM80_8x4      = Layout<Shape <Shape < _4,_8>,_1>,
                             Stride<Stride< _8,_1>,_0>>;
// (T32,V2) -> (M8,N8)
using SM80_8x8_Row  = Layout<Shape <Shape < _4,_8>,_2>,
                             Stride<Stride<_16,_1>,_8>>;
// (T32,V4) -> (M8,N16)
using SM80_8x16_Row = Layout<Shape <Shape < _4,_8>,_4>,
                             Stride<Stride<_32,_1>,_8>>;
// (T32,V4) -> (M16,N8)
using SM80_16x8_Row = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                             Stride<Stride<_32,_1>,Stride<_16,_8>>>;

}

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp16 = fp16 * fp16 + fp16 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_16x8x8_F16F16F16F16_TN>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_16,_8,_8>;
  using ThrID   = Layout<_32>;
  using ALayout = SM80_16x8_Row;
  using BLayout = SM80_8x8_Row;
  using CLayout = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM80_16x8x16_F16F16F16F16_TN>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_16,_8,_16>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2,  _2>>,
                         Stride<Stride<_32,_1>,Stride<_16,_8,_128>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,Shape <_2, _2>>,
                         Stride<Stride<_16,_1>,Stride<_8,_64>>>;
  using CLayout = SM80_16x8_Row;
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp32 = fp16 * fp16 + fp32 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_16x8x8_F32F16F16F32_TN>
     : MMA_Traits<SM80_16x8x8_F16F16F16F16_TN>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;
};

template <>
struct MMA_Traits<SM80_16x8x16_F32F16F16F32_TN>
     : MMA_Traits<SM80_16x8x16_F16F16F16F16_TN>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp32 = bf16 * bf16 + fp32 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_16x8x8_F32BF16BF16F32_TN>
     : MMA_Traits<SM80_16x8x8_F16F16F16F16_TN>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

template <>
struct MMA_Traits<SM80_16x8x16_F32BF16BF16F32_TN>
     : MMA_Traits<SM80_16x8x16_F16F16F16F16_TN>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp32 = tf32 * tf32 + fp32 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_16x8x4_F32TF32TF32F32_TN>
{
  using ValTypeD = float;
  using ValTypeA = cutlass::tfloat32_t;
  using ValTypeB = cutlass::tfloat32_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16,_8,_4>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,_2>,
                         Stride<Stride<_16,_1>,_8>>;
  using BLayout = SM80_8x4;
  using CLayout = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM80_16x8x8_F32TF32TF32F32_TN>
{
  using ValTypeD = float;
  using ValTypeA = cutlass::tfloat32_t;
  using ValTypeB = cutlass::tfloat32_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16,_8,_8>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape <_2, _2>>,
                         Stride<Stride<_16,_1>,Stride<_8,_64>>>;
  using BLayout = Layout<Shape <Shape <_4,_8>, _2>,
                         Stride<Stride<_8,_1>,_32>>;
  using CLayout = SM80_16x8_Row;
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////// fp64 = fp64 * fp64 + fp64 ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_8x8x4_F64F64F64F64_TN>
{
  using ValTypeD = double;
  using ValTypeA = double;
  using ValTypeB = double;
  using ValTypeC = double;

  using Shape_MNK = Shape<_8,_8,_4>;
  using ThrID   = Layout<_32>;
  using ALayout = SM80_8x4;
  using BLayout = SM80_8x4;
  using CLayout = SM80_8x8_Row;
};

// Custom complex fp64 MMA composed of 4 fp64 MMAs -- same layouts
template <>
struct MMA_Traits<SM80_8x8x4_C64C64C64C64_TN>
     : MMA_Traits<SM80_8x8x4_F64F64F64F64_TN>
{
  using ValTypeD = complex<double>;
  using ValTypeA = complex<double>;
  using ValTypeB = complex<double>;
  using ValTypeC = complex<double>;
};

// Custom complex fp64 MMA composed of 3 fp64 MMAs -- same layouts
template <>
struct MMA_Traits<SM80_8x8x4_GC64C64C64GC64_TN>
     : MMA_Traits<SM80_8x8x4_F64F64F64F64_TN>
{
  using ValTypeD = typename SM80_8x8x4_GC64C64C64GC64_TN::GaussComplex;
  using ValTypeA = complex<double>;
  using ValTypeB = complex<double>;
  using ValTypeC = typename SM80_8x8x4_GC64C64C64GC64_TN::GaussComplex;
};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = s8 * s8 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_8x8x16_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_8,_8,_16>;
  using ThrID   = Layout<_32>;
  using ALayout = SM80_8x16_Row;
  using BLayout = SM80_8x16_Row;
  using CLayout = SM80_8x8_Row;
};

template <>
struct MMA_Traits<SM80_8x8x16_S32S8S8S32_TN_SATURATE>
     : MMA_Traits<SM80_8x8x16_S32S8S8S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x16_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_16,_8,_16>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _4,_2>>,
                         Stride<Stride<_64,_1>,Stride<_16,_8>>>;
  using BLayout = SM80_8x16_Row;
  using CLayout = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM80_16x8x16_S32S8S8S32_TN_SATURATE>
     : MMA_Traits<SM80_16x8x16_S32S8S8S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x32_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_16,_8,_32>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _4,_2,  _2>>,
                         Stride<Stride<_64,_1>,Stride<_16,_8,_256>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>, Shape <_4,  _2>>,
                         Stride<Stride<_32,_1>, Stride<_8,_128>>>;
  using CLayout = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM80_16x8x32_S32S8S8S32_TN_SATURATE>
     : MMA_Traits<SM80_16x8x32_S32S8S8S32_TN> {};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = s8 * u8 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_8x8x16_S32S8U8S32_TN>
     : MMA_Traits<SM80_8x8x16_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<SM80_8x8x16_S32S8U8S32_TN_SATURATE>
     : MMA_Traits<SM80_8x8x16_S32S8U8S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x16_S32S8U8S32_TN>
     : MMA_Traits<SM80_16x8x16_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<SM80_16x8x16_S32S8U8S32_TN_SATURATE>
     : MMA_Traits<SM80_16x8x16_S32S8U8S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x32_S32S8U8S32_TN>
     : MMA_Traits<SM80_16x8x32_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<SM80_16x8x32_S32S8U8S32_TN_SATURATE>
     : MMA_Traits<SM80_16x8x32_S32S8U8S32_TN> {};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = u8 * s8 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_8x8x16_S32U8S8S32_TN>
     : MMA_Traits<SM80_8x8x16_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<SM80_8x8x16_S32U8S8S32_TN_SATURATE>
     : MMA_Traits<SM80_8x8x16_S32U8S8S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x16_S32U8S8S32_TN>
     : MMA_Traits<SM80_16x8x16_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<SM80_16x8x16_S32U8S8S32_TN_SATURATE>
     : MMA_Traits<SM80_16x8x16_S32U8S8S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x32_S32U8S8S32_TN>
     : MMA_Traits<SM80_16x8x32_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<SM80_16x8x32_S32U8S8S32_TN_SATURATE>
     : MMA_Traits<SM80_16x8x32_S32U8S8S32_TN> {};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = u8 * u8 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_8x8x16_S32U8U8S32_TN>
     : MMA_Traits<SM80_8x8x16_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<SM80_8x8x16_S32U8U8S32_TN_SATURATE>
     : MMA_Traits<SM80_8x8x16_S32U8U8S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x16_S32U8U8S32_TN>
     : MMA_Traits<SM80_16x8x16_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<SM80_16x8x16_S32U8U8S32_TN_SATURATE>
     : MMA_Traits<SM80_16x8x16_S32U8U8S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x32_S32U8U8S32_TN>
     : MMA_Traits<SM80_16x8x32_S32S8S8S32_TN>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<SM80_16x8x32_S32U8U8S32_TN_SATURATE>
     : MMA_Traits<SM80_16x8x32_S32U8U8S32_TN> {};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = s4 * s4 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_8x8x32_S32S4S4S32_TN> {
  using ValTypeD = int32_t;
  using ValTypeA = int4b_t;
  using ValTypeB = int4b_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_8, _8, _32>;
  using ThrID = Layout<_32>;
  // (T32,V8) -> (M8,N32)
  using ALayout = Layout<Shape <Shape < _4, _8>, Shape <_8>>,
                         Stride<Stride<_64, _1>, Stride<_8>>>;
  using BLayout = Layout<Shape <Shape < _4, _8>, Shape <_8>>,
                         Stride<Stride<_64, _1>, Stride<_8>>>;
  using CLayout = SM80_8x8_Row;
};

template <>
struct MMA_Traits<SM80_8x8x32_S32S4S4S32_TN_SATURATE>
     : MMA_Traits<SM80_8x8x32_S32S4S4S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x32_S32S4S4S32_TN> {
  using ValTypeD = int32_t;
  using ValTypeA = int4b_t;
  using ValTypeB = int4b_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_16, _8, _32>;
  using ThrID = Layout<_32>;
  // (T32,V16) -> (M16,N32)
  using ALayout = Layout<Shape <Shape <  _4, _8>, Shape < _8, _2>>,
                         Stride<Stride<_128, _1>, Stride<_16, _8>>>;
  // (T32,V8) -> (M8,N32)
  using BLayout = Layout<Shape <Shape < _4, _8>, Shape <_8>>,
                         Stride<Stride<_32, _1>, Stride<_8>>>;
  using CLayout = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM80_16x8x32_S32S4S4S32_TN_SATURATE>
     : MMA_Traits<SM80_16x8x32_S32S4S4S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x64_S32S4S4S32_TN> {
  using ValTypeD = int32_t;
  using ValTypeA = int4b_t;
  using ValTypeB = int4b_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_16, _8, _64>;
  using ThrID = Layout<_32>;
  // (T32,V32) -> (M16,N64)
  using ALayout = Layout<Shape <Shape <  _4, _8>, Shape < _8, _2, _2>>,
                         Stride<Stride<_128, _1>, Stride<_16, _8, _512>>>;
  // (T32,V16) -> (M8,N64)
  using BLayout = Layout<Shape <Shape < _4, _8>, Shape <_8, _2>>,
                         Stride<Stride<_64, _1>, Stride<_8, _256>>>;
  using CLayout = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM80_16x8x64_S32S4S4S32_TN_SATURATE>
     : MMA_Traits<SM80_16x8x64_S32S4S4S32_TN> {};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = s4 * u4 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_8x8x32_S32S4U4S32_TN>
     : MMA_Traits<SM80_8x8x32_S32S4S4S32_TN> {
  using ValTypeD = int32_t;
  using ValTypeA = int4b_t;
  using ValTypeB = uint4b_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<SM80_8x8x32_S32S4U4S32_TN_SATURATE>
     : MMA_Traits<SM80_8x8x32_S32S4U4S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x32_S32S4U4S32_TN>
    : MMA_Traits<SM80_16x8x32_S32S4S4S32_TN> {
  using ValTypeD = int32_t;
  using ValTypeA = int4b_t;
  using ValTypeB = uint4b_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<SM80_16x8x32_S32S4U4S32_TN_SATURATE>
     : MMA_Traits<SM80_16x8x32_S32S4U4S32_TN> {};
template <>
struct MMA_Traits<SM80_16x8x64_S32S4U4S32_TN>
     : MMA_Traits<SM80_16x8x64_S32S4S4S32_TN> {
  using ValTypeD = int32_t;
  using ValTypeA = int4b_t;
  using ValTypeB = uint4b_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<SM80_16x8x64_S32S4U4S32_TN_SATURATE>
     : MMA_Traits<SM80_16x8x64_S32S4U4S32_TN> {};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = u4 * s4 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_8x8x32_S32U4S4S32_TN>
     : MMA_Traits<SM80_8x8x32_S32S4S4S32_TN> {
  using ValTypeD = int32_t;
  using ValTypeA = uint4b_t;
  using ValTypeB = int4b_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<SM80_8x8x32_S32U4S4S32_TN_SATURATE>
     : MMA_Traits<SM80_8x8x32_S32U4S4S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x32_S32U4S4S32_TN>
     : MMA_Traits<SM80_16x8x32_S32S4S4S32_TN> {
  using ValTypeD = int32_t;
  using ValTypeA = uint4b_t;
  using ValTypeB = int4b_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<SM80_16x8x32_S32U4S4S32_TN_SATURATE>
     : MMA_Traits<SM80_16x8x32_S32U4S4S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x64_S32U4S4S32_TN>
     : MMA_Traits<SM80_16x8x64_S32S4S4S32_TN> {
  using ValTypeD = int32_t;
  using ValTypeA = uint4b_t;
  using ValTypeB = int4b_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<SM80_16x8x64_S32U4S4S32_TN_SATURATE>
    : MMA_Traits<SM80_16x8x64_S32U4S4S32_TN> {};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = u4 * u4 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_8x8x32_S32U4U4S32_TN>
     : MMA_Traits<SM80_8x8x32_S32S4S4S32_TN> {
  using ValTypeD = int32_t;
  using ValTypeA = uint4b_t;
  using ValTypeB = uint4b_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<SM80_8x8x32_S32U4U4S32_TN_SATURATE>
     : MMA_Traits<SM80_8x8x32_S32U4U4S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x32_S32U4U4S32_TN>
     : MMA_Traits<SM80_16x8x32_S32S4S4S32_TN> {
  using ValTypeD = int32_t;
  using ValTypeA = uint4b_t;
  using ValTypeB = uint4b_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<SM80_16x8x32_S32U4U4S32_TN_SATURATE>
     : MMA_Traits<SM80_16x8x32_S32U4U4S32_TN> {};

template <>
struct MMA_Traits<SM80_16x8x64_S32U4U4S32_TN>
     : MMA_Traits<SM80_16x8x64_S32S4S4S32_TN> {
  using ValTypeD = int32_t;
  using ValTypeA = uint4b_t;
  using ValTypeB = uint4b_t;
  using ValTypeC = int32_t;
};

template <>
struct MMA_Traits<SM80_16x8x64_S32U4U4S32_TN_SATURATE>
     : MMA_Traits<SM80_16x8x64_S32U4U4S32_TN> {};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = b1 ^ b1 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_16x8x256_S32U1U1S32_TN_XORPOPC>
{
  using ValTypeD = int32_t;
  using ValTypeA = cute::uint1b_t;
  using ValTypeB = cute::uint1b_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_16,_8,_256>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape<Shape<_4,_8>,Shape<_32,_2,_2>>,
                       Stride<Stride<_512,_1>,Stride<_16,_8,_2048>>>;
  using BLayout = Layout<Shape<Shape <_4,_8>,Shape<_32,_2>>,
                         Stride<Stride<_256,_1>,Stride< _8,_1024>>>;
  using CLayout = SM80_16x8_Row;
};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = b1 & b1 + s32 ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_16x8x256_S32U1U1S32_TN_ANDPOPC>
     : MMA_Traits<SM80_16x8x256_S32U1U1S32_TN_XORPOPC> {};

template<>
struct MMA_Traits<SM80_8x8x128_S32U1U1S32_TN_XORPOPC>
{
  using ValTypeD = int32_t;
  using ValTypeA = cute::uint1b_t;
  using ValTypeB = cute::uint1b_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_8,_8,_128>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape<Shape<_4,_8>,_32>,
                         Stride<Stride<_256,_1>,_8>>;
  using BLayout = Layout<Shape<Shape<_4,_8>,_32>,
                         Stride<Stride<_256,_1>,_8>>;
  using CLayout = SM80_8x8_Row;
};

template <>
struct MMA_Traits<SM80_8x8x128_S32U1U1S32_TN_ANDPOPC>
      :MMA_Traits<SM80_8x8x128_S32U1U1S32_TN_XORPOPC> {};

template<>
struct MMA_Traits<SM80_16x8x128_S32U1U1S32_TN_XORPOPC>
{
  using ValTypeD = int32_t;
  using ValTypeA = cute::uint1b_t;
  using ValTypeB = cute::uint1b_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_16,_8,_128>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape<Shape<_4,_8>,Shape<_32,_2>>,
                       Stride<Stride<_512,_1>,Stride<Stride<_16,_8>>>>;
  using BLayout = Layout<Shape <Shape<_4,_8>,_32>,
                         Stride<Stride<_256,_1>,_8>>;
  using CLayout = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM80_16x8x128_S32U1U1S32_TN_ANDPOPC>
      :MMA_Traits<SM80_16x8x128_S32U1U1S32_TN_XORPOPC> {};

} // end namespace cute
