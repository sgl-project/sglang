/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Definitions for GEMM structures
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/arch/wmma.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

////////////////////////////////////////////////////////////////////////////////

template <
  typename OperatorClass,
  typename ArchTag,
  typename ElementA, 
  typename ElementB, 
  typename ElementC,
  typename ElementAccumulator
>
struct DefaultGemmConfiguration;

////////////////////////////////////////////////////////////////////////////////

template <
  typename ArchTag,
  typename ElementA, 
  typename ElementB, 
  typename ElementC, 
  typename ElementAccumulator>
struct DefaultGemmConfiguration<
  arch::OpClassSimt, 
  ArchTag,
  ElementA, 
  ElementB, 
  ElementC, 
  ElementAccumulator> {
  
  static int const kAlignmentA = 1;
  static int const kAlignmentB = 1;
  using ThreadblockShape = GemmShape<128, 128, 8>;
  using WarpShape = GemmShape<32, 64, 8>;
  using InstructionShape = GemmShape<1, 1, 1>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    1,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ArchTag,
  typename ElementC>
struct DefaultGemmConfiguration<arch::OpClassSimt, ArchTag, int8_t, int8_t, ElementC, int32_t> {
  
  static int const kAlignmentA = 4;
  static int const kAlignmentB = 4;
  using ThreadblockShape = GemmShape<128, 128, 32>;
  using WarpShape = GemmShape<32, 64, 32>;
  using InstructionShape = GemmShape<1, 1, 4>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
    ElementC,
    1,
    int32_t,
    float
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

template <
  typename ArchTag,
  typename ElementA, 
  typename ElementB, 
  typename ElementC, 
  typename ElementAccumulator>
struct DefaultGemmConfiguration<
  arch::OpClassWmmaTensorOp, 
  ArchTag,
  ElementA, 
  ElementB, 
  ElementC, 
  ElementAccumulator> {
  
  static int const kAlignmentA = 128 / sizeof_bits<ElementA>::value;
  static int const kAlignmentB = 128 / sizeof_bits<ElementB>::value;

  static int const kStages = 2;
  
  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    128 / sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA, 
  typename ElementB, 
  typename ElementC, 
  typename ElementAccumulator>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm70,
  ElementA, 
  ElementB, 
  ElementC, 
  ElementAccumulator> {
  
  static int const kAlignmentA = 128 / sizeof_bits<ElementA>::value;
  static int const kAlignmentB = 128 / sizeof_bits<ElementB>::value;

  using ThreadblockShape = GemmShape<128, 256, 32>;
  using WarpShape = GemmShape<64, 64, 32>;
  using InstructionShape = GemmShape<8, 8, 4>;
  static int const kStages = 2;
  
  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    128 / sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA, 
  typename ElementB, 
  typename ElementC, 
  typename ElementAccumulator>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm75,
  ElementA, 
  ElementB, 
  ElementC, 
  ElementAccumulator> {

  static int const kAlignmentA = 128 / sizeof_bits<ElementA>::value;
  static int const kAlignmentB = 128 / sizeof_bits<ElementA>::value;
  using ThreadblockShape = GemmShape<128, 256, 32>;
  using WarpShape = GemmShape<64, 64, 32>;
  using InstructionShape = GemmShape<16, 8, 8>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    128 / sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Operator = typename platform::conditional<
      (platform::is_same<ElementA, int8_t>::value ||
       platform::is_same<ElementA, int4b_t>::value ||
       platform::is_same<ElementA, uint8_t>::value ||
       platform::is_same<ElementA, uint4b_t>::value),
      arch::OpMultiplyAddSaturate, arch::OpMultiplyAdd>::type;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm75, 
  int8_t, 
  int8_t, 
  ElementC, 
  int32_t> {
  
  static int const kAlignmentA = 128 / sizeof_bits<int8_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<int8_t>::value;

  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<8, 8, 16>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm75, 
  int8_t, 
  uint8_t, 
  ElementC, 
  int32_t> {
  
  static int const kAlignmentA = 128 / sizeof_bits<int8_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint8_t>::value;
 
  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<8, 8, 16>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm75, 
  uint8_t, 
  int8_t, 
  ElementC, 
  int32_t> {
  
  static int const kAlignmentA = 128 / sizeof_bits<uint8_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<int8_t>::value;
 
  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<8, 8, 16>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm75, 
  uint8_t, 
  uint8_t, 
  ElementC, 
  int32_t> {
  
  static int const kAlignmentA = 128 / sizeof_bits<uint8_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint8_t>::value;
 
  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<8, 8, 16>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm75, 
  int4b_t, 
  int4b_t, 
  ElementC, 
  int32_t> {
   
  static int const kAlignmentA = 128 / sizeof_bits<int4b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<int4b_t>::value;
 
  using ThreadblockShape = GemmShape<128, 256, 128>;
  using WarpShape = GemmShape<64, 64, 128>;
  using InstructionShape = GemmShape<8, 8, 32>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm75, 
  int4b_t, 
  uint4b_t, 
  ElementC, 
  int32_t> {
    
  static int const kAlignmentA = 128 / sizeof_bits<int4b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint4b_t>::value;
 
  using ThreadblockShape = GemmShape<128, 256, 128>;
  using WarpShape = GemmShape<64, 64, 128>;
  using InstructionShape = GemmShape<8, 8, 32>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm75, 
  uint4b_t, 
  int4b_t, 
  ElementC, 
  int32_t> {
  
  static int const kAlignmentA = 128 / sizeof_bits<uint4b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<int4b_t>::value;

  using ThreadblockShape = GemmShape<128, 256, 128>;
  using WarpShape = GemmShape<64, 64, 128>;
  using InstructionShape = GemmShape<8, 8, 32>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm75, 
  uint4b_t, 
  uint4b_t, 
  ElementC, 
  int32_t> {
   
  static int const kAlignmentA = 128 / sizeof_bits<uint4b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint4b_t>::value;
 
  using ThreadblockShape = GemmShape<128, 256, 128>;
  using WarpShape = GemmShape<64, 64, 128>;
  using InstructionShape = GemmShape<8, 8, 32>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm75, 
  uint1b_t, 
  uint1b_t, 
  ElementC, 
  int32_t> {
    
  static int const kAlignmentA = 128 / sizeof_bits<uint1b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint1b_t>::value;
 
  using ThreadblockShape = GemmShape<128, 256, 512>;
  using WarpShape = GemmShape<64, 64, 512>;
  using InstructionShape = GemmShape<8, 8, 128>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpXorPopc;
};

////////////////////////////////////////////////////////////////////////////////

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementAccumulator>
struct DefaultGemmConfiguration<arch::OpClassTensorOp, arch::Sm80, ElementA,
                                ElementB, ElementC, ElementAccumulator> {

  static int const kAlignmentA = 128 / sizeof_bits<ElementA>::value;
  static int const kAlignmentB = 128 / sizeof_bits<ElementA>::value;
  
  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<16, 8, 16>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
      ElementC, 128 / sizeof_bits<ElementC>::value, ElementAccumulator,
      ElementAccumulator>;

  using Operator = typename platform::conditional<
      (platform::is_same<ElementA, int8_t>::value ||
       platform::is_same<ElementA, int4b_t>::value ||
       platform::is_same<ElementA, uint8_t>::value ||
       platform::is_same<ElementA, uint4b_t>::value),
      arch::OpMultiplyAddSaturate, arch::OpMultiplyAdd>::type;
};

////////////////////////////////////////////////////////////////////////////////
template <typename ElementC,
          typename ElementAccumulator>
struct DefaultGemmConfiguration<arch::OpClassTensorOp, arch::Sm80, double,
                                double, ElementC, ElementAccumulator> {

  static int const kAlignmentA = 1;
  static int const kAlignmentB = 1;
  
  using ThreadblockShape = GemmShape<128, 128, 16>;
  using WarpShape = GemmShape<32, 64, 16>;
  using InstructionShape = GemmShape<8, 8, 4>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
      ElementC, 1, ElementAccumulator,
      ElementAccumulator>;

  using Operator = arch::OpMultiplyAdd;
};


template <>
struct DefaultGemmConfiguration<
    arch::OpClassTensorOp, 
    arch::Sm80, 
    complex<double>,
    complex<double>, 
    complex<double>,
    complex<double>
  > {

  static int const kAlignmentA = 1;
  static int const kAlignmentB = 1;
  
  using ThreadblockShape = GemmShape<64, 64, 16>;
  using WarpShape = GemmShape<32, 32, 16>;
  using InstructionShape = GemmShape<8, 8, 4>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
      complex<double>, 1, complex<double>,
      complex<double>>;

  using Operator = arch::OpMultiplyAddComplex;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm80, 
  int8_t, 
  int8_t, 
  ElementC, 
  int32_t> {
     
  static int const kAlignmentA = 128 / sizeof_bits<int8_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<int8_t>::value;
 
  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<16, 8, 32>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm80, 
  int8_t, 
  uint8_t, 
  ElementC, 
  int32_t> {
      
  static int const kAlignmentA = 128 / sizeof_bits<int8_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint8_t>::value;
  
  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<16, 8, 32>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm80, 
  uint8_t, 
  int8_t, 
  ElementC, 
  int32_t> {
      
  static int const kAlignmentA = 128 / sizeof_bits<uint8_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<int8_t>::value;
  
  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<16, 8, 32>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm80, 
  uint8_t, 
  uint8_t, 
  ElementC, 
  int32_t> {
      
  static int const kAlignmentA = 128 / sizeof_bits<uint8_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint8_t>::value;
  
  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<16, 8, 32>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm80, 
  int4b_t, 
  int4b_t, 
  ElementC, 
  int32_t> {
      
  static int const kAlignmentA = 128 / sizeof_bits<int4b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<int4b_t>::value;
  
  using ThreadblockShape = GemmShape<128, 256, 128>;
  using WarpShape = GemmShape<64, 64, 128>;
  using InstructionShape = GemmShape<16, 8, 64>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm80, 
  int4b_t, 
  uint4b_t, 
  ElementC, 
  int32_t> {
       
  static int const kAlignmentA = 128 / sizeof_bits<int4b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint4b_t>::value;
  
  using ThreadblockShape = GemmShape<128, 256, 128>;
  using WarpShape = GemmShape<64, 64, 128>;
  using InstructionShape = GemmShape<16, 8, 64>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm80, 
  uint4b_t, 
  int4b_t, 
  ElementC, 
  int32_t> {
       
  static int const kAlignmentA = 128 / sizeof_bits<uint4b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<int4b_t>::value;
  
  using ThreadblockShape = GemmShape<128, 256, 128>;
  using WarpShape = GemmShape<64, 64, 128>;
  using InstructionShape = GemmShape<16, 8, 64>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm80, 
  uint4b_t, 
  uint4b_t, 
  ElementC, 
  int32_t> {
       
  static int const kAlignmentA = 128 / sizeof_bits<uint4b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint4b_t>::value;
  
  using ThreadblockShape = GemmShape<128, 256, 128>;
  using WarpShape = GemmShape<64, 64, 128>;
  using InstructionShape = GemmShape<16, 8, 64>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Sm80, 
  uint1b_t, 
  uint1b_t, 
  ElementC, 
  int32_t> {
       
  static int const kAlignmentA = 128 / sizeof_bits<uint1b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint1b_t>::value;
  
  using ThreadblockShape = GemmShape<128, 256, 512>;
  using WarpShape = GemmShape<64, 64, 512>;
  using InstructionShape = GemmShape<16, 8, 256>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp,
  arch::Sm80,
  int4b_t,
  int8_t,
  ElementC,
  int32_t> {

  static int const kAlignmentA = 128 / sizeof_bits<int4b_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<int8_t>::value;

  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<16, 8, 32>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp,
  arch::Sm80,
  int8_t,
  int4b_t,
  ElementC,
  int32_t> {

  static int const kAlignmentA = 128 / sizeof_bits<int8_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<int4b_t>::value;

  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<16, 8, 32>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
      ElementC, 128 / sizeof_bits<ElementC>::value, int32_t, float>;

  using Operator = arch::OpMultiplyAddSaturate;
};

////////////////////////////////////////////////////////////////////////////////

/// Base configuration for all {fe4m3, fe5m2} x {fe4m3, fe5m2} combinations on SM89
template <
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename ElementAccumulator>
struct DefaultGemmConfigurationSm89F8 {
  static_assert((platform::is_same<ElementA, cutlass::float_e4m3_t>::value ||
                 platform::is_same<ElementA, cutlass::float_e5m2_t>::value),
                "ElementA must be of type float_e4m3_t or float_e5m2_t");
  static_assert((platform::is_same<ElementB, cutlass::float_e4m3_t>::value ||
                 platform::is_same<ElementB, cutlass::float_e5m2_t>::value),
                "ElementB must be of type float_e4m3_t or float_e5m2_t");

  static int const kAlignmentA = 128 / sizeof_bits<ElementA>::value;
  static int const kAlignmentB = 128 / sizeof_bits<ElementB>::value;

  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<16, 8, 32>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
      ElementC, 128 / sizeof_bits<ElementC>::value, ElementAccumulator,
      ElementAccumulator>;

  using Operator = arch::OpMultiplyAdd;
};

/// Partial specialization for SM89 fe4m3 x fe4m3
template <typename ElementC, typename ElementAccumulator>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp,
  arch::Sm89,
  cutlass::float_e4m3_t,
  cutlass::float_e4m3_t,
  ElementC,
  ElementAccumulator> : DefaultGemmConfigurationSm89F8<
                            cutlass::float_e4m3_t,
                            cutlass::float_e4m3_t,
                            ElementC,
                            ElementAccumulator> {};

/// Partial specialization for SM89 fe4m3 x fe5m2
template <typename ElementC, typename ElementAccumulator>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp,
  arch::Sm89,
  cutlass::float_e4m3_t,
  cutlass::float_e5m2_t,
  ElementC,
  ElementAccumulator> : DefaultGemmConfigurationSm89F8<
                            cutlass::float_e4m3_t,
                            cutlass::float_e5m2_t,
                            ElementC,
                            ElementAccumulator> {};

/// Partial specialization for SM89 fe5m2 x fe4m3
template <typename ElementC, typename ElementAccumulator>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp,
  arch::Sm89,
  cutlass::float_e5m2_t,
  cutlass::float_e4m3_t,
  ElementC,
  ElementAccumulator> : DefaultGemmConfigurationSm89F8<
                            cutlass::float_e5m2_t,
                            cutlass::float_e4m3_t,
                            ElementC,
                            ElementAccumulator> {};

/// Partial specialization for SM89 fe5m2 x fe5m2
template <typename ElementC, typename ElementAccumulator>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp,
  arch::Sm89,
  cutlass::float_e5m2_t,
  cutlass::float_e5m2_t,
  ElementC,
  ElementAccumulator> : DefaultGemmConfigurationSm89F8<
                            cutlass::float_e5m2_t,
                            cutlass::float_e5m2_t,
                            ElementC,
                            ElementAccumulator> {};

////////////////////////////////////////////////////////////////////////////////

template <typename ElementC,
          typename ElementAccumulator>
struct DefaultGemmConfiguration<arch::OpClassTensorOp, arch::Sm90, double,
                                double, ElementC, ElementAccumulator> {

  static int const kAlignmentA = 1;
  static int const kAlignmentB = 1;
  
  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<16, 8, 4>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
      ElementC, 1, ElementAccumulator,
      ElementAccumulator>;

  using Operator = arch::OpMultiplyAdd;
};

template <>
struct DefaultGemmConfiguration<
    arch::OpClassTensorOp, 
    arch::Sm90, 
    complex<double>,
    complex<double>, 
    complex<double>,
    complex<double>
  > {

  static int const kAlignmentA = 1;
  static int const kAlignmentB = 1;
  
  using ThreadblockShape = GemmShape<64, 64, 16>;
  using WarpShape = GemmShape<32, 32, 16>;
  using InstructionShape = GemmShape<16, 8, 4>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
      complex<double>, 1, complex<double>,
      complex<double>>;

  using Operator = arch::OpMultiplyAddComplex;
};

} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
