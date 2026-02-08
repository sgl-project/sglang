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
/*! 
  \file
  \brief The universal GEMM accommodates serial reductions, parallel reductions, batched strided, and 
    batched array variants.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA_, 
  typename LayoutA_, 
  ComplexTransform TransformA,
  int AlignmentA,
  typename ElementB_,
  typename LayoutB_,
  ComplexTransform TransformB,
  int AlignmentB,
  typename LayoutC_,
  bool Transpose
>
struct MapArguments {
  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  static ComplexTransform const kTransformA = TransformA;
  static int const kAlignmentA = AlignmentA; 
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  static ComplexTransform const kTransformB = TransformB;
  static int const kAlignmentB = AlignmentB; 
  using LayoutC = LayoutC_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA_, 
  typename LayoutA_, 
  ComplexTransform TransformA,
  int AlignmentA,
  typename ElementB_,
  typename LayoutB_,
  ComplexTransform TransformB,
  int AlignmentB,
  typename LayoutC_
>
struct MapArguments<
  ElementA_,
  LayoutA_,
  TransformA,
  AlignmentA, 
  ElementB_,
  LayoutB_,
  TransformB,
  AlignmentB,
  LayoutC_,
  true
> {
  using ElementA = ElementB_;
  using LayoutA = typename layout::LayoutTranspose<LayoutB_>::type;
  static ComplexTransform const kTransformA = TransformB;
  static int const kAlignmentA = AlignmentB; 
  using ElementB = ElementA_;
  using LayoutB = typename layout::LayoutTranspose<LayoutA_>::type;
  static ComplexTransform const kTransformB = TransformA;
  static int const kAlignmentB = AlignmentA; 
  using LayoutC = typename layout::LayoutTranspose<LayoutC_>::type;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}

/////////////////////////////////////////////////////////////////////////////////////////////////

}
}
}

/////////////////////////////////////////////////////////////////////////////////////////////////
