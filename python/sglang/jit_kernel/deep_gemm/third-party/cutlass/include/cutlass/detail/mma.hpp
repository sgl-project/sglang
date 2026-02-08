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

#include "cutlass/arch/mma.h"
#include "cute/layout.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <class TiledMma, class = void>
struct IsSparseTensorOp : cute::false_type { };

// TiledMma for sparse must have ValTypeE
template <class TiledMma>
struct IsSparseTensorOp<TiledMma, cute::void_t<typename TiledMma::ValTypeE>>
    : cute::true_type { };


template <class TiledMma, class = void>
struct IsBlockScaledTensorOp : cute::false_type { };

// TiledMma for blockScaled must have FrgTypeSFA
template <class TiledMma>
struct IsBlockScaledTensorOp<TiledMma, cute::void_t<typename TiledMma::FrgTypeSFA>>
    : cute::true_type { };


// The following metafunction is used to extract the OperatorClass from a cutlass 3.x kernel.
template <class TiledMma>
struct get_operator_class {
  static constexpr bool is_sparse_op = IsSparseTensorOp<TiledMma>::value;
  static constexpr bool is_block_scaled_op = IsBlockScaledTensorOp<TiledMma>::value;    
  // All tensorop operations have atom shape's M >= 8   
  static constexpr bool is_tensor_op = cute::size<0>(typename TiledMma::AtomShape_MNK{}) >= 8;
  using type = cute::conditional_t<
                is_tensor_op, 
                cute::conditional_t<
                  is_sparse_op,
                  cutlass::arch::OpClassSparseTensorOp,
                  cute::conditional_t<                          
                    is_block_scaled_op,                         
                    cutlass::arch::OpClassBlockScaledTensorOp,  
                    cutlass::arch::OpClassTensorOp
                    >                                           
                  >,
                cutlass::arch::OpClassSimt
                >;
};

template <class T>
using get_operator_class_t = typename get_operator_class<T>::type;

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::detail
