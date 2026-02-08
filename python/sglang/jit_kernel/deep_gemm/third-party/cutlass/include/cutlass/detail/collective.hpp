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

#include "cute/container/tuple.hpp"
#include "cute/layout.hpp" // cute::size(shape)
#include "cute/arch/mma_sm100_desc.hpp" // cute::UMMA::MXF4Format, cute::UMMA::MXF8F6F4Format 
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <size_t I, class Tuple>
struct deduce_mixed_width_dtype {
static_assert(I >= 0u && I <= 2u, "Valid indices are 0, 1, and 2, which represent Operand, Scale, and Bias, respectively.");

private:
  using underlying_tuple = cute::conditional_t<cute::is_tuple<Tuple>::value, Tuple, cute::tuple<Tuple>>;
  static constexpr size_t valid_index = cute::min(I, cute::tuple_size_v<underlying_tuple> - 1);

public:
  using type = cute::conditional_t<(I < cute::tuple_size_v<underlying_tuple>), 
                                    cute::tuple_element_t<valid_index, underlying_tuple>,
                                    void>;
};

template <size_t I, class Tuple>
using deduce_mixed_width_dtype_t = typename deduce_mixed_width_dtype<I, Tuple>::type;



template <class Element>
CUTLASS_HOST_DEVICE
static constexpr bool
is_sm10x_runtime_f8f6f4() {
  return (cute::is_same_v<Element, cutlass::type_erased_dynamic_float8_t> ||
          cute::is_same_v<Element, cutlass::type_erased_dynamic_float6_t> ||
          cute::is_same_v<Element, cutlass::type_erased_dynamic_float4_t>);
}

template <class ElementA, class ElementB>
CUTLASS_HOST_DEVICE
static constexpr bool
is_sm10x_f8f6f4_inputs() {
   return ( 
            
            cute::is_same_v<ElementA, cute::type_erased_dynamic_float8_t> || 
            cute::is_same_v<ElementA, cute::type_erased_dynamic_float6_t> ||
            cute::is_same_v<ElementA, cute::type_erased_dynamic_float4_t> ||
            
            cute::is_same_v<ElementA, cute::float_e4m3_t> ||
            cute::is_same_v<ElementA, cute::float_e5m2_t> 
            || cute::is_same_v<ElementA, cute::float_e3m2_t> ||
            cute::is_same_v<ElementA, cute::float_e2m3_t> ||
            cute::is_same_v<ElementA, cute::float_e2m1_t>
            
          ) &&
          ( 
            
            cute::is_same_v<ElementB, cute::type_erased_dynamic_float8_t> ||
            cute::is_same_v<ElementB, cute::type_erased_dynamic_float6_t> ||
            cute::is_same_v<ElementB, cute::type_erased_dynamic_float4_t> ||
            
            cute::is_same_v<ElementB, cute::float_e4m3_t> ||
            cute::is_same_v<ElementB, cute::float_e5m2_t> 
            || cute::is_same_v<ElementB, cute::float_e3m2_t> ||
            cute::is_same_v<ElementB, cute::float_e2m3_t> ||
            cute::is_same_v<ElementB, cute::float_e2m1_t>
            
          );
}

template <class TiledMma, class ElementA, class ElementB>
CUTLASS_HOST_DEVICE
static constexpr bool
is_sm100_mma_f8f6f4() {
  return (cute::size<2>(typename TiledMma::Shape_MNK{}) == 32) && is_sm10x_f8f6f4_inputs<ElementA, ElementB>();
}

template <class Element>
CUTLASS_HOST_DEVICE
static constexpr bool
is_sm10x_f8f6f4_element() {
  return (cute::is_same_v<Element, cute::float_e4m3_t> 
          || cute::is_same_v<Element, cute::float_e5m2_t> 
          || cute::is_same_v<Element, cute::float_e3m2_t>
          || cute::is_same_v<Element, cute::float_e2m3_t>
          || cute::is_same_v<Element, cute::float_e2m1_t>
          
        );
}


template <class Element>
CUTLASS_HOST_DEVICE
static constexpr bool
is_sm10x_f4_element() {
  return (cute::is_same_v<Element, cute::float_e2m1_t> 
  );
}

template <class ElementType>
CUTLASS_HOST_DEVICE
static constexpr bool
is_sm10x_mxf8f6f4_input() {
          // ElementType must be F8, F6, or F4
   return ( cute::is_same_v<ElementType, cutlass::type_erased_dynamic_float8_t> ||
            cute::is_same_v<ElementType, cutlass::detail::type_erased_dynamic_float6_unpacksmem_t> ||
            cute::is_same_v<ElementType, cutlass::detail::type_erased_dynamic_float4_unpacksmem_t> ||
            cute::is_same_v<ElementType, cutlass::float_e4m3_t> ||
            cute::is_same_v<ElementType, cutlass::float_e5m2_t> ||
            cute::is_same_v<ElementType, cutlass::detail::float_e2m3_unpacksmem_t> ||
            cute::is_same_v<ElementType, cutlass::detail::float_e3m2_unpacksmem_t> ||
            cute::is_same_v<ElementType, cutlass::detail::float_e2m1_unpacksmem_t>);
}

template <class ElementType>
CUTLASS_HOST_DEVICE
static constexpr bool
is_sm10x_mxf4nvf4_input() {
          // ElementType must be F4
   return ( cute::is_same_v<ElementType, cute::type_erased_dynamic_float4_t> ||
            cute::is_same_v<ElementType, cute::float_e2m1_t> 
          );
}

template <class ElementType, bool IsRuntimeDataType>
struct sm10x_block_scale_runtime_input_t {
  static constexpr bool IsF8F6F4MmaInput = is_sm10x_mxf8f6f4_input<ElementType>();
  static constexpr bool IsF4MmaInput = is_sm10x_mxf4nvf4_input<ElementType>();

  using Type = cute::conditional_t<IsRuntimeDataType && IsF8F6F4MmaInput, 
                                   cute::UMMA::MXF8F6F4Format, 
               cute::conditional_t<IsRuntimeDataType && IsF4MmaInput, 
                                   cute::UMMA::MXF4Format, 
                                   void*
                                   >
                                  >;
};


template <class TiledMma, class ElementA, class ElementB>
CUTLASS_HOST_DEVICE
static constexpr bool
is_sm120_f8f6f4() {
  return (cute::size<2>(typename TiledMma::Shape_MNK{}) == 32) && is_sm10x_f8f6f4_inputs<ElementA, ElementB>();
}

template <class TiledMma, class ElementA, class ElementB>
CUTLASS_HOST_DEVICE
static constexpr bool
is_sm100_sparse_f8f6f4() {
  return (cute::size<2>(typename TiledMma::Shape_MNK{}) == 64) && is_sm10x_f8f6f4_inputs<ElementA, ElementB>();
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective
