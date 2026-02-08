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
    \brief Statically sized array of elements that accommodates all CUTLASS-supported numeric types
           and is safe to use in a union.
*/

#pragma once

#include "cutlass/arch/wmma.h"

#if defined(CUTLASS_ARCH_WMMA_ENABLED)

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Wmma array type (WmmaFragmentArray holds elements of type nvcuda::wmma::fragment)
template <
  /// Element type
  typename T,
  /// Number of elements in the array
  int N,
  /// Whether the element type of T is half_t or __half
  bool IsHalfType = (platform::is_same<typename T::element_type, cutlass::half_t>::value ||
                     platform::is_same<typename T::element_type, __half>::value)
>
class WmmaFragmentArray: public Array<T, N, true> {
public:

  /// Efficient clear method (override Array::clear())
  CUTLASS_HOST_DEVICE
  void clear()
  {
    for(int i = 0; i < Array<T, N, true>::kElements; i++)
    {
      nvcuda::wmma::fill_fragment((*this)[i], (typename T::element_type)0);
    }
  }

  CUTLASS_HOST_DEVICE
  WmmaFragmentArray<T, N>& operator+=(const WmmaFragmentArray<T, N>& rhs)
  {
    using element_type = typename T::element_type;
    plus<T> add;

    for (int i = 0; i < Array<T, N, true>::kElements; i++)
    {
      (*this)[i] = add((*this)[i], rhs[i]);
    }

    return *this;
  }
};

/// Partial specialization for the case in which T::element_type is
/// half_t or __half. This is needed because the cast (typename T::element_type)0
/// in the primary template flags as an error when __CUDA_NO_HALF_CONVERSIONS__
/// is set.
template <
  /// Element type
  typename T,
  /// Number of elements in the array
  int N
>
class WmmaFragmentArray<T, N, true>: public Array<T, N, true> {
public:

  /// Efficient clear method (override Array::clear())
  CUTLASS_HOST_DEVICE
  void clear()
  {
    for(int i = 0; i < Array<T, N, true>::kElements; i++)
    {
      nvcuda::wmma::fill_fragment((*this)[i], __float2half(0.f));
    }
  }

  CUTLASS_HOST_DEVICE
  WmmaFragmentArray<T, N>& operator+=(const WmmaFragmentArray<T, N>& rhs)
  {
    using element_type = typename T::element_type;
    plus<T> add;

    for (int i = 0; i < Array<T, N, true>::kElements; i++)
    {
      (*this)[i] = add((*this)[i], rhs[i]);
    }

    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

#endif // if defined(CUTLASS_ARCH_WMMA_ENABLED)

