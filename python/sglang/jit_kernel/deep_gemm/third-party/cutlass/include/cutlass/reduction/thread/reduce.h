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
    \brief Defines basic thread level reduction with specializations for Array<T, N>.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/half.h"
#include "cutlass/functional.h"

namespace cutlass {
namespace reduction {
namespace thread {

/// Structure to compute the thread level reduction
template <typename Op, typename T>
struct Reduce;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial Specialization of Reduce for "plus" (a functional operator)
template <typename T>
struct Reduce< plus<T>, T > {

  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    plus<T> _op;
    return _op(lhs, rhs);
  } 
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization of Reduce for Array<T, N>
template <typename T, int N>
struct Reduce < plus<T>, Array<T, N>> {
  
  CUTLASS_HOST_DEVICE
  Array<T, 1> operator()(Array<T, N> const &in) const {

    Array<T, 1> result;
    Reduce< plus<T>, T > scalar_reduce;
    result.clear();

    CUTLASS_PRAGMA_UNROLL
    for (auto i = 0; i < N; ++i) {
      result[0] = scalar_reduce(result[0], in[i]);
    }

    return result;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specializations of Reduce for Array<half_t, N>
template <int N>
struct Reduce < plus<half_t>, Array<half_t, N> > {
  
  CUTLASS_HOST_DEVICE
  Array<half_t, 1> operator()(Array<half_t, N> const &input) {

    Array<half_t, 1> result;

    // If there is only 1 element - there is nothing to reduce
    if( N ==1 ){

      result[0] = input.front();

    } else {
    
      #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
        
        __half result_d;
        Array<half_t, 1> const *in_ptr_half = reinterpret_cast<Array<half_t, 1> const *>(&input);
        Array<half_t, 2> const *in_ptr_half2 = reinterpret_cast<Array<half_t, 2> const *>(&input);
        __half2 const *x_in_half2 = reinterpret_cast<__half2 const *>(in_ptr_half2);

        // Set initial result = first half2, in case N==2
        __half2 tmp_result = x_in_half2[0];

        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < N/2; ++i) {

          tmp_result = __hadd2(x_in_half2[i], tmp_result);

        }
        
        result_d = __hadd(__low2half(tmp_result), __high2half(tmp_result));
    
        // One final step is needed for odd "N" (to add the (N-1)th element)
        if( N%2 ){

          __half last_element;
          Array<half_t, 1> tmp_last;
          Array<half_t, 1> *tmp_last_ptr = &tmp_last;
          tmp_last_ptr[0] = in_ptr_half[N-1];
          last_element = reinterpret_cast<__half  const &>(tmp_last);

          result_d = __hadd(result_d, last_element);

        } 

        Array<half_t, 1> *result_ptr = &result;
        *result_ptr = reinterpret_cast<Array<half_t, 1> &>(result_d);

      #else
        
        Reduce< plus<half_t>, half_t > scalar_reduce;
        result.clear();

        CUTLASS_PRAGMA_UNROLL
        for (auto i = 0; i < N; ++i) {

          result[0] = scalar_reduce(result[0], input[i]);

        }

      #endif
    }

    return result;
      
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specializations of Reduce for AlignedArray<half_t, N>
template <int N>
struct Reduce < plus<half_t>, AlignedArray<half_t, N> > {
  
  CUTLASS_HOST_DEVICE
  Array<half_t, 1> operator()(AlignedArray<half_t, N> const &input) {

    Array<half_t, 1> result;

    // If there is only 1 element - there is nothing to reduce
    if( N ==1 ){

      result[0] = input.front();

    } else {
    
      #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
        
        __half result_d;
        AlignedArray<half_t, 1> const *in_ptr_half = reinterpret_cast<AlignedArray<half_t, 1> const *>(&input);
        AlignedArray<half_t, 2> const *in_ptr_half2 = reinterpret_cast<AlignedArray<half_t, 2> const *>(&input);
        __half2 const *x_in_half2 = reinterpret_cast<__half2 const *>(in_ptr_half2);

        // Set initial result = first half2, in case N==2
        __half2 tmp_result = x_in_half2[0];

        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < N/2; ++i) {

          tmp_result = __hadd2(x_in_half2[i], tmp_result);

        }
        
        result_d = __hadd(__low2half(tmp_result), __high2half(tmp_result));
    
        // One final step is needed for odd "N" (to add the (N-1)th element)
        if( N%2 ){

          __half last_element;
          AlignedArray<half_t, 1> tmp_last;
          AlignedArray<half_t, 1> *tmp_last_ptr = &tmp_last;
          tmp_last_ptr[0] = in_ptr_half[N-1];
          last_element = reinterpret_cast<__half  const &>(tmp_last);

          result_d = __hadd(result_d, last_element);

        } 

        Array<half_t, 1> *result_ptr = &result;
        *result_ptr = reinterpret_cast<Array<half_t, 1> &>(result_d);

      #else
        
        Reduce< plus<half_t>, half_t > scalar_reduce;
        result.clear();

        CUTLASS_PRAGMA_UNROLL
        for (auto i = 0; i < N; ++i) {

          result[0] = scalar_reduce(result[0], input[i]);

        }

      #endif
    }

    return result;
      
  }
};
}
}
}
