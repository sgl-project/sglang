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

#include <cute/config.hpp>
#include <cute/numeric/integral_constant.hpp>

#include <cuda_runtime.h>

namespace cutlass::fmha {

struct Pow2 {                                                                   
  int n;                                                                        
  int log2_n;                                                                   
                                                                                
  explicit CUTE_DEVICE Pow2(int n) : n(n) {
#ifdef __CUDA_ARCH__
    log2_n = __ffs(n) - 1;
#endif
  }                    
                                                                                
  template<class T>  
  CUTE_HOST_DEVICE T operator *(T const& b) const {
    return n * b;
  }

  template<int N>
  CUTE_HOST_DEVICE auto operator *(Int<N> const&) const {
    if constexpr (N & (N - 1) == 0) {
      return Pow2{n * N};
    }
    return n * N;
  }

};                                                                              

template<class T>
CUTE_HOST_DEVICE auto operator/(T const& a, Pow2 const& b) {
  return a >> b.log2_n;
}

template<class T>
CUTE_HOST_DEVICE auto operator%(T const& a, Pow2 const& b) {
  return a & (b.n - 1);
}

template<class T>
CUTE_HOST_DEVICE bool operator<(T const& a, Pow2 const& b) {
  return a < b.n;
}

CUTE_HOST_DEVICE void print(Pow2 const& a) {
  printf("2^%d", a.log2_n);
}

} // end namespace cutlass::fmha

namespace cute {

template <>
struct is_integral<cutlass::fmha::Pow2> : true_type {};

} // end namespace cute
