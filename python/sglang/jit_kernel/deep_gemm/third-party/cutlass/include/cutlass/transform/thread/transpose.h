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
    \brief Basic copy routines for tensor views
*/

#pragma once

namespace cutlass {
namespace transform {
namespace thread {

/// Transforms a fragment by doing a transpose
template <
  int ElementCount, 
  typename TransposeShape, 
  typename Element
> struct Transpose;

/// Specialization for int8_t 4x4 transpose
template <int ElementCount_>
struct Transpose<ElementCount_, layout::PitchLinearShape<4,4> , int8_t> {

    static const int kElementCount = ElementCount_;
    using TransposeShape = layout::PitchLinearShape<4,4>;
    using Element = int8_t;
    using Fragment = cutlass::Array<Element, kElementCount>;

    static_assert(!(kElementCount % TransposeShape::kCount), "Shape needs to be multiple of 16 elements to do a 4x4 transpose");

    CUTLASS_DEVICE 
    void transform(Fragment& dst, Fragment& src) {

    // Expose src/dst as int arrays.
    int* src_int = reinterpret_cast<int*>(&src);
    int* dst_int = reinterpret_cast<int*>(&dst);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementCount / TransposeShape::kCount; i++){
  
      int const i0 = 4 * i + 0;
      int const i1 = 4 * i + 1;
      int const i2 = 4 * i + 2;
      int const i3 = 4 * i + 3;

      int a0 = src_int[i0];
      int a1 = src_int[i1];
      int a2 = src_int[i2];
      int a3 = src_int[i3];

      int b0, b1, b2, b3, c0;
      b0 = __byte_perm(a0, a1, 0x0040);
      c0 = __byte_perm(a2, a3, 0x0040);
      b0 = __byte_perm(b0, c0, 0x5410);

      b1 = __byte_perm(a0, a1, 0x0051);
      c0 = __byte_perm(a2, a3, 0x0051);
      b1 = __byte_perm(b1, c0, 0x5410);

      b2 = __byte_perm(a0, a1, 0x0062);
      c0 = __byte_perm(a2, a3, 0x0062);
      b2 = __byte_perm(b2, c0, 0x5410);

      b3 = __byte_perm(a0, a1, 0x0073);
      c0 = __byte_perm(a2, a3, 0x0073);
      b3 = __byte_perm(b3, c0, 0x5410);

      dst_int[i0] = b0;
      dst_int[i1] = b1;
      dst_int[i2] = b2;
      dst_int[i3] = b3;
    }
  }
};

}  // namespace thread
}  // namespace layout
}  // namespace cutlass
