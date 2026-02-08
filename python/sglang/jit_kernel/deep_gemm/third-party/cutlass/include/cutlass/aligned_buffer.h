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
    \brief AlignedBuffer is a container for trivially copyable elements suitable for use in
      unions and shared memory.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Modifies semantics of cutlass::Array<> to provide guaranteed alignment. 
template <
  typename T,
  int N,
  int Align = 16
>
struct AlignedBuffer {
  
  /// Internal storage type
  using Storage = uint8_t;

  /// Number of logical elements held in buffer
  static int const kCount = N;

  /// Alignment requirement in bytes
  static int const kAlign = Align;

  /// Number of storage elements
  static int const kBytes = 
    (sizeof_bits<T>::value * N + 7) / 8;

private:

  /// Internal storage
  alignas(Align) Storage storage[kBytes];

public:

  //
  // C++ standard members
  //

  typedef T value_type;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef value_type *pointer;
  typedef value_type const * const_pointer;

  using Array = Array<T, N>;
  using reference = typename Array::reference;
  using const_reference = typename Array::const_reference;

public:

  CUTLASS_HOST_DEVICE
  pointer data() {
    return reinterpret_cast<pointer>(storage); 
  }

  CUTLASS_HOST_DEVICE
  const_pointer data() const {
    return reinterpret_cast<pointer>(storage); 
  }
  
  CUTLASS_HOST_DEVICE
  Storage * raw_data() {
    return storage;
  }

  CUTLASS_HOST_DEVICE
  Storage const * raw_data() const {
    return storage;
  }


  CUTLASS_HOST_DEVICE
  constexpr bool empty() const {
    return !kCount;
  }

  CUTLASS_HOST_DEVICE
  constexpr size_type size() const {
    return kCount;
  }

  CUTLASS_HOST_DEVICE
  constexpr size_type max_size() const {
    return kCount;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

