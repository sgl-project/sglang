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
    \brief Template wraps the vector access iterator concept to load whole vector from tensors in
      memory. This is typically used for per-channel scale and bias in convolution kernels.
*/

#pragma once

#include "cutlass/transform/threadblock/predicated_vector_access_iterator.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename VectorAccessIterator_>
class VectorIterator {
public:
  using VectorAccessIterator = VectorAccessIterator_;

  using Shape = typename VectorAccessIterator::Shape;
  using Element = typename VectorAccessIterator::Element;
  using Layout = typename VectorAccessIterator::Layout;
  using TensorCoord = typename Layout::TensorCoord;
  using AccessType = typename VectorAccessIterator::AccessType;
  using TensorRef = typename VectorAccessIterator::TensorRef;
  using Index = typename VectorAccessIterator::Index;
  using LongIndex = typename VectorAccessIterator::LongIndex;

  static int const kElementsPerAccess = VectorAccessIterator::kElementsPerAccess;
  static int const kRowsPerIteration = VectorAccessIterator::kRowsPerIteration;
  static int const kThreads = VectorAccessIterator::kThreads;
  static int const kIterations = VectorAccessIterator::kIterations;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<
    Element, kElementsPerAccess * kIterations>;

private:

  /// Internal state
  VectorAccessIterator vector_access_iterator_;

public:

  /// Constructor
  CUTLASS_HOST_DEVICE
  VectorIterator(
    Element const *ptr,
    TensorCoord extent,
    int thread_idx,
    int warp_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()
  ):
    vector_access_iterator_(ptr, extent, thread_idx, warp_idx, threadblock_offset) { }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  VectorIterator &operator++() {
    vector_access_iterator_.advance();
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  VectorIterator operator++(int) {
    VectorIterator self(*this);
    operator++();
    return self;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {

    frag.clear();
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < kIterations; ++c) {

        cutlass::arch::global_load<
          AccessType,
          sizeof(AccessType)
        >(
          frag_ptr[c],
          vector_access_iterator_.get() + pointer_offset,
          vector_access_iterator_.valid()
        );

        ++vector_access_iterator_;
      }
//    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    vector_access_iterator_.set_iteration_index(0);
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_DEVICE
  void advance() {
    vector_access_iterator_.advance();
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace transform
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

