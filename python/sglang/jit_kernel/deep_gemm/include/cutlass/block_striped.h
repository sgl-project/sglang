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
    \brief Utilities for performing block-striped access (load, store, reduce) of trivially-copyable,
    statically-sized array types to global memory.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/wmma_array.h"
#include "cutlass/functional.h"
#include "cutlass/complex.h"

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////
// AccessWidth
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes the maximal power-of-two that evenly divides the size of T, capped at Limit
template <
  typename T,
  int Limit>
struct AccessWidth
{
  // Inductive case
  template <
      int ObjectBytes,        /// Size of T in bytes
      int AlignBytes,         /// Template induction variable
      bool IsAligned  =       /// Whether ObjectBytes is an even multiple of AlignBytes
        ((AlignBytes <= Limit) &&  (ObjectBytes % AlignBytes == 0))>
  struct Detail
  {
      static const int value = Detail<ObjectBytes, AlignBytes * 2>::value;
  };

  // Base case (ObjectBytes is not an even multiple of AlignBytes)
  template <
      int ObjectBytes,        /// Size of T in bytes
      int AlignBytes>         /// Template induction variable
  struct Detail<ObjectBytes, AlignBytes, false>
  {
      static const int value = AlignBytes / 2;
  };

  /// The maximal power-of-two that evenly divides the size of T
  static const int value = Detail<
    (int) sizeof(T),
    1>::value;
};



/////////////////////////////////////////////////////////////////////////////////////////////////
// StripedAccessType
/////////////////////////////////////////////////////////////////////////////////////////////////

/// ReinterpretCast type for striping a trivially-copyable type in global memory
/// (Default specialization.  Striping granularity is type T.)
template <
    typename T,           /// Data type
    int TransferBytes =   /// Data access width (16 byte max for global memory access on current architectures)
      AccessWidth<T, 16>::value>
struct alignas(TransferBytes) StripedAccessType : public T
{};


/// ReinterpretCast type for striping a trivially-copyable type in global memory
/// (Specialization for cutlass::Array<T>.  Striping granularity is a multiple of T.)
template <
    typename T,           /// Array element type
    int N,                /// Number of elements in array
    bool RegisterSized,   /// T is register-sized
    int TransferBytes>    /// Data access width
struct StripedAccessType<
    Array<T, N, RegisterSized>,
    TransferBytes>
: public AlignedArray<
            T,                                                  // Element type of StripedAccessType
            __NV_STD_MAX(1, TransferBytes / (int) sizeof(T)),   // Number of elements T in StripedAccessType
            TransferBytes>                                      // Alignment of StripedAccessType
{};


#if defined(CUTLASS_ARCH_WMMA_ENABLED)

/// ReinterpretCast type for striping a trivially-copyable type in global memory
/// (Specialization for cutlass::WmmaFragmentArray<T>.  Striping granularity is a multiple of T.)
template<
    typename Use,
    int m,
    int n,
    int k,
    typename ElementT,
    typename Layout,
    int kFragments,
    int TransferBytes>
struct StripedAccessType<
    WmmaFragmentArray<nvcuda::wmma::fragment<Use, m, n, k, ElementT, Layout>, kFragments>,
    TransferBytes>
: public AlignedArray<
            ElementT,
            __NV_STD_MAX(1, TransferBytes / (int) sizeof(ElementT)),
            TransferBytes>
{};

#endif // if defined(CUTLASS_ARCH_WMMA_ENABLED)


/////////////////////////////////////////////////////////////////////////////////////////////////
// BlockStriped
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Utility for performing block-striped access (load, store) of trivially-copyable,
/// statically-sized array types to global memory
template <
  int BlockThreads,
  typename ArrayT,
  typename AccessT = StripedAccessType<ArrayT> >
struct BlockStriped
{
  /// Number of striped accesses
  static const int kStripes = int(sizeof(ArrayT) / sizeof(AccessT));
  static_assert(kStripes > 0, "AccessT type must be smaller than or equal to ArrayT type");

  /// Load
  CUTLASS_DEVICE
  static void load(ArrayT &data, ArrayT *ptr, int thread_idx)
  {
    AccessT *access_input = reinterpret_cast<AccessT*>(ptr);
    AccessT *access_data = reinterpret_cast<AccessT*>(&data);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kStripes; ++i) {
      access_data[i] = access_input[(BlockThreads * i) + thread_idx];
    }
  }

  /// Load & Add
  CUTLASS_DEVICE
  static void load_add(ArrayT &data, ArrayT *ptr, int thread_idx)
  {
    AccessT *access_input = reinterpret_cast<AccessT*>(ptr);
    AccessT *access_data = reinterpret_cast<AccessT*>(&data);

    plus<AccessT> add;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kStripes; ++i)
    {
      access_data[i] = add(access_data[i], access_input[(BlockThreads * i) + thread_idx]);
    }
  }

  /// Store
  CUTLASS_DEVICE
  static void store(ArrayT *ptr, const ArrayT &data, int thread_idx)
  {
    AccessT *access_output = reinterpret_cast<AccessT*>(ptr);
    const AccessT *access_data = reinterpret_cast<const AccessT*>(&data);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kStripes; ++i) {
      access_output[(BlockThreads * i) + thread_idx] = access_data[i];
    }
  }

};


/////////////////////////////////////////////////////////////////////////////////////////////////
// BlockStripedReduce
/////////////////////////////////////////////////////////////////////////////////////////////////


/// Utility for performing block-striped access (load, store, reduce) of trivially-copyable,
/// statically-sized array types to global memory.
/// (Default specialization)
template <
  int BlockThreads,
  typename ArrayT,
  typename ElementT = typename StripedAccessType<ArrayT>::Element>
struct BlockStripedReduce :
  BlockStriped<
    BlockThreads,
    ArrayT,
    ElementT>
{
  /// Reduce
  CUTLASS_DEVICE
  static void reduce(ArrayT *ptr, const ArrayT &data, int thread_idx)
  {
    cutlass::atomic_add<ElementT> reduce;
    ElementT *access_output = reinterpret_cast<ElementT*>(ptr);
    const ElementT *access_data = reinterpret_cast<const ElementT*>(&data);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < BlockStripedReduce::kStripes; ++i) {
      reduce(access_output + (BlockThreads * i) + thread_idx, access_data[i]);
    }
  }
};


/// Utility for performing block-striped access (load, store, reduce) of trivially-copyable,
/// statically-sized array types to global memory.
/// (Specialization for half_t.  Uses half2 vectorized-reduction.)
template <
  int BlockThreads,
  typename ArrayT>
struct BlockStripedReduce<BlockThreads, ArrayT, half_t> :
  BlockStriped<
    BlockThreads,
    ArrayT,
    half2>
{
  static_assert(BlockStripedReduce::kStripes % 2 == 0, "Array of half must be even number in length");

  /// Reduce
  CUTLASS_DEVICE
  static void reduce(ArrayT *ptr, const ArrayT &data, int thread_idx)
  {
    cutlass::atomic_add<half2> reduce;
    half2 *access_output = reinterpret_cast<half2*>(ptr);
    const half2 *access_data = reinterpret_cast<const half2*>(&data);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < BlockStripedReduce::kStripes; ++i)
    {
      reduce(access_output + (BlockThreads * i) + thread_idx, access_data[i]);
    }
  }
};


} // namespace cutlass

