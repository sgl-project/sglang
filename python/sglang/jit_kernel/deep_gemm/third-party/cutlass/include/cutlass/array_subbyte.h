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

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/platform/platform.h"

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Statically sized array for any data type
template <
  typename T,
  int N
>
struct Array<T, N, false> {
  static constexpr int kSizeBits = sizeof_bits<T>::value * N;

  /// Storage type
  using Storage = typename platform::conditional<
    ((kSizeBits % 32) != 0),
    typename platform::conditional<
      ((kSizeBits % 16) != 0),
      uint8_t,
      uint16_t
    >::type,
    uint32_t
  >::type;

  /// Element type
  using Element = T;

  /// Number of logical elements per stored object
  static constexpr int kElementsPerStoredItem = int(sizeof(Storage) * 8) / sizeof_bits<T>::value;

  /// Number of storage elements
  static constexpr size_t kStorageElements = (N + kElementsPerStoredItem - 1) / kElementsPerStoredItem;

  /// Number of logical elements
  static constexpr size_t kElements = N;

  /// Bitmask for covering one item
  static constexpr Storage kMask = ((Storage(1) << sizeof_bits<T>::value) - 1);

  //
  // C++ standard members with pointer types removed
  //

  typedef T value_type;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef value_type *pointer;
  typedef value_type const *const_pointer;

  //
  // References
  //

  /// Reference object inserts or extracts sub-byte items
  class reference {
    /// Pointer to storage element
    Storage *ptr_{nullptr};

    /// Index into elements packed into Storage object
    int idx_{0};

  public:

    reference() = default;

    /// Ctor
    CUTLASS_HOST_DEVICE
    reference(Storage *ptr, int idx = 0): ptr_(ptr), idx_(idx) { }

    /// Assignment
    CUTLASS_HOST_DEVICE
    reference &operator=(T x) {
    // `*ptr_ & kUpdateMask` will read ptr_ before write to it
    // This means code pattern like
    //
    // ```cpp
    // Array<half_t, N> result;
    // result[0] = xxx;
    // ```
    // 
    // Will leads to compiler warning on use of uninitialized member variable. Although we know
    //      this read of uninitialized member variable is harmeless.

#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wuninitialized"
#elif defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wuninitialized"
#  pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

      Storage item = (reinterpret_cast<Storage const &>(x) & kMask);

      Storage kUpdateMask = Storage(~(kMask << (idx_ * sizeof_bits<T>::value)));

      *ptr_ = Storage(((*ptr_ & kUpdateMask) | (item << idx_ * sizeof_bits<T>::value)));

#if defined(__clang__)
#  pragma clang diagnostic pop
#elif defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

      return *this;
    }

    CUTLASS_HOST_DEVICE
    T get() const {
      Storage item = Storage((*ptr_ >> (idx_ * sizeof_bits<T>::value)) & kMask);
      return reinterpret_cast<T const &>(item);
    }

    /// Extract
    CUTLASS_HOST_DEVICE
    operator T() const {
      return get();
    }

    /// Explicit cast to int
    CUTLASS_HOST_DEVICE
    explicit operator int() const {
      return int(get());
    }

    /// Explicit cast to float
    CUTLASS_HOST_DEVICE
    explicit operator float() const {
      return float(get());
    }
  };

  /// Reference object extracts sub-byte items
  class const_reference {

    /// Pointer to storage element
    Storage const *ptr_{nullptr};

    /// Index into elements packed into Storage object
    int idx_{0};

  public:

    const_reference() = default;

    /// Ctor
    CUTLASS_HOST_DEVICE
    const_reference(Storage const *ptr, int idx = 0): ptr_(ptr), idx_(idx) { }

    CUTLASS_HOST_DEVICE
    const T get() const {
      Storage item = (*ptr_ >> (idx_ * sizeof_bits<T>::value)) & kMask;
      return reinterpret_cast<T const &>(item);
    }

    /// Extract
    CUTLASS_HOST_DEVICE
    operator T() const {
      Storage item = Storage(Storage(*ptr_ >> Storage(idx_ * sizeof_bits<T>::value)) & kMask);
      return reinterpret_cast<T const &>(item);
    }

    /// Explicit cast to int
    CUTLASS_HOST_DEVICE
    explicit operator int() const {
      return int(get());
    }

    /// Explicit cast to float
    CUTLASS_HOST_DEVICE
    explicit operator float() const {
      return float(get());
    }
  };

  //
  // Iterators
  //

  /// Bidirectional iterator over elements
  class iterator {

    /// Pointer to storage element
    Storage *ptr_{nullptr};

    /// Index into elements packed into Storage object
    int idx_{0};

  public:

    iterator() = default;

    CUTLASS_HOST_DEVICE
    iterator(Storage *ptr, int idx = 0): ptr_(ptr), idx_(idx) { }

    CUTLASS_HOST_DEVICE
    iterator &operator++() {
      ++idx_;
      if (idx_ == kElementsPerStoredItem) {
        ++ptr_;
        idx_ = 0;
      }
      return *this;
    }

    CUTLASS_HOST_DEVICE
    iterator &operator--() {
      if (!idx_) {
        --ptr_;
        idx_ = kElementsPerStoredItem - 1;
      }
      else {
        --idx_;
      }
      return *this;
    }

    CUTLASS_HOST_DEVICE
    iterator operator++(int) {
      iterator ret(*this);
      ++idx_;
      if (idx_ == kElementsPerStoredItem) {
        ++ptr_;
        idx_ = 0;
      }
      return ret;
    }

    CUTLASS_HOST_DEVICE
    iterator operator--(int) {
      iterator ret(*this);
      if (!idx_) {
        --ptr_;
        idx_ = kElementsPerStoredItem - 1;
      }
      else {
        --idx_;
      }
      return ret;
    }

    CUTLASS_HOST_DEVICE
    reference operator*() const {
      return reference(ptr_, idx_);
    }

    CUTLASS_HOST_DEVICE
    bool operator==(iterator const &other) const {
      return ptr_ == other.ptr_ && idx_ == other.idx_;
    }

    CUTLASS_HOST_DEVICE
    bool operator!=(iterator const &other) const {
      return !(*this == other);
    }
  };

  /// Bidirectional constant iterator over elements
  class const_iterator {

    /// Pointer to storage element
    Storage const *ptr_{nullptr};

    /// Index into elements packed into Storage object
    int idx_{0};

  public:

    const_iterator() = default;

    CUTLASS_HOST_DEVICE
    const_iterator(Storage const *ptr, int idx = 0): ptr_(ptr), idx_(idx) { }

    CUTLASS_HOST_DEVICE
    iterator &operator++() {
      ++idx_;
      if (idx_ == kElementsPerStoredItem) {
        ++ptr_;
        idx_ = 0;
      }
      return *this;
    }

    CUTLASS_HOST_DEVICE
    iterator &operator--() {
      if (!idx_) {
        --ptr_;
        idx_ = kElementsPerStoredItem - 1;
      }
      else {
        --idx_;
      }
      return *this;
    }

    CUTLASS_HOST_DEVICE
    iterator operator++(int) {
      iterator ret(*this);
      ++idx_;
      if (idx_ == kElementsPerStoredItem) {
        ++ptr_;
        idx_ = 0;
      }
      return ret;
    }

    CUTLASS_HOST_DEVICE
    iterator operator--(int) {
      iterator ret(*this);
      if (!idx_) {
        --ptr_;
        idx_ = kElementsPerStoredItem - 1;
      }
      else {
        --idx_;
      }
      return ret;
    }

    CUTLASS_HOST_DEVICE
    const_reference operator*() const {
      return const_reference(ptr_, idx_);
    }

    CUTLASS_HOST_DEVICE
    bool operator==(iterator const &other) const {
      return ptr_ == other.ptr_ && idx_ == other.idx_;
    }

    CUTLASS_HOST_DEVICE
    bool operator!=(iterator const &other) const {
      return !(*this == other);
    }
  };

  /// Bidirectional iterator over elements
  class reverse_iterator {

    /// Pointer to storage element
    Storage *ptr_{nullptr};

    /// Index into elements packed into Storage object
    int idx_{0};

  public:

    reverse_iterator() = default;

    CUTLASS_HOST_DEVICE
    reverse_iterator(Storage *ptr, int idx = 0): ptr_(ptr), idx_(idx) { }
  };

  /// Bidirectional constant iterator over elements
  class const_reverse_iterator {

    /// Pointer to storage element
    Storage const *ptr_{nullptr};

    /// Index into elements packed into Storage object
    int idx_{0};

  public:

    const_reverse_iterator() = default;

    CUTLASS_HOST_DEVICE
    const_reverse_iterator(Storage const *ptr, int idx = 0): ptr_(ptr), idx_(idx) { }
  };

  /// Efficient clear method
  CUTLASS_HOST_DEVICE
  void clear() {

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(kStorageElements); ++i) {
      storage[i] = Storage(0);
    }
  }

  CUTLASS_HOST_DEVICE
  reference at(size_type pos) {
    return reference(storage + pos / kElementsPerStoredItem, pos % kElementsPerStoredItem);
  }

  CUTLASS_HOST_DEVICE
  const_reference at(size_type pos) const {
    return const_reference(storage + pos / kElementsPerStoredItem, pos % kElementsPerStoredItem);
  }

  CUTLASS_HOST_DEVICE
  reference operator[](size_type pos) {
    return at(pos);
  }

  CUTLASS_HOST_DEVICE
  const_reference operator[](size_type pos) const {
    return at(pos);
  }

  CUTLASS_HOST_DEVICE
  reference front() {
    return at(0);
  }

  CUTLASS_HOST_DEVICE
  const_reference front() const {
    return at(0);
  }

  CUTLASS_HOST_DEVICE
  reference back() {
    return reference(storage + kStorageElements - 1, kElementsPerStoredItem - 1);
  }

  CUTLASS_HOST_DEVICE
  const_reference back() const {
    return const_reference(storage + kStorageElements - 1, kElementsPerStoredItem - 1);
  }

  CUTLASS_HOST_DEVICE
  pointer data() {
    return reinterpret_cast<pointer>(storage);
  }

  CUTLASS_HOST_DEVICE
  const_pointer data() const {
    return reinterpret_cast<const_pointer>(storage);
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
    return !kElements;
  }

  CUTLASS_HOST_DEVICE
  constexpr size_type size() const {
    return kElements;
  }

  CUTLASS_HOST_DEVICE
  constexpr size_type max_size() const {
    return kElements;
  }

  CUTLASS_HOST_DEVICE
  void fill(T const &value) {

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerStoredItem; ++i) {
      reference ref(storage, i);
      ref = value;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < kStorageElements; ++i) {
      storage[i] = storage[0];
    }
  }

  CUTLASS_HOST_DEVICE
  iterator begin() {
    return iterator(storage);
  }

  CUTLASS_HOST_DEVICE
  const_iterator cbegin() const {
    return const_iterator(storage);
  }

  CUTLASS_HOST_DEVICE
  iterator end() {
    return iterator(storage + kStorageElements);
  }

  CUTLASS_HOST_DEVICE
  const_iterator cend() const {
    return const_iterator(storage + kStorageElements);
  }

  CUTLASS_HOST_DEVICE
  reverse_iterator rbegin() {
    return reverse_iterator(storage + kStorageElements);
  }

  CUTLASS_HOST_DEVICE
  const_reverse_iterator crbegin() const {
    return const_reverse_iterator(storage + kStorageElements);
  }

  CUTLASS_HOST_DEVICE
  reverse_iterator rend() {
    return reverse_iterator(storage);
  }

  CUTLASS_HOST_DEVICE
  const_reverse_iterator crend() const {
    return const_reverse_iterator(storage);
  }

private:
  /// Internal storage
  Storage storage[kStorageElements];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////
