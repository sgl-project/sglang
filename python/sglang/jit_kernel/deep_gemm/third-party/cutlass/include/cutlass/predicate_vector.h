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
    \brief Defines container classes and iterators for managing a statically sized vector
      of boolean predicates.
*/
#pragma once
#include "cutlass/cutlass.h"
#if defined(__CUDACC_RTC__)
#include CUDA_STD_HEADER(cstdint)
#else
#include <cstdint>
#endif

#include CUDA_STD_HEADER(cassert)

#include "cutlass/platform/platform.h"

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/*!@defgroup predicate_vector_concept Predicate Vector Concept
@{

Implementations of \ref predicate_vector_concept contain an ordered set of boolean predicates which
may be used as conditionals in other device-side operations. Both random access and iterators
offering sequential access are provided.

@par Predicate Vector
   A \ref predicate_vector_concept satisfies the following expressions
  - <b>at(int idx)</b> - returns the value of the indexed predicate
  - <b>set(int idx, bool value)</b> - sets the value of the indexed predicate
  - <b>begin()</b> - returns a \ref predicate_iterator_concept pointing to the first predicate

@}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////

/*!@defgroup predicate_iterator_concept Predicate Iterator Concept
@{

Implementations of \ref predicate_iterator_concept enables accessing and traversing elements of a
bit vector.

@par Const Predicate Iterator
  A const \ref predicate_iterator_concept satisfies the following expressions
 - <b>++it</b> increments the iterator to the next predicate
 - <b>*it</b> returns the value of the currently pointed-to predicate

@par Mutable Predicate Iterator
 A \ref predicate_iterator_concept that is non-const <b>also</b> satisfies the following expressions
 - <b>it.set(bool value)</b> sets the value of the currently pointed-to predicate

@}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////

/*!@defgroup predicate_tile_adapter Predicate Tile Adapter Concept
@{

Implementations of \ref predicate_tile_adapter provide a mapping between a the elements of a \ref
tile_traits_concept and a \ref predicate_vector_concept.

@par Predicate Tile Adapter
  A \ref predicate_tile_adapter satisfies the following expressions
 - <b>at(int d, int h, int w, int c)</b> - returns the value of a predicate corresponding to the
   access (d, h, w, c) within the tile.

@}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Statically sized array of bits implementing @concept{predicate_vector_concept}.
template <
    /// Number of predicates contained in predicate vector
    int kPredicates_,
    /// Number of predicates contained in each byte of internal storage
    int kPredicatesPerByte_ = 4,
    /// Location of first predicate within byte of internal storage
    int kPredicateStart_ = 0>
struct PredicateVector {
  /// Number of bits stored by the PredicateVector
  static constexpr int kPredicates = kPredicates_;

  /// Number of bits stored within each byte of the predicate bit vector
  static constexpr int kPredicatesPerByte = kPredicatesPerByte_;

  /// First bit within each byte containing predicates
  static constexpr int kPredicateStart = kPredicateStart_;

  // Make sure no one tries to put more than 8 bits in a byte :)
  static_assert(kPredicatesPerByte <= 8, "kPredicatesPerByte must fit within an actual byte");
  // Make sure the "offsetted" bits fit in one byte.
  static_assert(kPredicateStart + kPredicatesPerByte <= 8,
                "The offsetted predicates must fit within an actual byte.");

  /// Storage type of individual elements
  typedef uint32_t Storage;

  /// Number of bytes needed
  static constexpr int kBytes = (kPredicates + kPredicatesPerByte - 1) / kPredicatesPerByte;

  /// Number of storage elements needed
  static constexpr int kWordCount = (kBytes + int(sizeof(Storage)) - 1) / int(sizeof(Storage));

  /// The byte mask corresponding to predicates
  static constexpr Storage kByteMask = (((1 << kPredicatesPerByte) - 1) << kPredicateStart);

 private:
  //
  // Data members
  //

  /// Words of bit vector
  Storage storageData[kWordCount];

  //
  // Methods
  //

  /// Computes the word and bit corresponding to a logical predicate index
  CUTLASS_HOST_DEVICE void computeStorageOffset(int &word, int &bit, int idx) const {
    CUTLASS_ASSERT(idx < kPredicates);

    int byte = (idx / kPredicatesPerByte);
    int bit_offset = (idx % kPredicatesPerByte);

    word = byte / sizeof(Storage);
    int byte_offset = (byte % sizeof(Storage));

    bit = byte_offset * 8 + bit_offset + kPredicateStart;
  }

  /// Returns word mask.
  CUTLASS_HOST_DEVICE static constexpr bool computeWordMask() {
    Storage mask(0);
    CUTLASS_PRAGMA_UNROLL
    for (size_t byte = 0; byte < sizeof(Storage); ++byte) {
      mask |= (kByteMask << (byte * 8));
    }
    return mask;
  }

  /// Returns mask of last word.
  CUTLASS_HOST_DEVICE static constexpr bool computeLastWordMask() {
    Storage mask(0);
    CUTLASS_PRAGMA_UNROLL
    for (int byte = 0; byte < kBytes % sizeof(Storage); ++byte) {
      mask |= (kByteMask << (byte * 8));
    }
    return mask;
  }

  /// Accesses a given word with optional assertions
  CUTLASS_HOST_DEVICE Storage &storage(int word) {
    CUTLASS_ASSERT(word < kWordCount);
    return storageData[word];
  }

  /// Accesses a given word with optional assertions
  CUTLASS_HOST_DEVICE Storage const &storage(int word) const {
    CUTLASS_ASSERT(word < kWordCount);
    return storageData[word];
  }

 public:
  //
  // Iterator
  //

  /**
  * @brief An iterator implementing \ref predicate_iterator_concept enabling sequential
  * read and write access to predicates.
  * @concept{predicate_iterator_concept}
  */
  class Iterator {
    /// Reference to PredicateVector instance
    PredicateVector &vec_;

    /// Index into PredicateVector
    int bit_;

   public:
    /// Copy constructor
    CUTLASS_HOST_DEVICE
    Iterator(Iterator const &it) : vec_(it.vec_), bit_(it.bit_) {}

    /// Constructs an iterator from a PredicateVector
    CUTLASS_HOST_DEVICE
    Iterator(PredicateVector &vec, int _start = 0) : vec_(vec), bit_(_start) {}

    /// Pre-increment
    CUTLASS_HOST_DEVICE
    Iterator &operator++() {
      ++bit_;
      return *this;
    }

    /// Increment
    CUTLASS_HOST_DEVICE
    Iterator &operator+=(int offset) {
      bit_ += offset;
      return *this;
    }

    /// Pre-decrement
    CUTLASS_HOST_DEVICE
    Iterator &operator--() {
      --bit_;
      return *this;
    }

    /// Decrement
    CUTLASS_HOST_DEVICE
    Iterator &operator-=(int offset) {
      bit_ -= offset;
      return *this;
    }

    /// Post-increment
    CUTLASS_HOST_DEVICE
    Iterator operator++(int) {
      Iterator ret(*this);
      ret.bit_++;
      return ret;
    }

    /// Post-decrement
    CUTLASS_HOST_DEVICE
    Iterator operator--(int) {
      Iterator ret(*this);
      ret.bit_--;
      return ret;
    }

    /// Iterator advances by some amount
    CUTLASS_HOST_DEVICE
    Iterator operator+(int offset) {
      Iterator ret(*this);
      ret.bit_ += offset;
      return ret;
    }

    /// Iterator recedes by some amount
    CUTLASS_HOST_DEVICE
    Iterator operator-(int offset) {
      ConstIterator ret(*this);
      ret.bit_ -= offset;
      return ret;
    }

    /// Returns true if iterators point to the same bit
    CUTLASS_HOST_DEVICE
    bool operator==(Iterator const &it) const { return bit_ == it.bit_; }

    /// Returns false if iterators point to the same bit
    CUTLASS_HOST_DEVICE
    bool operator!=(Iterator const &it) const { return bit_ != it.bit_; }

    /// Gets the bit at the pointed to location
    CUTLASS_HOST_DEVICE
    bool get() { return vec_.at(bit_); }

    /// Gets the bit at the pointed to location
    CUTLASS_HOST_DEVICE
    bool at() const { return vec_.at(bit_); }

    /// Dereferences iterator
    CUTLASS_HOST_DEVICE
    bool operator*() const { return at(); }

    /// Sets the bit at the pointed to location
    CUTLASS_HOST_DEVICE
    void set(bool value = true) { vec_.set(bit_, value); }
  };

  /**
  * @brief An iterator implementing \ref predicate_iterator_concept enabling sequential
  * read and write access to predicates.
  * @concept{predicate_iterator_concept}
  */
  class ConstIterator {
    /// Reference to PredicateVector instance
    PredicateVector const &vec_;

    /// Index into PredicateVector
    int bit_;

   public:
    /// Copy constructor
    CUTLASS_HOST_DEVICE
    ConstIterator(ConstIterator const &it) : vec_(it.vec_), bit_(it.bit_) {}

    /// Constructs an iterator from a PredicateVector
    CUTLASS_HOST_DEVICE
    ConstIterator(PredicateVector const &vec, int _start = 0) : vec_(vec), bit_(_start) {}

    /// Pre-increment
    CUTLASS_HOST_DEVICE
    ConstIterator &operator++() {
      ++bit_;
      return *this;
    }

    /// Increment
    CUTLASS_HOST_DEVICE
    ConstIterator &operator+=(int offset) {
      bit_ += offset;
      return *this;
    }

    /// Pre-decrement
    CUTLASS_HOST_DEVICE
    ConstIterator &operator--() {
      --bit_;
      return *this;
    }

    /// Decrement
    CUTLASS_HOST_DEVICE
    ConstIterator &operator-=(int offset) {
      bit_ -= offset;
      return *this;
    }

    /// Post-increment
    CUTLASS_HOST_DEVICE
    ConstIterator operator++(int) {
      ConstIterator ret(*this);
      ret.bit_++;
      return ret;
    }

    /// Post-decrement
    CUTLASS_HOST_DEVICE
    ConstIterator operator--(int) {
      ConstIterator ret(*this);
      ret.bit_--;
      return ret;
    }

    /// Iterator advances by some amount
    CUTLASS_HOST_DEVICE
    ConstIterator operator+(int offset) {
      ConstIterator ret(*this);
      ret.bit_ += offset;
      return ret;
    }

    /// Iterator recedes by some amount
    CUTLASS_HOST_DEVICE
    ConstIterator operator-(int offset) {
      ConstIterator ret(*this);
      ret.bit_ -= offset;
      return ret;
    }

    /// Returns true if iterators point to the same bit
    CUTLASS_HOST_DEVICE
    bool operator==(ConstIterator const &it) const { return bit_ == it.bit_; }

    /// Returns false if iterators point to the same bit
    CUTLASS_HOST_DEVICE
    bool operator!=(ConstIterator const &it) const { return bit_ != it.bit_; }

    /// Gets the bit at the pointed to location
    CUTLASS_HOST_DEVICE
    bool get() { return vec_.at(bit_); }

    /// Gets the bit at the pointed to location
    CUTLASS_HOST_DEVICE
    bool at() const { return vec_.at(bit_); }

    /// Dereferences iterator
    CUTLASS_HOST_DEVICE
    bool operator*() const { return at(); }
  };

  /// Iterator that always returns true
  struct TrivialIterator {
    /// Constructor
    CUTLASS_HOST_DEVICE
    TrivialIterator() {}

    /// Copy constructor
    CUTLASS_HOST_DEVICE
    TrivialIterator(Iterator const &it) {}

    /// Constructs an iterator from a PredicateVector
    CUTLASS_HOST_DEVICE
    TrivialIterator(PredicateVector const &_vec) {}

    /// Pre-increment
    CUTLASS_HOST_DEVICE
    TrivialIterator &operator++() { return *this; }

    /// Post-increment
    CUTLASS_HOST_DEVICE
    TrivialIterator operator++(int) { return *this; }

    /// Dereferences iterator
    CUTLASS_HOST_DEVICE
    bool operator*() const { return true; }
  };

 public:
  //
  // Methods
  //

  /// Initialize the predicate vector
  CUTLASS_HOST_DEVICE PredicateVector(bool value = true) { fill(value); }

  /// Fills all predicates with a given value
  CUTLASS_HOST_DEVICE void fill(bool value = true) {
    Storage item = (value ? ~Storage(0) : Storage(0));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kWordCount; ++i) {
      storage(i) = item;
    }
  }

  /// Clears all predicates
  CUTLASS_HOST_DEVICE void clear() {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kWordCount; ++i) {
      storage(i) = 0;
    }
  }

  /// Sets all predicates to true
  CUTLASS_HOST_DEVICE void enable() {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kWordCount; ++i) {
      storage(i) = ~Storage(0);
    }
  }

  /// Accesses a bit within the predicate vector.
  CUTLASS_HOST_DEVICE bool operator[](int idx) const { return at(idx); }

  /// Accesses a bit within the predicate vector.
  CUTLASS_HOST_DEVICE bool at(int idx) const {
    int bit, word;
    computeStorageOffset(word, bit, idx);

    return ((storage(word) >> bit) & 1);
  }

  /// Set a bit within the predicate vector.
  CUTLASS_HOST_DEVICE void set(int idx, bool value = true) {
    int bit, word;
    computeStorageOffset(word, bit, idx);

    Storage disable_mask = (~(Storage(1) << bit));
    Storage enable_mask = (Storage(value) << bit);

    storage(word) = ((storage(word) & disable_mask) | enable_mask);
  }

  /// Computes the intersection of two identical predicate vectors.
  CUTLASS_HOST_DEVICE PredicateVector &operator&=(PredicateVector const &predicates) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kWordCount; ++i) {
      storage(i) = (storage(i) & predicates.storage(i));
    }
    return *this;
  }

  /// Computes the union of two identical predicate vectors.
  CUTLASS_HOST_DEVICE PredicateVector &operator|=(PredicateVector const &predicates) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kWordCount; ++i) {
      storage(i) = (storage(i) | predicates.storage(i));
    }
    return *this;
  }

  /// Returns true if entire predicate array is zero.
  CUTLASS_HOST_DEVICE bool is_zero() const {
   constexpr Storage mask = computeWordMask();
    Storage result = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int word = 0; word < kWordCount - 1; ++word) {
      result |= (storage(word) & mask);
    }
    constexpr Storage last_word_mask = computeLastWordMask();
    result |= (storage(kWordCount - 1) & last_word_mask);
    
    return result == 0;
  }

  /// Returns an iterator to the start of the bit vector
  CUTLASS_DEVICE
  Iterator begin() { return Iterator(*this); }

  /// Returns an iterator
  CUTLASS_DEVICE
  Iterator end() { return Iterator(*this, kPredicates); }

  /// Returns a ConstIterator
  CUTLASS_DEVICE
  ConstIterator const_begin() const { return ConstIterator(*this); }

  /// Returns a ConstIterator
  CUTLASS_DEVICE
  ConstIterator const_end() const { return ConstIterator(*this, kPredicates); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
