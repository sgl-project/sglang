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
/*! \file
    \brief Statically sized array of elements that accommodates subbyte trivial types
           in a packed storage.
*/

#pragma once

#include <cute/config.hpp>

#include <cute/numeric/numeric_types.hpp>
#include <cute/numeric/integral_constant.hpp>

namespace cute
{
//
// Underlying subbyte storage type
//
template <class T>
using subbyte_storage_type_t = conditional_t<(cute::sizeof_bits_v<T> <=   8), uint8_t,
                               conditional_t<(cute::sizeof_bits_v<T> <=  16), uint16_t,
                               conditional_t<(cute::sizeof_bits_v<T> <=  32), uint32_t,
                               conditional_t<(cute::sizeof_bits_v<T> <=  64), uint64_t,
                               conditional_t<(cute::sizeof_bits_v<T> <= 128), uint128_t,
                               T>>>>>;

template <class T> struct subbyte_iterator;
template <class, class> struct swizzle_ptr;

//
// subbyte_reference
//   Proxy object for sub-byte element references
//
template <class T>
struct subbyte_reference
{
  // Iterator Element type (const or non-const)
  using element_type = T;
  // Iterator Value type without type qualifier.
  using value_type   = remove_cv_t<T>;
  // Storage type (const or non-const)
  using storage_type = conditional_t<(is_const_v<T>), subbyte_storage_type_t<T> const, subbyte_storage_type_t<T>>;

  static_assert(sizeof_bits_v<storage_type> % 8 == 0, "Storage type is not supported");

  static_assert(sizeof_bits_v<element_type> <= sizeof_bits_v<storage_type>,
                "Size of Element must not be greater than Storage.");

private:

  // Bitmask for covering one item
  static constexpr storage_type BitMask = storage_type(storage_type(-1) >> (sizeof_bits_v<storage_type> - sizeof_bits_v<element_type>));
  // Flag for fast branching on straddled elements
  static constexpr bool is_storage_unaligned = ((sizeof_bits_v<storage_type> % sizeof_bits_v<element_type>) != 0);

  friend struct subbyte_iterator<T>;

  // Pointer to storage element
  storage_type* ptr_ = nullptr;

  // Bit index of value_type starting position within storage_type element.
  // RI: 0 <= idx_ < sizeof_bit<storage_type>
  uint8_t idx_ = 0;

  // Ctor
  template <class PointerType>
  CUTE_HOST_DEVICE constexpr
  subbyte_reference(PointerType* ptr, uint8_t idx = 0) : ptr_(reinterpret_cast<storage_type*>(ptr)), idx_(idx) {}

public:

  // Copy Ctor
  CUTE_HOST_DEVICE constexpr
  subbyte_reference(subbyte_reference<value_type> const& other) {
    *this = other.get();
  }

  CUTE_HOST_DEVICE constexpr
  subbyte_reference(subbyte_reference<value_type const> const& other) {
    *this = other.get();
  }

  // Copy Assignment
  CUTE_HOST_DEVICE constexpr
  subbyte_reference& operator=(subbyte_reference<value_type> const& other) {
    return *this = other.get();
  }

  CUTE_HOST_DEVICE constexpr
  subbyte_reference& operator=(subbyte_reference<value_type const> const& other) {
    return *this = other.get();
  }

  // Assignment
  template <class T_ = element_type>
  CUTE_HOST_DEVICE constexpr
  enable_if_t<!is_const_v<T_>, subbyte_reference&> operator=(value_type x)
  {
    static_assert(is_same_v<T_, element_type>, "Do not specify template arguments!");
    storage_type item = (reinterpret_cast<storage_type const&>(x) & BitMask);

    // Update the current storage element
    storage_type bit_mask_0 = storage_type(BitMask << idx_);
    ptr_[0] = storage_type((ptr_[0] & ~bit_mask_0) | (item << idx_));

    // If value_type is unaligned with storage_type (static) and this is a straddled value (dynamic)
    if (is_storage_unaligned && idx_ + sizeof_bits_v<value_type> > sizeof_bits_v<storage_type>) {
      uint8_t straddle_bits = uint8_t(sizeof_bits_v<storage_type> - idx_);
      storage_type bit_mask_1 = storage_type(BitMask >> straddle_bits);
      // Update the next storage element
      ptr_[1] = storage_type((ptr_[1] & ~bit_mask_1) | (item >> straddle_bits));
    }

    return *this;
  }

  // Comparison of referenced values
  CUTE_HOST_DEVICE constexpr friend
  bool operator==(subbyte_reference const& x, subbyte_reference const& y) { return x.get() == y.get(); }
  CUTE_HOST_DEVICE constexpr friend
  bool operator!=(subbyte_reference const& x, subbyte_reference const& y) { return x.get() != y.get(); }
  CUTE_HOST_DEVICE constexpr friend
  bool operator< (subbyte_reference const& x, subbyte_reference const& y) { return x.get() <  y.get(); }
  CUTE_HOST_DEVICE constexpr friend
  bool operator> (subbyte_reference const& x, subbyte_reference const& y) { return x.get() >  y.get(); }
  CUTE_HOST_DEVICE constexpr friend
  bool operator<=(subbyte_reference const& x, subbyte_reference const& y) { return x.get() <= y.get(); }
  CUTE_HOST_DEVICE constexpr friend
  bool operator>=(subbyte_reference const& x, subbyte_reference const& y) { return x.get() >= y.get(); }

  // Value
  CUTE_HOST_DEVICE
  value_type get() const
  {
    if constexpr (is_same_v<bool, value_type>) {      // Extract to bool -- potentially faster impl
      return bool((*ptr_) & (BitMask << idx_));
    } else {                                          // Extract to value_type
      // Extract from the current storage element
      auto item = storage_type((ptr_[0] >> idx_) & BitMask);

      // If value_type is unaligned with storage_type (static) and this is a straddled value (dynamic)
      if (is_storage_unaligned && idx_ + sizeof_bits_v<value_type> > sizeof_bits_v<storage_type>) {
        uint8_t straddle_bits = uint8_t(sizeof_bits_v<storage_type> - idx_);
        storage_type bit_mask_1 = storage_type(BitMask >> straddle_bits);
        // Extract from the next storage element
        item |= storage_type((ptr_[1] & bit_mask_1) << straddle_bits);
      }

      return reinterpret_cast<value_type&>(item);
    }
  }

  // Extract to type value_type
  CUTE_HOST_DEVICE constexpr
  operator value_type() const {
    return get();
  }

  // Address
  CUTE_HOST_DEVICE
  subbyte_iterator<T> operator&() const {
    return {ptr_, idx_};
  }
};

template <class T>
CUTE_HOST_DEVICE
void
print(subbyte_reference<T> ref) {
  cute::print(ref.get());
}

template <class T>
CUTE_HOST_DEVICE
void
pretty_print(subbyte_reference<T> ref) {
  cute::pretty_print(ref.get());
}

//
// subbyte_iterator
//   Random-access iterator over subbyte references
//
template <class T>
struct subbyte_iterator
{
  // Iterator Element type (const or non-const)
  using element_type = T;
  // Iterator Value type without type qualifier.
  using value_type   = remove_cv_t<T>;
  // Storage type (const or non-const)
  using storage_type = conditional_t<(is_const_v<T>), subbyte_storage_type_t<T> const, subbyte_storage_type_t<T>>;
  // Reference proxy type
  using reference = subbyte_reference<element_type>;

  static_assert(sizeof_bits_v<storage_type> % 8 == 0, "Storage type is not supported");

  static_assert(sizeof_bits_v<element_type> <= sizeof_bits_v<storage_type>,
                "Size of Element must not be greater than Storage.");

private:

  template <class, class> friend struct swizzle_ptr;
  template <class U> friend CUTE_HOST_DEVICE constexpr U* raw_pointer_cast(subbyte_iterator<U> const&);
  template <class N, class U> friend CUTE_HOST_DEVICE constexpr auto recast_ptr(subbyte_iterator<U> const&);
  template <class U> friend CUTE_HOST_DEVICE void print(subbyte_iterator<U> const&);

  // Pointer to storage element
  storage_type* ptr_;

  // Bit index of value_type starting position within storage_type element.
  // RI: 0 <= idx_ < sizeof_bit<storage_type>
  uint8_t idx_;

public:

  // Default Ctor
  CUTE_HOST_DEVICE constexpr
  subbyte_iterator() : ptr_{nullptr}, idx_{0} {};

  // Ctor
  template <class PointerType>
  CUTE_HOST_DEVICE constexpr
  subbyte_iterator(PointerType* ptr, uint8_t idx = 0) : ptr_(reinterpret_cast<storage_type*>(ptr)), idx_(idx) { }

  CUTE_HOST_DEVICE constexpr
  reference operator*() const {
    return reference(ptr_, idx_);
  }

  CUTE_HOST_DEVICE constexpr
  subbyte_iterator& operator+=(uint64_t k) {
    k = sizeof_bits_v<value_type> * k + idx_;
    ptr_ += k / sizeof_bits_v<storage_type>;
    idx_  = k % sizeof_bits_v<storage_type>;
    return *this;
  }

  CUTE_HOST_DEVICE constexpr
  subbyte_iterator operator+(uint64_t k) const {
    return subbyte_iterator(ptr_, idx_) += k;
  }

  CUTE_HOST_DEVICE constexpr
  reference operator[](uint64_t k) const {
    return *(*this + k);
  }

  CUTE_HOST_DEVICE constexpr
  subbyte_iterator& operator++() {
    idx_ += sizeof_bits_v<value_type>;
    if (idx_ >= sizeof_bits_v<storage_type>) {
      ++ptr_;
      idx_ -= sizeof_bits_v<storage_type>;
    }
    return *this;
  }

  CUTE_HOST_DEVICE constexpr
  subbyte_iterator operator++(int) {
    subbyte_iterator ret(*this);
    ++(*this);
    return ret;
  }

  CUTE_HOST_DEVICE constexpr
  subbyte_iterator& operator--() {
    if (idx_ >= sizeof_bits_v<value_type>) {
      idx_ -= sizeof_bits_v<value_type>;
    } else {
      --ptr_;
      idx_ += sizeof_bits_v<storage_type> - sizeof_bits_v<value_type>;
    }
    return *this;
  }

  CUTE_HOST_DEVICE constexpr
  subbyte_iterator operator--(int) {
    subbyte_iterator ret(*this);
    --(*this);
    return ret;
  }

  CUTE_HOST_DEVICE constexpr friend
  bool operator==(subbyte_iterator const& x, subbyte_iterator const& y) {
    return x.ptr_ == y.ptr_ && x.idx_ == y.idx_;
  }
  CUTE_HOST_DEVICE constexpr friend
  bool operator!=(subbyte_iterator const& x, subbyte_iterator const& y) { return !(x == y); }
  CUTE_HOST_DEVICE constexpr friend
  bool operator< (subbyte_iterator const& x, subbyte_iterator const& y) {
    return x.ptr_ < y.ptr_ || (x.ptr_ == y.ptr_ && x.idx_ < y.idx_);
  }
  CUTE_HOST_DEVICE constexpr friend
  bool operator<=(subbyte_iterator const& x, subbyte_iterator const& y) { return !(y <  x); }
  CUTE_HOST_DEVICE constexpr friend
  bool operator> (subbyte_iterator const& x, subbyte_iterator const& y) { return  (y <  x); }
  CUTE_HOST_DEVICE constexpr friend
  bool operator>=(subbyte_iterator const& x, subbyte_iterator const& y) { return !(x <  y); }
};

// Conversion to raw pointer with loss of subbyte index
template <class T>
CUTE_HOST_DEVICE constexpr
T*
raw_pointer_cast(subbyte_iterator<T> const& x) {
  assert(x.idx_ == 0);
  return reinterpret_cast<T*>(x.ptr_);
}

// Conversion to NewT_ with possible loss of subbyte index
template <class NewT_, class T>
CUTE_HOST_DEVICE constexpr
auto
recast_ptr(subbyte_iterator<T> const& x) {
  using NewT = conditional_t<(is_const_v<T>), NewT_ const, NewT_>;
  if constexpr (cute::is_subbyte_v<NewT>) {       // Making subbyte_iter, preserve the subbyte idx
    return subbyte_iterator<NewT>(x.ptr_, x.idx_);
  } else {                                       // Not subbyte, assume/assert subbyte idx 0
    return reinterpret_cast<NewT*>(raw_pointer_cast(x));
  }
  CUTE_GCC_UNREACHABLE;
}

// Dynamic pointers have unknown static alignment
template <class T>
CUTE_HOST_DEVICE constexpr
Int<0>
max_alignment(subbyte_iterator<T> const& x) {
  return {};
}

template <class T>
CUTE_HOST_DEVICE void
print(subbyte_iterator<T> const& x) {
  printf("subptr[%db](%p.%u)", int(sizeof_bits_v<T>), x.ptr_, x.idx_);
}

template <class T>
CUTE_HOST_DEVICE void
print(subbyte_reference<T> const& x) {
  print(x.get());
}

//
// array_subbyte
//   Statically sized array for non-byte-aligned data types
//
template <class T, size_t N>
struct array_subbyte
{
  using element_type    = T;
  using value_type      = remove_cv_t<T>;
  using pointer         = element_type*;
  using const_pointer   = element_type const*;

  using size_type       = size_t;
  using difference_type = ptrdiff_t;

  //
  // References
  //
  using reference       = subbyte_reference<element_type>;
  using const_reference = subbyte_reference<element_type const>;

  //
  // Iterators
  //
  using iterator        = subbyte_iterator<element_type>;
  using const_iterator  = subbyte_iterator<element_type const>;

  // Storage type (const or non-const)
  using storage_type = conditional_t<(is_const_v<T>), subbyte_storage_type_t<T> const, subbyte_storage_type_t<T>>;

  static_assert(sizeof_bits_v<storage_type> % 8 == 0, "Storage type is not supported");

private:

  // Number of storage elements, ceil_div
  static constexpr size_type StorageElements = (N * sizeof_bits_v<value_type> + sizeof_bits_v<storage_type> - 1) / sizeof_bits_v<storage_type>;

  // Internal storage
  storage_type storage[StorageElements];

public:

  CUTE_HOST_DEVICE constexpr
  size_type size() const {
    return N;
  }

  CUTE_HOST_DEVICE constexpr
  size_type max_size() const {
    return N;
  }

  CUTE_HOST_DEVICE constexpr
  bool empty() const {
    return !N;
  }

  // Efficient clear method
  CUTE_HOST_DEVICE constexpr
  void clear() {
    CUTE_UNROLL
    for (size_type i = 0; i < StorageElements; ++i) {
      storage[i] = storage_type(0);
    }
  }

  CUTE_HOST_DEVICE constexpr
  void fill(T const& value) {
    CUTE_UNROLL
    for (size_type i = 0; i < N; ++i) {
      at(i) = value;
    }
  }

  CUTE_HOST_DEVICE constexpr
  reference at(size_type pos) {
    return iterator(storage)[pos];
  }

  CUTE_HOST_DEVICE constexpr
  const_reference at(size_type pos) const {
    return const_iterator(storage)[pos];
  }

  CUTE_HOST_DEVICE constexpr
  reference operator[](size_type pos) {
    return at(pos);
  }

  CUTE_HOST_DEVICE constexpr
  const_reference operator[](size_type pos) const {
    return at(pos);
  }

  CUTE_HOST_DEVICE constexpr
  reference front() {
    return at(0);
  }

  CUTE_HOST_DEVICE constexpr
  const_reference front() const {
    return at(0);
  }

  CUTE_HOST_DEVICE constexpr
  reference back() {
    return at(N-1);
  }

  CUTE_HOST_DEVICE constexpr
  const_reference back() const {
    return at(N-1);
  }

  // In analogy to std::vector<bool>::data(), these functions are deleted to prevent bugs.
  // Instead, prefer
  //   auto* data = raw_pointer_cast(my_subbyte_array.begin());
  // where the type of auto* is implementation-defined and
  // with the knowledge that [data, data + my_subbyte_array.size()) may not be a valid range.
  CUTE_HOST_DEVICE constexpr
  pointer data() = delete;

  CUTE_HOST_DEVICE constexpr
  const_pointer data() const = delete;

  CUTE_HOST_DEVICE constexpr
  iterator begin() {
    return iterator(storage);
  }

  CUTE_HOST_DEVICE constexpr
  const_iterator begin() const {
    return const_iterator(storage);
  }

  CUTE_HOST_DEVICE constexpr
  const_iterator cbegin() const {
    return begin();
  }

  CUTE_HOST_DEVICE constexpr
  iterator end() {
    return iterator(storage) + N;
  }

  CUTE_HOST_DEVICE constexpr
  const_iterator end() const {
    return const_iterator(storage) + N;
  }

  CUTE_HOST_DEVICE constexpr
  const_iterator cend() const {
    return end();
  }

  //
  // Comparison operators
  //

};

//
// Operators
//

template <class T, size_t N>
CUTE_HOST_DEVICE constexpr
void clear(array_subbyte<T,N>& a)
{
  a.clear();
}

template <class T, size_t N>
CUTE_HOST_DEVICE constexpr
void fill(array_subbyte<T,N>& a, T const& value)
{
  a.fill(value);
}

} // namespace cute

//
// Specialize tuple-related functionality for cute::array_subbyte
//
#include "cutlass/cutlass.h"
#if defined(__CUDACC_RTC__)
#include CUDA_STD_HEADER(tuple)
#else
#include <tuple>
#endif

namespace cute
{

template <size_t I, class T, size_t N>
CUTE_HOST_DEVICE constexpr
T& get(array_subbyte<T,N>& a)
{
  static_assert(I < N, "Index out of range");
  return a[I];
}

template <size_t I, class T, size_t N>
CUTE_HOST_DEVICE constexpr
T const& get(array_subbyte<T,N> const& a)
{
  static_assert(I < N, "Index out of range");
  return a[I];
}

template <size_t I, class T, size_t N>
CUTE_HOST_DEVICE constexpr
T&& get(array_subbyte<T,N>&& a)
{
  static_assert(I < N, "Index out of range");
  return cute::move(a[I]);
}

} // end namespace cute

namespace CUTE_STL_NAMESPACE
{

template <class T>
struct is_reference<cute::subbyte_reference<T>>
    : CUTE_STL_NAMESPACE::true_type
{};


template <class T, size_t N>
struct tuple_size<cute::array_subbyte<T,N>>
    : CUTE_STL_NAMESPACE::integral_constant<size_t, N>
{};

template <size_t I, class T, size_t N>
struct tuple_element<I, cute::array_subbyte<T,N>>
{
  using type = T;
};

} // end namespace CUTE_STL_NAMESPACE

#ifdef CUTE_STL_NAMESPACE_IS_CUDA_STD
namespace std
{

#if defined(__CUDACC_RTC__)
template <class... _Tp>
struct tuple_size;

template <size_t _Ip, class... _Tp>
struct tuple_element;
#endif

template <class T, size_t N>
struct tuple_size<cute::array_subbyte<T,N>>
    : CUTE_STL_NAMESPACE::integral_constant<size_t, N>
{};

template <size_t I, class T, size_t N>
struct tuple_element<I, cute::array_subbyte<T,N>>
{
  using type = T;
};

} // end namespace std
#endif // CUTE_STL_NAMESPACE_IS_CUDA_STD
