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
#include "cutlass/functional.h"
#include "cutlass/numeric_types.h"
#include "cutlass/platform/platform.h"
namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Statically sized array for any data type
template <
  typename T,
  int N,
  bool RegisterSized = sizeof_bits<T>::value >= 32
>
struct Array;

namespace detail {

template<class T>
struct is_Array : platform::false_type {};

template <
  typename T,
  int N,
  bool RegisterSized
>
struct is_Array<Array<T, N, RegisterSized> > : platform::true_type {};

template<typename T>
constexpr bool is_Array_v = is_Array<T>::value;

} // namespace detail

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines the size of an Array<> in bits
template <typename T, int N, bool RegisterSized>
struct sizeof_bits<Array<T, N, RegisterSized> > {
  static constexpr int value = sizeof(Array<T, N, RegisterSized>) * 8;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns true if the argument is a power of 2
CUTLASS_HOST_DEVICE
constexpr bool ispow2(unsigned x) {
  return x && (!(x & (x - 1)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns the largest power of two not greater than the argument.
CUTLASS_HOST_DEVICE
constexpr unsigned floor_pow_2(unsigned x) {
  return (x == 0 || ispow2(x)) ? x : ((floor_pow_2(x >> 1)) << 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Statically sized array for any data type
template <
  typename T,
  int N
>
struct Array<T, N, true> {

  /// Storage type
  using Storage = T;

  /// Element type
  using Element = T;

  /// Number of storage elements
  //static std::size_t const kStorageElements = N;
  static constexpr size_t kStorageElements = N;

  /// Number of logical elements
  static constexpr size_t kElements = N;

  //
  // C++ standard members
  //

  typedef T value_type;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef value_type &reference;
  typedef value_type const & const_reference;
  typedef value_type *pointer;
  typedef value_type const * const_pointer;

  //
  // Iterators
  //

  /// Bidirectional iterator over elements
  class iterator {

    /// Pointer to object
    T *ptr_;

  public:

    CUTLASS_HOST_DEVICE
    iterator(): ptr_(nullptr) { }

    CUTLASS_HOST_DEVICE
    iterator(T *_ptr): ptr_(_ptr) { }

    CUTLASS_HOST_DEVICE
    iterator &operator++() {
      ++ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    iterator &operator--() {
      --ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    iterator operator++(int) {
      iterator ret(*this);
      ++ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    iterator operator--(int) {
      iterator ret(*this);
      --ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    T &operator*() const {
      return *ptr_;
    }

    CUTLASS_HOST_DEVICE
    bool operator==(iterator const &other) const {
      return ptr_ == other.ptr_;
    }

    CUTLASS_HOST_DEVICE
    bool operator!=(iterator const &other) const {
      return ptr_ != other.ptr_;
    }
  };

  /// Bidirectional constant iterator over elements
  class const_iterator {

    /// Pointer to object
    const T *ptr_;

  public:

    CUTLASS_HOST_DEVICE
    const_iterator(): ptr_(nullptr) { }

    CUTLASS_HOST_DEVICE
    const_iterator(T const *_ptr): ptr_(_ptr) { }

    CUTLASS_HOST_DEVICE
    const_iterator &operator++() {
      ++ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    const_iterator &operator--() {
      --ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    const_iterator operator++(int) {
      const_iterator ret(*this);
      ++ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    const_iterator operator--(int) {
      const_iterator ret(*this);
      --ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    T const &operator*() const {
      return *ptr_;
    }

    CUTLASS_HOST_DEVICE
    bool operator==(const_iterator const &other) const {
      return ptr_ == other.ptr_;
    }

    CUTLASS_HOST_DEVICE
    bool operator!=(const_iterator const &other) const {
      return ptr_ != other.ptr_;
    }
  };

  /// Bidirectional iterator over elements
  class reverse_iterator {

    /// Pointer to object
    T *ptr_;

  public:

    CUTLASS_HOST_DEVICE
    reverse_iterator(): ptr_(nullptr) { }

    CUTLASS_HOST_DEVICE
    reverse_iterator(T *_ptr): ptr_(_ptr) { }

    CUTLASS_HOST_DEVICE
    reverse_iterator &operator++() {
      --ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    reverse_iterator &operator--() {
      ++ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    reverse_iterator operator++(int) {
      iterator ret(*this);
      --ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    reverse_iterator operator--(int) {
      iterator ret(*this);
      ++ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    T &operator*() const {
      return *(ptr_ - 1);
    }

    CUTLASS_HOST_DEVICE
    bool operator==(reverse_iterator const &other) const {
      return ptr_ == other.ptr_;
    }

    CUTLASS_HOST_DEVICE
    bool operator!=(reverse_iterator const &other) const {
      return ptr_ != other.ptr_;
    }
  };

  /// Bidirectional constant iterator over elements
  class const_reverse_iterator {

    /// Pointer to object
    T const *ptr_;

  public:

    CUTLASS_HOST_DEVICE
    const_reverse_iterator(): ptr_(nullptr) { }

    CUTLASS_HOST_DEVICE
    const_reverse_iterator(T const *_ptr): ptr_(_ptr) { }

    CUTLASS_HOST_DEVICE
    const_reverse_iterator &operator++() {
      --ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    const_reverse_iterator &operator--() {
      ++ptr_;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    const_reverse_iterator operator++(int) {
      const_reverse_iterator ret(*this);
      --ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    const_reverse_iterator operator--(int) {
      const_reverse_iterator ret(*this);
      ++ptr_;
      return ret;
    }

    CUTLASS_HOST_DEVICE
    T const &operator*() const {
      return *(ptr_ - 1);
    }

    CUTLASS_HOST_DEVICE
    bool operator==(const_iterator const &other) const {
      return ptr_ == other.ptr_;
    }

    CUTLASS_HOST_DEVICE
    bool operator!=(const_iterator const &other) const {
      return ptr_ != other.ptr_;
    }
  };

  /// Internal storage
  Storage storage[kElements];

  /// Efficient clear method
  CUTLASS_HOST_DEVICE
  void clear() {
    fill(T(0));
  }

  CUTLASS_HOST_DEVICE
  reference at(size_type pos) {
    return reinterpret_cast<reference>(storage[pos]);
  }

  CUTLASS_HOST_DEVICE
  const_reference at(size_type pos) const {
    return reinterpret_cast<const_reference>(storage[pos]);
  }

  CUTLASS_HOST_DEVICE
  reference operator[](size_type pos) {
    return reinterpret_cast<reference>(storage[pos]);
  }

  CUTLASS_HOST_DEVICE
  const_reference operator[](size_type pos) const {
    return reinterpret_cast<const_reference>(storage[pos]);
  }

  CUTLASS_HOST_DEVICE
  reference front() {
    return reinterpret_cast<reference>(storage[0]);
  }

  CUTLASS_HOST_DEVICE
  const_reference front() const {
    return reinterpret_cast<const_reference>(storage[0]);
  }

  CUTLASS_HOST_DEVICE
  reference back() {
    return reinterpret_cast<reference>(storage[kStorageElements - 1]);
  }

  CUTLASS_HOST_DEVICE
  const_reference back() const {
    return reinterpret_cast<const_reference>(storage[kStorageElements - 1]);
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
  pointer raw_data() {
    return reinterpret_cast<pointer>(storage);
  }

  CUTLASS_HOST_DEVICE
  const_pointer raw_data() const {
    return reinterpret_cast<const_pointer>(storage);
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
    for (int i = 0; i < int(kElements); ++i) {
      storage[i] = static_cast<Storage>(value);
    }
  }

  CUTLASS_HOST_DEVICE
  iterator begin() {
    return iterator(storage);
  }

  CUTLASS_HOST_DEVICE
  const_iterator begin() const {
    return cbegin();
  }

  CUTLASS_HOST_DEVICE
  const_iterator cbegin() const {
    return const_iterator(storage);
  }

  CUTLASS_HOST_DEVICE
  iterator end() {
    return iterator(reinterpret_cast<pointer>(storage + kStorageElements));
  }

  CUTLASS_HOST_DEVICE
  const_iterator end() const {
    return cend();
  }

  CUTLASS_HOST_DEVICE
  const_iterator cend() const {
    return const_iterator(reinterpret_cast<const_pointer>(storage + kStorageElements));
  }

  CUTLASS_HOST_DEVICE
  reverse_iterator rbegin() {
    return reverse_iterator(reinterpret_cast<pointer>(storage + kStorageElements));
  }

  CUTLASS_HOST_DEVICE
  const_reverse_iterator rbegin() const {
    return crbegin();
  }

  CUTLASS_HOST_DEVICE
  const_reverse_iterator crbegin() const {
    return const_reverse_iterator(reinterpret_cast<const_pointer>(storage + kStorageElements));
  }

  CUTLASS_HOST_DEVICE
  reverse_iterator rend() {
    return reverse_iterator(reinterpret_cast<pointer>(storage));
  }

  CUTLASS_HOST_DEVICE
  const_reverse_iterator rend() const {
    return crend();
  }

  CUTLASS_HOST_DEVICE
  const_reverse_iterator crend() const {
    return const_reverse_iterator(reinterpret_cast<const_pointer>(storage));
  }

  //
  // Comparison operators
  //

};


////////////////////////////////////////////////////////////////////////////////////////////////////
// Factories
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element>
CUTLASS_HOST_DEVICE
Array<Element, 1> make_Array(Element x) {
  return {x};
}

template <typename Element>
CUTLASS_HOST_DEVICE
Array<Element, 2> make_Array(Element x, Element y) {
  return {x,y};
}

template <typename Element>
CUTLASS_HOST_DEVICE
Array<Element, 3> make_Array(Element x, Element y, Element z) {
  return {x,y,z};
}

template <typename Element>
CUTLASS_HOST_DEVICE
Array<Element, 4> make_Array(Element x, Element y, Element z, Element w) {
  return {x,y,z,w};
}


/////////////////////////////////////////////////////////////////////////////////////////////////
// functional.h numeric specializations
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int N>
struct absolute_value_op< Array<T, N> > {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs) const {

    Array<T, N> result;
    absolute_value_op<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i]);
    }

    return result;
  }
};

template <typename T, int N>
struct plus<Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {

    Array<T, N> result;
    plus<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {

    Array<T, N> result;
    plus<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()( T const &scalar, Array<T, N> const &rhs) const {

    Array<T, N> result;
    plus<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};
template <typename T, int N>
struct minus<Array<T, N>> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {

    Array<T, N> result;
    minus<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {

    Array<T, N> result;
    minus<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()( T const &scalar, Array<T, N> const &rhs) const {

    Array<T, N> result;
    minus<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T, int N>
struct multiplies<Array<T, N>> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {

    Array<T, N> result;
    multiplies<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {

    Array<T, N> result;
    multiplies<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()( T const &scalar, Array<T, N> const &rhs) const {

    Array<T, N> result;
    multiplies<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T, int N, bool PropogateNaN>
struct maximum_absolute_value_reduction<Array<T, N>, PropogateNaN> {

  CUTLASS_HOST_DEVICE
  T operator() (T const& scalar, Array<T, N> const& rhs) const {

    T result = scalar;
    maximum_absolute_value_reduction<T, PropogateNaN> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result = scalar_op(result, rhs[i]);
    }

    return result;
  }
};

template <typename T, int N>
struct scale<Array<T, N>> {
  T const scaling_factor_;

  CUTLASS_HOST_DEVICE
  scale(T scaling_factor) : scaling_factor_(scaling_factor) {
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const & rhs) const {
    Array<T, N> result;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = rhs[i] * scaling_factor_;
    }

    return result;
  }
};

template <typename T, int N>
struct divides<Array<T, N>> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {

    Array<T, N> result;
    divides<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {

    Array<T, N> result;
    divides<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()( T const &scalar, Array<T, N> const &rhs) const {

    Array<T, N> result;
    divides<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T, int N>
struct reciprocal_approximate<Array<T, N>> {
  
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs) const {

    Array<T, N> result;
    reciprocal_approximate<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i]);
    }

    return result;
  }
};

template <typename T, int N>
struct reciprocal_approximate_ftz<Array<T, N>> {
  
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs) const {

    Array<T, N> result;
    reciprocal_approximate_ftz<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i]);
    }

    return result;
  }
};

template <typename T, int N, bool PropagateNaN>
struct maximum<Array<T, N>, PropagateNaN> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {

    Array<T, N> result;
    maximum<T, PropagateNaN> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {

    Array<T, N> result;
    maximum<T, PropagateNaN> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &rhs) const {

    Array<T, N> result;
    maximum<T, PropagateNaN> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T, int N, bool PropagateNaN>
struct minimum<Array<T, N>, PropagateNaN> {

  CUTLASS_HOST_DEVICE
  static T scalar_op(T const &lhs, T const &rhs) {
    return (rhs < lhs ? rhs : lhs);
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {

    Array<T, N> result;
    minimum<T, PropagateNaN> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {

    Array<T, N> result;
    minimum<T, PropagateNaN> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &rhs) const {

    Array<T, N> result;
    minimum<T, PropagateNaN> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T, int N>
struct minimum_with_nan_propagation<Array<T, N>> : minimum<Array<T, N>, true> 
{};

template <typename T, int N>
struct negate<Array<T, N>> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs) const {

    Array<T, N> result;
    negate<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i]);
    }

    return result;
  }
};

/// Fused multiply-add
template <typename T, int N>
struct multiply_add<Array<T, N>, Array<T, N>, Array<T, N>> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a, Array<T, N> const &b, Array<T, N> const &c) const {

    Array<T, N> result;
    multiply_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(a[i], b[i], c[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a, T const &scalar, Array<T, N> const &c) const {

    Array<T, N> result;
    multiply_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(a[i], scalar, c[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &b, Array<T, N> const &c) const {

    Array<T, N> result;
    multiply_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, b[i], c[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a, Array<T, N> const &b, T const &scalar) const {

    Array<T, N> result;
    multiply_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(a[i], b[i], scalar);
    }

    return result;
  }


  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a, T const &scalar_b, T const &scalar_c) const {

    Array<T, N> result;
    multiply_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(a[i], scalar_b, scalar_c);
    }

    return result;
  }
};

/// Fused square-and-plus
template <typename T, int N>
struct square_and_plus<Array<T, N>> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
    multiply_add<Array<T, N>, Array<T, N>, Array<T, N>> ma_op;
    return ma_op(rhs, rhs, lhs);
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &rhs) const {
    plus<Array<T, N>> plus_op;
    multiplies<T> multiplies_op;
    return plus_op(multiplies_op(rhs, rhs), lhs);
  }
};

/// Inverse-square-root
template <typename T, int N>
struct inverse_square_root<Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a) const {
    Array<T, N> result;
    inverse_square_root<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(a[i]);
    }
    return result;
  }
};

template <int N>
struct inverse_square_root<Array<half_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & a) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *a_ptr = reinterpret_cast<__half2 const *>(&a);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = h2rsqrt(a_ptr[i]);
    }

    if constexpr (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&a);
      __half d_residual = hrsqrt(a_residual_ptr[N - 1]);
      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    inverse_square_root<half_t> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(a[i]);
    }

    #endif

    return result;
  }
};

/// Fused multiply-add-relu0
template <typename T, int N>
struct multiply_add_relu0<Array<T, N>, Array<T, N>, Array<T, N>> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a, Array<T, N> const &b, Array<T, N> const &c) const {

    Array<T, N> result;
    multiply_add<T> scalar_op;
    maximum<T> mx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = mx(scalar_op(a[i], b[i], c[i]), T(0));
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a, T const &scalar, Array<T, N> const &c) const {

    Array<T, N> result;
    multiply_add<T> scalar_op;
    maximum<T> mx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = mx(scalar_op(a[i], scalar, c[i]), T(0));
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &b, Array<T, N> const &c) const {

    Array<T, N> result;
    multiply_add<T> scalar_op;
    maximum<T> mx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = mx(scalar_op(scalar, b[i], c[i]), T(0));
    }

    return result;
  }
};


template <typename T, int N>
struct conjugate<Array<T, N> >  {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a) const {

    conjugate<T> conj_op;

    Array<T, N> ca;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      ca[i] = conj_op(a[i]);
    }
    return ca;
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
// functional.h numeric specializations targeting SIMD instructions in device code.
/////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
struct plus<Array<half_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hadd2(lhs_ptr[i], rhs_ptr[i]);
    }

    if constexpr (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);
      __half d_residual = __hadd(a_residual_ptr[N - 1], b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] + rhs[i];
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(half_t const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 lhs_pair = __half2half2(reinterpret_cast<__half const &>(lhs));
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hadd2(lhs_pair, rhs_ptr[i]);
    }

    if constexpr (N % 2) {
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);
      __half d_residual = __hadd(reinterpret_cast<__half const &>(lhs), b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs + rhs[i];
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, half_t const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 rhs_pair = __half2half2(reinterpret_cast<__half const &>(rhs));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hadd2(lhs_ptr[i], rhs_pair);
    }

    if constexpr (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half d_residual = __hadd(a_residual_ptr[N - 1], reinterpret_cast<__half const &>(rhs));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] + rhs;
    }
    #endif

    return result;
  }
};

template <int N>
struct minus<Array<half_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hsub2(lhs_ptr[i], rhs_ptr[i]);
    }

    if constexpr (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);
      __half d_residual = __hsub(a_residual_ptr[N - 1], b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] - rhs[i];
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(half_t const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 lhs_pair = __half2half2(reinterpret_cast<__half const &>(lhs));
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hsub2(lhs_pair, rhs_ptr[i]);
    }

    if constexpr (N % 2) {
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);
      __half d_residual = __hsub(reinterpret_cast<__half const &>(lhs), b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs - rhs[i];
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, half_t const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 rhs_pair = __half2half2(reinterpret_cast<__half const &>(rhs));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hsub2(lhs_ptr[i], rhs_pair);
    }

    if constexpr (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half d_residual = __hsub(a_residual_ptr[N - 1], reinterpret_cast<__half const &>(rhs));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] - rhs;
    }
    #endif

    return result;
  }
};

template <int N>
struct multiplies<Array<half_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hmul2(lhs_ptr[i], rhs_ptr[i]);
    }

    if constexpr (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);
      __half d_residual = __hmul(a_residual_ptr[N - 1], b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] * rhs[i];
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(half_t const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 lhs_pair = __half2half2(reinterpret_cast<__half const &>(lhs));
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hmul2(lhs_pair, rhs_ptr[i]);
    }

    if constexpr (N % 2) {
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);

      __half d_residual = __hmul(
        reinterpret_cast<__half const &>(lhs),
        b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs * rhs[i];
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, half_t const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 rhs_pair = __half2half2(reinterpret_cast<__half const &>(rhs));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hmul2(lhs_ptr[i], rhs_pair);
    }

    if constexpr (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);

      __half d_residual = __hmul(
        a_residual_ptr[N - 1],
        reinterpret_cast<__half const &>(rhs));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] * rhs;
    }
    #endif

    return result;
  }
};

template <int N>
struct divides<Array<half_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __h2div(lhs_ptr[i], rhs_ptr[i]);
    }

    if constexpr (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);

      __half d_residual = __hdiv(
        a_residual_ptr[N - 1],
        b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] / rhs[i];
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(half_t const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 lhs_pair = __half2half2(reinterpret_cast<__half const &>(lhs));
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __h2div(lhs_pair, rhs_ptr[i]);
    }

    if constexpr (N % 2) {
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);

      __half d_residual = __hdiv(
        reinterpret_cast<__half const &>(lhs),
        b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs / rhs[i];
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, half_t const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 rhs_pair = __half2half2(reinterpret_cast<__half const &>(rhs));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __h2div(lhs_ptr[i], rhs_pair);
    }

    if constexpr (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);

      __half d_residual = __hdiv(
        a_residual_ptr[N - 1],
        reinterpret_cast<__half const &>(rhs));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = lhs[i] / rhs;
    }
    #endif

    return result;
  }
};

template <int N>
struct negate<Array<half_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *source_ptr = reinterpret_cast<__half2 const *>(&lhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hneg2(source_ptr[i]);
    }

    if constexpr (N % 2) {
      half_t x = -lhs[N - 1];
      __half lhs_val = reinterpret_cast<__half const &>(x);
      result[N - 1] = reinterpret_cast<half_t const &>(lhs_val);
    }

    #else

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = -lhs[i];
    }
    #endif

    return result;
  }
};

/// Fused multiply-add
template <int N>
struct multiply_add<Array<half_t, N>, Array<half_t, N>, Array<half_t, N>> {

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(
    Array<half_t, N> const &a,
    Array<half_t, N> const &b,
    Array<half_t, N> const &c) const {

    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *a_ptr = reinterpret_cast<__half2 const *>(&a);
    __half2 const *b_ptr = reinterpret_cast<__half2 const *>(&b);
    __half2 const *c_ptr = reinterpret_cast<__half2 const *>(&c);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2(a_ptr[i], b_ptr[i], c_ptr[i]);
    }

    if constexpr (N % 2) {

      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&a);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&b);
      __half const *c_residual_ptr = reinterpret_cast<__half const *>(&c);

      __half d_residual = __hfma(
        a_residual_ptr[N - 1],
        b_residual_ptr[N - 1],
        c_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    multiply_add<half_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b[i], c[i]);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(
    half_t const &a,
    Array<half_t, N> const &b,
    Array<half_t, N> const &c) const {

    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 a_pair = __half2half2(reinterpret_cast<__half const &>(a));
    __half2 const *b_ptr = reinterpret_cast<__half2 const *>(&b);
    __half2 const *c_ptr = reinterpret_cast<__half2 const *>(&c);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2(a_pair, b_ptr[i], c_ptr[i]);
    }

    if constexpr (N % 2) {

      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&b);
      __half const *c_residual_ptr = reinterpret_cast<__half const *>(&c);
      __half d_residual = __hfma(
        reinterpret_cast<__half const &>(a),
        b_residual_ptr[N - 1],
        c_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    multiply_add<half_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a, b[i], c[i]);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(
    Array<half_t, N> const &a,
    half_t const &b,
    Array<half_t, N> const &c) const {

    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *a_ptr = reinterpret_cast<__half2 const *>(&a);
    __half2 b_pair = __half2half2(reinterpret_cast<__half const &>(b));
    __half2 const *c_ptr = reinterpret_cast<__half2 const *>(&c);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2(a_ptr[i], b_pair, c_ptr[i]);
    }

    if constexpr (N % 2) {

      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&a);
      __half const *c_residual_ptr = reinterpret_cast<__half const *>(&c);

      __half d_residual = __hfma(
        a_residual_ptr[N - 1],
        reinterpret_cast<__half const &>(b),
        c_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    multiply_add<half_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b, c[i]);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(
    Array<half_t, N> const &a,
    Array<half_t, N> const &b,
    half_t const &c) const {

    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *a_ptr = reinterpret_cast<__half2 const *>(&a);
    __half2 const *b_ptr = reinterpret_cast<__half2 const *>(&b);
    __half2 c_pair = __half2half2(reinterpret_cast<__half const &>(c));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2(a_ptr[i], b_ptr[i], c_pair);
    }

    if constexpr (N % 2) {

      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&a);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&b);

      __half d_residual = __hfma(
        a_residual_ptr[N - 1],
        b_residual_ptr[N - 1],
        reinterpret_cast<__half const &>(c));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    multiply_add<half_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b[i], c);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(
    Array<half_t, N> const &a,
    half_t const &b,
    half_t const &c) const {

    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *a_ptr = reinterpret_cast<__half2 const *>(&a);
    __half2 b_pair = __half2half2(reinterpret_cast<__half const &>(b));
    __half2 c_pair = __half2half2(reinterpret_cast<__half const &>(c));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2(a_ptr[i], b_pair, c_pair);
    }

    if constexpr (N % 2) {

      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&a);

      __half d_residual = __hfma(
        a_residual_ptr[N - 1],
        reinterpret_cast<__half const &>(b),
        reinterpret_cast<__half const &>(c));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    multiply_add<half_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b, c);
    }
    #endif

    return result;
  }
};

/// Fused multiply-add-relu0
template <int N>
struct multiply_add_relu0<Array<half_t, N>, Array<half_t, N>, Array<half_t, N>> {

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(
    Array<half_t, N> const &a,
    Array<half_t, N> const &b,
    Array<half_t, N> const &c) const {

    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *a_ptr = reinterpret_cast<__half2 const *>(&a);
    __half2 const *b_ptr = reinterpret_cast<__half2 const *>(&b);
    __half2 const *c_ptr = reinterpret_cast<__half2 const *>(&c);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2_relu(a_ptr[i], b_ptr[i], c_ptr[i]);
    }

    if constexpr (N % 2) {

      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&a);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&b);
      __half const *c_residual_ptr = reinterpret_cast<__half const *>(&c);

      __half d_residual = __hfma_relu(
        a_residual_ptr[N - 1],
        b_residual_ptr[N - 1],
        c_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    multiply_add<half_t> op;
    maximum<half_t> mx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = mx(op(a[i], b[i], c[i]), (half_t)0);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(
    half_t const &a,
    Array<half_t, N> const &b,
    Array<half_t, N> const &c) const {

    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 a_pair = __half2half2(reinterpret_cast<__half const &>(a));
    __half2 const *b_ptr = reinterpret_cast<__half2 const *>(&b);
    __half2 const *c_ptr = reinterpret_cast<__half2 const *>(&c);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2_relu(a_pair, b_ptr[i], c_ptr[i]);
    }

    if constexpr (N % 2) {

      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&b);
      __half const *c_residual_ptr = reinterpret_cast<__half const *>(&c);
      __half d_residual = __hfma_relu(
        reinterpret_cast<__half const &>(a),
        b_residual_ptr[N - 1],
        c_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    multiply_add<half_t> op;
    maximum<half_t> mx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = mx(op(a, b[i], c[i]), half_t(0));
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(
    Array<half_t, N> const &a,
    half_t const &b,
    Array<half_t, N> const &c) const {

    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *a_ptr = reinterpret_cast<__half2 const *>(&a);
    __half2 b_pair = __half2half2(reinterpret_cast<__half const &>(b));
    __half2 const *c_ptr = reinterpret_cast<__half2 const *>(&c);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2_relu(a_ptr[i], b_pair, c_ptr[i]);
    }

    if constexpr (N % 2) {

      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&a);
      __half const *c_residual_ptr = reinterpret_cast<__half const *>(&c);

      __half d_residual = __hfma_relu(
        a_residual_ptr[N - 1],
        reinterpret_cast<__half const &>(b),
        c_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    multiply_add<half_t> op;
    maximum<half_t> mx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = mx(op(a[i], b, c[i]), half_t(0));
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(
    Array<half_t, N> const &a,
    Array<half_t, N> const &b,
    half_t const &c) const {

    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *a_ptr = reinterpret_cast<__half2 const *>(&a);
    __half2 const *b_ptr = reinterpret_cast<__half2 const *>(&b);
    __half2 c_pair = __half2half2(reinterpret_cast<__half const &>(c));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = __hfma2_relu(a_ptr[i], b_ptr[i], c_pair);
    }

    if constexpr (N % 2) {

      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&a);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&b);

      __half d_residual = __hfma_relu(
        a_residual_ptr[N - 1],
        b_residual_ptr[N - 1],
        reinterpret_cast<__half const &>(c));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    multiply_add<half_t> op;
    maximum<half_t> mx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = mx(op(a[i], b[i], c), half_t(0));
    }
    #endif

    return result;
  }
};

template <int N, bool PropagateNaN>
struct minimum<Array<half_t, N>, PropagateNaN> {
  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = PropagateNaN ? __hmin2_nan(lhs_ptr[i], rhs_ptr[i])
                                   : __hmin2(lhs_ptr[i], rhs_ptr[i]);
    }

    if constexpr (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);

      __half d_residual = PropagateNaN ? __hmin_nan(a_residual_ptr[N - 1], b_residual_ptr[N - 1])
                                       : __hmin(a_residual_ptr[N - 1], b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    minimum<half_t,PropagateNaN> mn;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = mn(lhs[i],rhs[i]);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(half_t const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 lhs_pair = __half2half2(reinterpret_cast<__half const &>(lhs));
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = PropagateNaN ? __hmin2_nan(lhs_pair, rhs_ptr[i])
                                   : __hmin2(lhs_pair, rhs_ptr[i]);
    }

    if constexpr (N % 2) {
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);

      __half d_residual = PropagateNaN ? __hmin_nan(reinterpret_cast<__half const &>(lhs), b_residual_ptr[N - 1])
                                       : __hmin(reinterpret_cast<__half const &>(lhs), b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    minimum<half_t,PropagateNaN> mn;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = mn(lhs, rhs[i]);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, half_t const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 rhs_pair = __half2half2(reinterpret_cast<__half const &>(rhs));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = PropagateNaN ? __hmin2_nan(lhs_ptr[i], rhs_pair)
                                   : __hmin2(lhs_ptr[i], rhs_pair);
    }

    if constexpr (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);

      __half d_residual = PropagateNaN ? __hmin_nan(a_residual_ptr[N - 1], reinterpret_cast<__half const &>(rhs))
                                       : __hmin(a_residual_ptr[N - 1], reinterpret_cast<__half const &>(rhs));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    minimum<half_t, PropagateNaN> mn;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = mn(lhs[i], rhs);
    }
    #endif

    return result;
  }
};

template <int N, bool PropagateNaN>
struct maximum<Array<half_t, N>, PropagateNaN> {
  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = PropagateNaN ? __hmax2_nan(lhs_ptr[i], rhs_ptr[i])
                                   : __hmax2(lhs_ptr[i], rhs_ptr[i]);
    }

    if constexpr (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);

      __half d_residual = PropagateNaN ? __hmax(a_residual_ptr[N - 1], b_residual_ptr[N - 1])
                                       : __hmax_nan(a_residual_ptr[N - 1], b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    maximum<half_t,PropagateNaN> mx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = mx(lhs[i], rhs[i]);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(half_t const & lhs, Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 lhs_pair = __half2half2(reinterpret_cast<__half const &>(lhs));
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = PropagateNaN ? __hmax2_nan(lhs_pair, rhs_ptr[i])
                                   : __hmax2(lhs_pair, rhs_ptr[i]);
    }

    if constexpr (N % 2) {
      __half const *b_residual_ptr = reinterpret_cast<__half const *>(&rhs);

      __half d_residual = PropagateNaN ? __hmax_nan(reinterpret_cast<__half const &>(lhs), b_residual_ptr[N - 1])
                                       : __hmax(reinterpret_cast<__half const &>(lhs), b_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    maximum<half_t,PropagateNaN> mx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = mx(lhs, rhs[i]);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const & lhs, half_t const &rhs) const {
    Array<half_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(&lhs);
    __half2 rhs_pair = __half2half2(reinterpret_cast<__half const &>(rhs));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = PropagateNaN ? __hmax2_nan(lhs_ptr[i], rhs_pair)
                                   : __hmax2(lhs_ptr[i], rhs_pair);
    }

    if constexpr (N % 2) {
      __half const *a_residual_ptr = reinterpret_cast<__half const *>(&lhs);

      __half d_residual = PropagateNaN ? __hmax_nan(a_residual_ptr[N - 1], reinterpret_cast<__half const &>(rhs))
                                       : __hmax(a_residual_ptr[N - 1], reinterpret_cast<__half const &>(rhs));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

    #else

    maximum<half_t,PropagateNaN> mx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = mx(lhs[i], rhs);
    }
    #endif

    return result;
  }
};

/// Fused multiply-add
template <int N>
struct multiply_add<Array<bfloat16_t, N>, Array<bfloat16_t, N>, Array<bfloat16_t, N>> {

  CUTLASS_HOST_DEVICE
  Array<bfloat16_t, N> operator()(
    Array<bfloat16_t, N> const &a,
    Array<bfloat16_t, N> const &b,
    Array<bfloat16_t, N> const &c) const {

    Array<bfloat16_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    unsigned *result_ptr = reinterpret_cast<unsigned *>(&result);
    unsigned const *a_ptr = reinterpret_cast<unsigned const *>(&a);
    unsigned const *b_ptr = reinterpret_cast<unsigned const *>(&b);
    unsigned const *c_ptr = reinterpret_cast<unsigned const *>(&c);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      asm ("fma.rn.bf16x2 %0, %1, %2, %3;\n"
        : "=r"(result_ptr[i])
        : "r"(a_ptr[i]), "r"(b_ptr[i]), "r"(c_ptr[i])
      );
    }

    if constexpr (N % 2) {

      uint16_t *result_ptr = reinterpret_cast<uint16_t *>(&result);
      uint16_t const *a_residual_ptr = reinterpret_cast<uint16_t const *>(&a);
      uint16_t const *b_residual_ptr = reinterpret_cast<uint16_t const *>(&b);
      uint16_t const *c_residual_ptr = reinterpret_cast<uint16_t const *>(&c);

      asm ("fma.rn.bf16 %0, %1, %2, %3;\n"
        : "=h"(result_ptr[N - 1])
        : "h"(a_residual_ptr[N - 1]), "h"(b_residual_ptr[N - 1]), "h"(c_residual_ptr[N - 1])
      );
    }

    #else

    multiply_add<bfloat16_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b[i], c[i]);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<bfloat16_t, N> operator()(
    bfloat16_t const &a,
    Array<bfloat16_t, N> const &b,
    Array<bfloat16_t, N> const &c) const {

    Array<bfloat16_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    unsigned *result_ptr = reinterpret_cast<unsigned *>(&result);

    unsigned const *b_ptr = reinterpret_cast<unsigned const *>(&b);
    unsigned const *c_ptr = reinterpret_cast<unsigned const *>(&c);

    unsigned a_packed = static_cast<unsigned>(a.raw());
    a_packed = (a_packed | (a_packed << 16));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      asm ("fma.rn.bf16x2 %0, %1, %2, %3;\n"
        : "=r"(result_ptr[i])
        : "r"(a_packed), "r"(b_ptr[i]), "r"(c_ptr[i])
      );
    }

    if constexpr (N % 2) {

      uint16_t *result_ptr = reinterpret_cast<uint16_t *>(&result);
      uint16_t const *a_residual_ptr = reinterpret_cast<uint16_t const *>(&a);
      uint16_t const *b_residual_ptr = reinterpret_cast<uint16_t const *>(&b);
      uint16_t const *c_residual_ptr = reinterpret_cast<uint16_t const *>(&c);

      asm ("fma.rn.bf16 %0, %1, %2, %3;\n"
        : "=h"(result_ptr[N - 1])
        : "h"(a_residual_ptr[0]), "h"(b_residual_ptr[N - 1]), "h"(c_residual_ptr[N - 1])
      );
    }

    #else

    multiply_add<bfloat16_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a, b[i], c[i]);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<bfloat16_t, N> operator()(
    Array<bfloat16_t, N> const &a,
    bfloat16_t const &b,
    Array<bfloat16_t, N> const &c) const {

    Array<bfloat16_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    unsigned *result_ptr = reinterpret_cast<unsigned *>(&result);

    unsigned const *a_ptr = reinterpret_cast<unsigned const *>(&a);
    unsigned const *c_ptr = reinterpret_cast<unsigned const *>(&c);

    unsigned b_packed = static_cast<unsigned>(b.raw());
    b_packed = (b_packed | (b_packed << 16));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      asm ("fma.rn.bf16x2 %0, %1, %2, %3;\n"
        : "=r"(result_ptr[i])
        : "r"(a_ptr[i]), "r"(b_packed), "r"(c_ptr[i])
      );
    }

    if constexpr (N % 2) {

      uint16_t *result_ptr = reinterpret_cast<uint16_t *>(&result);
      uint16_t const *a_residual_ptr = reinterpret_cast<uint16_t const *>(&a);
      uint16_t const *b_residual_ptr = reinterpret_cast<uint16_t const *>(&b);
      uint16_t const *c_residual_ptr = reinterpret_cast<uint16_t const *>(&c);

      asm ("fma.rn.bf16 %0, %1, %2, %3;\n"
        : "=h"(result_ptr[N - 1])
        : "h"(a_residual_ptr[N - 1]), "h"(b_residual_ptr[0]), "h"(c_residual_ptr[N - 1])
      );
    }

    #else

    multiply_add<bfloat16_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b, c[i]);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<bfloat16_t, N> operator()(
    Array<bfloat16_t, N> const &a,
    Array<bfloat16_t, N> const &b,
    bfloat16_t const &c) const {

    Array<bfloat16_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    unsigned *result_ptr = reinterpret_cast<unsigned *>(&result);

    unsigned const *a_ptr = reinterpret_cast<unsigned const *>(&a);
    unsigned const *b_ptr = reinterpret_cast<unsigned const *>(&b);

    unsigned c_packed = static_cast<unsigned>(c.raw());
    c_packed = (c_packed | (c_packed << 16));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      asm ("fma.rn.bf16x2 %0, %1, %2, %3;\n"
        : "=r"(result_ptr[i])
        : "r"(a_ptr[i]), "r"(b_ptr[i]), "r"(c_packed)
      );
    }

    if constexpr (N % 2) {

      uint16_t *result_ptr = reinterpret_cast<uint16_t *>(&result);
      uint16_t const *a_residual_ptr = reinterpret_cast<uint16_t const *>(&a);
      uint16_t const *b_residual_ptr = reinterpret_cast<uint16_t const *>(&b);
      uint16_t const *c_residual_ptr = reinterpret_cast<uint16_t const *>(&c);

      asm ("fma.rn.bf16 %0, %1, %2, %3;\n"
        : "=h"(result_ptr[N - 1])
        : "h"(a_residual_ptr[N - 1]), "h"(b_residual_ptr[N - 1]), "h"(c_residual_ptr[0])
      );
    }

    #else

    multiply_add<bfloat16_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b[i], c);
    }
    #endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<bfloat16_t, N> operator()(
    Array<bfloat16_t, N> const &a,
    bfloat16_t const &b,
    bfloat16_t const &c) const {

    Array<bfloat16_t, N> result;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

    unsigned *result_ptr = reinterpret_cast<unsigned *>(&result);

    unsigned const *a_ptr = reinterpret_cast<unsigned const *>(&a);

    unsigned b_packed = static_cast<unsigned>(b.raw());
    b_packed = (b_packed | (b_packed << 16));

    unsigned c_packed = static_cast<unsigned>(c.raw());
    c_packed = (c_packed | (c_packed << 16));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      asm ("fma.rn.bf16x2 %0, %1, %2, %3;\n"
        : "=r"(result_ptr[i])
        : "r"(a_ptr[i]), "r"(b_packed), "r"(c_packed)
      );
    }

    if constexpr (N % 2) {

      uint16_t *result_ptr = reinterpret_cast<uint16_t *>(&result);
      uint16_t const *a_residual_ptr = reinterpret_cast<uint16_t const *>(&a);
      uint16_t const *b_residual_ptr = reinterpret_cast<uint16_t const *>(&b);
      uint16_t const *c_residual_ptr = reinterpret_cast<uint16_t const *>(&c);

      asm ("fma.rn.bf16 %0, %1, %2, %3;\n"
        : "=h"(result_ptr[N - 1])
        : "h"(a_residual_ptr[N - 1]), "h"(b_residual_ptr[0]), "h"(c_residual_ptr[0])
      );
    }


    #else

    multiply_add<bfloat16_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b, c);
    }
    #endif

    return result;
  }
};


/// bit_and
template <int N>
struct bit_and<Array<uint1b_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<uint1b_t, N> operator()(Array<uint1b_t, N> const &a, Array<uint1b_t, N> const &b) const {
    using ArrayType = Array<uint1b_t, N>;
    using Storage = typename ArrayType::Storage;
    ArrayType result;

    Storage *result_data = result.raw_data();
    Storage const *a_data = a.raw_data();
    Storage const *b_data = b.raw_data();

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ArrayType::kStorageElements; ++i) {
      result_data[i] = (a_data[i] & b_data[i]);
    }

    return result;
  }
};


/// bit_or
template <int N>
struct bit_or<Array<uint1b_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<uint1b_t, N> operator()(Array<uint1b_t, N> const &a, Array<uint1b_t, N> const &b) const {
    using ArrayType = Array<uint1b_t, N>;
    using Storage = typename ArrayType::Storage;
    ArrayType result;

    Storage *result_data = result.raw_data();
    Storage const *a_data = a.raw_data();
    Storage const *b_data = b.raw_data();

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ArrayType::kStorageElements; ++i) {
      result_data[i] = (a_data[i] | b_data[i]);
    }

    return result;
  }
};


/// bit_not
template <int N>
struct bit_not<Array<uint1b_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<uint1b_t, N> operator()(Array<uint1b_t, N> const &a) const {
    using ArrayType = Array<uint1b_t, N>;
    using Storage = typename ArrayType::Storage;
    ArrayType result;

    Storage *result_data = result.raw_data();
    Storage const *a_data = a.raw_data();

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ArrayType::kStorageElements; ++i) {
      result_data[i] = (~a_data[i]);
    }

    return result;
  }
};

/// bit_xor
template <int N>
struct bit_xor<Array<uint1b_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<uint1b_t, N> operator()(Array<uint1b_t, N> const &a, Array<uint1b_t, N> const &b) const {
    using ArrayType = Array<uint1b_t, N>;
    using Storage = typename ArrayType::Storage;
    ArrayType result;

    Storage *result_data = result.raw_data();
    Storage const *a_data = a.raw_data();
    Storage const *b_data = b.raw_data();

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ArrayType::kStorageElements; ++i) {
      result_data[i] = (a_data[i] ^ b_data[i]);
    }

    return result;
  }
};

/// Fused and-popc-add
template <typename T, int N>
struct and_popc_add<Array<T, N>, Array<T, N>, Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a, Array<T, N> const &b, Array<T, N> const &c) const {
    Array<T, N> result;
    and_popc_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(a[i], b[i], c[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a, T const &scalar, Array<T, N> const &c) const {
    Array<T, N> result;
    and_popc_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(a[i], scalar, c[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &b, Array<T, N> const &c) const {
    Array<T, N> result;
    and_popc_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, b[i], c[i]);
    }

    return result;
  }
};


/// Fused or-popc-add
template <typename T, int N>
struct or_popc_add<Array<T, N>, Array<T, N>, Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a, Array<T, N> const &b, Array<T, N> const &c) const {
    Array<T, N> result;
    or_popc_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(a[i], b[i], c[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a, T const &scalar, Array<T, N> const &c) const {
    Array<T, N> result;
    or_popc_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(a[i], scalar, c[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &b, Array<T, N> const &c) const {
    Array<T, N> result;
    or_popc_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, b[i], c[i]);
    }

    return result;
  }
};

/// Fused xor-popc-add
template <typename T, int N>
struct xor_popc_add<Array<T, N>, Array<T, N>, Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a, Array<T, N> const &b, Array<T, N> const &c) const {
    Array<T, N> result;
    xor_popc_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(a[i], b[i], c[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &a, T const &scalar, Array<T, N> const &c) const {
    Array<T, N> result;
    xor_popc_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(a[i], scalar, c[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &b, Array<T, N> const &c) const {
    Array<T, N> result;
    xor_popc_add<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, b[i], c[i]);
    }

    return result;
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
// Operator overloads
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> operator+(Array<T, N> const &lhs, Array<T, N> const &rhs) {
  plus<Array<T, N>> op;
  return op(lhs, rhs);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> operator+(T const &lhs, Array<T, N> const &rhs) {
  plus<Array<T, N>> op;
  return op(lhs, rhs);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> operator+(Array<T, N> const &lhs, T const &rhs) {
  plus<Array<T, N>> op;
  return op(lhs, rhs);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> operator-(Array<T, N> const &lhs, Array<T, N> const &rhs) {
  minus<Array<T, N>> op;
  return op(lhs, rhs);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> operator-(Array<T, N> const &lhs) {
  negate<Array<T, N>> op;
  return op(lhs);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> operator*(Array<T, N> const &lhs, Array<T, N> const &rhs) {
  multiplies<Array<T, N>> op;
  return op(lhs, rhs);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> operator*(T lhs, Array<T, N> const &rhs) {
  multiplies<Array<T, N>> op;
  return op(lhs, rhs);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> operator*(Array<T, N> const &lhs, T rhs) {
  multiplies<Array<T, N>> op;
  return op(lhs, rhs);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> operator/(Array<T, N> const &lhs, Array<T, N> const &rhs) {
  divides<Array<T, N>> op;
  return op(lhs, rhs);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> fma(Array<T, N> const &a, Array<T, N> const &b, Array<T, N> const &c) {
  multiply_add<Array<T, N>> op;
  return op(a, b, c);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> fma(T a, Array<T, N> const &b, Array<T, N> const &c) {
  multiply_add<Array<T, N>> op;
  return op(a, b, c);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> fma(Array<T, N> const &a, T b, Array<T, N> const &c) {
  multiply_add<Array<T, N>> op;
  return op(a, b, c);
}

template <typename T, int N>
CUTLASS_HOST_DEVICE
Array<T, N> fma(Array<T, N> const &a, Array<T, N> const &b, T c) {
  multiply_add<Array<T, N>> op;
  return op(a, b, c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
  

////////////////////////////////////////////////////////////////////////////////////////////////////
// AlignedArray
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Aligned array type
template <
  /// Element type
  typename T,
  /// Number of elements in the array
  int N,
  /// Alignment requirement in bytes
  int Alignment = ( sizeof_bits<T>::value * N + 7 ) / 8
>
class alignas(Alignment) AlignedArray: public Array<T, N> {
public:

};

} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/array_subbyte.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
