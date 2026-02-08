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

#include <cute/config.hpp>                     // CUTE_HOST_DEVICE
#include <cute/numeric/numeric_types.hpp>      // cute::sizeof_bits
#include <cute/numeric/integral_constant.hpp>  // Int<0>
#include <cute/util/type_traits.hpp>           // cute::declval, cute::void_t, etc

namespace cute
{

//
// C++20 <iterator> iterator_traits
//

namespace detail {
// Default reference type of an iterator
template <class T, class = void>
struct iter_ref { using type = decltype(*declval<T&>()); };
// Prefer to propagate ::reference
template <class T>
struct iter_ref<T,void_t<typename T::reference>> { using type = typename T::reference; };
} // end namespace detail

template <class T>
using iter_reference = detail::iter_ref<T>;
template <class T>
using iter_reference_t = typename iter_reference<T>::type;

namespace detail {
// Default element_type of an iterator
template <class T, class = void>
struct iter_e { using type = remove_reference_t<typename iter_ref<T>::type>; };
// Prefer to propagate ::element_type
template <class T>
struct iter_e<T,void_t<typename T::element_type>> { using type = typename T::element_type; };
} // end namespace detail

template <class T>
using iter_element = detail::iter_e<T>;
template <class T>
using iter_element_t = typename iter_element<T>::type;

namespace detail {
// Default value_type of an iterator
template <class T, class = void>
struct iter_v { using type = remove_cv_t<typename iter_e<T>::type>; };
// Prefer to propagate ::value_type
template <class T>
struct iter_v<T,void_t<typename T::value_type>> { using type = typename T::value_type; };
} // end namespace detail

template <class T>
using iter_value = detail::iter_v<T>;
template <class T>
using iter_value_t = typename iter_value<T>::type;

template <class Iterator>
struct iterator_traits {
  using reference    = iter_reference_t<Iterator>;
  using element_type = iter_element_t<Iterator>;
  using value_type   = iter_value_t<Iterator>;
};

//
// has_dereference to determine if a type is an iterator concept
//

namespace detail {
template <class T, class = void>
struct has_dereference : CUTE_STL_NAMESPACE::false_type {};
template <class T>
struct has_dereference<T, void_t<decltype(*declval<T&>())>> : CUTE_STL_NAMESPACE::true_type {};
} // end namespace detail

template <class T>
using has_dereference = detail::has_dereference<T>;

//
// raw_pointer_cast
//

template <class T>
CUTE_HOST_DEVICE constexpr
T*
raw_pointer_cast(T* ptr) {
  return ptr;
}

// The statically-known alignment of a dynamic pointer is unknown
template <class T>
CUTE_HOST_DEVICE constexpr
Int<0>
max_alignment(T*) {
  return {};
}

//
// A very simplified iterator adaptor.
// Derived classed may override methods, but be careful to reproduce interfaces exactly.
// Clients should never have an instance of this class. Do not write methods that take this as a param.
//

template <class Iterator, class DerivedType>
struct iter_adaptor
{
  using iterator     = Iterator;
  using reference    = typename iterator_traits<iterator>::reference;
  using element_type = typename iterator_traits<iterator>::element_type;
  using value_type   = typename iterator_traits<iterator>::value_type;

  iterator ptr_;

  CUTE_HOST_DEVICE constexpr
  iter_adaptor(iterator ptr = {}) : ptr_(ptr) {}

  CUTE_HOST_DEVICE constexpr
  reference operator*() const { return *ptr_; }

  template <class Index>
  CUTE_HOST_DEVICE constexpr
  reference operator[](Index const& i) const { return ptr_[i]; }

  template <class Index>
  CUTE_HOST_DEVICE constexpr
  DerivedType operator+(Index const& i) const { return {ptr_ + i}; }

  CUTE_HOST_DEVICE constexpr
  iterator get() const { return ptr_; }

  CUTE_HOST_DEVICE constexpr
  friend bool operator==(DerivedType const& x, DerivedType const& y) { return x.ptr_ == y.ptr_; }
  CUTE_HOST_DEVICE constexpr
  friend bool operator!=(DerivedType const& x, DerivedType const& y) { return x.ptr_ != y.ptr_; }
  CUTE_HOST_DEVICE constexpr
  friend bool operator< (DerivedType const& x, DerivedType const& y) { return x.ptr_ <  y.ptr_; }
  CUTE_HOST_DEVICE constexpr
  friend bool operator<=(DerivedType const& x, DerivedType const& y) { return x.ptr_ <= y.ptr_; }
  CUTE_HOST_DEVICE constexpr
  friend bool operator> (DerivedType const& x, DerivedType const& y) { return x.ptr_ >  y.ptr_; }
  CUTE_HOST_DEVICE constexpr
  friend bool operator>=(DerivedType const& x, DerivedType const& y) { return x.ptr_ >= y.ptr_; }
};

template <class I, class D>
CUTE_HOST_DEVICE constexpr
auto
raw_pointer_cast(iter_adaptor<I,D> const& x) {
  return raw_pointer_cast(x.ptr_);
}

template <class I, class D>
CUTE_HOST_DEVICE constexpr
auto
max_alignment(iter_adaptor<I,D> const& x) {
  return max_alignment(x.ptr_);
}

//
// counting iterator -- quick and dirty
//

template <class T = int>
struct counting_iterator
{
  using index_type = T;
  using value_type = T;
  using reference  = T;

  index_type n_;

  CUTE_HOST_DEVICE constexpr
  counting_iterator(index_type n = 0) : n_(n) {}

  CUTE_HOST_DEVICE constexpr
  index_type operator*() const { return n_; }

  CUTE_HOST_DEVICE constexpr
  index_type operator[](index_type i) const { return n_ + i; }

  CUTE_HOST_DEVICE constexpr
  counting_iterator operator+(index_type i) const { return {n_ + i}; }
  CUTE_HOST_DEVICE constexpr
  counting_iterator& operator++() { ++n_; return *this; }
  CUTE_HOST_DEVICE constexpr
  counting_iterator operator++(int) { counting_iterator ret = *this; ++n_; return ret; }

  CUTE_HOST_DEVICE constexpr
  friend bool operator==(counting_iterator const& x, counting_iterator const& y) { return x.n_ == y.n_; }
  CUTE_HOST_DEVICE constexpr
  friend bool operator!=(counting_iterator const& x, counting_iterator const& y) { return x.n_ != y.n_; }
  CUTE_HOST_DEVICE constexpr
  friend bool operator< (counting_iterator const& x, counting_iterator const& y) { return x.n_ <  y.n_; }
  CUTE_HOST_DEVICE constexpr
  friend bool operator<=(counting_iterator const& x, counting_iterator const& y) { return x.n_ <= y.n_; }
  CUTE_HOST_DEVICE constexpr
  friend bool operator> (counting_iterator const& x, counting_iterator const& y) { return x.n_ >  y.n_; }
  CUTE_HOST_DEVICE constexpr
  friend bool operator>=(counting_iterator const& x, counting_iterator const& y) { return x.n_ >= y.n_; }
};

template <class T>
CUTE_HOST_DEVICE constexpr
T
raw_pointer_cast(counting_iterator<T> const& x) {
  return x.n_;
}

//
// transform_iterator
//

template <class Fn, class Iter>
struct transform_iter
{
  using iterator     = Iter;
  // using reference    = typename iterator_traits<iterator>::reference;
  // using element_type = typename iterator_traits<iterator>::element_type;
  // using value_type   = typename iterator_traits<iterator>::value_type;

  Fn fn_;
  iterator ptr_;

  CUTE_HOST_DEVICE constexpr
  transform_iter(Fn fn, iterator ptr = {}) : fn_(fn), ptr_(ptr) {}

  CUTE_HOST_DEVICE constexpr
  decltype(auto) operator*() const { return fn_(*ptr_); }

  template <class Index>
  CUTE_HOST_DEVICE constexpr
  decltype(auto) operator[](Index const& i) const { return fn_(ptr_[i]); }

  template <class Index>
  CUTE_HOST_DEVICE constexpr
  auto operator+(Index const& i) const { return transform_iter<Fn, decltype(ptr_+i)>{fn_, ptr_+i}; }

  template <class IterY>
  CUTE_HOST_DEVICE constexpr
  friend bool operator==(transform_iter<Fn,Iter> const& x, transform_iter<Fn,IterY> const& y) { return x.ptr_ == y.ptr_; }
  template <class IterY>
  CUTE_HOST_DEVICE constexpr
  friend bool operator!=(transform_iter<Fn,Iter> const& x, transform_iter<Fn,IterY> const& y) { return x.ptr_ != y.ptr_; }
  template <class IterY>
  CUTE_HOST_DEVICE constexpr
  friend bool operator< (transform_iter<Fn,Iter> const& x, transform_iter<Fn,IterY> const& y) { return x.ptr_ <  y.ptr_; }
  template <class IterY>
  CUTE_HOST_DEVICE constexpr
  friend bool operator<=(transform_iter<Fn,Iter> const& x, transform_iter<Fn,IterY> const& y) { return x.ptr_ <= y.ptr_; }
  template <class IterY>
  CUTE_HOST_DEVICE constexpr
  friend bool operator> (transform_iter<Fn,Iter> const& x, transform_iter<Fn,IterY> const& y) { return x.ptr_ >  y.ptr_; }
  template <class IterY>
  CUTE_HOST_DEVICE constexpr
  friend bool operator>=(transform_iter<Fn,Iter> const& x, transform_iter<Fn,IterY> const& y) { return x.ptr_ >= y.ptr_; }
};

template <class Fn, class Iterator>
CUTE_HOST_DEVICE constexpr
auto
make_transform_iter(Fn const& fn, Iterator const& ptr)
{
  return transform_iter<Fn,Iterator>(fn,ptr);
}

//
// Display utilities
//

template <class T>
CUTE_HOST_DEVICE void print(T const* const ptr)
{
  printf("ptr["); print(sizeof_bits<T>::value); printf("b](%p)", ptr);
}

template <class T>
CUTE_HOST_DEVICE void print(counting_iterator<T> ptr)
{
  printf("counting_iter("); print(ptr.n_); printf(")");
}

template <class Fn, class Iterator>
CUTE_HOST_DEVICE void print(transform_iter<Fn,Iterator> ptr)
{
  printf("trans_"); print(ptr.ptr_);
}

#if !defined(__CUDACC_RTC__)
template <class T>
CUTE_HOST std::ostream& operator<<(std::ostream& os, counting_iterator<T> ptr)
{
  return os << "counting_iter(" << ptr.n_ << ")";
}

template <class Fn, class Iterator>
CUTE_HOST std::ostream& operator<<(std::ostream& os, transform_iter<Fn,Iterator> ptr)
{
  return os << "trans_" << ptr.ptr_;
}
#endif // !defined(__CUDACC_RTC__)

} // end namespace cute
