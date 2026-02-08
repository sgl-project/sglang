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
#include <cute/pointer_base.hpp>               // cute::iter_adaptor
#include <cute/pointer_sparse.hpp>
#include <cute/numeric/integral_constant.hpp>  // cute::true_type, cute::false_type
#include <cute/numeric/numeric_types.hpp>      // sizeof_bits
#include <cute/container/array_subbyte.hpp>    // cute::subbyte_iterator

namespace cute
{

//
// recast_ptr<T> -- Create an iterator over values of type T.
// For most types this will simply be T*, but certain types require more care.
// Subbyte Types: uint2_t, uint4_t, etc
//   Requires construction of a subbyte_iterator<T> in order to properly
//   resolve each element in byte-addressed memory.
// Sparse Types: sparse_elem<int S, class T>
//   A type that holds one physical element meant to represent S number of logical elements.
//   Requires construction of a sparse_ptr that emulates access to the S logical elements.
//

template <class NewT_, class T>
CUTE_HOST_DEVICE constexpr
auto
recast_ptr(T* ptr)
{
  using NewT = copy_cv_t<T, NewT_>;

  if constexpr (is_sparse<NewT>::value) {
    constexpr int sparsity = NewT::sparsity;
    NewT* p = reinterpret_cast<NewT*>(ptr);
    return make_sparse_ptr<sparsity>(p);
  } else
  if constexpr (cute::is_subbyte_v<NewT>) {
    return subbyte_iterator<NewT>(ptr);
  } else {
    return reinterpret_cast<NewT*>(ptr);
  }
  CUTE_GCC_UNREACHABLE;
}

// Disambiguate nullptr
template <class NewT>
CUTE_HOST_DEVICE constexpr
auto
recast_ptr(decltype(nullptr)) {   // nullptr_t
  return recast_ptr<NewT>(static_cast<NewT*>(nullptr));
}

//
// gmem_ptr
//

template <class P>
struct gmem_ptr : iter_adaptor<P, gmem_ptr<P>> {
  using iter_adaptor<P, gmem_ptr<P>>::iter_adaptor;
};

template <class T, class = void>
struct is_gmem : false_type {};
template <class P>                     // Found the gmem
struct is_gmem<gmem_ptr<P>> : true_type {};
template <class P>                     // Recurse on ::iterator, if possible
struct is_gmem<P, void_t<typename P::iterator>> : is_gmem<typename P::iterator> {};
template <class P>
constexpr bool is_gmem_v = is_gmem<P>::value;

// Idempotent gmem tag on an iterator
template <class Iterator>
CUTE_HOST_DEVICE constexpr
auto
make_gmem_ptr(Iterator iter) {
  if constexpr (is_gmem<Iterator>::value) {
    return iter;
  } else {
    return gmem_ptr<Iterator>{iter};
  }
  CUTE_GCC_UNREACHABLE;
}

// Explicitly typed construction from a raw pointer
template <class T>
CUTE_HOST_DEVICE constexpr
auto
make_gmem_ptr(void* ptr) {
  return make_gmem_ptr(recast_ptr<T>(ptr));
}

// Explicitly typed construction from a raw pointer
template <class T>
CUTE_HOST_DEVICE constexpr
auto
make_gmem_ptr(void const* ptr) {
  return make_gmem_ptr(recast_ptr<T const>(ptr));
}

// nullptr_t overload for make_gmem_ptr<float>(nullptr) disambiguation
template <class T>
CUTE_HOST_DEVICE constexpr
auto
make_gmem_ptr(decltype(nullptr)) { // nullptr_t
  return make_gmem_ptr(recast_ptr<T>(nullptr));
}

// The gmem tag is invariant over type-recast
template <class NewT, class P>
CUTE_HOST_DEVICE constexpr
auto
recast_ptr(gmem_ptr<P> const& ptr) {
  return make_gmem_ptr(recast_ptr<NewT>(ptr.get()));
}

//
// smem_ptr
//

template <class P>
struct smem_ptr : iter_adaptor<P, smem_ptr<P>> {
  using iter_adaptor<P, smem_ptr<P>>::iter_adaptor;
};

template <class T, class = void>
struct is_smem : false_type {};
template <class P>                     // Found the smem
struct is_smem<smem_ptr<P>> : true_type {};
template <class P>                     // Recurse on ::iterator, if possible
struct is_smem<P, void_t<typename P::iterator>> : is_smem<typename P::iterator> {};
template <class P>
constexpr bool is_smem_v = is_smem<P>::value;

// Idempotent smem tag on an iterator
template <class Iterator>
CUTE_HOST_DEVICE constexpr
auto
make_smem_ptr(Iterator iter) {
  if constexpr (is_smem<Iterator>::value) {
    return iter;
  } else {
    return smem_ptr<Iterator>{iter};
  }
  CUTE_GCC_UNREACHABLE;
}

// Make a smem swizzle pointer, common operation
template <class Iterator, class Swizzle>
CUTE_HOST_DEVICE constexpr
auto
make_smem_ptr(Iterator ptr, Swizzle sw)
{
  return make_swizzle_ptr(make_smem_ptr(ptr), sw);
}

// Explicitly typed construction from a raw pointer
template <class T>
CUTE_HOST_DEVICE constexpr
auto
make_smem_ptr(void* ptr) {
  return make_smem_ptr(recast_ptr<T>(ptr));
}

// Explicitly typed construction from a raw pointer
template <class T>
CUTE_HOST_DEVICE constexpr
auto
make_smem_ptr(void const* ptr) {
  return make_smem_ptr(recast_ptr<T const>(ptr));
}

// nullptr_t overload for make_smem_ptr<float>(nullptr) disambiguation
template <class T>
CUTE_HOST_DEVICE constexpr
auto
make_smem_ptr(decltype(nullptr)) { // nullptr_t
  return make_smem_ptr(recast_ptr<T>(nullptr));
}

// The smem tag is invariant over type-recast
template <class NewT, class P>
CUTE_HOST_DEVICE constexpr
auto
recast_ptr(smem_ptr<P> const& ptr) {
  return make_smem_ptr(recast_ptr<NewT>(ptr.get()));
}

//
// rmem_ptr
//

template <class P>
struct rmem_ptr : iter_adaptor<P, rmem_ptr<P>> {
  using iter_adaptor<P, rmem_ptr<P>>::iter_adaptor;
};

// Anything that is not gmem or smem is rmem
template <class T, class = void>
struct is_rmem : bool_constant<not (is_gmem<T>::value || is_smem<T>::value)> {};
template <class P>
struct is_rmem<rmem_ptr<P>> : true_type {};
template <class P>
constexpr bool is_rmem_v = is_rmem<P>::value;

// Idempotent rmem tag on an iterator
template <class Iterator>
CUTE_HOST_DEVICE constexpr
auto
make_rmem_ptr(Iterator iter) {
  if constexpr (is_rmem<Iterator>::value) {
    return iter;
  } else {
    return rmem_ptr<Iterator>{iter};
  }
  CUTE_GCC_UNREACHABLE;
}

// Explicitly typed construction from a raw pointer
template <class T>
CUTE_HOST_DEVICE constexpr
auto
make_rmem_ptr(void* ptr) {
  return make_rmem_ptr(recast_ptr<T>(ptr));
}

// Explicitly typed construction from a raw pointer
template <class T>
CUTE_HOST_DEVICE constexpr
auto
make_rmem_ptr(void const* ptr) {
  return make_rmem_ptr(recast_ptr<T const>(ptr));
}

// The rmem tag is invariant over type-recast
template <class NewT, class P>
CUTE_HOST_DEVICE constexpr
auto
recast_ptr(rmem_ptr<P> const& ptr) {
  return make_rmem_ptr(recast_ptr<NewT>(ptr.get()));
}


//
// tmem_ptr -- a typed, word-addressed, non-dereferencable "pointer"
//

template <class T>
struct tmem_ptr
{
  using value_type   = remove_cv_t<T>;
  using element_type = T;
  using reference    = T;

  // Right-shift value for the offset scaling -- TMEM uses word-addressing
  static constexpr int32_t OffsetShift = log_2(trait_ratio(sizeof_bits<uint32_t>{}, sizeof_bits<T>{}));

  CUTE_HOST_DEVICE constexpr
  tmem_ptr(uint32_t addr = 0) : addr_(addr) {}

  CUTE_HOST_DEVICE constexpr
  uint32_t const& get() const {
    return addr_;
  }
  CUTE_HOST_DEVICE constexpr
  uint32_t& get() {
    return addr_;
  }

  template <class T_ = T>
  CUTE_HOST_DEVICE constexpr
  value_type operator*() const {
    static_assert(dependent_false<T_>, "Attempting to dereference a tmem_ptr, want raw_pointer_cast() for address instead?");
    return value_type{};
  }

  CUTE_HOST_DEVICE constexpr
  reference operator[](uint32_t const& i) const { return *(*this + i); }

  CUTE_HOST_DEVICE constexpr
  tmem_ptr operator+(uint32_t const& i) const {
    //return {addr_ + shiftr(i, OffsetShift)};  // Shift the offset for word-addressing
    return {addr_ + rotr(i, OffsetShift)};    // Rotate the offset to keep subword indices in the unused high 8bits for debug
  }

  // TMEM "Address" with active mask 0x007F.01FF
  // The upper 16 bits, the 0x007F portion, refers to the 128  DP lanes
  // The lower 16 bits, the 0x01FF portion, refers to the 512 COL lanes
  union {
    uint32_t addr_;
    struct {
      uint16_t col_;
      uint8_t  dp_;
      uint8_t  idx_;  // Hijack the top 8bits for the sub-word idx to avoid an extra reg.
                      // Assert this is 0 on every access?
    };
  };
};

template <class T, class = void>
struct is_tmem : false_type {};
template <class T>                     // Found the tmem
struct is_tmem<tmem_ptr<T>> : true_type {};
template <class P>                     // Recurse on ::iterator, if possible
struct is_tmem<P, void_t<typename P::iterator>> : is_tmem<typename P::iterator> {};
template <class P>
constexpr bool is_tmem_v = is_tmem<P>::value;

template <class T>
CUTE_HOST_DEVICE constexpr
tmem_ptr<T>
make_tmem_ptr(uint32_t addr = 0) {
  return tmem_ptr<T>(addr);
}

template <class T>
CUTE_HOST_DEVICE constexpr
uint32_t
raw_pointer_cast(tmem_ptr<T> const& ptr) {
  return ptr.get();
}

// TMEM accounts for subword/superword elements already due to the offset shift based on sizeof_bits
//   Thus, this is a trivial recast equivalent to reinterpret_cast<NewT*>
template <class NewT, class T>
CUTE_HOST_DEVICE constexpr
auto
recast_ptr(tmem_ptr<T> const& ptr) {
  return tmem_ptr<NewT>{ptr.addr_};
}


//
// Display utilities
//

template <class T>
CUTE_HOST_DEVICE void print(gmem_ptr<T> ptr)
{
  printf("gmem_"); print(ptr.get());
}

template <class T>
CUTE_HOST_DEVICE void print(smem_ptr<T> ptr)
{
  printf("smem_"); print(ptr.get());
}

template <class T>
CUTE_HOST_DEVICE void print(rmem_ptr<T> ptr)
{
  printf("rmem_"); print(ptr.get());
}


template <class T>
CUTE_HOST_DEVICE void print(tmem_ptr<T> ptr)
{
  printf("tmem_["); print(sizeof_bits<T>::value); printf("b](0x%04x.%04x)", ptr.addr_ >> 16, ptr.addr_ & 0xFFFF);
}


#if !defined(__CUDACC_RTC__)
template <class T>
CUTE_HOST std::ostream& operator<<(std::ostream& os, gmem_ptr<T> ptr)
{
  return os << "gmem_[" << int(sizeof_bits<iter_value_t<T>>::value) << "b]";
}

template <class T>
CUTE_HOST std::ostream& operator<<(std::ostream& os, smem_ptr<T> ptr)
{
  return os << "smem_[" << int(sizeof_bits<iter_value_t<T>>::value) << "b]";
}

template <class T>
CUTE_HOST std::ostream& operator<<(std::ostream& os, rmem_ptr<T> ptr)
{
  return os << "rmem_[" << int(sizeof_bits<iter_value_t<T>>::value) << "b]";
}


template <class T>
CUTE_HOST std::ostream& operator<<(std::ostream& os, tmem_ptr<T> ptr)
{
  return os << "tmem_[" << int(sizeof_bits<T>::value) << "b](" << ptr.addr_ << ")";
}

#endif // !defined(__CUDACC_RTC__)

} // end namespace cute
