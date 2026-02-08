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

#pragma once

/**
 * \file
 * \brief C++ features that may be otherwise unimplemented for CUDA device functions.
 *
 * This file has three components:
 *
 *   (1) Macros:
 *       - Empty macro defines for C++ keywords not supported by the current
 *         version of C++. These simply allow compilation to proceed (but do
 *         not provide the added semantics).
 *           - \p noexcept
 *           - \p constexpr
 *           - \p nullptr
 *           - \p static_assert
 *
 *       - Macro functions that we need in constant expressions because the
 *         C++ equivalents require constexpr compiler support.  These are
 *         prefixed with \p __NV_STD_*
 *           - \p __NV_STD_MAX
 *           - \p __NV_STD_MIN
 *
 *   (2) Re-implementations of STL functions and types:
 *       - C++ features that need the \p __device__ annotation.  These are
 *         placed into the \p platform namespace.
 *           - \p abs
 *           - \p plus
 *           - \p less
 *           - \p greater
 *           - \p min
 *           - \p max
 *           - \p methods on std::pair (==, !=, <, <=, >, >=, and make_pair())
 *
 *   (3) Stop-gap implementations of unsupported STL functions and types:
 *       - STL functions and types defined by C++ 11/14/17/etc. that are not
 *         provided by the current version of C++. These are placed into the
 *         \p platform namespace
 *           - \p integral_constant
 *           - \p nullptr_t
 *           - \p true_type
 *           - \p false_type
 *           - \p bool_constant
 *           - \p enable_if
 *           - \p conditional
 *           - \p is_same
 *           - \p is_base_of
 *           - \p remove_const
 *           - \p remove_volatile
 *           - \p remove_cv
 *           - \p is_volatile
 *           - \p is_pointer
 *           - \p is_void
 *           - \p is_integral
 *           - \p is_floating_point
 *           - \p is_arithmetic
 *           - \p is_fundamental
 *           - \p is_trivially_copyable
 *           - \p alignment_of
 *           - \p aligned_storage
 *
 * The idea is that, as we drop support for older compilers, we can simply #define
 * the \p __NV_STD_XYZ macros and \p platform namespace to alias their C++
 * counterparts (or trivially find-and-replace their occurrences in code text).
 */

//-----------------------------------------------------------------------------
// Dependencies
//-----------------------------------------------------------------------------
#include <cutlass/cutlass.h>
#if defined(__CUDACC_RTC__)
#include CUDA_STD_HEADER(type_traits)
#include CUDA_STD_HEADER(utility)
#include CUDA_STD_HEADER(cstddef)
#include CUDA_STD_HEADER(cstdint)
#include CUDA_STD_HEADER(limits)
#else
#include <type_traits>
#include <utility>
#include <cstddef>
#include <cstdint>
#include <limits>
#endif

#if !defined(__CUDACC_RTC__)
//-----------------------------------------------------------------------------
// Include STL files that platform provides functionality for
//-----------------------------------------------------------------------------

#include <algorithm>   // Minimum/maximum operations
#include <cstddef>     // nullptr_t
#include <functional>  // Arithmetic operations
#include <utility>     // For methods on std::pair
#include <limits>      // float_round_style, float_denorm_style
#if (!defined(_MSC_VER) && (__cplusplus >= 201103L)) || (defined(_MSC_VER) && (_MS_VER >= 1500))
#include <type_traits>  // For integral constants, conditional metaprogramming, and type traits
#endif

#include <vector_types.h>

#endif

//-----------------------------------------------------------------------------
// OS
//-----------------------------------------------------------------------------
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#define CUTLASS_OS_WINDOWS
#endif

#if defined(__clang__) && defined(__CUDA__)
#define CUTLASS_CLANG_CUDA 1
#endif

/******************************************************************************
 * Macros
 ******************************************************************************/
/// std
#if !defined(CUTLASS_STL_NAMESPACE)
#if defined(__CUDACC_RTC__)
#define CUTLASS_STL_NAMESPACE cuda::std
#else
#define CUTLASS_STL_NAMESPACE std
#endif
#endif

/// builtin_unreachable
#if !defined(CUTLASS_GCC_UNREACHABLE)
#  if defined(__GNUC__)
#    define CUTLASS_GCC_UNREACHABLE __builtin_unreachable()
#  else
#    define CUTLASS_GCC_UNREACHABLE
#  endif
#endif

//-----------------------------------------------------------------------------
// Keywords
//-----------------------------------------------------------------------------

/// noexcept, constexpr
#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1900))
#ifndef noexcept
#define noexcept
#endif
#ifndef constexpr
#define constexpr
#endif
#endif

/// nullptr
#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1310))
#ifndef nullptr
#define nullptr 0
#endif
#endif

/// static_assert
#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1600))
#ifndef static_assert
#define __platform_cat_(a, b) a##b
#define __platform_cat(a, b) __platform_cat_(a, b)
#define static_assert(__e, __m) typedef int __platform_cat(AsSeRt, __LINE__)[(__e) ? 1 : -1]
#endif
#endif

//-----------------------------------------------------------------------------
// Functions
//-----------------------------------------------------------------------------

/// Select maximum(a, b)
#ifndef __NV_STD_MAX
#define __NV_STD_MAX(a, b) (((b) > (a)) ? (b) : (a))
#endif

/// Select minimum(a, b)
#ifndef __NV_STD_MIN
#define __NV_STD_MIN(a, b) (((b) < (a)) ? (b) : (a))
#endif

/******************************************************************************
 * Re-implementations
 ******************************************************************************/
namespace cutlass {
namespace platform {

//-----------------------------------------------------------------------------
// Abs operations <algorithm>
//-----------------------------------------------------------------------------

#if defined(__CUDACC_RTC__)
/// std::abs
CUTLASS_HOST_DEVICE constexpr int abs(int a) {
    return (a < 0) ? -a : a;
}
CUTLASS_HOST_DEVICE constexpr long long abs(long long a) {
    return (a < 0) ? -a : a;
}
#else
using std::abs;
#endif

//-----------------------------------------------------------------------------
// Minimum/maximum operations <algorithm>
//-----------------------------------------------------------------------------

/// std::min
template <typename T>
CUTLASS_HOST_DEVICE constexpr const T& min(const T& a, const T& b) {
  return (b < a) ? b : a;
}

/// std::max
template <typename T>
CUTLASS_HOST_DEVICE constexpr const T& max(const T& a, const T& b) {
  return (a < b) ? b : a;
}

#if !defined(__CUDACC_RTC__)
//-----------------------------------------------------------------------------
// Methods on std::pair
//-----------------------------------------------------------------------------

using std::pair;

template <class T1, class T2>
CUTLASS_HOST_DEVICE constexpr bool operator==(const pair<T1, T2>& lhs, const pair<T1, T2>& rhs) {
  return (lhs.first == rhs.first) && (lhs.second == rhs.second);
}

template <class T1, class T2>
CUTLASS_HOST_DEVICE constexpr bool operator!=(const pair<T1, T2>& lhs, const pair<T1, T2>& rhs) {
  return (lhs.first != rhs.first) && (lhs.second != rhs.second);
}

template <class T1, class T2>
CUTLASS_HOST_DEVICE constexpr bool operator<(const pair<T1, T2>& lhs, const pair<T1, T2>& rhs) {
  return (lhs.first < rhs.first) ? true : (rhs.first < lhs.first) ? false
                                                                  : (lhs.second < rhs.second);
}

template <class T1, class T2>
CUTLASS_HOST_DEVICE constexpr bool operator<=(const pair<T1, T2>& lhs, const pair<T1, T2>& rhs) {
  return !(rhs < lhs);
}

template <class T1, class T2>
CUTLASS_HOST_DEVICE constexpr bool operator>(const pair<T1, T2>& lhs, const pair<T1, T2>& rhs) {
  return (rhs < lhs);
}

template <class T1, class T2>
CUTLASS_HOST_DEVICE constexpr bool operator>=(const pair<T1, T2>& lhs, const pair<T1, T2>& rhs) {
  return !(lhs < rhs);
}

template <class T1, class T2>
CUTLASS_HOST_DEVICE std::pair<T1, T2> make_pair(T1 t, T2 u) {
  std::pair<T1, T2> retval;
  retval.first = t;
  retval.second = u;
  return retval;
}
#endif

}  // namespace platform

/******************************************************************************
 * Implementations of C++ 11/14/17/... STL features
 ******************************************************************************/

namespace platform {

//-----------------------------------------------------------------------------
// Integral constant helper types <type_traits>
//-----------------------------------------------------------------------------

#if defined(__CUDACC_RTC__) || (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1500))

#else

using std::pair;

#endif

using CUTLASS_STL_NAMESPACE::integral_constant;
using CUTLASS_STL_NAMESPACE::bool_constant;
using CUTLASS_STL_NAMESPACE::true_type;
using CUTLASS_STL_NAMESPACE::false_type;

#if defined(__CUDACC_RTC__) || (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1700))

/// std::nullptr_t
struct nullptr_t {};

#else

using std::nullptr_t;

#endif

//-----------------------------------------------------------------------------
// Conditional metaprogramming <type_traits>
//-----------------------------------------------------------------------------

using CUTLASS_STL_NAMESPACE::conditional;
using CUTLASS_STL_NAMESPACE::conditional_t;
using CUTLASS_STL_NAMESPACE::enable_if;
using CUTLASS_STL_NAMESPACE::enable_if_t;
using CUTLASS_STL_NAMESPACE::void_t;

//-----------------------------------------------------------------------------
// Const/volatility specifiers <type_traits>
//-----------------------------------------------------------------------------

using CUTLASS_STL_NAMESPACE::remove_const;
using CUTLASS_STL_NAMESPACE::remove_const_t;
using CUTLASS_STL_NAMESPACE::remove_cv;
using CUTLASS_STL_NAMESPACE::remove_cv_t;
using CUTLASS_STL_NAMESPACE::remove_reference;
using CUTLASS_STL_NAMESPACE::remove_reference_t;
using CUTLASS_STL_NAMESPACE::remove_volatile;
using CUTLASS_STL_NAMESPACE::remove_volatile_t;

// remove_cvref and remove_cvref_t are C++20 features,
// but CUTLASS finds them useful enough to back-port.
#if defined(__cpp_lib_remove_cvref)

using CUTLASS_STL_NAMESPACE::remove_cvref;
using CUTLASS_STL_NAMESPACE::remove_cvref_t;

#else

template <class T>
struct remove_cvref {
  using type = remove_cv_t<remove_reference_t<T>>;
};

template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;

#endif

//-----------------------------------------------------------------------------
// Type relationships <type_traits>
//-----------------------------------------------------------------------------

using CUTLASS_STL_NAMESPACE::is_same;
using CUTLASS_STL_NAMESPACE::is_same_v;

#if defined(__CUDACC_RTC__) || (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1500))

/// Helper for std::is_base_of
template <typename BaseT, typename DerivedT>
struct is_base_of_helper {
  typedef char (&yes)[1];
  typedef char (&no)[2];

  template <typename B, typename D>
  struct dummy {
    CUTLASS_HOST_DEVICE operator B*() const;
    CUTLASS_HOST_DEVICE operator D*();
  };

  template <typename T>
  CUTLASS_HOST_DEVICE static yes check(DerivedT*, T);

  CUTLASS_HOST_DEVICE static no check(BaseT*, int);

  static const bool value = sizeof(check(dummy<BaseT, DerivedT>(), int())) == sizeof(yes);
};

/// std::is_base_of
template <typename BaseT, typename DerivedT>
struct is_base_of
    : integral_constant<bool,
                        (is_base_of_helper<typename remove_cv<BaseT>::type,
                                           typename remove_cv<DerivedT>::type>::value) ||
                            (is_same<typename remove_cv<BaseT>::type,
                                     typename remove_cv<DerivedT>::type>::value)> {};

#else

using std::is_base_of;

#endif

//-----------------------------------------------------------------------------
// Type properties <type_traits>
//-----------------------------------------------------------------------------

using CUTLASS_STL_NAMESPACE::is_arithmetic;
using CUTLASS_STL_NAMESPACE::is_arithmetic_v;
using CUTLASS_STL_NAMESPACE::is_void;
using CUTLASS_STL_NAMESPACE::is_void_v;

#if defined(__CUDACC_RTC__) || (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1500))

/// std::is_volatile
template <typename T>
struct is_volatile : false_type {};
template <typename T>
struct is_volatile<volatile T> : true_type {};

/// Helper for std::is_pointer (false specialization)
template <typename T>
struct is_pointer_helper : false_type {};

/// Helper for std::is_pointer (true specialization)
template <typename T>
struct is_pointer_helper<T*> : true_type {};

/// std::is_pointer
template <typename T>
struct is_pointer : is_pointer_helper<typename remove_cv<T>::type> {};

/// std::is_integral
template <typename T>
struct is_integral : false_type {};
template <>
struct is_integral<char> : true_type {};
template <>
struct is_integral<signed char> : true_type {};
template <>
struct is_integral<unsigned char> : true_type {};
template <>
struct is_integral<short> : true_type {};
template <>
struct is_integral<unsigned short> : true_type {};
template <>
struct is_integral<int> : true_type {};
template <>
struct is_integral<unsigned int> : true_type {};
template <>
struct is_integral<long> : true_type {};
template <>
struct is_integral<unsigned long> : true_type {};
template <>
struct is_integral<long long> : true_type {};
template <>
struct is_integral<unsigned long long> : true_type {};
template <typename T>
struct is_integral<volatile T> : is_integral<T> {};
template <typename T>
struct is_integral<const T> : is_integral<T> {};
template <typename T>
struct is_integral<const volatile T> : is_integral<T> {};

/// std::is_floating_point
template <typename T>
struct is_floating_point
    : integral_constant<bool,
                        (is_same<float, typename remove_cv<T>::type>::value ||
                         is_same<double, typename remove_cv<T>::type>::value)> {};

/// std::is_fundamental
template <typename T>
struct is_fundamental
    : integral_constant<bool,
                        (is_arithmetic<T>::value || is_void<T>::value ||
                         is_same<nullptr_t, typename remove_cv<T>::type>::value)> {};

#else

using std::is_volatile;
using std::is_pointer;
using std::is_integral;
using std::is_floating_point;
using std::is_fundamental;

#endif

#if defined(__CUDACC_RTC__) || (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1800)) || \
    (defined(__GNUG__) && (__GNUC__ < 5))

/**
     * std::is_trivially_copyable
     *
     * This implementation only evaluates true if T is fundamental or pointer
     *
     * Without help from partial template specializations provided by the user for
     * a specific class or struct, this trait will never report that the specified
     * class or struct  is trivially-copyable ; this is always safe,
     * if possibly sub-optimal.
     */
template <typename T>
struct is_trivially_copyable
    : integral_constant<bool, (is_fundamental<T>::value || is_pointer<T>::value)> {};

#else

using std::is_trivially_copyable;

#endif

#if (CUTLASS_CXX17_OR_LATER)

/// std::is_unsigned_v
using CUTLASS_STL_NAMESPACE::is_integral_v;
/// std::is_unsigned_v
using CUTLASS_STL_NAMESPACE::is_unsigned_v;

#endif

//-----------------------------------------------------------------------------
// <utility>
//-----------------------------------------------------------------------------

using CUTLASS_STL_NAMESPACE::declval;

//-----------------------------------------------------------------------------
// bit_cast <bit>
//-----------------------------------------------------------------------------

template< class To, class From >
constexpr To CUTLASS_HOST_DEVICE bit_cast(const From& from ) noexcept;

template <class To, class From>
constexpr To CUTLASS_HOST_DEVICE bit_cast(const From& src) noexcept
{
  static_assert(sizeof(To) == sizeof(From), "sizes must match");
  return reinterpret_cast<To const &>(src);
}

//-----------------------------------------------------------------------------
// Convertable
//-----------------------------------------------------------------------------
using CUTLASS_STL_NAMESPACE::is_convertible;
using CUTLASS_STL_NAMESPACE::is_convertible_v;

//-----------------------------------------------------------------------------
// Alignment and layout utilities
//-----------------------------------------------------------------------------

#if defined(__CUDACC_RTC__) || (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1500))

/// std::alignment_of
template <typename value_t>
struct alignment_of {
  struct pad {
    value_t val;
    char byte;
  };

  enum { value = sizeof(pad) - sizeof(value_t) };
};

#else

template <typename value_t>
struct alignment_of : std::alignment_of<value_t> {};

#endif

/* 16B specializations where 32-bit Win32 host compiler disagrees with device compiler */
template <>
struct alignment_of<int4> {
  enum { value = 16 };
};
template <>
struct alignment_of<uint4> {
  enum { value = 16 };
};
template <>
struct alignment_of<float4> {
  enum { value = 16 };
};
template <>
struct alignment_of<longlong2> {
  enum { value = 16 };
};
template <>
struct alignment_of<ulonglong2> {
  enum { value = 16 };
};
template <>
struct alignment_of<double2> {
  enum { value = 16 };
};

#if !defined(CUDA_VECTOR_TYPE_ALIGNMENT_16_32_ENABLED)
#define CUDA_VECTOR_TYPE_ALIGNMENT_16_32_ENABLED (__CUDACC_VER_MAJOR__ >= 13)
#endif

#if (CUDA_VECTOR_TYPE_ALIGNMENT_16_32_ENABLED)
template <>
struct alignment_of<long4_16a> {
  enum { value = 16 };
};
template <>
struct alignment_of<ulong4_16a> {
  enum { value = 16 };
};
template <>
struct alignment_of<longlong4_16a> {
  enum { value = 16 };
};
template <>
struct alignment_of<ulonglong4_16a> {
  enum { value = 16 };
};
template <>
struct alignment_of<double4_16a> {
  enum { value = 16 };
};
template <>
struct alignment_of<long4_32a> {
  enum { value = 32 };
};
template <>
struct alignment_of<ulong4_32a> {
  enum { value = 32 };
};
template <>
struct alignment_of<longlong4_32a> {
  enum { value = 32 };
};
template <>
struct alignment_of<ulonglong4_32a> {
  enum { value = 32 };
};
template <>
struct alignment_of<double4_32a> {
  enum { value = 32 };
};
#else
template <>
struct alignment_of<long4> {
  enum { value = 16 };
};
template <>
struct alignment_of<ulong4> {
  enum { value = 16 };
};
template <>
struct alignment_of<longlong4> {
  enum { value = 16 };
};
template <>
struct alignment_of<ulonglong4> {
  enum { value = 16 };
};
template <>
struct alignment_of<double4> {
  enum { value = 16 };
};

#endif

// Specializations for volatile/const qualified types
template <typename value_t>
struct alignment_of<volatile value_t> : alignment_of<value_t> {};
template <typename value_t>
struct alignment_of<const value_t> : alignment_of<value_t> {};
template <typename value_t>
struct alignment_of<const volatile value_t> : alignment_of<value_t> {};

#if defined(__CUDACC_RTC__) || (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1800))

template <size_t Align>
struct aligned_chunk;
template <>
struct __align__(1) aligned_chunk<1> {
  uint8_t buff;
};
template <>
struct __align__(2) aligned_chunk<2> {
  uint16_t buff;
};
template <>
struct __align__(4) aligned_chunk<4> {
  uint32_t buff;
};
template <>
struct __align__(8) aligned_chunk<8> {
  uint32_t buff[2];
};
template <>
struct __align__(16) aligned_chunk<16> {
  uint32_t buff[4];
};
template <>
struct __align__(32) aligned_chunk<32> {
  uint32_t buff[8];
};
template <>
struct __align__(64) aligned_chunk<64> {
  uint32_t buff[16];
};
template <>
struct __align__(128) aligned_chunk<128> {
  uint32_t buff[32];
};
template <>
struct __align__(256) aligned_chunk<256> {
  uint32_t buff[64];
};
template <>
struct __align__(512) aligned_chunk<512> {
  uint32_t buff[128];
};
template <>
struct __align__(1024) aligned_chunk<1024> {
  uint32_t buff[256];
};
template <>
struct __align__(2048) aligned_chunk<2048> {
  uint32_t buff[512];
};
template <>
struct __align__(4096) aligned_chunk<4096> {
  uint32_t buff[1024];
};

/// std::aligned_storage
template <size_t Len, size_t Align>
struct aligned_storage {
  typedef aligned_chunk<Align> type[Len / sizeof(aligned_chunk<Align>)];
};

#else

using std::aligned_storage;

#endif

#if !defined(__CUDACC_RTC__)
/// Default deleter
template <typename T>
struct default_delete {
  void operator()(T* ptr) const { delete ptr; }
};

/// Partial specialization for deleting array types
template <typename T>
struct default_delete<T[]> {
  void operator()(T* ptr) const { delete[] ptr; }
};

/// std::unique_ptr
template <class T, class Deleter = default_delete<T> >
class unique_ptr {
 public:
  typedef T* pointer;
  typedef T element_type;
  typedef Deleter deleter_type;

 private:
  /// Pointer to memory
  pointer _ptr;

  /// Deleter
  deleter_type _deleter;

 public:
  unique_ptr() : _ptr(nullptr) {}
  unique_ptr(pointer p) : _ptr(p) {}

  ~unique_ptr() {
    if (_ptr) {
      _deleter(_ptr);
    }
  }
  /// Returns a pointer to the managed object or nullptr if no object is owned.
  pointer get() const noexcept { return _ptr; }

  /// Releases ownership of the managed object, if any
  pointer release() noexcept {
    pointer p(_ptr);
    _ptr = nullptr;
    return p;
  }

  /// Replaces the managed object, deleting the old object.
  void reset(pointer p = pointer()) noexcept {
    pointer old_ptr = _ptr;
    _ptr = p;
    if (old_ptr != nullptr) {
      get_deleter()(old_ptr);
    }
  }

  /// Swaps the managed objects with *this and another unique_ptr
  void swap(unique_ptr& other) noexcept { std::swap(_ptr, other._ptr); }

  /// Returns the deleter object
  Deleter& get_deleter() noexcept { return _deleter; }

  /// Returns the deleter object
  Deleter const& get_deleter() const noexcept { return _deleter; }

  /// Checks whether an object is owned
  operator bool() const noexcept { return _ptr != nullptr; }

  /// Dereferences the unique_ptr
  T& operator*() const { return *_ptr; }

  /// Returns a pointer to the managed object
  pointer operator->() const noexcept { return _ptr; }

  /// Array access to managed object
  T& operator[](size_t i) const { return _ptr[i]; }
};

/// Specializes the swap algorithm
template <typename T, typename Deleter>
void swap(unique_ptr<T, Deleter>& lhs, unique_ptr<T, Deleter>& rhs) noexcept {
  lhs.swap(rhs);
}
#endif

/// std::numeric_limits
template <class T>
struct numeric_limits;

template <>
struct numeric_limits<int32_t> {
  CUTLASS_HOST_DEVICE
  static constexpr int32_t lowest() noexcept { return -2147483647 - 1;}
  CUTLASS_HOST_DEVICE
  static constexpr int32_t max() noexcept { return 2147483647;}
  static constexpr bool is_integer = true;
  static constexpr bool has_infinity = false;
};

template <>
struct numeric_limits<int16_t> {
  CUTLASS_HOST_DEVICE
  static constexpr int16_t lowest() noexcept { return -32768;}
  CUTLASS_HOST_DEVICE
  static constexpr int16_t max() noexcept { return 32767;}
  static constexpr bool is_integer = true;
  static constexpr bool has_infinity = false;
};

template <>
struct numeric_limits<int8_t> {
  CUTLASS_HOST_DEVICE
  static constexpr int8_t lowest() noexcept { return -128;}
  CUTLASS_HOST_DEVICE
  static constexpr int8_t max() noexcept { return 127;}
  static constexpr bool is_integer = true;
  static constexpr bool has_infinity = false;
};


template <>
struct numeric_limits<uint32_t> {
  CUTLASS_HOST_DEVICE
  static constexpr uint32_t lowest() noexcept { return 0;}
  CUTLASS_HOST_DEVICE
  static constexpr uint32_t max() noexcept { return 4294967295U;}
  static constexpr bool is_integer = true;
  static constexpr bool has_infinity = false;
};

template <>
struct numeric_limits<uint16_t> {
  CUTLASS_HOST_DEVICE
  static constexpr uint16_t lowest() noexcept { return 0;}
  CUTLASS_HOST_DEVICE
  static constexpr uint16_t max() noexcept { return 65535U;}
  static constexpr bool is_integer = true;
  static constexpr bool has_infinity = false;
};

template <>
struct numeric_limits<uint8_t> {
  CUTLASS_HOST_DEVICE
  static constexpr uint8_t lowest() noexcept { return 0;}
  CUTLASS_HOST_DEVICE
  static constexpr uint8_t max() noexcept { return 255U;}
  static constexpr bool is_integer = true;
  static constexpr bool has_infinity = false;
};

template <>
struct numeric_limits<float> {
  CUTLASS_HOST_DEVICE
  static constexpr float infinity() noexcept { return bit_cast<float, int32_t>(0x7f800000);}
  CUTLASS_HOST_DEVICE
  static constexpr float max() noexcept { return bit_cast<float, int32_t>(0x7f7fffff);}
  static constexpr bool is_integer = false;
  static constexpr bool has_infinity = true;
};

/// Returns a value that curries the `std::maximum()` function into the identity
/// function. No value will compare < than this value.
template <typename T>
constexpr T identity_for_maximum() {
  if constexpr (numeric_limits<T>::has_infinity) {
    return -numeric_limits<T>::infinity();
  } else {
    return numeric_limits<T>::lowest();
  }
}

/// Returns a value that curries the `std::minimum()` function into the identity
/// function. No value will compare > than this value.
template <typename T>
constexpr T identity_for_minimum() {
  if constexpr (numeric_limits<T>::has_infinity) {
    return numeric_limits<T>::infinity();
  } else {
    return numeric_limits<T>::max();
  }
}

/// std::float_round_style
using CUTLASS_STL_NAMESPACE::float_round_style;
using CUTLASS_STL_NAMESPACE::round_indeterminate;
using CUTLASS_STL_NAMESPACE::round_toward_zero;
using CUTLASS_STL_NAMESPACE::round_to_nearest;
using CUTLASS_STL_NAMESPACE::round_toward_infinity;
using CUTLASS_STL_NAMESPACE::round_toward_neg_infinity;

/// std::float_denorm_style
using CUTLASS_STL_NAMESPACE::float_denorm_style;
using CUTLASS_STL_NAMESPACE::denorm_indeterminate;
using CUTLASS_STL_NAMESPACE::denorm_absent;
using CUTLASS_STL_NAMESPACE::denorm_present;

}  // namespace platform
}  // namespace cutlass
