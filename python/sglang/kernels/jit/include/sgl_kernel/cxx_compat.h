#pragma once

#if defined(__has_include)
#if __has_include(<version>)
#include <version>
#endif
#else
#define __has_include(...) 0
#endif

#if defined(__has_cpp_attribute)
#define SGL_HAS_CPP_ATTRIBUTE(name) __has_cpp_attribute(name)
#else
#define SGL_HAS_CPP_ATTRIBUTE(name) 0
#endif

#if defined(__MUSACC__) && defined(__clang_major__) && __clang_major__ == 14
#define SGL_MCC_CLANG14_RANGES_BROKEN 1
#else
#define SGL_MCC_CLANG14_RANGES_BROKEN 0
#endif

#ifndef SGL_USE_CONCEPTS
#if __cplusplus >= 202002L && defined(__cpp_concepts) && __cpp_concepts >= 201907L
#define SGL_USE_CONCEPTS 1
#else
#define SGL_USE_CONCEPTS 0
#endif
#endif

#ifndef SGL_USE_RANGES
#if !SGL_MCC_CLANG14_RANGES_BROKEN && __cplusplus >= 202002L && __has_include(<ranges>) && \
    defined(__cpp_lib_ranges) && __cpp_lib_ranges >= 201911L
#define SGL_USE_RANGES 1
#else
#define SGL_USE_RANGES 0
#endif
#endif

#ifndef SGL_USE_SPAN
#if __cplusplus >= 202002L && __has_include(<span>) && defined(__cpp_lib_span) && __cpp_lib_span >= 202002L
#define SGL_USE_SPAN 1
#else
#define SGL_USE_SPAN 0
#endif
#endif

#ifndef SGL_USE_BIT_CAST
#if __cplusplus >= 202002L && __has_include(<bit>) && defined(__cpp_lib_bit_cast) && __cpp_lib_bit_cast >= 201806L
#define SGL_USE_BIT_CAST 1
#else
#define SGL_USE_BIT_CAST 0
#endif
#endif

#ifndef SGL_USE_BITOPS
#if __cplusplus >= 202002L && __has_include(<bit>) && defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L
#define SGL_USE_BITOPS 1
#else
#define SGL_USE_BITOPS 0
#endif
#endif

#ifndef SGL_USE_TYPE_IDENTITY
#if __cplusplus >= 202002L && defined(__cpp_lib_type_identity) && __cpp_lib_type_identity >= 201806L
#define SGL_USE_TYPE_IDENTITY 1
#else
#define SGL_USE_TYPE_IDENTITY 0
#endif
#endif

#ifndef SGL_LIKELY
#if SGL_HAS_CPP_ATTRIBUTE(likely)
#define SGL_LIKELY [[likely]]
#else
#define SGL_LIKELY
#endif
#endif

#undef SGL_HAS_CPP_ATTRIBUTE
#undef SGL_MCC_CLANG14_RANGES_BROKEN
