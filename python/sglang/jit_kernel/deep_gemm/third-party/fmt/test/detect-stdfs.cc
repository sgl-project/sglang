// Formatting library for C++ - tests of formatters for standard library types
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#include <exception>  // _GLIBCXX_RELEASE & _LIBCPP_VERSION

#if defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE == 8
#  error libfound "stdc++fs"
#elif !defined(__apple_build_version__) && defined(_LIBCPP_VERSION) && \
    _LIBCPP_VERSION >= 7000 && _LIBCPP_VERSION < 9000
#  error libfound "c++fs"
#else
// none if std::filesystem does not require additional libraries
#  error libfound ""
#endif
