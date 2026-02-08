#------------------------------------------------------------------------------
# Build the google test library

# We compile Google Test ourselves instead of using pre-compiled libraries.
# See the Google Test FAQ "Why is it not recommended to install a
# pre-compiled copy of Google Test (for example, into /usr/local)?"
# at http://code.google.com/p/googletest/wiki/FAQ for more details.
add_library(gtest STATIC
  gmock-gtest-all.cc gmock/gmock.h gtest/gtest.h gtest/gtest-spi.h)
target_compile_definitions(gtest PUBLIC GTEST_HAS_STD_WSTRING=1)
target_include_directories(gtest SYSTEM PUBLIC .)
target_compile_features(gtest PUBLIC cxx_std_11)

find_package(Threads)
if (Threads_FOUND)
  target_link_libraries(gtest ${CMAKE_THREAD_LIBS_INIT})
else ()
  target_compile_definitions(gtest PUBLIC GTEST_HAS_PTHREAD=0)
endif ()

if (MSVC)
  # Disable MSVC warnings of _CRT_INSECURE_DEPRECATE functions.
  target_compile_definitions(gtest PRIVATE _CRT_SECURE_NO_WARNINGS)
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Disable MSVC warnings of POSIX functions.
    target_compile_options(gtest PUBLIC -Wno-deprecated-declarations)
  endif ()
endif ()

# Silence MSVC tr1 deprecation warning in gmock.
target_compile_definitions(gtest
  PUBLIC _SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING=1)
