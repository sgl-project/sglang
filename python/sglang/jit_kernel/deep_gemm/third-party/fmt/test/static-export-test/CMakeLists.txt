cmake_minimum_required(VERSION 3.8...3.25)

project(fmt-link CXX)

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")

# Broken LTO on GCC 4
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5)
  set(BROKEN_LTO ON)
endif ()

if (NOT BROKEN_LTO AND CMAKE_VERSION VERSION_GREATER "3.8")
  # CMake 3.9+
  include(CheckIPOSupported)
  check_ipo_supported(RESULT HAVE_IPO)
  if (HAVE_IPO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif ()
endif ()

add_subdirectory(../.. fmt)
set_property(TARGET fmt PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(library-test SHARED library.cc)
target_link_libraries(library-test PRIVATE fmt::fmt)

add_executable(exe-test main.cc)
target_link_libraries(exe-test PRIVATE library-test)
