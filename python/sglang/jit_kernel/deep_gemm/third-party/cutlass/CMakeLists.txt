# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
cmake_policy(SET CMP0112 NEW)

if(cutlass_LOADED)
  # If CUTLASS has been previously fetched and loaded, don't do it again.
  return()
else()
  set(cutlass_LOADED ON)
  set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "CUTLASS Repository Directory")
endif()

message(STATUS "CMake Version: ${CMAKE_VERSION}")
set(IMPLICIT_CMAKE_CXX_STANDARD OFF CACHE BOOL "Do not explicitly specify -std=c++17 if set")

# To reduce duplicate version locations, parse the version out of the
# main versions.h file and reuse it here.

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/include/cutlass/version.h VERSION_FILE_CONTENTS)
string(REGEX MATCH "#define CUTLASS_MAJOR ([0-9]+)" _CUTLASS_VERSION_MAJOR "${VERSION_FILE_CONTENTS}")
set(_CUTLASS_VERSION_MAJOR ${CMAKE_MATCH_1})
string(REGEX MATCH "#define CUTLASS_MINOR ([0-9]+)" _CUTLASS_VERSION_MINOR "${VERSION_FILE_CONTENTS}")
set(_CUTLASS_VERSION_MINOR ${CMAKE_MATCH_1})
string(REGEX MATCH "#define CUTLASS_PATCH ([0-9]+)" _CUTLASS_VERSION_PATCH "${VERSION_FILE_CONTENTS}")
set(_CUTLASS_VERSION_PATCH ${CMAKE_MATCH_1})

message(STATUS "CUTLASS ${_CUTLASS_VERSION_MAJOR}.${_CUTLASS_VERSION_MINOR}.${_CUTLASS_VERSION_PATCH}")

## CUTLASS PROJECT #############################################################

project(CUTLASS VERSION ${_CUTLASS_VERSION_MAJOR}.${_CUTLASS_VERSION_MINOR}.${_CUTLASS_VERSION_PATCH} LANGUAGES CXX)

################################################################################

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(CUTLASS_GNU_HOST_COMPILE ON CACHE BOOL "Using GNU tools for host code compilation")
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "[Cc]lang")
  set(CUTLASS_CLANG_HOST_COMPILE ON CACHE BOOL "Using Clang tools for host code compilation")
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  set(CUTLASS_MSVC_HOST_COMPILE ON CACHE BOOL "Using MSVC tools for host code compilation")
endif()

################################################################################

include(${CMAKE_CURRENT_SOURCE_DIR}/CUDA.cmake)

if (CUDA_VERSION VERSION_LESS 11.3)
  message(WARNING "CUTLASS ${CUTLASS_VERSION} requires CUDA 11.4 or higher, and strongly recommends CUDA 11.8 or higher.")
elseif (CUDA_VERSION VERSION_LESS 11.4)
  message(WARNING "CUTLASS ${CUTLASS_VERSION} support for CUDA ${CUDA_VERSION} is deprecated, please use CUDA 11.8 or higher.")
endif()

if(CUTLASS_GNU_HOST_COMPILE AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3)
  message(FATAL_ERROR "GCC version must be at least 7.3!")
endif()

if (CUTLASS_CLANG_DEVICE_COMPILE AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
  message(FATAL_ERROR "Clang 7.0+ required for GPU compilation")
endif()
find_package(Doxygen QUIET)

################################################################################

#
# CUTLASS 3.x requires C++17
#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

list(APPEND CUTLASS_CUDA_NVCC_FLAGS --expt-relaxed-constexpr)

list(APPEND CUTLASS_CUDA_NVCC_FLAGS -ftemplate-backtrace-limit=0)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX install CACHE PATH "Default installation location." FORCE)
endif()

message(STATUS "Default Install Location: ${CMAKE_INSTALL_PREFIX}")

set(CUTLASS_TEST_LEVEL "0" CACHE STRING "Level of tests to compile.")
# 0 - Sanity, 1 - Release-Quality, 2 - Exhaustive

find_package(Python3 3.5 COMPONENTS Interpreter REQUIRED)

################################################################################


include(customConfigs.cmake)

################################################################################


set(CUTLASS_ENABLE_HEADERS_ONLY OFF CACHE BOOL "Enable only the header library")

if(CUTLASS_ENABLE_HEADERS_ONLY)
  set(CUTLASS_ENABLE_EXAMPLES_INIT OFF)
  set(CUTLASS_ENABLE_TOOLS_INIT ON)
  set(CUTLASS_ENABLE_LIBRARY_INIT OFF)
  set(CUTLASS_ENABLE_TESTS_INIT OFF)
else()
  set(CUTLASS_ENABLE_EXAMPLES_INIT ON)
  set(CUTLASS_ENABLE_TOOLS_INIT ON)
  set(CUTLASS_ENABLE_LIBRARY_INIT ON)
  if(${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
    set(CUTLASS_ENABLE_TESTS_INIT ON)
  else()
    set(CUTLASS_ENABLE_TESTS_INIT OFF)
  endif()
endif()

set(CUTLASS_TEST_UNIT_ENABLE_WARNINGS OFF CACHE BOOL "Enable warnings on waived unit tests.")

set(CUTLASS_ENABLE_EXAMPLES ${CUTLASS_ENABLE_EXAMPLES_INIT} CACHE BOOL "Enable CUTLASS Examples")
set(CUTLASS_ENABLE_TOOLS ${CUTLASS_ENABLE_TOOLS_INIT} CACHE BOOL "Enable CUTLASS Tools")
set(CUTLASS_ENABLE_LIBRARY ${CUTLASS_ENABLE_LIBRARY_INIT} CACHE BOOL "Enable CUTLASS Library")
set(CUTLASS_ENABLE_PROFILER ${CUTLASS_ENABLE_LIBRARY} CACHE BOOL "Enable CUTLASS Profiler")
set(CUTLASS_ENABLE_PERFORMANCE ${CUTLASS_ENABLE_PROFILER} CACHE BOOL "Enable CUTLASS Performance")

set(CUTLASS_ENABLE_TESTS ${CUTLASS_ENABLE_TESTS_INIT} CACHE BOOL "Enable CUTLASS Tests")
set(CUTLASS_ENABLE_GTEST_UNIT_TESTS ${CUTLASS_ENABLE_TESTS} CACHE BOOL "Enable CUTLASS GTest-based Unit Tests")
set(CUTLASS_USE_SYSTEM_GOOGLETEST OFF CACHE BOOL "Use system/external installation of GTest")

if (CUTLASS_ENABLE_TESTS AND CUTLASS_ENABLE_PROFILER)
  set(CUTLASS_ENABLE_PROFILER_UNIT_TESTS_INIT ON)
else()
  set(CUTLASS_ENABLE_PROFILER_UNIT_TESTS_INIT OFF)
endif()
set(CUTLASS_ENABLE_PROFILER_UNIT_TESTS ${CUTLASS_ENABLE_PROFILER_UNIT_TESTS_INIT} CACHE BOOL "Enable CUTLASS Profiler-based Unit Tests")
set(CUTLASS_ENABLE_SELF_CONTAINED_INCLUDES_CHECK ON CACHE BOOL "Enable CUTLASS check for self-contained header includes")

################################################################################

set(CUTLASS_NVCC_ARCHS_SUPPORTED "")
if (CUDA_VERSION VERSION_GREATER_EQUAL 11.4)
  list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 70 72 75 80 86 87)
endif()
if (CUDA_VERSION VERSION_GREATER_EQUAL 11.8)
  list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 89 90)
endif()
if (CUDA_VERSION VERSION_GREATER_EQUAL 12.0)
  list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 90a)
endif()

if (CUDA_VERSION VERSION_GREATER_EQUAL 12.8)
  list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 100 100a 120 120a 121 121a)
  if (CUDA_VERSION VERSION_LESS 13.0)
    list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 101 101a)
  else()
    list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 110 110a)
  endif()
endif()

if (CUDA_VERSION VERSION_GREATER_EQUAL 12.9)
  list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 100f 120f 121f 103a 103f)
  if (CUDA_VERSION VERSION_LESS 13.0)
    list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 101f)
  else()
    list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 110f)
  endif()
endif()

if (CUDA_VERSION VERSION_GREATER_EQUAL 13.0)
  list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 110 110a)
endif()

set(CUTLASS_NVCC_ARCHS ${CUTLASS_NVCC_ARCHS_SUPPORTED} CACHE STRING "The SM architectures requested.")
set(CUTLASS_NVCC_ARCHS_ENABLED ${CUTLASS_NVCC_ARCHS} CACHE STRING "The SM architectures to build code for.")

# Find unsupported and deprecated compute capabilities
if (CUTLASS_NVCC_ARCHS_SUPPORTED)
  set(CUTLASS_NVCC_ARCHS_UNSUPPORTED ${CUTLASS_NVCC_ARCHS})
  list(REMOVE_ITEM CUTLASS_NVCC_ARCHS_UNSUPPORTED ${CUTLASS_NVCC_ARCHS_SUPPORTED})
  if (CUTLASS_NVCC_ARCHS_UNSUPPORTED)
    message(WARNING "Using unsupported or deprecated compute capabilities ${CUTLASS_NVCC_ARCHS_UNSUPPORTED}. Support may be removed in future versions.")
  endif()
else()
  message(WARNING "No supported compute capabilities for CUDA ${CUDA_VERSION}.")
endif()

# Special policy introduced in CMake 3.13
if (POLICY CMP0076)
  cmake_policy(SET CMP0076 NEW)
endif()

include(GNUInstallDirs)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

###################################################################################################
#
# Configure CMake variables
#
###################################################################################################

message(STATUS "CUDA Compilation Architectures: ${CUTLASS_NVCC_ARCHS_ENABLED}")

if (NOT (CMAKE_BUILD_TYPE OR CONFIGURATION_TYPES))
  # By default we want to build in Release mode to ensure that we're getting best performance.
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build level" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "RelWithDebInfo" "Release")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (DEFINED CMAKE_DEBUG_POSTFIX)
  set(CUTLASS_LIBRARY_DEBUG_POSTFIX_INIT ${CMAKE_DEBUG_POSTFIX})
else()
  set(CUTLASS_LIBRARY_DEBUG_POSTFIX_INIT .debug)
endif()
set(CUTLASS_LIBRARY_DEBUG_POSTFIX ${CUTLASS_LIBRARY_DEBUG_POSTFIX_INIT} CACHE STRING "Default postfix value for debug libraries")

if(WIN32)
  # On Windows we link against the shared (DLL) runtime. Change gtest settings to match this.
  set(gtest_force_shared_crt ON CACHE BOOL "Use shared (DLL) run-time lib even when Google Test is built as static lib" FORCE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUTLASS_VERSIONS_GENERATED")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUTLASS_VERSIONS_GENERATED")

if (WIN32)
  # Enable more warnings.  Add "-Xcompiler=/WX" to enable warnings as errors.
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -Xcompiler=/W3)

  # Disable warning on Unicode characters
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -Xcompiler=/wd4819)

  # Disable excess x86 floating point precision that can lead to results being labeled incorrectly
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -Xcompiler=/fp:strict)
endif(WIN32)

if (${CUTLASS_NVCC_VERBOSE})
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -v)
endif()

#
# CUTLASS NAMESPACE
#
set(CUTLASS_NAMESPACE "cutlass" CACHE STRING "Top level namespace of CUTLASS")

set(CUTLASS_NVCC_EMBED_CUBIN ON CACHE BOOL "Embed compiled CUDA kernel binaries into executables.")
set(CUTLASS_NVCC_EMBED_PTX ON CACHE BOOL "Embed compiled PTX into executables.")
set(CUTLASS_NVCC_KEEP OFF CACHE BOOL "Keep intermediate files generated by NVCC.")
set(CUTLASS_ENABLE_F16C OFF CACHE BOOL "Enable F16C x86 extensions in host code.")

################################################################################
#
# CUTLASS generator cmake configuration
#

# Kernel unified filter file

set(KERNEL_FILTER_FILE "" CACHE STRING "KERNEL FILTER FILE FULL PATH")

if (KERNEL_FILTER_FILE AND NOT CUTLASS_LIBRARY_KERNELS)
  # If a kernel filter file is specified, we want to generate and then
  # filter on the entire kernel set, not the default kernel
  # (sub)set. The user may have overridden CUTLASS_LIBRARY_KERNELS, in which
  # case the resulting kernel set will be the intersection of the two
  # options differenced against CUTLASS_LIBRARY_IGNORE_KERNELS.
  set(CUTLASS_LIBRARY_KERNELS_INIT "*")
else()
  set(CUTLASS_LIBRARY_KERNELS_INIT "")
endif()

if (KERNEL_FILTER_FILE)
  get_filename_component(KERNEL_FILTER_FILE "${KERNEL_FILTER_FILE}" ABSOLUTE)
  set(KERNEL_FILTER_FILE "${KERNEL_FILTER_FILE}" CACHE STRING "KERNEL FILTER FILE FULL PATH" FORCE)
endif()

if (CUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE)
  get_filename_component(CUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE "${CUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE}" ABSOLUTE)
  set(CUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE "${CUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE}" CACHE STRING "HEURISTICS FILE FULL PATH" FORCE)
endif()

set(SELECTED_KERNEL_LIST "selected" CACHE STRING "Name of the filtered kernel list")

if(KERNEL_FILTER_FILE)
  message(STATUS "Full path of filter file: ${KERNEL_FILTER_FILE}")
endif()

if(CUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE)
  message(STATUS "Full path of heuristics problems file: ${CUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE}")
  if(DEFINED CUTLASS_NVMMH_URL)
    message(STATUS "CUTLASS_NVVMH_URL is set. Fetching dependency")
    include(FetchContent)
    FetchContent_Declare(
      nvmmh
      URL ${CUTLASS_NVMMH_URL}
    )
    FetchContent_MakeAvailable(nvmmh)
    FetchContent_GetProperties(nvmmh SOURCE_DIR nvmmh_dir)
    set(CUTLASS_NVMMH_PATH "${nvmmh_dir}")
  endif()

  if(DEFINED CUTLASS_NVMMH_PATH)
    message(STATUS "CUTLASS_NVMMH_PATH is set. Using package at: ${CUTLASS_NVMMH_PATH}")

    set(CUTLASS_NVMMH_PY_DIR "${CUTLASS_NVMMH_PATH}/python/")
    set(ENV{CUTLASS_NVMMH_SO_PATH} "${CUTLASS_NVMMH_PATH}/lib/libnvMatmulHeuristics.so")
  endif()
endif()

set(CUTLASS_LIBRARY_OPERATIONS "all" CACHE STRING "Comma-delimited list of operation name filters. Default '' means all operations are enabled.")
set(CUTLASS_LIBRARY_KERNELS ${CUTLASS_LIBRARY_KERNELS_INIT} CACHE STRING "Comma-delimited list of kernel name filters. If unspecified, only the largest tile size is enabled. If the string 'all' is specified, all kernels are enabled.")
set(CUTLASS_LIBRARY_IGNORE_KERNELS "" CACHE STRING "Comma-delimited list of kernels to exclude from build. This option ONLY takes effect if CUTLASS_LIBRARY_KERNELS is set.")
set(CUTLASS_LIBRARY_EXCLUDE_KERNELS "" CACHE STRING "Comma-delimited list of kernels to exclude from build. This option always takes effect, whether or not CUTLASS_LIBRARY_KERNELS is set. It also can exclude kernels from the filter file (see KERNEL_FILTER_FILE).")
set(CUTLASS_LIBRARY_INSTANTIATION_LEVEL "" CACHE STRING "Instantiation level for SM90 and SM100 kernels. Set to `max` and make sure CUTLASS_LIBRARY_KERNELS is non-empty to stamp all possible kernel configurations.")

if(CUTLASS_LIBRARY_INSTANTIATION_LEVEL OR CUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE)
  message(STATUS "Enable extended SM90 WGMMA instruction shapes for instantiation levels")
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
endif()

################################################################################

set(CUTLASS_TEST_ENABLE_CACHED_RESULTS ON CACHE BOOL "Enable caching and reuse of test results in unit tests")

set_property(CACHE CUTLASS_TEST_LEVEL PROPERTY STRINGS 0 1 2)
list(APPEND CUTLASS_CUDA_NVCC_FLAGS -DCUTLASS_TEST_LEVEL=${CUTLASS_TEST_LEVEL})
list(APPEND CUTLASS_CUDA_CLANG_FLAGS -DCUTLASS_TEST_LEVEL=${CUTLASS_TEST_LEVEL})

if (CUTLASS_TEST_ENABLE_CACHED_RESULTS)
  message(STATUS "Enable caching of reference results in conv unit tests")
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1)
endif()

set(CUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED ON CACHE BOOL "Enable/Disable rigorous conv problem sizes in conv unit tests")

if (CUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED)
  message(STATUS "Enable rigorous conv problem sizes in conv unit tests")
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1)
endif()

################################################################################

#
# CUDA 10.1 introduces "mma" in PTX performing collective matrix multiply operations.
#

if (CUDA_VERSION VERSION_LESS 10.1)
  set(CUTLASS_ENABLE_TENSOR_CORE_MMA_DEFAULT OFF)
else()
  set(CUTLASS_ENABLE_TENSOR_CORE_MMA_DEFAULT ON)
endif()

# Trace levels for debugging
set(CUTLASS_DEBUG_TRACE_LEVEL "0" CACHE STRING "Level of debug tracing to perform.")
list(APPEND CUTLASS_CUDA_NVCC_FLAGS -DCUTLASS_DEBUG_TRACE_LEVEL=${CUTLASS_DEBUG_TRACE_LEVEL})

set(CUTLASS_ENABLE_TENSOR_CORE_MMA ${CUTLASS_ENABLE_TENSOR_CORE_MMA_DEFAULT} CACHE BOOL
  "Enable PTX mma instruction for collective matrix multiply operations.")

set(CUTLASS_ENABLE_SM90_EXTENDED_MMA_SHAPES OFF CACHE BOOL
  "Enable an extended set of SM90 WGMMA instruction shapes (may lead to increased compilation times)")
if(CUTLASS_ENABLE_SM90_EXTENDED_MMA_SHAPES)
  message(STATUS "Enabled extended SM90 WGMMA instruction shapes")
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
endif()

if (CUTLASS_NVCC_ARCHS MATCHES 100f OR CUTLASS_NVCC_ARCHS MATCHES 101f)
list(APPEND CUTLASS_CUDA_NVCC_FLAGS -DCUTLASS_SM100_FAMILY_ARCHS_ENABLED)
endif()

if (CUTLASS_NVCC_ARCHS MATCHES 110f)
list(APPEND CUTLASS_CUDA_NVCC_FLAGS -DCUTLASS_SM100_FAMILY_ARCHS_ENABLED)
endif()

set(CUTLASS_SKIP_REDUCTION_INIT OFF CACHE BOOL "Disable init reduction workspace")

#
# NOTE: running with asan and CUDA requires the following environment variable:
#
#  ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0
#
# without the above environment setting, an error like the following may be generated:
#
#  *** Error: Could not detect active GPU device ID [out of memory]
#  ...
#  ==9149==ERROR: LeakSanitizer: detected memory leaks
#  ...
#
if(ENABLE_ASAN)  # https://github.com/google/sanitizers/wiki/AddressSanitizer
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS --compiler-options=-fsanitize=address --compiler-options=-fno-omit-frame-pointer)
  string(APPEND CMAKE_EXE_LINKER_FLAGS " -fsanitize=address")
endif()

###################################################################################################
#
# Configure CUDA build options
#
###################################################################################################

if(CUTLASS_NVCC_EMBED_PTX)
  list(APPEND CUTLASS_CUDA_CLANG_FLAGS --cuda-include-ptx=all)
endif()

if (CUTLASS_SKIP_REDUCTION_INIT)
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -DCUTLASS_SKIP_REDUCTION_INIT=1)
endif()

if (CUTLASS_ENABLE_TENSOR_CORE_MMA)
  list(APPEND CUTLASS_CUDA_FLAGS -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1)
endif()

set(CUTLASS_PROFILER_DISABLE_REFERENCE OFF CACHE BOOL "Disable compilation of reference kernels in the CUTLASS profiler.")
if (CUTLASS_PROFILER_DISABLE_REFERENCE)
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -DCUTLASS_PROFILER_DISABLE_REFERENCE=1)
endif()

if (CUTLASS_ENABLE_GDC_FOR_SM90)
  message(STATUS "Grid Dependency Control (GDC) is enabled for SM90 kernels (required for programmatic dependent launches).")
  list(APPEND CUTLASS_CUDA_FLAGS -DCUTLASS_ENABLE_GDC_FOR_SM90=1)
endif()

if (NOT DEFINED CUTLASS_ENABLE_GDC_FOR_SM100_DEFAULT)
  set(CUTLASS_ENABLE_GDC_FOR_SM100_DEFAULT ON)
endif()

set(CUTLASS_ENABLE_GDC_FOR_SM100
    ${CUTLASS_ENABLE_GDC_FOR_SM100_DEFAULT}
    CACHE BOOL
    "Enables Grid Dependency Control (GDC) for SM100 kernels (required for PDL).")

if (CUTLASS_ENABLE_GDC_FOR_SM100)
  message(STATUS "Grid Dependency Control (GDC) is enabled for SM100 kernels (required for programmatic dependent launches).")
  list(APPEND CUTLASS_CUDA_FLAGS -DCUTLASS_ENABLE_GDC_FOR_SM100=1)
endif()

set(CUTLASS_ENABLE_SYNCLOG OFF CACHE BOOL "Enable synchronization event logging for race condition debugging. WARNING: This redefines __syncthreads() and __syncwarp() in all downstream code!")

if (CUTLASS_ENABLE_SYNCLOG)
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
  string(APPEND CMAKE_CXX_FLAGS " -DCUTLASS_ENABLE_SYNCLOG=1")
  string(APPEND CMAKE_CUDA_FLAGS " -DCUTLASS_ENABLE_SYNCLOG=1")
endif()




###################################################################################################
#
# Blackwell features
#
###################################################################################################

# Warnings-as-error exceptions and warning suppressions for Clang builds
if (CUTLASS_CLANG_HOST_COMPILE)

  set(FLAGS_TO_ADD
    "-Wno-error=implicit-int-conversion"
    "-Wno-error=pass-failed"
    "-Wno-error=inconsistent-missing-override"
    "-Wno-sign-conversion"
    "-Wno-unused-parameter"
  )

  foreach(FLAG ${FLAGS_TO_ADD})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG}")
    list(APPEND CUTLASS_CUDA_NVCC_FLAGS "${FLAG}")
    list(APPEND CUTLASS_CUDA_CLANG_FLAGS "${FLAG}")
  endforeach()

endif()

if (NOT MSVC AND CUTLASS_NVCC_KEEP)
  # MSVC flow handles caching already, but for other generators we handle it here.
  set(CUTLASS_NVCC_KEEP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp CACHE PATH "Location to store NVCC scratch files")
  file(MAKE_DIRECTORY ${CUTLASS_NVCC_KEEP_DIR})
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS --keep -v -objtemp) # --keep-dir may not work with nvcc for some directories.
  list(APPEND CUTLASS_CUDA_CLANG_FLAGS -save-temps=${CUTLASS_NVCC_KEEP_DIR})
endif()

if (CUTLASS_ENABLE_F16C AND NOT CMAKE_CROSSCOMPILING)
  list(APPEND CUTLASS_CUDA_FLAGS -DCUTLASS_ENABLE_F16C=1)
  if (CUTLASS_GNU_HOST_COMPILE OR CUTLASS_CLANG_HOST_COMPILE)
    list(APPEND CUTLASS_CUDA_NVCC_FLAGS -Xcompiler=-mf16c)
  elseif(CUTLASS_MSVC_HOST_COMPILE)
    list(APPEND CUTLASS_CUDA_NVCC_FLAGS -Xcompiler=/arch:AVX2)
  endif()
endif()

if (CUTLASS_ENABLE_OPENMP_TESTS)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    list(APPEND CUTLASS_CUDA_NVCC_FLAGS -Xcompiler=${OpenMP_CXX_FLAGS})
  else()
    message(WARNING "CUTLASS_ENABLE_OPENMP_TESTS set but OpenMP not found.")
  endif()
endif()

if(UNIX)
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -Xcompiler=-Wconversion)
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -Xcompiler=-fno-strict-aliasing)
endif()

# Known ctk11.4 issue (fixed later)
# Also see https://stackoverflow.com/questions/64523302/cuda-missing-return-statement-at-end-of-non-void-function-in-constexpr-if-fun
if (CUDA_VERSION VERSION_LESS 11.5.0)
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -Xcudafe "--diag_suppress=implicit_return_from_non_void_function" )
  message("CUDA_VERSION check pass ${CUDA_VERSION}")
endif()

# Don't leak lineinfo in release builds
if (NOT CMAKE_BUILD_TYPE MATCHES "Release")
  list(APPEND CUTLASS_CUDA_CLANG_FLAGS -gmlt)
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -lineinfo)
endif()

if (CUTLASS_CLANG_DEVICE_COMPILE)
  if (NOT CUTLASS_CLANG_HOST_COMPILE)
    message(FATAL_ERROR "Clang CUDA compilation requires Clang CXX compilation. Currently CMAKE_CXX_COMPILER is ${CMAKE_CXX_COMPILER_ID}" )
  endif()

  # There are numerous Clang versions that can work with each CUDA toolkit and the
  # the checks are not very useful so we are turning them off and using testing to
  # ensure the various combinations work properly.

  list(APPEND CUTLASS_CUDA_CLANG_FLAGS --cuda-path=${CUDA_TOOLKIT_ROOT_DIR})
  list(APPEND CUTLASS_CUDA_CLANG_FLAGS -D__NV_NO_HOST_COMPILER_CHECK=1)
  list(APPEND CUTLASS_CUDA_CLANG_FLAGS -Wno-unknown-cuda-version)

  list(APPEND CUTLASS_CUDA_CLANG_FLAGS -mllvm -pragma-unroll-threshold=100000)
  list(APPEND CUTLASS_CUDA_CLANG_FLAGS -mllvm -unroll-threshold=5000)
  list(APPEND CUTLASS_CUDA_CLANG_FLAGS -Wno-unused-command-line-argument)

  list(APPEND CUTLASS_CUDA_CLANG_FLAGS -D__CUDACC_VER_MAJOR__=${CUDA_VERSION_MAJOR} -D__CUDACC_VER_MINOR__=${CUDA_VERSION_MINOR})

  # needed for libcublasLt.so in case it's installed in the same location as libcudart.so
  # dynamic linker can find it if linker sets RPATH (forced by --disable-new-tags)
  # Otherwise linker uses RUNPATH and that does not propagate to loaded libs.
  list(APPEND CUTLASS_CUDA_CLANG_FLAGS -Wl,--disable-new-dtags)

  link_libraries(nvidia::cudart)
  link_libraries(nvidia::cuda_driver)

endif()

#Report CUDA build flags
if (CUTLASS_CLANG_DEVICE_COMPILE AND CUTLASS_CUDA_CLANG_FLAGS)
  set(__FLAG_GROUP Clang)
  set(__FLAG_LIST CUTLASS_CUDA_CLANG_FLAGS)
else(CUTLASS_NVCC_DEVICE_COMPILE AND CUTLASS_CUDA_NVCC_FLAGS)
  set(__FLAG_GROUP NVCC)
  set(__FLAG_LIST CUTLASS_CUDA_NVCC_FLAGS)
endif()

set(__FLAG_DISPLAY_STRING "")
set(__FLAG_DISPLAY_SEPARATOR)
list(JOIN ${__FLAG_LIST} "\n  " __FLAG_DISPLAY_STRING)
message(STATUS "Using the following ${__FLAG_GROUP} flags: \n  ${__FLAG_DISPLAY_STRING}")

# Known gcc 8.1-8.3 SFINAE issue (fixed in gcc 8.4), check https://gcc.gnu.org/bugzilla/show_bug.cgi?id=87748
# Also see https://github.com/NVIDIA/nccl/issues/835 for nvtx3.hpp
if (CUTLASS_GNU_HOST_COMPILE AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8.1 AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS_EQUAL 8.3)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNVTX3_USE_CHECKED_OVERLOADS_FOR_GET=0")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNVTX3_USE_CHECKED_OVERLOADS_FOR_GET=0")
endif()

# Support for 128-bit integers if using NVIDIA C++ compiler
if (${CMAKE_CXX_COMPILER_ID} MATCHES "PGI" OR ${CMAKE_CXX_COMPILER_ID} MATCHES "NVHPC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Mint128 ")
endif()

# CMake 3.18 added support for CUDA_ARCHITECTURES target property. We will use this
# property for CMake 3.18+, so we request the NEW behavior for correct compatibility.
# https://cmake.org/cmake/help/v3.18/policy/CMP0104.html#policy:CMP0104
cmake_policy(SET CMP0104 NEW)

if (MSVC)

  # MSVC by default does not apply the correct __cplusplus version as specified by the C++ standard
  # because MSVC is not a completely compliant implementation. This option forces MSVC to use the
  # appropriate value given the requested --std option. This fixes a compilation issue mismatch
  # between GCC/Clang and MSVC.
  #
  # error : a constexpr function cannot have a nonliteral return type "dim3"
  #
  # See https://developercommunity.visualstudio.com/t/msvc-incorrectly-defines-cplusplus/139261

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler  /Zc:__cplusplus")

endif()

# Some tests require this build option in order to link.
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /bigobj")
endif()

function(cutlass_apply_cuda_gencode_flags TARGET)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs SM_ARCHS)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (__SM_ARCHS)
    set(ARCHS_ENABLED ${__SM_ARCHS})
  else()
    set(ARCHS_ENABLED ${CUTLASS_NVCC_ARCHS_ENABLED})
  endif()

  set(__CMAKE_CUDA_ARCHS)
  foreach(ARCH ${ARCHS_ENABLED})
    set(CODES)
    if(CUTLASS_NVCC_EMBED_CUBIN)
      list(APPEND __CMAKE_CUDA_ARCHS ${ARCH}-real)
    endif()
    if(CUTLASS_NVCC_EMBED_PTX AND NOT CUTLASS_CLANG_DEVICE_COMPILE)
      # If we're using clang for device compilation, the ptx is inserted
      # via another command line option and the `-virtual` flags will cause an error.
      list(APPEND __CMAKE_CUDA_ARCHS ${ARCH}-virtual)
    endif()
    list(JOIN CODES "," CODES_STR)
  endforeach()

  set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES ${__CMAKE_CUDA_ARCHS})

endfunction()

# Cache the flags so they are available when the function below is called anywhere globally.

set(__CUTLASS_CUDA_FLAGS ${CUTLASS_CUDA_FLAGS} CACHE INTERNAL "")
set(__CUTLASS_CUDA_FLAGS_RELEASE ${CUTLASS_CUDA_FLAGS_RELEASE} CACHE INTERNAL "")
set(__CUTLASS_CUDA_FLAGS_RELWITHDEBINFO ${CUTLASS_CUDA_FLAGS_RELWITHDEBINFO} CACHE INTERNAL "")
set(__CUTLASS_CUDA_FLAGS_DEBUG ${CUTLASS_CUDA_FLAGS_DEBUG} CACHE INTERNAL "")
set(__CUTLASS_CUDA_CLANG_FLAGS ${CUTLASS_CUDA_CLANG_FLAGS} CACHE INTERNAL "")
set(__CUTLASS_CUDA_CLANG_FLAGS_RELEASE ${CUTLASS_CUDA_CLANG_FLAGS_RELEASE} CACHE INTERNAL "")
set(__CUTLASS_CUDA_CLANG_FLAGS_RELWITHDEBINFO ${CUTLASS_CUDA_CLANG_FLAGS_RELWITHDEBINFO} CACHE INTERNAL "")
set(__CUTLASS_CUDA_CLANG_FLAGS_DEBUG ${CUTLASS_CUDA_CLANG_FLAGS_DEBUG} CACHE INTERNAL "")
set(__CUTLASS_CUDA_NVCC_FLAGS ${CUTLASS_CUDA_NVCC_FLAGS} CACHE INTERNAL "")
set(__CUTLASS_CUDA_NVCC_FLAGS_RELEASE ${CUTLASS_CUDA_NVCC_FLAGS_RELEASE} CACHE INTERNAL "")
set(__CUTLASS_CUDA_NVCC_FLAGS_RELWITHDEBINFO ${CUTLASS_CUDA_NVCC_FLAGS_RELWITHDEBINFO} CACHE INTERNAL "")
set(__CUTLASS_CUDA_NVCC_FLAGS_DEBUG ${CUTLASS_CUDA_NVCC_FLAGS_DEBUG} CACHE INTERNAL "")

function(cutlass_apply_standard_compile_options TARGET)

  if(CUTLASS_CLANG_DEVICE_COMPILE)
    set(CUDA_COMPILE_LANGUAGE CUDA)
    set(_FLAGS ${__CUTLASS_CUDA_FLAGS} ${__CUTLASS_CUDA_CLANG_FLAGS})
    set(_FLAGS_RELEASE ${__CUTLASS_CUDA_FLAGS_RELEASE} ${__CUTLASS_CUDA_CLANG_FLAGS_RELEASE})
    set(_FLAGS_RELWITHDEBINFO ${__CUTLASS_CUDA_FLAGS_RELWITHDEBINFO} ${__CUTLASS_CUDA_CLANG_FLAGS_RELWITHDEBINFO})
    set(_FLAGS_DEBUG ${__CUTLASS_CUDA_FLAGS_DEBUG} ${__CUTLASS_CUDA_CLANG_FLAGS_DEBUG})
  else()
    set(CUDA_COMPILE_LANGUAGE CUDA)
    set(_FLAGS ${__CUTLASS_CUDA_FLAGS} ${__CUTLASS_CUDA_NVCC_FLAGS})
    set(_FLAGS_RELEASE ${__CUTLASS_CUDA_FLAGS_RELEASE} ${__CUTLASS_CUDA_NVCC_FLAGS_RELEASE})
    set(_FLAGS_RELWITHDEBINFO ${__CUTLASS_CUDA_FLAGS_RELWITHDEBINFO} ${__CUTLASS_CUDA_NVCC_FLAGS_RELWITHDEBINFO})
    set(_FLAGS_DEBUG ${__CUTLASS_CUDA_FLAGS_DEBUG} ${__CUTLASS_CUDA_NVCC_FLAGS_DEBUG})
  endif()

  target_link_libraries(${TARGET} PRIVATE CUTLASS)

  target_compile_options(
    ${TARGET}
    PRIVATE
    $<$<COMPILE_LANGUAGE:${CUDA_COMPILE_LANGUAGE}>:${_FLAGS}>
    $<$<COMPILE_LANGUAGE:${CUDA_COMPILE_LANGUAGE}>:$<$<CONFIG:RELEASE>:${_FLAGS_RELEASE}>>
    $<$<COMPILE_LANGUAGE:${CUDA_COMPILE_LANGUAGE}>:$<$<CONFIG:RELWITHDEBINFO>:${_FLAGS_RELWITHDEBINFO}>>
    $<$<COMPILE_LANGUAGE:${CUDA_COMPILE_LANGUAGE}>:$<$<CONFIG:DEBUG>:${_FLAGS_DEBUG}>>
    )

endfunction()

#
# The following items should eventually be pushed into cutlass/CMakeLists.txt
#

# GLOB for CUTLASS header files. Should we use a static list instead?
file(GLOB_RECURSE CUTLASS_INCLUDE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/cutlass/*.h)
file(GLOB_RECURSE CUTLASS_CUTLASS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/include include/cutlass/*.h include/cutlass/*.hpp include/cutlass/*.inl)
file(GLOB_RECURSE CUTLASS_CUTE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/include include/cute/*.h*)
file(GLOB_RECURSE CUTLASS_NVRTC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/test test/unit/nvrtc/kernel/*.h)

###################################################################################################
#
# Define build targets
#
###################################################################################################

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/include REGULAR_EXPRESSION ".*\.h")

add_library(CUTLASS INTERFACE)
add_library(nvidia::cutlass::cutlass ALIAS CUTLASS)
set_target_properties(CUTLASS PROPERTIES EXPORT_NAME cutlass)

set(CUTLASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE PATH "CUTLASS Header Library")

set(CUTLASS_GENERATOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tools/library CACHE INTERNAL "Location of generator scripts")

# The following utility directory is needed even if the tools build is disabled, so it exists here.
set(CUTLASS_TOOLS_UTIL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tools/util/include CACHE INTERNAL "")

include_directories(${CUTLASS_INCLUDE_DIR})

target_compile_features(CUTLASS INTERFACE cxx_std_11)

if (NOT CUTLASS_NAMESPACE STREQUAL "cutlass")
  target_compile_definitions(CUTLASS INTERFACE CUTLASS_NAMESPACE=${CUTLASS_NAMESPACE})
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version_extended.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/cutlass/version_extended.h
  @ONLY)

target_include_directories(
  CUTLASS
  INTERFACE
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CUTLASS_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  )

# Mark CTK headers as system to supress warnings from them
target_include_directories(
  CUTLASS
  SYSTEM INTERFACE
  $<BUILD_INTERFACE:${CUDA_TOOLKIT_ROOT_DIR}/include>
  )

if(CUDA_VERSION VERSION_GREATER_EQUAL 13.0)
  target_include_directories(
    CUTLASS
    SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CUDA_TOOLKIT_ROOT_DIR}/include/cccl>
    )
endif()

install(
  DIRECTORY
  ${CUTLASS_INCLUDE_DIR}/
  ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(
  TARGETS CUTLASS
  EXPORT NvidiaCutlass
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

################################################################################

# Doxygen is available. Generate documentation
if (DOXYGEN_FOUND)
    # DOT is available. Enable graph generation in the documentation
    if (DOXYGEN_DOT_EXECUTABLE)
        set(CUTLASS_ENABLE_DOXYGEN_DOT ON CACHE BOOL "Use dot to generate graphs in the doxygen documentation.")
    else()
        set(CUTLASS_ENABLE_DOXYGEN_DOT OFF CACHE BOOL "Use dot to generate graphs in the doxygen documentation." FORCE)
    endif()

    if (CUTLASS_ENABLE_DOXYGEN_DOT)
        set(HAVE_DOT "YES")
    else()
        set(HAVE_DOT "NO")
    endif()

    # Add custom target for Doxygen.
    add_custom_target(cutlass_docs ${CMAKE_COMMAND} -E env
        "DOT_PATH=${DOXYGEN_DOT_EXECUTABLE}"
        "HAVE_DOT=${HAVE_DOT}"
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM
    )
endif()

if(NOT WIN32)
  # Add common library search paths so executables and libraries can load and run
  # without LD_LIBRARY_PATH being set.
  link_libraries(
    "-Wl,-rpath,'$ORIGIN'"
    "-Wl,-rpath,'$ORIGIN/../lib64'"
    "-Wl,-rpath,'$ORIGIN/../lib'"
    "-Wl,-rpath,'${CUDA_TOOLKIT_ROOT_DIR}/lib64'"
    "-Wl,-rpath,'${CUDA_TOOLKIT_ROOT_DIR}/lib'"
    ${CMAKE_DL_LIBS}
    )
endif()

################################################################################

include(CTest)
enable_testing()

if (CUTLASS_ENABLE_GTEST_UNIT_TESTS)
  if (CUTLASS_USE_SYSTEM_GOOGLETEST)
    find_package(GTest REQUIRED)
  else()
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/googletest.cmake)
  endif()
endif()

if (NOT TARGET test_all)
  add_custom_target(test_all)
endif()

set(CUTLASS_INSTALL_TESTS ON CACHE BOOL "Install test executables")
set(CUTLASS_TEST_EXECUTION_ENVIRONMENT "" CACHE BOOL "Environment in which to invoke unit test executables")

set(CMAKE_TEST_INSTALL_PREFIX test CACHE STRING "Test root install location, relative to CMAKE_INSTALL_PREFIX.")
set(CUTLASS_TEST_INSTALL_PREFIX ${CMAKE_TEST_INSTALL_PREFIX}/cutlass CACHE STRING "Test root install location, relative to CMAKE_INSTALL_PREFIX.")
set(CUTLASS_TEST_INSTALL_BINDIR ${CUTLASS_TEST_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR} CACHE STRING "Test root install location, relative to CMAKE_INSTALL_PREFIX.")
set(CUTLASS_TEST_INSTALL_LIBDIR ${CUTLASS_TEST_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} CACHE STRING "Test root install location, relative to CMAKE_INSTALL_PREFIX.")

install(DIRECTORY DESTINATION ${CUTLASS_TEST_INSTALL_PREFIX})
install(DIRECTORY DESTINATION ${CUTLASS_TEST_INSTALL_BINDIR})
install(DIRECTORY DESTINATION ${CUTLASS_TEST_INSTALL_LIBDIR})
install(DIRECTORY DESTINATION ${CUTLASS_TEST_INSTALL_PREFIX}/ctest)

################################################################################

set(CUTLASS_ENABLE_CUBLAS OFF CACHE BOOL "cuBLAS usage for tests")
set(CUTLASS_ENABLE_CUDNN OFF CACHE BOOL "cuDNN usage for tests")

include(${CMAKE_CURRENT_SOURCE_DIR}/cuBLAS.cmake)

if (CUTLASS_ENABLE_CUBLAS)
  target_compile_definitions(CUTLASS INTERFACE CUTLASS_ENABLE_CUBLAS=1)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cuDNN.cmake)

if (CUTLASS_ENABLE_CUDNN)
  target_compile_definitions(CUTLASS INTERFACE CUTLASS_ENABLE_CUDNN=1)
endif()

################################################################################

set(CUTLASS_DEFAULT_ACTIVE_TEST_SETS "default" CACHE STRING "Default
  activated test sets. In `make test` mode, this string determines the
  active set of tests. In `ctest` mode, this value can be overriden
  with CUTLASS_TEST_SETS environment variable when running the ctest
  executable.")

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CUTLASS_CTEST_TEMPLATE_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/CTestTestfile.configure.cmake)
set(CUTLASS_CTEST_GENERATED_FILES "" CACHE INTERNAL "")

function(cutlass_add_executable_tests NAME TARGET)
#
# Generates test rules for `make test`, `make test_all`, and `ctest` invoked from either the
# <CMAKE_BINARY_DIR> or the <CMAKE_INSTALL_PREFIX>/<CUTLASS_TEST_INSTALL_PREFIX> after installation.
#
# NAME: The base name for the test. Can be run with `make <NAME>` or `ctest -R 'c<NAME>'`.
# TARGET: The target corresponding to the executable under test.
# DISABLE_EXECUTABLE_INSTALL_RULE: An option, if given, that disables creating an install rule for TARGET.
# DEPENDS: A list of targets or files on which this test is dependent.
# DEPENDEES: A list of targets which should depend on this test.
# TEST_COMMAND_OPTIONS: A list of variables (i.e. by reference params) which contain command line arguments
#   to pass to the test executable. A unique test is generated for each set of
#   options given. If this option is not used, a single test with no arguments is generated.
# TEST_COMMAND_OPTIONS_PREFIX: If provided, is added as a prefix to each TEST_COMMAND_OPTIONS value for
#   generating the full variable name to be referenced.
# RESULT_CACHE_FILE: A file to be installed alongside the test executable with pre-computed
#   test results to speed up test runtime.
# TEST_SETS_SUPPORTED: A list of test set names these tests support.
#

  set(options DISABLE_EXECUTABLE_INSTALL_RULE DO_NOT_LOWERCASE_TEST_NAME)
  set(oneValueArgs DISABLE_TESTS RESULT_CACHE_FILE TEST_COMMAND_OPTIONS_PREFIX)
  set(multiValueArgs DEPENDS DEPENDEES TEST_COMMAND_OPTIONS TEST_SETS_SUPPORTED)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED __DISABLE_TESTS)
    set(__DISABLE_TESTS OFF)
  endif()

  set(TEST_EXE $<TARGET_FILE_NAME:${TARGET}>)
  set(TEST_EXE_WORKING_DIRECTORY ./${CMAKE_INSTALL_BINDIR})

  if (NOT DEFINED __TEST_SETS_SUPPORTED)
    set(__TEST_SETS_SUPPORTED ${CUTLASS_DEFAULT_ACTIVE_TEST_SETS})
  endif()

  set(TEST_SETS_SUPPORTED ${__TEST_SETS_SUPPORTED})

  if (__RESULT_CACHE_FILE)

    add_custom_command(
      TARGET ${TARGET}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      ARGS -E copy ${__RESULT_CACHE_FILE} "$<TARGET_FILE_DIR:${TARGET}>"
      )

  endif()

  if (NOT __DISABLE_EXECUTABLE_INSTALL_RULE AND CUTLASS_INSTALL_TESTS)

    # file(RELATIVE_PATH CMAKE_CURRENT_BINARY_RELATIVE_DIR ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})

    install(
      TARGETS ${TARGET}
      RUNTIME DESTINATION ${CUTLASS_TEST_INSTALL_BINDIR}
      )

    if (__RESULT_CACHE_FILE)

     install(
       FILES ${__RESULT_CACHE_FILE}
       DESTINATION ${CUTLASS_TEST_INSTALL_BINDIR}/
       )

    endif()

  endif()

  if (NOT __TEST_COMMAND_OPTIONS)
    set(__TEST_COMMAND_OPTIONS " ")
  endif()

  list(LENGTH __TEST_COMMAND_OPTIONS CMD_COUNT)

  if (CMD_COUNT GREATER 1)
    add_custom_target(${NAME} DEPENDS ${TARGET} ${__DEPENDS})
    foreach(DEPENDEE ${__DEPENDEES})
      add_dependencies(${DEPENDEE} ${NAME})
    endforeach()
  endif()

  if (CUTLASS_INSTALL_TESTS)

    set(_INLINE_PER_TEST_CODE)

    file(READ "${PROJECT_SOURCE_DIR}/cmake/CTestTestfile.test.configure.cmake" _INLINE_PER_TEST_CODE_TEMPLATE)

  endif()

  set(TEST_GROUP_NAME ${NAME})

  # To run the tests from an install package with tests enabled, we need to generate test files
  # that don't rely on the current directory structure in build.

  set(TEST_NAME c${NAME})
  set(TEST_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/ctest/${TEST_NAME})
  file(MAKE_DIRECTORY ${TEST_GEN_DIR})

  set(TEST_EXE_PATH $<TARGET_FILE:${TARGET}>)
  set(TEST_USE_EXTENDED_FORMAT ON)
  configure_file("${CUTLASS_CTEST_TEMPLATE_FILE}" "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.cmake" @ONLY)

  set(TEST_EXE_PATH $<TARGET_FILE_NAME:${TARGET}>)
  set(TEST_USE_EXTENDED_FORMAT OFF) # ctest does not support extended add_test format.
  configure_file("${CUTLASS_CTEST_TEMPLATE_FILE}" "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.install.cmake.in" @ONLY)

  foreach(CMD_OPTIONS_VAR IN LISTS __TEST_COMMAND_OPTIONS)

    if (CMD_COUNT GREATER 1)
      set(TESTCASE_NAME "${NAME}_${CMD_OPTIONS_VAR}")
    else()
      set(TESTCASE_NAME "${NAME}")
    endif()

    if (NOT __DO_NOT_LOWERCASE_TEST_NAME)
      string(TOLOWER "${TESTCASE_NAME}" TESTCASE_NAME)
    endif()

    # The following rigmarole is needed to deal with spaces and possible quotes in
    # command line arguments. The options are passed "by reference" as the actual
    # variable names holding the real options. We then expand these in a way that
    # preserves any quotes. Note, they have to be in this order for it to work for
    # all the use cases below.

    set(TEST_COMMAND_OPTIONS ${${__TEST_COMMAND_OPTIONS_PREFIX}${CMD_OPTIONS_VAR}})
    list(JOIN TEST_COMMAND_OPTIONS " " TEST_COMMAND_OPTIONS)
    separate_arguments(TEST_COMMAND_OPTIONS)

    add_custom_target(
      ${TESTCASE_NAME}
      COMMAND
      ${CUTLASS_TEST_EXECUTION_ENVIRONMENT} $<TARGET_FILE:${TARGET}> ${TEST_COMMAND_OPTIONS}
      DEPENDS
      ${TARGET}
      )

    if (CMD_COUNT GREATER 1)
      add_dependencies(${NAME} ${TESTCASE_NAME})
    endif()

    foreach(DEPENDEE ${__DEPENDEES})
      add_dependencies(${DEPENDEE} ${TESTCASE_NAME})
    endforeach()

    set(TESTCASE_NAME c${TESTCASE_NAME})
    string(CONFIGURE "${_INLINE_PER_TEST_CODE_TEMPLATE}" _TEST_CODE @ONLY)
    file(APPEND "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.cmake" "${_TEST_CODE}")
    file(APPEND "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.install.cmake.in" "${_TEST_CODE}")

  endforeach()

  # The following line imports the tests for immediate run via `make test`.

  include(${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.cmake)

  set(CUTLASS_CTEST_GENERATED_FILES ${CUTLASS_CTEST_GENERATED_FILES};ctest/${TEST_NAME}/CTestTestfile.${TEST_NAME}.cmake CACHE INTERNAL "")

    if (CUTLASS_INSTALL_TESTS)

    file(GENERATE
      OUTPUT "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.install.cmake"
      INPUT "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.install.cmake.in"
      )

    install(
      FILES "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.install.cmake"
      DESTINATION ${CUTLASS_TEST_INSTALL_PREFIX}/ctest/${TEST_NAME}
      RENAME CTestTestfile.${TEST_NAME}.cmake
      )

    endif()

endfunction()



function(cutlass_generate_profiler_tests NAME)

  set(options)
  set(oneValueArgs)
  set(multiValueArgs DEPENDS DEPENDEES CUTLASS_PROFILER_EXTRA_OPTIONS)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT CUTLASS_BUILD_FOR_PROFILER_REGRESSIONS AND NOT CUTLASS_BUILD_FOR_PROFILER_PERFORMANCE_REGRESSIONS)
    return()
  endif()

  install(
    FILES ${CUTLASS_PROFILER_REGRESSION_LIST_FILE}
    DESTINATION ${CMAKE_INSTALL_INFODIR}/cutlass/
    RENAME profiler_regressions.csv
    )

  # Generate cmake test targets for each entry in the testlist csv

  if (NOT EXISTS "${CUTLASS_PROFILER_REGRESSION_LIST_FILE}")
    message(SEND_ERROR "Profiler unit tests list path is invalid: CUTLASS_PROFILER_REGRESSION_LIST_FILE = ${CUTLASS_PROFILER_REGRESSION_LIST_FILE}")
  else()
    message(STATUS "Using ${CUTLASS_PROFILER_REGRESSION_LIST_FILE} to generate profiler-based tests.")
  endif()

  file(STRINGS ${CUTLASS_PROFILER_REGRESSION_LIST_FILE} TEST_LIST)
  foreach(TEST IN LISTS TEST_LIST)
    set(TEMP_TEST ${TEST})
    if ("${TEST}" MATCHES " *cutlass_profiler.*")

      # Generate a flattened name for the test from the test command line.
      string(REPLACE "," ";" TEST_NAME_LIST ${TEMP_TEST})
      string(REGEX REPLACE "\\*" "_" TEST_NAME "${TEMP_TEST}")
      string(REGEX REPLACE "\\\"\\{\\\"\\\"input_params.*\\{.*\\}\\}\\\"" "" TEST_NAME "${TEST_NAME}")
      string(REGEX REPLACE "\\\"\\{\\\"\\\"input_params.*\\{.*\\}\\}\\\"" "" TEST "${TEST}")
      string(REGEX REPLACE "," ";" TEST "${TEST}")
      string(REGEX MATCHALL "[a-zA-Z0-9_=]+" TEST_NAME "${TEST_NAME}")
        list(FILTER TEST_NAME EXCLUDE REGEX "cutlass_profiler|mode=trace|providers=cutlass")
      list(JOIN TEST_NAME "_" TEST_NAME)
      string(REGEX REPLACE "_verification_required=(true|false)" "" TEST_NAME "${TEST_NAME}")
      string(REPLACE "_verification_providers=device" "" TEST_NAME "${TEST_NAME}")
      string(REPLACE "batch_count=" "batch" TEST_NAME "${TEST_NAME}")
      string(REPLACE "cluster_m=" "" TEST_NAME "${TEST_NAME}")
      string(REPLACE "_cluster_n=" "x" TEST_NAME "${TEST_NAME}")
      string(REGEX REPLACE "_cluster_k=[0-9]+" "" TEST_NAME "${TEST_NAME}")
      string(REPLACE "cluster_m_fallback=" "" TEST_NAME "${TEST_NAME}")
      string(REPLACE "_cluster_n_fallback=" "x" TEST_NAME "${TEST_NAME}")
      string(REGEX REPLACE "_cluster_k_fallback=[0-9]+" "" TEST_NAME "${TEST_NAME}")
      string(REPLACE "runtime_input_datatype_a=" "" TEST_NAME "${TEST_NAME}")
      string(REPLACE "runtime_input_datatype_b=" "" TEST_NAME "${TEST_NAME}")
      string(REPLACE "swizzle_size=" "" TEST_NAME "${TEST_NAME}")
      string(REGEX REPLACE "verification_enabled=(true|false)" "" TEST_NAME "${TEST_NAME}")
      string(REGEX REPLACE "warmup_iterations=[0-9]+" "" TEST_NAME "${TEST_NAME}")
      string(REGEX REPLACE "profiling_iterations=[0-9]+" "" TEST_NAME "${TEST_NAME}")
      string(REGEX REPLACE "sleep_duration=[0-9]+" "" TEST_NAME "${TEST_NAME}")
      string(REGEX REPLACE "profiling_enabled=(true|false)" "" TEST_NAME "${TEST_NAME}")
      string(REPLACE "=" "" TEST_NAME "${TEST_NAME}")
      string(REPLACE "_error_on_no_match" "" TEST_NAME "${TEST_NAME}")
      string(REPLACE "_error_if_nothing_is_profiled" "" TEST_NAME "${TEST_NAME}")
      string(REPLACE "kernels" "" TEST_NAME "${TEST_NAME}")
      string(REPLACE "operation" "" TEST_NAME "${TEST_NAME}")

      if (NOT __DO_NOT_LOWERCASE_TEST_NAME)
        string(TOLOWER "${TEST_NAME}" TEST_NAME)
      endif()

      # Munge the test command

      string(REPLACE "cutlass_profiler" "" TEST "${TEST}")
      set(TEST "${TEST}" ${__CUTLASS_PROFILER_EXTRA_OPTIONS} "--junit-output=${TEST_NAME}")
      set(TEST_COMMAND_${TEST_NAME} "${TEST}")
      list(APPEND TEST_COMMAND_VARS ${TEST_NAME})
    endif()

  endforeach()

  cutlass_add_executable_tests(
    ${NAME} cutlass_profiler
    DEPENDS ${__DEPENDS}
    DEPENDEES ${__DEPENDEES}
    TEST_COMMAND_OPTIONS ${TEST_COMMAND_VARS}
    TEST_COMMAND_OPTIONS_PREFIX TEST_COMMAND_
    DISABLE_EXECUTABLE_INSTALL_RULE
    # Uncomment the following line when alloc/dealloc tracking
    # is fixed for all configurations.
    # TEST_SETS_SUPPORTED tmem_alloc_tracking
    )

endfunction()



if (CUTLASS_ENABLE_TOOLS)
  add_subdirectory(tools)
  if (CUTLASS_ENABLE_PROFILER)
    add_dependencies(test_all test_profiler)
  endif()
endif()

if (CUTLASS_ENABLE_EXAMPLES)
  add_subdirectory(examples)
  add_dependencies(test_all test_examples)
endif()

if (CUTLASS_ENABLE_TESTS)
  add_subdirectory(test)
  if (CUTLASS_ENABLE_GTEST_UNIT_TESTS)
  add_dependencies(test_all test_unit)
  endif()
  if (CUTLASS_ENABLE_PROFILER_UNIT_TESTS AND CUTLASS_BUILD_FOR_PROFILER_REGRESSIONS)
    # Generate profiler based unit test
    cutlass_generate_profiler_tests(
      tup
      DEPENDEES test_unit
    )
  endif()

endif()

if (CUTLASS_INSTALL_TESTS)

  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/ctest")

  file(WRITE "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake" "# Generated File\n\n")
  file(APPEND "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake" "cmake_policy(SET CMP0057 NEW) # Allow IN_LIST for if()\n\n")
  file(APPEND "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake" "if (NOT DEFINED ENV{CUTLASS_TEST_SETS})\n")
  file(APPEND "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake" "  set(ENV{CUTLASS_TEST_SETS} ${CUTLASS_DEFAULT_ACTIVE_TEST_SETS})\n")
  file(APPEND "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake" "endif()\n\n")

  foreach(GENERATED_FILE ${CUTLASS_CTEST_GENERATED_FILES})
    file(APPEND "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake" "include(${GENERATED_FILE})\n")
  endforeach()

  install(
    FILES "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake"
    DESTINATION "${CUTLASS_TEST_INSTALL_PREFIX}/"
    )

endif()

################################################################################

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/NvidiaCutlassConfigVersion.cmake
  COMPATIBILITY AnyNewerVersion)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/NvidiaCutlassConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/NvidiaCutlassConfig.cmake
  @ONLY
  )

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/NvidiaCutlassConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/NvidiaCutlassConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NvidiaCutlass/
  )

install(
  EXPORT NvidiaCutlass
  NAMESPACE nvidia::cutlass::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NvidiaCutlass/
  FILE NvidiaCutlassTargets.cmake
  )

################################################################################

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/NvidiaCutlassPackageConfig.cmake)

