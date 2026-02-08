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

# This example demonstrates building against and utilizing an
# installed CUTLASS library. Unlike the other examples, this example
# is not built within the standard CUTLASS CMake flow, but rather
# relies on a pre-installed CUTLASS package. If the CUTLASS package is
# not installed in a standard location, provide the root location of
# the install with "-DCUTLASS_DIR=<root-of-cutlass-install>" CMake
# argument or any of the other features CMake allows for specifying
# locations of installed CMake packages via find_package().

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(cutlass_import_example VERSION 0.2 LANGUAGES CXX CUDA)

if (CUTLASS_DIR)
  message(STATUS "Using CUTLASS specified at ${CUTLASS_DIR}.")
  list(APPEND CMAKE_PREFIX_PATH ${CUTLASS_DIR})
endif()

find_package(NvidiaCutlass 2.0 REQUIRED)

message(STATUS "CUTLASS: ${NvidiaCutlass_DIR}")

add_executable(example)

target_sources(example PRIVATE main.cpp)

target_include_directories(
  example
  PRIVATE
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

target_link_libraries(
  example 
  PRIVATE
  nvidia::cutlass::cutlass
  nvidia::cutlass::library
  )
