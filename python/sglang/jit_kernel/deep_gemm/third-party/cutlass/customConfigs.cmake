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





# Profiler based functional testing
set(CUTLASS_BUILD_FOR_PROFILER_REGRESSIONS OFF CACHE BOOL "Utilize profiler-based functional regressions")
set(CUTLASS_PROFILER_REGRESSION_TEST_LEVEL  ${CUTLASS_TEST_LEVEL} CACHE STRING "Profiler functional regression test level")

find_package(Python3 3.5 COMPONENTS Interpreter REQUIRED)

function(cutlass_generate_kernel_filter_and_testlist_files)

  set(options)
  set(oneValueArgs TEST_SET_NAME)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CUTLASS_LIBRARY_PACKAGE_DIR}
      ${Python3_EXECUTABLE} ${CUTLASS_SOURCE_DIR}/python/cutlass_library/generator.py 
      --generator-target=${__TEST_SET_NAME} 
      --cuda-version=${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}
      --architectures=${CUTLASS_NVCC_ARCHS}
      --kernels=\*
      --disable-cutlass-package-imports
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RESULT_VARIABLE cutlass_FILTER_GENERATION_RESULT
    OUTPUT_VARIABLE cutlass_FILTER_GENERATION_OUTPUT
    OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/library_filter_generation.log
    ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/library_filter_generation.log
  )

  if(NOT cutlass_FILTER_GENERATION_RESULT EQUAL 0)
    message(FATAL_ERROR "Error generating kernel filters and testlist files. See ${CMAKE_CURRENT_BINARY_DIR}/library_filter_generation.log")
  endif()
endfunction()

if(CUTLASS_BUILD_FOR_PROFILER_REGRESSIONS)

    set(PROFILER_ARCH_LIST 100a 100f 103a 120a 120f 121a)
    if (CUDA_VERSION VERSION_LESS 13.0)
      list(APPEND PROFILER_ARCH_LIST 101a 101f)
    else()
      list(APPEND PROFILER_ARCH_LIST 110a 110f)
    endif()
    foreach(ARCH IN LISTS CUTLASS_NVCC_ARCHS)
      if(NOT (ARCH IN_LIST PROFILER_ARCH_LIST))
        message(FATAL_ERROR "Only SM${PROFILER_ARCH_LIST} compute capabilities are supported with profiler-based unit tests")
      endif()
    endforeach()

    if(CUTLASS_PROFILER_REGRESSION_TEST_LEVEL  EQUAL 0)

      message(STATUS "Building for L0 profiler-based functional regressions")
      cutlass_generate_kernel_filter_and_testlist_files(TEST_SET_NAME kernel_testlist_l0)
      set(KERNEL_FILTER_FILE ${CMAKE_CURRENT_BINARY_DIR}/FK_functional_L0_testlist_SM${CUTLASS_NVCC_ARCHS}_cutlass3x_gemm_kernel_filter.list CACHE STRING "Kernel set")
      set(CUTLASS_PROFILER_REGRESSION_LIST_FILE ${CMAKE_CURRENT_BINARY_DIR}/FK_functional_L0_testlist_SM${CUTLASS_NVCC_ARCHS}_cutlass3x_gemm.csv CACHE STRING "Regression set")

    elseif (CUTLASS_PROFILER_REGRESSION_TEST_LEVEL  EQUAL 1)
      
      message(STATUS "Building for L1 profiler-based functional regressions")
      cutlass_generate_kernel_filter_and_testlist_files(TEST_SET_NAME kernel_testlist_l1)
      set(KERNEL_FILTER_FILE ${CMAKE_CURRENT_BINARY_DIR}/FK_functional_L1_testlist_SM${CUTLASS_NVCC_ARCHS}_cutlass3x_gemm_kernel_filter.list CACHE STRING "Kernel set")
      set(CUTLASS_PROFILER_REGRESSION_LIST_FILE ${CMAKE_CURRENT_BINARY_DIR}/FK_functional_L1_testlist_SM${CUTLASS_NVCC_ARCHS}_cutlass3x_gemm.csv CACHE STRING "Regression set")

    endif()
endif()


