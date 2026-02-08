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

include(CTest)

set(CUTLASS_UNIT_TEST_COMMON_DIR ${CMAKE_CURRENT_LIST_DIR}/common)

cutlass_add_library(
  cutlass_test_unit_infra 
  OBJECT
  common/filter_architecture.cpp
  )

target_link_libraries(
  cutlass_test_unit_infra
  PUBLIC
  CUTLASS
  cutlass_tools_util_includes
  $<$<BOOL:${CUTLASS_ENABLE_CUBLAS}>:nvidia::cublas>
  GTest::gtest
  cudart
  cuda_driver
  )

cutlass_add_library(
  cutlass_test_unit_infra_lib
  OBJECT
  test_unit.cpp
  )

target_link_libraries(
  cutlass_test_unit_infra_lib
  PUBLIC
  cutlass_test_unit_infra
  )

set(CUTLASS_TEST_UNIT_RESULTS_CACHE_DIR ${CMAKE_CURRENT_LIST_DIR}/data/hashes)

function(cutlass_test_unit_add_executable NAME)

  set(options WITHOUT_CUDA DO_NOT_LOWERCASE_TEST_NAME)
  set(oneValueArgs)
  set(multiValueArgs TEST_SETS_SUPPORTED EXTRA_INCLUDE_DIRS)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  cutlass_add_executable(${NAME} ${__UNPARSED_ARGUMENTS} BATCH_SOURCES OFF)

  target_compile_definitions(${NAME} PUBLIC CUTLASS_TARGET_NAME="${NAME}")

  target_include_directories(
    ${NAME}
    PRIVATE
    ${CUTLASS_UNIT_TEST_COMMON_DIR}
    ${__EXTRA_INCLUDE_DIRS}
  )
  if (__WITHOUT_CUDA)
    # Avoid CUDA dependencies for host-only unit tests that provide the
    # WITHOUT_CUDA argument.
    target_link_libraries(
      ${NAME}
      PUBLIC
      GTest::gtest 
    )
  else()
    target_link_libraries(
      ${NAME}
      PRIVATE
      cutlass_test_unit_infra
      cutlass_test_unit_infra_lib
    )
  endif()

  if (CUTLASS_ENABLE_OPENMP_TESTS AND OpenMP_CXX_FOUND)
    target_link_libraries(${NAME} PRIVATE OpenMP::OpenMP_CXX)
  endif()

  string(REGEX REPLACE cutlass_ "" NAME_STEM ${NAME})

  set(RESULT_CACHE_FILE "${CUTLASS_TEST_UNIT_RESULTS_CACHE_DIR}/cached_results_${NAME}.txt")

  if (EXISTS ${RESULT_CACHE_FILE})
    set(RESULT_CACHE_FILE_ARGS RESULT_CACHE_FILE ${RESULT_CACHE_FILE})
  endif()

  set(CUTLASS_TEST_UNIT_TEST_COMMAND_OPTIONS --gtest_output=xml:${NAME_STEM}.gtest.xml)

  if (__DO_NOT_LOWERCASE_TEST_NAME)
    set(DO_NOT_LOWERCASE_TEST_NAME DO_NOT_LOWERCASE_TEST_NAME)
  else()
    set(DO_NOT_LOWERCASE_TEST_NAME)
  endif()
  
  cutlass_add_executable_tests(
    ${NAME_STEM} ${NAME}
    TEST_SETS_SUPPORTED ${__TEST_SETS_SUPPORTED}
    TEST_COMMAND_OPTIONS CUTLASS_TEST_UNIT_TEST_COMMAND_OPTIONS
    ${RESULT_CACHE_FILE_ARGS}
    ${DO_NOT_LOWERCASE_TEST_NAME}
    )

endfunction()
  

function(cutlass_test_unit_add_executable_split_file NAME)
  # Given the input arguments to cutlass_test_unit_add_executable, creates
  # a new set of arguments in which each file has at most one TEST definition,
  # and calls cutlass_test_unit_add_executable with the newly-formed arguments.
  # The goal of this is to reduce the memory consumed while building CUTLASS
  # tests with a high degree of parallelism while not requiring developers
  # to split unit tests across multiple files artificially.

  # Get all arguments other than the NAME of the target
  list(SUBLIST ARGV 1 ${ARGC} SUBARGV)

  if (CUTLASS_UNIT_TEST_SPLIT_FILES)
    execute_process(
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      COMMAND ${Python3_EXECUTABLE} ${CUTLASS_SOURCE_DIR}/tools/util/scripts/split_test_cmake.py
        ${NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}
        --src_files ${SUBARGV}
        --dst_dir ${CMAKE_CURRENT_BINARY_DIR}
      RESULT_VARIABLE cutlass_test_SPLIT_RESULT
      OUTPUT_VARIABLE cutlass_test_SPLIT_OUTPUT
      OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/test_split_files.txt
      ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/test_split_error.log
    )

    if(NOT cutlass_test_SPLIT_RESULT EQUAL 0)
      message(FATAL_ERROR "Error splitting unit test. See ${CMAKE_CURRENT_BINARY_DIR}/test_split_error.log")
    endif()

    # Forward the values printed by split_test_cmake.py as arguments to cutlass_test_unit_add_executable.
    # We additionally specify to add -I${CMAKE_CURRENT_SOURCE_DIR} to the target. This is necessary because
    # the splitting process writes new files to ${CMAKE_CURRENT_BINARY_DIR}, but many CUTLASS unit tests
    # use relative imports for including testbeds (e.g., '#include "../testbed.hpp"'). These headers are
    # not written to ${CMAKE_CURRENT_BINARY_DIR} during the splitting process, so we must indicate that
    # headers can also be searched for from ${CMAKE_CURRENT_SOURCE_DIR}.
    file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/test_split_files.txt NEW_OPTIONS)
    cutlass_test_unit_add_executable(${NAME} ${NEW_OPTIONS} EXTRA_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
  else()
    # Simply pass arguments through
    cutlass_test_unit_add_executable(${ARGV})
  endif()
endfunction() 

add_custom_target(cutlass_test_unit)
add_custom_target(test_unit)

set(SUBDIRS
  core
  cute
  gemm
  conv
  layout
  transform
  epilogue
  reduction
  util
  pipeline
  substrate
  cluster_launch
  )

if(TARGET nvidia::nvrtc AND TARGET nvidia::cuda_driver)
  set(CUTLASS_NVRTC_ENABLE_INIT ON)
else()
  set(CUTLASS_NVRTC_ENABLE_INIT OFF)
endif()  

set(CUTLASS_NVRTC_ENABLE ${CUTLASS_NVRTC_ENABLE_INIT} CACHE BOOL "Enable NVRTC support")

if (CUTLASS_NVRTC_ENABLE)
  list(APPEND SUBDIRS nvrtc)
endif()

foreach(SUBDIR ${SUBDIRS})

  add_subdirectory(${SUBDIR})
  add_dependencies(cutlass_test_unit cutlass_test_unit_${SUBDIR})
  add_dependencies(test_unit test_unit_${SUBDIR})

endforeach()

