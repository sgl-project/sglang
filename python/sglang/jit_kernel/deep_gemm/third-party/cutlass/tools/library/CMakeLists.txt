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

cmake_policy(SET CMP0112 NEW)

include(GNUInstallDirs)

################################################################################

set(CUTLASS_BUILD_MONO_LIBRARY OFF CACHE BOOL 
  "Determines whether the cutlass library is generated as a single file or multiple files.")

################################################################################

add_library(cutlass_library_includes INTERFACE)
add_library(nvidia::cutlass::library::includes ALIAS cutlass_library_includes)
set_target_properties(cutlass_library_includes PROPERTIES EXPORT_NAME library::includes)

target_include_directories(
  cutlass_library_includes
  INTERFACE
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )

target_link_libraries(
  cutlass_library_includes 
  INTERFACE 
  CUTLASS
  cutlass_tools_util_includes
  )

install(
  TARGETS cutlass_library_includes
  EXPORT NvidiaCutlass
  )

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/
  )

add_library(cutlass_library_internal_interface INTERFACE)
add_library(nvidia::cutlass::library::obj_interface ALIAS cutlass_library_internal_interface)

target_include_directories(
  cutlass_library_internal_interface
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  )

target_link_libraries(
  cutlass_library_internal_interface
  INTERFACE
  cutlass_library_includes
  )

################################################################################

function(cutlass_add_cutlass_library)
#
# Generates static and shared libraries with the given SOURCES. The public CMake
# targets produces will be cutlass_library(_${SUFFIX})? and 
# cutlass_library(_${SUFFIX})?_static.
# 
# SUFFIX: An additional string to be joined to the default names. If suffix is given,
#   the generated libraries will be linked as a dependency of the main cutlass library.

  set(options)
  set(oneValueArgs SUFFIX)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(DEFAULT_NAME cutlass_library)

  set(__NAME ${DEFAULT_NAME})
  set(__OUTPUT_NAME cutlass)
  set(__EXPORT_NAME library)

  if (__SUFFIX)
    string(APPEND __NAME _${__SUFFIX})
    string(APPEND __OUTPUT_NAME _${__SUFFIX})
    string(APPEND __EXPORT_NAME _${__SUFFIX})
  endif()

  cutlass_add_library(
    ${__NAME}_objs
    OBJECT
    ${__UNPARSED_ARGUMENTS}
    )  

  target_link_libraries(${__NAME}_objs
    PUBLIC cutlass_library_includes
    PRIVATE cutlass_library_internal_interface
    )

  if (CUTLASS_BUILD_MONO_LIBRARY AND __SUFFIX)

    # If we're only building a single monolithic library then we
    # simply link the generated object files to the default library. 

    target_link_libraries(${DEFAULT_NAME} PRIVATE $<BUILD_INTERFACE:${__NAME}_objs>)
    target_link_libraries(${DEFAULT_NAME}_static PRIVATE $<BUILD_INTERFACE:${__NAME}_objs>)

  else()

    cutlass_add_library(
      ${__NAME} 
      SHARED
      EXPORT_NAME ${__EXPORT_NAME}
      ""
      )

    target_compile_features(${__NAME} INTERFACE cxx_std_17)
    
    set_target_properties(
      ${__NAME}
      PROPERTIES
      OUTPUT_NAME ${__OUTPUT_NAME}
      WINDOWS_EXPORT_ALL_SYMBOLS 1
      )
    
    target_link_libraries(
      ${__NAME}
      PUBLIC cutlass_library_includes
      PRIVATE $<BUILD_INTERFACE:${__NAME}_objs>
      cuda_driver
      )
    
    set_target_properties(${__NAME} PROPERTIES DEBUG_POSTFIX "${CUTLASS_LIBRARY_DEBUG_POSTFIX}")
    
    cutlass_add_library(
      ${__NAME}_static
      STATIC
      EXPORT_NAME ${__EXPORT_NAME}_static
      ""
      )

    target_compile_features(${__NAME}_static INTERFACE cxx_std_17)
    
    if (WIN32)
      set(STATIC_OUTPUT_NAME ${__OUTPUT_NAME}.static)
    else()
      set(STATIC_OUTPUT_NAME ${__OUTPUT_NAME})
    endif()
    
    set_target_properties(
      ${__NAME}_static
      PROPERTIES
      OUTPUT_NAME ${STATIC_OUTPUT_NAME}
      WINDOWS_EXPORT_ALL_SYMBOLS 1
      )
    
    target_link_libraries(
      ${__NAME}_static
      PUBLIC cutlass_library_includes
      PRIVATE $<BUILD_INTERFACE:${__NAME}_objs>
      cuda_driver
      )
    
    set_target_properties(${__NAME}_static PROPERTIES DEBUG_POSTFIX "${CUTLASS_LIBRARY_DEBUG_POSTFIX}")
    
    install(
      TARGETS ${__NAME} ${__NAME}_static
      EXPORT NvidiaCutlass
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      )
    
    if (__SUFFIX)
    
      # The partial libraries generated will be registered as linked libraries
      # to the main cutlass library so users automatically get the necessary link
      # commands to pull in all kernels by default.
    
      target_link_libraries(${DEFAULT_NAME} PUBLIC ${__NAME})
      target_link_libraries(${DEFAULT_NAME}_static PUBLIC ${__NAME}_static)
    
    endif()

  endif()

endfunction()

################################################################################

cutlass_add_cutlass_library(

  src/handle.cu
  src/manifest.cpp
  src/operation_table.cu
  src/singleton.cu
  src/util.cu

  # files split for parallel compilation
  src/reference/gemm_int4.cu
  
  src/reference/block_scaled_gemm_fp4a_vs16.cu   
  src/reference/block_scaled_gemm_fp4a_vs32.cu   
  src/reference/block_scaled_gemm_mixed8bitsa.cu
  src/reference/gemm_f4_f4_f32.cu
  src/reference/gemm_f4_f6_f32.cu
  src/reference/gemm_f4_f8_f32.cu
  src/reference/gemm_f6_f4_f32.cu
  src/reference/gemm_f6_f6_f32.cu
  src/reference/gemm_f6_f8_f32.cu
  src/reference/gemm_f8_f4_f32.cu
  src/reference/gemm_f8_f6_f32.cu

  src/reference/blockwise_gemm_fp8_fp16out.cu   
  src/reference/blockwise_gemm_fp8_fp32out.cu   
  src/reference/blockwise_gemm_fp8_bf16out.cu   
  
  src/reference/gemm_s8_s8_s32.cu
  src/reference/gemm_u8_u8_s32.cu
  src/reference/gemm_int8_interleaved_32.cu
  src/reference/gemm_int8_interleaved_64.cu
  src/reference/gemm_e4m3a_e4m3out.cu
  src/reference/gemm_e5m2a_e4m3out.cu
  src/reference/gemm_e4m3a_e5m2out.cu
  src/reference/gemm_e5m2a_e5m2out.cu
  src/reference/gemm_fp8in_fp16out.cu
  src/reference/gemm_fp8in_bf16out.cu
  src/reference/gemm_fp8in_fp32out.cu
  src/reference/gemm_fp32out.cu
  src/reference/gemm_fp_other.cu
  src/reference/gemm_fp_mixed_input.cu
  src/reference/gemm_int_mixed_input.cu
  src/reference/initialize_reference_operations.cu

  # cutlass reduction instances in cutlass library

  src/reduction/reduction_device.cu
  src/reduction/init_reduction_operations.cu
  
  # cutlass conv reference instances in cutlass library

  src/reference/conv2d.cu
  src/reference/conv3d.cu

  )

# For backward compatibility with the old name
add_library(cutlass_lib ALIAS cutlass_library)
add_library(cutlass_lib_static ALIAS cutlass_library_static)

################################################################################

file(GLOB_RECURSE GENERATOR_PYTHON_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.py)

#
# auto-instantiation of CUTLASS kernels
#

# set cutlass generator compiler version to filter kernels in the generator not supported by a specific toolkit. 
set(CUTLASS_GENERATOR_CUDA_COMPILER_VERSION ${CMAKE_CUDA_COMPILER_VERSION})
set(CUTLASS_LIBRARY_GENERATED_KERNEL_LIST_FILE ${CMAKE_CURRENT_BINARY_DIR}/generated_kernels.txt CACHE STRING "Generated kernel listing file")

set(CUTLASS_LIBRARY_HEURISTICS_TESTLIST_FILE ${CMAKE_CURRENT_BINARY_DIR}/heuristics.csv CACHE STRING "Generated heuristics configs CSV")
set(CUTLASS_LIBRARY_HEURISTICS_GPU "" CACHE STRING "GPU to use for GEMM heuristics")
set(CUTLASS_LIBRARY_HEURISTICS_RESTRICT_KERNELS OFF CACHE BOOL
  "Restrict heuristics kernels to only the default set of kernels emitted by generator.py")


if(CUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE)
  set(HEURISTICS_ARGS
    --heuristics-problems-file "${CUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE}"
    --heuristics-testlist-file "${CUTLASS_LIBRARY_HEURISTICS_TESTLIST_FILE}"
    --heuristics-configs-per-problem "${CUTLASS_LIBRARY_HEURISTICS_CONFIGS_PER_PROBLEM}"
  )

  if(CUTLASS_LIBRARY_HEURISTICS_RESTRICT_KERNELS)
    list(APPEND HEURISTICS_ARGS --heuristics-restrict-kernels)
  endif()

  if(CUTLASS_LIBRARY_HEURISTICS_GPU)
    list(APPEND HEURISTICS_ARGS --heuristics-gpu "${CUTLASS_LIBRARY_HEURISTICS_GPU}")
  endif()
endif()

# --log-level is set to DEBUG to enable printing information about which kernels were excluded
# from generation in /python/cutlass_library/manifest.py. To avoid having this information appear
# in ${CMAKE_CURRENT_BINARY_DIR}/library_instance_generation.log, set this parameter to INFO
execute_process(
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../python/cutlass_library
  COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CUTLASS_LIBRARY_PACKAGE_DIR}:${CUTLASS_NVMMH_PY_DIR}
    ${Python3_EXECUTABLE} ${CUTLASS_SOURCE_DIR}/python/cutlass_library/generator.py
    --operations "${CUTLASS_LIBRARY_OPERATIONS}" 
    --build-dir ${PROJECT_BINARY_DIR}
    --curr-build-dir ${CMAKE_CURRENT_BINARY_DIR}
    --generator-target library
    --architectures "${CUTLASS_NVCC_ARCHS_ENABLED}"
    --kernels "${CUTLASS_LIBRARY_KERNELS}"
    --instantiation-level "${CUTLASS_LIBRARY_INSTANTIATION_LEVEL}"
    --ignore-kernels "${CUTLASS_LIBRARY_IGNORE_KERNELS}"
    --exclude-kernels "${CUTLASS_LIBRARY_EXCLUDE_KERNELS}"
    --kernel-filter-file "${KERNEL_FILTER_FILE}"
    --selected-kernel-list "${CUTLASS_LIBRARY_GENERATED_KERNEL_LIST_FILE}"
    --cuda-version "${CUTLASS_GENERATOR_CUDA_COMPILER_VERSION}"
    --log-level INFO
    --disable-cutlass-package-imports
    ${HEURISTICS_ARGS}
  RESULT_VARIABLE cutlass_lib_INSTANCE_GENERATION_RESULT
  OUTPUT_VARIABLE cutlass_lib_INSTANCE_GENERATION_OUTPUT
  OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/library_instance_generation.log
  ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/library_instance_generation.log
)

if(NOT cutlass_lib_INSTANCE_GENERATION_RESULT EQUAL 0)
  message(FATAL_ERROR "Error generating library instances. See ${CMAKE_CURRENT_BINARY_DIR}/library_instance_generation.log")
endif()

message(STATUS "Completed generation of library instances. See ${CMAKE_CURRENT_BINARY_DIR}/library_instance_generation.log for more information.")

# include auto-instantiated kernels in he CUTLASS Deliverables Library
set(CUTLASS_LIBRARY_MANIFEST_CMAKE_FILE ${CMAKE_CURRENT_BINARY_DIR}/generated/manifest.cmake)
if(EXISTS "${CUTLASS_LIBRARY_MANIFEST_CMAKE_FILE}")
  include(${CUTLASS_LIBRARY_MANIFEST_CMAKE_FILE})
else()
  message(STATUS "auto-generated library manifest cmake file (${CUTLASS_LIBRARY_MANIFEST_CMAKE_FILE}) not found.")
endif()

################################################################################

install(
  FILES ${CUTLASS_LIBRARY_GENERATED_KERNEL_LIST_FILE}
  DESTINATION ${CMAKE_INSTALL_INFODIR}/cutlass/
  )

