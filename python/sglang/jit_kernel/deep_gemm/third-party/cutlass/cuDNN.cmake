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

if(DEFINED CUDNN_ENABLED)
    set(CUTLASS_ENABLE_CUDNN ${CUDNN_ENABLED} CACHE BOOL "Enable CUTLASS to build with cuDNN library.")
endif()

if(DEFINED CUTLASS_ENABLE_CUDNN AND NOT CUTLASS_ENABLE_CUDNN)
  return()
endif()
  
message(STATUS "Configuring cuDNN ...")

find_path(
    _CUDNN_INCLUDE_DIR cudnn.h
    PATHS
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    $ENV{CUDNN_PATH}/include
    $ENV{CUDA_PATH}/include
    ${CUDNN_PATH}/include
    /usr/include)

find_library(
    _CUDNN_LIBRARY cudnn
    HINTS
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
    ${CUDA_TOOLKIT_ROOT_DIR}/lib
    $ENV{CUDNN_PATH}/lib64
    $ENV{CUDNN_PATH}/lib/x64
    $ENV{CUDNN_PATH}/lib
    $ENV{CUDA_PATH}/lib64
    $ENV{CUDA_PATH}/lib/x64
    $ENV{CUDA_PATH}/lib
    ${CUDNN_PATH}/lib64
    ${CUDNN_PATH}/lib/x64
    ${CUDNN_PATH}/lib
    /usr/lib/x86_64-linux-gnu
    /usr/lib)

if(_CUDNN_INCLUDE_DIR AND _CUDNN_LIBRARY)

    message(STATUS "cuDNN: ${_CUDNN_LIBRARY}")
    message(STATUS "cuDNN: ${_CUDNN_INCLUDE_DIR}")
    
    set(CUDNN_FOUND ON CACHE INTERNAL "cuDNN Library Found")

else()

    message(STATUS "cuDNN not found.")
    set(CUDNN_FOUND OFF CACHE INTERNAL "cuDNN Library Found")

endif()

set(CUTLASS_ENABLE_CUDNN ${CUDNN_FOUND} CACHE BOOL "Enable CUTLASS to build with cuDNN library.")

if (CUTLASS_ENABLE_CUDNN AND NOT TARGET cudnn)

  set(CUDNN_INCLUDE_DIR ${_CUDNN_INCLUDE_DIR})
  set(CUDNN_LIBRARY ${_CUDNN_LIBRARY})

  if(WIN32)
    add_library(cudnn STATIC IMPORTED GLOBAL)
  else()
    add_library(cudnn SHARED IMPORTED GLOBAL)
  endif()

  add_library(nvidia::cudnn ALIAS cudnn)

  set_property(
    TARGET cudnn
    PROPERTY IMPORTED_LOCATION
    ${CUDNN_LIBRARY})
    
  target_include_directories(
    cudnn
    INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CUDNN_INCLUDE_DIR}>)

endif()

if(CUTLASS_ENABLE_CUDNN AND NOT CUDNN_FOUND)
  message(FATAL_ERROR "CUTLASS_ENABLE_CUDNN enabled but cuDNN library could not be found.")
endif()

message(STATUS "Configuring cuDNN ... done.")
