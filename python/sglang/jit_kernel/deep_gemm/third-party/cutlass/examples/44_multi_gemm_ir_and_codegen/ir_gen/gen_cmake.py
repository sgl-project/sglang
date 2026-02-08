#################################################################################################
#
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
#
#################################################################################################

class gen_build_sys:
    def __init__(self, cutlass_deps_dir, output_dir = "../"):
        self.output_dir = output_dir
        self.cutlass_deps_dir = cutlass_deps_dir

    def gen_top(self):
        code = ""
        code += '''\
# Auto Generated code - Do not edit.

cmake_minimum_required(VERSION 3.8)
project(CUTLASS_MULTI_GEMMS LANGUAGES CXX CUDA)
find_package(CUDAToolkit)
set(CUDA_PATH ${{CUDA_TOOLKIT_ROOT_DIR}})
set(CUTLASS_PATH \"{cutlass_deps_dir}/include\")
set(CUTLASS_UTIL_PATH \"{cutlass_deps_dir}/tools/util/include\")
list(APPEND CMAKE_MODULE_PATH ${{CUDAToolkit_LIBRARY_DIR}})
'''.format(cutlass_deps_dir=self.cutlass_deps_dir)

        code += '''\
set(GPU_ARCHS \"\" CACHE STRING
  \"List of GPU architectures (semicolon-separated) to be compiled for.\")

if(\"${GPU_ARCHS}\" STREQUAL \"\")
	set(GPU_ARCHS \"70\")
endif()

foreach(arch ${GPU_ARCHS})
  set(CMAKE_CUDA_FLAGS \"${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}\")
	if(SM STREQUAL 70 OR SM STREQUAL 75)
    set(CMAKE_C_FLAGS    \"${CMAKE_C_FLAGS}    -DWMMA\")
    set(CMAKE_CXX_FLAGS  \"${CMAKE_CXX_FLAGS}  -DWMMA\")
    set(CMAKE_CUDA_FLAGS \"${CMAKE_CUDA_FLAGS} -DWMMA\")
	endif()
endforeach()

set(CMAKE_C_FLAGS    \"${CMAKE_C_FLAGS}\")
set(CMAKE_CXX_FLAGS  \"${CMAKE_CXX_FLAGS}\")
set(CMAKE_CUDA_FLAGS \"${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall\")

set(CMAKE_C_FLAGS_DEBUG    \"${CMAKE_C_FLAGS_DEBUG}    -Wall -O0\")
set(CMAKE_CXX_FLAGS_DEBUG  \"${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0\")
set(CMAKE_CUDA_FLAGS_DEBUG \"${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall\")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_STANDARD STREQUAL \"11\")
  set(CMAKE_CUDA_FLAGS \"${CMAKE_CUDA_FLAGS} --expt-extended-lambda\")
  set(CMAKE_CUDA_FLAGS \"${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr\")
endif()

set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} -g -O3\")
set(CMAKE_CUDA_FLAGS \"${CMAKE_CUDA_FLAGS} -Xcompiler -O3\")
set(CMAKE_CUDA_FLAGS \"${CMAKE_CUDA_FLAGS} -Xcompiler=-fno-strict-aliasing\")

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

set(COMMON_LIB_DIRS
  ${CUDAToolkit_LIBRARY_DIR}
)
list(APPEND COMMON_HEADER_DIRS ${CUTLASS_PATH})
list(APPEND COMMON_HEADER_DIRS ${CUTLASS_UTIL_PATH})
'''
        code += '''\
include_directories(
  ${COMMON_HEADER_DIRS}
)

link_directories(
  ${COMMON_LIB_DIRS}
)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-DGOOGLE_CUDA=1)

add_executable(sample
  sample/sample.cu
  one_api.cu
)
target_link_libraries(sample PRIVATE
  -lcudart
  -lnvToolsExt
  ${CMAKE_THREAD_LIBS_INIT}
)

if(NOT DEFINED LIB_INSTALL_PATH)
	set(LIB_INSTALL_PATH ${CMAKE_CURRENT_BINARY_DIR})
endif()
'''
        return code

    def gen_code(self):
        top_code = self.gen_top()
        with open(self.output_dir + "CMakeLists.txt", "w") as f:
            f.write(top_code)
