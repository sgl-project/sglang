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

set(CUTLASS_NVRTC_HAS_CUDA_FP16 FALSE)

# CUTLASS NVRTC target
macro(add_nvrtc_headers BASE_DIR FILES)
  foreach(CUTLASS_FILE ${FILES})
    set(OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/nvrtc/${CUTLASS_FILE}")

    string(REPLACE "/" "_" VARIABLE_NAME ${CUTLASS_FILE})
    string(REPLACE "." "_" VARIABLE_NAME ${VARIABLE_NAME})

    add_custom_command(OUTPUT ${OUTPUT_FILE}
      COMMAND ${CMAKE_COMMAND}
      -DFILE_IN="${BASE_DIR}/${CUTLASS_FILE}"
      -DFILE_OUT="${OUTPUT_FILE}"
      -DVARIABLE_NAME="${VARIABLE_NAME}"
      -P ${PROJECT_SOURCE_DIR}/bin2hex.cmake
      DEPENDS ${BASE_DIR}/${CUTLASS_FILE}
    )

    list(APPEND GENERATED_HEADER_FILES "${OUTPUT_FILE}")

    string(APPEND NVRTC_INCLUDES_HEADERS "#include <${OUTPUT_FILE}>\n")
    string(APPEND NVRTC_INCLUDES_STRINGS "  ${VARIABLE_NAME},\n")
    string(APPEND NVRTC_INCLUDES_NAMES "  \"${CUTLASS_FILE}\",\n")
  endforeach()
endmacro()

string(APPEND NVRTC_INCLUDES_STRINGS "char const *kCutlassHeaders[] = {\n")
string(APPEND NVRTC_INCLUDES_NAMES "char const *kCutlassHeaderNames[] = {\n")


file(GLOB_RECURSE NVRTC_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} kernel/thread/*.hpp)

add_nvrtc_headers(${PROJECT_SOURCE_DIR}/include "${CUTLASS_CUTLASS};${CUTLASS_UTIL};${CUTLASS_DEVICE}")
add_nvrtc_headers(${PROJECT_SOURCE_DIR}/include "${CUTLASS_CUTE}")
add_nvrtc_headers(${PROJECT_SOURCE_DIR}/test "${CUTLASS_NVRTC};${CUTLASS_UTIL};${CUTLASS_DEVICE}")
add_nvrtc_headers(${CMAKE_CURRENT_SOURCE_DIR} "${NVRTC_SOURCES}")

add_nvrtc_headers("${CMAKE_CURRENT_SOURCE_DIR}/stdlib" "assert.h;stdint.h")
if(CUTLASS_NVRTC_HAS_CUDA_FP16)
  add_nvrtc_headers("${CMAKE_CURRENT_SOURCE_DIR}/stdlib" "cuda_fp16.h;cuda_fp16.hpp")
endif()

string(APPEND NVRTC_INCLUDES_STRINGS "};\n")
string(APPEND NVRTC_INCLUDES_NAMES "};\n")

string(APPEND NVRTC_INCLUDES_STRINGS "const size_t kCutlassHeaderCount = sizeof(kCutlassHeaders) / sizeof(*kCutlassHeaders);\n")

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cutlass/nvrtc/environment.cpp"
  "#include <cutlass/nvrtc/environment.h>\n"
  "${NVRTC_INCLUDES_HEADERS}"
  "\n"
  "namespace cutlass {\n"
  "namespace nvrtc {\n"
  "\n"
  "${NVRTC_INCLUDES_STRINGS}"
  "\n"
  "${NVRTC_INCLUDES_NAMES}"
  "\n"
  "} // namespace nvrtc\n"
  "} // namespace cutlass\n"
)

set(GENERATED_SOURCE_FILES "${CMAKE_CURRENT_BINARY_DIR}/cutlass/nvrtc/environment.cpp")

source_group("Generated\\Header Files" FILES ${GENERATED_HEADER_FILES})
source_group("Generated\\Source Files" FILES ${GENERATED_SOURCE_FILES})

cutlass_add_library(cutlass_nvrtc STATIC
  cutlass/nvrtc/environment.h
  ${GENERATED_SOURCE_FILES}
  ${GENERATED_HEADER_FILES}
  )

target_include_directories(
  cutlass_nvrtc 
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
  )

target_link_libraries(cutlass_nvrtc PUBLIC nvidia::nvrtc nvidia::cuda_driver)

add_subdirectory(thread)

add_custom_target(cutlass_test_unit_nvrtc DEPENDS cutlass_test_unit_nvrtc_thread)
add_custom_target(test_unit_nvrtc DEPENDS test_unit_nvrtc_thread)
