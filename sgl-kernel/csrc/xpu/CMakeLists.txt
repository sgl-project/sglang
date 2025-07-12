cmake_minimum_required(VERSION 3.19.2)
project(sgl_kernel)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)

# Torch
find_package(Python3 COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

execute_process(
  COMMAND ${PYTHON3_EXECUTABLE}
          -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_PY_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS ${TORCH_PY_PREFIX})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PY_PREFIX}/Torch)
find_package(Torch REQUIRED)


set(SGL_OPS_XPU_ROOT ${PROJECT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH ${SGL_OPS_XPU_ROOT}/cmake/Modules)

include(${SGL_OPS_XPU_ROOT}/cmake/SYCL.cmake)
include(${SGL_OPS_XPU_ROOT}/cmake/BuildFlags.cmake)

include(FetchContent)

# # cutlass
# FetchContent_Declare(
#     repo-cutlass-sycl
#     GIT_REPOSITORY https://github.com/codeplaysoftware/cutlass-sycl.git
#     GIT_TAG        ef9797f4327886ad231bfe853099ca022060c293
#     GIT_SHALLOW    OFF
# )
# FetchContent_Populate(repo-cutlass-sycl)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc
    # ${repo-cutlass-sycl_SOURCE_DIR}/include
    # ${repo-cutlass-sycl_SOURCE_DIR}/tools/util/include
)

add_subdirectory(${SGL_OPS_XPU_ROOT}/src)
