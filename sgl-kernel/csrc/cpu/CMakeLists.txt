cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(sgl_kernel)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Torch
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE}
            -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PY_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS ${TORCH_PY_PREFIX})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PY_PREFIX}/Torch)

include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INSTALL_PREFIX}/include
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/csrc
)

# Platform-specific library directory
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(PLAT_LIB_DIR "/usr/lib/x86_64-linux-gnu")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(PLAT_LIB_DIR "/usr/lib/aarch64-linux-gnu")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le|ppc64")
    set(PLAT_LIB_DIR "/usr/lib/powerpc64le-linux-gnu")
else()
    set(PLAT_LIB_DIR "/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")
endif()
link_directories(${PLAT_LIB_DIR})

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_compile_options(
    -O3
    -Wno-unknown-pragmas
    -march=native
    -fopenmp
)

add_library(sgl_kernel_common_ops SHARED ${SOURCES})
pybind11_add_module(common_ops ${SOURCES})

execute_process(
    COMMAND ${Python_EXECUTABLE}
            -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
    OUTPUT_VARIABLE TORCH_LIB_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

file(GLOB TORCH_PYTHON_LIB "${TORCH_LIB_PREFIX}/libtorch_python.so")
target_link_libraries(sgl_kernel_common_ops
    PRIVATE
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
    ${TORCH_PYTHON_LIB}
)

set_target_properties(sgl_kernel_common_ops PROPERTIES
    INSTALL_RPATH "$ORIGIN/../torch/lib"
    PREFIX ""
    OUTPUT_NAME "common_ops"
)

target_compile_definitions(sgl_kernel_common_ops PRIVATE TORCH_EXTENSION_NAME=common_ops TORCH_API_INCLUDE_EXTENSION_H)

# Install
install(TARGETS sgl_kernel_common_ops
    LIBRARY DESTINATION ${Python3_SITEARCH}/sgl_kernel
)
