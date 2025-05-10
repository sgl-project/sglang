cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(sgl_kernel)

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
find_package(Torch REQUIRED)

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

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/activation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bmm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/decode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/extend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm_int8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moe.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moe_int8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/norm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/qkv_proj.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/topk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rope.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/torch_extension_cpu.cpp
)

add_compile_options(
    -O3
    -Wno-unknown-pragmas
    -march=native
    -fopenmp
)

add_library(sgl_kernel_common_ops SHARED ${SOURCES})

target_link_libraries(sgl_kernel_common_ops
    PRIVATE
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
    c10
)

set_target_properties(sgl_kernel_common_ops PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../torch/lib"
    PREFIX ""
    OUTPUT_NAME "sgl_kernel.common_ops"
)

target_compile_definitions(sgl_kernel_common_ops PRIVATE TORCH_API_INCLUDE_EXTENSION_H)

# Install
install(TARGETS sgl_kernel_common_ops
    LIBRARY DESTINATION ${Python3_SITEARCH}
)
