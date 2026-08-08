cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(sgl_kernel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE}
            -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PY_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS ${TORCH_PY_PREFIX})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PY_PREFIX}/Torch)
find_package(Torch REQUIRED)

# Paths relative to this file (csrc/cpu/riscv64/)
get_filename_component(CPU_SRC_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/.."   ABSOLUTE)
get_filename_component(SGLKERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INSTALL_PREFIX}/include
    ${Python_INCLUDE_DIRS}
    ${SGLKERNEL_DIR}/csrc
    ${SGLKERNEL_DIR}/include
    ${CPU_SRC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

link_directories("/usr/lib/riscv64-linux-gnu")

# Conda / venv paths (no NUMA on RISC-V)
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_LIB_DIR "$ENV{CONDA_PREFIX}/lib")
    message(STATUS "Using Conda lib dir: ${CONDA_LIB_DIR}")
    link_directories(${CONDA_LIB_DIR})
    include_directories("$ENV{CONDA_PREFIX}/include")
else()
    if(DEFINED ENV{VIRTUAL_ENV})
        set(VENV_LIB_DIR "$ENV{VIRTUAL_ENV}/lib")
        message(STATUS "Using venv lib dir: ${VENV_LIB_DIR}")
        link_directories(${VENV_LIB_DIR})
        include_directories("$ENV{VIRTUAL_ENV}/include")
    endif()
endif()

# Compiler check
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "RISC-V build requires Clang for RVV intrinsics. Current: ${CMAKE_CXX_COMPILER_ID}")
elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.0)
    message(FATAL_ERROR "Clang >= 19.0 required for RVV. Found: ${CMAKE_CXX_COMPILER_VERSION}")
endif()

# Auto-detect VLEN and ISA extensions from /proc/cpuinfo
set(RVV_HAS_ZVFH OFF)
if(NOT DEFINED RVV_VECTOR_BITS)
    if(EXISTS "/proc/cpuinfo")
        file(READ "/proc/cpuinfo" _cpuinfo)
        if(_cpuinfo MATCHES "zvfh")
            set(RVV_HAS_ZVFH ON)
        endif()
        if(_cpuinfo MATCHES "zvl1024b")
            set(RVV_VECTOR_BITS 1024)
        elseif(_cpuinfo MATCHES "zvl512b")
            set(RVV_VECTOR_BITS 512)
        elseif(_cpuinfo MATCHES "zvl256b")
            set(RVV_VECTOR_BITS 256)
        elseif(_cpuinfo MATCHES "zvl128b")
            set(RVV_VECTOR_BITS 128)
        else()
            set(RVV_VECTOR_BITS 256)
        endif()
    else()
        set(RVV_VECTOR_BITS 256)
    endif()
else()
    if(EXISTS "/proc/cpuinfo")
        file(READ "/proc/cpuinfo" _cpuinfo)
        if(_cpuinfo MATCHES "zvfh")
            set(RVV_HAS_ZVFH ON)
        endif()
    endif()
endif()
message(STATUS "RISC-V: VLEN=${RVV_VECTOR_BITS} (-mrvv-vector-bits=${RVV_VECTOR_BITS})")
message(STATUS "RISC-V: zvfh support = ${RVV_HAS_ZVFH}")

# L1 cache size (KB) for K-tiling. Override: cmake -DRVV_L1_CACHE_KB=64
if(NOT DEFINED RVV_L1_CACHE_KB)
    set(RVV_L1_CACHE_KB 32)
endif()
message(STATUS "RISC-V: L1 cache = ${RVV_L1_CACHE_KB}KB (override: -DRVV_L1_CACHE_KB=N)")

# Source collection
# Top-level cpu/ files replaced by riscv64/ equivalents or not supported on RISC-V
set(RISCV64_SKIP_TOPLEVEL
    activation decode extend gemm gemm_int8 norm rope
    moe moe_int8 moe_fp8 moe_int4
    gemm_fp8 gemm_int4
    bmm
    qkv_proj preprocessor conv3d
    shm interface flash_attn
    torch_extension_cpu
)

file(GLOB_RECURSE SOURCES "${CPU_SRC_DIR}/*.cpp")
# Exclude arch-specific subdirectories
list(FILTER SOURCES EXCLUDE REGEX "${CPU_SRC_DIR}/(x86_64|aarch64|riscv64)/")
# Exclude top-level files replaced by riscv64/ equivalents
foreach(_basename IN LISTS RISCV64_SKIP_TOPLEVEL)
    list(FILTER SOURCES EXCLUDE REGEX "${CPU_SRC_DIR}/${_basename}\\.cpp$")
endforeach()
# Add RVV-specific sources
file(GLOB RVV_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
list(APPEND SOURCES ${RVV_SOURCES})

# FP8 flush-to-zero
if(NOT DEFINED ENV{SGLANG_CPU_FP8_CVT_FTZ})
    set(ENV{SGLANG_CPU_FP8_CVT_FTZ} "1")
endif()
if("$ENV{SGLANG_CPU_FP8_CVT_FTZ}" STREQUAL "1")
    message(STATUS "Enabling macro: SGLANG_CPU_FP8_CVT_FTZ")
    add_compile_definitions(SGLANG_CPU_FP8_CVT_FTZ)
endif()

add_compile_options(-O3 -Wno-unknown-pragmas -fopenmp)
if(RVV_HAS_ZVFH)
    set(RVV_MARCH "rv64gcv_zvfh")
    set(RVV_ZVFH_ENABLED "ON")
    add_compile_options(-march=${RVV_MARCH} -mabi=lp64d -mrvv-vector-bits=${RVV_VECTOR_BITS})
    add_compile_definitions(RVV_HAS_ZVFH=1)
else()
    set(RVV_MARCH "rv64gcv")
    set(RVV_ZVFH_ENABLED "OFF")
    add_compile_options(-march=${RVV_MARCH} -mabi=lp64d -mrvv-vector-bits=${RVV_VECTOR_BITS})
endif()
add_compile_definitions(CPU_CAPABILITY_RVV)
add_compile_definitions(RVV_L1_CACHE_KB=${RVV_L1_CACHE_KB})

message(STATUS "RVV configure summary:")
message(STATUS "  march: ${RVV_MARCH}")
message(STATUS "  mabi: lp64d")
message(STATUS "  vector-bits: ${RVV_VECTOR_BITS}")
message(STATUS "  zvfh enabled: ${RVV_ZVFH_ENABLED}")

message(STATUS "Build Architecture: riscv64 (RVV enabled)")

# Link libraries: PyTorch + OpenMP (no NUMA)
set(LINK_LIBS ${TORCH_LIBRARIES})
set(OMP_LIB_PATH "$ENV{RISCV_OMP_LIB_PATH}")
if(EXISTS "${OMP_LIB_PATH}")
    list(APPEND LINK_LIBS "${OMP_LIB_PATH}")
else()
    find_library(OMP_LIB omp)
    if(OMP_LIB)
        list(APPEND LINK_LIBS ${OMP_LIB})
    endif()
endif()

Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})
target_link_libraries(common_ops PRIVATE ${LINK_LIBS})
target_include_directories(common_ops PRIVATE ${TORCH_INCLUDE_DIRS})

install(TARGETS common_ops LIBRARY DESTINATION sgl_kernel)
