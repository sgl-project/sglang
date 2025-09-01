cmake_minimum_required(VERSION 3.20)
project(mxfp4_grouped LANGUAGES CXX CUDA)

# Options for different backends
option(USE_CUTLASS_FP4 "Use CUTLASS 3.x FP4 weight-only kernels" OFF)
option(USE_FLASHINFER_BACKEND "Use FlashInfer FP4 backend" OFF)
option(USE_CUTLASS_ADVANCED "Use advanced CUTLASS implementation" OFF)

# Find required packages
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find CUTLASS if requested
if(USE_CUTLASS_FP4 OR USE_CUTLASS_ADVANCED)
  find_path(CUTLASS_INCLUDE_DIR 
    NAMES cutlass/cutlass.h
    PATHS 
      ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../third_party/cutlass/include
      /usr/local/include
      /opt/cutlass/include
  )
  if(NOT CUTLASS_INCLUDE_DIR)
    message(WARNING "CUTLASS not found, falling back to stub implementation")
    set(USE_CUTLASS_FP4 OFF)
    set(USE_CUTLASS_ADVANCED OFF)
  endif()
endif()

# Set standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Use the same ABI as PyTorch
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

# Select source files based on backend
set(SOURCES mxfp4_grouped.cpp)

if(USE_CUTLASS_ADVANCED)
  list(APPEND SOURCES mxfp4_grouped_cutlass.cu)
  add_definitions(-DUSE_CUTLASS_FP4)
  message(STATUS "Using advanced CUTLASS FP4 implementation")
elseif(USE_CUTLASS_FP4)
  list(APPEND SOURCES mxfp4_grouped.cu)
  add_definitions(-DUSE_CUTLASS_FP4)
  message(STATUS "Using CUTLASS FP4 backend")
elseif(USE_FLASHINFER_BACKEND)
  list(APPEND SOURCES mxfp4_grouped.cu)
  add_definitions(-DUSE_FLASHINFER_BACKEND)
  message(STATUS "Using FlashInfer backend")
else()
  list(APPEND SOURCES mxfp4_grouped.cu)
  message(STATUS "Using stub implementation (build sanity only)")
endif()

# Create shared library
add_library(_mxfp4_kernels SHARED ${SOURCES})

# Include directories
target_include_directories(_mxfp4_kernels PRIVATE 
  ${TORCH_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
)

# Add CUTLASS include path if found
if(CUTLASS_INCLUDE_DIR)
  target_include_directories(_mxfp4_kernels PRIVATE ${CUTLASS_INCLUDE_DIR})
endif()

# Set CUDA architectures (SM120 for RTX 5090)
set_target_properties(_mxfp4_kernels PROPERTIES
  CUDA_ARCHITECTURES "120"
  PREFIX ""  # Remove lib prefix for Python module
  POSITION_INDEPENDENT_CODE ON
)

# Compile flags
target_compile_options(_mxfp4_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    --use_fast_math
    -lineinfo
    --expt-extended-lambda
    --expt-relaxed-constexpr
    --threads 0
    --maxrregcount=255
    $<$<BOOL:${USE_CUTLASS_FP4}>:-DUSE_CUTLASS_FP4>
    $<$<BOOL:${USE_FLASHINFER_BACKEND}>:-DUSE_FLASHINFER_BACKEND>
  >
  $<$<COMPILE_LANGUAGE:CXX>:
    -O3
    -fPIC
    -Wall
    -march=native
  >
)

# Link libraries
target_link_libraries(_mxfp4_kernels PRIVATE 
  ${TORCH_LIBRARIES}
  ${Python3_LIBRARIES}
)

# Link FlashInfer if using that backend
if(USE_FLASHINFER_BACKEND)
  find_library(FLASHINFER_LIB flashinfer PATHS /usr/local/lib)
  if(FLASHINFER_LIB)
    target_link_libraries(_mxfp4_kernels PRIVATE ${FLASHINFER_LIB})
  endif()
endif()

# Installation
install(TARGETS _mxfp4_kernels
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
)