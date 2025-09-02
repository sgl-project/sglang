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

# Derive ABI flag from PyTorch to match the installed wheel
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE TORCH_CXX11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE TORCH_ABI_RESULT
)
if(NOT TORCH_ABI_RESULT EQUAL 0)
  # Fallback if PyTorch query fails
  set(TORCH_CXX11_ABI 1)
  message(WARNING "Could not detect PyTorch ABI, defaulting to _GLIBCXX_USE_CXX11_ABI=1")
endif()
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})

# Select source files based on backend
set(SOURCES mxfp4_grouped.cpp)

if(USE_CUTLASS_ADVANCED)
  list(APPEND SOURCES mxfp4_grouped_cutlass.cu)
  add_definitions(-DUSE_CUTLASS_FP4)
  message(STATUS "Using advanced CUTLASS FP4 implementation")
elseif(USE_CUTLASS_FP4)
  list(APPEND SOURCES mxfp4_grouped_impl.cu)
  add_definitions(-DUSE_CUTLASS_FP4)
  message(STATUS "Using CUTLASS FP4 backend")
elseif(USE_FLASHINFER_BACKEND)
  list(APPEND SOURCES mxfp4_grouped_impl.cu)
  add_definitions(-DUSE_FLASHINFER_BACKEND)
  message(STATUS "Using FlashInfer backend")
else()
  list(APPEND SOURCES mxfp4_grouped_impl.cu)
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

# Set CUDA architectures - honor TORCH_CUDA_ARCH_LIST if set
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Check for TORCH_CUDA_ARCH_LIST environment variable
  if(DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    string(REPLACE " " ";" ARCH_LIST "$ENV{TORCH_CUDA_ARCH_LIST}")
    set(CMAKE_CUDA_ARCHITECTURES "")
    foreach(ARCH IN LISTS ARCH_LIST)
      # Convert format like "12.0" to "120"
      string(REPLACE "." "" ARCH_CLEAN "${ARCH}")
      list(APPEND CMAKE_CUDA_ARCHITECTURES "${ARCH_CLEAN}")
    endforeach()
    message(STATUS "Using CUDA architectures from TORCH_CUDA_ARCH_LIST: ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    set(CMAKE_CUDA_ARCHITECTURES "120")  # Default to SM120 for RTX 5090
    message(STATUS "Using default CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
  endif()
endif()
set_target_properties(_mxfp4_kernels PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
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
  # Check for custom FlashInfer location via environment
  if(DEFINED ENV{FLASHINFER_DIR})
    find_library(FLASHINFER_LIB flashinfer 
      PATHS $ENV{FLASHINFER_DIR}/lib $ENV{FLASHINFER_DIR}
      NO_DEFAULT_PATH)
  else()
    find_library(FLASHINFER_LIB flashinfer PATHS /usr/local/lib)
  endif()
  if(FLASHINFER_LIB)
    target_link_libraries(_mxfp4_kernels PRIVATE ${FLASHINFER_LIB})
    message(STATUS "Found FlashInfer library: ${FLASHINFER_LIB}")
  else()
    message(WARNING "FlashInfer library not found, backend may not work")
  endif()
endif()

# Installation - use proper subdirectory
if(NOT DEFINED CMAKE_INSTALL_LIBDIR)
  set(CMAKE_INSTALL_LIBDIR "lib")
endif()
install(TARGETS _mxfp4_kernels
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
)