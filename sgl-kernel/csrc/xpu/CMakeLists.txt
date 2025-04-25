cmake_minimum_required(VERSION 3.19.2)
project(sgl-kernel)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)

# Torch
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

include(FetchContent)

# cutlass
FetchContent_Declare(
    repo-cutlass-sycl
    GIT_REPOSITORY https://github.com/codeplaysoftware/cutlass-sycl.git
    GIT_TAG        ef9797f4327886ad231bfe853099ca022060c293
    GIT_SHALLOW    OFF
)
FetchContent_Populate(repo-cutlass-sycl)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc
    ${repo-cutlass-fork_SOURCE_DIR}/include
    ${repo-cutlass-fork_SOURCE_DIR}/tools/util/include
)

add_subdirectory(sycl)

set(SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/torch_extension_sycl.cc"
)

message(STATUS "${SOURCES}")

add_library(host_obj OBJECT ${SOURCES})
target_link_libraries(host_obj PRIVATE ${TORCH_LIBRARIES} c10 torch torch_cpu)
target_include_directories(host_obj PRIVATE ${TORCH_INCLUDE_DIRS})
target_include_directories(host_obj PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(host_obj PRIVATE ${Python_LIBRARIES})

get_target_property(SYCL_OBJ_SOURCES sycl_obj SOURCES)
message(STATUS "sycl_obj SOURCES: ${SYCL_OBJ_SOURCES}")

Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI
    $<TARGET_OBJECTS:host_obj>
    $<TARGET_OBJECTS:sycl_obj>
)

# torch related
target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} c10 torch torch_cpu)
target_include_directories(common_ops PRIVATE ${TORCH_INCLUDE_DIRS})

# install
install(TARGETS common_ops LIBRARY DESTINATION sgl_kernel)

install(DIRECTORY "${repo-cutlass-fork_SOURCE_DIR}/include/cute/"
        DESTINATION "deep_gemm/include/cute")

install(DIRECTORY "${repo-cutlass-fork_SOURCE_DIR}/include/cutlass/"
        DESTINATION "deep_gemm/include/cutlass")
