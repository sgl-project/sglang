cmake_minimum_required(VERSION 3.16)
project(SYCLPart LANGUAGES CXX)


# —————————————————————————————————————————————
execute_process(
  COMMAND which icpx
  OUTPUT_VARIABLE ICPCX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT ICPCX_PATH)
  message(FATAL_ERROR "icpx not found in PATH; please source oneAPI setvars.sh or adjust your PATH")
endif()
message(STATUS "Found icpx at: ${ICPCX_PATH}")

set(SYCL_OBJECTS "")
file(GLOB SYCL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.sycl")

foreach(SYCL_FILE ${SYCL_SOURCES})
    get_filename_component(FNAME ${SYCL_FILE} NAME_WE)
    set(OBJ_FILE ${CMAKE_CURRENT_BINARY_DIR}/${FNAME}.o)
    set_source_files_properties(${OBJ_FILE} PROPERTIES GENERATED TRUE)

    add_custom_command(
        OUTPUT ${OBJ_FILE}
        COMMAND ${ICPCX_PATH} -fsycl -c ${SYCL_FILE} -o ${OBJ_FILE}
        DEPENDS ${SYCL_FILE}
        COMMENT "Compiling ${SYCL_FILE} with icpx"
    )
    list(APPEND SYCL_OBJECTS ${OBJ_FILE})
endforeach()

message(STATUS "SYCL_OBJECTS = ${SYCL_OBJECTS}")
add_library(sycl_obj OBJECT ${SYCL_OBJECTS})

set_target_properties(sycl_obj PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(sycl_obj PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(sycl_obj PRIVATE ${Python_LIBRARIES})

target_include_directories(sycl_obj PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(sycl_obj PRIVATE ${TORCH_LIBRARIES} c10 torch torch_cpu ${SYCL_LIBRARY})
