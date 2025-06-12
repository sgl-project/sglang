# This will define the following variables:
# XCCL_FOUND               : True if the system has the XCCL library.
# XCCL_INCLUDE_DIR         : Include directories needed to use XCCL.
# XCCL_LIBRARY_DIR         ：The path to the XCCL library.
# XCCL_LIBRARY             : XCCL library fullname.

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux")
  set(XCCL_FOUND False)
  set(XCCL_NOT_FOUND_MESSAGE "OneCCL library is only supported on Linux!")
  return()
endif()

# we need source OneCCL environment before building.
set(XCCL_ROOT $ENV{CCL_ROOT})

# Find include path from binary.
find_file(
  XCCL_INCLUDE_DIR
  NAMES include
  HINTS ${XCCL_ROOT}
  NO_DEFAULT_PATH
)

# Find include/oneapi path from include path.
find_file(
  XCCL_INCLUDE_ONEAPI_DIR
  NAMES oneapi
  HINTS ${XCCL_ROOT}/include/
  NO_DEFAULT_PATH
)

list(APPEND XCCL_INCLUDE_DIR ${XCCL_INCLUDE_ONEAPI_DIR})

# Find library directory from binary.
find_file(
  XCCL_LIBRARY_DIR
  NAMES lib
  HINTS ${XCCL_ROOT}
  NO_DEFAULT_PATH
)

# Find XCCL library fullname.
find_library(
  XCCL_LIBRARY
  NAMES ccl
  HINTS ${XCCL_LIBRARY_DIR}
  NO_DEFAULT_PATH
)

if((NOT XCCL_INCLUDE_DIR) OR (NOT XCCL_LIBRARY_DIR) OR (NOT XCCL_LIBRARY))
  set(XCCL_FOUND False)
  set(XCCL_NOT_FOUND_MESSAGE "OneCCL library not found!!")
  return()
endif()

SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH}
  "${XCCL_INCLUDE_DIR}")
SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
  "${XCCL_LIBRARY_DIR}")

find_package_handle_standard_args(
  XCCL
  FOUND_VAR XCCL_FOUND
  REQUIRED_VARS XCCL_INCLUDE_DIR XCCL_LIBRARY_DIR XCCL_LIBRARY
  REASON_FAILURE_MESSAGE "${XCCL_NOT_FOUND_MESSAGE}"
)
