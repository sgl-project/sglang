#[=======================================================================[.rst:
SYCLConfig
-------

Library to verify SYCL compatibility of CMAKE_CXX_COMPILER
and passes relevant compiler flags.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``SYCLTOOLKIT_FOUND``
  True if the system has the SYCL library.
``SYCL_COMPILER``
  SYCL compiler executable.
``SYCL_INCLUDE_DIR``
  Include directories needed to use SYCL.
``SYCL_LIBRARY_DIR``
  Libaray directories needed to use SYCL.
``SYCL_FLAGS``
  SYCL specific flags for the compiler.
``SYCL_LANGUAGE_VERSION``
  The SYCL language spec version by Compiler.

#]=======================================================================]

include(${TORCH_INSTALL_PREFIX}/share/cmake/Caffe2/FindSYCLToolkit.cmake)

if(NOT SYCL_FOUND)
  set(SYCLTOOLKIT_FOUND FALSE)
  return()
endif()

if(SYCLTOOLKIT_FOUND)
  return()
endif()

set(SYCLTOOLKIT_FOUND TRUE)

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

if(WIN32)
  set(SYCL_EXECUTABLE_NAME icx)
else()
  set(SYCL_EXECUTABLE_NAME icpx)
endif()

if(NOT SYCL_ROOT)
  execute_process(
    COMMAND which ${SYCL_EXECUTABLE_NAME}
    OUTPUT_VARIABLE SYCL_CMPLR_FULL_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT EXISTS "${SYCL_CMPLR_FULL_PATH}")
    message(WARNING "Cannot find ENV{CMPLR_ROOT} or icpx, please setup SYCL compiler Tool kit environment before building!!")
    return()
  endif()

  get_filename_component(SYCL_BIN_DIR "${SYCL_CMPLR_FULL_PATH}" DIRECTORY)
  set(SYCL_ROOT ${SYCL_BIN_DIR}/..)
endif()

find_program(
  SYCL_COMPILER
  NAMES ${SYCL_EXECUTABLE_NAME}
  PATHS "${SYCL_ROOT}"
  PATH_SUFFIXES bin bin64
  NO_DEFAULT_PATH
  )

string(COMPARE EQUAL "${SYCL_COMPILER}" "" nocmplr)
if(nocmplr)
  set(SYCLTOOLKIT_FOUND False)
  set(SYCL_REASON_FAILURE "SYCL: CMAKE_CXX_COMPILER not set!!")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
endif()

# Function to write a test case to verify SYCL features.

function(SYCL_CMPLR_TEST_WRITE src macro_name)

  set(cpp_macro_if "#if")
  set(cpp_macro_endif "#endif")

  set(SYCL_CMPLR_TEST_CONTENT "")
  string(APPEND SYCL_CMPLR_TEST_CONTENT "#include <iostream>\nusing namespace std;\n")
  string(APPEND SYCL_CMPLR_TEST_CONTENT "int main(){\n")

  # Feature tests goes here

  string(APPEND SYCL_CMPLR_TEST_CONTENT "${cpp_macro_if} defined(${macro_name})\n")
  string(APPEND SYCL_CMPLR_TEST_CONTENT "cout << \"${macro_name}=\"<<${macro_name}<<endl;\n")
  string(APPEND SYCL_CMPLR_TEST_CONTENT "${cpp_macro_endif}\n")

  string(APPEND SYCL_CMPLR_TEST_CONTENT "return 0;}\n")

  file(WRITE ${src} "${SYCL_CMPLR_TEST_CONTENT}")

endfunction()

# Function to Build the feature check test case.

function(SYCL_CMPLR_TEST_BUILD error TEST_SRC_FILE TEST_EXE)

  set(SYCL_CXX_FLAGS_LIST "${SYCL_CXX_FLAGS}")
  string(REPLACE "-Wno-stringop-overflow" "" SYCL_CXX_FLAGS_LIST "${SYCL_CXX_FLAGS_LIST}")
  separate_arguments(SYCL_CXX_FLAGS_LIST)

  execute_process(
    COMMAND "${SYCL_COMPILER}"
    ${SYCL_CXX_FLAGS_LIST}
    ${TEST_SRC_FILE}
    "-o"
    ${TEST_EXE}
    WORKING_DIRECTORY ${SYCL_CMPLR_TEST_DIR}
    OUTPUT_VARIABLE output ERROR_VARIABLE output
    OUTPUT_FILE ${SYCL_CMPLR_TEST_DIR}/Compile.log
    RESULT_VARIABLE result
    TIMEOUT 60
    )

  # Verify if test case build properly.
  if(result)
    message("SYCL: feature test compile failed!!")
    message("compile output is: ${output}")
  endif()

  set(${error} ${result} PARENT_SCOPE)

endfunction()

function(SYCL_CMPLR_TEST_RUN error TEST_EXE)

  execute_process(
    COMMAND ${TEST_EXE}
    WORKING_DIRECTORY ${SYCL_CMPLR_TEST_DIR}
    OUTPUT_VARIABLE output ERROR_VARIABLE output
    RESULT_VARIABLE result
    TIMEOUT 60
    )

  if(test_result)
    set(SYCLTOOLKIT_FOUND False)
    set(SYCL_REASON_FAILURE "SYCL: feature test execution failed!!")
  endif()

  set(test_result "${result}" PARENT_SCOPE)
  set(test_output "${output}" PARENT_SCOPE)

  set(${error} ${result} PARENT_SCOPE)

endfunction()

function(SYCL_CMPLR_TEST_EXTRACT test_output macro_name)

  string(REGEX REPLACE "\n" ";" test_output_list "${test_output}")

  set(${macro_name} "")
  foreach(strl ${test_output_list})
     if(${strl} MATCHES "^${macro_name}=([A-Za-z0-9_]+)$")
       string(REGEX REPLACE "^${macro_name}=" "" extracted_sycl_lang "${strl}")
       set(${macro_name} ${extracted_sycl_lang})
     endif()
  endforeach()

  set(${macro_name} "${extracted_sycl_lang}" PARENT_SCOPE)
endfunction()

set(SYCL_FLAGS "")
set(SYCL_LINK_FLAGS "")
list(APPEND SYCL_FLAGS "-fsycl")
list(APPEND SYCL_LINK_FLAGS "-fsycl")

set(SYCL_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYCL_FLAGS}")

string(FIND "${CMAKE_CXX_FLAGS}" "-Werror" has_werror)
if(${has_werror} EQUAL -1)
  # Create a clean working directory.
  set(SYCL_CMPLR_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/TESTSYCLCMPLR")
  file(REMOVE_RECURSE ${SYCL_CMPLR_TEST_DIR})
  file(MAKE_DIRECTORY ${SYCL_CMPLR_TEST_DIR})

  # Create the test source file
  set(TEST_SRC_FILE "${SYCL_CMPLR_TEST_DIR}/sycl_features.cpp")
  set(TEST_EXE "${TEST_SRC_FILE}.exe")
  SYCL_CMPLR_TEST_WRITE(${TEST_SRC_FILE} "SYCL_LANGUAGE_VERSION")

  # Build the test and create test executable
  SYCL_CMPLR_TEST_BUILD(error ${TEST_SRC_FILE} ${TEST_EXE})
  if(error)
    return()
  endif()

  # Execute the test to extract information
  SYCL_CMPLR_TEST_RUN(error ${TEST_EXE})
  if(error)
    return()
  endif()

  # Extract test output for information
  SYCL_CMPLR_TEST_EXTRACT(${test_output} "SYCL_LANGUAGE_VERSION")

  # As per specification, all the SYCL compatible compilers should
  # define macro  SYCL_LANGUAGE_VERSION
  string(COMPARE EQUAL "${SYCL_LANGUAGE_VERSION}" "" nosycllang)
  if(nosycllang)
    set(SYCLTOOLKIT_FOUND False)
    set(SYCL_REASON_FAILURE "SYCL: It appears that the ${SYCL_COMPILER} does not support SYCL")
    set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
  endif()

  message(DEBUG "The SYCL Language Version is ${SYCL_LANGUAGE_VERSION}")

  # Include in Cache
  set(SYCL_LANGUAGE_VERSION "${SYCL_LANGUAGE_VERSION}" CACHE STRING "SYCL Language version")
endif()

# Create a clean working directory.
set(SYCL_CMPLR_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/TESTSYCLCMPLR")
file(REMOVE_RECURSE ${SYCL_CMPLR_TEST_DIR})
file(MAKE_DIRECTORY ${SYCL_CMPLR_TEST_DIR})
# Create the test source file
set(TEST_SRC_FILE "${SYCL_CMPLR_TEST_DIR}/llvm_features.cpp")
set(TEST_EXE "${TEST_SRC_FILE}.exe")
SYCL_CMPLR_TEST_WRITE(${TEST_SRC_FILE} "__INTEL_LLVM_COMPILER")
# Build the test and create test executable
SYCL_CMPLR_TEST_BUILD(error ${TEST_SRC_FILE} ${TEST_EXE})
if(error)
  message(FATAL_ERROR "Can not build SYCL_CMPLR_TEST")
endif()
# Execute the test to extract information
SYCL_CMPLR_TEST_RUN(error ${TEST_EXE})
if(error)
  message(FATAL_ERROR "Can not run SYCL_CMPLR_TEST")
endif()
# Extract test output for information
SYCL_CMPLR_TEST_EXTRACT(${test_output} "__INTEL_LLVM_COMPILER")

# Check whether the value of __INTEL_LLVM_COMPILER macro was successfully extracted
string(COMPARE EQUAL "${__INTEL_LLVM_COMPILER}" "" nosycllang)
if(nosycllang)
  set(SYCLTOOLKIT_FOUND False)
  set(SYCL_REASON_FAILURE "Can not find __INTEL_LLVM_COMPILER}")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
endif()


# Include in Cache
set(__INTEL_LLVM_COMPILER "${__INTEL_LLVM_COMPILER}" CACHE STRING "Intel llvm compiler")

message(DEBUG "The SYCL compiler is ${SYCL_COMPILER}")
message(DEBUG "The SYCL Flags are ${SYCL_FLAGS}")
