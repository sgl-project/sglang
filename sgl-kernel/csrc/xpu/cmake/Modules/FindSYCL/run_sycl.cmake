##########################################################################
# This file runs the SYCL compiler commands to produce the desired output file
# along with the dependency file needed by CMake to compute dependencies.
# In addition the file checks the output of each command and if the command fails
# it deletes the output files.

# Input variables
#
# verbose:BOOL=<>          OFF: Be as quiet as possible (default)
#                          ON : Describe each step
#
# generated_file:STRING=<> File to generate.  This argument must be passed in.

cmake_policy(PUSH)
cmake_policy(SET CMP0007 NEW)
cmake_policy(SET CMP0010 NEW)
if(NOT generated_file)
  message(FATAL_ERROR "You must specify generated_file on the command line")
endif()

set(CMAKE_COMMAND "@CMAKE_COMMAND@") # path
set(source_file "@source_file@") # path
set(SYCL_generated_dependency_file "@SYCL_generated_dependency_file@") # path
set(cmake_dependency_file "@cmake_dependency_file@") # path
set(SYCL_make2cmake "@SYCL_make2cmake@") # path
set(SYCL_host_compiler "@SYCL_HOST_COMPILER@") # path
set(generated_file_path "@generated_file_path@") # path
set(generated_file_internal "@generated_file@") # path
set(SYCL_executable "@SYCL_EXECUTABLE@") # path
set(SYCL_flags @SYCL_FLAGS@) # list
set(SYCL_include_dirs [==[@SYCL_include_dirs@]==]) # list
set(SYCL_compile_definitions [==[@SYCL_compile_definitions@]==]) # list

list(REMOVE_DUPLICATES SYCL_INCLUDE_DIRS)

set(SYCL_host_compiler_flags "-fsycl-host-compiler-options=")
set(SYCL_include_args)

foreach(dir ${SYCL_include_dirs})
  # Args with spaces need quotes around them to get them to be parsed as a single argument.
  if(dir MATCHES " ")
    list(APPEND SYCL_include_args "-I\"${dir}\"")
    string(APPEND SYCL_host_compiler_flags "-I\"${dir}\" ")
  else()
    list(APPEND SYCL_include_args -I${dir})
    string(APPEND SYCL_host_compiler_flags "-I${dir} ")
  endif()
endforeach()

# Clean up list of compile definitions, add -D flags, and append to SYCL_flags
list(REMOVE_DUPLICATES SYCL_compile_definitions)
foreach(def ${SYCL_compile_definitions})
  list(APPEND SYCL_flags "-D${def}")
endforeach()

# Choose host flags in FindSYCL.cmake
@SYCL_host_flags@

# Adding permissive flag for MSVC build to overcome ambiguous symbol error.
if(WIN32)
  string(APPEND SYCL_host_compiler_flags "/permissive- ")
endif()


list(REMOVE_DUPLICATES CMAKE_HOST_FLAGS)
foreach(flag ${CMAKE_HOST_FLAGS})
  # Extra quotes are added around each flag to help SYCL parse out flags with spaces.
  string(APPEND SYCL_host_compiler_flags "${flag} ")
endforeach()
foreach(def ${SYCL_compile_definitions})
  string(APPEND SYCL_host_compiler_flags "-D${def} ")
endforeach()

# string(APPEND SYCL_host_compiler_flags "\"")
set(SYCL_host_compiler "-fsycl-host-compiler=${SYCL_host_compiler}")

# SYCL_execute_process - Executes a command with optional command echo and status message.
#
#   status  - Status message to print if verbose is true
#   command - COMMAND argument from the usual execute_process argument structure
#   ARGN    - Remaining arguments are the command with arguments
#
#   SYCL_result - return value from running the command
#
# Make this a macro instead of a function, so that things like RESULT_VARIABLE
# and other return variables are present after executing the process.
macro(SYCL_execute_process status command)
  set(_command ${command})
  if(NOT "x${_command}" STREQUAL "xCOMMAND")
    message(FATAL_ERROR "Malformed call to SYCL_execute_process.  Missing COMMAND as second argument. (command = ${command})")
  endif()
  if(verbose)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E echo -- ${status})
    # Now we need to build up our command string.  We are accounting for quotes
    # and spaces, anything else is left up to the user to fix if they want to
    # copy and paste a runnable command line.
    set(SYCL_execute_process_string)
    foreach(arg ${ARGN})
      # If there are quotes, escape them, so they come through.
      string(REPLACE "\"" "\\\"" arg ${arg})
      # Args with spaces need quotes around them to get them to be parsed as a single argument.
      if(arg MATCHES " ")
        list(APPEND SYCL_execute_process_string "\"${arg}\"")
      else()
        list(APPEND SYCL_execute_process_string ${arg})
      endif()
    endforeach()
    # Echo the command
    execute_process(COMMAND ${CMAKE_COMMAND} -E echo ${SYCL_execute_process_string})
  endif()
  # Run the command
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE SYCL_result )
endmacro()

# Delete the target file
SYCL_execute_process(
  "Removing ${generated_file}"
  COMMAND "${CMAKE_COMMAND}" -E remove "${generated_file}"
  )

# Generate the code
if(WIN32)
  set(SYCL_dependency_file_args /clang:-MD /clang:-MF /clang:${SYCL_generated_dependency_file})
else()
  set(SYCL_dependency_file_args -MD -MF "${SYCL_generated_dependency_file}")
endif()
SYCL_execute_process(
  "Generating ${generated_file}"
  COMMAND "${SYCL_executable}"
  ${SYCL_dependency_file_args}
  -c
  "${source_file}"
  -o "${generated_file}"
  ${SYCL_include_args}
  ${SYCL_host_compiler}
  ${SYCL_host_compiler_flags}
  ${SYCL_flags}
  )

if(SYCL_result)
  SYCL_execute_process(
    "Removing ${generated_file}"
    COMMAND "${CMAKE_COMMAND}" -E remove "${generated_file}"
    )
  message(FATAL_ERROR "Error generating file ${generated_file}")
endif()

# Parse *.d file to retrieve included headers. These headers are dependencies
# of custom compilation command. Inform cmake to scan these files and
# retrigger compilation if anything change in these headers.
SYCL_execute_process(
  "Generating temporary cmake readable file: ${cmake_dependency_file}.tmp"
  COMMAND "${CMAKE_COMMAND}"
  -D "input_file:FILEPATH=${SYCL_generated_dependency_file}"
  -D "output_file:FILEPATH=${cmake_dependency_file}.tmp"
  -D "verbose=${verbose}"
  -P "${SYCL_make2cmake}"
  )

if(SYCL_result)
  message(FATAL_ERROR "Error generating ${generated_file}")
endif()

# Update dependencies list. When we remove some header in .cpp, then the
# header should be removed from dependencies list. Or unnecessary re-compilation
# will be triggered, when the header changes.
SYCL_execute_process(
  "Copy if different ${cmake_dependency_file}.tmp to ${cmake_dependency_file}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${cmake_dependency_file}.tmp" "${cmake_dependency_file}"
  )

if(SYCL_result)
  message(FATAL_ERROR "Error generating ${generated_file}")
endif()

SYCL_execute_process(
  "Removing ${cmake_dependency_file}.tmp and ${SYCL_generated_dependency_file}"
  COMMAND "${CMAKE_COMMAND}" -E remove "${cmake_dependency_file}.tmp" "${SYCL_generated_dependency_file}"
  )

if(SYCL_result)
  message(FATAL_ERROR "Error generating ${generated_file}")
endif()

if(verbose)
  message("Generated ${generated_file} successfully.")
endif()

cmake_policy(POP)
