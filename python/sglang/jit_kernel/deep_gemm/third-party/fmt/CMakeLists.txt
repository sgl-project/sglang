cmake_minimum_required(VERSION 3.8...3.28)

# Fallback for using newer policies on CMake <3.12.
if (${CMAKE_VERSION} VERSION_LESS 3.12)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif ()

# Determine if fmt is built as a subproject (using add_subdirectory)
# or if it is the master project.
if (NOT DEFINED FMT_MASTER_PROJECT)
  set(FMT_MASTER_PROJECT OFF)
  if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(FMT_MASTER_PROJECT ON)
    message(STATUS "CMake version: ${CMAKE_VERSION}")
  endif ()
endif ()

# Joins arguments and places the results in ${result_var}.
function(join result_var)
  set(result "")
  foreach (arg ${ARGN})
    set(result "${result}${arg}")
  endforeach ()
  set(${result_var} "${result}" PARENT_SCOPE)
endfunction()

# DEPRECATED! Should be merged into add_module_library.
function(enable_module target)
  if (MSVC)
    if(NOT CMAKE_GENERATOR STREQUAL "Ninja")
      set(BMI_DIR "${CMAKE_CURRENT_BINARY_DIR}")
      file(TO_NATIVE_PATH "${BMI_DIR}/${target}.ifc" BMI)
      target_compile_options(${target}
        PRIVATE /interface /ifcOutput ${BMI}
        INTERFACE /reference fmt=${BMI})
      set_target_properties(${target} PROPERTIES ADDITIONAL_CLEAN_FILES ${BMI})
      set_source_files_properties(${BMI} PROPERTIES GENERATED ON)
    endif()
  endif ()
endfunction()

set(FMT_USE_CMAKE_MODULES FALSE)
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.28 AND
    CMAKE_GENERATOR STREQUAL "Ninja")
  set(FMT_USE_CMAKE_MODULES TRUE)
endif ()

# Adds a library compiled with C++20 module support.
# `enabled` is a CMake variables that specifies if modules are enabled.
# If modules are disabled `add_module_library` falls back to creating a
# non-modular library.
#
# Usage:
#   add_module_library(<name> [sources...] FALLBACK [sources...] [IF enabled])
function(add_module_library name)
  cmake_parse_arguments(AML "" "IF" "FALLBACK" ${ARGN})
  set(sources ${AML_UNPARSED_ARGUMENTS})

  add_library(${name})
  set_target_properties(${name} PROPERTIES LINKER_LANGUAGE CXX)

  if (NOT ${${AML_IF}})
    # Create a non-modular library.
    target_sources(${name} PRIVATE ${AML_FALLBACK})
    set_target_properties(${name} PROPERTIES CXX_SCAN_FOR_MODULES OFF)
    return()
  endif ()

  # Modules require C++20.
  target_compile_features(${name} PUBLIC cxx_std_20)
  if (CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(${name} PUBLIC -fmodules-ts)
  endif ()

  if (FMT_USE_CMAKE_MODULES)
    target_sources(${name} PUBLIC FILE_SET fmt TYPE CXX_MODULES
                   FILES ${sources})
  else()
    # `std` is affected by CMake options and may be higher than C++20.
    get_target_property(std ${name} CXX_STANDARD)

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      set(pcms)
      foreach (src ${sources})
        get_filename_component(pcm ${src} NAME_WE)
        set(pcm ${pcm}.pcm)

        # Propagate -fmodule-file=*.pcm to targets that link with this library.
        target_compile_options(
          ${name} PUBLIC -fmodule-file=${CMAKE_CURRENT_BINARY_DIR}/${pcm})

        # Use an absolute path to prevent target_link_libraries prepending -l
        # to it.
        set(pcms ${pcms} ${CMAKE_CURRENT_BINARY_DIR}/${pcm})
        add_custom_command(
          OUTPUT ${pcm}
          COMMAND ${CMAKE_CXX_COMPILER}
                  -std=c++${std} -x c++-module --precompile -c
                  -o ${pcm} ${CMAKE_CURRENT_SOURCE_DIR}/${src}
                  "-I$<JOIN:$<TARGET_PROPERTY:${name},INCLUDE_DIRECTORIES>,;-I>"
          # Required by the -I generator expression above.
          COMMAND_EXPAND_LISTS
          DEPENDS ${src})
      endforeach ()

      # Add .pcm files as sources to make sure they are built before the library.
      set(sources)
      foreach (pcm ${pcms})
        get_filename_component(pcm_we ${pcm} NAME_WE)
        set(obj ${pcm_we}.o)
        # Use an absolute path to prevent target_link_libraries prepending -l.
        set(sources ${sources} ${pcm} ${CMAKE_CURRENT_BINARY_DIR}/${obj})
        add_custom_command(
          OUTPUT ${obj}
          COMMAND ${CMAKE_CXX_COMPILER} $<TARGET_PROPERTY:${name},COMPILE_OPTIONS>
                  -c -o ${obj} ${pcm}
          DEPENDS ${pcm})
      endforeach ()
    endif ()
    target_sources(${name} PRIVATE ${sources})
  endif()
endfunction()

include(CMakeParseArguments)

# Sets a cache variable with a docstring joined from multiple arguments:
#   set(<variable> <value>... CACHE <type> <docstring>...)
# This allows splitting a long docstring for readability.
function(set_verbose)
  # cmake_parse_arguments is broken in CMake 3.4 (cannot parse CACHE) so use
  # list instead.
  list(GET ARGN 0 var)
  list(REMOVE_AT ARGN 0)
  list(GET ARGN 0 val)
  list(REMOVE_AT ARGN 0)
  list(REMOVE_AT ARGN 0)
  list(GET ARGN 0 type)
  list(REMOVE_AT ARGN 0)
  join(doc ${ARGN})
  set(${var} ${val} CACHE ${type} ${doc})
endfunction()

# Set the default CMAKE_BUILD_TYPE to Release.
# This should be done before the project command since the latter can set
# CMAKE_BUILD_TYPE itself (it does so for nmake).
if (FMT_MASTER_PROJECT AND NOT CMAKE_BUILD_TYPE)
  set_verbose(CMAKE_BUILD_TYPE Release CACHE STRING
              "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or "
              "CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
endif ()

project(FMT CXX)
include(GNUInstallDirs)
set_verbose(FMT_INC_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE STRING
            "Installation directory for include files, a relative path that "
            "will be joined with ${CMAKE_INSTALL_PREFIX} or an absolute path.")

option(FMT_PEDANTIC "Enable extra warnings and expensive tests." OFF)
option(FMT_WERROR "Halt the compilation with an error on compiler warnings."
       OFF)

# Options that control generation of various targets.
option(FMT_DOC "Generate the doc target." ${FMT_MASTER_PROJECT})
option(FMT_INSTALL "Generate the install target." ON)
option(FMT_TEST "Generate the test target." ${FMT_MASTER_PROJECT})
option(FMT_FUZZ "Generate the fuzz target." OFF)
option(FMT_CUDA_TEST "Generate the cuda-test target." OFF)
option(FMT_OS "Include OS-specific APIs." ON)
option(FMT_MODULE "Build a module instead of a traditional library." OFF)
option(FMT_SYSTEM_HEADERS "Expose headers with marking them as system." OFF)
option(FMT_UNICODE "Enable Unicode support." ON)

if (FMT_TEST AND FMT_MODULE)
  # The tests require {fmt} to be compiled as traditional library
  message(STATUS "Testing is incompatible with build mode 'module'.")
endif ()
set(FMT_SYSTEM_HEADERS_ATTRIBUTE "")
if (FMT_SYSTEM_HEADERS)
  set(FMT_SYSTEM_HEADERS_ATTRIBUTE SYSTEM)
endif ()
if (CMAKE_SYSTEM_NAME STREQUAL "MSDOS")
  set(FMT_TEST OFF)
  message(STATUS "MSDOS is incompatible with gtest")
endif ()

# Get version from base.h
file(READ include/fmt/base.h base_h)
if (NOT base_h MATCHES "FMT_VERSION ([0-9]+)([0-9][0-9])([0-9][0-9])")
  message(FATAL_ERROR "Cannot get FMT_VERSION from base.h.")
endif ()
# Use math to skip leading zeros if any.
math(EXPR CPACK_PACKAGE_VERSION_MAJOR ${CMAKE_MATCH_1})
math(EXPR CPACK_PACKAGE_VERSION_MINOR ${CMAKE_MATCH_2})
math(EXPR CPACK_PACKAGE_VERSION_PATCH ${CMAKE_MATCH_3})
join(FMT_VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.
                 ${CPACK_PACKAGE_VERSION_PATCH})
message(STATUS "{fmt} version: ${FMT_VERSION}")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif ()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/support/cmake")

include(CheckCXXCompilerFlag)
include(JoinPaths)

if (FMT_MASTER_PROJECT AND NOT DEFINED CMAKE_CXX_VISIBILITY_PRESET)
  set_verbose(CMAKE_CXX_VISIBILITY_PRESET hidden CACHE STRING
              "Preset for the export of private symbols")
  set_property(CACHE CMAKE_CXX_VISIBILITY_PRESET PROPERTY STRINGS
               hidden default)
endif ()

if (FMT_MASTER_PROJECT AND NOT DEFINED CMAKE_VISIBILITY_INLINES_HIDDEN)
  set_verbose(CMAKE_VISIBILITY_INLINES_HIDDEN ON CACHE BOOL
              "Whether to add a compile flag to hide symbols of inline functions")
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(PEDANTIC_COMPILE_FLAGS -pedantic-errors -Wall -Wextra -pedantic
      -Wold-style-cast -Wundef
      -Wredundant-decls -Wwrite-strings -Wpointer-arith
      -Wcast-qual -Wformat=2 -Wmissing-include-dirs
      -Wcast-align
      -Wctor-dtor-privacy -Wdisabled-optimization
      -Winvalid-pch -Woverloaded-virtual
      -Wconversion -Wundef
      -Wno-ctor-dtor-privacy -Wno-format-nonliteral)
  if (NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.6)
      set(PEDANTIC_COMPILE_FLAGS ${PEDANTIC_COMPILE_FLAGS}
         -Wno-dangling-else -Wno-unused-local-typedefs)
  endif ()
  if (NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
      set(PEDANTIC_COMPILE_FLAGS ${PEDANTIC_COMPILE_FLAGS} -Wdouble-promotion
          -Wtrampolines -Wzero-as-null-pointer-constant -Wuseless-cast
          -Wvector-operation-performance -Wsized-deallocation -Wshadow)
  endif ()
  if (NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
      set(PEDANTIC_COMPILE_FLAGS ${PEDANTIC_COMPILE_FLAGS} -Wshift-overflow=2
          -Wduplicated-cond)
      # Workaround for GCC regression
      # [12/13/14/15 regression] New (since gcc 12) false positive null-dereference in vector.resize
      # https://gcc.gnu.org/bugzilla/show_bug.cgi?id=108860
      if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12.0)
        set(PEDANTIC_COMPILE_FLAGS ${PEDANTIC_COMPILE_FLAGS} -Wnull-dereference)
      endif ()
  endif ()
  set(WERROR_FLAG -Werror)
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(PEDANTIC_COMPILE_FLAGS -Wall -Wextra -pedantic -Wconversion -Wundef
      -Wdeprecated -Wweak-vtables -Wshadow
      -Wno-gnu-zero-variadic-macro-arguments)
  check_cxx_compiler_flag(-Wzero-as-null-pointer-constant HAS_NULLPTR_WARNING)
  if (HAS_NULLPTR_WARNING)
    set(PEDANTIC_COMPILE_FLAGS ${PEDANTIC_COMPILE_FLAGS}
        -Wzero-as-null-pointer-constant)
  endif ()
  set(WERROR_FLAG -Werror)
endif ()

if (MSVC)
  set(PEDANTIC_COMPILE_FLAGS /W3)
  set(WERROR_FLAG /WX)
endif ()

if (FMT_MASTER_PROJECT AND CMAKE_GENERATOR MATCHES "Visual Studio")
  # If Microsoft SDK is installed create script run-msbuild.bat that
  # calls SetEnv.cmd to set up build environment and runs msbuild.
  # It is useful when building Visual Studio projects with the SDK
  # toolchain rather than Visual Studio.
  include(FindSetEnv)
  if (WINSDK_SETENV)
    set(MSBUILD_SETUP "call \"${WINSDK_SETENV}\"")
  endif ()
  # Set FrameworkPathOverride to get rid of MSB3644 warnings.
  join(netfxpath
       "C:\\Program Files\\Reference Assemblies\\Microsoft\\Framework\\"
       ".NETFramework\\v4.0")
  file(WRITE run-msbuild.bat "
    ${MSBUILD_SETUP}
    ${CMAKE_MAKE_PROGRAM} -p:FrameworkPathOverride=\"${netfxpath}\" %*")
endif ()

function(add_headers VAR)
  set(headers ${${VAR}})
  foreach (header ${ARGN})
    set(headers ${headers} include/fmt/${header})
  endforeach()
  set(${VAR} ${headers} PARENT_SCOPE)
endfunction()

# Define the fmt library, its includes and the needed defines.
set(FMT_HEADERS)
add_headers(FMT_HEADERS args.h base.h chrono.h color.h compile.h core.h format.h
                        format-inl.h os.h ostream.h printf.h ranges.h std.h
                        xchar.h)
set(FMT_SOURCES src/format.cc)

add_module_library(fmt src/fmt.cc FALLBACK
                   ${FMT_SOURCES} ${FMT_HEADERS} README.md ChangeLog.md
                   IF FMT_MODULE)
add_library(fmt::fmt ALIAS fmt)
if (FMT_MODULE)
  enable_module(fmt)
elseif (FMT_OS)
  target_sources(fmt PRIVATE src/os.cc)
else()
  target_compile_definitions(fmt PRIVATE FMT_OS=0)
endif ()

if (FMT_WERROR)
  target_compile_options(fmt PRIVATE ${WERROR_FLAG})
endif ()
if (FMT_PEDANTIC)
  target_compile_options(fmt PRIVATE ${PEDANTIC_COMPILE_FLAGS})
endif ()

if (cxx_std_11 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
  target_compile_features(fmt PUBLIC cxx_std_11)
else ()
  message(WARNING "Feature cxx_std_11 is unknown for the CXX compiler")
endif ()

target_include_directories(fmt ${FMT_SYSTEM_HEADERS_ATTRIBUTE} BEFORE PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${FMT_INC_DIR}>)

set(FMT_DEBUG_POSTFIX d CACHE STRING "Debug library postfix.")

set_target_properties(fmt PROPERTIES
  VERSION ${FMT_VERSION} SOVERSION ${CPACK_PACKAGE_VERSION_MAJOR}
  PUBLIC_HEADER "${FMT_HEADERS}"
  DEBUG_POSTFIX "${FMT_DEBUG_POSTFIX}"

  # Workaround for Visual Studio 2017:
  # Ensure the .pdb is created with the same name and in the same directory
  # as the .lib. Newer VS versions already do this by default, but there is no
  # harm in setting it for those too. Ignored by other generators.
  COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  COMPILE_PDB_NAME "fmt"
  COMPILE_PDB_NAME_DEBUG "fmt${FMT_DEBUG_POSTFIX}")

# Set FMT_LIB_NAME for pkg-config fmt.pc. We cannot use the OUTPUT_NAME target
# property because it's not set by default.
set(FMT_LIB_NAME fmt)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(FMT_LIB_NAME ${FMT_LIB_NAME}${FMT_DEBUG_POSTFIX})
endif ()

if (BUILD_SHARED_LIBS)
  target_compile_definitions(fmt PRIVATE FMT_LIB_EXPORT INTERFACE FMT_SHARED)
endif ()
if (FMT_SAFE_DURATION_CAST)
  target_compile_definitions(fmt PUBLIC FMT_SAFE_DURATION_CAST)
endif ()

add_library(fmt-header-only INTERFACE)
add_library(fmt::fmt-header-only ALIAS fmt-header-only)

if (NOT MSVC)
  # Unicode is always supported on compilers other than MSVC.
elseif (FMT_UNICODE)
  # Unicode support requires compiling with /utf-8.
  target_compile_options(fmt PUBLIC $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>)
  target_compile_options(fmt-header-only INTERFACE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>)
else ()
  target_compile_definitions(fmt PUBLIC FMT_UNICODE=0)
endif ()

target_compile_definitions(fmt-header-only INTERFACE FMT_HEADER_ONLY=1)
target_compile_features(fmt-header-only INTERFACE cxx_std_11)

target_include_directories(fmt-header-only
  ${FMT_SYSTEM_HEADERS_ATTRIBUTE} BEFORE INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${FMT_INC_DIR}>)

# Install targets.
if (FMT_INSTALL)
  include(CMakePackageConfigHelpers)
  set_verbose(FMT_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/fmt CACHE STRING
              "Installation directory for cmake files, a relative path that "
              "will be joined with ${CMAKE_INSTALL_PREFIX} or an absolute "
              "path.")
  set(version_config ${PROJECT_BINARY_DIR}/fmt-config-version.cmake)
  set(project_config ${PROJECT_BINARY_DIR}/fmt-config.cmake)
  set(pkgconfig ${PROJECT_BINARY_DIR}/fmt.pc)
  set(targets_export_name fmt-targets)

  set_verbose(FMT_LIB_DIR ${CMAKE_INSTALL_LIBDIR} CACHE STRING
              "Installation directory for libraries, a relative path that "
              "will be joined to ${CMAKE_INSTALL_PREFIX} or an absolute path.")

  set_verbose(FMT_PKGCONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/pkgconfig CACHE STRING
              "Installation directory for pkgconfig (.pc) files, a relative "
              "path that will be joined with ${CMAKE_INSTALL_PREFIX} or an "
              "absolute path.")

  # Generate the version, config and target files into the build directory.
  write_basic_package_version_file(
    ${version_config}
    VERSION ${FMT_VERSION}
    COMPATIBILITY AnyNewerVersion)

  join_paths(libdir_for_pc_file "\${exec_prefix}" "${FMT_LIB_DIR}")
  join_paths(includedir_for_pc_file "\${prefix}" "${FMT_INC_DIR}")

  configure_file(
    "${PROJECT_SOURCE_DIR}/support/cmake/fmt.pc.in"
    "${pkgconfig}"
    @ONLY)
  configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/support/cmake/fmt-config.cmake.in
    ${project_config}
    INSTALL_DESTINATION ${FMT_CMAKE_DIR})

  set(INSTALL_TARGETS fmt fmt-header-only)

  set(INSTALL_FILE_SET)
  if (FMT_USE_CMAKE_MODULES)
    set(INSTALL_FILE_SET FILE_SET fmt DESTINATION "${FMT_INC_DIR}/fmt")
  endif()

  # Install the library and headers.
  install(TARGETS ${INSTALL_TARGETS}
          COMPONENT fmt_core
          EXPORT ${targets_export_name}
          LIBRARY DESTINATION ${FMT_LIB_DIR}
          ARCHIVE DESTINATION ${FMT_LIB_DIR}
          PUBLIC_HEADER DESTINATION "${FMT_INC_DIR}/fmt"
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
          ${INSTALL_FILE_SET})

  # Use a namespace because CMake provides better diagnostics for namespaced
  # imported targets.
  export(TARGETS ${INSTALL_TARGETS} NAMESPACE fmt::
         FILE ${PROJECT_BINARY_DIR}/${targets_export_name}.cmake)

  # Install version, config and target files.
  install(FILES ${project_config} ${version_config}
          DESTINATION ${FMT_CMAKE_DIR}
          COMPONENT fmt_core)
  install(EXPORT ${targets_export_name} DESTINATION ${FMT_CMAKE_DIR}
          NAMESPACE fmt::
          COMPONENT fmt_core)

  install(FILES "${pkgconfig}" DESTINATION "${FMT_PKGCONFIG_DIR}"
          COMPONENT fmt_core)
endif ()

function(add_doc_target)
  find_program(DOXYGEN doxygen
    PATHS "$ENV{ProgramFiles}/doxygen/bin"
          "$ENV{ProgramFiles\(x86\)}/doxygen/bin")
  if (NOT DOXYGEN)
    message(STATUS "Target 'doc' disabled because doxygen not found")
    return ()
  endif ()

  find_program(MKDOCS mkdocs)
  if (NOT MKDOCS)
    message(STATUS "Target 'doc' disabled because mkdocs not found")
    return ()
  endif ()

  set(sources )
  foreach (source api.md index.md syntax.md get-started.md fmt.css fmt.js)
    set(sources ${sources} doc/${source})
  endforeach()

  add_custom_target(
    doc
    COMMAND
      ${CMAKE_COMMAND}
        -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/support/python
        ${MKDOCS} build -f ${CMAKE_CURRENT_SOURCE_DIR}/support/mkdocs.yml
        # MkDocs requires the site dir to be outside of the doc dir.
                        --site-dir ${CMAKE_CURRENT_BINARY_DIR}/doc-html
                        --no-directory-urls
    SOURCES ${sources})

  include(GNUInstallDirs)
  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc-html/
          DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/fmt
          COMPONENT fmt_doc OPTIONAL)
endfunction()

if (FMT_DOC)
  add_doc_target()
endif ()

if (FMT_TEST)
  enable_testing()
  add_subdirectory(test)
endif ()

# Control fuzzing independent of the unit tests.
if (FMT_FUZZ)
  add_subdirectory(test/fuzzing)

  # The FMT_FUZZ macro is used to prevent resource exhaustion in fuzzing
  # mode and make fuzzing practically possible. It is similar to
  # FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION but uses a different name to
  # avoid interfering with fuzzing of projects that use {fmt}.
  # See also https://llvm.org/docs/LibFuzzer.html#fuzzer-friendly-build-mode.
  target_compile_definitions(fmt PUBLIC FMT_FUZZ)
endif ()

set(gitignore ${PROJECT_SOURCE_DIR}/.gitignore)
if (FMT_MASTER_PROJECT AND EXISTS ${gitignore})
  # Get the list of ignored files from .gitignore.
  file (STRINGS ${gitignore} lines)
  list(REMOVE_ITEM lines /doc/html)
  foreach (line ${lines})
    string(REPLACE "." "[.]" line "${line}")
    string(REPLACE "*" ".*" line "${line}")
    set(ignored_files ${ignored_files} "${line}$" "${line}/")
  endforeach ()
  set(ignored_files ${ignored_files} /.git /build/doxyxml .vagrant)

  set(CPACK_SOURCE_GENERATOR ZIP)
  set(CPACK_SOURCE_IGNORE_FILES ${ignored_files})
  set(CPACK_SOURCE_PACKAGE_FILE_NAME fmt-${FMT_VERSION})
  set(CPACK_PACKAGE_NAME fmt)
  set(CPACK_RESOURCE_FILE_README ${PROJECT_SOURCE_DIR}/README.md)
  include(CPack)
endif ()
