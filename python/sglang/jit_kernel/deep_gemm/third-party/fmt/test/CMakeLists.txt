add_subdirectory(gtest)

include(CheckSymbolExists)

set(TEST_MAIN_SRC test-main.cc gtest-extra.cc gtest-extra.h util.cc)
add_library(test-main STATIC ${TEST_MAIN_SRC})
target_include_directories(test-main PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(test-main gtest fmt)

# Adds a test.
# Usage: add_fmt_test(name srcs...)
function(add_fmt_test name)
  cmake_parse_arguments(ADD_FMT_TEST "HEADER_ONLY;MODULE" "" "" ${ARGN})

  set(sources ${name}.cc ${ADD_FMT_TEST_UNPARSED_ARGUMENTS})
  if (ADD_FMT_TEST_HEADER_ONLY)
    set(sources ${sources} ${TEST_MAIN_SRC} ../src/os.cc)
    set(libs gtest fmt-header-only)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      set(PEDANTIC_COMPILE_FLAGS ${PEDANTIC_COMPILE_FLAGS} -Wno-weak-vtables)
    endif ()
  elseif (ADD_FMT_TEST_MODULE)
    set(libs test-main test-module)
    set_source_files_properties(${name}.cc PROPERTIES OBJECT_DEPENDS test-module)
  else ()
    set(libs test-main fmt)
  endif ()
  add_executable(${name} ${sources})
  target_link_libraries(${name} ${libs})

  if (ADD_FMT_TEST_HEADER_ONLY AND NOT FMT_UNICODE)
    target_compile_definitions(${name} PUBLIC FMT_UNICODE=0)
  endif ()

  # Define if certain C++ features can be used.
  if (FMT_PEDANTIC)
    target_compile_options(${name} PRIVATE ${PEDANTIC_COMPILE_FLAGS})
  endif ()
  if (FMT_WERROR)
    target_compile_options(${name} PRIVATE ${WERROR_FLAG})
  endif ()
  add_test(NAME ${name} COMMAND ${name})
endfunction()

if (FMT_MODULE)
  return ()
endif ()

add_fmt_test(args-test)
add_fmt_test(base-test)
add_fmt_test(assert-test)
add_fmt_test(chrono-test)
add_fmt_test(color-test)
add_fmt_test(gtest-extra-test)
add_fmt_test(format-test mock-allocator.h)
if (MSVC)
  target_compile_options(format-test PRIVATE /bigobj)
endif ()
if (NOT (MSVC AND BUILD_SHARED_LIBS))
  add_fmt_test(format-impl-test HEADER_ONLY header-only-test.cc)
endif ()
add_fmt_test(ostream-test)
add_fmt_test(compile-test)
add_fmt_test(compile-fp-test)
if (MSVC)
  # Without this option, MSVC returns 199711L for the __cplusplus macro.
  target_compile_options(compile-fp-test PRIVATE /Zc:__cplusplus)
endif()
add_fmt_test(printf-test)
add_fmt_test(ranges-test ranges-odr-test.cc)
add_fmt_test(no-builtin-types-test HEADER_ONLY)

add_fmt_test(scan-test HEADER_ONLY)
check_symbol_exists(strptime "time.h" HAVE_STRPTIME)
if (HAVE_STRPTIME)
  target_compile_definitions(scan-test PRIVATE FMT_HAVE_STRPTIME)
endif ()

add_fmt_test(std-test)
try_compile(compile_result_unused
            ${CMAKE_CURRENT_BINARY_DIR}
            SOURCES ${CMAKE_CURRENT_LIST_DIR}/detect-stdfs.cc
            OUTPUT_VARIABLE RAWOUTPUT)
string(REGEX REPLACE ".*libfound \"([^\"]*)\".*" "\\1" STDLIBFS "${RAWOUTPUT}")
if (STDLIBFS)
  target_link_libraries(std-test ${STDLIBFS})
endif ()
add_fmt_test(unicode-test HEADER_ONLY)
if (MSVC)
  target_compile_options(unicode-test PRIVATE /utf-8)
endif ()
add_fmt_test(xchar-test)
add_fmt_test(enforce-checks-test)
target_compile_definitions(enforce-checks-test PRIVATE
                           -DFMT_ENFORCE_COMPILE_STRING)

add_executable(perf-sanity perf-sanity.cc)
target_link_libraries(perf-sanity fmt::fmt)

if (FMT_MODULE)
  # The tests need {fmt} to be compiled as traditional library
  # because of visibility of implementation details.
  # If module support is present the module tests require a
  # test-only module to be built from {fmt}
  add_library(test-module OBJECT ${CMAKE_SOURCE_DIR}/src/fmt.cc)
  target_compile_features(test-module PUBLIC cxx_std_11)
  target_include_directories(test-module PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
  enable_module(test-module)

  add_fmt_test(module-test MODULE test-main.cc)
  if (MSVC)
    target_compile_options(test-module PRIVATE /utf-8 /Zc:__cplusplus
                           /Zc:externConstexpr /Zc:inline)
    target_compile_options(module-test PRIVATE /utf-8 /Zc:__cplusplus
                           /Zc:externConstexpr /Zc:inline)
  endif ()
endif ()

if (NOT DEFINED MSVC_STATIC_RUNTIME AND MSVC)
  foreach (flag_var
			 CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
			 CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
	if (${flag_var} MATCHES "^(/|-)(MT|MTd)")
	  set(MSVC_STATIC_RUNTIME ON)
	  break()
	endif()
  endforeach()
endif()

if (NOT MSVC_STATIC_RUNTIME)
  add_executable(posix-mock-test
    posix-mock-test.cc ../src/format.cc ${TEST_MAIN_SRC})
  target_include_directories(
    posix-mock-test PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(posix-mock-test gtest)
  if (FMT_PEDANTIC)
    target_compile_options(posix-mock-test PRIVATE ${PEDANTIC_COMPILE_FLAGS})
  endif ()
  if (MSVC)
    target_compile_options(posix-mock-test PRIVATE /utf-8)
  endif ()
  add_test(NAME posix-mock-test COMMAND posix-mock-test)
  add_fmt_test(os-test)
endif ()

message(STATUS "FMT_PEDANTIC: ${FMT_PEDANTIC}")

if (FMT_PEDANTIC)
  # Test that the library can be compiled with exceptions disabled.
  # -fno-exception is broken in icc: https://github.com/fmtlib/fmt/issues/822.
  if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    check_cxx_compiler_flag(-fno-exceptions HAVE_FNO_EXCEPTIONS_FLAG)
  endif ()
  if (HAVE_FNO_EXCEPTIONS_FLAG)
    add_library(noexception-test ../src/format.cc noexception-test.cc)
    target_include_directories(
      noexception-test PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_compile_options(noexception-test PRIVATE -fno-exceptions)
    target_compile_options(noexception-test PRIVATE ${PEDANTIC_COMPILE_FLAGS})
  endif ()

  # Test that the library compiles without locale.
  add_library(nolocale-test ../src/format.cc)
  target_include_directories(
    nolocale-test PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_compile_definitions(
    nolocale-test PRIVATE FMT_STATIC_THOUSANDS_SEPARATOR=1)
endif ()

# These tests are disabled on Windows because they take too long.
# They are disabled on GCC < 4.9 because it can not parse UDLs without
# a space after `operator""` but that is an incorrect syntax for any more
# modern compiler.
if (FMT_PEDANTIC AND NOT WIN32 AND NOT (
    CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND
    CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9))
  # Test if incorrect API usages produce compilation error.
  add_test(compile-error-test ${CMAKE_CTEST_COMMAND}
    --build-and-test
    "${CMAKE_CURRENT_SOURCE_DIR}/compile-error-test"
    "${CMAKE_CURRENT_BINARY_DIR}/compile-error-test"
    --build-generator ${CMAKE_GENERATOR}
    --build-makeprogram ${CMAKE_MAKE_PROGRAM}
    --build-options
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
    "-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}"
    "-DCXX_STANDARD_FLAG=${CXX_STANDARD_FLAG}"
    "-DFMT_DIR=${CMAKE_SOURCE_DIR}")

  # Test if the targets are found from the build directory.
  add_test(find-package-test ${CMAKE_CTEST_COMMAND}
    -C ${CMAKE_BUILD_TYPE}
    --build-and-test
    "${CMAKE_CURRENT_SOURCE_DIR}/find-package-test"
    "${CMAKE_CURRENT_BINARY_DIR}/find-package-test"
    --build-generator ${CMAKE_GENERATOR}
    --build-makeprogram ${CMAKE_MAKE_PROGRAM}
    --build-options
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
    "-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}"
    "-DFMT_DIR=${PROJECT_BINARY_DIR}"
    "-DPEDANTIC_COMPILE_FLAGS=${PEDANTIC_COMPILE_FLAGS}"
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

  # Test if the targets are found when add_subdirectory is used.
  add_test(add-subdirectory-test ${CMAKE_CTEST_COMMAND}
    -C ${CMAKE_BUILD_TYPE}
    --build-and-test
    "${CMAKE_CURRENT_SOURCE_DIR}/add-subdirectory-test"
    "${CMAKE_CURRENT_BINARY_DIR}/add-subdirectory-test"
    --build-generator ${CMAKE_GENERATOR}
    --build-makeprogram ${CMAKE_MAKE_PROGRAM}
    --build-options
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}"
    "-DPEDANTIC_COMPILE_FLAGS=${PEDANTIC_COMPILE_FLAGS}"
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
endif ()

# This test is disabled on Windows because it is POSIX-specific.
if (FMT_PEDANTIC AND NOT WIN32)
  add_test(static-export-test ${CMAKE_CTEST_COMMAND}
    -C ${CMAKE_BUILD_TYPE}
    --build-and-test
    "${CMAKE_CURRENT_SOURCE_DIR}/static-export-test"
    "${CMAKE_CURRENT_BINARY_DIR}/static-export-test"
    --build-generator ${CMAKE_GENERATOR}
    --build-makeprogram ${CMAKE_MAKE_PROGRAM}
    --build-options
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}"
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
endif ()

# Activate optional CUDA tests if CUDA is found. For version selection see
# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cpp14-language-features
if (FMT_CUDA_TEST)
  if (${CMAKE_VERSION} VERSION_LESS 3.15)
    find_package(CUDA 9.0)
  else ()
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
      enable_language(CUDA OPTIONAL)
      set(CUDA_FOUND TRUE)
    endif ()
  endif ()

  if (CUDA_FOUND)
    add_subdirectory(cuda-test)
    add_test(NAME cuda-test COMMAND fmt-in-cuda-test)
  endif ()
endif ()
