# Test if compile errors are produced where necessary.

cmake_minimum_required(VERSION 3.8...3.25)
project(compile-error-test CXX)

set(fmt_headers "
  #include <fmt/format.h>
  #include <fmt/xchar.h>
  #include <fmt/ostream.h>
  #include <iostream>
")

set(error_test_names "")
set(non_error_test_content "")

# For error tests (we expect them to produce compilation error):
#  * adds a name of test into `error_test_names` list
#  * generates a single source file (with the same name) for each test
# For non-error tests (we expect them to compile successfully):
#  * adds a code segment as separate function to `non_error_test_content`
function (expect_compile name code_fragment)
  cmake_parse_arguments(EXPECT_COMPILE "ERROR" "" "" ${ARGN})
  string(MAKE_C_IDENTIFIER "${name}" test_name)

  if (EXPECT_COMPILE_ERROR)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/test/${test_name}.cc" "
      ${fmt_headers}
      void ${test_name}() {
        ${code_fragment}
      }
    ")
    set(error_test_names_copy "${error_test_names}")
    list(APPEND error_test_names_copy "${test_name}")
    set(error_test_names "${error_test_names_copy}" PARENT_SCOPE)
  else()
    set(non_error_test_content "
      ${non_error_test_content}
      void ${test_name}() {
        ${code_fragment}
      }" PARENT_SCOPE)
  endif()
endfunction ()

# Generates a source file for non-error test with `non_error_test_content` and
# CMake project file with all error and single non-error test targets.
function (run_tests)
  set(cmake_targets "")
  foreach(test_name IN LISTS error_test_names)
    set(cmake_targets "
      ${cmake_targets}
      add_library(test-${test_name} ${test_name}.cc)
      target_link_libraries(test-${test_name} PRIVATE fmt::fmt)
    ")
  endforeach()

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/test/non_error_test.cc" "
    ${fmt_headers}
    ${non_error_test_content}
  ")
  set(cmake_targets "
    ${cmake_targets}
    add_library(non-error-test non_error_test.cc)
    target_link_libraries(non-error-test PRIVATE fmt::fmt)
  ")

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/test/CMakeLists.txt" "
    cmake_minimum_required(VERSION 3.8...3.25)
    project(tests CXX)
    add_subdirectory(${FMT_DIR} fmt)
    ${cmake_targets}
  ")

  set(build_directory "${CMAKE_CURRENT_BINARY_DIR}/test/build")
  file(MAKE_DIRECTORY "${build_directory}")
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}"
      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
      "-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}"
      "-DCMAKE_GENERATOR=${CMAKE_GENERATOR}"
      "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
      "-DFMT_DIR=${FMT_DIR}"
      "${CMAKE_CURRENT_BINARY_DIR}/test"
    WORKING_DIRECTORY "${build_directory}"
    RESULT_VARIABLE result_var
    OUTPUT_VARIABLE output_var
    ERROR_VARIABLE output_var)
  if (NOT result_var EQUAL 0)
    message(FATAL_ERROR "Unable to configure:\n${output_var}")
  endif()

  foreach(test_name IN LISTS error_test_names)
    execute_process(
      COMMAND
        "${CMAKE_COMMAND}" --build "${build_directory}" --target "test-${test_name}"
      WORKING_DIRECTORY "${build_directory}"
      RESULT_VARIABLE result_var
      OUTPUT_VARIABLE output_var
      ERROR_QUIET)
    if (result_var EQUAL 0)
      message(SEND_ERROR "No compile error for \"${test_name}\":\n${output_var}")
    endif ()
  endforeach()

  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" --build "${build_directory}" --target "non-error-test"
    WORKING_DIRECTORY "${build_directory}"
    RESULT_VARIABLE result_var
    OUTPUT_VARIABLE output_var
    ERROR_VARIABLE output_var)
  if (NOT result_var EQUAL 0)
    message(SEND_ERROR "Compile error for combined non-error test:\n${output_var}")
  endif ()
endfunction ()

# Check if the source file skeleton compiles.
expect_compile(check "")
expect_compile(check-error "compilation_error" ERROR)

# Formatting a wide character with a narrow format string is forbidden.
expect_compile(wide-character-narrow-format-string "fmt::format(L\"{}\", L'a');")
expect_compile(wide-character-narrow-format-string-error "fmt::format(\"{}\", L'a');" ERROR)

# Formatting a wide string with a narrow format string is forbidden.
expect_compile(wide-string-narrow-format-string "fmt::format(L\"{}\", L\"foo\");")
expect_compile(wide-string-narrow-format-string-error "fmt::format(\"{}\", L\"foo\");" ERROR)

# Formatting a narrow string with a wide format string is forbidden because
# mixing UTF-8 with UTF-16/32 can result in an invalid output.
expect_compile(narrow-string-wide-format-string "fmt::format(L\"{}\", L\"foo\");")
expect_compile(narrow-string-wide-format-string-error "fmt::format(L\"{}\", \"foo\");" ERROR)

expect_compile(cast-to-string "
  struct S {
    operator std::string() const { return std::string(); }
  };
  fmt::format(\"{}\", std::string(S()));
")
expect_compile(cast-to-string-error "
  struct S {
    operator std::string() const { return std::string(); }
  };
  fmt::format(\"{}\", S());
" ERROR)

# Formatting a function
expect_compile(format-function "
  void (*f)();
  fmt::format(\"{}\", fmt::ptr(f));
")
expect_compile(format-function-error "
  void (*f)();
  fmt::format(\"{}\", f);
" ERROR)

# Formatting an unformattable argument should always be a compile time error
expect_compile(format-lots-of-arguments-with-unformattable "
  struct E {};
  fmt::format(\"\", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, E());
" ERROR)
expect_compile(format-lots-of-arguments-with-function "
  void (*f)();
  fmt::format(\"\", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, f);
" ERROR)

if (CMAKE_CXX_STANDARD GREATER_EQUAL 20)
  # Compile-time argument type check
  expect_compile(format-string-number-spec "
    #ifdef FMT_HAS_CONSTEVAL
      fmt::format(\"{:d}\", 42);
    #endif
  ")
  expect_compile(format-string-number-spec-error "
    #ifdef FMT_HAS_CONSTEVAL
      fmt::format(\"{:d}\", \"I am not a number\");
    #else
      #error
    #endif
  " ERROR)
  expect_compile(print-string-number-spec-error "
    #ifdef FMT_HAS_CONSTEVAL
      fmt::print(\"{:d}\", \"I am not a number\");
    #else
      #error
    #endif
  " ERROR)
  expect_compile(print-stream-string-number-spec-error "
  #ifdef FMT_HAS_CONSTEVAL
    fmt::print(std::cout, \"{:d}\", \"I am not a number\");
  #else
    #error
  #endif
  " ERROR)

  # Compile-time argument name check
  expect_compile(format-string-name "
    #if defined(FMT_HAS_CONSTEVAL) && FMT_USE_NONTYPE_TEMPLATE_ARGS
      using namespace fmt::literals;
      fmt::print(\"{foo}\", \"foo\"_a=42);
    #endif
  ")
  expect_compile(format-string-name-error "
    #if defined(FMT_HAS_CONSTEVAL) && FMT_USE_NONTYPE_TEMPLATE_ARGS
      using namespace fmt::literals;
      fmt::print(\"{foo}\", \"bar\"_a=42);
    #else
      #error
    #endif
  " ERROR)
endif ()

# Run all tests
run_tests()
