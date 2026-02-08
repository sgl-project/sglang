// Formatting library for C++ - custom Google Test assertions
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#ifndef FMT_GTEST_EXTRA_H_
#define FMT_GTEST_EXTRA_H_

#include <stdlib.h>  // _invalid_parameter_handler

#include <string>

#include "fmt/os.h"
#include "gmock/gmock.h"

#define FMT_TEST_THROW_(statement, expected_exception, expected_message, fail) \
  GTEST_AMBIGUOUS_ELSE_BLOCKER_                                                \
  if (::testing::AssertionResult gtest_ar = ::testing::AssertionSuccess()) {   \
    std::string gtest_expected_message = expected_message;                     \
    bool gtest_caught_expected = false;                                        \
    try {                                                                      \
      GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement);               \
    } catch (expected_exception const& e) {                                    \
      if (gtest_expected_message != e.what()) {                                \
        gtest_ar << #statement                                                 \
            " throws an exception with a different message.\n"                 \
                 << "Expected: " << gtest_expected_message << "\n"             \
                 << "  Actual: " << e.what();                                  \
        goto GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__);            \
      }                                                                        \
      gtest_caught_expected = true;                                            \
    } catch (...) {                                                            \
      gtest_ar << "Expected: " #statement                                      \
                  " throws an exception of type " #expected_exception          \
                  ".\n  Actual: it throws a different type.";                  \
      goto GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__);              \
    }                                                                          \
    if (!gtest_caught_expected) {                                              \
      gtest_ar << "Expected: " #statement                                      \
                  " throws an exception of type " #expected_exception          \
                  ".\n  Actual: it throws nothing.";                           \
      goto GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__);              \
    }                                                                          \
  } else                                                                       \
    GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__)                      \
        : fail(gtest_ar.failure_message())

// Tests that the statement throws the expected exception and the exception's
// what() method returns expected message.
#define EXPECT_THROW_MSG(statement, expected_exception, expected_message) \
  FMT_TEST_THROW_(statement, expected_exception, expected_message,        \
                  GTEST_NONFATAL_FAILURE_)

inline std::string system_error_message(int error_code,
                                        const std::string& message) {
  auto ec = std::error_code(error_code, std::generic_category());
  return std::system_error(ec, message).what();
}

#define EXPECT_SYSTEM_ERROR(statement, error_code, message) \
  EXPECT_THROW_MSG(statement, std::system_error,            \
                   system_error_message(error_code, message))

#if FMT_USE_FCNTL

// Captures file output by redirecting it to a pipe.
// The output it can handle is limited by the pipe capacity.
class output_redirect {
 private:
  FILE* file_;
  fmt::file original_;  // Original file passed to redirector.
  fmt::file read_end_;  // Read end of the pipe where the output is redirected.

  void flush();
  void restore();

 public:
  explicit output_redirect(FILE* file, bool flush = true);
  ~output_redirect() noexcept;

  output_redirect(const output_redirect&) = delete;
  void operator=(const output_redirect&) = delete;

  // Restores the original file, reads output from the pipe into a string
  // and returns it.
  std::string restore_and_read();
};

#  define FMT_TEST_WRITE_(statement, expected_output, file, fail)              \
    GTEST_AMBIGUOUS_ELSE_BLOCKER_                                              \
    if (::testing::AssertionResult gtest_ar = ::testing::AssertionSuccess()) { \
      std::string gtest_expected_output = expected_output;                     \
      output_redirect gtest_redir(file);                                       \
      GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement);               \
      std::string gtest_output = gtest_redir.restore_and_read();               \
      if (gtest_output != gtest_expected_output) {                             \
        gtest_ar << #statement " produces different output.\n"                 \
                 << "Expected: " << gtest_expected_output << "\n"              \
                 << "  Actual: " << gtest_output;                              \
        goto GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__);            \
      }                                                                        \
    } else                                                                     \
      GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__)                    \
          : fail(gtest_ar.failure_message())

// Tests that the statement writes the expected output to file.
#  define EXPECT_WRITE(file, statement, expected_output) \
    FMT_TEST_WRITE_(statement, expected_output, file, GTEST_NONFATAL_FAILURE_)

#  ifdef _MSC_VER

// Suppresses Windows assertions on invalid file descriptors, making
// POSIX functions return proper error codes instead of crashing on Windows.
class suppress_assert {
 private:
  _invalid_parameter_handler original_handler_;
  int original_report_mode_;

  static void handle_invalid_parameter(const wchar_t*, const wchar_t*,
                                       const wchar_t*, unsigned, uintptr_t) {}

 public:
  suppress_assert()
      : original_handler_(
            _set_invalid_parameter_handler(handle_invalid_parameter)),
        original_report_mode_(_CrtSetReportMode(_CRT_ASSERT, 0)) {}
  ~suppress_assert() {
    _set_invalid_parameter_handler(original_handler_);
    _CrtSetReportMode(_CRT_ASSERT, original_report_mode_);
    (void)original_report_mode_;
  }
};

#    define SUPPRESS_ASSERT(statement) \
      {                                \
        suppress_assert sa;            \
        statement;                     \
      }
#  else
#    define SUPPRESS_ASSERT(statement) statement
#  endif  // _MSC_VER

#  define EXPECT_SYSTEM_ERROR_NOASSERT(statement, error_code, message) \
    EXPECT_SYSTEM_ERROR(SUPPRESS_ASSERT(statement), error_code, message)

// Attempts to read count characters from a file.
std::string read(fmt::file& f, size_t count);

#  define EXPECT_READ(file, expected_content) \
    EXPECT_EQ(expected_content,               \
              read(file, fmt::string_view(expected_content).size()))

#else
#  define EXPECT_WRITE(file, statement, expected_output) \
    do {                                                 \
      (void)(file);                                      \
      (void)(statement);                                 \
      (void)(expected_output);                           \
      SUCCEED();                                         \
    } while (false)
#endif  // FMT_USE_FCNTL

#endif  // FMT_GTEST_EXTRA_H_
