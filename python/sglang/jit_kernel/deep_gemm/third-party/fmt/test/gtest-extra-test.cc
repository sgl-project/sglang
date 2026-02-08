// Formatting library for C++ - tests of custom Google Test assertions
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#include "gtest-extra.h"

#include <gtest/gtest-spi.h>

#include <cstring>
#include <memory>
#include <stdexcept>

#include "fmt/os.h"
#include "util.h"

// Tests that assertion macros evaluate their arguments exactly once.
namespace {
class single_evaluation_test : public ::testing::Test {
 protected:
  single_evaluation_test() {
    p_ = s_;
    a_ = 0;
    b_ = 0;
  }

  static const char* const s_;
  static const char* p_;

  static int a_;
  static int b_;
};
}  // namespace

const char* const single_evaluation_test::s_ = "01234";
const char* single_evaluation_test::p_;
int single_evaluation_test::a_;
int single_evaluation_test::b_;

void do_nothing() {}

FMT_NORETURN void throw_exception() { throw std::runtime_error("test"); }

FMT_NORETURN void throw_system_error() {
  throw fmt::system_error(EDOM, "test");
}

// Tests that when EXPECT_THROW_MSG fails, it evaluates its message argument
// exactly once.
TEST_F(single_evaluation_test, failed_expect_throw_msg) {
  EXPECT_NONFATAL_FAILURE(
      EXPECT_THROW_MSG(throw_exception(), std::exception, p_++), "01234");
  EXPECT_EQ(s_ + 1, p_);
}

// Tests that when EXPECT_SYSTEM_ERROR fails, it evaluates its message argument
// exactly once.
TEST_F(single_evaluation_test, failed_expect_system_error) {
  EXPECT_NONFATAL_FAILURE(EXPECT_SYSTEM_ERROR(throw_system_error(), EDOM, p_++),
                          "01234");
  EXPECT_EQ(s_ + 1, p_);
}

// Tests that assertion arguments are evaluated exactly once.
TEST_F(single_evaluation_test, exception_tests) {
  // successful EXPECT_THROW_MSG
  EXPECT_THROW_MSG(
      {  // NOLINT
        a_++;
        throw_exception();
      },
      std::exception, (b_++, "test"));
  EXPECT_EQ(1, a_);
  EXPECT_EQ(1, b_);

  // failed EXPECT_THROW_MSG, throws different type
  EXPECT_NONFATAL_FAILURE(EXPECT_THROW_MSG(
                              {  // NOLINT
                                a_++;
                                throw_exception();
                              },
                              std::logic_error, (b_++, "test")),
                          "throws a different type");
  EXPECT_EQ(2, a_);
  EXPECT_EQ(2, b_);

  // failed EXPECT_THROW_MSG, throws an exception with different message
  EXPECT_NONFATAL_FAILURE(EXPECT_THROW_MSG(
                              {  // NOLINT
                                a_++;
                                throw_exception();
                              },
                              std::exception, (b_++, "other")),
                          "throws an exception with a different message");
  EXPECT_EQ(3, a_);
  EXPECT_EQ(3, b_);

  // failed EXPECT_THROW_MSG, throws nothing
  EXPECT_NONFATAL_FAILURE(
      EXPECT_THROW_MSG(a_++, std::exception, (b_++, "test")), "throws nothing");
  EXPECT_EQ(4, a_);
  EXPECT_EQ(4, b_);
}

TEST_F(single_evaluation_test, system_error_tests) {
  // successful EXPECT_SYSTEM_ERROR
  EXPECT_SYSTEM_ERROR(
      {  // NOLINT
        a_++;
        throw_system_error();
      },
      EDOM, (b_++, "test"));
  EXPECT_EQ(1, a_);
  EXPECT_EQ(1, b_);

  // failed EXPECT_SYSTEM_ERROR, throws different type
  EXPECT_NONFATAL_FAILURE(EXPECT_SYSTEM_ERROR(
                              {  // NOLINT
                                a_++;
                                throw_exception();
                              },
                              EDOM, (b_++, "test")),
                          "throws a different type");
  EXPECT_EQ(2, a_);
  EXPECT_EQ(2, b_);

  // failed EXPECT_SYSTEM_ERROR, throws an exception with different message
  EXPECT_NONFATAL_FAILURE(EXPECT_SYSTEM_ERROR(
                              {  // NOLINT
                                a_++;
                                throw_system_error();
                              },
                              EDOM, (b_++, "other")),
                          "throws an exception with a different message");
  EXPECT_EQ(3, a_);
  EXPECT_EQ(3, b_);

  // failed EXPECT_SYSTEM_ERROR, throws nothing
  EXPECT_NONFATAL_FAILURE(EXPECT_SYSTEM_ERROR(a_++, EDOM, (b_++, "test")),
                          "throws nothing");
  EXPECT_EQ(4, a_);
  EXPECT_EQ(4, b_);
}

#if FMT_USE_FCNTL
// Tests that when EXPECT_WRITE fails, it evaluates its message argument
// exactly once.
TEST_F(single_evaluation_test, failed_expect_write) {
  EXPECT_NONFATAL_FAILURE(EXPECT_WRITE(stdout, std::printf("test"), p_++),
                          "01234");
  EXPECT_EQ(s_ + 1, p_);
}

// Tests that assertion arguments are evaluated exactly once.
TEST_F(single_evaluation_test, write_tests) {
  // successful EXPECT_WRITE
  EXPECT_WRITE(
      stdout,
      {  // NOLINT
        a_++;
        std::printf("test");
      },
      (b_++, "test"));
  EXPECT_EQ(1, a_);
  EXPECT_EQ(1, b_);

  // failed EXPECT_WRITE
  EXPECT_NONFATAL_FAILURE(EXPECT_WRITE(
                              stdout,
                              {  // NOLINT
                                a_++;
                                std::printf("test");
                              },
                              (b_++, "other")),
                          "Actual: test");
  EXPECT_EQ(2, a_);
  EXPECT_EQ(2, b_);
}

// Tests EXPECT_WRITE.
TEST(gtest_extra_test, expect_write) {
  EXPECT_WRITE(stdout, do_nothing(), "");
  EXPECT_WRITE(stdout, std::printf("test"), "test");
  EXPECT_WRITE(stderr, std::fprintf(stderr, "test"), "test");
  EXPECT_NONFATAL_FAILURE(EXPECT_WRITE(stdout, std::printf("that"), "this"),
                          "Expected: this\n"
                          "  Actual: that");
}

TEST(gtest_extra_test, expect_write_streaming) {
  EXPECT_WRITE(stdout, std::printf("test"), "test") << "unexpected failure";
  EXPECT_NONFATAL_FAILURE(EXPECT_WRITE(stdout, std::printf("test"), "other")
                              << "expected failure",
                          "expected failure");
}
#endif  // FMT_USE_FCNTL

// Tests that the compiler will not complain about unreachable code in the
// EXPECT_THROW_MSG macro.
TEST(gtest_extra_test, expect_throw_no_unreachable_code_warning) {
  int n = 0;
  (void)n;
  using std::runtime_error;
  EXPECT_THROW_MSG(throw runtime_error(""), runtime_error, "");
  EXPECT_NONFATAL_FAILURE(EXPECT_THROW_MSG(n++, runtime_error, ""), "");
  EXPECT_NONFATAL_FAILURE(EXPECT_THROW_MSG(throw 1, runtime_error, ""), "");
  EXPECT_NONFATAL_FAILURE(
      EXPECT_THROW_MSG(throw runtime_error("a"), runtime_error, "b"), "");
}

// Tests that the compiler will not complain about unreachable code in the
// EXPECT_SYSTEM_ERROR macro.
TEST(gtest_extra_test, expect_system_error_no_unreachable_code_warning) {
  int n = 0;
  (void)n;
  EXPECT_SYSTEM_ERROR(throw fmt::system_error(EDOM, "test"), EDOM, "test");
  EXPECT_NONFATAL_FAILURE(EXPECT_SYSTEM_ERROR(n++, EDOM, ""), "");
  EXPECT_NONFATAL_FAILURE(EXPECT_SYSTEM_ERROR(throw 1, EDOM, ""), "");
  EXPECT_NONFATAL_FAILURE(
      EXPECT_SYSTEM_ERROR(throw fmt::system_error(EDOM, "aaa"), EDOM, "bbb"),
      "");
}

TEST(gtest_extra_test, expect_throw_behaves_like_single_statement) {
  if (::testing::internal::AlwaysFalse())
    EXPECT_THROW_MSG(do_nothing(), std::exception, "");

  if (::testing::internal::AlwaysTrue())
    EXPECT_THROW_MSG(throw_exception(), std::exception, "test");
  else
    do_nothing();
}

TEST(gtest_extra_test, expect_system_error_behaves_like_single_statement) {
  if (::testing::internal::AlwaysFalse())
    EXPECT_SYSTEM_ERROR(do_nothing(), EDOM, "");

  if (::testing::internal::AlwaysTrue())
    EXPECT_SYSTEM_ERROR(throw_system_error(), EDOM, "test");
  else
    do_nothing();
}

TEST(gtest_extra_test, expect_write_behaves_like_single_statement) {
  if (::testing::internal::AlwaysFalse())
    EXPECT_WRITE(stdout, std::printf("x"), "x");

  if (::testing::internal::AlwaysTrue())
    EXPECT_WRITE(stdout, std::printf("x"), "x");
  else
    do_nothing();
}

// Tests EXPECT_THROW_MSG.
TEST(gtest_extra_test, expect_throw_msg) {
  EXPECT_THROW_MSG(throw_exception(), std::exception, "test");
  EXPECT_NONFATAL_FAILURE(
      EXPECT_THROW_MSG(throw_exception(), std::logic_error, "test"),
      "Expected: throw_exception() throws an exception of "
      "type std::logic_error.\n  Actual: it throws a different type.");
  EXPECT_NONFATAL_FAILURE(
      EXPECT_THROW_MSG(do_nothing(), std::exception, "test"),
      "Expected: do_nothing() throws an exception of type std::exception.\n"
      "  Actual: it throws nothing.");
  EXPECT_NONFATAL_FAILURE(
      EXPECT_THROW_MSG(throw_exception(), std::exception, "other"),
      "throw_exception() throws an exception with a different message.\n"
      "Expected: other\n"
      "  Actual: test");
}

// Tests EXPECT_SYSTEM_ERROR.
TEST(gtest_extra_test, expect_system_error) {
  EXPECT_SYSTEM_ERROR(throw_system_error(), EDOM, "test");
  EXPECT_NONFATAL_FAILURE(
      EXPECT_SYSTEM_ERROR(throw_exception(), EDOM, "test"),
      "Expected: throw_exception() throws an exception of "
      "type std::system_error.\n  Actual: it throws a different type.");
  EXPECT_NONFATAL_FAILURE(
      EXPECT_SYSTEM_ERROR(do_nothing(), EDOM, "test"),
      "Expected: do_nothing() throws an exception of type std::system_error.\n"
      "  Actual: it throws nothing.");
  EXPECT_NONFATAL_FAILURE(
      EXPECT_SYSTEM_ERROR(throw_system_error(), EDOM, "other"),
      fmt::format(
          "throw_system_error() throws an exception with a different message.\n"
          "Expected: {}\n"
          "  Actual: {}",
          system_error_message(EDOM, "other"),
          system_error_message(EDOM, "test")));
}

TEST(gtest_extra_test, expect_throw_msg_streaming) {
  EXPECT_THROW_MSG(throw_exception(), std::exception, "test")
      << "unexpected failure";
  EXPECT_NONFATAL_FAILURE(
      EXPECT_THROW_MSG(throw_exception(), std::exception, "other")
          << "expected failure",
      "expected failure");
}

TEST(gtest_extra_test, expect_system_error_streaming) {
  EXPECT_SYSTEM_ERROR(throw_system_error(), EDOM, "test")
      << "unexpected failure";
  EXPECT_NONFATAL_FAILURE(
      EXPECT_SYSTEM_ERROR(throw_system_error(), EDOM, "other")
          << "expected failure",
      "expected failure");
}

#if FMT_USE_FCNTL

using fmt::buffered_file;
using fmt::file;

TEST(output_redirect_test, scoped_redirect) {
  auto pipe = fmt::pipe();
  {
    buffered_file file(pipe.write_end.fdopen("w"));
    std::fprintf(file.get(), "[[[");
    {
      output_redirect redir(file.get());
      std::fprintf(file.get(), "censored");
    }
    std::fprintf(file.get(), "]]]");
  }
  EXPECT_READ(pipe.read_end, "[[[]]]");
}

// Test that output_redirect handles errors in flush correctly.
TEST(output_redirect_test, flush_error_in_ctor) {
  auto pipe = fmt::pipe();
  int write_fd = pipe.write_end.descriptor();
  file write_copy = pipe.write_end.dup(write_fd);
  buffered_file f = pipe.write_end.fdopen("w");
  // Put a character in a file buffer.
  EXPECT_EQ('x', fputc('x', f.get()));
  FMT_POSIX(close(write_fd));
  std::unique_ptr<output_redirect> redir{nullptr};
  EXPECT_SYSTEM_ERROR_NOASSERT(redir.reset(new output_redirect(f.get())), EBADF,
                               "cannot flush stream");
  redir.reset(nullptr);
  write_copy.dup2(write_fd);  // "undo" close or dtor will fail
}

TEST(output_redirect_test, dup_error_in_ctor) {
  buffered_file f = open_buffered_file();
  int fd = (f.descriptor)();
  file copy = file::dup(fd);
  FMT_POSIX(close(fd));
  std::unique_ptr<output_redirect> redir{nullptr};
  EXPECT_SYSTEM_ERROR_NOASSERT(
      redir.reset(new output_redirect(f.get(), false)), EBADF,
      fmt::format("cannot duplicate file descriptor {}", fd));
  copy.dup2(fd);  // "undo" close or dtor will fail
}

TEST(output_redirect_test, restore_and_read) {
  auto pipe = fmt::pipe();
  buffered_file file(pipe.write_end.fdopen("w"));
  std::fprintf(file.get(), "[[[");
  output_redirect redir(file.get());
  std::fprintf(file.get(), "censored");
  EXPECT_EQ("censored", redir.restore_and_read());
  EXPECT_EQ("", redir.restore_and_read());
  std::fprintf(file.get(), "]]]");
  file = buffered_file();
  EXPECT_READ(pipe.read_end, "[[[]]]");
}

// Test that OutputRedirect handles errors in flush correctly.
TEST(output_redirect_test, flush_error_in_restore_and_read) {
  auto pipe = fmt::pipe();
  int write_fd = pipe.write_end.descriptor();
  file write_copy = pipe.write_end.dup(write_fd);
  buffered_file f = pipe.write_end.fdopen("w");
  output_redirect redir(f.get());
  // Put a character in a file buffer.
  EXPECT_EQ('x', fputc('x', f.get()));
  FMT_POSIX(close(write_fd));
  EXPECT_SYSTEM_ERROR_NOASSERT(redir.restore_and_read(), EBADF,
                               "cannot flush stream");
  write_copy.dup2(write_fd);  // "undo" close or dtor will fail
}

TEST(output_redirect_test, error_in_dtor) {
  auto pipe = fmt::pipe();
  int write_fd = pipe.write_end.descriptor();
  file write_copy = pipe.write_end.dup(write_fd);
  buffered_file f = pipe.write_end.fdopen("w");
  std::unique_ptr<output_redirect> redir(new output_redirect(f.get()));
  // Put a character in a file buffer.
  EXPECT_EQ('x', fputc('x', f.get()));
  EXPECT_WRITE(
      stderr,
      {
        // The close function must be called inside EXPECT_WRITE,
        // otherwise the system may recycle closed file descriptor when
        // redirecting the output in EXPECT_STDERR and the second close
        // will break output redirection.
        FMT_POSIX(close(write_fd));
        SUPPRESS_ASSERT(redir.reset(nullptr));
      },
      system_error_message(EBADF, "cannot flush stream"));
  write_copy.dup2(write_fd);  // "undo" close or dtor of buffered_file will fail
}

#endif  // FMT_USE_FCNTL
