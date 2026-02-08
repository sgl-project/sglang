// Tests of the C++ interface to POSIX functions that require mocks
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

// Disable bogus MSVC warnings.
#if !defined(_CRT_SECURE_NO_WARNINGS) && defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS
#endif

#include "posix-mock.h"

#include <errno.h>
#include <fcntl.h>

#include <climits>
#include <memory>

#include "../src/os.cc"

#ifdef _WIN32
#  include <io.h>
#  undef max
#endif

#include "gmock/gmock.h"
#include "gtest-extra.h"
#include "util.h"

using fmt::buffered_file;

using testing::_;
using testing::Return;
using testing::StrEq;

template <typename Mock> struct scoped_mock : testing::StrictMock<Mock> {
  scoped_mock() { Mock::instance = this; }
  ~scoped_mock() { Mock::instance = nullptr; }
};

namespace {
int open_count;
int close_count;
int dup_count;
int dup2_count;
int fdopen_count;
int read_count;
int write_count;
int pipe_count;
int fopen_count;
int fclose_count;
int fileno_count;
size_t read_nbyte;
size_t write_nbyte;
bool sysconf_error;

enum { none, max_size, error } fstat_sim;
}  // namespace

#define EMULATE_EINTR(func, error_result) \
  if (func##_count != 0) {                \
    if (func##_count++ != 3) {            \
      errno = EINTR;                      \
      return error_result;                \
    }                                     \
  }

#ifndef _MSC_VER
int test::open(const char* path, int oflag, int mode) {
  EMULATE_EINTR(open, -1);
  return ::open(path, oflag, mode);
}
#endif

#ifndef _WIN32

long test::sysconf(int name) {
  long result = ::sysconf(name);
  if (!sysconf_error) return result;
  // Simulate an error.
  errno = EINVAL;
  return -1;
}

static off_t max_file_size() { return std::numeric_limits<off_t>::max(); }

int test::fstat(int fd, struct stat* buf) {
  int result = ::fstat(fd, buf);
  if (fstat_sim == max_size) buf->st_size = max_file_size();
  return result;
}

#else

static LONGLONG max_file_size() { return std::numeric_limits<LONGLONG>::max(); }

DWORD test::GetFileSize(HANDLE hFile, LPDWORD lpFileSizeHigh) {
  if (fstat_sim == error) {
    SetLastError(ERROR_ACCESS_DENIED);
    return INVALID_FILE_SIZE;
  }
  if (fstat_sim == max_size) {
    DWORD max = std::numeric_limits<DWORD>::max();
    *lpFileSizeHigh = max >> 1;
    return max;
  }
  return ::GetFileSize(hFile, lpFileSizeHigh);
}

#endif

int test::close(int fildes) {
  // Close the file first because close shouldn't be retried.
  int result = ::FMT_POSIX(close(fildes));
  EMULATE_EINTR(close, -1);
  return result;
}

int test::dup(int fildes) {
  EMULATE_EINTR(dup, -1);
  return ::FMT_POSIX(dup(fildes));
}

int test::dup2(int fildes, int fildes2) {
  EMULATE_EINTR(dup2, -1);
  return ::FMT_POSIX(dup2(fildes, fildes2));
}

FILE* test::fdopen(int fildes, const char* mode) {
  EMULATE_EINTR(fdopen, nullptr);
  return ::FMT_POSIX(fdopen(fildes, mode));
}

test::ssize_t test::read(int fildes, void* buf, test::size_t nbyte) {
  read_nbyte = nbyte;
  EMULATE_EINTR(read, -1);
  return ::FMT_POSIX(read(fildes, buf, nbyte));
}

test::ssize_t test::write(int fildes, const void* buf, test::size_t nbyte) {
  write_nbyte = nbyte;
  EMULATE_EINTR(write, -1);
  return ::FMT_POSIX(write(fildes, buf, nbyte));
}

#ifndef _WIN32
int test::pipe(int fildes[2]) {
  EMULATE_EINTR(pipe, -1);
  return ::pipe(fildes);
}
#else
int test::pipe(int* pfds, unsigned psize, int textmode) {
  EMULATE_EINTR(pipe, -1);
  return _pipe(pfds, psize, textmode);
}
#endif

FILE* test::fopen(const char* filename, const char* mode) {
  EMULATE_EINTR(fopen, nullptr);
  return ::fopen(filename, mode);
}

int test::fclose(FILE* stream) {
  EMULATE_EINTR(fclose, EOF);
  return ::fclose(stream);
}

int(test::fileno)(FILE* stream) {
  EMULATE_EINTR(fileno, -1);
#ifdef fileno
  return FMT_POSIX(fileno(stream));
#else
  return ::FMT_POSIX(fileno(stream));
#endif
}

#ifndef _WIN32
#  define EXPECT_RETRY(statement, func, message) \
    func##_count = 1;                            \
    statement;                                   \
    EXPECT_EQ(4, func##_count);                  \
    func##_count = 0;
#  define EXPECT_EQ_POSIX(expected, actual) EXPECT_EQ(expected, actual)
#else
#  define EXPECT_RETRY(statement, func, message)    \
    func##_count = 1;                               \
    EXPECT_SYSTEM_ERROR(statement, EINTR, message); \
    func##_count = 0;
#  define EXPECT_EQ_POSIX(expected, actual)
#endif

#if FMT_USE_FCNTL
void write_file(fmt::cstring_view filename, fmt::string_view content) {
  fmt::buffered_file f(filename, "w");
  f.print("{}", content);
}

using fmt::file;

TEST(os_test, getpagesize) {
#  ifdef _WIN32
  SYSTEM_INFO si = {};
  GetSystemInfo(&si);
  EXPECT_EQ(si.dwPageSize, fmt::getpagesize());
#  else
  EXPECT_EQ(sysconf(_SC_PAGESIZE), fmt::getpagesize());
  sysconf_error = true;
  EXPECT_SYSTEM_ERROR(fmt::getpagesize(), EINVAL,
                      "cannot get memory page size");
  sysconf_error = false;
#  endif
}

TEST(file_test, open_retry) {
#  ifndef _WIN32
  write_file("temp", "there must be something here");
  std::unique_ptr<file> f{nullptr};
  EXPECT_RETRY(f.reset(new file("temp", file::RDONLY)), open,
               "cannot open file temp");
  char c = 0;
  f->read(&c, 1);
#  endif
}

TEST(file_test, close_no_retry_in_dtor) {
  auto pipe = fmt::pipe();
  std::unique_ptr<file> f(new file(std::move(pipe.read_end)));
  int saved_close_count = 0;
  EXPECT_WRITE(
      stderr,
      {
        close_count = 1;
        f.reset(nullptr);
        saved_close_count = close_count;
        close_count = 0;
      },
      system_error_message(EINTR, "cannot close file") + "\n");
  EXPECT_EQ(2, saved_close_count);
}

TEST(file_test, close_no_retry) {
  auto pipe = fmt::pipe();
  close_count = 1;
  EXPECT_SYSTEM_ERROR(pipe.read_end.close(), EINTR, "cannot close file");
  EXPECT_EQ(2, close_count);
  close_count = 0;
}

TEST(file_test, size) {
  std::string content = "top secret, destroy before reading";
  write_file("temp", content);
  file f("temp", file::RDONLY);
  EXPECT_GE(f.size(), 0);
  EXPECT_EQ(content.size(), static_cast<unsigned long long>(f.size()));
#  ifdef _WIN32
  auto error_code = std::error_code();
  fstat_sim = error;
  try {
    f.size();
  } catch (const std::system_error& e) {
    error_code = e.code();
  }
  fstat_sim = none;
  EXPECT_EQ(error_code,
            std::error_code(ERROR_ACCESS_DENIED, fmt::system_category()));
#  else
  f.close();
  EXPECT_SYSTEM_ERROR(f.size(), EBADF, "cannot get file attributes");
#  endif
}

TEST(file_test, max_size) {
  write_file("temp", "");
  file f("temp", file::RDONLY);
  fstat_sim = max_size;
  EXPECT_GE(f.size(), 0);
  EXPECT_EQ(max_file_size(), f.size());
  fstat_sim = none;
}

TEST(file_test, read_retry) {
  auto pipe = fmt::pipe();
  enum { SIZE = 4 };
  pipe.write_end.write("test", SIZE);
  pipe.write_end.close();
  char buffer[SIZE];
  size_t count = 0;
  EXPECT_RETRY(count = pipe.read_end.read(buffer, SIZE), read,
               "cannot read from file");
  EXPECT_EQ_POSIX(static_cast<std::streamsize>(SIZE), count);
}

TEST(file_test, write_retry) {
  auto pipe = fmt::pipe();
  enum { SIZE = 4 };
  size_t count = 0;
  EXPECT_RETRY(count = pipe.write_end.write("test", SIZE), write,
               "cannot write to file");
  pipe.write_end.close();
#  ifndef _WIN32
  EXPECT_EQ(static_cast<std::streamsize>(SIZE), count);
  char buffer[SIZE + 1];
  pipe.read_end.read(buffer, SIZE);
  buffer[SIZE] = '\0';
  EXPECT_STREQ("test", buffer);
#  endif
}

#  ifdef _WIN32
TEST(file_test, convert_read_count) {
  auto pipe = fmt::pipe();
  char c;
  size_t size = UINT_MAX;
  if (sizeof(unsigned) != sizeof(size_t)) ++size;
  read_count = 1;
  read_nbyte = 0;
  EXPECT_THROW(pipe.read_end.read(&c, size), std::system_error);
  read_count = 0;
  EXPECT_EQ(UINT_MAX, read_nbyte);
}

TEST(file_test, convert_write_count) {
  auto pipe = fmt::pipe();
  char c;
  size_t size = UINT_MAX;
  if (sizeof(unsigned) != sizeof(size_t)) ++size;
  write_count = 1;
  write_nbyte = 0;
  EXPECT_THROW(pipe.write_end.write(&c, size), std::system_error);
  write_count = 0;
  EXPECT_EQ(UINT_MAX, write_nbyte);
}
#  endif

TEST(file_test, dup_no_retry) {
  int stdout_fd = FMT_POSIX(fileno(stdout));
  dup_count = 1;
  EXPECT_SYSTEM_ERROR(
      file::dup(stdout_fd), EINTR,
      fmt::format("cannot duplicate file descriptor {}", stdout_fd));
  dup_count = 0;
}

TEST(file_test, dup2_retry) {
  int stdout_fd = FMT_POSIX(fileno(stdout));
  file f1 = file::dup(stdout_fd), f2 = file::dup(stdout_fd);
  EXPECT_RETRY(f1.dup2(f2.descriptor()), dup2,
               fmt::format("cannot duplicate file descriptor {} to {}",
                           f1.descriptor(), f2.descriptor()));
}

TEST(file_test, dup2_no_except_retry) {
  int stdout_fd = FMT_POSIX(fileno(stdout));
  file f1 = file::dup(stdout_fd), f2 = file::dup(stdout_fd);
  std::error_code ec;
  dup2_count = 1;
  f1.dup2(f2.descriptor(), ec);
#  ifndef _WIN32
  EXPECT_EQ(4, dup2_count);
#  else
  EXPECT_EQ(EINTR, ec.value());
#  endif
  dup2_count = 0;
}

TEST(file_test, pipe_no_retry) {
  pipe_count = 1;
  EXPECT_SYSTEM_ERROR(fmt::pipe(), EINTR, "cannot create pipe");
  pipe_count = 0;
}

TEST(file_test, fdopen_no_retry) {
  auto pipe = fmt::pipe();
  fdopen_count = 1;
  EXPECT_SYSTEM_ERROR(pipe.read_end.fdopen("r"), EINTR,
                      "cannot associate stream with file descriptor");
  fdopen_count = 0;
}

TEST(buffered_file_test, open_retry) {
  write_file("temp", "there must be something here");
  std::unique_ptr<buffered_file> f{nullptr};
  EXPECT_RETRY(f.reset(new buffered_file("temp", "r")), fopen,
               "cannot open file temp");
#  ifndef _WIN32
  char c = 0;
  if (fread(&c, 1, 1, f->get()) < 1)
    throw fmt::system_error(errno, "fread failed");
#  endif
}

TEST(buffered_file_test, close_no_retry_in_dtor) {
  auto pipe = fmt::pipe();
  std::unique_ptr<buffered_file> f(
      new buffered_file(pipe.read_end.fdopen("r")));
  int saved_fclose_count = 0;
  EXPECT_WRITE(
      stderr,
      {
        fclose_count = 1;
        f.reset(nullptr);
        saved_fclose_count = fclose_count;
        fclose_count = 0;
      },
      system_error_message(EINTR, "cannot close file") + "\n");
  EXPECT_EQ(2, saved_fclose_count);
}

TEST(buffered_file_test, close_no_retry) {
  auto pipe = fmt::pipe();
  buffered_file f = pipe.read_end.fdopen("r");
  fclose_count = 1;
  EXPECT_SYSTEM_ERROR(f.close(), EINTR, "cannot close file");
  EXPECT_EQ(2, fclose_count);
  fclose_count = 0;
}

TEST(buffered_file_test, fileno_no_retry) {
  auto pipe = fmt::pipe();
  buffered_file f = pipe.read_end.fdopen("r");
  fileno_count = 1;
  EXPECT_SYSTEM_ERROR((f.descriptor)(), EINTR, "cannot get file descriptor");
  EXPECT_EQ(2, fileno_count);
  fileno_count = 0;
}
#endif  // FMT_USE_FCNTL

struct test_mock {
  static test_mock* instance;
}* test_mock::instance;

TEST(scoped_mock, scope) {
  {
    scoped_mock<test_mock> mock;
    EXPECT_EQ(&mock, test_mock::instance);
    test_mock& copy = mock;
    static_cast<void>(copy);
  }
  EXPECT_EQ(nullptr, test_mock::instance);
}
