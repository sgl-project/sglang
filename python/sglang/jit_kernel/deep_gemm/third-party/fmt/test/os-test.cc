// Formatting library for C++ - tests of the OS-specific functionality
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#include "fmt/os.h"

#include <cstdlib>  // std::exit
#include <cstring>
#include <memory>
#include <thread>

#include "gtest-extra.h"
#include "util.h"

using fmt::buffered_file;
using testing::HasSubstr;
using wstring_view = fmt::basic_string_view<wchar_t>;

static auto uniq_file_name(unsigned line_number) -> std::string {
  return "test-file" + std::to_string(line_number);
}

auto safe_fopen(const char* filename, const char* mode) -> FILE* {
#if defined(_WIN32) && !defined(__MINGW32__)
  // Fix MSVC warning about "unsafe" fopen.
  FILE* f = nullptr;
  errno = fopen_s(&f, filename, mode);
  return f;
#else
  return std::fopen(filename, mode);
#endif
}

#ifdef _WIN32

#  include <windows.h>

TEST(os_test, format_windows_error) {
  LPWSTR message = nullptr;
  auto result = FormatMessageW(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, ERROR_FILE_EXISTS, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      reinterpret_cast<LPWSTR>(&message), 0, nullptr);
  auto utf8_message =
      fmt::detail::to_utf8<wchar_t>(wstring_view(message, result - 2));
  LocalFree(message);
  fmt::memory_buffer actual_message;
  fmt::detail::format_windows_error(actual_message, ERROR_FILE_EXISTS, "test");
  EXPECT_EQ(fmt::format("test: {}", utf8_message.str()),
            fmt::to_string(actual_message));
  actual_message.resize(0);
}

TEST(os_test, format_long_windows_error) {
  LPWSTR message = nullptr;
  // this error code is not available on all Windows platforms and
  // Windows SDKs, so do not fail the test if the error string cannot
  // be retrieved.
  int provisioning_not_allowed = 0x80284013L;  // TBS_E_PROVISIONING_NOT_ALLOWED
  auto result = FormatMessageW(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, static_cast<DWORD>(provisioning_not_allowed),
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      reinterpret_cast<LPWSTR>(&message), 0, nullptr);
  if (result == 0) {
    LocalFree(message);
    return;
  }
  auto utf8_message =
      fmt::detail::to_utf8<wchar_t>(wstring_view(message, result - 2));
  LocalFree(message);
  fmt::memory_buffer actual_message;
  fmt::detail::format_windows_error(actual_message, provisioning_not_allowed,
                                    "test");
  EXPECT_EQ(fmt::format("test: {}", utf8_message.str()),
            fmt::to_string(actual_message));
}

TEST(os_test, windows_error) {
  auto error = std::system_error(std::error_code());
  try {
    throw fmt::windows_error(ERROR_FILE_EXISTS, "test {}", "error");
  } catch (const std::system_error& e) {
    error = e;
  }
  fmt::memory_buffer message;
  fmt::detail::format_windows_error(message, ERROR_FILE_EXISTS, "test error");
  EXPECT_THAT(error.what(), HasSubstr(to_string(message)));
  EXPECT_EQ(ERROR_FILE_EXISTS, error.code().value());
}

TEST(os_test, report_windows_error) {
  fmt::memory_buffer out;
  fmt::detail::format_windows_error(out, ERROR_FILE_EXISTS, "test error");
  out.push_back('\n');
  EXPECT_WRITE(stderr,
               fmt::report_windows_error(ERROR_FILE_EXISTS, "test error"),
               fmt::to_string(out));
}

#  if FMT_USE_FCNTL && !defined(__MINGW32__)
TEST(file_test, open_windows_file) {
  using fmt::file;
  file out = file::open_windows_file(L"test-file",
                                     file::WRONLY | file::CREATE | file::TRUNC);
  out.write("x", 1);
  file in = file::open_windows_file(L"test-file", file::RDONLY);
  EXPECT_READ(in, "x");
}
#  endif  // FMT_USE_FCNTL && !defined(__MINGW32__)

#endif  // _WIN32

#if FMT_USE_FCNTL

using fmt::file;

auto isclosed(int fd) -> bool {
  char buffer;
  auto result = std::streamsize();
  SUPPRESS_ASSERT(result = FMT_POSIX(read(fd, &buffer, 1)));
  return result == -1 && errno == EBADF;
}

// Opens a file for reading.
auto open_file() -> file {
  auto pipe = fmt::pipe();
  pipe.write_end.write(file_content, std::strlen(file_content));
  pipe.write_end.close();
  return std::move(pipe.read_end);
}

// Attempts to write a string to a file.
void write(file& f, fmt::string_view s) {
  size_t num_chars_left = s.size();
  const char* ptr = s.data();
  do {
    size_t count = f.write(ptr, num_chars_left);
    ptr += count;
    // We can't write more than size_t bytes since num_chars_left
    // has type size_t.
    num_chars_left -= count;
  } while (num_chars_left != 0);
}

TEST(buffered_file_test, default_ctor) {
  auto f = buffered_file();
  EXPECT_TRUE(f.get() == nullptr);
}

TEST(buffered_file_test, move_ctor) {
  buffered_file bf = open_buffered_file();
  FILE* fp = bf.get();
  EXPECT_TRUE(fp != nullptr);
  buffered_file bf2(std::move(bf));
  EXPECT_EQ(fp, bf2.get());
  EXPECT_TRUE(bf.get() == nullptr);
}

TEST(buffered_file_test, move_assignment) {
  buffered_file bf = open_buffered_file();
  FILE* fp = bf.get();
  EXPECT_TRUE(fp != nullptr);
  buffered_file bf2;
  bf2 = std::move(bf);
  EXPECT_EQ(fp, bf2.get());
  EXPECT_TRUE(bf.get() == nullptr);
}

TEST(buffered_file_test, move_assignment_closes_file) {
  buffered_file bf = open_buffered_file();
  buffered_file bf2 = open_buffered_file();
  int old_fd = bf2.descriptor();
  bf2 = std::move(bf);
  EXPECT_TRUE(isclosed(old_fd));
}

TEST(buffered_file_test, move_from_temporary_in_ctor) {
  FILE* fp = nullptr;
  buffered_file f = open_buffered_file(&fp);
  EXPECT_EQ(fp, f.get());
}

TEST(buffered_file_test, move_from_temporary_in_assignment) {
  FILE* fp = nullptr;
  auto f = buffered_file();
  f = open_buffered_file(&fp);
  EXPECT_EQ(fp, f.get());
}

TEST(buffered_file_test, move_from_temporary_in_assignment_closes_file) {
  buffered_file f = open_buffered_file();
  int old_fd = f.descriptor();
  f = open_buffered_file();
  EXPECT_TRUE(isclosed(old_fd));
}

TEST(buffered_file_test, close_file_in_dtor) {
  int fd = 0;
  {
    buffered_file f = open_buffered_file();
    fd = f.descriptor();
  }
  EXPECT_TRUE(isclosed(fd));
}

TEST(buffered_file_test, close_error_in_dtor) {
  auto f =
      std::unique_ptr<buffered_file>(new buffered_file(open_buffered_file()));
  EXPECT_WRITE(
      stderr,
      {
        // The close function must be called inside EXPECT_WRITE,
        // otherwise the system may recycle closed file descriptor when
        // redirecting the output in EXPECT_STDERR and the second close
        // will break output redirection.
        FMT_POSIX(close(f->descriptor()));
        SUPPRESS_ASSERT(f.reset(nullptr));
      },
      system_error_message(EBADF, "cannot close file") + "\n");
}

TEST(buffered_file_test, close) {
  buffered_file f = open_buffered_file();
  int fd = f.descriptor();
  f.close();
  EXPECT_TRUE(f.get() == nullptr);
  EXPECT_TRUE(isclosed(fd));
}

TEST(buffered_file_test, close_error) {
  buffered_file f = open_buffered_file();
  FMT_POSIX(close(f.descriptor()));
  EXPECT_SYSTEM_ERROR_NOASSERT(f.close(), EBADF, "cannot close file");
  EXPECT_TRUE(f.get() == nullptr);
}

TEST(buffered_file_test, descriptor) {
  auto f = open_buffered_file();
  EXPECT_TRUE(f.descriptor() != -1);
  file copy = file::dup(f.descriptor());
  EXPECT_READ(copy, file_content);
}

TEST(ostream_test, move) {
  fmt::ostream out = fmt::output_file(uniq_file_name(__LINE__));
  fmt::ostream moved(std::move(out));
  moved.print("hello");
}

TEST(ostream_test, move_while_holding_data) {
  auto test_file = uniq_file_name(__LINE__);
  {
    fmt::ostream out = fmt::output_file(test_file);
    out.print("Hello, ");
    fmt::ostream moved(std::move(out));
    moved.print("world!\n");
  }
  {
    file in(test_file, file::RDONLY);
    EXPECT_READ(in, "Hello, world!\n");
  }
}

TEST(ostream_test, print) {
  auto test_file = uniq_file_name(__LINE__);
  fmt::ostream out = fmt::output_file(test_file);
  out.print("The answer is {}.\n", 42);
  out.close();
  file in(test_file, file::RDONLY);
  EXPECT_READ(in, "The answer is 42.\n");
}

TEST(ostream_test, buffer_boundary) {
  auto str = std::string(4096, 'x');
  auto test_file = uniq_file_name(__LINE__);
  fmt::ostream out = fmt::output_file(test_file);
  out.print("{}", str);
  out.print("{}", str);
  out.close();
  file in(test_file, file::RDONLY);
  EXPECT_READ(in, str + str);
}

TEST(ostream_test, buffer_size) {
  auto test_file = uniq_file_name(__LINE__);
  fmt::ostream out = fmt::output_file(test_file, fmt::buffer_size = 1);
  out.print("{}", "foo");
  out.close();
  file in(test_file, file::RDONLY);
  EXPECT_READ(in, "foo");
}

TEST(ostream_test, truncate) {
  auto test_file = uniq_file_name(__LINE__);
  {
    fmt::ostream out = fmt::output_file(test_file);
    out.print("0123456789");
  }
  {
    fmt::ostream out = fmt::output_file(test_file);
    out.print("foo");
  }
  file in(test_file, file::RDONLY);
  EXPECT_EQ("foo", read(in, 4));
}

TEST(ostream_test, flush) {
  auto test_file = uniq_file_name(__LINE__);
  auto out = fmt::output_file(test_file);
  out.print("x");
  out.flush();
  auto in = fmt::file(test_file, file::RDONLY);
  EXPECT_READ(in, "x");
}

TEST(file_test, default_ctor) {
  file f;
  EXPECT_EQ(-1, f.descriptor());
}

TEST(file_test, open_buffered_file_in_ctor) {
  auto test_file = uniq_file_name(__LINE__);
  FILE* fp = safe_fopen(test_file.c_str(), "w");
  std::fputs(file_content, fp);
  std::fclose(fp);
  file f(test_file.c_str(), file::RDONLY);
  // Check if the file is open by reading one character from it.
  char buffer;
  bool isopen = FMT_POSIX(read(f.descriptor(), &buffer, 1)) == 1;
  ASSERT_TRUE(isopen);
}

TEST(file_test, open_buffered_file_error) {
  EXPECT_SYSTEM_ERROR(file("nonexistent", file::RDONLY), ENOENT,
                      "cannot open file nonexistent");
}

TEST(file_test, move_ctor) {
  file f = open_file();
  int fd = f.descriptor();
  EXPECT_NE(-1, fd);
  file f2(std::move(f));
  EXPECT_EQ(fd, f2.descriptor());
  EXPECT_EQ(-1, f.descriptor());
}

TEST(file_test, move_assignment) {
  file f = open_file();
  int fd = f.descriptor();
  EXPECT_NE(-1, fd);
  file f2;
  f2 = std::move(f);
  EXPECT_EQ(fd, f2.descriptor());
  EXPECT_EQ(-1, f.descriptor());
}

TEST(file_test, move_assignment_closes_file) {
  file f = open_file();
  file f2 = open_file();
  int old_fd = f2.descriptor();
  f2 = std::move(f);
  EXPECT_TRUE(isclosed(old_fd));
}

file open_buffered_file(int& fd) {
  file f = open_file();
  fd = f.descriptor();
  return f;
}

TEST(file_test, move_from_temporary_in_ctor) {
  int fd = 0xdead;
  file f(open_buffered_file(fd));
  EXPECT_EQ(fd, f.descriptor());
}

TEST(file_test, move_from_temporary_in_assignment) {
  int fd = 0xdead;
  file f;
  f = open_buffered_file(fd);
  EXPECT_EQ(fd, f.descriptor());
}

TEST(file_test, move_from_temporary_in_assignment_closes_file) {
  int fd = 0xdead;
  file f = open_file();
  int old_fd = f.descriptor();
  f = open_buffered_file(fd);
  EXPECT_TRUE(isclosed(old_fd));
}

TEST(file_test, close_file_in_dtor) {
  int fd = 0;
  {
    file f = open_file();
    fd = f.descriptor();
  }
  EXPECT_TRUE(isclosed(fd));
}

TEST(file_test, close_error_in_dtor) {
  std::unique_ptr<file> f(new file(open_file()));
  EXPECT_WRITE(
      stderr,
      {
        // The close function must be called inside EXPECT_WRITE,
        // otherwise the system may recycle closed file descriptor when
        // redirecting the output in EXPECT_STDERR and the second close
        // will break output redirection.
        FMT_POSIX(close(f->descriptor()));
        SUPPRESS_ASSERT(f.reset(nullptr));
      },
      system_error_message(EBADF, "cannot close file") + "\n");
}

TEST(file_test, close) {
  file f = open_file();
  int fd = f.descriptor();
  f.close();
  EXPECT_EQ(-1, f.descriptor());
  EXPECT_TRUE(isclosed(fd));
}

TEST(file_test, close_error) {
  file f = open_file();
  FMT_POSIX(close(f.descriptor()));
  EXPECT_SYSTEM_ERROR_NOASSERT(f.close(), EBADF, "cannot close file");
  EXPECT_EQ(-1, f.descriptor());
}

TEST(file_test, read) {
  file f = open_file();
  EXPECT_READ(f, file_content);
}

TEST(file_test, read_error) {
  file f(uniq_file_name(__LINE__), file::WRONLY | file::CREATE);
  char buf;
  // We intentionally read from a file opened in the write-only mode to
  // cause error.
  EXPECT_SYSTEM_ERROR(f.read(&buf, 1), EBADF, "cannot read from file");
}

TEST(file_test, write) {
  auto pipe = fmt::pipe();
  write(pipe.write_end, "test");
  pipe.write_end.close();
  EXPECT_READ(pipe.read_end, "test");
}

TEST(file_test, write_error) {
  file f(uniq_file_name(__LINE__), file::RDONLY | file::CREATE);
  // We intentionally write to a file opened in the read-only mode to
  // cause error.
  EXPECT_SYSTEM_ERROR(f.write(" ", 1), EBADF, "cannot write to file");
}

TEST(file_test, dup) {
  file f = open_file();
  file copy = file::dup(f.descriptor());
  EXPECT_NE(f.descriptor(), copy.descriptor());
  EXPECT_EQ(file_content, read(copy, std::strlen(file_content)));
}

#  ifndef __COVERITY__
TEST(file_test, dup_error) {
  int value = -1;
  EXPECT_SYSTEM_ERROR_NOASSERT(file::dup(value), EBADF,
                               "cannot duplicate file descriptor -1");
}
#  endif

TEST(file_test, dup2) {
  file f = open_file();
  file copy = open_file();
  f.dup2(copy.descriptor());
  EXPECT_NE(f.descriptor(), copy.descriptor());
  EXPECT_READ(copy, file_content);
}

TEST(file_test, dup2_error) {
  file f = open_file();
  EXPECT_SYSTEM_ERROR_NOASSERT(
      f.dup2(-1), EBADF,
      fmt::format("cannot duplicate file descriptor {} to -1", f.descriptor()));
}

TEST(file_test, dup2_noexcept) {
  file f = open_file();
  file copy = open_file();
  std::error_code ec;
  f.dup2(copy.descriptor(), ec);
  EXPECT_EQ(ec.value(), 0);
  EXPECT_NE(f.descriptor(), copy.descriptor());
  EXPECT_READ(copy, file_content);
}

TEST(file_test, dup2_noexcept_error) {
  file f = open_file();
  std::error_code ec;
  SUPPRESS_ASSERT(f.dup2(-1, ec));
  EXPECT_EQ(EBADF, ec.value());
}

TEST(file_test, pipe) {
  auto pipe = fmt::pipe();
  EXPECT_NE(-1, pipe.read_end.descriptor());
  EXPECT_NE(-1, pipe.write_end.descriptor());
  write(pipe.write_end, "test");
  EXPECT_READ(pipe.read_end, "test");
}

TEST(file_test, fdopen) {
  auto pipe = fmt::pipe();
  int read_fd = pipe.read_end.descriptor();
  EXPECT_EQ(read_fd, FMT_POSIX(fileno(pipe.read_end.fdopen("r").get())));
}

// Windows CRT implements _IOLBF incorrectly (full buffering).
#  ifndef _WIN32
TEST(file_test, line_buffering) {
  auto pipe = fmt::pipe();

  int write_fd = pipe.write_end.descriptor();
  auto write_end = pipe.write_end.fdopen("w");
  setvbuf(write_end.get(), nullptr, _IOLBF, 4096);
  write_end.print("42\n");
  close(write_fd);
  try {
    write_end.close();
  } catch (const std::system_error&) {
  }

  auto read_end = pipe.read_end.fdopen("r");
  std::thread reader([&]() {
    int n = 0;
    int result = fscanf(read_end.get(), "%d", &n);
    (void)result;
    EXPECT_EQ(n, 42);
  });

  reader.join();
}
#  endif  // _WIN32

TEST(file_test, buffer_boundary) {
  auto pipe = fmt::pipe();

  auto write_end = pipe.write_end.fdopen("w");
  setvbuf(write_end.get(), nullptr, _IOFBF, 4096);
  for (int i = 3; i < 4094; i++)
    write_end.print("{}", (i % 73) != 0 ? 'x' : '\n');
  write_end.print("{} {}", 1234, 567);
  write_end.close();

  auto read_end = pipe.read_end.fdopen("r");
  char buf[4091] = {};
  size_t n = fread(buf, 1, sizeof(buf), read_end.get());
  EXPECT_EQ(n, sizeof(buf));
  EXPECT_STREQ(fgets(buf, sizeof(buf), read_end.get()), "1234 567");
}

TEST(file_test, io_putting) {
  auto pipe = fmt::pipe();
  auto read_end = pipe.read_end.fdopen("r");
  auto write_end = pipe.write_end.fdopen("w");

  size_t read_size = 0;
  auto reader = std::thread([&]() {
    size_t n = 0;
    do {
      char buf[4096] = {};
      n = fread(buf, 1, sizeof(buf), read_end.get());
      read_size += n;
    } while (n != 0);
  });

  // This initialize buffers but doesn't set _IO_CURRENTLY_PUTTING.
  fseek(write_end.get(), 0, SEEK_SET);

  size_t write_size = 0;
  for (int i = 0; i <= 20000; ++i) {
    auto s = fmt::format("{}\n", i);
    fmt::print(write_end.get(), "{}", s);
    write_size += s.size();
  }

  write_end.close();
  reader.join();
  EXPECT_EQ(read_size, write_size);
}

#endif  // FMT_USE_FCNTL
