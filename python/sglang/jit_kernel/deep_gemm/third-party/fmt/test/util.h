// Formatting library for C++ - test utilities
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#include <cstdarg>
#include <cstdio>
#include <locale>
#include <string>

#include "fmt/os.h"

#ifdef _MSC_VER
#  define FMT_VSNPRINTF vsprintf_s
#else
#  define FMT_VSNPRINTF vsnprintf
#endif

template <size_t SIZE>
void safe_sprintf(char (&buffer)[SIZE], const char* format, ...) {
  std::va_list args;
  va_start(args, format);
  FMT_VSNPRINTF(buffer, SIZE, format, args);
  va_end(args);
}

extern const char* const file_content;

// Opens a buffered file for reading.
auto open_buffered_file(FILE** fp = nullptr) -> fmt::buffered_file;

template <typename Char> class basic_test_string {
 private:
  std::basic_string<Char> value_;

  static const Char empty[];

 public:
  explicit basic_test_string(const Char* value = empty) : value_(value) {}

  auto value() const -> const std::basic_string<Char>& { return value_; }
};

template <typename Char> const Char basic_test_string<Char>::empty[] = {0};

using test_string = basic_test_string<char>;
using test_wstring = basic_test_string<wchar_t>;

template <typename Char>
auto operator<<(std::basic_ostream<Char>& os, const basic_test_string<Char>& s)
    -> std::basic_ostream<Char>& {
  os << s.value();
  return os;
}

class date {
  int year_, month_, day_;

 public:
  date(int year, int month, int day) : year_(year), month_(month), day_(day) {}

  auto year() const -> int { return year_; }
  auto month() const -> int { return month_; }
  auto day() const -> int { return day_; }
};

// Returns a locale with the given name if available or classic locale
// otherwise.
auto get_locale(const char* name, const char* alt_name = nullptr)
    -> std::locale;
