// Formatting library for C++ - formatting library tests
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#include "fmt/compile.h"
#include "gmock/gmock.h"

#if FMT_USE_CONSTEVAL

template <size_t max_string_length, typename Char = char> struct test_string {
  template <typename T> constexpr bool operator==(const T& rhs) const noexcept {
    return fmt::basic_string_view<Char>(rhs).compare(buffer) == 0;
  }
  Char buffer[max_string_length]{};
};

template <size_t max_string_length, typename Char = char, typename... Args>
consteval auto test_format(auto format, const Args&... args) {
  test_string<max_string_length, Char> string{};
  fmt::format_to(string.buffer, format, args...);
  return string;
}

TEST(compile_time_formatting_test, floating_point) {
  EXPECT_EQ("0", test_format<2>(FMT_COMPILE("{}"), 0.0f));
  EXPECT_EQ("392.500000", test_format<11>(FMT_COMPILE("{0:f}"), 392.5f));

  EXPECT_EQ("0", test_format<2>(FMT_COMPILE("{:}"), 0.0));
  EXPECT_EQ("0.000000", test_format<9>(FMT_COMPILE("{:f}"), 0.0));
  EXPECT_EQ("0", test_format<2>(FMT_COMPILE("{:g}"), 0.0));
  EXPECT_EQ("392.65", test_format<7>(FMT_COMPILE("{:}"), 392.65));
  EXPECT_EQ("392.65", test_format<7>(FMT_COMPILE("{:g}"), 392.65));
  EXPECT_EQ("392.65", test_format<7>(FMT_COMPILE("{:G}"), 392.65));
  EXPECT_EQ("4.9014e+06", test_format<11>(FMT_COMPILE("{:g}"), 4.9014e6));
  EXPECT_EQ("-392.650000", test_format<12>(FMT_COMPILE("{:f}"), -392.65));
  EXPECT_EQ("-392.650000", test_format<12>(FMT_COMPILE("{:F}"), -392.65));

  EXPECT_EQ("3.926500e+02", test_format<13>(FMT_COMPILE("{0:e}"), 392.65));
  EXPECT_EQ("3.926500E+02", test_format<13>(FMT_COMPILE("{0:E}"), 392.65));
  EXPECT_EQ("+0000392.6", test_format<11>(FMT_COMPILE("{0:+010.4g}"), 392.65));
  EXPECT_EQ("9223372036854775808.000000",
            test_format<27>(FMT_COMPILE("{:f}"), 9223372036854775807.0));

  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ("nan", test_format<4>(FMT_COMPILE("{}"), nan));
  EXPECT_EQ("+nan", test_format<5>(FMT_COMPILE("{:+}"), nan));
  if (std::signbit(-nan))
    EXPECT_EQ("-nan", test_format<5>(FMT_COMPILE("{}"), -nan));
  else
    fmt::print("Warning: compiler doesn't handle negative NaN correctly");

  constexpr double inf = std::numeric_limits<double>::infinity();
  EXPECT_EQ("inf", test_format<4>(FMT_COMPILE("{}"), inf));
  EXPECT_EQ("+inf", test_format<5>(FMT_COMPILE("{:+}"), inf));
  EXPECT_EQ("-inf", test_format<5>(FMT_COMPILE("{}"), -inf));
}

#endif  // FMT_USE_CONSTEVAL
