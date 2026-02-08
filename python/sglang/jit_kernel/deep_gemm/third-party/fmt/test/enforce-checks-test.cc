// Formatting library for C++ - formatting library tests
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#include <iterator>
#include <vector>

#define I 42  // simulate https://en.cppreference.com/w/c/numeric/complex/I
#include "fmt/chrono.h"
#include "fmt/color.h"
#include "fmt/format.h"
#include "fmt/ostream.h"
#include "fmt/ranges.h"
#include "fmt/xchar.h"
#undef I

// Exercise the API to verify that everything we expect to can compile.
void test_format_api() {
  (void)fmt::format(FMT_STRING("{}"), 42);
  (void)fmt::format(FMT_STRING(L"{}"), 42);
  (void)fmt::format(FMT_STRING("noop"));

  (void)fmt::to_string(42);
  (void)fmt::to_wstring(42);

  std::vector<char> out;
  fmt::format_to(std::back_inserter(out), FMT_STRING("{}"), 42);

  char buffer[4];
  fmt::format_to_n(buffer, 3, FMT_STRING("{}"), 12345);

  wchar_t wbuffer[4];
  fmt::format_to_n(wbuffer, 3, FMT_STRING(L"{}"), 12345);
}

void test_chrono() {
  (void)fmt::format(FMT_STRING("{}"), std::chrono::seconds(42));
  (void)fmt::format(FMT_STRING(L"{}"), std::chrono::seconds(42));
}

void test_text_style() {
  fmt::print(fg(fmt::rgb(255, 20, 30)), FMT_STRING("{}"), "rgb(255,20,30)");
  (void)fmt::format(fg(fmt::rgb(255, 20, 30)), FMT_STRING("{}"),
                    "rgb(255,20,30)");

  fmt::text_style ts = fg(fmt::rgb(255, 20, 30));
  std::string out;
  fmt::format_to(std::back_inserter(out), ts,
                 FMT_STRING("rgb(255,20,30){}{}{}"), 1, 2, 3);
}

void test_range() {
  std::vector<char> hello = {'h', 'e', 'l', 'l', 'o'};
  (void)fmt::format(FMT_STRING("{}"), hello);
}

int main() {
  test_format_api();
  test_chrono();
  test_text_style();
  test_range();
}
