// Formatting library for C++ - formatting library tests
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#include "fmt/xchar.h"

#include <algorithm>
#include <complex>
#include <cwchar>
#include <vector>

#include "fmt/chrono.h"
#include "fmt/color.h"
#include "fmt/ostream.h"
#include "fmt/ranges.h"
#include "fmt/std.h"
#include "gtest-extra.h"  // Contains
#include "util.h"         // get_locale

using fmt::detail::max_value;
using testing::Contains;

#if defined(__MINGW32__) && !defined(_UCRT)
// Only C89 conversion specifiers when using MSVCRT instead of UCRT
#  define FMT_HAS_C99_STRFTIME 0
#else
#  define FMT_HAS_C99_STRFTIME 1
#endif

struct non_string {};

template <typename T> class has_to_string_view_test : public testing::Test {};

using string_char_types = testing::Types<char, wchar_t, char16_t, char32_t>;
TYPED_TEST_SUITE(has_to_string_view_test, string_char_types);

template <typename Char>
struct derived_from_string_view : fmt::basic_string_view<Char> {};

TYPED_TEST(has_to_string_view_test, has_to_string_view) {
  EXPECT_TRUE(fmt::detail::has_to_string_view<TypeParam*>::value);
  EXPECT_TRUE(fmt::detail::has_to_string_view<const TypeParam*>::value);
  EXPECT_TRUE(fmt::detail::has_to_string_view<TypeParam[2]>::value);
  EXPECT_TRUE(fmt::detail::has_to_string_view<const TypeParam[2]>::value);
  EXPECT_TRUE(
      fmt::detail::has_to_string_view<std::basic_string<TypeParam>>::value);
  EXPECT_TRUE(fmt::detail::has_to_string_view<
              fmt::basic_string_view<TypeParam>>::value);
  EXPECT_TRUE(fmt::detail::has_to_string_view<
              derived_from_string_view<TypeParam>>::value);
  using fmt_string_view = fmt::detail::std_string_view<TypeParam>;
  EXPECT_TRUE(std::is_empty<fmt_string_view>::value !=
              fmt::detail::has_to_string_view<fmt_string_view>::value);
  EXPECT_FALSE(fmt::detail::has_to_string_view<non_string>::value);
}

// std::is_constructible is broken in MSVC until version 2015.
#if !FMT_MSC_VERSION || FMT_MSC_VERSION >= 1900
struct explicitly_convertible_to_wstring_view {
  explicit operator fmt::wstring_view() const { return L"foo"; }
};

TEST(xchar_test, format_explicitly_convertible_to_wstring_view) {
  // Types explicitly convertible to wstring_view are not formattable by
  // default because it may introduce ODR violations.
  static_assert(
      !fmt::is_formattable<explicitly_convertible_to_wstring_view>::value, "");
}
#endif

TEST(xchar_test, format) {
  EXPECT_EQ(fmt::format(L"{}", 42), L"42");
  EXPECT_EQ(fmt::format(L"{}", 4.2), L"4.2");
  EXPECT_EQ(fmt::format(L"{}", L"abc"), L"abc");
  EXPECT_EQ(fmt::format(L"{}", L'z'), L"z");
  EXPECT_THROW(fmt::format(fmt::runtime(L"{:*\x343E}"), 42), fmt::format_error);
  EXPECT_EQ(fmt::format(L"{}", true), L"true");
  EXPECT_EQ(fmt::format(L"{0}", L'a'), L"a");
  EXPECT_EQ(fmt::format(L"Letter {}", L'\x40e'), L"Letter \x40e");  // Ў
  if (sizeof(wchar_t) == 4)
    EXPECT_EQ(fmt::format(fmt::runtime(L"{:𓀨>3}"), 42), L"𓀨42");
  EXPECT_EQ(fmt::format(L"{}c{}", L"ab", 1), L"abc1");
}

TEST(xchar_test, is_formattable) {
  static_assert(!fmt::is_formattable<const wchar_t*>::value, "");
}

TEST(xchar_test, compile_time_string) {
  EXPECT_EQ(fmt::format(fmt::wformat_string<int>(L"{}"), 42), L"42");
#if defined(FMT_USE_STRING_VIEW) && FMT_CPLUSPLUS >= 201703L
  EXPECT_EQ(fmt::format(FMT_STRING(std::wstring_view(L"{}")), 42), L"42");
#endif
}

TEST(xchar_test, format_to) {
  auto buf = std::vector<wchar_t>();
  fmt::format_to(std::back_inserter(buf), L"{}{}", 42, L'\0');
  EXPECT_STREQ(buf.data(), L"42");
}

TEST(xchar_test, compile_time_string_format_to) {
  std::wstring ws;
  fmt::format_to(std::back_inserter(ws), FMT_STRING(L"{}"), 42);
  EXPECT_EQ(L"42", ws);
}

TEST(xchar_test, vformat_to) {
  int n = 42;
  auto args = fmt::make_wformat_args(n);
  auto w = std::wstring();
  fmt::vformat_to(std::back_inserter(w), L"{}", args);
  EXPECT_EQ(L"42", w);
}

namespace test {
struct struct_as_wstring_view {};
auto format_as(struct_as_wstring_view) -> fmt::wstring_view { return L"foo"; }
}  // namespace test

TEST(xchar_test, format_as) {
  EXPECT_EQ(fmt::format(L"{}", test::struct_as_wstring_view()), L"foo");
}

TEST(format_test, wide_format_to_n) {
  wchar_t buffer[4];
  buffer[3] = L'x';
  auto result = fmt::format_to_n(buffer, 3, L"{}", 12345);
  EXPECT_EQ(5u, result.size);
  EXPECT_EQ(buffer + 3, result.out);
  EXPECT_EQ(L"123x", fmt::wstring_view(buffer, 4));
  buffer[0] = L'x';
  buffer[1] = L'x';
  buffer[2] = L'x';
  result = fmt::format_to_n(buffer, 3, L"{}", L'A');
  EXPECT_EQ(1u, result.size);
  EXPECT_EQ(buffer + 1, result.out);
  EXPECT_EQ(L"Axxx", fmt::wstring_view(buffer, 4));
  result = fmt::format_to_n(buffer, 3, L"{}{} ", L'B', L'C');
  EXPECT_EQ(3u, result.size);
  EXPECT_EQ(buffer + 3, result.out);
  EXPECT_EQ(L"BC x", fmt::wstring_view(buffer, 4));
}

TEST(xchar_test, named_arg_udl) {
  using namespace fmt::literals;
  auto udl_a =
      fmt::format(L"{first}{second}{first}{third}", L"first"_a = L"abra",
                  L"second"_a = L"cad", L"third"_a = 99);
  EXPECT_EQ(
      fmt::format(L"{first}{second}{first}{third}", fmt::arg(L"first", L"abra"),
                  fmt::arg(L"second", L"cad"), fmt::arg(L"third", 99)),
      udl_a);
}

TEST(xchar_test, print) {
  // Check that the wide print overload compiles.
  if (fmt::detail::const_check(false)) {
    fmt::print(L"test");
    fmt::println(L"test");
  }
}

TEST(xchar_test, join) {
  int v[3] = {1, 2, 3};
  EXPECT_EQ(fmt::format(L"({})", fmt::join(v, v + 3, L", ")), L"(1, 2, 3)");
  auto t = std::tuple<wchar_t, int, float>('a', 1, 2.0f);
  EXPECT_EQ(fmt::format(L"({})", fmt::join(t, L", ")), L"(a, 1, 2)");
}

#ifdef __cpp_lib_byte
TEST(xchar_test, join_bytes) {
  auto v = std::vector<std::byte>{std::byte(1), std::byte(2), std::byte(3)};
  EXPECT_EQ(fmt::format(L"{}", fmt::join(v, L", ")), L"1, 2, 3");
}
#endif

enum streamable_enum {};

std::wostream& operator<<(std::wostream& os, streamable_enum) {
  return os << L"streamable_enum";
}

namespace fmt {
template <>
struct formatter<streamable_enum, wchar_t> : basic_ostream_formatter<wchar_t> {
};
}  // namespace fmt

enum unstreamable_enum {};
auto format_as(unstreamable_enum e) -> int { return e; }

TEST(xchar_test, enum) {
  EXPECT_EQ(L"streamable_enum", fmt::format(L"{}", streamable_enum()));
  EXPECT_EQ(L"0", fmt::format(L"{}", unstreamable_enum()));
}

struct streamable_and_unformattable {};

auto operator<<(std::wostream& os, streamable_and_unformattable)
    -> std::wostream& {
  return os << L"foo";
}

TEST(xchar_test, streamed) {
  EXPECT_FALSE(fmt::is_formattable<streamable_and_unformattable>());
  EXPECT_EQ(fmt::format(L"{}", fmt::streamed(streamable_and_unformattable())),
            L"foo");
}

TEST(xchar_test, sign_not_truncated) {
  wchar_t format_str[] = {
      L'{', L':',
      '+' | static_cast<wchar_t>(1 << fmt::detail::num_bits<char>()), L'}', 0};
  EXPECT_THROW(fmt::format(fmt::runtime(format_str), 42), fmt::format_error);
}

TEST(xchar_test, chrono) {
  auto tm = std::tm();
  tm.tm_year = 116;
  tm.tm_mon = 3;
  tm.tm_mday = 25;
  tm.tm_hour = 11;
  tm.tm_min = 22;
  tm.tm_sec = 33;
  EXPECT_EQ(fmt::format("The date is {:%Y-%m-%d %H:%M:%S}.", tm),
            "The date is 2016-04-25 11:22:33.");
  EXPECT_EQ(L"42s", fmt::format(L"{}", std::chrono::seconds(42)));
  EXPECT_EQ(fmt::format(L"{:%F}", tm), L"2016-04-25");
  EXPECT_EQ(fmt::format(L"{:%T}", tm), L"11:22:33");

  auto t = fmt::sys_time<std::chrono::seconds>(std::chrono::seconds(290088000));
  EXPECT_EQ(fmt::format("{:%Y-%m-%d %H:%M:%S}", t), "1979-03-12 12:00:00");
}

TEST(xchar_test, color) {
  EXPECT_EQ(fmt::format(fg(fmt::rgb(255, 20, 30)), L"rgb(255,20,30) wide"),
            L"\x1b[38;2;255;020;030mrgb(255,20,30) wide\x1b[0m");
}

TEST(xchar_test, ostream) {
#if !FMT_GCC_VERSION || FMT_GCC_VERSION >= 409
  {
    std::wostringstream wos;
    fmt::print(wos, L"Don't {}!", L"panic");
    EXPECT_EQ(wos.str(), L"Don't panic!");
  }

  {
    std::wostringstream wos;
    fmt::println(wos, L"Don't {}!", L"panic");
    EXPECT_EQ(wos.str(), L"Don't panic!\n");
  }
#endif
}

TEST(xchar_test, format_map) {
  auto m = std::map<std::wstring, int>{{L"one", 1}, {L"t\"wo", 2}};
  EXPECT_EQ(fmt::format(L"{}", m), L"{\"one\": 1, \"t\\\"wo\": 2}");
}

TEST(xchar_test, escape_string) {
  using vec = std::vector<std::wstring>;
  EXPECT_EQ(fmt::format(L"{}", vec{L"\n\r\t\"\\"}), L"[\"\\n\\r\\t\\\"\\\\\"]");
  EXPECT_EQ(fmt::format(L"{}", vec{L"понедельник"}), L"[\"понедельник\"]");
}

TEST(xchar_test, to_wstring) { EXPECT_EQ(L"42", fmt::to_wstring(42)); }

#ifndef FMT_STATIC_THOUSANDS_SEPARATOR

template <typename Char> struct numpunct : std::numpunct<Char> {
 protected:
  Char do_decimal_point() const override { return '?'; }
  std::string do_grouping() const override { return "\03"; }
  Char do_thousands_sep() const override { return '~'; }
};

template <typename Char> struct no_grouping : std::numpunct<Char> {
 protected:
  Char do_decimal_point() const override { return '.'; }
  std::string do_grouping() const override { return ""; }
  Char do_thousands_sep() const override { return ','; }
};

template <typename Char> struct special_grouping : std::numpunct<Char> {
 protected:
  Char do_decimal_point() const override { return '.'; }
  std::string do_grouping() const override { return "\03\02"; }
  Char do_thousands_sep() const override { return ','; }
};

template <typename Char> struct small_grouping : std::numpunct<Char> {
 protected:
  Char do_decimal_point() const override { return '.'; }
  std::string do_grouping() const override { return "\01"; }
  Char do_thousands_sep() const override { return ','; }
};

TEST(locale_test, localized_double) {
  auto loc = std::locale(std::locale(), new numpunct<char>());
  EXPECT_EQ(fmt::format(loc, "{:L}", 1.23), "1?23");
  EXPECT_EQ(fmt::format(loc, "{:Lf}", 1.23), "1?230000");
  EXPECT_EQ(fmt::format(loc, "{:L}", 1234.5), "1~234?5");
  EXPECT_EQ(fmt::format(loc, "{:L}", 12000.0), "12~000");
  EXPECT_EQ(fmt::format(loc, "{:8L}", 1230.0), "   1~230");
  EXPECT_EQ(fmt::format(loc, "{:15.6Lf}", 0.1), "       0?100000");
  EXPECT_EQ(fmt::format(loc, "{:15.6Lf}", 1.0), "       1?000000");
  EXPECT_EQ(fmt::format(loc, "{:15.6Lf}", 1e3), "   1~000?000000");
}

TEST(locale_test, format) {
  auto loc = std::locale(std::locale(), new numpunct<char>());
  EXPECT_EQ("1234567", fmt::format(std::locale(), "{:L}", 1234567));
  EXPECT_EQ("1~234~567", fmt::format(loc, "{:L}", 1234567));
  EXPECT_EQ("-1~234~567", fmt::format(loc, "{:L}", -1234567));
  EXPECT_EQ("-256", fmt::format(loc, "{:L}", -256));
  auto n = 1234567;
  EXPECT_EQ("1~234~567", fmt::vformat(loc, "{:L}", fmt::make_format_args(n)));
  auto s = std::string();
  fmt::format_to(std::back_inserter(s), loc, "{:L}", 1234567);
  EXPECT_EQ("1~234~567", s);

  auto no_grouping_loc = std::locale(std::locale(), new no_grouping<char>());
  EXPECT_EQ("1234567", fmt::format(no_grouping_loc, "{:L}", 1234567));

  auto special_grouping_loc =
      std::locale(std::locale(), new special_grouping<char>());
  EXPECT_EQ("1,23,45,678", fmt::format(special_grouping_loc, "{:L}", 12345678));
  EXPECT_EQ("12,345", fmt::format(special_grouping_loc, "{:L}", 12345));

  auto small_grouping_loc =
      std::locale(std::locale(), new small_grouping<char>());
  EXPECT_EQ("4,2,9,4,9,6,7,2,9,5",
            fmt::format(small_grouping_loc, "{:L}", max_value<uint32_t>()));
}

TEST(locale_test, format_default_align) {
  auto loc = std::locale({}, new special_grouping<char>());
  EXPECT_EQ("  12,345", fmt::format(loc, "{:8L}", 12345));
}

TEST(locale_test, format_plus) {
  auto loc = std::locale({}, new special_grouping<char>());
  EXPECT_EQ("+100", fmt::format(loc, "{:+L}", 100));
}

TEST(locale_test, wformat) {
  auto loc = std::locale(std::locale(), new numpunct<wchar_t>());
  EXPECT_EQ(L"1234567", fmt::format(std::locale(), L"{:L}", 1234567));
  EXPECT_EQ(L"1~234~567", fmt::format(loc, L"{:L}", 1234567));
  int n = 1234567;
  EXPECT_EQ(L"1~234~567",
            fmt::vformat(loc, L"{:L}", fmt::make_wformat_args(n)));
  EXPECT_EQ(L"1234567", fmt::format(std::locale("C"), L"{:L}", 1234567));

  auto no_grouping_loc = std::locale(std::locale(), new no_grouping<wchar_t>());
  EXPECT_EQ(L"1234567", fmt::format(no_grouping_loc, L"{:L}", 1234567));

  auto special_grouping_loc =
      std::locale(std::locale(), new special_grouping<wchar_t>());
  EXPECT_EQ(L"1,23,45,678",
            fmt::format(special_grouping_loc, L"{:L}", 12345678));

  auto small_grouping_loc =
      std::locale(std::locale(), new small_grouping<wchar_t>());
  EXPECT_EQ(L"4,2,9,4,9,6,7,2,9,5",
            fmt::format(small_grouping_loc, L"{:L}", max_value<uint32_t>()));
}

TEST(locale_test, int_formatter) {
  auto loc = std::locale(std::locale(), new special_grouping<char>());
  auto f = fmt::formatter<int>();
  auto parse_ctx = fmt::format_parse_context("L");
  f.parse(parse_ctx);
  auto buf = fmt::memory_buffer();
  fmt::basic_format_context<fmt::appender, char> format_ctx(
      fmt::appender(buf), {}, fmt::detail::locale_ref(loc));
  f.format(12345, format_ctx);
  EXPECT_EQ(fmt::to_string(buf), "12,345");
}

TEST(locale_test, chrono_weekday) {
  auto loc = get_locale("es_ES.UTF-8", "Spanish_Spain.1252");
  auto loc_old = std::locale::global(loc);
  auto sat = fmt::weekday(6);
  EXPECT_EQ(fmt::format(L"{}", sat), L"Sat");
  if (loc != std::locale::classic()) {
    // L'\341' is 'á'.
    auto saturdays =
        std::vector<std::wstring>{L"s\341b", L"s\341.", L"s\341b."};
    EXPECT_THAT(saturdays, Contains(fmt::format(loc, L"{:L}", sat)));
  }
  std::locale::global(loc_old);
}

TEST(locale_test, sign) {
  EXPECT_EQ(fmt::format(std::locale(), L"{:L}", -50), L"-50");
}

TEST(std_test_xchar, format_bitset) {
  auto bs = std::bitset<6>(42);
  EXPECT_EQ(fmt::format(L"{}", bs), L"101010");
  EXPECT_EQ(fmt::format(L"{:0>8}", bs), L"00101010");
  EXPECT_EQ(fmt::format(L"{:-^12}", bs), L"---101010---");
}

TEST(std_test_xchar, complex) {
  auto s = fmt::format(L"{}", std::complex<double>(1, 2));
  EXPECT_EQ(s, L"(1+2i)");
  EXPECT_EQ(fmt::format(L"{:.2f}", std::complex<double>(1, 2)),
            L"(1.00+2.00i)");
  EXPECT_EQ(fmt::format(L"{:8}", std::complex<double>(1, 2)), L"(1+2i)  ");
  EXPECT_EQ(fmt::format(L"{:-<8}", std::complex<double>(1, 2)), L"(1+2i)--");
}

TEST(std_test_xchar, optional) {
#  ifdef __cpp_lib_optional
  EXPECT_EQ(fmt::format(L"{}", std::optional{L'C'}), L"optional(\'C\')");
  EXPECT_EQ(fmt::format(L"{}", std::optional{std::wstring{L"wide string"}}),
            L"optional(\"wide string\")");
#  endif
}

#endif  // FMT_STATIC_THOUSANDS_SEPARATOR
