// Formatting library for C++ - formatting library implementation tests
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#include <algorithm>
#include <cstring>

// clang-format off
#include "test-assert.h"
// clang-format on

#include "fmt/format.h"
#include "gmock/gmock.h"
#include "util.h"

using fmt::detail::bigint;
using fmt::detail::fp;
using fmt::detail::max_value;

static_assert(!std::is_copy_constructible<bigint>::value, "");
static_assert(!std::is_copy_assignable<bigint>::value, "");

TEST(bigint_test, construct) {
  EXPECT_EQ(fmt::to_string(bigint()), "");
  EXPECT_EQ(fmt::to_string(bigint(0x42)), "42");
  EXPECT_EQ(fmt::to_string(bigint(0x123456789abcedf0)), "123456789abcedf0");
}

TEST(bigint_test, compare) {
  bigint n1(42);
  bigint n2(42);
  EXPECT_EQ(compare(n1, n2), 0);
  n2 <<= 32;
  EXPECT_LT(compare(n1, n2), 0);
  bigint n3(43);
  EXPECT_LT(compare(n1, n3), 0);
  EXPECT_GT(compare(n3, n1), 0);
  bigint n4(42 * 0x100000001);
  EXPECT_LT(compare(n2, n4), 0);
  EXPECT_GT(compare(n4, n2), 0);
}

TEST(bigint_test, add_compare) {
  EXPECT_LT(
      add_compare(bigint(0xffffffff), bigint(0xffffffff), bigint(1) <<= 64), 0);
  EXPECT_LT(add_compare(bigint(1) <<= 32, bigint(1), bigint(1) <<= 96), 0);
  EXPECT_GT(add_compare(bigint(1) <<= 32, bigint(0), bigint(0xffffffff)), 0);
  EXPECT_GT(add_compare(bigint(0), bigint(1) <<= 32, bigint(0xffffffff)), 0);
  EXPECT_GT(add_compare(bigint(42), bigint(1), bigint(42)), 0);
  EXPECT_GT(add_compare(bigint(0xffffffff), bigint(1), bigint(0xffffffff)), 0);
  EXPECT_LT(add_compare(bigint(10), bigint(10), bigint(22)), 0);
  EXPECT_LT(add_compare(bigint(0x100000010), bigint(0x100000010),
                        bigint(0x300000010)),
            0);
  EXPECT_GT(add_compare(bigint(0x1ffffffff), bigint(0x100000002),
                        bigint(0x300000000)),
            0);
  EXPECT_EQ(add_compare(bigint(0x1ffffffff), bigint(0x100000002),
                        bigint(0x300000001)),
            0);
  EXPECT_LT(add_compare(bigint(0x1ffffffff), bigint(0x100000002),
                        bigint(0x300000002)),
            0);
  EXPECT_LT(add_compare(bigint(0x1ffffffff), bigint(0x100000002),
                        bigint(0x300000003)),
            0);
}

TEST(bigint_test, shift_left) {
  bigint n(0x42);
  n <<= 0;
  EXPECT_EQ(fmt::to_string(n), "42");
  n <<= 1;
  EXPECT_EQ(fmt::to_string(n), "84");
  n <<= 25;
  EXPECT_EQ(fmt::to_string(n), "108000000");
}

TEST(bigint_test, multiply) {
  bigint n(0x42);
  EXPECT_THROW(n *= 0, assertion_failure);
  n *= 1;
  EXPECT_EQ(fmt::to_string(n), "42");

  n *= 2;
  EXPECT_EQ(fmt::to_string(n), "84");
  n *= 0x12345678;
  EXPECT_EQ(fmt::to_string(n), "962fc95e0");

  bigint bigmax(max_value<uint32_t>());
  bigmax *= max_value<uint32_t>();
  EXPECT_EQ(fmt::to_string(bigmax), "fffffffe00000001");

  const auto max64 = max_value<uint64_t>();
  bigmax = max64;
  bigmax *= max64;
  EXPECT_EQ(fmt::to_string(bigmax), "fffffffffffffffe0000000000000001");

  const auto max128 = (fmt::detail::uint128_t(max64) << 64) | max64;
  bigmax = max128;
  bigmax *= max128;
  EXPECT_EQ(fmt::to_string(bigmax),
            "fffffffffffffffffffffffffffffffe00000000000000000000000000000001");
}

TEST(bigint_test, square) {
  bigint n0(0);
  n0.square();
  EXPECT_EQ(fmt::to_string(n0), "0");
  bigint n1(0x100);
  n1.square();
  EXPECT_EQ(fmt::to_string(n1), "10000");
  bigint n2(0xfffffffff);
  n2.square();
  EXPECT_EQ(fmt::to_string(n2), "ffffffffe000000001");
  bigint n3(max_value<uint64_t>());
  n3.square();
  EXPECT_EQ(fmt::to_string(n3), "fffffffffffffffe0000000000000001");
  bigint n4;
  n4.assign_pow10(10);
  EXPECT_EQ(fmt::to_string(n4), "2540be400");
}

TEST(bigint_test, divmod_assign_zero_divisor) {
  bigint zero(0);
  EXPECT_THROW(bigint(0).divmod_assign(zero), assertion_failure);
  EXPECT_THROW(bigint(42).divmod_assign(zero), assertion_failure);
}

TEST(bigint_test, divmod_assign_self) {
  bigint n(100);
  EXPECT_THROW(n.divmod_assign(n), assertion_failure);
}

TEST(bigint_test, divmod_assign_unaligned) {
  // (42 << 340) / pow(10, 100):
  bigint n1(42);
  n1 <<= 340;
  bigint n2;
  n2.assign_pow10(100);
  int result = n1.divmod_assign(n2);
  EXPECT_EQ(result, 9406);
  EXPECT_EQ(fmt::to_string(n1),
            "10f8353019583bfc29ffc8f564e1b9f9d819dbb4cf783e4507eca1539220p96");
}

TEST(bigint_test, divmod_assign) {
  // 100 / 10:
  bigint n1(100);
  int result = n1.divmod_assign(bigint(10));
  EXPECT_EQ(result, 10);
  EXPECT_EQ(fmt::to_string(n1), "0");
  // pow(10, 100) / (42 << 320):
  n1.assign_pow10(100);
  result = n1.divmod_assign(bigint(42) <<= 320);
  EXPECT_EQ(result, 111);
  EXPECT_EQ(fmt::to_string(n1),
            "13ad2594c37ceb0b2784c4ce0bf38ace408e211a7caab24308a82e8f10p96");
  // 42 / 100:
  bigint n2(42);
  n1.assign_pow10(2);
  result = n2.divmod_assign(n1);
  EXPECT_EQ(result, 0);
  EXPECT_EQ(fmt::to_string(n2), "2a");
}

template <bool is_iec559> void run_double_tests() {
  fmt::print("warning: double is not IEC559, skipping FP tests\n");
}

template <> void run_double_tests<true>() {
  // Construct from double.
  EXPECT_EQ(fp(1.23), fp(0x13ae147ae147aeu, -52));
}

TEST(fp_test, double_tests) {
  run_double_tests<std::numeric_limits<double>::is_iec559>();
}

TEST(fp_test, normalize) {
  const auto v = fp(0xbeef, 42);
  auto normalized = normalize(v);
  EXPECT_EQ(normalized.f, 0xbeef000000000000);
  EXPECT_EQ(normalized.e, -6);
}

TEST(fp_test, multiply) {
  auto v = fp(123ULL << 32, 4) * fp(56ULL << 32, 7);
  EXPECT_EQ(v.f, 123u * 56u);
  EXPECT_EQ(v.e, 4 + 7 + 64);
  v = fp(123ULL << 32, 4) * fp(567ULL << 31, 8);
  EXPECT_EQ(v.f, (123 * 567 + 1u) / 2);
  EXPECT_EQ(v.e, 4 + 8 + 64);
}

TEST(fp_test, dragonbox_max_k) {
  using fmt::detail::dragonbox::floor_log10_pow2;
  using float_info = fmt::detail::dragonbox::float_info<float>;
  EXPECT_EQ(
      fmt::detail::const_check(float_info::max_k),
      float_info::kappa -
          floor_log10_pow2(std::numeric_limits<float>::min_exponent -
                           fmt::detail::num_significand_bits<float>() - 1));
  using double_info = fmt::detail::dragonbox::float_info<double>;
  EXPECT_EQ(fmt::detail::const_check(double_info::max_k),
            double_info::kappa -
                floor_log10_pow2(
                    std::numeric_limits<double>::min_exponent -
                    2 * fmt::detail::num_significand_bits<double>() - 1));
}

TEST(format_impl_test, format_error_code) {
  std::string msg = "error 42", sep = ": ";
  {
    auto buffer = fmt::memory_buffer();
    fmt::format_to(fmt::appender(buffer), "garbage");
    fmt::detail::format_error_code(buffer, 42, "test");
    EXPECT_EQ(to_string(buffer), "test: " + msg);
  }
  {
    auto buffer = fmt::memory_buffer();
    auto prefix =
        std::string(fmt::inline_buffer_size - msg.size() - sep.size() + 1, 'x');
    fmt::detail::format_error_code(buffer, 42, prefix);
    EXPECT_EQ(msg, to_string(buffer));
  }
  int codes[] = {42, -1};
  for (size_t i = 0, n = sizeof(codes) / sizeof(*codes); i < n; ++i) {
    // Test maximum buffer size.
    msg = fmt::format("error {}", codes[i]);
    fmt::memory_buffer buffer;
    auto prefix =
        std::string(fmt::inline_buffer_size - msg.size() - sep.size(), 'x');
    fmt::detail::format_error_code(buffer, codes[i], prefix);
    EXPECT_EQ(prefix + sep + msg, to_string(buffer));
    size_t size = fmt::inline_buffer_size;
    EXPECT_EQ(size, buffer.size());
    buffer.resize(0);
    // Test with a message that doesn't fit into the buffer.
    prefix += 'x';
    fmt::detail::format_error_code(buffer, codes[i], prefix);
    EXPECT_EQ(to_string(buffer), msg);
  }
}

// Tests fmt::detail::count_digits for integer type Int.
template <typename Int> void test_count_digits() {
  for (Int i = 0; i < 10; ++i) EXPECT_EQ(1u, fmt::detail::count_digits(i));
  for (Int i = 1, n = 1, end = max_value<Int>() / 10; n <= end; ++i) {
    n *= 10;
    EXPECT_EQ(fmt::detail::count_digits(n - 1), i);
    EXPECT_EQ(fmt::detail::count_digits(n), i + 1);
  }
}

TEST(format_impl_test, count_digits) {
  test_count_digits<uint32_t>();
  test_count_digits<uint64_t>();
}

TEST(format_impl_test, countl_zero) {
  constexpr auto num_bits = fmt::detail::num_bits<uint32_t>();
  uint32_t n = 1u;
  for (int i = 1; i < num_bits - 1; i++) {
    n <<= 1;
    EXPECT_EQ(fmt::detail::countl_zero(n - 1), num_bits - i);
    EXPECT_EQ(fmt::detail::countl_zero(n), num_bits - i - 1);
  }
}

#if FMT_USE_FLOAT128
TEST(format_impl_test, write_float128) {
  auto s = std::string();
  fmt::detail::write<char>(std::back_inserter(s), __float128(42));
  EXPECT_EQ(s, "42");
}
#endif

struct double_double {
  double a;
  double b;

  constexpr explicit double_double(double a_val = 0, double b_val = 0)
      : a(a_val), b(b_val) {}

  operator double() const { return a + b; }
  auto operator-() const -> double_double { return double_double(-a, -b); }
};

auto format_as(double_double d) -> double { return d; }

bool operator>=(const double_double& lhs, const double_double& rhs) {
  return lhs.a + lhs.b >= rhs.a + rhs.b;
}

struct slow_float {
  float value;

  constexpr explicit slow_float(float val = 0) : value(val) {}
  operator float() const { return value; }
  auto operator-() const -> slow_float { return slow_float(-value); }
};

auto format_as(slow_float f) -> float { return f; }

namespace std {
template <> struct numeric_limits<double_double> {
  // is_iec559 is true for double-double in libstdc++.
  static constexpr bool is_iec559 = true;
  static constexpr int digits = 106;
  static constexpr int digits10 = 33;
};

template <> struct numeric_limits<slow_float> : numeric_limits<float> {};
}  // namespace std

FMT_BEGIN_NAMESPACE
namespace detail {
template <> struct is_floating_point<double_double> : std::true_type {};
template <> struct is_floating_point<slow_float> : std::true_type {};
template <> struct is_fast_float<slow_float> : std::false_type {};
namespace dragonbox {
template <> struct float_info<slow_float> {
  using carrier_uint = uint32_t;
  static const int exponent_bits = 8;
};
}  // namespace dragonbox
}  // namespace detail
FMT_END_NAMESPACE

TEST(format_impl_test, write_double_double) {
  auto s = std::string();
  fmt::detail::write<char>(std::back_inserter(s), double_double(42), {});
  // Specializing is_floating_point is broken in MSVC.
  if (!FMT_MSC_VERSION) EXPECT_EQ(s, "42");
}

TEST(format_impl_test, write_dragon_even) {
  auto s = std::string();
  fmt::detail::write<char>(std::back_inserter(s), slow_float(33554450.0f), {});
  // Specializing is_floating_point is broken in MSVC.
  if (!FMT_MSC_VERSION) EXPECT_EQ(s, "3.355445e+07");
}

#if defined(_WIN32) && !defined(FMT_USE_WRITE_CONSOLE)
#  include <windows.h>

TEST(format_impl_test, write_console_signature) {
  decltype(::WriteConsoleW)* p = fmt::detail::WriteConsoleW;
  (void)p;
}
#endif

// A public domain branchless UTF-8 decoder by Christopher Wellons:
// https://github.com/skeeto/branchless-utf8
constexpr bool unicode_is_surrogate(uint32_t c) {
  return c >= 0xD800U && c <= 0xDFFFU;
}

FMT_CONSTEXPR char* utf8_encode(char* s, uint32_t c) {
  if (c >= (1UL << 16)) {
    s[0] = static_cast<char>(0xf0 | (c >> 18));
    s[1] = static_cast<char>(0x80 | ((c >> 12) & 0x3f));
    s[2] = static_cast<char>(0x80 | ((c >> 6) & 0x3f));
    s[3] = static_cast<char>(0x80 | ((c >> 0) & 0x3f));
    return s + 4;
  } else if (c >= (1UL << 11)) {
    s[0] = static_cast<char>(0xe0 | (c >> 12));
    s[1] = static_cast<char>(0x80 | ((c >> 6) & 0x3f));
    s[2] = static_cast<char>(0x80 | ((c >> 0) & 0x3f));
    return s + 3;
  } else if (c >= (1UL << 7)) {
    s[0] = static_cast<char>(0xc0 | (c >> 6));
    s[1] = static_cast<char>(0x80 | ((c >> 0) & 0x3f));
    return s + 2;
  } else {
    s[0] = static_cast<char>(c);
    return s + 1;
  }
}

// Make sure it can decode every character
TEST(format_impl_test, utf8_decode_decode_all) {
  for (uint32_t i = 0; i < 0x10ffff; i++) {
    if (!unicode_is_surrogate(i)) {
      int e;
      uint32_t c;
      char buf[8] = {0};
      char* end = utf8_encode(buf, i);
      const char* res = fmt::detail::utf8_decode(buf, &c, &e);
      EXPECT_EQ(end, res);
      EXPECT_EQ(c, i);
      EXPECT_EQ(e, 0);
    }
  }
}

// Reject everything outside of U+0000..U+10FFFF
TEST(format_impl_test, utf8_decode_out_of_range) {
  for (uint32_t i = 0x110000; i < 0x1fffff; i++) {
    int e;
    uint32_t c;
    char buf[8] = {0};
    utf8_encode(buf, i);
    const char* end = fmt::detail::utf8_decode(buf, &c, &e);
    EXPECT_NE(e, 0);
    EXPECT_EQ(end - buf, 4);
  }
}

// Does it reject all surrogate halves?
TEST(format_impl_test, utf8_decode_surrogate_halves) {
  for (uint32_t i = 0xd800; i <= 0xdfff; i++) {
    int e;
    uint32_t c;
    char buf[8] = {0};
    utf8_encode(buf, i);
    fmt::detail::utf8_decode(buf, &c, &e);
    EXPECT_NE(e, 0);
  }
}

// How about non-canonical encodings?
TEST(format_impl_test, utf8_decode_non_canonical_encodings) {
  int e;
  uint32_t c;
  const char* end;

  char buf2[8] = {char(0xc0), char(0xA4)};
  end = fmt::detail::utf8_decode(buf2, &c, &e);
  EXPECT_NE(e, 0);           // non-canonical len 2
  EXPECT_EQ(end, buf2 + 2);  // non-canonical recover 2

  char buf3[8] = {char(0xe0), char(0x80), char(0xA4)};
  end = fmt::detail::utf8_decode(buf3, &c, &e);
  EXPECT_NE(e, 0);           // non-canonical len 3
  EXPECT_EQ(end, buf3 + 3);  // non-canonical recover 3

  char buf4[8] = {char(0xf0), char(0x80), char(0x80), char(0xA4)};
  end = fmt::detail::utf8_decode(buf4, &c, &e);
  EXPECT_NE(e, 0);           // non-canonical encoding len 4
  EXPECT_EQ(end, buf4 + 4);  // non-canonical recover 4
}

// Let's try some bogus byte sequences
TEST(format_impl_test, utf8_decode_bogus_byte_sequences) {
  int e;
  uint32_t c;

  // Invalid first byte
  char buf0[4] = {char(0xff)};
  auto len = fmt::detail::utf8_decode(buf0, &c, &e) - buf0;
  EXPECT_NE(e, 0);    // "bogus [ff] 0x%02x U+%04lx", e, (unsigned long)c);
  EXPECT_EQ(len, 1);  // "bogus [ff] recovery %d", len);

  // Invalid first byte
  char buf1[4] = {char(0x80)};
  len = fmt::detail::utf8_decode(buf1, &c, &e) - buf1;
  EXPECT_NE(e, 0);    // "bogus [80] 0x%02x U+%04lx", e, (unsigned long)c);
  EXPECT_EQ(len, 1);  // "bogus [80] recovery %d", len);

  // Looks like a two-byte sequence but second byte is wrong
  char buf2[4] = {char(0xc0), char(0x0a)};
  len = fmt::detail::utf8_decode(buf2, &c, &e) - buf2;
  EXPECT_NE(e, 0);    // "bogus [c0 0a] 0x%02x U+%04lx", e, (unsigned long)c
  EXPECT_EQ(len, 2);  // "bogus [c0 0a] recovery %d", len);
}

TEST(format_impl_test, to_utf8) {
  auto s = std::string("ёжик");
  auto u = fmt::detail::to_utf8<wchar_t>(L"\x0451\x0436\x0438\x043A");
  EXPECT_EQ(s, u.str());
  EXPECT_EQ(s.size(), u.size());
}
