// Formatting library for C++ - formatting library tests
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

// Check if fmt/format.h compiles with windows.h included before it.
#ifdef _WIN32
#  include <windows.h>
#endif
// clang-format off
#include "fmt/format.h"
// clang-format on

#include <stdint.h>  // uint32_t

#include <cfenv>               // fegetexceptflag and FE_ALL_EXCEPT
#include <climits>             // INT_MAX
#include <cmath>               // std::signbit
#include <condition_variable>  // std::condition_variable
#include <cstring>             // std::strlen
#include <iterator>            // std::back_inserter
#include <list>                // std::list
#include <mutex>               // std::mutex
#include <string>              // std::string
#include <thread>              // std::thread
#include <type_traits>         // std::is_default_constructible
#if FMT_CPLUSPLUS > 201703L && FMT_HAS_INCLUDE(<version>)
#  include <version>
#endif

#include <limits.h>

#include <limits>

#include "gtest-extra.h"
#include "mock-allocator.h"
#include "util.h"
using fmt::basic_memory_buffer;
using fmt::format_error;
using fmt::memory_buffer;
using fmt::runtime;
using fmt::string_view;
using fmt::detail::max_value;
using fmt::detail::uint128_fallback;

using testing::Return;
using testing::StrictMock;

#ifdef __cpp_lib_concepts
static_assert(std::output_iterator<fmt::appender, char>);
#endif

enum { buffer_size = 256 };

TEST(uint128_test, ctor) {
  auto n = uint128_fallback();
  EXPECT_EQ(n, 0);
  n = uint128_fallback(42);
  EXPECT_EQ(n, 42);
  EXPECT_EQ(static_cast<uint64_t>(n), 42);
}

TEST(uint128_test, shift) {
  auto n = uint128_fallback(42);
  n = n << 64;
  EXPECT_EQ(static_cast<uint64_t>(n), 0);
  n = n >> 64;
  EXPECT_EQ(static_cast<uint64_t>(n), 42);
  n = n << 62;
  EXPECT_EQ(static_cast<uint64_t>(n >> 64), 0xa);
  EXPECT_EQ(static_cast<uint64_t>(n), 0x8000000000000000);
  n = n >> 62;
  EXPECT_EQ(static_cast<uint64_t>(n), 42);
  EXPECT_EQ(uint128_fallback(1) << 112, uint128_fallback(0x1000000000000, 0));
  EXPECT_EQ(uint128_fallback(0x1000000000000, 0) >> 112, uint128_fallback(1));
}

TEST(uint128_test, minus) {
  auto n = uint128_fallback(42);
  EXPECT_EQ(n - 2, 40);
}

TEST(uint128_test, plus_assign) {
  auto n = uint128_fallback(32);
  n += uint128_fallback(10);
  EXPECT_EQ(n, 42);
  n = uint128_fallback(max_value<uint64_t>());
  n += uint128_fallback(1);
  EXPECT_EQ(n, uint128_fallback(1) << 64);
}

TEST(uint128_test, multiply) {
  auto n = uint128_fallback(2251799813685247);
  n = n * 3611864890;
  EXPECT_EQ(static_cast<uint64_t>(n >> 64), 440901);
}

template <typename Float> void check_isfinite() {
  using fmt::detail::isfinite;
  EXPECT_TRUE(isfinite(Float(0.0)));
  EXPECT_TRUE(isfinite(Float(42.0)));
  EXPECT_TRUE(isfinite(Float(-42.0)));
  EXPECT_TRUE(isfinite(Float(fmt::detail::max_value<double>())));
  // Use double because std::numeric_limits is broken for __float128.
  using limits = std::numeric_limits<double>;
  FMT_CONSTEXPR20 auto result = isfinite(Float(limits::infinity()));
  EXPECT_FALSE(result);
  EXPECT_FALSE(isfinite(Float(limits::infinity())));
  EXPECT_FALSE(isfinite(Float(-limits::infinity())));
  EXPECT_FALSE(isfinite(Float(limits::quiet_NaN())));
  EXPECT_FALSE(isfinite(Float(-limits::quiet_NaN())));
}

TEST(float_test, isfinite) {
  check_isfinite<double>();
#if FMT_USE_FLOAT128
  check_isfinite<fmt::detail::float128>();
#endif
}

void check_no_fp_exception() {
  fexcept_t fe;
  fegetexceptflag(&fe, FE_ALL_EXCEPT);

  // No exception flags should have been set
  EXPECT_TRUE(fe == 0);
}

template <typename Float> void check_isnan() {
  using fmt::detail::isnan;
  EXPECT_FALSE(isnan(Float(0.0)));
  EXPECT_FALSE(isnan(Float(42.0)));
  EXPECT_FALSE(isnan(Float(-42.0)));
  EXPECT_FALSE(isnan(Float(fmt::detail::max_value<double>())));
  // Use double because std::numeric_limits is broken for __float128.
  using limits = std::numeric_limits<double>;
  EXPECT_FALSE(isnan(Float(limits::infinity())));
  EXPECT_FALSE(isnan(Float(-limits::infinity())));
  EXPECT_TRUE(isnan(Float(limits::quiet_NaN())));
  EXPECT_TRUE(isnan(Float(-limits::quiet_NaN())));

  // Sanity check: make sure no error has occurred before we start
  check_no_fp_exception();

  // Check that no exception is raised for the non-NaN case
  isnan(Float(42.0));
  check_no_fp_exception();

  // Check that no exception is raised for the NaN case
  isnan(Float(limits::quiet_NaN()));
  check_no_fp_exception();
}

TEST(float_test, isnan) {
  check_isnan<double>();
#if FMT_USE_FLOAT128
  check_isnan<fmt::detail::float128>();
#endif
}

struct uint32_pair {
  uint32_t u[2];
};

TEST(util_test, bit_cast) {
  auto s = fmt::detail::bit_cast<uint32_pair>(uint64_t{42});
  EXPECT_EQ(fmt::detail::bit_cast<uint64_t>(s), 42ull);
  s = fmt::detail::bit_cast<uint32_pair>(~uint64_t{0});
  EXPECT_EQ(fmt::detail::bit_cast<uint64_t>(s), ~0ull);
}

// Increment a number in a string.
void increment(char* s) {
  for (int i = static_cast<int>(std::strlen(s)) - 1; i >= 0; --i) {
    if (s[i] != '9') {
      ++s[i];
      break;
    }
    s[i] = '0';
  }
}

TEST(util_test, increment) {
  char s[10] = "123";
  increment(s);
  EXPECT_STREQ("124", s);
  s[2] = '8';
  increment(s);
  EXPECT_STREQ("129", s);
  increment(s);
  EXPECT_STREQ("130", s);
  s[1] = s[2] = '9';
  increment(s);
  EXPECT_STREQ("200", s);
}

TEST(util_test, parse_nonnegative_int) {
  auto s = fmt::string_view("10000000000");
  auto begin = s.begin(), end = s.end();
  EXPECT_EQ(fmt::detail::parse_nonnegative_int(begin, end, -1), -1);
  s = "2147483649";
  begin = s.begin();
  end = s.end();
  EXPECT_EQ(fmt::detail::parse_nonnegative_int(begin, end, -1), -1);
}

TEST(util_test, utf8_to_utf16) {
  auto u = fmt::detail::utf8_to_utf16("лошадка");
  EXPECT_EQ(L"\x043B\x043E\x0448\x0430\x0434\x043A\x0430", u.str());
  EXPECT_EQ(7, u.size());
  // U+10437 { DESERET SMALL LETTER YEE }
  EXPECT_EQ(L"\xD801\xDC37", fmt::detail::utf8_to_utf16("𐐷").str());
  EXPECT_THROW_MSG(fmt::detail::utf8_to_utf16("\xc3\x28"), std::runtime_error,
                   "invalid utf8");
  EXPECT_THROW_MSG(fmt::detail::utf8_to_utf16(fmt::string_view("л", 1)),
                   std::runtime_error, "invalid utf8");
  EXPECT_EQ(L"123456", fmt::detail::utf8_to_utf16("123456").str());
}

TEST(util_test, utf8_to_utf16_empty_string) {
  auto s = std::string();
  auto u = fmt::detail::utf8_to_utf16(s.c_str());
  EXPECT_EQ(L"", u.str());
  EXPECT_EQ(s.size(), u.size());
}

TEST(util_test, allocator_ref) {
  using test_allocator_ref = allocator_ref<mock_allocator<int>>;
  auto check_forwarding = [](mock_allocator<int>& alloc,
                             test_allocator_ref& ref) {
    int mem;
    // Check if value_type is properly defined.
    allocator_ref<mock_allocator<int>>::value_type* ptr = &mem;
    // Check forwarding.
    EXPECT_CALL(alloc, allocate(42)).WillOnce(Return(ptr));
    ref.allocate(42);
    EXPECT_CALL(alloc, deallocate(ptr, 42));
    ref.deallocate(ptr, 42);
  };

  StrictMock<mock_allocator<int>> alloc;
  auto ref = test_allocator_ref(&alloc);
  // Check if allocator_ref forwards to the underlying allocator.
  check_forwarding(alloc, ref);
  test_allocator_ref ref2(ref);
  check_forwarding(alloc, ref2);
  test_allocator_ref ref3;
  EXPECT_EQ(nullptr, ref3.get());
  ref3 = ref;
  check_forwarding(alloc, ref3);
}

TEST(util_test, format_system_error) {
  fmt::memory_buffer message;
  fmt::format_system_error(message, EDOM, "test");
  auto ec = std::error_code(EDOM, std::generic_category());
  EXPECT_EQ(to_string(message), std::system_error(ec, "test").what());
  message = fmt::memory_buffer();

  // Check if std::allocator throws on allocating max size_t / 2 chars.
  size_t max_size = max_value<size_t>() / 2;
  bool throws_on_alloc = false;
  try {
    auto alloc = std::allocator<char>();
    alloc.deallocate(alloc.allocate(max_size), max_size);
  } catch (const std::bad_alloc&) {
    throws_on_alloc = true;
  }
  if (!throws_on_alloc) {
    fmt::print(stderr, "warning: std::allocator allocates {} chars\n",
               max_size);
    return;
  }
}

TEST(util_test, system_error) {
  auto test_error = fmt::system_error(EDOM, "test");
  auto ec = std::error_code(EDOM, std::generic_category());
  EXPECT_STREQ(test_error.what(), std::system_error(ec, "test").what());
  EXPECT_EQ(test_error.code(), ec);

  auto error = std::system_error(std::error_code());
  try {
    throw fmt::system_error(EDOM, "test {}", "error");
  } catch (const std::system_error& e) {
    error = e;
  }
  fmt::memory_buffer message;
  fmt::format_system_error(message, EDOM, "test error");
  EXPECT_EQ(error.what(), to_string(message));
  EXPECT_EQ(error.code(), std::error_code(EDOM, std::generic_category()));
}

TEST(util_test, report_system_error) {
  fmt::memory_buffer out;
  fmt::format_system_error(out, EDOM, "test error");
  out.push_back('\n');
  EXPECT_WRITE(stderr, fmt::report_system_error(EDOM, "test error"),
               to_string(out));
}

TEST(memory_buffer_test, ctor) {
  basic_memory_buffer<char, 123> buffer;
  EXPECT_EQ(static_cast<size_t>(0), buffer.size());
  EXPECT_EQ(123u, buffer.capacity());
}

using std_allocator = allocator_ref<std::allocator<char>>;

TEST(memory_buffer_test, move_ctor_inline_buffer) {
  auto check_move_buffer =
      [](const char* str, basic_memory_buffer<char, 5, std_allocator>& buffer) {
        std::allocator<char>* alloc = buffer.get_allocator().get();
        basic_memory_buffer<char, 5, std_allocator> buffer2(std::move(buffer));
        // Move shouldn't destroy the inline content of the first buffer.
        EXPECT_EQ(str, std::string(&buffer[0], buffer.size()));
        EXPECT_EQ(str, std::string(&buffer2[0], buffer2.size()));
        EXPECT_EQ(5u, buffer2.capacity());
        // Move should transfer allocator.
        EXPECT_EQ(nullptr, buffer.get_allocator().get());
        EXPECT_EQ(alloc, buffer2.get_allocator().get());
      };

  auto alloc = std::allocator<char>();
  basic_memory_buffer<char, 5, std_allocator> buffer((std_allocator(&alloc)));
  const char test[] = "test";
  buffer.append(string_view(test, 4));
  check_move_buffer("test", buffer);
  // Adding one more character fills the inline buffer, but doesn't cause
  // dynamic allocation.
  buffer.push_back('a');
  check_move_buffer("testa", buffer);
}

TEST(memory_buffer_test, move_ctor_dynamic_buffer) {
  auto alloc = std::allocator<char>();
  basic_memory_buffer<char, 4, std_allocator> buffer((std_allocator(&alloc)));
  const char test[] = "test";
  buffer.append(test, test + 4);
  const char* inline_buffer_ptr = &buffer[0];
  // Adding one more character causes the content to move from the inline to
  // a dynamically allocated buffer.
  buffer.push_back('a');
  basic_memory_buffer<char, 4, std_allocator> buffer2(std::move(buffer));
  // Move should rip the guts of the first buffer.
  EXPECT_EQ(&buffer[0], inline_buffer_ptr);
  EXPECT_EQ(buffer.size(), 0);
  EXPECT_EQ(std::string(&buffer2[0], buffer2.size()), "testa");
  EXPECT_GT(buffer2.capacity(), 4u);
}

void check_move_assign_buffer(const char* str,
                              basic_memory_buffer<char, 5>& buffer) {
  basic_memory_buffer<char, 5> buffer2;
  buffer2 = std::move(buffer);
  // Move shouldn't destroy the inline content of the first buffer.
  EXPECT_EQ(str, std::string(&buffer[0], buffer.size()));
  EXPECT_EQ(str, std::string(&buffer2[0], buffer2.size()));
  EXPECT_EQ(5u, buffer2.capacity());
}

TEST(memory_buffer_test, move_assignment) {
  basic_memory_buffer<char, 5> buffer;
  const char test[] = "test";
  buffer.append(test, test + 4);
  check_move_assign_buffer("test", buffer);
  // Adding one more character fills the inline buffer, but doesn't cause
  // dynamic allocation.
  buffer.push_back('a');
  check_move_assign_buffer("testa", buffer);
  const char* inline_buffer_ptr = &buffer[0];
  // Adding one more character causes the content to move from the inline to
  // a dynamically allocated buffer.
  buffer.push_back('b');
  basic_memory_buffer<char, 5> buffer2;
  buffer2 = std::move(buffer);
  // Move should rip the guts of the first buffer.
  EXPECT_EQ(inline_buffer_ptr, &buffer[0]);
  EXPECT_EQ("testab", std::string(&buffer2[0], buffer2.size()));
  EXPECT_GT(buffer2.capacity(), 5u);
}

TEST(memory_buffer_test, grow) {
  using allocator = allocator_ref<mock_allocator<int>>;
  mock_allocator<int> alloc;
  basic_memory_buffer<int, 10, allocator> buffer((allocator(&alloc)));
  buffer.resize(7);
  using fmt::detail::to_unsigned;
  for (int i = 0; i < 7; ++i) buffer[to_unsigned(i)] = i * i;
  EXPECT_EQ(10u, buffer.capacity());
  int mem[20];
  mem[7] = 0xdead;
  EXPECT_CALL(alloc, allocate(20)).WillOnce(Return(mem));
  buffer.try_reserve(20);
  EXPECT_EQ(20u, buffer.capacity());
  // Check if size elements have been copied
  for (int i = 0; i < 7; ++i) EXPECT_EQ(i * i, buffer[to_unsigned(i)]);
  // and no more than that.
  EXPECT_EQ(0xdead, buffer[7]);
  EXPECT_CALL(alloc, deallocate(mem, 20));
}

TEST(memory_buffer_test, allocator) {
  using test_allocator = allocator_ref<mock_allocator<char>>;
  basic_memory_buffer<char, 10, test_allocator> buffer;
  EXPECT_EQ(nullptr, buffer.get_allocator().get());
  StrictMock<mock_allocator<char>> alloc;
  char mem;
  {
    basic_memory_buffer<char, 10, test_allocator> buffer2(
        (test_allocator(&alloc)));
    EXPECT_EQ(&alloc, buffer2.get_allocator().get());
    size_t size = 2 * fmt::inline_buffer_size;
    EXPECT_CALL(alloc, allocate(size)).WillOnce(Return(&mem));
    buffer2.reserve(size);
    EXPECT_CALL(alloc, deallocate(&mem, size));
  }
}

TEST(memory_buffer_test, exception_in_deallocate) {
  using test_allocator = allocator_ref<mock_allocator<char>>;
  StrictMock<mock_allocator<char>> alloc;
  basic_memory_buffer<char, 10, test_allocator> buffer(
      (test_allocator(&alloc)));
  size_t size = 2 * fmt::inline_buffer_size;
  auto mem = std::vector<char>(size);
  {
    EXPECT_CALL(alloc, allocate(size)).WillOnce(Return(&mem[0]));
    buffer.resize(size);
    std::fill(&buffer[0], &buffer[0] + size, 'x');
  }
  auto mem2 = std::vector<char>(2 * size);
  {
    EXPECT_CALL(alloc, allocate(2 * size)).WillOnce(Return(&mem2[0]));
    auto e = std::exception();
    EXPECT_CALL(alloc, deallocate(&mem[0], size)).WillOnce(testing::Throw(e));
    EXPECT_THROW(buffer.reserve(2 * size), std::exception);
    EXPECT_EQ(&mem2[0], &buffer[0]);
    // Check that the data has been copied.
    for (size_t i = 0; i < size; ++i) EXPECT_EQ('x', buffer[i]);
  }
  EXPECT_CALL(alloc, deallocate(&mem2[0], 2 * size));
}

template <typename Allocator, size_t MaxSize>
class max_size_allocator : public Allocator {
 public:
  using typename Allocator::value_type;
  size_t max_size() const noexcept { return MaxSize; }
  value_type* allocate(size_t n) {
    if (n > max_size()) {
      throw std::length_error("size > max_size");
    }
    return std::allocator_traits<Allocator>::allocate(
        *static_cast<Allocator*>(this), n);
  }
  void deallocate(value_type* p, size_t n) {
    std::allocator_traits<Allocator>::deallocate(*static_cast<Allocator*>(this),
                                                 p, n);
  }
};

TEST(memory_buffer_test, max_size_allocator) {
  // 160 = 128 + 32
  using test_allocator = max_size_allocator<std::allocator<char>, 160>;
  basic_memory_buffer<char, 10, test_allocator> buffer;
  buffer.resize(128);
  // new_capacity = 128 + 128/2 = 192 > 160
  buffer.resize(160);  // Shouldn't throw.
}

TEST(memory_buffer_test, max_size_allocator_overflow) {
  using test_allocator = max_size_allocator<std::allocator<char>, 160>;
  basic_memory_buffer<char, 10, test_allocator> buffer;
  EXPECT_THROW(buffer.resize(161), std::exception);
}

TEST(memory_buffer_test, back_insert_iterator) {
  fmt::memory_buffer buf;
  using iterator = decltype(std::back_inserter(buf));
  EXPECT_TRUE(fmt::detail::is_back_insert_iterator<iterator>::value);
}

TEST(format_test, digits2_alignment) {
  auto p =
      fmt::detail::bit_cast<fmt::detail::uintptr_t>(fmt::detail::digits2(0));
  EXPECT_EQ(p % 2, 0);
}

TEST(format_test, exception_from_lib) {
  EXPECT_THROW_MSG(fmt::report_error("test"), format_error, "test");
}

TEST(format_test, escape) {
  EXPECT_EQ(fmt::format("{{"), "{");
  EXPECT_EQ(fmt::format("before {{"), "before {");
  EXPECT_EQ(fmt::format("{{ after"), "{ after");
  EXPECT_EQ(fmt::format("before {{ after"), "before { after");

  EXPECT_EQ(fmt::format("}}"), "}");
  EXPECT_EQ(fmt::format("before }}"), "before }");
  EXPECT_EQ(fmt::format("}} after"), "} after");
  EXPECT_EQ(fmt::format("before }} after"), "before } after");

  EXPECT_EQ(fmt::format("{{}}"), "{}");
  EXPECT_EQ(fmt::format("{{{0}}}", 42), "{42}");
}

TEST(format_test, unmatched_braces) {
  EXPECT_THROW_MSG((void)fmt::format(runtime("{")), format_error,
                   "invalid format string");
  EXPECT_THROW_MSG((void)fmt::format(runtime("}")), format_error,
                   "unmatched '}' in format string");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0{}")), format_error,
                   "invalid format string");
}

TEST(format_test, no_args) { EXPECT_EQ(fmt::format("test"), "test"); }

TEST(format_test, args_in_different_positions) {
  EXPECT_EQ(fmt::format("{0}", 42), "42");
  EXPECT_EQ(fmt::format("before {0}", 42), "before 42");
  EXPECT_EQ(fmt::format("{0} after", 42), "42 after");
  EXPECT_EQ(fmt::format("before {0} after", 42), "before 42 after");
  EXPECT_EQ(fmt::format("{0} = {1}", "answer", 42), "answer = 42");
  EXPECT_EQ(fmt::format("{1} is the {0}", "answer", 42), "42 is the answer");
  EXPECT_EQ(fmt::format("{0}{1}{0}", "abra", "cad"), "abracadabra");
}

TEST(format_test, arg_errors) {
  EXPECT_THROW_MSG((void)fmt::format(runtime("{")), format_error,
                   "invalid format string");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{?}")), format_error,
                   "invalid format string");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0")), format_error,
                   "invalid format string");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0}")), format_error,
                   "argument not found");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{00}"), 42), format_error,
                   "invalid format string");

  auto int_max = std::to_string(INT_MAX);
  EXPECT_THROW_MSG((void)fmt::format(runtime("{" + int_max)), format_error,
                   "invalid format string");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{" + int_max + "}")),
                   format_error, "argument not found");

  auto int_maxer = std::to_string(INT_MAX + 1u);
  EXPECT_THROW_MSG((void)fmt::format(runtime("{" + int_maxer)), format_error,
                   "invalid format string");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{" + int_maxer + "}")),
                   format_error, "argument not found");
}

TEST(format_test, display_width_precision) {
  EXPECT_EQ(fmt::format("{:.5}", "🐱🐱🐱"), "🐱🐱");
}

template <int N> struct test_format {
  template <typename... T>
  static auto format(fmt::string_view fmt, const T&... args) -> std::string {
    return test_format<N - 1>::format(fmt, N - 1, args...);
  }
};

template <> struct test_format<0> {
  template <typename... T>
  static auto format(fmt::string_view fmt, const T&... args) -> std::string {
    return fmt::format(runtime(fmt), args...);
  }
};

TEST(format_test, many_args) {
  EXPECT_EQ("19", test_format<20>::format("{19}"));
  EXPECT_THROW_MSG(test_format<20>::format("{20}"), format_error,
                   "argument not found");
  EXPECT_THROW_MSG(test_format<21>::format("{21}"), format_error,
                   "argument not found");
  using fmt::detail::max_packed_args;
  std::string format_str = fmt::format("{{{}}}", max_packed_args + 1);
  EXPECT_THROW_MSG(test_format<max_packed_args>::format(format_str),
                   format_error, "argument not found");
}

TEST(format_test, named_arg) {
  EXPECT_EQ("1/a/A", fmt::format("{_1}/{a_}/{A_}", fmt::arg("a_", 'a'),
                                 fmt::arg("A_", "A"), fmt::arg("_1", 1)));
  EXPECT_EQ(fmt::format("{0:{width}}", -42, fmt::arg("width", 4)), " -42");
  EXPECT_EQ(fmt::format("{value:{width}}", fmt::arg("value", -42),
                        fmt::arg("width", 4)),
            " -42");
  EXPECT_EQ("st",
            fmt::format("{0:.{precision}}", "str", fmt::arg("precision", 2)));
  EXPECT_EQ(fmt::format("{} {two}", 1, fmt::arg("two", 2)), "1 2");
  EXPECT_EQ("42",
            fmt::format("{c}", fmt::arg("a", 0), fmt::arg("b", 0),
                        fmt::arg("c", 42), fmt::arg("d", 0), fmt::arg("e", 0),
                        fmt::arg("f", 0), fmt::arg("g", 0), fmt::arg("h", 0),
                        fmt::arg("i", 0), fmt::arg("j", 0), fmt::arg("k", 0),
                        fmt::arg("l", 0), fmt::arg("m", 0), fmt::arg("n", 0),
                        fmt::arg("o", 0), fmt::arg("p", 0)));
  EXPECT_THROW_MSG((void)fmt::format(runtime("{a}")), format_error,
                   "argument not found");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{a}"), 42), format_error,
                   "argument not found");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{a} {}"), fmt::arg("a", 2), 42),
                   format_error,
                   "cannot switch from manual to automatic argument indexing");
  EXPECT_THROW_MSG(
      (void)fmt::format("{a}", fmt::arg("a", 1), fmt::arg("a", 10)),
      format_error, "duplicate named arg");
}

TEST(format_test, auto_arg_index) {
  EXPECT_EQ(fmt::format("{}{}{}", 'a', 'b', 'c'), "abc");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0}{}"), 'a', 'b'), format_error,
                   "cannot switch from manual to automatic argument indexing");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{}{0}"), 'a', 'b'), format_error,
                   "cannot switch from automatic to manual argument indexing");
  EXPECT_EQ(fmt::format("{:.{}}", 1.2345, 2), "1.2");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0}:.{}"), 1.2345, 2),
                   format_error,
                   "cannot switch from manual to automatic argument indexing");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:.{0}}"), 1.2345, 2),
                   format_error,
                   "cannot switch from automatic to manual argument indexing");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{}")), format_error,
                   "argument not found");
}

TEST(format_test, empty_specs) { EXPECT_EQ(fmt::format("{0:}", 42), "42"); }

TEST(format_test, left_align) {
  EXPECT_EQ(fmt::format("{0:<4}", 42), "42  ");
  EXPECT_EQ(fmt::format("{0:<4o}", 042), "42  ");
  EXPECT_EQ(fmt::format("{0:<4x}", 0x42), "42  ");
  EXPECT_EQ(fmt::format("{0:<5}", -42), "-42  ");
  EXPECT_EQ(fmt::format("{0:<5}", 42u), "42   ");
  EXPECT_EQ(fmt::format("{0:<5}", -42l), "-42  ");
  EXPECT_EQ(fmt::format("{0:<5}", 42ul), "42   ");
  EXPECT_EQ(fmt::format("{0:<5}", -42ll), "-42  ");
  EXPECT_EQ(fmt::format("{0:<5}", 42ull), "42   ");
  EXPECT_EQ(fmt::format("{0:<5}", -42.0), "-42  ");
  EXPECT_EQ(fmt::format("{0:<5}", -42.0l), "-42  ");
  EXPECT_EQ(fmt::format("{0:<5}", 'c'), "c    ");
  EXPECT_EQ(fmt::format("{0:<5}", "abc"), "abc  ");
  EXPECT_EQ(fmt::format("{0:<8}", reinterpret_cast<void*>(0xface)), "0xface  ");
}

TEST(format_test, right_align) {
  EXPECT_EQ(fmt::format("{0:>4}", 42), "  42");
  EXPECT_EQ(fmt::format("{0:>4o}", 042), "  42");
  EXPECT_EQ(fmt::format("{0:>4x}", 0x42), "  42");
  EXPECT_EQ(fmt::format("{0:>5}", -42), "  -42");
  EXPECT_EQ(fmt::format("{0:>5}", 42u), "   42");
  EXPECT_EQ(fmt::format("{0:>5}", -42l), "  -42");
  EXPECT_EQ(fmt::format("{0:>5}", 42ul), "   42");
  EXPECT_EQ(fmt::format("{0:>5}", -42ll), "  -42");
  EXPECT_EQ(fmt::format("{0:>5}", 42ull), "   42");
  EXPECT_EQ(fmt::format("{0:>5}", -42.0), "  -42");
  EXPECT_EQ(fmt::format("{0:>5}", -42.0l), "  -42");
  EXPECT_EQ(fmt::format("{0:>5}", 'c'), "    c");
  EXPECT_EQ(fmt::format("{0:>5}", "abc"), "  abc");
  EXPECT_EQ(fmt::format("{0:>8}", reinterpret_cast<void*>(0xface)), "  0xface");
}

TEST(format_test, center_align) {
  EXPECT_EQ(fmt::format("{0:^5}", 42), " 42  ");
  EXPECT_EQ(fmt::format("{0:^5o}", 042), " 42  ");
  EXPECT_EQ(fmt::format("{0:^5x}", 0x42), " 42  ");
  EXPECT_EQ(fmt::format("{0:^5}", -42), " -42 ");
  EXPECT_EQ(fmt::format("{0:^5}", 42u), " 42  ");
  EXPECT_EQ(fmt::format("{0:^5}", -42l), " -42 ");
  EXPECT_EQ(fmt::format("{0:^5}", 42ul), " 42  ");
  EXPECT_EQ(fmt::format("{0:^5}", -42ll), " -42 ");
  EXPECT_EQ(fmt::format("{0:^5}", 42ull), " 42  ");
  EXPECT_EQ(fmt::format("{0:^5}", -42.0), " -42 ");
  EXPECT_EQ(fmt::format("{0:^5}", -42.0l), " -42 ");
  EXPECT_EQ(fmt::format("{0:^5}", 'c'), "  c  ");
  EXPECT_EQ(fmt::format("{0:^6}", "abc"), " abc  ");
  EXPECT_EQ(fmt::format("{0:^8}", reinterpret_cast<void*>(0xface)), " 0xface ");
}

TEST(format_test, fill) {
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:{<5}"), 'c'), format_error,
                   "invalid fill character '{'");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:{<5}}"), 'c'), format_error,
                   "invalid fill character '{'");
  EXPECT_EQ(fmt::format("{0:*>4}", 42), "**42");
  EXPECT_EQ(fmt::format("{0:*>5}", -42), "**-42");
  EXPECT_EQ(fmt::format("{0:*>5}", 42u), "***42");
  EXPECT_EQ(fmt::format("{0:*>5}", -42l), "**-42");
  EXPECT_EQ(fmt::format("{0:*>5}", 42ul), "***42");
  EXPECT_EQ(fmt::format("{0:*>5}", -42ll), "**-42");
  EXPECT_EQ(fmt::format("{0:*>5}", 42ull), "***42");
  EXPECT_EQ(fmt::format("{0:*>5}", -42.0), "**-42");
  EXPECT_EQ(fmt::format("{0:*>5}", -42.0l), "**-42");
  EXPECT_EQ(fmt::format("{0:*<5}", 'c'), "c****");
  EXPECT_EQ(fmt::format("{0:*<5}", "abc"), "abc**");
  EXPECT_EQ("**0xface",
            fmt::format("{0:*>8}", reinterpret_cast<void*>(0xface)));
  EXPECT_EQ(fmt::format("{:}=", "foo"), "foo=");
  EXPECT_EQ(std::string("\0\0\0*", 4),
            fmt::format(string_view("{:\0>4}", 6), '*'));
  EXPECT_EQ(fmt::format("{0:ж>4}", 42), "жж42");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:\x80\x80\x80\x80\x80>}"), 0),
                   format_error, "invalid format specifier");
}

TEST(format_test, plus_sign) {
  EXPECT_EQ(fmt::format("{0:+}", 42), "+42");
  EXPECT_EQ(fmt::format("{0:+}", -42), "-42");
  EXPECT_EQ(fmt::format("{0:+}", 42), "+42");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:+}"), 42u), format_error,
                   "invalid format specifier");
  EXPECT_EQ(fmt::format("{0:+}", 42l), "+42");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:+}"), 42ul), format_error,
                   "invalid format specifier");
  EXPECT_EQ(fmt::format("{0:+}", 42ll), "+42");
#if FMT_USE_INT128
  EXPECT_EQ(fmt::format("{0:+}", __int128_t(42)), "+42");
#endif
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:+}"), 42ull), format_error,
                   "invalid format specifier");
  EXPECT_EQ(fmt::format("{0:+}", 42.0), "+42");
  EXPECT_EQ(fmt::format("{0:+}", 42.0l), "+42");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:+}"), 'c'), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:+}"), "abc"), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG(
      (void)fmt::format(runtime("{0:+}"), reinterpret_cast<void*>(0x42)),
      format_error, "invalid format specifier");
}

TEST(format_test, minus_sign) {
  EXPECT_EQ(fmt::format("{0:-}", 42), "42");
  EXPECT_EQ(fmt::format("{0:-}", -42), "-42");
  EXPECT_EQ(fmt::format("{0:-}", 42), "42");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:-}"), 42u), format_error,
                   "invalid format specifier");
  EXPECT_EQ(fmt::format("{0:-}", 42l), "42");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:-}"), 42ul), format_error,
                   "invalid format specifier");
  EXPECT_EQ(fmt::format("{0:-}", 42ll), "42");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:-}"), 42ull), format_error,
                   "invalid format specifier");
  EXPECT_EQ(fmt::format("{0:-}", 42.0), "42");
  EXPECT_EQ(fmt::format("{0:-}", 42.0l), "42");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:-}"), 'c'), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:-}"), "abc"), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG(
      (void)fmt::format(runtime("{0:-}"), reinterpret_cast<void*>(0x42)),
      format_error, "invalid format specifier");
}

TEST(format_test, space_sign) {
  EXPECT_EQ(fmt::format("{0: }", 42), " 42");
  EXPECT_EQ(fmt::format("{0: }", -42), "-42");
  EXPECT_EQ(fmt::format("{0: }", 42), " 42");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0: }"), 42u), format_error,
                   "invalid format specifier");
  EXPECT_EQ(fmt::format("{0: }", 42l), " 42");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0: }"), 42ul), format_error,
                   "invalid format specifier");
  EXPECT_EQ(fmt::format("{0: }", 42ll), " 42");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0: }"), 42ull), format_error,
                   "invalid format specifier");
  EXPECT_EQ(fmt::format("{0: }", 42.0), " 42");
  EXPECT_EQ(fmt::format("{0: }", 42.0l), " 42");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0: }"), 'c'), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0: }"), "abc"), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG(
      (void)fmt::format(runtime("{0: }"), reinterpret_cast<void*>(0x42)),
      format_error, "invalid format specifier");
}

TEST(format_test, hash_flag) {
  EXPECT_EQ(fmt::format("{0:#}", 42), "42");
  EXPECT_EQ(fmt::format("{0:#}", -42), "-42");
  EXPECT_EQ(fmt::format("{0:#b}", 42), "0b101010");
  EXPECT_EQ(fmt::format("{0:#B}", 42), "0B101010");
  EXPECT_EQ(fmt::format("{0:#b}", -42), "-0b101010");
  EXPECT_EQ(fmt::format("{0:#x}", 0x42), "0x42");
  EXPECT_EQ(fmt::format("{0:#X}", 0x42), "0X42");
  EXPECT_EQ(fmt::format("{0:#x}", -0x42), "-0x42");
  EXPECT_EQ(fmt::format("{0:#o}", 0), "0");
  EXPECT_EQ(fmt::format("{0:#o}", 042), "042");
  EXPECT_EQ(fmt::format("{0:#o}", -042), "-042");
  EXPECT_EQ(fmt::format("{0:#}", 42u), "42");
  EXPECT_EQ(fmt::format("{0:#x}", 0x42u), "0x42");
  EXPECT_EQ(fmt::format("{0:#o}", 042u), "042");

  EXPECT_EQ(fmt::format("{0:#}", -42l), "-42");
  EXPECT_EQ(fmt::format("{0:#x}", 0x42l), "0x42");
  EXPECT_EQ(fmt::format("{0:#x}", -0x42l), "-0x42");
  EXPECT_EQ(fmt::format("{0:#o}", 042l), "042");
  EXPECT_EQ(fmt::format("{0:#o}", -042l), "-042");
  EXPECT_EQ(fmt::format("{0:#}", 42ul), "42");
  EXPECT_EQ(fmt::format("{0:#x}", 0x42ul), "0x42");
  EXPECT_EQ(fmt::format("{0:#o}", 042ul), "042");

  EXPECT_EQ(fmt::format("{0:#}", -42ll), "-42");
  EXPECT_EQ(fmt::format("{0:#x}", 0x42ll), "0x42");
  EXPECT_EQ(fmt::format("{0:#x}", -0x42ll), "-0x42");
  EXPECT_EQ(fmt::format("{0:#o}", 042ll), "042");
  EXPECT_EQ(fmt::format("{0:#o}", -042ll), "-042");
  EXPECT_EQ(fmt::format("{0:#}", 42ull), "42");
  EXPECT_EQ(fmt::format("{0:#x}", 0x42ull), "0x42");
  EXPECT_EQ(fmt::format("{0:#o}", 042ull), "042");

  EXPECT_EQ(fmt::format("{0:#}", -42.0), "-42.");
  EXPECT_EQ(fmt::format("{0:#}", -42.0l), "-42.");
  EXPECT_EQ(fmt::format("{:#.0e}", 42.0), "4.e+01");
  EXPECT_EQ(fmt::format("{:#.0f}", 0.01), "0.");
  EXPECT_EQ(fmt::format("{:#.2g}", 0.5), "0.50");
  EXPECT_EQ(fmt::format("{:#.0f}", 0.5), "0.");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:#"), 'c'), format_error,
                   "invalid format specifier for char");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:#}"), 'c'), format_error,
                   "invalid format specifier for char");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:#}"), "abc"), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG(
      (void)fmt::format(runtime("{0:#}"), reinterpret_cast<void*>(0x42)),
      format_error, "invalid format specifier");
}

TEST(format_test, zero_flag) {
  EXPECT_EQ(fmt::format("{0:0}", 42), "42");
  EXPECT_EQ(fmt::format("{0:05}", -42), "-0042");
  EXPECT_EQ(fmt::format("{0:05}", 42u), "00042");
  EXPECT_EQ(fmt::format("{0:05}", -42l), "-0042");
  EXPECT_EQ(fmt::format("{0:05}", 42ul), "00042");
  EXPECT_EQ(fmt::format("{0:05}", -42ll), "-0042");
  EXPECT_EQ(fmt::format("{0:05}", 42ull), "00042");
  EXPECT_EQ(fmt::format("{0:07}", -42.0), "-000042");
  EXPECT_EQ(fmt::format("{0:07}", -42.0l), "-000042");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:0"), 'c'), format_error,
                   "invalid format specifier for char");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:05}"), 'c'), format_error,
                   "invalid format specifier for char");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:05}"), "abc"), format_error,
                   "format specifier requires numeric argument");
  EXPECT_THROW_MSG(
      (void)fmt::format(runtime("{0:05}"), reinterpret_cast<void*>(0x42)),
      format_error, "format specifier requires numeric argument");
}

TEST(format_test, zero_flag_and_align) {
  // If the 0 character and an align option both appear, the 0 character is
  // ignored.
  EXPECT_EQ(fmt::format("{:<05}", 42), "42   ");
  EXPECT_EQ(fmt::format("{:<05}", -42), "-42  ");
  EXPECT_EQ(fmt::format("{:^05}", 42), " 42  ");
  EXPECT_EQ(fmt::format("{:^05}", -42), " -42 ");
  EXPECT_EQ(fmt::format("{:>05}", 42), "   42");
  EXPECT_EQ(fmt::format("{:>05}", -42), "  -42");
}

TEST(format_test, width) {
  auto int_maxer = std::to_string(INT_MAX + 1u);
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:" + int_maxer), 0),
                   format_error, "number is too big");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:" + int_maxer + "}"), 0),
                   format_error, "number is too big");

  EXPECT_EQ(fmt::format("{:4}", -42), " -42");
  EXPECT_EQ(fmt::format("{:5}", 42u), "   42");
  EXPECT_EQ(fmt::format("{:6}", -42l), "   -42");
  EXPECT_EQ(fmt::format("{:7}", 42ul), "     42");
  EXPECT_EQ(fmt::format("{:6}", -42ll), "   -42");
  EXPECT_EQ(fmt::format("{:7}", 42ull), "     42");
  EXPECT_EQ(fmt::format("{:8}", -1.23), "   -1.23");
  EXPECT_EQ(fmt::format("{:9}", -1.23l), "    -1.23");
  EXPECT_EQ(fmt::format("{:10}", reinterpret_cast<void*>(0xcafe)),
            "    0xcafe");
  EXPECT_EQ(fmt::format("{:11}", 'x'), "x          ");
  EXPECT_EQ(fmt::format("{:12}", "str"), "str         ");
  EXPECT_EQ(fmt::format("{:*^5}", "🤡"), "*🤡**");
  EXPECT_EQ(fmt::format("{:*^6}", "🤡"), "**🤡**");
  EXPECT_EQ(fmt::format("{:*^8}", "你好"), "**你好**");
  EXPECT_EQ(fmt::format("{:#6}", 42.0), "   42.");
  EXPECT_EQ(fmt::format("{:6c}", static_cast<int>('x')), "x     ");
  EXPECT_EQ(fmt::format("{:>06.0f}", 0.00884311), "     0");
}

TEST(format_test, debug_presentation) {
  EXPECT_EQ(fmt::format("{:?}", ""), R"("")");

  EXPECT_EQ(fmt::format("{:*<5.0?}", "\n"), R"(*****)");
  EXPECT_EQ(fmt::format("{:*<5.1?}", "\n"), R"("****)");
  EXPECT_EQ(fmt::format("{:*<5.2?}", "\n"), R"("\***)");
  EXPECT_EQ(fmt::format("{:*<5.3?}", "\n"), R"("\n**)");
  EXPECT_EQ(fmt::format("{:*<5.4?}", "\n"), R"("\n"*)");

  EXPECT_EQ(fmt::format("{:*<5.1?}", "Σ"), R"("****)");
  EXPECT_EQ(fmt::format("{:*<5.2?}", "Σ"), R"("Σ***)");
  EXPECT_EQ(fmt::format("{:*<5.3?}", "Σ"), R"("Σ"**)");

  EXPECT_EQ(fmt::format("{:*<5.1?}", "笑"), R"("****)");
  EXPECT_EQ(fmt::format("{:*<5.2?}", "笑"), R"("****)");
  EXPECT_EQ(fmt::format("{:*<5.3?}", "笑"), R"("笑**)");
  EXPECT_EQ(fmt::format("{:*<5.4?}", "笑"), R"("笑"*)");

  EXPECT_EQ(fmt::format("{:*<8?}", "туда"), R"("туда"**)");
  EXPECT_EQ(fmt::format("{:*>8?}", "сюда"), R"(**"сюда")");
  EXPECT_EQ(fmt::format("{:*^8?}", "中心"), R"(*"中心"*)");

  EXPECT_EQ(fmt::format("{:*^14?}", "A\t👈🤯ы猫"), R"(*"A\t👈🤯ы猫"*)");
}

auto bad_dynamic_spec_msg = FMT_BUILTIN_TYPES
                                ? "width/precision is out of range"
                                : "width/precision is not integer";

TEST(format_test, runtime_width) {
  auto int_maxer = std::to_string(INT_MAX + 1u);
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:{" + int_maxer), 0),
                   format_error, "invalid format string");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:{" + int_maxer + "}"), 0),
                   format_error, "argument not found");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:{" + int_maxer + "}}"), 0),
                   format_error, "argument not found");

  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:{"), 0), format_error,
                   "invalid format string");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:{}"), 0), format_error,
                   "cannot switch from manual to automatic argument indexing");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:{?}}"), 0), format_error,
                   "invalid format string");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:{1}}"), 0), format_error,
                   "argument not found");

  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:{0:}}"), 0), format_error,
                   "invalid format string");

  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:{1}}"), 0, -1), format_error,
                   "width/precision is out of range");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:{1}}"), 0, (INT_MAX + 1u)),
                   format_error, bad_dynamic_spec_msg);
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:{1}}"), 0, -1l), format_error,
                   bad_dynamic_spec_msg);
  if (fmt::detail::const_check(sizeof(long) > sizeof(int))) {
    long value = INT_MAX;
    EXPECT_THROW_MSG((void)fmt::format(runtime("{0:{1}}"), 0, (value + 1)),
                     format_error, bad_dynamic_spec_msg);
  }
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:{1}}"), 0, (INT_MAX + 1ul)),
                   format_error, bad_dynamic_spec_msg);

  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:{1}}"), 0, '0'), format_error,
                   "width/precision is not integer");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:{1}}"), 0, 0.0), format_error,
                   "width/precision is not integer");

  EXPECT_EQ(fmt::format("{0:{1}}", -42, 4), " -42");
  EXPECT_EQ(fmt::format("{0:{1}}", 42u, 5), "   42");
  EXPECT_EQ(fmt::format("{0:{1}}", -42l, 6), "   -42");
  EXPECT_EQ(fmt::format("{0:{1}}", 42ul, 7), "     42");
  EXPECT_EQ(fmt::format("{0:{1}}", -42ll, 6), "   -42");
  EXPECT_EQ(fmt::format("{0:{1}}", 42ull, 7), "     42");
  EXPECT_EQ(fmt::format("{0:{1}}", -1.23, 8), "   -1.23");
  EXPECT_EQ(fmt::format("{0:{1}}", -1.23l, 9), "    -1.23");
  EXPECT_EQ("    0xcafe",
            fmt::format("{0:{1}}", reinterpret_cast<void*>(0xcafe), 10));
  EXPECT_EQ(fmt::format("{0:{1}}", 'x', 11), "x          ");
  EXPECT_EQ(fmt::format("{0:{1}}", "str", 12), "str         ");
  EXPECT_EQ(fmt::format("{:{}}", 42, short(4)), "  42");
}

TEST(format_test, exponent_range) {
  for (int e = -1074; e <= 1023; ++e) (void)fmt::format("{}", std::ldexp(1, e));
}

TEST(format_test, precision) {
  char format_str[buffer_size];
  safe_sprintf(format_str, "{0:.%u", UINT_MAX);
  increment(format_str + 4);
  EXPECT_THROW_MSG((void)fmt::format(runtime(format_str), 0.0), format_error,
                   "number is too big");
  size_t size = std::strlen(format_str);
  format_str[size] = '}';
  format_str[size + 1] = 0;
  EXPECT_THROW_MSG((void)fmt::format(runtime(format_str), 0.0), format_error,
                   "number is too big");

  safe_sprintf(format_str, "{0:.%u", INT_MAX + 1u);
  EXPECT_THROW_MSG((void)fmt::format(runtime(format_str), 0.0), format_error,
                   "number is too big");
  safe_sprintf(format_str, "{0:.%u}", INT_MAX + 1u);
  EXPECT_THROW_MSG((void)fmt::format(runtime(format_str), 0.0), format_error,
                   "number is too big");

  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:."), 0.0), format_error,
                   "invalid precision");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.}"), 0.0), format_error,
                   "invalid format string");

  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.2"), 0), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.2}"), 42), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.2f}"), 42), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.2}"), 42u), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.2f}"), 42u), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.2}"), 42l), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.2f}"), 42l), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.2}"), 42ul), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.2f}"), 42ul), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.2}"), 42ll), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.2f}"), 42ll), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.2}"), 42ull), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.2f}"), 42ull), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:3.0}"), 'x'), format_error,
                   "invalid format specifier");
  EXPECT_EQ(fmt::format("{0:.2}", 1.2345), "1.2");
  EXPECT_EQ(fmt::format("{0:.2}", 1.2345l), "1.2");
  EXPECT_EQ(fmt::format("{:.2}", 1.234e56), "1.2e+56");
  EXPECT_EQ(fmt::format("{0:.3}", 1.1), "1.1");
  EXPECT_EQ(fmt::format("{:.0e}", 1.0L), "1e+00");
  EXPECT_EQ(fmt::format("{:9.1e}", 0.0), "  0.0e+00");
  EXPECT_EQ(fmt::format("{:.7f}", 0.0000000000000071054273576010018587L),
            "0.0000000");

  EXPECT_EQ(
      fmt::format("{:.494}", 4.9406564584124654E-324),
      "4.9406564584124654417656879286822137236505980261432476442558568250067550"
      "727020875186529983636163599237979656469544571773092665671035593979639877"
      "479601078187812630071319031140452784581716784898210368871863605699873072"
      "305000638740915356498438731247339727316961514003171538539807412623856559"
      "117102665855668676818703956031062493194527159149245532930545654440112748"
      "012970999954193198940908041656332452475714786901472678015935523861155013"
      "480352649347201937902681071074917033322268447533357208324319361e-324");
  EXPECT_EQ(
      fmt::format("{:.1074f}", 1.1125369292536e-308),
      "0.0000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000111253692925360019747947051741965785554081512200979"
      "355021686109411883779182127659725163430929750364498219730822952552570601"
      "152163505899912777129583674906301179059298598412303893909188340988729019"
      "014361467448914817838555156840459458527907308695109202499990850735085304"
      "478476991912072201449236975063640913461919914396877093174125167509869762"
      "482369631100360266123742648159508919592746619553246586039571522788247697"
      "156360766271842991667238355464496455107749716934387136380536472531224398"
      "559833794807213172371254492216255558078524900147957309382830827524104234"
      "530961756787819847850302379672357738807808384667004752163416921762619527"
      "462847642037420991432005657440259928195996762610375541867198059294212446"
      "81962777939941034720757232455434770912461317493580281734466552734375");

  std::string outputs[] = {
      "-0X1.41FE3FFE71C9E000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000P+127",
      "-0XA.0FF1FFF38E4F0000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000P+124"};
  EXPECT_THAT(outputs,
              testing::Contains(fmt::format("{:.838A}", -2.14001164E+38)));

  if (std::numeric_limits<long double>::digits == 64) {
    auto ld = (std::numeric_limits<long double>::min)();
    EXPECT_EQ(fmt::format("{:.0}", ld), "3e-4932");
    EXPECT_EQ(
        fmt::format("{:0g}", std::numeric_limits<long double>::denorm_min()),
        "3.6452e-4951");
  }

  EXPECT_EQ(fmt::format("{:#.0f}", 123.0), "123.");
  EXPECT_EQ(fmt::format("{:.02f}", 1.234), "1.23");
  EXPECT_EQ(fmt::format("{:.1g}", 0.001), "0.001");
  EXPECT_EQ(fmt::format("{}", 123456789.0f), "1.2345679e+08");
  EXPECT_EQ(fmt::format("{}", 1019666432.0f), "1.0196664e+09");
  EXPECT_EQ(fmt::format("{:.0e}", 9.5), "1e+01");
  EXPECT_EQ(fmt::format("{:.1e}", 1e-34), "1.0e-34");

  EXPECT_THROW_MSG(
      (void)fmt::format(runtime("{0:.2}"), reinterpret_cast<void*>(0xcafe)),
      format_error, "invalid format specifier");
  EXPECT_THROW_MSG(
      (void)fmt::format(runtime("{0:.2f}"), reinterpret_cast<void*>(0xcafe)),
      format_error, "invalid format specifier");
  EXPECT_THROW_MSG(
      (void)fmt::format("{:.2147483646f}", -2.2121295195081227E+304),
      format_error, "number is too big");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:.f}"), 42.0), format_error,
                   "invalid format string");

  EXPECT_EQ(fmt::format("{0:.2}", "str"), "st");
  EXPECT_EQ(fmt::format("{0:.5}", "вожыкі"), "вожык");
  EXPECT_EQ(fmt::format("{0:.6}", "123456\xad"), "123456");
}

TEST(format_test, large_precision) {
  // Iterator used to abort the actual output.
  struct throwing_iterator {
    auto operator=(char) -> throwing_iterator& {
      throw std::runtime_error("aborted");
      return *this;
    }
    auto operator*() -> throwing_iterator& { return *this; }
    auto operator++() -> throwing_iterator& { return *this; }
    auto operator++(int) -> throwing_iterator { return *this; }
  };
  auto it = throwing_iterator();

  EXPECT_THROW_MSG(fmt::format_to(it, fmt::runtime("{:#.{}}"), 1.0,
                                  fmt::detail::max_value<int>()),
                   std::runtime_error, "aborted");

  EXPECT_THROW_MSG(fmt::format_to(it, fmt::runtime("{:#.{}e}"), 1.0,
                                  fmt::detail::max_value<int>() - 1),
                   std::runtime_error, "aborted");

  EXPECT_THROW_MSG((void)fmt::format(fmt::runtime("{:.{}e}"), 42.0,
                                     fmt::detail::max_value<int>()),
                   format_error, "number is too big");
}

TEST(format_test, utf8_precision) {
  auto result = fmt::format("{:.4}", "caf\u00e9s");  // cafés
  EXPECT_EQ(result, "caf\u00e9");
}

TEST(format_test, runtime_precision) {
  char format_str[buffer_size];
  safe_sprintf(format_str, "{0:.{%u", UINT_MAX);
  increment(format_str + 5);
  EXPECT_THROW_MSG((void)fmt::format(runtime(format_str), 0.0), format_error,
                   "invalid format string");
  size_t size = std::strlen(format_str);
  format_str[size] = '}';
  format_str[size + 1] = 0;
  EXPECT_THROW_MSG((void)fmt::format(runtime(format_str), 0.0), format_error,
                   "argument not found");
  format_str[size + 1] = '}';
  format_str[size + 2] = 0;
  EXPECT_THROW_MSG((void)fmt::format(runtime(format_str), 0.0), format_error,
                   "argument not found");

  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{"), 0.0), format_error,
                   "invalid format string");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{}"), 0.0), format_error,
                   "cannot switch from manual to automatic argument indexing");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{?}}"), 0.0), format_error,
                   "invalid format string");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}"), 0, 0), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}}"), 0.0), format_error,
                   "argument not found");

  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{0:}}"), 0.0), format_error,
                   "invalid format string");

  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}}"), 0.0, -1),
                   format_error, "width/precision is out of range");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}}"), 0.0, (INT_MAX + 1u)),
                   format_error, bad_dynamic_spec_msg);
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}}"), 0.0, -1l),
                   format_error, bad_dynamic_spec_msg);
  if (fmt::detail::const_check(sizeof(long) > sizeof(int))) {
    long value = INT_MAX;
    EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}}"), 0.0, (value + 1)),
                     format_error, bad_dynamic_spec_msg);
  }
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}}"), 0.0, (INT_MAX + 1ul)),
                   format_error, bad_dynamic_spec_msg);

  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}}"), 0.0, '0'),
                   format_error, "width/precision is not integer");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}}"), 0.0, 0.0),
                   format_error, "width/precision is not integer");

  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}}"), 42, 2), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}f}"), 42, 2), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}}"), 42u, 2), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}f}"), 42u, 2),
                   format_error, "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}}"), 42l, 2), format_error,
                   "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}f}"), 42l, 2),
                   format_error, "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}}"), 42ul, 2),
                   format_error, "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}f}"), 42ul, 2),
                   format_error, "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}}"), 42ll, 2),
                   format_error, "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}f}"), 42ll, 2),
                   format_error, "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}}"), 42ull, 2),
                   format_error, "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}f}"), 42ull, 2),
                   format_error, "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:3.{1}}"), 'x', 0),
                   format_error, "invalid format specifier");
  EXPECT_EQ(fmt::format("{0:.{1}}", 1.2345, 2), "1.2");
  EXPECT_EQ(fmt::format("{1:.{0}}", 2, 1.2345l), "1.2");

  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}}"),
                                     reinterpret_cast<void*>(0xcafe), 2),
                   format_error, "invalid format specifier");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:.{1}f}"),
                                     reinterpret_cast<void*>(0xcafe), 2),
                   format_error, "invalid format specifier");

  EXPECT_EQ(fmt::format("{0:.{1}}", "str", 2), "st");
}

TEST(format_test, format_bool) {
  EXPECT_EQ(fmt::format("{}", true), "true");
  EXPECT_EQ(fmt::format("{}", false), "false");
  EXPECT_EQ(fmt::format("{:d}", true), "1");
  EXPECT_EQ(fmt::format("{:5}", true), "true ");
  EXPECT_EQ(fmt::format("{:s}", true), "true");
  EXPECT_EQ(fmt::format("{:s}", false), "false");
  EXPECT_EQ(fmt::format("{:6s}", false), "false ");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:c}"), false), format_error,
                   "invalid format specifier");
}

TEST(format_test, format_short) {
  short s = 42;
  EXPECT_EQ(fmt::format("{0:d}", s), "42");
  unsigned short us = 42;
  EXPECT_EQ(fmt::format("{0:d}", us), "42");
}

template <typename T>
void check_unknown_types(const T& value, const char* types, const char*) {
  char format_str[buffer_size];
  const char* special = ".0123456789L?}";
  for (int i = CHAR_MIN; i <= CHAR_MAX; ++i) {
    char c = static_cast<char>(i);
    if (std::strchr(types, c) || std::strchr(special, c) || !c) continue;
    safe_sprintf(format_str, "{0:10%c}", c);
    const char* message = "invalid format specifier";
    EXPECT_THROW_MSG((void)fmt::format(runtime(format_str), value),
                     format_error, message)
        << format_str << " " << message;
  }
}

TEST(format_test, format_int) {
  EXPECT_THROW_MSG((void)fmt::format(runtime("{0:v"), 42), format_error,
                   "invalid format specifier");
  check_unknown_types(42, "bBdoxXnLc", "integer");
  EXPECT_EQ(fmt::format("{:c}", static_cast<int>('x')), "x");
}

TEST(format_test, format_bin) {
  EXPECT_EQ(fmt::format("{0:b}", 0), "0");
  EXPECT_EQ(fmt::format("{0:b}", 42), "101010");
  EXPECT_EQ(fmt::format("{0:b}", 42u), "101010");
  EXPECT_EQ(fmt::format("{0:b}", -42), "-101010");
  EXPECT_EQ(fmt::format("{0:b}", 12345), "11000000111001");
  EXPECT_EQ(fmt::format("{0:b}", 0x12345678), "10010001101000101011001111000");
  EXPECT_EQ("10010000101010111100110111101111",
            fmt::format("{0:b}", 0x90ABCDEF));
  EXPECT_EQ("11111111111111111111111111111111",
            fmt::format("{0:b}", max_value<uint32_t>()));
}

#if FMT_USE_INT128
constexpr auto int128_max = static_cast<__int128_t>(
    (static_cast<__uint128_t>(1) << ((__SIZEOF_INT128__ * CHAR_BIT) - 1)) - 1);
constexpr auto int128_min = -int128_max - 1;

constexpr auto uint128_max = ~static_cast<__uint128_t>(0);
#endif

TEST(format_test, format_dec) {
  EXPECT_EQ(fmt::format("{0}", 0), "0");
  EXPECT_EQ(fmt::format("{0}", 42), "42");
  EXPECT_EQ(fmt::format("{:}>", 42), "42>");
  EXPECT_EQ(fmt::format("{0:d}", 42), "42");
  EXPECT_EQ(fmt::format("{0}", 42u), "42");
  EXPECT_EQ(fmt::format("{0}", -42), "-42");
  EXPECT_EQ(fmt::format("{0}", 12345), "12345");
  EXPECT_EQ(fmt::format("{0}", 67890), "67890");
#if FMT_USE_INT128
  EXPECT_EQ(fmt::format("{0}", static_cast<__int128_t>(0)), "0");
  EXPECT_EQ(fmt::format("{0}", static_cast<__uint128_t>(0)), "0");
  EXPECT_EQ("9223372036854775808",
            fmt::format("{0}", static_cast<__int128_t>(INT64_MAX) + 1));
  EXPECT_EQ("-9223372036854775809",
            fmt::format("{0}", static_cast<__int128_t>(INT64_MIN) - 1));
  EXPECT_EQ("18446744073709551616",
            fmt::format("{0}", static_cast<__int128_t>(UINT64_MAX) + 1));
  EXPECT_EQ("170141183460469231731687303715884105727",
            fmt::format("{0}", int128_max));
  EXPECT_EQ("-170141183460469231731687303715884105728",
            fmt::format("{0}", int128_min));
  EXPECT_EQ("340282366920938463463374607431768211455",
            fmt::format("{0}", uint128_max));
#endif

  char buffer[buffer_size];
  safe_sprintf(buffer, "%d", INT_MIN);
  EXPECT_EQ(buffer, fmt::format("{0}", INT_MIN));
  safe_sprintf(buffer, "%d", INT_MAX);
  EXPECT_EQ(buffer, fmt::format("{0}", INT_MAX));
  safe_sprintf(buffer, "%u", UINT_MAX);
  EXPECT_EQ(buffer, fmt::format("{0}", UINT_MAX));
  safe_sprintf(buffer, "%ld", 0 - static_cast<unsigned long>(LONG_MIN));
  EXPECT_EQ(buffer, fmt::format("{0}", LONG_MIN));
  safe_sprintf(buffer, "%ld", LONG_MAX);
  EXPECT_EQ(buffer, fmt::format("{0}", LONG_MAX));
  safe_sprintf(buffer, "%lu", ULONG_MAX);
  EXPECT_EQ(buffer, fmt::format("{0}", ULONG_MAX));
}

TEST(format_test, format_hex) {
  EXPECT_EQ(fmt::format("{0:x}", 0), "0");
  EXPECT_EQ(fmt::format("{0:x}", 0x42), "42");
  EXPECT_EQ(fmt::format("{0:x}", 0x42u), "42");
  EXPECT_EQ(fmt::format("{0:x}", -0x42), "-42");
  EXPECT_EQ(fmt::format("{0:x}", 0x12345678), "12345678");
  EXPECT_EQ(fmt::format("{0:x}", 0x90abcdef), "90abcdef");
  EXPECT_EQ(fmt::format("{0:X}", 0x12345678), "12345678");
  EXPECT_EQ(fmt::format("{0:X}", 0x90ABCDEF), "90ABCDEF");
#if FMT_USE_INT128
  EXPECT_EQ(fmt::format("{0:x}", static_cast<__int128_t>(0)), "0");
  EXPECT_EQ(fmt::format("{0:x}", static_cast<__uint128_t>(0)), "0");
  EXPECT_EQ("8000000000000000",
            fmt::format("{0:x}", static_cast<__int128_t>(INT64_MAX) + 1));
  EXPECT_EQ("-8000000000000001",
            fmt::format("{0:x}", static_cast<__int128_t>(INT64_MIN) - 1));
  EXPECT_EQ("10000000000000000",
            fmt::format("{0:x}", static_cast<__int128_t>(UINT64_MAX) + 1));
  EXPECT_EQ("7fffffffffffffffffffffffffffffff",
            fmt::format("{0:x}", int128_max));
  EXPECT_EQ("-80000000000000000000000000000000",
            fmt::format("{0:x}", int128_min));
  EXPECT_EQ("ffffffffffffffffffffffffffffffff",
            fmt::format("{0:x}", uint128_max));
#endif

  char buffer[buffer_size];
  safe_sprintf(buffer, "-%x", 0 - static_cast<unsigned>(INT_MIN));
  EXPECT_EQ(buffer, fmt::format("{0:x}", INT_MIN));
  safe_sprintf(buffer, "%x", INT_MAX);
  EXPECT_EQ(buffer, fmt::format("{0:x}", INT_MAX));
  safe_sprintf(buffer, "%x", UINT_MAX);
  EXPECT_EQ(buffer, fmt::format("{0:x}", UINT_MAX));
  safe_sprintf(buffer, "-%lx", 0 - static_cast<unsigned long>(LONG_MIN));
  EXPECT_EQ(buffer, fmt::format("{0:x}", LONG_MIN));
  safe_sprintf(buffer, "%lx", LONG_MAX);
  EXPECT_EQ(buffer, fmt::format("{0:x}", LONG_MAX));
  safe_sprintf(buffer, "%lx", ULONG_MAX);
  EXPECT_EQ(buffer, fmt::format("{0:x}", ULONG_MAX));
}

TEST(format_test, format_oct) {
  EXPECT_EQ(fmt::format("{0:o}", 0), "0");
  EXPECT_EQ(fmt::format("{0:o}", 042), "42");
  EXPECT_EQ(fmt::format("{0:o}", 042u), "42");
  EXPECT_EQ(fmt::format("{0:o}", -042), "-42");
  EXPECT_EQ(fmt::format("{0:o}", 012345670), "12345670");
#if FMT_USE_INT128
  EXPECT_EQ(fmt::format("{0:o}", static_cast<__int128_t>(0)), "0");
  EXPECT_EQ(fmt::format("{0:o}", static_cast<__uint128_t>(0)), "0");
  EXPECT_EQ("1000000000000000000000",
            fmt::format("{0:o}", static_cast<__int128_t>(INT64_MAX) + 1));
  EXPECT_EQ("-1000000000000000000001",
            fmt::format("{0:o}", static_cast<__int128_t>(INT64_MIN) - 1));
  EXPECT_EQ("2000000000000000000000",
            fmt::format("{0:o}", static_cast<__int128_t>(UINT64_MAX) + 1));
  EXPECT_EQ("1777777777777777777777777777777777777777777",
            fmt::format("{0:o}", int128_max));
  EXPECT_EQ("-2000000000000000000000000000000000000000000",
            fmt::format("{0:o}", int128_min));
  EXPECT_EQ("3777777777777777777777777777777777777777777",
            fmt::format("{0:o}", uint128_max));
#endif

  char buffer[buffer_size];
  safe_sprintf(buffer, "-%o", 0 - static_cast<unsigned>(INT_MIN));
  EXPECT_EQ(buffer, fmt::format("{0:o}", INT_MIN));
  safe_sprintf(buffer, "%o", INT_MAX);
  EXPECT_EQ(buffer, fmt::format("{0:o}", INT_MAX));
  safe_sprintf(buffer, "%o", UINT_MAX);
  EXPECT_EQ(buffer, fmt::format("{0:o}", UINT_MAX));
  safe_sprintf(buffer, "-%lo", 0 - static_cast<unsigned long>(LONG_MIN));
  EXPECT_EQ(buffer, fmt::format("{0:o}", LONG_MIN));
  safe_sprintf(buffer, "%lo", LONG_MAX);
  EXPECT_EQ(buffer, fmt::format("{0:o}", LONG_MAX));
  safe_sprintf(buffer, "%lo", ULONG_MAX);
  EXPECT_EQ(buffer, fmt::format("{0:o}", ULONG_MAX));
}

TEST(format_test, format_int_locale) {
  EXPECT_EQ(fmt::format("{:L}", 1234), "1234");
}

TEST(format_test, format_float) {
  EXPECT_EQ(fmt::format("{}", 0.0f), "0");
  EXPECT_EQ(fmt::format("{0:f}", 392.5f), "392.500000");
}

TEST(format_test, format_double) {
  EXPECT_EQ(fmt::format("{}", 0.0), "0");
  check_unknown_types(1.2, "eEfFgGaAnL%", "double");
  EXPECT_EQ(fmt::format("{:}", 0.0), "0");
  EXPECT_EQ(fmt::format("{:f}", 0.0), "0.000000");
  EXPECT_EQ(fmt::format("{:g}", 0.0), "0");
  EXPECT_EQ(fmt::format("{:}", 392.65), "392.65");
  EXPECT_EQ(fmt::format("{:g}", 392.65), "392.65");
  EXPECT_EQ(fmt::format("{:G}", 392.65), "392.65");
  EXPECT_EQ(fmt::format("{:g}", 4.9014e6), "4.9014e+06");
  EXPECT_EQ(fmt::format("{:f}", 392.65), "392.650000");
  EXPECT_EQ(fmt::format("{:F}", 392.65), "392.650000");
  EXPECT_EQ(fmt::format("{:L}", 42.0), "42");
  EXPECT_EQ(fmt::format("{:24a}", 4.2f), "           0x1.0cccccp+2");
  EXPECT_EQ(fmt::format("{:24a}", 4.2), "    0x1.0cccccccccccdp+2");
  EXPECT_EQ(fmt::format("{:<24a}", 4.2), "0x1.0cccccccccccdp+2    ");
  EXPECT_EQ(fmt::format("{0:e}", 392.65), "3.926500e+02");
  EXPECT_EQ(fmt::format("{0:E}", 392.65), "3.926500E+02");
  EXPECT_EQ(fmt::format("{0:+010.4g}", 392.65), "+0000392.6");

#if FMT_CPLUSPLUS >= 201703L
  double xd = 0x1.ffffffffffp+2;
  EXPECT_EQ(fmt::format("{:.10a}", xd), "0x1.ffffffffffp+2");
  EXPECT_EQ(fmt::format("{:.9a}", xd), "0x2.000000000p+2");

  if (std::numeric_limits<long double>::digits == 64) {
    auto ld = 0xf.ffffffffffp-3l;
    EXPECT_EQ(fmt::format("{:a}", ld), "0xf.ffffffffffp-3");
    EXPECT_EQ(fmt::format("{:.10a}", ld), "0xf.ffffffffffp-3");
    EXPECT_EQ(fmt::format("{:.9a}", ld), "0x1.000000000p+1");
  }
#endif

  if (fmt::detail::const_check(std::numeric_limits<double>::is_iec559)) {
    double d = (std::numeric_limits<double>::min)();
    EXPECT_EQ(fmt::format("{:a}", d), "0x1p-1022");
    EXPECT_EQ(fmt::format("{:#a}", d), "0x1.p-1022");

    d = (std::numeric_limits<double>::max)();
    EXPECT_EQ(fmt::format("{:a}", d), "0x1.fffffffffffffp+1023");

    d = std::numeric_limits<double>::denorm_min();
    EXPECT_EQ(fmt::format("{:a}", d), "0x0.0000000000001p-1022");
  }

  if (std::numeric_limits<long double>::digits == 64) {
    auto ld = (std::numeric_limits<long double>::min)();
    EXPECT_EQ(fmt::format("{:a}", ld), "0x8p-16385");

    ld = (std::numeric_limits<long double>::max)();
    EXPECT_EQ(fmt::format("{:a}", ld), "0xf.fffffffffffffffp+16380");

    ld = std::numeric_limits<long double>::denorm_min();
    EXPECT_EQ(fmt::format("{:a}", ld), "0x0.000000000000001p-16382");
  }

  EXPECT_EQ(fmt::format("{:.10a}", 4.2), "0x1.0ccccccccdp+2");

  EXPECT_EQ(fmt::format("{:a}", -42.0), "-0x1.5p+5");
  EXPECT_EQ(fmt::format("{:A}", -42.0), "-0X1.5P+5");

  EXPECT_EQ(fmt::format("{:f}", 9223372036854775807.0),
            "9223372036854775808.000000");
}

TEST(format_test, precision_rounding) {
  EXPECT_EQ(fmt::format("{:.0f}", 0.0), "0");
  EXPECT_EQ(fmt::format("{:.0f}", 0.01), "0");
  EXPECT_EQ(fmt::format("{:.0f}", 0.1), "0");
  EXPECT_EQ(fmt::format("{:.3f}", 0.00049), "0.000");
  EXPECT_EQ(fmt::format("{:.3f}", 0.0005), "0.001");
  EXPECT_EQ(fmt::format("{:.3f}", 0.00149), "0.001");
  EXPECT_EQ(fmt::format("{:.3f}", 0.0015), "0.002");
  EXPECT_EQ(fmt::format("{:.3f}", 0.9999), "1.000");
  EXPECT_EQ(fmt::format("{:.3}", 0.00123), "0.00123");
  EXPECT_EQ(fmt::format("{:.16g}", 0.1), "0.1");
  EXPECT_EQ(fmt::format("{:.0}", 1.0), "1");
  EXPECT_EQ("225.51575035152063720",
            fmt::format("{:.17f}", 225.51575035152064));
  EXPECT_EQ(fmt::format("{:.1f}", -761519619559038.2), "-761519619559038.2");
  EXPECT_EQ("1.9156918820264798e-56",
            fmt::format("{}", 1.9156918820264798e-56));
  EXPECT_EQ(fmt::format("{:.4f}", 7.2809479766055470e-15), "0.0000");
}

TEST(format_test, prettify_float) {
  EXPECT_EQ(fmt::format("{}", 1e-4), "0.0001");
  EXPECT_EQ(fmt::format("{}", 1e-5), "1e-05");
  EXPECT_EQ(fmt::format("{}", 1e15), "1000000000000000");
  EXPECT_EQ(fmt::format("{}", 1e16), "1e+16");
  EXPECT_EQ(fmt::format("{}", 9.999e-5), "9.999e-05");
  EXPECT_EQ(fmt::format("{}", 1e10), "10000000000");
  EXPECT_EQ(fmt::format("{}", 1e11), "100000000000");
  EXPECT_EQ(fmt::format("{}", 1234e7), "12340000000");
  EXPECT_EQ(fmt::format("{}", 1234e-2), "12.34");
  EXPECT_EQ(fmt::format("{}", 1234e-6), "0.001234");
  EXPECT_EQ(fmt::format("{}", 0.1f), "0.1");
  EXPECT_EQ(fmt::format("{}", 1.35631564e-19f), "1.3563156e-19");
}

TEST(format_test, format_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(fmt::format("{}", nan), "nan");
  EXPECT_EQ(fmt::format("{:+}", nan), "+nan");
  EXPECT_EQ(fmt::format("{:+06}", nan), "  +nan");
  EXPECT_EQ(fmt::format("{:<+06}", nan), "+nan  ");
  EXPECT_EQ(fmt::format("{:^+06}", nan), " +nan ");
  EXPECT_EQ(fmt::format("{:>+06}", nan), "  +nan");
  if (std::signbit(-nan)) {
    EXPECT_EQ(fmt::format("{}", -nan), "-nan");
    EXPECT_EQ(fmt::format("{:+06}", -nan), "  -nan");
  } else {
    fmt::print("Warning: compiler doesn't handle negative NaN correctly");
  }
  EXPECT_EQ(fmt::format("{: }", nan), " nan");
  EXPECT_EQ(fmt::format("{:F}", nan), "NAN");
  EXPECT_EQ(fmt::format("{:<7}", nan), "nan    ");
  EXPECT_EQ(fmt::format("{:^7}", nan), "  nan  ");
  EXPECT_EQ(fmt::format("{:>7}", nan), "    nan");
}

TEST(format_test, format_infinity) {
  double inf = std::numeric_limits<double>::infinity();
  EXPECT_EQ(fmt::format("{}", inf), "inf");
  EXPECT_EQ(fmt::format("{:+}", inf), "+inf");
  EXPECT_EQ(fmt::format("{}", -inf), "-inf");
  EXPECT_EQ(fmt::format("{:+06}", inf), "  +inf");
  EXPECT_EQ(fmt::format("{:+06}", -inf), "  -inf");
  EXPECT_EQ(fmt::format("{:<+06}", inf), "+inf  ");
  EXPECT_EQ(fmt::format("{:^+06}", inf), " +inf ");
  EXPECT_EQ(fmt::format("{:>+06}", inf), "  +inf");
  EXPECT_EQ(fmt::format("{: }", inf), " inf");
  EXPECT_EQ(fmt::format("{:F}", inf), "INF");
  EXPECT_EQ(fmt::format("{:<7}", inf), "inf    ");
  EXPECT_EQ(fmt::format("{:^7}", inf), "  inf  ");
  EXPECT_EQ(fmt::format("{:>7}", inf), "    inf");
}

TEST(format_test, format_long_double) {
  EXPECT_EQ(fmt::format("{0:}", 0.0l), "0");
  EXPECT_EQ(fmt::format("{0:f}", 0.0l), "0.000000");
  EXPECT_EQ(fmt::format("{:.1f}", 0.000000001l), "0.0");
  EXPECT_EQ(fmt::format("{:.2f}", 0.099l), "0.10");
  EXPECT_EQ(fmt::format("{0:}", 392.65l), "392.65");
  EXPECT_EQ(fmt::format("{0:g}", 392.65l), "392.65");
  EXPECT_EQ(fmt::format("{0:G}", 392.65l), "392.65");
  EXPECT_EQ(fmt::format("{0:f}", 392.65l), "392.650000");
  EXPECT_EQ(fmt::format("{0:F}", 392.65l), "392.650000");
  char buffer[buffer_size];
  safe_sprintf(buffer, "%Le", 392.65l);
  EXPECT_EQ(buffer, fmt::format("{0:e}", 392.65l));
  EXPECT_EQ(fmt::format("{0:+010.4g}", 392.64l), "+0000392.6");

  auto ld = 3.31l;
  if (fmt::detail::is_double_double<decltype(ld)>::value) {
    safe_sprintf(buffer, "%a", static_cast<double>(ld));
    EXPECT_EQ(buffer, fmt::format("{:a}", ld));
  } else if (std::numeric_limits<long double>::digits == 64) {
    EXPECT_EQ(fmt::format("{:a}", ld), "0xd.3d70a3d70a3d70ap-2");
  }
}

TEST(format_test, format_char) {
  const char types[] = "cbBdoxX";
  check_unknown_types('a', types, "char");
  EXPECT_EQ(fmt::format("{0}", 'a'), "a");
  EXPECT_EQ(fmt::format("{0:c}", 'z'), "z");
  int n = 'x';
  for (const char* type = types + 1; *type; ++type) {
    std::string format_str = fmt::format("{{:{}}}", *type);
    EXPECT_EQ(fmt::format(runtime(format_str), n),
              fmt::format(runtime(format_str), 'x'))
        << format_str;
  }
  EXPECT_EQ(fmt::format("{:02X}", n), fmt::format("{:02X}", 'x'));

  EXPECT_EQ(fmt::format("{}", '\n'), "\n");
  EXPECT_EQ(fmt::format("{:?}", '\n'), "'\\n'");
  EXPECT_EQ(fmt::format("{:x}", '\xff'), "ff");
}

TEST(format_test, format_volatile_char) {
  volatile char c = 'x';
  EXPECT_EQ(fmt::format("{}", c), "x");
}

TEST(format_test, format_unsigned_char) {
  EXPECT_EQ(fmt::format("{}", static_cast<unsigned char>(42)), "42");
  EXPECT_EQ(fmt::format("{}", static_cast<uint8_t>(42)), "42");
}

TEST(format_test, format_cstring) {
  check_unknown_types("test", "sp", "string");
  EXPECT_EQ(fmt::format("{0}", "test"), "test");
  EXPECT_EQ(fmt::format("{0:s}", "test"), "test");
  char nonconst[] = "nonconst";
  EXPECT_EQ(fmt::format("{0}", nonconst), "nonconst");
  auto nullstr = static_cast<const char*>(nullptr);
  EXPECT_THROW_MSG((void)fmt::format("{}", nullstr), format_error,
                   "string pointer is null");
  EXPECT_THROW_MSG((void)fmt::format("{:s}", nullstr), format_error,
                   "string pointer is null");
}

void function_pointer_test(int, double, std::string) {}

TEST(format_test, format_pointer) {
  check_unknown_types(reinterpret_cast<void*>(0x1234), "p", "pointer");
  EXPECT_EQ(fmt::format("{0}", static_cast<void*>(nullptr)), "0x0");
  EXPECT_EQ(fmt::format("{0}", reinterpret_cast<void*>(0x1234)), "0x1234");
  EXPECT_EQ(fmt::format("{0:p}", reinterpret_cast<void*>(0x1234)), "0x1234");
  // On CHERI (or other fat-pointer) systems, the size of a pointer is greater
  // than the size an integer that can hold a virtual address.  There is no
  // portable address-as-an-integer type (yet) in C++, so we use `size_t` as
  // the closest equivalent for now.
  EXPECT_EQ("0x" + std::string(sizeof(size_t) * CHAR_BIT / 4, 'f'),
            fmt::format("{0}", reinterpret_cast<void*>(~uintptr_t())));
  EXPECT_EQ("0x1234",
            fmt::format("{}", fmt::ptr(reinterpret_cast<int*>(0x1234))));
  EXPECT_EQ(fmt::format("{}", fmt::detail::bit_cast<const void*>(
                                  &function_pointer_test)),
            fmt::format("{}", fmt::ptr(function_pointer_test)));
  EXPECT_EQ(fmt::format("{}", nullptr), "0x0");
}

TEST(format_test, write_uintptr_fallback) {
  // Test that formatting a pointer by converting it to uint128_fallback works.
  // This is needed to support systems without uintptr_t.
  auto s = std::string();
  fmt::detail::write_ptr<char>(
      std::back_inserter(s),
      fmt::detail::bit_cast<fmt::detail::uint128_fallback>(
          reinterpret_cast<void*>(0xface)),
      nullptr);
  EXPECT_EQ(s, "0xface");
}

enum class color { red, green, blue };

namespace test_ns {
enum class color { red, green, blue };
using fmt::enums::format_as;
}  // namespace test_ns

TEST(format_test, format_enum_class) {
  EXPECT_EQ(fmt::format("{}", fmt::underlying(color::red)), "0");
  EXPECT_EQ(fmt::format("{}", test_ns::color::red), "0");
}

TEST(format_test, format_string) {
  EXPECT_EQ(fmt::format("{0}", std::string("test")), "test");
  EXPECT_EQ(fmt::format("{0}", std::string("test")), "test");
  EXPECT_EQ(fmt::format("{:?}", std::string("test")), "\"test\"");
  EXPECT_EQ(fmt::format("{:*^10?}", std::string("test")), "**\"test\"**");
  EXPECT_EQ(fmt::format("{:?}", std::string("\test")), "\"\\test\"");
  EXPECT_THROW((void)fmt::format(fmt::runtime("{:x}"), std::string("test")),
               fmt::format_error);
}

TEST(format_test, format_string_view) {
  EXPECT_EQ(fmt::format("{}", string_view("test")), "test");
  EXPECT_EQ(fmt::format("{:?}", string_view("t\nst")), "\"t\\nst\"");
  EXPECT_EQ(fmt::format("{}", string_view()), "");
}

#ifdef FMT_USE_STRING_VIEW
struct string_viewable {};

FMT_BEGIN_NAMESPACE
template <> struct formatter<string_viewable> : formatter<std::string_view> {
  auto format(string_viewable, format_context& ctx) const
      -> decltype(ctx.out()) {
    return formatter<std::string_view>::format("foo", ctx);
  }
};
FMT_END_NAMESPACE

TEST(format_test, format_std_string_view) {
  EXPECT_EQ(fmt::format("{}", std::string_view("test")), "test");
  EXPECT_EQ(fmt::format("{}", string_viewable()), "foo");
}

struct explicitly_convertible_to_std_string_view {
  explicit operator std::string_view() const { return "foo"; }
};

template <>
struct fmt::formatter<explicitly_convertible_to_std_string_view>
    : formatter<std::string_view> {
  auto format(explicitly_convertible_to_std_string_view v,
              format_context& ctx) const -> decltype(ctx.out()) {
    return fmt::format_to(ctx.out(), "'{}'", std::string_view(v));
  }
};

TEST(format_test, format_explicitly_convertible_to_std_string_view) {
  EXPECT_EQ("'foo'",
            fmt::format("{}", explicitly_convertible_to_std_string_view()));
}

struct convertible_to_std_string_view {
  operator std::string_view() const noexcept { return "Hi there"; }
};
FMT_BEGIN_NAMESPACE
template <>
class formatter<convertible_to_std_string_view>
    : public formatter<std::string_view> {};
FMT_END_NAMESPACE

TEST(format_test, format_implicitly_convertible_and_inherits_string_view) {
  static_assert(fmt::is_formattable<convertible_to_std_string_view>{}, "");
  EXPECT_EQ("Hi there", fmt::format("{}", convertible_to_std_string_view{}));
}
#endif

class Answer {};

FMT_BEGIN_NAMESPACE
template <> struct formatter<date> {
  template <typename ParseContext>
  FMT_CONSTEXPR auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    auto it = ctx.begin();
    if (it != ctx.end() && *it == 'd') ++it;
    return it;
  }

  auto format(const date& d, format_context& ctx) const -> decltype(ctx.out()) {
    // Namespace-qualify to avoid ambiguity with std::format_to.
    fmt::format_to(ctx.out(), "{}-{}-{}", d.year(), d.month(), d.day());
    return ctx.out();
  }
};

template <> struct formatter<Answer> : formatter<int> {
  template <typename FormatContext>
  auto format(Answer, FormatContext& ctx) const -> decltype(ctx.out()) {
    return formatter<int>::format(42, ctx);
  }
};
FMT_END_NAMESPACE

TEST(format_test, format_custom) {
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:s}"), date(2012, 12, 9)),
                   format_error, "unknown format specifier");
  EXPECT_EQ(fmt::format("{0}", Answer()), "42");
  EXPECT_EQ(fmt::format("{:04}", Answer()), "0042");
}

TEST(format_test, format_to_custom) {
  char buf[10] = {};
  auto end = fmt::format_to(buf, "{}", Answer());
  EXPECT_EQ(end, buf + 2);
  EXPECT_STREQ(buf, "42");
}

TEST(format_test, format_string_from_speed_test) {
  EXPECT_EQ("1.2340000000:0042:+3.13:str:0x3e8:X:%",
            fmt::format("{0:0.10f}:{1:04}:{2:+g}:{3}:{4}:{5}:%", 1.234, 42,
                        3.13, "str", reinterpret_cast<void*>(1000), 'X'));
}

TEST(format_test, format_examples) {
  std::string message = fmt::format("The answer is {}", 42);
  EXPECT_EQ("The answer is 42", message);

  EXPECT_EQ(fmt::format("{}", 42), "42");

  memory_buffer out;
  fmt::format_to(std::back_inserter(out), "The answer is {}.", 42);
  EXPECT_EQ("The answer is 42.", to_string(out));

  EXPECT_THROW(
      {
        const char* filename = "madeup";
        FILE* file = fopen(filename, "r");
        if (!file)
          throw fmt::system_error(errno, "cannot open file '{}'", filename);
        fclose(file);
      },
      std::system_error);

  EXPECT_EQ("First, thou shalt count to three",
            fmt::format("First, thou shalt count to {0}", "three"));
  EXPECT_EQ(fmt::format("Bring me a {}", "shrubbery"), "Bring me a shrubbery");
  EXPECT_EQ(fmt::format("From {} to {}", 1, 3), "From 1 to 3");

  char buffer[buffer_size];
  safe_sprintf(buffer, "%03.2f", -1.2);
  EXPECT_EQ(buffer, fmt::format("{:03.2f}", -1.2));

  EXPECT_EQ(fmt::format("{0}, {1}, {2}", 'a', 'b', 'c'), "a, b, c");
  EXPECT_EQ(fmt::format("{}, {}, {}", 'a', 'b', 'c'), "a, b, c");
  EXPECT_EQ(fmt::format("{2}, {1}, {0}", 'a', 'b', 'c'), "c, b, a");
  EXPECT_EQ(fmt::format("{0}{1}{0}", "abra", "cad"), "abracadabra");

  EXPECT_EQ("left aligned                  ",
            fmt::format("{:<30}", "left aligned"));
  EXPECT_EQ("                 right aligned",
            fmt::format("{:>30}", "right aligned"));
  EXPECT_EQ("           centered           ",
            fmt::format("{:^30}", "centered"));
  EXPECT_EQ("***********centered***********",
            fmt::format("{:*^30}", "centered"));

  EXPECT_EQ(fmt::format("{:+f}; {:+f}", 3.14, -3.14), "+3.140000; -3.140000");
  EXPECT_EQ(fmt::format("{: f}; {: f}", 3.14, -3.14), " 3.140000; -3.140000");
  EXPECT_EQ(fmt::format("{:-f}; {:-f}", 3.14, -3.14), "3.140000; -3.140000");

  EXPECT_EQ("int: 42;  hex: 2a;  oct: 52",
            fmt::format("int: {0:d};  hex: {0:x};  oct: {0:o}", 42));
  EXPECT_EQ("int: 42;  hex: 0x2a;  oct: 052",
            fmt::format("int: {0:d};  hex: {0:#x};  oct: {0:#o}", 42));

  EXPECT_EQ(fmt::format("The answer is {}", 42), "The answer is 42");
  EXPECT_THROW_MSG(
      (void)fmt::format(runtime("The answer is {:d}"), "forty-two"),
      format_error, "invalid format specifier");

  EXPECT_WRITE(
      stdout, fmt::print("{}", std::numeric_limits<double>::infinity()), "inf");
}

TEST(format_test, print) {
  EXPECT_WRITE(stdout, fmt::print("Don't {}!", "panic"), "Don't panic!");
  EXPECT_WRITE(stderr, fmt::print(stderr, "Don't {}!", "panic"),
               "Don't panic!");
  EXPECT_WRITE(stdout, fmt::println("Don't {}!", "panic"), "Don't panic!\n");
  EXPECT_WRITE(stderr, fmt::println(stderr, "Don't {}!", "panic"),
               "Don't panic!\n");
}

TEST(format_test, big_print) {
  enum { count = 5000 };
  auto big_print = []() {
    for (int i = 0; i < count / 5; ++i) fmt::print("xxxxx");
  };
  EXPECT_WRITE(stdout, big_print(), std::string(count, 'x'));
}

struct deadlockable {
  int value = 0;
  mutable std::mutex mutex;
};

FMT_BEGIN_NAMESPACE
template <> struct formatter<deadlockable> {
  FMT_CONSTEXPR auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    return ctx.begin();
  }

  auto format(const deadlockable& d, format_context& ctx) const
      -> decltype(ctx.out()) {
    std::lock_guard<std::mutex> lock(d.mutex);
    return format_to(ctx.out(), "{}", d.value);
  }
};
FMT_END_NAMESPACE

TEST(format_test, locking_formatter) {
  auto f = fmt::buffered_file();
  try {
    f = fmt::buffered_file("/dev/null", "w");
  } catch (const std::system_error&) {
    fmt::print(stderr, "warning: /dev/null is not supported\n");
    return;
  }
  deadlockable d;
  auto t = std::thread([&]() {
    fmt::print(f.get(), "start t\n");
    std::lock_guard<std::mutex> lock(d.mutex);
    for (int i = 0; i < 1000000; ++i) d.value += 10;
    fmt::print(f.get(), "done\n");
  });
  for (int i = 0; i < 100; ++i) fmt::print(f.get(), "{}", d);
  t.join();
}

TEST(format_test, variadic) {
  EXPECT_EQ(fmt::format("{}c{}", "ab", 1), "abc1");
}

TEST(format_test, bytes) {
  auto s = fmt::format("{:10}", fmt::bytes("ёжик"));
  EXPECT_EQ("ёжик  ", s);
  EXPECT_EQ(10, s.size());
}

TEST(format_test, group_digits_view) {
  EXPECT_EQ(fmt::format("{}", fmt::group_digits(10000000)), "10,000,000");
  EXPECT_EQ(fmt::format("{:8}", fmt::group_digits(1000)), "   1,000");
  EXPECT_EQ(fmt::format("{}", fmt::group_digits(-10000000)), "-10,000,000");
  EXPECT_EQ(fmt::format("{:8}", fmt::group_digits(-1000)), "  -1,000");
  EXPECT_EQ(fmt::format("{:8}", fmt::group_digits(-100)), "    -100");
}

#ifdef __cpp_generic_lambdas
struct point {
  double x, y;
};

FMT_BEGIN_NAMESPACE
template <> struct formatter<point> : nested_formatter<double> {
  auto format(point p, format_context& ctx) const -> decltype(ctx.out()) {
    return write_padded(ctx, [this, p](auto out) -> decltype(out) {
      return fmt::format_to(out, "({}, {})", this->nested(p.x),
                            this->nested(p.y));
    });
  }
};
FMT_END_NAMESPACE

TEST(format_test, nested_formatter) {
  EXPECT_EQ(fmt::format("{:>16.2f}", point{1, 2}), "    (1.00, 2.00)");
}
#endif  // __cpp_generic_lambdas

enum test_enum { foo, bar };
auto format_as(test_enum e) -> int { return e; }

std::string vformat_message(int id, const char* format, fmt::format_args args) {
  auto buffer = fmt::memory_buffer();
  fmt::format_to(fmt::appender(buffer), "[{}] ", id);
  vformat_to(fmt::appender(buffer), format, args);
  return to_string(buffer);
}

template <typename... Args>
std::string format_message(int id, const char* format, const Args&... args) {
  auto va = fmt::make_format_args(args...);
  return vformat_message(id, format, va);
}

TEST(format_test, format_message_example) {
  EXPECT_EQ("[42] something happened",
            format_message(42, "{} happened", "something"));
}

template <typename... Args>
void print_error(const char* file, int line, const char* format,
                 const Args&... args) {
  fmt::print("{}: {}: ", file, line);
  fmt::print(format, args...);
}

TEST(format_test, unpacked_args) {
  EXPECT_EQ("0123456789abcdefg",
            fmt::format("{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}", 0, 1, 2, 3, 4, 5,
                        6, 7, 8, 9, 'a', 'b', 'c', 'd', 'e', 'f', 'g'));
}

constexpr char with_null[3] = {'{', '}', '\0'};
constexpr char no_null[2] = {'{', '}'};
static constexpr char static_with_null[3] = {'{', '}', '\0'};
static constexpr char static_no_null[2] = {'{', '}'};

TEST(format_test, compile_time_string) {
  EXPECT_EQ(fmt::format(FMT_STRING("foo")), "foo");
  EXPECT_EQ(fmt::format(FMT_STRING("{}"), 42), "42");

#if FMT_USE_NONTYPE_TEMPLATE_ARGS
  using namespace fmt::literals;
  EXPECT_EQ("foobar", fmt::format(FMT_STRING("{foo}{bar}"), "bar"_a = "bar",
                                  "foo"_a = "foo"));
  EXPECT_EQ(fmt::format(FMT_STRING("")), "");
  EXPECT_EQ(fmt::format(FMT_STRING(""), "arg"_a = 42), "");
  EXPECT_EQ(fmt::format(FMT_STRING("{answer}"), "answer"_a = Answer()), "42");
  EXPECT_EQ(fmt::format(FMT_STRING("{} {two}"), 1, "two"_a = 2), "1 2");
#endif

  (void)static_with_null;
  (void)static_no_null;
#ifndef _MSC_VER
  EXPECT_EQ(fmt::format(FMT_STRING(static_with_null), 42), "42");
  EXPECT_EQ(fmt::format(FMT_STRING(static_no_null), 42), "42");
#endif

  (void)with_null;
  (void)no_null;
#if FMT_CPLUSPLUS >= 201703L
  EXPECT_EQ(fmt::format(FMT_STRING(with_null), 42), "42");
  EXPECT_EQ(fmt::format(FMT_STRING(no_null), 42), "42");
#endif
#if defined(FMT_USE_STRING_VIEW) && FMT_CPLUSPLUS >= 201703L
  EXPECT_EQ(fmt::format(FMT_STRING(std::string_view("{}")), 42), "42");
#endif
}

TEST(format_test, custom_format_compile_time_string) {
  EXPECT_EQ(fmt::format(FMT_STRING("{}"), Answer()), "42");
  auto answer = Answer();
  EXPECT_EQ(fmt::format(FMT_STRING("{}"), answer), "42");
  char buf[10] = {};
  fmt::format_to(buf, FMT_STRING("{}"), answer);
  const Answer const_answer = Answer();
  EXPECT_EQ(fmt::format(FMT_STRING("{}"), const_answer), "42");
}

TEST(format_test, named_arg_udl) {
  using namespace fmt::literals;
  auto udl_a = fmt::format("{first}{second}{first}{third}", "first"_a = "abra",
                           "second"_a = "cad", "third"_a = 99);
  EXPECT_EQ(
      fmt::format("{first}{second}{first}{third}", fmt::arg("first", "abra"),
                  fmt::arg("second", "cad"), fmt::arg("third", 99)),
      udl_a);

  EXPECT_EQ(fmt::format("{answer}", "answer"_a = Answer()), "42");
}

TEST(format_test, enum) { EXPECT_EQ(fmt::format("{}", foo), "0"); }

TEST(format_test, formatter_not_specialized) {
  static_assert(!fmt::is_formattable<fmt::formatter<test_enum>,
                                     fmt::format_context>::value,
                "");
}

#if FMT_HAS_FEATURE(cxx_strong_enums)
enum big_enum : unsigned long long { big_enum_value = 5000000000ULL };
auto format_as(big_enum e) -> unsigned long long { return e; }

TEST(format_test, strong_enum) {
  auto arg = fmt::basic_format_arg<fmt::context>(big_enum_value);
  EXPECT_EQ(arg.type(), fmt::detail::type::ulong_long_type);
  EXPECT_EQ(fmt::format("{}", big_enum_value), "5000000000");
}
#endif

TEST(format_test, non_null_terminated_format_string) {
  EXPECT_EQ(fmt::format(string_view("{}foo", 2), 42), "42");
}

namespace adl_test {
namespace fmt {
namespace detail {
struct foo {};
template <typename, typename OutputIt> void write(OutputIt, foo) = delete;
}  // namespace detail
}  // namespace fmt
}  // namespace adl_test

FMT_BEGIN_NAMESPACE
template <>
struct formatter<adl_test::fmt::detail::foo> : formatter<std::string> {
  auto format(adl_test::fmt::detail::foo, format_context& ctx) const
      -> decltype(ctx.out()) {
    return formatter<std::string>::format("foo", ctx);
  }
};
FMT_END_NAMESPACE

TEST(format_test, to_string) {
  EXPECT_EQ(fmt::to_string(42), "42");
  EXPECT_EQ(fmt::to_string(reinterpret_cast<void*>(0x1234)), "0x1234");
  EXPECT_EQ(fmt::to_string(adl_test::fmt::detail::foo()), "foo");
  EXPECT_EQ(fmt::to_string(foo), "0");

#if FMT_USE_FLOAT128
  EXPECT_EQ(fmt::to_string(__float128(0.5)), "0.5");
#endif

#if defined(FMT_USE_STRING_VIEW) && FMT_CPLUSPLUS >= 201703L
  EXPECT_EQ(fmt::to_string(std::string_view()), "");
#endif
}

TEST(format_test, output_iterators) {
  std::list<char> out;
  fmt::format_to(std::back_inserter(out), "{}", 42);
  EXPECT_EQ("42", std::string(out.begin(), out.end()));
  std::stringstream s;
  fmt::format_to(std::ostream_iterator<char>(s), "{}", 42);
  EXPECT_EQ("42", s.str());

  std::stringstream s2;
  fmt::format_to(std::ostreambuf_iterator<char>(s2), "{}.{:06d}", 42, 43);
  EXPECT_EQ("42.000043", s2.str());
}

TEST(format_test, fill_via_appender) {
  fmt::memory_buffer buf;
  auto it = fmt::appender(buf);
  std::fill_n(it, 3, '~');
  EXPECT_EQ(fmt::to_string(buf), "~~~");
}

TEST(format_test, formatted_size) {
  EXPECT_EQ(2u, fmt::formatted_size("{}", 42));
  EXPECT_EQ(2u, fmt::formatted_size(std::locale(), "{}", 42));
}

TEST(format_test, format_to_no_args) {
  std::string s;
  fmt::format_to(std::back_inserter(s), "test");
  EXPECT_EQ("test", s);
}

TEST(format_test, format_to) {
  std::string s;
  fmt::format_to(std::back_inserter(s), "part{0}", 1);
  EXPECT_EQ("part1", s);
  fmt::format_to(std::back_inserter(s), "part{0}", 2);
  EXPECT_EQ("part1part2", s);
}

TEST(format_test, format_to_memory_buffer) {
  auto buf = fmt::basic_memory_buffer<char, 100>();
  fmt::format_to(fmt::appender(buf), "{}", "foo");
  EXPECT_EQ("foo", to_string(buf));
}

TEST(format_test, format_to_vector) {
  std::vector<char> v;
  fmt::format_to(std::back_inserter(v), "{}", "foo");
  EXPECT_EQ(string_view(v.data(), v.size()), "foo");
}

struct nongrowing_container {
  using value_type = char;
  void push_back(char) { throw std::runtime_error("can't take it any more"); }
};

TEST(format_test, format_to_propagates_exceptions) {
  auto c = nongrowing_container();
  EXPECT_THROW(fmt::format_to(std::back_inserter(c), "{}", 42),
               std::runtime_error);
}

TEST(format_test, format_to_n) {
  char buffer[4];
  buffer[3] = 'x';
  auto result = fmt::format_to_n(buffer, 3, "{}", 12345);
  EXPECT_EQ(5u, result.size);
  EXPECT_EQ(buffer + 3, result.out);
  EXPECT_EQ("123x", fmt::string_view(buffer, 4));

  result = fmt::format_to_n(buffer, 3, "{:s}", "foobar");
  EXPECT_EQ(6u, result.size);
  EXPECT_EQ(buffer + 3, result.out);
  EXPECT_EQ("foox", fmt::string_view(buffer, 4));

  buffer[0] = 'x';
  buffer[1] = 'x';
  buffer[2] = 'x';
  result = fmt::format_to_n(buffer, 3, "{}", 'A');
  EXPECT_EQ(1u, result.size);
  EXPECT_EQ(buffer + 1, result.out);
  EXPECT_EQ("Axxx", fmt::string_view(buffer, 4));

  result = fmt::format_to_n(buffer, 3, "{}{} ", 'B', 'C');
  EXPECT_EQ(3u, result.size);
  EXPECT_EQ(buffer + 3, result.out);
  EXPECT_EQ("BC x", fmt::string_view(buffer, 4));

  result = fmt::format_to_n(buffer, 4, "{}", "ABCDE");
  EXPECT_EQ(5u, result.size);
  EXPECT_EQ("ABCD", fmt::string_view(buffer, 4));

  buffer[3] = 'x';
  result = fmt::format_to_n(buffer, 3, "{}", std::string(1000, '*'));
  EXPECT_EQ(1000u, result.size);
  EXPECT_EQ("***x", fmt::string_view(buffer, 4));
}

struct test_output_iterator {
  char* data;

  using iterator_category = std::output_iterator_tag;
  using value_type = void;
  using difference_type = void;
  using pointer = void;
  using reference = void;

  auto operator++() -> test_output_iterator& {
    ++data;
    return *this;
  }
  auto operator++(int) -> test_output_iterator {
    auto tmp = *this;
    ++data;
    return tmp;
  }
  auto operator*() -> char& { return *data; }
};

TEST(format_test, format_to_n_output_iterator) {
  char buf[10] = {};
  fmt::format_to_n(test_output_iterator{buf}, 10, "{}", 42);
  EXPECT_STREQ(buf, "42");
}

TEST(format_test, vformat_to) {
  using context = fmt::format_context;
  int n = 42;
  auto args = fmt::make_format_args<context>(n);
  auto s = std::string();
  fmt::vformat_to(std::back_inserter(s), "{}", args);
  EXPECT_EQ(s, "42");
  s.clear();
  fmt::vformat_to(std::back_inserter(s), "{}", args);
  EXPECT_EQ(s, "42");
}

TEST(format_test, char_traits_not_ambiguous) {
  // Test that we don't inject detail names into the std namespace.
  using namespace std;
  auto c = char_traits<char>::char_type();
  (void)c;
#if FMT_CPLUSPLUS >= 201103L
  auto s = std::string();
  auto lval = begin(s);
  (void)lval;
#endif
}

struct check_back_appender {};

FMT_BEGIN_NAMESPACE
template <> struct formatter<check_back_appender> {
  FMT_CONSTEXPR auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    return ctx.begin();
  }

  template <typename Context>
  auto format(check_back_appender, Context& ctx) const -> decltype(ctx.out()) {
    auto out = ctx.out();
    static_assert(std::is_same<decltype(++out), decltype(out)&>::value,
                  "needs to satisfy weakly_incrementable");
    *out = 'y';
    return ++out;
  }
};
FMT_END_NAMESPACE

TEST(format_test, back_insert_slicing) {
  EXPECT_EQ(fmt::format("{}", check_back_appender{}), "y");
}

namespace test {
enum class scoped_enum_as_int {};
auto format_as(scoped_enum_as_int) -> int { return 42; }

enum class scoped_enum_as_string_view {};
auto format_as(scoped_enum_as_string_view) -> fmt::string_view { return "foo"; }

enum class scoped_enum_as_string {};
auto format_as(scoped_enum_as_string) -> std::string { return "foo"; }

struct struct_as_int {};
auto format_as(struct_as_int) -> int { return 42; }

struct struct_as_const_reference {
  const std::string name = "foo";
};
auto format_as(const struct_as_const_reference& s) -> const std::string& {
  return s.name;
}
}  // namespace test

TEST(format_test, format_as) {
  EXPECT_EQ(fmt::format("{}", test::scoped_enum_as_int()), "42");
  EXPECT_EQ(fmt::format("{}", test::scoped_enum_as_string_view()), "foo");
  EXPECT_EQ(fmt::format("{}", test::scoped_enum_as_string()), "foo");
  EXPECT_EQ(fmt::format("{}", test::struct_as_int()), "42");
  EXPECT_EQ(fmt::format("{}", test::struct_as_const_reference()), "foo");
}

TEST(format_test, format_as_to_string) {
  EXPECT_EQ(fmt::to_string(test::scoped_enum_as_int()), "42");
  EXPECT_EQ(fmt::to_string(test::scoped_enum_as_string_view()), "foo");
  EXPECT_EQ(fmt::to_string(test::scoped_enum_as_string()), "foo");
  EXPECT_EQ(fmt::to_string(test::struct_as_int()), "42");
}

template <typename Char, typename T> auto check_enabled_formatter() -> bool {
  static_assert(std::is_default_constructible<fmt::formatter<T, Char>>::value,
                "");
  return true;
}

template <typename Char, typename... T> void check_enabled_formatters() {
  auto dummy = {check_enabled_formatter<Char, T>()...};
  (void)dummy;
}

TEST(format_test, test_formatters_enabled) {
  using custom_string =
      std::basic_string<char, std::char_traits<char>, mock_allocator<char>>;
  using custom_wstring = std::basic_string<wchar_t, std::char_traits<wchar_t>,
                                           mock_allocator<wchar_t>>;

  check_enabled_formatters<char, bool, char, signed char, unsigned char, short,
                           unsigned short, int, unsigned, long, unsigned long,
                           long long, unsigned long long, float, double,
                           long double, void*, const void*, char*, const char*,
                           std::string, custom_string, std::nullptr_t>();
  check_enabled_formatters<
      wchar_t, bool, wchar_t, signed char, unsigned char, short, unsigned short,
      int, unsigned, long, unsigned long, long long, unsigned long long, float,
      double, long double, void*, const void*, wchar_t*, const wchar_t*,
      std::wstring, custom_wstring, std::nullptr_t>();
}

TEST(format_int_test, data) {
  fmt::format_int format_int(42);
  EXPECT_EQ(std::string(format_int.data(), format_int.size()), "42");
}

TEST(format_int_test, format_int) {
  EXPECT_EQ(fmt::format_int(42).str(), "42");
  EXPECT_EQ(fmt::format_int(42).size(), 2u);
  EXPECT_EQ(fmt::format_int(-42).str(), "-42");
  EXPECT_EQ(fmt::format_int(-42).size(), 3u);
  EXPECT_EQ(fmt::format_int(42ul).str(), "42");
  EXPECT_EQ(fmt::format_int(-42l).str(), "-42");
  EXPECT_EQ(fmt::format_int(42ull).str(), "42");
  EXPECT_EQ(fmt::format_int(-42ll).str(), "-42");
  EXPECT_EQ(fmt::format_int(max_value<int64_t>()).str(),
            std::to_string(max_value<int64_t>()));
}

#ifndef FMT_STATIC_THOUSANDS_SEPARATOR

#  include <locale>

class format_facet : public fmt::format_facet<std::locale> {
 protected:
  struct int_formatter {
    fmt::appender out;

    template <typename T, FMT_ENABLE_IF(fmt::detail::is_integer<T>::value)>
    auto operator()(T value) -> bool {
      fmt::format_to(out, "[{}]", value);
      return true;
    }

    template <typename T, FMT_ENABLE_IF(!fmt::detail::is_integer<T>::value)>
    auto operator()(T) -> bool {
      return false;
    }
  };

  auto do_put(fmt::appender out, fmt::loc_value val,
              const fmt::format_specs&) const -> bool override;
};

auto format_facet::do_put(fmt::appender out, fmt::loc_value val,
                          const fmt::format_specs&) const -> bool {
  return val.visit(int_formatter{out});
}

TEST(format_test, format_facet) {
  auto loc = std::locale(std::locale(), new format_facet());
  EXPECT_EQ(fmt::format(loc, "{:L}", 42), "[42]");
  EXPECT_EQ(fmt::format(loc, "{:L}", -42), "[-42]");
}

TEST(format_test, format_facet_separator) {
  // U+2019 RIGHT SINGLE QUOTATION MARK is a digit separator in the de_CH
  // locale.
  auto loc =
      std::locale({}, new fmt::format_facet<std::locale>("\xe2\x80\x99"));
  EXPECT_EQ(fmt::format(loc, "{:L}", 1000),
            "1\xe2\x80\x99"
            "000");
}

TEST(format_test, format_facet_grouping) {
  auto loc =
      std::locale({}, new fmt::format_facet<std::locale>(",", {1, 2, 3}));
  EXPECT_EQ(fmt::format(loc, "{:L}", 1234567890), "1,234,567,89,0");
}

TEST(format_test, format_named_arg_with_locale) {
  EXPECT_EQ(fmt::format(std::locale(), "{answer}", fmt::arg("answer", 42)),
            "42");
}

TEST(format_test, format_locale) {
  auto loc = std::locale({}, new fmt::format_facet<std::locale>(","));
  EXPECT_EQ(fmt::format(loc, "{:Lx}", 123456789), "7,5bc,d15");
  EXPECT_EQ(fmt::format(loc, "{:#Lb}", -123456789),
            "-0b111,010,110,111,100,110,100,010,101");
  EXPECT_EQ(fmt::format(loc, "{:10Lo}", 12345), "    30,071");
}

#endif  // FMT_STATIC_THOUSANDS_SEPARATOR

struct convertible_to_nonconst_cstring {
  operator char*() const {
    static char c[] = "bar";
    return c;
  }
};

FMT_BEGIN_NAMESPACE
template <>
struct formatter<convertible_to_nonconst_cstring> : formatter<char*> {};
FMT_END_NAMESPACE

TEST(format_test, formatter_nonconst_char) {
  EXPECT_EQ(fmt::format("{}", convertible_to_nonconst_cstring()), "bar");
}

namespace adl_test {
template <typename... T> void make_format_args(const T&...) = delete;

struct string : std::string {};
auto format_as(const string& s) -> std::string { return s; }
}  // namespace adl_test

// Test that formatting functions compile when make_format_args is found by ADL.
TEST(format_test, adl) {
  // Only check compilation and don't run the code to avoid polluting the output
  // and since the output is tested elsewhere.
  if (fmt::detail::const_check(true)) return;
  auto s = adl_test::string();
  char buf[10];
  (void)fmt::format("{}", s);
  fmt::format_to(buf, "{}", s);
  fmt::format_to_n(buf, 10, "{}", s);
  (void)fmt::formatted_size("{}", s);
  fmt::print("{}", s);
  fmt::print(stdout, "{}", s);
}

struct convertible_to_int {
  operator int() const { return 42; }
};

struct convertible_to_cstring {
  operator const char*() const { return "foo"; }
};

FMT_BEGIN_NAMESPACE
template <> struct formatter<convertible_to_int> {
  FMT_CONSTEXPR auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    return ctx.begin();
  }
  auto format(convertible_to_int, format_context& ctx) const
      -> decltype(ctx.out()) {
    auto out = ctx.out();
    *out++ = 'x';
    return out;
  }
};

template <> struct formatter<convertible_to_cstring> {
  FMT_CONSTEXPR auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    return ctx.begin();
  }
  auto format(convertible_to_cstring, format_context& ctx) const
      -> decltype(ctx.out()) {
    auto out = ctx.out();
    *out++ = 'y';
    return out;
  }
};
FMT_END_NAMESPACE

TEST(format_test, formatter_overrides_implicit_conversion) {
  EXPECT_EQ(fmt::format("{}", convertible_to_int()), "x");
  EXPECT_EQ(fmt::format("{}", convertible_to_cstring()), "y");
}

struct ustring {
  using value_type = unsigned;

  auto find_first_of(value_type, size_t) const -> size_t;

  auto data() const -> const char*;
  auto size() const -> size_t;
};

FMT_BEGIN_NAMESPACE
template <> struct formatter<ustring> : formatter<std::string> {
  auto format(const ustring&, format_context& ctx) const
      -> decltype(ctx.out()) {
    return formatter<std::string>::format("ustring", ctx);
  }
};
FMT_END_NAMESPACE

TEST(format_test, ustring) {
  EXPECT_EQ(fmt::format("{}", ustring()), "ustring");
}

TEST(format_test, writer) {
  auto write_to_stdout = []() {
    auto w = fmt::writer(stdout);
    w.print("{}", 42);
  };
  EXPECT_WRITE(stdout, write_to_stdout(), "42");

#if FMT_USE_FCNTL
  auto pipe = fmt::pipe();
  auto write_end = pipe.write_end.fdopen("w");
  fmt::writer(write_end.get()).print("42");
  write_end.close();
  auto read_end = pipe.read_end.fdopen("r");
  int n = 0;
  int result = fscanf(read_end.get(), "%d", &n);
  (void)result;
  EXPECT_EQ(n, 42);
#endif

  auto s = fmt::string_buffer();
  fmt::writer(s).print("foo");
  EXPECT_EQ(s.str(), "foo");
}

#if FMT_USE_FCNTL && !defined(_WIN32)
TEST(format_test, invalid_glibc_buffer) {
  auto pipe = fmt::pipe();
  auto write_end = pipe.write_end.fdopen("w");
  auto file = write_end.get();

  // This results in _IO_write_ptr < _IO_write_end.
  fprintf(file, "111\n");
  setvbuf(file, nullptr, _IOLBF, 0);

  fmt::print(file, "------\n");
}
#endif  // FMT_USE_FCNTL

#if FMT_USE_BITINT
FMT_PRAGMA_CLANG(diagnostic ignored "-Wbit-int-extension")

TEST(format_test, bitint) {
  using fmt::detail::bitint;
  using fmt::detail::ubitint;

  EXPECT_EQ(fmt::format("{}", ubitint<3>(7)), "7");
  EXPECT_EQ(fmt::format("{}", bitint<7>()), "0");

  EXPECT_EQ(fmt::format("{}", ubitint<15>(31000)), "31000");
  EXPECT_EQ(fmt::format("{}", bitint<16>(INT16_MIN)), "-32768");
  EXPECT_EQ(fmt::format("{}", bitint<16>(INT16_MAX)), "32767");

  EXPECT_EQ(fmt::format("{}", ubitint<32>(4294967295)), "4294967295");

  EXPECT_EQ(fmt::format("{}", ubitint<47>(140737488355327ULL)),
            "140737488355327");
  EXPECT_EQ(fmt::format("{}", bitint<47>(-40737488355327LL)),
            "-40737488355327");

  // Check lvalues and const
  auto a = bitint<8>(0);
  auto b = ubitint<32>(4294967295);
  const auto c = bitint<7>(0);
  const auto d = ubitint<32>(4294967295);
  EXPECT_EQ(fmt::format("{}", a), "0");
  EXPECT_EQ(fmt::format("{}", b), "4294967295");
  EXPECT_EQ(fmt::format("{}", c), "0");
  EXPECT_EQ(fmt::format("{}", d), "4294967295");

  static_assert(fmt::is_formattable<bitint<64>, char>{}, "");
  static_assert(fmt::is_formattable<ubitint<64>, char>{}, "");
}
#endif

#ifdef __cpp_lib_byte
TEST(base_test, format_byte) {
  auto s = std::string();
  fmt::format_to(std::back_inserter(s), "{}", std::byte(42));
  EXPECT_EQ(s, "42");
}
#endif

// Only defined after the test case.
struct incomplete_type;
extern const incomplete_type& external_instance;

FMT_BEGIN_NAMESPACE
template <> struct formatter<incomplete_type> : formatter<int> {
  auto format(const incomplete_type& x, context& ctx) const -> appender;
};
FMT_END_NAMESPACE

TEST(incomplete_type_test, format) {
  EXPECT_EQ(fmt::format("{}", external_instance), "42");
}

struct incomplete_type {};
const incomplete_type& external_instance = {};

auto fmt::formatter<incomplete_type>::format(const incomplete_type&,
                                             fmt::context& ctx) const
    -> fmt::appender {
  return formatter<int>::format(42, ctx);
}
