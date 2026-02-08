// Formatting library for C++ - tests of formatters for standard library types
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#include "fmt/std.h"

#include <bitset>
#include <stdexcept>
#include <string>
#include <vector>

#include "fmt/os.h"       // fmt::system_category
#include "gtest-extra.h"  // StartsWith

#ifdef __cpp_lib_filesystem
TEST(std_test, path) {
  using std::filesystem::path;
  EXPECT_EQ(fmt::format("{}", path("/usr/bin")), "/usr/bin");

  // see #4303
  const path p = "/usr/bin";
  EXPECT_EQ(fmt::format("{}", p), "/usr/bin");

  EXPECT_EQ(fmt::format("{:?}", path("/usr/bin")), "\"/usr/bin\"");
  EXPECT_EQ(fmt::format("{:8}", path("foo")), "foo     ");

  EXPECT_EQ(fmt::format("{}", path("foo\"bar")), "foo\"bar");
  EXPECT_EQ(fmt::format("{:?}", path("foo\"bar")), "\"foo\\\"bar\"");

  EXPECT_EQ(fmt::format("{:g}", path("/usr/bin")), "/usr/bin");
#  ifdef _WIN32
  EXPECT_EQ(fmt::format("{}", path("C:\\foo")), "C:\\foo");
  EXPECT_EQ(fmt::format("{:g}", path("C:\\foo")), "C:/foo");

  EXPECT_EQ(fmt::format("{}", path(L"\x0428\x0447\x0443\x0447\x044B\x043D\x0448"
                                   L"\x0447\x044B\x043D\x0430")),
            "Шчучыншчына");
  EXPECT_EQ(fmt::format("{}", path(L"\xd800")), "�");
  EXPECT_EQ(fmt::format("{}", path(L"HEAD \xd800 TAIL")), "HEAD � TAIL");
  EXPECT_EQ(fmt::format("{}", path(L"HEAD \xD83D\xDE00 TAIL")),
            "HEAD \xF0\x9F\x98\x80 TAIL");
  EXPECT_EQ(fmt::format("{}", path(L"HEAD \xD83D\xD83D\xDE00 TAIL")),
            "HEAD �\xF0\x9F\x98\x80 TAIL");
  EXPECT_EQ(fmt::format("{:?}", path(L"\xd800")), "\"\\ud800\"");
#  endif
}

// Intentionally delayed include to test #4303
#  include "fmt/ranges.h"

// Test ambiguity problem described in #2954.
TEST(ranges_std_test, format_vector_path) {
  auto p = std::filesystem::path("foo/bar.txt");
  auto c = std::vector<std::string>{"abc", "def"};
  EXPECT_EQ(fmt::format("path={}, range={}", p, c),
            "path=foo/bar.txt, range=[\"abc\", \"def\"]");
}

// Test that path is not escaped twice in the debug mode.
TEST(ranges_std_test, format_quote_path) {
  auto vec =
      std::vector<std::filesystem::path>{"path1/file1.txt", "path2/file2.txt"};
  EXPECT_EQ(fmt::format("{}", vec),
            "[\"path1/file1.txt\", \"path2/file2.txt\"]");
#  ifdef __cpp_lib_optional
  auto o = std::optional<std::filesystem::path>("path/file.txt");
  EXPECT_EQ(fmt::format("{}", o), "optional(\"path/file.txt\")");
  EXPECT_EQ(fmt::format("{:?}", o), "optional(\"path/file.txt\")");
#  endif
}
#endif

TEST(std_test, thread_id) {
  EXPECT_FALSE(fmt::format("{}", std::this_thread::get_id()).empty());
}

TEST(std_test, complex) {
  using limits = std::numeric_limits<double>;
  EXPECT_EQ(fmt::format("{}", std::complex<double>(1, limits::quiet_NaN())),
            "(1+nan i)");
  EXPECT_EQ(fmt::format("{}", std::complex<double>(1, -limits::infinity())),
            "(1-inf i)");

  EXPECT_EQ(fmt::format("{}", std::complex<int>(1, 2)), "(1+2i)");

  EXPECT_EQ(fmt::format("{}", std::complex<double>(1, 2.2)), "(1+2.2i)");
  EXPECT_EQ(fmt::format("{}", std::complex<double>(1, -2.2)), "(1-2.2i)");
  EXPECT_EQ(fmt::format("{}", std::complex<double>(0, 2.2)), "2.2i");
  EXPECT_EQ(fmt::format("{}", std::complex<double>(0, -2.2)), "-2.2i");

  EXPECT_EQ(fmt::format("{:+}", std::complex<double>(0, 2.2)), "+2.2i");
  EXPECT_EQ(fmt::format("{:+}", std::complex<double>(0, -2.2)), "-2.2i");
  EXPECT_EQ(fmt::format("{:+}", std::complex<double>(1, -2.2)), "(+1-2.2i)");
  EXPECT_EQ(fmt::format("{:+}", std::complex<double>(1, 2.2)), "(+1+2.2i)");
  EXPECT_EQ(fmt::format("{: }", std::complex<double>(1, 2.2)), "( 1+2.2i)");
  EXPECT_EQ(fmt::format("{: }", std::complex<double>(1, -2.2)), "( 1-2.2i)");

  EXPECT_EQ(fmt::format("{:8}", std::complex<double>(1, 2)), "(1+2i)  ");
  EXPECT_EQ(fmt::format("{:-<8}", std::complex<double>(1, 2)), "(1+2i)--");

  EXPECT_EQ(fmt::format("{:>20.2f}", std::complex<double>(1, 2.2)),
            "        (1.00+2.20i)");
  EXPECT_EQ(fmt::format("{:<20.2f}", std::complex<double>(1, 2.2)),
            "(1.00+2.20i)        ");
  EXPECT_EQ(fmt::format("{:<20.2f}", std::complex<double>(1, -2.2)),
            "(1.00-2.20i)        ");
  EXPECT_EQ(fmt::format("{:<{}.{}f}", std::complex<double>(1, -2.2), 20, 2),
            "(1.00-2.20i)        ");
}

#ifdef __cpp_lib_source_location
TEST(std_test, source_location) {
  std::source_location loc = std::source_location::current();
  EXPECT_EQ(fmt::format("{}", loc),
            fmt::format("{}:{}:{}: {}", loc.file_name(), loc.line(),
                        loc.column(), loc.function_name()));
}
#endif

TEST(std_test, optional) {
#ifdef __cpp_lib_optional
  EXPECT_EQ(fmt::format("{}", std::optional<int>{}), "none");
  EXPECT_EQ(fmt::format("{}", std::pair{1, "second"}), "(1, \"second\")");
  EXPECT_EQ(fmt::format("{}", std::vector{std::optional{1}, std::optional{2},
                                          std::optional{3}}),
            "[optional(1), optional(2), optional(3)]");
  EXPECT_EQ(
      fmt::format("{}", std::optional<std::optional<const char*>>{{"nested"}}),
      "optional(optional(\"nested\"))");
  EXPECT_EQ(
      fmt::format("{:<{}}", std::optional{std::string{"left aligned"}}, 30),
      "optional(\"left aligned\"                )");
  EXPECT_EQ(
      fmt::format("{::d}", std::optional{std::vector{'h', 'e', 'l', 'l', 'o'}}),
      "optional([104, 101, 108, 108, 111])");
  EXPECT_EQ(fmt::format("{}", std::optional{std::string{"string"}}),
            "optional(\"string\")");
  EXPECT_EQ(fmt::format("{}", std::optional{'C'}), "optional(\'C\')");
  EXPECT_EQ(fmt::format("{:.{}f}", std::optional{3.14}, 1), "optional(3.1)");

  struct unformattable {};
  EXPECT_FALSE((fmt::is_formattable<unformattable>::value));
  EXPECT_FALSE((fmt::is_formattable<std::optional<unformattable>>::value));
  EXPECT_TRUE((fmt::is_formattable<std::optional<int>>::value));
#endif
}

TEST(std_test, expected) {
#ifdef __cpp_lib_expected
  EXPECT_EQ(fmt::format("{}", std::expected<void, int>{}), "expected()");
  EXPECT_EQ(fmt::format("{}", std::expected<int, int>{1}), "expected(1)");
  EXPECT_EQ(fmt::format("{}", std::expected<int, int>{std::unexpected(1)}),
            "unexpected(1)");
  EXPECT_EQ(fmt::format("{}", std::expected<std::string, int>{"test"}),
            "expected(\"test\")");
  EXPECT_EQ(fmt::format(
                "{}", std::expected<int, std::string>{std::unexpected("test")}),
            "unexpected(\"test\")");
  EXPECT_EQ(fmt::format("{}", std::expected<char, int>{'a'}), "expected('a')");
  EXPECT_EQ(fmt::format("{}", std::expected<int, char>{std::unexpected('a')}),
            "unexpected('a')");

  struct unformattable1 {};
  struct unformattable2 {};
  EXPECT_FALSE((fmt::is_formattable<unformattable1>::value));
  EXPECT_FALSE((fmt::is_formattable<unformattable2>::value));
  EXPECT_FALSE((fmt::is_formattable<
                std::expected<unformattable1, unformattable2>>::value));
  EXPECT_FALSE(
      (fmt::is_formattable<std::expected<unformattable1, int>>::value));
  EXPECT_FALSE(
      (fmt::is_formattable<std::expected<int, unformattable2>>::value));
  EXPECT_TRUE((fmt::is_formattable<std::expected<int, int>>::value));
  EXPECT_TRUE((fmt::is_formattable<std::expected<void, int>>::value));
#endif
}

namespace my_nso {
enum class my_number {
  one,
  two,
};
auto format_as(my_number number) -> fmt::string_view {
  return number == my_number::one ? "first" : "second";
}

class my_class {
 public:
  int av;

 private:
  friend auto format_as(const my_class& elm) -> std::string {
    return fmt::to_string(elm.av);
  }
};
}  // namespace my_nso
TEST(std_test, optional_format_as) {
#ifdef __cpp_lib_optional
  EXPECT_EQ(fmt::format("{}", std::optional<my_nso::my_number>{}), "none");
  EXPECT_EQ(fmt::format("{}", std::optional{my_nso::my_number::one}),
            "optional(\"first\")");
  EXPECT_EQ(fmt::format("{}", std::optional<my_nso::my_class>{}), "none");
  EXPECT_EQ(fmt::format("{}", std::optional{my_nso::my_class{7}}),
            "optional(\"7\")");
#endif
}

struct throws_on_move {
  throws_on_move() = default;

  [[noreturn]] throws_on_move(throws_on_move&&) {
    throw std::runtime_error("Thrown by throws_on_move");
  }

  throws_on_move(const throws_on_move&) = default;
};

namespace fmt {
template <> struct formatter<throws_on_move> : formatter<string_view> {
  auto format(const throws_on_move&, format_context& ctx) const
      -> decltype(ctx.out()) {
    string_view str("<throws_on_move>");
    return formatter<string_view>::format(str, ctx);
  }
};
}  // namespace fmt

TEST(std_test, variant) {
#ifdef __cpp_lib_variant
  EXPECT_EQ(fmt::format("{}", std::monostate{}), "monostate");
  using V0 = std::variant<int, float, std::string, char>;
  V0 v0(42);
  V0 v1(1.5f);
  V0 v2("hello");
  V0 v3('i');
  EXPECT_EQ(fmt::format("{}", v0), "variant(42)");
  EXPECT_EQ(fmt::format("{}", v1), "variant(1.5)");
  EXPECT_EQ(fmt::format("{}", v2), "variant(\"hello\")");
  EXPECT_EQ(fmt::format("{}", v3), "variant('i')");

  struct unformattable {};
  EXPECT_FALSE((fmt::is_formattable<unformattable>::value));
  EXPECT_FALSE((fmt::is_formattable<std::variant<unformattable>>::value));
  EXPECT_FALSE((fmt::is_formattable<std::variant<unformattable, int>>::value));
  EXPECT_FALSE((fmt::is_formattable<std::variant<int, unformattable>>::value));
  EXPECT_FALSE(
      (fmt::is_formattable<std::variant<unformattable, unformattable>>::value));
  EXPECT_TRUE((fmt::is_formattable<std::variant<int, float>>::value));

  using V1 = std::variant<std::monostate, std::string, std::string>;
  V1 v4{};
  V1 v5{std::in_place_index<1>, "yes, this is variant"};

  EXPECT_EQ(fmt::format("{}", v4), "variant(monostate)");
  EXPECT_EQ(fmt::format("{}", v5), "variant(\"yes, this is variant\")");

  volatile int i = 42;  // Test compile error before GCC 11 described in #3068.
  EXPECT_EQ(fmt::format("{}", i), "42");

  std::variant<std::monostate, throws_on_move> v6;

  try {
    throws_on_move thrower;
    v6.emplace<throws_on_move>(std::move(thrower));
  } catch (const std::runtime_error&) {
  }
  // v6 is now valueless by exception

  EXPECT_EQ(fmt::format("{}", v6), "variant(valueless by exception)");

#endif
}

TEST(std_test, error_code) {
  auto& generic = std::generic_category();
  EXPECT_EQ(fmt::format("{}", std::error_code(42, generic)), "generic:42");
  EXPECT_EQ(fmt::format("{:>12}", std::error_code(42, generic)),
            "  generic:42");
  EXPECT_EQ(fmt::format("{:12}", std::error_code(42, generic)), "generic:42  ");
  EXPECT_EQ(fmt::format("{}", std::error_code(42, fmt::system_category())),
            "system:42");
  EXPECT_EQ(fmt::format("{}", std::error_code(-42, fmt::system_category())),
            "system:-42");
  auto ec = std::make_error_code(std::errc::value_too_large);
  EXPECT_EQ(fmt::format("{:s}", ec), ec.message());
  EXPECT_EQ(fmt::format("{:?}", std::error_code(42, generic)),
            "\"generic:42\"");
}

template <typename Catch> void exception_test() {
  try {
    throw std::runtime_error("Test Exception");
  } catch (const Catch& ex) {
    EXPECT_EQ("Test Exception", fmt::format("{}", ex));
    EXPECT_EQ("std::runtime_error: Test Exception", fmt::format("{:t}", ex));
  }
}

namespace my_ns1 {
namespace my_ns2 {
struct my_exception : public std::exception {
 private:
  std::string msg;

 public:
  my_exception(const std::string& s) : msg(s) {}
  const char* what() const noexcept override;
};
const char* my_exception::what() const noexcept { return msg.c_str(); }
}  // namespace my_ns2
}  // namespace my_ns1

TEST(std_test, exception) {
  using testing::StartsWith;
  exception_test<std::exception>();
  exception_test<std::runtime_error>();

  try {
    using namespace my_ns1::my_ns2;
    throw my_exception("My Exception");
  } catch (const std::exception& ex) {
    EXPECT_EQ("my_ns1::my_ns2::my_exception: My Exception",
              fmt::format("{:t}", ex));
    EXPECT_EQ("My Exception", fmt::format("{:}", ex));
  }

  try {
    throw std::system_error(std::error_code(), "message");
  } catch (const std::system_error& ex) {
    EXPECT_THAT(fmt::format("{:t}", ex), StartsWith("std::system_error: "));
  }

#ifdef __cpp_lib_filesystem
  // Tests that the inline namespace is stripped out, e.g.
  // std::filesystem::__cxx11::* -> std::filesystem::*.
  try {
    throw std::filesystem::filesystem_error("message", std::error_code());
  } catch (const std::filesystem::filesystem_error& ex) {
    EXPECT_THAT(fmt::format("{:t}", ex),
                StartsWith("std::filesystem::filesystem_error: "));
  }
#endif
}

#if FMT_USE_RTTI
TEST(std_test, type_info) {
  EXPECT_EQ(fmt::format("{}", typeid(std::runtime_error)),
            "std::runtime_error");
}
#endif

TEST(std_test, format_bit_reference) {
  std::bitset<2> bs(1);
  EXPECT_EQ(fmt::format("{} {}", bs[0], bs[1]), "true false");
  std::vector<bool> v = {true, false};
  EXPECT_EQ(fmt::format("{} {}", v[0], v[1]), "true false");
}

TEST(std_test, format_const_bit_reference) {
  const std::bitset<2> bs(1);
  EXPECT_EQ(fmt::format("{} {}", bs[0], bs[1]), "true false");
  const std::vector<bool> v = {true, false};
  EXPECT_EQ(fmt::format("{} {}", v[0], v[1]), "true false");
}

TEST(std_test, format_bitset) {
  auto bs = std::bitset<6>(42);
  EXPECT_EQ(fmt::format("{}", bs), "101010");
  EXPECT_EQ(fmt::format("{:0>8}", bs), "00101010");
  EXPECT_EQ(fmt::format("{:-^12}", bs), "---101010---");
}

TEST(std_test, format_atomic) {
  std::atomic<bool> b(false);
  EXPECT_EQ(fmt::format("{}", b), "false");

  const std::atomic<bool> cb(true);
  EXPECT_EQ(fmt::format("{}", cb), "true");
}

#ifdef __cpp_lib_atomic_flag_test
TEST(std_test, format_atomic_flag) {
  std::atomic_flag f;
  (void)f.test_and_set();
  EXPECT_EQ(fmt::format("{}", f), "true");

  f.clear();
  const std::atomic_flag& cf = f;
  EXPECT_EQ(fmt::format("{}", cf), "false");
}
#endif  // __cpp_lib_atomic_flag_test

TEST(std_test, format_unique_ptr) {
  std::unique_ptr<int> up(new int(1));
  EXPECT_EQ(fmt::format("{}", fmt::ptr(up.get())),
            fmt::format("{}", fmt::ptr(up)));
  struct custom_deleter {
    void operator()(int* p) const { delete p; }
  };
  std::unique_ptr<int, custom_deleter> upcd(new int(1));
  EXPECT_EQ(fmt::format("{}", fmt::ptr(upcd.get())),
            fmt::format("{}", fmt::ptr(upcd)));
}

TEST(std_test, format_shared_ptr) {
  std::shared_ptr<int> sp(new int(1));
  EXPECT_EQ(fmt::format("{}", fmt::ptr(sp.get())),
            fmt::format("{}", fmt::ptr(sp)));
}

TEST(std_test, format_reference_wrapper) {
  int num = 35;
  EXPECT_EQ(fmt::to_string(std::cref(num)), "35");
  EXPECT_EQ(fmt::to_string(std::ref(num)), "35");
  EXPECT_EQ(fmt::format("{}", std::cref(num)), "35");
  EXPECT_EQ(fmt::format("{}", std::ref(num)), "35");
}

// Regression test for https://github.com/fmtlib/fmt/issues/4424.
struct type_with_format_as {};
int format_as(type_with_format_as) { return 20; }

TEST(std_test, format_reference_wrapper_with_format_as) {
  type_with_format_as t;
  EXPECT_EQ(fmt::to_string(std::cref(t)), "20");
  EXPECT_EQ(fmt::to_string(std::ref(t)), "20");
  EXPECT_EQ(fmt::format("{}", std::cref(t)), "20");
  EXPECT_EQ(fmt::format("{}", std::ref(t)), "20");
}

struct type_with_format_as_string {};
std::string format_as(type_with_format_as_string) { return "foo"; }

TEST(std_test, format_reference_wrapper_with_format_as_string) {
  type_with_format_as_string t;
  EXPECT_EQ(fmt::to_string(std::cref(t)), "foo");
  EXPECT_EQ(fmt::to_string(std::ref(t)), "foo");
  EXPECT_EQ(fmt::format("{}", std::cref(t)), "foo");
  EXPECT_EQ(fmt::format("{}", std::ref(t)), "foo");
}
