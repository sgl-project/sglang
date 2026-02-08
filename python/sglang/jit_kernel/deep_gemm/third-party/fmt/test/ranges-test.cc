// Formatting library for C++ - ranges tests
//
// Copyright (c) 2012 - present, Victor Zverovich and {fmt} contributors
// All rights reserved.
//
// For the license information refer to format.h.

#include "fmt/ranges.h"

#include <array>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#if FMT_CPLUSPLUS > 201703L && FMT_HAS_INCLUDE(<ranges>)
#  include <ranges>
#endif

#include "fmt/format.h"
#include "gtest/gtest.h"

#if !FMT_GCC_VERSION || FMT_GCC_VERSION >= 601
#  define FMT_RANGES_TEST_ENABLE_C_STYLE_ARRAY
#endif

#ifdef FMT_RANGES_TEST_ENABLE_C_STYLE_ARRAY
TEST(ranges_test, format_array) {
  int arr[] = {1, 2, 3, 5, 7, 11};
  EXPECT_EQ(fmt::format("{}", arr), "[1, 2, 3, 5, 7, 11]");
}

TEST(ranges_test, format_2d_array) {
  int arr[][2] = {{1, 2}, {3, 5}, {7, 11}};
  EXPECT_EQ(fmt::format("{}", arr), "[[1, 2], [3, 5], [7, 11]]");
}

TEST(ranges_test, format_array_of_literals) {
  const char* arr[] = {"1234", "abcd"};
  EXPECT_EQ(fmt::format("{}", arr), "[\"1234\", \"abcd\"]");
  EXPECT_EQ(fmt::format("{:n}", arr), "\"1234\", \"abcd\"");
  EXPECT_EQ(fmt::format("{:n:}", arr), "1234, abcd");
}
#endif  // FMT_RANGES_TEST_ENABLE_C_STYLE_ARRAY

struct unformattable {};

TEST(ranges_test, format_vector) {
  auto v = std::vector<int>{1, 2, 3, 5, 7, 11};
  EXPECT_EQ(fmt::format("{}", v), "[1, 2, 3, 5, 7, 11]");
  EXPECT_EQ(fmt::format("{::#x}", v), "[0x1, 0x2, 0x3, 0x5, 0x7, 0xb]");
  EXPECT_EQ(fmt::format("{:n:#x}", v), "0x1, 0x2, 0x3, 0x5, 0x7, 0xb");

  auto vc = std::vector<char>{'a', 'b', 'c'};
  auto vec = std::vector<char>{'a', '\n', '\t'};
  auto vvc = std::vector<std::vector<char>>{vc, vc};
  EXPECT_EQ(fmt::format("{}", vc), "['a', 'b', 'c']");
  EXPECT_EQ(fmt::format("{:s}", vc), "\"abc\"");
  EXPECT_EQ(fmt::format("{:?s}", vec), "\"a\\n\\t\"");
  EXPECT_EQ(fmt::format("{:s}", vec), "\"a\n\t\"");
  EXPECT_EQ(fmt::format("{::s}", vvc), "[\"abc\", \"abc\"]");
  EXPECT_EQ(fmt::format("{}", vvc), "[['a', 'b', 'c'], ['a', 'b', 'c']]");
  EXPECT_EQ(fmt::format("{:n}", vvc), "['a', 'b', 'c'], ['a', 'b', 'c']");
  EXPECT_EQ(fmt::format("{:n:n}", vvc), "'a', 'b', 'c', 'a', 'b', 'c'");
  EXPECT_EQ(fmt::format("{:n:n:}", vvc), "a, b, c, a, b, c");

  EXPECT_FALSE(fmt::is_formattable<unformattable>::value);
  EXPECT_FALSE(fmt::is_formattable<std::vector<unformattable>>::value);
}

TEST(ranges_test, format_nested_vector) {
  auto v = std::vector<std::vector<int>>{{1, 2}, {3, 5}, {7, 11}};
  EXPECT_EQ(fmt::format("{}", v), "[[1, 2], [3, 5], [7, 11]]");
  EXPECT_EQ(fmt::format("{:::#x}", v), "[[0x1, 0x2], [0x3, 0x5], [0x7, 0xb]]");
  EXPECT_EQ(fmt::format("{:n:n:#x}", v), "0x1, 0x2, 0x3, 0x5, 0x7, 0xb");
}

TEST(ranges_test, to_string_vector) {
  auto v = std::vector<std::string>{"a", "b", "c"};
  EXPECT_EQ(fmt::to_string(v), "[\"a\", \"b\", \"c\"]");
}

TEST(ranges_test, format_map) {
  auto m = std::map<std::string, int>{{"one", 1}, {"two", 2}};
  EXPECT_EQ(fmt::format("{}", m), "{\"one\": 1, \"two\": 2}");
  EXPECT_EQ(fmt::format("{:n}", m), "\"one\": 1, \"two\": 2");

  EXPECT_FALSE((fmt::is_formattable<std::map<int, unformattable>>::value));
}

struct test_map_value {};

FMT_BEGIN_NAMESPACE
template <> struct formatter<test_map_value> : formatter<string_view> {
  auto format(test_map_value, format_context& ctx) const
      -> format_context::iterator {
    return formatter<string_view>::format("foo", ctx);
  }
};

template <typename K>
struct formatter<std::pair<K, test_map_value>> : formatter<string_view> {
  auto format(std::pair<K, test_map_value>, format_context& ctx) const
      -> format_context::iterator {
    return ctx.out();
  }
};

template <typename K>
struct is_tuple_formattable<std::pair<K, test_map_value>, char>
    : std::false_type {};

FMT_END_NAMESPACE

TEST(ranges_test, format_map_custom_pair) {
  EXPECT_EQ(fmt::format("{}", std::map<int, test_map_value>{{42, {}}}),
            "{42: \"foo\"}");
}

TEST(ranges_test, format_set) {
  EXPECT_EQ(fmt::format("{}", std::set<std::string>{"one", "two"}),
            "{\"one\", \"two\"}");
}

// Models std::flat_set close enough to test if no ambiguous lookup of a
// formatter happens due to the flat_set type matching is_set and
// is_container_adaptor_like.
template <typename T> class flat_set {
 public:
  using key_type = T;
  using container_type = std::vector<T>;

  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;

  template <typename... Ts>
  explicit flat_set(Ts&&... args) : c{std::forward<Ts>(args)...} {}

  auto begin() -> iterator { return c.begin(); }
  auto end() -> iterator { return c.end(); }

  auto begin() const -> const_iterator { return c.begin(); }
  auto end() const -> const_iterator { return c.end(); }

 private:
  std::vector<T> c;
};

TEST(ranges_test, format_flat_set) {
  EXPECT_EQ(fmt::format("{}", flat_set<std::string>{"one", "two"}),
            "{\"one\", \"two\"}");
  EXPECT_FALSE(fmt::is_formattable<flat_set<unformattable>>::value);
}

namespace adl {
struct box {
  int value;
};

auto begin(const box& b) -> const int* { return &b.value; }
auto end(const box& b) -> const int* { return &b.value + 1; }
}  // namespace adl

TEST(ranges_test, format_adl_begin_end) {
  auto b = adl::box{42};
  EXPECT_EQ(fmt::format("{}", b), "[42]");
}

TEST(ranges_test, format_pair) {
  auto p = std::pair<int, float>(42, 1.5f);
  EXPECT_EQ(fmt::format("{}", p), "(42, 1.5)");
  EXPECT_EQ(fmt::format("{:}", p), "(42, 1.5)");
  EXPECT_EQ(fmt::format("{:n}", p), "421.5");
}

TEST(ranges_test, format_tuple) {
  auto t =
      std::tuple<int, float, std::string, char>(42, 1.5f, "this is tuple", 'i');
  EXPECT_EQ(fmt::format("{}", t), "(42, 1.5, \"this is tuple\", 'i')");
  EXPECT_EQ(fmt::format("{:n}", t), "421.5\"this is tuple\"'i'");

  EXPECT_EQ(fmt::format("{}", std::tuple<>()), "()");

  EXPECT_TRUE((fmt::is_formattable<std::tuple<>>::value));
  EXPECT_FALSE((fmt::is_formattable<std::tuple<unformattable>>::value));
  EXPECT_FALSE((fmt::is_formattable<std::tuple<unformattable, int>>::value));
  EXPECT_FALSE((fmt::is_formattable<std::tuple<int, unformattable>>::value));
  EXPECT_FALSE(
      (fmt::is_formattable<std::tuple<unformattable, unformattable>>::value));
  EXPECT_TRUE((fmt::is_formattable<std::tuple<int, float>>::value));
}

struct not_default_formattable {};
struct bad_format {};

FMT_BEGIN_NAMESPACE
template <> struct formatter<not_default_formattable> {
  auto parse(format_parse_context&) -> const char* { throw bad_format(); }
  auto format(not_default_formattable, format_context& ctx)
      -> format_context::iterator {
    return ctx.out();
  }
};
FMT_END_NAMESPACE

TEST(ranges_test, tuple_parse_calls_element_parse) {
  auto f = fmt::formatter<std::tuple<not_default_formattable>>();
  auto ctx = fmt::format_parse_context("");
  EXPECT_THROW(f.parse(ctx), bad_format);
}

struct tuple_like {
  int i;
  std::string str;

  template <size_t N>
  auto get() const noexcept -> fmt::enable_if_t<N == 0, int> {
    return i;
  }
  template <size_t N>
  auto get() const noexcept -> fmt::enable_if_t<N == 1, fmt::string_view> {
    return str;
  }
};

template <size_t N>
auto get(const tuple_like& t) noexcept -> decltype(t.get<N>()) {
  return t.get<N>();
}

// https://github.com/llvm/llvm-project/issues/39218
FMT_PRAGMA_CLANG(diagnostic ignored "-Wmismatched-tags")

namespace std {
template <>
struct tuple_size<tuple_like> : public std::integral_constant<size_t, 2> {};

template <size_t N> struct tuple_element<N, tuple_like> {
  using type = decltype(std::declval<tuple_like>().get<N>());
};
}  // namespace std

TEST(ranges_test, format_struct) {
  auto t = tuple_like{42, "foo"};
  EXPECT_EQ(fmt::format("{}", t), "(42, \"foo\")");
}

TEST(ranges_test, format_to) {
  char buf[10];
  auto end = fmt::format_to(buf, "{}", std::vector<int>{1, 2, 3});
  *end = '\0';
  EXPECT_STREQ(buf, "[1, 2, 3]");
}

template <typename Char> struct path_like {
  auto begin() const -> const path_like*;
  auto end() const -> const path_like*;

  operator std::basic_string<Char>() const;
};

TEST(ranges_test, disabled_range_formatting_of_path) {
  // Range formatting of path is disabled because of infinite recursion
  // (path element is itself a path).
  EXPECT_EQ((fmt::range_format_kind<path_like<char>, char>::value),
            fmt::range_format::disabled);
  EXPECT_EQ((fmt::range_format_kind<path_like<wchar_t>, char>::value),
            fmt::range_format::disabled);
}

struct vector_string : std::vector<char> {
  using base = std::vector<char>;
  using base::base;
};
struct vector_debug_string : std::vector<char> {
  using base = std::vector<char>;
  using base::base;
};
FMT_BEGIN_NAMESPACE
template <>
struct range_format_kind<vector_string, char>
    : std::integral_constant<range_format, range_format::string> {};
template <>
struct range_format_kind<vector_debug_string, char>
    : std::integral_constant<range_format, range_format::debug_string> {};
FMT_END_NAMESPACE

TEST(ranges_test, range_format_string) {
  const vector_string v{'f', 'o', 'o'};
  EXPECT_EQ(fmt::format("{}", v), "foo");
}

TEST(ranges_test, range_format_debug_string) {
  const vector_debug_string v{'f', 'o', 'o'};
  EXPECT_EQ(fmt::format("{}", v), "\"foo\"");
}

// A range that provides non-const only begin()/end() to test fmt::join
// handles that.
//
// Some ranges (e.g. those produced by range-v3's views::filter()) can cache
// information during iteration so they only provide non-const begin()/end().
template <typename T> class non_const_only_range {
 private:
  std::vector<T> vec;

 public:
  using const_iterator = typename ::std::vector<T>::const_iterator;

  template <typename... Args>
  explicit non_const_only_range(Args&&... args)
      : vec(std::forward<Args>(args)...) {}

  auto begin() -> const_iterator { return vec.begin(); }
  auto end() -> const_iterator { return vec.end(); }
};

template <typename T> class noncopyable_range {
 private:
  std::vector<T> vec;

 public:
  using iterator = typename ::std::vector<T>::iterator;

  template <typename... Args>
  explicit noncopyable_range(Args&&... args)
      : vec(std::forward<Args>(args)...) {}

  noncopyable_range(const noncopyable_range&) = delete;
  noncopyable_range(noncopyable_range&) = delete;

  auto begin() -> iterator { return vec.begin(); }
  auto end() -> iterator { return vec.end(); }
};

TEST(ranges_test, range) {
  auto&& w = noncopyable_range<int>(3u, 0);
  EXPECT_EQ(fmt::format("{}", w), "[0, 0, 0]");
  EXPECT_EQ(fmt::format("{}", noncopyable_range<int>(3u, 0)), "[0, 0, 0]");

  auto x = non_const_only_range<int>(3u, 0);
  EXPECT_EQ(fmt::format("{}", x), "[0, 0, 0]");
  EXPECT_EQ(fmt::format("{}", non_const_only_range<int>(3u, 0)), "[0, 0, 0]");

  auto y = std::vector<int>(3u, 0);
  EXPECT_EQ(fmt::format("{}", y), "[0, 0, 0]");
  EXPECT_EQ(fmt::format("{}", std::vector<int>(3u, 0)), "[0, 0, 0]");

  const auto z = std::vector<int>(3u, 0);
  EXPECT_EQ(fmt::format("{}", z), "[0, 0, 0]");
}

enum test_enum { foo, bar };
auto format_as(test_enum e) -> int { return e; }

TEST(ranges_test, enum_range) {
  auto v = std::vector<test_enum>{test_enum::foo};
  EXPECT_EQ(fmt::format("{}", v), "[0]");
}

#if !FMT_MSC_VERSION
TEST(ranges_test, unformattable_range) {
  EXPECT_FALSE((fmt::is_formattable<std::vector<unformattable>, char>::value));
}
#endif

TEST(ranges_test, join) {
  using fmt::join;
  int v1[3] = {1, 2, 3};
  auto v2 = std::vector<float>();
  v2.push_back(1.2f);
  v2.push_back(3.4f);
  void* v3[2] = {&v1[0], &v1[1]};

  EXPECT_EQ(fmt::format("({})", join(v1, v1 + 3, ", ")), "(1, 2, 3)");
  EXPECT_EQ(fmt::format("({})", join(v1, v1 + 1, ", ")), "(1)");
  EXPECT_EQ(fmt::format("({})", join(v1, v1, ", ")), "()");
  EXPECT_EQ(fmt::format("({:03})", join(v1, v1 + 3, ", ")), "(001, 002, 003)");
  EXPECT_EQ("(+01.20, +03.40)",
            fmt::format("({:+06.2f})", join(v2.begin(), v2.end(), ", ")));

  EXPECT_EQ(fmt::format("{0:{1}}", join(v1, v1 + 3, ", "), 1), "1, 2, 3");

  EXPECT_EQ(fmt::format("{}, {}", v3[0], v3[1]),
            fmt::format("{}", join(v3, v3 + 2, ", ")));

  EXPECT_EQ(fmt::format("({})", join(v1, ", ")), "(1, 2, 3)");
  EXPECT_EQ(fmt::format("({:+06.2f})", join(v2, ", ")), "(+01.20, +03.40)");

  auto v4 = std::vector<test_enum>{foo, bar, foo};
  EXPECT_EQ(fmt::format("{}", join(v4, " ")), "0 1 0");
}

#ifdef __cpp_lib_byte
TEST(ranges_test, join_bytes) {
  auto v = std::vector<std::byte>{std::byte(1), std::byte(2), std::byte(3)};
  EXPECT_EQ(fmt::format("{}", fmt::join(v, ", ")), "1, 2, 3");
}
#endif

TEST(ranges_test, join_tuple) {
  // Value tuple args.
  auto t1 = std::tuple<char, int, float>('a', 1, 2.0f);
  EXPECT_EQ(fmt::format("({})", fmt::join(t1, ", ")), "(a, 1, 2)");

  // Testing lvalue tuple args.
  int x = 4;
  auto t2 = std::tuple<char, int&>('b', x);
  EXPECT_EQ(fmt::format("{}", fmt::join(t2, " + ")), "b + 4");

  // Empty tuple.
  auto t3 = std::tuple<>();
  EXPECT_EQ(fmt::format("{}", fmt::join(t3, "|")), "");

  // Single element tuple.
  auto t4 = std::tuple<float>(4.0f);
  EXPECT_EQ(fmt::format("{}", fmt::join(t4, "/")), "4");

  // Tuple-like.
  auto t5 = tuple_like{42, "foo"};
  EXPECT_EQ(fmt::format("{}", fmt::join(t5, ", ")), "42, foo");

#if FMT_TUPLE_JOIN_SPECIFIERS
  // Specs applied to each element.
  auto t5 = std::tuple<int, int, long>(-3, 100, 1);
  EXPECT_EQ(fmt::format("{:+03}", fmt::join(t5, ", ")), "-03, +100, +01");

  auto t6 = std::tuple<float, double, long double>(3, 3.14, 3.1415);
  EXPECT_EQ(fmt::format("{:5.5f}", fmt::join(t6, ", ")),
            "3.00000, 3.14000, 3.14150");

  // Testing lvalue tuple args.
  int y = -1;
  auto t7 = std::tuple<int, int&, const int&>(3, y, y);
  EXPECT_EQ(fmt::format("{:03}", fmt::join(t7, ", ")), "003, -01, -01");
#endif
}

TEST(ranges_test, join_initializer_list) {
  EXPECT_EQ(fmt::format("{}", fmt::join({1, 2, 3}, ", ")), "1, 2, 3");
  EXPECT_EQ(fmt::format("{}", fmt::join({"fmt", "rocks", "!"}, " ")),
            "fmt rocks !");
}

struct zstring_sentinel {};

bool operator==(const char* p, zstring_sentinel) { return *p == '\0'; }
bool operator!=(const char* p, zstring_sentinel) { return *p != '\0'; }

struct zstring {
  const char* p;
  auto begin() const -> const char* { return p; }
  auto end() const -> zstring_sentinel { return {}; }
};

#ifdef __cpp_lib_ranges
struct cpp20_only_range {
  struct iterator {
    int val = 0;

    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using iterator_concept = std::input_iterator_tag;

    iterator() = default;
    iterator(int i) : val(i) {}
    auto operator*() const -> int { return val; }
    auto operator++() -> iterator& {
      ++val;
      return *this;
    }
    void operator++(int) { ++*this; }
    auto operator==(const iterator& rhs) const -> bool {
      return val == rhs.val;
    }
  };

  int lo;
  int hi;

  auto begin() const -> iterator { return iterator(lo); }
  auto end() const -> iterator { return iterator(hi); }
};

static_assert(std::input_iterator<cpp20_only_range::iterator>);
#endif

TEST(ranges_test, join_sentinel) {
  auto hello = zstring{"hello"};
  EXPECT_EQ(fmt::format("{}", hello), "['h', 'e', 'l', 'l', 'o']");
  EXPECT_EQ(fmt::format("{::}", hello), "[h, e, l, l, o]");
  EXPECT_EQ(fmt::format("{}", fmt::join(hello, "_")), "h_e_l_l_o");
}

TEST(ranges_test, join_range) {
  auto&& w = noncopyable_range<int>(3u, 0);
  EXPECT_EQ(fmt::format("{}", fmt::join(w, ",")), "0,0,0");
  EXPECT_EQ(fmt::format("{}", fmt::join(noncopyable_range<int>(3u, 0), ",")),
            "0,0,0");

  auto x = non_const_only_range<int>(3u, 0);
  EXPECT_EQ(fmt::format("{}", fmt::join(x, ",")), "0,0,0");
  EXPECT_EQ(fmt::format("{}", fmt::join(non_const_only_range<int>(3u, 0), ",")),
            "0,0,0");

  auto y = std::vector<int>(3u, 0);
  EXPECT_EQ(fmt::format("{}", fmt::join(y, ",")), "0,0,0");
  EXPECT_EQ(fmt::format("{}", fmt::join(std::vector<int>(3u, 0), ",")),
            "0,0,0");

  const auto z = std::vector<int>(3u, 0);
  EXPECT_EQ(fmt::format("{}", fmt::join(z, ",")), "0,0,0");

#ifdef __cpp_lib_ranges
  EXPECT_EQ(fmt::format("{}", cpp20_only_range{.lo = 0, .hi = 5}),
            "[0, 1, 2, 3, 4]");
  EXPECT_EQ(
      fmt::format("{}", fmt::join(cpp20_only_range{.lo = 0, .hi = 5}, ",")),
      "0,1,2,3,4");
#endif
}

namespace adl {
struct vec {
  int n[2] = {42, 43};
};

auto begin(const vec& v) -> const int* { return v.n; }
auto end(const vec& v) -> const int* { return v.n + 2; }
}  // namespace adl

TEST(ranges_test, format_join_adl_begin_end) {
  EXPECT_EQ(fmt::format("{}", fmt::join(adl::vec(), "/")), "42/43");
}

#if defined(__cpp_lib_ranges) && __cpp_lib_ranges >= 202207L
TEST(ranges_test, nested_ranges) {
  auto l = std::list{1, 2, 3};
  auto r = std::views::iota(0, 3) | std::views::transform([&l](auto i) {
             return std::views::take(std::ranges::subrange(l), i);
           }) |
           std::views::transform(std::views::reverse);
  EXPECT_EQ(fmt::format("{}", r), "[[], [1], [2, 1]]");
}
#endif

TEST(ranges_test, is_printable) {
  using fmt::detail::is_printable;
  EXPECT_TRUE(is_printable(0x0323));
  EXPECT_FALSE(is_printable(0x0378));
  EXPECT_FALSE(is_printable(0x110000));
}

TEST(ranges_test, escape) {
  using vec = std::vector<std::string>;
  EXPECT_EQ(fmt::format("{}", vec{"\n\r\t\"\\"}), "[\"\\n\\r\\t\\\"\\\\\"]");
  EXPECT_EQ(fmt::format("{}", vec{"\x07"}), "[\"\\x07\"]");
  EXPECT_EQ(fmt::format("{}", vec{"\x7f"}), "[\"\\x7f\"]");
  EXPECT_EQ(fmt::format("{}", vec{"n\xcc\x83"}), "[\"n\xcc\x83\"]");

  if (fmt::detail::use_utf8) {
    EXPECT_EQ(fmt::format("{}", vec{"\xcd\xb8"}), "[\"\\u0378\"]");
    // Unassigned Unicode code points.
    EXPECT_EQ(fmt::format("{}", vec{"\xf0\xaa\x9b\x9e"}), "[\"\\U0002a6de\"]");
    // Broken utf-8.
    EXPECT_EQ(fmt::format("{}", vec{"\xf4\x8f\xbf\xc0"}),
              "[\"\\xf4\\x8f\\xbf\\xc0\"]");
    EXPECT_EQ(fmt::format("{}", vec{"\xf0\x28"}), "[\"\\xf0(\"]");
    EXPECT_EQ(fmt::format("{}", vec{"\xe1\x28"}), "[\"\\xe1(\"]");
    EXPECT_EQ(fmt::format("{}", vec{std::string("\xf0\x28\0\0anything", 12)}),
              "[\"\\xf0(\\x00\\x00anything\"]");

    // Correct utf-8.
    EXPECT_EQ(fmt::format("{}", vec{"🦄"}), "[\"🦄\"]");
  }

  EXPECT_EQ(fmt::format("{}", std::vector<std::vector<char>>{{'x'}}),
            "[['x']]");

// Disabled due to a clang 17 bug: https://github.com/fmtlib/fmt/issues/4144.
#if FMT_CLANG_VERSION >= 1800
  EXPECT_EQ(fmt::format("{}", std::tuple<std::vector<char>>{{'x'}}), "(['x'])");
#endif
}

template <typename R> struct fmt_ref_view {
  R* r;

  auto begin() const -> decltype(r->begin()) { return r->begin(); }
  auto end() const -> decltype(r->end()) { return r->end(); }
};

TEST(ranges_test, range_of_range_of_mixed_const) {
  auto v = std::vector<std::vector<int>>{{1, 2, 3}, {4, 5}};
  EXPECT_EQ(fmt::format("{}", v), "[[1, 2, 3], [4, 5]]");

  auto r = fmt_ref_view<decltype(v)>{&v};
  EXPECT_EQ(fmt::format("{}", r), "[[1, 2, 3], [4, 5]]");
}

TEST(ranges_test, vector_char) {
  EXPECT_EQ(fmt::format("{}", std::vector<char>{'a', 'b'}), "['a', 'b']");
}

TEST(ranges_test, container_adaptor) {
  {
    using fmt::detail::is_container_adaptor_like;
    using T = std::nullptr_t;
    static_assert(is_container_adaptor_like<std::stack<T>>::value, "");
    static_assert(is_container_adaptor_like<std::queue<T>>::value, "");
    static_assert(is_container_adaptor_like<std::priority_queue<T>>::value, "");
    static_assert(!is_container_adaptor_like<std::vector<T>>::value, "");
  }

  {
    auto s = std::stack<int>();
    s.push(1);
    s.push(2);
    EXPECT_EQ(fmt::format("{}", s), "[1, 2]");
    EXPECT_EQ(fmt::format("{}", const_cast<const decltype(s)&>(s)), "[1, 2]");
  }

  {
    auto q = std::queue<int>();
    q.push(1);
    q.push(2);
    EXPECT_EQ(fmt::format("{}", q), "[1, 2]");
  }

  {
    auto q = std::priority_queue<int>();
    q.push(3);
    q.push(1);
    q.push(2);
    q.push(4);
    EXPECT_EQ(fmt::format("{}", q), "[4, 3, 2, 1]");
  }

  {
    auto s = std::stack<char, std::string>();
    s.push('a');
    s.push('b');
    // See https://cplusplus.github.io/LWG/issue3881.
    EXPECT_EQ(fmt::format("{}", s), "['a', 'b']");
  }

  {
    struct my_container_adaptor {
      using value_type = int;
      using container_type = std::vector<value_type>;
      void push(const value_type& v) { c.push_back(v); }

     protected:
      container_type c;
    };

    auto m = my_container_adaptor();
    m.push(1);
    m.push(2);
    EXPECT_EQ(fmt::format("{}", m), "[1, 2]");
  }

  EXPECT_FALSE(fmt::is_formattable<std::stack<unformattable>>::value);
}

struct tieable {
  int a = 3;
  double b = 0.42;
};

auto format_as(const tieable& t) -> std::tuple<int, double> {
  return std::tie(t.a, t.b);
}

TEST(ranges_test, format_as_tie) {
  EXPECT_EQ(fmt::format("{}", tieable()), "(3, 0.42)");
}

struct lvalue_qualified_begin_end {
  int arr[5] = {1, 2, 3, 4, 5};

  auto begin() & -> const int* { return arr; }
  auto end() & -> const int* { return arr + 5; }
};

TEST(ranges_test, lvalue_qualified_begin_end) {
  EXPECT_EQ(fmt::format("{}", lvalue_qualified_begin_end{}), "[1, 2, 3, 4, 5]");
}

#if !defined(__cpp_lib_ranges) || __cpp_lib_ranges <= 202106L
#  define ENABLE_STD_VIEWS_TESTS 0
#elif FMT_CLANG_VERSION
#  if FMT_CLANG_VERSION > 1500
#    define ENABLE_STD_VIEWS_TESTS 1
#  else
#    define ENABLE_STD_VIEWS_TESTS 0
#  endif
#else
#  define ENABLE_STD_VIEWS_TESTS 1
#endif

#if ENABLE_STD_VIEWS_TESTS
TEST(ranges_test, input_range_join) {
  auto iss = std::istringstream("1 2 3 4 5");
  auto view = std::views::istream<std::string>(iss);
  EXPECT_EQ("1, 2, 3, 4, 5",
            fmt::format("{}", fmt::join(view.begin(), view.end(), ", ")));
}

TEST(ranges_test, input_range_join_overload) {
  auto iss = std::istringstream("1 2 3 4 5");
  EXPECT_EQ(
      "1.2.3.4.5",
      fmt::format("{}", fmt::join(std::views::istream<std::string>(iss), ".")));
}

namespace views_filter_view_test {
struct codec_mask {
  static constexpr auto codecs = std::array{0, 1, 2, 3};
  int except = 0;
};

auto format_as(codec_mask mask) {
  // Careful not to capture param by reference here, it will dangle.
  return codec_mask::codecs |
         std::views::filter([mask](auto c) { return c != mask.except; });
}
}  // namespace views_filter_view_test

TEST(ranges_test, format_as_with_ranges_mutable_begin_end) {
  using namespace views_filter_view_test;
  {
    auto make_filter_view = []() {
      return codec_mask::codecs |
             std::views::filter([](auto c) { return c != 2; });
    };
    auto r = make_filter_view();
    EXPECT_EQ("[0, 1, 3]", fmt::format("{}", r));
    EXPECT_EQ("[0, 1, 3]", fmt::format("{}", make_filter_view()));
  }

  {
    auto mask = codec_mask{2};
    const auto const_mask = codec_mask{2};

    EXPECT_EQ("[0, 1, 3]", fmt::format("{}", mask));
    EXPECT_EQ("[0, 1, 3]", fmt::format("{}", const_mask));
    EXPECT_EQ("[0, 1, 3]", fmt::format("{}", codec_mask{2}));
  }
}

#endif

TEST(ranges_test, std_istream_iterator_join) {
  auto&& iss = std::istringstream("1 2 3 4 5");
  auto first = std::istream_iterator<int>(iss);
  auto last = std::istream_iterator<int>();
  EXPECT_EQ("1, 2, 3, 4, 5", fmt::format("{}", fmt::join(first, last, ", ")));
}

// Mirrors C++20 std::ranges::basic_istream_view::iterator.
struct noncopyable_istream_iterator : std::istream_iterator<int> {
  using base = std::istream_iterator<int>;
  explicit noncopyable_istream_iterator(std::istringstream& iss) : base{iss} {}
  noncopyable_istream_iterator(const noncopyable_istream_iterator&) = delete;
  noncopyable_istream_iterator(noncopyable_istream_iterator&&) = default;
};
static_assert(!std::is_copy_constructible<noncopyable_istream_iterator>::value,
              "");

TEST(ranges_test, movable_only_istream_iter_join) {
  auto&& iss = std::istringstream("1 2 3 4 5");
  auto first = noncopyable_istream_iterator(iss);
  auto last = std::istream_iterator<int>();
  EXPECT_EQ("1, 2, 3, 4, 5",
            fmt::format("{}", fmt::join(std::move(first), last, ", ")));
}

struct movable_iter_range {
  std::istringstream iss{"1 2 3 4 5"};
  noncopyable_istream_iterator begin() {
    return noncopyable_istream_iterator{iss};
  }
  std::istream_iterator<int> end() { return {}; }
};

TEST(ranges_test, movable_only_istream_iter_join2) {
  EXPECT_EQ("[1, 2, 3, 4, 5]", fmt::format("{}", movable_iter_range{}));
}

struct not_range {
  void begin() const {}
  void end() const {}
};
static_assert(!fmt::is_formattable<not_range>{}, "");
