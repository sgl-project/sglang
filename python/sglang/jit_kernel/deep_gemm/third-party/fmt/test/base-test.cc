// Formatting library for C++ - core tests
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

// Turn assertion failures into exceptions for testing.
// clang-format off
#include "test-assert.h"
// clang-format on

#include "fmt/base.h"

#include <limits.h>  // INT_MAX
#include <string.h>  // strlen

#include <functional>   // std::equal_to
#include <iterator>     // std::back_insert_iterator, std::distance
#include <limits>       // std::numeric_limits
#include <string>       // std::string
#include <type_traits>  // std::is_same

#include "gmock/gmock.h"

#ifdef FMT_FORMAT_H_
#  error base-test includes format.h
#endif

using testing::_;
using testing::Invoke;
using testing::Return;

auto copy(fmt::string_view s, fmt::appender out) -> fmt::appender {
  for (char c : s) *out++ = c;
  return out;
}

TEST(string_view_test, value_type) {
  static_assert(std::is_same<fmt::string_view::value_type, char>::value, "");
}

TEST(string_view_test, ctor) {
  EXPECT_STREQ(fmt::string_view("abc").data(), "abc");
  EXPECT_EQ(fmt::string_view("abc").size(), 3u);

  EXPECT_STREQ(fmt::string_view(std::string("defg")).data(), "defg");
  EXPECT_EQ(fmt::string_view(std::string("defg")).size(), 4u);
}

TEST(string_view_test, length) {
  // Test that string_view::size() returns string length, not buffer size.
  char str[100] = "some string";
  EXPECT_EQ(fmt::string_view(str).size(), strlen(str));
  EXPECT_LT(strlen(str), sizeof(str));
}

// Check string_view's comparison operator.
template <template <typename> class Op> void check_op() {
  const char* inputs[] = {"foo", "fop", "fo"};
  size_t num_inputs = sizeof(inputs) / sizeof(*inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    for (size_t j = 0; j < num_inputs; ++j) {
      fmt::string_view lhs(inputs[i]), rhs(inputs[j]);
      EXPECT_EQ(Op<int>()(lhs.compare(rhs), 0),
                Op<fmt::string_view>()(lhs, rhs));
    }
  }
}

TEST(string_view_test, compare) {
  using fmt::string_view;

  EXPECT_EQ(string_view("foo").compare(string_view("foo")), 0);
  EXPECT_GT(string_view("fop").compare(string_view("foo")), 0);
  EXPECT_LT(string_view("foo").compare(string_view("fop")), 0);
  EXPECT_GT(string_view("foo").compare(string_view("fo")), 0);
  EXPECT_LT(string_view("fo").compare(string_view("foo")), 0);

  EXPECT_TRUE(string_view("foo").starts_with('f'));
  EXPECT_FALSE(string_view("foo").starts_with('o'));
  EXPECT_FALSE(string_view().starts_with('o'));

  EXPECT_TRUE(string_view("foo").starts_with("fo"));
  EXPECT_TRUE(string_view("foo").starts_with("foo"));
  EXPECT_FALSE(string_view("foo").starts_with("fooo"));
  EXPECT_FALSE(string_view().starts_with("fooo"));

  check_op<std::equal_to>();
  check_op<std::not_equal_to>();
  check_op<std::less>();
  check_op<std::less_equal>();
  check_op<std::greater>();
  check_op<std::greater_equal>();
}

#if FMT_USE_CONSTEVAL
TEST(string_view_test, from_constexpr_fixed_string) {
  constexpr int size = 4;

  struct fixed_string {
    char data[size] = {};

    constexpr fixed_string(const char (&m)[size]) {
      for (size_t i = 0; i != size; ++i) data[i] = m[i];
    }
  };

  static constexpr auto fs = fixed_string("foo");
  static constexpr auto sv = fmt::string_view(fs.data);
  EXPECT_EQ(sv, "foo");
}
#endif  // FMT_USE_CONSTEVAL

TEST(buffer_test, noncopyable) {
  EXPECT_FALSE(std::is_copy_constructible<fmt::detail::buffer<char>>::value);
  EXPECT_FALSE(std::is_copy_assignable<fmt::detail::buffer<char>>::value);
}

TEST(buffer_test, nonmoveable) {
  EXPECT_FALSE(std::is_move_constructible<fmt::detail::buffer<char>>::value);
  EXPECT_FALSE(std::is_move_assignable<fmt::detail::buffer<char>>::value);
}

TEST(buffer_test, indestructible) {
  static_assert(!std::is_destructible<fmt::detail::buffer<int>>(),
                "buffer's destructor is protected");
}

template <typename T> struct mock_buffer final : fmt::detail::buffer<T> {
  MOCK_METHOD(size_t, do_grow, (size_t));

  static void grow(fmt::detail::buffer<T>& buf, size_t capacity) {
    auto& self = static_cast<mock_buffer&>(buf);
    self.set(buf.data(), self.do_grow(capacity));
  }

  mock_buffer(T* data = nullptr, size_t buf_capacity = 0)
      : fmt::detail::buffer<T>(grow) {
    this->set(data, buf_capacity);
    ON_CALL(*this, do_grow(_)).WillByDefault(Invoke([](size_t capacity) {
      return capacity;
    }));
  }
};

TEST(buffer_test, ctor) {
  {
    mock_buffer<int> buffer;
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 0u);
    EXPECT_EQ(buffer.capacity(), 0u);
  }
  {
    int data;
    mock_buffer<int> buffer(&data);
    EXPECT_EQ(&buffer[0], &data);
    EXPECT_EQ(buffer.size(), 0u);
    EXPECT_EQ(buffer.capacity(), 0u);
  }
  {
    int data;
    size_t capacity = std::numeric_limits<size_t>::max();
    mock_buffer<int> buffer(&data, capacity);
    EXPECT_EQ(&buffer[0], &data);
    EXPECT_EQ(buffer.size(), 0u);
    EXPECT_EQ(buffer.capacity(), capacity);
  }
}

TEST(buffer_test, access) {
  char data[10];
  mock_buffer<char> buffer(data, sizeof(data));
  buffer[0] = 11;
  EXPECT_EQ(buffer[0], 11);
  buffer[3] = 42;
  EXPECT_EQ(*(&buffer[0] + 3), 42);
  const fmt::detail::buffer<char>& const_buffer = buffer;
  EXPECT_EQ(const_buffer[3], 42);
}

TEST(buffer_test, try_resize) {
  char data[123];
  mock_buffer<char> buffer(data, sizeof(data));
  buffer[10] = 42;
  EXPECT_EQ(buffer[10], 42);
  buffer.try_resize(20);
  EXPECT_EQ(buffer.size(), 20u);
  EXPECT_EQ(buffer.capacity(), 123u);
  EXPECT_EQ(buffer[10], 42);
  buffer.try_resize(5);
  EXPECT_EQ(buffer.size(), 5u);
  EXPECT_EQ(buffer.capacity(), 123u);
  EXPECT_EQ(buffer[10], 42);
  // Check if try_resize calls grow.
  EXPECT_CALL(buffer, do_grow(124));
  buffer.try_resize(124);
  EXPECT_CALL(buffer, do_grow(200));
  buffer.try_resize(200);
}

TEST(buffer_test, try_resize_partial) {
  char data[10];
  mock_buffer<char> buffer(data, sizeof(data));
  EXPECT_CALL(buffer, do_grow(20)).WillOnce(Return(15));
  buffer.try_resize(20);
  EXPECT_EQ(buffer.capacity(), 15);
  EXPECT_EQ(buffer.size(), 15);
}

TEST(buffer_test, clear) {
  mock_buffer<char> buffer;
  EXPECT_CALL(buffer, do_grow(20));
  buffer.try_resize(20);
  buffer.try_resize(0);
  EXPECT_EQ(buffer.size(), 0u);
  EXPECT_EQ(buffer.capacity(), 20u);
}

TEST(buffer_test, append) {
  char data[15];
  mock_buffer<char> buffer(data, 10);
  auto test = "test";
  buffer.append(test, test + 5);
  EXPECT_STREQ(&buffer[0], test);
  EXPECT_EQ(buffer.size(), 5u);
  buffer.try_resize(10);
  EXPECT_CALL(buffer, do_grow(12));
  buffer.append(test, test + 2);
  EXPECT_EQ(buffer[10], 't');
  EXPECT_EQ(buffer[11], 'e');
  EXPECT_EQ(buffer.size(), 12u);
}

TEST(buffer_test, append_partial) {
  char data[10];
  mock_buffer<char> buffer(data, sizeof(data));
  testing::InSequence seq;
  EXPECT_CALL(buffer, do_grow(15)).WillOnce(Return(10));
  EXPECT_CALL(buffer, do_grow(15)).WillOnce(Invoke([&buffer](size_t) {
    EXPECT_EQ(fmt::string_view(buffer.data(), buffer.size()), "0123456789");
    buffer.clear();
    return 10;
  }));
  auto test = "0123456789abcde";
  buffer.append(test, test + 15);
}

TEST(buffer_test, append_allocates_enough_storage) {
  char data[19];
  mock_buffer<char> buffer(data, 10);
  auto test = "abcdefgh";
  buffer.try_resize(10);
  EXPECT_CALL(buffer, do_grow(19));
  buffer.append(test, test + 9);
}

TEST(base_test, is_locking) {
  EXPECT_FALSE(fmt::detail::is_locking<const char(&)[3]>());
}

TEST(base_test, is_output_iterator) {
  EXPECT_TRUE((fmt::detail::is_output_iterator<char*, char>::value));
  EXPECT_FALSE((fmt::detail::is_output_iterator<const char*, char>::value));
  EXPECT_FALSE((fmt::detail::is_output_iterator<std::string, char>::value));
  EXPECT_TRUE(
      (fmt::detail::is_output_iterator<std::back_insert_iterator<std::string>,
                                       char>::value));
  EXPECT_TRUE(
      (fmt::detail::is_output_iterator<std::string::iterator, char>::value));
  EXPECT_FALSE((fmt::detail::is_output_iterator<std::string::const_iterator,
                                                char>::value));
}

TEST(base_test, is_back_insert_iterator) {
  EXPECT_TRUE(fmt::detail::is_back_insert_iterator<
              std::back_insert_iterator<std::string>>::value);
  EXPECT_FALSE(fmt::detail::is_back_insert_iterator<
               std::front_insert_iterator<std::string>>::value);
}

struct minimal_container {
  using value_type = char;
  void push_back(char) {}
};

TEST(base_test, copy) {
  minimal_container c;
  static constexpr char str[] = "a";
  fmt::detail::copy<char>(str, str + 1, std::back_inserter(c));
}

TEST(base_test, get_buffer) {
  mock_buffer<char> buffer;
  void* buffer_ptr = &buffer;
  auto&& appender_result = fmt::detail::get_buffer<char>(fmt::appender(buffer));
  EXPECT_EQ(&appender_result, buffer_ptr);
  auto&& back_inserter_result =
      fmt::detail::get_buffer<char>(std::back_inserter(buffer));
  EXPECT_EQ(&back_inserter_result, buffer_ptr);
}

struct test_struct {};

FMT_BEGIN_NAMESPACE
template <typename Char> struct formatter<test_struct, Char> {
  FMT_CONSTEXPR auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    return ctx.begin();
  }

  auto format(test_struct, format_context& ctx) const -> decltype(ctx.out()) {
    return copy("test", ctx.out());
  }
};
FMT_END_NAMESPACE

// Use a unique result type to make sure that there are no undesirable
// conversions.
struct test_result {};

template <typename T> struct mock_visitor {
  template <typename U> struct result {
    using type = test_result;
  };

  mock_visitor() {
    ON_CALL(*this, visit(_)).WillByDefault(Return(test_result()));
  }

  MOCK_METHOD(test_result, visit, (T));
  MOCK_METHOD(void, unexpected, ());

  auto operator()(T value) -> test_result { return visit(value); }

  template <typename U> auto operator()(U) -> test_result {
    unexpected();
    return test_result();
  }
};

template <typename T> struct visit_type {
  using type = T;
};

#define VISIT_TYPE(type_, visit_type_)   \
  template <> struct visit_type<type_> { \
    using type = visit_type_;            \
  }

VISIT_TYPE(signed char, int);
VISIT_TYPE(unsigned char, unsigned);
VISIT_TYPE(short, int);
VISIT_TYPE(unsigned short, unsigned);

#if LONG_MAX == INT_MAX
VISIT_TYPE(long, int);
VISIT_TYPE(unsigned long, unsigned);
#else
VISIT_TYPE(long, long long);
VISIT_TYPE(unsigned long, unsigned long long);
#endif

#if FMT_BUILTIN_TYPES
#  define CHECK_ARG(expected, value)                                  \
    {                                                                 \
      testing::StrictMock<mock_visitor<decltype(expected)>> visitor;  \
      EXPECT_CALL(visitor, visit(expected));                          \
      auto var = value;                                               \
      fmt::basic_format_arg<fmt::format_context>(var).visit(visitor); \
    }
#else
#  define CHECK_ARG(expected, value)
#endif

#define CHECK_ARG_SIMPLE(value)                             \
  {                                                         \
    using value_type = decltype(value);                     \
    typename visit_type<value_type>::type expected = value; \
    CHECK_ARG(expected, value)                              \
  }

TEST(arg_test, format_args) {
  auto args = fmt::format_args();
  EXPECT_FALSE(args.get(1));
}

TEST(arg_test, char_arg) { CHECK_ARG('a', 'a'); }

TEST(arg_test, string_arg) {
  char str_data[] = "test";
  char* str = str_data;
  const char* cstr = str;
  CHECK_ARG(cstr, str);

  auto sv = fmt::string_view(str);
  CHECK_ARG(sv, std::string(str));
}

TEST(arg_test, pointer_arg) {
  void* p = nullptr;
  const void* cp = nullptr;
  CHECK_ARG(cp, p);
  CHECK_ARG_SIMPLE(cp);
}

TEST(arg_test, volatile_pointer_arg) {
  const void* p = nullptr;
  volatile int* vip = nullptr;
  const volatile int* cvip = nullptr;
  CHECK_ARG(p, static_cast<volatile void*>(vip));
  CHECK_ARG(p, static_cast<const volatile void*>(cvip));
}

struct check_custom {
  auto operator()(fmt::basic_format_arg<fmt::format_context>::handle h) const
      -> test_result {
    struct test_buffer final : fmt::detail::buffer<char> {
      char data[10];
      test_buffer()
          : fmt::detail::buffer<char>([](buffer<char>&, size_t) {}, data, 0,
                                      10) {}
    } buffer;
    auto parse_ctx = fmt::format_parse_context("");
    auto ctx = fmt::format_context(fmt::appender(buffer), fmt::format_args());
    h.format(parse_ctx, ctx);
    EXPECT_EQ(std::string(buffer.data, buffer.size()), "test");
    return test_result();
  }
};

TEST(arg_test, custom_arg) {
  auto test = test_struct();
  using visitor =
      mock_visitor<fmt::basic_format_arg<fmt::format_context>::handle>;
  auto&& v = testing::StrictMock<visitor>();
  EXPECT_CALL(v, visit(_)).WillOnce(Invoke(check_custom()));
  fmt::basic_format_arg<fmt::format_context>(test).visit(v);
}

TEST(arg_test, visit_invalid_arg) {
  auto&& visitor = testing::StrictMock<mock_visitor<fmt::monostate>>();
  EXPECT_CALL(visitor, visit(_));
  fmt::basic_format_arg<fmt::format_context>().visit(visitor);
}

template <typename T> class numeric_arg_test : public testing::Test {};

#if FMT_BUILTIN_TYPES
using test_types =
    testing::Types<bool, signed char, unsigned char, short, unsigned short, int,
                   unsigned, long, unsigned long, long long, unsigned long long,
                   float, double, long double>;
#else
using test_types = testing::Types<int>;
#endif
TYPED_TEST_SUITE(numeric_arg_test, test_types);

template <typename T, fmt::enable_if_t<std::is_integral<T>::value, int> = 0>
auto test_value() -> T {
  return static_cast<T>(42);
}

template <typename T,
          fmt::enable_if_t<std::is_floating_point<T>::value, int> = 0>
auto test_value() -> T {
  return static_cast<T>(4.2);
}

TYPED_TEST(numeric_arg_test, make_and_visit) {
  CHECK_ARG_SIMPLE(test_value<TypeParam>());
  CHECK_ARG_SIMPLE(std::numeric_limits<TypeParam>::min());
  CHECK_ARG_SIMPLE(std::numeric_limits<TypeParam>::max());
}

#if FMT_USE_CONSTEXPR

enum class arg_id_result { none, index, name };

struct test_arg_id_handler {
  arg_id_result res = arg_id_result::none;
  int index = 0;
  fmt::string_view name;

  constexpr void on_index(int i) {
    res = arg_id_result::index;
    index = i;
  }

  constexpr void on_name(fmt::string_view n) {
    res = arg_id_result::name;
    name = n;
  }
};

template <size_t N>
constexpr auto parse_arg_id(const char (&s)[N]) -> test_arg_id_handler {
  auto h = test_arg_id_handler();
  fmt::detail::parse_arg_id(s, s + N, h);
  return h;
}

TEST(base_test, constexpr_parse_arg_id) {
  static_assert(parse_arg_id("42:").res == arg_id_result::index, "");
  static_assert(parse_arg_id("42:").index == 42, "");
  static_assert(parse_arg_id("foo:").res == arg_id_result::name, "");
  static_assert(parse_arg_id("foo:").name.size() == 3, "");
}

template <size_t N> constexpr auto parse_test_specs(const char (&s)[N]) {
  auto ctx = fmt::detail::compile_parse_context<char>(fmt::string_view(s, N),
                                                      43, nullptr);
  auto specs = fmt::detail::dynamic_format_specs<>();
  fmt::detail::parse_format_specs(s, s + N - 1, specs, ctx,
                                  fmt::detail::type::float_type);
  return specs;
}

TEST(base_test, constexpr_parse_format_specs) {
  static_assert(parse_test_specs("<").align() == fmt::align::left, "");
  static_assert(parse_test_specs("*^").fill_unit<char>() == '*', "");
  static_assert(parse_test_specs("+").sign() == fmt::sign::plus, "");
  static_assert(parse_test_specs("-").sign() == fmt::sign::none, "");
  static_assert(parse_test_specs(" ").sign() == fmt::sign::space, "");
  static_assert(parse_test_specs("#").alt(), "");
  static_assert(parse_test_specs("0").align() == fmt::align::numeric, "");
  static_assert(parse_test_specs("L").localized(), "");
  static_assert(parse_test_specs("42").width == 42, "");
  static_assert(parse_test_specs("{42}").width_ref.index == 42, "");
  static_assert(parse_test_specs(".42").precision == 42, "");
  static_assert(parse_test_specs(".{42}").precision_ref.index == 42, "");
  static_assert(parse_test_specs("f").type() == fmt::presentation_type::fixed,
                "");
}

struct test_format_string_handler {
  constexpr void on_text(const char*, const char*) {}

  constexpr auto on_arg_id() -> int { return 0; }

  template <typename T> constexpr auto on_arg_id(T) -> int { return 0; }

  constexpr void on_replacement_field(int, const char*) {}

  constexpr auto on_format_specs(int, const char* begin, const char*) -> const
      char* {
    return begin;
  }

  constexpr void on_error(const char*) { error = true; }

  bool error = false;
};

template <size_t N> constexpr auto parse_string(const char (&s)[N]) -> bool {
  auto h = test_format_string_handler();
  fmt::detail::parse_format_string(fmt::string_view(s, N - 1), h);
  return !h.error;
}

TEST(base_test, constexpr_parse_format_string) {
  static_assert(parse_string("foo"), "");
  static_assert(!parse_string("}"), "");
  static_assert(parse_string("{}"), "");
  static_assert(parse_string("{42}"), "");
  static_assert(parse_string("{foo}"), "");
  static_assert(parse_string("{:}"), "");
}

#endif  // FMT_USE_CONSTEXPR

struct enabled_formatter {};
struct enabled_ptr_formatter {};
struct disabled_formatter {};
struct disabled_formatter_convertible {
  operator int() const { return 42; }
};

FMT_BEGIN_NAMESPACE
template <> struct formatter<enabled_formatter> {
  FMT_CONSTEXPR auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    return ctx.begin();
  }
  auto format(enabled_formatter, format_context& ctx) const
      -> decltype(ctx.out()) {
    return ctx.out();
  }
};

template <> struct formatter<enabled_ptr_formatter*> {
  FMT_CONSTEXPR auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    return ctx.begin();
  }
  auto format(enabled_ptr_formatter*, format_context& ctx) const
      -> decltype(ctx.out()) {
    return ctx.out();
  }
};
FMT_END_NAMESPACE

struct const_formattable {};
struct nonconst_formattable {};

FMT_BEGIN_NAMESPACE
template <> struct formatter<const_formattable> {
  FMT_CONSTEXPR auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    return ctx.begin();
  }

  auto format(const const_formattable&, format_context& ctx) const
      -> decltype(ctx.out()) {
    return copy("test", ctx.out());
  }
};

template <> struct formatter<nonconst_formattable> {
  FMT_CONSTEXPR auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    return ctx.begin();
  }

  auto format(nonconst_formattable&, format_context& ctx) const
      -> decltype(ctx.out()) {
    return copy("test", ctx.out());
  }
};
FMT_END_NAMESPACE

struct convertible_to_pointer {
  operator const int*() const { return nullptr; }
};

struct convertible_to_pointer_formattable {
  operator const int*() const { return nullptr; }
};

FMT_BEGIN_NAMESPACE
template <> struct formatter<convertible_to_pointer_formattable> {
  FMT_CONSTEXPR auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    return ctx.begin();
  }

  auto format(convertible_to_pointer_formattable, format_context& ctx) const
      -> decltype(ctx.out()) {
    return copy("test", ctx.out());
  }
};
FMT_END_NAMESPACE

enum class unformattable_scoped_enum {};

TEST(base_test, is_formattable) {
  EXPECT_FALSE(fmt::is_formattable<void>::value);
  EXPECT_FALSE(fmt::is_formattable<wchar_t>::value);
#ifdef __cpp_char8_t
  EXPECT_FALSE(fmt::is_formattable<char8_t>::value);
#endif
  EXPECT_FALSE(fmt::is_formattable<char16_t>::value);
  EXPECT_FALSE(fmt::is_formattable<char32_t>::value);
  EXPECT_FALSE(fmt::is_formattable<signed char*>::value);
  EXPECT_FALSE(fmt::is_formattable<unsigned char*>::value);
  EXPECT_FALSE(fmt::is_formattable<const signed char*>::value);
  EXPECT_FALSE(fmt::is_formattable<const unsigned char*>::value);
  EXPECT_FALSE(fmt::is_formattable<const wchar_t*>::value);
  EXPECT_FALSE(fmt::is_formattable<const wchar_t[3]>::value);
  EXPECT_FALSE(fmt::is_formattable<fmt::basic_string_view<wchar_t>>::value);
  EXPECT_FALSE(fmt::is_formattable<enabled_ptr_formatter*>::value);
  EXPECT_FALSE(fmt::is_formattable<disabled_formatter>::value);
  EXPECT_FALSE(fmt::is_formattable<disabled_formatter_convertible>::value);

  EXPECT_TRUE(fmt::is_formattable<enabled_formatter>::value);
  EXPECT_TRUE(fmt::is_formattable<const_formattable&>::value);
  EXPECT_TRUE(fmt::is_formattable<const const_formattable&>::value);

  EXPECT_TRUE(fmt::is_formattable<nonconst_formattable&>::value);
  EXPECT_FALSE(fmt::is_formattable<const nonconst_formattable&>::value);

  EXPECT_FALSE(fmt::is_formattable<convertible_to_pointer>::value);
  const auto f = convertible_to_pointer_formattable();
  auto str = std::string();
  fmt::format_to(std::back_inserter(str), "{}", f);
  EXPECT_EQ(str, "test");

  EXPECT_FALSE(fmt::is_formattable<void (*)()>::value);

  struct s;
  EXPECT_FALSE(fmt::is_formattable<int(s::*)>::value);
  EXPECT_FALSE(fmt::is_formattable<int (s::*)()>::value);
  EXPECT_FALSE(fmt::is_formattable<unformattable_scoped_enum>::value);
  EXPECT_FALSE(fmt::is_formattable<unformattable_scoped_enum>::value);
}

#ifdef __cpp_concepts
TEST(base_test, formattable_concept) {
  static_assert(fmt::formattable<char>);
  static_assert(fmt::formattable<char&>);
  static_assert(fmt::formattable<char&&>);
  static_assert(fmt::formattable<const char>);
  static_assert(fmt::formattable<const char&>);
  static_assert(fmt::formattable<const char&&>);
  static_assert(fmt::formattable<int>);
  static_assert(!fmt::formattable<wchar_t>);
}
#endif

TEST(base_test, format_to) {
  auto s = std::string();
  fmt::format_to(std::back_inserter(s), "{}", 42);
  EXPECT_EQ(s, "42");
}

TEST(base_test, format_to_array) {
  char buffer[4];
  auto result = fmt::format_to(buffer, "{}", 12345);
  EXPECT_EQ(std::distance(&buffer[0], result.out), 4);
  EXPECT_TRUE(result.truncated);
  EXPECT_EQ(result.out, buffer + 4);
  EXPECT_EQ(fmt::string_view(buffer, 4), "1234");

  char* out = nullptr;
  EXPECT_THROW(out = result, std::runtime_error);
  (void)out;

  result = fmt::format_to(buffer, "{:s}", "foobar");
  EXPECT_EQ(std::distance(&buffer[0], result.out), 4);
  EXPECT_TRUE(result.truncated);
  EXPECT_EQ(result.out, buffer + 4);
  EXPECT_EQ(fmt::string_view(buffer, 4), "foob");

  buffer[0] = 'x';
  buffer[1] = 'x';
  buffer[2] = 'x';
  buffer[3] = 'x';
  result = fmt::format_to(buffer, "{}", 'A');
  EXPECT_EQ(std::distance(&buffer[0], result.out), 1);
  EXPECT_FALSE(result.truncated);
  EXPECT_EQ(result.out, buffer + 1);
  EXPECT_EQ(fmt::string_view(buffer, 4), "Axxx");

  result = fmt::format_to(buffer, "{}{} ", 'B', 'C');
  EXPECT_EQ(std::distance(&buffer[0], result.out), 3);
  EXPECT_FALSE(result.truncated);
  EXPECT_EQ(result.out, buffer + 3);
  EXPECT_EQ(fmt::string_view(buffer, 4), "BC x");

  result = fmt::format_to(buffer, "{}", "ABCDE");
  EXPECT_EQ(std::distance(&buffer[0], result.out), 4);
  EXPECT_TRUE(result.truncated);
  EXPECT_EQ(fmt::string_view(buffer, 4), "ABCD");

  result = fmt::format_to(buffer, "{}", std::string(1000, '*').c_str());
  EXPECT_EQ(std::distance(&buffer[0], result.out), 4);
  EXPECT_TRUE(result.truncated);
  EXPECT_EQ(fmt::string_view(buffer, 4), "****");
}

// Test that check is not found by ADL.
template <typename T> void check(T);
TEST(base_test, adl_check) {
  auto s = std::string();
  fmt::format_to(std::back_inserter(s), "{}", test_struct());
  EXPECT_EQ(s, "test");
}

struct implicitly_convertible_to_string_view {
  operator fmt::string_view() const { return "foo"; }
};

TEST(base_test, no_implicit_conversion_to_string_view) {
  EXPECT_FALSE(
      fmt::is_formattable<implicitly_convertible_to_string_view>::value);
}

struct explicitly_convertible_to_string_view {
  explicit operator fmt::string_view() const { return "foo"; }
};

TEST(base_test, format_explicitly_convertible_to_string_view) {
  // Types explicitly convertible to string_view are not formattable by
  // default because it may introduce ODR violations.
  static_assert(
      !fmt::is_formattable<explicitly_convertible_to_string_view>::value, "");
}

#if FMT_CPLUSPLUS >= 201703L
struct implicitly_convertible_to_std_string_view {
  operator std::string_view() const { return "foo"; }
};

TEST(base_test, no_implicit_conversion_to_std_string_view) {
  EXPECT_FALSE(
      fmt::is_formattable<implicitly_convertible_to_std_string_view>::value);
}

struct explicitly_convertible_to_std_string_view {
  explicit operator std::string_view() const { return "foo"; }
};

TEST(base_test, format_explicitly_convertible_to_std_string_view) {
  // Types explicitly convertible to string_view are not formattable by
  // default because it may introduce ODR violations.
  static_assert(
      !fmt::is_formattable<explicitly_convertible_to_std_string_view>::value,
      "");
}
#endif  // FMT_CPLUSPLUS >= 201703L

TEST(base_test, has_formatter) {
  EXPECT_TRUE((fmt::detail::has_formatter<const const_formattable, char>()));
  EXPECT_FALSE(
      (fmt::detail::has_formatter<const nonconst_formattable, char>()));
}

TEST(base_test, format_nonconst) {
  auto s = std::string();
  fmt::format_to(std::back_inserter(s), "{}", nonconst_formattable());
  EXPECT_EQ(s, "test");
}

TEST(base_test, throw_in_buffer_dtor) {
  constexpr int buffer_size = 256;

  struct throwing_iterator {
    int& count;

    auto operator=(char) -> throwing_iterator& {
      if (++count > buffer_size) throw std::exception();
      return *this;
    }
    auto operator*() -> throwing_iterator& { return *this; }
    auto operator++() -> throwing_iterator& { return *this; }
    auto operator++(int) -> throwing_iterator { return *this; }
  };

  try {
    int count = 0;
    fmt::format_to(throwing_iterator{count}, fmt::runtime("{:{}}{"), "",
                   buffer_size + 1);
  } catch (const std::exception&) {
  }
}

struct convertible_to_any_type_with_member_x {
  template <typename T> operator T() const {
    auto v = T();
    v.x = 42;
    return v;
  }
};

FMT_BEGIN_NAMESPACE
template <> struct formatter<convertible_to_any_type_with_member_x> {
  FMT_CONSTEXPR auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    return ctx.begin();
  }

  auto format(convertible_to_any_type_with_member_x, format_context& ctx) const
      -> decltype(ctx.out()) const {
    auto out = ctx.out();
    *out++ = 'x';
    return out;
  }
};
FMT_END_NAMESPACE

TEST(base_test, promiscuous_conversions) {
  auto s = std::string();
  fmt::format_to(std::back_inserter(s), "{}",
                 convertible_to_any_type_with_member_x());
  EXPECT_EQ(s, "x");
}

struct custom_container {
  char data;

  using value_type = char;

  size_t size() const { return 0; }
  void resize(size_t) {}

  void push_back(char) {}
  char& operator[](size_t) { return data; }
};

FMT_BEGIN_NAMESPACE
template <> struct is_contiguous<custom_container> : std::true_type {};
FMT_END_NAMESPACE

TEST(base_test, format_to_custom_container) {
  auto c = custom_container();
  fmt::format_to(std::back_inserter(c), "");
}

TEST(base_test, no_repeated_format_string_conversions) {
  struct nondeterministic_format_string {
    mutable int i = 0;
    FMT_CONSTEXPR operator fmt::string_view() const {
      return {"{}", i++ != 0 ? 2u : 0u};
    }
  };

#if !FMT_GCC_VERSION || FMT_GCC_VERSION >= 1200
  char buf[10];
  fmt::format_to(buf, nondeterministic_format_string());
#endif
}

TEST(base_test, format_context_accessors) {
  auto copy = [](fmt::appender app, const fmt::format_context& ctx) {
    return fmt::format_context(app, ctx.args(), ctx.locale());
  };
  fmt::detail::ignore_unused(copy);
}
