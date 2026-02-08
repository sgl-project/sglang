// Formatting library for C++ - dynamic argument store tests
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#include "fmt/args.h"

#include <memory>

#include "gtest/gtest.h"

TEST(args_test, basic) {
  fmt::dynamic_format_arg_store<fmt::format_context> store;
  store.push_back(42);
  store.push_back("abc1");
  store.push_back(1.5f);
  EXPECT_EQ("42 and abc1 and 1.5", fmt::vformat("{} and {} and {}", store));
}

TEST(args_test, strings_and_refs) {
  // Unfortunately the tests are compiled with old ABI so strings use COW.
  fmt::dynamic_format_arg_store<fmt::format_context> store;
  char str[] = "1234567890";
  store.push_back(str);
  store.push_back(std::cref(str));
  store.push_back(fmt::string_view{str});
  str[0] = 'X';

  auto result = fmt::vformat("{} and {} and {}", store);
  EXPECT_EQ("1234567890 and X234567890 and X234567890", result);
}

struct custom_type {
  int i = 0;
};

FMT_BEGIN_NAMESPACE
template <> struct formatter<custom_type> {
  auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const custom_type& p, FormatContext& ctx) const
      -> decltype(ctx.out()) {
    return fmt::format_to(ctx.out(), "cust={}", p.i);
  }
};
FMT_END_NAMESPACE

TEST(args_test, custom_format) {
  fmt::dynamic_format_arg_store<fmt::format_context> store;
  auto c = custom_type();
  store.push_back(c);
  ++c.i;
  store.push_back(c);
  ++c.i;
  store.push_back(std::cref(c));
  ++c.i;
  auto result = fmt::vformat("{} and {} and {}", store);
  EXPECT_EQ("cust=0 and cust=1 and cust=3", result);
}

struct to_stringable {
  friend fmt::string_view to_string_view(to_stringable) { return {}; }
};

FMT_BEGIN_NAMESPACE
template <> struct formatter<to_stringable> {
  auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    return ctx.begin();
  }

  auto format(to_stringable, format_context& ctx) const -> decltype(ctx.out()) {
    return ctx.out();
  }
};
FMT_END_NAMESPACE

TEST(args_test, to_string_and_formatter) {
  fmt::dynamic_format_arg_store<fmt::format_context> store;
  auto s = to_stringable();
  store.push_back(s);
  store.push_back(std::cref(s));
  fmt::vformat("", store);
}

TEST(args_test, named_int) {
  fmt::dynamic_format_arg_store<fmt::format_context> store;
  store.push_back(fmt::arg("a1", 42));
  EXPECT_EQ("42", fmt::vformat("{a1}", store));
}

TEST(args_test, named_strings) {
  fmt::dynamic_format_arg_store<fmt::format_context> store;
  char str[] = "1234567890";
  store.push_back(fmt::arg("a1", str));
  store.push_back(fmt::arg("a2", std::cref(str)));
  str[0] = 'X';
  EXPECT_EQ("1234567890 and X234567890", fmt::vformat("{a1} and {a2}", store));
}

TEST(args_test, named_arg_by_ref) {
  fmt::dynamic_format_arg_store<fmt::format_context> store;
  char band[] = "Rolling Stones";
  store.push_back(fmt::arg("band", std::cref(band)));
  band[9] = 'c';  // Changing band affects the output.
  EXPECT_EQ(fmt::vformat("{band}", store), "Rolling Scones");
}

TEST(args_test, named_custom_format) {
  fmt::dynamic_format_arg_store<fmt::format_context> store;
  auto c = custom_type();
  store.push_back(fmt::arg("c1", c));
  ++c.i;
  store.push_back(fmt::arg("c2", c));
  ++c.i;
  store.push_back(fmt::arg("c_ref", std::cref(c)));
  ++c.i;
  auto result = fmt::vformat("{c1} and {c2} and {c_ref}", store);
  EXPECT_EQ("cust=0 and cust=1 and cust=3", result);
}

TEST(args_test, clear) {
  fmt::dynamic_format_arg_store<fmt::format_context> store;
  store.push_back(42);

  auto result = fmt::vformat("{}", store);
  EXPECT_EQ("42", result);

  store.push_back(43);
  result = fmt::vformat("{} and {}", store);
  EXPECT_EQ("42 and 43", result);

  store.clear();
  store.push_back(44);
  result = fmt::vformat("{}", store);
  EXPECT_EQ("44", result);
}

TEST(args_test, reserve) {
  fmt::dynamic_format_arg_store<fmt::format_context> store;
  store.reserve(2, 1);
  store.push_back(1.5f);
  store.push_back(fmt::arg("a", 42));
  auto result = fmt::vformat("{} and {a}", store);
  EXPECT_EQ("1.5 and 42", result);
}

struct copy_throwable {
  copy_throwable() {}
  copy_throwable(const copy_throwable&) { throw "deal with it"; }
};

FMT_BEGIN_NAMESPACE
template <> struct formatter<copy_throwable> {
  auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    return ctx.begin();
  }
  auto format(copy_throwable, format_context& ctx) const
      -> decltype(ctx.out()) {
    return ctx.out();
  }
};
FMT_END_NAMESPACE

TEST(args_test, throw_on_copy) {
  fmt::dynamic_format_arg_store<fmt::format_context> store;
  store.push_back(std::string("foo"));
  try {
    store.push_back(copy_throwable());
  } catch (...) {
  }
  EXPECT_EQ(fmt::vformat("{}", store), "foo");
}

TEST(args_test, move_constructor) {
  using store_type = fmt::dynamic_format_arg_store<fmt::format_context>;
  auto store = std::unique_ptr<store_type>(new store_type());
  store->push_back(42);
  store->push_back(std::string("foo"));
  store->push_back(fmt::arg("a1", "foo"));
  auto moved_store = std::move(*store);
  store.reset();
  EXPECT_EQ(fmt::vformat("{} {} {a1}", moved_store), "42 foo foo");
}

TEST(args_test, size) {
  fmt::dynamic_format_arg_store<fmt::format_context> store;
  EXPECT_EQ(store.size(), 0);

  store.push_back(42);
  EXPECT_EQ(store.size(), 1);

  store.push_back("Molybdenum");
  EXPECT_EQ(store.size(), 2);

  store.clear();
  EXPECT_EQ(store.size(), 0);
}
