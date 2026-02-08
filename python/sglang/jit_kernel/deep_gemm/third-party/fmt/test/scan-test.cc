// Formatting library for C++ - scanning API test
//
// Copyright (c) 2019 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#include "scan.h"

#include <time.h>

#include <climits>
#include <thread>

#include "fmt/os.h"
#include "gmock/gmock.h"
#include "gtest-extra.h"

TEST(scan_test, read_text) {
  fmt::string_view s = "foo";
  auto end = fmt::scan_to(s, "foo");
  EXPECT_EQ(end, s.end());
  EXPECT_THROW_MSG(fmt::scan<int>("fob", "foo"), fmt::format_error,
                   "invalid input");
}

TEST(scan_test, read_int) {
  EXPECT_EQ(fmt::scan<int>("42", "{}")->value(), 42);
  EXPECT_EQ(fmt::scan<int>("-42", "{}")->value(), -42);
  EXPECT_EQ(fmt::scan<int>("42", "{:}")->value(), 42);
  EXPECT_THROW_MSG(fmt::scan<int>(std::to_string(INT_MAX + 1u), "{}"),
                   fmt::format_error, "number is too big");
}

TEST(scan_test, read_long_long) {
  EXPECT_EQ(fmt::scan<long long>("42", "{}")->value(), 42);
  EXPECT_EQ(fmt::scan<long long>("-42", "{}")->value(), -42);
}

TEST(scan_test, read_uint) {
  EXPECT_EQ(fmt::scan<unsigned>("42", "{}")->value(), 42);
  EXPECT_THROW_MSG(fmt::scan<unsigned>("-42", "{}"), fmt::format_error,
                   "invalid input");
}

TEST(scan_test, read_ulong_long) {
  EXPECT_EQ(fmt::scan<unsigned long long>("42", "{}")->value(), 42);
  EXPECT_THROW_MSG(fmt::scan<unsigned long long>("-42", "{}")->value(),
                   fmt::format_error, "invalid input");
}

TEST(scan_test, read_hex) {
  EXPECT_EQ(fmt::scan<unsigned>("2a", "{:x}")->value(), 42);
  auto num_digits = std::numeric_limits<unsigned>::digits / 4;
  EXPECT_THROW_MSG(
      fmt::scan<unsigned>(fmt::format("1{:0{}}", 0, num_digits), "{:x}")
          ->value(),
      fmt::format_error, "number is too big");
}

TEST(scan_test, read_string) {
  EXPECT_EQ(fmt::scan<std::string>("foo", "{}")->value(), "foo");
}

TEST(scan_test, read_string_view) {
  EXPECT_EQ(fmt::scan<fmt::string_view>("foo", "{}")->value(), "foo");
}

TEST(scan_test, separator) {
  int n1 = 0, n2 = 0;
  fmt::scan_to("10 20", "{} {}", n1, n2);
  EXPECT_EQ(n1, 10);
  EXPECT_EQ(n2, 20);
}

struct num {
  int value;
};

namespace fmt {
template <> struct scanner<num> {
  bool hex = false;

  auto parse(scan_parse_context& ctx) -> scan_parse_context::iterator {
    auto it = ctx.begin(), end = ctx.end();
    if (it != end && *it == 'x') {
      hex = true;
      ++it;
    }
    if (it != end && *it != '}') report_error("invalid format");
    return it;
  }

  template <class ScanContext>
  auto scan(num& n, ScanContext& ctx) const -> typename ScanContext::iterator {
    return hex ? scan_to(ctx, "{:x}", n.value) : scan_to(ctx, "{}", n.value);
  }
};
}  // namespace fmt

TEST(scan_test, read_custom) {
  EXPECT_EQ(fmt::scan<num>("42", "{}")->value().value, 42);
  EXPECT_EQ(fmt::scan<num>("2a", "{:x}")->value().value, 42);
}

TEST(scan_test, invalid_format) {
  EXPECT_THROW_MSG(fmt::scan_to("", "{}"), fmt::format_error,
                   "argument index out of range");
  EXPECT_THROW_MSG(fmt::scan_to("", "{"), fmt::format_error,
                   "invalid format string");
}

namespace std {
using fmt::scan;
using fmt::scan_error;
}  // namespace std

TEST(scan_test, example) {
  // Example from https://wg21.link/p1729r3.
  if (auto result = std::scan<std::string, int>("answer = 42", "{} = {}")) {
    auto range = result->range();
    EXPECT_EQ(range.begin(), range.end());
    EXPECT_EQ(result->begin(), result->end());
#ifdef __cpp_structured_bindings
    const auto& [key, value] = result->values();
    EXPECT_EQ(key, "answer");
    EXPECT_EQ(value, 42);
#endif
  } else {
    std::scan_error error = result.error();
    (void)error;
    FAIL();
  }
}

TEST(scan_test, end_of_input) { fmt::scan<int>("", "{}"); }

#if FMT_USE_FCNTL
TEST(scan_test, file) {
  auto pipe = fmt::pipe();

  fmt::string_view input = "10 20";
  pipe.write_end.write(input.data(), input.size());
  pipe.write_end.close();

  int n1 = 0, n2 = 0;
  fmt::buffered_file f = pipe.read_end.fdopen("r");
  fmt::scan_to(f.get(), "{} {}", n1, n2);
  EXPECT_EQ(n1, 10);
  EXPECT_EQ(n2, 20);
}

TEST(scan_test, lock) {
  auto pipe = fmt::pipe();

  std::thread producer([&]() {
    fmt::string_view input = "42 ";
    for (int i = 0; i < 1000; ++i)
      pipe.write_end.write(input.data(), input.size());
    pipe.write_end.close();
  });

  std::atomic<int> count(0);
  fmt::buffered_file f = pipe.read_end.fdopen("r");
  auto fun = [&]() {
    int value = 0;
    while (fmt::scan_to(f.get(), "{}", value)) {
      if (value != 42) {
        pipe.read_end.close();
        EXPECT_EQ(value, 42);
        break;
      }
      ++count;
    }
  };
  std::thread consumer1(fun);
  std::thread consumer2(fun);

  producer.join();
  consumer1.join();
  consumer2.join();
  EXPECT_EQ(count, 1000);
}
#endif  // FMT_USE_FCNTL
