// Formatting library for C++ - the core API
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#include <vector>

#include "fmt/format.h"
#include "fmt/ranges.h"
#include "gtest/gtest.h"

// call fmt::format from another translation unit to test ODR
TEST(ranges_odr_test, format_vector) {
  auto v = std::vector<int>{1, 2, 3, 5, 7, 11};
  EXPECT_EQ(fmt::format("{}", v), "[1, 2, 3, 5, 7, 11]");
}
