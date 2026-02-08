// Header-only configuration test

#include "fmt/base.h"
#include "fmt/ostream.h"
#include "gtest/gtest.h"

#ifndef FMT_HEADER_ONLY
#  error "Not in the header-only mode."
#endif

TEST(header_only_test, format) { EXPECT_EQ(fmt::format("foo"), "foo"); }
