// Formatting library for C++ - test version of FMT_ASSERT
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#ifndef FMT_TEST_ASSERT_H_
#define FMT_TEST_ASSERT_H_

#include <stdexcept>

void throw_assertion_failure(const char* message);
#define FMT_ASSERT(condition, message) \
  ((condition) ? (void)0 : throw_assertion_failure(message))

#include "gtest/gtest.h"

class assertion_failure : public std::logic_error {
 public:
  explicit assertion_failure(const char* message) : std::logic_error(message) {}

 private:
  virtual void avoid_weak_vtable();
};

void assertion_failure::avoid_weak_vtable() {}

// We use a separate function (rather than throw directly from FMT_ASSERT) to
// avoid GCC's -Wterminate warning when FMT_ASSERT is used in a destructor.
inline void throw_assertion_failure(const char* message) {
  throw assertion_failure(message);
}

// Expects an assertion failure.
#define EXPECT_ASSERT(stmt, message) \
  FMT_TEST_THROW_(stmt, assertion_failure, message, GTEST_NONFATAL_FAILURE_)

#endif  // FMT_TEST_ASSERT_H_
