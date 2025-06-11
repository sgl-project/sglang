#pragma once
#include <stdexcept>
#include <string>

namespace radix_tree_v2 {

inline void _assert(bool condition, const char* message = "Assertion failed") {
  if (!condition) {
    [[unlikely]] throw std::runtime_error(message);
  }
}

inline void _assert(bool condition, const std::string& message) {
  if (!condition) {
    [[unlikely]] throw std::runtime_error(message);
  }
}

}  // namespace radix_tree_v2
