#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace radix_tree_v2 {

using token_t = std::int32_t;
using token_vec_t = std::vector<token_t>;
using NodeHandle = std::intptr_t;

// so that this can be passed to the torch
static_assert(std::is_same_v<NodeHandle, std::int64_t>);

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
