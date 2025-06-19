#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace radix_tree_v2 {

using token_t = std::int32_t;
using token_vec_t = std::vector<token_t>;
using NodeHandle = std::size_t;
using IOTicket = std::uint32_t;

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
