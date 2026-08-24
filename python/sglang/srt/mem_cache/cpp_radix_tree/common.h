#pragma once
#include <cstddef>
#include <cstdint>
#include <source_location>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace radix_tree_v2 {

// RadixKey is backed by Python's array("q").  Keeping the native token type
// identical lets pybind expose that storage as a read-only span instead of
// materializing a std::vector for every tree operation.
using token_t = std::int64_t;
using token_vec_t = std::vector<token_t>;
using token_slice = std::span<const token_t>;
using NodeHandle = std::size_t;
using IOTicket = std::uint32_t;

inline void _assert(
    bool condition,
    const char* message = "Assertion failed",
    std::source_location loc = std::source_location::current()) {
  if (!condition) [[unlikely]] {
    std::string msg = message;
    msg = msg + " at " + loc.file_name() + ":" + std::to_string(loc.line()) + " in " + loc.function_name();
    throw std::runtime_error(msg);
  }
}

}  // namespace radix_tree_v2
