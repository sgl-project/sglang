#pragma once
#include <version>

/// NOTE: fallback to a minimal source_location implementation
#if defined(__cpp_lib_source_location)
#include <source_location>

using source_location_t = std::source_location;

#else

struct source_location_fallback {
 public:
  static constexpr source_location_fallback current() noexcept {
    return source_location_fallback{};
  }
  constexpr source_location_fallback() noexcept = default;
  constexpr unsigned line() const noexcept {
    return 0;
  }
  constexpr unsigned column() const noexcept {
    return 0;
  }
  constexpr const char* file_name() const noexcept {
    return "";
  }
  constexpr const char* function_name() const noexcept {
    return "";
  }
};

using source_location_t = source_location_fallback;

#endif
