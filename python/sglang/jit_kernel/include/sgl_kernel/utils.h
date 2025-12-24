#pragma once

// ref: https://forums.developer.nvidia.com/t/c-20s-source-location-compilation-error-when-using-nvcc-12-1/258026/3
#ifdef __CUDACC__
#pragma push_macro("__cpp_consteval")
#pragma push_macro("_NODISCARD")
#pragma push_macro("__builtin_LINE")

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbuiltin-macro-redefined"
#define __cpp_consteval 201811L
#pragma clang diagnostic pop

#ifdef _NODISCARD
#undef _NODISCARD
#define _NODISCARD
#endif

#define consteval constexpr

#include <source_location>

#undef consteval
#pragma pop_macro("__cpp_consteval")
#pragma pop_macro("_NODISCARD")
#else
#include <source_location>
#endif

#include <dlpack/dlpack.h>

#include <concepts>
#include <cstddef>
#include <ostream>
#include <ranges>
#include <source_location>
#include <sstream>
#include <utility>

namespace host {

struct DebugInfo : public std::source_location {
  DebugInfo(std::source_location loc = std::source_location::current()) : std::source_location(loc) {}
};

struct PanicError : public std::runtime_error {
 public:
  explicit PanicError(std::string msg) : runtime_error(msg), m_message(std::move(msg)) {}
  auto root_cause() const -> std::string_view {
    const auto str = std::string_view{m_message};
    const auto pos = str.find(": ");
    return pos == std::string_view::npos ? str : str.substr(pos + 2);
  }

 private:
  std::string m_message;
};

template <typename... Args>
[[noreturn]]
inline auto panic(DebugInfo location, Args&&... args) -> void {
  std::ostringstream os;
  os << "Runtime check failed at " << location.file_name() << ":" << location.line();
  if constexpr (sizeof...(args) > 0) {
    os << ": ";
    (os << ... << std::forward<Args>(args));
  } else {
    os << " in " << location.function_name();
  }
  throw PanicError(std::move(os).str());
}

template <typename... Args>
struct RuntimeCheck {
  template <typename Cond>
  explicit RuntimeCheck(Cond&& condition, Args&&... args, DebugInfo location = {}) {
    if (condition) return;
    [[unlikely]] ::host::panic(location, std::forward<Args>(args)...);
  }
  template <typename Cond>
  explicit RuntimeCheck(DebugInfo location, Cond&& condition, Args&&... args) {
    if (condition) return;
    [[unlikely]] ::host::panic(location, std::forward<Args>(args)...);
  }
};

template <typename... Args>
struct Panic {
  explicit Panic(Args&&... args, DebugInfo location = {}) {
    ::host::panic(location, std::forward<Args>(args)...);
  }
  explicit Panic(DebugInfo location, Args&&... args) {
    ::host::panic(location, std::forward<Args>(args)...);
  }
  [[noreturn]] ~Panic() {
    std::terminate();
  }
};

template <typename Cond, typename... Args>
explicit RuntimeCheck(Cond&&, Args&&...) -> RuntimeCheck<Args...>;

template <typename Cond, typename... Args>
explicit RuntimeCheck(DebugInfo, Cond&&, Args&&...) -> RuntimeCheck<Args...>;

template <typename... Args>
explicit Panic(Args&&...) -> Panic<Args...>;

template <typename... Args>
explicit Panic(DebugInfo, Args&&...) -> Panic<Args...>;

namespace pointer {

// we only allow void * pointer arithmetic for safety

template <typename T, std::integral... U>
inline auto offset(T* ptr, U... offset) -> void* {
  static_assert(std::is_same_v<T, void>, "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<char*>(ptr) + (... + offset);
}

template <typename T, std::integral... U>
inline auto offset(const T* ptr, U... offset) -> const void* {
  static_assert(std::is_same_v<T, void>, "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<const char*>(ptr) + (... + offset);
}

}  // namespace pointer

template <std::integral T, std::integral U>
inline constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

inline auto dtype_bytes(DLDataType dtype) -> std::size_t {
  return static_cast<std::size_t>(dtype.bits / 8);
}

namespace stdr = std::ranges;
namespace stdv = stdr::views;

template <std::integral T>
inline auto irange(T end) {
  return stdv::iota(static_cast<T>(0), end);
}

template <std::integral T>
inline auto irange(T start, T end) {
  return stdv::iota(start, end);
}

}  // namespace host
