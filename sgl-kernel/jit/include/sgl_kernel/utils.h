#pragma once

#include <dlpack/dlpack.h>

#include <concepts>
#include <ostream>
#include <source_location>
#include <sstream>
#include <utility>

namespace host {

struct PanicError : public std::runtime_error {
 public:
  // copy and move constructors
  PanicError(std::string msg) : runtime_error(msg), m_message(std::move(msg)) {}
  auto detail() const -> std::string_view {
    const auto sv = std::string_view{m_message};
    const auto pos = sv.find(": ");
    return pos == std::string_view::npos ? sv : sv.substr(pos + 2);
  }

 private:
  std::string m_message;
};

template <typename... Args>
[[noreturn]]
inline auto panic(std::source_location location, Args&&... args) -> void {
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
  template <typename T>
  explicit RuntimeCheck(
      T&& condition, Args&&... args, std::source_location location = std::source_location::current()) {
    if (!condition) {
      [[unlikely]];
      ::host::panic(location, std::forward<Args>(args)...);
    }
  }
};

template <typename T, typename... Args>
explicit RuntimeCheck(T&&, Args&&...) -> RuntimeCheck<Args...>;

template <std::integral T, std::integral U>
inline constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

inline auto dtype_bytes(DLDataType dtype) -> std::size_t {
  return static_cast<std::size_t>(dtype.bits / 8);
}

namespace pointer {

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

}  // namespace host
