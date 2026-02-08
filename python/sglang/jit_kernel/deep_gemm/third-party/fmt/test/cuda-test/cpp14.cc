#include <fmt/base.h>

// The purpose of this part is to ensure NVCC's host compiler also supports
// the standard version. See 'cuda-cpp14.cu'.
//
// https://en.cppreference.com/w/cpp/preprocessor/replace#Predefined_macros
static_assert(__cplusplus >= 201402L, "expect C++ 2014 for host compiler");

auto make_message_cpp() -> std::string {
  return fmt::format("host compiler \t: __cplusplus == {}", __cplusplus);
}
