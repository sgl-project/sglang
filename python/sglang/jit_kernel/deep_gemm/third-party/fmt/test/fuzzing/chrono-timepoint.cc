// Copyright (c) 2021, Paul Dreik
// For license information refer to format.h.
#include <fmt/chrono.h>

#include "fuzzer-common.h"

/*
 * a fuzzer for the chrono timepoints formatters
 * C is a clock (std::chrono::system_clock etc)
 */
template <typename C> void doit(const uint8_t* data, size_t size) {
  using Rep = typename C::time_point::rep;
  constexpr auto N = sizeof(Rep);
  if (size < N) return;

  const auto x = assign_from_buf<Rep>(data);
  typename C::duration dur{x};
  typename C::time_point timepoint{dur};
  data += N;
  size -= N;
  data_to_string format_str(data, size);

  std::string message = fmt::format(format_str.get(), timepoint);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  try {
    doit<std::chrono::system_clock>(data, size);
  } catch (...) {
  }
  return 0;
}
