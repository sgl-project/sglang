// A fuzzer for floating-point formatter.
// For the license information refer to format.h.

#include <fmt/format.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#include "fuzzer-common.h"

void check_round_trip(fmt::string_view format_str, double value) {
  auto buffer = fmt::memory_buffer();
  fmt::format_to(std::back_inserter(buffer), format_str, value);

  if (std::isnan(value)) {
    auto nan = std::signbit(value) ? "-nan" : "nan";
    if (fmt::string_view(buffer.data(), buffer.size()) != nan)
      throw std::runtime_error("round trip failure");
    return;
  }

  buffer.push_back('\0');
  char* ptr = nullptr;
  if (std::strtod(buffer.data(), &ptr) != value)
    throw std::runtime_error("round trip failure");
  if (ptr + 1 != buffer.end()) throw std::runtime_error("unparsed output");
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size <= sizeof(double) || !std::numeric_limits<double>::is_iec559)
    return 0;
  check_round_trip("{}", assign_from_buf<double>(data));
  // A larger than necessary precision is used to trigger the fallback
  // formatter.
  check_round_trip("{:.50g}", assign_from_buf<double>(data));
  return 0;
}
