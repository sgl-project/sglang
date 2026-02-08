// Copyright (c) 2019, Paul Dreik
// For the license information refer to format.h.

#include <fmt/chrono.h>

#include <cstdint>

#include "fuzzer-common.h"

template <typename Period, typename Rep>
void invoke_inner(fmt::string_view format_str, Rep rep) {
  auto value = std::chrono::duration<Rep, Period>(rep);
  try {
#if FMT_FUZZ_FORMAT_TO_STRING
    std::string message = fmt::format(format_str, value);
#else
    auto buf = fmt::memory_buffer();
    fmt::format_to(std::back_inserter(buf), format_str, value);
#endif
  } catch (std::exception&) {
  }
}

// Rep is a duration's representation type.
template <typename Rep>
void invoke_outer(const uint8_t* data, size_t size, int period) {
  // Always use a fixed location of the data.
  static_assert(sizeof(Rep) <= fixed_size, "fixed size is too small");
  if (size <= fixed_size + 1) return;

  const Rep rep = assign_from_buf<Rep>(data);
  data += fixed_size;
  size -= fixed_size;

  // data is already allocated separately in libFuzzer so reading past the end
  // will most likely be detected anyway.
  const auto format_str = fmt::string_view(as_chars(data), size);

  // yocto, zepto, zetta and yotta are not handled.
  switch (period) {
  case 1:
    invoke_inner<std::atto>(format_str, rep);
    break;
  case 2:
    invoke_inner<std::femto>(format_str, rep);
    break;
  case 3:
    invoke_inner<std::pico>(format_str, rep);
    break;
  case 4:
    invoke_inner<std::nano>(format_str, rep);
    break;
  case 5:
    invoke_inner<std::micro>(format_str, rep);
    break;
  case 6:
    invoke_inner<std::milli>(format_str, rep);
    break;
  case 7:
    invoke_inner<std::centi>(format_str, rep);
    break;
  case 8:
    invoke_inner<std::deci>(format_str, rep);
    break;
  case 9:
    invoke_inner<std::deca>(format_str, rep);
    break;
  case 10:
    invoke_inner<std::kilo>(format_str, rep);
    break;
  case 11:
    invoke_inner<std::mega>(format_str, rep);
    break;
  case 12:
    invoke_inner<std::giga>(format_str, rep);
    break;
  case 13:
    invoke_inner<std::tera>(format_str, rep);
    break;
  case 14:
    invoke_inner<std::peta>(format_str, rep);
    break;
  case 15:
    invoke_inner<std::exa>(format_str, rep);
    break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size <= 4) return 0;

  const auto representation = data[0];
  const auto period = data[1];
  data += 2;
  size -= 2;

  switch (representation) {
  case 1:
    invoke_outer<char>(data, size, period);
    break;
  case 2:
    invoke_outer<signed char>(data, size, period);
    break;
  case 3:
    invoke_outer<unsigned char>(data, size, period);
    break;
  case 4:
    invoke_outer<short>(data, size, period);
    break;
  case 5:
    invoke_outer<unsigned short>(data, size, period);
    break;
  case 6:
    invoke_outer<int>(data, size, period);
    break;
  case 7:
    invoke_outer<unsigned int>(data, size, period);
    break;
  case 8:
    invoke_outer<long>(data, size, period);
    break;
  case 9:
    invoke_outer<unsigned long>(data, size, period);
    break;
  case 10:
    invoke_outer<float>(data, size, period);
    break;
  case 11:
    invoke_outer<double>(data, size, period);
    break;
  case 12:
    invoke_outer<long double>(data, size, period);
    break;
  }
  return 0;
}
