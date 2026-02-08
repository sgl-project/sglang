// Copyright (c) 2019, Paul Dreik
// For the license information refer to format.h.

#include <fmt/chrono.h>

#include <cstdint>
#include <type_traits>
#include <vector>

#include "fuzzer-common.h"

template <typename T>
void invoke_fmt(const uint8_t* data, size_t size, unsigned arg_name_size) {
  static_assert(sizeof(T) <= fixed_size, "fixed_size too small");
  if (size <= fixed_size) return;
  const T value = assign_from_buf<T>(data);
  data += fixed_size;
  size -= fixed_size;

  if (arg_name_size <= 0 || arg_name_size >= size) return;
  data_to_string arg_name(data, arg_name_size, true);
  data += arg_name_size;
  size -= arg_name_size;

  data_to_string format_str(data, size);
  try {
#if FMT_FUZZ_FORMAT_TO_STRING
    std::string message =
        fmt::format(format_str.get(), fmt::arg(arg_name.data(), value));
#else
    fmt::memory_buffer out;
    fmt::format_to(std::back_inserter(out), format_str.get(),
                   fmt::arg(arg_name.data(), value));
#endif
  } catch (std::exception&) {
  }
}

// For dynamic dispatching to an explicit instantiation.
template <typename Callback> void invoke(int type, Callback callback) {
  switch (type) {
  case 0:
    callback(bool());
    break;
  case 1:
    callback(char());
    break;
  case 2:
    using sc = signed char;
    callback(sc());
    break;
  case 3:
    using uc = unsigned char;
    callback(uc());
    break;
  case 4:
    callback(short());
    break;
  case 5:
    using us = unsigned short;
    callback(us());
    break;
  case 6:
    callback(int());
    break;
  case 7:
    callback(unsigned());
    break;
  case 8:
    callback(long());
    break;
  case 9:
    using ul = unsigned long;
    callback(ul());
    break;
  case 10:
    callback(float());
    break;
  case 11:
    callback(double());
    break;
  case 12:
    using LD = long double;
    callback(LD());
    break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size <= 3) return 0;

  // Switch types depending on the first byte of the input.
  const auto type = data[0] & 0x0F;
  const unsigned arg_name_size = (data[0] & 0xF0) >> 4;
  data++;
  size--;

  invoke(type, [=](auto arg) {
    invoke_fmt<decltype(arg)>(data, size, arg_name_size);
  });
  return 0;
}
