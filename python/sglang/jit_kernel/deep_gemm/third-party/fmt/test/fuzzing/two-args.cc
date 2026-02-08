// Copyright (c) 2019, Paul Dreik
// For the license information refer to format.h.

#include <fmt/format.h>

#include <cstdint>
#include <exception>
#include <string>

#include "fuzzer-common.h"

template <typename Item1, typename Item2>
void invoke_fmt(const uint8_t* data, size_t size) {
  static_assert(sizeof(Item1) <= fixed_size, "size1 exceeded");
  static_assert(sizeof(Item2) <= fixed_size, "size2 exceeded");
  if (size <= fixed_size + fixed_size) return;

  const Item1 item1 = assign_from_buf<Item1>(data);
  data += fixed_size;
  size -= fixed_size;

  const Item2 item2 = assign_from_buf<Item2>(data);
  data += fixed_size;
  size -= fixed_size;

  auto format_str = fmt::string_view(as_chars(data), size);
#if FMT_FUZZ_FORMAT_TO_STRING
  std::string message = fmt::format(format_str, item1, item2);
#else
  auto buf = fmt::memory_buffer();
  fmt::format_to(std::back_inserter(buf), format_str, item1, item2);
#endif
}

// For dynamic dispatching to an explicit instantiation.
template <typename Callback> void invoke(int index, Callback callback) {
  switch (index) {
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
  case 13:
    using ptr = void*;
    callback(ptr());
    break;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size <= 3) return 0;

  // Switch types depending on the first byte of the input.
  const auto type1 = data[0] & 0x0F;
  const auto type2 = (data[0] & 0xF0) >> 4;
  data++;
  size--;
  try {
    invoke(type1, [=](auto param1) {
      invoke(type2, [=](auto param2) {
        invoke_fmt<decltype(param1), decltype(param2)>(data, size);
      });
    });
  } catch (std::exception&) {
  }
  return 0;
}
