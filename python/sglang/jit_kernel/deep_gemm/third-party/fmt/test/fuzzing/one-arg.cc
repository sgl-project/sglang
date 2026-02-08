// Copyright (c) 2019, Paul Dreik
// For the license information refer to format.h.

#include <fmt/chrono.h>

#include <cstdint>
#include <exception>

#include "fuzzer-common.h"

template <typename T, typename Repr> const T* from_repr(const Repr& r) {
  return &r;
}

template <> const std::tm* from_repr<std::tm>(const std::time_t& t) {
  return std::localtime(&t);
}

template <typename T, typename Repr = T>
void invoke_fmt(const uint8_t* data, size_t size) {
  static_assert(sizeof(Repr) <= fixed_size, "Nfixed is too small");
  if (size <= fixed_size) return;
  auto repr = assign_from_buf<Repr>(data);
  const T* value = from_repr<T>(repr);
  if (!value) return;
  data += fixed_size;
  size -= fixed_size;
  data_to_string format_str(data, size);
  try {
#if FMT_FUZZ_FORMAT_TO_STRING
    std::string message = fmt::format(format_str.get(), *value);
#else
    auto buf = fmt::memory_buffer();
    fmt::format_to(std::back_inserter(buf), format_str.get(), *value);
#endif
  } catch (std::exception&) {
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size <= 3) return 0;

  const auto first = data[0];
  data++;
  size--;

  switch (first) {
  case 0:
    invoke_fmt<bool>(data, size);
    break;
  case 1:
    invoke_fmt<char>(data, size);
    break;
  case 2:
    invoke_fmt<unsigned char>(data, size);
    break;
  case 3:
    invoke_fmt<signed char>(data, size);
    break;
  case 4:
    invoke_fmt<short>(data, size);
    break;
  case 5:
    invoke_fmt<unsigned short>(data, size);
    break;
  case 6:
    invoke_fmt<int>(data, size);
    break;
  case 7:
    invoke_fmt<unsigned int>(data, size);
    break;
  case 8:
    invoke_fmt<long>(data, size);
    break;
  case 9:
    invoke_fmt<unsigned long>(data, size);
    break;
  case 10:
    invoke_fmt<float>(data, size);
    break;
  case 11:
    invoke_fmt<double>(data, size);
    break;
  case 12:
    invoke_fmt<long double>(data, size);
    break;
  case 13:
    invoke_fmt<std::tm, std::time_t>(data, size);
    break;
  }
  return 0;
}
