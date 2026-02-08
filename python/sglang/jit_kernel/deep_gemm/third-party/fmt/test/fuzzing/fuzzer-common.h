// Copyright (c) 2019, Paul Dreik
// For the license information refer to format.h.

#ifndef FUZZER_COMMON_H
#define FUZZER_COMMON_H

#include <fmt/base.h>

#include <cstdint>  // std::uint8_t
#include <cstring>  // memcpy
#include <vector>

// One can format to either a string, or a buffer. The latter is faster, but
// one may be interested in formatting to a string instead to verify it works
// as intended. To avoid a combinatoric explosion, select this at compile time
// instead of dynamically from the fuzz data.
#define FMT_FUZZ_FORMAT_TO_STRING 0

// If {fmt} is given a buffer that is separately allocated, chances that address
// sanitizer detects out of bound reads is much higher. However, it slows down
// the fuzzing.
#define FMT_FUZZ_SEPARATE_ALLOCATION 1

// The size of the largest possible type in use.
// To let the the fuzzer mutation be efficient at cross pollinating between
// different types, use a fixed size format. The same bit pattern, interpreted
// as another type, is likely interesting.
constexpr auto fixed_size = 16;

// Casts data to a char pointer.
template <typename T> inline const char* as_chars(const T* data) {
  return reinterpret_cast<const char*>(data);
}

// Casts data to a byte pointer.
template <typename T> inline const std::uint8_t* as_bytes(const T* data) {
  return reinterpret_cast<const std::uint8_t*>(data);
}

// Blits bytes from data to form an (assumed trivially constructible) object
// of type Item.
template <class Item> inline Item assign_from_buf(const std::uint8_t* data) {
  auto item = Item();
  std::memcpy(&item, data, sizeof(Item));
  return item;
}

// Reads a boolean value by looking at the first byte from data.
template <> inline bool assign_from_buf<bool>(const std::uint8_t* data) {
  return *data != 0;
}

struct data_to_string {
#if FMT_FUZZ_SEPARATE_ALLOCATION
  std::vector<char> buffer;

  data_to_string(const uint8_t* data, size_t size, bool add_terminator = false)
      : buffer(size + (add_terminator ? 1 : 0)) {
    if (size) {
      std::memcpy(buffer.data(), data, size);
    }
  }

  fmt::string_view get() const { return {buffer.data(), buffer.size()}; }
#else
  fmt::string_view sv;

  data_to_string(const uint8_t* data, size_t size, bool = false)
      : str(as_chars(data), size) {}

  fmt::string_view get() const { return sv; }
#endif

  const char* data() const { return get().data(); }
};

#endif  // FUZZER_COMMON_H
