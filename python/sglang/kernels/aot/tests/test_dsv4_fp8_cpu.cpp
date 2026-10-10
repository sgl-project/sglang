// Standalone, CPU-only test of the REAL production integer encoder. Example:
// From the repo root (join these two command lines):
// c++ -std=c++17 -O2 -Ipython/sglang/kernels/jit/include
//   python/sglang/kernels/aot/tests/test_dsv4_fp8_cpu.cpp -o /tmp/test_dsv4_fp8
// /tmp/test_dsv4_fp8
// test_dsv4_fp8_cpu.py additionally extracts and tests the actual AOT/JIT
// clipping/packing wrappers, without requiring HIP, CUDA, torch or a GPU.
#include <sgl_kernel/deepseek_v4/fp8_e4m3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SGL_FP8_WRAPPER_HEADER
#include SGL_FP8_WRAPPER_HEADER
#endif

using sglang::deepseek_v4::fp8::f32_to_fp8_e4m3_bits;

static uint32_t bits_of(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static float float_of(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// Independent golden decoder: the mathematical E4M3 definition, using exact
// powers of two. This does NOT reverse or reproduce the production encoder.
static double decode(uint8_t byte, bool fnuz) {
  const int magnitude = byte & 127;
  if ((fnuz && byte == 128) || (!fnuz && magnitude == 127)) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return (!fnuz && (byte & 128)) ? -nan : nan;
  }
  const int exponent = magnitude / 8;
  const int fraction = magnitude % 8;
  const int bias = fnuz ? 8 : 7;
  const double value = exponent == 0 ? std::ldexp(double(fraction), 1 - bias - 3)
                                     : std::ldexp(1.0 + double(fraction) / 8.0, exponent - bias);
  return (byte & 128) ? -value : value;
}

struct Golden {
  bool fnuz;
  std::vector<double> positive;

  explicit Golden(bool use_fnuz) : fnuz(use_fnuz) {
    for (int b = 0; b <= (fnuz ? 127 : 126); ++b)
      positive.push_back(decode(b, fnuz));
  }

  uint8_t encode(float value) const {
    const uint8_t sign = std::signbit(value) ? 128 : 0;
    if (std::isnan(value)) return fnuz ? 128 : (sign | 127);
    const double x = std::abs(double(value));
    if (x >= positive.back()) return sign | uint8_t(positive.size() - 1);
    const auto upper = std::lower_bound(positive.begin(), positive.end(), x);
    int code = int(upper - positive.begin());
    if (code > 0) {
      const double midpoint = (positive[code - 1] + positive[code]) / 2;
      if (x < midpoint || (x == midpoint && (code & 1))) --code;
    }
    return (fnuz && code == 0) ? 0 : (sign | code);
  }

  uint8_t clipped(float value) const {
    // Model the caller's explicit contract independently of fmin/fmax and the
    // production bit encoder. In particular, NaN maps to POSITIVE clamp max.
    const float limit = fnuz ? 224.0f : 448.0f;
    if (std::isnan(value)) value = limit;
    if (value > limit) value = limit;
    if (value < -limit) value = -limit;
    return encode(value);
  }
};

static void expect(uint32_t bits, uint32_t actual, uint32_t expected, const char* where) {
  if (actual == expected) return;
  std::cerr << where << " input=0x" << std::hex << bits << " actual=0x" << actual << " expected=0x" << expected
            << std::dec << '\n';
  throw std::runtime_error("FP8 byte mismatch");
}

static std::vector<uint32_t> corpus(const Golden& fn, const Golden& fnuz) {
  std::vector<uint32_t> values;
  const auto add = [&](float x) {
    values.push_back(bits_of(x));
    values.push_back(bits_of(-x));
  };
  for (const auto* golden : {&fn, &fnuz}) {
    // All 256 byte decodes, including both FN NaNs, FNUZ NaN and signed zeros.
    for (int b = 0; b < 256; ++b)
      add(float(decode(b, golden->fnuz)));
    for (size_t b = 0; b < golden->positive.size(); ++b) {
      const float x = float(golden->positive[b]);
      add(x);
      add(std::nextafter(x, -std::numeric_limits<float>::infinity()));
      add(std::nextafter(x, std::numeric_limits<float>::infinity()));
      if (b == 0) continue;
      const float midpoint = float((golden->positive[b - 1] + golden->positive[b]) / 2);
      // Every adjacent-code midpoint and the immediately adjacent binary32s,
      // both signs: sticky bits, even/odd ties, and all exponent carries.
      add(midpoint);
      add(std::nextafter(midpoint, 0.0f));
      add(std::nextafter(midpoint, std::numeric_limits<float>::infinity()));
    }
  }
  // All bfloat16 bit patterns: includes infinities and MANY signed NaN payloads.
  for (uint32_t b = 0; b < 65536; ++b)
    values.push_back(b << 16);
  // All binary32 exponents with strategically chosen round/sticky bit patterns.
  for (uint32_t e = 0; e < 256; ++e) {
    for (uint32_t m : {0u, 1u, 0x7ffffu, 0x80000u, 0x80001u, 0xfffffu, 0x100000u, 0x780000u, 0x7fffffu}) {
      values.push_back((e << 23) | m);
      values.push_back(0x80000000u | (e << 23) | m);
    }
  }
  // Reproducible raw binary32 samples (including very small/large magnitudes).
  uint32_t state = 0x42c0ffeeu;
  for (int i = 0; i < 100000; ++i) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    values.push_back(state);
  }
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

int main(int argc, char** argv) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
  const Golden fn(false), fnuz(true);
  // Literal known-answer regressions, not just agreement with the decoder.
  const std::array<float, 7> top_fn{256, 288, 320, 352, 384, 416, 448};
  const std::array<float, 8> top_fnuz{128, 144, 160, 176, 192, 208, 224, 240};
  for (size_t i = 0; i < top_fn.size(); ++i) {
    expect(bits_of(top_fn[i]), f32_to_fp8_e4m3_bits<false>(bits_of(top_fn[i])), 0x78 + i, "FN top exponent");
#ifdef SGL_FP8_WRAPPER_HEADER
    expect(bits_of(top_fn[i]), aot::cvt_float_to_fp8_e4m3(top_fn[i]), 0x78 + i, "AOT top exponent");
    expect(bits_of(top_fn[i]), jit_fn::cvt_float_to_fp8_e4m3(top_fn[i]), 0x78 + i, "JIT FN top exponent");
#endif
  }
  for (size_t i = 0; i < top_fnuz.size(); ++i) {
    expect(bits_of(top_fnuz[i]), f32_to_fp8_e4m3_bits<true>(bits_of(top_fnuz[i])), 0x78 + i, "FNUZ top exponent");
#ifdef SGL_FP8_WRAPPER_HEADER
    expect(
        bits_of(top_fnuz[i]),
        jit_fnuz::cvt_float_to_fp8_e4m3(top_fnuz[i]),
        std::min<size_t>(0x78 + i, 0x7e),
        "JIT FNUZ top exponent/clamp");
#endif
  }
  expect(bits_of(1.0625f), f32_to_fp8_e4m3_bits<false>(bits_of(1.0625f)), 0x38, "FN even tie");
  expect(bits_of(1.1875f), f32_to_fp8_e4m3_bits<false>(bits_of(1.1875f)), 0x3a, "FN odd tie");
  expect(bits_of(0.0146484375f), f32_to_fp8_e4m3_bits<false>(bits_of(0.0146484375f)), 8, "FN subnormal carry");
  expect(bits_of(0.00732421875f), f32_to_fp8_e4m3_bits<true>(bits_of(0.00732421875f)), 8, "FNUZ subnormal carry");
  expect(0x80000000u, f32_to_fp8_e4m3_bits<false>(0x80000000u), 0x80, "FN negative zero");
  expect(0x80000000u, f32_to_fp8_e4m3_bits<true>(0x80000000u), 0, "FNUZ negative zero");

  // Exhaust every FP8 byte, and require exact finite roundtrip/canonical NaN.
  for (const auto* golden : {&fn, &fnuz}) {
    for (int b = 0; b < 256; ++b) {
      const float x = float(decode(b, golden->fnuz));
      const uint8_t actual =
          golden->fnuz ? f32_to_fp8_e4m3_bits<true>(bits_of(x)) : f32_to_fp8_e4m3_bits<false>(bits_of(x));
      expect(bits_of(x), actual, std::isnan(x) ? golden->encode(x) : b, "all-byte roundtrip");
    }
  }

  const auto inputs = corpus(fn, fnuz);
  const bool emit = argc == 2 && std::string(argv[1]) == "--emit";
  for (size_t i = 0; i < inputs.size(); ++i) {
    const uint32_t bits = inputs[i];
    const float x = float_of(bits);
    const uint8_t want_fn = fn.encode(x), want_fnuz = fnuz.encode(x);
    expect(bits, f32_to_fp8_e4m3_bits<false>(bits), want_fn, "raw FN");
    expect(bits, f32_to_fp8_e4m3_bits<true>(bits), want_fnuz, "raw FNUZ");
#ifdef SGL_FP8_WRAPPER_HEADER
    expect(bits, aot::cvt_float_to_fp8_e4m3(x), fn.clipped(x), "AOT FN");
    expect(bits, jit_fn::cvt_float_to_fp8_e4m3(x), fn.clipped(x), "JIT FN");
    expect(bits, jit_fnuz::cvt_float_to_fp8_e4m3(x), fnuz.clipped(x), "JIT FNUZ");
    const float y = float_of(inputs[(i + 997) % inputs.size()]);
    const uint16_t packed_fn = fn.clipped(x) | (uint16_t(fn.clipped(y)) << 8);
    const uint16_t packed_fnuz = fnuz.clipped(x) | (uint16_t(fnuz.clipped(y)) << 8);
    expect(bits, aot::pack_fp8(x, y), packed_fn, "AOT pack");
    expect(bits, jit_fn::pack_fp8(x, y), packed_fn, "JIT FN pack");
    expect(bits, jit_fnuz::pack_fp8(x, y), packed_fnuz, "JIT FNUZ pack");
#endif
    if (emit) {
      std::cout << bits << ' ' << unsigned(want_fn) << ' ' << unsigned(want_fnuz) << ' ' << unsigned(fn.clipped(x))
                << ' ' << unsigned(fnuz.clipped(x)) << '\n';
    }
  }
  if (!emit) {
    std::cout << "PASS: FN/FNUZ all 256 bytes each, all adjacent halfways +/-1 ULP, " << inputs.size()
              << " distinct binary32 inputs (including all 65536 BF16 patterns)";
#ifdef SGL_FP8_WRAPPER_HEADER
    std::cout << "; actual AOT/JIT clip + encode + pack wrappers";
#endif
    std::cout << '\n';
  }
}
