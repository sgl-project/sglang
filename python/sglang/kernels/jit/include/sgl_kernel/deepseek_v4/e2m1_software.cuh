#pragma once

#include <sgl_kernel/utils.cuh>

#include <cmath>
#include <cstdint>

namespace sglang::deepseek_v4::e2m1 {

SGL_DEVICE uint8_t encode_rne_satfinite(float x) {
  const float magnitude = fminf(fabsf(x), 6.0f);
  constexpr float kMagnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

  uint8_t best_code = 0;
  float best_distance = magnitude;
#pragma unroll
  for (uint8_t code = 1; code < 8; ++code) {
    const float distance = fabsf(magnitude - kMagnitudes[code]);
    if (distance < best_distance || (distance == best_distance && (code & 1u) == 0)) {
      best_code = code;
      best_distance = distance;
    }
  }
  const uint8_t sign = static_cast<uint8_t>((__float_as_uint(x) >> 28) & 0x8u);
  return sign | best_code;
}

SGL_DEVICE float decode(uint8_t code) {
  float magnitude;
  switch (code & 0x7u) {
    case 0:
      magnitude = 0.0f;
      break;
    case 1:
      magnitude = 0.5f;
      break;
    case 2:
      magnitude = 1.0f;
      break;
    case 3:
      magnitude = 1.5f;
      break;
    case 4:
      magnitude = 2.0f;
      break;
    case 5:
      magnitude = 3.0f;
      break;
    case 6:
      magnitude = 4.0f;
      break;
    default:
      magnitude = 6.0f;
      break;
  }
  return (code & 0x8u) != 0 ? -magnitude : magnitude;
}

SGL_DEVICE uint8_t pack_pair(float low, float high) {
  return encode_rne_satfinite(low) | (encode_rne_satfinite(high) << 4);
}

}  // namespace sglang::deepseek_v4::e2m1
