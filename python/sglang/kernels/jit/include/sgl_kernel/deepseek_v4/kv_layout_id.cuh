#pragma once

#include <cstdint>

namespace sglang::deepseek_v4 {

enum class KVLayout : int32_t {
  V4 = 0,
  V41 = 1,
  V41_FP4 = 2,
  UNIFORM_FP8 = 3,
  V41_NEW = 4,
  V41_FP4_NEW = 5,
  DSV41_MAIN_KV_E2M1_BLOCK16_ROPE_BF16_V1 = 6,
};

}  // namespace sglang::deepseek_v4
