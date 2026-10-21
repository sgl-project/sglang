#pragma once

#include <cstdint>

namespace sglang {

namespace device::engram::config {

inline constexpr uint32_t kNumCols = 24;                          // 3 n-gram sizes x 8 heads
inline constexpr uint32_t kRowBytes = 256;                        // head_dim fp8 e4m3 values
inline constexpr uint32_t kScaleBytes = 8;                        // one e8m0 per 32 values
inline constexpr uint32_t kTokenScales = kNumCols * kScaleBytes;  // 192 scale columns per token
inline constexpr uint32_t kSfTileBytes = 512;
inline constexpr uint32_t kSfTilesPerRow = kTokenScales / 4;  // 48 column tiles

}  // namespace device::engram::config

}  // namespace sglang
