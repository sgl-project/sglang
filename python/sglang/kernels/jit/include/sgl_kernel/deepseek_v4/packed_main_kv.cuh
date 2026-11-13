#pragma once

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/deepseek_v4/kv_layout_id.cuh>

#include <cstdint>

namespace sglang::deepseek_v4 {

struct PackedMainKVTraits {
  static constexpr KVLayout kLayout = KVLayout::DSV41_MAIN_KV_E2M1_BLOCK16_ROPE_BF16_V1;
  static constexpr int64_t kPayloadBytesPerSlot = 224;
  static constexpr int64_t kScaleBytesPerSlot = 32;
  static constexpr int64_t kRopeBytesPerSlot = 128;
  static constexpr int64_t kBytesPerSlot = kPayloadBytesPerSlot + kScaleBytesPerSlot + kRopeBytesPerSlot;
  static constexpr int64_t kValidScalesPerSlot = 28;
};

template <int64_t kPageSlots>
struct PackedMainKVPaged {
  using Traits = PackedMainKVTraits;
  static_assert(kPageSlots == 128 || kPageSlots == 256);

  static constexpr int64_t kPayloadBase = 0;
  static constexpr int64_t kScaleBase = kPageSlots * Traits::kPayloadBytesPerSlot;
  static constexpr int64_t kRopeBase = kScaleBase + kPageSlots * Traits::kScaleBytesPerSlot;
  static constexpr int64_t kPageBytes = kPageSlots * Traits::kBytesPerSlot;

  struct Row {
    uint8_t* payload;
    uint8_t* scales;
    uint8_t* rope;
  };

  template <typename LocT>
  SGL_DEVICE static Row row(uint8_t* storage, LocT loc) {
    const int64_t index = static_cast<int64_t>(loc);
    const int64_t page_id = index / kPageSlots;
    const int64_t slot = index % kPageSlots;
    uint8_t* page = storage + page_id * kPageBytes;
    return {
        page + kPayloadBase + slot * Traits::kPayloadBytesPerSlot,
        page + kScaleBase + slot * Traits::kScaleBytesPerSlot,
        page + kRopeBase + slot * Traits::kRopeBytesPerSlot,
    };
  }
};

}  // namespace sglang::deepseek_v4
