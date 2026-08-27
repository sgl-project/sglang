#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

namespace device::hisparse {

/// NOTE: We call nope+rope as a "value" here.
/// Paged C4 cache layout:
/// VALUE 0, VALUE 1, ..., VALUE 63,
/// SCALE 0, SCALE 1, ..., SCALE 63,
/// [Padding to align to 576 bytes]
inline constexpr int64_t kPageSize = 64;
inline constexpr int64_t kPageBits = 6;  // log2(kPageSize)
inline constexpr int64_t kValueBytes = 576;
inline constexpr int64_t kScaleBytes = 8;
/// NOTE: FlashMLA requires each page to be aligned to 576 bytes
inline constexpr int64_t kItemBytes = kValueBytes + kScaleBytes;
inline constexpr int64_t kPageBytes = host::div_ceil(kItemBytes * kPageSize, 576) * 576;
inline constexpr int64_t kScaleOffset = kValueBytes * kPageSize;

struct PointerInfo {
  int64_t* value_ptr;
  int64_t* scale_ptr;
};

SGL_DEVICE PointerInfo get_pointer_paged(void* cache, int32_t index) {
  using namespace device;
  static_assert(1 << kPageBits == kPageSize);
  const int32_t page_num = index >> kPageBits;
  const int32_t page_offset = index & (kPageSize - 1);
  const auto page_ptr = pointer::offset(cache, page_num * kPageBytes);
  const auto value_ptr = pointer::offset(page_ptr, page_offset * kValueBytes);
  const auto scale_ptr = pointer::offset(page_ptr, kScaleOffset + page_offset * kScaleBytes);
  return {static_cast<int64_t*>(value_ptr), static_cast<int64_t*>(scale_ptr)};
}

SGL_DEVICE void transfer_item(void* dst_cache, void* src_cache, const int32_t dst_index, const int32_t src_index) {
  const auto [dst_value_ptr, dst_scale_ptr] = get_pointer_paged(dst_cache, dst_index);
  const auto [src_value_ptr, src_scale_ptr] = get_pointer_paged(src_cache, src_index);

  int64_t local_items[2];
  const int64_t* tail_src_ptr;
  int64_t* tail_dst_ptr;

  const int32_t lane_id = threadIdx.x % 32;

  for (int i = 0; i < 2; ++i) {
    const auto j = lane_id + i * 32;
    local_items[i] = src_value_ptr[j];
  }

  if (lane_id < 8) {  // handle the tail element safely
    const auto last_id = 64 + lane_id;
    tail_src_ptr = src_value_ptr + last_id;
    tail_dst_ptr = dst_value_ptr + last_id;
  } else {  // broadcast load/store is safe
    tail_src_ptr = src_scale_ptr;
    tail_dst_ptr = dst_scale_ptr;
  }

  const auto tail_item = *tail_src_ptr;

  // store first 512 bytes of value
  for (int i = 0; i < 2; ++i) {
    const auto j = lane_id + i * 32;
    dst_value_ptr[j] = local_items[i];
  }

  // store the tail element
  *tail_dst_ptr = tail_item;
}

}  // namespace device::hisparse
