#include "utils.metal"
#include <metal_stdlib>

using namespace metal;

template <typename T>
[[kernel]] void copy_blocks(device T *key_cache [[buffer(0)]],
                            device T *value_cache [[buffer(1)]],
                            const device int64_t *block_mapping [[buffer(2)]],
                            device const int &numel_per_block_key,
                            device const int &numel_per_block_value,
                            uint gid [[thread_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint threads_per_threadgroup
                            [[threads_per_threadgroup]]) {
  const int pair_idx = gid;

  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset_key = src_block_number * numel_per_block_key;
  const int64_t dst_block_offset_key = dst_block_number * numel_per_block_key;

  // Copy key cache blocks
  for (int i = tid; i < numel_per_block_key; i += threads_per_threadgroup) {
    int64_t src_offset = src_block_offset_key + i;
    int64_t dst_offset = dst_block_offset_key + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }

  const int64_t src_block_offset_value =
      src_block_number * numel_per_block_value;
  const int64_t dst_block_offset_value =
      dst_block_number * numel_per_block_value;

  // Copy value cache blocks
  for (int i = tid; i < numel_per_block_value; i += threads_per_threadgroup) {
    int64_t src_offset = src_block_offset_value + i;
    int64_t dst_offset = dst_block_offset_value + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}

#define instantiate_copy_blocks(type)                                          \
  template [[host_name("copy_blocks_" #type)]] [[kernel]] void                 \
  copy_blocks<type>(device type * key_cache_ptrs [[buffer(0)]],                \
                    device type * value_cache_ptrs [[buffer(1)]],              \
                    const device int64_t *block_mapping [[buffer(2)]],         \
                    device const int &numel_per_block_key,                     \
                    device const int &numel_per_block_value,                   \
                    uint gid [[thread_position_in_grid]],                      \
                    uint tid [[thread_position_in_threadgroup]],               \
                    uint threads_per_threadgroup [[threads_per_threadgroup]]);

instantiate_copy_blocks(float);
instantiate_copy_blocks(bfloat16_t);
instantiate_copy_blocks(half);
instantiate_copy_blocks(uchar);