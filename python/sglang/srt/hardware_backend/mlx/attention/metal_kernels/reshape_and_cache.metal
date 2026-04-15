#include "utils.metal"
#include <metal_stdlib>

using namespace metal;

template <typename KV_T, typename CACHE_T>
inline CACHE_T to_cache(KV_T v) = delete;

template <> inline uchar to_cache<float, uchar>(float v) {
  return 0; // TODO
}

template <> inline uchar to_cache<bfloat16_t, uchar>(bfloat16_t v) {
  return 0; // TODO
}

template <> inline uchar to_cache<half, uchar>(half v) {
  return 0; // TODO
}

template <> inline float to_cache<float, float>(float v) { return v; }

template <> inline bfloat16_t to_cache<bfloat16_t, bfloat16_t>(bfloat16_t v) {
  return v;
}

template <> inline half to_cache<half, half>(half v) { return v; }

constant bool use_fp8_scales [[function_constant(10)]];

// Cache layout: [num_blocks, block_size, num_heads, head_size]
// Both key and value caches use the same token-contiguous layout.
template <typename KV_T, typename CACHE_T>
[[kernel]] void reshape_and_cache(
    const device KV_T *__restrict__ key
    [[buffer(0)]], // [num_tokens, num_heads, head_size]
    const device KV_T *__restrict__ value
    [[buffer(1)]], // [num_tokens, num_heads, head_size]
    device CACHE_T *__restrict__ key_cache
    [[buffer(2)]], // [num_blocks, block_size, num_heads, head_size]
    device CACHE_T *__restrict__ value_cache
    [[buffer(3)]], // [num_blocks, block_size, num_heads, head_size]
    const device int64_t *__restrict__ slot_mapping
    [[buffer(4)]], // [num_tokens]
    const device float *__restrict__ k_scale
    [[buffer(5), function_constant(use_fp8_scales)]], // [1]
    const device float *__restrict__ v_scale
    [[buffer(6), function_constant(use_fp8_scales)]], // [1]
    device const int &key_stride [[buffer(7)]],
    device const int &value_stride [[buffer(8)]],
    device const int &num_heads [[buffer(9)]],
    device const int &head_size [[buffer(10)]],
    device const int &block_size [[buffer(11)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
  const int64_t token_idx = gid;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = tid; i < n; i += threads_per_threadgroup) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;

    // Target index: [block_idx, block_offset, head_idx, head_offset]
    const int64_t tgt_idx =
        block_idx * block_size * num_heads * head_size +
        block_offset * num_heads * head_size +
        head_idx * head_size +
        head_offset;

    if (use_fp8_scales) {
      key_cache[tgt_idx] =
          to_cache<KV_T, CACHE_T>(KV_T((float)key[src_key_idx] / *k_scale));
      value_cache[tgt_idx] =
          to_cache<KV_T, CACHE_T>(KV_T((float)value[src_value_idx] / *v_scale));
    } else {
      key_cache[tgt_idx] = to_cache<KV_T, CACHE_T>(key[src_key_idx]);
      value_cache[tgt_idx] =
          to_cache<KV_T, CACHE_T>(value[src_value_idx]);
    }
  }
}

#define instantiate_reshape_and_cache(kv_type, cache_type)                     \
  template [[host_name("reshape_and_cache_kv_" #kv_type                        \
                       "_cache_" #cache_type)]] [[kernel]] void                \
  reshape_and_cache<kv_type, cache_type>(                                      \
      const device kv_type *__restrict__ key [[buffer(0)]],                    \
      const device kv_type *__restrict__ value [[buffer(1)]],                  \
      device cache_type *__restrict__ key_cache [[buffer(2)]],                 \
      device cache_type *__restrict__ value_cache [[buffer(3)]],               \
      const device int64_t *__restrict__ slot_mapping [[buffer(4)]],           \
      const device float *__restrict__ k_scale                                 \
      [[buffer(5), function_constant(use_fp8_scales)]],                        \
      const device float *__restrict__ v_scale                                 \
      [[buffer(6), function_constant(use_fp8_scales)]],                        \
      device const int &key_stride [[buffer(7)]],                              \
      device const int &value_stride [[buffer(8)]],                            \
      device const int &num_heads [[buffer(9)]],                               \
      device const int &head_size [[buffer(10)]],                              \
      device const int &block_size [[buffer(11)]],                             \
      uint gid [[threadgroup_position_in_grid]],                               \
      uint tid [[thread_position_in_threadgroup]],                             \
      uint threads_per_threadgroup [[threads_per_threadgroup]]);

instantiate_reshape_and_cache(float, float);
instantiate_reshape_and_cache(bfloat16_t, bfloat16_t);
instantiate_reshape_and_cache(half, half);

// instantiate_reshape_and_cache(float, uchar);
// instantiate_reshape_and_cache(bfloat16_t, uchar);
// instantiate_reshape_and_cache(half, uchar);