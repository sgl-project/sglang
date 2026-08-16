/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_LAYOUT_CUH_
#define FLASHINFER_LAYOUT_CUH_

#include <cstdint>
#include <string>
#include <tuple>

namespace flashinfer {

/*!
 * \brief The Layout of QKV matrices
 */
enum class QKVLayout {
  // [seq_len, num_heads, head_dim]
  kNHD = 0U,
  // [num_heads, seq_len, head_dim]
  kHND = 1U,
};

__host__ __device__ __forceinline__ size_t get_elem_offset_impl(size_t elem_idx, size_t head_idx,
                                                                size_t feat_idx, size_t stride_n,
                                                                size_t stride_h) {
  return elem_idx * stride_n + head_idx * stride_h + feat_idx;
}

__host__ __forceinline__ auto get_qkv_strides(QKVLayout kv_layout, uint32_t kv_len,
                                              uint32_t num_qo_heads, uint32_t num_kv_heads,
                                              uint32_t head_dim) {
  const uint32_t q_stride_n = num_qo_heads * head_dim, q_stride_h = head_dim,
                 kv_stride_n = (kv_layout == QKVLayout::kNHD) ? num_kv_heads * head_dim : head_dim,
                 kv_stride_h = (kv_layout == QKVLayout::kNHD) ? head_dim : kv_len * head_dim;
  return std::make_tuple(q_stride_n, q_stride_h, kv_stride_n, kv_stride_h);
}

struct tensor_info_t {
  uint32_t qo_len;
  uint32_t kv_len;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t q_stride_n;
  uint32_t q_stride_h;
  uint32_t kv_stride_n;
  uint32_t kv_stride_h;
  uint32_t head_dim;
  __host__ __device__ __forceinline__ tensor_info_t(uint32_t qo_len, uint32_t kv_len,
                                                    uint32_t num_qo_heads, uint32_t num_kv_heads,
                                                    uint32_t q_stride_n, uint32_t q_stride_h,
                                                    uint32_t kv_stride_n, uint32_t kv_stride_h,
                                                    uint32_t head_dim)
      : qo_len(qo_len),
        kv_len(kv_len),
        num_qo_heads(num_qo_heads),
        num_kv_heads(num_kv_heads),
        q_stride_n(q_stride_n),
        q_stride_h(q_stride_h),
        kv_stride_n(kv_stride_n),
        kv_stride_h(kv_stride_h),
        head_dim(head_dim) {}

  __host__ __device__ __forceinline__ tensor_info_t(uint32_t qo_len, uint32_t kv_len,
                                                    uint32_t num_qo_heads, uint32_t num_kv_heads,
                                                    QKVLayout kv_layout, uint32_t head_dim)
      : qo_len(qo_len),
        kv_len(kv_len),
        num_qo_heads(num_qo_heads),
        num_kv_heads(num_kv_heads),
        head_dim(head_dim) {
    q_stride_n = num_qo_heads * head_dim;
    q_stride_h = head_dim;
    kv_stride_n = (kv_layout == QKVLayout::kNHD) ? num_kv_heads * head_dim : head_dim;
    kv_stride_h = (kv_layout == QKVLayout::kNHD) ? head_dim : kv_len * head_dim;
  }

  __host__ __device__ __forceinline__ size_t get_q_elem_offset(uint32_t qo_idx,
                                                               uint32_t qo_head_idx,
                                                               uint32_t feat_idx) const {
    return get_elem_offset_impl(qo_idx, qo_head_idx, feat_idx, q_stride_n, q_stride_h);
  }

  __host__ __device__ __forceinline__ size_t get_o_elem_offset(uint32_t qo_idx,
                                                               uint32_t qo_head_idx,
                                                               uint32_t feat_idx) const {
    return get_elem_offset_impl(qo_idx, qo_head_idx, feat_idx, num_qo_heads * head_dim, head_dim);
  }

  __host__ __device__ __forceinline__ size_t get_kv_elem_offset(uint32_t kv_idx,
                                                                uint32_t kv_head_idx,
                                                                uint32_t feat_idx) const {
    return get_elem_offset_impl(kv_idx, kv_head_idx, feat_idx, kv_stride_n, kv_stride_h);
  }

  __host__ __device__ __forceinline__ uint32_t get_group_size() const {
    return num_qo_heads / num_kv_heads;
  }
};

/*!
 * \brief Convert QKVLayout to string
 * \param layout The QKVLayout to convert
 */
inline std::string QKVLayoutToString(const QKVLayout& layout) {
  switch (layout) {
    case QKVLayout::kNHD:
      return "NHD";
    case QKVLayout::kHND:
      return "HND";
    default:
      return "Unknown";
  }
}

}  // namespace flashinfer
#endif  // FLASHINFER_LAYOUT_CUH_
