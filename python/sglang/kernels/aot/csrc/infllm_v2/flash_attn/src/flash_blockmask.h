#pragma once

namespace flash {

class fwdIterator {
 public:
  template <typename Params, typename BlockInfo>
  __device__ fwdIterator(
      const Params& params,
      const BlockInfo& binfo,
      const int kBlockM,
      const int kBlockN,
      const int batch_idx,
      const int head_idx,
      const int loop_step_idx,
      int n_block_min,
      int n_block_max) {  // row first
    if (params.blockmask == nullptr) {
      blockmask_ptr = nullptr;
      return;
    }
    this->cache_seqlen_k = binfo.actual_seqlen_k - binfo.actual_seqlen_q / params.m_block_dim;
    this->max_block_idx = cute::ceil_div(binfo.actual_seqlen_k, params.n_block_dim);
    this->m_block_dim = params.m_block_dim;
    this->n_block_dim = params.n_block_dim;
    this->n_block_min = n_block_min;
    this->n_block_max = n_block_max;
    this->batch_idx = batch_idx;  // Store batch_idx for debugging
    this->head_idx = head_idx;

    // Calculate the offset for the uint64 blockmask
    const int num_blocks_m = params.num_blocks_m;
    const int num_blocks_n = params.num_blocks_n;
    const int uint64_per_row = (num_blocks_n + 64 - 1) / 64;
    const int row_offset = params.cu_seqlens_q != nullptr ? binfo.blockmask_q_offset(m_block_dim, batch_idx)
                                                          : batch_idx * params.num_k_heads * params.num_blocks_m;

    blockmask_ptr = params.blockmask + head_idx * params.num_blocks_m * uint64_per_row + row_offset * uint64_per_row +
                    loop_step_idx * uint64_per_row;

    // printf("blockmask_ptr = %d\n", blockmask_ptr);

    const int q_block_idx = loop_step_idx + cache_seqlen_k;
  }

  __device__ int max_no_larger(int target) const {
    if (blockmask_ptr == nullptr) {
      // printf("blockmask_ptr is nullptr\n");
      return target;
    }
    // printf("blockmask_ptr is NOT!!!! nullptr\n");
    if (max_block_idx == 0) {
      return -1;
    };

    // 目标值不能超过最大块索引
    target = min(target, max_block_idx - 1);

    // 计算相对于当前q_bit_position的实际位置
    int target_bit_pos = target;

    // 确定此块在哪个uint64中
    int uint64_offset = target_bit_pos / 64;

    // 确定此块在uint64中的哪一位
    int bit_pos = target_bit_pos % 64;

    // 创建一个掩码，保留target及更低位的所有位
    uint64_t mask = bit_pos != 63 ? (1ULL << (bit_pos + 1)) - 1 : 0xFFFFFFFFFFFFFFFFULL;

    // 检查当前uint64中target及以下的位
    uint64_t value = blockmask_ptr[uint64_offset] & mask;

    // 如果当前uint64中有设置的位
    int result = -1;
    if (value != 0) {
      // 找到最高位的1（即不大于target的最大设置位）
      int highest_bit = 63 - __clzll(value);  // __clzll计算前导0的数量
      result = highest_bit + (uint64_offset * 64);
    } else {
      // 如果当前uint64中没有找到，检查更低的uint64块
      for (int i = uint64_offset - 1; i >= 0; i--) {
        value = blockmask_ptr[i];
        if (value != 0) {
          // 找到最高位的1
          int highest_bit = 63 - __clzll(value);
          // 计算相对于q_bit_position的偏移
          result = highest_bit + (i * 64);
          break;
        }
      }
    }

    // 没有找到设置位
    return result;
  }

  uint64_t* blockmask_ptr;
  int row_offset;      // 行偏移量
  int uint64_per_row;  // 每行使用的uint64数量
  int cache_seqlen_k;
  int max_block_idx;
  int m_block_dim, n_block_dim;
  int n_block_min, n_block_max;
  int batch_idx, head_idx;
};

}  // namespace flash
