#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>

namespace sglang {

// `topk` is a *runtime* value (<= kMaxTopK), so one module serves every k. It
// used to be baked in via -DSGL_TOPK, which built a separate module per k --
// and because `kTopK` came from a macro rather than a template parameter, both
// modules exported identically mangled symbols. The function-local static in
// setup_kernel_smem_once() is emitted as STB_GNU_UNIQUE, which the loader
// merges across every loaded object, so whichever module was used second
// skipped its cudaFuncSetAttribute opt-in and then failed to launch with 64 KB
// of dynamic shared memory ("invalid argument").
constexpr uint32_t kMaxTopK = 1024;
// Fixed, and deliberately not tied to `topk`: run_cumsum() and the histogram
// init below index up to RADIX + 1 == 257 threads, so a block sized after a
// small topk would silently skip part of the histogram.
constexpr uint32_t kTopKBlockSize = kMaxTopK;
#ifdef USE_ROCM
// Match the ROCm AOT kernel. gfx1100 exposes 64 KiB LDS per workgroup, and
// this kernel also owns static histograms/counters and the top-k index buffer.
// Reserving the CUDA path's full 64 KiB dynamically would exceed that limit.
constexpr uint32_t kSMEM = 12 * 1024 * sizeof(uint32_t);  // 48KB (bytes)
#else
constexpr uint32_t kSMEM = 16 * 1024 * sizeof(uint32_t);  // 64KB (bytes)
#endif

struct TopKParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int32_t* __restrict__ raw_indices;  // optional: output raw abs position indices before page transform
  const int64_t score_stride;
  const int64_t page_table_stride;
  uint32_t page_bits;
  uint32_t topk;
};

SGL_DEVICE uint8_t convert_to_uint8(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
#ifdef USE_ROCM
  // IEEE -0.0 and +0.0 compare equal. Normalize their radix keys so the
  // deterministic boundary-tie rule, rather than the sign bit, picks indices.
  if ((bits & 0x7FFFu) == 0) {
    bits = 0;
  }
#endif
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

SGL_DEVICE uint32_t convert_to_uint32(float x) {
  uint32_t bits = __float_as_uint(x);
#ifdef USE_ROCM
  if ((bits & 0x7FFFFFFFu) == 0) {
    bits = 0;
  }
#endif
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

SGL_DEVICE int32_t page_to_indices(const int32_t* __restrict__ page_table, uint32_t i, uint32_t page_bits) {
  const uint32_t mask = (1u << page_bits) - 1u;
  return (page_table[i >> page_bits] << page_bits) | (i & mask);
}

#ifdef USE_ROCM
SGL_DEVICE void canonicalize_topk_indices(
    const float* __restrict__ input, int32_t* values, const uint32_t length, const uint32_t topk) {
  const auto tx = threadIdx.x;
  constexpr uint32_t kMinWaveSize = 32;
  constexpr uint32_t kMaxNumWaves = (kTopKBlockSize + kMinWaveSize - 1) / kMinWaveSize;
  alignas(128) __shared__ uint32_t s_boundary_key;
  alignas(128) __shared__ uint32_t s_num_greater;
  alignas(128) __shared__ uint32_t s_num_equal;
  alignas(128) __shared__ uint32_t s_ties_seen;
  alignas(128) __shared__ uint32_t s_output_count;
  alignas(128) __shared__ uint32_t s_wave_counts[kMaxNumWaves];
  alignas(128) __shared__ uint32_t s_wave_offsets[kMaxNumWaves];

  if (tx == 0) {
    s_boundary_key = ~uint32_t{0};
    s_num_greater = 0;
    s_num_equal = 0;
  }
  __syncthreads();

  for (uint32_t i = tx; i < topk; i += kTopKBlockSize) {
    ::atomicMin(&s_boundary_key, convert_to_uint32(input[values[i]]));
  }
  __syncthreads();

  const auto boundary_key = s_boundary_key;
  const auto lane = tx % warpSize;
  const auto wave = tx / warpSize;
  const auto num_waves = (kTopKBlockSize + warpSize - 1) / warpSize;
  for (uint32_t tile = 0; tile < length; tile += kTopKBlockSize) {
    const auto index = tile + tx;
    uint32_t key = 0;
    if (index < length) {
      key = convert_to_uint32(input[index]);
    }
    const auto greater_mask = static_cast<unsigned long long>(__ballot(index < length && key > boundary_key));
    const auto equal_mask = static_cast<unsigned long long>(__ballot(index < length && key == boundary_key));
    if (lane == 0) {
      ::atomicAdd(&s_num_greater, static_cast<uint32_t>(__popcll(greater_mask)));
      ::atomicAdd(&s_num_equal, static_cast<uint32_t>(__popcll(equal_mask)));
    }
  }
  __syncthreads();

  const auto tie_needed = topk - s_num_greater;
  if (length <= 2 * kTopKBlockSize || s_num_equal > tie_needed) {
    if (tx == 0) {
      s_ties_seen = 0;
      s_output_count = 0;
    }
    __syncthreads();

    // Compact in logical-index order. Exact-score ties crossing the top-k
    // boundary consume the first `tie_needed` indices, matching the stable
    // tie rule without relying on atomic scheduling in radix_topk().
    for (uint32_t tile = 0; tile < length; tile += kTopKBlockSize) {
      const auto index = tile + tx;
      uint32_t key = 0;
      if (index < length) {
        key = convert_to_uint32(input[index]);
      }
      const bool equal = index < length && key == boundary_key;
      const auto equal_mask = static_cast<unsigned long long>(__ballot(equal));
      if (lane == 0) {
        s_wave_counts[wave] = static_cast<uint32_t>(__popcll(equal_mask));
      }
      __syncthreads();

      if (tx == 0) {
        auto offset = s_ties_seen;
        for (uint32_t wave_id = 0; wave_id < num_waves; ++wave_id) {
          s_wave_offsets[wave_id] = offset;
          offset += s_wave_counts[wave_id];
        }
        s_ties_seen = offset;
      }
      __syncthreads();

      const auto lower_mask = lane == 0 ? 0ull : ((1ull << lane) - 1ull);
      const auto tie_rank = s_wave_offsets[wave] + static_cast<uint32_t>(__popcll(equal_mask & lower_mask));
      const bool selected = index < length && (key > boundary_key || (equal && tie_rank < tie_needed));
      const auto selected_mask = static_cast<unsigned long long>(__ballot(selected));
      if (lane == 0) {
        s_wave_counts[wave] = static_cast<uint32_t>(__popcll(selected_mask));
      }
      __syncthreads();

      if (tx == 0) {
        auto offset = s_output_count;
        for (uint32_t wave_id = 0; wave_id < num_waves; ++wave_id) {
          s_wave_offsets[wave_id] = offset;
          offset += s_wave_counts[wave_id];
        }
        s_output_count = offset;
      }
      __syncthreads();

      if (selected) {
        const auto position = s_wave_offsets[wave] + static_cast<uint32_t>(__popcll(selected_mask & lower_mask));
        values[position] = static_cast<int32_t>(index);
      }
      __syncthreads();
    }
    return;
  }

  // The API intentionally leaves top-k unordered. Sort a fixed, power-of-two
  // buffer so runtime top-k values need not themselves be powers of two.
  if (tx >= topk) {
    values[tx] = static_cast<int32_t>(0x7FFFFFFFu);
  }
  __syncthreads();
  for (uint32_t size = 2; size <= kMaxTopK; size <<= 1) {
    for (uint32_t stride = size >> 1; stride > 0; stride >>= 1) {
      const auto peer = tx ^ stride;
      if (peer > tx) {
        const bool ascending = (tx & size) == 0;
        const auto lhs = values[tx];
        const auto rhs = values[peer];
        if ((ascending && lhs > rhs) || (!ascending && lhs < rhs)) {
          values[tx] = rhs;
          values[peer] = lhs;
        }
      }
      __syncthreads();
    }
  }
}
#endif

[[maybe_unused]]
SGL_DEVICE void naive_transform(
    const float* __restrict__,  // unused
    const int32_t* __restrict__ page_table,
    int32_t* __restrict__ indices,
    int32_t* __restrict__ raw_indices,  // optional: output raw abs position indices
    const uint32_t length,
    const uint32_t page_bits,
    const uint32_t topk) {
  if (const auto tx = threadIdx.x; tx < length) {
    indices[tx] = page_to_indices(page_table, tx, page_bits);
    if (raw_indices != nullptr) {
      raw_indices[tx] = tx;
    }
  } else if (tx < topk) {
    indices[tx] = -1;  // fill invalid indices to -1
    if (raw_indices != nullptr) {
      raw_indices[tx] = -1;
    }
  }
}

[[maybe_unused]]
SGL_DEVICE void
radix_topk(const float* __restrict__ input, int32_t* __restrict__ output, const uint32_t length, const uint32_t topk) {
  constexpr uint32_t RADIX = 256;
  constexpr uint32_t BLOCK_SIZE = kTopKBlockSize;
  constexpr uint32_t SMEM_INPUT_SIZE = kSMEM / (2 * sizeof(int32_t));

  alignas(128) __shared__ uint32_t _s_histogram_buf[2][RADIX + 32];
  alignas(128) __shared__ uint32_t s_counter;
  alignas(128) __shared__ uint32_t s_threshold_bin_id;
  alignas(128) __shared__ uint32_t s_num_input[2];
  alignas(128) __shared__ int32_t s_last_remain;
#ifdef USE_ROCM
  alignas(128) __shared__ uint32_t s_num_exact_ties;
#endif

  extern __shared__ uint32_t s_input_idx[][kSMEM / (2 * sizeof(int32_t))];

  const uint32_t tx = threadIdx.x;
  uint32_t remain_topk = topk;
  auto& s_histogram = _s_histogram_buf[0];

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int32_t i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (tx < RADIX) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = _s_histogram_buf[k][tx];
        if (tx + j < RADIX) {
          value += _s_histogram_buf[k][tx + j];
        }
        _s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  // stage 1: 8bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();
  for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();
  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  remain_topk -= s_histogram[threshold_bin + 1];
  if (remain_topk == 0) {
    for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
      const uint32_t bin = convert_to_uint8(input[idx]);
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        output[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
      const float raw_input = input[idx];
      const uint32_t bin = convert_to_uint8(raw_input);
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        output[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        if (pos < SMEM_INPUT_SIZE) {
          [[likely]] s_input_idx[0][pos] = idx;
          const auto bin = convert_to_uint32(raw_input);
          const auto sub_bin = (bin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

#ifdef USE_ROCM
  if (s_num_input[0] > SMEM_INPUT_SIZE) {
    // The fast path keeps the coarse-threshold candidates in LDS. A dense
    // FP16 bin can exceed that bounded buffer (for example at long context),
    // and clipping it silently drops valid high-scoring FP32 values. Re-scan
    // the row one radix byte at a time only for this overflow case. Elements
    // above the FP16 threshold were already emitted and remain disjoint from
    // every set handled below.
    const auto coarse_threshold_bin = threshold_bin;
    uint32_t score_prefix = 0;

    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();
    for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto value = input[idx];
      if (convert_to_uint8(value) == coarse_threshold_bin) {
        const auto bin = (convert_to_uint32(value) >> 24) & 0xFFu;
        ::atomicAdd(&s_histogram[bin], 1);
      }
    }
    __syncthreads();

#pragma unroll 4
    for (int score_round = 0; score_round < 4; ++score_round) {
      run_cumsum();
      if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
        s_threshold_bin_id = tx;
      }
      __syncthreads();

      const auto score_threshold_bin = s_threshold_bin_id;
      const auto num_greater = s_histogram[score_threshold_bin + 1];
      const auto num_equal = s_histogram[score_threshold_bin] - num_greater;
      remain_topk -= num_greater;
      const bool include_equal = remain_topk == num_equal;
      const bool done = remain_topk == 0 || include_equal;
      const auto score_offset = 24 - score_round * 8;
      const auto previous_score_prefix = score_prefix;
      score_prefix |= score_threshold_bin << score_offset;

      if (!done) {
        if (tx < RADIX + 1) {
          s_histogram[tx] = 0;
        }
      }
      __syncthreads();

      for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto value = input[idx];
        if (convert_to_uint8(value) != coarse_threshold_bin) {
          continue;
        }
        const auto exact_key = convert_to_uint32(value);
        if (score_round > 0 && (exact_key >> (score_offset + 8)) != (previous_score_prefix >> (score_offset + 8))) {
          continue;
        }
        const auto bin = (exact_key >> score_offset) & 0xFFu;
        if (bin > score_threshold_bin || (include_equal && bin == score_threshold_bin)) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          output[pos] = idx;
        } else if (!done && bin == score_threshold_bin) {
          if (score_round < 3) {
            const auto next_bin = (exact_key >> (score_offset - 8)) & 0xFFu;
            ::atomicAdd(&s_histogram[next_bin], 1);
          } else {
            const auto index_bin = 0xFFu - ((idx >> 24) & 0xFFu);
            ::atomicAdd(&s_histogram[index_bin], 1);
          }
        }
      }
      __syncthreads();

      if (done) {
        return;
      }
    }

    // The score threshold itself is tied and only remain_topk entries fit.
    // Continue radix selection on the inverted raw index, so larger keys mean
    // smaller logical indices. Unlike the fast path, this never materializes
    // the potentially unbounded tie set in LDS.
    uint32_t index_prefix = 0;
#pragma unroll 4
    for (int index_round = 0; index_round < 4; ++index_round) {
      run_cumsum();
      if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
        s_threshold_bin_id = tx;
      }
      __syncthreads();

      const auto index_threshold_bin = s_threshold_bin_id;
      const auto num_greater = s_histogram[index_threshold_bin + 1];
      const auto num_equal = s_histogram[index_threshold_bin] - num_greater;
      remain_topk -= num_greater;
      const bool include_equal = remain_topk == num_equal;
      const bool done = remain_topk == 0 || include_equal;
      const auto index_offset = 24 - index_round * 8;
      const auto previous_index_prefix = index_prefix;
      index_prefix |= index_threshold_bin << index_offset;

      if (!done) {
        if (tx < RADIX + 1) {
          s_histogram[tx] = 0;
        }
      }
      __syncthreads();

      for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto exact_key = convert_to_uint32(input[idx]);
        if (exact_key != score_prefix) {
          continue;
        }
        const auto inverted_index = ~idx;
        if (index_round > 0 &&
            (inverted_index >> (index_offset + 8)) != (previous_index_prefix >> (index_offset + 8))) {
          continue;
        }
        const auto bin = (inverted_index >> index_offset) & 0xFFu;
        if (bin > index_threshold_bin || (include_equal && bin == index_threshold_bin)) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          output[pos] = idx;
        } else if (!done && bin == index_threshold_bin && index_round < 3) {
          const auto next_bin = (inverted_index >> (index_offset - 8)) & 0xFFu;
          ::atomicAdd(&s_histogram[next_bin], 1);
        }
      }
      __syncthreads();

      if (done) {
        return;
      }
    }
    return;
  }
#endif

  // stage 2: refine with 8bit radix passes
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    const auto r_idx = round % 2;

    // clip here to prevent overflow
    const auto raw_num_input = s_num_input[r_idx];
    const auto num_input = raw_num_input < SMEM_INPUT_SIZE ? raw_num_input : SMEM_INPUT_SIZE;

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = remain_topk - s_histogram[tx + 1];
#ifdef USE_ROCM
      if (round == 3) {
        s_num_exact_ties = s_histogram[tx] - s_histogram[tx + 1];
      }
#endif
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    remain_topk -= s_histogram[threshold_bin + 1];

    if (remain_topk == 0) {
      for (uint32_t i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          output[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
#ifdef USE_ROCM
      if (round == 3 && tx == 0) {
        s_num_input[r_idx ^ 1] = 0;
      }
#endif
      __syncthreads();
      for (uint32_t i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          output[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
#ifdef USE_ROCM
            const auto tie_needed = static_cast<uint32_t>(s_last_remain);
            if (tie_needed == s_num_exact_ties) {
              const auto pos = ::atomicAdd(&s_counter, 1);
              output[pos] = idx;
            } else {
              // A strict subset of an exact-score boundary tie belongs to
              // top-k. Seed an index radix pass so lower logical indices win,
              // matching the exact tie rule in the CUDA top-k v2 kernel.
              const auto index_bin = 0xFFu - ((idx >> 24) & 0xFFu);
              ::atomicAdd(&s_histogram[index_bin], 1);
            }
#else
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              output[topk - pos] = idx;
            }
#endif
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (pos < SMEM_INPUT_SIZE) {
              /// NOTE: (dark) fuse the histogram computation here
              [[likely]] s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin = convert_to_uint32(raw_input);
              const auto sub_bin = (bin >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();

#ifdef USE_ROCM
      if (round == 3 && static_cast<uint32_t>(s_last_remain) < s_num_exact_ties) {
        uint32_t tie_remain = static_cast<uint32_t>(s_last_remain);
        uint32_t source = r_idx;
        uint32_t source_size = num_input;

#pragma unroll 4
        for (int index_round = 0; index_round < 4; ++index_round) {
          run_cumsum();
          if (tx < RADIX && s_histogram[tx] > tie_remain && s_histogram[tx + 1] <= tie_remain) {
            s_threshold_bin_id = tx;
          }
          __syncthreads();

          const auto index_threshold_bin = s_threshold_bin_id;
          const auto num_greater = s_histogram[index_threshold_bin + 1];
          const auto num_equal = s_histogram[index_threshold_bin] - num_greater;
          tie_remain -= num_greater;
          const bool include_equal = tie_remain == num_equal;
          const bool done = tie_remain == 0 || include_equal;
          const auto next = source ^ 1;

          if (!done) {
            if (tx < RADIX + 1) {
              s_histogram[tx] = 0;
            }
            if (tx == 0) {
              s_num_input[next] = 0;
            }
          }
          __syncthreads();

          const auto index_offset = 24 - index_round * 8;
          for (uint32_t i = tx; i < source_size; i += BLOCK_SIZE) {
            const auto idx = s_input_idx[source][i];
            if (index_round == 0 && (convert_to_uint32(input[idx]) & 0xFFu) != threshold_bin) {
              continue;
            }
            const auto index_bin = 0xFFu - ((idx >> index_offset) & 0xFFu);
            if (index_bin > index_threshold_bin || (include_equal && index_bin == index_threshold_bin)) {
              const auto pos = ::atomicAdd(&s_counter, 1);
              output[pos] = idx;
            } else if (!done && index_bin == index_threshold_bin && index_round < 3) {
              const auto pos = ::atomicAdd(&s_num_input[next], 1);
              if (pos < SMEM_INPUT_SIZE) {
                [[likely]] s_input_idx[next][pos] = idx;
                const auto next_bin = 0xFFu - ((idx >> (index_offset - 8)) & 0xFFu);
                ::atomicAdd(&s_histogram[next_bin], 1);
              }
            }
          }
          __syncthreads();

          if (done) {
            break;
          }
          source = next;
          const auto raw_source_size = s_num_input[source];
          source_size = raw_source_size < SMEM_INPUT_SIZE ? raw_source_size : SMEM_INPUT_SIZE;
        }
      }
#endif
    }
  }
}

template <bool kUsePDL>
__global__ void topk_transform_kernel(const __grid_constant__ TopKParams params) {
  const auto &[
    scores, seq_lens, page_table, page_indices, raw_indices, // pointers
    score_stride, page_table_stride, page_bits, topk // sizes
  ] = params;
  const uint32_t work_id = blockIdx.x;

  /// NOTE: dangerous prefetch seq_len before PDL wait
  const uint32_t seq_len = seq_lens[work_id];
  const auto score_ptr = scores + work_id * score_stride;
  const auto page_ptr = page_table + work_id * page_table_stride;
  const auto indices_ptr = page_indices + work_id * topk;
  const auto raw_indices_ptr = raw_indices != nullptr ? raw_indices + work_id * topk : nullptr;

  device::PDLWaitPrimary<kUsePDL>();

  if (seq_len <= topk) {
    naive_transform(score_ptr, page_ptr, indices_ptr, raw_indices_ptr, seq_len, page_bits, topk);
  } else {
    __shared__ int32_t s_topk_indices[kMaxTopK];
    radix_topk(score_ptr, s_topk_indices, seq_len, topk);
#ifdef USE_ROCM
    canonicalize_topk_indices(score_ptr, s_topk_indices, seq_len, topk);
#endif
    const auto tx = threadIdx.x;
    if (tx < topk) {
      indices_ptr[tx] = page_to_indices(page_ptr, s_topk_indices[tx], page_bits);
      if (raw_indices_ptr != nullptr) {
        raw_indices_ptr[tx] = s_topk_indices[tx];
      }
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <auto* f, size_t kMaxDynamicSMEM>
void setup_kernel_smem_once(host::DebugInfo where = {}) {
  [[maybe_unused]]
  static const auto result = [] {
    const auto fptr = std::bit_cast<const void*>(f);
#ifdef USE_ROCM
    return ::hipFuncSetAttribute(fptr, ::hipFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSMEM);
#else
    return ::cudaFuncSetAttribute(fptr, ::cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSMEM);
#endif
  }();
  host::RuntimeDeviceCheck(result, where);
}

template <bool kUsePDL>
struct TopKKernel {
  static constexpr auto kernel = topk_transform_kernel<kUsePDL>;

  static void transform(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView page_indices,
      const uint32_t page_size,
      const tvm::ffi::Optional<tvm::ffi::TensorView> raw_indices) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto S = SymbolicSize{"score_stride"};
    auto P = SymbolicSize{"page_table_stride"};
    auto K = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, -1})  // strided scores
        .with_strides({S, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(scores);
    TensorMatcher({B})  // seq_lens, must be contiguous
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(seq_lens);
    TensorMatcher({B, -1})  // strided page table
        .with_strides({P, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_table);
    TensorMatcher({B, K})  // output, must be contiguous
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_indices);

    int32_t* raw_indices_ptr = nullptr;
    if (raw_indices.has_value()) {
      TensorMatcher({B, K})  // optional raw indices output, must be contiguous
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(raw_indices.value());
      raw_indices_ptr = static_cast<int32_t*>(raw_indices.value().data_ptr());
    }

    RuntimeCheck(std::has_single_bit(page_size), "page_size must be power of 2");
    const auto page_bits = static_cast<uint32_t>(std::countr_zero(page_size));
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 1024]");
    const auto params = TopKParams{
        .scores = static_cast<float*>(scores.data_ptr()),
        .seq_lens = static_cast<int32_t*>(seq_lens.data_ptr()),
        .page_table = static_cast<int32_t*>(page_table.data_ptr()),
        .page_indices = static_cast<int32_t*>(page_indices.data_ptr()),
        .raw_indices = raw_indices_ptr,
        .score_stride = S.unwrap(),
        .page_table_stride = P.unwrap(),
        .page_bits = page_bits,
        .topk = topk,
    };
    constexpr auto kSMEM_ = kSMEM + sizeof(int32_t);  // align up a little
    setup_kernel_smem_once<kernel, kSMEM_>();
    LaunchKernel(batch_size, kTopKBlockSize, device.unwrap(), kSMEM_).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
