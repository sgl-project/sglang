#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cfloat>
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
constexpr uint32_t kSMEM = 16 * 1024 * sizeof(uint32_t);  // 64KB (bytes)

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
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

SGL_DEVICE uint32_t convert_to_uint32(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

SGL_DEVICE int32_t page_to_indices(const int32_t* __restrict__ page_table, uint32_t i, uint32_t page_bits) {
  const uint32_t mask = (1u << page_bits) - 1u;
  return (page_table[i >> page_bits] << page_bits) | (i & mask);
}

// ============================================================
// v3: RegisterTopK-based implementation (1024 threads, 12-bit histogram)
// ============================================================

namespace v3 {
constexpr uint32_t kBlockSize = 1024;
constexpr uint32_t kNumWarps = kBlockSize / 32;
constexpr uint32_t kHistBits = 12;
constexpr uint32_t kHistBins = 1 << kHistBits;  // 4096
constexpr uint32_t kVecsPerThread = 4;
constexpr uint32_t kMax1PassLength = kVecsPerThread * 4 * kBlockSize;  // 16384
constexpr uint32_t kMax2PassLength = 2 * kMax1PassLength;              // 32768
constexpr uint32_t kMaxTies = 1024;
}  // namespace v3

using Vec4 = device::AlignedVector<float, 4>;

struct alignas(16) MatchBinV3 {
  uint32_t bin;
  uint32_t above_count;
  uint32_t equal_count;
};

struct alignas(8) TieV3 {
  uint32_t idx;
  float score;
};

template <uint32_t kBits>
SGL_DEVICE uint32_t extract_coarse_bin_v3(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return key >> (16 - kBits);
}

SGL_DEVICE uint32_t warp_inclusive_sum_v3(uint32_t lane_id, uint32_t val) {
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
    if (lane_id >= offset) val += n;
  }
  return val;
}

SGL_DEVICE uint32_t extract_exact_bin_v3(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// cp.async PTX helpers
SGL_DEVICE void cp_async_cg_shared_global_16(void* smem_dst, const void* global_src) {
  uint32_t dst;
  asm volatile(
      "{ .reg .u64 u64addr;\n\t"
      "  cvta.to.shared.u64 u64addr, %1;\n\t"
      "  cvt.u32.u64 %0, u64addr; }\n"
      : "=r"(dst)
      : "l"(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(global_src));
}

SGL_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}
SGL_DEVICE void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;");
}
// Wait until at most kKeep prior cp.async groups remain in flight.
template <uint32_t kKeep>
SGL_DEVICE void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;" ::"n"(kKeep) : "memory");
}

struct RegisterTopKSmem {
  using HistVec = device::AlignedVector<uint32_t, v3::kHistBins / v3::kBlockSize>;
  alignas(128) uint32_t counter_gt;
  alignas(128) uint32_t counter_eq;
  MatchBinV3 match;
  uint32_t warp_sum[v3::kNumWarps];
  union {
    uint32_t histogram[v3::kHistBins];
    HistVec histogram_vec[v3::kBlockSize];
    TieV3 tie_buffer[v3::kMaxTies];
  };  // 16KB
  alignas(16) float score_buffer[v3::kMax1PassLength];  // 64KB
};

template <bool kIs2Pass>
[[maybe_unused]]
SGL_DEVICE void register_topk(
    const float* __restrict__ scores,
    int32_t* __restrict__ indices,
    const uint32_t length,
    const uint32_t topk,
    void* _smem) {
  auto* smem = static_cast<RegisterTopKSmem*>(_smem);
  const auto tx = threadIdx.x;
  const auto lane_id = tx % 32;
  const auto warp_id = tx / 32;

  // Initialize shared memory histogram
  {
    typename RegisterTopKSmem::HistVec hist_vec;
    hist_vec.fill(0);
    smem->histogram_vec[tx] = hist_vec;
    if (tx == 0) {
      smem->counter_gt = smem->counter_eq = 0;
    }
    __syncthreads();
  }

  // Load scores into registers
  Vec4 local[v3::kVecsPerThread];
#pragma unroll
  for (uint32_t v = 0; v < v3::kVecsPerThread; ++v) {
    const uint32_t base = (tx + v * v3::kBlockSize) * 4;
    if (base >= length) break;
    local[v].load(scores, tx + v * v3::kBlockSize);
  }

  // Fetch the next chunk of scores via cp.async (replaces TMA)
  if constexpr (kIs2Pass) {
    const uint32_t extra_length = length - v3::kMax1PassLength;
    const uint32_t extra_vec4s = (extra_length + 3) / 4;
    for (uint32_t i = tx; i < extra_vec4s; i += v3::kBlockSize) {
      cp_async_cg_shared_global_16(&smem->score_buffer[i * 4], scores + v3::kMax1PassLength + i * 4);
    }
    cp_async_commit();
  }

  // Accumulate histogram via shared-memory atomics
#pragma unroll
  for (uint32_t v = 0; v < v3::kVecsPerThread; ++v) {
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
      if constexpr (!kIs2Pass) {
        const uint32_t idx = (tx + v * v3::kBlockSize) * 4 + e;
        if (idx >= length) goto LABEL_ACC_FINISH;
      }
      atomicAdd(&smem->histogram[extract_coarse_bin_v3<v3::kHistBits>(local[v][e])], 1);
    }
  }
  if constexpr (kIs2Pass) {
    // Wait for cp.async to complete
    cp_async_wait_all();
    __syncthreads();
    for (uint32_t i = tx; i + v3::kMax1PassLength < length; i += v3::kBlockSize) {
      const auto val = smem->score_buffer[i];  // Stall instruction stall
      atomicAdd(&smem->histogram[extract_coarse_bin_v3<v3::kHistBits>(val)], 1);
    }
  }
[[maybe_unused]] LABEL_ACC_FINISH:
  __syncthreads();

  // Phase 2: Exclusive prefix scan -> find threshold bin
  {
    constexpr uint32_t kItems = v3::kHistBins / v3::kBlockSize;
    uint32_t orig[kItems];
    const auto hist_vec = smem->histogram_vec[tx];
    uint32_t tmp_local_sum = 0;

#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      orig[i] = hist_vec[i];
      tmp_local_sum += orig[i];
    }

    const auto warp_inc = warp_inclusive_sum_v3(lane_id, tmp_local_sum);
    const auto warp_exc = warp_inc - tmp_local_sum;
    if (lane_id == 31) {
      smem->warp_sum[warp_id] = warp_inc;
    }

    __syncthreads();

    const auto tmp = smem->warp_sum[lane_id];
    uint32_t prefix_sum = device::warp::reduce_sum(lane_id < warp_id ? tmp : 0u);
    prefix_sum += warp_exc;
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      prefix_sum += orig[i];
      const auto above = length - prefix_sum;
      if (above < topk && above + orig[i] >= topk) {
        smem->match = {
            .bin = tx * kItems + i,
            .above_count = above,
            .equal_count = orig[i],
        };
      }
    }
    __syncthreads();
  }

  const auto [thr_bin, num_above, num_equal] = smem->match;

  // Phase 3: Scatter
  constexpr uint32_t kMaxTolerance = 0;
  const bool need_tiebreak = (num_equal + num_above > topk + kMaxTolerance);
  const auto topk_indices = indices;
  const auto tie_buffer = smem->tie_buffer;

#pragma unroll
  for (uint32_t v = 0; v < v3::kVecsPerThread; ++v) {
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
      const uint32_t idx = (tx + v * v3::kBlockSize) * 4 + e;
      if constexpr (!kIs2Pass) {
        if (idx >= length) goto LABEL_SCATTER_DONE;
      }
      const uint32_t bin = extract_coarse_bin_v3<v3::kHistBits>(local[v][e]);
      if (bin > thr_bin) {
        topk_indices[atomicAdd(&smem->counter_gt, 1)] = idx;
      } else if (bin == thr_bin) {
        const auto pos = atomicAdd(&smem->counter_eq, 1);
        if (need_tiebreak) {
          if (pos < v3::kMaxTies) {
            tie_buffer[pos] = {.idx = idx, .score = local[v][e]};
          }
        } else {
          if (const auto which = pos + num_above; which < topk) {
            topk_indices[which] = idx;
          }
        }
      }
    }
    // prefetch the next scores from smem into registers (for 2-pass)
    if constexpr (kIs2Pass) {
      local[v].load(smem->score_buffer, tx + v * v3::kBlockSize);
    }
  }

  // 16K ~ 32K: process second chunk from registers (now loaded from score_buffer)
  if constexpr (kIs2Pass) {
#pragma unroll
    for (uint32_t v = 0; v < v3::kVecsPerThread; ++v) {
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) {
        const uint32_t idx = (tx + v * v3::kBlockSize) * 4 + e + v3::kMax1PassLength;
        if (idx >= length) goto LABEL_SCATTER_DONE;
        const uint32_t bin = extract_coarse_bin_v3<v3::kHistBits>(local[v][e]);
        if (bin > thr_bin) {
          topk_indices[atomicAdd(&smem->counter_gt, 1)] = idx;
        } else if (bin == thr_bin) {
          const auto pos = atomicAdd(&smem->counter_eq, 1);
          if (need_tiebreak) {
            if (pos < v3::kMaxTies) {
              tie_buffer[pos] = {.idx = idx, .score = local[v][e]};
            }
          } else {
            if (const auto which = pos + num_above; which < topk) {
              topk_indices[which] = idx;
            }
          }
        }
      }
    }
  }

[[maybe_unused]] LABEL_SCATTER_DONE:
  if (!need_tiebreak) return;

  // Phase 4: Tie-breaking within the threshold bin
  __syncthreads();

  const uint32_t num_ties = min(num_equal, v3::kMaxTies);
  const uint32_t topk_remain = topk - num_above;

  const auto is_greater = [](const TieV3& a, const TieV3& b) {
    return (a.score > b.score) || (a.score == b.score && a.idx < b.idx);
  };

  if (num_ties <= 32u) {
    if (lane_id >= num_ties || warp_id >= num_ties) return;
    const uint32_t mask = (1ull << num_ties) - 1u;
    const auto tie = tie_buffer[lane_id];
    const auto target_tie = tie_buffer[warp_id];
    const bool pred = is_greater(tie, target_tie);
    const auto rank = static_cast<uint32_t>(__popc(__ballot_sync(mask, pred)));
    if (lane_id == 0 && rank < topk_remain) {
      topk_indices[num_above + rank] = target_tie.idx;
    }
  } else if (num_ties <= 64u) {
    const auto lane_id_1 = lane_id + 32;
    const auto warp_id_1 = warp_id + 32;
    const auto invalid = TieV3{.idx = 0xFFFFFFFF, .score = -FLT_MAX};
    const auto tie_0 = tie_buffer[lane_id];
    const auto tie_1 = lane_id_1 < num_ties ? tie_buffer[lane_id_1] : invalid;
    if (true) {
      const auto target = tie_buffer[warp_id];
      const bool pred_0 = is_greater(tie_0, target);
      const bool pred_1 = is_greater(tie_1, target);
      const auto rank_0 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_0)));
      const auto rank_1 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_1)));
      const auto rank = rank_0 + rank_1;
      if (lane_id == 0 && rank < topk_remain) {
        topk_indices[num_above + rank] = target.idx;
      }
    }
    if (warp_id_1 < num_ties) {
      const auto target = tie_buffer[warp_id_1];
      const bool pred_0 = is_greater(tie_0, target);
      const bool pred_1 = is_greater(tie_1, target);
      const auto rank_0 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_0)));
      const auto rank_1 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_1)));
      const auto rank = rank_0 + rank_1;
      if (lane_id == 0 && rank < topk_remain) {
        topk_indices[num_above + rank] = target.idx;
      }
    }
  } else {
    [[unlikely]];
    for (auto i = warp_id; i < num_ties; i += v3::kNumWarps) {
      const auto target_tie = tie_buffer[i];
      uint32_t local_rank = 0;
      for (auto j = lane_id; j < num_ties; j += 32) {
        const auto tie = tie_buffer[j];
        if (is_greater(tie, target_tie)) local_rank++;
      }
      const auto rank = device::warp::reduce_sum(local_rank);
      if (lane_id == 0 && rank < topk_remain) {
        topk_indices[num_above + rank] = target_tie.idx;
      }
    }
  }
}

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
  constexpr uint32_t BLOCK_SIZE = v3::kBlockSize;
  constexpr uint32_t SMEM_INPUT_SIZE = kSMEM / (2 * sizeof(int32_t));

  alignas(128) __shared__ uint32_t _s_histogram_buf[2][RADIX + 32];
  alignas(128) __shared__ uint32_t s_counter;
  alignas(128) __shared__ uint32_t s_threshold_bin_id;
  alignas(128) __shared__ uint32_t s_num_input[2];
  alignas(128) __shared__ int32_t s_last_remain;

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

  // cp.async pipelined chunk loader, shared by stage 1 and stage 1b.
  // Each thread issues one 16-byte cp.async per chunk (4 floats), so chunk size
  // = BLOCK_SIZE * (16 B / sizeof(float)) = BLOCK_SIZE * 4 = 4096 floats at 1024 threads.
  constexpr uint32_t kChunkFloats = BLOCK_SIZE * 4;
  alignas(16) __shared__ float s_chunk_buf[2][kChunkFloats];
  const uint32_t num_chunks = (length + kChunkFloats - 1) / kChunkFloats;
  auto issue_chunk = [&](uint32_t ci, uint32_t bi) {
    const uint32_t base = ci * kChunkFloats + tx * 4;
    if (base + 3 < length) {
      cp_async_cg_shared_global_16(&s_chunk_buf[bi][tx * 4], input + base);
    } else if (base < length) {
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) {
        s_chunk_buf[bi][tx * 4 + e] = (base + e < length) ? input[base + e] : 0.0f;
      }
    }
  };

  // stage 1: 8bit coarse histogram, pipelined via the chunk loader above.
  // Always issue + commit. Past-end chunks are bounded out inside issue_chunk
  // (no cp.async fired); their empty commits keep two groups in flight, so
  // wait_group<1> correctly blocks for the current chunk every iteration.
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();
  {
    issue_chunk(0, 0);
    cp_async_commit();
    issue_chunk(1, 1);
    cp_async_commit();
    for (uint32_t ci = 0; ci < num_chunks; ++ci) {
      const uint32_t bi = ci & 1;
      cp_async_wait_group<1>();
      __syncthreads();
      const uint32_t base = ci * kChunkFloats;
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) {
        const uint32_t idx = base + tx * 4 + e;
        if (idx < length) {
          ::atomicAdd(&s_histogram[convert_to_uint8(s_chunk_buf[bi][tx * 4 + e])], 1);
        }
      }
      __syncthreads();
      issue_chunk(ci + 2, bi);
      cp_async_commit();
    }
    cp_async_wait_all();
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

    // stage 1b: same pipeline pattern as stage 1, reusing s_chunk_buf + issue_chunk.
    {
      issue_chunk(0, 0);
      cp_async_commit();
      issue_chunk(1, 1);
      cp_async_commit();
      for (uint32_t ci = 0; ci < num_chunks; ++ci) {
        const uint32_t bi = ci & 1;
        cp_async_wait_group<1>();
        __syncthreads();
        const uint32_t base = ci * kChunkFloats;
#pragma unroll
        for (uint32_t e = 0; e < 4; ++e) {
          const uint32_t idx = base + tx * 4 + e;
          if (idx >= length) continue;
          const float raw_input = s_chunk_buf[bi][tx * 4 + e];
          const uint32_t bin = convert_to_uint8(raw_input);
          if (bin > threshold_bin) {
            const auto pos = ::atomicAdd(&s_counter, 1);
            output[pos] = idx;
          } else if (bin == threshold_bin) {
            const auto pos = ::atomicAdd(&s_num_input[0], 1);
            if (pos < SMEM_INPUT_SIZE) {
              [[likely]] s_input_idx[0][pos] = idx;
              const auto bin2 = convert_to_uint32(raw_input);
              const auto sub_bin = (bin2 >> 24) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
        __syncthreads();
        issue_chunk(ci + 2, bi);
        cp_async_commit();
      }
      cp_async_wait_all();
    }
    __syncthreads();
  }

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
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              output[topk - pos] = idx;
            }
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
    }
  }
}

template <bool kUsePDL>
__global__ __launch_bounds__(v3::kBlockSize, 2) void topk_512_transform_v3(const __grid_constant__ TopKParams params) {
  alignas(128) extern __shared__ uint8_t _smem[];
  __shared__ int32_t s_topk_indices[kMaxTopK];
  const auto& [scores, seq_lens, page_table, page_indices, raw_indices, score_stride, page_table_stride, page_bits, topk] =
      params;
  const uint32_t work_id = blockIdx.x;

  const uint32_t seq_len = seq_lens[work_id];
  const auto score_ptr = scores + work_id * score_stride;
  const auto page_ptr = page_table + work_id * page_table_stride;
  const auto indices_ptr = page_indices + work_id * topk;
  const auto raw_indices_ptr = raw_indices != nullptr ? raw_indices + work_id * topk : nullptr;

  device::PDLWaitPrimary<kUsePDL>();

  if (seq_len <= topk) {
    // Trivial case: just transform indices
    const auto tx = threadIdx.x;
    if (tx < seq_len) {
      indices_ptr[tx] = page_to_indices(page_ptr, tx, page_bits);
      if (raw_indices_ptr != nullptr) {
        raw_indices_ptr[tx] = tx;
      }
    } else if (tx < topk) {
      indices_ptr[tx] = -1;
      if (raw_indices_ptr != nullptr) {
        raw_indices_ptr[tx] = -1;
      }
    }
  } else if (seq_len <= v3::kMax1PassLength) {
    register_topk<false>(score_ptr, s_topk_indices, seq_len, topk, _smem);
    __syncthreads();
    const auto tx = threadIdx.x;
    if (tx < topk) {
      indices_ptr[tx] = page_to_indices(page_ptr, s_topk_indices[tx], page_bits);
      if (raw_indices_ptr != nullptr) {
        raw_indices_ptr[tx] = s_topk_indices[tx];
      }
    }
  } else if (seq_len <= v3::kMax2PassLength) {  // Stall instruction fetch: consider tagged as likely?
    [[likely]];
    register_topk<true>(score_ptr, s_topk_indices, seq_len, topk, _smem);
    __syncthreads();
    const auto tx = threadIdx.x;
    if (tx < topk) {
      indices_ptr[tx] = page_to_indices(page_ptr, s_topk_indices[tx], page_bits);
      if (raw_indices_ptr != nullptr) {
        raw_indices_ptr[tx] = s_topk_indices[tx];
      }
    }
  } else {
    // Fallback to radix_topk for sequences exceeding 32K
    radix_topk(score_ptr, s_topk_indices, seq_len, topk);
    __syncthreads();
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
    return ::cudaFuncSetAttribute(fptr, ::cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSMEM);
  }();
  host::RuntimeDeviceCheck(result, where);
}

template <bool kUsePDL>
struct TopKKernel {
  static constexpr auto kernel = topk_512_transform_v3<kUsePDL>;

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
    constexpr auto kSMEM_ = sizeof(RegisterTopKSmem) + 128;  // align up a little
    setup_kernel_smem_once<kernel, kSMEM_>();
    LaunchKernel(batch_size, v3::kBlockSize, device.unwrap(), kSMEM_).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
