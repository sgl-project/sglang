#include "topk_v1.cuh"

namespace sglang {

struct DCPTopKCandidateParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ local_lens;
  int64_t* __restrict__ candidates;
  int64_t score_stride;
  int64_t candidate_stride;
  uint32_t score_width;
  uint32_t topk;
  uint32_t dcp_size;
  uint32_t dcp_rank;
};

struct DCPTopKMergeParams {
  const int64_t* __restrict__ candidates;
  const int32_t* __restrict__ local_page_table;
  int32_t* __restrict__ page_indices;
  int32_t* __restrict__ local_raw_indices;
  int32_t* __restrict__ local_lens;
  int64_t page_table_stride;
  int64_t candidate_stride;
  uint32_t batch_size;
  uint32_t page_bits;
  uint32_t topk;
  uint32_t dcp_rank;
};

SGL_DEVICE int64_t pack_dcp_candidate(float score, int32_t global_index) {
  const uint64_t score_bits = static_cast<uint64_t>(__float_as_uint(score));
  const uint64_t index_bits = static_cast<uint32_t>(global_index);
  return static_cast<int64_t>((score_bits << 32) | index_bits);
}

SGL_DEVICE float unpack_dcp_candidate_score(int64_t candidate) {
  return __uint_as_float(static_cast<uint64_t>(candidate) >> 32);
}

SGL_DEVICE int32_t unpack_dcp_candidate_index(int64_t candidate) {
  return static_cast<int32_t>(static_cast<uint32_t>(candidate));
}

SGL_DEVICE float canonicalize_dcp_score(float score) {
  if (score != score) return __uint_as_float(0xff800000u);
  return score == 0.0f ? 0.0f : score;
}

SGL_DEVICE uint32_t ordered_dcp_score(float score) {
  return convert_to_uint32(canonicalize_dcp_score(score));
}

SGL_DEVICE uint32_t dcp_block_inclusive_sum(uint32_t value, int32_t* scratch, uint32_t* total) {
  constexpr uint32_t kLogicalWarpSize = 32;
  constexpr uint32_t kLogicalWarps = kTopKBlockSize / kLogicalWarpSize;
  const uint32_t tx = threadIdx.x;
  const uint32_t warp_id = tx / kLogicalWarpSize;
  const uint32_t lane_id = tx % kLogicalWarpSize;

#pragma unroll
  for (uint32_t offset = 1; offset < kLogicalWarpSize; offset <<= 1) {
#ifdef USE_ROCM
    const uint32_t addend = __shfl_up(value, offset, kLogicalWarpSize);
#else
    const uint32_t addend = __shfl_up_sync(0xffffffffu, value, offset, kLogicalWarpSize);
#endif
    if (lane_id >= offset) {
      value += addend;
    }
  }
  if (lane_id == kLogicalWarpSize - 1) {
    scratch[warp_id] = static_cast<int32_t>(value);
  }
  __syncthreads();

  if (warp_id == 0) {
    const uint32_t warp_value = lane_id < kLogicalWarps ? static_cast<uint32_t>(scratch[lane_id]) : 0u;
    uint32_t warp_prefix = warp_value;
#pragma unroll
    for (uint32_t offset = 1; offset < kLogicalWarpSize; offset <<= 1) {
#ifdef USE_ROCM
      const uint32_t addend = __shfl_up(warp_prefix, offset, kLogicalWarpSize);
#else
      const uint32_t addend = __shfl_up_sync(0xffffffffu, warp_prefix, offset, kLogicalWarpSize);
#endif
      if (lane_id >= offset) {
        warp_prefix += addend;
      }
    }
    if (lane_id < kLogicalWarps) {
      scratch[lane_id] = static_cast<int32_t>(warp_prefix - warp_value);
    }
    if (lane_id == kLogicalWarps - 1) {
      *total = warp_prefix;
    }
  }
  __syncthreads();
  return static_cast<uint32_t>(scratch[warp_id]) + value;
}

template <bool kUsePDL>
__global__ void dcp_topk_candidates_kernel(const __grid_constant__ DCPTopKCandidateParams params) {
  const uint32_t batch_idx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const int32_t raw_local_len = params.local_lens[batch_idx];
  const uint32_t local_len = raw_local_len > 0 ? min(static_cast<uint32_t>(raw_local_len), params.score_width) : 0;
  const auto score_ptr = params.scores + batch_idx * params.score_stride;
  const auto candidate_ptr = params.candidates + batch_idx * params.candidate_stride;

  device::PDLWaitPrimary<kUsePDL>();

  __shared__ int32_t selected[kMaxTopK];
  __shared__ uint32_t threshold_key;
  __shared__ uint32_t output_count;
  __shared__ uint32_t tile_count;
  __shared__ uint32_t radix_prefix;
  __shared__ uint32_t radix_mask;
  __shared__ uint32_t radix_remaining;
  __shared__ uint32_t radix_bin;

  if (tx < params.topk) {
    candidate_ptr[tx] = pack_dcp_candidate(__uint_as_float(0xff800000u), -1);
  }

  if (local_len <= params.topk) {
    if (tx < local_len) {
      const int32_t global_index = static_cast<int32_t>(tx * params.dcp_size + params.dcp_rank);
      candidate_ptr[tx] = pack_dcp_candidate(canonicalize_dcp_score(score_ptr[tx]), global_index);
    }
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }

  const bool radix_scratch_overflow = radix_topk(score_ptr, selected, local_len, params.topk, true);
  if (!radix_scratch_overflow) {
    if (tx == 0) {
      threshold_key = 0xffffffffu;
      output_count = 0;
    }
    __syncthreads();

    if (tx < params.topk) {
      atomicMin(&threshold_key, ordered_dcp_score(score_ptr[selected[tx]]));
    }
    __syncthreads();
  } else {
    // topk_v1 reports when its fixed index scratch clipped a dense coarse
    // bucket. Select the exact 32-bit threshold with four bounded histogram
    // passes only for those score distributions; wide rows that fit the fast
    // path keep its lower latency.
    constexpr uint32_t kRadix = 256;
    constexpr uint32_t kHistogramStride = kRadix + 32;
    constexpr uint32_t kWaveSize = 64;
    constexpr uint32_t kWaveCount = kTopKBlockSize / kWaveSize;
    auto* histogram_0 = reinterpret_cast<uint32_t*>(selected);
    auto* histogram_1 = histogram_0 + kHistogramStride;
    extern __shared__ uint32_t dynamic_smem[];
    auto* wave_histograms = dynamic_smem;

    if (tx == 0) {
      radix_prefix = 0;
      radix_mask = 0;
      radix_remaining = params.topk;
    }
    __syncthreads();

#pragma unroll
    for (int32_t shift = 24; shift >= 0; shift -= 8) {
      for (uint32_t index = tx; index < kWaveCount * kRadix; index += kTopKBlockSize) {
        wave_histograms[index] = 0;
      }
      __syncthreads();

      const uint32_t prefix = radix_prefix;
      const uint32_t mask = radix_mask;
      const uint32_t wave_id = tx / kWaveSize;
      uint32_t pending_bin = 0;
      uint32_t pending_count = 0;
      for (uint32_t local_index = tx; local_index < local_len; local_index += kTopKBlockSize) {
        const uint32_t key = ordered_dcp_score(score_ptr[local_index]);
        if ((key & mask) == prefix) {
          const uint32_t bin = (key >> shift) & 0xffu;
          if (pending_count != 0 && bin != pending_bin) {
            atomicAdd(&wave_histograms[wave_id * kRadix + pending_bin], pending_count);
            pending_count = 0;
          }
          pending_bin = bin;
          ++pending_count;
        }
      }
      if (pending_count != 0) {
        atomicAdd(&wave_histograms[wave_id * kRadix + pending_bin], pending_count);
      }
      __syncthreads();

      if (tx < kRadix) {
        uint32_t count = 0;
#pragma unroll
        for (uint32_t wave = 0; wave < kWaveCount; ++wave) {
          count += wave_histograms[wave * kRadix + tx];
        }
        histogram_0[tx] = count;
      } else if (tx == kRadix) {
        histogram_0[kRadix] = 0;
      }
      __syncthreads();

#pragma unroll
      for (int32_t level = 0; level < 8; ++level) {
        const int32_t offset = 1 << level;
        auto* input_histogram = level & 1 ? histogram_1 : histogram_0;
        auto* output_histogram = level & 1 ? histogram_0 : histogram_1;
        if (tx < kRadix) {
          output_histogram[tx] = input_histogram[tx] + (tx + offset < kRadix ? input_histogram[tx + offset] : 0u);
        }
        __syncthreads();
      }

      if (tx < kRadix && histogram_0[tx] >= radix_remaining && histogram_0[tx + 1] < radix_remaining) {
        radix_bin = tx;
      }
      __syncthreads();
      if (tx == 0) {
        radix_remaining -= histogram_0[radix_bin + 1];
        radix_prefix |= radix_bin << shift;
        radix_mask |= 0xffu << shift;
      }
      __syncthreads();
    }

    if (tx == 0) {
      threshold_key = radix_prefix;
      output_count = 0;
    }
    __syncthreads();
  }

  for (uint32_t tile_start = 0; tile_start < local_len; tile_start += kTopKBlockSize) {
    if (output_count >= params.topk) break;
    const uint32_t local_index = tile_start + tx;
    const bool is_strict = local_index < local_len && ordered_dcp_score(score_ptr[local_index]) > threshold_key;
    const uint32_t tile_base = output_count;
    const uint32_t position = dcp_block_inclusive_sum(is_strict ? 1u : 0u, selected, &tile_count);
    if (is_strict && tile_base + position <= params.topk) {
      const int32_t global_index = static_cast<int32_t>(local_index * params.dcp_size + params.dcp_rank);
      candidate_ptr[tile_base + position - 1] =
          pack_dcp_candidate(canonicalize_dcp_score(score_ptr[local_index]), global_index);
    }
    if (tx == 0) {
      output_count = min(params.topk, tile_base + tile_count);
    }
    __syncthreads();
  }

  for (uint32_t tile_start = 0; tile_start < local_len; tile_start += kTopKBlockSize) {
    if (output_count >= params.topk) break;
    const uint32_t local_index = tile_start + tx;
    const bool is_threshold = local_index < local_len && ordered_dcp_score(score_ptr[local_index]) == threshold_key;
    const uint32_t tile_base = output_count;
    const uint32_t position = dcp_block_inclusive_sum(is_threshold ? 1u : 0u, selected, &tile_count);
    if (is_threshold && tile_base + position <= params.topk) {
      const int32_t global_index = static_cast<int32_t>(local_index * params.dcp_size + params.dcp_rank);
      candidate_ptr[tile_base + position - 1] =
          pack_dcp_candidate(canonicalize_dcp_score(score_ptr[local_index]), global_index);
    }
    if (tx == 0) {
      output_count = min(params.topk, tile_base + tile_count);
    }
    __syncthreads();
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL, uint32_t kDCPSize>
__global__ void dcp_topk_merge_kernel(const __grid_constant__ DCPTopKMergeParams params) {
  static_assert(kDCPSize == 2 || kDCPSize == 4 || kDCPSize == 8);
  constexpr uint32_t kMaxCandidateCount = kDCPSize * kMaxTopK;

  const uint32_t batch_idx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const uint32_t candidate_count = kDCPSize * params.topk;
  const auto page_table = params.local_page_table + batch_idx * params.page_table_stride;
  const auto page_indices = params.page_indices + batch_idx * params.topk;
  const auto local_raw_indices =
      params.local_raw_indices != nullptr ? params.local_raw_indices + batch_idx * params.topk : nullptr;

  __shared__ float candidate_scores[kMaxCandidateCount];
  __shared__ int32_t candidate_indices[kMaxCandidateCount];
  __shared__ int32_t selected[kMaxTopK];
  __shared__ uint32_t output_count;
  __shared__ uint32_t threshold_key;
  __shared__ uint32_t greater_count;
  __shared__ uint32_t threshold_valid_count;
  __shared__ uint32_t tie_count;
  __shared__ uint32_t tie_min_index;
  __shared__ uint32_t tie_max_index;
  __shared__ uint32_t compact_count;

  device::PDLWaitPrimary<kUsePDL>();

  if (tx == 0) {
    output_count = 0;
    threshold_key = 0xffffffffu;
    greater_count = 0;
    threshold_valid_count = 0;
    tie_count = 0;
    tie_min_index = 0xffffffffu;
    tie_max_index = 0;
  }
  if (tx < params.topk) {
    page_indices[tx] = -1;
    if (local_raw_indices != nullptr) {
      local_raw_indices[tx] = -1;
    }
  }

  for (uint32_t candidate_id = tx; candidate_id < candidate_count; candidate_id += kTopKBlockSize) {
    const uint32_t source_rank = candidate_id / params.topk;
    const uint32_t local_id = candidate_id - source_rank * params.topk;
    const auto candidate_row =
        params.candidates + (source_rank * params.batch_size + batch_idx) * params.candidate_stride;
    const int64_t candidate = candidate_row[local_id];
    const int32_t global_index = unpack_dcp_candidate_index(candidate);
    candidate_scores[candidate_id] = canonicalize_dcp_score(unpack_dcp_candidate_score(candidate));
    candidate_indices[candidate_id] = global_index;
  }
  __syncthreads();

  radix_topk(candidate_scores, selected, candidate_count, params.topk);
  __syncthreads();

  if (tx < params.topk) {
    atomicMin(&threshold_key, ordered_dcp_score(candidate_scores[selected[tx]]));
  }
  __syncthreads();

  uint32_t thread_greater_count = 0;
  uint32_t thread_threshold_valid_count = 0;
  for (uint32_t candidate_id = tx; candidate_id < candidate_count; candidate_id += kTopKBlockSize) {
    const int32_t global_index = candidate_indices[candidate_id];
    const uint32_t score_key = ordered_dcp_score(candidate_scores[candidate_id]);
    if (global_index >= 0) {
      if (score_key > threshold_key) {
        ++thread_greater_count;
      } else if (score_key == threshold_key) {
        ++thread_threshold_valid_count;
      }
    }
  }
  if (thread_greater_count != 0) {
    atomicAdd(&greater_count, thread_greater_count);
  }
  if (thread_threshold_valid_count != 0) {
    atomicAdd(&threshold_valid_count, thread_threshold_valid_count);
  }
  __syncthreads();

  if (tx == 0) {
    const uint32_t remaining = greater_count < params.topk ? params.topk - greater_count : 0;
    tie_count = min(remaining, threshold_valid_count);
  }
  __syncthreads();

  const uint32_t owner_candidate_id = params.dcp_rank * params.topk + min(tx, params.topk - 1);
  const int32_t owner_global_index = tx < params.topk ? candidate_indices[owner_candidate_id] : -1;
  const bool owner_is_strict =
      owner_global_index >= 0 && ordered_dcp_score(candidate_scores[owner_candidate_id]) > threshold_key;
  const uint32_t owner_position = dcp_block_inclusive_sum(owner_is_strict ? 1u : 0u, selected, &compact_count);
  if (owner_is_strict) {
    const uint32_t local_raw = static_cast<uint32_t>(owner_global_index) / kDCPSize;
    page_indices[owner_position - 1] = page_to_indices(page_table, local_raw, params.page_bits);
    if (local_raw_indices != nullptr) {
      local_raw_indices[owner_position - 1] = static_cast<int32_t>(local_raw);
    }
  }
  if (tx == 0) {
    output_count = compact_count;
  }
  __syncthreads();

  if (tie_count == 1) {
    // Random model scores almost always have one item at the top-k boundary.
    // Avoid a second full radix selection in that common case.
    for (uint32_t candidate_id = tx; candidate_id < candidate_count; candidate_id += kTopKBlockSize) {
      const int32_t global_index = candidate_indices[candidate_id];
      if (global_index >= 0 && ordered_dcp_score(candidate_scores[candidate_id]) == threshold_key) {
        atomicMin(&tie_min_index, static_cast<uint32_t>(global_index));
      }
    }
    __syncthreads();
    if (tx == 0 && tie_min_index % kDCPSize == params.dcp_rank) {
      const uint32_t local_raw = tie_min_index / kDCPSize;
      page_indices[output_count] = page_to_indices(page_table, local_raw, params.page_bits);
      if (local_raw_indices != nullptr) {
        local_raw_indices[output_count] = static_cast<int32_t>(local_raw);
      }
      ++output_count;
    }
  } else if (tie_count > 1) {
    for (uint32_t candidate_id = tx; candidate_id < candidate_count; candidate_id += kTopKBlockSize) {
      const int32_t global_index = candidate_indices[candidate_id];
      const bool is_threshold = global_index >= 0 && ordered_dcp_score(candidate_scores[candidate_id]) == threshold_key;
      candidate_scores[candidate_id] = is_threshold ? -static_cast<float>(global_index) : __uint_as_float(0xff800000u);
    }
    __syncthreads();
    radix_topk(candidate_scores, selected, candidate_count, tie_count);
    __syncthreads();
    if (tx < tie_count) {
      const int32_t global_index = candidate_indices[selected[tx]];
      atomicMax(&tie_max_index, static_cast<uint32_t>(global_index));
    }
    __syncthreads();

    const bool owner_is_selected_tie = owner_global_index >= 0 &&
                                       candidate_scores[owner_candidate_id] != __uint_as_float(0xff800000u) &&
                                       static_cast<uint32_t>(owner_global_index) <= tie_max_index;
    const uint32_t tie_position = dcp_block_inclusive_sum(owner_is_selected_tie ? 1u : 0u, selected, &compact_count);
    const uint32_t tie_base = output_count;
    if (owner_is_selected_tie) {
      const uint32_t local_raw = static_cast<uint32_t>(owner_global_index) / kDCPSize;
      page_indices[tie_base + tie_position - 1] = page_to_indices(page_table, local_raw, params.page_bits);
      if (local_raw_indices != nullptr) {
        local_raw_indices[tie_base + tie_position - 1] = static_cast<int32_t>(local_raw);
      }
    }
    if (tx == 0) {
      output_count = tie_base + compact_count;
    }
  }
  __syncthreads();

  if (tx == 0) {
    params.local_lens[batch_idx] = static_cast<int32_t>(output_count);
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct DCPTopKKernel {
  static constexpr auto candidate_kernel = dcp_topk_candidates_kernel<kUsePDL>;

  static void candidates(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView local_lens,
      const tvm::ffi::TensorView candidates,
      const uint32_t dcp_size,
      const uint32_t dcp_rank) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto W = SymbolicSize{"score_width"};
    auto S = SymbolicSize{"score_stride"};
    auto CS = SymbolicSize{"candidate_stride"};
    auto K = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();

    TensorMatcher({B, W}).with_strides({S, 1}).with_dtype<float>().with_device(device).verify(scores);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(local_lens);
    TensorMatcher({B, K}).with_strides({CS, 1}).with_dtype<int64_t>().with_device(device).verify(candidates);

    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 1024]");
    RuntimeCheck(dcp_size == 2 || dcp_size == 4 || dcp_size == 8, "dcp_size must be in {2, 4, 8}");
    RuntimeCheck(dcp_rank < dcp_size, "dcp_rank must be smaller than dcp_size");

    const auto params = DCPTopKCandidateParams{
        .scores = static_cast<const float*>(scores.data_ptr()),
        .local_lens = static_cast<const int32_t*>(local_lens.data_ptr()),
        .candidates = static_cast<int64_t*>(candidates.data_ptr()),
        .score_stride = S.unwrap(),
        .candidate_stride = CS.unwrap(),
        .score_width = static_cast<uint32_t>(W.unwrap()),
        .topk = topk,
        .dcp_size = dcp_size,
        .dcp_rank = dcp_rank,
    };
    constexpr auto kDynamicSMEM = kSMEM + sizeof(int32_t);
    setup_kernel_smem_once<candidate_kernel, kDynamicSMEM>();
    LaunchKernel(B.unwrap(), kTopKBlockSize, device.unwrap(), kDynamicSMEM)
        .enable_pdl(kUsePDL)(candidate_kernel, params);
  }

  template <uint32_t kDCPSize>
  static void launch_merge(const DCPTopKMergeParams& params, uint32_t batch_size, DLDevice device) {
    constexpr auto kernel = dcp_topk_merge_kernel<kUsePDL, kDCPSize>;
    constexpr auto kDynamicSMEM = kSMEM + sizeof(int32_t);
    setup_kernel_smem_once<kernel, kDynamicSMEM>();
    host::LaunchKernel(batch_size, kTopKBlockSize, device, kDynamicSMEM).enable_pdl(kUsePDL)(kernel, params);
  }

  static void merge(
      const tvm::ffi::TensorView candidates,
      const tvm::ffi::TensorView local_page_table,
      const tvm::ffi::TensorView page_indices,
      const tvm::ffi::TensorView local_lens,
      const uint32_t page_size,
      const uint32_t dcp_size,
      const uint32_t dcp_rank,
      const tvm::ffi::Optional<tvm::ffi::TensorView> local_raw_indices) {
    using namespace host;
    auto C = SymbolicSize{"candidate_rows"};
    auto B = SymbolicSize{"batch_size"};
    auto P = SymbolicSize{"page_table_stride"};
    auto CS = SymbolicSize{"candidate_stride"};
    auto K = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();

    TensorMatcher({C, K}).with_strides({CS, 1}).with_dtype<int64_t>().with_device(device).verify(candidates);
    TensorMatcher({B, -1}).with_strides({P, 1}).with_dtype<int32_t>().with_device(device).verify(local_page_table);
    TensorMatcher({B, K}).with_dtype<int32_t>().with_device(device).verify(page_indices);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(local_lens);

    int32_t* local_raw_ptr = nullptr;
    if (local_raw_indices.has_value()) {
      TensorMatcher({B, K}).with_dtype<int32_t>().with_device(device).verify(local_raw_indices.value());
      local_raw_ptr = static_cast<int32_t*>(local_raw_indices.value().data_ptr());
    }

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 1024]");
    RuntimeCheck(C.unwrap() == B.unwrap() * dcp_size, "candidate rows must equal batch_size * dcp_size");
    RuntimeCheck(std::has_single_bit(page_size), "page_size must be power of 2");
    RuntimeCheck(dcp_size == 2 || dcp_size == 4 || dcp_size == 8, "dcp_size must be in {2, 4, 8}");
    RuntimeCheck(dcp_rank < dcp_size, "dcp_rank must be smaller than dcp_size");

    const auto params = DCPTopKMergeParams{
        .candidates = static_cast<const int64_t*>(candidates.data_ptr()),
        .local_page_table = static_cast<const int32_t*>(local_page_table.data_ptr()),
        .page_indices = static_cast<int32_t*>(page_indices.data_ptr()),
        .local_raw_indices = local_raw_ptr,
        .local_lens = static_cast<int32_t*>(local_lens.data_ptr()),
        .page_table_stride = P.unwrap(),
        .candidate_stride = CS.unwrap(),
        .batch_size = batch_size,
        .page_bits = static_cast<uint32_t>(std::countr_zero(page_size)),
        .topk = topk,
        .dcp_rank = dcp_rank,
    };

    switch (dcp_size) {
      case 2:
        launch_merge<2>(params, batch_size, device.unwrap());
        break;
      case 4:
        launch_merge<4>(params, batch_size, device.unwrap());
        break;
      case 8:
        launch_merge<8>(params, batch_size, device.unwrap());
        break;
      default:
        host::RuntimeCheck(false, "Unsupported dcp_size=", dcp_size);
    }
  }
};

}  // namespace sglang
