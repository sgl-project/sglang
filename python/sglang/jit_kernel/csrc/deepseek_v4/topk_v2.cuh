#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/object.h>

#include <cfloat>
#include <cstdint>

namespace {

constexpr uint32_t K = 512;
constexpr uint32_t kBlockSize = 1024;
constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
static_assert(K <= kBlockSize);

// always use float4 to load from global memory
using Vec4 = device::AlignedVector<float, 4>;

// ---------------------------------------------------------------------------
// Order-preserving FP16 key -> histogram bin
// ---------------------------------------------------------------------------

template <uint32_t kBits>
SGL_DEVICE uint32_t extract_bin(float x) {
  static_assert(0 < kBits && kBits < 15);
  const auto hx = device::cast<fp16_t>(x);
  const uint16_t bits = *reinterpret_cast<const uint16_t*>(&hx);
  const uint16_t key = (bits & 0x8000) ? ~bits : bits | 0x8000;
  return key >> (16 - kBits);
}

SGL_DEVICE uint32_t warp_inclusive_sum(uint32_t lane_id, uint32_t val) {
  static_assert(device::kWarpThreads == 32);
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
    if (lane_id >= offset) val += n;
  }
  return val;
}

struct TopKProblem {
  const float* __restrict__ scores;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ indices;
  const uint32_t length;
  const uint32_t page_bits;
};

struct SmallTopKImpl {
  static constexpr uint32_t kHistBits = 12;
  static constexpr uint32_t kHistBins = 1 << kHistBits;
  static constexpr uint32_t kVecsPerThread = 4;
  static constexpr uint32_t kMaxTolerance = 2;
  [[maybe_unused]]
  static constexpr uint32_t kMaxSeqLen = kVecsPerThread * 4 * kBlockSize;

  struct alignas(16) MatchBin {
    uint32_t bin;
    uint32_t above_count;
    uint32_t equal_count;
  };

  struct alignas(8) Tie {
    uint32_t idx;
    float score;
  };

  struct Smem {
    using HistVec = device::AlignedVector<uint32_t, kHistBins / kBlockSize>;
    alignas(128) uint32_t counter_gt;
    alignas(128) uint32_t counter_eq;
    alignas(128) MatchBin match;
    alignas(128) uint32_t warp_sum[kNumWarps];
    alignas(16) union {
      uint32_t histogram[kHistBins];
      HistVec histogram_vec[kBlockSize];
      Tie tie_buffer[kBlockSize];
    };
  };

  SGL_DEVICE static void run(const TopKProblem problem, void* _smem) {
    using namespace device;

    const auto smem = static_cast<Smem*>(_smem);
    const auto tx = threadIdx.x;
    Smem::HistVec hist_vec;
    hist_vec.fill(0);
    smem->histogram_vec[tx] = hist_vec;
    __syncthreads();

    PDLWaitPrimary<true>();

    // Load scores into registers
    Vec4 local[kVecsPerThread];
#pragma unroll
    for (uint32_t v = 0; v < kVecsPerThread; ++v) {
      const uint32_t base = (tx + v * kBlockSize) * 4;
      if (base >= problem.length) break;
      local[v].load(problem.scores, tx + v * kBlockSize);
    }

    // Accumulate histogram via shared-memory atomics
#pragma unroll
    for (uint32_t v = 0; v < kVecsPerThread; ++v) {
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) {
        const uint32_t idx = (tx + v * kBlockSize) * 4 + e;
        if (idx >= problem.length) goto LABEL_ACC_FINISH;
        atomicAdd(&smem->histogram[extract_bin<kHistBits>(local[v][e])], 1);
      }
    }
  LABEL_ACC_FINISH:
    __syncthreads();

    // Phase 2: Exclusive prefix scan -> find threshold bin
    constexpr uint32_t kItems = kHistBins / kBlockSize;

    const auto lane_id = tx % kWarpThreads;
    const auto warp_id = tx / kWarpThreads;

    {
      smem->counter_gt = smem->counter_eq = 0;

      uint32_t orig[kItems];
      const auto hist_vec = smem->histogram_vec[tx];
      uint32_t tmp_local_sum = 0;

#pragma unroll
      for (uint32_t i = 0; i < kItems; ++i) {
        orig[i] = hist_vec[i];
        tmp_local_sum += orig[i];
      }

      const auto warp_inclusive = warp_inclusive_sum(lane_id, tmp_local_sum);
      const auto warp_exclusive = warp_inclusive - tmp_local_sum;
      if (lane_id == device::kWarpThreads - 1) {
        smem->warp_sum[warp_id] = warp_inclusive;
      }

      __syncthreads();

      const auto tmp = smem->warp_sum[lane_id];
      // Exactly one bin satisfies: above < K && above + count >= K
      uint32_t prefix_sum = warp::reduce_sum(lane_id < warp_id ? tmp : 0);
      prefix_sum += warp_exclusive;
#pragma unroll
      for (uint32_t i = 0; i < kItems; ++i) {
        prefix_sum += orig[i];
        const auto above = problem.length - prefix_sum;
        if (above < K && above + orig[i] >= K) {
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
    const bool need_tiebreak = (num_equal + num_above > K + kMaxTolerance);

    // Phase 3: Scatter
    // Elements strictly above threshold go directly to output.
    // Tied elements: simple path admits first-come; tiebreak path collects into tie_buffer.
#pragma unroll
    for (uint32_t v = 0; v < kVecsPerThread; ++v) {
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) {
        const uint32_t idx = (tx + v * kBlockSize) * 4 + e;
        if (idx >= problem.length) goto LABEL_SCATTER_DONE;
        const uint32_t bin = extract_bin<kHistBits>(local[v][e]);
        if (bin > thr_bin) {
          problem.indices[atomicAdd(&smem->counter_gt, 1)] = idx;
        } else if (bin == thr_bin) {
          const auto pos = atomicAdd(&smem->counter_eq, 1);
          if (need_tiebreak) {
            if (pos < kBlockSize) {
              smem->tie_buffer[pos] = {.idx = idx, .score = local[v][e]};
            }
          } else {
            if (const auto which = pos + num_above; which < K) {
              problem.indices[which] = idx;
            }
          }
        }
      }
    }
  LABEL_SCATTER_DONE:
    if (!need_tiebreak) return;

    // Phase 4: Tie-breaking within the threshold bin.
    // Assume num_ties <= kBlockSize (at most 1 block of ties).
    // Each thread takes one tied element, computes its rank (number of
    // elements with strictly higher score, breaking exact float ties by
    // original index), and writes to output if rank < topk_remain.
    __syncthreads();

    const uint32_t num_ties = num_equal < kBlockSize ? num_equal : kBlockSize;
    const uint32_t topk_remain = K - num_above;

    const auto is_greater = [](const Tie& a, const Tie& b) {
      return (a.score > b.score) || (a.score == b.score && a.idx < b.idx);
    };

    if (num_ties <= kWarpThreads) {
      static_assert(kWarpThreads <= kNumWarps);
      if (lane_id >= num_ties || warp_id >= num_ties) return;  // some threads are idle
      /// NOTE: use long long to avoid mask overflow when num_ties == 32
      const uint32_t mask = (1ull << num_ties) - 1u;
      const auto tie = smem->tie_buffer[lane_id];
      const auto target_tie = smem->tie_buffer[warp_id];
      const bool pred = is_greater(tie, target_tie);
      const auto rank = static_cast<uint32_t>(__popc(__ballot_sync(mask, pred)));
      if (lane_id == 0 && rank < topk_remain) {
        problem.indices[num_above + rank] = target_tie.idx;
      }
    } else if (num_ties <= kWarpThreads * 2) {
      [[unlikely]];
      // 64 x 64 topk implementation: each thread takes 2 elements
      const auto lane_id_1 = lane_id + kWarpThreads;
      const auto warp_id_1 = warp_id + kWarpThreads;
      const auto invalid = Tie{.idx = 0xFFFFFFFF, .score = -FLT_MAX};
      const auto tie_0 = smem->tie_buffer[lane_id];
      const auto tie_1 = lane_id_1 < num_ties ? smem->tie_buffer[lane_id_1] : invalid;
      if (true) {
        const auto target = smem->tie_buffer[warp_id];
        const bool pred_0 = is_greater(tie_0, target);
        const bool pred_1 = is_greater(tie_1, target);
        const auto rank_0 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_0)));
        const auto rank_1 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_1)));
        const auto rank = rank_0 + rank_1;
        if (lane_id == 0 && rank < topk_remain) {
          problem.indices[num_above + rank] = target.idx;
        }
      }
      if (warp_id_1 < num_ties) {
        const auto target = smem->tie_buffer[warp_id_1];
        const bool pred_0 = is_greater(tie_0, target);
        const bool pred_1 = is_greater(tie_1, target);
        const auto rank_0 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_0)));
        const auto rank_1 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_1)));
        const auto rank = rank_0 + rank_1;
        if (lane_id == 0 && rank < topk_remain) {
          problem.indices[num_above + rank] = target.idx;
        }
      }
    } else {
      [[unlikely]];
      // Block-level: each thread reads from tie_buffer in shared memory
      if (tx >= num_ties) return;
      const auto target_tie = smem->tie_buffer[tx];
      uint32_t rank = 0;
      for (uint32_t i = 0; i < num_ties; i++) {
        const auto tie = smem->tie_buffer[i];
        if (is_greater(tie, target_tie)) rank++;
      }
      if (rank < topk_remain) {
        problem.indices[num_above + rank] = target_tie.idx;
      }
    }
  }
};

struct TopKParams {
  const uint32_t* __restrict__ seq_lens;
  const float* __restrict__ scores;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  const int64_t score_stride;
  const int64_t page_table_stride;
  /// NOTE: indices stride must = K
  uint32_t page_bits;
};

SGL_DEVICE int32_t page_to_indices(const int32_t* __restrict__ page_table, uint32_t i, uint32_t page_bits) {
  const uint32_t mask = (1u << page_bits) - 1u;
  return (page_table[i >> page_bits] << page_bits) | (i & mask);
}

[[maybe_unused]]
SGL_DEVICE void naive_transform(
    const float* __restrict__,  // unused
    const int32_t* __restrict__ page_table,
    int32_t* __restrict__ indices,
    const uint32_t length,
    const uint32_t page_bits) {
  if (const auto tx = threadIdx.x; tx < length) {
    indices[tx] = page_to_indices(page_table, tx, page_bits);
  } else if (tx < K) {
    indices[tx] = -1;  // fill invalid indices to -1
  }
}

__global__ __launch_bounds__(kBlockSize, 2)  // optimize prefill
    void topk_transform_v2(const __grid_constant__ TopKParams params) {
  alignas(128) extern __shared__ uint8_t smem[];
  const auto batch_id = blockIdx.x;
  const auto seq_len = params.seq_lens[batch_id];
  const auto score_ptr = params.scores + batch_id * params.score_stride;
  const auto page_ptr = params.page_table + batch_id * params.page_table_stride;
  const auto indices_ptr = params.page_indices + batch_id * K;
  if (seq_len <= K) return naive_transform(score_ptr, page_ptr, indices_ptr, seq_len, params.page_bits);
  __shared__ int32_t s_topk_indices[K];
  const auto problem = TopKProblem{
      .scores = score_ptr,
      .page_table = page_ptr,
      .indices = s_topk_indices,
      .length = seq_len,
      .page_bits = params.page_bits,
  };
  SmallTopKImpl::run(problem, smem);
  device::PDLTriggerSecondary<true>();
  __syncthreads();
  if (const auto tx = threadIdx.x; tx < K) {
    indices_ptr[tx] = page_to_indices(page_ptr, s_topk_indices[tx], params.page_bits);
  }
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

struct TopK512Kernel {
  static constexpr auto kSMEM = sizeof(typename SmallTopKImpl::Smem) + 128;
  static void transform(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView page_indices,
      const uint32_t page_size,
      const tvm::ffi::Optional<tvm::ffi::TensorView> unused) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto S = SymbolicSize{"score_stride"};
    auto P = SymbolicSize{"page_table_stride"};
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
    TensorMatcher({B, 512})  // output, must be contiguous
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_indices);

    RuntimeCheck(!unused.has_value(), "topk_transform_v2 only accepts 5 arguments");
    RuntimeCheck(std::has_single_bit(page_size), "page_size must be power of 2");
    const auto page_bits = static_cast<uint32_t>(std::countr_zero(page_size));
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto params = TopKParams{
        .seq_lens = static_cast<uint32_t*>(seq_lens.data_ptr()),
        .scores = static_cast<float*>(scores.data_ptr()),
        .page_table = static_cast<int32_t*>(page_table.data_ptr()),
        .page_indices = static_cast<int32_t*>(page_indices.data_ptr()),
        .score_stride = S.unwrap(),
        .page_table_stride = P.unwrap(),
        .page_bits = page_bits,
    };
    RuntimeCheck(std::bit_cast<uintptr_t>(params.scores) % 16 == 0, "scores must be 16-byte aligned");
    RuntimeCheck(params.score_stride % 4 == 0, "score_stride must be a multiple of 4");
    constexpr auto kernel = topk_transform_v2;
    setup_kernel_smem_once<kernel, kSMEM>();
    LaunchKernel(batch_size, kBlockSize, device.unwrap(), kSMEM)  //
        .enable_pdl(true)(kernel, params);
  }
};

}  // namespace
