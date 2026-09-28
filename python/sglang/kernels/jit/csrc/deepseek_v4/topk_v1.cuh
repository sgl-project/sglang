#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>

namespace sglang {

// Tie handling of the radix top-k below:
//   1. the last radix round fills the remaining slots with the LOWEST
//      POSITIONS among the exactly-equal scores (bitmap + block prefix-sum)
//      instead of atomic arrival order, so the selected SET is a pure function
//      of (scores, length, topk) and equals a stable sort by
//      (score desc, position asc);
//   2. when a threshold bin holds more candidates than the 8192-entry smem
//      candidate buffer, the next round re-scans the input (filtering by the
//      accumulated key prefix) instead of working on an arrival-ordered
//      subset, and the sub-bin histogram always counts every candidate, so the
//      result stays exact.

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

// Warp-inclusive prefix sum (all 32 lanes participate).
SGL_DEVICE uint32_t warp_inclusive_sum(uint32_t v) {
#pragma unroll
  for (uint32_t d = 1; d < 32; d <<= 1) {
    const uint32_t n = __shfl_up_sync(0xffffffffu, v, d);
    if ((threadIdx.x & 31) >= d) v += n;
  }
  return v;
}

[[maybe_unused]]
SGL_DEVICE void
radix_topk(const float* __restrict__ input, int32_t* __restrict__ output, const uint32_t length, const uint32_t topk) {
  constexpr uint32_t RADIX = 256;
  constexpr uint32_t BLOCK_SIZE = kTopKBlockSize;
  constexpr uint32_t SMEM_INPUT_SIZE = kSMEM / (2 * sizeof(int32_t));
  constexpr uint32_t kBitmapBits = SMEM_INPUT_SIZE * 32;  // one candidate buffer reused as a position bitmap

  alignas(128) __shared__ uint32_t _s_histogram_buf[2][RADIX + 32];
  alignas(128) __shared__ uint32_t s_counter;
  alignas(128) __shared__ uint32_t s_threshold_bin_id;
  alignas(128) __shared__ uint32_t s_num_input[2];
  alignas(128) __shared__ int32_t s_last_remain;

  extern __shared__ uint32_t s_input_idx[][kSMEM / (2 * sizeof(int32_t))];

  const uint32_t tx = threadIdx.x;
  uint32_t remain_topk = topk;
  auto& s_histogram = _s_histogram_buf[0];

  // The fp16 coarse bin and the accumulated fp32 key prefix of the surviving
  // candidates (uniform across the block); used to re-derive the candidate set
  // from the input when the smem candidate buffer overflowed.
  uint32_t coarse_bin = 0;
  uint32_t key_prefix = 0;

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
  coarse_bin = threshold_bin;
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
        const auto bin = convert_to_uint32(raw_input);
        const auto sub_bin = (bin >> 24) & 0xFF;
        if (pos < SMEM_INPUT_SIZE) {
          [[likely]] s_input_idx[0][pos] = idx;
        }
        // the histogram must cover every candidate, stored or not
        ::atomicAdd(&s_histogram[sub_bin], 1);
      }
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
    // When the candidate buffer overflowed, re-derive the candidate set from
    // the input (coarse bin match + key prefix match) instead of the subset.
    const bool rescan = raw_num_input > SMEM_INPUT_SIZE;
    const auto offset = 24 - round * 8;

    // Iterate the round's candidates as (idx, key32), exact in every case.
    const auto for_each_candidate = [&](auto&& f) {
      if (!rescan) {
        for (uint32_t i = tx; i < num_input; i += BLOCK_SIZE) {
          const auto idx = s_input_idx[r_idx][i];
          f(idx, convert_to_uint32(input[idx]));
        }
      } else {
        for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
          const float x = input[idx];
          if (convert_to_uint8(x) != coarse_bin) continue;
          const auto key = convert_to_uint32(x);
          if (round > 0 && (key >> (32 - 8 * round)) != key_prefix) continue;
          f(idx, key);
        }
      }
    };

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = remain_topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    remain_topk -= s_histogram[threshold_bin + 1];
    const uint32_t need = static_cast<uint32_t>(s_last_remain);  // uniform; == remain_topk here
    // The rescan filter of THIS round uses the prefix of rounds < round;
    // this round's threshold is appended after its candidate loops (below).

    if (remain_topk == 0) {
      for_each_candidate([&](uint32_t idx, uint32_t key) {
        const auto bin = (key >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          output[pos] = idx;
        }
      });
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      if (round == 3) {
        // (1) strictly-above candidates: the set is deterministic, order irrelevant
        for_each_candidate([&](uint32_t idx, uint32_t key) {
          if (((key >> offset) & 0xFF) > threshold_bin) {
            const auto pos = ::atomicAdd(&s_counter, 1);
            output[pos] = idx;
          }
        });
        // (2) exactly-equal candidates (all 32 key bits match): fill the
        // remaining `need` slots with the lowest positions, in position order.
        uint32_t* bitmap = s_input_idx[r_idx ^ 1];    // free in the last round
        uint32_t* s_warp_incl = _s_histogram_buf[0];  // [32] warp totals (inclusive)
        uint32_t* s_warp_excl = _s_histogram_buf[1];  // [32] warp offsets (exclusive)
        const uint32_t out_base = topk - need;
        const uint32_t lane = tx & 31, warp = tx >> 5;
        uint32_t carry = 0;
        for (uint32_t chunk = 0; chunk < length; chunk += kBitmapBits) {
          const uint32_t chunk_len = min(length - chunk, kBitmapBits);
          const uint32_t nwords = (chunk_len + 31) >> 5;
          __syncthreads();  // previous pass's readers of bitmap / s_warp_* are done
          for (uint32_t w = tx; w < nwords; w += BLOCK_SIZE)
            bitmap[w] = 0;
          __syncthreads();
          for_each_candidate([&](uint32_t idx, uint32_t key) {
            if (((key >> offset) & 0xFF) == threshold_bin && idx >= chunk && idx < chunk + chunk_len) {
              ::atomicOr(&bitmap[(idx - chunk) >> 5], 1u << (idx & 31));
            }
          });
          __syncthreads();
          for (uint32_t wb = 0; wb < nwords; wb += BLOCK_SIZE) {
            const uint32_t w = wb + tx;
            const uint32_t word = w < nwords ? bitmap[w] : 0u;
            const uint32_t cnt = __popc(word);
            const uint32_t incl = warp_inclusive_sum(cnt);
            if (lane == 31) s_warp_incl[warp] = incl;
            __syncthreads();
            if (warp == 0) {
              const uint32_t v = s_warp_incl[lane];
              const uint32_t vi = warp_inclusive_sum(v);
              s_warp_excl[lane] = vi - v;
            }
            __syncthreads();
            const uint32_t pass_total = s_warp_excl[31] + s_warp_incl[31];
            uint32_t r = carry + s_warp_excl[warp] + incl - cnt;
            uint32_t wd = word;
            while (wd != 0u && r < need) {
              const uint32_t b = __ffs(wd) - 1u;
              output[out_base + r] = static_cast<int32_t>(chunk + w * 32u + b);
              ++r;
              wd &= wd - 1u;
            }
            carry += pass_total;
            __syncthreads();  // s_warp_* reused by the next pass
          }
        }
        __syncthreads();
        break;
      }
      for_each_candidate([&](uint32_t idx, uint32_t key) {
        const auto bin = (key >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          output[pos] = idx;
        } else if (bin == threshold_bin) {
          const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
          const auto sub_bin = (key >> (offset - 8)) & 0xFF;
          if (pos < SMEM_INPUT_SIZE) {
            /// NOTE: (dark) fuse the histogram computation here
            [[likely]] s_input_idx[r_idx ^ 1][pos] = idx;
          }
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      });
      __syncthreads();
      key_prefix = (key_prefix << 8) | threshold_bin;
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
