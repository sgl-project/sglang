#include <sgl_kernel/tensor.h>  // For TensorMatcher and symbolic tensor metadata
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, PDL helpers, and SGL_DEVICE

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>

namespace {

#ifndef SGL_TOPK
#define SGL_TOPK 512
#endif

constexpr uint32_t kTopK = SGL_TOPK;
constexpr uint32_t kTopKBlockSize = SGL_TOPK;
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
};

struct DCPTopKCandidateParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  int64_t* __restrict__ candidates;
  int64_t score_stride;
  uint32_t page_bits;
  uint32_t dcp_size;
  uint32_t dcp_rank;
};

struct DCPTopKMergeParams {
  const int64_t* __restrict__ candidates;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int32_t* __restrict__ raw_indices;
  int64_t page_table_stride;
  uint32_t batch_size;
  uint32_t page_bits;
  uint32_t dcp_size;
};

SGL_DEVICE int64_t pack_candidate(float score, int32_t raw_index) {
  const uint64_t score_bits = static_cast<uint64_t>(__float_as_uint(score));
  const uint64_t index_bits = static_cast<uint32_t>(raw_index);
  return static_cast<int64_t>((score_bits << 32) | index_bits);
}

SGL_DEVICE float unpack_candidate_score(int64_t candidate) {
  return __uint_as_float(static_cast<uint64_t>(candidate) >> 32);
}

SGL_DEVICE int32_t unpack_candidate_index(int64_t candidate) {
  return static_cast<int32_t>(static_cast<uint32_t>(candidate));
}

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
    const uint32_t page_bits) {
  static_assert(kTopK <= kTopKBlockSize);
  if (const auto tx = threadIdx.x; tx < length) {
    indices[tx] = page_to_indices(page_table, tx, page_bits);
    if (raw_indices != nullptr) {
      raw_indices[tx] = tx;
    }
  } else if (kTopK == kTopKBlockSize || tx < kTopK) {
    indices[tx] = -1;  // fill invalid indices to -1
    if (raw_indices != nullptr) {
      raw_indices[tx] = -1;
    }
  }
}

[[maybe_unused]]
SGL_DEVICE void radix_topk(const float* __restrict__ input, int32_t* __restrict__ output, const uint32_t length) {
  constexpr uint32_t RADIX = 256;
  constexpr uint32_t BLOCK_SIZE = kTopKBlockSize;
  constexpr uint32_t SMEM_INPUT_SIZE = kSMEM / (2 * sizeof(int32_t));

  alignas(128) __shared__ uint32_t _s_histogram_buf[2][RADIX + 32];
  alignas(128) __shared__ uint32_t s_counter;
  alignas(128) __shared__ uint32_t s_threshold_bin_id;
  alignas(128) __shared__ uint32_t s_num_input[2];
  alignas(128) __shared__ int32_t s_last_remain;

  extern __shared__ uint32_t s_input_idx[][kSMEM / (2 * sizeof(int32_t))];

  const uint32_t tx = threadIdx.x;
  uint32_t remain_topk = kTopK;
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
              output[kTopK - pos] = idx;
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
__global__ void topk_transform_kernel(const __grid_constant__ TopKParams params) {
  const auto &[
    scores, seq_lens, page_table, page_indices, raw_indices, // pointers
    score_stride, page_table_stride, page_bits // sizes
  ] = params;
  const uint32_t work_id = blockIdx.x;

  /// NOTE: dangerous prefetch seq_len before PDL wait
  const uint32_t seq_len = seq_lens[work_id];
  const auto score_ptr = scores + work_id * score_stride;
  const auto page_ptr = page_table + work_id * page_table_stride;
  const auto indices_ptr = page_indices + work_id * kTopK;
  const auto raw_indices_ptr = raw_indices != nullptr ? raw_indices + work_id * kTopK : nullptr;

  device::PDLWaitPrimary<kUsePDL>();

  if (seq_len <= kTopK) {
    naive_transform(score_ptr, page_ptr, indices_ptr, raw_indices_ptr, seq_len, page_bits);
  } else {
    __shared__ int32_t s_topk_indices[kTopK];
    radix_topk(score_ptr, s_topk_indices, seq_len);
    static_assert(kTopK <= kTopKBlockSize);
    const auto tx = threadIdx.x;
    if (kTopK == kTopKBlockSize || tx < kTopK) {
      indices_ptr[tx] = page_to_indices(page_ptr, s_topk_indices[tx], page_bits);
      if (raw_indices_ptr != nullptr) {
        raw_indices_ptr[tx] = s_topk_indices[tx];
      }
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
__global__ void dcp_topk_candidates_kernel(const __grid_constant__ DCPTopKCandidateParams params) {
  const uint32_t work_id = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const uint32_t seq_len = params.seq_lens[work_id];
  const auto score_ptr = params.scores + work_id * params.score_stride;
  const auto candidate_ptr = params.candidates + work_id * kTopK;

  device::PDLWaitPrimary<kUsePDL>();

  uint32_t local_raw = tx;
  bool valid = tx < seq_len;
  if (seq_len > kTopK) {
    __shared__ int32_t s_topk_indices[kTopK];
    radix_topk(score_ptr, s_topk_indices, seq_len);
    local_raw = s_topk_indices[tx];
    valid = true;
  }

  if (tx < kTopK) {
    int32_t global_raw = -1;
    float score = __int_as_float(static_cast<int>(0xff800000u));
    if (valid) {
      const uint32_t page = local_raw >> params.page_bits;
      const uint32_t offset = local_raw & ((1u << params.page_bits) - 1u);
      global_raw = static_cast<int32_t>(
          (((page * params.dcp_size) + params.dcp_rank) << params.page_bits) | offset);
      score = score_ptr[local_raw];
    }
    candidate_ptr[tx] = pack_candidate(score, global_raw);
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL, uint32_t kDCPSize>
__global__ void dcp_topk_merge_kernel(const __grid_constant__ DCPTopKMergeParams params) {
  static_assert(kDCPSize == 2 || kDCPSize == 4 || kDCPSize == 8);
  constexpr uint32_t kCandidateCount = kDCPSize * kTopK;
  const uint32_t work_id = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const uint32_t seq_len = params.seq_lens[work_id];
  const auto page_ptr = params.page_table + work_id * params.page_table_stride;
  const auto page_indices_ptr = params.page_indices + work_id * kTopK;
  const auto raw_indices_ptr = params.raw_indices != nullptr ? params.raw_indices + work_id * kTopK : nullptr;

  __shared__ float s_candidate_scores[kCandidateCount];
  __shared__ int32_t s_candidate_raw[kCandidateCount];
  __shared__ int32_t s_topk_indices[kTopK];

  device::PDLWaitPrimary<kUsePDL>();

  for (uint32_t candidate_id = tx; candidate_id < kCandidateCount; candidate_id += kTopKBlockSize) {
    const uint32_t rank = candidate_id / kTopK;
    const uint32_t local_id = candidate_id % kTopK;
    const auto candidate = params.candidates[(rank * params.batch_size + work_id) * kTopK + local_id];
    s_candidate_scores[candidate_id] = unpack_candidate_score(candidate);
    s_candidate_raw[candidate_id] = unpack_candidate_index(candidate);
  }
  __syncthreads();

  if (seq_len <= kTopK) {
    if (tx < seq_len) {
      page_indices_ptr[tx] = page_to_indices(page_ptr, tx, params.page_bits);
      if (raw_indices_ptr != nullptr) raw_indices_ptr[tx] = tx;
    } else {
      page_indices_ptr[tx] = -1;
      if (raw_indices_ptr != nullptr) raw_indices_ptr[tx] = -1;
    }
  } else {
    radix_topk(s_candidate_scores, s_topk_indices, kCandidateCount);
    const auto raw_index = s_candidate_raw[s_topk_indices[tx]];
    page_indices_ptr[tx] = page_to_indices(page_ptr, raw_index, params.page_bits);
    if (raw_indices_ptr != nullptr) raw_indices_ptr[tx] = raw_index;
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
  static constexpr auto candidate_kernel = dcp_topk_candidates_kernel<kUsePDL>;

  template <uint32_t kDCPSize>
  static void launch_merge_kernel(const DCPTopKMergeParams& params, uint32_t batch_size, DLDevice device) {
    constexpr auto merge_kernel = dcp_topk_merge_kernel<kUsePDL, kDCPSize>;
    constexpr auto kSMEM_ = kSMEM + sizeof(int32_t);
    setup_kernel_smem_once<merge_kernel, kSMEM_>();
    host::LaunchKernel(batch_size, kTopKBlockSize, device, kSMEM_).enable_pdl(kUsePDL)(merge_kernel, params);
  }

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
    TensorMatcher({B, kTopK})  // output, must be contiguous
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_indices);

    int32_t* raw_indices_ptr = nullptr;
    if (raw_indices.has_value()) {
      TensorMatcher({B, kTopK})  // optional raw indices output, must be contiguous
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(raw_indices.value());
      raw_indices_ptr = static_cast<int32_t*>(raw_indices.value().data_ptr());
    }

    RuntimeCheck(std::has_single_bit(page_size), "page_size must be power of 2");
    const auto page_bits = static_cast<uint32_t>(std::countr_zero(page_size));
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto params = TopKParams{
        .scores = static_cast<float*>(scores.data_ptr()),
        .seq_lens = static_cast<int32_t*>(seq_lens.data_ptr()),
        .page_table = static_cast<int32_t*>(page_table.data_ptr()),
        .page_indices = static_cast<int32_t*>(page_indices.data_ptr()),
        .raw_indices = raw_indices_ptr,
        .score_stride = S.unwrap(),
        .page_table_stride = P.unwrap(),
        .page_bits = page_bits,
    };
    constexpr auto kSMEM_ = kSMEM + sizeof(int32_t);  // align up a little
    setup_kernel_smem_once<kernel, kSMEM_>();
    LaunchKernel(batch_size, kTopKBlockSize, device.unwrap(), kSMEM_).enable_pdl(kUsePDL)(kernel, params);
  }

  static void candidates(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView candidates,
      const uint32_t page_size,
      const uint32_t dcp_size,
      const uint32_t dcp_rank) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto S = SymbolicSize{"score_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, -1})  // strided scores
        .with_strides({S, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(scores);
    TensorMatcher({B})  // contiguous local sequence lengths
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(seq_lens);
    TensorMatcher({B, kTopK})  // packed score/index payload
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(candidates);

    RuntimeCheck(std::has_single_bit(page_size), "page_size must be power of 2");
    RuntimeCheck(
        dcp_size == 2 || dcp_size == 4 || dcp_size == 8,
        "DCP candidate path requires dcp_size in {2, 4, 8}");
    RuntimeCheck(dcp_rank < dcp_size, "dcp_rank must be smaller than dcp_size");
    const auto params = DCPTopKCandidateParams{
        .scores = static_cast<const float*>(scores.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .candidates = static_cast<int64_t*>(candidates.data_ptr()),
        .score_stride = S.unwrap(),
        .page_bits = static_cast<uint32_t>(std::countr_zero(page_size)),
        .dcp_size = dcp_size,
        .dcp_rank = dcp_rank,
    };
    constexpr auto kSMEM_ = kSMEM + sizeof(int32_t);
    setup_kernel_smem_once<candidate_kernel, kSMEM_>();
    LaunchKernel(B.unwrap(), kTopKBlockSize, device.unwrap(), kSMEM_)
        .enable_pdl(kUsePDL)(candidate_kernel, params);
  }

  static void merge_dcp_candidates(
      const tvm::ffi::TensorView candidates,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView page_indices,
      const uint32_t page_size,
      const uint32_t dcp_size,
      const tvm::ffi::Optional<tvm::ffi::TensorView> raw_indices) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto P = SymbolicSize{"page_table_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(seq_lens);
    TensorMatcher({B, -1})
        .with_strides({P, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_table);
    TensorMatcher({B, kTopK})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_indices);
    TensorMatcher({-1, kTopK})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(candidates);

    int32_t* raw_indices_ptr = nullptr;
    if (raw_indices.has_value()) {
      TensorMatcher({B, kTopK})
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(raw_indices.value());
      raw_indices_ptr = static_cast<int32_t*>(raw_indices.value().data_ptr());
    }

    RuntimeCheck(std::has_single_bit(page_size), "page_size must be power of 2");
    RuntimeCheck(
        dcp_size == 2 || dcp_size == 4 || dcp_size == 8,
        "DCP candidate merge requires dcp_size in {2, 4, 8}");
    RuntimeCheck(
        candidates.size(0) == B.unwrap() * dcp_size,
        "invalid gathered candidate shape");
    const auto params = DCPTopKMergeParams{
        .candidates = static_cast<const int64_t*>(candidates.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .page_indices = static_cast<int32_t*>(page_indices.data_ptr()),
        .raw_indices = raw_indices_ptr,
        .page_table_stride = P.unwrap(),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
        .page_bits = static_cast<uint32_t>(std::countr_zero(page_size)),
        .dcp_size = dcp_size,
    };
    switch (dcp_size) {
      case 2:
        launch_merge_kernel<2>(params, B.unwrap(), device.unwrap());
        break;
      case 4:
        launch_merge_kernel<4>(params, B.unwrap(), device.unwrap());
        break;
      case 8:
        launch_merge_kernel<8>(params, B.unwrap(), device.unwrap());
        break;
      default:
        Panic("unsupported dcp_size: ", dcp_size);
    }
  }
};

}  // namespace
