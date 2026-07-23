/**
 * @NOTE: The radix top-k core (fast_topk_cuda_tl_impl) is adapted from
 * https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/topk_selector.py
 * and was previously shipped as an AOT sgl-kernel op (fast_kpool_topk_transform_fused).
 * It is re-implemented here as a lightweight JIT kernel for the NSA kpool indexer:
 * select pool groups at pool granularity, expand each group to `pool_size` token
 * indices, and optionally transform those indices through a page table or ragged offset.
 *
 * The pool-level top-k value is a compile-time constant injected via -DSGL_GROUP_TOPK.
 */
#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice, is_type
#include <sgl_kernel/utils.h>   // For RuntimeCheck, RuntimeDeviceCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, type aliases

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>

namespace {

#ifndef C10_LIKELY
#define C10_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#endif

#ifndef SGL_GROUP_TOPK
#define SGL_GROUP_TOPK 256
#endif

// Compile-time pool-level top-k (number of groups selected per row).
inline constexpr int kGroupTopK = SGL_GROUP_TOPK;
inline constexpr int kThreadsPerBlock = 1024;

// Reduced from 128KB to 32KB to improve occupancy.
// Each radix pass needs at most ~K candidates in the threshold bin,
// so 4K entries per round (2 rounds = 8K entries = 32KB) is sufficient.
inline constexpr std::size_t kSmem = 8 * 1024 * sizeof(uint32_t);  // 32KB (bytes)

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B] or nullptr
  int32_t* __restrict__ indices;           // unused here (kept for layout parity)
  const int32_t* __restrict__ lengths;     // [B]
  int64_t input_stride;
};

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <int K>
__device__ void
fast_topk_cuda_tl_impl(const float* __restrict__ input, int* __restrict__ index, int row_start, int length) {
  // An optimized topk kernel copied from tilelang kernel
  // We assume length > K here, or it will crash
  int topk = K;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
  // allocate for two rounds
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  // stage 1: 8bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx + row_start]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (C10_LIKELY(tx < RADIX)) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto bin = static_cast<int>(convert_to_uint8(input[idx + row_start]));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
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

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = input[idx + row_start];
      const auto bin = static_cast<int>(convert_to_uint8(raw_input));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        /// NOTE: (dark) fuse the histogram computation here
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
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
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

    // clip here to prevent overflow
    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    topk -= s_histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx + row_start]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
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
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx + row_start];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              index[K - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              /// NOTE: (dark) fuse the histogram computation here
              s_input_idx[r_idx ^ 1][pos] = idx;
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

__device__ __forceinline__ int32_t transform_kpool_token(
    int32_t raw_token,
    const int32_t* __restrict__ page_table_entry,
    const int32_t* __restrict__ topk_indices_offset,
    int32_t offset) {
  if (page_table_entry != nullptr) {
    return page_table_entry[raw_token];
  }
  if (topk_indices_offset != nullptr) {
    return raw_token + offset;
  }
  return raw_token;
}

template <int K>
__global__ __launch_bounds__(kThreadsPerBlock) void kpool_topk_transform_kernel(
    const __grid_constant__ FastTopKParams params,
    int32_t* __restrict__ dst_token_indices,
    const int64_t dst_stride,
    const int32_t pool_size,
    const int32_t token_topk,
    const int32_t out_cols,
    const int32_t* __restrict__ page_table,
    const int64_t page_table_stride,
    const int32_t* __restrict__ topk_indices_offset,
    const int32_t* __restrict__ seq_lens) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto score = input + bid * input_stride;
  const auto dst = dst_token_indices + bid * dst_stride;
  const auto page_table_entry = page_table == nullptr ? nullptr : page_table + bid * page_table_stride;
  const auto offset = topk_indices_offset == nullptr ? 0 : topk_indices_offset[bid];
  const bool append_tail = seq_lens != nullptr;
  const auto full_pool_token_len = length * pool_size;
  const auto history_len = full_pool_token_len < token_topk ? full_pool_token_len : token_topk;
  const auto tail_count = append_tail ? seq_lens[bid] % pool_size : 0;

  if (length <= K) {
    for (int col = tid; col < out_cols; col += kThreadsPerBlock) {
      if (col < history_len) {
        const auto group_rank = col / pool_size;
        const auto slot = col % pool_size;
        const auto raw_token = group_rank * pool_size + slot;
        dst[col] = transform_kpool_token(raw_token, page_table_entry, topk_indices_offset, offset);
      } else if (append_tail && col < history_len + tail_count) {
        const auto raw_token = length * pool_size + (col - history_len);
        dst[col] = transform_kpool_token(raw_token, page_table_entry, topk_indices_offset, offset);
      } else {
        dst[col] = -1;
      }
    }
    return;
  }

  __shared__ int s_indices[K];
  fast_topk_cuda_tl_impl<K>(score, s_indices, row_start, length);
  for (int col = tid; col < out_cols; col += kThreadsPerBlock) {
    if (col < history_len) {
      const auto group_rank = col / pool_size;
      const auto group_id = s_indices[group_rank];
      const auto slot = col % pool_size;
      const auto raw_token = group_id * pool_size + slot;
      dst[col] = transform_kpool_token(raw_token, page_table_entry, topk_indices_offset, offset);
    } else if (append_tail && col < history_len + tail_count) {
      const auto raw_token = length * pool_size + (col - history_len);
      dst[col] = transform_kpool_token(raw_token, page_table_entry, topk_indices_offset, offset);
    } else {
      dst[col] = -1;
    }
  }
}

template <auto* f, std::size_t kMaxDynamicSMEM>
void setup_kernel_smem_once(host::DebugInfo where = {}) {
  [[maybe_unused]]
  static const auto result = [] {
    const auto fptr = std::bit_cast<const void*>(f);
    return ::cudaFuncSetAttribute(fptr, ::cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSMEM);
  }();
  host::RuntimeDeviceCheck(result, where);
}

template <typename T>
const T* optional_data_ptr(const tvm::ffi::Optional<tvm::ffi::TensorView>& opt) {
  if (!opt.has_value()) {
    return nullptr;
  }
  return static_cast<const T*>(opt.value().data_ptr());
}

struct KpoolTopKTransformKernel {
  static constexpr auto kernel = kpool_topk_transform_kernel<kGroupTopK>;

  // Pool-level radix top-k for the NSA kpool indexer.
  //   score                : [B, S] strided float32 scores (one score per pool group)
  //   lengths              : [B] int32 valid group count per row
  //   dst_token_indices    : [B, out_cols] int32 output token indices (contiguous)
  //   pool_size            : tokens per pool group
  //   page_table  (opt)    : [B, P] strided int32 raw-token -> real-token map
  //   topk_indices_offset  : [B] int32 per-row offset added to raw tokens (ragged)
  //   row_starts  (opt)    : [B] int32 score row start offsets
  //   seq_lens    (opt)    : [B] int32 sequence lengths; enables tail append
  static void transform(
      const tvm::ffi::TensorView score,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView dst_token_indices,
      const int64_t pool_size,
      const tvm::ffi::Optional<tvm::ffi::TensorView> page_table_opt,
      const tvm::ffi::Optional<tvm::ffi::TensorView> topk_indices_offset_opt,
      const tvm::ffi::Optional<tvm::ffi::TensorView> row_starts_opt,
      const tvm::ffi::Optional<tvm::ffi::TensorView> seq_lens_opt) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto S = SymbolicSize{"score_stride"};
    auto out_cols_sym = SymbolicSize{"out_cols"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, -1})  // strided scores
        .with_strides({S, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(score);
    TensorMatcher({B})  // lengths, contiguous int32
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(lengths);
    TensorMatcher({B, out_cols_sym})  // output, contiguous int32
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(dst_token_indices);

    RuntimeCheck(pool_size > 1, "pool_size must be > 1, got ", pool_size);
    RuntimeCheck(
        !(page_table_opt.has_value() && topk_indices_offset_opt.has_value()),
        "page_table and topk_indices_offset are mutually exclusive");

    const auto out_cols = static_cast<int32_t>(out_cols_sym.unwrap());
    const auto tail_cols = seq_lens_opt.has_value() ? static_cast<int32_t>(pool_size) - 1 : 0;
    RuntimeCheck(out_cols > tail_cols, "dst_token_indices columns ", out_cols, " must exceed tail ", tail_cols);
    const auto token_topk = out_cols - tail_cols;
    RuntimeCheck(token_topk % static_cast<int32_t>(pool_size) == 0, "token_topk must be a multiple of pool_size");
    RuntimeCheck(
        token_topk / static_cast<int32_t>(pool_size) == kGroupTopK,
        "this module is built for group_topk=",
        kGroupTopK,
        " but got ",
        token_topk / static_cast<int32_t>(pool_size));

    const auto batch_size = static_cast<uint32_t>(B.unwrap());

    int64_t page_table_stride = 0;
    const int32_t* page_table_ptr = nullptr;
    if (page_table_opt.has_value()) {
      auto P = SymbolicSize{"page_table_stride"};
      TensorMatcher({B, -1})  // strided page table
          .with_strides({P, 1})
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(page_table_opt.value());
      page_table_ptr = static_cast<const int32_t*>(page_table_opt.value().data_ptr());
      page_table_stride = static_cast<int64_t>(P.unwrap());
    }

    if (topk_indices_offset_opt.has_value()) {
      TensorMatcher({B})  //
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(topk_indices_offset_opt.value());
    }
    if (row_starts_opt.has_value()) {
      TensorMatcher({B})  //
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(row_starts_opt.value());
    }
    if (seq_lens_opt.has_value()) {
      TensorMatcher({B})  //
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(seq_lens_opt.value());
    }

    const auto params = FastTopKParams{
        .input = static_cast<const float*>(score.data_ptr()),
        .row_starts = optional_data_ptr<int32_t>(row_starts_opt),
        .indices = nullptr,
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .input_stride = static_cast<int64_t>(S.unwrap()),
    };

    setup_kernel_smem_once<kernel, kSmem>();
    LaunchKernel(batch_size, kThreadsPerBlock, device.unwrap(), kSmem)(
        kernel,
        params,
        static_cast<int32_t*>(dst_token_indices.data_ptr()),
        static_cast<int64_t>(dst_token_indices.strides()[0]),
        static_cast<int32_t>(pool_size),
        token_topk,
        out_cols,
        page_table_ptr,
        page_table_stride,
        optional_data_ptr<int32_t>(topk_indices_offset_opt),
        optional_data_ptr<int32_t>(seq_lens_opt));
  }
};

}  // namespace
