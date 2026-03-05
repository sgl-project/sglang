#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace {

static constexpr int TopK = 2048;
static constexpr int kThreadsPerBlock = 1024;
static constexpr size_t kSmem = 8 * 1024 * sizeof(uint32_t);  // 32KB

struct FastTopKParams {
  const float* __restrict__ input;
  const int32_t* __restrict__ row_starts;
  int32_t* __restrict__ indices;
  int32_t* __restrict__ lengths;
  int64_t input_stride;
};

__device__ void naive_topk_cuda(
    const float* __restrict__ score,
    int32_t* __restrict__ indice,
    int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < TopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

__device__ void naive_topk_transform(
    const float* __restrict__ score,
    int32_t length,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    dst_page_table[i] = (i < length) ? src_page_table[i] : -1;
  }
}

__device__ void naive_topk_transform_ragged(
    const float* __restrict__ score,
    int32_t length,
    int32_t* __restrict__ topk_indices_ragged,
    int32_t offset) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    topk_indices_ragged[i] =
        (i < length) ? static_cast<int32_t>(i) + offset : -1;
  }
}

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ void fast_topk_cuda_tl(
    const float* __restrict__ input,
    int* __restrict__ index,
    int row_start,
    int length) {
  int topk = TopK;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE =
      static_cast<int>(kSmem / (2 * sizeof(int)));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
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
      if (__builtin_expect(tx < RADIX, 1)) {
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
      const auto bin =
          static_cast<int>(convert_to_uint8(input[idx + row_start]));
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
      const auto bin =
          static_cast<int>(convert_to_uint8(raw_input));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        if (__builtin_expect(pos < SMEM_INPUT_SIZE, 1)) {
          s_input_idx[0][pos] = idx;
          const auto bin2 = convert_to_uint32(raw_input);
          const auto sub_bin = (bin2 >> 24) & 0xFF;
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

    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE))
                               ? _raw_num_input
                               : int(SMEM_INPUT_SIZE);

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
        const auto bin = (convert_to_uint32(input[idx + row_start]) >> offset) &
                         0xFF;
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
        const auto bin =
            (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              index[TopK - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (__builtin_expect(pos < SMEM_INPUT_SIZE, 1)) {
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin2 = convert_to_uint32(raw_input);
              const auto sub_bin = (bin2 >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)
void topk_kernel(const FastTopKParams params) {
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto row_start =
      params.row_starts == nullptr ? 0 : params.row_starts[bid];
  const auto length = params.lengths[bid];
  const auto indice = params.indices + bid * TopK;
  const auto score = params.input + bid * params.input_stride;
  if (length <= TopK) {
    return naive_topk_cuda(score, indice, length);
  } else {
    return fast_topk_cuda_tl(score, indice, row_start, length);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)
void topk_transform_decode_kernel(
    const FastTopKParams params,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table,
    const int64_t src_stride) {
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto length = params.lengths[bid];
  const auto src_page_entry = src_page_table + bid * src_stride;
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = params.input + bid * params.input_stride;
  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, 0, length);
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    dst_page_entry[idx_0] = src_page_entry[s_indices[idx_0]];
    const auto idx_1 = tid + kThreadsPerBlock;
    dst_page_entry[idx_1] = src_page_entry[s_indices[idx_1]];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)
void topk_transform_prefill_kernel(
    const FastTopKParams params,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table,
    const int64_t src_stride,
    const int32_t* __restrict__ cu_seqlens_q,
    const int64_t prefill_bs) {
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto length = params.lengths[bid];
  const auto row_start =
      params.row_starts == nullptr ? 0 : params.row_starts[bid];
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = params.input + bid * params.input_stride;

  __shared__ const int32_t* s_src_page_entry;
  if (__builtin_expect(prefill_bs <= kThreadsPerBlock, 1)) {
    if (tid < prefill_bs) {
      if (bid >= static_cast<uint64_t>(cu_seqlens_q[tid]) &&
          bid < static_cast<uint64_t>(cu_seqlens_q[tid + 1])) {
        s_src_page_entry = src_page_table + tid * src_stride;
      }
    }
  } else {
    for (int64_t i = tid; i < prefill_bs; i += kThreadsPerBlock) {
      if (bid >= static_cast<uint64_t>(cu_seqlens_q[i]) &&
          bid < static_cast<uint64_t>(cu_seqlens_q[i + 1])) {
        s_src_page_entry = src_page_table + i * src_stride;
      }
    }
  }
  __syncthreads();
  const auto src_page_entry = s_src_page_entry;

  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    dst_page_entry[idx_0] = src_page_entry[s_indices[idx_0]];
    const auto idx_1 = tid + kThreadsPerBlock;
    dst_page_entry[idx_1] = src_page_entry[s_indices[idx_1]];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)
void topk_transform_prefill_ragged_kernel(
    const FastTopKParams params,
    int32_t* __restrict__ topk_indices_ragged,
    const int32_t* __restrict__ topk_indices_offset) {
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start =
      params.row_starts == nullptr ? 0 : params.row_starts[bid];
  const auto length = params.lengths[bid];
  const auto dst_indices_entry = topk_indices_ragged + bid * TopK;
  const auto score = params.input + bid * params.input_stride;
  const auto offset = topk_indices_offset[bid];

  if (length <= TopK) {
    return naive_topk_transform_ragged(score, length, dst_indices_entry, offset);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    dst_indices_entry[idx_0] = s_indices[idx_0] + offset;
    const auto idx_1 = tid + kThreadsPerBlock;
    dst_indices_entry[idx_1] = s_indices[idx_1] + offset;
  }
}

// Host wrapper: fast_topk
void fast_topk(
    tvm::ffi::TensorView score,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView lengths,
    tvm::ffi::TensorView row_starts,
    bool has_row_starts) {
  using namespace host;

  auto B = SymbolicSize{"batch"};
  auto L = SymbolicSize{"input_stride"};
  auto K = SymbolicSize{"topk"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({B, L}).with_dtype<fp32_t>().with_device(device).verify(score);
  TensorMatcher({B, K}).with_dtype<int32_t>().with_device(device).verify(indices);
  TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(lengths);

  const auto batch = static_cast<int64_t>(B.unwrap());
  const auto input_stride = static_cast<int64_t>(L.unwrap());
  const auto dev = device.unwrap();

  FastTopKParams params{
      .input = static_cast<const float*>(score.data_ptr()),
      .row_starts = has_row_starts
                        ? static_cast<const int32_t*>(row_starts.data_ptr())
                        : nullptr,
      .indices = static_cast<int32_t*>(indices.data_ptr()),
      .lengths = static_cast<int32_t*>(lengths.data_ptr()),
      .input_stride = input_stride,
  };

  LaunchKernel(
      dim3(static_cast<unsigned>(batch)),
      dim3(kThreadsPerBlock),
      dev,
      kSmem)(topk_kernel, params);
}

// Host wrapper: fast_topk_transform
void fast_topk_transform(
    tvm::ffi::TensorView score,
    tvm::ffi::TensorView lengths,
    tvm::ffi::TensorView dst_page_table,
    tvm::ffi::TensorView src_page_table,
    tvm::ffi::TensorView cu_seqlens_q,
    tvm::ffi::TensorView row_starts,
    bool has_row_starts,
    bool is_decode) {
  using namespace host;

  auto B = SymbolicSize{"batch"};
  auto L = SymbolicSize{"input_stride"};
  auto K = SymbolicSize{"topk"};
  auto PBS = SymbolicSize{"prefill_bs_plus_1"};
  auto SRC_S = SymbolicSize{"src_stride"};
  auto SRC_B = SymbolicSize{"src_batch"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({B, L}).with_dtype<fp32_t>().with_device(device).verify(score);
  TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(lengths);
  TensorMatcher({B, K}).with_dtype<int32_t>().with_device(device).verify(dst_page_table);
  TensorMatcher({SRC_B, SRC_S}).with_dtype<int32_t>().with_device(device).verify(src_page_table);
  TensorMatcher({PBS}).with_dtype<int32_t>().with_device(device).verify(cu_seqlens_q);

  const auto batch = static_cast<int64_t>(B.unwrap());
  const auto input_stride = static_cast<int64_t>(L.unwrap());
  const auto prefill_bs = static_cast<int64_t>(PBS.unwrap()) - 1;
  const auto src_stride = static_cast<int64_t>(SRC_S.unwrap());
  const auto dev = device.unwrap();

  FastTopKParams params{
      .input = static_cast<const float*>(score.data_ptr()),
      .row_starts = has_row_starts
                        ? static_cast<const int32_t*>(row_starts.data_ptr())
                        : nullptr,
      .indices = nullptr,
      .lengths = static_cast<int32_t*>(lengths.data_ptr()),
      .input_stride = input_stride,
  };

  const auto grid = dim3(static_cast<unsigned>(batch));
  const auto block = dim3(kThreadsPerBlock);

  if (is_decode) {
    LaunchKernel(grid, block, dev, kSmem)(
        topk_transform_decode_kernel,
        params,
        static_cast<int32_t*>(dst_page_table.data_ptr()),
        static_cast<const int32_t*>(src_page_table.data_ptr()),
        src_stride);
  } else {
    LaunchKernel(grid, block, dev, kSmem)(
        topk_transform_prefill_kernel,
        params,
        static_cast<int32_t*>(dst_page_table.data_ptr()),
        static_cast<const int32_t*>(src_page_table.data_ptr()),
        src_stride,
        static_cast<const int32_t*>(cu_seqlens_q.data_ptr()),
        prefill_bs);
  }
}

// Host wrapper: fast_topk_transform_ragged
void fast_topk_transform_ragged(
    tvm::ffi::TensorView score,
    tvm::ffi::TensorView lengths,
    tvm::ffi::TensorView topk_indices_ragged,
    tvm::ffi::TensorView topk_indices_offset,
    tvm::ffi::TensorView row_starts,
    bool has_row_starts) {
  using namespace host;

  auto B = SymbolicSize{"batch"};
  auto L = SymbolicSize{"input_stride"};
  auto K = SymbolicSize{"topk"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({B, L}).with_dtype<fp32_t>().with_device(device).verify(score);
  TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(lengths);
  TensorMatcher({B, K})
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(topk_indices_ragged);
  TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(
      topk_indices_offset);

  const auto batch = static_cast<int64_t>(B.unwrap());
  const auto input_stride = static_cast<int64_t>(L.unwrap());
  const auto dev = device.unwrap();

  FastTopKParams params{
      .input = static_cast<const float*>(score.data_ptr()),
      .row_starts = has_row_starts
                        ? static_cast<const int32_t*>(row_starts.data_ptr())
                        : nullptr,
      .indices = nullptr,
      .lengths = static_cast<int32_t*>(lengths.data_ptr()),
      .input_stride = input_stride,
  };

  LaunchKernel(
      dim3(static_cast<unsigned>(batch)),
      dim3(kThreadsPerBlock),
      dev,
      kSmem)(
      topk_transform_prefill_ragged_kernel,
      params,
      static_cast<int32_t*>(topk_indices_ragged.data_ptr()),
      static_cast<const int32_t*>(topk_indices_offset.data_ptr()));
}

}  // namespace
