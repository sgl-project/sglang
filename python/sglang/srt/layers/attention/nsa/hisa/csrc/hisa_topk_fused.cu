/**
 * HISA — fused topk + coord-transform kernel.
 *
 * Adapted from sgl-kernel/csrc/elementwise/topk.cu. The radix-select machinery
 * (``TopK`` / ``kThreadsPerBlock`` / ``kSmem`` / ``FastTopKParams`` /
 * ``naive_topk_*`` / ``convert_to_uint*`` / ``fast_topk_cuda_tl`` / ``get_params``
 * / ``setup_kernel_smem_once``) is copied verbatim from upstream so any perf
 * delta vs ``fast_topk_v2`` comes only from the modified epilogue, not from
 * helper drift.
 *
 * The two new kernels — ``topk_coord_transform_fused_paged_kernel`` and
 * ``topk_coord_transform_fused_ragged_kernel`` — mirror upstream's
 * ``topk_transform_decode_kernel`` / ``topk_transform_prefill_ragged_kernel``
 * 1:1, replacing only the gather tail with HISA's coord-transform logic
 * (``raw = topk_block_idx[batch, slot] * K_BLK + (r % K_BLK)``).
 */
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace {

// =============================================================================
// BEGIN verbatim copy from sgl-kernel/csrc/elementwise/topk.cu (line 21-252).
// Do NOT modify — keep helpers byte-equal so any perf delta is from the new
// epilogue alone.
// =============================================================================

constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 1024;

#ifdef USE_ROCM
#ifdef SGL_TOPK_DYNAMIC_SMEM_BYTES
constexpr size_t kSmem = static_cast<size_t>(SGL_TOPK_DYNAMIC_SMEM_BYTES);
#else
constexpr size_t kSmem = 48 * 1024;  // bytes
#endif
#else
constexpr size_t kSmem = 8 * 1024 * sizeof(uint32_t);  // 32KB (bytes)
#endif

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, TopK]
  int32_t* __restrict__ lengths;           // [B]
  int64_t input_stride;
};

// when length <= TopK, we can directly write the indices
__device__ void naive_topk_cuda(const float* __restrict__ score, int32_t* __restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < TopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

// keep the first `length` entries, set others to -1
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

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform_ragged(
    const float* __restrict__ score, int32_t length, int32_t* __restrict__ topk_indices_ragged, int32_t offset) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    topk_indices_ragged[i] = (i < length) ? static_cast<int32_t>(i) + offset : -1;
  }
}

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

__device__ void fast_topk_cuda_tl(const float* __restrict__ input, int* __restrict__ index, int row_start, int length) {
  // An optimized topk kernel copied from tilelang kernel
  // We assume length > TopK here, or it will crash
  int topk = TopK;
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
              index[TopK - pos] = idx;
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

// =============================================================================
// END verbatim copy from upstream.
// =============================================================================


// =============================================================================
// HISA-specific kernels. Skeleton modeled 1:1 after upstream's
// ``topk_transform_decode_kernel`` (paged) and
// ``topk_transform_prefill_ragged_kernel`` (ragged); only the gather tail is
// replaced with HISA's coord transform.
// =============================================================================

__global__ __launch_bounds__(kThreadsPerBlock)  // hisa paged decode
    void topk_coord_transform_fused_paged_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ output,                       // [B, TopK] i32
        const int32_t* __restrict__ topk_block_idx,         // [B, BLOCK_TOPK] i32
        const int32_t* __restrict__ seq_lens,               // [B] i32 — per-req absolute seq_len
        int32_t k_block_size,
        int32_t block_topk) {
  const auto& [input, _1, _2, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = 0;
  const auto length = lengths[bid];
  const auto out_entry = output + bid * TopK;
  const auto score = input + bid * input_stride;
  if (length <= TopK) {
    // Mirror naive_topk_transform but apply the HISA coord transform inline.
    const int32_t batch_seq_len = seq_lens[bid];
    const int32_t* abs_blocks = topk_block_idx + bid * block_topk;
    for (int i = tid; i < TopK; i += kThreadsPerBlock) {
      if (i < length) {
        const int slot = i / k_block_size;
        const int abs_block = abs_blocks[slot];
        const int raw = abs_block * k_block_size + (i - slot * k_block_size);
        const bool pos_valid = (raw >= 0) && (raw < batch_seq_len);
        out_entry[i] = pos_valid ? raw : -1;
      } else {
        out_entry[i] = -1;
      }
    }
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    // ── HISA coord transform tail (replaces upstream's page-table gather) ──
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const int32_t batch_seq_len = seq_lens[bid];
    const int32_t* abs_blocks = topk_block_idx + bid * block_topk;

    // unrolled iter 0
    {
      const int idx_0 = tid;
      const int r = s_indices[idx_0];
      const bool r_valid = (r != -1);
      const int r_safe = max(r, 0);
      const int slot = r_safe / k_block_size;
      const int abs_block = abs_blocks[slot];
      const int raw = abs_block * k_block_size + (r_safe - slot * k_block_size);
      const bool pos_valid = (raw >= 0) && (raw < batch_seq_len);
      out_entry[idx_0] = (r_valid && pos_valid) ? raw : -1;
    }
    // unrolled iter 1
    {
      const int idx_1 = tid + kThreadsPerBlock;
      const int r = s_indices[idx_1];
      const bool r_valid = (r != -1);
      const int r_safe = max(r, 0);
      const int slot = r_safe / k_block_size;
      const int abs_block = abs_blocks[slot];
      const int raw = abs_block * k_block_size + (r_safe - slot * k_block_size);
      const bool pos_valid = (raw >= 0) && (raw < batch_seq_len);
      out_entry[idx_1] = (r_valid && pos_valid) ? raw : -1;
    }
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // hisa ragged prefill
    void topk_coord_transform_fused_ragged_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ output,                       // [B, TopK] i32
        const int32_t* __restrict__ topk_block_idx,         // [B, BLOCK_TOPK] i32
        const int32_t* __restrict__ ks,                     // [B] i32 — per-row kv start
        const int32_t* __restrict__ ke,                     // [B] i32 — per-row kv end
        int32_t k_block_size,
        int32_t block_topk) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto out_entry = output + bid * TopK;
  const auto score = input + bid * input_stride;

  const int32_t row_ks = ks[bid];
  const int32_t row_ke = ke[bid];
  const int32_t row_extent = row_ke - row_ks;
  const int32_t* abs_blocks = topk_block_idx + bid * block_topk;

  if (length <= TopK) {
    for (int i = tid; i < TopK; i += kThreadsPerBlock) {
      if (i < length) {
        const int slot = i / k_block_size;
        const int abs_block = abs_blocks[slot];
        const int raw = abs_block * k_block_size + (i - slot * k_block_size);
        const int raw_rel = raw - row_ks;
        const bool pos_valid = (raw_rel >= 0) && (raw_rel < row_extent);
        out_entry[i] = pos_valid ? raw_rel : -1;
      } else {
        out_entry[i] = -1;
      }
    }
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    // ── HISA ragged coord transform tail ──
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);

    // unrolled iter 0
    {
      const int idx_0 = tid;
      const int r = s_indices[idx_0];
      const bool r_valid = (r != -1);
      const int r_safe = max(r, 0);
      const int slot = r_safe / k_block_size;
      const int abs_block = abs_blocks[slot];
      const int raw = abs_block * k_block_size + (r_safe - slot * k_block_size);
      const int raw_rel = raw - row_ks;
      const bool pos_valid = (raw_rel >= 0) && (raw_rel < row_extent);
      out_entry[idx_0] = (r_valid && pos_valid) ? raw_rel : -1;
    }
    // unrolled iter 1
    {
      const int idx_1 = tid + kThreadsPerBlock;
      const int r = s_indices[idx_1];
      const bool r_valid = (r != -1);
      const int r_safe = max(r, 0);
      const int slot = r_safe / k_block_size;
      const int abs_block = abs_blocks[slot];
      const int raw = abs_block * k_block_size + (r_safe - slot * k_block_size);
      const int raw_rel = raw - row_ks;
      const bool pos_valid = (raw_rel >= 0) && (raw_rel < row_extent);
      out_entry[idx_1] = (r_valid && pos_valid) ? raw_rel : -1;
    }
  }
}


// =============================================================================
// Reserved stubs for SGLANG_NSA_FUSE_TOPK=1 (output = physical page_table_1
// indices, mirroring upstream's ``fast_topk_transform_fused`` /
// ``fast_topk_transform_ragged_fused``). Not yet implemented — when the
// epilogue is added it will reuse the same radix-select front and append a
// final ``page_table_1[batch, raw]`` (paged) or ``raw + topk_indices_offset``
// (ragged) gather.
// =============================================================================

__global__ __launch_bounds__(kThreadsPerBlock)  // TODO: SGLANG_NSA_FUSE_TOPK=1 paged
    void topk_transform_paged_kernel(
        const FastTopKParams /*params*/,
        int32_t* __restrict__ /*output*/,
        const int32_t* __restrict__ /*topk_block_idx*/,
        const int32_t* __restrict__ /*page_table_1*/,
        int32_t /*k_block_size*/,
        int32_t /*block_topk*/,
        int64_t /*page_table_stride*/) {
  // Reserved name. Body intentionally empty.
}

__global__ __launch_bounds__(kThreadsPerBlock)  // TODO: SGLANG_NSA_FUSE_TOPK=1 ragged
    void topk_transform_ragged_kernel(
        const FastTopKParams /*params*/,
        int32_t* __restrict__ /*output*/,
        const int32_t* __restrict__ /*topk_block_idx*/,
        const int32_t* __restrict__ /*topk_indices_offset*/,
        int32_t /*k_block_size*/,
        int32_t /*block_topk*/) {
  // Reserved name. Body intentionally empty.
}


// =============================================================================
// BEGIN verbatim copy from upstream (line 383-431).
// =============================================================================

auto get_params(
    const at::Tensor& score,
    const at::Tensor& lengths,
    std::optional<at::Tensor> row_starts_opt = std::nullopt,
    std::optional<at::Tensor> indices_opt = std::nullopt) -> FastTopKParams {
  const auto B = score.size(0);
  TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1);
  if (row_starts_opt.has_value()) {
    const auto& row_starts = row_starts_opt.value();
    TORCH_CHECK(row_starts.dim() == 1);
    TORCH_CHECK(row_starts.size(0) == B);
  }
  TORCH_CHECK(lengths.dim() == 1 && lengths.is_contiguous());
  TORCH_CHECK(lengths.size(0) == B);
  int32_t* indices_data_ptr = nullptr;
  if (indices_opt.has_value()) {
    const auto& indices = indices_opt.value();
    TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous());
    TORCH_CHECK(indices.size(0) == B);
    TORCH_CHECK(indices.size(1) == TopK);
    indices_data_ptr = indices.data_ptr<int32_t>();
  }

  return FastTopKParams{
      .input = score.data_ptr<float>(),
      .row_starts = row_starts_opt.has_value() ? row_starts_opt->data_ptr<int32_t>() : nullptr,
      .indices = indices_data_ptr,
      .lengths = lengths.data_ptr<int32_t>(),
      .input_stride = score.stride(0),
  };
}

template <auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
  [[maybe_unused]]
  static const auto result = [] {
#ifdef USE_ROCM
    return ::cudaFuncSetAttribute(
        reinterpret_cast<const void*>(f), ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#else
    return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#endif
  }();
  TORCH_CHECK(result == cudaSuccess, "set_up_kernel_once failed:", ::cudaGetErrorString(result));
}

// =============================================================================
// END verbatim copy from upstream.
// =============================================================================

}  // namespace

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

void topk_coord_transform_fused_paged_interface(
    const at::Tensor& score,                  // [B, sparse_len] f32
    const at::Tensor& lengths,                // [B] i32 — per-row valid length of `score`
    const at::Tensor& topk_block_idx,         // [B, block_topk] i32
    const at::Tensor& seq_lens,               // [B] i32 — per-req absolute seq_len for OOB mask
    at::Tensor& output,                       // [B, TopK] i32 (out)
    int64_t k_block_size) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(topk_block_idx);
  CHECK_CUDA(seq_lens);
  CHECK_CUDA(output);

  const auto params = get_params(score, lengths, std::nullopt);
  const auto B = score.size(0);
  TORCH_CHECK(output.dim() == 2 && output.is_contiguous());
  TORCH_CHECK(output.size(0) == B && output.size(1) == TopK);
  TORCH_CHECK(topk_block_idx.dim() == 2 && topk_block_idx.is_contiguous());
  TORCH_CHECK(topk_block_idx.size(0) == B);
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.is_contiguous() && seq_lens.size(0) == B);
  const int32_t block_topk = static_cast<int32_t>(topk_block_idx.size(1));

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  setup_kernel_smem_once<topk_coord_transform_fused_paged_kernel, kSmem>();
  topk_coord_transform_fused_paged_kernel<<<grid, block, kSmem, stream>>>(
      params,
      output.data_ptr<int32_t>(),
      topk_block_idx.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(),
      static_cast<int32_t>(k_block_size),
      block_topk);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "topk_coord_transform_fused_paged kernel failed");
}

void topk_coord_transform_fused_ragged_interface(
    const at::Tensor& score,                  // [B, sparse_len] f32
    const at::Tensor& lengths,                // [B] i32 — per-row valid length of `score`
    const at::Tensor& topk_block_idx,         // [B, block_topk] i32
    const at::Tensor& ks,                     // [B] i32 — per-row kv start
    const at::Tensor& ke,                     // [B] i32 — per-row kv end
    at::Tensor& output,                       // [B, TopK] i32 (out)
    int64_t k_block_size) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(topk_block_idx);
  CHECK_CUDA(ks);
  CHECK_CUDA(ke);
  CHECK_CUDA(output);

  const auto params = get_params(score, lengths, std::nullopt);
  const auto B = score.size(0);
  TORCH_CHECK(output.dim() == 2 && output.is_contiguous());
  TORCH_CHECK(output.size(0) == B && output.size(1) == TopK);
  TORCH_CHECK(topk_block_idx.dim() == 2 && topk_block_idx.is_contiguous());
  TORCH_CHECK(topk_block_idx.size(0) == B);
  TORCH_CHECK(ks.dim() == 1 && ks.is_contiguous() && ks.size(0) == B);
  TORCH_CHECK(ke.dim() == 1 && ke.is_contiguous() && ke.size(0) == B);
  const int32_t block_topk = static_cast<int32_t>(topk_block_idx.size(1));

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  setup_kernel_smem_once<topk_coord_transform_fused_ragged_kernel, kSmem>();
  topk_coord_transform_fused_ragged_kernel<<<grid, block, kSmem, stream>>>(
      params,
      output.data_ptr<int32_t>(),
      topk_block_idx.data_ptr<int32_t>(),
      ks.data_ptr<int32_t>(),
      ke.data_ptr<int32_t>(),
      static_cast<int32_t>(k_block_size),
      block_topk);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "topk_coord_transform_fused_ragged kernel failed");
}

void topk_transform_paged_interface(
    const at::Tensor& /*score*/,
    const at::Tensor& /*lengths*/,
    const at::Tensor& /*topk_block_idx*/,
    const at::Tensor& /*page_table_1*/,
    at::Tensor& /*output*/,
    int64_t /*k_block_size*/) {
  TORCH_CHECK(false,
              "topk_transform_paged not implemented yet. This name is reserved "
              "for the SGLANG_NSA_FUSE_TOPK=1 path (page_table_1-output). Use "
              "topk_coord_transform_fused_paged for the current "
              "(token-position-output) variant.");
}

void topk_transform_ragged_interface(
    const at::Tensor& /*score*/,
    const at::Tensor& /*lengths*/,
    const at::Tensor& /*topk_block_idx*/,
    const at::Tensor& /*topk_indices_offset*/,
    at::Tensor& /*output*/,
    int64_t /*k_block_size*/) {
  TORCH_CHECK(false,
              "topk_transform_ragged not implemented yet. This name is reserved "
              "for the SGLANG_NSA_FUSE_TOPK=1 path (ragged page-table-style "
              "output). Use topk_coord_transform_fused_ragged for the current "
              "(token-position-output) variant.");
}


TORCH_LIBRARY(hisa_topk_fused, m) {
  // Currently implemented (token-position output — replaces fast_topk_v2 +
  // hisa_coord_transform).
  m.def(
      "topk_coord_transform_fused_paged(Tensor score, Tensor lengths, Tensor topk_block_idx, "
      "Tensor seq_lens, Tensor(a!) output, int k_block_size) -> ()");
  m.def(
      "topk_coord_transform_fused_ragged(Tensor score, Tensor lengths, Tensor topk_block_idx, "
      "Tensor ks, Tensor ke, Tensor(a!) output, int k_block_size) -> ()");
  // Reserved for SGLANG_NSA_FUSE_TOPK=1 — page_table_1 output. Names match
  // upstream's family (`topk_transform_decode_kernel` /
  // `topk_transform_prefill_ragged_kernel`). Stubs raise at call time.
  m.def(
      "topk_transform_paged(Tensor score, Tensor lengths, Tensor topk_block_idx, "
      "Tensor page_table_1, Tensor(a!) output, int k_block_size) -> ()");
  m.def(
      "topk_transform_ragged(Tensor score, Tensor lengths, Tensor topk_block_idx, "
      "Tensor topk_indices_offset, Tensor(a!) output, int k_block_size) -> ()");
}

TORCH_LIBRARY_IMPL(hisa_topk_fused, CUDA, m) {
  m.impl("topk_coord_transform_fused_paged", topk_coord_transform_fused_paged_interface);
  m.impl("topk_coord_transform_fused_ragged", topk_coord_transform_fused_ragged_interface);
  m.impl("topk_transform_paged", topk_transform_paged_interface);
  m.impl("topk_transform_ragged", topk_transform_ragged_interface);
}
