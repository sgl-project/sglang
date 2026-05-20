/**
 * HISA — fused topk + coord-transform kernel.
 *
 * Adapted from sgl-kernel/csrc/elementwise/topk.cu. The radix-select machinery
 * (``TopK`` / ``kThreadsPerBlock`` / ``kSmem`` / ``FastTopKParams`` /
 * ``naive_topk_*`` / ``convert_to_uint*`` / ``fast_topk_cuda_tl`` /
 * ``get_params`` / ``setup_kernel_smem_once``) is copied verbatim from upstream
 * so any perf delta vs ``fast_topk_v2`` comes only from the modified epilogue,
 * not from helper drift.
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
#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <torch/library.h>

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
// Upper bound for block_topk so we can statically allocate s_abs_blocks.
// Production max: K=8 -> block_topk = 8192/8 = 1024.
constexpr int kMaxBlockTopk = 1024;

#ifdef USE_ROCM
#ifdef SGL_TOPK_DYNAMIC_SMEM_BYTES
constexpr size_t kSmem = static_cast<size_t>(SGL_TOPK_DYNAMIC_SMEM_BYTES);
#else
constexpr size_t kSmem = 48 * 1024; // bytes
#endif
#else
constexpr size_t kSmem = 8 * 1024 * sizeof(uint32_t); // 32KB (bytes)
#endif

struct FastTopKParams {
  const float *__restrict__ input;        // [B, input_stride]
  const int32_t *__restrict__ row_starts; // [B]
  int32_t *__restrict__ indices;          // [B, TopK]
  int32_t *__restrict__ lengths;          // [B]
  int64_t input_stride;
};

// when length <= TopK, we can directly write the indices
__device__ void naive_topk_cuda(const float *__restrict__ score,
                                int32_t *__restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < TopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void
naive_topk_transform(const float *__restrict__ score, int32_t length,
                     int32_t *__restrict__ dst_page_table,
                     const int32_t *__restrict__ src_page_table) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    dst_page_table[i] = (i < length) ? src_page_table[i] : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void
naive_topk_transform_ragged(const float *__restrict__ score, int32_t length,
                            int32_t *__restrict__ topk_indices_ragged,
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

__device__ void fast_topk_cuda_tl(const float *__restrict__ input,
                                  int *__restrict__ index, int row_start,
                                  int length) {
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

  auto &s_histogram = s_histogram_buf[0];
  // allocate for two rounds
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  // stage 1: 8bit coarse histogram
  if (tx < RADIX + 1)
    s_histogram[tx] = 0;
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
        const auto bin =
            (convert_to_uint32(input[idx + row_start]) >> offset) & 0xFF;
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

__global__ __launch_bounds__(kThreadsPerBlock) // hisa paged decode
    void topk_coord_transform_fused_paged_kernel(
        const FastTopKParams params,
        int32_t *__restrict__ output,               // [B, TopK] i32
        const int32_t *__restrict__ topk_block_idx, // [B, BLOCK_TOPK] i32
        const int32_t
            *__restrict__ seq_lens, // [B] i32 — per-req absolute seq_len
        int32_t k_block_size, int32_t block_topk) {
  const auto &[input, _1, _2, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = 0;
  const auto length = lengths[bid];
  const auto out_entry = output + bid * TopK;
  const auto score = input + bid * input_stride;

  // Preload abs_blocks (per-batch, <= kMaxBlockTopk ints) into smem so the
  // epilogue's `abs_blocks[slot]` reads are smem-resident.
  // (c) Skip explicit sync after preload in slow path; fast_topk_cuda_tl's
  // first internal sync establishes the barrier.
  __shared__ int s_abs_blocks[kMaxBlockTopk];
  {
    const int32_t *g_abs_blocks = topk_block_idx + bid * block_topk;
    for (int i = tid; i < block_topk; i += kThreadsPerBlock) {
      s_abs_blocks[i] = g_abs_blocks[i];
    }
  }

  if (length <= TopK) {
    __syncthreads();
    const int32_t batch_seq_len = seq_lens[bid];
    for (int i = tid; i < TopK; i += kThreadsPerBlock) {
      if (i < length) {
        const int slot = i / k_block_size;
        const int abs_block = s_abs_blocks[slot];
        // (a) `i % K` for AND-mask parallelism with `i / K`.
        const int raw = abs_block * k_block_size + (i % k_block_size);
        const bool pos_valid = raw < batch_seq_len;
        out_entry[i] = pos_valid ? raw : -1;
      } else {
        out_entry[i] = -1;
      }
    }
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const int32_t batch_seq_len = seq_lens[bid];

    // (b) Manually interleave two iters for ILP. Removed defensive checks:
    // `r != -1` (radix-select fills all 2048 slots when length > TopK) and
    // `raw >= 0` (abs_block, intra both >= 0).
    const int idx_0 = tid;
    const int idx_1 = tid + kThreadsPerBlock;
    const int r0 = s_indices[idx_0];
    const int r1 = s_indices[idx_1];
    const int slot0 = r0 / k_block_size;
    const int slot1 = r1 / k_block_size;
    const int abs0 = s_abs_blocks[slot0];
    const int abs1 = s_abs_blocks[slot1];
    // (a) `r % K` for AND-mask.
    const int raw0 = abs0 * k_block_size + (r0 % k_block_size);
    const int raw1 = abs1 * k_block_size + (r1 % k_block_size);
    const bool valid0 = raw0 < batch_seq_len;
    const bool valid1 = raw1 < batch_seq_len;
    out_entry[idx_0] = valid0 ? raw0 : -1;
    out_entry[idx_1] = valid1 ? raw1 : -1;
  }
}

__global__ __launch_bounds__(kThreadsPerBlock) // hisa ragged prefill
    void topk_coord_transform_fused_ragged_kernel(
        const FastTopKParams params,
        int32_t *__restrict__ output,               // [B, TopK] i32
        const int32_t *__restrict__ topk_block_idx, // [B, BLOCK_TOPK] i32
        const int32_t *__restrict__ ks, // [B] i32 — per-row kv start
        const int32_t *__restrict__ ke, // [B] i32 — per-row kv end
        int32_t k_block_size, int32_t block_topk) {
  const auto &[input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto out_entry = output + bid * TopK;
  const auto score = input + bid * input_stride;

  const int32_t row_ks = ks[bid];
  const int32_t row_ke = ke[bid];
  const int32_t row_extent = row_ke - row_ks;

  // (c) Skip explicit sync after preload in slow path.
  __shared__ int s_abs_blocks[kMaxBlockTopk];
  {
    const int32_t *g_abs_blocks = topk_block_idx + bid * block_topk;
    for (int i = tid; i < block_topk; i += kThreadsPerBlock) {
      s_abs_blocks[i] = g_abs_blocks[i];
    }
  }

  if (length <= TopK) {
    __syncthreads();
    for (int i = tid; i < TopK; i += kThreadsPerBlock) {
      if (i < length) {
        const int slot = i / k_block_size;
        const int abs_block = s_abs_blocks[slot];
        const int raw = abs_block * k_block_size + (i % k_block_size); // (a)
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
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);

    // (b) Interleaved + (a) `r % K` + dropped defensive checks.
    const int idx_0 = tid;
    const int idx_1 = tid + kThreadsPerBlock;
    const int r0 = s_indices[idx_0];
    const int r1 = s_indices[idx_1];
    const int slot0 = r0 / k_block_size;
    const int slot1 = r1 / k_block_size;
    const int abs0 = s_abs_blocks[slot0];
    const int abs1 = s_abs_blocks[slot1];
    const int raw0 = abs0 * k_block_size + (r0 % k_block_size);
    const int raw1 = abs1 * k_block_size + (r1 % k_block_size);
    const int rel0 = raw0 - row_ks;
    const int rel1 = raw1 - row_ks;
    const bool valid0 = (rel0 >= 0) && (rel0 < row_extent);
    const bool valid1 = (rel1 >= 0) && (rel1 < row_extent);
    out_entry[idx_0] = valid0 ? rel0 : -1;
    out_entry[idx_1] = valid1 ? rel1 : -1;
  }
}

// =============================================================================
// SGLANG_NSA_FUSE_TOPK=1 kernels — output matches upstream's
// `fast_topk_transform_{fused,ragged_fused}` contract so the indexer's
// output can be used as a transformed page_table_1 directly.
//   PAGED:  out[i] = page_table_1[batch, raw]
//   RAGGED: out[i] = raw_rel + topk_indices_offset[batch]
// where raw / raw_rel come from HISA's coord transform.
// =============================================================================

// Removed defensive checks vs an earlier draft:
//   - `r != -1`: slow path's s_indices is filled by fast_topk_cuda_tl which
//     guarantees all 2048 entries are valid (length > TopK invariant).
//   - `raw >= 0`: abs_block >= 0 (from topk_block_idx) and intra >= 0, so
//     raw = abs_block*K + intra is always >= 0.
// Templating on K_BLOCK was tried and is a no-op — nvcc already lowers
// `r / k_block_size` to magic-number-multiply for runtime int32, no IDIV
// in the SASS. The remaining ~6us F-C gap is from elsewhere (smem bank
// conflicts on s_abs_blocks, sync barrier, register pressure).
__global__ __launch_bounds__(kThreadsPerBlock) // SGLANG_NSA_FUSE_TOPK=1 paged
    void topk_transform_paged_kernel(
        const FastTopKParams params,
        int32_t *__restrict__ output,               // [B, TopK] i32
        const int32_t *__restrict__ topk_block_idx, // [B, BLOCK_TOPK] i32
        const int32_t *__restrict__ seq_lens,       // [B] i32
        const int32_t *__restrict__ page_table_1, // [B, page_table_stride] i32
        int32_t k_block_size, int32_t block_topk, int64_t page_table_stride) {
  const auto &[input, _1, _2, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = 0;
  const auto length = lengths[bid];
  const auto out_entry = output + bid * TopK;
  const auto score = input + bid * input_stride;
  const auto pt_row = page_table_1 + bid * page_table_stride;

  // (c) Skip the explicit __syncthreads() after preload in the slow path;
  // fast_topk_cuda_tl's first internal __syncthreads() (post histogram-zero)
  // suffices because s_abs_blocks isn't read until the epilogue. Fast path
  // syncs explicitly inside its branch.
  __shared__ int s_abs_blocks[kMaxBlockTopk];
  {
    const int32_t *g_abs_blocks = topk_block_idx + bid * block_topk;
    for (int i = tid; i < block_topk; i += kThreadsPerBlock) {
      s_abs_blocks[i] = g_abs_blocks[i];
    }
  }

  if (length <= TopK) {
    __syncthreads(); // fast path needs sync; no fast_topk_cuda_tl to provide
                     // one
    const int32_t batch_seq_len = seq_lens[bid];
    for (int i = tid; i < TopK; i += kThreadsPerBlock) {
      if (i < length) {
        const int slot = i / k_block_size;
        const int abs_block = s_abs_blocks[slot];
        const int raw = abs_block * k_block_size + (i % k_block_size);
        const bool pos_valid = raw < batch_seq_len;
        // OOB-safe gather: clamp index, unconditional load, then select. The
        // ternary `pos_valid ? pt_row[raw] : -1` could let nvcc emit an
        // unconditional load with the OOB raw → illegal memory access if
        // raw exceeds page_table_1's row stride (page_table_1 may be sized
        // to actual seq_len, not max_seqlen_k).
        const int raw_safe = pos_valid ? raw : 0;
        const int g = pt_row[raw_safe];
        out_entry[i] = pos_valid ? g : -1;
      } else {
        out_entry[i] = -1;
      }
    }
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const int32_t batch_seq_len = seq_lens[bid];

    // Interleaved 2 iters for ILP. OOB-safe gather (see fast path comment).
    const int idx_0 = tid;
    const int idx_1 = tid + kThreadsPerBlock;
    const int r0 = s_indices[idx_0];
    const int r1 = s_indices[idx_1];
    const int slot0 = r0 / k_block_size;
    const int slot1 = r1 / k_block_size;
    const int abs0 = s_abs_blocks[slot0];
    const int abs1 = s_abs_blocks[slot1];
    const int raw0 = abs0 * k_block_size + (r0 % k_block_size);
    const int raw1 = abs1 * k_block_size + (r1 % k_block_size);
    const bool valid0 = raw0 < batch_seq_len;
    const bool valid1 = raw1 < batch_seq_len;
    const int raw0_safe = valid0 ? raw0 : 0;
    const int raw1_safe = valid1 ? raw1 : 0;
    const int g0 = pt_row[raw0_safe];
    const int g1 = pt_row[raw1_safe];
    out_entry[idx_0] = valid0 ? g0 : -1;
    out_entry[idx_1] = valid1 ? g1 : -1;
  }
}

// SGLANG_NSA_FUSE_TOPK=1 paged PREFILL. CLONED FROM upstream's
// `topk_transform_prefill_kernel`
// (sgl-kernel/csrc/elementwise/topk.cu:300-350). The only HISA additions are:
//   1. Preload `topk_block_idx[bid, :block_topk]` into smem (`s_abs_blocks`).
//   2. Replace the trivial `dst[i] = src_page_entry[s_indices[i]]` gather
//      with `dst[i] = src_page_entry[raw]` where
//      `raw = s_abs_blocks[s_indices[i] / K] * K + s_indices[i] % K`,
//      masked by the per-token seq_lens for OOB.
__global__
__launch_bounds__(kThreadsPerBlock) void topk_transform_paged_prefill_kernel(
    const FastTopKParams params,
    int32_t *__restrict__ dst_page_table,       // [M, TopK] i32
    const int32_t *__restrict__ src_page_table, // [B, src_stride] i32
    const int64_t src_stride,
    const int32_t *__restrict__ cu_seqlens_q, // [B+1] i32 (cumulative)
    const int64_t prefill_bs,
    // ── HISA additions ──
    const int32_t *__restrict__ topk_block_idx, // [M, block_topk] i32
    const int32_t *__restrict__ seq_lens,       // [M] i32 (per-token)
    const int32_t k_block_size, const int32_t block_topk) {
  const auto &[input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto length = lengths[bid];
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;

  // === verbatim from upstream: bid → src_page_entry pointer ===
  __shared__ const int32_t *s_src_page_entry;
  if (C10_LIKELY(prefill_bs <= kThreadsPerBlock)) {
    if (tid < prefill_bs) {
      if (bid >= cu_seqlens_q[tid] && bid < cu_seqlens_q[tid + 1]) {
        s_src_page_entry = src_page_table + tid * src_stride;
      }
    }
  } else {
    for (int64_t i = tid; i < prefill_bs; i += kThreadsPerBlock) {
      if (bid >= cu_seqlens_q[i] && bid < cu_seqlens_q[i + 1]) {
        s_src_page_entry = src_page_table + i * src_stride;
      }
    }
  }
  __syncthreads();
  const auto src_page_entry = s_src_page_entry;
  // === end verbatim ===

  // ── HISA: preload abs_blocks for coord transform ──
  __shared__ int s_abs_blocks[kMaxBlockTopk];
  {
    const int32_t *g_abs_blocks = topk_block_idx + bid * block_topk;
    for (int i = tid; i < block_topk; i += kThreadsPerBlock) {
      s_abs_blocks[i] = g_abs_blocks[i];
    }
  }
  __syncthreads();
  const int32_t batch_seq_len = seq_lens[bid];

  if (length <= TopK) {
    // HISA fast path: index `i` is the s_indices for fast path.
    for (int i = tid; i < TopK; i += kThreadsPerBlock) {
      if (i < length) {
        const int slot = i / k_block_size;
        const int abs_block = s_abs_blocks[slot];
        const int raw = abs_block * k_block_size + (i % k_block_size);
        const bool pos_valid = raw < batch_seq_len;
        const int raw_safe =
            pos_valid ? raw : 0; // OOB-safe (see decode kernel)
        const int g = src_page_entry[raw_safe];
        dst_page_entry[i] = pos_valid ? g : -1;
      } else {
        dst_page_entry[i] = -1;
      }
    }
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);

    // Insert HISA coord transform between s_indices and the gather
    // (upstream did `dst[i] = src_page_entry[s_indices[i]]` directly).
    const int idx_0 = tid;
    const int idx_1 = tid + kThreadsPerBlock;
    const int r0 = s_indices[idx_0];
    const int r1 = s_indices[idx_1];
    const int slot0 = r0 / k_block_size;
    const int slot1 = r1 / k_block_size;
    const int abs0 = s_abs_blocks[slot0];
    const int abs1 = s_abs_blocks[slot1];
    const int raw0 = abs0 * k_block_size + (r0 % k_block_size);
    const int raw1 = abs1 * k_block_size + (r1 % k_block_size);
    const bool valid0 = raw0 < batch_seq_len;
    const bool valid1 = raw1 < batch_seq_len;
    // OOB-safe gather (see decode kernel).
    const int raw0_safe = valid0 ? raw0 : 0;
    const int raw1_safe = valid1 ? raw1 : 0;
    const int g0 = src_page_entry[raw0_safe];
    const int g1 = src_page_entry[raw1_safe];
    dst_page_entry[idx_0] = valid0 ? g0 : -1;
    dst_page_entry[idx_1] = valid1 ? g1 : -1;
  }
}

// SGLANG_NSA_FUSE_TOPK=1 ragged. Mirrors upstream's
// `topk_transform_prefill_ragged_kernel` epilogue contract
// (`out[i] = pos + offset`) but with HISA's coord transform inserted in
// front: `pos = (abs_block * K + (s % K)) - row_ks` (ks-relative).
__global__
__launch_bounds__(kThreadsPerBlock) void topk_transform_ragged_kernel(
    const FastTopKParams params,
    int32_t *__restrict__ output,                    // [B, TopK] i32
    const int32_t *__restrict__ topk_block_idx,      // [B, BLOCK_TOPK] i32
    const int32_t *__restrict__ ks,                  // [B] i32
    const int32_t *__restrict__ ke,                  // [B] i32
    const int32_t *__restrict__ topk_indices_offset, // [B] i32
    int32_t k_block_size, int32_t block_topk) {
  const auto &[input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto out_entry = output + bid * TopK;
  const auto score = input + bid * input_stride;

  const int32_t row_ks = ks[bid];
  const int32_t row_ke = ke[bid];
  const int32_t row_extent = row_ke - row_ks;
  const int32_t row_offset = topk_indices_offset[bid];

  // (c) See PAGED kernel comment — same sync optimization here.
  __shared__ int s_abs_blocks[kMaxBlockTopk];
  {
    const int32_t *g_abs_blocks = topk_block_idx + bid * block_topk;
    for (int i = tid; i < block_topk; i += kThreadsPerBlock) {
      s_abs_blocks[i] = g_abs_blocks[i];
    }
  }

  if (length <= TopK) {
    __syncthreads();
    for (int i = tid; i < TopK; i += kThreadsPerBlock) {
      if (i < length) {
        const int slot = i / k_block_size;
        const int abs_block = s_abs_blocks[slot];
        // (a) `i % K` for AND-mask parallelism with `i / K`.
        const int raw = abs_block * k_block_size + (i % k_block_size);
        const int raw_rel = raw - row_ks;
        const bool pos_valid = (raw_rel >= 0) && (raw_rel < row_extent);
        out_entry[i] = pos_valid ? (raw_rel + row_offset) : -1;
      } else {
        out_entry[i] = -1;
      }
    }
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);

    // (b) Manually interleave two unrolled iters for ILP.
    const int idx_0 = tid;
    const int idx_1 = tid + kThreadsPerBlock;
    const int r0 = s_indices[idx_0];
    const int r1 = s_indices[idx_1];
    const int slot0 = r0 / k_block_size;
    const int slot1 = r1 / k_block_size;
    const int abs0 = s_abs_blocks[slot0];
    const int abs1 = s_abs_blocks[slot1];
    // (a) `r % K` for AND-mask.
    const int raw0 = abs0 * k_block_size + (r0 % k_block_size);
    const int raw1 = abs1 * k_block_size + (r1 % k_block_size);
    const int rel0 = raw0 - row_ks;
    const int rel1 = raw1 - row_ks;
    const bool valid0 = (rel0 >= 0) && (rel0 < row_extent);
    const bool valid1 = (rel1 >= 0) && (rel1 < row_extent);
    out_entry[idx_0] = valid0 ? (rel0 + row_offset) : -1;
    out_entry[idx_1] = valid1 ? (rel1 + row_offset) : -1;
  }
}

// =============================================================================
// BEGIN verbatim copy from upstream (line 383-431).
// =============================================================================

auto get_params(const at::Tensor &score, const at::Tensor &lengths,
                std::optional<at::Tensor> row_starts_opt = std::nullopt,
                std::optional<at::Tensor> indices_opt = std::nullopt)
    -> FastTopKParams {
  const auto B = score.size(0);
  TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1);
  if (row_starts_opt.has_value()) {
    const auto &row_starts = row_starts_opt.value();
    TORCH_CHECK(row_starts.dim() == 1);
    TORCH_CHECK(row_starts.size(0) == B);
  }
  TORCH_CHECK(lengths.dim() == 1 && lengths.is_contiguous());
  TORCH_CHECK(lengths.size(0) == B);
  int32_t *indices_data_ptr = nullptr;
  if (indices_opt.has_value()) {
    const auto &indices = indices_opt.value();
    TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous());
    TORCH_CHECK(indices.size(0) == B);
    TORCH_CHECK(indices.size(1) == TopK);
    indices_data_ptr = indices.data_ptr<int32_t>();
  }

  return FastTopKParams{
      .input = score.data_ptr<float>(),
      .row_starts = row_starts_opt.has_value()
                        ? row_starts_opt->data_ptr<int32_t>()
                        : nullptr,
      .indices = indices_data_ptr,
      .lengths = lengths.data_ptr<int32_t>(),
      .input_stride = score.stride(0),
  };
}

template <auto *f, size_t max_dynamic_smem> void setup_kernel_smem_once() {
  [[maybe_unused]]
  static const auto result = [] {
#ifdef USE_ROCM
    return ::cudaFuncSetAttribute(reinterpret_cast<const void *>(f),
                                  ::cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  max_dynamic_smem);
#else
    return ::cudaFuncSetAttribute(
        f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#endif
  }();
  TORCH_CHECK(result == cudaSuccess,
              "set_up_kernel_once failed:", ::cudaGetErrorString(result));
}

// =============================================================================
// END verbatim copy from upstream.
// =============================================================================

} // namespace

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

void topk_coord_transform_fused_paged_interface(
    const at::Tensor &score,   // [B, sparse_len] f32
    const at::Tensor &lengths, // [B] i32 — per-row valid length of `score`
    const at::Tensor &topk_block_idx, // [B, block_topk] i32
    const at::Tensor
        &seq_lens,      // [B] i32 — per-req absolute seq_len for OOB mask
    at::Tensor &output, // [B, TopK] i32 (out)
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
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.is_contiguous() &&
              seq_lens.size(0) == B);
  const int32_t block_topk = static_cast<int32_t>(topk_block_idx.size(1));
  TORCH_CHECK(block_topk <= kMaxBlockTopk, "block_topk=", block_topk,
              " exceeds kMaxBlockTopk=", kMaxBlockTopk);

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  setup_kernel_smem_once<topk_coord_transform_fused_paged_kernel, kSmem>();
  topk_coord_transform_fused_paged_kernel<<<grid, block, kSmem, stream>>>(
      params, output.data_ptr<int32_t>(), topk_block_idx.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(), static_cast<int32_t>(k_block_size),
      block_topk);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "topk_coord_transform_fused_paged kernel failed");
}

void topk_coord_transform_fused_ragged_interface(
    const at::Tensor &score,   // [B, sparse_len] f32
    const at::Tensor &lengths, // [B] i32 — per-row valid length of `score`
    const at::Tensor &topk_block_idx, // [B, block_topk] i32
    const at::Tensor &ks,             // [B] i32 — per-row kv start
    const at::Tensor &ke,             // [B] i32 — per-row kv end
    at::Tensor &output,               // [B, TopK] i32 (out)
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
  TORCH_CHECK(block_topk <= kMaxBlockTopk, "block_topk=", block_topk,
              " exceeds kMaxBlockTopk=", kMaxBlockTopk);

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  setup_kernel_smem_once<topk_coord_transform_fused_ragged_kernel, kSmem>();
  topk_coord_transform_fused_ragged_kernel<<<grid, block, kSmem, stream>>>(
      params, output.data_ptr<int32_t>(), topk_block_idx.data_ptr<int32_t>(),
      ks.data_ptr<int32_t>(), ke.data_ptr<int32_t>(),
      static_cast<int32_t>(k_block_size), block_topk);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "topk_coord_transform_fused_ragged kernel failed");
}

void topk_transform_paged_interface(
    const at::Tensor &score,   // [M, sparse_len] f32 (M = total query tokens)
    const at::Tensor &lengths, // [M] i32 — per-token valid length of `score`
    const at::Tensor &topk_block_idx, // [M, block_topk] i32 (per-token)
    const at::Tensor &seq_lens, // [M] i32 — per-token K end (seq_lens) for OOB
    const at::Tensor &page_table_1, // [B, max_seqlen_k] i32 (per-batch!)
    const at::Tensor &cu_seqlens_q, // [B+1] i32 (cumulative)
    at::Tensor &output,             // [M, TopK] i32 (out)
    int64_t k_block_size) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(topk_block_idx);
  CHECK_CUDA(seq_lens);
  CHECK_CUDA(page_table_1);
  CHECK_CUDA(cu_seqlens_q);
  CHECK_CUDA(output);

  const auto params = get_params(score, lengths, std::nullopt);
  const auto M = score.size(0);
  TORCH_CHECK(output.dim() == 2 && output.is_contiguous());
  TORCH_CHECK(output.size(0) == M && output.size(1) == TopK);
  TORCH_CHECK(topk_block_idx.dim() == 2 && topk_block_idx.is_contiguous());
  TORCH_CHECK(topk_block_idx.size(0) == M);
  TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.is_contiguous() &&
              seq_lens.size(0) == M);
  TORCH_CHECK(page_table_1.dim() == 2,
              "page_table_1 must be 2D, got dim=", page_table_1.dim());
  TORCH_CHECK(page_table_1.stride(1) == 1,
              "page_table_1 must have stride(1)==1, got ",
              page_table_1.stride(1));
  TORCH_CHECK(
      cu_seqlens_q.dim() == 1 && cu_seqlens_q.is_contiguous(),
      "cu_seqlens_q must be 1D contiguous, got dim=", cu_seqlens_q.dim());
  const int32_t prefill_bs = static_cast<int32_t>(cu_seqlens_q.size(0) - 1);
  TORCH_CHECK(page_table_1.size(0) == prefill_bs, "page_table_1.size(0) (",
              page_table_1.size(0),
              ") must equal prefill_bs = cu_seqlens_q.size(0)-1 (", prefill_bs,
              ")");
  TORCH_CHECK(prefill_bs <= M, "prefill_bs (", prefill_bs,
              ") cannot exceed M=score.size(0) (", M, ")");
  const int32_t block_topk = static_cast<int32_t>(topk_block_idx.size(1));
  TORCH_CHECK(block_topk <= kMaxBlockTopk, "block_topk=", block_topk,
              " exceeds kMaxBlockTopk=", kMaxBlockTopk);
  const int64_t page_table_stride = page_table_1.stride(0);

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(M)};
  const auto block = dim3{kThreadsPerBlock};

  // Dispatch decode (1:1) vs prefill (cu_seqlens_q-based batch lookup),
  // mirroring upstream's `is_decode = !row_starts && prefill_bs == B`.
  const bool is_decode = (prefill_bs == M);
  if (is_decode) {
    setup_kernel_smem_once<topk_transform_paged_kernel, kSmem>();
    topk_transform_paged_kernel<<<grid, block, kSmem, stream>>>(
        params, output.data_ptr<int32_t>(), topk_block_idx.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(), page_table_1.data_ptr<int32_t>(),
        static_cast<int32_t>(k_block_size), block_topk, page_table_stride);
  } else {
    setup_kernel_smem_once<topk_transform_paged_prefill_kernel, kSmem>();
    topk_transform_paged_prefill_kernel<<<grid, block, kSmem, stream>>>(
        params,
        output.data_ptr<int32_t>(),       // dst_page_table
        page_table_1.data_ptr<int32_t>(), // src_page_table
        page_table_stride,                // src_stride
        cu_seqlens_q.data_ptr<int32_t>(), static_cast<int64_t>(prefill_bs),
        topk_block_idx.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
        static_cast<int32_t>(k_block_size), block_topk);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "topk_transform_paged kernel failed");
}

void topk_transform_ragged_interface(
    const at::Tensor &score,               // [B, sparse_len] f32
    const at::Tensor &lengths,             // [B] i32
    const at::Tensor &topk_block_idx,      // [B, block_topk] i32
    const at::Tensor &ks,                  // [B] i32
    const at::Tensor &ke,                  // [B] i32
    const at::Tensor &topk_indices_offset, // [B] i32
    at::Tensor &output,                    // [B, TopK] i32 (out)
    int64_t k_block_size) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(topk_block_idx);
  CHECK_CUDA(ks);
  CHECK_CUDA(ke);
  CHECK_CUDA(topk_indices_offset);
  CHECK_CUDA(output);

  const auto params = get_params(score, lengths, std::nullopt);
  const auto B = score.size(0);
  TORCH_CHECK(output.dim() == 2 && output.is_contiguous());
  TORCH_CHECK(output.size(0) == B && output.size(1) == TopK);
  TORCH_CHECK(topk_block_idx.dim() == 2 && topk_block_idx.is_contiguous());
  TORCH_CHECK(topk_block_idx.size(0) == B);
  TORCH_CHECK(ks.dim() == 1 && ks.is_contiguous() && ks.size(0) == B);
  TORCH_CHECK(ke.dim() == 1 && ke.is_contiguous() && ke.size(0) == B);
  TORCH_CHECK(topk_indices_offset.dim() == 1 &&
              topk_indices_offset.is_contiguous() &&
              topk_indices_offset.size(0) == B);
  const int32_t block_topk = static_cast<int32_t>(topk_block_idx.size(1));
  TORCH_CHECK(block_topk <= kMaxBlockTopk, "block_topk=", block_topk,
              " exceeds kMaxBlockTopk=", kMaxBlockTopk);

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  setup_kernel_smem_once<topk_transform_ragged_kernel, kSmem>();
  topk_transform_ragged_kernel<<<grid, block, kSmem, stream>>>(
      params, output.data_ptr<int32_t>(), topk_block_idx.data_ptr<int32_t>(),
      ks.data_ptr<int32_t>(), ke.data_ptr<int32_t>(),
      topk_indices_offset.data_ptr<int32_t>(),
      static_cast<int32_t>(k_block_size), block_topk);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "topk_transform_ragged kernel failed");
}

TORCH_LIBRARY(hisa_topk_fused, m) {
  // Currently implemented (token-position output — replaces fast_topk_v2 +
  // hisa_coord_transform).
  m.def("topk_coord_transform_fused_paged(Tensor score, Tensor lengths, Tensor "
        "topk_block_idx, "
        "Tensor seq_lens, Tensor(a!) output, int k_block_size) -> ()");
  m.def("topk_coord_transform_fused_ragged(Tensor score, Tensor lengths, "
        "Tensor topk_block_idx, "
        "Tensor ks, Tensor ke, Tensor(a!) output, int k_block_size) -> ()");
  // Reserved for SGLANG_NSA_FUSE_TOPK=1 — page_table_1 output. Names match
  // upstream's family (`topk_transform_decode_kernel` /
  // `topk_transform_prefill_ragged_kernel`). Stubs raise at call time.
  m.def("topk_transform_paged(Tensor score, Tensor lengths, Tensor "
        "topk_block_idx, "
        "Tensor seq_lens, Tensor page_table_1, Tensor cu_seqlens_q, "
        "Tensor(a!) output, int k_block_size) -> ()");
  m.def("topk_transform_ragged(Tensor score, Tensor lengths, Tensor "
        "topk_block_idx, "
        "Tensor ks, Tensor ke, Tensor topk_indices_offset, Tensor(a!) output, "
        "int k_block_size) -> ()");
}

TORCH_LIBRARY_IMPL(hisa_topk_fused, CUDA, m) {
  m.impl("topk_coord_transform_fused_paged",
         topk_coord_transform_fused_paged_interface);
  m.impl("topk_coord_transform_fused_ragged",
         topk_coord_transform_fused_ragged_interface);
  m.impl("topk_transform_paged", topk_transform_paged_interface);
  m.impl("topk_transform_ragged", topk_transform_ragged_interface);
}
