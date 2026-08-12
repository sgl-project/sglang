#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/compress.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/object.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>

namespace device::compress {

/// \brief Plan entry for online compress 128 prefill.
/// Each entry describes a contiguous segment of tokens that lies inside a
/// single 128-chunk. Multiple segments can map to the same batch id when the
/// extend tokens span chunk boundaries.
///
/// **Layout compatibility:** the field order/types match `PrefillPlan` so that
/// downstream kernels (e.g. `fused_norm_rope` in `CompressExtend` mode) can
/// consume the compress_plan tensor as-if it were a `PrefillPlan` tensor --
/// they only read `ragged_id` and `position`, both of which carry identical
/// semantics here (the LAST token of the segment in q-ragged and global
/// coordinates respectively).
///
/// Note that `window_len` here means "number of real tokens in this segment"
/// (1..128), which differs from `PrefillPlan::window_len`. Downstream kernels
/// that share the tensor MUST NOT read it under that name.
struct alignas(16) OnlinePrefillPlan {
  /// \brief Ragged-q position of the LAST token in this segment.
  /// Equal to `segment_start_ragged + window_len - 1`.
  uint32_t ragged_id;
  /// \brief Index into the `indices` / `load_indices` arrays.
  uint32_t batch_id;
  /// \brief Global position of the LAST token in this segment.
  /// For compress plans, `position % 128 == 127` (chunk-closing); for write
  /// plans, `position % 128 < 127`.
  uint32_t position;
  /// \brief Number of real tokens in this segment (1..128).
  /// The first segment token sits at `position - window_len + 1` (global) and
  /// at `ragged_id - window_len + 1` (ragged).
  uint32_t window_len;
};

static_assert(alignof(OnlinePrefillPlan) == alignof(PrefillPlan));
static_assert(sizeof(OnlinePrefillPlan) == sizeof(PrefillPlan));

}  // namespace device::compress

namespace host::compress {

using device::compress::OnlinePrefillPlan;
using OnlinePrefillPlanTensorDtype = uint8_t;
inline constexpr int64_t kOnlinePrefillPlanDim = 16;

static_assert(alignof(OnlinePrefillPlan) == sizeof(OnlinePrefillPlan));
static_assert(sizeof(OnlinePrefillPlan) == kOnlinePrefillPlanDim * sizeof(OnlinePrefillPlanTensorDtype));

}  // namespace host::compress

namespace {

using OnlinePlan = device::compress::OnlinePrefillPlan;
using IndiceT = int32_t;

/// \brief Need to reduce register usage to increase occupancy
struct Compress128OnlineDecodeParams {
  /** \brief Shape: `[num_indices, 1, head_dim * 3 (max, sum, kv) ]` \n */
  void* __restrict__ kv_score_buffer;
  /** \brief Shape: `[batch_size, head_dim * 2]` */
  const void* __restrict__ kv_score_input;
  /** \brief Shape: `[batch_size, head_dim]` */
  void* __restrict__ kv_compressed_output;
  /** \brief Shape: `[128, head_dim]` (called `ape`) */
  const void* __restrict__ score_bias;
  /** \brief Shape: `[batch_size, ]`*/
  const IndiceT* __restrict__ indices;
  /** \brief Shape: `[batch_size, ]` */
  const IndiceT* __restrict__ seq_lens;
  /** \NOTE: `batch_size` <= `num_indices` */
  uint32_t batch_size;
};

/// \brief Need to reduce register usage to increase occupancy
struct Compress128OnlinePrefillParams {
  /** \brief Shape: `[num_indices, 1, head_dim * 3 (max, sum, kv) ]` \n */
  void* __restrict__ kv_score_buffer;
  /** \brief Shape: `[num_q_tokens, head_dim * 2]` */
  const void* __restrict__ kv_score_input;
  /** \brief Shape: `[num_q_tokens, head_dim]` */
  void* __restrict__ kv_compressed_output;
  /** \brief Shape: `[128, head_dim]` (called `ape`) */
  const void* __restrict__ score_bias;
  /** \brief Shape: `[batch_size, ]`*/
  const IndiceT* __restrict__ indices;
  /** \brief Shape: `[batch_size, ]`*/
  const IndiceT* __restrict__ load_indices;
  /// \brief Plan for segments that close a chunk (write to `kv_compressed_output`).
  /// Shape: `[num_compress, 16]` (uint8).
  const OnlinePlan* __restrict__ compress_plan;
  /// \brief Plan for the trailing partial segment of each batch (write back to
  /// `kv_score_buffer`). Shape: `[num_write, 16]` (uint8).
  const OnlinePlan* __restrict__ write_plan;
  uint32_t num_compress;
  uint32_t num_write;
};

// 4 elements per thread, kHeadDim / 4 threads per block
template <int64_t kHeadDim, bool kUsePDL>
__global__ void flash_c128_online_decode(const __grid_constant__ Compress128OnlineDecodeParams params) {
  using namespace device;
  constexpr uint32_t kVecSize = 4;
  constexpr uint32_t kBlockSize = kHeadDim / kVecSize;
  using Vec = AlignedVector<float, kVecSize>;
  const auto gmem = tile::Memory<Vec>::cta(kBlockSize);
  const auto batch_id = blockIdx.x;
  const auto index = params.indices[batch_id];
  const auto seq_len = params.seq_lens[batch_id];

  const auto kv_score_buffer = static_cast<float*>(params.kv_score_buffer);
  const auto kv_buf = kv_score_buffer + index * (kHeadDim * 3);
  const auto kv_score_input = static_cast<const float*>(params.kv_score_input);
  const auto kv_src = kv_score_input + batch_id * (kHeadDim * 2);

  /// NOTE: kv_score_buffer layout is [max, sum, kv] (slot 0 / 1 / 2). Reads,
  /// writes, and the prefill kernel must all agree on this order.
  const auto max_score_vec = gmem.load(kv_buf, 0);
  const auto sum_score_vec = gmem.load(kv_buf, 1);
  const auto old_kv_vec = gmem.load(kv_buf, 2);

  /// NOTE: kv_score_input layout is | kv | score | (head_dim each), matching
  /// the offline c128 kernel and the online prefill kernel.
  const auto new_kv_vec = gmem.load(kv_src, 0);
  const auto new_score_raw_vec = gmem.load(kv_src, 1);

  /// NOTE: the new token sits at global position `seq_len - 1`, so its
  /// position inside the 128-chunk is `(seq_len - 1) % 128`. The previous
  /// `seq_len % 128` was off by one (`bias[127]` vs `bias[0]`, etc.).
  const auto pos_in_chunk = (seq_len - 1) % 128;
  const auto bias_vec = gmem.load(params.score_bias, pos_in_chunk);

  Vec out_kv_vec;
  Vec out_max_vec;
  Vec out_sum_vec;
  if (pos_in_chunk != 0) {
    // Mid-chunk: combine prior partial state with the new token via online softmax.
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const auto old_max = max_score_vec[i];
      const auto old_kv = old_kv_vec[i];
      const auto new_score = new_score_raw_vec[i] + bias_vec[i];
      const auto new_kv = new_kv_vec[i];
      const auto new_max = fmax(old_max, new_score);
      const auto old_sum = sum_score_vec[i] * expf(old_max - new_max);
      const auto new_exp = expf(new_score - new_max);
      const auto new_sum = old_sum + new_exp;
      out_kv_vec[i] = (old_kv * old_sum + new_kv * new_exp) / new_sum;
      out_max_vec[i] = new_max;
      out_sum_vec[i] = new_sum;
    }
  } else {
    // First token of a new 128-chunk: initialize state with this token alone.
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      out_kv_vec[i] = new_kv_vec[i];
      out_max_vec[i] = new_score_raw_vec[i] + bias_vec[i];
      out_sum_vec[i] = 1.0f;  // exp(score - max) with max == score
    }
  }

  if (pos_in_chunk == 127) {
    // Chunk just closed: emit the compressed kv. No need to update the buffer
    // -- the next chunk's first token will overwrite it.
    const auto kv_out = static_cast<float*>(params.kv_compressed_output) + batch_id * kHeadDim;
    gmem.store(kv_out, out_kv_vec);
  } else {
    // Otherwise persist the running [max, sum, kv] state for the next step.
    gmem.store(kv_buf, out_max_vec, 0);
    gmem.store(kv_buf, out_sum_vec, 1);
    gmem.store(kv_buf, out_kv_vec, 2);
  }
}

constexpr int32_t kTileElements = 2;  // split (along head-dim)
/// \brief Each warp will handle this many elements (split along softmax-128)
constexpr int32_t kElementsPerWarp = 8;
constexpr uint32_t kNumWarps = 128 / kElementsPerWarp;
constexpr uint32_t kPrefillBlockSize = device::kWarpThreads * kNumWarps;
using PrefillStorage = device::AlignedVector<float, kTileElements>;

struct Compress128SharedBuffer {
  using Storage = device::AlignedVector<float, 4>;
  Storage data[kNumWarps][device::kWarpThreads + 1];  // padding to avoid bank conflict
  SGL_DEVICE Storage& operator()(uint32_t warp_id, uint32_t lane_id) {
    return data[warp_id][lane_id];
  }
  SGL_DEVICE float& operator()(uint32_t warp_id, uint32_t lane_id, uint32_t tile_id) {
    return data[warp_id][lane_id][tile_id];
  }
};

template <bool kNeedData>
SGL_DEVICE void c128_prefill_forward(
    const PrefillStorage (&kv)[kElementsPerWarp],
    const PrefillStorage (&score)[kElementsPerWarp],
    float* kv_out,
    float* max_out,
    float* sum_out,
    const uint32_t warp_id,
    const uint32_t lane_id) {
  using namespace device;

  /// NOTE: part 2: safe online softmax + weighted sum
  using TmpStorage = typename Compress128SharedBuffer::Storage;
  __shared__ Compress128SharedBuffer s_local_val_max;
  __shared__ Compress128SharedBuffer s_local_exp_sum;
  __shared__ Compress128SharedBuffer s_local_product;

  TmpStorage tmp_val_max;
  TmpStorage tmp_exp_sum;
  TmpStorage tmp_product;

#pragma unroll
  for (int32_t i = 0; i < kTileElements; ++i) {
    float score_fp32[kElementsPerWarp];

#pragma unroll
    for (int32_t j = 0; j < kElementsPerWarp; ++j) {
      score_fp32[j] = score[j][i];
    }

    float max_value = score_fp32[0];
    float sum_exp_value = 0.0f;

#pragma unroll
    for (int32_t j = 1; j < kElementsPerWarp; ++j) {
      const auto fp32_score = score_fp32[j];
      max_value = fmaxf(max_value, fp32_score);
    }

    float sum_product = 0.0f;
#pragma unroll
    for (int32_t j = 0; j < 8; ++j) {
      const auto fp32_score = score_fp32[j];
      const auto exp_score = expf(fp32_score - max_value);
      sum_product += cast<float>(kv[j][i]) * exp_score;
      sum_exp_value += exp_score;
    }

    tmp_val_max[i] = max_value;
    tmp_exp_sum[i] = sum_exp_value;
    tmp_product[i] = sum_product;
  }

  // naturally aligned, so no bank conflict
  s_local_val_max(warp_id, lane_id) = tmp_val_max;
  s_local_exp_sum(warp_id, lane_id) = tmp_exp_sum;
  s_local_product(warp_id, lane_id) = tmp_product;

  __syncthreads();

  /// NOTE: part 3: online softmax
  /// NOTE: We have `kTileElements * kWarpThreads * kNumWarps` values to reduce
  /// each reduce will consume `kNumWarps` threads (use partial warp reduction)
  constexpr uint32_t kReductionCount = kTileElements * kWarpThreads * kNumWarps;
  constexpr uint32_t kIteration = kReductionCount / kPrefillBlockSize;

#pragma unroll
  for (uint32_t i = 0; i < kIteration; ++i) {
    /// NOTE: Range `[0, kTileElements * kWarpThreads * kNumWarps)`
    const uint32_t j = i * kPrefillBlockSize + warp_id * kWarpThreads + lane_id;
    /// NOTE: Range `[0, kNumWarps)`
    const uint32_t local_warp_id = j % kNumWarps;
    /// NOTE: Range `[0, kTileElements * kWarpThreads)`
    const uint32_t local_elem_id = j / kNumWarps;
    /// NOTE: Range `[0, kTileElements)`
    const uint32_t local_tile_id = local_elem_id % kTileElements;
    /// NOTE: Range `[0, kWarpThreads)`
    const uint32_t local_lane_id = local_elem_id / kTileElements;
    /// NOTE: each warp will access the whole tile (all `kTileElements`)
    /// and for different lanes, the memory access only differ in `local_warp_id`
    /// so there's no bank conflict in shared memory access.
    static_assert(kTileElements * kNumWarps == kWarpThreads, "TODO: support other configs");
    const auto local_val_max = s_local_val_max(local_warp_id, local_lane_id, local_tile_id);
    const auto local_exp_sum = s_local_exp_sum(local_warp_id, local_lane_id, local_tile_id);
    const auto local_product = s_local_product(local_warp_id, local_lane_id, local_tile_id);
    const auto global_val_max = warp::reduce_max<kNumWarps>(local_val_max);
    const auto rescale = expf(local_val_max - global_val_max);
    const auto global_exp_sum = warp::reduce_sum<kNumWarps>(local_exp_sum * rescale);
    const auto final_scale = rescale / global_exp_sum;
    const auto global_product = warp::reduce_sum<kNumWarps>(local_product * final_scale);
    kv_out[local_elem_id] = global_product;
    if constexpr (kNeedData) {
      max_out[local_elem_id] = global_val_max;
      sum_out[local_elem_id] = global_exp_sum;
    }
  }
  if constexpr (kNeedData) __syncthreads();
}

/// \brief Sentinel score for padded positions in a 128-segment.
/// Must be finite so that `score - max` never produces NaN even when an
/// entire warp has only padded positions.
constexpr float kPadScore = -FLT_MAX;

/// \brief Online compress 128 prefill. Two passes share this body:
/// - `kWrite=false` (compress pass): handles segments that close a chunk.
///   May load prior partial state from the buffer, but never writes to it,
///   so concurrent blocks can read the same slot without racing.
/// - `kWrite=true` (write pass): handles the trailing partial segment of each
///   batch. Each batch contributes at most one such plan, so concurrent blocks
///   touch disjoint buffer slots.
///
/// The two passes MUST run as separate kernel launches (in stream order) so
/// that all reads in pass 1 finish before any writes in pass 2 start.
template <int64_t kHeadDim, bool kWrite, bool kUsePDL>
__global__ __launch_bounds__(kPrefillBlockSize, 2)  //
    void flash_c128_online_prefill(const __grid_constant__ Compress128OnlinePrefillParams params) {
  using namespace device;

  constexpr int64_t kTileDim = kTileElements * kWarpThreads;  // 64
  constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  static_assert(kHeadDim % kTileDim == 0, "Head dim must be multiple of tile dim");

  /// NOTE: the compiler folds the if-else at compile time.
  const auto num_plans = kWrite ? params.num_write : params.num_compress;
  const auto plan_ptr = kWrite ? params.write_plan : params.compress_plan;
  const uint32_t global_id = blockIdx.x;
  const uint32_t global_pid = global_id / kNumSplit;  // plan id
  const uint32_t global_sid = global_id % kNumSplit;  // split id
  if (global_pid >= num_plans) return;
  const auto [ragged_id, batch_id, position, window_len] = plan_ptr[global_pid];
  if (ragged_id == 0xFFFFFFFFu) [[unlikely]]
    return;

  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  const int32_t split_offset = global_sid * kTileDim;  // int32 is enough

  const auto kv_score_buffer = static_cast<float*>(params.kv_score_buffer);
  const auto kv_score_input = static_cast<const float*>(params.kv_score_input);
  const auto kv_compressed_output = static_cast<float*>(params.kv_compressed_output);
  const auto score_bias_base = static_cast<const float*>(params.score_bias);

  constexpr int64_t kElementSize = kHeadDim * 2;  // | kv | score |
  const uint32_t chunk_offset = (position % 128u) + 1u - window_len;
  const uint32_t window_end = chunk_offset + window_len;        // exclusive, in [1, 128]
  const int32_t segment_start = ragged_id - (position % 128u);  // can be negative, but safe
  const int32_t load_index = chunk_offset != 0 ? params.load_indices[batch_id] : -1;
  const int32_t store_index = kWrite ? params.indices[batch_id] : -1;

  PDLWaitPrimary<kUsePDL>();

  // 2 * 8 = 16 register per elem. in theory we should consume 48 register here
  PrefillStorage kv[kElementsPerWarp];
  PrefillStorage score[kElementsPerWarp];
  PrefillStorage bias[kElementsPerWarp];
  const auto warp_offset = warp_id * kElementsPerWarp;

#pragma unroll
  for (uint32_t i = 0; i < kElementsPerWarp; ++i) {
    const uint32_t j = i + warp_offset;
    if (j >= chunk_offset && j < window_end) {
      const auto kv_src_ptr = kv_score_input + (segment_start + j) * kElementSize + split_offset;
      const auto score_src_ptr = kv_src_ptr + kHeadDim;
      const auto bias_src_ptr = score_bias_base + j * kHeadDim + split_offset;
      kv[i].load(kv_src_ptr, lane_id);
      score[i].load(score_src_ptr, lane_id);
      bias[i].load(bias_src_ptr, lane_id);
    }
  }

#pragma unroll
  for (uint32_t i = 0; i < kElementsPerWarp; ++i) {
    const uint32_t j = i + warp_offset;
    const bool is_valid = (j >= chunk_offset && j < window_end);
#pragma unroll
    for (uint32_t ii = 0; ii < kTileElements; ++ii) {
      score[i][ii] = is_valid ? score[i][ii] + bias[i][ii] : kPadScore;
      /// NOTE: must zero out kv on padded slots -- `c128_prefill_forward`
      /// computes `kv * exp_score` where `exp_score = expf(-FLT_MAX - max) ??? 0`,
      /// and IEEE-754 makes `NaN * 0 = NaN` / `+-inf * 0 = NaN`. An
      /// uninitialized register can hold a NaN/inf bit pattern, so without
      /// this reset a single padded warp can poison the whole softmax.
      kv[i][ii] = is_valid ? kv[i][ii] : 0.0f;
    }
  }

  __shared__ alignas(16) float seg_kv[kTileDim];
  __shared__ alignas(16) float seg_max[kTileDim];
  __shared__ alignas(16) float seg_sum[kTileDim];

  c128_prefill_forward<true>(kv, score, seg_kv, seg_max, seg_sum, warp_id, lane_id);

  PDLTriggerSecondary<kUsePDL>();

  if (warp_id == 0) {
    PrefillStorage out_kv_vec, out_max_vec, out_sum_vec;
    out_kv_vec.load(seg_kv, lane_id);
    out_max_vec.load(seg_max, lane_id);
    out_sum_vec.load(seg_sum, lane_id);
    if (chunk_offset != 0) {
      /// NOTE: load (max, sum, kv) of the in-progress chunk for this index.
      /// `load_indices` may differ from `indices` when the prior partial state
      /// lives on a different slot than the slot we ultimately write to.
      const auto buf_load = kv_score_buffer + load_index * (kHeadDim * 3) + split_offset;
      PrefillStorage buf_max_vec, buf_sum_vec, buf_kv_vec;
      buf_max_vec.load(buf_load + 0 * kHeadDim, lane_id);
      buf_sum_vec.load(buf_load + 1 * kHeadDim, lane_id);
      buf_kv_vec.load(buf_load + 2 * kHeadDim, lane_id);
#pragma unroll
      for (uint32_t ii = 0; ii < kTileElements; ++ii) {
        const float m1 = buf_max_vec[ii];
        const float s1 = buf_sum_vec[ii];
        const float k1 = buf_kv_vec[ii];
        const float m2 = out_max_vec[ii];
        const float s2 = out_sum_vec[ii];
        const float k2 = out_kv_vec[ii];
        const float new_max = fmaxf(m1, m2);
        const float new_s1 = s1 * expf(m1 - new_max);
        const float new_s2 = s2 * expf(m2 - new_max);
        const float new_sum = new_s1 + new_s2;
        const float new_kv = (k1 * new_s1 + k2 * new_s2) / new_sum;
        out_max_vec[ii] = new_max;
        out_sum_vec[ii] = new_sum;
        out_kv_vec[ii] = new_kv;
      }
    }

    if constexpr (kWrite) {
      const auto buf_store = kv_score_buffer + store_index * (kHeadDim * 3) + split_offset;
      reinterpret_cast<PrefillStorage*>(buf_store + 0 * kHeadDim)[lane_id] = out_max_vec;
      reinterpret_cast<PrefillStorage*>(buf_store + 1 * kHeadDim)[lane_id] = out_sum_vec;
      reinterpret_cast<PrefillStorage*>(buf_store + 2 * kHeadDim)[lane_id] = out_kv_vec;
    } else {
      const auto out_ptr = kv_compressed_output + ragged_id * kHeadDim + split_offset;
      reinterpret_cast<PrefillStorage*>(out_ptr)[lane_id] = out_kv_vec;
    }
  }
}

template <int64_t kHeadDim, bool kUsePDL>
struct FlashCompress128OnlineKernel {
  static constexpr auto decode_kernel = flash_c128_online_decode<kHeadDim, kUsePDL>;
  template <bool kWrite>
  static constexpr auto prefill_kernel = flash_c128_online_prefill<kHeadDim, kWrite, kUsePDL>;
  static constexpr auto prefill_c_kernel = prefill_kernel</*kWrite=*/false>;
  static constexpr auto prefill_w_kernel = prefill_kernel</*kWrite=*/true>;
  static constexpr int64_t kTileDim = kTileElements * device::kWarpThreads;  // 64
  static constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  static constexpr uint32_t kDecodeBlockSize = kHeadDim / 4;

  static void run_decode(
      const tvm::ffi::TensorView kv_score_buffer,
      const tvm::ffi::TensorView kv_score_input,
      const tvm::ffi::TensorView kv_compressed_output,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::Optional<tvm::ffi::TensorView> /* UNUSED */) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({-1, 1, kHeadDim * 3})  // kv score buffer (max, sum, kv)
        .with_dtype<float>()
        .with_device(device)
        .verify(kv_score_buffer);
    TensorMatcher({B, kHeadDim * 2})  // kv score input
        .with_dtype<float>()
        .with_device(device)
        .verify(kv_score_input);
    TensorMatcher({B, kHeadDim})  // kv compressed output
        .with_dtype<float>()
        .with_device(device)
        .verify(kv_compressed_output);
    TensorMatcher({128, kHeadDim})  // ape
        .with_dtype<float>()
        .with_device(device)
        .verify(ape);
    TensorMatcher({B}).with_dtype<IndiceT>().with_device(device).verify(indices);
    TensorMatcher({B}).with_dtype<IndiceT>().with_device(device).verify(seq_lens);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto params = Compress128OnlineDecodeParams{
        .kv_score_buffer = kv_score_buffer.data_ptr(),
        .kv_score_input = kv_score_input.data_ptr(),
        .kv_compressed_output = kv_compressed_output.data_ptr(),
        .score_bias = ape.data_ptr(),
        .indices = static_cast<const IndiceT*>(indices.data_ptr()),
        .seq_lens = static_cast<const IndiceT*>(seq_lens.data_ptr()),
        .batch_size = batch_size,
    };
    LaunchKernel(batch_size, kDecodeBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(decode_kernel, params);
  }

  static void run_prefill(
      const tvm::ffi::TensorView kv_score_buffer,
      const tvm::ffi::TensorView kv_score_input,
      const tvm::ffi::TensorView kv_compressed_output,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView compress_plan,
      const tvm::ffi::TensorView write_plan,
      const tvm::ffi::Optional<tvm::ffi::TensorView> extra) {
    using namespace host;
    using host::compress::kOnlinePrefillPlanDim;
    using host::compress::OnlinePrefillPlanTensorDtype;

    auto B = SymbolicSize{"batch_size"};
    auto N = SymbolicSize{"num_q_tokens"};
    auto X = SymbolicSize{"compress_tokens"};
    auto Y = SymbolicSize{"write_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({-1, 1, kHeadDim * 3})  // kv score buffer (max, sum, kv) ??? 2D
        .with_dtype<float>()
        .with_device(device_)
        .verify(kv_score_buffer);
    TensorMatcher({N, kHeadDim * 2})  // kv score input
        .with_dtype<float>()
        .with_device(device_)
        .verify(kv_score_input);
    TensorMatcher({N, kHeadDim})  // kv compressed output
        .with_dtype<float>()
        .with_device(device_)
        .verify(kv_compressed_output);
    TensorMatcher({128, kHeadDim})  // ape
        .with_dtype<float>()
        .with_device(device_)
        .verify(ape);
    TensorMatcher({B})  // indices
        .with_dtype<IndiceT>()
        .with_device(device_)
        .verify(indices);
    TensorMatcher({X, kOnlinePrefillPlanDim})  // compress plan
        .with_dtype<OnlinePrefillPlanTensorDtype>()
        .with_device(device_)
        .verify(compress_plan);
    TensorMatcher({Y, kOnlinePrefillPlanDim})  // write plan
        .with_dtype<OnlinePrefillPlanTensorDtype>()
        .with_device(device_)
        .verify(write_plan);

    /// NOTE: `extra` is `load_indices`. When the previous partial state lives
    /// on a slot different from the destination slot (e.g. paged buffers), the
    /// caller must supply this; otherwise it defaults to `indices`.
    const auto load_indices = extra.value_or(indices);
    TensorMatcher({B}).with_dtype<IndiceT>().with_device(device_).verify(load_indices);

    const auto device = device_.unwrap();
    const auto num_c = static_cast<uint32_t>(X.unwrap());
    const auto num_w = static_cast<uint32_t>(Y.unwrap());
    const auto params = Compress128OnlinePrefillParams{
        .kv_score_buffer = kv_score_buffer.data_ptr(),
        .kv_score_input = kv_score_input.data_ptr(),
        .kv_compressed_output = kv_compressed_output.data_ptr(),
        .score_bias = ape.data_ptr(),
        .indices = static_cast<const IndiceT*>(indices.data_ptr()),
        .load_indices = static_cast<const IndiceT*>(load_indices.data_ptr()),
        .compress_plan = static_cast<const OnlinePlan*>(compress_plan.data_ptr()),
        .write_plan = static_cast<const OnlinePlan*>(write_plan.data_ptr()),
        .num_compress = num_c,
        .num_write = num_w,
    };

    /// NOTE: pass 1 reads the buffer (for the first segment of each batch
    /// that started mid-chunk) and writes only to `kv_compressed_output`.
    /// Pass 2 then writes the trailing partial state of each batch back to
    /// the buffer. Stream serialization between the two launches enforces
    /// read-before-write on shared buffer slots.
    if (const auto num_c_blocks = num_c * kNumSplit) {
      LaunchKernel(num_c_blocks, kPrefillBlockSize, device)  //
          .enable_pdl(kUsePDL)(prefill_c_kernel, params);
    }
    if (const auto num_w_blocks = num_w * kNumSplit) {
      LaunchKernel(num_w_blocks, kPrefillBlockSize, device)  //
          .enable_pdl(kUsePDL)(prefill_w_kernel, params);
    }
  }
};

}  // namespace

namespace host::compress {

using OnlinePlanResult = tvm::ffi::Tuple<uint32_t, uint32_t>;

struct OnlinePrefillCompressParams {
  OnlinePrefillPlan* __restrict__ compress_plan;
  OnlinePrefillPlan* __restrict__ write_plan;
  const int64_t* __restrict__ seq_lens;
  const int64_t* __restrict__ extend_lens;
  uint32_t batch_size;
  uint32_t num_tokens;
};

/// \brief Build the compress + write plans for online compress 128 prefill.
///
/// Each batch's `[prefix_len, prefix_len + extend_len)` range is split at
/// 128-aligned boundaries. Every resulting segment falls into one of:
/// - **compress**: closes a 128-chunk (`chunk_offset + window_len == 128`).
///   These plans only read the buffer (when starting mid-chunk) and write the
///   compressed kv to `kv_compressed_output`.
/// - **write**: trailing partial of the batch (`chunk_offset + window_len < 128`).
///   May read the buffer and always writes the new partial state back to it.
///   Each batch produces at most one such plan.
///
/// The two plans MUST be dispatched as separate kernel launches in stream
/// order so that pass-1 reads of a buffer slot complete before any pass-2
/// write of the same slot.
inline OnlinePlanResult plan_online_prefill_host(const OnlinePrefillCompressParams& params, const bool use_cuda_graph) {
  const auto& [compress_plan, write_plan, seq_lens, extend_lens, batch_size, num_tokens] = params;

  uint32_t counter = 0;
  uint32_t compress_count = 0;
  uint32_t write_count = 0;
  for (const auto i : irange(batch_size)) {
    const uint32_t seq_len = static_cast<uint32_t>(seq_lens[i]);
    const uint32_t extend_len = static_cast<uint32_t>(extend_lens[i]);
    RuntimeCheck(0 < extend_len && extend_len <= seq_len);
    const uint32_t prefix_len = seq_len - extend_len;
    const uint32_t end_pos = prefix_len + extend_len;
    /// NOTE: split the extend range into per-128-chunk segments. Each segment
    /// stays inside one chunk, so the kernel can decide load/store from
    /// `chunk_offset` and `window_len` alone.
    uint32_t pos = prefix_len;
    while (pos < end_pos) {
      const uint32_t chunk_start = (pos / 128u) * 128u;
      const uint32_t seg_end = std::min(end_pos, chunk_start + 128u);  // exclusive
      const uint32_t seg_len = seg_end - pos;
      const uint32_t chunk_off = pos - chunk_start;
      /// NOTE: store last-token coordinates so that downstream consumers
      /// (e.g. `fused_norm_rope`) can read `ragged_id` and `position` with the
      /// same semantics as `PrefillPlan`. The segment start is recoverable as
      /// `ragged_id - window_len + 1` and `position - window_len + 1`.
      const uint32_t last_pos = seg_end - 1;
      const uint32_t last_ragged = counter + (last_pos - prefix_len);
      const auto plan = OnlinePrefillPlan{
          .ragged_id = last_ragged,
          .batch_id = i,
          .position = last_pos,
          .window_len = seg_len,
      };
      if (chunk_off + seg_len == 128u) {
        // full chunk, must be complete, maybe read the buffer, no write
        RuntimeCheck(compress_count < num_tokens);
        compress_plan[compress_count++] = plan;
      } else {
        // last chunk, must be incomplete, maybe read the buffer, must write
        RuntimeCheck(write_count < num_tokens);
        write_plan[write_count++] = plan;
      }
      pos = seg_end;
    }
    counter += extend_len;
  }
  RuntimeCheck(counter == num_tokens, "input size ", counter, " != num_q_tokens ", num_tokens);
  if (!use_cuda_graph) return OnlinePlanResult{compress_count, write_count};
  /// NOTE: pad both plans with sentinel entries so cuda-graph runs always see
  /// the same number of blocks. The kernel skips plans whose `ragged_id` is -1.
  constexpr auto kInvalid = static_cast<uint32_t>(-1);
  constexpr auto kInvalidPlan = OnlinePrefillPlan{kInvalid, kInvalid, kInvalid, kInvalid};
  for (const auto i : irange(compress_count, num_tokens)) {
    compress_plan[i] = kInvalidPlan;
  }
  for (const auto i : irange(write_count, num_tokens)) {
    write_plan[i] = kInvalidPlan;
  }
  return OnlinePlanResult{num_tokens, num_tokens};
}

inline OnlinePlanResult plan_online_prefill(
    const tvm::ffi::TensorView extend_lens,
    const tvm::ffi::TensorView seq_lens,
    const tvm::ffi::TensorView compress_plan,
    const tvm::ffi::TensorView write_plan,
    const bool use_cuda_graph) {
  auto N = SymbolicSize{"batch_size"};
  auto M = SymbolicSize{"num_tokens"};
  auto device = SymbolicDevice{};
  /// NOTE: only host (CPU/cuda-host) planning is implemented for now. The
  device.set_options<kDLCPU, kDLCUDAHost>();
  TensorMatcher({N})  //
      .with_dtype<int64_t>()
      .with_device(device)
      .verify(extend_lens)
      .verify(seq_lens);
  TensorMatcher({M, kOnlinePrefillPlanDim})  //
      .with_dtype<OnlinePrefillPlanTensorDtype>()
      .with_device(device)
      .verify(compress_plan)
      .verify(write_plan);
  const auto params = OnlinePrefillCompressParams{
      .compress_plan = static_cast<OnlinePrefillPlan*>(compress_plan.data_ptr()),
      .write_plan = static_cast<OnlinePrefillPlan*>(write_plan.data_ptr()),
      .seq_lens = static_cast<const int64_t*>(seq_lens.data_ptr()),
      .extend_lens = static_cast<const int64_t*>(extend_lens.data_ptr()),
      .batch_size = static_cast<uint32_t>(N.unwrap()),
      .num_tokens = static_cast<uint32_t>(M.unwrap()),
  };
  return plan_online_prefill_host(params, use_cuda_graph);
}

}  // namespace host::compress

namespace {

[[maybe_unused]]
constexpr auto& plan_compress_online_prefill = host::compress::plan_online_prefill;

}  // namespace
