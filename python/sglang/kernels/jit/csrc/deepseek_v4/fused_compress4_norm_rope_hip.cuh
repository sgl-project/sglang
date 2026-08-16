/**
 * \brief C4 decode compress fused with the norm + RoPE + fp8 store epilogue.
 *
 * The unfused chain runs flash_c4_decode, writes the compressed row to a
 * temporary, and immediately reads it back in fused_norm_rope_flashmla. That
 * temporary has no other consumer, so the round trip buys nothing but a second
 * kernel launch -- which at these sizes is most of the cost, since the norm/rope
 * kernel measures 4.3us against a launch floor of about 4us.
 *
 * Both halves already run one block per token, so fusing needs only a common
 * thread->element mapping. This uses 2 elements per lane (256 threads for
 * head_dim 512) rather than the compressor's 4, which puts thread tx on elements
 * [2tx, 2tx+1] -- exactly where the epilogue expects them, and leaves the fp8
 * groups at one warp / 64 elements / one UE8M0 scale so the bytes written to the
 * cache are unchanged.
 *
 * Buffer layouts are inherited unchanged from c4_v2.cuh:
 *   kv_buffer: [num_indices, 8, head_dim * 4]  (| kv overlap | kv | score overlap | score |)
 *   kv_input:  [batch_size, head_dim * 4]
 *   ape:       [8, head_dim]
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/c4_compress_core.cuh>
#include <sgl_kernel/deepseek_v4/compress_v2.cuh>
#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/object.h>

#include <bit>
#include <cfloat>
#include <cstdint>
#include <type_traits>

namespace sglang {

using PlanD = device::compress::DecodePlan;
using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

constexpr uint32_t kFusedBlockSize = 256;
constexpr uint32_t kFusedNumWarps = kFusedBlockSize / device::kWarpThreads;

#ifdef USE_ROCM
/// A `warp` is 32 lanes everywhere in this tree, but the gfx9 wavefront is 64.
/// The indexer fusion below cares about the difference: it decides both whether
/// an early-out is uniform and whether a cross-lane step needs a barrier.
constexpr uint32_t kWaveThreads = 64;

/// `warp::reduce_*` asserts its width down to kWarpThreads, so the wavefront
/// reductions the indexer fusion needs are spelled out here. The wavefront is a
/// single lockstep unit, so these need no barrier even when the rest of the
/// block has already exited.
SGL_DEVICE float wave_reduce_sum(float value) {
#pragma unroll
  for (uint32_t mask = kWaveThreads / 2; mask >= 1; mask >>= 1) {
    value += __shfl_xor(value, mask, kWaveThreads);
  }
  return value;
}

SGL_DEVICE float wave_reduce_max(float value) {
#pragma unroll
  for (uint32_t mask = kWaveThreads / 2; mask >= 1; mask >>= 1) {
    value = fmaxf(value, __shfl_xor(value, mask, kWaveThreads));
  }
  return value;
}
#endif

#define FUSED_C4_KERNEL __global__ __launch_bounds__(kFusedBlockSize, 4)

struct FusedCompress4NormRopeParams {
  void* __restrict__ kv_buffer;
  const void* __restrict__ kv_input;
  const void* __restrict__ score_bias;
  const void* __restrict__ norm_weight;
  const float* __restrict__ freqs_cis;
  const int64_t* __restrict__ out_loc;
  uint8_t* __restrict__ kvcache;
  const PlanD* __restrict__ plan_d;
  float eps;
  uint32_t compress_ratio;
  uint32_t batch_size;
};

/// \brief `kTileElements_` is elements per lane, and it is what reconciles the
/// compressor's decomposition with the epilogue it is being fused into:
///   - head_dim 512 (flashmla): 2, so a 256-thread block spans the whole row and
///     thread tx lands on elements [2tx, 2tx+1]. The compressor alone uses 4.
///   - head_dim 128 (indexer): 4, one warp per token, which is already what both
///     halves do -- no remap at all.
template <int64_t kHeadDim_, int32_t kTileElements_>
struct FusedC4Trait {
  static constexpr int32_t kTileElements = kTileElements_;
  static constexpr int64_t kTileDim = kTileElements * device::kWarpThreads;
  static constexpr int64_t kHeadDim = kHeadDim_;
  static constexpr int64_t kOverlapOffset = kHeadDim;
  static constexpr int64_t kScoreOffset = kHeadDim * 2;
  static constexpr int64_t kElementSize = kHeadDim * 4;
  static constexpr int64_t kPageElementSize = 4 * kElementSize;  // page size = 4
  static constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  static_assert(kHeadDim % kTileDim == 0);
};

// The c4 compress load + softmax core and the ring-buffer write are shared with
// the standalone c4 kernels: this fused HIP epilogue calls c4_compress_core() and
// c4_write_decode() from c4_compress_core.cuh (included above) so the math stays
// bit-identical to the two-kernel path and there is no duplicated copy to drift.

/// \brief compress -> RMSNorm -> RoPE / fp8 quant -> paged store, in one launch.
template <
    int64_t kHeadDim,
    typename BufferFloat,
    typename InputFloat,
    typename DType,
    int32_t kPageBits,
    bool kUsePDL,
    bool kBf16Store>
FUSED_C4_KERNEL void flash_c4_decode_norm_rope(const __grid_constant__ FusedCompress4NormRopeParams params) {
  using namespace device;
  using Trait = FusedC4Trait<kHeadDim, 2>;

  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = Trait::kTileElements;
  constexpr uint32_t kRopeWarp = kFusedNumWarps - 1;
  constexpr int64_t kPageBytes =
      kBf16Store ? (kHeadDim * 2ll << kPageBits) : host::div_ceil(584ll << kPageBits, 576) * 576;
  static_assert(kHeadDim == kFusedBlockSize * kVecSize);
  static_assert(kRopeDim == kWarpThreads * kVecSize);
  static_assert(kHeadDim - kRopeDim == kRopeWarp * kWarpThreads * kVecSize);
  using Float2 = AlignedVector<float, kVecSize>;
  using Storage = AlignedVector<DType, kVecSize>;

  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;
  const auto token_id = blockIdx.x;
  if (token_id >= params.batch_size) return;

  // One warp per head_dim split, one block per token.
  const int64_t split_offset = static_cast<int64_t>(warp_id) * Trait::kTileDim;

  const auto plan = params.plan_d[token_id];
  const auto kv_input = static_cast<const InputFloat*>(params.kv_input) + split_offset;
  const auto kv_buffer = static_cast<BufferFloat*>(params.kv_buffer) + split_offset;
  const auto score_bias = static_cast<const InputFloat*>(params.score_bias) + split_offset;

  const auto kv_src = kv_input + token_id * Trait::kElementSize;
  const auto kv_buf_0 = kv_buffer + plan.read_page_0 * Trait::kPageElementSize;
  const auto kv_buf_1 = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;
  const auto kv_dst = kv_buffer + plan.write_loc * Trait::kElementSize;

  PDLWaitPrimary<kUsePDL>();
  c4_write_decode<Trait, BufferFloat, InputFloat>(kv_dst, kv_src);

  // Matches the standalone pair: compress only emits on ratio boundaries, and
  // the norm/rope kernel returns early on exactly the same condition.
  if (plan.seq_len % params.compress_ratio != 0) return;

  const auto need_overlap = plan.seq_len > 4;
  Float2 data = c4_compress_core<Trait, BufferFloat, InputFloat>(
      kv_buf_0, kv_buf_1, kv_src, score_bias, need_overlap, 8);

  const auto position = static_cast<int32_t>(plan.seq_len - params.compress_ratio);
  const auto out_loc = params.out_loc[token_id];
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  Float2 freq;
  if (warp_id == kRopeWarp) freq.load(freqs_cis, lane_id);

  // part 1: RMSNorm. Sum of squares reduced across the block, which holds the
  // whole head_dim row.
  {
    __shared__ float partial_sums[kFusedNumWarps];

    Storage weight_vec;
    weight_vec.load(params.norm_weight, tx);

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      sum_of_squares += data[i] * data[i];
    }

    const auto warp_sum = warp::reduce_sum(sum_of_squares);
    if (lane_id == 0) partial_sums[warp_id] = warp_sum;
    __syncthreads();
    sum_of_squares = warp::reduce_sum<kFusedNumWarps>(partial_sums[lane_id % kFusedNumWarps]);
    const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_weight = cast<float>(weight_vec[i]);
      data[i] = data[i] * norm_factor * fp32_weight;
    }
  }

  const int64_t page = out_loc >> kPageBits;
  const int64_t offset = out_loc & ((1 << kPageBits) - 1);
  const auto page_ptr = params.kvcache + page * kPageBytes;
  const auto value_ptr = page_ptr + offset * (kBf16Store ? (kHeadDim * 2) : 576);

  PDLTriggerSecondary<kUsePDL>();

  // part 2: rope on the last warp, then either a plain bf16 store or a
  // per-warp fp8 group quant on the non-rope warps.
  if constexpr (kBf16Store) {
    Float2 d = data;
    if (warp_id == kRopeWarp) {
      const auto x_real = data[0];
      const auto x_imag = data[1];
      const auto freq_real = freq[0];
      const auto freq_imag = freq[1];
      d[0] = x_real * freq_real - x_imag * freq_imag;
      d[1] = x_real * freq_imag + x_imag * freq_real;
    }
    reinterpret_cast<bf16x2_t*>(value_ptr)[tx] = cast<bf16x2_t>(fp32x2_t{d[0], d[1]});
  } else if (warp_id == kRopeWarp) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto freq_real = freq[0];
    const auto freq_imag = freq[1];
    data[0] = x_real * freq_real - x_imag * freq_imag;
    data[1] = x_real * freq_imag + x_imag * freq_real;
    const auto result = cast<bf16x2_t>(fp32x2_t{data[0], data[1]});
    const auto rope_ptr = value_ptr + 448;
    reinterpret_cast<bf16x2_t*>(rope_ptr)[lane_id] = result;
  } else {
    // BF16 round-trip to match the precision of the non-fused path.
    const auto x = cast<float>(cast<bf16_t>(data[0]));
    const auto y = cast<float>(cast<bf16_t>(data[1]));
    const auto abs_max = warp::reduce_max(fmaxf(fabs(x), fabs(y)));
    const auto scale_raw = fmaxf(1e-4f, abs_max) / kFP8E4M3Max;
    const auto scale_ue8m0 = cast_to_ue8m0(scale_raw);
    const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);
    const auto result = pack_fp8(x * inv_scale, y * inv_scale);
    const auto scale_ptr = page_ptr + (576 << kPageBits) + offset * 8;
    reinterpret_cast<fp8x2_e4m3_t*>(value_ptr)[tx] = result;
    if (lane_id == 0) static_cast<uint8_t*>(scale_ptr)[warp_id] = scale_ue8m0;
  }
}

template <
    int64_t kHeadDim,
    typename BufferFloat,
    typename InputFloat,
    typename DType,
    uint32_t kPageSize,
    bool kUsePDL,
    bool kBf16Store>
struct FusedCompress4NormRopeKernel {
  static constexpr int32_t kLogPageSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes =
      kBf16Store ? (kHeadDim * 2 * kPageSize) : host::div_ceil(584 * kPageSize, 576) * 576;
  static constexpr auto kernel =
      flash_c4_decode_norm_rope<kHeadDim, BufferFloat, InputFloat, DType, kLogPageSize, kUsePDL, kBf16Store>;
  using Trait = FusedC4Trait<kHeadDim, 2>;

  static_assert(kHeadDim == 512, "fused c4 epilogue is defined for flashmla head_dim=512");
  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");

  static void run_decode(
      const tvm::ffi::TensorView kv_buffer,
      const tvm::ffi::TensorView kv_input,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView plan_d_,
      const tvm::ffi::TensorView norm_weight,
      const double eps,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView kvcache,
      const int64_t compress_ratio) {
    using namespace host;

    auto N = SymbolicSize{"batch_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    TensorMatcher({-1, 4, Trait::kElementSize})  // kv score
        .with_dtype<BufferFloat>()
        .with_device(device_)
        .verify(kv_buffer);
    TensorMatcher({N, Trait::kElementSize})  // kv score input
        .with_dtype<InputFloat>()
        .with_device(device_)
        .verify(kv_input);
    TensorMatcher({8, kHeadDim})  // ape
        .with_dtype<InputFloat>()
        .with_device(device_)
        .verify(ape);
    TensorMatcher({kHeadDim})  // norm weight
        .with_dtype<DType>()
        .with_device(device_)
        .verify(norm_weight);
    TensorMatcher({-1, 64})  // freqs_cis
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs_cis);
    TensorMatcher({-1})  // out_loc
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(out_loc);
    TensorMatcher({-1, -1})  // cache
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(kvcache);

    const auto plan_d = compress::verify_plan_d(plan_d_, N, device_);
    const auto batch_size = static_cast<uint32_t>(N.unwrap());
    if (batch_size == 0) return;
    RuntimeCheck(out_loc.size(0) == N.unwrap());

    const auto params = FusedCompress4NormRopeParams{
        .kv_buffer = kv_buffer.data_ptr(),
        .kv_input = kv_input.data_ptr(),
        .score_bias = ape.data_ptr(),
        .norm_weight = norm_weight.data_ptr(),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .out_loc = static_cast<const int64_t*>(out_loc.data_ptr()),
        .kvcache = static_cast<uint8_t*>(kvcache.data_ptr()),
        .plan_d = plan_d,
        .eps = static_cast<float>(eps),
        .compress_ratio = static_cast<uint32_t>(compress_ratio),
        .batch_size = batch_size,
    };
    // One block per token: the block owns the whole head_dim row.
    LaunchKernel(batch_size, kFusedBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

/// \brief compress -> RMSNorm -> RoPE -> hadamard -> fp8 quant -> paged store,
/// for the indexer (head_dim 128). Unlike the flashmla variant this needs no
/// remap: the compressor and the epilogue both already put one token on one
/// warp with 4 elements per lane, and every reduction here is warp-local, so
/// block size is free. 8 warps = 8 tokens per block, matching the epilogue.
template <
    typename BufferFloat,
    typename InputFloat,
    typename DType,
    int32_t kPageBits,
    bool kUsePDL,
    int32_t kPreshuffleSize>
FUSED_C4_KERNEL void flash_c4_decode_norm_rope_indexer(const __grid_constant__ FusedCompress4NormRopeParams params) {
  using namespace device;

  constexpr int64_t kHeadDim = 128;
  using Trait = FusedC4Trait<kHeadDim, 4>;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = Trait::kTileElements;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  constexpr int64_t kPageBytes = 132ll << kPageBits;
  static_assert(kHeadDim == kWarpThreads * kVecSize);
  static_assert(kRopeDim == kWarpThreads * 2);
  static_assert(Trait::kNumSplit == 1, "indexer fusion runs one warp per token");
  using Float4 = AlignedVector<float, kVecSize>;
  using Storage = AlignedVector<DType, kVecSize>;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto token_id = blockIdx.x * kFusedNumWarps + warp_id;
  // Lanes whose 4-elem pack lies in the rope tail (= last `kRopeSize` packs).
  const bool is_rope_lane = lane_id >= kWarpThreads - kRopeSize;

  if (token_id >= params.batch_size) return;

  const auto plan = params.plan_d[token_id];
  const auto kv_input = static_cast<const InputFloat*>(params.kv_input);
  const auto kv_buffer = static_cast<BufferFloat*>(params.kv_buffer);
  const auto score_bias = static_cast<const InputFloat*>(params.score_bias);

  const auto kv_src = kv_input + token_id * Trait::kElementSize;
  const auto kv_buf_0 = kv_buffer + plan.read_page_0 * Trait::kPageElementSize;
  const auto kv_buf_1 = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;
  const auto kv_dst = kv_buffer + plan.write_loc * Trait::kElementSize;

  PDLWaitPrimary<kUsePDL>();
  c4_write_decode<Trait, BufferFloat, InputFloat>(kv_dst, kv_src);

  // Warp-uniform, so the cross-lane butterflies below still see a full warp.
  if (plan.seq_len % params.compress_ratio != 0) return;

  Float4 data = c4_compress_core<Trait, BufferFloat, InputFloat>(
      kv_buf_0, kv_buf_1, kv_src, score_bias, plan.seq_len > 4, 8);

  const auto position = static_cast<int32_t>(plan.seq_len - params.compress_ratio);
  const auto out_loc = params.out_loc[token_id];
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  Float4 freq;

  // part 1: norm
  {
    Storage weight_vec;
    weight_vec.load(params.norm_weight, lane_id);
    if (is_rope_lane) freq.load(freqs_cis, lane_id - (kWarpThreads - kRopeSize));

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      sum_of_squares += data[i] * data[i];
    }

    sum_of_squares = warp::reduce_sum(sum_of_squares);
    const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_weight = cast<float>(weight_vec[i]);
      data[i] = data[i] * norm_factor * fp32_weight;
    }
  }

  // part 2: rope (rope-lane only, 4 elems per lane = 2 (real, imag) pairs)
  if (is_rope_lane) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto y_real = data[2];
    const auto y_imag = data[3];
    const auto freq_x_real = freq[0];
    const auto freq_x_imag = freq[1];
    const auto freq_y_real = freq[2];
    const auto freq_y_imag = freq[3];
    data[0] = x_real * freq_x_real - x_imag * freq_x_imag;
    data[1] = x_real * freq_x_imag + x_imag * freq_x_real;
    data[2] = y_real * freq_y_real - y_imag * freq_y_imag;
    data[3] = y_real * freq_y_imag + y_imag * freq_y_real;
  }

  // part 3: hadamard transform
  {
    {
      const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a1;
      data[1] = a0 - a1;
      data[2] = a2 + a3;
      data[3] = a2 - a3;
    }
    {
      const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a2;
      data[1] = a1 + a3;
      data[2] = a0 - a2;
      data[3] = a1 - a3;
    }
#pragma unroll
    for (uint32_t mask = 1; mask < kWarpThreads; mask <<= 1) {
#pragma unroll
      for (int i = 0; i < kVecSize; ++i) {
#ifndef USE_ROCM
        const float other = __shfl_xor_sync(kFullMask, data[i], mask, kWarpThreads);
#else
        const float other = __shfl_xor(data[i], mask, kWarpThreads);
#endif
        data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
      }
    }
    const float kHadamardScale = math::rsqrt(static_cast<float>(kHeadDim));
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      data[i] *= kHadamardScale;
  }

  // part 4: per-warp UE8M0 quant + store (128 elements -> one fp8 group).
  {
    using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;
    float local_max = math::abs(data[0]);
#pragma unroll
    for (int i = 1; i < kVecSize; ++i) {
      local_max = math::max(local_max, math::abs(data[i]));
    }
    const auto abs_max = warp::reduce_max(local_max);
    const auto scale = fmaxf(1e-4f, abs_max) / kFP8E4M3Max;
    const auto inv_scale = 1.0f / scale;
    const int64_t page = out_loc >> kPageBits;
    const int64_t offset = out_loc & ((1 << kPageBits) - 1);
    const auto page_ptr = params.kvcache + page * kPageBytes;
    const auto value_ptr = page_ptr + offset * 128;
    const auto scale_ptr = page_ptr + (128 << kPageBits) + offset * 4;
    OutStorage result;
    result[0] = pack_fp8(data[0] * inv_scale, data[1] * inv_scale);
    result[1] = pack_fp8(data[2] * inv_scale, data[3] * inv_scale);
    PDLTriggerSecondary<kUsePDL>();
    if constexpr (kPreshuffleSize != 0) {
      constexpr int32_t kTile = kPreshuffleSize;
      const int32_t dim_base = lane_id * kVecSize;
      const int32_t token_tile_id = offset / kTile;
      const int32_t token_in_tile = offset % kTile;
      const int32_t col_tile_id = dim_base / kTile;
      const int32_t col_in_tile = dim_base % kTile;
      const int32_t value_offset = token_tile_id * (kTile * static_cast<int32_t>(kHeadDim)) +
                                   col_tile_id * (kTile * kTile) + token_in_tile * kTile + col_in_tile;
      result.store(page_ptr + value_offset, 0);
    } else {
      result.store(value_ptr, lane_id);
    }
    if (lane_id == 0) reinterpret_cast<float*>(scale_ptr)[0] = scale;
  }
}

#ifdef USE_ROCM
/// \brief Same fusion as `flash_c4_decode_norm_rope_indexer`, but one token per
/// *wavefront* instead of one token per 32-lane warp.
///
/// The warp-mapped version above loses to the two kernels it replaces, for two
/// reasons that are both about the 32-vs-64 mismatch:
///
///   - Two tokens share a wavefront, so `seq_len % ratio` is not wave-uniform.
///     Only one token in four compresses, so 1 - (3/4)^2 = 44% of wavefronts
///     run the whole compress + norm + rope + quant tail for (usually) a single
///     live token. The standalone norm/rope kernel launches over the compressed
///     rows only and never pays this.
///   - 4 elements per lane keeps `score_fp32[4][8]` live across the softmax,
///     which put the kernel at 85 VGPRs / 5 waves per SIMD against 76 / 6 and
///     26 / 8 for the pair it replaced.
///
/// Handing the token a whole wavefront at 2 elements per lane fixes both at
/// once: the early-out becomes wave-uniform, and the live set halves. It also
/// makes every cross-lane step here span exactly one wavefront, so the
/// reductions and the Hadamard butterfly are plain shuffles. That matters more
/// than it looks -- a block holds 4 independent tokens that exit at different
/// times, so a __syncthreads() would be unusable here.
template <
    typename BufferFloat,
    typename InputFloat,
    typename DType,
    int32_t kPageBits,
    bool kUsePDL,
    int32_t kPreshuffleSize>
FUSED_C4_KERNEL void
flash_c4_decode_norm_rope_indexer_w64(const __grid_constant__ FusedCompress4NormRopeParams params) {
  using namespace device;

  constexpr int64_t kHeadDim = 128;
  using Trait = FusedC4Trait<kHeadDim, 2>;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = Trait::kTileElements;
  constexpr uint32_t kWarpsPerToken = Trait::kNumSplit;
  constexpr uint32_t kThreadsPerToken = kWarpsPerToken * kWarpThreads;
  constexpr uint32_t kTokensPerBlock = kFusedBlockSize / kThreadsPerToken;
  constexpr int64_t kPageBytes = 132ll << kPageBits;
  static_assert(kThreadsPerToken == kWaveThreads, "one token must own one wavefront");
  static_assert(kHeadDim == kThreadsPerToken * kVecSize);
  static_assert(kRopeDim == kWarpThreads * kVecSize, "rope tail is the token's last warp");
  using Float2 = AlignedVector<float, kVecSize>;
  using Storage = AlignedVector<DType, kVecSize>;

  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;
  const auto local_token = warp_id / kWarpsPerToken;
  const auto split_id = warp_id % kWarpsPerToken;
  // Element slot within the token's row, and equally the lane's index in the
  // wavefront, which is what the butterflies below index by.
  const auto wave_lane = split_id * kWarpThreads + lane_id;
  const auto token_id = blockIdx.x * kTokensPerBlock + local_token;

  if (token_id >= params.batch_size) return;

  const int64_t split_offset = static_cast<int64_t>(split_id) * Trait::kTileDim;

  const auto plan = params.plan_d[token_id];
  const auto kv_input = static_cast<const InputFloat*>(params.kv_input) + split_offset;
  const auto kv_buffer = static_cast<BufferFloat*>(params.kv_buffer) + split_offset;
  const auto score_bias = static_cast<const InputFloat*>(params.score_bias) + split_offset;

  const auto kv_src = kv_input + token_id * Trait::kElementSize;
  const auto kv_buf_0 = kv_buffer + plan.read_page_0 * Trait::kPageElementSize;
  const auto kv_buf_1 = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;
  const auto kv_dst = kv_buffer + plan.write_loc * Trait::kElementSize;

  PDLWaitPrimary<kUsePDL>();
  c4_write_decode<Trait, BufferFloat, InputFloat>(kv_dst, kv_src);

  // Wave-uniform: all 64 lanes of this token take the same branch.
  if (plan.seq_len % params.compress_ratio != 0) return;

  Float2 data = c4_compress_core<Trait, BufferFloat, InputFloat>(
      kv_buf_0, kv_buf_1, kv_src, score_bias, plan.seq_len > 4, 8);

  const auto position = static_cast<int32_t>(plan.seq_len - params.compress_ratio);
  const auto out_loc = params.out_loc[token_id];
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;
  const bool is_rope_warp = split_id == kWarpsPerToken - 1;

  Float2 freq;

  // part 1: norm
  {
    Storage weight_vec;
    weight_vec.load(params.norm_weight, wave_lane);
    if (is_rope_warp) freq.load(freqs_cis, lane_id);

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      sum_of_squares += data[i] * data[i];
    }

    sum_of_squares = wave_reduce_sum(sum_of_squares);
    const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto fp32_weight = cast<float>(weight_vec[i]);
      data[i] = data[i] * norm_factor * fp32_weight;
    }
  }

  // part 2: rope on the token's last warp, 1 (real, imag) pair per lane
  if (is_rope_warp) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto freq_real = freq[0];
    const auto freq_imag = freq[1];
    data[0] = x_real * freq_real - x_imag * freq_imag;
    data[1] = x_real * freq_imag + x_imag * freq_real;
  }

  // part 3: hadamard. Same 128-point transform as the warp-mapped kernel: the
  // element index is still `lane * kVecSize + i`, so moving a bit out of the
  // register loop and into the lane loop leaves the result unchanged.
  {
    {
      const float a0 = data[0], a1 = data[1];
      data[0] = a0 + a1;
      data[1] = a0 - a1;
    }
#pragma unroll
    for (uint32_t mask = 1; mask < kWaveThreads; mask <<= 1) {
#pragma unroll
      for (int i = 0; i < kVecSize; ++i) {
        const float other = __shfl_xor(data[i], mask, kWaveThreads);
        data[i] = (wave_lane & mask) ? (other - data[i]) : (data[i] + other);
      }
    }
    const float kHadamardScale = math::rsqrt(static_cast<float>(kHeadDim));
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      data[i] *= kHadamardScale;
  }

  // part 4: one fp8 group per 128-element row, so the scale reduces over the
  // whole wavefront rather than over a warp.
  {
    float local_max = math::abs(data[0]);
#pragma unroll
    for (int i = 1; i < kVecSize; ++i) {
      local_max = math::max(local_max, math::abs(data[i]));
    }
    const auto abs_max = wave_reduce_max(local_max);
    const auto scale = fmaxf(1e-4f, abs_max) / kFP8E4M3Max;
    const auto inv_scale = 1.0f / scale;
    const int64_t page = out_loc >> kPageBits;
    const int64_t offset = out_loc & ((1 << kPageBits) - 1);
    const auto page_ptr = params.kvcache + page * kPageBytes;
    const auto value_ptr = page_ptr + offset * 128;
    const auto scale_ptr = page_ptr + (128 << kPageBits) + offset * 4;
    const auto result = pack_fp8(data[0] * inv_scale, data[1] * inv_scale);
    PDLTriggerSecondary<kUsePDL>();
    if constexpr (kPreshuffleSize != 0) {
      constexpr int32_t kTile = kPreshuffleSize;
      const int32_t dim_base = wave_lane * kVecSize;
      const int32_t token_tile_id = offset / kTile;
      const int32_t token_in_tile = offset % kTile;
      const int32_t col_tile_id = dim_base / kTile;
      const int32_t col_in_tile = dim_base % kTile;
      const int32_t value_offset = token_tile_id * (kTile * static_cast<int32_t>(kHeadDim)) +
                                   col_tile_id * (kTile * kTile) + token_in_tile * kTile + col_in_tile;
      *reinterpret_cast<fp8x2_e4m3_t*>(page_ptr + value_offset) = result;
    } else {
      reinterpret_cast<fp8x2_e4m3_t*>(value_ptr)[wave_lane] = result;
    }
    if (wave_lane == 0) reinterpret_cast<float*>(scale_ptr)[0] = scale;
  }
}
#endif  // USE_ROCM

template <
    typename BufferFloat,
    typename InputFloat,
    typename DType,
    uint32_t kPageSize,
    bool kUsePDL,
    int32_t kPreshuffleSize>
struct FusedCompress4NormRopeIndexerKernel {
  static constexpr int64_t kHeadDim = 128;
  static constexpr int32_t kLogPageSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes = 132 * kPageSize;
#ifdef USE_ROCM
  static constexpr auto kernel = flash_c4_decode_norm_rope_indexer_w64<
      BufferFloat,
      InputFloat,
      DType,
      kLogPageSize,
      kUsePDL,
      kPreshuffleSize>;
  static constexpr uint32_t kTokensPerBlock = kFusedBlockSize / kWaveThreads;
  using Trait = FusedC4Trait<kHeadDim, 2>;
#else
  static constexpr auto kernel = flash_c4_decode_norm_rope_indexer<
      BufferFloat,
      InputFloat,
      DType,
      kLogPageSize,
      kUsePDL,
      kPreshuffleSize>;
  static constexpr uint32_t kTokensPerBlock = kFusedNumWarps;
  using Trait = FusedC4Trait<kHeadDim, 4>;
#endif

  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");

  static void run_decode(
      const tvm::ffi::TensorView kv_buffer,
      const tvm::ffi::TensorView kv_input,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView plan_d_,
      const tvm::ffi::TensorView norm_weight,
      const double eps,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView kvcache,
      const int64_t compress_ratio) {
    using namespace host;

    auto N = SymbolicSize{"batch_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    TensorMatcher({-1, 4, Trait::kElementSize})  // kv score
        .with_dtype<BufferFloat>()
        .with_device(device_)
        .verify(kv_buffer);
    TensorMatcher({N, Trait::kElementSize})  // kv score input
        .with_dtype<InputFloat>()
        .with_device(device_)
        .verify(kv_input);
    TensorMatcher({8, kHeadDim})  // ape
        .with_dtype<InputFloat>()
        .with_device(device_)
        .verify(ape);
    TensorMatcher({kHeadDim})  // norm weight
        .with_dtype<DType>()
        .with_device(device_)
        .verify(norm_weight);
    TensorMatcher({-1, 64})  // freqs_cis
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs_cis);
    TensorMatcher({-1})  // out_loc
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(out_loc);
    TensorMatcher({-1, -1})  // cache
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(kvcache);

    const auto plan_d = compress::verify_plan_d(plan_d_, N, device_);
    const auto batch_size = static_cast<uint32_t>(N.unwrap());
    if (batch_size == 0) return;
    RuntimeCheck(out_loc.size(0) == N.unwrap());

    const auto params = FusedCompress4NormRopeParams{
        .kv_buffer = kv_buffer.data_ptr(),
        .kv_input = kv_input.data_ptr(),
        .score_bias = ape.data_ptr(),
        .norm_weight = norm_weight.data_ptr(),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .out_loc = static_cast<const int64_t*>(out_loc.data_ptr()),
        .kvcache = static_cast<uint8_t*>(kvcache.data_ptr()),
        .plan_d = plan_d,
        .eps = static_cast<float>(eps),
        .compress_ratio = static_cast<uint32_t>(compress_ratio),
        .batch_size = batch_size,
    };
    // A 256-thread block either way: 4 tokens of one wavefront each on HIP, or
    // 8 tokens of one warp each elsewhere. Splitting into smaller blocks to
    // spread across more CUs backfires: the tighter launch bound costs
    // registers and the kernel spills (117 VGPRs clean at 256/occ4 vs 64 + 372B
    // scratch at 128/occ8), which measured 2x slower in situ.
    const uint32_t num_blocks = div_ceil(batch_size, kTokensPerBlock);
    LaunchKernel(num_blocks, kFusedBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
