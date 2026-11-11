#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp4_utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>
#include <optional>

namespace sglang {

/// \brief RMSNorm, RoPE, two FP4 stages and a 68-byte index-K cache store.
///
/// A ratio-r group's latent uses its first position, `positions & ~(r - 1)`,
/// for power-of-two ratios.
struct IndexKParams {
  const bf16_t* __restrict__ input;        // [num_tokens, kHeadDim] bf16, pre-norm
  const bf16_t* __restrict__ norm_weight;  // [kHeadDim] bf16
  const float* __restrict__ freqs_cis;     // [max_pos, kRopeDim] fp32, real/imag interleaved
  const void* __restrict__ positions;      // [num_tokens] PosT
  const int64_t* __restrict__ loc;         // [num_tokens] index-K slot; 0 publishes nothing
  uint8_t* __restrict__ cache;             // [npages, kPageSize * 68] uint8
  uint32_t num_tokens;
  float eps;
};

/// \brief RoPE and two-stage FP4 packing for index-Q, after `wq_b`.
///
/// Each (token, head) row uses its token's own position, without RMSNorm
/// or a ratio mask.
struct IndexQParams {
  const bf16_t* __restrict__ input;     // [num_tokens, heads, kHeadDim] bf16
  const float* __restrict__ freqs_cis;  // [max_pos, kRopeDim] fp32, real/imag interleaved
  const void* __restrict__ positions;   // [num_tokens] PosT
  int8_t* __restrict__ payload;         // [num_tokens * heads, kHeadDim / 2] int8
  int32_t* __restrict__ scale;          // [num_tokens * heads] int32, four ue8m0 bytes
  // Optional head-weight epilogue (kWeights): the raw `weights_proj` output for
  // the same (token, head) rows.
  const bf16_t* __restrict__ head_weights;  // [num_tokens * heads] bf16, or nullptr
  float* __restrict__ weights_out;          // [num_tokens * heads] fp32, or nullptr
  float weight_scale;
  uint32_t num_rows;
  uint32_t heads;
};

/// One warp owns one row. Chosen from B200 decode measurements, where occupancy
/// has little effect; multiple warps avoid starving larger batches.
constexpr uint32_t kFp4RopeWarpsPerCTA = 4;

/// \brief Indexer packer scale: `_ceil_ue8m0_exp(max(amax / 6, 1e-4))`.
///
/// Unlike fake quantization, the floor follows the divide, division is not
/// replaced by multiplication by 1/6, and the exponent is clamped.
/// The two stages therefore require separate scales.
SGL_DEVICE uint32_t index_pack_exponent(float amax) {
  const auto exponent = deepseek_v4::fp8::cast_to_ue8m0(fmaxf(amax / 6.0f, 1.0e-4f));
  // Neither bound is reachable for finite fp32 inputs: the 1e-4 floor keeps
  // the exponent above 1, and reaching 254 requires absmax > 6 * 2^126.
  return static_cast<uint32_t>(min(max(exponent, 1), 254));
}

/// \brief Clear the sign of every packed nibble whose magnitude rounded to zero.
///
/// `cvt.rn.satfinite.e2m1x2.f32` keeps the sign of a small negative, giving the
/// `-0` code `0x8`; the reference packer drops it (`sign = (x < 0) & (idx != 0)`).
/// Both dequantize to zero, but the stored byte differs, so match the reference.
/// Branchless and independent of how many nibbles the word holds: bit 4k+3 of
/// each nibble survives only if one of bits 4k..4k+2 is set.
SGL_DEVICE uint32_t clear_negative_zero(uint32_t packed) {
  const auto any_magnitude = (packed | (packed >> 1) | (packed >> 2)) & 0x11111111u;
  return packed & ((any_magnitude << 3) | 0x77777777u);
}

/// One lane's share of a packed 128-element row: two payload bytes and the two
/// block exponents its half of the warp owns.
struct IndexPacked {
  uint32_t payload[2];   // the head pair's byte, then the tail pair's
  uint32_t exponent[2];  // blocks {0, 1} then {2, 3}, by half of the warp
};

/// \brief The whole shared body of both directions: RoPE tail, both fp4 stages,
///        and the indexer's pack.
///
/// `head` and `tail` are the lane's two bf16 pairs, widened and already rounded
/// to bf16 by whatever produced them (the caller's RMSNorm, or the load itself).
/// `tail` is pre-rotation and `freq` is its matching `(real, imag)`.
///
/// A 32-element fp4 block is 16 lanes of *one* half -- blocks 0/1 are the head
/// on lanes 0-15 / 16-31 and blocks 2/3 the tail -- so each quantization stage
/// costs two `reduce_max<16>`, not four, and nothing here spans the row.
SGL_DEVICE IndexPacked index_rope_quant_pack(fp32x2_t head, fp32x2_t tail, fp32x2_t freq) {
  using namespace device;
  namespace fp4 = deepseek_v4::fp4;

  constexpr uint32_t kHalfLanes = kWarpThreads / 2;
  static_assert(fp4::kBlockSize == kHalfLanes * 2, "an fp4 block must be half a warp of one half");

  float data[4];
  data[0] = head.x;
  data[1] = head.y;
  // The contraction is spelled out so `index_q_block_kernel` matches it bit for bit;
  // `rope_tail` ends in `.to(x.dtype)`, so the rotated pair is rounded again.
  const auto rotated = cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{
      __fmaf_rn(tail.x, freq.x, -__fmul_rn(tail.y, freq.y)), __fmaf_rn(tail.x, freq.y, __fmul_rn(tail.y, freq.x))}));
  data[2] = rotated.x;
  data[3] = rotated.y;

  // The fake-quant result is already exact in bf16: e2m1 needs at most three
  // significant bits, and the power-of-two scales admitted by the amax floor fit bf16.
#pragma unroll
  for (uint32_t half = 0; half < 2; ++half) {
    const auto amax = warp::reduce_max<kHalfLanes>(fmaxf(fabsf(data[half * 2]), fabsf(data[half * 2 + 1])));
    const auto [scale, inv_scale] = fp4::block_scale(amax);
    const auto q = fp4::fake_quant_x2({data[half * 2], data[half * 2 + 1]}, scale, inv_scale);
    data[half * 2 + 0] = q.x;
    data[half * 2 + 1] = q.y;
  }

  // The packer's scale floor differs from fake quantization; keep both stages.
  // Each packed byte puts `.x` in the low nibble.
  IndexPacked out;
#pragma unroll
  for (uint32_t half = 0; half < 2; ++half) {
    const auto amax = warp::reduce_max<kHalfLanes>(fmaxf(fabsf(data[half * 2]), fabsf(data[half * 2 + 1])));
    out.exponent[half] = index_pack_exponent(amax);
    // `inv_scale_ue8m0` instead of the reference's division: the scale is a
    // power of two, so both are exact except at the unreachable exponent 254.
    const auto inv_scale = deepseek_v4::fp8::inv_scale_ue8m0(static_cast<int32_t>(out.exponent[half]));
#ifndef USE_ROCM
    const auto code = __nv_cvt_float2_to_fp4x2(
        fp32x2_t{data[half * 2] * inv_scale, data[half * 2 + 1] * inv_scale}, __NV_E2M1, cudaRoundNearest);
#else
    const auto code = fp4::e2m1x2_code(fp32x2_t{data[half * 2] * inv_scale, data[half * 2 + 1] * inv_scale});
#endif
    out.payload[half] = clear_negative_zero(static_cast<uint32_t>(code));
  }
  return out;
}

/// \brief The row's four block exponents, packed little-endian into one word.
///
/// They live on two lanes -- 0 holds blocks 0 and 2, 16 holds 1 and 3 -- so this
/// costs two shuffles, and the result is the word only for the lower half of
/// the warp. Every lane must reach it: the shuffles are warp-wide.
SGL_DEVICE uint32_t index_scale_word(const uint32_t (&exponent)[2]) {
  using namespace device;
#ifndef USE_ROCM
  const auto exp_1 = __shfl_sync(warp::kFullMask, exponent[0], kWarpThreads / 2);
  const auto exp_3 = __shfl_sync(warp::kFullMask, exponent[1], kWarpThreads / 2);
#else
  // A wave holds two of these 32-lane rows, so the shuffle width has to be the
  // row, not the wave, for lane 16 to be this row's lane 16.
  const auto exp_1 = __shfl(exponent[0], kWarpThreads / 2, kWarpThreads);
  const auto exp_3 = __shfl(exponent[1], kWarpThreads / 2, kWarpThreads);
#endif
  return exponent[0] | (exp_1 << 8) | (exponent[1] << 16) | (exp_3 << 24);
}

/// \brief One warp per token; grid = ceil(num_tokens / kFp4RopeWarpsPerCTA).
///
/// Lane L owns head elements {2L, 2L+1} and tail elements {64+2L, 64+2L+1},
/// so each lane carries one complex RoPE pair. Only RMSNorm spans the full
/// row; the remaining reductions use the FP4 block layout.
template <bool kUsePDL, int64_t kHeadDim, int64_t kRopeDim, uint32_t kPageSize, uint32_t kRatio, typename PosT>
__global__ __launch_bounds__(kFp4RopeWarpsPerCTA* device::kWarpThreads) void index_k_kernel(const IndexKParams params) {
  using namespace device;
  namespace fp4 = deepseek_v4::fp4;

  constexpr uint32_t kPayloadBytes = kHeadDim / 2;
  constexpr uint32_t kScaleBytes = kHeadDim / fp4::kBlockSize;
  constexpr uint32_t kSlotBytes = kPayloadBytes + kScaleBytes;

  static_assert(
      kHeadDim == 128 && kRopeDim == 64,
      "the one-warp tiling is specific to a 128-wide row whose second half is the RoPE tail");
  static_assert(kHeadDim == 2 * kWarpThreads * 2, "a lane owns one bf16x2 of each half");
  static_assert(kScaleBytes == 4, "the four block exponents are packed into one uint32 store");
  static_assert(std::has_single_bit(kRatio), "group_pos is derived by masking, so the ratio must be a power of two");

  using bf16_vec_t = AlignedVector<bf16x2_t, 1>;
  using fp32_vec_t = AlignedVector<float, 2>;

  const auto lane = threadIdx.x % kWarpThreads;
  const auto row = blockIdx.x * kFp4RopeWarpsPerCTA + threadIdx.x / kWarpThreads;
  // Warp-uniform, so the reductions below still see a full warp.
  if (row >= params.num_tokens) return;

  // Both come from the step's metadata rather than the predecessor, so
  // prefetching them ahead of the PDL gate overlaps with the `wk` GEMM's tail.
  const auto slot_id = params.loc[row];
  const auto position = static_cast<int64_t>(static_cast<const PosT*>(params.positions)[row]);
  PDLWaitPrimary<kUsePDL>();

  bf16_vec_t head_in, tail_in, head_w, tail_w;
  head_in.load(params.input + row * kHeadDim, lane);
  tail_in.load(params.input + row * kHeadDim, lane + kWarpThreads);
  head_w.load(params.norm_weight, lane);
  tail_w.load(params.norm_weight, lane + kWarpThreads);
  fp32_vec_t freq;
  freq.load(params.freqs_cis + (position & ~static_cast<int64_t>(kRatio - 1)) * kRopeDim, lane);

  fp32x2_t head, tail;
  {
    const auto [h0, h1] = cast<fp32x2_t>(head_in[0]);
    const auto [t0, t1] = cast<fp32x2_t>(tail_in[0]);
    const auto sqrsum = warp::reduce_sum(h0 * h0 + h1 * h1 + t0 * t0 + t1 * t1);
    const auto inv_rms = math::rsqrt(sqrsum * (1.0f / static_cast<float>(kHeadDim)) + params.eps);
    const auto [wh0, wh1] = cast<fp32x2_t>(head_w[0]);
    const auto [wt0, wt1] = cast<fp32x2_t>(tail_w[0]);
    // `k_norm` materializes a bf16 tensor, so the norm result is rounded before
    // anything downstream sees it -- the RoPE below included.
    head = cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{wh0 * (h0 * inv_rms), wh1 * (h1 * inv_rms)}));
    tail = cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{wt0 * (t0 * inv_rms), wt1 * (t1 * inv_rms)}));
  }

  const auto packed = index_rope_quant_pack(head, tail, fp32x2_t{freq[0], freq[1]});
  const auto scale_word = index_scale_word(packed.exponent);

  // A padded graph row, and at ratio > 1 a row completing no group, carry the
  // reserved slot 0 and must publish nothing.
  if (slot_id <= 0) return;
  const auto page = slot_id / kPageSize;
  const auto slot = slot_id % kPageSize;
  const auto page_ptr = params.cache + page * (kPageSize * kSlotBytes);

  // Byte i of the payload covers elements (2i, 2i+1), so a lane's head pair is
  // byte `lane` and its tail pair byte `lane + 32`: two coalesced 32-byte runs.
  const auto payload_ptr = page_ptr + slot * kPayloadBytes;
  payload_ptr[lane] = static_cast<uint8_t>(packed.payload[0]);
  payload_ptr[lane + kWarpThreads] = static_cast<uint8_t>(packed.payload[1]);

  if (lane == 0) {
    *reinterpret_cast<uint32_t*>(page_ptr + kPageSize * kPayloadBytes + slot * kScaleBytes) = scale_word;
  }
}

/// \brief One warp per (token, head); grid = ceil(num_rows / kFp4RopeWarpsPerCTA).
///
/// Input is contiguous [num_tokens, heads, kHeadDim]; row r uses token r / heads
/// and head r % heads.
template <bool kUsePDL, int64_t kHeadDim, int64_t kRopeDim, typename PosT, bool kWeights>
__global__ __launch_bounds__(kFp4RopeWarpsPerCTA* device::kWarpThreads) void index_q_kernel(const IndexQParams params) {
  using namespace device;

  constexpr uint32_t kPayloadBytes = kHeadDim / 2;

  static_assert(
      kHeadDim == 128 && kRopeDim == 64,
      "the one-warp tiling is specific to a 128-wide row whose second half is the RoPE tail");
  static_assert(kHeadDim == 2 * kWarpThreads * 2, "a lane owns one bf16x2 of each half");

  using bf16_vec_t = AlignedVector<bf16x2_t, 1>;
  using fp32_vec_t = AlignedVector<float, 2>;

  const auto lane = threadIdx.x % kWarpThreads;
  const auto row = blockIdx.x * kFp4RopeWarpsPerCTA + threadIdx.x / kWarpThreads;
  // Warp-uniform, so the reductions below still see a full warp.
  if (row >= params.num_rows) return;

  // The position lookup is independent of the PDL producer.
  const auto position = static_cast<int64_t>(static_cast<const PosT*>(params.positions)[row / params.heads]);
  PDLWaitPrimary<kUsePDL>();

  bf16_vec_t head_in, tail_in;
  head_in.load(params.input + row * kHeadDim, lane);
  tail_in.load(params.input + row * kHeadDim, lane + kWarpThreads);
  fp32_vec_t freq;
  freq.load(params.freqs_cis + position * kRopeDim, lane);

  const auto packed =
      index_rope_quant_pack(cast<fp32x2_t>(head_in[0]), cast<fp32x2_t>(tail_in[0]), fp32x2_t{freq[0], freq[1]});
  const auto scale_word = index_scale_word(packed.exponent);

  const auto payload_ptr = params.payload + row * kPayloadBytes;
  payload_ptr[lane] = static_cast<int8_t>(packed.payload[0]);
  payload_ptr[lane + kWarpThreads] = static_cast<int8_t>(packed.payload[1]);

  if (lane == 0) params.scale[row] = static_cast<int32_t>(scale_word);

  if constexpr (kWeights) {
    // Match `head_weights(x).float()`: the bf16 round-trip is part of the reference.
    if (lane == 0) {
      const auto w = cast<float>(params.head_weights[row]) * params.weight_scale;
      params.weights_out[row] = cast<float>(cast<bf16_t>(w));
    }
  }
}

#ifndef USE_ROCM
/// Threads per CTA of `index_q_block_kernel`: 64 (token, head) rows.
constexpr uint32_t kIndexQBlockThreads = 256;

/// \brief Absmax of one 32-element block as a tree; seeded with 0 so a NaN is
///        dropped exactly as the `fmaxf` chain from 0 drops it.
SGL_DEVICE float index_block_absmax(const float (&v)[deepseek_v4::fp4::kBlockSize]) {
  static_assert(deepseek_v4::fp4::kBlockSize == 32);
  float m[8];
#pragma unroll
  for (int i = 0; i < 8; ++i)
    m[i] = fmaxf(fmaxf(fabsf(v[4 * i]), fabsf(v[4 * i + 1])), fmaxf(fabsf(v[4 * i + 2]), fabsf(v[4 * i + 3])));
#pragma unroll
  for (int i = 0; i < 4; ++i)
    m[i] = fmaxf(m[i], m[i + 4]);
  return fmaxf(fmaxf(m[0], m[2]), fmaxf(fmaxf(m[1], m[3]), 0.0f));
}

/// \brief `index_pack_exponent(amax)` without the division, for an `amax` that is
///        an e2m1 grid value times a power of two.
///
/// Such an amax is 1 or 1.5 times 2^E, so `ceil_ue8m0(amax / 6)` is its biased
/// exponent minus 2; the 1e-4 floor (biased 114) covers zero and subnormals, and
/// 255 (inf/nan) still lands on the 254 clamp.
SGL_DEVICE uint32_t index_pack_exponent_grid(float amax) {
  const auto biased = static_cast<int32_t>((__float_as_uint(amax) >> 23) & 0xFF);
  const auto exponent = biased == 255 ? 255 : biased - 2;
  return static_cast<uint32_t>(min(max(exponent, 114), 254));
}

/// \brief `index_q_kernel` for prefill-sized row counts: one thread per 32-element
///        fp4 block instead of one warp per row.
///
/// Every fp4 stage of `index_rope_quant_pack` is per element except the two block
/// absmaxes, and a max is exact in any order, so a thread holding its whole block
/// computes the same bytes with no shuffles. Warps 0-3 of a CTA take the head
/// blocks of its 64 rows and warps 4-7 the RoPE blocks, so the RoPE branch is
/// warp-uniform; each thread issues all of its loads before any math.
template <bool kUsePDL, int64_t kHeadDim, int64_t kRopeDim, typename PosT, bool kWeights>
__global__ __launch_bounds__(kIndexQBlockThreads) void index_q_block_kernel(const IndexQParams params) {
  using namespace device;
  namespace fp4 = deepseek_v4::fp4;

  static_assert(kHeadDim == 128 && kRopeDim == 64, "the block split assumes a 64 + 64 (RoPE tail) row");
  constexpr uint32_t kBlock = fp4::kBlockSize;
  constexpr uint32_t kBlocksPerRow = kHeadDim / kBlock;
  constexpr uint32_t kRopeBlock = (kHeadDim - kRopeDim) / kBlock;
  constexpr uint32_t kHalf = kIndexQBlockThreads / 2;
  static_assert(kBlocksPerRow == 2 * kRopeBlock, "each half of the CTA owns two blocks per row");

  const bool rope = threadIdx.x >= kHalf;
  const auto t = threadIdx.x % kHalf;
  const auto row = blockIdx.x * (kIndexQBlockThreads / kBlocksPerRow) + t / kRopeBlock;
  const auto blk = (rope ? kRopeBlock : 0) + t % kRopeBlock;
  if (row >= params.num_rows) return;

  // The position lookup is independent of the PDL producer.
  const auto position = static_cast<int64_t>(static_cast<const PosT*>(params.positions)[row / params.heads]);
  PDLWaitPrimary<kUsePDL>();

  constexpr uint32_t kVecWords = kMaxVecBytes / sizeof(uint32_t);
  using vec_t = AlignedVector<uint32_t, kVecWords>;
  const auto* in = params.input + static_cast<int64_t>(row) * kHeadDim + blk * kBlock;
  vec_t raw[kBlock / 2 / kVecWords];
#pragma unroll
  for (uint32_t i = 0; i < kBlock / 2 / kVecWords; ++i)
    raw[i].load(in, i);
  vec_t freq[kBlock / kVecWords];
  if (rope) {
    const auto* f = params.freqs_cis + position * kRopeDim + (blk - kRopeBlock) * kBlock;
#pragma unroll
    for (uint32_t i = 0; i < kBlock / kVecWords; ++i)
      freq[i].load(f, i);
  }

  float v[kBlock];
#pragma unroll
  for (uint32_t j = 0; j < kBlock / 2; ++j) {
    const auto word = raw[j / kVecWords][j % kVecWords];
    const auto f = cast<fp32x2_t>(*reinterpret_cast<const bf16x2_t*>(&word));
    v[2 * j] = f.x;
    v[2 * j + 1] = f.y;
  }
  if (rope) {
#pragma unroll
    for (uint32_t j = 0; j < kBlock / 2; ++j) {
      const float x = v[2 * j], y = v[2 * j + 1];
      const float fx = __uint_as_float(freq[2 * j / kVecWords][2 * j % kVecWords]);
      const float fy = __uint_as_float(freq[2 * j / kVecWords][2 * j % kVecWords + 1]);
      // `index_rope_quant_pack`'s rotation, fma(x, fx, -(y * fy)) and fma(x, fy, y * fx),
      // rounded to bf16 again as `rope_tail` does.
      const auto rotated = cast<fp32x2_t>(
          cast<bf16x2_t>(fp32x2_t{__fmaf_rn(x, fx, -__fmul_rn(y, fy)), __fmaf_rn(x, fy, __fmul_rn(y, fx))}));
      v[2 * j] = rotated.x;
      v[2 * j + 1] = rotated.y;
    }
  }

  // `fp4::fake_quant_x2` up to its final `* scale`: v becomes the e2m1 grid value.
  const auto [scale, inv_scale] = fp4::block_scale(index_block_absmax(v));
#pragma unroll
  for (uint32_t i = 0; i < kBlock; i += 2) {
    const auto code = __nv_cvt_float2_to_fp4x2(
        fp32x2_t{__fmaf_rn(v[i], inv_scale, 0.0f), __fmaf_rn(v[i + 1], inv_scale, 0.0f)}, __NV_E2M1, cudaRoundNearest);
    const auto grid = cast<fp32x2_t>(fp16x2_t{__nv_cvt_fp4x2_to_halfraw2(code, __NV_E2M1)});
    v[i] = grid.x;
    v[i + 1] = grid.y;
  }
  // The fake-quant value is grid * scale and both scales are powers of two, so
  // (grid * scale) * pack_inv_scale == grid * (scale * pack_inv_scale) exactly.
  const auto exponent = index_pack_exponent_grid(__fmul_rn(index_block_absmax(v), scale));
  const auto factor = __fmul_rn(scale, deepseek_v4::fp8::inv_scale_ue8m0(static_cast<int32_t>(exponent)));

  using out_vec_t = AlignedVector<uint32_t, 4>;
  out_vec_t out;
#pragma unroll
  for (uint32_t w = 0; w < 4; ++w) {
    uint32_t word = 0;
#pragma unroll
    for (uint32_t b = 0; b < 4; ++b) {
      const auto i = (w * 4 + b) * 2;
      const auto code = __nv_cvt_float2_to_fp4x2(
          fp32x2_t{__fmul_rn(v[i], factor), __fmul_rn(v[i + 1], factor)}, __NV_E2M1, cudaRoundNearest);
      word |= (static_cast<uint32_t>(code) & 0xFFu) << (b * 8);
    }
    out[w] = clear_negative_zero(word);
  }
  out.store(params.payload + static_cast<int64_t>(row) * (kHeadDim / 2) + blk * (kBlock / 2));
  reinterpret_cast<uint8_t*>(params.scale)[static_cast<int64_t>(row) * kBlocksPerRow + blk] =
      static_cast<uint8_t>(exponent);

  if constexpr (kWeights) {
    // Match `head_weights(x).float()`: the bf16 round-trip is part of the reference.
    if (blk == 0) {
      const auto w = cast<float>(params.head_weights[row]) * params.weight_scale;
      params.weights_out[row] = cast<float>(cast<bf16_t>(w));
    }
  }
}
#endif  // USE_ROCM

/// \brief Host side of `index_k_kernel`.
template <int64_t kHeadDim, int64_t kRopeDim, uint32_t kPageSize, uint32_t kRatio, bool kUsePDL>
struct IndexKKernel {
  static constexpr uint32_t kBlockSize = kFp4RopeWarpsPerCTA * device::kWarpThreads;
  static constexpr int64_t kSlotBytes = kHeadDim / 2 + kHeadDim / deepseek_v4::fp4::kBlockSize;

  template <typename PosT>
  static constexpr auto kernel = index_k_kernel<kUsePDL, kHeadDim, kRopeDim, kPageSize, kRatio, PosT>;

  /// \param input `[num_tokens, kHeadDim]` bf16, `wk(latent)` before `k_norm`.
  /// \param norm_weight `[kHeadDim]` bf16, `k_norm.weight`.
  /// \param freqs_cis `[max_pos, kRopeDim]` fp32, real/imag interleaved.
  /// \param positions `[num_tokens]` int32 or int64, the *token* position; the
  ///        group position is derived from it and the ratio.
  /// \param loc `[num_tokens]` int64, the index-K slot; `0` publishes nothing.
  /// \param cache `[npages, kPageSize * 68]` uint8.
  static void run_index_k(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView norm_weight,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView loc,
      const tvm::ffi::TensorView cache,
      const float eps) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    TensorMatcher({N, kHeadDim}).with_dtype<bf16_t>().with_device(device_).verify(input);
    TensorMatcher({kHeadDim}).with_dtype<bf16_t>().with_device(device_).verify(norm_weight);
    // Real/imag interleaved, so the trailing dim is kRopeDim, not kRopeDim / 2.
    TensorMatcher({-1, kRopeDim}).with_dtype<fp32_t>().with_device(device_).verify(freqs_cis);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(pos_dtype).with_device(device_).verify(positions);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device_).verify(loc);
    TensorMatcher({-1, kPageSize * kSlotBytes}).with_dtype<uint8_t>().with_device(device_).verify(cache);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_tokens == 0) return;

    const auto params = IndexKParams{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .norm_weight = static_cast<const bf16_t*>(norm_weight.data_ptr()),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .positions = positions.data_ptr(),
        .loc = static_cast<const int64_t*>(loc.data_ptr()),
        .cache = static_cast<uint8_t*>(cache.data_ptr()),
        .num_tokens = num_tokens,
        .eps = eps,
    };
    const auto k_int32 = kernel<int32_t>;
    const auto k_int64 = kernel<int64_t>;
    const auto k = pos_dtype.is_type<int32_t>() ? k_int32 : k_int64;
    LaunchKernel(div_ceil(num_tokens, kFp4RopeWarpsPerCTA), kBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(k, params);
  }
};

/// \brief Host side of `index_q_kernel`.
template <int64_t kHeadDim, int64_t kRopeDim, bool kUsePDL>
struct IndexQKernel {
  static constexpr uint32_t kBlockSize = kFp4RopeWarpsPerCTA * device::kWarpThreads;

  template <typename PosT, bool kWeights>
  static constexpr auto kernel = index_q_kernel<kUsePDL, kHeadDim, kRopeDim, PosT, kWeights>;
#ifndef USE_ROCM
  template <typename PosT, bool kWeights>
  static constexpr auto block_kernel = index_q_block_kernel<kUsePDL, kHeadDim, kRopeDim, PosT, kWeights>;
  /// (token, head) rows from which SM100-family GPUs take `index_q_block_kernel`:
  /// prefill chunks, and decode / verify batches that reach it (e.g. 64 requests x
  /// 4-token verify x 32 heads). Smaller batches keep the warp-per-row kernel.
  static constexpr uint32_t kBlockKernelMinRows = 8192;
#endif

  /// \param input `[num_tokens, heads, kHeadDim]` bf16, `wq_b(q_lora)`.
  /// \param freqs_cis `[max_pos, kRopeDim]` fp32, real/imag interleaved.
  /// \param positions `[num_tokens]` int32 or int64, the query's own position.
  /// \param payload `[num_tokens * heads, kHeadDim / 2]` int8.
  /// \param scale `[num_tokens * heads]` int32, the four ue8m0 block exponents
  ///        packed little-endian.
  /// \param head_weights `[num_tokens, heads]` bf16, the raw `weights_proj(x)`.
  /// \param weights_out `[num_tokens, heads]` fp32, receives
  ///        `float(bf16(head_weights * weight_scale))`, i.e. `head_weights(x).float()`.
  /// \param weight_scale `softmax_scale * heads^-0.5`, applied in fp32.
  static void run_index_q_weights(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView payload,
      const tvm::ffi::TensorView scale,
      const tvm::ffi::TensorView head_weights,
      const tvm::ffi::TensorView weights_out,
      const double weight_scale) {
    launch(input, freqs_cis, positions, payload, scale, head_weights, weights_out, static_cast<float>(weight_scale));
  }

 private:
  using MaybeTensor = std::optional<tvm::ffi::TensorView>;

  static void launch(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView payload,
      const tvm::ffi::TensorView scale,
      const MaybeTensor head_weights,
      const MaybeTensor weights_out,
      const float weight_scale) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"heads"};
    auto R = SymbolicSize{"num_rows"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    TensorMatcher({N, H, kHeadDim}).with_dtype<bf16_t>().with_device(device_).verify(input);
    // Real/imag interleaved, so the trailing dim is kRopeDim, not kRopeDim / 2.
    TensorMatcher({-1, kRopeDim}).with_dtype<fp32_t>().with_device(device_).verify(freqs_cis);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(pos_dtype).with_device(device_).verify(positions);
    TensorMatcher({R, kHeadDim / 2}).with_dtype<int8_t>().with_device(device_).verify(payload);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(scale);
    const auto weights = head_weights.has_value();
    RuntimeCheck(weights == weights_out.has_value(), "head_weights and weights_out come together");
    if (weights) {
      TensorMatcher({N, H}).with_dtype<bf16_t>().with_device(device_).verify(*head_weights);
      TensorMatcher({N, H}).with_dtype<fp32_t>().with_device(device_).verify(*weights_out);
    }
    RuntimeCheck(
        R.unwrap() == N.unwrap() * H.unwrap(),
        "payload holds ",
        R.unwrap(),
        " rows, but the input is ",
        N.unwrap(),
        " tokens x ",
        H.unwrap(),
        " heads");

    const auto num_rows = static_cast<uint32_t>(R.unwrap());
    if (num_rows == 0) return;

    const auto params = IndexQParams{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .positions = positions.data_ptr(),
        .payload = static_cast<int8_t*>(payload.data_ptr()),
        .scale = static_cast<int32_t*>(scale.data_ptr()),
        .head_weights = weights ? static_cast<const bf16_t*>(head_weights->data_ptr()) : nullptr,
        .weights_out = weights ? static_cast<float*>(weights_out->data_ptr()) : nullptr,
        .weight_scale = weight_scale,
        .num_rows = num_rows,
        .heads = static_cast<uint32_t>(H.unwrap()),
    };
    const auto i32 = pos_dtype.is_type<int32_t>();
#ifndef USE_ROCM
    if (num_rows >= kBlockKernelMinRows && runtime::get_cc_major(device_.unwrap().device_id) == 10) {
      // The block kernel moves input and freqs_cis rows in 32-byte vectors and payload
      // rows in 16-byte vectors; a misaligned view fails here instead of faulting.
      TensorMatcher({N, H, kHeadDim}).with_dtype<bf16_t>().ensure_alignment(32).verify(input);
      TensorMatcher({-1, kRopeDim}).with_dtype<fp32_t>().ensure_alignment(32).verify(freqs_cis);
      TensorMatcher({R, kHeadDim / 2}).with_dtype<int8_t>().ensure_alignment(16).verify(payload);
      const auto bk = weights ? (i32 ? block_kernel<int32_t, true> : block_kernel<int64_t, true>)
                              : (i32 ? block_kernel<int32_t, false> : block_kernel<int64_t, false>);
      constexpr uint32_t kRowsPerCTA = kIndexQBlockThreads / (kHeadDim / deepseek_v4::fp4::kBlockSize);
      LaunchKernel(div_ceil(num_rows, kRowsPerCTA), kIndexQBlockThreads, device_.unwrap())  //
          .enable_pdl(kUsePDL)(bk, params);
      return;
    }
#endif
    const auto k = weights ? (i32 ? kernel<int32_t, true> : kernel<int64_t, true>)
                           : (i32 ? kernel<int32_t, false> : kernel<int64_t, false>);
    LaunchKernel(div_ceil(num_rows, kFp4RopeWarpsPerCTA), kBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(k, params);
  }
};

}  // namespace sglang
