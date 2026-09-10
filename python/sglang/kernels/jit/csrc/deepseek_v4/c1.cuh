#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp4_utils.cuh>
#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>

namespace sglang {

/// \brief Ratio-1 decode compressor: RMSNorm and the whole main-KV write.
///
/// The ratio-1 compressor pools nothing -- `project` is a single bf16 `wkv`
/// GEMM and the latent it produces stands for the token itself -- so the
/// kernel's input is the GEMM output and its RoPE position is `positions`, not
/// `positions - 1`. `kv_output` is the pre-RoPE latent, for the index-K
/// branch's `wk` projection.
struct C1Params {
  const bf16_t* __restrict__ kv_input;     // [num_tokens, kHeadDim] bf16
  bf16_t* __restrict__ kv_output;          // [num_tokens, kHeadDim] bf16, pre-RoPE
  const bf16_t* __restrict__ norm_weight;  // [kHeadDim] bf16
  const float* __restrict__ freqs_cis;     // [max_pos, kRopeDim] fp32, real/imag interleaved
  const void* __restrict__ positions;      // [num_tokens] PosT
  const void* __restrict__ out_loc;        // [num_tokens] LocT compressed slot; 0 marks a padded row
  uint8_t* __restrict__ kvcache;           // [npages, kPageBytes] uint8
  float eps;
};

/// Elements per thread; 256 threads per token was measured fastest on B200 decode batches.
/// At head_dim 512, (512 - 64) / 2 = 224 threads keeps the nope/rope split warp-aligned;
/// the fp8 amax reduction requires every lane in its full-warp mask to participate.
constexpr uint32_t kC1VecSize = 2;

/// \brief RMSNorm + RoPE tail + fp4 fake-quant + the 584-byte FlashMLA store.
///
/// One CTA per token, `kHeadDim / kC1VecSize` threads over the row.
///
/// The three reductions have three different widths and are not
/// interchangeable: the RMSNorm statistic spans the row, an fp8 store scale
/// spans 64 elements, an fp4 block spans 16. All asserted below.
template <bool kUsePDL, int64_t kHeadDim, int64_t kRopeDim, int32_t kPageBits, typename PosT, typename LocT>
__global__
__launch_bounds__(kHeadDim / kC1VecSize) void flash_c1_decode_kernel(const __grid_constant__ C1Params params) {
  using namespace device;
  using deepseek_v4::fp8::cast_to_ue8m0;
  using deepseek_v4::fp8::inv_scale_ue8m0;
  using deepseek_v4::fp8::pack_fp8;

  /// Threads over one token, and the leading ones of those that carry the fp8
  /// nope part; the rest carry the bf16 RoPE tail.
  constexpr uint32_t kVecSize = kC1VecSize;
  constexpr uint32_t kRowLanes = kHeadDim / kVecSize;
  constexpr uint32_t kNopeLanes = (kHeadDim - kRopeDim) / kVecSize;
  constexpr uint32_t kRowWarps = kRowLanes / kWarpThreads;
  constexpr uint32_t kFp8Lanes = 64 / kVecSize;
  constexpr uint32_t kFp4Lanes = deepseek_v4::fp4::kCompressedKVBlockSize / kVecSize;
  constexpr int64_t kPageBytes = host::div_ceil(584ll << kPageBits, 576) * 576;
  static_assert(kHeadDim == 512 && kRopeDim == 64, "the 584-byte layout requires (512, 64)");
  static_assert(kHeadDim % kVecSize == 0 && kVecSize % 2 == 0);
  static_assert(kRowLanes % kWarpThreads == 0, "a token owns a whole number of warps");
  static_assert(kNopeLanes % kFp8Lanes == 0, "the nope part must end on an fp8 scale block");
  static_assert(
      (kHeadDim - kRopeDim) % deepseek_v4::fp4::kCompressedKVBlockSize == 0,
      "no fp4 block may straddle the nope/rope seam");
  static_assert(kFp8Lanes <= kWarpThreads && kFp4Lanes <= kWarpThreads);

  using bf16_vec_t = AlignedVector<bf16x2_t, kVecSize / 2>;
  using fp8_vec_t = AlignedVector<fp8x2_e4m3_t, kVecSize / 2>;
  using freq_vec_t = AlignedVector<float, kVecSize>;

  const uint32_t tx = threadIdx.x;
  const uint32_t row = blockIdx.x;

  // `out_loc` and `positions` are step metadata, independent of the PDL producer.
  // Slots fit in int32; padded rows are suppressed at the cache store.
  const auto out_loc = static_cast<int32_t>(static_cast<const LocT*>(params.out_loc)[row]);
  const auto position = static_cast<int64_t>(static_cast<const PosT*>(params.positions)[row]);
  PDLWaitPrimary<kUsePDL>();

  float data[kVecSize];
  bf16_vec_t latent;
  {
    bf16_vec_t input, weight;
    input.load(params.kv_input + row * kHeadDim, tx);
    weight.load(params.norm_weight, tx);

    // `project` already returns bf16 at ratio 1, so `finish`'s `.to(bfloat16)`
    // is a no-op and the statistic is taken over the loaded values as they are.
    float local_sqrsum = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kVecSize / 2; ++j) {
      const auto [x, y] = cast<fp32x2_t>(input[j]);
      local_sqrsum += x * x;
      local_sqrsum += y * y;
      data[j * 2 + 0] = x;
      data[j * 2 + 1] = y;
    }

    __shared__ float s_warp_sum[kRowWarps];
    s_warp_sum[tx / kWarpThreads] = warp::reduce_sum(local_sqrsum);
    __syncthreads();
    float sqrsum = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kRowWarps; ++i) {
      sqrsum += s_warp_sum[i];
    }
    constexpr float kInvHeadDim = 1.0f / static_cast<float>(kHeadDim);
    const auto norm_factor = math::rsqrt(sqrsum * kInvHeadDim + params.eps);

#pragma unroll
    for (uint32_t j = 0; j < kVecSize / 2; ++j) {
      const auto [wx, wy] = cast<fp32x2_t>(weight[j]);
      const auto x = data[j * 2 + 0] * norm_factor * wx;
      const auto y = data[j * 2 + 1] * norm_factor * wy;
      latent[j] = cast<bf16x2_t>(fp32x2_t{x, y});
    }
  }

  // Publish the pre-RoPE latent before the PDL trigger for the index-K `wk` GEMM.
  // Padded rows publish a latent too; the caller discards it.
  latent.store(params.kv_output + row * kHeadDim, tx);
  PDLTriggerSecondary<kUsePDL>();

  // Match finish()'s bf16 rounding before the main-KV RoPE.
#pragma unroll
  for (uint32_t j = 0; j < kVecSize / 2; ++j) {
    const auto [x, y] = cast<fp32x2_t>(latent[j]);
    data[j * 2 + 0] = x;
    data[j * 2 + 1] = y;
  }

  if (tx >= kNopeLanes) {
    // `rope_tail` ends in `.to(x.dtype)`, so the rotated value is rounded back
    // to bf16 before the fake-quant widens it again.
    freq_vec_t freq;
    freq.load(params.freqs_cis + position * kRopeDim, tx - kNopeLanes);
#pragma unroll
    for (uint32_t j = 0; j < kVecSize / 2; ++j) {
      const auto k = j * 2;
      const auto x_real = data[k + 0];
      const auto x_imag = data[k + 1];
      const auto f_real = freq[k + 0];
      const auto f_imag = freq[k + 1];
      const auto rotated =
          cast<bf16x2_t>(fp32x2_t{x_real * f_real - x_imag * f_imag, x_real * f_imag + x_imag * f_real});
      const auto [r0, r1] = cast<fp32x2_t>(rotated);
      data[k + 0] = r0;
      data[k + 1] = r1;
    }
  }

  // FP4/E4M3 fake-quant over 16 elements, i.e. kFp4Lanes threads.
  {
    float amax = fabsf(data[0]);
#pragma unroll
    for (uint32_t i = 1; i < kVecSize; ++i) {
      amax = fmaxf(amax, fabsf(data[i]));
    }
    amax = warp::reduce_max<kFp4Lanes>(amax);
    const auto scale = deepseek_v4::fp4::compressed_kv_scale(amax);
#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      const auto [x, y] = deepseek_v4::fp4::fake_quant_compressed_kv_x2({data[i * 2 + 0], data[i * 2 + 1]}, scale);
      data[i * 2 + 0] = x;
      data[i * 2 + 1] = y;
    }
  }

  // A padded CUDA-graph row carries `out_loc == 0`, the reserved dummy slot,
  // and must publish nothing: at ratio 1 the compressed slot *is* the FULL
  // slot, so there is nothing to divide and no other marker to read.
  if (out_loc <= 0) return;
  const int32_t page = out_loc >> kPageBits;
  const int32_t slot = out_loc & ((1 << kPageBits) - 1);
  const auto page_ptr = params.kvcache + page * kPageBytes;
  const auto value_ptr = page_ptr + slot * 576;

  if (tx >= kNopeLanes) {
    bf16_vec_t rope_out;
#pragma unroll
    for (uint32_t j = 0; j < kVecSize / 2; ++j) {
      rope_out[j] = cast<bf16x2_t>(fp32x2_t{data[j * 2 + 0], data[j * 2 + 1]});
    }
    rope_out.store(value_ptr + (kHeadDim - kRopeDim), tx - kNopeLanes);
  } else {
    // fp8 e4m3 with one ue8m0 scale per 64 elements.
    float abs_max = fabsf(data[0]);
#pragma unroll
    for (uint32_t i = 1; i < kVecSize; ++i) {
      abs_max = fmaxf(abs_max, fabsf(data[i]));
    }
    abs_max = warp::reduce_max<kFp8Lanes>(abs_max);
    const auto scale_ue8m0 = cast_to_ue8m0(fmaxf(1e-4f, abs_max) / math::FP8_E4M3_MAX);
    const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);
    fp8_vec_t nope_out;
#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      nope_out[i] = pack_fp8(data[i * 2 + 0] * inv_scale, data[i * 2 + 1] * inv_scale);
    }
    nope_out.store(value_ptr, tx);
    if (tx % kFp8Lanes == 0) {
      (page_ptr + (576 << kPageBits) + slot * 8)[tx / kFp8Lanes] = scale_ue8m0;
    }
  }
}

/// \brief Host side of `flash_c1_decode_kernel`.
template <int64_t kHeadDim, int64_t kRopeDim, uint32_t kPageSize, bool kUsePDL>
struct FlashC1DecodeKernel {
  static constexpr int32_t kPageBits = std::bit_width(kPageSize) - 1;
  static constexpr int64_t kPageBytes = host::div_ceil(584ll * kPageSize, 576) * 576;
  static constexpr uint32_t kBlockSize = kHeadDim / kC1VecSize;

  static_assert(std::has_single_bit(kPageSize), "the page/slot split needs a power-of-two page");
  static_assert(kBlockSize % device::kWarpThreads == 0 && kBlockSize <= 1024);

  template <typename PosT, typename LocT>
  static constexpr auto kernel = flash_c1_decode_kernel<kUsePDL, kHeadDim, kRopeDim, kPageBits, PosT, LocT>;

  /// \brief The (`positions`, `out_loc`) dtype pair, resolved at run time.
  static auto select(const bool pos_i32, const bool loc_i32) {
    if (pos_i32) return loc_i32 ? kernel<int32_t, int32_t> : kernel<int32_t, int64_t>;
    return loc_i32 ? kernel<int64_t, int32_t> : kernel<int64_t, int64_t>;
  }

  /// \brief RMSNorm + RoPE + fp4 fake-quant + the 584-byte store, one launch.
  ///
  /// \param kv_input `[num_tokens, kHeadDim]` bf16, the `wkv` projection.
  /// \param kv_output `[num_tokens, kHeadDim]` bf16, the pre-RoPE latent.
  /// \param norm_weight `[kHeadDim]` bf16.
  /// \param freqs_cis `[max_pos, kRopeDim]` fp32, real/imag interleaved.
  /// \param positions `[num_tokens]` int32 or int64, indexed as-is.
  /// \param out_loc `[num_tokens]` int32 or int64, the compressed slot; `0` is a padded row.
  /// \param kvcache `[npages, kPageBytes]` uint8, or the pool's fp8 view of it.
  static void run_decode_fusion(
      const tvm::ffi::TensorView kv_input,
      const tvm::ffi::TensorView kv_output,
      const tvm::ffi::TensorView norm_weight,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView kvcache,
      const float eps) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({N, kHeadDim})  //
        .with_dtype<bf16_t>()
        .with_device(device_)
        .verify(kv_input)
        .verify(kv_output);
    TensorMatcher({kHeadDim}).with_dtype<bf16_t>().with_device(device_).verify(norm_weight);
    // Real/imag interleaved, so the trailing dim is kRopeDim, not kRopeDim / 2.
    TensorMatcher({-1, kRopeDim}).with_dtype<fp32_t>().with_device(device_).verify(freqs_cis);
    // The scheduler's `out_cache_loc` (which `c1_out_loc` aliases at ratio 1)
    // is int64; the unit tests hand int32. Both are indexed as-is.
    auto pos_dtype = SymbolicDType{};
    auto loc_dtype = SymbolicDType{};
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(pos_dtype).with_device(device_).verify(positions);
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(loc_dtype).with_device(device_).verify(out_loc);
    // The pool allocates the buffer as uint8 and hands it out viewed as its fp8
    // dtype (`get_extra_key_buffer`); both are one byte per element.
    TensorMatcher({-1, kPageBytes}).with_dtype<uint8_t, fp8_e4m3_t>().with_device(device_).verify(kvcache);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_tokens == 0) return;

    const auto params = C1Params{
        .kv_input = static_cast<const bf16_t*>(kv_input.data_ptr()),
        .kv_output = static_cast<bf16_t*>(kv_output.data_ptr()),
        .norm_weight = static_cast<const bf16_t*>(norm_weight.data_ptr()),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .positions = positions.data_ptr(),
        .out_loc = out_loc.data_ptr(),
        .kvcache = static_cast<uint8_t*>(kvcache.data_ptr()),
        .eps = eps,
    };
    const auto k = select(pos_dtype.is_type<int32_t>(), loc_dtype.is_type<int32_t>());
    LaunchKernel(num_tokens, kBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(k, params);
  }
};

}  // namespace sglang
