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
#include <optional>

namespace sglang {

/// \brief Ratio-2 decode compressor: pair-pool, RMSNorm and the main-KV write.
///
/// `kv_input` and `kv_state` rows are `2 * kHeadDim` floats, kv then score.
/// `kv_output` is the pre-RoPE latent, for the index-K branch's `wk`.
struct C2Params {
  const float* __restrict__ kv_input;  // [num_tokens, 2 * kHeadDim] fp32
  /// `CompressStatePool`'s flat `KVAndScore` buffer, `[size, 2 * kHeadDim]`
  /// fp32 with kv in the low half and score in the high half. A request's
  /// pending pair lives at `req * ring_size + pos % ring_size`.
  float* __restrict__ kv_state;
  bf16_t* __restrict__ kv_output;          // [num_tokens, kHeadDim] bf16, pre-RoPE
  const bf16_t* __restrict__ norm_weight;  // [kHeadDim] bf16
  const float* __restrict__ freqs_cis;     // [max_pos, kRopeDim] fp32, real/imag interleaved
  const void* __restrict__ positions;      // [num_tokens] PosT
  const int64_t* __restrict__ req;         // [num_tokens], req_pool_idx per token
  const void* __restrict__ raw_out_loc;    // [num_tokens] LocT, the FULL slot; 0 marks a padded row
  uint8_t* __restrict__ kvcache;           // [npages, kPageBytes] uint8
  /// Positions per request slot in the pair-state ring.
  uint32_t ring_size;
  float eps;
};

/// Elements per thread; 256 threads per token was measured fastest on B200 decode batches.
/// The launcher and launch bounds share this value. At head_dim 512, the nope/rope split
/// is (512 - 64) / 2 = 224 threads, keeping the fp8 amax reduction's full-warp mask valid.
constexpr uint32_t kC2VecSize = 2;

/// \brief grid = num_tokens, block = kHeadDim / kC2VecSize.
///
/// An odd position completes a group with its even predecessor; an even one
/// parks itself in the state.
///
/// Target-verify runs that same schedule with `draft_len` consecutive positions
/// per request instead of one, so a row's partner is usually the row before it
/// in `kv_input` rather than the ring. That is the whole difference, and a 2D
/// grid answers it without arithmetic: `blockIdx.x` is the position inside the
/// block, `blockIdx.y` the request.
///
/// The three reductions below have three different widths and are not
/// interchangeable: the RMSNorm statistic spans the row, an fp8 store scale
/// spans 64 elements, an fp4 block spans 16. All asserted.
template <
    bool kUsePDL,
    bool kStore,
    bool kVerify,
    int64_t kHeadDim,
    int64_t kRopeDim,
    int32_t kPageBits,
    typename PosT,
    typename LocT>
__global__ __launch_bounds__(kHeadDim / kC2VecSize) void flash_c2_decode_kernel(const C2Params params) {
  using namespace device;
  using deepseek_v4::fp8::cast_to_ue8m0;
  using deepseek_v4::fp8::inv_scale_ue8m0;
#ifndef USE_ROCM
  using deepseek_v4::fp8::pack_fp8;
#else
  using deepseek_v4::fp8::rn::pack_fp8;  // the hardware RNE pack, rounding as CUDA's does
#endif

  constexpr uint32_t kVecSize = kC2VecSize;
  constexpr uint32_t kCTASize = kHeadDim / kVecSize;
  constexpr int64_t kStride = kHeadDim * 2;
  /// Threads covering the fp8 nope part; the rest carry the bf16 RoPE tail.
  constexpr uint32_t kNopeThreads = (kHeadDim - kRopeDim) / kVecSize;
  constexpr uint32_t kFp8Lanes = 64 / kVecSize;
  constexpr uint32_t kFp4Lanes = deepseek_v4::fp4::kCompressedKVBlockSize / kVecSize;
  constexpr int64_t kPageBytes = host::div_ceil(584ll << kPageBits, 576) * 576;

  static_assert(kHeadDim == (kVecSize * kCTASize));
  static_assert(kCTASize % kWarpThreads == 0);
  static_assert(kNopeThreads % kFp8Lanes == 0, "the nope part must end on an fp8 scale block");
  static_assert(kWarpThreads % kFp8Lanes == 0 && kWarpThreads % kFp4Lanes == 0);
  static_assert(kHeadDim == 512 && kRopeDim == 64, "the 584-byte layout requires (512, 64)");
  using fp32_vec_t = AlignedVector<float, kVecSize>;
  using bf16_vec_t = AlignedVector<bf16x2_t, kVecSize / 2>;

  const auto tx = threadIdx.x;
  // Verify gives each request a CTA column; decode a flat grid of one row each.
  const auto row = kVerify ? blockIdx.y * gridDim.x + blockIdx.x : blockIdx.x;
  // Slots fit in int32 whatever width the scheduler hands them in.
  const auto raw_out_loc = static_cast<int32_t>(static_cast<const LocT*>(params.raw_out_loc)[row]);
  const auto pos = static_cast<const PosT*>(params.positions)[row];
  // A completing row reads the slot left by `pos - 1`;
  // a pending row writes its own slot, so reads and writes stay disjoint.
  const auto ring = static_cast<int64_t>(params.req[row]) * params.ring_size;
  const auto read_row = ring + (pos - 1 + params.ring_size) % params.ring_size;
  const auto write_row = ring + pos % params.ring_size;
  PDLWaitPrimary<kUsePDL>();

  fp32_vec_t kv_new, score_new;
  kv_new.load(params.kv_input + row * kStride, tx);
  score_new.load(params.kv_input + row * kStride, tx + kCTASize);

  fp32_vec_t kv_old, score_old;
  // Only a verify block's first row carries over from the ring; the rest pair
  // with the row before them, which is already in `kv_input` under the same
  // `| kv | score |` layout as the state, so this is a pointer swap. Taking the
  // in-block partner from the input is also what keeps it race-free: the CTA
  // that publishes that ring slot belongs to this very launch.
  const float* partner = params.kv_state + read_row * kStride;
  if constexpr (kVerify) {
    if (blockIdx.x != 0) partner = params.kv_input + static_cast<int64_t>(row - 1) * kStride;
  }
  kv_old.load(partner, tx);
  score_old.load(partner, tx + kCTASize);

  if ((pos & 1) == 0) {
    // padded case
    if (raw_out_loc == 0) return PDLTriggerSecondary<kUsePDL>();
    kv_new.store(params.kv_state + write_row * kStride, tx);
    score_new.store(params.kv_state + write_row * kStride, tx + kCTASize);
    return PDLTriggerSecondary<kUsePDL>();
  }

  constexpr uint32_t kNumWarps = kCTASize / kWarpThreads;
  __shared__ float s_warp_sum[kNumWarps];
  fp32_vec_t staged, freq;
  bf16_vec_t weight, out;
  weight.load(params.norm_weight, tx);
  if constexpr (kStore) {
    if (tx >= kNopeThreads) freq.load(params.freqs_cis + (pos - 1) * kRopeDim, tx - kNopeThreads);
  }

  // With two scores `exp(-|s0 - s1|)` is the whole softmax: one exp, argument
  // always <= 0, so no max-subtraction pass and no overflow.
#pragma unroll
  for (uint32_t i = 0; i < kVecSize; ++i) {
    const auto delta = score_old[i] - score_new[i];
    const auto scale = expf(-fabsf(delta));
    const auto scale_0 = delta > 0 ? 1.0f : scale;
    const auto scale_1 = delta > 0 ? scale : 1.0f;
    staged[i] = (kv_old[i] * scale_0 + kv_new[i] * scale_1) / (1.0f + scale);
  }

  // `finish` casts to bf16 before the norm, so the sum of squares has to see
  // the rounded values.
  float local_sqrsum = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < kVecSize / 2; ++i) {
    const auto packed = fp32x2_t{staged[i * 2 + 0], staged[i * 2 + 1]};
    const auto [x, y] = cast<fp32x2_t>(cast<bf16x2_t>(packed));
    local_sqrsum += x * x;
    local_sqrsum += y * y;
    staged[i * 2 + 0] = x;
    staged[i * 2 + 1] = y;
  }
  const auto warp_sum = warp::reduce_sum(local_sqrsum);
  s_warp_sum[tx / kWarpThreads] = warp_sum;
  __syncthreads();

  float sqrsum = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < kNumWarps; ++i) {
    sqrsum += s_warp_sum[i];
  }
  constexpr float kInvScale = 1.0f / static_cast<float>(kHeadDim);
  const auto norm_factor = math::rsqrt(sqrsum * kInvScale + params.eps);

#pragma unroll
  for (uint32_t i = 0; i < kVecSize / 2; ++i) {
    const auto [wx, wy] = cast<fp32x2_t>(weight[i]);
    const auto x = staged[i * 2 + 0] * norm_factor * wx;
    const auto y = staged[i * 2 + 1] * norm_factor * wy;
    out[i] = cast<bf16x2_t>(fp32x2_t{x, y});
  }
  // The pre-RoPE latent, for the index-K branch's `wk` projection. Published
  // before the trigger because that GEMM is the successor that reads it.
  out.store(params.kv_output, static_cast<int64_t>(row) * kCTASize + tx);
  PDLTriggerSecondary<kUsePDL>();

  if constexpr (kStore) {
    // ---- main-KV branch: RoPE tail, fp4 fake-quant, 584-byte store ----
    // Match finish()'s bf16 rounding before RoPE.
#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      const auto [x, y] = cast<fp32x2_t>(out[i]);
      staged[i * 2 + 0] = x;
      staged[i * 2 + 1] = y;
    }

    if (tx >= kNopeThreads) {
      // Match rope_tail()'s bf16 rounding before fake quantization.
      // Only odd positions reach here; the latent represents `pos - 1`.
      freq.load(params.freqs_cis + (pos - 1) * kRopeDim, tx - kNopeThreads);
#pragma unroll
      for (uint32_t i = 0; i < kVecSize / 2; ++i) {
        const auto x_real = staged[i * 2 + 0];
        const auto x_imag = staged[i * 2 + 1];
        const auto f_real = x_real * freq[i * 2 + 0] - x_imag * freq[i * 2 + 1];
        const auto f_imag = x_real * freq[i * 2 + 1] + x_imag * freq[i * 2 + 0];
        const auto rotated = cast<bf16x2_t>(fp32x2_t{f_real, f_imag});
        const auto [r0, r1] = cast<fp32x2_t>(rotated);
        staged[i * 2 + 0] = r0;
        staged[i * 2 + 1] = r1;
      }
    }

    // FP4/E4M3 fake-quant over 16 elements, i.e. kFp4Lanes threads.
    {
      float amax = fabsf(staged[0]);
#pragma unroll
      for (uint32_t i = 1; i < kVecSize; ++i) {
        amax = fmaxf(amax, fabsf(staged[i]));
      }
      amax = warp::reduce_max<kFp4Lanes>(amax);
      const auto scale = deepseek_v4::fp4::compressed_kv_scale(amax);
#pragma unroll
      for (uint32_t i = 0; i < kVecSize / 2; ++i) {
        const auto [x, y] =
            deepseek_v4::fp4::fake_quant_compressed_kv_x2({staged[i * 2 + 0], staged[i * 2 + 1]}, scale);
        staged[i * 2 + 0] = x;
        staged[i * 2 + 1] = y;
      }
    }

    // padded case
    if (raw_out_loc == 0) return;
    // `raw_out_loc / ratio`; ratio 2 makes it a shift.
    const int32_t out_loc = raw_out_loc >> 1;
    const int32_t page = out_loc >> kPageBits;
    const int32_t slot = out_loc & ((1 << kPageBits) - 1);
    const auto page_ptr = params.kvcache + page * kPageBytes;
    const auto value_ptr = page_ptr + slot * 576;

    if (tx >= kNopeThreads) {
      bf16_vec_t rope_out;
#pragma unroll
      for (uint32_t i = 0; i < kVecSize / 2; ++i) {
        rope_out[i] = cast<bf16x2_t>(fp32x2_t{staged[i * 2 + 0], staged[i * 2 + 1]});
      }
      rope_out.store(value_ptr + (kHeadDim - kRopeDim), tx - kNopeThreads);
    } else {
      // fp8 e4m3 with one ue8m0 scale per 64 elements.
      auto abs_max = fabsf(staged[0]);
#pragma unroll
      for (uint32_t i = 1; i < kVecSize; ++i) {
        abs_max = fmaxf(abs_max, fabsf(staged[i]));
      }
      abs_max = warp::reduce_max<kFp8Lanes>(abs_max);
      const auto scale_ue8m0 = cast_to_ue8m0(fmaxf(1e-4f, abs_max) / math::FP8_E4M3_MAX);
      const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);
#pragma unroll
      for (uint32_t i = 0; i < kVecSize / 2; ++i) {
        reinterpret_cast<fp8x2_e4m3_t*>(value_ptr)[tx * (kVecSize / 2) + i] =
            pack_fp8(staged[i * 2 + 0] * inv_scale, staged[i * 2 + 1] * inv_scale);
      }
      if (tx % kFp8Lanes == 0) {
        (page_ptr + (576 << kPageBits) + slot * 8)[tx / kFp8Lanes] = scale_ue8m0;
      }
    }
  }
}

template <int64_t kHeadDim, int64_t kRopeDim, uint32_t kPageSize, bool kUsePDL>
struct FlashC2DecodeKernel {
  static constexpr uint32_t kBlockSize = kHeadDim / kC2VecSize;
  static constexpr int32_t kPageBits = std::bit_width(kPageSize) - 1;
  static constexpr int64_t kPageBytes = host::div_ceil(584ll * kPageSize, 576) * 576;
  template <bool kStore, bool kVerify, typename PosT, typename LocT>
  static constexpr auto kernel =
      flash_c2_decode_kernel<kUsePDL, kStore, kVerify, kHeadDim, kRopeDim, kPageBits, PosT, LocT>;

  /// \brief The (`positions`, `raw_out_loc`) dtype pair, resolved at run time.
  template <bool kStore, bool kVerify>
  static auto select(const bool pos_i32, const bool loc_i32) {
    if (pos_i32) return loc_i32 ? kernel<kStore, kVerify, int32_t, int32_t> : kernel<kStore, kVerify, int32_t, int64_t>;
    return loc_i32 ? kernel<kStore, kVerify, int64_t, int32_t> : kernel<kStore, kVerify, int64_t, int64_t>;
  }

  // The sum of squares is reduced through a fixed-size shared array, so the CTA
  // has to be a whole number of warps.
  static_assert(kHeadDim % (4 * device::kWarpThreads) == 0, "head_dim must be a multiple of 128");
  static_assert(std::has_single_bit(kPageSize), "the page/slot split needs a power-of-two page");

  /// \brief Pool + norm only. The main-KV write stays with the caller.
  static void run_decode(
      const tvm::ffi::TensorView kv_input,
      const tvm::ffi::TensorView kv_state,
      const tvm::ffi::TensorView kv_output,
      const tvm::ffi::TensorView norm_weight,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView req,
      const tvm::ffi::TensorView raw_out_loc,
      const float eps,
      const int64_t ring_size) {
    launch(
        kv_input,
        kv_state,
        kv_output,
        norm_weight,
        positions,
        req,
        raw_out_loc,
        eps,
        ring_size,
        std::nullopt,
        std::nullopt);
  }

  /// \brief Pool + norm + the RoPE tail, fp4 fake-quant and 584-byte store.
  static void run_decode_fusion(
      const tvm::ffi::TensorView kv_input,
      const tvm::ffi::TensorView kv_state,
      const tvm::ffi::TensorView kv_output,
      const tvm::ffi::TensorView norm_weight,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView req,
      const tvm::ffi::TensorView raw_out_loc,
      const float eps,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView kvcache,
      const int64_t ring_size) {
    launch(kv_input, kv_state, kv_output, norm_weight, positions, req, raw_out_loc, eps, ring_size, freqs_cis, kvcache);
  }

  /// \brief `run_decode_fusion` for a target-verify block.
  ///
  /// `draft_len` consecutive positions per request, request-major, which the
  /// grid reproduces as `draft_len x batch`.
  static void run_verify_fusion(
      const tvm::ffi::TensorView kv_input,
      const tvm::ffi::TensorView kv_state,
      const tvm::ffi::TensorView kv_output,
      const tvm::ffi::TensorView norm_weight,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView req,
      const tvm::ffi::TensorView raw_out_loc,
      const float eps,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView kvcache,
      const int64_t ring_size,
      const int64_t draft_len) {
    launch(
        kv_input,
        kv_state,
        kv_output,
        norm_weight,
        positions,
        req,
        raw_out_loc,
        eps,
        ring_size,
        freqs_cis,
        kvcache,
        draft_len);
  }

 private:
  using MaybeTensor = std::optional<tvm::ffi::TensorView>;

  static void launch(
      const tvm::ffi::TensorView kv_input,
      const tvm::ffi::TensorView kv_state,
      const tvm::ffi::TensorView kv_output,
      const tvm::ffi::TensorView norm_weight,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView req,
      const tvm::ffi::TensorView raw_out_loc,
      const float eps,
      const int64_t ring_size,
      const MaybeTensor freqs_cis,
      const MaybeTensor kvcache,
      const int64_t draft_len = 1) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    TensorMatcher({N, kHeadDim * 2}).with_dtype<fp32_t>().with_device(device_).verify(kv_input);
    TensorMatcher({-1, kHeadDim * 2}).with_dtype<fp32_t>().with_device(device_).verify(kv_state);
    // Only rows that complete a group are written.
    TensorMatcher({N, kHeadDim}).with_dtype<bf16_t>().with_device(device_).verify(kv_output);
    TensorMatcher({kHeadDim}).with_dtype<bf16_t>().with_device(device_).verify(norm_weight);
    // Metadata retains its original dtypes: the scheduler uses int64 locations,
    // while callers may also supply int32 locations and positions.
    auto pos_dtype = SymbolicDType{};
    auto loc_dtype = SymbolicDType{};
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(pos_dtype).with_device(device_).verify(positions);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device_).verify(req);
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(loc_dtype).with_device(device_).verify(raw_out_loc);

    const auto store = freqs_cis.has_value();
    if (store) {
      // Real/imag interleaved, so the trailing dim is kRopeDim, not kRopeDim / 2.
      TensorMatcher({-1, kRopeDim}).with_dtype<fp32_t>().with_device(device_).verify(*freqs_cis);
      // The pool allocates the buffer as uint8 and hands it out viewed as its
      // fp8 dtype (`get_extra_key_buffer`); both are one byte per element.
      TensorMatcher({-1, kPageBytes}).with_dtype<uint8_t, fp8_e4m3_t>().with_device(device_).verify(*kvcache);
    }

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_tokens == 0) return;
    RuntimeCheck(ring_size > 0, "the pair-state ring must have at least one position");
    RuntimeCheck(draft_len >= 1, "the draft length ", draft_len, " must be positive");
    const auto is_verify = draft_len > 1;
    RuntimeCheck(!is_verify || num_tokens % draft_len == 0, "verify rows must be a whole number of blocks");
    // A block publishes its own even positions, so the slot its first row reads
    // stays out of the launch's reach only while the ring is wider than the
    // block. `get_compress_state_ring_size` satisfies this by construction.
    RuntimeCheck(
        !is_verify || ring_size > draft_len,
        "the pair-state ring (",
        ring_size,
        ") must be wider than the draft length (",
        draft_len,
        ")");

    const auto params = C2Params{
        .kv_input = static_cast<const float*>(kv_input.data_ptr()),
        .kv_state = static_cast<float*>(kv_state.data_ptr()),
        .kv_output = static_cast<bf16_t*>(kv_output.data_ptr()),
        .norm_weight = static_cast<const bf16_t*>(norm_weight.data_ptr()),
        .freqs_cis = store ? static_cast<const float*>(freqs_cis->data_ptr()) : nullptr,
        .positions = positions.data_ptr(),
        .req = static_cast<const int64_t*>(req.data_ptr()),
        .raw_out_loc = raw_out_loc.data_ptr(),
        .kvcache = store ? static_cast<uint8_t*>(kvcache->data_ptr()) : nullptr,
        .ring_size = static_cast<uint32_t>(ring_size),
        .eps = eps,
    };
    // `LaunchKernel` is move-only, so each arm builds its own.
    const auto pos_i32 = pos_dtype.is_type<int32_t>();
    const auto loc_i32 = loc_dtype.is_type<int32_t>();
    if (is_verify) {
      const auto block = static_cast<uint32_t>(draft_len);
      const auto k = select<true, true>(pos_i32, loc_i32);
      LaunchKernel(dim3{block, num_tokens / block}, kBlockSize, device_.unwrap())  //
          .enable_pdl(kUsePDL)(k, params);
    } else if (store) {
      const auto k = select<true, false>(pos_i32, loc_i32);
      LaunchKernel(num_tokens, kBlockSize, device_.unwrap())  //
          .enable_pdl(kUsePDL)(k, params);
    } else {
      const auto k = select<false, false>(pos_i32, loc_i32);
      LaunchKernel(num_tokens, kBlockSize, device_.unwrap())  //
          .enable_pdl(kUsePDL)(k, params);
    }
  }
};

}  // namespace sglang
