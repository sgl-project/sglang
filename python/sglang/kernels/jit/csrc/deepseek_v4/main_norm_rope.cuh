#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>

namespace sglang {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

SGL_DEVICE uint8_t quant_fp4_e2m1(float x) {
  const float ax = fminf(fabsf(x), 6.0f);
  uint8_t idx = 0;
  idx += ax > 0.25f;
  idx += ax > 0.75f;
  idx += ax > 1.25f;
  idx += ax > 1.75f;
  idx += ax > 2.5f;
  idx += ax > 3.5f;
  idx += ax > 5.0f;
  if (x < 0.0f && idx != 0) idx |= 0x8;
  return idx;
}

// 4 warps per block: warp-per-(token, head) work-item dispatch (Q kernel).
constexpr uint32_t kFusedQBlockSize = 128;
constexpr uint32_t kFusedQNumWarps = kFusedQBlockSize / device::kWarpThreads;

// 8 warps per block: block-per-token work-item dispatch (K kernel).
constexpr uint32_t kFusedKBlockSize = 256;
constexpr uint32_t kFusedKNumWarps = kFusedKBlockSize / device::kWarpThreads;

#define Q_KERNEL __global__ __launch_bounds__(kFusedQBlockSize, 16)
#define K_KERNEL __global__ __launch_bounds__(kFusedKBlockSize, 8)

template <int64_t kRopeDim>
SGL_DEVICE device::AlignedVector<float, 4>
load_rope_first_cos_sin(const float* __restrict__ cos_sin_cache, int32_t lane_id) {
  constexpr int64_t kHalfRopeDim = kRopeDim / 2;
  const int32_t pair0 = lane_id * 2;
  const int32_t pair1 = pair0 + 1;
  device::AlignedVector<float, 4> freq;
  freq[0] = cos_sin_cache[pair0];
  freq[1] = cos_sin_cache[kHalfRopeDim + pair0];
  freq[2] = cos_sin_cache[pair1];
  freq[3] = cos_sin_cache[kHalfRopeDim + pair1];
  return freq;
}

// ============================================================================
// Q kernel: warp-per-(token, head) rmsnorm-self + RoPE + write to q_out.
// ============================================================================

struct FusedQNormRopeParams {
  const void* __restrict__ q_input;     // (B, num_q_heads, kHeadDim) DType
  void* __restrict__ q_output;          // (B, num_q_heads, kHeadDim) DType
  const float* __restrict__ freqs_cis;  // (max_pos, kRopeDim) fp32 (re/im interleaved)
  const void* __restrict__ positions;   // (B,) PosT
  int64_t q_input_stride_batch;
  int64_t q_output_stride_batch;
  uint32_t batch_size;
  uint32_t num_q_heads;
  float eps;
};

template <typename DType, int64_t kHeadDim, int64_t kRopeDim, typename PosT, bool kUsePDL, bool kBlockPerToken = true>
Q_KERNEL void fused_q_norm_rope(const __grid_constant__ FusedQNormRopeParams params) {
  using namespace device;

  constexpr int64_t kMaxVecSize = 16 / sizeof(DType);
  constexpr int64_t kVecSize = std::min(kMaxVecSize, kHeadDim / kWarpThreads);
  constexpr int64_t kLocalSize = kHeadDim / (kWarpThreads * kVecSize);
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  static_assert(kHeadDim % (kWarpThreads * kVecSize) == 0);
  static_assert(kLocalSize * kVecSize * kWarpThreads == kHeadDim);
  static_assert(kRopeDim % kVecSize == 0);
  static_assert(kRopeSize <= kWarpThreads);
  static_assert(kRopeDim == kWarpThreads * 2, "1 (real, imag) pair per lane");

  // fp8: 2 work-items/warp so their loads overlap (latency-bound); bf16: 1
  // (bandwidth-bound, a 2nd item just adds register pressure). Per-item math is
  // identical either way -> the output is bitwise-identical for both.
  constexpr uint32_t kWorkPerWarp = (sizeof(DType) == 1) ? 2u : 1u;

  using Storage = AlignedVector<DType, kVecSize>;
  using DType2 = packed_t<DType>;
  constexpr bool kVecConvert = (sizeof(DType) == 1);  // fp8 only
  // fp8 shares one freq row per token across its heads (block-per-token), but
  // only when the grid fills the SMs; the host picks kBlockPerToken=false at
  // small N (crossover total_works>=kBPTMinWorks). bf16 never uses it.
  // Dispatch-only, so the per-work-item math is byte-identical.
  constexpr bool kUseBPT = kVecConvert && kBlockPerToken;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const uint32_t total_works = params.batch_size * params.num_q_heads;

  // rope-tail staging. fp8's part-2 read is 32 lanes x fp8x2 (2B) -> 2 lanes
  // share a 4B SMEM bank; pad each pair to its own 4B slot to avoid that. bf16
  // pairs are already 4B, so bf16 keeps the packed layout.
  __shared__ Storage s_rope[kFusedQNumWarps][kRopeSize];
  __shared__ uint32_t s_rope_pad[kFusedQNumWarps][kVecConvert ? kWarpThreads : 1];
  // fp8 block-per-token: one freq row shared by all heads of the block's token.
  __shared__ fp32x2_t s_freq[kVecConvert ? kWarpThreads : 1];
  const auto gmem = tile::Memory<Storage>{lane_id, kWarpThreads};
  const auto mem_freq = tile::Memory<fp32x2_t>{lane_id, kWarpThreads};
  const bool is_rope_lane = lane_id >= kWarpThreads - kRopeSize;

  // Per-work-item context + a hoisted input load (all loads issued first).
  Storage input_vec[kWorkPerWarp][kLocalSize];
  const DType* out_ptr[kWorkPerWarp];
  int32_t position[kWorkPerWarp];  // bf16 path only; fp8 uses shared s_freq
  uint32_t n_work;

  if constexpr (kUseBPT) {
    // ---- fp8: block-per-token dispatch. A block is pinned to one token and
    // covers up to kHeadsPerBlock consecutive heads; the token's freq row is
    // loaded ONCE into s_freq (warp 0) and reused. freq is exact fp32 so the
    // shared value is bit-neutral -> the output is unchanged. All warps reach
    // __syncthreads (no divergent return); out-of-range warps just do zero
    // work-items.
    constexpr uint32_t kHeadsPerBlock = kFusedQNumWarps * kWorkPerWarp;
    const uint32_t blocks_per_token = (params.num_q_heads + kHeadsPerBlock - 1) / kHeadsPerBlock;
    const uint32_t token = blockIdx.x / blocks_per_token;
    const uint32_t head_block = blockIdx.x % blocks_per_token;
    const uint32_t head_base = head_block * kHeadsPerBlock + warp_id * kWorkPerWarp;
    const int32_t tpos = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[token]);

    PDLWaitPrimary<kUsePDL>();

    if (warp_id == 0) {
      s_freq[lane_id] = mem_freq.load(params.freqs_cis + tpos * kRopeDim);
    }

    n_work = (head_base >= params.num_q_heads)
                 ? 0u
                 : ((params.num_q_heads - head_base < kWorkPerWarp) ? (params.num_q_heads - head_base) : kWorkPerWarp);

#pragma unroll
    for (uint32_t w = 0; w < kWorkPerWarp; ++w) {
      const uint32_t head_id = head_base + w;
      const uint32_t chead = (head_id < params.num_q_heads) ? head_id : (params.num_q_heads - 1);
      const auto in_ptr =
          static_cast<const DType*>(params.q_input) + token * params.q_input_stride_batch + chead * kHeadDim;
      out_ptr[w] = static_cast<DType*>(params.q_output) + token * params.q_output_stride_batch + chead * kHeadDim;
#pragma unroll
      for (int i = 0; i < kLocalSize; ++i) {
        input_vec[w][i] = gmem.load(in_ptr, i);
      }
    }
    __syncthreads();  // s_freq visible to all warps before part 2
  } else {
    // ---- warp-per-(token, head) dispatch (bf16 always; fp8 at small N). -----
    const auto warp_slot = blockIdx.x * kFusedQNumWarps + warp_id;
    const auto work_base = warp_slot * kWorkPerWarp;
    if (work_base >= total_works) return;

    PDLWaitPrimary<kUsePDL>();

    n_work = (total_works - work_base < kWorkPerWarp) ? (total_works - work_base) : kWorkPerWarp;
#pragma unroll
    for (uint32_t w = 0; w < kWorkPerWarp; ++w) {
      const uint32_t work_id = work_base + w;
      const uint32_t clamped = (work_id < total_works) ? work_id : (total_works - 1);
      const uint32_t batch_id = clamped / params.num_q_heads;
      const uint32_t head_id = clamped % params.num_q_heads;
      const auto in_ptr =
          static_cast<const DType*>(params.q_input) + batch_id * params.q_input_stride_batch + head_id * kHeadDim;
      out_ptr[w] = static_cast<DType*>(params.q_output) + batch_id * params.q_output_stride_batch + head_id * kHeadDim;
      position[w] = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[batch_id]);
#pragma unroll
      for (int i = 0; i < kLocalSize; ++i) {
        input_vec[w][i] = gmem.load(in_ptr, i);
      }
    }
  }

  // Process each in-range work-item; math identical to the 1-item baseline.
#pragma unroll
  for (uint32_t w = 0; w < kWorkPerWarp; ++w) {
    if (w >= n_work) break;
    const auto output_ptr = const_cast<DType*>(out_ptr[w]);
    fp32x2_t freq;
    if constexpr (kUseBPT) {
      freq = s_freq[lane_id];
    } else {
      freq = mem_freq.load(params.freqs_cis + position[w] * kRopeDim);
    }

    float sum_of_squares = 0.0f;
    if constexpr (kVecConvert) {
      // Vectorized fp8<->fp32: one hardware convert per pair instead of two
      // scalar ones (fp8 is instruction-issue bound). device::cast<fp32x2_t> and
      // device::cast<DType2> use the same round as scalar static_cast, and the
      // accumulation order is preserved (pair p -> elem 2p then 2p+1), so the
      // output stays bitwise-identical.
      constexpr int kPairs = kVecSize / 2;
#pragma unroll
      for (int i = 0; i < kLocalSize; ++i) {
        const auto* vp = reinterpret_cast<const DType2*>(input_vec[w][i].data());
#pragma unroll
        for (int p = 0; p < kPairs; ++p) {
          const auto f = cast<fp32x2_t>(vp[p]);
          sum_of_squares += f.x * f.x;
          sum_of_squares += f.y * f.y;
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < kLocalSize; ++i) {
#pragma unroll
        for (int j = 0; j < kVecSize; ++j) {
          const auto x = cast<float>(input_vec[w][i][j]);
          sum_of_squares += x * x;
        }
      }
    }
    sum_of_squares = warp::reduce_sum(sum_of_squares);
    const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);

    if constexpr (kVecConvert) {
      constexpr int kPairs = kVecSize / 2;
#pragma unroll
      for (int i = 0; i < kLocalSize; ++i) {
        auto* vp = reinterpret_cast<DType2*>(input_vec[w][i].data());
#pragma unroll
        for (int p = 0; p < kPairs; ++p) {
          const auto f = cast<fp32x2_t>(vp[p]);
          vp[p] = cast<DType2>(fp32x2_t{f.x * norm_factor, f.y * norm_factor});
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < kLocalSize; ++i) {
#pragma unroll
        for (int j = 0; j < kVecSize; ++j) {
          const auto x = cast<float>(input_vec[w][i][j]);
          input_vec[w][i][j] = cast<DType>(x * norm_factor);
        }
      }
    }

    // Streaming store (st.cs) of nope tiles; stash rope tile to shared.
#pragma unroll
    for (int i = 0; i < kLocalSize; ++i) {
      if (i == kLocalSize - 1 && is_rope_lane) {
        const auto rope_id = lane_id - (kWarpThreads - kRopeSize);
        if constexpr (kVecConvert) {
          // fp8: spread this lane's pairs into 4B slots (bank-conflict-free read).
          const auto* pr = reinterpret_cast<const DType2*>(input_vec[w][i].data());
#pragma unroll
          for (int p = 0; p < kVecSize / 2; ++p) {
            s_rope_pad[warp_id][rope_id * (kVecSize / 2) + p] =
                static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&pr[p]));
          }
        } else {
          s_rope[warp_id][rope_id] = input_vec[w][i];
        }
      } else {
        static_assert(sizeof(Storage) == 16, "expect 128-bit nope tile");
        __stcs(
            reinterpret_cast<int4*>(output_ptr) + (lane_id + i * kWarpThreads),
            *reinterpret_cast<const int4*>(&input_vec[w][i]));
      }
    }
    __syncwarp();

    // part 2: RoPE on all 32 lanes -- one (real, imag) pair per lane.
    const auto mem_elem = tile::Memory<DType2>{lane_id, kWarpThreads};
    DType2 elem;
    if constexpr (kVecConvert) {
      const uint16_t raw = static_cast<uint16_t>(s_rope_pad[warp_id][lane_id]);
      elem = *reinterpret_cast<const DType2*>(&raw);
    } else {
      elem = mem_elem.load(s_rope[warp_id]);
    }
    const auto [x_real, x_imag] = cast<fp32x2_t>(elem);
    const auto [freq_real, freq_imag] = freq;
    const fp32x2_t rotated = {
        x_real * freq_real - x_imag * freq_imag,
        x_real * freq_imag + x_imag * freq_real,
    };
    mem_elem.store(output_ptr + (kHeadDim - kRopeDim), cast<DType2>(rotated));
    if (w + 1 < n_work) __syncwarp();  // fence only before next item reuses s_rope
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <typename DType, int64_t kHeadDim, int64_t kRopeDim, bool kUsePDL>
struct FusedQNormRopeKernel {
  template <typename PosT, bool kBlockPerToken>
  static constexpr auto kernel = fused_q_norm_rope<DType, kHeadDim, kRopeDim, PosT, kUsePDL, kBlockPerToken>;

  static void forward(
      const tvm::ffi::TensorView q_input,
      const tvm::ffi::TensorView q_output,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView positions,
      float eps) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto H = SymbolicSize{"num_q_heads"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, H, kHeadDim})  //
        .with_strides({-1, kHeadDim, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(q_input);
    TensorMatcher({B, H, kHeadDim})  //
        .with_strides({-1, kHeadDim, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(q_output);
    TensorMatcher({-1, kRopeDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs_cis);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({B})  //
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device_)
        .verify(positions);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto num_q_heads = static_cast<uint32_t>(H.unwrap());
    if (batch_size == 0) return;

    const auto params = FusedQNormRopeParams{
        .q_input = q_input.data_ptr(),
        .q_output = q_output.data_ptr(),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .positions = positions.data_ptr(),
        .q_input_stride_batch = q_input.stride(0),
        .q_output_stride_batch = q_output.stride(0),
        .batch_size = batch_size,
        .num_q_heads = num_q_heads,
        .eps = eps,
    };
    const auto total_works = batch_size * num_q_heads;
    constexpr uint32_t kWorkPerWarp = (sizeof(DType) == 1) ? 2u : 1u;
    constexpr bool kVecConvert = (sizeof(DType) == 1);
    constexpr uint32_t kHeadsPerBlock = kFusedQNumWarps * kWorkPerWarp;
    // Shape dispatch (fp8 only): block-per-token wins once the grid fills the
    // SMs but loses to warp-per-work at small N (crossover total_works~4096).
    constexpr uint32_t kBPTMinWorks = 4096;
    const bool use_bpt = kVecConvert && (total_works >= kBPTMinWorks);

    uint32_t num_blocks;
    if (use_bpt) {
      num_blocks = batch_size * div_ceil(num_q_heads, kHeadsPerBlock);
    } else {
      num_blocks = div_ceil(total_works, kFusedQNumWarps * kWorkPerWarp);
    }
    const auto k = pos_dtype.is_type<int32_t>() ? (use_bpt ? kernel<int32_t, true> : kernel<int32_t, false>)
                                                : (use_bpt ? kernel<int64_t, true> : kernel<int64_t, false>);
    LaunchKernel(num_blocks, kFusedQBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(k, params);
  }
};

// ============================================================================
// K kernel: block-per-token rmsnorm (with kv_weight) + RoPE + FlashMLA store.
// ============================================================================

struct FusedKNormRopeFlashMLAParams {
  const void* __restrict__ kv;          // (B, kHeadDim) DType
  const void* __restrict__ kv_weight;   // (kHeadDim,) DType
  const float* __restrict__ freqs_cis;  // (max_pos, kRopeDim) fp32
  const void* __restrict__ positions;   // (B,) PosT
  const int32_t* __restrict__ out_loc;  // (B,) int32 -> cache slot id
  uint8_t* __restrict__ kvcache;        // (npages, kPageBytes) uint8
  // Row stride for `kv` in elements. Required because the upstream caller often
  // passes `qkv_a[..., q_lora_rank:]`, a non-contiguous slice whose stride[0]
  // equals `q_lora_rank + kHeadDim` rather than `kHeadDim`.
  int64_t kv_stride_batch;
  uint32_t batch_size;
  float eps;
};

template <typename DType, int64_t kHeadDim, int64_t kRopeDim, typename PosT, int32_t kPageBits, bool kUsePDL>
K_KERNEL void fused_k_norm_rope_flashmla(const __grid_constant__ FusedKNormRopeFlashMLAParams params) {
  using namespace device;

  constexpr int64_t kVecSize = 2;
  constexpr uint32_t kRopeWarp = kFusedKNumWarps - 1;
  constexpr int64_t kPageBytes = host::div_ceil(584ll << kPageBits, 576) * 576;
  static_assert(kHeadDim == kFusedKBlockSize * kVecSize);
  static_assert(kRopeDim == kWarpThreads * kVecSize);
  static_assert(kHeadDim - kRopeDim == kRopeWarp * kWarpThreads * kVecSize);
  using Storage = AlignedVector<DType, kVecSize>;
  using Float2 = AlignedVector<float, kVecSize>;

  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;
  const auto work_id = blockIdx.x;
  if (work_id >= params.batch_size) return;

  const auto input_ptr = static_cast<const DType*>(params.kv) + work_id * params.kv_stride_batch;
  const auto position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[work_id]);
  const auto out_loc = params.out_loc[work_id];
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  Float2 data, freq;

  // part 1: norm. Each thread owns one 2-elem pack (the `tx`-th).
  // Sum-of-squares is reduced block-wide via per-warp partials.
  {
    __shared__ float partial_sums[kFusedKNumWarps];

    Storage input_vec, weight_vec;
    input_vec.load(input_ptr, tx);
    weight_vec.load(params.kv_weight, tx);
    if (warp_id == kRopeWarp) freq.load(freqs_cis, lane_id);

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto x = cast<float>(input_vec[i]);
      sum_of_squares += x * x;
    }
    const auto warp_sum = warp::reduce_sum(sum_of_squares);
    if (lane_id == 0) partial_sums[warp_id] = warp_sum;
    __syncthreads();
    // Replicate the per-warp partial sums onto all lanes of one warp and
    // reduce. Every group of `kBlockItemNumWarps` lanes ends up with the
    // global sum.
    sum_of_squares = warp::reduce_sum<kFusedKNumWarps>(partial_sums[lane_id % kFusedKNumWarps]);
    const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto x = cast<float>(input_vec[i]);
      const auto w = cast<float>(weight_vec[i]);
      data[i] = x * norm_factor * w;
    }
  }

  // A negative out_loc marks a slot with no KV write target (e.g. the -1
  // sentinel from the full->SWA translation for out-of-window tokens or
  // padded rows); skip the row instead of writing out of bounds. Checked
  // here, not at the load, so the out_loc prefetch overlaps the norm above.
  if (out_loc < 0) return;

  const int32_t page = out_loc >> kPageBits;
  const int32_t offset = out_loc & ((1 << kPageBits) - 1);
  const auto page_ptr = params.kvcache + page * kPageBytes;
  const auto value_ptr = page_ptr + offset * 576;

  PDLTriggerSecondary<kUsePDL>();

  // part 2: rope on warp 7 (BF16 store), per-warp UE8M0 quant + store on warps 0..6.
  if (warp_id == kRopeWarp) {
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
    const auto x = data[0];
    const auto y = data[1];
    const auto abs_max = warp::reduce_max(fmaxf(fabs(x), fabs(y)));
    const auto scale_raw = fmaxf(1e-4f, abs_max) / math::FP8_E4M3_MAX;
    const auto scale_ue8m0 = cast_to_ue8m0(scale_raw);
    const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);
    const auto result = pack_fp8(x * inv_scale, y * inv_scale);
    const auto scale_ptr = page_ptr + (576 << kPageBits) + offset * 8;
    reinterpret_cast<fp8x2_e4m3_t*>(value_ptr)[tx] = result;
    if (lane_id == 0) static_cast<uint8_t*>(scale_ptr)[warp_id] = scale_ue8m0;
  }
}

template <typename DType, int64_t kHeadDim, int64_t kRopeDim, uint32_t kPageSize, bool kUsePDL>
struct FusedKNormRopeFlashMLAKernel {
  static constexpr int32_t kLogPageSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes = host::div_ceil(584 * kPageSize, 576) * 576;
  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");
  static_assert(1 << kLogPageSize == kPageSize);
  static_assert(kHeadDim == 512 && kRopeDim == 64, "FlashMLA layout requires (512, 64)");

  template <typename PosT>
  static constexpr auto kernel = fused_k_norm_rope_flashmla<DType, kHeadDim, kRopeDim, PosT, kLogPageSize, kUsePDL>;

  static void forward(
      const tvm::ffi::TensorView kv,
      const tvm::ffi::TensorView kv_weight,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView kvcache,
      float eps) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, kHeadDim})  //
        .with_strides({-1, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(kv);
    TensorMatcher({kHeadDim})  //
        .with_dtype<DType>()
        .with_device(device_)
        .verify(kv_weight);
    TensorMatcher({-1, kRopeDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs_cis);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({B})  //
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device_)
        .verify(positions);
    TensorMatcher({B})  //
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(out_loc);
    TensorMatcher({-1, -1})  //
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(kvcache);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    if (batch_size == 0) return;

    const auto params = FusedKNormRopeFlashMLAParams{
        .kv = kv.data_ptr(),
        .kv_weight = kv_weight.data_ptr(),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .positions = positions.data_ptr(),
        .out_loc = static_cast<const int32_t*>(out_loc.data_ptr()),
        .kvcache = static_cast<uint8_t*>(kvcache.data_ptr()),
        .kv_stride_batch = kv.stride(0),
        .batch_size = batch_size,
        .eps = eps,
    };
    const auto k_int32 = kernel<int32_t>;
    const auto k_int64 = kernel<int64_t>;
    const auto k = pos_dtype.is_type<int32_t>() ? k_int32 : k_int64;
    LaunchKernel(batch_size, kFusedKBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(k, params);
  }
};

// ============================================================================
// Indexer Q kernel: warp-per-(token, head) RoPE + Hadamard + fp8 act-quant.
// ============================================================================

struct FusedQIndexerRopeHadamardQuantParams {
  const void* __restrict__ q_input;  // (B, num_heads, 128) DType
  void* __restrict__ q_fp8;          // (B, num_heads, 128) fp8_e4m3
  // weights_out[b, h] = weight[b, h] * weight_scale * q_scale[b, h].
  // q_scale is computed internally and not exposed -- the only consumer of
  // it is `weights_out`.
  const void* __restrict__ weight;  // (B, num_heads) DType
  float* __restrict__ weights_out;  // (B, num_heads) fp32 (== (B, H, 1) flat)
  float weight_scale;               // scalar c4_indexer.weight_scale
  // Template-dependent layout:
  //   kRopeFirst=false: (max_pos, 64) fp32 interleaved [cos0, sin0, ...]
  //   kRopeFirst=true : (max_pos, 64) fp32 halves [cos..., sin...]
  const float* __restrict__ rope_cache;
  const void* __restrict__ positions;  // (B,) PosT
  // Row stride for `weight` (caller passes the non-contiguous wk slice directly).
  int64_t weight_stride_batch;
  uint32_t batch_size;
  uint32_t num_heads;
};

template <typename DType, typename PosT, bool kUsePDL, bool kRopeFirst = false, bool kHadamard = true>
Q_KERNEL void fused_q_indexer_rope_hadamard_quant(const __grid_constant__ FusedQIndexerRopeHadamardQuantParams params) {
  using namespace device;

  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;  // = 16
  static_assert(kHeadDim == kWarpThreads * kVecSize);
  static_assert(kRopeDim == kWarpThreads * 2);
  static_assert(kRopeSize <= kWarpThreads);

  using Storage = AlignedVector<DType, kVecSize>;
  using Float4 = AlignedVector<float, kVecSize>;
  using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;  // 4 fp8 / lane

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto work_id = blockIdx.x * kFusedQNumWarps + warp_id;
  // V4 ropes the trailing kRopeDim dims (kRopeFirst=false); V3.2 ropes the
  // leading kRopeDim dims (kRopeFirst=true). Select the owning lanes per layout.
  const bool is_rope_lane = kRopeFirst ? (lane_id < kRopeSize) : (lane_id >= kWarpThreads - kRopeSize);

  const uint32_t total_works = params.batch_size * params.num_heads;
  if (work_id >= total_works) return;

  const uint32_t batch_id = work_id / params.num_heads;
  const auto input_ptr = static_cast<const DType*>(params.q_input) + work_id * kHeadDim;
  const auto position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[batch_id]);
  const auto rope_cache = params.rope_cache + position * kRopeDim;

  // Lane 0 prefetches the weight scalar for this (token, head) work item.
  // Weight is (B, num_heads) DType; we need one scalar per warp -- offload
  // the load to lane 0 only. The multiply + store happens once the q_scale
  // is known (part 4).

  PDLWaitPrimary<kUsePDL>();
  Float4 data, freq;
  const uint32_t head_id = work_id - batch_id * params.num_heads;
  const auto weight_val =
      cast<float>(static_cast<const DType*>(params.weight)[batch_id * params.weight_stride_batch + head_id]);

  // part 1: load (no norm). Each lane owns a 4-elem pack.
  {
    Storage input_vec;
    input_vec.load(input_ptr, lane_id);
    if (is_rope_lane) {
      if constexpr (kRopeFirst) {
        freq = load_rope_first_cos_sin<kRopeDim>(rope_cache, lane_id);
      } else {
        freq.load(rope_cache, lane_id - (kWarpThreads - kRopeSize));
      }
    }
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      data[i] = cast<float>(input_vec[i]);
    }
  }

  // part 2: rope on rope lanes only (4 elems / lane = 2 (real, imag) pairs).
  if (is_rope_lane) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto y_real = data[2];
    const auto y_imag = data[3];
    const auto fxr = freq[0];
    const auto fxi = freq[1];
    const auto fyr = freq[2];
    const auto fyi = freq[3];
    data[0] = x_real * fxr - x_imag * fxi;
    data[1] = x_real * fxi + x_imag * fxr;
    data[2] = y_real * fyr - y_imag * fyi;
    data[3] = y_real * fyi + y_imag * fyr;
  }

  PDLTriggerSecondary<kUsePDL>();

  // part 3: 128-point Hadamard (2 local stages + 5 cross-lane shfl_xor stages).
  // Same recipe as `fused_norm_rope_indexer`; see comments there for the
  // butterfly invariants and the early-return safety argument. V3.2 omits the
  // rotation (kHadamard=false): it is logit-preserving (H orthonormal, applied
  // to both q and k), so dropping it only trades fp8 quant accuracy.
  if constexpr (kHadamard) {
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
        const float other = __shfl_xor_sync(0xFFFFFFFFu, data[i], mask, kWarpThreads);
        data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
      }
    }
    const float kHadamardScale = math::rsqrt(static_cast<float>(kHeadDim));
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      data[i] *= kHadamardScale;
  }

  {
    float local_max = math::abs(data[0]);
#pragma unroll
    for (int i = 1; i < kVecSize; ++i) {
      local_max = math::max(local_max, math::abs(data[i]));
    }
    const auto abs_max = warp::reduce_max(local_max);
    const auto scale = fmaxf(1e-4f, abs_max) / math::FP8_E4M3_MAX;
    const auto inv_scale = 1.0f / scale;
    OutStorage result;
    result[0] = pack_fp8(data[0] * inv_scale, data[1] * inv_scale);
    result[1] = pack_fp8(data[2] * inv_scale, data[3] * inv_scale);

    // q_fp8 row pointer: 128 fp8 / row = 32 OutStorage / row, one per lane.
    auto out_row = static_cast<uint8_t*>(params.q_fp8) + work_id * kHeadDim;
    result.store(out_row, lane_id);
    params.weights_out[work_id] = weight_val * params.weight_scale * scale;
  }
}

template <typename DType, bool kUsePDL, bool kRopeFirst = false, bool kHadamard = true>
struct FusedQIndexerRopeHadamardQuantKernel {
  template <typename PosT>
  static constexpr auto kernel = fused_q_indexer_rope_hadamard_quant<DType, PosT, kUsePDL, kRopeFirst, kHadamard>;

  static void forward(
      const tvm::ffi::TensorView q_input,
      const tvm::ffi::TensorView q_fp8,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView weights_out,
      double weight_scale,
      const tvm::ffi::TensorView rope_cache,
      const tvm::ffi::TensorView positions) {
    using namespace host;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kRopeDim = 64;

    auto B = SymbolicSize{"batch_size"};
    auto H = SymbolicSize{"num_heads"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    // Caller path is `wq_b(q_lora).view(-1, H, D)` -> contiguous; the kernel
    // assumes a flat `(B*H, kHeadDim)` layout for both q_input and q_fp8.
    // Pin the head/innermost strides; assert the batch stride below.
    TensorMatcher({B, H, kHeadDim})  //
        .with_strides({-1, kHeadDim, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(q_input);
    TensorMatcher({B, H, kHeadDim})  //
        .with_strides({-1, kHeadDim, 1})
        .with_dtype<fp8_e4m3_t>()
        .with_device(device_)
        .verify(q_fp8);
    TensorMatcher({B, H})  //
        .with_strides({-1, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(weight);
    TensorMatcher({B, H, 1})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(weights_out);
    TensorMatcher({-1, kRopeDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(rope_cache);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({B})  //
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device_)
        .verify(positions);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto num_heads = static_cast<uint32_t>(H.unwrap());
    if (batch_size == 0) return;

    // The kernel computes row pointers as `base + work_id * kHeadDim`, so
    // both inputs must be contiguous in (batch, head, elem) order.
    const int64_t expected_batch_stride = static_cast<int64_t>(num_heads) * kHeadDim;
    RuntimeCheck(
        q_input.stride(0) == expected_batch_stride,
        "q_input must be contiguous (B, H, kHeadDim); got stride[0]=",
        q_input.stride(0));
    RuntimeCheck(
        q_fp8.stride(0) == expected_batch_stride,
        "q_fp8 must be contiguous (B, H, kHeadDim); got stride[0]=",
        q_fp8.stride(0));

    const auto params = FusedQIndexerRopeHadamardQuantParams{
        .q_input = q_input.data_ptr(),
        .q_fp8 = q_fp8.data_ptr(),
        .weight = weight.data_ptr(),
        .weights_out = static_cast<float*>(weights_out.data_ptr()),
        .weight_scale = static_cast<float>(weight_scale),
        .rope_cache = static_cast<const float*>(rope_cache.data_ptr()),
        .positions = positions.data_ptr(),
        .weight_stride_batch = weight.stride(0),
        .batch_size = batch_size,
        .num_heads = num_heads,
    };
    const auto total_works = batch_size * num_heads;
    const auto num_blocks = div_ceil(total_works, kFusedQNumWarps);
    const auto k_int32 = kernel<int32_t>;
    const auto k_int64 = kernel<int64_t>;
    const auto k = pos_dtype.is_type<int32_t>() ? k_int32 : k_int64;
    LaunchKernel(num_blocks, kFusedQBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(k, params);
  }
};

struct FusedQIndexerRopeHadamardFp4QuantParams {
  const void* __restrict__ q_input;
  void* __restrict__ q_fp4;
  int32_t* __restrict__ q_sf;
  const void* __restrict__ weight;
  float* __restrict__ weights_out;
  float weight_scale;
  const float* __restrict__ freqs_cis;
  const void* __restrict__ positions;
  uint32_t batch_size;
  uint32_t num_heads;
};

template <typename DType, typename PosT, bool kUsePDL>
Q_KERNEL void
fused_q_indexer_rope_hadamard_fp4_quant(const __grid_constant__ FusedQIndexerRopeHadamardFp4QuantParams params) {
  using namespace device;

  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  static_assert(kHeadDim == kWarpThreads * kVecSize);
  static_assert(kRopeDim == kWarpThreads * 2);
  static_assert(kRopeSize <= kWarpThreads);

  using Storage = AlignedVector<DType, kVecSize>;
  using Float4 = AlignedVector<float, kVecSize>;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto work_id = blockIdx.x * kFusedQNumWarps + warp_id;
  const bool is_rope_lane = lane_id >= kWarpThreads - kRopeSize;

  const uint32_t total_works = params.batch_size * params.num_heads;
  if (work_id >= total_works) return;

  const uint32_t batch_id = work_id / params.num_heads;
  const auto input_ptr = static_cast<const DType*>(params.q_input) + work_id * kHeadDim;
  const auto position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[batch_id]);
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  Float4 data, freq;
  const auto weight_val = cast<float>(static_cast<const DType*>(params.weight)[work_id]);

  {
    Storage input_vec;
    input_vec.load(input_ptr, lane_id);
    if (is_rope_lane) freq.load(freqs_cis, lane_id - (kWarpThreads - kRopeSize));
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      data[i] = cast<float>(input_vec[i]);
    }
  }

  if (is_rope_lane) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto y_real = data[2];
    const auto y_imag = data[3];
    const auto fxr = freq[0];
    const auto fxi = freq[1];
    const auto fyr = freq[2];
    const auto fyi = freq[3];
    data[0] = x_real * fxr - x_imag * fxi;
    data[1] = x_real * fxi + x_imag * fxr;
    data[2] = y_real * fyr - y_imag * fyi;
    data[3] = y_real * fyi + y_imag * fyr;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      data[i] = cast<float>(cast<DType>(data[i]));
  }

  PDLTriggerSecondary<kUsePDL>();

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
        const float other = __shfl_xor_sync(0xFFFFFFFFu, data[i], mask, kWarpThreads);
        data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
      }
    }
    const float kHadamardScale = math::rsqrt(static_cast<float>(kHeadDim));
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      data[i] *= kHadamardScale;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      data[i] = cast<float>(cast<DType>(data[i]));
  }

  {
    float local_max = math::abs(data[0]);
#pragma unroll
    for (int i = 1; i < kVecSize; ++i) {
      local_max = math::max(local_max, math::abs(data[i]));
    }
    local_max = warp::reduce_max<8>(local_max);
    const auto scale_raw = fmaxf(1e-4f, local_max) / 6.0f;
    const auto scale_ue8m0 = static_cast<uint8_t>(cast_to_ue8m0(scale_raw));
    const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);
    const uint8_t packed0 = quant_fp4_e2m1(data[0] * inv_scale) | (quant_fp4_e2m1(data[1] * inv_scale) << 4);
    const uint8_t packed1 = quant_fp4_e2m1(data[2] * inv_scale) | (quant_fp4_e2m1(data[3] * inv_scale) << 4);
    const uint16_t packed = static_cast<uint16_t>(packed0) | (static_cast<uint16_t>(packed1) << 8);
    auto out_row = static_cast<uint8_t*>(params.q_fp4) + work_id * (kHeadDim / 2);
    reinterpret_cast<uint16_t*>(out_row)[lane_id] = packed;
    if ((lane_id & 7) == 0) {
      reinterpret_cast<uint8_t*>(params.q_sf + work_id)[lane_id >> 3] = scale_ue8m0;
    }
    params.weights_out[work_id] = weight_val * params.weight_scale;
  }
}

template <typename DType, bool kUsePDL>
struct FusedQIndexerRopeHadamardFp4QuantKernel {
  template <typename PosT>
  static constexpr auto kernel = fused_q_indexer_rope_hadamard_fp4_quant<DType, PosT, kUsePDL>;

  static void forward(
      const tvm::ffi::TensorView q_input,
      const tvm::ffi::TensorView q_fp4,
      const tvm::ffi::TensorView q_sf,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView weights_out,
      double weight_scale,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView positions) {
    using namespace host;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kRopeDim = 64;
    constexpr int64_t kFp4Dim = kHeadDim / 2;

    auto B = SymbolicSize{"batch_size"};
    auto H = SymbolicSize{"num_heads"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, H, kHeadDim})
        .with_strides({-1, kHeadDim, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(q_input);
    TensorMatcher({B, H, kFp4Dim})
        .with_strides({-1, kFp4Dim, 1})
        .with_dtype<int8_t>()
        .with_device(device_)
        .verify(q_fp4);
    TensorMatcher({B, H}).with_dtype<int32_t>().with_device(device_).verify(q_sf);
    TensorMatcher({B, H}).with_dtype<DType>().with_device(device_).verify(weight);
    TensorMatcher({B, H, 1}).with_dtype<float>().with_device(device_).verify(weights_out);
    TensorMatcher({-1, kRopeDim}).with_dtype<float>().with_device(device_).verify(freqs_cis);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({B}).with_dtype<int32_t, int64_t>(pos_dtype).with_device(device_).verify(positions);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto num_heads = static_cast<uint32_t>(H.unwrap());
    if (batch_size == 0) return;

    const int64_t expected_q_stride = static_cast<int64_t>(num_heads) * kHeadDim;
    const int64_t expected_fp4_stride = static_cast<int64_t>(num_heads) * kFp4Dim;
    RuntimeCheck(q_input.stride(0) == expected_q_stride, "q_input must be contiguous");
    RuntimeCheck(q_fp4.stride(0) == expected_fp4_stride, "q_fp4 must be contiguous");
    RuntimeCheck(q_sf.stride(0) == static_cast<int64_t>(num_heads) && q_sf.stride(1) == 1, "q_sf must be contiguous");

    const auto params = FusedQIndexerRopeHadamardFp4QuantParams{
        .q_input = q_input.data_ptr(),
        .q_fp4 = q_fp4.data_ptr(),
        .q_sf = static_cast<int32_t*>(q_sf.data_ptr()),
        .weight = weight.data_ptr(),
        .weights_out = static_cast<float*>(weights_out.data_ptr()),
        .weight_scale = static_cast<float>(weight_scale),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .positions = positions.data_ptr(),
        .batch_size = batch_size,
        .num_heads = num_heads,
    };
    const auto total_works = batch_size * num_heads;
    const auto num_blocks = div_ceil(total_works, kFusedQNumWarps);
    const auto k_int32 = kernel<int32_t>;
    const auto k_int64 = kernel<int64_t>;
    const auto k = pos_dtype.is_type<int32_t>() ? k_int32 : k_int64;
    LaunchKernel(num_blocks, kFusedQBlockSize, device_.unwrap()).enable_pdl(kUsePDL)(k, params);
  }
};

}  // namespace sglang
