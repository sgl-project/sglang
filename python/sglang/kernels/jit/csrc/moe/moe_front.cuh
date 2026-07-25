// K3 MoE front: the routing top-k, and the merged gate + down-proj epilogue.
//
// `KimiK3MoE._forward_unfused` -- the path every EP-a2a / WideEP deployment
// takes -- runs three ops over the same `hidden_states [T, 7168]`:
//
//     router_logits = gate(hidden_states)                    // [896, 7168]  12.85 MB
//     topk_output   = topk(hidden_states, router_logits)
//     routed_input  = routed_expert_down_proj(hidden_states)  // [3584, 7168] 51.4 MB
//
// Two entry points, both consuming fp32 logits:
//
//   route_radix_fp32     radix-select top-k over [M, 896] fp32 logits.
//                        route_radix.cuh is bf16-only, so the fp32 logits the
//                        cuBLAS gate GEMV produces used to fall back to the
//                        Triton router -- 7.04 us instead of 2.4 us per layer at
//                        T=1 on a GB300, across 92 MoE layers.
//   fused_front_epilogue the merged-front epilogue. The gate and the down-proj
//                        read the same activations, so their weights are merged
//                        and one cuBLAS GEMM emits [M, 896 + 3584]. It can only
//                        emit one dtype: bf16 would round the router logits and
//                        move the selected expert set on a few percent of rows,
//                        so it emits fp32 and this kernel folds the remaining
//                        cast into the top-k launch. The front then costs two
//                        kernels with routing unchanged, and routed_input comes
//                        out dense (no strided-slice copy).
//
// Measured in-graph on a GB300, us per MoE layer, against today's path (gate
// GEMM + Triton router + down GEMM), with the copy a dense-input runner needs:
//
//   T          1     16    256   1024   2048   4096   16384
//   baseline  22.3   21.9   31.2   61.4  109.5  185.1   735.8
//   merged    13.1   12.0   17.6   47.2   89.7  181.8   702.8
//   unfused+  17.3   17.5   24.2   51.3   92.9  154.5   621.8
//
// The merge stops paying once the GEMM is compute-bound (the fp32 output traffic
// is no longer free), so callers switch to the router-only entry above ~1024
// tokens; see kernels/ops/moe/moe_front.py.
//
// Routing semantics match route_radix / the Triton router exactly: sigmoid
// scoring, correction bias for ranking only while the emitted weight is
// bias-free, lowest expert id wins ties, NaN floored, optional renormalize and
// routed-scaling, winners in expert-id order.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

inline constexpr uint32_t kFGTNumExperts = 896;
inline constexpr uint32_t kFGTTopK = 16;
// 7 warps: 896 experts / 224 threads = 4 experts per thread in the epilogue, and
// one expert per warp per pass in phase 1.  Keeping the block at route_radix's
// shape lets the epilogue reuse its radix-select verbatim.
inline constexpr uint32_t kFGTBlockSize = 224;
inline constexpr uint32_t kFGTWarps = kFGTBlockSize / 32;
// Largest batch the fused kernel serves; above this the gate GEMM is already
// near the memory-read floor and cuBLAS + route_radix_fp32 wins.
inline constexpr uint32_t kFGTMaxTokens = 16;

struct FusedGateTopKTrait {
  static constexpr uint32_t kNumExperts = kFGTNumExperts;
  static constexpr uint32_t kTopK = kFGTTopK;
  static constexpr uint32_t kVecSize = 4;  // epilogue: experts per thread

  static constexpr uint32_t kRadixBits = 8;
  static constexpr uint32_t kRadixSize = 1 << kRadixBits;
  static constexpr uint32_t kRadixRounds = 32 / kRadixBits;
  static constexpr uint32_t kBlockSize = kFGTBlockSize;
  static constexpr uint32_t kNumWarps = kFGTWarps;

  struct alignas(16) MatchBin {
    uint32_t bin;
    uint32_t above_count;
    uint32_t equal_count;
  };
  struct Smem {
    uint32_t warp_sum[3][kNumWarps];
    MatchBin match[kRadixRounds];
    uint32_t histogram[kRadixSize];
    int32_t wid[kTopK];
    uint32_t wkey[kTopK];
    fp32_t wact[kTopK];
  };
};

struct MoEFrontParams {
  const fp32_t* __restrict__ bias;    // [E] fp32 correction bias
  const fp32_t* __restrict__ logits;  // [M, logits_stride] fp32, gate slice first
  fp32_t* __restrict__ out_w;         // [M, topk] fp32
  int32_t* __restrict__ out_i;        // [M, topk] int32
  int M;
  int logits_stride;  // E for the router-only entry, E + latent for the front
  long long out_w_stride;
  long long out_i_stride;
  float routed_scaling_factor;
  int renormalize;
  int apply_scale;
  // Merged-front entry only: the [M, latent] bf16 routed_input to emit.
  bf16_t* __restrict__ routed_out;
  int latent;
  long long routed_stride;
};

inline constexpr float kFGTNanFloor = -1e30f;

SGL_DEVICE uint32_t fgt_biased_to_key(float biased) {
  if (biased == 0.0f) biased = 0.0f;
  uint32_t u = __float_as_uint(biased);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Must stay instruction-identical to route_radix's sigmoid_match so both
// kernels rank and weight identically.
SGL_DEVICE float fgt_sigmoid_match(float x) {
  return __fdividef(1.0f, 1.0f + __expf(-x));
}

SGL_DEVICE float fgt_nan_floor(float x) {
  return (x == x) ? x : kFGTNanFloor;
}

SGL_DEVICE void fgt_bar_sync(uint32_t id, uint32_t num_threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(num_threads) : "memory");
}

SGL_DEVICE uint32_t fgt_warp_inclusive_sum(uint32_t lane_id, uint32_t val) {
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
    if (lane_id >= offset) val += n;
  }
  return val;
}

SGL_DEVICE uint32_t
fgt_block_exclusive_sum(uint32_t cnt, uint32_t lane_id, uint32_t warp_id, uint32_t* smem_warp_sum) {
  const uint32_t inc = fgt_warp_inclusive_sum(lane_id, cnt);
  if (lane_id == 31) smem_warp_sum[warp_id] = inc;
  __syncthreads();
  const auto base = __reduce_add_sync(0xFFFFFFFF, lane_id < warp_id ? smem_warp_sum[lane_id] : 0u);
  return base + inc - cnt;
}

/// Radix-select top-k over one token's fp32 logits.  Lifted from
/// route_radix.cuh; the only change is the input dtype (fp32 in place of bf16).
template <bool kUsePDL>
SGL_DEVICE void fgt_select_topk(
    const MoEFrontParams& params,
    typename FusedGateTopKTrait::Smem& smem,
    int m,
    uint32_t tx,
    uint32_t warp_id,
    uint32_t lane_id) {
  using T = FusedGateTopKTrait;
  constexpr uint32_t kVecSize = T::kVecSize;
  constexpr uint32_t kRadixLanes = T::kRadixSize / 2;
  enum { BAR_SUM = 1 };

  uint32_t keys[kVecSize];
  float act[kVecSize];
  {
    // 4 experts per thread as two fp32x2 (16B) loads, matching route_radix.
    device::AlignedVector<fp32x2_t, kVecSize / 2> bias_vec;
    bias_vec.load(params.bias, tx);
    device::AlignedVector<fp32x2_t, kVecSize / 2> lv;
    lv.load(params.logits + (long long)m * params.logits_stride, tx);
    const float logit[kVecSize] = {lv[0].x, lv[0].y, lv[1].x, lv[1].y};
#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      const float sx = fgt_sigmoid_match(logit[2 * i + 0]);
      const float sy = fgt_sigmoid_match(logit[2 * i + 1]);
      act[2 * i + 0] = sx;
      act[2 * i + 1] = sy;
      keys[2 * i + 0] = fgt_biased_to_key(fgt_nan_floor(sx + bias_vec[i].x));
      keys[2 * i + 1] = fgt_biased_to_key(fgt_nan_floor(sy + bias_vec[i].y));
    }
  }

  bool active[kVecSize];
#pragma unroll
  for (uint32_t i = 0; i < kVecSize; ++i) {
    active[i] = true;
  }

  uint32_t total_active = T::kNumExperts;
  uint32_t topk = T::kTopK;
  uint32_t threshold = 0;
  uint32_t examined_mask = 0;
  bool take_all_equals = false;

  {
    device::AlignedVector<uint32_t, 2> zero;
    zero.fill(0);
    if (tx < kRadixLanes) zero.store(smem.histogram, tx);

#pragma unroll
    for (uint32_t round = 0; round < T::kRadixRounds; ++round) {
      __syncthreads();
      const uint32_t shift = 24 - round * 8;
      uint32_t bin[kVecSize];
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) {
        bin[i] = (keys[i] >> shift) & 0xff;
      }
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) {
        if (active[i]) atomicAdd(&smem.histogram[bin[i]], 1);
      }
      __syncthreads();

      if (tx < kRadixLanes) {
        device::AlignedVector<uint32_t, 2> hist;
        hist.load(smem.histogram, tx);
        const auto local_val = hist[0] + hist[1];
        const auto warp_inc = fgt_warp_inclusive_sum(lane_id, local_val);
        if (lane_id == 31) smem.warp_sum[0][warp_id] = warp_inc;
        fgt_bar_sync(BAR_SUM, kRadixLanes);
        const auto inter =
            __reduce_add_sync(0xFFFFFFFF, lane_id < warp_id ? smem.warp_sum[0][lane_id] : 0u);
        const auto prefix = inter + warp_inc;
        const auto above_r = total_active - prefix;
        const auto above_m = above_r + hist[1];
        const auto above_l = above_m + hist[0];
        if (above_r < topk && above_m >= topk) {
          smem.match[round] = {tx * 2 + 1, above_r, hist[1]};
        } else if (above_m < topk && above_l >= topk) {
          smem.match[round] = {tx * 2 + 0, above_m, hist[0]};
        }
      }
      __syncthreads();

      const auto [threshold_bin, above_count, equal_count] = smem.match[round];
      threshold |= threshold_bin << shift;
      examined_mask |= 0xffu << shift;
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) {
        active[i] &= (bin[i] == threshold_bin);
      }
      total_active = equal_count;
      topk -= above_count;
      if (topk == equal_count) {
        take_all_equals = true;
        break;
      }
      if (round + 1 < T::kRadixRounds && tx < kRadixLanes) zero.store(smem.histogram, tx);
    }
  }

  bool selected[kVecSize];
  if (take_all_equals) {
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      selected[i] = active[i] || (keys[i] & examined_mask) > threshold;
    }
  } else {
    uint32_t cnt = 0;
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      cnt += active[i] ? 1 : 0;
    }
    uint32_t rank = fgt_block_exclusive_sum(cnt, lane_id, warp_id, smem.warp_sum[1]);
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      const bool eq_win = active[i] && rank < topk;
      if (active[i]) ++rank;
      selected[i] = eq_win || (keys[i] & examined_mask) > threshold;
    }
  }

  uint32_t selected_cnt = 0;
#pragma unroll
  for (uint32_t i = 0; i < kVecSize; ++i) {
    selected_cnt += selected[i] ? 1 : 0;
  }
  uint32_t slot = fgt_block_exclusive_sum(selected_cnt, lane_id, warp_id, smem.warp_sum[2]);
#pragma unroll
  for (uint32_t i = 0; i < kVecSize; ++i) {
    if (selected[i] && slot < T::kTopK) {
      smem.wid[slot] = (int32_t)(tx * kVecSize + i);
      smem.wact[slot] = act[i];
      ++slot;
    }
  }
  __syncthreads();

  static_assert(T::kTopK <= 32);
  if (tx < T::kTopK) {
    auto wv = smem.wact[tx];
    const auto id = smem.wid[tx];
    float sum = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < T::kTopK; ++i) {
      sum += smem.wact[i];
    }
    const auto norm = (sum > 0.0f) ? sum : 1.0f;
    if (params.renormalize) wv = wv / norm;
    if (params.apply_scale) wv = wv * params.routed_scaling_factor;
    params.out_w[m * params.out_w_stride + tx] = wv;
    params.out_i[m * params.out_i_stride + tx] = id;
  }
  __syncthreads();  // smem reuse across the token loop
}

/// Router-only entry: radix-select top-k over **fp32** logits.
///
/// route_radix.cuh is bf16-only, so K3's unfused-front path (which produces fp32
/// logits from the cuBLAS gate GEMM) silently falls back to the Triton router --
/// 7.04 us instead of 2.4 us per layer at T=1 on a GB300.  This is the same
/// radix-select, reading the fp32 logits in place.
template <bool kUsePDL>
__global__ __launch_bounds__(kFGTBlockSize)  //
    void route_radix_fp32_kernel(const __grid_constant__ MoEFrontParams params) {
  using namespace device;
  using T = FusedGateTopKTrait;
  __shared__ typename T::Smem smem;
  const uint32_t tx = threadIdx.x;
  // grid.x == M exactly; no row guard (an early return would deadlock the
  // block-wide barriers inside the select).
  PDLWaitPrimary<kUsePDL>();
  fgt_select_topk<kUsePDL>(params, smem, (int)blockIdx.x, tx, tx / 32, tx % 32);
}

/// Merged-front epilogue: consumes one fp32 GEMM output [M, E + latent] and
/// emits both routing results and the bf16 routed_input.
///
/// The K3 MoE front runs two GEMMs over the same `hidden_states`: the router
/// gate ([896, 7168], 12.85 MB) and routed_expert_down_proj ([3584, 7168],
/// 51.4 MB).  Merging their weights lets cuBLAS read the activations once and
/// keeps the tensor-core GEMM (which beats a hand-written cudacore GEMV by ~3x
/// at this size), but the merged GEMM can only emit one dtype.  Emitting bf16
/// would round the router logits and change routing on a few percent of rows;
/// emitting fp32 keeps routing exact and leaves a cast to do -- which this
/// kernel folds into the top-k launch, so the front still costs two kernels.
///
/// grid.x == M, one CTA per token, matching route_radix's shape.
template <bool kUsePDL>
__global__ __launch_bounds__(kFGTBlockSize)  //
    void fused_front_epilogue_kernel(const __grid_constant__ MoEFrontParams params) {
  using namespace device;
  using T = FusedGateTopKTrait;
  __shared__ typename T::Smem smem;
  const uint32_t tx = threadIdx.x;
  const int m = (int)blockIdx.x;

  PDLWaitPrimary<kUsePDL>();

  // Cast the latent slice: [E, E + latent) fp32 -> [0, latent) bf16.  Done
  // before the select so these loads are in flight during the radix rounds.
  {
    const fp32_t* src = params.logits + (long long)m * params.logits_stride + T::kNumExperts;
    bf16_t* dst = params.routed_out + (long long)m * params.routed_stride;
    for (int i = (int)tx * 4; i < params.latent; i += (int)kFGTBlockSize * 4) {
      AlignedVector<fp32x2_t, 2> v;
      v.load(src, i / 4);
      AlignedVector<bf16_t, 4> o;
      o[0] = cast<bf16_t>(v[0].x);
      o[1] = cast<bf16_t>(v[0].y);
      o[2] = cast<bf16_t>(v[1].x);
      o[3] = cast<bf16_t>(v[1].y);
      o.store(dst, i / 4);
    }
  }

  fgt_select_topk<kUsePDL>(params, smem, m, tx, tx / 32, tx % 32);
}

}  // namespace sglang

template <bool kUsePDL>
struct FusedFrontEpilogueKernel {
  static void
  run(const tvm::ffi::TensorView merged,   // [M, E + latent] fp32, row-dense
      const tvm::ffi::TensorView bias,     // [E] fp32
      const tvm::ffi::TensorView out_w,    // [M, topk] fp32
      const tvm::ffi::TensorView out_i,    // [M, topk] int32
      const tvm::ffi::TensorView routed,   // [M, latent] bf16
      int64_t topk,
      double routed_scaling_factor,
      bool renormalize,
      bool apply_scale) {
    using namespace host;

    auto M_ = SymbolicSize{"num_tokens"};
    auto W_ = SymbolicSize{"merged_width"};
    auto L_ = SymbolicSize{"latent"};
    auto E_ = SymbolicSize{"num_experts"};
    auto K_ = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M_, W_}).with_dtype<fp32_t>().with_device(device).with_strides({-1, 1}).verify(merged);
    TensorMatcher({E_}).with_dtype<fp32_t>().with_device(device).verify(bias);
    TensorMatcher({M_, K_}).with_dtype<fp32_t>().with_device(device).verify(out_w);
    TensorMatcher({M_, K_}).with_dtype<int32_t>().with_device(device).verify(out_i);
    TensorMatcher({M_, L_}).with_dtype<bf16_t>().with_device(device).with_strides({-1, 1}).verify(routed);

    const auto M = static_cast<int>(M_.unwrap());
    const auto latent = static_cast<int>(L_.unwrap());
    RuntimeCheck(
        E_.unwrap() == sglang::kFGTNumExperts && K_.unwrap() == sglang::kFGTTopK &&
            topk == sglang::kFGTTopK,
        "fused_front_epilogue is specialized for E=896, topk=16");
    RuntimeCheck(
        static_cast<int>(W_.unwrap()) == static_cast<int>(sglang::kFGTNumExperts) + latent,
        "fused_front_epilogue: merged width must be num_experts + latent");
    // 16B vectorized reads of the fp32 rows and 8B writes of the bf16 rows.
    RuntimeCheck(latent % 4 == 0, "fused_front_epilogue: latent must be a multiple of 4");
    RuntimeCheck(
        merged.stride(0) % 4 == 0 && routed.stride(0) % 4 == 0,
        "fused_front_epilogue: row strides must be a multiple of 4");
    if (M == 0) return;

    auto params = sglang::MoEFrontParams{};
    params.bias = static_cast<const fp32_t*>(bias.data_ptr());
    params.logits = static_cast<fp32_t*>(merged.data_ptr());
    params.out_w = static_cast<fp32_t*>(out_w.data_ptr());
    params.out_i = static_cast<int32_t*>(out_i.data_ptr());
    params.M = M;
    params.out_w_stride = static_cast<long long>(out_w.stride(0));
    params.out_i_stride = static_cast<long long>(out_i.stride(0));
    params.routed_scaling_factor = static_cast<float>(routed_scaling_factor);
    params.renormalize = renormalize ? 1 : 0;
    params.apply_scale = apply_scale ? 1 : 0;
    params.logits_stride = static_cast<int>(merged.stride(0));
    params.routed_out = static_cast<bf16_t*>(routed.data_ptr());
    params.latent = latent;
    params.routed_stride = static_cast<long long>(routed.stride(0));

    LaunchKernel(M, sglang::kFGTBlockSize, device.unwrap())
        .enable_pdl(kUsePDL)(sglang::fused_front_epilogue_kernel<kUsePDL>, params);
  }
};

template <bool kUsePDL>
struct RouteRadixFp32Kernel {
  static void
  run(const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView out_w,
      const tvm::ffi::TensorView out_i,
      int64_t topk,
      double routed_scaling_factor,
      bool renormalize,
      bool apply_scale) {
    using namespace host;

    auto M_ = SymbolicSize{"num_tokens"};
    auto E_ = SymbolicSize{"num_experts"};
    auto K_ = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M_, E_}).with_dtype<fp32_t>().with_device(device).with_strides({-1, 1}).verify(logits);
    TensorMatcher({E_}).with_dtype<fp32_t>().with_device(device).verify(bias);
    TensorMatcher({M_, K_}).with_dtype<fp32_t>().with_device(device).verify(out_w);
    TensorMatcher({M_, K_}).with_dtype<int32_t>().with_device(device).verify(out_i);

    RuntimeCheck(
        E_.unwrap() == sglang::kFGTNumExperts && K_.unwrap() == sglang::kFGTTopK &&
            topk == sglang::kFGTTopK,
        "route_radix_fp32 is specialized for E=896, topk=16");
    // fgt_select_topk indexes rows by num_experts, so the logits must be densely
    // packed (which also keeps the 16B vectorized row loads aligned).
    RuntimeCheck(
        logits.stride(0) == static_cast<int64_t>(sglang::kFGTNumExperts),
        "route_radix_fp32: logits must be row-dense [M, 896]");

    const auto M = static_cast<int>(M_.unwrap());
    if (M == 0) return;

    auto params = sglang::MoEFrontParams{};
    params.bias = static_cast<const fp32_t*>(bias.data_ptr());
    params.logits = static_cast<fp32_t*>(logits.data_ptr());
    params.out_w = static_cast<fp32_t*>(out_w.data_ptr());
    params.out_i = static_cast<int32_t*>(out_i.data_ptr());
    params.M = M;
    params.out_w_stride = static_cast<long long>(out_w.stride(0));
    params.out_i_stride = static_cast<long long>(out_i.stride(0));
    params.routed_scaling_factor = static_cast<float>(routed_scaling_factor);
    params.renormalize = renormalize ? 1 : 0;
    params.apply_scale = apply_scale ? 1 : 0;
    params.logits_stride = static_cast<int>(sglang::kFGTNumExperts);

    LaunchKernel(M, sglang::kFGTBlockSize, device.unwrap())
        .enable_pdl(kUsePDL)(sglang::route_radix_fp32_kernel<kUsePDL>, params);
  }
};
