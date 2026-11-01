// MoE router: scoring + (grouped) top-k + renormalize, one warp per token.
//
// Scoring is sigmoid or sqrtsoftplus plus a correction bias, or softmax of the
// logits plus an optional correction bias (weights keep the bias). Each lane
// owns E / 32 contiguous experts as order-preserving u32 keys, sorts them once,
// and every selection round only looks at the 32 lane heads: redux max of the
// heads, redux min of the lanes holding it (lowest expert id wins ties), pop.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/algorithm.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <bit>
#include <cstdint>

namespace sglang {

namespace device::route_warp::ptx {

SGL_DEVICE float ex2_approx_ftz(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

SGL_DEVICE float rcp_approx_ftz(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

SGL_DEVICE float sqrt_approx_ftz(float x) {
  float y;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// a * rcp(b), with b rescaled when it is subnormal or above 2^126.
SGL_DEVICE float div_full(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

}  // namespace device::route_warp::ptx

namespace moe::route_warp {

namespace ptx = device::route_warp::ptx;

struct Params {
  const void* __restrict__ logits;             // [T, E], row stride `logits_stride`
  const void* __restrict__ bias;               // [E] fp32 / bf16 / fp16, optional for softmax
  float* __restrict__ out_w;                   // [T, K + num_shared]
  int32_t* __restrict__ out_i;                 // [T, K + num_shared]
  int32_t* __restrict__ out_packed;            // optional [T, K + num_shared]: id << 16 | bf16(w)
  const int32_t* __restrict__ num_non_padded;  // optional device scalar
  int64_t logits_stride;
  uint32_t num_tokens;
  uint32_t num_shared;
  float scale;          // routed_scaling_factor
  float shared_weight;  // weight of the fused shared-expert slots
  float renorm_eps;     // > 0: sum + eps instead of the zero-sum guard
  bool renormalize;
  bool apply_scale;
  // kMuseSpark only
  float logit_scale;                       // router_output_multiplier, applied before the sigmoid
  const int32_t* __restrict__ expert_map;  // optional [E]: EP global -> local expert id
  // kTokenBias only: rows whose input id is `bias_alt_token_id` rank with `bias_alt`
  const int32_t* __restrict__ input_ids;  // int32 or the low words of int64 ids
  const void* __restrict__ bias_alt;      // [E], same dtype as `bias`
  int32_t input_ids_stride;               // in int32 words
  int32_t bias_alt_token_id;
};

// Larger value -> larger key. NaN is floored and -0 canonicalized before.
SGL_DEVICE uint32_t order_key(float x) {
  const auto u = __float_as_uint(x);
  return u ^ (static_cast<uint32_t>(static_cast<int32_t>(u) >> 31) | 0x80000000u);
}

SGL_DEVICE float key_to_float(uint32_t k) {
  return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

inline constexpr float kLog2e = 1.4426950408889634f;
inline constexpr float kLn2 = 0.6931471805599453f;

// 1 + e >= 1: rcp.approx equals div.full, except a result below 2^-126 flushes to 0.
SGL_DEVICE float sigmoid(float x) {
  return ptx::rcp_approx_ftz(1.0f + ptx::ex2_approx_ftz(x * -kLog2e));
}

SGL_DEVICE float softmax_exp(float x, float mx) {
  return ptx::ex2_approx_ftz((x - mx) * kLog2e);
}

// sqrt(max(x, 0) + log1p(exp(-|x|))), branch-free: w = exp(-|x| / 2) = 2^n * 2^f with
// a degree-7 polynomial for 2^f, z = w^2, log1p(z) = 2 atanh(z / (2 + z)) as an odd
// series. For x <= 0 the result is w * sqrt(log1p(z) / z), which stays normal down to
// x = -174 where z itself underflows.
SGL_DEVICE float sqrtsoftplus(float x) {
  constexpr float kHalfL2e = 0.5f * kLog2e;
  constexpr float kHalfL2eLo = 0.5f * 1.925963033500011e-08f;  // (log2(e) - kLog2e) / 2
  // Below -174 the result exp(x / 2) is subnormal; the clamp also keeps 2^n normal.
  const auto nx = fmaxf(-fabsf(x), -174.0f);
  const auto t_hi = nx * kHalfL2e;
  const auto t_lo = fmaf(nx, kHalfL2e, -t_hi) + nx * kHalfL2eLo;
  // n = rint(t_hi) via 1.5 * 2^23, whose low mantissa bits then hold n.
  const auto n_shift = t_hi + 12582912.0f;
  const auto n = n_shift - 12582912.0f;
  const auto f = (t_hi - n) + t_lo;
  // 2^f = e^(f ln2), Taylor to degree 7 (error < 5e-9 for |f| <= 1/2).
  auto e = 1.0f / 5040.0f;
  e = fmaf(e, f * kLn2, 1.0f / 720.0f);
  e = fmaf(e, f * kLn2, 1.0f / 120.0f);
  e = fmaf(e, f * kLn2, 1.0f / 24.0f);
  e = fmaf(e, f * kLn2, 1.0f / 6.0f);
  e = fmaf(e, f * kLn2, 0.5f);
  e = fmaf(e, f * kLn2, 1.0f);
  e = fmaf(e, f * kLn2, 1.0f);
  // w = e * 2^n by an exponent add: 1.5 * 2^23 << 23 wraps to 0, and w stays normal.
  const auto w = __int_as_float(__float_as_int(e) + (__float_as_int(n_shift) << 23));
  const auto z = w * w;
  // 1 / (2 + z): rcp.approx and one Newton step, ~0.5 ulp.
  const auto d = 2.0f + z;
  const auto inv0 = ptx::rcp_approx_ftz(d);
  const auto inv = fmaf(inv0, fmaf(-d, inv0, 1.0f), inv0);
  const auto s = z * inv;
  const auto s2 = s * s;
  auto poly = 1.0f / 13.0f;
  poly = fmaf(poly, s2, 1.0f / 11.0f);
  poly = fmaf(poly, s2, 1.0f / 9.0f);
  poly = fmaf(poly, s2, 1.0f / 7.0f);
  poly = fmaf(poly, s2, 1.0f / 5.0f);
  poly = fmaf(poly, s2, 1.0f / 3.0f);
  // r = log1p(z) / z = 2 / (2 + z) * (1 + s^2 poly).
  const auto two_inv = 2.0f * inv;
  const auto r = fmaf(two_inv * s2, poly, two_inv);
  // NaN takes the x > 0 side and stays NaN. Selecting before the MUFU keeps it
  // unconditional; a select after it makes nvcc branch around the MUFU.
  const bool pos = !(x <= 0.0f);
  return ptx::sqrt_approx_ftz(pos ? fmaf(z, r, x) : r) * (pos ? 1.0f : w);
}

enum class Scoring : uint32_t { SIGMOID = 0, SOFTMAX = 1, SQRTSOFTPLUS = 2 };

// Lane row pitch in words: an odd number of 16 B units when L is a multiple of 8.
constexpr uint32_t smem_pitch(uint32_t L) {
  return L % 8 == 0 ? L + 4 : L;
}

// Keys reserved below every real value (the smallest real key, -inf's, is 0x007fffff).
inline constexpr uint32_t kKeyTaken = 0;
inline constexpr uint32_t kKeyDropped = 1;

// Widest vector that divides a lane's L elements.
template <typename T, uint32_t L>
inline constexpr uint32_t kVecSize = [] {
  uint32_t v = 1;
  while (L % (v * 2) == 0 && v * 2 * sizeof(T) <= device::kMaxVecBytes) {
    v *= 2;
  }
  return v;
}();

template <
    uint32_t E,
    uint32_t K,
    uint32_t G,
    uint32_t TG,
    Scoring kScoring,
    typename T,
    typename TB,
    bool kBias,
    bool kUsePDL,
    bool kMuseSpark,
    bool kTokenBias,
    uint32_t kNumWarps>
__global__ __launch_bounds__(kNumWarps * 32) void route_warp_kernel(const __grid_constant__ Params p) {
  using namespace device;
  constexpr bool kSoftmax = kScoring == Scoring::SOFTMAX;
  constexpr uint32_t L = E / 32;          // experts per lane
  constexpr uint32_t M = std::min(L, K);  // pops one lane can take
  constexpr uint32_t V = kVecSize<T, L>;
  constexpr uint32_t VB = kVecSize<TB, L>;
  static_assert(E % 32 == 0 && L <= 16 && K >= 1 && K <= 32);
  static_assert(!kMuseSpark || kScoring == Scoring::SIGMOID);
  static_assert(kBias || kSoftmax);
  static_assert(!kTokenBias || (kBias && !kMuseSpark));

  // Lane rows padded so 128-bit accesses stay conflict-free.
  constexpr uint32_t kPitch = smem_pitch(L);
  __shared__ float s_val[kNumWarps][32 * kPitch];     // weights (sigmoid) or logits (softmax)
  __shared__ uint32_t s_key[kNumWarps][32 * kPitch];  // unsorted keys, for the id lookup

  const uint32_t lane_id = threadIdx.x % 32;
  const uint32_t warp_id = threadIdx.x / 32;
  const uint32_t token = blockIdx.x * kNumWarps + warp_id;
  const uint32_t base = lane_id * L;
  const uint32_t k_total = K + p.num_shared;

  if (token >= p.num_tokens) return;
  auto bias = static_cast<const TB*>(p.bias);
  if constexpr (kTokenBias) {
    // Before the wait: input ids are written before the layer stack, never by a PDL predecessor.
    if (p.input_ids[token * p.input_ids_stride] == p.bias_alt_token_id) {
      bias = static_cast<const TB*>(p.bias_alt);
    }
  }
  PDLWaitPrimary<kUsePDL>();

  const auto out_w = p.out_w + static_cast<int64_t>(token) * k_total;
  const auto out_i = p.out_i + static_cast<int64_t>(token) * k_total;
  if (p.num_non_padded != nullptr && token >= static_cast<uint32_t>(*p.num_non_padded)) {
    if (lane_id < k_total) {
      if constexpr (!kMuseSpark) out_w[lane_id] = 0.0f;
      out_i[lane_id] = -1;
      if (p.out_packed != nullptr) {
        p.out_packed[static_cast<int64_t>(token) * k_total + lane_id] = static_cast<int32_t>(0xffff0000u);
      }
    }
    return;
  }

  // ---- load and score ----
  const auto row = static_cast<const T*>(p.logits) + static_cast<int64_t>(token) * p.logits_stride + base;
  AlignedVector<T, V> xv[L / V];
#pragma unroll
  for (uint32_t i = 0; i < L / V; ++i) {
    xv[i].load(row, i);
  }
  AlignedVector<TB, VB> bv[L / VB];
  if constexpr (kBias) {
#pragma unroll
    for (uint32_t i = 0; i < L / VB; ++i) {
      bv[i].load(bias + base, i);
    }
  }

  float val[L];
  float biased[L];
  uint32_t key[L];
#pragma unroll
  for (uint32_t j = 0; j < L; ++j) {
    auto x = cast<float>(xv[j / V][j % V]);
    if constexpr (kMuseSpark) x *= p.logit_scale;
    if constexpr (kSoftmax) {
      // Softmax ranks and weighs the biased logit.
      val[j] = kBias ? x + cast<float>(bv[j / VB][j % VB]) : x;
      biased[j] = val[j];
    } else {
      val[j] = kScoring == Scoring::SIGMOID ? sigmoid(x) : sqrtsoftplus(x);
      biased[j] = val[j] + cast<float>(bv[j / VB][j % VB]);
    }
    // NaN ranks last (as DeepGEMM's mega-gate); + 0 turns -0 into +0.
    biased[j] = (biased[j] == biased[j]) ? biased[j] + 0.0f : -1e30f;
    key[j] = order_key(biased[j]);
    s_val[warp_id][lane_id * kPitch + j] = val[j];
  }

  // ---- group selection (noaux_tc): keep TG groups by the sum of their top two ----
  if constexpr (G > 1) {
    constexpr uint32_t kLanesPerGroup = 32 / G;
    static_assert(32 % G == 0 && E / G == kLanesPerGroup * L, "a group must span whole lanes");
    float t1 = -INFINITY, t2 = -INFINITY;
#pragma unroll
    for (uint32_t j = 0; j < L; ++j) {
      t2 = fmaxf(t2, fminf(t1, biased[j]));
      t1 = fmaxf(t1, biased[j]);
    }
#pragma unroll
    for (uint32_t off = 1; off < kLanesPerGroup; off *= 2) {
      const auto o1 = __shfl_xor_sync(0xffffffffu, t1, off);
      const auto o2 = __shfl_xor_sync(0xffffffffu, t2, off);
      t2 = fmaxf(fminf(t1, o1), fmaxf(t2, o2));
      t1 = fmaxf(t1, o1);
    }
    const auto score = t1 + t2;
    const uint32_t group = lane_id / kLanesPerGroup;
    uint32_t rank = 0;
#pragma unroll
    for (uint32_t g = 0; g < G; ++g) {
      const auto s = __shfl_sync(0xffffffffu, score, g * kLanesPerGroup);
      rank += (s > score || (s == score && g < group)) ? 1u : 0u;
    }
    if (rank >= TG) {
#pragma unroll
      for (uint32_t j = 0; j < L; ++j) {
        key[j] = kKeyDropped;
      }
    }
  }

  // ---- top-K over the lane heads ----
  // Lane k records round k's (key, lane, earlier pops of the same key from that
  // lane) and looks the expert id up after the loop.
#pragma unroll
  for (uint32_t j = 0; j < L; ++j) {
    s_key[warp_id][lane_id * kPitch + j] = key[j];
  }
  merge_sort(key);
  uint32_t my_key = 0, my_lane = 0, my_run = 0;
  uint32_t top_key = 0, prev_key = 0, prev_lane = 32, run = 0;
#pragma unroll
  for (uint32_t k = 0; k < K; ++k) {
    const auto m = warp::reduce_max(key[0]);
    const auto winner = warp::reduce_min(key[0] == m ? lane_id : 32u);
    const bool pop = lane_id == winner;
#pragma unroll
    for (uint32_t i = 0; i + 1 < M; ++i) {
      key[i] = pop ? key[i + 1] : key[i];
    }
    key[M - 1] = pop ? kKeyTaken : key[M - 1];
    run = (m == prev_key && winner == prev_lane) ? run + 1 : 0;
    prev_key = m;
    prev_lane = winner;
    my_key = (lane_id == k) ? m : my_key;
    my_lane = (lane_id == k) ? winner : my_lane;
    my_run = (lane_id == k) ? run : my_run;
    if (k == 0) top_key = m;
  }

  __syncwarp();
  uint32_t expert = 0;
  float w = 0.0f;
  if (lane_id < K) {
    uint32_t hits = 0;
#pragma unroll
    for (uint32_t j = 0; j < L; ++j) {
      hits |= (s_key[warp_id][my_lane * kPitch + j] == my_key ? 1u : 0u) << j;
    }
    // Equal keys of one lane pop in id order.
    for (uint32_t r = 0; r < my_run; ++r) {
      hits &= hits - 1;
    }
    const uint32_t j = __ffs(hits) - 1;
    expert = my_lane * L + j;
    w = s_val[warp_id][my_lane * kPitch + j];
  }

  // ---- weights ----
  if constexpr (kMuseSpark) {
    // MuseSpark: weights summed in selection order, `scale / (sum + eps)` applied once.
    float sum = __shfl_sync(0xffffffffu, w, 0);
#pragma unroll
    for (uint32_t k = 1; k < K; ++k) {
      sum += __shfl_sync(0xffffffffu, w, k);
    }
    w *= ptx::div_full(p.scale, sum + p.renorm_eps);
    if (lane_id < K && p.expert_map != nullptr) expert = static_cast<uint32_t>(p.expert_map[expert]);
  } else if constexpr (kSoftmax) {
    const auto mx = key_to_float(top_key);
    if (lane_id < K) w = softmax_exp(w, mx);
    if (!p.renormalize) {
      float part = 0.0f;
#pragma unroll
      for (uint32_t j = 0; j < L; ++j) {
        part += softmax_exp(val[j], mx);
      }
      w = ptx::div_full(w, warp::reduce_sum(part));
    }
  }
  if (!kMuseSpark && p.renormalize) {
    const auto sum = warp::reduce_sum<std::bit_ceil(K)>(w);
    const auto norm = p.renorm_eps > 0.0f ? sum + p.renorm_eps : (sum > 0.0f ? sum : 1.0f);
    w = ptx::div_full(w, norm);
  }
  if (!kMuseSpark && p.apply_scale) w *= p.scale;
  if (lane_id >= K) {
    expert = E + (lane_id - K);
    w = p.shared_weight;
  }
  if (lane_id < k_total) {
    out_w[lane_id] = w;
    out_i[lane_id] = static_cast<int32_t>(expert);
    if (p.out_packed != nullptr) {
      const auto w_bits = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(w)));
      p.out_packed[static_cast<int64_t>(token) * k_total + lane_id] = static_cast<int32_t>((expert << 16) | w_bits);
    }
  }
  // No launch_dependents: an explicit trigger let the expert GEMMs overlap the
  // shared-expert stream and run slower (Moonlight bs1 decode on H200).
}

template <
    uint32_t E,
    uint32_t K,
    uint32_t G,
    uint32_t TG,
    Scoring kScoring,
    typename T,
    typename TB,
    bool kBias,
    bool kUsePDL,
    bool kMuseSpark,
    bool kTokenBias>
struct RouteWarpKernel {
  static void
  run(const tvm::ffi::TensorView logits,
      const tvm::ffi::Optional<tvm::ffi::TensorView> bias,
      const tvm::ffi::TensorView out_w,
      const tvm::ffi::TensorView out_i,
      const tvm::ffi::Optional<tvm::ffi::TensorView> out_packed,
      const tvm::ffi::Optional<tvm::ffi::TensorView> num_non_padded,
      const int32_t num_shared,
      const float scale,
      const float renorm_eps,
      const bool renormalize,
      const bool apply_scale,
      const float logit_scale,
      const tvm::ffi::Optional<tvm::ffi::TensorView> expert_map,
      const tvm::ffi::Optional<tvm::ffi::TensorView> input_ids,
      const tvm::ffi::Optional<tvm::ffi::TensorView> bias_alt,
      const int32_t bias_alt_token_id) {
    using namespace host;
    constexpr uint32_t L = E / 32;
    const auto k_total = static_cast<int64_t>(K) + num_shared;
    auto N_ = SymbolicSize{"num_tokens"};
    auto device_sym = SymbolicDevice{};
    device_sym.set_options<kDLCUDA>();
    TensorMatcher({N_, E})  //
        .with_strides({-1, 1})
        .with_dtype<T>()
        .with_device(device_sym)
        .ensure_alignment(kVecSize<T, L> * sizeof(T))
        .verify(logits);
    TensorMatcher({N_, k_total})  //
        .with_dtype<float>()
        .with_device(device_sym)
        .verify(out_w);
    TensorMatcher({N_, k_total})  //
        .with_dtype<int32_t>()
        .with_device(device_sym)
        .verify(out_i);
    CHECK_HOST(k_total <= 32) << "route_warp: topk + shared slots must fit in a warp";
    if (out_packed.has_value()) {
      TensorMatcher({N_, k_total})  //
          .with_dtype<int32_t>()
          .with_device(device_sym)
          .verify(out_packed.value());
    }

    auto p = Params{};
    if constexpr (kBias) {
      CHECK_HOST(bias.has_value()) << "route_warp: the biased instantiation needs a bias";
      TensorMatcher({E})  //
          .with_dtype<TB>()
          .with_device(device_sym)
          .ensure_alignment(kVecSize<TB, L> * sizeof(TB))
          .verify(bias.value());
      p.bias = bias.value().data_ptr();
    } else {
      CHECK_HOST(!bias.has_value()) << "route_warp: a bias needs the biased instantiation";
    }
    if (num_non_padded.has_value()) {
      // A 0-d or [1] device scalar, so no shape matcher.
      const auto& n = num_non_padded.value();
      CHECK_HOST(is_type<int32_t>(n.dtype()) && n.numel() == 1 && n.device().device_type == kDLCUDA);
      p.num_non_padded = static_cast<const int32_t*>(num_non_padded.value().data_ptr());
    }
    p.logits = logits.data_ptr();
    p.out_w = static_cast<float*>(out_w.data_ptr());
    p.out_i = static_cast<int32_t*>(out_i.data_ptr());
    if (out_packed.has_value()) p.out_packed = static_cast<int32_t*>(out_packed.value().data_ptr());
    p.logits_stride = logits.stride(0);
    p.num_tokens = static_cast<uint32_t>(logits.size(0));
    p.num_shared = static_cast<uint32_t>(num_shared);
    p.scale = scale;
    p.shared_weight = apply_scale ? 1.0f : 1.0f / scale;
    p.renorm_eps = renorm_eps;
    p.renormalize = renormalize;
    p.apply_scale = apply_scale;
    if constexpr (kMuseSpark) {
      CHECK_HOST(num_shared == 0) << "route_warp: MuseSpark routing has no fused shared experts";
      p.logit_scale = logit_scale;
      if (expert_map.has_value()) {
        TensorMatcher({E})  //
            .with_dtype<int32_t>()
            .with_device(device_sym)
            .verify(expert_map.value());
        p.expert_map = static_cast<const int32_t*>(expert_map.value().data_ptr());
      }
    } else {
      CHECK_HOST(!expert_map.has_value()) << "route_warp: the expert map is MuseSpark only";
    }
    if constexpr (kTokenBias) {
      CHECK_HOST(input_ids.has_value() && bias_alt.has_value())
          << "route_warp: token bias needs input_ids and bias_alt";
      TensorMatcher({N_})  //
          .with_strides({-1})
          .with_dtype<int32_t, int64_t>()
          .with_device(device_sym)
          .verify(input_ids.value());
      TensorMatcher({E})  //
          .with_dtype<TB>()
          .with_device(device_sym)
          .ensure_alignment(kVecSize<TB, L> * sizeof(TB))
          .verify(bias_alt.value());
      // Token ids are non-negative and below 2^31: an int64 id is its low (little-endian) word.
      const auto words = is_type<int64_t>(input_ids.value().dtype()) ? 2 : 1;
      p.input_ids = static_cast<const int32_t*>(input_ids.value().data_ptr());
      p.input_ids_stride = static_cast<int32_t>(input_ids.value().stride(0) * words);
      p.bias_alt = bias_alt.value().data_ptr();
      p.bias_alt_token_id = bias_alt_token_id;
    } else {
      CHECK_HOST(!input_ids.has_value()) << "route_warp: input_ids need the token-bias instantiation";
    }
    if (p.num_tokens == 0) return;
    const auto device = device_sym.unwrap();
    const auto kernel_1 = route_warp_kernel<E, K, G, TG, kScoring, T, TB, kBias, kUsePDL, kMuseSpark, kTokenBias, 1>;
    const auto kernel_4 = route_warp_kernel<E, K, G, TG, kScoring, T, TB, kBias, kUsePDL, kMuseSpark, kTokenBias, 4>;
    // One warp per CTA (64 registers under a 32-thread bound, vs 40) while all tokens
    // fit in one wave; past that, four warps per CTA (B200 sweep, 256..16384 tokens).
    static const uint32_t s_single_warp_occupancy = runtime::get_blocks_per_sm(kernel_1, 32);
    if (p.num_tokens <= s_single_warp_occupancy * runtime::get_sm_count(device.device_id)) {
      const auto num_blocks = p.num_tokens;
      LaunchKernel(num_blocks, 32, device).enable_pdl(kUsePDL)(kernel_1, p);
    } else {
      const auto num_blocks = div_ceil(p.num_tokens, 4u);
      LaunchKernel(num_blocks, 128, device).enable_pdl(kUsePDL)(kernel_4, p);
    }
  }
};

}  // namespace moe::route_warp

// The JIT module names and wrappers spell the scoring as a bare enumerator.
using enum moe::route_warp::Scoring;

}  // namespace sglang
