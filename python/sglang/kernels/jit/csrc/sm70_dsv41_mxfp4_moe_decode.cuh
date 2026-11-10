// SPDX-License-Identifier: Apache-2.0
// SM70 decode GEMV for DeepSeek-V4.1-Flash MXFP4 MoE (M<=4).
// Reads marlin_v100 packed weights [E, K/16, N*2] + logical UE8M0 [E, K/32, N].
// Do not reuse sm70_nvfp4_moe_decode (NVFP4 E4M3 g16 + FP32 global_scale).
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp16.h>

namespace sglang::sm70_dsv41 {

inline constexpr int32_t kHidden = 5120;
inline constexpr int32_t kIntermediate = 2304;
inline constexpr int32_t kGateUp = 2 * kIntermediate;  // 4608
inline constexpr int32_t kTopK = 6;
inline constexpr int32_t kGroupSize = 32;
inline constexpr int32_t kMaxM = 4;
inline constexpr int32_t kSplitKGate = 8;
inline constexpr int32_t kSplitKDown = 8;
inline constexpr int32_t kThreads = 64;
inline constexpr int32_t kPackedMacroN = 256;
inline constexpr int32_t kQuantTileK = 16;
inline constexpr int32_t kQuantTileN = 64;
inline constexpr int32_t kGroupTiles = kPackedMacroN / kQuantTileN;  // 4
inline constexpr int32_t kW13KTiles = kHidden / kQuantTileK;         // 320
inline constexpr int32_t kW2KTiles = kIntermediate / kQuantTileK;    // 144
inline constexpr int32_t kW13Groups = kHidden / kGroupSize;          // 160
inline constexpr int32_t kW2Groups = kIntermediate / kGroupSize;     // 72

/**
 * \brief Marlin SM70 U4 packed-macro-N qword index (PackedMacroN=256).
 *
 * Same formula as marlin_v100
 * ``u4_packed_macro_n_qweight_offset_from_logical<256>``.
 */
SGL_DEVICE int32_t qweight_offset(int32_t size_n, int32_t logical_k, int32_t logical_n) {
  const int32_t k_tile = logical_k / kQuantTileK;
  const int32_t local_k = logical_k - k_tile * kQuantTileK;
  const int32_t n_tile = logical_n / kQuantTileN;
  const int32_t group_n_tile = n_tile / kGroupTiles;
  const int32_t subtile = n_tile - group_n_tile * kGroupTiles;
  const int32_t local_n_vec = (logical_n - n_tile * kQuantTileN) / 8;
  const int32_t local_word = local_k * (kQuantTileN / 8) + local_n_vec;
  return k_tile * (size_n * 2) + group_n_tile * kGroupTiles * (kQuantTileK * kQuantTileN / 8) +
         local_word * kGroupTiles + subtile;
}

/**
 * \brief Marlin ``dequant<half2, kFE2M1f, skip_flop=false>`` on one packed word.
 *
 * \param packed One uint32 holding eight E2M1 values at a single K, eight N.
 * \param frag Eight fp16 values as four half2, N-major, same order as Marlin.
 */
SGL_DEVICE void dequant_fe2m1_word(uint32_t packed, half2 frag[4]) {
  auto dequant4 = [](int q, half2* frag_b) {
    constexpr int MASK = 0x70007000;
    int Out1 = (q & 0x80008000) | ((q & MASK) >> 3);
    q <<= 4;
    int Out2 = (q & 0x80008000) | ((q & MASK) >> 3);
    frag_b[1] = *reinterpret_cast<const half2*>(&Out1);
    frag_b[0] = *reinterpret_cast<const half2*>(&Out2);
    const half2 bias = __float2half2_rn(static_cast<float>(1 << 14));
    frag_b[1] = __hmul2(frag_b[1], bias);
    frag_b[0] = __hmul2(frag_b[0], bias);
  };
  dequant4(static_cast<int>(packed << 8), frag + 0);
  dequant4(static_cast<int>(packed), frag + 2);
}

/**
 * \brief SM70 UE8M0 -> float, matching marlin_v100 ``e8m0x2_to_half2_fast``.
 *
 * Exact iff byte in {0} union [113, 142]. Byte 0 becomes +0, not 2^-127.
 */
SGL_DEVICE float e8m0_to_float(uint8_t byte) {
  int e = static_cast<int>(byte) - 112;
  e = e < 0 ? 0 : (e > 31 ? 31 : e);
  const uint16_t bits = static_cast<uint16_t>(e << 10);
  return __half2float(*reinterpret_cast<const __half*>(&bits));
}

SGL_DEVICE void load_e8m0_scales(const uint8_t* __restrict__ scale_row, int32_t n_base, float out[8]) {
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    out[p] = e8m0_to_float(scale_row[n_base + p]);
  }
}

SGL_DEVICE void unpack_fe2m1(uint32_t packed, float out[8]) {
  half2 frag[4];
  dequant_fe2m1_word(packed, frag);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[2 * i] = __low2float(frag[i]);
    out[2 * i + 1] = __high2float(frag[i]);
  }
}

/**
 * \brief Split-K GEMV: each work item is (route, split, 8 output columns).
 *
 * \tparam kSizeN Output columns of this GEMM (w13=4608, w2=5120).
 * \tparam kSizeK Reduction length (w13=5120, w2=2304).
 * \tparam kSplitK Must divide ``kSizeK / 32``.
 * \tparam kApplyWeight If true, multiply the split partial by ``route_weights``.
 */
template <int kSizeN, int kSizeK, int kSplitK, bool kApplyWeight>
__global__ __launch_bounds__(kThreads, 8) void mxfp4_gemv_splitk_kernel(
    const fp16_t* __restrict__ input,
    const uint32_t* __restrict__ qweight,
    const uint8_t* __restrict__ scales,
    const int32_t* __restrict__ expert_ids,
    const float* __restrict__ route_weights,
    float* __restrict__ partials,
    int32_t num_routes,
    int32_t input_ld,
    int32_t partials_ld,
    int32_t n_experts,
    int32_t k_tiles,
    int32_t num_groups,
    int32_t topk,
    int32_t input_is_per_route) {
  static_assert(kSizeK % (kGroupSize * kSplitK) == 0, "split-K must cover whole MXFP4 groups");
  constexpr int32_t kNq = kSizeN / 8;
  constexpr int32_t kGroupsPerSplit = (kSizeK / kGroupSize) / kSplitK;

  const int32_t work = static_cast<int32_t>(blockIdx.x) * kThreads + static_cast<int32_t>(threadIdx.x);
  const int32_t n_work = num_routes * kSplitK * kNq;
  if (work >= n_work) {
    return;
  }
  const int32_t qword = work % kNq;
  const int32_t split_route = work / kNq;
  const int32_t split = split_route % kSplitK;
  const int32_t route = split_route / kSplitK;
  const int32_t n_base = qword * 8;
  const int32_t eid = expert_ids[route];
  float* dst = partials + (static_cast<int32_t>(split) * partials_ld + route) * kSizeN + n_base;

  if (eid < 0 || eid >= n_experts) {
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      dst[p] = 0.f;
    }
    return;
  }

  const int32_t token = input_is_per_route ? route : (route / topk);
  const fp16_t* x_row = input + token * input_ld;
  const uint32_t* qw_e = qweight + static_cast<int64_t>(eid) * k_tiles * (kSizeN * 2);
  const uint8_t* sc_e = scales + static_cast<int64_t>(eid) * num_groups * kSizeN;

  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const int32_t g0 = split * kGroupsPerSplit;
  const int32_t g1 = g0 + kGroupsPerSplit;
  for (int32_t g = g0; g < g1; ++g) {
    float scale[8];
    load_e8m0_scales(sc_e + g * kSizeN, n_base, scale);
    const int32_t k0 = g * kGroupSize;
#pragma unroll 4
    for (int32_t kk = 0; kk < kGroupSize; ++kk) {
      const int32_t k = k0 + kk;
      const float x = static_cast<float>(x_row[k]);
      const uint32_t packed = qw_e[qweight_offset(kSizeN, k, n_base)];
      float w[8];
      unpack_fe2m1(packed, w);
#pragma unroll
      for (int p = 0; p < 8; ++p) {
        acc[p] = fmaf(x, w[p] * scale[p], acc[p]);
      }
    }
  }
  if constexpr (kApplyWeight) {
    const float rw = route_weights[route];
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      acc[p] *= rw;
    }
  }
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    dst[p] = acc[p];
  }
}

/**
 * \brief Sum split-K gate/up partials, then SiLU(gate)*up.
 *
 * \param gate_partials [split_k, partials_ld, 4608]
 * \param activated     [num_routes, 2304]
 */
__global__ __launch_bounds__(kThreads, 8) void silu_mul_from_partials_kernel(
    const float* __restrict__ gate_partials,
    fp16_t* __restrict__ activated,
    int32_t num_routes,
    int32_t split_k,
    int32_t partials_ld,
    float swiglu_limit) {
  constexpr int32_t kNq = kIntermediate / 8;
  const int32_t work = static_cast<int32_t>(blockIdx.x) * kThreads + static_cast<int32_t>(threadIdx.x);
  if (work >= num_routes * kNq) {
    return;
  }
  const int32_t qword = work % kNq;
  const int32_t route = work / kNq;
  const int32_t n_base = qword * 8;
  float gate[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float up[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int32_t s = 0; s < split_k; ++s) {
    const float* row = gate_partials + (s * partials_ld + route) * kGateUp;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      gate[p] += row[n_base + p];
      up[p] += row[kIntermediate + n_base + p];
    }
  }
  fp16_t* dst = activated + route * kIntermediate + n_base;
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    // Match fused_marlin_moe.swiglu_limit_func: clamp gate high and up
    // both sides, then SiLU(gate)*up. DSV4.1-Flash uses limit 10.
    float g = gate[p];
    float u = up[p];
    if (swiglu_limit > 0.f) {
      g = fminf(g, swiglu_limit);
      u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
    }
    const float sig = 1.f / (1.f + expf(-g));
    dst[p] = static_cast<fp16_t>(g * sig * u);
  }
}

/**
 * \brief Sum weighted down split-K partials into per-token hidden.
 *
 * \param down_partials [split_k, partials_ld, 5120]
 * \param output        [M, 5120]
 */
__global__ __launch_bounds__(kThreads, 8) void down_reduce_kernel(
    const float* __restrict__ down_partials,
    fp16_t* __restrict__ output,
    int32_t n_tok,
    int32_t topk,
    int32_t split_k,
    int32_t partials_ld) {
  constexpr int32_t kNq = kHidden / 8;
  const int32_t work = static_cast<int32_t>(blockIdx.x) * kThreads + static_cast<int32_t>(threadIdx.x);
  if (work >= n_tok * kNq) {
    return;
  }
  const int32_t qword = work % kNq;
  const int32_t token = work / kNq;
  const int32_t n_base = qword * 8;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const int32_t route0 = token * topk;
  for (int32_t r = 0; r < topk; ++r) {
    const int32_t route = route0 + r;
    for (int32_t s = 0; s < split_k; ++s) {
      const float* row = down_partials + (s * partials_ld + route) * kHidden;
#pragma unroll
      for (int p = 0; p < 8; ++p) {
        acc[p] += row[n_base + p];
      }
    }
  }
  fp16_t* dst = output + token * kHidden + n_base;
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    dst[p] = static_cast<fp16_t>(acc[p]);
  }
}

/**
 * \brief Decode-shaped MXFP4 MoE: gated SiLU, top-6, hidden=5120, I=2304.
 *
 * \param hidden        fp16 [M, 5120], M in 1..4
 * \param w13           int32 [E, 320, 9216]
 * \param w2            int32 [E, 144, 10240]
 * \param s13           uint8 [E, 160, 4608]
 * \param s2            uint8 [E, 72, 5120]
 * \param topk_ids      int32 [M, 6]; negative ids are skipped
 * \param topk_weights  fp32 [M, 6] (routed scale already folded if used)
 * \param gate_partials fp32 [8, P, 4608], P >= M*6
 * \param activated     fp16 [P, 2304]
 * \param down_partials fp32 [8, P, 5120]
 * \param output        fp16 [M, 5120]
 * \param swiglu_limit  Marlin clamp; 0 = unclamped SiLU. DSV4.1-Flash is 10.
 */
inline void mxfp4_moe_decode(tvm::ffi::TensorView hidden,
                             tvm::ffi::TensorView w13,
                             tvm::ffi::TensorView w2,
                             tvm::ffi::TensorView s13,
                             tvm::ffi::TensorView s2,
                             tvm::ffi::TensorView topk_ids,
                             tvm::ffi::TensorView topk_weights,
                             tvm::ffi::TensorView gate_partials,
                             tvm::ffi::TensorView activated,
                             tvm::ffi::TensorView down_partials,
                             tvm::ffi::TensorView output,
                             double swiglu_limit) {
  using namespace host;
  SymbolicSize n_tok = {"n_tok"};
  SymbolicSize n_experts = {"n_experts"};
  SymbolicSize n_partials = {"n_partials"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({n_tok, kHidden})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(hidden)
      .verify(output);
  TensorMatcher({n_tok, kTopK})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(topk_ids);
  TensorMatcher({n_tok, kTopK})  //
      .with_dtype<float>()
      .with_device<kDLCUDA>(device_)
      .verify(topk_weights);
  TensorMatcher({n_experts, kW13KTiles, kGateUp * 2})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(w13);
  TensorMatcher({n_experts, kW2KTiles, kHidden * 2})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(w2);
  TensorMatcher({n_experts, kW13Groups, kGateUp})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(s13);
  TensorMatcher({n_experts, kW2Groups, kHidden})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(s2);
  TensorMatcher({kSplitKGate, n_partials, kGateUp})  //
      .with_dtype<float>()
      .with_device<kDLCUDA>(device_)
      .verify(gate_partials);
  TensorMatcher({n_partials, kIntermediate})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(activated);
  TensorMatcher({kSplitKDown, n_partials, kHidden})  //
      .with_dtype<float>()
      .with_device<kDLCUDA>(device_)
      .verify(down_partials);

  const int32_t m = static_cast<int32_t>(n_tok.unwrap());
  const int32_t e = static_cast<int32_t>(n_experts.unwrap());
  const int32_t pld = static_cast<int32_t>(n_partials.unwrap());
  const int32_t num_routes = m * kTopK;
  CHECK_HOST(m >= 1 && m <= kMaxM) << "sm70_dsv41 mxfp4_moe_decode: M " << m;
  CHECK_HOST(e >= 1) << "sm70_dsv41 mxfp4_moe_decode: empty expert table";
  CHECK_HOST(pld >= num_routes) << "sm70_dsv41 mxfp4_moe_decode: partials " << pld
                                << " < routes " << num_routes;
  CHECK_HOST(hidden.is_contiguous()) << "sm70_dsv41 mxfp4_moe_decode: hidden not contiguous";
  CHECK_HOST(w13.is_contiguous() && w2.is_contiguous())
      << "sm70_dsv41 mxfp4_moe_decode: qweight not contiguous";
  CHECK_HOST(s13.is_contiguous() && s2.is_contiguous())
      << "sm70_dsv41 mxfp4_moe_decode: scales not contiguous";
  CHECK_HOST(output.is_contiguous()) << "sm70_dsv41 mxfp4_moe_decode: output not contiguous";
  CHECK_HOST(swiglu_limit >= 0.0 && swiglu_limit == swiglu_limit)
      << "sm70_dsv41 mxfp4_moe_decode: swiglu_limit " << swiglu_limit;

  const DLDevice dev = device_.unwrap();
  const float limit = static_cast<float>(swiglu_limit);
  const int32_t gate_work = num_routes * kSplitKGate * (kGateUp / 8);
  const int32_t silu_work = num_routes * (kIntermediate / 8);
  const int32_t down_work = num_routes * kSplitKDown * (kHidden / 8);
  const int32_t red_work = m * (kHidden / 8);

  LaunchKernel(div_ceil(gate_work, kThreads), kThreads, dev)(
      mxfp4_gemv_splitk_kernel<kGateUp, kHidden, kSplitKGate, false>,
      static_cast<const fp16_t*>(hidden.data_ptr()),
      static_cast<const uint32_t*>(w13.data_ptr()),
      static_cast<const uint8_t*>(s13.data_ptr()),
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<const float*>(nullptr),
      static_cast<float*>(gate_partials.data_ptr()),
      num_routes,
      kHidden,
      pld,
      e,
      kW13KTiles,
      kW13Groups,
      kTopK,
      0);
  LaunchKernel(div_ceil(silu_work, kThreads), kThreads, dev)(
      silu_mul_from_partials_kernel,
      static_cast<const float*>(gate_partials.data_ptr()),
      static_cast<fp16_t*>(activated.data_ptr()),
      num_routes,
      kSplitKGate,
      pld,
      limit);
  LaunchKernel(div_ceil(down_work, kThreads), kThreads, dev)(
      mxfp4_gemv_splitk_kernel<kHidden, kIntermediate, kSplitKDown, true>,
      static_cast<const fp16_t*>(activated.data_ptr()),
      static_cast<const uint32_t*>(w2.data_ptr()),
      static_cast<const uint8_t*>(s2.data_ptr()),
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<const float*>(topk_weights.data_ptr()),
      static_cast<float*>(down_partials.data_ptr()),
      num_routes,
      kIntermediate,
      pld,
      e,
      kW2KTiles,
      kW2Groups,
      kTopK,
      1);
  LaunchKernel(div_ceil(red_work, kThreads), kThreads, dev)(
      down_reduce_kernel,
      static_cast<const float*>(down_partials.data_ptr()),
      static_cast<fp16_t*>(output.data_ptr()),
      m,
      kTopK,
      kSplitKDown,
      pld);
}

}  // namespace sglang::sm70_dsv41
