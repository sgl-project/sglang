// TP8 all-gather of Kimi-K3's column-parallel MoE front (ROCm, wave64).
//
// With the router and the latent down-projection split by output columns over TP8, rank r's
// front GEMM yields [T, 112] router logits (experts [112 r, 112 r + 112)) and [T, 448] latent
// columns ([448 r, 448 r + 448)), side by side in its output. This gathers them into the full
// [T, 896] logits and [T, 3584] latent on every rank, one workgroup per token row, over the LL
// protocol (comm/ll_hip.cuh) in the gather scratch's second region. With kQuant it also emits
// the stage-1 MXFP8 activations of the latent (fp8 e4m3 + one e8m0 scale per 32 columns, as
// aiter's fused_dynamic_mxfp8_quant_moe_sort computes them); with a TBias it also runs the
// router top-k of the token (route_topk.cuh) for the route sort; with kSitu it also applies the
// shared expert's SiTU to its [T, 2 * 768] gate/up columns of the same GEMM output
// (situ_and_mul.cuh's math, so the shared down GEMM reads the activation directly).

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include "../route_topk.cuh"
#include "../situ_and_mul.cuh"
#include "ll_hip.cuh"
#include <cstdint>
#include <custom_all_reduce.cuh>  // AITER csrc/include: CustomAllreduce, Signal
#include <type_traits>

#ifndef USE_ROCM
#error "kimi_k3/comm/front_gather_hip.cuh is a wave64 (ROCm) kernel"
#endif

namespace sglang {
namespace k3_front_gather {

using namespace k3_ll;

inline constexpr int kWorld = 8;
inline constexpr int kExperts = 896;
inline constexpr int kLatent = 3584;
inline constexpr int kVec = 8;     // bf16 per 16-byte vector
inline constexpr int kGroup = 32;  // MXFP8 scale group
inline constexpr int kExpertVecs = kExperts / kWorld / kVec;
inline constexpr int kLatentVecs = kLatent / kWorld / kVec;
// A rank's slice as thread vectors: latent first, then logits, then 2 idle; the 72-vector
// stride keeps each 32-column latent group in one aligned lane quad (4 lanes x 8 columns).
inline constexpr int kSliceVecs = 72;
static_assert(kLatentVecs + kExpertVecs <= kSliceVecs && kSliceVecs % 4 == 0, "slice layout");
inline constexpr int kBlock = kSliceVecs * kWorld;
inline constexpr int kMaxRows = 64;
inline constexpr int kLinesPerRow = 2 * kSliceVecs;
inline constexpr int64_t kEpochLines = int64_t{kWorld} * kMaxRows * kLinesPerRow;
// The region after the attention-residual gather's (read_hip.cuh: 2 epochs x world x rows x
// 224 lines), and its flags after that gather's rows; Python's scratch size covers both.
inline constexpr int64_t kRegionLine = 2 * int64_t{kWorld} * kMaxRows * 224;
inline constexpr int kEpochBase = kMaxRows;
inline constexpr int kEpochs = 2 * kMaxRows;
inline constexpr int kTopK = 16;
inline constexpr int kSharedInter = 768;  // the shared expert's TP8 intermediate width
inline constexpr int kSituThreads = kSharedInter / kVec;

struct Params {
  const bf16_t* __restrict__ slices;  // [T, 560] this rank's [logits | latent] columns
  bf16_t* __restrict__ logits;        // [T, 896]
  bf16_t* __restrict__ latent;        // [T, 3584]
  uint8_t* __restrict__ a1;           // [T, 3584] fp8 e4m3 (kQuant)
  uint8_t* __restrict__ a1_scale;     // [T, 112] e8m0, per token and 32-column group (kQuant)
  const void* __restrict__ bias;      // [896] TBias router correction bias (top-k)
  int32_t* __restrict__ topk_ids;     // [T, 16] (top-k)
  float* __restrict__ topk_weights;   // [T, 16] (top-k)
  float routed_scaling_factor;
  bool renormalize;
  const bf16_t* __restrict__ gate_up;  // [T, 2 * 768], row stride slice_stride (kSitu)
  bf16_t* __restrict__ situ_out;       // [T, 768] (kSitu)
  float situ_beta;
  float situ_inv_beta;
  float situ_linear_beta;
  float situ_inv_linear_beta;
  // Constant-index only: a runtime index makes the kernarg read a vector load whose wait
  // also waits for in-flight remote stores.
  uint4* scratch[kWorld];
  const uint4* self_scratch;
  int32_t* __restrict__ epochs;
  int64_t slice_stride;
  int32_t rank;
};

template <int kCtrl>
SGL_DEVICE float quad_dpp(float v) {
  return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), kCtrl, 0xf, 0xf, false));
}

// The 8 columns this lane holds of a 32-column group spread over its lane quad, quantized as
// kimi_k3/route_sort.cuh's mxfp8_quant_group does (bit-exact with aiter): amax floored at
// 1e-10, scale ceil_pow2(amax / 448) as e8m0, elements times its reciprocal through pk_fp8.
SGL_DEVICE uint2 mxfp8_quant_lane(const uint4& v, uint32_t& exponent) {
  float f[kVec];
  f[0] = __uint_as_float(v.x << 16), f[1] = __uint_as_float(v.x & 0xffff0000u);
  f[2] = __uint_as_float(v.y << 16), f[3] = __uint_as_float(v.y & 0xffff0000u);
  f[4] = __uint_as_float(v.z << 16), f[5] = __uint_as_float(v.z & 0xffff0000u);
  f[6] = __uint_as_float(v.w << 16), f[7] = __uint_as_float(v.w & 0xffff0000u);
  float amax = 1e-10f;
#pragma unroll
  for (int e = 0; e < kVec; ++e)
    amax = fmaxf(amax, fabsf(f[e]));
  amax = fmaxf(amax, quad_dpp<0xb1>(amax));  // quad_perm [1,0,3,2]
  amax = fmaxf(amax, quad_dpp<0x4e>(amax));  // quad_perm [2,3,0,1]
  const uint32_t u = __float_as_uint(amax * (1.0f / 448.0f));
  exponent = (u >> 23) & 0xffu;
  if (exponent < 0xffu && (u & 0x7fffffu)) exponent += 1;
  const float inv = __builtin_amdgcn_rcpf(__uint_as_float(exponent << 23));
  int w0 = __builtin_amdgcn_cvt_pk_fp8_f32(f[0] * inv, f[1] * inv, 0, false);
  w0 = __builtin_amdgcn_cvt_pk_fp8_f32(f[2] * inv, f[3] * inv, w0, true);
  int w1 = __builtin_amdgcn_cvt_pk_fp8_f32(f[4] * inv, f[5] * inv, 0, false);
  w1 = __builtin_amdgcn_cvt_pk_fp8_f32(f[6] * inv, f[7] * inv, w1, true);
  return make_uint2(static_cast<uint32_t>(w0), static_cast<uint32_t>(w1));
}

// TBias void: no top-k.
template <bool kQuant, typename TBias, bool kSitu>
__global__ __launch_bounds__(kBlock, 1) void k3_front_gather_kernel(const Params p) {
  constexpr bool kRoute = !std::is_void_v<TBias>;
  __shared__ __attribute__((aligned(16))) bf16_t s_logits[kRoute ? kExperts : 1];
  __shared__ __attribute__((aligned(16))) uint64_t s_ckey[kRoute ? kRouteTopkMaxCand : 1];
  __shared__ float s_csig[kRoute ? kRouteTopkMaxCand : 1];
  __shared__ int16_t s_ids[kRoute ? kTopK : 1];
  __shared__ float s_tsig[kRoute ? kTopK : 1];
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const uint32_t flag = next_flag(p.epochs, kEpochBase + t);
  const int64_t epoch_off = kRegionLine + static_cast<int64_t>(flag & 1u) * kEpochLines;
  const int src = tid / kSliceVecs;
  const int vec = tid % kSliceVecs;
  const bool live = vec < kLatentVecs + kExpertVecs;
  const bool is_latent = vec < kLatentVecs;
  // This vector's columns in the rank's GEMM output ([logits | latent]).
  const int slice_col = is_latent ? kExpertVecs * kVec + vec * kVec : (vec - kLatentVecs) * kVec;
  // The shared expert's gate/up loads go out first; their SiTU runs after the exchange.
  uint4 sg = make_uint4(0, 0, 0, 0), su = make_uint4(0, 0, 0, 0);
  if constexpr (kSitu) {
    if (tid < kSituThreads) {
      const bf16_t* row = p.gate_up + t * p.slice_stride + tid * kVec;
      sg = *reinterpret_cast<const uint4*>(row);
      su = *reinterpret_cast<const uint4*>(row + kSharedInter);
    }
  }
  // Pushers and pollers share waves, and a wave runs one side of a branch at a time: the push
  // must come first in program order, or a wave polling a peer holds back its own lanes'
  // pushes that the peer is waiting for.
  uint4 v = make_uint4(0, 0, 0, 0);
  if (live && src == p.rank) {
    v = *reinterpret_cast<const uint4*>(p.slices + t * p.slice_stride + slice_col);
    const int64_t row_line = epoch_off + (static_cast<int64_t>(p.rank) * kMaxRows + t) * kLinesPerRow;
#pragma unroll
    for (int peer = 0; peer < kWorld; ++peer) {
      if (peer != p.rank) ll_push(p.scratch[peer] + row_line, vec, kSliceVecs, v, flag);
    }
  }
  if (live && src != p.rank) {
    const uint4* s = p.self_scratch + epoch_off + (static_cast<int64_t>(src) * kMaxRows + t) * kLinesPerRow + vec;
    uint4 a, b;
    do {
      a = ll_load(s);
      b = ll_load(s + kSliceVecs);
    } while (!ll_ready(a, b, flag));
    v = ll_payload(a, b);
  }
  if (is_latent) {
    const int col = (src * kLatentVecs + vec) * kVec;
    *reinterpret_cast<uint4*>(p.latent + t * kLatent + col) = v;
    if constexpr (kQuant) {
      // Whole quads are latent (kLatentVecs % 4 == 0), so the quad exchange sees live lanes only.
      uint32_t exponent;
      *reinterpret_cast<uint2*>(p.a1 + t * kLatent + col) = mxfp8_quant_lane(v, exponent);
      if (vec % 4 == 0) p.a1_scale[t * (kLatent / kGroup) + col / kGroup] = static_cast<uint8_t>(exponent);
    }
  } else if (live) {
    const int col = (src * kExpertVecs + vec - kLatentVecs) * kVec;
    *reinterpret_cast<uint4*>(p.logits + t * kExperts + col) = v;
    if constexpr (kRoute) *reinterpret_cast<uint4*>(s_logits + col) = v;
  }
  if constexpr (kSitu) {
    if (tid < kSituThreads) {
      const bf16_t* g = reinterpret_cast<const bf16_t*>(&sg);
      const bf16_t* u = reinterpret_cast<const bf16_t*>(&su);
      bf16_t o[kVec];
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        o[e] = device::cast<bf16_t>(kimi_k3::situ_activate<true>(
            device::cast<fp32_t>(g[e]),
            device::cast<fp32_t>(u[e]),
            p.situ_beta,
            p.situ_inv_beta,
            p.situ_linear_beta,
            p.situ_inv_linear_beta));
      }
      *reinterpret_cast<uint4*>(p.situ_out + t * kSharedInter + tid * kVec) = *reinterpret_cast<const uint4*>(o);
    }
  }
  // Every wave has read the flag before it advances (and the logits are in LDS).
  lds_barrier();
  if constexpr (kRoute) {
    if (tid < kRouteTopkWave) {
      int id = 0;
      float w = 0.0f;
      route_topk_wave<kExperts, kTopK, TBias>(
          reinterpret_cast<const bf16x2_t*>(s_logits),
          p.bias,
          p.routed_scaling_factor,
          p.renormalize,
          s_ckey,
          s_csig,
          s_ids,
          s_tsig,
          tid,
          id,
          w);
      if (tid < kTopK) {
        p.topk_ids[t * kTopK + tid] = id;
        p.topk_weights[t * kTopK + tid] = w;
      }
    }
  }
  if (tid == 0) p.epochs[kEpochBase + t] = static_cast<int32_t>(flag);
}

struct K3FrontGatherKernel {
  // `comm` is the dedicated aiter CustomAllreduce* whose meta buffers hold the LL scratch.
  // With quant, a1 [T, 3584] and a1_scale [T, 112] (uint8) receive the latent's MXFP8 form; with
  // route, topk_ids / topk_weights [T, 16] the router top-k under bias [896] (bf16 or fp32).
  static void
  run(int64_t comm,
      const tvm::ffi::TensorView slices,
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView latent,
      const tvm::ffi::TensorView a1,
      const tvm::ffi::TensorView a1_scale,
      const tvm::ffi::TensorView epochs,
      bool quant,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView topk_ids,
      const tvm::ffi::TensorView topk_weights,
      bool route,
      double routed_scaling_factor,
      bool renormalize,
      const tvm::ffi::TensorView gate_up,
      const tvm::ffi::TensorView situ_out,
      bool situ,
      double situ_beta,
      double situ_linear_beta) {
    using namespace host;
    auto T = SymbolicSize{"num_tokens"};
    auto S = SymbolicSize{"slice_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLROCM>();
    TensorMatcher({T, (kExpertVecs + kLatentVecs) * kVec})
        .with_dtype<bf16_t>()
        .with_device(device)
        .with_strides({S, 1})
        .verify(slices);
    if (quant) {
      TensorMatcher({T, kLatent}).with_dtype<uint8_t>().with_device(device).verify(a1);
      TensorMatcher({T, kLatent / kGroup}).with_dtype<uint8_t>().with_device(device).verify(a1_scale);
    }
    TensorMatcher({T, kExperts}).with_dtype<bf16_t>().with_device(device).verify(logits);
    TensorMatcher({T, kLatent}).with_dtype<bf16_t>().with_device(device).verify(latent);
    TensorMatcher({kEpochs}).with_dtype<int32_t>().with_device(device).verify(epochs);
    const bool bf16_bias = route && bias.dtype().code == kDLBfloat;
    if (route) {
      if (bf16_bias) {
        TensorMatcher({kExperts}).with_dtype<bf16_t>().with_device(device).verify(bias);
      } else {
        TensorMatcher({kExperts}).with_dtype<float>().with_device(device).verify(bias);
      }
      TensorMatcher({T, kTopK}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
      TensorMatcher({T, kTopK}).with_dtype<float>().with_device(device).verify(topk_weights);
    }
    if (situ) {
      TensorMatcher({T, 2 * kSharedInter})
          .with_dtype<bf16_t>()
          .with_device(device)
          .with_strides({S, 1})
          .verify(gate_up);
      TensorMatcher({T, kSharedInter}).with_dtype<bf16_t>().with_device(device).verify(situ_out);
      RuntimeCheck(reinterpret_cast<uintptr_t>(gate_up.data_ptr()) % 16 == 0, "gate/up rows must be 16-byte aligned");
    }
    const int64_t rows = T.unwrap();
    RuntimeCheck(1 <= rows && rows <= kMaxRows, "the front gather takes 1-", kMaxRows, " rows");
    RuntimeCheck(
        S.unwrap() % kVec == 0 && reinterpret_cast<uintptr_t>(slices.data_ptr()) % 16 == 0,
        "front slices must be 16-byte aligned");
    auto* communicator = reinterpret_cast<aiter::CustomAllreduce*>(comm);
    RuntimeCheck(communicator->world_size_ == kWorld, "the front gather requires TP", kWorld);
    Params params{
        .slices = static_cast<const bf16_t*>(slices.data_ptr()),
        .logits = static_cast<bf16_t*>(logits.data_ptr()),
        .latent = static_cast<bf16_t*>(latent.data_ptr()),
        .a1 = quant ? static_cast<uint8_t*>(a1.data_ptr()) : nullptr,
        .a1_scale = quant ? static_cast<uint8_t*>(a1_scale.data_ptr()) : nullptr,
        .bias = route ? bias.data_ptr() : nullptr,
        .topk_ids = route ? static_cast<int32_t*>(topk_ids.data_ptr()) : nullptr,
        .topk_weights = route ? static_cast<float*>(topk_weights.data_ptr()) : nullptr,
        .routed_scaling_factor = static_cast<float>(routed_scaling_factor),
        .renormalize = renormalize,
        .gate_up = situ ? static_cast<const bf16_t*>(gate_up.data_ptr()) : nullptr,
        .situ_out = situ ? static_cast<bf16_t*>(situ_out.data_ptr()) : nullptr,
        // situ_and_mul.cuh's host values, so the activation matches it bit for bit
        .situ_beta = static_cast<float>(situ_beta),
        .situ_inv_beta = 1.0f / static_cast<float>(situ_beta),
        .situ_linear_beta = static_cast<float>(situ_linear_beta),
        .situ_inv_linear_beta = 1.0f / static_cast<float>(situ_linear_beta),
        .scratch = {},
        .self_scratch = nullptr,
        .epochs = static_cast<int32_t*>(epochs.data_ptr()),
        .slice_stride = S.unwrap(),
        .rank = communicator->rank_,
    };
    for (int i = 0; i < kWorld; ++i) {
      params.scratch[i] =
          reinterpret_cast<uint4*>(reinterpret_cast<char*>(communicator->sg_.signals[i]) + sizeof(aiter::Signal));
    }
    params.self_scratch = params.scratch[communicator->rank_];
    const auto launch = LaunchKernel(static_cast<uint32_t>(rows), kBlock, device.unwrap());
    const auto pick = [&](auto bias_tag) {
      using TBias = typename decltype(bias_tag)::type;
      if (situ) {
        quant ? launch(k3_front_gather_kernel<true, TBias, true>, params)
              : launch(k3_front_gather_kernel<false, TBias, true>, params);
      } else {
        quant ? launch(k3_front_gather_kernel<true, TBias, false>, params)
              : launch(k3_front_gather_kernel<false, TBias, false>, params);
      }
    };
    if (!route) {
      pick(std::type_identity<void>{});
    } else if (bf16_bias) {
      pick(std::type_identity<bf16_t>{});
    } else {
      pick(std::type_identity<float>{});
    }
  }
};

}  // namespace k3_front_gather
}  // namespace sglang
