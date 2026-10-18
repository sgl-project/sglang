// Fused Kimi-K3 router top-k + aiter moe_sorting for decode-sized batches (ROCm, wave64).
//
// Replaces aiter's topk_reg_kernel + the two-phase opus moe_sorting with one
// workgroup: wave w selects the top-k experts of tokens w, w + 16, ...; the
// block then writes aiter's block-padded sorted routes and zeroes moe_buf.
// Output contract (matches moe_sorting_opus_fwd, verified against it):
//   sorted_expert_ids[b]  active experts ascending, ceil(count / block_m) blocks each
//   sorted_ids[p]         (topk_slot << 24) | token, tokens ascending inside an
//                         expert's rows; padding (topk << 24) | num_tokens, weight 0
//   num_valid_ids         {num_blocks * block_m, num_tokens}
// Entries past the valid range are left untouched, as aiter's kernel does.
// Workgroups past the first compute the next layer's attention-residual bank partials
// (attn_res/bank_partial_hip.cuh), one per token row, while the first sorts.
// Scores follow topk_reg_kernel: sig = rcp(1 + 2^(-log2e * x)), the selection
// key is sig + bias, the weight is the pre-bias sig, renormalized in descending
// order; exact score ties go to the lower expert id.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include "attn_res/bank_partial_hip.cuh"
#include "route_topk.cuh"
#include <cstdint>

#ifndef USE_ROCM
#error "kimi_k3/route_sort.cuh is a wave64 (ROCm) kernel"
#endif

namespace sglang {

struct K3RouteSortParams {
  const bf16_t* __restrict__ logits;  // [M, E], row stride logits_stride elements
  const void* __restrict__ bias;      // [E], bf16 or fp32
  float* __restrict__ topk_weights;   // [M, K]
  int32_t* __restrict__ topk_ids;     // [M, K]
  int32_t* __restrict__ sorted_ids;
  float* __restrict__ sorted_weights;
  int32_t* __restrict__ sorted_expert_ids;
  int32_t* __restrict__ num_valid_ids;  // [2]
  uint4* __restrict__ moe_buf;          // zeroed as moe_buf_vecs 16-byte vectors
  // Optional fused stage-1 activation quant (aiter fused_dynamic_mxfp8_quant_moe_sort):
  const bf16_t* __restrict__ quant_input;  // [M, quant_cols], row stride quant_input_stride
  uint8_t* __restrict__ a1;                // [M, quant_cols] fp8 e4m3, token-major
  uint8_t* __restrict__ a1_scale;          // e8m0 bytes, mx_scale_shuffle_idx(scale_n_pad, row, g)
  int64_t logits_stride;
  int64_t moe_buf_vecs;
  int64_t quant_input_stride;
  int32_t quant_cols;
  int32_t scale_n_pad;
  int32_t num_tokens;
  int32_t block_m;
  float routed_scaling_factor;
  bool renormalize;
  k3_attn_res_read::PartialParams partials;
};

inline constexpr int kRouteSortThreads = 1024;
inline constexpr int kRouteSortWave = 64;
inline constexpr int kRouteSortWaves = kRouteSortThreads / kRouteSortWave;
// Token masks are one uint64 per expert.
inline constexpr int kRouteSortMaxTokens = 64;
inline constexpr int kRouteSortMinBlockM = 16;
inline constexpr int kRouteSortMaxCand = kRouteTopkMaxCand;

// aiter mx_scale_shuffle_idx: the e8m0 tile layout the MXFP8/MXFP4 MoE GEMMs read.
SGL_DEVICE int mx_scale_shuffle_idx(int scale_n_pad, int x, int y) {
  return (x / 32 * scale_n_pad) * 32 + (y / 8) * 256 + (y % 4) * 64 + (x % 16) * 4 + (y % 8) / 4 * 2 + (x % 32) / 16;
}

// One 32-element MXFP8 group as aiter's fused quant kernel computes it: amax
// floored at 1e-10, round-up e8m0 scale ceil_pow2(amax / 448), elements scaled
// by its reciprocal and converted with the hardware pk_fp8 instruction.
SGL_DEVICE uint8_t mxfp8_quant_group(const bf16_t* __restrict__ src, uint8_t* __restrict__ dst) {
  float v[32];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint4 raw = reinterpret_cast<const uint4*>(src)[i];
    const bf16x2_t* pairs = reinterpret_cast<const bf16x2_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const auto [a, b] = device::cast<fp32x2_t>(pairs[j]);
      v[i * 8 + j * 2] = a;
      v[i * 8 + j * 2 + 1] = b;
    }
  }
  float amax = 1e-10f;
#pragma unroll
  for (int i = 0; i < 32; ++i)
    amax = fmaxf(amax, fabsf(v[i]));
  const uint32_t u = __float_as_uint(amax * (1.0f / 448.0f));
  uint32_t exponent = (u >> 23) & 0xffu;
  if (exponent < 0xffu && (u & 0x7fffffu)) exponent += 1;
  const float inv = __builtin_amdgcn_rcpf(__uint_as_float(exponent << 23));
  uint32_t words[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int w = __builtin_amdgcn_cvt_pk_fp8_f32(v[4 * i] * inv, v[4 * i + 1] * inv, 0, false);
    w = __builtin_amdgcn_cvt_pk_fp8_f32(v[4 * i + 2] * inv, v[4 * i + 3] * inv, w, true);
    words[i] = static_cast<uint32_t>(w);
  }
  reinterpret_cast<uint4*>(dst)[0] = make_uint4(words[0], words[1], words[2], words[3]);
  reinterpret_cast<uint4*>(dst)[1] = make_uint4(words[4], words[5], words[6], words[7]);
  return static_cast<uint8_t>(exponent);
}

inline constexpr int kRouteSortMaxGroups = 128;  // quant_cols / 32

// kPrecomputed: topk_ids / topk_weights are inputs (an earlier kernel ran the top-k).
template <int kExperts, int kTopK, typename TBias, bool kFuseQuant, bool kPrecomputed>
__global__ __launch_bounds__(kRouteSortThreads) void k3_route_sort_kernel(const __grid_constant__ K3RouteSortParams p) {
  static_assert(kExperts % (2 * kRouteSortWave) == 0 && kExperts <= kRouteSortThreads);
  static_assert(kTopK <= kRouteSortWave && kTopK < 256);

  __shared__ unsigned long long s_mask[kExperts];  // bit t: token t routes to the expert
  __shared__ int16_t s_ids[kRouteSortMaxTokens][kTopK];
  __shared__ float s_tsig[kRouteSortMaxTokens][kTopK];  // selected pre-bias sig, rank order
  __shared__ float s_w[kRouteSortMaxTokens][kTopK];
  __shared__ int32_t s_boff[kExperts];
  __shared__ int32_t s_wave_prefix[kRouteSortWaves];
  // per-wave selection scratch
  __shared__ __attribute__((aligned(16))) uint64_t s_ckey[kRouteSortWaves][kRouteSortMaxCand];
  __shared__ float s_csig[kRouteSortWaves][kRouteSortMaxCand];
  __shared__ uint8_t s_qscale[kFuseQuant ? kRouteSortMaxTokens : 1][kFuseQuant ? kRouteSortMaxGroups : 1];
  __shared__ int32_t s_route_row[kFuseQuant ? kRouteSortMaxTokens * kTopK : 1];  // sorted row of each route

  static_assert(k3_attn_res_read::kThreads <= kRouteSortThreads, "a partial workgroup covers a row");
  if (blockIdx.x > 0) {
    __shared__ k3_attn_res_read::PartialShared s_partial;
    k3_attn_res_read::attn_res_partials(p.partials, s_partial, static_cast<int>(blockIdx.x) - 1);
    return;
  }
  const int tid = threadIdx.x;
  const int lane = tid % kRouteSortWave;
  const int wave = tid / kRouteSortWave;
  const int num_tokens = p.num_tokens;
  const uint64_t lanemask_lt = (1ull << lane) - 1ull;
#ifdef K3RS_TIMING
  uint64_t k3rs_ts[16] = {};
  if (tid == 0) k3rs_ts[0] = __builtin_amdgcn_s_memtime();
#endif

  for (int e = tid; e < kExperts; e += kRouteSortThreads)
    s_mask[e] = 0ull;
  __syncthreads();

#ifdef K3RS_TIMING
  if (tid == 0) k3rs_ts[1] = __builtin_amdgcn_s_memtime();
#endif

  // Stage-1 activation quant: one 32-element group per task. The waves that
  // have no token to route (num_tokens < 16) run it alongside the top-k.
  const int quant_groups = kFuseQuant ? p.quant_cols / 32 : 0;
  auto quant_tasks = [&](int worker, int num_workers) {
    for (int task = worker; task < num_tokens * quant_groups; task += num_workers) {
      const int t = task / quant_groups;
      const int g = task % quant_groups;
      s_qscale[t][g] = mxfp8_quant_group(
          p.quant_input + t * p.quant_input_stride + g * 32, p.a1 + static_cast<int64_t>(t) * p.quant_cols + g * 32);
    }
  };
  if (kFuseQuant && num_tokens < kRouteSortWaves && wave >= num_tokens) {
    quant_tasks(tid - num_tokens * kRouteSortWave, (kRouteSortWaves - num_tokens) * kRouteSortWave);
  }

  if constexpr (kPrecomputed) {
    for (int route = tid; route < num_tokens * kTopK; route += kRouteSortThreads) {
      const int t = route / kTopK;
      const int k = route % kTopK;
      const int id = p.topk_ids[route];
      s_ids[t][k] = static_cast<int16_t>(id);
      s_w[t][k] = p.topk_weights[route];
      atomicOr(&s_mask[id], 1ull << t);
    }
  } else {
    // ---- top-k: one wave per token
    for (int t = wave; t < num_tokens; t += kRouteSortWaves) {
      int id = 0;
      float w = 0.0f;
      route_topk_wave<kExperts, kTopK, TBias>(
          reinterpret_cast<const bf16x2_t*>(p.logits + t * p.logits_stride),
          p.bias,
          p.routed_scaling_factor,
          p.renormalize,
          s_ckey[wave],
          s_csig[wave],
          s_ids[t],
          s_tsig[t],
          lane,
          id,
          w);
      if (lane < kTopK) {
        p.topk_weights[t * kTopK + lane] = w;
        p.topk_ids[t * kTopK + lane] = id;
        s_w[t][lane] = w;
        atomicOr(&s_mask[id], 1ull << t);
      }
    }
  }
  if (kFuseQuant && num_tokens >= kRouteSortWaves) quant_tasks(tid, kRouteSortThreads);
  __syncthreads();

#ifdef K3RS_TIMING
  if (tid == 0) k3rs_ts[7] = __builtin_amdgcn_s_memtime();
#endif

  const int block_m = p.block_m;
  const int pad_id = (kTopK << 24) | num_tokens;
  // ---- block layout: exclusive scan of ceil(count / block_m) over experts (one per thread)
  const int count = tid < kExperts ? __popcll(s_mask[tid]) : 0;
  const int nblk = (count + block_m - 1) / block_m;
  int incl;
  if (num_tokens <= block_m) {
    // nblk is 0 or 1: the in-wave prefix is a popcount of the ballot.
    const uint64_t active = __ballot(nblk > 0);
    incl = __popcll(active & lanemask_lt) + nblk;
    if (lane == 0) s_wave_prefix[wave] = __popcll(active);
  } else {
    incl = nblk;
#pragma unroll
    for (int offset = 1; offset < kRouteSortWave; offset <<= 1) {
      const int v = __shfl_up(incl, offset, kRouteSortWave);
      if (lane >= offset) incl += v;
    }
    if (lane == kRouteSortWave - 1) s_wave_prefix[wave] = incl;
  }
  __syncthreads();
  int wave_base = 0;
  int total_blocks = 0;
#pragma unroll
  for (int w = 0; w < kRouteSortWaves; ++w) {
    const int v = s_wave_prefix[w];
    wave_base += w < wave ? v : 0;
    total_blocks += v;
  }

  // Every valid row first gets the padding entry (16-byte stores spread over
  // all threads); after the barrier the route rows overwrite theirs.
  if (tid < kExperts && nblk > 0) {
    const int boff = incl - nblk + wave_base;
    s_boff[tid] = boff;
    for (int j = 0; j < nblk; ++j)
      p.sorted_expert_ids[boff + j] = tid;
  }
  const int total_rows = total_blocks * block_m;
  if (tid == 0) {
    p.num_valid_ids[0] = total_rows;
    p.num_valid_ids[1] = num_tokens;
  }
  for (int v = tid; v < total_rows / 4; v += kRouteSortThreads) {
    reinterpret_cast<int4*>(p.sorted_ids)[v] = make_int4(pad_id, pad_id, pad_id, pad_id);
    reinterpret_cast<float4*>(p.sorted_weights)[v] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  __syncthreads();

  for (int route = tid; route < num_tokens * kTopK; route += kRouteSortThreads) {
    const int t = route / kTopK;
    const int k = route % kTopK;
    const int e = s_ids[t][k];
    const int row = s_boff[e] * block_m + __popcll(s_mask[e] & ((1ull << t) - 1ull));
    p.sorted_ids[row] = (k << 24) | t;
    p.sorted_weights[row] = s_w[t][k];
    if constexpr (kFuseQuant) s_route_row[route] = row;
  }
  if constexpr (kFuseQuant) {
    __syncthreads();
    // Every route's sorted row carries its token's group scales.
    const auto scale = [&](int t, int g) -> uint32_t { return s_qscale[t][g]; };
    if (block_m == 32 && num_tokens <= 16 && quant_groups % 8 == 0) {
      // A row's scales for groups g and g + 4 (g % 8 < 4) share a 4-byte word with those of
      // row + 16 of its 32-row block; with at most 16 tokens per expert that row is padding,
      // so one word store places two scales (byte stores here are one transaction each).
      const int words = quant_groups / 2;
      for (int item = tid; item < num_tokens * kTopK * words; item += kRouteSortThreads) {
        const int route = item / words;
        const int w = item % words;
        const int t = route / kTopK;
        const int g = w / 4 * 8 + w % 4;
        const uint32_t pad = 127u;  // e8m0 1.0
        const uint32_t word = scale(t, g) | (pad << 8) | (scale(t, g + 4) << 16) | (pad << 24);
        *reinterpret_cast<uint32_t*>(p.a1_scale + mx_scale_shuffle_idx(p.scale_n_pad, s_route_row[route], g)) = word;
      }
    } else {
      for (int item = tid; item < num_tokens * kTopK * quant_groups; item += kRouteSortThreads) {
        const int route = item / quant_groups;
        const int g = item % quant_groups;
        p.a1_scale[mx_scale_shuffle_idx(p.scale_n_pad, s_route_row[route], g)] =
            static_cast<uint8_t>(scale(route / kTopK, g));
      }
    }
  }

#ifdef K3RS_TIMING
  if (tid == 0) k3rs_ts[10] = __builtin_amdgcn_s_memtime();
#endif

  // ---- zero the stage-2 accumulation buffer
  for (int64_t v = tid; v < p.moe_buf_vecs; v += kRouteSortThreads)
    p.moe_buf[v] = make_uint4(0u, 0u, 0u, 0u);
#ifdef K3RS_TIMING
  __syncthreads();
  if (tid == 0) {
    k3rs_ts[11] = __builtin_amdgcn_s_memtime();
    for (int i = 0; i < 12; ++i)
      reinterpret_cast<uint64_t*>(p.moe_buf)[i] = k3rs_ts[i];
  }
#endif
}

template <int kExperts, int kTopK, typename TBias, bool kFuseQuant, bool kPrecomputed>
struct K3RouteSortKernel {
  static void
  run(const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView topk_weights,
      const tvm::ffi::TensorView topk_ids,
      const tvm::ffi::TensorView sorted_ids,
      const tvm::ffi::TensorView sorted_weights,
      const tvm::ffi::TensorView sorted_expert_ids,
      const tvm::ffi::TensorView num_valid_ids,
      const tvm::ffi::TensorView moe_buf,
      const tvm::ffi::TensorView quant_input,
      const tvm::ffi::TensorView a1,
      const tvm::ffi::TensorView a1_scale,
      const int64_t block_m,
      const double routed_scaling_factor,
      const bool renormalize,
      const tvm::ffi::TensorView bank,
      const tvm::ffi::TensorView cw0,
      const tvm::ffi::TensorView cw1,
      const tvm::ffi::TensorView bank_acc,
      const tvm::ffi::TensorView bank_ms,
      const int64_t nvb,
      const int64_t sides,
      const int64_t partial_rows,
      const double score_eps) {
    using namespace host;

    auto M = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLROCM>();
    TensorMatcher({M, kExperts}).with_dtype<bf16_t>().with_device(device_).with_strides({-1, 1}).verify(logits);
    TensorMatcher({kExperts}).with_dtype<TBias>().with_device(device_).verify(bias);
    TensorMatcher({M, kTopK}).with_dtype<float>().with_device(device_).verify(topk_weights);
    TensorMatcher({M, kTopK}).with_dtype<int32_t>().with_device(device_).verify(topk_ids);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device_).verify(num_valid_ids);

    const auto num_tokens = static_cast<int32_t>(M.unwrap());
    RuntimeCheck(
        num_tokens > 0 && num_tokens <= kRouteSortMaxTokens, "route_sort covers 1..64 tokens, got ", num_tokens);
    RuntimeCheck(block_m >= kRouteSortMinBlockM, "block_m must be >= 16");
    RuntimeCheck(logits.stride(0) % 2 == 0, "logits row stride must be even (bf16x2 loads)");
    const int64_t max_rows = sorted_ids.numel();
    RuntimeCheck(
        max_rows >= num_tokens * kTopK + kExperts * block_m - kTopK && sorted_weights.numel() >= max_rows &&
            sorted_expert_ids.numel() * block_m >= max_rows,
        "sorted buffers are smaller than aiter's moe_sorting sizes");
    const int64_t moe_buf_bytes = moe_buf.numel() * static_cast<int64_t>(moe_buf.dtype().bits / 8);
    RuntimeCheck(moe_buf_bytes % 16 == 0, "moe_buf must be a whole number of 16-byte vectors");
    int32_t quant_cols = 0, scale_n_pad = 0;
    if constexpr (kFuseQuant) {
      TensorMatcher({M, -1}).with_dtype<bf16_t>().with_device(device_).with_strides({-1, 1}).verify(quant_input);
      quant_cols = static_cast<int32_t>(quant_input.size(1));
      RuntimeCheck(quant_cols % 32 == 0 && quant_cols / 32 <= kRouteSortMaxGroups, "quant cols must be 32k, k <= 128");
      RuntimeCheck(
          quant_input.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(quant_input.data_ptr()) % 16 == 0,
          "quant input rows must be 16-byte aligned");
      RuntimeCheck(a1.numel() >= static_cast<int64_t>(num_tokens) * quant_cols, "a1 too small");
      scale_n_pad = (quant_cols / 32 + 7) / 8 * 8;
      RuntimeCheck(a1_scale.numel() >= (max_rows + 31) / 32 * 32 * scale_n_pad, "a1_scale too small");
    }

    k3_attn_res_read::PartialParams partials{};
    if (partial_rows > 0) {
      // bank [R, NB, H] bf16 read over rows [0, nvb); cw0 (and cw1 with two sides) [H] fp32;
      // bank_acc [2, >= R, H] and bank_ms [2, >= R, 2] fp32, side s in [s].
      namespace ar = k3_attn_res_read;
      auto R = SymbolicSize{"partial_rows"};
      auto NB = SymbolicSize{"bank_blocks"};
      auto RA = SymbolicSize{"partial_capacity"};
      TensorMatcher({R, NB, ar::kHidden})
          .with_dtype<bf16_t>()
          .with_device(device_)
          .with_strides({-1, -1, 1})
          .verify(bank);
      TensorMatcher({ar::kHidden}).with_dtype<float>().with_device(device_).verify(cw0);
      if (sides == 2) TensorMatcher({ar::kHidden}).with_dtype<float>().with_device(device_).verify(cw1);
      TensorMatcher({2, RA, ar::kHidden}).with_dtype<float>().with_device(device_).verify(bank_acc);
      TensorMatcher({2, RA, 2}).with_dtype<float>().with_device(device_).verify(bank_ms);
      RuntimeCheck(R.unwrap() == partial_rows && partial_rows <= RA.unwrap(), "bank partial rows");
      RuntimeCheck(sides == 1 || sides == 2, "one or two score vectors");
      RuntimeCheck(1 <= nvb && nvb <= ar::kMaxNvb && nvb <= NB.unwrap(), "bank partials take 1-8 bank rows");
      RuntimeCheck(
          bank.stride(0) % ar::kVec == 0 && bank.stride(1) % ar::kVec == 0, "bank rows must be 16-byte aligned");
      auto* acc = static_cast<float*>(bank_acc.data_ptr());
      auto* ms = static_cast<float*>(bank_ms.data_ptr());
      const int64_t capacity = RA.unwrap();
      partials = ar::PartialParams{
          .bank = static_cast<const bf16_t*>(bank.data_ptr()),
          .bank_stride_t = bank.stride(0),
          .bank_stride_b = bank.stride(1),
          .cw =
              {static_cast<const float4*>(cw0.data_ptr()),
               sides == 2 ? static_cast<const float4*>(cw1.data_ptr()) : nullptr},
          .acc = {reinterpret_cast<float4*>(acc), reinterpret_cast<float4*>(acc + capacity * ar::kHidden)},
          .ms = {reinterpret_cast<float2*>(ms), reinterpret_cast<float2*>(ms + capacity * 2)},
          .score_eps = static_cast<float>(score_eps),
          .nvb = static_cast<int32_t>(nvb),
          .rows = static_cast<int32_t>(partial_rows),
      };
    }
    const auto params = K3RouteSortParams{
        .logits = static_cast<const bf16_t*>(logits.data_ptr()),
        .bias = bias.data_ptr(),
        .topk_weights = static_cast<float*>(topk_weights.data_ptr()),
        .topk_ids = static_cast<int32_t*>(topk_ids.data_ptr()),
        .sorted_ids = static_cast<int32_t*>(sorted_ids.data_ptr()),
        .sorted_weights = static_cast<float*>(sorted_weights.data_ptr()),
        .sorted_expert_ids = static_cast<int32_t*>(sorted_expert_ids.data_ptr()),
        .num_valid_ids = static_cast<int32_t*>(num_valid_ids.data_ptr()),
        .moe_buf = static_cast<uint4*>(moe_buf.data_ptr()),
        .quant_input = kFuseQuant ? static_cast<const bf16_t*>(quant_input.data_ptr()) : nullptr,
        .a1 = kFuseQuant ? static_cast<uint8_t*>(a1.data_ptr()) : nullptr,
        .a1_scale = kFuseQuant ? static_cast<uint8_t*>(a1_scale.data_ptr()) : nullptr,
        .logits_stride = logits.stride(0),
        .moe_buf_vecs = moe_buf_bytes / 16,
        .quant_input_stride = kFuseQuant ? quant_input.stride(0) : 0,
        .quant_cols = quant_cols,
        .scale_n_pad = scale_n_pad,
        .num_tokens = num_tokens,
        .block_m = static_cast<int32_t>(block_m),
        .routed_scaling_factor = static_cast<float>(routed_scaling_factor),
        .renormalize = renormalize,
        .partials = partials,
    };
    LaunchKernel(1 + static_cast<uint32_t>(partial_rows), kRouteSortThreads, device_.unwrap())(
        k3_route_sort_kernel<kExperts, kTopK, TBias, kFuseQuant, kPrecomputed>, params);
  }
};

}  // namespace sglang
