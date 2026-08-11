/// Four-bit radix-select router for K3 routing on CDNA (gfx942/gfx950).
///
/// aiter's grouped_topk_kernel spends one round per selected expert, each round
/// rescanning every score from LDS and ending in a cross-lane max whose winner
/// is written back as -INFINITY. At 896 experts and top-16 that fits ~0.72us per
/// round, so the kernel scales with topk and lands near 11us. aiter does avoid
/// that loop for DeepSeek, sorting the group scores to get a pivot and counting
/// against it, but that path is gated on DeepSeek's exact shape (256 experts, 8
/// groups, top-8, top-4 groups) and its innards are specialized to it, so K3
/// falls to the generic loop.
///
/// The pivot idea is the right one; what does not carry over is finding the pivot
/// by sorting, since 896 scores will not sort in registers the way 8 group scores
/// do. So the pivot is approached four key bits at a time instead, which also
/// means the round count follows the key width rather than topk. Three properties
/// of the hardware shape the rest:
///
///   * A round's cost is dominated by the VALU->SALU->VALU trip that broadcasting
///     a cross-lane count requires, and that trip costs the same whether the
///     round resolves one bit or four. Hence a 16-bin histogram per round.
///   * With the round count cut, a lone wave64 becomes issue-bound rather than
///     latency-bound (one VALU op per 4 cycles with nothing to interleave), so
///     the experts are spread over several waves on separate SIMDs.
///   * What is left per round is the work that does not shrink with more waves.
///     Transposing the bin totals into lanes turns the 16-step walk over the bins
///     into a 4-step DPP prefix sum plus a ballot.
///
/// Measured on MI355X at [8, 896] top-16: 5.4us against aiter's 10.4us. The gap
/// widens with topk (3.3x at top-32) since only aiter's cost tracks it.

#pragma once

#ifndef USE_ROCM
#error "route_radix4_hip.cuh targets CDNA; it uses amdgcn DPP and wave64 ballots"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

inline constexpr uint32_t kRadix4NumExperts = 896;
inline constexpr uint32_t kRadix4TopK = 16;
inline constexpr uint32_t kRadix4Block = 256;
inline constexpr uint32_t kRadix4Wave = 64;

struct RouteRadix4Params {
  const void* __restrict__ scores;
  const void* __restrict__ bias;
  fp32_t* __restrict__ out_w;
  int32_t* __restrict__ out_i;
  uint32_t stride_scores;
  uint32_t stride_out;
  fp32_t routed_scaling_factor;
  bool renormalize;
};

namespace radix4 {

SGL_DEVICE float load_score(const bf16_t* p, int i) {
  return __uint_as_float(static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(p)[i]) << 16);
}

SGL_DEVICE float load_score(const fp32_t* p, int i) {
  return p[i];
}

/// Monotonic float -> uint32 map, so unsigned compares order the floats.
SGL_DEVICE uint32_t sortable(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

template <int CTRL, int RM, int BM, int N>
SGL_DEVICE void dpp_add_stage(uint32_t (&x)[N]) {
#pragma unroll
  for (int j = 0; j < N; ++j)
    x[j] += static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x[j]), CTRL, RM, BM, false));
}

/// Leaves the wave totals in lane 63. Deliberately does not broadcast: a caller
/// wanting 16 bin counts must reduce them to one scalar itself rather than
/// paying a readlane per word.
template <int N>
SGL_DEVICE void wave_sum_dpp(uint32_t (&x)[N]) {
  dpp_add_stage<0x111, 0xf, 0xf>(x);  // row_shr:1
  dpp_add_stage<0x112, 0xf, 0xf>(x);  // row_shr:2
  dpp_add_stage<0x114, 0xf, 0xe>(x);  // row_shr:4
  dpp_add_stage<0x118, 0xf, 0xc>(x);  // row_shr:8
  dpp_add_stage<0x142, 0xa, 0xf>(x);  // row_bcast:15
  dpp_add_stage<0x143, 0xc, 0xf>(x);  // row_bcast:31
}

}  // namespace radix4

template <typename T, int EXPERTS, int TOPK, int BLOCK>
__global__ __launch_bounds__(BLOCK) void route_radix4_kernel(__grid_constant__ const RouteRadix4Params params) {
  constexpr int WAVE = static_cast<int>(kRadix4Wave);
  constexpr int NWAVE = BLOCK / WAVE;
  constexpr int VPT = (EXPERTS + BLOCK - 1) / BLOCK;
  // A nibble caps a lane at 15 tallies per bin, hence one accumulator per 15
  // values held.
  constexpr int CHUNK = 15;
  constexpr int NACC = (VPT + CHUNK - 1) / CHUNK;
  static_assert(BLOCK % WAVE == 0, "block must be whole waves");
  static_assert(TOPK <= WAVE, "topk must fit in one lane-indexed row");
  static_assert(VPT <= 32, "alive mask is 32 bits");

  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid % WAVE;
  const int wid = tid / WAVE;
  const auto* srow = static_cast<const T*>(params.scores) + static_cast<size_t>(token) * params.stride_scores;
  const auto* sbias = static_cast<const T*>(params.bias);

  float sig[VPT];
  uint32_t key[VPT];
  // BLOCK * VPT overshoots EXPERTS; the tail slots stay out of the histogram and
  // out of the final compaction.
  uint32_t valid = (VPT >= 32) ? 0xffffffffu : ((1u << VPT) - 1u);

  uint32_t or_all = 0u, and_all = 0xffffffffu;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int e = tid + i * BLOCK;
    if (e < EXPERTS) {
      const float g = 1.0f / (1.0f + __expf(-radix4::load_score(srow, e)));
      sig[i] = g;
      key[i] = radix4::sortable(g + radix4::load_score(sbias, e));
      or_all |= key[i];
      and_all &= key[i];
    } else {
      sig[i] = 0.0f;
      key[i] = 0u;
      valid &= ~(1u << i);
    }
  }
  uint32_t alive = valid;

  __shared__ uint32_t s_hist[2][NWAVE][8];
  __shared__ uint32_t s_pre[2][NWAVE];
  __shared__ float s_w[TOPK];
  __shared__ int s_id[TOPK];
  __shared__ int s_cnt;

#pragma unroll
  for (int s = 32; s > 0; s >>= 1) {
    or_all |= __shfl_xor(or_all, s, WAVE);
    and_all &= __shfl_xor(and_all, s, WAVE);
  }
  if (lane == 0) {
    s_pre[0][wid] = or_all;
    s_pre[1][wid] = and_all;
  }
  __syncthreads();
#pragma unroll
  for (int w = 0; w < NWAVE; ++w) {
    or_all |= s_pre[0][w];
    and_all &= s_pre[1][w];
  }

  // Bits shared by every key carry no information; skipping them shortens the
  // search before it starts.
  const uint32_t diff = or_all ^ and_all;
  const int start = diff ? (31 - __clz(diff)) : -1;
  uint32_t pivot = (start >= 31) ? 0u : (and_all & ~((1u << (start + 1)) - 1u));

  int need = TOPK;
  // Lowest bit the pivot is resolved down to; the tail compares only the prefix
  // above it, so an early exit needs no extra bookkeeping.
  int bend = 0;

#pragma unroll 1
  for (int b = (start < 0) ? -4 : (start >> 2) << 2; b >= 0; b -= 4) {
    // Alternating buffers: a round's writes then only have to clear the reads of
    // the round before last, which an intervening barrier covers.
    const int buf = (b >> 2) & 1;

    uint64_t h[NACC];
#pragma unroll
    for (int a = 0; a < NACC; ++a) h[a] = 0ull;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      // Variable 64-bit shift, so tallying needs no dynamic array index; one
      // would push key[]/sig[] out to scratch.
      const uint32_t nib = (key[i] >> b) & 15u;
      h[i / CHUNK] += static_cast<uint64_t>((alive >> i) & 1u) << (4 * nib);
    }

    // Two bins per word: 64 lanes x 15 tops out at 960, so the low 16-bit field
    // cannot carry into the high one.
    uint32_t p[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint32_t lo = 0u, hi = 0u;
#pragma unroll
      for (int a = 0; a < NACC; ++a) {
        lo += static_cast<uint32_t>((h[a] >> (8 * j)) & 0xfull);
        hi += static_cast<uint32_t>((h[a] >> (8 * j + 4)) & 0xfull);
      }
      p[j] = lo | (hi << 16);
    }
    radix4::wave_sum_dpp(p);
    if (lane == WAVE - 1) {
#pragma unroll
      for (int j = 0; j < 8; ++j) s_hist[buf][wid][j] = p[j];
    }
    __syncthreads();

    // Reading bin 15-L in lane L transposes the packed histogram for the price
    // of one LDS access. Reversing the order makes "cumulative from the top bin
    // down" a plain prefix, the direction row_shr moves.
    int c = 0;
    if (lane < 16) {
      const int d = 15 - lane;
#pragma unroll
      for (int w = 0; w < NWAVE; ++w)
        c += static_cast<int>((s_hist[buf][w][d >> 1] >> ((d & 1) * 16)) & 0xffffu);
    }
    // Full bank_mask: unlike the reduce above, a prefix needs every lane of the
    // row correct, not just the last one.
    int cum = c;
    cum += __builtin_amdgcn_update_dpp(0, cum, 0x111, 0xf, 0xf, false);  // row_shr:1
    cum += __builtin_amdgcn_update_dpp(0, cum, 0x112, 0xf, 0xf, false);  // row_shr:2
    cum += __builtin_amdgcn_update_dpp(0, cum, 0x114, 0xf, 0xf, false);  // row_shr:4
    cum += __builtin_amdgcn_update_dpp(0, cum, 0x118, 0xf, 0xf, false);  // row_shr:8

    // cum rises monotonically with L, so the predicate turns on once and stays
    // on; its lowest lane is the bin holding the k-th key.
    const unsigned long long mk = __ballot(lane < 16 && cum >= need);
    const int L0 = __ffsll(mk) - 1;
    const int sel = 15 - L0;
    const int nsel = __builtin_amdgcn_readlane(c, L0);
    const int above = __builtin_amdgcn_readlane(cum, L0) - nsel;

    need -= above;
    pivot |= static_cast<uint32_t>(sel) << b;
    bend = b;
#pragma unroll
    for (int i = 0; i < VPT; ++i)
      if (((key[i] >> b) & 15u) != static_cast<uint32_t>(sel)) alive &= ~(1u << i);

    // Every survivor is a winner: the remaining bits cannot change the set.
    // Uniform across the block, since every thread scanned the same LDS totals.
    if (nsel == need) break;
  }

  // Winners span waves, so ballot cannot number them; an LDS bump counter can.
  // Order within the top-k is irrelevant to the sorting stage downstream.
  const uint32_t pmask = ~((1u << bend) - 1u);
  if (tid == 0) s_cnt = 0;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    if (((valid >> i) & 1u) && (key[i] & pmask) > pivot) {
      const int pos = atomicAdd(&s_cnt, 1);
      if (pos < TOPK) {
        s_w[pos] = sig[i];
        s_id[pos] = tid + i * BLOCK;
      }
    }
  }
  // Ties may only fill slots the strictly-greater keys did not take.
  __syncthreads();
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    if (((valid >> i) & 1u) && (key[i] & pmask) == pivot) {
      const int pos = atomicAdd(&s_cnt, 1);
      if (pos < TOPK) {
        s_w[pos] = sig[i];
        s_id[pos] = tid + i * BLOCK;
      }
    }
  }
  __syncthreads();

  if (tid < TOPK) {
    // The weight is the plain sigmoid; the bias only ever ranked the experts.
    float scale = params.routed_scaling_factor;
    if (params.renormalize) {
      float sum = 0.0f;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) sum += s_w[k];
      scale /= sum;
    }
    const size_t o = static_cast<size_t>(token) * params.stride_out + tid;
    params.out_w[o] = s_w[tid] * scale;
    params.out_i[o] = s_id[tid];
  }
}

}  // namespace sglang

struct RouteRadix4Kernel {
  static void run(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView out_w,
      const tvm::ffi::TensorView out_i,
      int64_t topk,
      double routed_scaling_factor,
      bool renormalize) {
    using namespace host;

    auto M_ = SymbolicSize{"num_tokens"};
    auto N_ = SymbolicSize{"num_experts"};
    auto K_ = SymbolicSize{"topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    auto score_dtype = SymbolicDType{};
    TensorMatcher({M_, N_})
        .with_dtype<bf16_t, fp32_t>(score_dtype)
        .with_device(device_)
        .with_strides({-1, 1})
        .verify(scores);
    // Rebinding the same symbolic dtype makes the bias track the scores.
    TensorMatcher({N_}).with_dtype<bf16_t, fp32_t>(score_dtype).with_device(device_).verify(bias);
    TensorMatcher({M_, K_}).with_dtype<fp32_t>().with_device(device_).verify(out_w);
    TensorMatcher({M_, K_}).with_dtype<int32_t>().with_device(device_).verify(out_i);

    RuntimeCheck(
        N_.unwrap() == sglang::kRadix4NumExperts && K_.unwrap() == sglang::kRadix4TopK &&
            topk == sglang::kRadix4TopK,
        "route_radix4 is specialized for N=896, K=16");

    const auto M = static_cast<uint32_t>(M_.unwrap());
    if (M == 0) return;

    const auto params = sglang::RouteRadix4Params{
        .scores = scores.data_ptr(),
        .bias = bias.data_ptr(),
        .out_w = static_cast<fp32_t*>(out_w.data_ptr()),
        .out_i = static_cast<int32_t*>(out_i.data_ptr()),
        .stride_scores = static_cast<uint32_t>(scores.stride(0)),
        .stride_out = static_cast<uint32_t>(out_w.stride(0)),
        .routed_scaling_factor = static_cast<fp32_t>(routed_scaling_factor),
        .renormalize = renormalize,
    };

    constexpr auto kExperts = static_cast<int>(sglang::kRadix4NumExperts);
    constexpr auto kTopK = static_cast<int>(sglang::kRadix4TopK);
    constexpr auto kBlock = static_cast<int>(sglang::kRadix4Block);
    const auto device = device_.unwrap();
    if (score_dtype.is_type<bf16_t>()) {
      LaunchKernel(M, kBlock, device)(sglang::route_radix4_kernel<bf16_t, kExperts, kTopK, kBlock>, params);
    } else {
      LaunchKernel(M, kBlock, device)(sglang::route_radix4_kernel<fp32_t, kExperts, kTopK, kBlock>, params);
    }
  }
};
