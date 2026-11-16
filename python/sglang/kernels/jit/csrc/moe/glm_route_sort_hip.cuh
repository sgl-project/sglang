/// GLM-5.2 routing plus AITER's general MoE sort in one launch (gfx942/gfx950).
///
/// Routing: 256 experts, ungrouped sigmoid + correction bias, top-8, with the
/// fused shared expert appended as column 8 (global id 256). It reproduces what
/// the aiter path produces bit for bit: aiter's biased_grouped_topk register
/// kernel (exp2f + rcpf sigmoid, winners by key descending with ties broken by
/// legacy_topk_tie_rank, weights the pre-bias sigmoid renormalized by a serial
/// sum in output order), then fused_append_shared_experts (shared weight
/// appended, CUDA-graph padding rows reduced to routed id 0 and weight 0).
///
/// Sort: what aiter's moe_sorting_opus writes for those ids under an EP expert
/// mask that keeps global experts [routed_begin, routed_begin + num_routed) and
/// the shared expert: local expert ids (routed g - routed_begin, shared
/// num_routed), a block_m-padded block per expert with routes, routes within an
/// expert in (token, slot) order, padding rows (topk << 24) | M with weight 0.
/// Opus keeps one route per (token, expert), so a padding row's eight expert-0
/// routes become one; which slot it keeps is its own business (the weight is 0
/// and both GEMMs read the slot from the same sort), and this kernel keeps the
/// last. Rows past num_valid_ids[0] are left unwritten, as aiter leaves them.
///
/// Scores are the router logits, fp32 or bf16. aiter routes in fp32 once the
/// bias is fp32, so bf16 logits are widened on load as the caller's upcast would.
///
/// Routing blocks take tokens_per_block tokens each, one wave per token. With
/// one routing block it sorts from LDS and no block waits on another; with
/// more, each stores its rows at the coherent point and the last to arrive
/// reads them back and sorts. Wave scheduling bounds the single-block case: a
/// CU's four SIMDs route its tokens, so past a few tokens per SIMD spreading
/// over CUs wins despite the hand-off. The remaining blocks zero one row of
/// the MoE accumulation buffer each.

#pragma once

#ifndef USE_ROCM
#error "glm_route_sort_hip.cuh targets CDNA; it uses wave64 DPP and ballots"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

inline constexpr int kGlmExperts = 256;
inline constexpr int kGlmRoutedTopK = 8;
inline constexpr int kGlmTopK = kGlmRoutedTopK + 1;
inline constexpr int kGlmWave = 64;
inline constexpr int kGlmBlock = 1024;
inline constexpr int kGlmWaves = kGlmBlock / kGlmWave;
inline constexpr int kGlmMaxTokens = 64;
inline constexpr int kGlmTokensPerWave = kGlmMaxTokens / kGlmWaves;
inline constexpr int kGlmMaxRoutes = kGlmMaxTokens * kGlmTopK;
inline constexpr int kGlmRouteWords = (kGlmMaxRoutes + 31) / 32;
// Every routed expert local (no EP) plus the shared one.
inline constexpr int kGlmMaxLocal = kGlmExperts + 1;
// aiter's C_LOG2E is a double literal, so its -C_LOG2E * x is a double product
// rounded once to float; a float product rounds differently.
inline constexpr double kGlmSigmoidLog2E = 1.44269504088896340736;

struct GlmRouteSortParams {
  const void* __restrict__ scores;
  const fp32_t* __restrict__ bias;
  fp32_t* __restrict__ out_w;
  int32_t* __restrict__ out_i;
  uint32_t stride_scores;
  bool scores_bf16;
  fp32_t routed_scaling_factor;
  fp32_t shared_weight;
  bool renormalize;
  // [arrival] for multi-block routing, zeroed once; the sorting block restores it.
  int32_t* __restrict__ sort_ws;
  int32_t* __restrict__ sorted_token_ids;
  fp32_t* __restrict__ sorted_weights;
  int32_t* __restrict__ sorted_expert_ids;
  int32_t* __restrict__ num_valid_ids;
  void* __restrict__ moe_buf;
  uint32_t moe_dim;
  const int32_t* __restrict__ num_non_padded;
  int32_t num_tokens;
  // Blocks [0, route_blocks) route tokens_per_block tokens each; the rest zero moe_buf rows.
  int32_t route_blocks;
  int32_t tokens_per_block;
  int32_t routed_begin;
  int32_t num_routed;
  int32_t block_m;
};

namespace glm_route {

// aiter topk_softmax_kernels_group.cu: legacy_topk_lane_rank / legacy_topk_tie_rank.
SGL_DEVICE int legacy_lane_rank(int lane) {
  const int low2 = (lane & 3) ^ ((lane & 4) ? 3 : 0);
  const int bit2 = (((lane >> 2) ^ (lane >> 3)) & 1) << 2;
  return ((~lane) & 0x38) | bit2 | low2;
}

SGL_DEVICE int legacy_tie_rank(int expert) {
  const int vec4_id = expert >> 2;
  const int lane = vec4_id & 63;
  const int local_pos = ((vec4_id >> 6) << 2) | (expert & 3);
  return (legacy_lane_rank(lane) << 5) | local_pos;
}

/// Monotonic float -> uint32 map, so unsigned max picks the larger float. Keys
/// are never -0 (sigmoid > 0), so +0 and -0 cannot split a tie aiter would see.
SGL_DEVICE uint32_t ordered_bits(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

template <int CTRL, int ROW_MASK, int BANK_MASK>
SGL_DEVICE uint32_t dpp_move(uint32_t x, uint32_t identity) {
  return static_cast<uint32_t>(
      __builtin_amdgcn_update_dpp(static_cast<int>(identity), static_cast<int>(x), CTRL, ROW_MASK, BANK_MASK, false));
}

/// Whole-wave reduction under `op` (identity `identity`), returned uniformly.
/// The wave must be fully active.
template <typename Op>
SGL_DEVICE uint32_t wave_reduce_dpp(uint32_t x, uint32_t identity, Op op) {
  x = op(x, dpp_move<0x111, 0xf, 0xf>(x, identity));  // row_shr:1
  x = op(x, dpp_move<0x112, 0xf, 0xf>(x, identity));  // row_shr:2
  x = op(x, dpp_move<0x114, 0xf, 0xe>(x, identity));  // row_shr:4
  x = op(x, dpp_move<0x118, 0xf, 0xc>(x, identity));  // row_shr:8
  x = op(x, dpp_move<0x142, 0xa, 0xf>(x, identity));  // row_bcast:15
  x = op(x, dpp_move<0x143, 0xc, 0xf>(x, identity));  // row_bcast:31
  return static_cast<uint32_t>(__builtin_amdgcn_readlane(static_cast<int>(x), kGlmWave - 1));
}

SGL_DEVICE int expert_of(int lane, int q) {
  return (lane + (q >> 1) * kGlmWave) * 2 + (q & 1);
}

/// One wave routes one token from its preloaded scores and writes the
/// [kGlmTopK] row to LDS and to the output tensors.
template <bool kWriteThrough>
SGL_DEVICE void route_token(
    const GlmRouteSortParams& p, const float (&x)[4], const float (&bias)[4], int token, bool padded,
    int32_t* s_ids, float* s_w) {
  const int lane = threadIdx.x & (kGlmWave - 1);
  // Lane L owns experts 2L, 2L + 1, 128 + 2L, 129 + 2L, as aiter's EPL=4 kernel does.
  uint32_t key[4];
  float sig[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float s = __builtin_amdgcn_rcpf(1.0f + exp2f(static_cast<float>(-kGlmSigmoidLog2E * static_cast<double>(x[q]))));
    sig[q] = s;
    key[q] = ordered_bits(s + bias[q]);
  }
  const auto max_op = [](uint32_t a, uint32_t b) { return a > b ? a : b; };
  const auto min_op = [](uint32_t a, uint32_t b) { return a < b ? a : b; };

  int out_id = 0;
  float out_w = 0.0f;
  float total = 0.0f;
#pragma unroll 1
  for (int k = 0; k < kGlmRoutedTopK; ++k) {
    uint32_t v = key[0];
#pragma unroll
    for (int q = 1; q < 4; ++q)
      v = max_op(v, key[q]);
    const uint32_t best = wave_reduce_dpp(v, 0u, max_op);
    int n_eq = 0;
    int cand = 0;
#pragma unroll
    for (int q = 0; q < 4; ++q)
      if (key[q] == best) {
        ++n_eq;
        cand = expert_of(lane, q);
      }
    const uint64_t holders = __ballot(n_eq > 0);
    const int first = static_cast<int>(__builtin_ctzll(holders));
    int id;
    if ((holders & (holders - 1)) == 0 && __builtin_amdgcn_readlane(n_eq, first) == 1) {
      id = __builtin_amdgcn_readlane(cand, first);
    } else {
      // An exact tie on the key: the smallest tie rank wins, and tie ranks are
      // unique, so (rank << 8) | expert names the winner.
      uint32_t t = 0xffffffffu;
#pragma unroll
      for (int q = 0; q < 4; ++q)
        if (key[q] == best) {
          const int e = expert_of(lane, q);
          t = min_op(t, (static_cast<uint32_t>(legacy_tie_rank(e)) << 8) | static_cast<uint32_t>(e));
        }
      id = static_cast<int>(wave_reduce_dpp(t, 0xffffffffu, min_op) & 0xffu);
    }
    // id is wave-uniform: its owner retires it and hands out its sigmoid.
    const int owner = (id >> 1) & (kGlmWave - 1);
    const int slot = ((id >> 7) << 1) | (id & 1);
    float mine = sig[0];
#pragma unroll
    for (int q = 1; q < 4; ++q)
      if (q == slot) mine = sig[q];
#pragma unroll
    for (int q = 0; q < 4; ++q)
      if (q == slot && lane == owner) key[q] = 0u;
    const float w = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(mine), owner));
    total += w;
    if (lane == k) {
      out_id = id;
      out_w = w;
    }
  }
  if (p.renormalize)
    out_w = total > 0.0f ? out_w * (p.routed_scaling_factor / total) : 0.0f;
  else
    out_w *= p.routed_scaling_factor;

  if (lane < kGlmTopK) {
    int id;
    float w;
    if (lane == kGlmRoutedTopK) {
      id = kGlmExperts;
      w = padded ? 0.0f : p.shared_weight;
    } else {
      id = padded ? 0 : out_id;
      w = padded ? 0.0f : out_w;
    }
    const int r = token * kGlmTopK + lane;
    s_ids[r] = id;
    s_w[r] = w;
    if constexpr (kWriteThrough) {
      // The sorting block on another XCD reads the row back: store it at the
      // coherent point, so retiring the stores releases it without an L2 write-back.
      __hip_atomic_store(&p.out_i[r], id, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(&p.out_w[r], w, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } else {
      p.out_i[r] = id;
      p.out_w[r] = w;
    }
  }
}

SGL_DEVICE void zero_row(void* buf, int row, int dim) {
  int4* v = reinterpret_cast<int4*>(buf) + static_cast<long long>(row) * dim / 8;
  const int nvec = dim / 8;
  const int4 zero = make_int4(0, 0, 0, 0);
  for (int i = threadIdx.x; i < nvec; i += kGlmBlock)
    v[i] = zero;
}

SGL_DEVICE int local_expert(int g, int routed_begin, int num_routed) {
  if (g == kGlmExperts) return num_routed;
  const int l = g - routed_begin;
  return (l >= 0 && l < num_routed) ? l : -1;
}

}  // namespace glm_route

template <bool kMultiBlock>
__global__ void __launch_bounds__(kGlmBlock) glm_route_sort_kernel(const GlmRouteSortParams p) {
  if (static_cast<int>(blockIdx.x) >= p.route_blocks) {
    glm_route::zero_row(p.moe_buf, blockIdx.x - p.route_blocks, static_cast<int>(p.moe_dim));
    return;
  }

  __shared__ int32_t s_ids[kGlmMaxRoutes];
  __shared__ float s_w[kGlmMaxRoutes];
  __shared__ int s_cnt[kGlmMaxLocal];
  __shared__ int s_off[kGlmMaxLocal];
  __shared__ uint32_t s_bits[kGlmMaxLocal][kGlmRouteWords];
  __shared__ int s_wtot[kGlmWaves];
  __shared__ int s_leader;

  const int tid = threadIdx.x;
  const int lane = tid & (kGlmWave - 1);
  const int wave = tid / kGlmWave;
  const int M = p.num_tokens;
  const int num_local = p.num_routed + 1;
  int n_real = M;
  if (p.num_non_padded != nullptr) n_real = *p.num_non_padded < M ? *p.num_non_padded : M;
  const int t_begin = static_cast<int>(blockIdx.x) * p.tokens_per_block;
  const int t_end = t_begin + p.tokens_per_block < M ? t_begin + p.tokens_per_block : M;

  // Issue every score load this wave needs before routing its first token.
  float bias[4];
  float x[kGlmTokensPerWave][4];
#pragma unroll
  for (int q = 0; q < 4; ++q)
    bias[q] = p.bias[glm_route::expert_of(lane, q)];
#pragma unroll
  for (int i = 0; i < kGlmTokensPerWave; ++i) {
    const int t = t_begin + wave + i * kGlmWaves;
    if (t < t_end) {
      const size_t row = static_cast<size_t>(t) * p.stride_scores;
      if (p.scores_bf16) {
        const uint16_t* bits = static_cast<const uint16_t*>(p.scores) + row;
#pragma unroll
        for (int q = 0; q < 4; ++q)
          x[i][q] = __uint_as_float(static_cast<uint32_t>(bits[glm_route::expert_of(lane, q)]) << 16);
      } else {
        const fp32_t* vals = static_cast<const fp32_t*>(p.scores) + row;
#pragma unroll
        for (int q = 0; q < 4; ++q)
          x[i][q] = vals[glm_route::expert_of(lane, q)];
      }
    }
  }

  for (int i = tid; i < num_local * kGlmRouteWords; i += kGlmBlock)
    (&s_bits[0][0])[i] = 0u;
  for (int i = tid; i < num_local; i += kGlmBlock)
    s_cnt[i] = 0;
#pragma unroll
  for (int i = 0; i < kGlmTokensPerWave; ++i) {
    const int t = t_begin + wave + i * kGlmWaves;
    if (t < t_end) glm_route::route_token<kMultiBlock>(p, x[i], bias, t, t >= n_real, s_ids, s_w);
  }

  if constexpr (kMultiBlock) {
    __builtin_amdgcn_s_waitcnt(0x0f70);  // vmcnt(0): this wave's row stores have retired
    __syncthreads();
    if (tid == 0) {
      const bool last = atomicAdd(&p.sort_ws[0], 1) == p.route_blocks - 1;
      // This XCD's L2 may hold stale lines of the other blocks' rows.
      if (last) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
      s_leader = last ? 1 : 0;
    }
    __syncthreads();
    if (!s_leader) return;
    if (tid < M * kGlmTopK) {
      s_ids[tid] = p.out_i[tid];
      s_w[tid] = p.out_w[tid];
    }
    if (tid == 0) p.sort_ws[0] = 0;
  }
  __syncthreads();

  // Count and mark every route; a padding row keeps only its last expert-0 route.
  const int routes = M * kGlmTopK;
  const int first_pad_route = n_real * kGlmTopK;
  const int r = tid;
  int l = -1;
  if (r < routes && !(r >= first_pad_route && (r % kGlmTopK) < kGlmRoutedTopK - 1)) {
    l = glm_route::local_expert(s_ids[r], p.routed_begin, p.num_routed);
    if (l >= 0) {
      atomicAdd(&s_cnt[l], 1);
      atomicOr(&s_bits[l][r >> 5], 1u << (r & 31));
    }
  }
  __syncthreads();

  // Padded expert prefix in expert order, one expert per thread.
  const int bm = p.block_m;
  const int e = tid;
  const int cnt = e < num_local ? s_cnt[e] : 0;
  const int padded = (cnt + bm - 1) / bm * bm;
  int incl = padded;
#pragma unroll
  for (int off = 1; off < kGlmWave; off <<= 1) {
    const int y = __shfl_up(incl, off, kGlmWave);
    if (lane >= off) incl += y;
  }
  if (lane == kGlmWave - 1) s_wtot[wave] = incl;
  __syncthreads();
  int begin = incl - padded;
  for (int w = 0; w < wave; ++w)
    begin += s_wtot[w];
  if (e < num_local) {
    s_off[e] = begin;
    const int pad_id = (kGlmTopK << 24) | M;
    for (int row = begin + cnt; row < begin + padded; ++row) {
      p.sorted_token_ids[row] = pad_id;
      p.sorted_weights[row] = 0.0f;
    }
    for (int b = begin / bm; b < (begin + padded) / bm; ++b)
      p.sorted_expert_ids[b] = e;
  }
  if (tid == kGlmBlock - 1) {
    p.num_valid_ids[0] = begin + padded;
    p.num_valid_ids[1] = M;
  }
  __syncthreads();

  // A route's rank within its expert is the number of marked routes before it.
  if (l >= 0) {
    const int word = r >> 5;
    int rank = __popc(s_bits[l][word] & ((1u << (r & 31)) - 1u));
    for (int k = 0; k < word; ++k)
      rank += __popc(s_bits[l][k]);
    const int t = r / kGlmTopK;
    const int pos = s_off[l] + rank;
    p.sorted_token_ids[pos] = ((r - t * kGlmTopK) << 24) | t;
    p.sorted_weights[pos] = s_w[r];
  }
}

struct GlmRouteSortKernel {
  static void
  run(const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView out_w,
      const tvm::ffi::TensorView out_i,
      const tvm::ffi::TensorView sort_ws,
      const tvm::ffi::TensorView sorted_token_ids,
      const tvm::ffi::TensorView sorted_weights,
      const tvm::ffi::TensorView sorted_expert_ids,
      const tvm::ffi::TensorView num_valid_ids,
      const tvm::ffi::TensorView moe_buf,
      const tvm::ffi::TensorView num_non_padded,
      int64_t routed_begin,
      int64_t num_routed,
      int64_t block_m,
      int64_t tokens_per_block,
      double routed_scaling_factor,
      double shared_weight,
      bool renormalize) {
    using namespace host;

    auto M_ = SymbolicSize{"num_tokens"};
    auto N_ = SymbolicSize{"num_experts"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();
    auto score_dtype = SymbolicDType{};
    TensorMatcher({M_, N_})
        .with_dtype<fp32_t, bf16_t>(score_dtype)
        .with_device(device_)
        .with_strides({-1, 1})
        .verify(scores);
    TensorMatcher({N_}).with_dtype<fp32_t>().with_device(device_).with_strides({1}).verify(bias);
    TensorMatcher({M_, kGlmTopK}).with_dtype<fp32_t>().with_device(device_).with_strides({kGlmTopK, 1}).verify(out_w);
    TensorMatcher({M_, kGlmTopK}).with_dtype<int32_t>().with_device(device_).with_strides({kGlmTopK, 1}).verify(out_i);
    auto S_ = SymbolicSize{"sorted_len"};
    TensorMatcher({S_}).with_dtype<int32_t>().with_device(device_).verify(sorted_token_ids);
    TensorMatcher({S_}).with_dtype<fp32_t>().with_device(device_).verify(sorted_weights);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device_).verify(sorted_expert_ids);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device_).verify(num_valid_ids);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device_).verify(sort_ws);

    RuntimeCheck(N_.unwrap() == kGlmExperts, "glm_route_sort is specialized for 256 routed experts");
    const int64_t M = M_.unwrap();
    RuntimeCheck(M <= kGlmMaxTokens, "glm_route_sort routes at most 64 tokens");
    RuntimeCheck(
        routed_begin >= 0 && num_routed >= 1 && routed_begin + num_routed <= kGlmExperts,
        "local routed experts must be a range of the 256");
    RuntimeCheck(block_m >= 1, "block_m must be positive");
    RuntimeCheck(
        tokens_per_block >= 1 && tokens_per_block <= kGlmWaves * kGlmTokensPerWave,
        "a block routes 1 to 64 tokens");
    // aiter's moe_sorting allocation for the 257 global experts.
    const int64_t global_experts = kGlmExperts + 1;
    const int64_t padded_rows = M * kGlmTopK + global_experts * block_m - kGlmTopK;
    const int64_t capacity = (padded_rows + block_m - 1) / block_m * block_m;
    RuntimeCheck(S_.unwrap() == capacity, "sorted rows must be sized as aiter's moe_sorting sizes them");
    RuntimeCheck(sorted_expert_ids.size(0) * block_m == capacity, "one expert id per block_m sorted rows");
    if (M == 0) return;

    void* moe_ptr = nullptr;
    uint32_t moe_dim = 0;
    if (moe_buf.numel() > 0) {
      TensorMatcher({M_, -1}).with_dtype<bf16_t>().with_device(device_).with_strides({-1, 1}).verify(moe_buf);
      RuntimeCheck(
          moe_buf.stride(0) == moe_buf.size(1) && moe_buf.size(1) % 8 == 0 &&
              reinterpret_cast<uintptr_t>(moe_buf.data_ptr()) % 16 == 0,
          "moe_buf must be packed, 16-byte aligned rows of a multiple of 8 bf16");
      moe_ptr = moe_buf.data_ptr();
      moe_dim = static_cast<uint32_t>(moe_buf.size(1));
    }
    const int32_t* non_padded_ptr = nullptr;
    if (num_non_padded.numel() > 0) {
      TensorMatcher({-1}).with_dtype<int32_t>().with_device(device_).verify(num_non_padded);
      non_padded_ptr = static_cast<const int32_t*>(num_non_padded.data_ptr());
    }

    const auto route_blocks = static_cast<int32_t>((M + tokens_per_block - 1) / tokens_per_block);
    RuntimeCheck(route_blocks == 1 || sort_ws.numel() >= 1, "multi-block routing needs the arrival counter");
    const auto params = GlmRouteSortParams{
        .scores = scores.data_ptr(),
        .bias = static_cast<const fp32_t*>(bias.data_ptr()),
        .out_w = static_cast<fp32_t*>(out_w.data_ptr()),
        .out_i = static_cast<int32_t*>(out_i.data_ptr()),
        .stride_scores = static_cast<uint32_t>(scores.stride(0)),
        .scores_bf16 = score_dtype.is_type<bf16_t>(),
        .routed_scaling_factor = static_cast<fp32_t>(routed_scaling_factor),
        .shared_weight = static_cast<fp32_t>(shared_weight),
        .renormalize = renormalize,
        .sort_ws = static_cast<int32_t*>(sort_ws.data_ptr()),
        .sorted_token_ids = static_cast<int32_t*>(sorted_token_ids.data_ptr()),
        .sorted_weights = static_cast<fp32_t*>(sorted_weights.data_ptr()),
        .sorted_expert_ids = static_cast<int32_t*>(sorted_expert_ids.data_ptr()),
        .num_valid_ids = static_cast<int32_t*>(num_valid_ids.data_ptr()),
        .moe_buf = moe_ptr,
        .moe_dim = moe_dim,
        .num_non_padded = non_padded_ptr,
        .num_tokens = static_cast<int32_t>(M),
        .route_blocks = route_blocks,
        .tokens_per_block = static_cast<int32_t>(tokens_per_block),
        .routed_begin = static_cast<int32_t>(routed_begin),
        .num_routed = static_cast<int32_t>(num_routed),
        .block_m = static_cast<int32_t>(block_m),
    };
    const auto grid = static_cast<uint32_t>(route_blocks + (moe_ptr != nullptr ? M : 0));
    if (route_blocks > 1)
      LaunchKernel(grid, kGlmBlock, device_.unwrap())(glm_route_sort_kernel<true>, params);
    else
      LaunchKernel(grid, kGlmBlock, device_.unwrap())(glm_route_sort_kernel<false>, params);
  }
};

}  // namespace sglang
