// Qwen3.8 decode MoE front in one launch (SGLANG_Q38_FUSED_ROUTING):
//   * CTA 0 selects each row's top-10 of 512 BF16 router logits, computes the
//     RenormalizeNaive weights, and builds the trtllm-gen routing metadata for a
//     prepared-routing (DA) body: the routed MoE then runs only its GEMMs and
//     FlashInfer's routing kernel is skipped.
//   * CTAs 1.. quantize the routed activations to NVFP4 (2560 wide, 16-element
//     E4M3 block scales in linear layout, one per-tensor global scale).
//
// Adapted from upstream sgl-project/sglang 0fc97bc07c (#48586,
// kimi_k3_fused_routing.cuh): the routing/quant CTA split, the early PDL
// trigger and the metadata contract (trtllm-gen routingPermutation: local
// experts in id order, each padded to whole tiles, padding rows of the token
// map set to -1). Retargeted from sigmoid+bias top-16 of 896 with MXFP8 quant
// to Qwen3.8's softmax top-10 of 512 with NVFP4 quant, and for M <= 32 rows one
// CTA builds the metadata from per-expert token bitmasks (no grid-wide counter).
//
// Bit-exactness contract (checked against the unfused chain by the H502 bench):
//   * Selection is FlashInfer 0.7.2's TopKRedType order: key =
//     (cub-twiddled BF16 bits << 16) | (65535 - expert), largest first, so ties
//     go to the smaller expert id and column k holds the k-th largest.
//   * Weights are FlashInfer's scalar calcSoftmax over the 10 selected BF16
//     logits (RoutingKernel.cuh), the same source expressions and cg::reduce
//     calls; this TU builds with --use_fast_math as FlashInfer's MoE module does.
//   * Within an expert, rows are in token order, as the Block and DynBlock
//     routing kernels place them.
//   * The quant is FlashInfer's CuTe-DSL fp4_quantize (linear SF layout, the
//     standard fast-math recipe): every float operation is inline PTX, so the
//     TU's fast-math flags do not reach it.
//
// 33-512 rows (q38_fused_routing_cluster_kernel): the rows are split in token
// order over one thread-block cluster of C CTAs (CTA r owns a contiguous run of
// at most 512 / C rows, a warp per row, a 64-bit row mask per expert). Each CTA
// publishes its per-expert row counts in shared memory; after a cluster barrier
// every CTA reads all C over DSMEM, so it knows each expert's total and how many
// of the expert's rows earlier CTAs own. Every CTA then runs the same exclusive
// scan of padded tiles over the 512 experts, writes the metadata of the experts
// e with e % C == r, and places its own rows at expert offset + earlier CTAs'
// rows + earlier rows of its own mask: rows of an expert stay in token order, as
// in the 1-32 row kernel. Keys, the exact top-10 order, weights and quant are
// the same as above (the selection caches three keys per lane, see select_row).
// FlashInfer's Cluster (<= 256 rows) and Coop (> 256 rows) kernels place an
// expert's rows by shared-memory atomics, so their order inside an expert is not
// fixed; every other metadata field is a function of the routing and equals theirs.

#pragma once

#include <sgl_kernel/tensor.h>  // TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // RuntimeCheck, div_ceil

#include <sgl_kernel/utils.cuh>  // LaunchKernel, SGL_DEVICE, PDL helpers, bf16_t

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>

namespace sglang {

namespace q38_fused_routing {

inline constexpr int kNumExperts = 512;
inline constexpr int kTopK = 10;
inline constexpr int kHidden = 2560;
inline constexpr int kSfVec = 16;
inline constexpr int kBlocksPerRow = kHidden / kSfVec;  // 160 NVFP4 blocks per row
inline constexpr int kThreads = 512;
inline constexpr int kWarps = kThreads / 32;
inline constexpr int kMaxTokens = 32;  // one bit per row in the expert masks
inline constexpr int kTokensPerWarp = kMaxTokens / kWarps;
inline constexpr int kPerLane = kNumExperts / 32;  // 16 contiguous logits per lane
static_assert(kThreads == kNumExperts, "one metadata thread per expert");
static_assert(kMaxTokens % kWarps == 0);
static_assert(kPerLane == 16, "two 16-byte loads per lane");

// 33-512 rows: one cluster of CTAs routes and builds the metadata, the CTAs after it quantize.
inline constexpr int kClusterMaxTokens = 512;
inline constexpr int kExpertWarps = kNumExperts / 32;  // threads 0-511 own one expert each

struct Params {
  const bf16_t* __restrict__ logits;  // [M, 512]
  const bf16_t* __restrict__ x;       // [M, 2560]
  const float* __restrict__ global_scale;
  uint8_t* __restrict__ x_fp4;   // [M, 1280]
  uint8_t* __restrict__ x_sf;    // [M, 160] E4M3
  int32_t* __restrict__ ids;     // [M, 10]
  bf16_t* __restrict__ weights;  // [M, 10]
  int32_t* __restrict__ expanded_idx_to_permuted_idx;
  int32_t* __restrict__ permuted_idx_to_token_idx;
  int32_t* __restrict__ cta_idx_xy_to_batch_idx;
  int32_t* __restrict__ cta_idx_xy_to_mn_limit;
  int32_t* __restrict__ num_non_exiting_ctas;
  int32_t* __restrict__ total_num_padded_tokens;
  int32_t* __restrict__ num_tokens_per_expert;
  int64_t logits_stride;
  int64_t x_stride;
  int32_t num_tokens;
  int32_t tile_tokens;
};

// cub::Traits<__nv_bfloat16>::TwiddleIn / TwiddleOut.
SGL_DEVICE uint32_t twiddle_in(uint32_t bits) {
  return bits ^ ((bits & 0x8000u) ? 0xFFFFu : 0x8000u);
}
SGL_DEVICE uint32_t twiddle_out(uint32_t bits) {
  return bits ^ ((bits & 0x8000u) ? 0x8000u : 0xFFFFu);
}

// FlashInfer TopKRedType<__nv_bfloat16>::makeCmpVal.
SGL_DEVICE uint32_t make_key(uint32_t bf16_bits, int expert) {
  return (twiddle_in(bf16_bits & 0xFFFFu) << 16) | (0xFFFFu & static_cast<uint32_t>(65535 - expert));
}

// FlashInfer calcSoftmax(warp, score, laneIdx, NumTopExperts) with DataType =
// __nv_bfloat16 (RoutingKernel.cuh), kept expression for expression.
template <typename Warp>
SGL_DEVICE bf16_t renormalize_naive(Warp const& warp, bf16_t score, int lane) {
  namespace cg = cooperative_groups;
  float maxScore = -INFINITY;
  if (lane < kTopK) {
    float si = __bfloat162float(score);
    maxScore = si >= maxScore ? si : maxScore;
  }
  maxScore = cg::reduce(warp, maxScore, cg::greater<float>());

  float sumScore = 0.f;
  float newScore = 0.f;
  if (lane < kTopK) {
    newScore = __bfloat162float(score) - maxScore;
    newScore = expf(newScore);
    sumScore += newScore;
  }
  sumScore = cg::reduce(warp, sumScore, cg::plus<float>());

  if (lane < kTopK) {
    score = __float2bfloat16(newScore / sumScore);
  }
  return score;
}

// ---- NVFP4 quant: FlashInfer CuTe-DSL fp4_quantize, instruction for instruction ----

SGL_DEVICE uint32_t bf16x2_max(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("max.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

SGL_DEVICE float bf16x2_hmax_to_f32(uint32_t v) {
  float r;
  asm("{\n"
      "  .reg .b32 lo, hi;\n"
      "  .reg .f32 f0, f1;\n"
      "  and.b32 lo, %1, 0xFFFF;\n"
      "  shr.b32 hi, %1, 16;\n"
      "  shl.b32 lo, lo, 16;\n"
      "  shl.b32 hi, hi, 16;\n"
      "  mov.b32 f0, lo;\n"
      "  mov.b32 f1, hi;\n"
      "  max.f32 %0, f0, f1;\n"
      "}"
      : "=f"(r)
      : "r"(v));
  return r;
}

SGL_DEVICE float rcp_approx_ftz(float a) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}

SGL_DEVICE float mul_f32(float a, float b) {
  float r;
  asm("mul.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

SGL_DEVICE uint32_t cvt_f32_to_e4m3(float a) {
  uint32_t r;
  asm("{\n"
      "  .reg .b16 pair;\n"
      "  .reg .f32 zero;\n"
      "  mov.f32 zero, 0f00000000;\n"
      "  cvt.rn.satfinite.e4m3x2.f32 pair, zero, %1;\n"
      "  cvt.u32.u16 %0, pair;\n"
      "}"
      : "=r"(r)
      : "f"(a));
  return r;
}

// fp4_common.nvfp4_compute_output_scale: rcp(f32(e4m3) * rcp(global_scale)), 0 for a zero scale.
SGL_DEVICE float nvfp4_output_scale(uint32_t fp8, float global_scale) {
  float r;
  asm("{\n"
      "  .reg .pred p_zero;\n"
      "  .reg .b16 fp8_pair;\n"
      "  .reg .b32 h2_32;\n"
      "  .reg .b16 h_lo, h_hi;\n"
      "  .reg .f32 scale_f32, rcp_gs, product, result;\n"
      "  cvt.u16.u32 fp8_pair, %1;\n"
      "  cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;\n"
      "  mov.b32 {h_lo, h_hi}, h2_32;\n"
      "  cvt.f32.f16 scale_f32, h_lo;\n"
      "  rcp.approx.ftz.f32 rcp_gs, %2;\n"
      "  mul.f32 product, scale_f32, rcp_gs;\n"
      "  rcp.approx.ftz.f32 result, product;\n"
      "  setp.eq.f32 p_zero, scale_f32, 0f00000000;\n"
      "  selp.f32 %0, 0f00000000, result, p_zero;\n"
      "}"
      : "=f"(r)
      : "r"(fp8), "f"(global_scale));
  return r;
}

// fp4_common.bfloat2_to_float2_scaled.
SGL_DEVICE void bf16x2_to_f32x2_scaled(uint32_t v, float s, float& f0, float& f1) {
  asm("{\n"
      "  .reg .b32 lo, hi;\n"
      "  .reg .f32 a0, a1;\n"
      "  and.b32 lo, %2, 0xFFFF;\n"
      "  shr.b32 hi, %2, 16;\n"
      "  shl.b32 lo, lo, 16;\n"
      "  shl.b32 hi, hi, 16;\n"
      "  mov.b32 a0, lo;\n"
      "  mov.b32 a1, hi;\n"
      "  mul.f32 %0, a0, %3;\n"
      "  mul.f32 %1, a1, %3;\n"
      "}"
      : "=f"(f0), "=f"(f1)
      : "r"(v), "f"(s));
}

// fp4_common.cvt_e2m1x8_f32: element 2j in the low nibble of byte j.
SGL_DEVICE uint32_t cvt_e2m1x8(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7) {
  uint32_t r;
  asm("{\n"
      "  .reg .b8 b0, b1, b2, b3;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "  mov.b32 %0, {b0, b1, b2, b3};\n"
      "}"
      : "=r"(r)
      : "f"(v0), "f"(v1), "f"(v2), "f"(v3), "f"(v4), "f"(v5), "f"(v6), "f"(v7));
  return r;
}

// L2-coherent 16-byte load: the producer may still have been running at launch (PDL).
SGL_DEVICE uint4 load_v4(const void* p) {
  return __ldcg(reinterpret_cast<const uint4*>(p));
}

// One 16-element block: quantization_cute_dsl_utils._quantize_nvfp4_from_h2x8_bfloat
// (bfloat2_max_abs_8 tree, the fast-math scale, nvfp4_compute_output_scale).
SGL_DEVICE void quantize_block(const Params& p, int block) {
  const int row = block / kBlocksPerRow;
  const int col = block - row * kBlocksPerRow;
  const bf16_t* src = p.x + row * p.x_stride + col * kSfVec;
  const uint4 a = load_v4(src);
  const uint4 b = load_v4(src + 8);
  const uint32_t h[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};

  const uint32_t max01 = bf16x2_max(h[0] & 0x7FFF7FFFu, h[1] & 0x7FFF7FFFu);
  const uint32_t max23 = bf16x2_max(h[2] & 0x7FFF7FFFu, h[3] & 0x7FFF7FFFu);
  const uint32_t max45 = bf16x2_max(h[4] & 0x7FFF7FFFu, h[5] & 0x7FFF7FFFu);
  const uint32_t max67 = bf16x2_max(h[6] & 0x7FFF7FFFu, h[7] & 0x7FFF7FFFu);
  const float block_max = bf16x2_hmax_to_f32(bf16x2_max(bf16x2_max(max01, max23), bf16x2_max(max45, max67)));

  const float global_scale = p.global_scale[0];
  const float fp4_max_rcp = rcp_approx_ftz(6.0f);
  const float scale_float = mul_f32(global_scale, mul_f32(block_max, fp4_max_rcp));
  const uint32_t fp8 = cvt_f32_to_e4m3(scale_float) & 0xFFu;
  const float out_scale = nvfp4_output_scale(fp8, global_scale);

  float s[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    bf16x2_to_f32x2_scaled(h[i], out_scale, s[2 * i], s[2 * i + 1]);
  }
  const uint32_t lo = cvt_e2m1x8(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);
  const uint32_t hi = cvt_e2m1x8(s[8], s[9], s[10], s[11], s[12], s[13], s[14], s[15]);
  *reinterpret_cast<uint2*>(p.x_fp4 + static_cast<int64_t>(row) * (kHidden / 2) + col * 8) = make_uint2(lo, hi);
  p.x_sf[static_cast<int64_t>(row) * kBlocksPerRow + col] = static_cast<uint8_t>(fp8);
}

}  // namespace q38_fused_routing

template <bool kUsePDL>
__global__ __launch_bounds__(q38_fused_routing::kThreads)  //
    void q38_fused_routing_kernel(const __grid_constant__ q38_fused_routing::Params params) {
  using namespace q38_fused_routing;
  namespace cg = cooperative_groups;
  const int tid = threadIdx.x;
  const int M = params.num_tokens;

  if (blockIdx.x > 0) {
    // Quant CTAs read the activations the router GEMM's predecessors wrote.
    device::PDLWaitPrimary<kUsePDL>();
    device::PDLTriggerSecondary<kUsePDL>();
    const int block = (blockIdx.x - 1) * kThreads + tid;
    if (block < M * kBlocksPerRow) quantize_block(params, block);
    return;
  }

  __shared__ uint32_t s_mask[kNumExperts];       // bit t: row t routes to the expert
  __shared__ int32_t s_padded_off[kNumExperts];  // first permuted row of the expert
  __shared__ uint32_t s_warp_sum[kWarps];

  s_mask[tid] = 0u;
  __syncthreads();
  device::PDLWaitPrimary<kUsePDL>();

  const int lane = tid % 32;
  const int warp_id = tid / 32;
  auto warp = cg::tiled_partition<32>(cg::this_thread_block());

  int my_expert[kTokensPerWarp];
#pragma unroll
  for (int j = 0; j < kTokensPerWarp; ++j) {
    my_expert[j] = 0;
    const int t = warp_id + j * kWarps;  // warp-uniform
    if (t >= M) continue;
    const bf16_t* row = params.logits + t * params.logits_stride + lane * kPerLane;
    const uint4 a = load_v4(row);
    const uint4 b = load_v4(row + 8);
    const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    uint32_t keys[kPerLane];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      keys[2 * i] = make_key(w[i] & 0xFFFFu, lane * kPerLane + 2 * i);
      keys[2 * i + 1] = make_key(w[i] >> 16, lane * kPerLane + 2 * i + 1);
    }
    // Ten rounds of the warp-wide max key; a taken key drops to 0, below every real key.
    uint32_t mine = 0u;
#pragma unroll
    for (int r = 0; r < kTopK; ++r) {
      uint32_t local = keys[0];
#pragma unroll
      for (int i = 1; i < kPerLane; ++i) local = max(local, keys[i]);
      const uint32_t top = __reduce_max_sync(0xFFFFFFFFu, local);
#pragma unroll
      for (int i = 0; i < kPerLane; ++i) keys[i] = keys[i] == top ? 0u : keys[i];
      mine = lane == r ? top : mine;
    }
    const int expert = 65535 - static_cast<int>(mine & 0xFFFFu);
    const bf16_t score = __ushort_as_bfloat16(static_cast<unsigned short>(twiddle_out(mine >> 16)));
    const bf16_t weight = renormalize_naive(warp, score, lane);
    if (lane < kTopK) {
      params.weights[t * kTopK + lane] = weight;
      params.ids[t * kTopK + lane] = expert;
      atomicOr(&s_mask[expert], 1u << t);
      my_expert[j] = expert;
    }
  }
  __syncthreads();
  // Early trigger (as upstream): the routed MoE's griddepcontrol.wait still waits
  // for this whole grid, metadata included.
  device::PDLTriggerSecondary<kUsePDL>();

  // Metadata: thread e owns expert e (all experts are local at EP1).
  const int tile = params.tile_tokens;
  const uint32_t mask = s_mask[tid];
  const int count = __popc(mask);
  const int num_ctas = (count + tile - 1) / tile;
  int incl = num_ctas;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int n = __shfl_up_sync(0xFFFFFFFFu, incl, o);
    if (lane >= o) incl += n;
  }
  if (lane == 31) s_warp_sum[warp_id] = static_cast<uint32_t>(incl);
  __syncthreads();
  const int base = static_cast<int>(__reduce_add_sync(0xFFFFFFFFu, lane < warp_id ? s_warp_sum[lane] : 0u));
  const int cta_off = base + incl - num_ctas;
  s_padded_off[tid] = cta_off * tile;
  params.num_tokens_per_expert[tid] = count;
  const int limit = cta_off * tile + count;
  for (int c = 0; c < num_ctas; ++c) {
    const int cta = cta_off + c;
    const int tile_end = (cta + 1) * tile;
    const int mn_limit = min(tile_end, limit);
    params.cta_idx_xy_to_batch_idx[cta] = tid;
    params.cta_idx_xy_to_mn_limit[cta] = mn_limit;
    for (int p = mn_limit; p < tile_end; ++p) params.permuted_idx_to_token_idx[p] = -1;
  }
  if (tid == kThreads - 1) {
    const int total_ctas = cta_off + num_ctas;
    params.num_non_exiting_ctas[0] = total_ctas;
    params.total_num_padded_tokens[0] = total_ctas * tile;
  }
  __syncthreads();

  // Permutation: rows of an expert in token order (the Block/DynBlock kernels' order).
#pragma unroll
  for (int j = 0; j < kTokensPerWarp; ++j) {
    const int t = warp_id + j * kWarps;
    if (t >= M || lane >= kTopK) continue;
    const int e = my_expert[j];
    const int permuted = s_padded_off[e] + __popc(s_mask[e] & ((1u << t) - 1u));
    params.expanded_idx_to_permuted_idx[t * kTopK + lane] = permuted;
    params.permuted_idx_to_token_idx[permuted] = t;
  }
}

namespace q38_fused_routing {

// Inserts k into a lane's sorted cache a1 > a2 > a3 (0 = empty; real keys are never 0).
SGL_DEVICE void cache_insert(uint32_t k, uint32_t& a1, uint32_t& a2, uint32_t& a3) {
  const uint32_t d1 = min(a1, k);
  a1 = max(a1, k);
  const uint32_t d2 = min(a2, d1);
  a2 = max(a2, d1);
  a3 = max(a3, d2);
}

// One row's top-10 and RenormalizeNaive weights, called by a whole warp; returns this lane's expert (lanes 0-9).
// Same keys, same exact descending top-10 and same weight expressions as the 1-32 row kernel. Each lane caches
// its three largest keys, so a round is one warp max; a lane whose cache runs dry refills it from its keys below
// the last one it gave up (a lane holds 16 keys and gives up at most 10, so a dry cache always has more).
template <typename Warp>
SGL_DEVICE int select_row(const Params& p, Warp const& warp, int t, int lane) {
  const bf16_t* row = p.logits + t * p.logits_stride + lane * kPerLane;
  const uint4 a = load_v4(row);
  const uint4 b = load_v4(row + 8);
  const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  uint32_t keys[kPerLane];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    keys[2 * i] = make_key(w[i] & 0xFFFFu, lane * kPerLane + 2 * i);
    keys[2 * i + 1] = make_key(w[i] >> 16, lane * kPerLane + 2 * i + 1);
  }
  uint32_t a1 = 0u, a2 = 0u, a3 = 0u;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) cache_insert(keys[i], a1, a2, a3);
  uint32_t last = 0u;  // the smallest key this lane has given up
  uint32_t mine = 0u;
#pragma unroll
  for (int r = 0; r < kTopK; ++r) {
    const uint32_t top = __reduce_max_sync(0xFFFFFFFFu, a1);
    mine = lane == r ? top : mine;
    const bool won = a1 == top;
    last = won ? top : last;
    a1 = won ? a2 : a1;
    a2 = won ? a3 : a2;
    a3 = won ? 0u : a3;
    if (r + 1 < kTopK && __any_sync(0xFFFFFFFFu, a1 == 0u)) {
      if (a1 == 0u) {
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) cache_insert(keys[i] < last ? keys[i] : 0u, a1, a2, a3);
      }
    }
  }
  const int expert = 65535 - static_cast<int>(mine & 0xFFFFu);
  const bf16_t score = __ushort_as_bfloat16(static_cast<unsigned short>(twiddle_out(mine >> 16)));
  const bf16_t weight = renormalize_naive(warp, score, lane);
  if (lane < kTopK) {
    p.weights[t * kTopK + lane] = weight;
    p.ids[t * kTopK + lane] = expert;
  }
  return expert;
}

}  // namespace q38_fused_routing

// 33-512 rows. CTAs 0..kCtas-1 are one cluster that routes and builds the metadata; the CTAs after it quantize.
// The grid is a whole number of clusters; the quant CTAs never touch a cluster barrier.
template <bool kUsePDL, int kCtas, int kThreads>
__global__ void __cluster_dims__(kCtas, 1, 1) __launch_bounds__(kThreads)  //
    q38_fused_routing_cluster_kernel(const __grid_constant__ q38_fused_routing::Params params) {
  using namespace q38_fused_routing;
  namespace cg = cooperative_groups;
  constexpr int kWarpsC = kThreads / 32;
  constexpr int kRowsPerCta = kClusterMaxTokens / kCtas;
  constexpr int kRowsPerWarp = kRowsPerCta / kWarpsC;
  static_assert(kRowsPerCta <= 64, "one u64 mask bit per row");
  static_assert(kRowsPerWarp >= 1 && kRowsPerCta % kWarpsC == 0);
  static_assert(kThreads >= kNumExperts && kThreads <= 1024);
  const int tid = threadIdx.x;
  const int M = params.num_tokens;

  if (blockIdx.x >= kCtas) {
    device::PDLWaitPrimary<kUsePDL>();
    device::PDLTriggerSecondary<kUsePDL>();
    const int block = (blockIdx.x - kCtas) * kThreads + tid;
    if (block < M * kBlocksPerRow) quantize_block(params, block);
    return;
  }

  __shared__ unsigned long long s_mask[kNumExperts];  // bit j: this CTA's row j routes to the expert
  __shared__ int32_t s_count[kNumExperts];            // this CTA's rows per expert, read by the cluster
  __shared__ int32_t s_offset[kNumExperts];           // permuted index of this CTA's first row of the expert
  __shared__ int32_t s_warp_sum[kExpertWarps];

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int lane = tid % 32;
  const int warp_id = tid / 32;
  auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  const int rows_per_cta = (M + kCtas - 1) / kCtas;
  const int row0 = rank * rows_per_cta;
  const int rows = max(0, min(rows_per_cta, M - row0));

  if (tid < kNumExperts) s_mask[tid] = 0ull;
  __syncthreads();
  device::PDLWaitPrimary<kUsePDL>();

  int my_expert[kRowsPerWarp];
#pragma unroll
  for (int j = 0; j < kRowsPerWarp; ++j) {
    my_expert[j] = 0;
    const int r = warp_id + j * kWarpsC;  // warp-uniform
    if (r >= rows) continue;
    const int expert = select_row(params, warp, row0 + r, lane);
    if (lane < kTopK) {
      atomicOr(&s_mask[expert], 1ull << r);
      my_expert[j] = expert;
    }
  }
  __syncthreads();
  // Early trigger, as the 1-32 row kernel: the dependent's griddepcontrol.wait still waits for this whole grid.
  device::PDLTriggerSecondary<kUsePDL>();

  if (tid < kNumExperts) s_count[tid] = __popcll(s_mask[tid]);
  cluster.sync();

  // Thread e (< 512): expert e's total, and its rows owned by lower ranks (lower tokens).
  int total = 0;
  int before = 0;
  if (tid < kNumExperts) {
#pragma unroll
    for (int q = 0; q < kCtas; ++q) {
      const int c = cluster.map_shared_rank(s_count, q)[tid];
      before += q < rank ? c : 0;
      total += c;
    }
  }
  // Done with remote shared memory; the matching wait is the last thing before exit.
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");

  // Exclusive scan of padded tiles over the experts, identical in every CTA.
  const int tile = params.tile_tokens;
  const int num_ctas = (total + tile - 1) / tile;
  int incl = num_ctas;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int n = __shfl_up_sync(0xFFFFFFFFu, incl, o);
    if (lane >= o) incl += n;
  }
  if (warp_id < kExpertWarps && lane == 31) s_warp_sum[warp_id] = incl;
  __syncthreads();
  const int base = static_cast<int>(
      __reduce_add_sync(0xFFFFFFFFu, lane < warp_id && lane < kExpertWarps ? static_cast<uint32_t>(s_warp_sum[lane]) : 0u));
  const int cta_off = base + incl - num_ctas;

  if (tid < kNumExperts) {
    s_offset[tid] = cta_off * tile + before;
    if (tid % kCtas == rank) {
      params.num_tokens_per_expert[tid] = total;
      const int limit = cta_off * tile + total;
      for (int c = 0; c < num_ctas; ++c) {
        const int cta = cta_off + c;
        const int tile_end = (cta + 1) * tile;
        const int mn_limit = min(tile_end, limit);
        params.cta_idx_xy_to_batch_idx[cta] = tid;
        params.cta_idx_xy_to_mn_limit[cta] = mn_limit;
        for (int p = mn_limit; p < tile_end; ++p) params.permuted_idx_to_token_idx[p] = -1;
      }
    }
    if (rank == 0 && tid == kNumExperts - 1) {
      const int total_ctas = cta_off + num_ctas;
      params.num_non_exiting_ctas[0] = total_ctas;
      params.total_num_padded_tokens[0] = total_ctas * tile;
    }
  }
  __syncthreads();

#pragma unroll
  for (int j = 0; j < kRowsPerWarp; ++j) {
    const int r = warp_id + j * kWarpsC;
    if (r >= rows || lane >= kTopK) continue;
    const int e = my_expert[j];
    const int t = row0 + r;
    const int permuted = s_offset[e] + __popcll(s_mask[e] & ((1ull << r) - 1ull));
    params.expanded_idx_to_permuted_idx[t * kTopK + lane] = permuted;
    params.permuted_idx_to_token_idx[permuted] = t;
  }
  // Other CTAs may still be reading s_count over DSMEM.
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// kCtas x kThreads: the 33-512 row cluster's shape. 16 x 512 measured fastest on VR200 (H560 bench); a 16-CTA
// cluster is non-portable, so a device that cannot run one gets the portable 8 x 1024 shape (same outputs).
template <bool kUsePDL, int kCtas = 16, int kThreads = 512>
struct Q38FusedRoutingKernel {
  static bool shape_supported() {
    if constexpr (kCtas <= 8) {
      return true;
    } else {
      static const bool ok = [] {
        const auto kernel = q38_fused_routing_cluster_kernel<kUsePDL, kCtas, kThreads>;
        cudaLaunchConfig_t config = {};
        config.gridDim = dim3(kCtas);
        config.blockDim = dim3(kThreads);
        int clusters = 0;
        const bool fits =
            cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) == cudaSuccess &&
            cudaOccupancyMaxActiveClusters(&clusters, kernel, &config) == cudaSuccess && clusters > 0;
        if (!fits) (void)cudaGetLastError();
        return fits;
      }();
      return ok;
    }
  }

  // CTAs per routing cluster this process launches for 33-512 rows.
  static int64_t cluster_ctas() {
    return shape_supported() ? kCtas : 8;
  }

  template <int C, int T>
  static void launch_cluster(int64_t M, DLDevice device, const q38_fused_routing::Params& params) {
    // One routing cluster, then the quant CTAs rounded up to whole clusters (__cluster_dims__ is compiled in).
    const int64_t quant_ctas = host::div_ceil(M * q38_fused_routing::kBlocksPerRow, int64_t{T});
    const auto grid = static_cast<uint32_t>(C + host::div_ceil(quant_ctas, int64_t{C}) * C);
    host::LaunchKernel(grid, T, device).enable_pdl(kUsePDL)(q38_fused_routing_cluster_kernel<kUsePDL, C, T>, params);
  }

  static int64_t max_num_ctas(int64_t num_tokens, int64_t tile) {
    // FlashInfer Routing::getMaxNumCtasInBatchDim(num_tokens, 10, 512, tile).
    int64_t remaining = num_tokens * q38_fused_routing::kTopK;
    const int64_t filled = remaining < q38_fused_routing::kNumExperts ? remaining : q38_fused_routing::kNumExperts;
    remaining -= filled;
    return filled + (remaining > 0 ? remaining / tile : 0);
  }

  static void
  run(const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView global_scale,
      const tvm::ffi::TensorView x_fp4,
      const tvm::ffi::TensorView x_sf,
      const tvm::ffi::TensorView ids,
      const tvm::ffi::TensorView weights,
      const tvm::ffi::TensorView expanded_idx_to_permuted_idx,
      const tvm::ffi::TensorView permuted_idx_to_token_idx,
      const tvm::ffi::TensorView cta_idx_xy_to_batch_idx,
      const tvm::ffi::TensorView cta_idx_xy_to_mn_limit,
      const tvm::ffi::TensorView num_non_exiting_ctas,
      const tvm::ffi::TensorView total_num_padded_tokens,
      const tvm::ffi::TensorView num_tokens_per_expert,
      int64_t tile_tokens) {
    using namespace host;
    namespace R = q38_fused_routing;

    auto M_ = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M_, R::kNumExperts}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device).verify(logits);
    TensorMatcher({M_, R::kHidden}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(global_scale);
    TensorMatcher({M_, R::kHidden / 2}).with_strides({R::kHidden / 2, 1}).with_dtype<uint8_t>().with_device(device).verify(
        x_fp4);
    TensorMatcher({M_, R::kBlocksPerRow})
        .with_strides({R::kBlocksPerRow, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(x_sf);
    TensorMatcher({M_, R::kTopK}).with_strides({R::kTopK, 1}).with_dtype<int32_t>().with_device(device).verify(ids);
    TensorMatcher({M_, R::kTopK}).with_strides({R::kTopK, 1}).with_dtype<bf16_t>().with_device(device).verify(weights);
    for (const auto* t :
         {&expanded_idx_to_permuted_idx,
          &permuted_idx_to_token_idx,
          &cta_idx_xy_to_batch_idx,
          &cta_idx_xy_to_mn_limit,
          &num_non_exiting_ctas,
          &total_num_padded_tokens,
          &num_tokens_per_expert}) {
      TensorMatcher({-1}).with_strides({1}).with_dtype<int32_t>().with_device(device).verify(*t);
    }

    const int64_t M = M_.unwrap();
    RuntimeCheck(M >= 1 && M <= R::kClusterMaxTokens, "q38_fused_routing: 1-512 rows");
    RuntimeCheck(tile_tokens >= 1 && tile_tokens <= 256, "q38_fused_routing: tile_tokens must be in [1, 256]");
    const int64_t max_ctas = max_num_ctas(M, tile_tokens);
    RuntimeCheck(expanded_idx_to_permuted_idx.size(0) >= M * R::kTopK, "q38_fused_routing: expanded map too small");
    RuntimeCheck(
        permuted_idx_to_token_idx.size(0) >= max_ctas * tile_tokens + 1, "q38_fused_routing: token map too small");
    RuntimeCheck(cta_idx_xy_to_batch_idx.size(0) >= max_ctas, "q38_fused_routing: CTA batch map too small");
    RuntimeCheck(cta_idx_xy_to_mn_limit.size(0) >= max_ctas, "q38_fused_routing: CTA limit map too small");
    RuntimeCheck(num_non_exiting_ctas.size(0) >= 1 && total_num_padded_tokens.size(0) >= 1, "q38_fused_routing: scalars");
    RuntimeCheck(num_tokens_per_expert.size(0) >= R::kNumExperts, "q38_fused_routing: per-expert counts too small");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0 && logits.stride(0) % 8 == 0,
        "q38_fused_routing: logits rows must be 16-byte aligned");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && x.stride(0) % 8 == 0,
        "q38_fused_routing: activation rows must be 16-byte aligned");
    RuntimeCheck(reinterpret_cast<uintptr_t>(x_fp4.data_ptr()) % 8 == 0, "q38_fused_routing: x_fp4 alignment");

    const auto params = R::Params{
        .logits = static_cast<const bf16_t*>(logits.data_ptr()),
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .global_scale = static_cast<const float*>(global_scale.data_ptr()),
        .x_fp4 = static_cast<uint8_t*>(x_fp4.data_ptr()),
        .x_sf = static_cast<uint8_t*>(x_sf.data_ptr()),
        .ids = static_cast<int32_t*>(ids.data_ptr()),
        .weights = static_cast<bf16_t*>(weights.data_ptr()),
        .expanded_idx_to_permuted_idx = static_cast<int32_t*>(expanded_idx_to_permuted_idx.data_ptr()),
        .permuted_idx_to_token_idx = static_cast<int32_t*>(permuted_idx_to_token_idx.data_ptr()),
        .cta_idx_xy_to_batch_idx = static_cast<int32_t*>(cta_idx_xy_to_batch_idx.data_ptr()),
        .cta_idx_xy_to_mn_limit = static_cast<int32_t*>(cta_idx_xy_to_mn_limit.data_ptr()),
        .num_non_exiting_ctas = static_cast<int32_t*>(num_non_exiting_ctas.data_ptr()),
        .total_num_padded_tokens = static_cast<int32_t*>(total_num_padded_tokens.data_ptr()),
        .num_tokens_per_expert = static_cast<int32_t*>(num_tokens_per_expert.data_ptr()),
        .logits_stride = logits.stride(0),
        .x_stride = x.stride(0),
        .num_tokens = static_cast<int32_t>(M),
        .tile_tokens = static_cast<int32_t>(tile_tokens),
    };
    if (M <= R::kMaxTokens) {
      const auto grid = static_cast<uint32_t>(1 + div_ceil(M * R::kBlocksPerRow, int64_t{R::kThreads}));
      LaunchKernel(grid, R::kThreads, device.unwrap()).enable_pdl(kUsePDL)(q38_fused_routing_kernel<kUsePDL>, params);
    } else {
      if (shape_supported()) {
        launch_cluster<kCtas, kThreads>(M, device.unwrap(), params);
      } else {
        launch_cluster<8, 1024>(M, device.unwrap(), params);
      }
    }
  }
};

}  // namespace sglang
