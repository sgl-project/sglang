/// Four-bit radix-select router for K3 routing on CDNA (gfx942/gfx950).
///
/// One block routes one token. Each thread keeps its slice of the 896 scores in
/// registers, and the block fixes four key bits of the pivot at a time: a round
/// tallies the still-live keys into a 16-bin histogram, accumulates from the top
/// bin down, and keeps the bin the k-th key falls in. The round count therefore
/// follows the key width rather than topk.
///
/// A round's cost is dominated by the vector->scalar->vector unit trip that
/// broadcasting a cross-lane count requires, and that trip is the same price
/// whether the round resolves one bit or four, hence 16 bins. What is left is
/// the per-round work that does not shrink as the experts are spread over more
/// waves; transposing the bin totals onto lanes turns the 16-step walk over
/// them into a 4-step DPP prefix sum plus a ballot.
///
/// Selection contract, shared with the CUDA router in route_radix.cuh: experts
/// rank by sigmoid(score) + bias, a NaN ranking value is floored so it can never
/// displace a finite expert, and among experts whose whole key is identical the
/// lowest id wins. Ties are settled by rank and the renorm divisor is summed
/// across threads rather than read back off the staged row, so the same input
/// always yields the same experts with the same weights, bit for bit. Where in a
/// row a winner lands is not stable, and the sorting stage downstream does not
/// read it.

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

/// Sets NaN to a very small value, so the expert it belongs to sorts last and is
/// never selected.
inline constexpr float kNanFloor = -1e30f;

SGL_DEVICE float nan_floor(float x) {
  return (x == x) ? x : kNanFloor;
}

/// Monotonic float -> uint32 map, so unsigned compares order the floats.
SGL_DEVICE uint32_t sortable(float f) {
  uint32_t u = __float_as_uint(f);
  // Map -0.0 and +0.0 to the same value.
  if (u == 0x80000000u) u = 0u;
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

template <int CTRL, int RM, int BM, int N>
SGL_DEVICE void dpp_add_stage(uint32_t (&x)[N]) {
#pragma unroll
  for (int j = 0; j < N; ++j)
    x[j] += static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x[j]), CTRL, RM, BM, false));
}

/// Wave-wide inclusive prefix sum over N uint32 at a time: afterwards lane L is
/// the sum of lanes 0..L and lane 63 is the total. The narrower bank masks on the
/// row_shr:4 and row_shr:8 stages switch off exactly the lanes whose source lane
/// falls outside the row, which would have added zero, so masking them or not
/// makes no difference.
template <int N>
SGL_DEVICE void wave_sum_dpp(uint32_t (&x)[N]) {
  dpp_add_stage<0x111, 0xf, 0xf>(x);  // row_shr:1
  dpp_add_stage<0x112, 0xf, 0xf>(x);  // row_shr:2
  dpp_add_stage<0x114, 0xf, 0xe>(x);  // row_shr:4
  dpp_add_stage<0x118, 0xf, 0xc>(x);  // row_shr:8
  dpp_add_stage<0x142, 0xa, 0xf>(x);  // row_bcast:15
  dpp_add_stage<0x143, 0xc, 0xf>(x);  // row_bcast:31
}

template <int CTRL, int RM, int BM>
SGL_DEVICE float dpp_fadd_stage(float x) {
  const int moved = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x), CTRL, RM, BM, false);
  return x + __builtin_bit_cast(float, moved);
}

/// Sums v within the wave and leaves the total in out[wid]. The ladder fixes the
/// order the addition happens in, so the same values give the same float on two
/// runs. Not __shfl_xor: that turns into six ds_bpermute round trips through LDS,
/// measured slower.
SGL_DEVICE void stage_wave_sum(float v, int lane, int wid, float* out) {
  v = dpp_fadd_stage<0x111, 0xf, 0xf>(v);  // row_shr:1
  v = dpp_fadd_stage<0x112, 0xf, 0xf>(v);  // row_shr:2
  v = dpp_fadd_stage<0x114, 0xf, 0xe>(v);  // row_shr:4
  v = dpp_fadd_stage<0x118, 0xf, 0xc>(v);  // row_shr:8
  v = dpp_fadd_stage<0x142, 0xa, 0xf>(v);  // row_bcast:15
  v = dpp_fadd_stage<0x143, 0xc, 0xf>(v);  // row_bcast:31
  if (lane == static_cast<int>(kRadix4Wave) - 1) out[wid] = v;
}

}  // namespace radix4

template <typename T, int EXPERTS, int TOPK, int BLOCK>
__global__ __launch_bounds__(BLOCK) void route_radix4_kernel(__grid_constant__ const RouteRadix4Params params) {
  constexpr int WAVE = static_cast<int>(kRadix4Wave);
  constexpr int NWAVE = BLOCK / WAVE;
  constexpr int VPT = (EXPERTS + BLOCK - 1) / BLOCK;
  // When the histogram below is tallied, each thread counts the bins of its VPT
  // (values per thread, how many experts one thread handles) experts into one
  // 64-bit register: 16 bins of 4 bits each, and an expert adds 1 to the 4 bits
  // it belongs to.
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
  // BLOCK * VPT overshoots EXPERTS; hence the mask.
  uint32_t valid = (VPT >= 32) ? 0xffffffffu : ((1u << VPT) - 1u);

  // Read the scores and the bias and compute sig[i] and key[i]: sig[i] is the
  // sigmoid without the bias and is what gets emitted, key[i] is sigmoid + bias
  // and is used to rank. Also accumulate the bitwise OR and AND of every key,
  // needed at diff below.
  uint32_t or_all = 0u, and_all = 0xffffffffu;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int e = tid + i * BLOCK;
    if (e < EXPERTS) {
      const float g = 1.0f / (1.0f + __expf(-radix4::load_score(srow, e)));
      sig[i] = g;
      key[i] = radix4::sortable(radix4::nan_floor(g + radix4::load_score(sbias, e)));
      or_all |= key[i];
      and_all &= key[i];
    } else {
      sig[i] = 0.0f;
      key[i] = 0u;
      valid &= ~(1u << i);
    }
  }
  uint32_t alive = valid;

  constexpr int PACK_WORDS = (VPT + 1) / 2;
  static_assert(BLOCK <= 0xffff, "epilogue counters are 16-bit fields");

  __shared__ uint32_t s_hist[2][NWAVE][8];
  __shared__ uint32_t s_pre[2][NWAVE];
  __shared__ uint32_t s_scan[NWAVE][2 * PACK_WORDS];
  __shared__ float s_w[TOPK];
  __shared__ float s_wsum[NWAVE];
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

  // The XOR marks the bits the keys disagree on (a disagreeing bit is 1 in the
  // OR and 0 in the AND); the highest one is where the search starts. Above it
  // every key is the same, and that value comes out of and_all as the initial
  // pivot.
  const uint32_t diff = or_all ^ and_all;
  const int start = diff ? (31 - __clz(diff)) : -1;
  uint32_t pivot = (start >= 31) ? 0u : (and_all & ~((1u << (start + 1)) - 1u));

  int need = TOPK;  // how many still have to be picked out of the live set
  // lowest bit the pivot is resolved down to
  int bend = 0;
  bool capped = (start < 0);

  // Main loop: fixes the TOPK-th largest key (the pivot) without sorting. A round
  // takes 4 bits as the bin index, tallies the histogram and accumulates from the
  // top bin down; the bin the need-th key falls in fixes those 4 pivot bits, the
  // higher bins are in for good and come off need, and alive narrows to that bin.
  // Survivors exactly equal to need exit early; still more than need at the
  // lowest bit sets capped.
#pragma unroll 1
  for (int b = (start < 0) ? -4 : (start >> 2) << 2; b >= 0; b -= 4) {
    // Two buffers used alternately, so a round's writes only have to wait for the
    // reads of the round before last.
    const int buf = (b >> 2) & 1;

    uint64_t h[NACC];
#pragma unroll
    for (int a = 0; a < NACC; ++a)
      h[a] = 0ull;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      const uint32_t bin = (key[i] >> b) & 15u;
      h[i / CHUNK] += static_cast<uint64_t>((alive >> i) & 1u) << (4 * bin);
    }

    // Spread the 16 four-bit counts over 8 uint32, two bins each.
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
      for (int j = 0; j < 8; ++j)
        s_hist[buf][wid][j] = p[j];
    }
    __syncthreads();

    // One LDS read puts each of the 16 bins on a lane of its own, which turns the
    // walk over the bins into a prefix sum across lanes. Lane L takes bin 15 - L:
    // with the bins reversed, "how many keys are in this bin or a higher one" is
    // exactly the direction row_shr adds in.
    int c = 0;
    if (lane < 16) {
      const int d = 15 - lane;
#pragma unroll
      for (int w = 0; w < NWAVE; ++w)
        c += static_cast<int>((s_hist[buf][w][d >> 1] >> ((d & 1) * 16)) & 0xffffu);
    }

    // A prefix sum of c across lanes, so cum is the number of keys in this lane's
    // bin and in every higher bin.
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

    // Every survivor is a winner: the remaining bits cannot change the set. The
    // test is uniform across the block, since every thread scanned the same LDS
    // totals.
    if (nsel == need) break;
    if (b == 0) capped = true;
  }

  const uint32_t pmask = ~((1u << bend) - 1u);
  // The renorm divisor is accumulated while the winners are picked instead of
  // read back off the staged row: the order that row gets filled in can differ
  // from run to run, and the order of the float additions with it, whereas
  // reducing across threads always adds in lane order.
  float wsum = 0.0f;
  if (!capped) {
    // The survivors exactly fill the quota, so a key wins as soon as it reaches
    // the pivot prefix. Winners span waves, so ballot cannot number them; an LDS
    // bump counter can.
    if (tid == 0) s_cnt = 0;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      if (((valid >> i) & 1u) && (key[i] & pmask) >= pivot) {
        const int pos = atomicAdd(&s_cnt, 1);
        if (pos < TOPK) {
          s_w[pos] = sig[i];
          s_id[pos] = tid + i * BLOCK;
          wsum += sig[i];
        }
      }
    }
    radix4::stage_wave_sum(wsum, lane, wid, s_wsum);
    __syncthreads();
  } else {
    // All 32 pivot bits are fixed and the survivors still outnumber the quota,
    // which means keys that are equal bit for bit. The lowest ids win then. The
    // expert a thread tid holds is tid + i * BLOCK, so for a fixed i the thread
    // order already is the id order, and the i's follow one another: scan each i
    // separately, then add the totals of the smaller i's.
    uint32_t win = 0u, eqm = 0u;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      if (!((valid >> i) & 1u)) continue;
      const uint32_t k = key[i] & pmask;
      if (k > pivot)
        win |= 1u << i;
      else if (k == pivot)
        eqm |= 1u << i;
    }

    // Packed into uint32 the same way: the low 16 bits hold i = 2j, the high 16
    // bits i = 2j + 1; the first PACK_WORDS hold the ties, the next PACK_WORDS the
    // outright winners. own keeps the original value, because the prefix the scan
    // returns includes the thread itself and has to come back off later.
    uint32_t own[2 * PACK_WORDS], sc[2 * PACK_WORDS];
#pragma unroll
    for (int j = 0; j < PACK_WORDS; ++j) {
      own[j] = ((eqm >> (2 * j)) & 1u) | (((eqm >> (2 * j + 1)) & 1u) << 16);
      own[PACK_WORDS + j] = ((win >> (2 * j)) & 1u) | (((win >> (2 * j + 1)) & 1u) << 16);
      sc[j] = own[j];
      sc[PACK_WORDS + j] = own[PACK_WORDS + j];
    }
    // Unlike the histogram above, this caller wants every lane's prefix, not just
    // the wave total the last lane carries.
    radix4::wave_sum_dpp(sc);
    if (lane == WAVE - 1) {
#pragma unroll
      for (int j = 0; j < 2 * PACK_WORDS; ++j)
        s_scan[wid][j] = sc[j];
    }
    __syncthreads();

    // Neither 16-bit half ever reaches 2^16, so the packed uint32 can be added
    // and subtracted whole without carrying between the two counts it holds.
    uint32_t tot[2 * PACK_WORDS];
#pragma unroll
    for (int j = 0; j < 2 * PACK_WORDS; ++j) {
      uint32_t below = 0u, all = 0u;
#pragma unroll
      for (int w = 0; w < NWAVE; ++w) {
        const uint32_t v = s_scan[w][j];
        if (w < wid) below += v;
        all += v;
      }
      sc[j] += below - own[j];  // inclusive within the wave -> exclusive block-wide
      tot[j] = all;
    }

    uint32_t eq_rank[VPT], win_rank[VPT];
    uint32_t eq_ahead = 0u, win_ahead = 0u;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      const int j = i >> 1, sh = 16 * (i & 1);
      eq_rank[i] = eq_ahead + ((sc[j] >> sh) & 0xffffu);
      win_rank[i] = win_ahead + ((sc[PACK_WORDS + j] >> sh) & 0xffffu);
      eq_ahead += (tot[j] >> sh) & 0xffffu;
      win_ahead += (tot[PACK_WORDS + j] >> sh) & 0xffffu;
    }

    // By this point the outright winners can never fill the quota on their own
    // and the ties always cover the rest, so the two ranks address exactly the
    // TOPK positions of the output row, with no gaps and no collisions.
    const uint32_t need_eq = (win_ahead < TOPK) ? (TOPK - win_ahead) : 0u;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      const bool taken = ((win >> i) & 1u) || (((eqm >> i) & 1u) && eq_rank[i] < need_eq);
      const uint32_t slot = win_rank[i] + (eq_rank[i] < need_eq ? eq_rank[i] : need_eq);
      if (taken && slot < TOPK) {
        s_w[slot] = sig[i];
        s_id[slot] = tid + i * BLOCK;
        wsum += sig[i];
      }
    }
    radix4::stage_wave_sum(wsum, lane, wid, s_wsum);
    __syncthreads();
  }

  if (tid < TOPK) {
    // The weight is the plain sigmoid; the bias only ever ranked the experts.
    float scale = params.routed_scaling_factor;
    if (params.renormalize) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < NWAVE; ++w)
        sum += s_wsum[w];
      // Every sigmoid underflows to zero on a row of saturated scores, and a row
      // of NaN sums to NaN; neither may turn a finite weight into an inf.
      scale /= (sum > 0.0f) ? sum : 1.0f;
    }
    const size_t o = static_cast<size_t>(token) * params.stride_out + tid;
    params.out_w[o] = s_w[tid] * scale;
    params.out_i[o] = s_id[tid];
  }
}

struct RouteRadix4Kernel {
  static void
  run(const tvm::ffi::TensorView scores,
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
        N_.unwrap() == kRadix4NumExperts && K_.unwrap() == kRadix4TopK && topk == kRadix4TopK,
        "route_radix4 is specialized for N=896, K=16");

    const auto M = static_cast<uint32_t>(M_.unwrap());
    if (M == 0) return;

    const auto params = RouteRadix4Params{
        .scores = scores.data_ptr(),
        .bias = bias.data_ptr(),
        .out_w = static_cast<fp32_t*>(out_w.data_ptr()),
        .out_i = static_cast<int32_t*>(out_i.data_ptr()),
        .stride_scores = static_cast<uint32_t>(scores.stride(0)),
        .stride_out = static_cast<uint32_t>(out_w.stride(0)),
        .routed_scaling_factor = static_cast<fp32_t>(routed_scaling_factor),
        .renormalize = renormalize,
    };

    constexpr auto kExperts = static_cast<int>(kRadix4NumExperts);
    constexpr auto kTopK = static_cast<int>(kRadix4TopK);
    constexpr auto kBlock = static_cast<int>(kRadix4Block);
    const auto device = device_.unwrap();
    if (score_dtype.is_type<bf16_t>()) {
      LaunchKernel(M, kBlock, device)(route_radix4_kernel<bf16_t, kExperts, kTopK, kBlock>, params);
    } else {
      LaunchKernel(M, kBlock, device)(route_radix4_kernel<fp32_t, kExperts, kTopK, kBlock>, params);
    }
  }
};

}  // namespace sglang
