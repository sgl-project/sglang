#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cfloat>
#include <climits>
#include <cstdint>
#include <cuda_bf16.h>

namespace {

// Fixed constants for the Inkling model
static constexpr int kInklingRoutedExperts = 256;
static constexpr int kInklingSharedExperts = 2;
static constexpr int kInklingTotalExperts = kInklingRoutedExperts + kInklingSharedExperts;
static constexpr int kInklingTopK = 6;
static constexpr int kInklingTopPow2 = 8;
static constexpr int kInklingWarpSize = 32;
static constexpr int kInklingValuesPerLane = kInklingRoutedExperts / kInklingWarpSize;

__device__ __forceinline__ float inkling_sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ bool inkling_score_better(float score, int idx, float best_score, int best_idx) {
  return score > best_score || (score == best_score && idx < best_idx);
}

// Order-preserving uint32 key of an fp32 (same total order as the triton
// kernel's fpval_to_key): flip sign bit for positives, all bits for negatives.
__device__ __forceinline__ uint32_t inkling_fp_key(float f) {
  const uint32_t u = __float_as_uint(f);
  return u ^ (static_cast<uint32_t>(-static_cast<int32_t>(u >> 31)) | 0x80000000u);
}

// FlashInfer routed-MoE pack: low 16 bits = bf16(weight) bits (round-to-nearest-even,
// same as torch/triton `.to(bfloat16)`), high 16 bits = int16 expert id.
__device__ __forceinline__ int32_t inkling_pack_routed(int32_t id, float w) {
  const uint32_t wbits = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(w)));
  return static_cast<int32_t>((static_cast<uint32_t>(id) << 16) | wbits);
}

template <int WarpsPerBlock, bool ReturnPacked>
__launch_bounds__(kInklingWarpSize* WarpsPerBlock) __global__ void inkling_gate_topk_renorm_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const float* __restrict__ global_scale,
    float* __restrict__ routed_w,
    float* __restrict__ shared_w,
    int64_t* __restrict__ indices,
    int32_t* __restrict__ packed,
    int64_t M,
    int64_t logits_stride_m,
    float route_scale) {
  const int lane = threadIdx.x;
  const int warp_in_block = threadIdx.y;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * WarpsPerBlock + warp_in_block;
  if (row >= M) {
    return;
  }

  const int64_t row_base = row * logits_stride_m;
  float local_scores[kInklingValuesPerLane];
#pragma unroll
  for (int i = 0; i < kInklingValuesPerLane; ++i) {
    const int expert = lane + i * kInklingWarpSize;
    const float raw = logits[row_base + expert];
    local_scores[i] = inkling_sigmoid(raw) + bias[expert];
  }

  int selected_idx[kInklingTopK];
#pragma unroll
  for (int k = 0; k < kInklingTopK; ++k) {
    float best_score = -FLT_MAX;
    int best_idx = INT_MAX;
#pragma unroll
    for (int i = 0; i < kInklingValuesPerLane; ++i) {
      const int expert = lane + i * kInklingWarpSize;
      const float score = local_scores[i];
      if (inkling_score_better(score, expert, best_score, best_idx)) {
        best_score = score;
        best_idx = expert;
      }
    }

#pragma unroll
    for (int offset = kInklingWarpSize / 2; offset > 0; offset >>= 1) {
      const float other_score = __shfl_xor_sync(0xffffffff, best_score, offset);
      const int other_idx = __shfl_xor_sync(0xffffffff, best_idx, offset);
      if (inkling_score_better(other_score, other_idx, best_score, best_idx)) {
        best_score = other_score;
        best_idx = other_idx;
      }
    }

    selected_idx[k] = best_idx;
    if (best_idx % kInklingWarpSize == lane) {
      local_scores[best_idx / kInklingWarpSize] = -FLT_MAX;
    }
    __syncwarp();
  }

  if (lane != 0) {
    return;
  }

  float active[kInklingTopPow2];
#pragma unroll
  for (int i = 0; i < kInklingTopK; ++i) {
    active[i] = inkling_sigmoid(logits[row_base + selected_idx[i]]);
  }
#pragma unroll
  for (int i = 0; i < kInklingSharedExperts; ++i) {
    active[kInklingTopK + i] = inkling_sigmoid(logits[row_base + kInklingRoutedExperts + i]);
  }

  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kInklingTopPow2; ++i) {
    sum += active[i];
  }
  const float scale = route_scale * global_scale[0] / sum;

#pragma unroll
  for (int i = 0; i < kInklingTopK; ++i) {
    const float w = active[i] * scale;
    if constexpr (ReturnPacked) {
      packed[row * kInklingTopK + i] = inkling_pack_routed(selected_idx[i], w);
    } else {
      routed_w[row * kInklingTopK + i] = w;
      indices[row * kInklingTopK + i] = static_cast<int64_t>(selected_idx[i]);
    }
  }
#pragma unroll
  for (int i = 0; i < kInklingSharedExperts; ++i) {
    shared_w[row * kInklingSharedExperts + i] = active[kInklingTopK + i] * scale;
  }
}

template <int WarpsPerBlock, bool ReturnPacked>
void launch_inkling_gate_topk_renorm(
    const float* logits,
    const float* bias,
    const float* global_scale,
    float* routed_w,
    float* shared_w,
    int64_t* indices,
    int32_t* packed,
    int64_t tokens,
    int64_t logits_stride_m,
    float route_scale,
    DLDevice device) {
  using namespace host;
  const dim3 block(kInklingWarpSize, WarpsPerBlock);
  const dim3 grid(static_cast<unsigned int>(div_ceil(tokens, static_cast<int64_t>(WarpsPerBlock))));
  LaunchKernel(grid, block, device)(
      inkling_gate_topk_renorm_kernel<WarpsPerBlock, ReturnPacked>,
      logits,
      bias,
      global_scale,
      routed_w,
      shared_w,
      indices,
      packed,
      tokens,
      logits_stride_m,
      route_scale);
}

template <bool ReturnPacked>
void dispatch_inkling_gate_topk_renorm(
    const float* logits,
    const float* bias,
    const float* global_scale,
    float* routed_w,
    float* shared_w,
    int64_t* indices,
    int32_t* packed,
    int64_t tokens,
    int64_t logits_stride_m,
    float route_scale,
    DLDevice device) {
  if (tokens <= 64) {
    launch_inkling_gate_topk_renorm<1, ReturnPacked>(
        logits, bias, global_scale, routed_w, shared_w, indices, packed, tokens, logits_stride_m, route_scale, device);
  } else if (tokens <= 1024) {
    launch_inkling_gate_topk_renorm<4, ReturnPacked>(
        logits, bias, global_scale, routed_w, shared_w, indices, packed, tokens, logits_stride_m, route_scale, device);
  } else {
    launch_inkling_gate_topk_renorm<8, ReturnPacked>(
        logits, bias, global_scale, routed_w, shared_w, indices, packed, tokens, logits_stride_m, route_scale, device);
  }
}

// Warp-per-row gate kernel with optional expert-per-block GEMV fusion.

// Inkling gate GEMM shape: x [M, 6144] bf16 x W [264 (row-padded), 6144] bf16.
static constexpr int kInklingHidden = 6144;
static constexpr int kInklingGemvThreads = 256;
static constexpr int kInklingGemvWarps = kInklingGemvThreads / kInklingWarpSize;
// fp32 logits row pitch: 264 floats = 1056 bytes, a multiple of 32B, so every
// row supports the widest vector loads. Matches the production [M, 264]
// padded-GEMM output that InklingGate slices to [:, :258].
static constexpr int kInklingLogitsPad = 264;
// The fused GEMV epilogue runs in a single block (8 warps looping over rows).
static constexpr int kInklingFusedMaxTokens = 64;

// Widest vector loads: 32B on Blackwell, 16B before.
static constexpr int kVecF32 = static_cast<int>(device::kMaxVecBytes / sizeof(float));
static constexpr int kVecBf16 = static_cast<int>(device::kMaxVecBytes / sizeof(bf16_t));
// v2 per-lane expert layout: expert = lane * 8 + j (contiguous per lane so one
// or two wide loads cover a lane's slice; the warp covers 256 experts).
static constexpr int kLanePitch = kInklingValuesPerLane;
static_assert(kLanePitch == 8 && kLanePitch % kVecF32 == 0);

// Launch-invariant gate inputs (weights): safe to read before the PDL wait.
struct InklingGateStatics {
  float bias[kLanePitch];  // selection bias for experts lane*8 + j
  float scale;             // route_scale * global_scale[0]
};

__device__ __forceinline__ InklingGateStatics inkling_gate_load_statics(
    const float* __restrict__ bias, const float* __restrict__ global_scale, float route_scale, int lane) {
  InklingGateStatics st;
#pragma unroll
  for (int i = 0; i < kLanePitch / kVecF32; ++i) {
    device::AlignedVector<float, kVecF32> v;
    v.load(bias, lane * (kLanePitch / kVecF32) + i);
#pragma unroll
    for (int j = 0; j < kVecF32; ++j) {
      st.bias[i * kVecF32 + j] = v[j];
    }
  }
  st.scale = route_scale * SGLANG_LDG(global_scale);
  return st;
}

// Whole-warp gate for one token row: top-6 of sigmoid(logit)+bias over experts
// 0..255 (exact fp32 compare, ties -> smaller expert id, matching the triton
// kernel), then renorm over sigmoid(raw selected) ++ sigmoid(shared 256..257).
// `row` must be device::kMaxVecBytes-aligned. Raw logits ride along in registers, so
// nothing is re-gathered from memory. The epilogue is spread over 8 lanes.
template <bool kPacked>
__device__ __forceinline__ void inkling_gate_row(
    const float* __restrict__ row,
    const InklingGateStatics& st,
    int lane,
    int64_t m,
    float* __restrict__ routed_w,
    int32_t* __restrict__ indices,
    int32_t* __restrict__ packed,
    float* __restrict__ shared_w) {
  float raw[kLanePitch];
  float sel[kLanePitch];
#pragma unroll
  for (int i = 0; i < kLanePitch / kVecF32; ++i) {
    device::AlignedVector<float, kVecF32> v;
    v.load(row, lane * (kLanePitch / kVecF32) + i);
#pragma unroll
    for (int j = 0; j < kVecF32; ++j) {
      raw[i * kVecF32 + j] = v[j];
      sel[i * kVecF32 + j] = inkling_sigmoid(v[j]) + st.bias[i * kVecF32 + j];
    }
  }

  // Shared-expert logits live at columns 256/257; lane 0 fetches, all receive.
  float sh0 = 0.0f;
  float sh1 = 0.0f;
  if (lane == 0) {
    const float2 sh = *reinterpret_cast<const float2*>(row + kInklingRoutedExperts);
    sh0 = sh.x;
    sh1 = sh.y;
  }
  sh0 = __shfl_sync(0xffffffff, sh0, 0);
  sh1 = __shfl_sync(0xffffffff, sh1, 0);

  int sel_idx[kInklingTopK];
  float sel_raw[kInklingTopK];
#pragma unroll
  for (int k = 0; k < kInklingTopK; ++k) {
    float best_s = -FLT_MAX;
    int best_j = 0;
#pragma unroll
    for (int j = 0; j < kLanePitch; ++j) {
      // ascending j => first strict max keeps the smallest expert id
      if (sel[j] > best_s) {
        best_s = sel[j];
        best_j = j;
      }
    }
    // Cross-lane argmax via one hardware redux + ballot. Exact fp32 order via
    // the monotonic key; ties pick the lowest lane, which is the smallest
    // expert id under the lane-major layout (expert = lane*8 + j).
    const uint32_t key = inkling_fp_key(best_s);
    int win_lane;
#if SGL_CUDA_ARCH >= 800
    const uint32_t key_max = __reduce_max_sync(0xffffffff, key);
#else
    uint32_t key_max = key;
#pragma unroll
    for (int offset = kInklingWarpSize / 2; offset > 0; offset >>= 1) {
      key_max = max(key_max, __shfl_xor_sync(0xffffffff, key_max, offset));
    }
#endif
    win_lane = __ffs(__ballot_sync(0xffffffff, key == key_max)) - 1;
    // The owning lane retires the winner and forwards (expert id, raw logit);
    // every local-array index stays static so nothing spills to local memory.
    float win_raw = 0.0f;
    int win_idx = 0;
    if (lane == win_lane) {
      win_idx = lane * kLanePitch + best_j;
#pragma unroll
      for (int j = 0; j < kLanePitch; ++j) {
        if (j == best_j) {
          sel[j] = -FLT_MAX;
          win_raw = raw[j];
        }
      }
    }
    sel_idx[k] = __shfl_sync(0xffffffff, win_idx, win_lane);
    sel_raw[k] = __shfl_sync(0xffffffff, win_raw, win_lane);
  }

  // Renorm, replicated on all lanes (registers only, no cross-lane traffic).
  float active[kInklingTopK + kInklingSharedExperts];
#pragma unroll
  for (int k = 0; k < kInklingTopK; ++k) {
    active[k] = inkling_sigmoid(sel_raw[k]);
  }
  active[kInklingTopK] = inkling_sigmoid(sh0);
  active[kInklingTopK + 1] = inkling_sigmoid(sh1);
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kInklingTopK + kInklingSharedExperts; ++i) {
    sum += active[i];
  }
  const float scale = st.scale / sum;

  // Lane a < 8 owns active slot a (static-index select, then one store each).
  float my_active = 0.0f;
  int my_idx = 0;
#pragma unroll
  for (int a = 0; a < kInklingTopK + kInklingSharedExperts; ++a) {
    if (a == lane) {
      my_active = active[a];
      my_idx = a < kInklingTopK ? sel_idx[a] : 0;
    }
  }
  const float w = my_active * scale;
  if (lane < kInklingTopK) {
    if constexpr (kPacked) {
      packed[m * kInklingTopK + lane] = inkling_pack_routed(my_idx, w);
    } else {
      routed_w[m * kInklingTopK + lane] = w;
      indices[m * kInklingTopK + lane] = my_idx;
    }
  } else if (lane < kInklingTopK + kInklingSharedExperts) {
    shared_w[m * kInklingSharedExperts + (lane - kInklingTopK)] = w;
  }
}

template <int kWarpsPerBlock, bool kPacked, bool kUsePDL>
__launch_bounds__(kInklingWarpSize* kWarpsPerBlock) __global__ void inkling_gate_topk_renorm_v2_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const float* __restrict__ global_scale,
    float* __restrict__ routed_w,
    int32_t* __restrict__ indices,
    int32_t* __restrict__ packed,
    float* __restrict__ shared_w,
    int64_t M,
    int64_t logits_stride_m,
    float route_scale) {
  const int lane = threadIdx.x;
  const int64_t m = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.y;
  const InklingGateStatics st = inkling_gate_load_statics(bias, global_scale, route_scale, lane);
  device::PDLWaitPrimary<kUsePDL>();
  if (m < M) {
    inkling_gate_row<kPacked>(logits + m * logits_stride_m, st, lane, m, routed_w, indices, packed, shared_w);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

// Vectorized fp32 dot of one x row against one smem-staged W row, over the
// vector-index range [v_lo, v_hi) with stride 32 (one warp lane per vector).
__device__ __forceinline__ float
inkling_gemv_partial(const bf16_t* __restrict__ x_row, const bf16_t* __restrict__ w_row, int v_lo, int v_hi, int lane) {
  float acc = 0.0f;
  for (int v = v_lo + lane; v < v_hi; v += kInklingWarpSize) {
    device::AlignedVector<bf16_t, kVecBf16> xv;
    device::AlignedVector<bf16_t, kVecBf16> wv;
    xv.load(x_row, v);
    wv.load(w_row, v);
#pragma unroll
    for (int p = 0; p < kVecBf16 / 2; ++p) {
      const float2 xf = __bfloat1622float2(reinterpret_cast<const bf16x2_t*>(xv.data())[p]);
      const float2 wf = __bfloat1622float2(reinterpret_cast<const bf16x2_t*>(wv.data())[p]);
      acc = fmaf(xf.x, wf.x, acc);
      acc = fmaf(xf.y, wf.y, acc);
    }
  }
  return acc;
}

// Experts-per-block GEMV: block b computes logits[:, b*kEpb : b*kEpb+kEpb] =
// x @ W[those experts].T. The W rows are staged to smem before the PDL wait
// (weights are launch-invariant), so the weight fetch overlaps the producer
// kernel's tail, and every x vector load is reused for kEpb experts. Token
// assignment: warp-per-token when M >= 8; for smaller M the warps split the
// hidden dim (partials combined through smem) so a single token still uses
// the whole block. With kFused, the last block to finish (atomic ticket) runs
// the gate epilogue over the workspace and resets the ticket so CUDA-graph
// replays need no re-initialization.
template <int kEpb, bool kFused, bool kPacked, bool kUsePDL>
__launch_bounds__(kInklingGemvThreads) __global__ void inkling_gate_gemv_kernel(
    const bf16_t* __restrict__ x,       // [M, 6144]
    const bf16_t* __restrict__ weight,  // [>=258, 6144]
    const float* __restrict__ bias,
    const float* __restrict__ global_scale,
    float* __restrict__ logits,  // [M, kInklingLogitsPad]
    float* __restrict__ routed_w,
    int32_t* __restrict__ indices,
    int32_t* __restrict__ packed,
    float* __restrict__ shared_w,
    int32_t* __restrict__ ticket,
    int64_t M,
    float route_scale) {
  constexpr int kHiddenVecs = kInklingHidden / kVecBf16;
  __shared__ alignas(device::kMaxVecBytes) bf16_t w_rows[kEpb][kInklingHidden];
  __shared__ float s_part[kInklingGemvWarps][kEpb];

  const int tid = threadIdx.x;
  const int lane = tid % kInklingWarpSize;
  const int warp = tid / kInklingWarpSize;
  const int e0 = blockIdx.x * kEpb;
  const int n_e = min(kEpb, kInklingTotalExperts - e0);  // tail block: fewer experts

  // M <= 2: each warp's W slice fits in registers, so it is preloaded there
  // before the PDL wait -- no smem staging round-trip on the critical path.
  // Sized for the worst case among the reg_sliced wpt values {8, 4}: wpt=4
  // gives the larger per-warp slice (kHiddenVecs/4 vectors). Derived from
  // kHiddenVecs (not hardcoded) because it depends on kMaxVecBytes, which is
  // 32B on Blackwell but only 16B pre-Blackwell -- a fixed literal sized for
  // Blackwell's narrower kHiddenVecs silently drops the back half of the
  // hidden dim on pre-Blackwell architectures.
  constexpr int kRegVecs = (kHiddenVecs / 4 + kInklingWarpSize - 1) / kInklingWarpSize;
  const bool reg_sliced = M <= 2;
  const int wpt = M <= 1 ? 8 : (M <= 2 ? 4 : (M <= 4 ? 2 : 1));
  const int v_span = kHiddenVecs / wpt;
  const int slice = warp % wpt;
  device::AlignedVector<bf16_t, kVecBf16> w_reg[kEpb][kRegVecs];
  if (reg_sliced) {
#pragma unroll
    for (int j = 0; j < kEpb; ++j) {
      if (j < n_e) {
#pragma unroll
        for (int r = 0; r < kRegVecs; ++r) {
          const int v = slice * v_span + r * kInklingWarpSize + lane;
          if (v < (slice + 1) * v_span) {
            w_reg[j][r].load(weight + static_cast<int64_t>(e0 + j) * kInklingHidden, v);
          }
        }
      }
    }
  } else {
#pragma unroll
    for (int j = 0; j < kEpb; ++j) {
      if (j < n_e) {
        for (int v = tid; v < kHiddenVecs; v += kInklingGemvThreads) {
          device::AlignedVector<bf16_t, kVecBf16> wv;
          wv.load(weight + static_cast<int64_t>(e0 + j) * kInklingHidden, v);
          wv.store(w_rows[j], v);
        }
      }
    }
  }
  device::PDLWaitPrimary<kUsePDL>();
  if (!reg_sliced) {  // uniform: publish the smem-staged W rows
    __syncthreads();
  }

  if (reg_sliced) {
    const int m = warp / wpt;
    if (m < M) {
      const bf16_t* x_row = x + static_cast<int64_t>(m) * kInklingHidden;
#pragma unroll
      for (int j = 0; j < kEpb; ++j) {
        float acc = 0.0f;
#pragma unroll
        for (int r = 0; r < kRegVecs; ++r) {
          const int v = slice * v_span + r * kInklingWarpSize + lane;
          if (v < (slice + 1) * v_span) {
            device::AlignedVector<bf16_t, kVecBf16> xv;
            xv.load(x_row, v);
#pragma unroll
            for (int p = 0; p < kVecBf16 / 2; ++p) {
              const float2 xf = __bfloat1622float2(reinterpret_cast<const bf16x2_t*>(xv.data())[p]);
              const float2 wf = __bfloat1622float2(reinterpret_cast<const bf16x2_t*>(w_reg[j][r].data())[p]);
              acc = fmaf(xf.x, wf.x, acc);
              acc = fmaf(xf.y, wf.y, acc);
            }
          }
        }
        const float r = device::warp::reduce_sum(acc);
        if (lane == 0) {
          s_part[warp][j] = r;
        }
      }
    }
    __syncthreads();
    if (warp == 0) {
      const int total = static_cast<int>(M) * kEpb;
      if (lane < total) {
        const int m_out = lane / kEpb;
        const int j_out = lane % kEpb;
        float r = 0.0f;
        for (int s = 0; s < wpt; ++s) {
          r += s_part[m_out * wpt + s][j_out];
        }
        if (j_out < n_e) {
          logits[static_cast<int64_t>(m_out) * kInklingLogitsPad + e0 + j_out] = r;
        }
      }
    }
  } else if (M >= kInklingGemvWarps) {
    for (int64_t m = warp; m < M; m += kInklingGemvWarps) {
      const bf16_t* x_row = x + m * kInklingHidden;
      float acc[kEpb];
#pragma unroll
      for (int j = 0; j < kEpb; ++j) {
        acc[j] = 0.0f;
      }
      // Single pass over the x row; each vector feeds all kEpb experts.
      for (int v = lane; v < kHiddenVecs; v += kInklingWarpSize) {
        device::AlignedVector<bf16_t, kVecBf16> xv;
        xv.load(x_row, v);
#pragma unroll
        for (int j = 0; j < kEpb; ++j) {
          device::AlignedVector<bf16_t, kVecBf16> wv;
          wv.load(w_rows[j], v);
#pragma unroll
          for (int p = 0; p < kVecBf16 / 2; ++p) {
            const float2 xf = __bfloat1622float2(reinterpret_cast<const bf16x2_t*>(xv.data())[p]);
            const float2 wf = __bfloat1622float2(reinterpret_cast<const bf16x2_t*>(wv.data())[p]);
            acc[j] = fmaf(xf.x, wf.x, acc[j]);
            acc[j] = fmaf(xf.y, wf.y, acc[j]);
          }
        }
      }
#pragma unroll
      for (int j = 0; j < kEpb; ++j) {
        const float r = device::warp::reduce_sum(acc[j]);
        if (lane == 0 && j < n_e) {
          logits[m * kInklingLogitsPad + e0 + j] = r;
        }
      }
    }
  } else {
    // 2 < M < 8: warps_per_token warps split the hidden dim of one token,
    // reading W from the smem staging.
    const int m = warp / wpt;
    if (m < M) {
      const bf16_t* x_row = x + static_cast<int64_t>(m) * kInklingHidden;
#pragma unroll
      for (int j = 0; j < kEpb; ++j) {
        const float p = inkling_gemv_partial(x_row, w_rows[j], slice * v_span, (slice + 1) * v_span, lane);
        const float r = device::warp::reduce_sum(p);
        if (lane == 0) {
          s_part[warp][j] = r;
        }
      }
    }
    __syncthreads();
    // One lane per (token, expert) folds the wpt partials and stores.
    if (warp == 0) {
      const int total = static_cast<int>(M) * kEpb;
      if (lane < total) {
        const int m_out = lane / kEpb;
        const int j_out = lane % kEpb;
        float r = 0.0f;
        for (int s = 0; s < wpt; ++s) {
          r += s_part[m_out * wpt + s][j_out];
        }
        if (j_out < n_e) {
          logits[static_cast<int64_t>(m_out) * kInklingLogitsPad + e0 + j_out] = r;
        }
      }
    }
  }

  if constexpr (!kFused) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }

  // Single-pass fused epilogue (CUB-style threadfence + ticket pattern).
  __shared__ int s_ticket;
  __syncthreads();
  __threadfence();
  if (tid == 0) {
    s_ticket = atomicAdd(ticket, 1);
  }
  __syncthreads();
  if (s_ticket != static_cast<int>(gridDim.x) - 1) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  __threadfence();  // acquire side: other blocks' workspace stores are visible
  const InklingGateStatics st = inkling_gate_load_statics(bias, global_scale, route_scale, lane);
  for (int64_t m = warp; m < M; m += kInklingGemvWarps) {
    inkling_gate_row<kPacked>(logits + m * kInklingLogitsPad, st, lane, m, routed_w, indices, packed, shared_w);
  }
  __syncthreads();
  if (tid == 0) {
    *ticket = 0;  // replay-safe self-reset (CUDA graphs)
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kPacked>
void dispatch_inkling_gate_topk_renorm_v2(
    const float* logits,
    const float* bias,
    const float* global_scale,
    float* routed_w,
    int32_t* indices,
    int32_t* packed,
    float* shared_w,
    int64_t tokens,
    int64_t logits_stride_m,
    float route_scale,
    bool enable_pdl,
    int64_t warps_per_block,
    DLDevice device) {
  using namespace host;
  if (warps_per_block == 0) {
    // Four warps balance occupancy and per-row parallelism.
    warps_per_block = 4;
  }
  const dim3 block(kInklingWarpSize, static_cast<unsigned int>(warps_per_block));
  const dim3 grid(static_cast<unsigned int>(div_ceil(tokens, warps_per_block)));
  auto launch = [&](auto kernel) {
    LaunchKernel(grid, block, device)
        .enable_pdl(enable_pdl)(
            kernel,
            logits,
            bias,
            global_scale,
            routed_w,
            indices,
            packed,
            shared_w,
            tokens,
            logits_stride_m,
            route_scale);
  };
  switch (warps_per_block) {
    case 1:
      return enable_pdl ? launch(inkling_gate_topk_renorm_v2_kernel<1, kPacked, true>)
                        : launch(inkling_gate_topk_renorm_v2_kernel<1, kPacked, false>);
    case 2:
      return enable_pdl ? launch(inkling_gate_topk_renorm_v2_kernel<2, kPacked, true>)
                        : launch(inkling_gate_topk_renorm_v2_kernel<2, kPacked, false>);
    case 4:
      return enable_pdl ? launch(inkling_gate_topk_renorm_v2_kernel<4, kPacked, true>)
                        : launch(inkling_gate_topk_renorm_v2_kernel<4, kPacked, false>);
    case 8:
      return enable_pdl ? launch(inkling_gate_topk_renorm_v2_kernel<8, kPacked, true>)
                        : launch(inkling_gate_topk_renorm_v2_kernel<8, kPacked, false>);
    default:
      ::host::panic({}, "warps_per_block must be one of {0, 1, 2, 4, 8}");
  }
}

template <bool kFused, bool kPacked>
void dispatch_inkling_gate_gemv(
    const bf16_t* x,
    const bf16_t* weight,
    const float* bias,
    const float* global_scale,
    float* logits,
    float* routed_w,
    int32_t* indices,
    int32_t* packed,
    float* shared_w,
    int32_t* ticket,
    int64_t tokens,
    float route_scale,
    bool enable_pdl,
    int64_t experts_per_block,
    DLDevice device) {
  using namespace host;
  if (experts_per_block == 0) {  // auto policy, tuned on B200
    experts_per_block = 2;
  }
  const dim3 block(kInklingGemvThreads);
  const dim3 grid(static_cast<unsigned int>(div_ceil(kInklingTotalExperts, static_cast<int>(experts_per_block))));
  auto launch = [&](auto kernel) {
    LaunchKernel(grid, block, device)
        .enable_pdl(enable_pdl)(
            kernel,
            x,
            weight,
            bias,
            global_scale,
            logits,
            routed_w,
            indices,
            packed,
            shared_w,
            ticket,
            tokens,
            route_scale);
  };
  switch (experts_per_block) {
    case 1:
      return enable_pdl ? launch(inkling_gate_gemv_kernel<1, kFused, kPacked, true>)
                        : launch(inkling_gate_gemv_kernel<1, kFused, kPacked, false>);
    case 2:
      return enable_pdl ? launch(inkling_gate_gemv_kernel<2, kFused, kPacked, true>)
                        : launch(inkling_gate_gemv_kernel<2, kFused, kPacked, false>);
    default:
      // 4 experts/block would need >48KB static smem (dynamic smem territory).
      ::host::panic({}, "experts_per_block must be one of {0, 1, 2}");
  }
}

void check_gate_row_alignment(const void* ptr, int64_t stride_m) {
  host::RuntimeCheck(
      reinterpret_cast<uintptr_t>(ptr) % device::kMaxVecBytes == 0 &&
          (stride_m * static_cast<int64_t>(sizeof(float))) % device::kMaxVecBytes == 0,
      "logits rows must be device::kMaxVecBytes-aligned (production pitch is 264 floats)");
}

}  // namespace

void inkling_gate_topk_renorm(
    tvm::ffi::TensorView logits,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView global_scale,
    tvm::ffi::TensorView routed_w,
    tvm::ffi::TensorView shared_w,
    tvm::ffi::TensorView indices,
    double route_scale) {
  using namespace host;

  SymbolicSize M{"tokens"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({M, kInklingTotalExperts})
      .with_strides({-1, 1})
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(logits);
  TensorMatcher({kInklingRoutedExperts}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(bias);
  TensorMatcher({1}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(global_scale);
  TensorMatcher({M, kInklingTopK}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(routed_w);
  TensorMatcher({M, kInklingSharedExperts}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(shared_w);
  TensorMatcher({M, kInklingTopK}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(indices);

  RuntimeCheck(logits.stride(1) == 1, "logits must be contiguous along the expert dimension");
  RuntimeCheck(bias.stride(0) == 1, "bias must be contiguous");
  RuntimeCheck(routed_w.stride(1) == 1, "routed_w must be contiguous along top-k dimension");
  RuntimeCheck(shared_w.stride(1) == 1, "shared_w must be contiguous along shared dimension");
  RuntimeCheck(indices.stride(1) == 1, "indices must be contiguous along top-k dimension");

  const int64_t tokens = M.unwrap();
  if (tokens == 0) {
    return;
  }

  const auto* logits_ptr = static_cast<const float*>(logits.data_ptr());
  const auto* bias_ptr = static_cast<const float*>(bias.data_ptr());
  const auto* global_scale_ptr = static_cast<const float*>(global_scale.data_ptr());
  auto* routed_w_ptr = static_cast<float*>(routed_w.data_ptr());
  auto* shared_w_ptr = static_cast<float*>(shared_w.data_ptr());
  auto* indices_ptr = static_cast<int64_t*>(indices.data_ptr());
  const float route_scale_f = static_cast<float>(route_scale);
  const DLDevice device = device_.unwrap();

  dispatch_inkling_gate_topk_renorm<false>(
      logits_ptr,
      bias_ptr,
      global_scale_ptr,
      routed_w_ptr,
      shared_w_ptr,
      indices_ptr,
      nullptr,
      tokens,
      logits.stride(0),
      route_scale_f,
      device);
}

// Packed variant: emits packed[M, kInklingTopK] int32 ((id<<16)|bf16 weight) instead of
// the routed_w + indices pair; shared_w still written.
void inkling_gate_topk_renorm_packed(
    tvm::ffi::TensorView logits,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView global_scale,
    tvm::ffi::TensorView packed,
    tvm::ffi::TensorView shared_w,
    double route_scale) {
  using namespace host;

  SymbolicSize M{"tokens"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({M, kInklingTotalExperts})
      .with_strides({-1, 1})
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(logits);
  TensorMatcher({kInklingRoutedExperts}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(bias);
  TensorMatcher({1}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(global_scale);
  TensorMatcher({M, kInklingTopK}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(packed);
  TensorMatcher({M, kInklingSharedExperts}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(shared_w);

  RuntimeCheck(logits.stride(1) == 1, "logits must be contiguous along the expert dimension");
  RuntimeCheck(bias.stride(0) == 1, "bias must be contiguous");
  RuntimeCheck(packed.stride(1) == 1, "packed must be contiguous along top-k dimension");
  RuntimeCheck(shared_w.stride(1) == 1, "shared_w must be contiguous along shared dimension");

  const int64_t tokens = M.unwrap();
  if (tokens == 0) {
    return;
  }

  const auto* logits_ptr = static_cast<const float*>(logits.data_ptr());
  const auto* bias_ptr = static_cast<const float*>(bias.data_ptr());
  const auto* global_scale_ptr = static_cast<const float*>(global_scale.data_ptr());
  auto* packed_ptr = static_cast<int32_t*>(packed.data_ptr());
  auto* shared_w_ptr = static_cast<float*>(shared_w.data_ptr());
  const float route_scale_f = static_cast<float>(route_scale);
  const DLDevice device = device_.unwrap();

  dispatch_inkling_gate_topk_renorm<true>(
      logits_ptr,
      bias_ptr,
      global_scale_ptr,
      nullptr,
      shared_w_ptr,
      nullptr,
      packed_ptr,
      tokens,
      logits.stride(0),
      route_scale_f,
      device);
}

// Uses int32 indices for MoeRunner, optional PDL, and tunable warps (0 = auto).
// Logit rows must be device::kMaxVecBytes-aligned (the standard pitch is 264).
void inkling_gate_topk_renorm_v2(
    tvm::ffi::TensorView logits,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView global_scale,
    tvm::ffi::TensorView routed_w,
    tvm::ffi::TensorView shared_w,
    tvm::ffi::TensorView indices,
    double route_scale,
    bool enable_pdl,
    int64_t warps_per_block) {
  using namespace host;

  SymbolicSize M{"tokens"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({M, kInklingTotalExperts})
      .with_strides({-1, 1})
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(logits);
  TensorMatcher({kInklingRoutedExperts}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(bias);
  TensorMatcher({1}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(global_scale);
  TensorMatcher({M, kInklingTopK}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(routed_w);
  TensorMatcher({M, kInklingSharedExperts}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(shared_w);
  TensorMatcher({M, kInklingTopK}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(indices);

  RuntimeCheck(logits.stride(1) == 1, "logits must be contiguous along the expert dimension");
  RuntimeCheck(bias.stride(0) == 1, "bias must be contiguous");
  RuntimeCheck(
      routed_w.stride(1) == 1 && shared_w.stride(1) == 1 && indices.stride(1) == 1, "outputs must be contiguous");
  check_gate_row_alignment(logits.data_ptr(), logits.stride(0));

  const int64_t tokens = M.unwrap();
  if (tokens == 0) {
    return;
  }

  dispatch_inkling_gate_topk_renorm_v2<false>(
      static_cast<const float*>(logits.data_ptr()),
      static_cast<const float*>(bias.data_ptr()),
      static_cast<const float*>(global_scale.data_ptr()),
      static_cast<float*>(routed_w.data_ptr()),
      static_cast<int32_t*>(indices.data_ptr()),
      nullptr,
      static_cast<float*>(shared_w.data_ptr()),
      tokens,
      logits.stride(0),
      static_cast<float>(route_scale),
      enable_pdl,
      warps_per_block,
      device_.unwrap());
}

void inkling_gate_topk_renorm_v2_packed(
    tvm::ffi::TensorView logits,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView global_scale,
    tvm::ffi::TensorView packed,
    tvm::ffi::TensorView shared_w,
    double route_scale,
    bool enable_pdl,
    int64_t warps_per_block) {
  using namespace host;

  SymbolicSize M{"tokens"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({M, kInklingTotalExperts})
      .with_strides({-1, 1})
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(logits);
  TensorMatcher({kInklingRoutedExperts}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(bias);
  TensorMatcher({1}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(global_scale);
  TensorMatcher({M, kInklingTopK}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(packed);
  TensorMatcher({M, kInklingSharedExperts}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(shared_w);

  RuntimeCheck(logits.stride(1) == 1, "logits must be contiguous along the expert dimension");
  RuntimeCheck(bias.stride(0) == 1, "bias must be contiguous");
  RuntimeCheck(packed.stride(1) == 1 && shared_w.stride(1) == 1, "outputs must be contiguous");
  check_gate_row_alignment(logits.data_ptr(), logits.stride(0));

  const int64_t tokens = M.unwrap();
  if (tokens == 0) {
    return;
  }

  dispatch_inkling_gate_topk_renorm_v2<true>(
      static_cast<const float*>(logits.data_ptr()),
      static_cast<const float*>(bias.data_ptr()),
      static_cast<const float*>(global_scale.data_ptr()),
      nullptr,
      nullptr,
      static_cast<int32_t*>(packed.data_ptr()),
      static_cast<float*>(shared_w.data_ptr()),
      tokens,
      logits.stride(0),
      static_cast<float>(route_scale),
      enable_pdl,
      warps_per_block,
      device_.unwrap());
}

// Standalone gate GEMV: logits[:, :258] = x @ W[:258].T (fp32 accumulate), for
// the PDL split pair (GEMV kernel -> v2 gate kernel). `logits` must use the
// production [M, 264] padded layout; columns 258..263 are left untouched.
void inkling_gate_gemv(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView logits,
    bool enable_pdl,
    int64_t experts_per_block) {
  using namespace host;

  SymbolicSize M{"tokens"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({M, kInklingHidden}).with_dtype<bf16_t>().with_device<kDLCUDA>(device_).verify(x);
  TensorMatcher({-1, kInklingHidden}).with_dtype<bf16_t>().with_device<kDLCUDA>(device_).verify(weight);
  TensorMatcher({M, kInklingLogitsPad}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(logits);

  RuntimeCheck(weight.size(0) >= kInklingTotalExperts, "gate weight must cover 258 experts");
  RuntimeCheck(x.stride(1) == 1 && x.stride(0) == kInklingHidden, "x must be contiguous");
  RuntimeCheck(weight.stride(1) == 1 && weight.stride(0) == kInklingHidden, "weight must be contiguous");
  RuntimeCheck(logits.stride(1) == 1 && logits.stride(0) == kInklingLogitsPad, "logits must be contiguous");

  const int64_t tokens = M.unwrap();
  if (tokens == 0) {
    return;
  }

  dispatch_inkling_gate_gemv<false, false>(
      static_cast<const bf16_t*>(x.data_ptr()),
      static_cast<const bf16_t*>(weight.data_ptr()),
      nullptr,
      nullptr,
      static_cast<float*>(logits.data_ptr()),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      tokens,
      0.0f,
      enable_pdl,
      experts_per_block,
      device_.unwrap());
}

namespace {

// Shared verification for the fused GEMV entry points; returns tokens.
int64_t verify_inkling_gate_gemv_fused_common(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView global_scale,
    tvm::ffi::TensorView workspace,
    tvm::ffi::TensorView ticket,
    tvm::ffi::TensorView shared_w,
    host::SymbolicSize& M,
    host::SymbolicDevice& device_) {
  using namespace host;

  TensorMatcher({M, kInklingHidden}).with_dtype<bf16_t>().with_device<kDLCUDA>(device_).verify(x);
  TensorMatcher({-1, kInklingHidden}).with_dtype<bf16_t>().with_device<kDLCUDA>(device_).verify(weight);
  TensorMatcher({kInklingRoutedExperts}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(bias);
  TensorMatcher({1}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(global_scale);
  TensorMatcher({-1, kInklingLogitsPad}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(workspace);
  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(ticket);
  TensorMatcher({M, kInklingSharedExperts}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(shared_w);

  const int64_t tokens = M.unwrap();
  RuntimeCheck(weight.size(0) >= kInklingTotalExperts, "gate weight must cover 258 experts");
  RuntimeCheck(tokens <= kInklingFusedMaxTokens, "fused gate GEMV supports at most 64 tokens");
  RuntimeCheck(workspace.size(0) >= tokens, "workspace too small");
  RuntimeCheck(x.stride(1) == 1 && x.stride(0) == kInklingHidden, "x must be contiguous");
  RuntimeCheck(weight.stride(1) == 1 && weight.stride(0) == kInklingHidden, "weight must be contiguous");
  RuntimeCheck(workspace.stride(1) == 1 && workspace.stride(0) == kInklingLogitsPad, "workspace must be contiguous");
  RuntimeCheck(bias.stride(0) == 1, "bias must be contiguous");
  RuntimeCheck(shared_w.stride(1) == 1, "shared_w must be contiguous");
  return tokens;
}

}  // namespace

// Fully fused gate: GEMV + top-k + renorm in a single launch. `workspace` is a
// [>=M, 264] fp32 scratch and `ticket` a zero-initialized int32[1] that the
// kernel resets after use (both may be cached across calls / graph replays).
void inkling_gate_gemv_fused(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView global_scale,
    tvm::ffi::TensorView workspace,
    tvm::ffi::TensorView ticket,
    tvm::ffi::TensorView routed_w,
    tvm::ffi::TensorView shared_w,
    tvm::ffi::TensorView indices,
    double route_scale,
    bool enable_pdl,
    int64_t experts_per_block) {
  using namespace host;

  SymbolicSize M{"tokens"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  const int64_t tokens =
      verify_inkling_gate_gemv_fused_common(x, weight, bias, global_scale, workspace, ticket, shared_w, M, device_);
  TensorMatcher({M, kInklingTopK}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(routed_w);
  TensorMatcher({M, kInklingTopK}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(indices);
  RuntimeCheck(routed_w.stride(1) == 1 && indices.stride(1) == 1, "outputs must be contiguous");
  if (tokens == 0) {
    return;
  }

  dispatch_inkling_gate_gemv<true, false>(
      static_cast<const bf16_t*>(x.data_ptr()),
      static_cast<const bf16_t*>(weight.data_ptr()),
      static_cast<const float*>(bias.data_ptr()),
      static_cast<const float*>(global_scale.data_ptr()),
      static_cast<float*>(workspace.data_ptr()),
      static_cast<float*>(routed_w.data_ptr()),
      static_cast<int32_t*>(indices.data_ptr()),
      nullptr,
      static_cast<float*>(shared_w.data_ptr()),
      static_cast<int32_t*>(ticket.data_ptr()),
      tokens,
      static_cast<float>(route_scale),
      enable_pdl,
      experts_per_block,
      device_.unwrap());
}

void inkling_gate_gemv_fused_packed(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView global_scale,
    tvm::ffi::TensorView workspace,
    tvm::ffi::TensorView ticket,
    tvm::ffi::TensorView packed,
    tvm::ffi::TensorView shared_w,
    double route_scale,
    bool enable_pdl,
    int64_t experts_per_block) {
  using namespace host;

  SymbolicSize M{"tokens"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  const int64_t tokens =
      verify_inkling_gate_gemv_fused_common(x, weight, bias, global_scale, workspace, ticket, shared_w, M, device_);
  TensorMatcher({M, kInklingTopK}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(packed);
  RuntimeCheck(packed.stride(1) == 1, "packed must be contiguous");
  if (tokens == 0) {
    return;
  }

  dispatch_inkling_gate_gemv<true, true>(
      static_cast<const bf16_t*>(x.data_ptr()),
      static_cast<const bf16_t*>(weight.data_ptr()),
      static_cast<const float*>(bias.data_ptr()),
      static_cast<const float*>(global_scale.data_ptr()),
      static_cast<float*>(workspace.data_ptr()),
      nullptr,
      nullptr,
      static_cast<int32_t*>(packed.data_ptr()),
      static_cast<float*>(shared_w.data_ptr()),
      static_cast<int32_t*>(ticket.data_ptr()),
      tokens,
      static_cast<float>(route_scale),
      enable_pdl,
      experts_per_block,
      device_.unwrap());
}
