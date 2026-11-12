// SPDX-License-Identifier: Apache-2.0
// Small-batch routed MoE GEMM for Hopper on Marlin-repacked NVFP4 / MXFP8 expert
// weights, reading the Marlin layout directly. A lane's 4-byte word of a Marlin
// 16 (k) x 64 (n) tile holds exactly an mma.m16n8k16 A fragment of the transposed
// tile (16 n rows x 16 k), so the weights are the A operand and 8 routed rows
// per MMA are the B operand. The K splits of an n tile form a cluster and are
// summed in split order through distributed shared memory.

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cooperative_groups.h>
#include <cstdint>
#include <cuda_bf16.h>

namespace sglang::inkling2_marlin_gemv {

namespace cg = cooperative_groups;

constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kMaxExperts = 128;

__device__ __forceinline__ void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ void pdl_launch() {
  asm volatile("griddepcontrol.launch_dependents;");
}

__device__ __forceinline__ uint32_t bf16x2_mul(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// BF16 pair of the nibbles at bits 12-15 and 28-31 of q << shift, at 2^-126 of the
// e2m1 values.
template <int kShift>
__device__ __forceinline__ uint32_t e2m1_pair(uint32_t q) {
  uint32_t sign = kShift ? (q << kShift) & 0x80008000u : q & 0x80008000u;
  uint32_t magnitude = (kShift <= 6 ? q >> (6 - kShift) : q << (kShift - 6)) & 0x01C001C0u;
  return sign | magnitude;
}

// BF16 pair of the bytes at bits 8-15 and 24-31 of q << shift, at 2^-120 of the
// e4m3 values.
template <int kShift>
__device__ __forceinline__ uint32_t e4m3_pair(uint32_t q) {
  uint32_t sign = kShift ? (q << kShift) & 0x80008000u : q & 0x80008000u;
  uint32_t magnitude = (kShift <= 4 ? q >> (4 - kShift) : q << (kShift - 4)) & 0x07F007F0u;
  return sign | magnitude;
}

// Marlin NVFP4 scale byte j of a word (high byte of the fp16 of scale * 2^7) in
// both halves of a BF16 pair, as scale * 2^118: with the e2m1 bias, products are
// at 2^-8.
template <int kByte>
__device__ __forceinline__ uint32_t nvfp4_scale(uint32_t word) {
  return __byte_perm(word, 0, 0x4040 | (kByte << 8) | kByte) * 16u + 0x6F806F80u;
}

__device__ __forceinline__ uint32_t broadcast_lo(uint32_t v) {
  return __byte_perm(v, 0, 0x1010);
}
__device__ __forceinline__ uint32_t broadcast_hi(uint32_t v) {
  return __byte_perm(v, 0, 0x3232);
}

template <bool kNvfp4>
struct Layout {
  static constexpr int kWordsPerLane = kNvfp4 ? 4 : 8;
  static constexpr int kScaleWords = kNvfp4 ? 2 : 4;
  static constexpr int kTileWords = kTileN * kTileK / (kNvfp4 ? 8 : 4);
  static constexpr float kOutScale = kNvfp4 ? 256.0f : 1.0f;
};

// A fragments of the 4 (n16 x k16) blocks of one lane's words, scaled per n row.
template <bool kNvfp4>
__device__ __forceinline__ void fragments(
    const uint32_t (&q)[Layout<kNvfp4>::kWordsPerLane],
    const uint32_t (&s)[Layout<kNvfp4>::kScaleWords],
    uint32_t (&a)[4][4]) {
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    uint32_t s_lo, s_hi;
    if constexpr (kNvfp4) {
      if (w & 1) {
        s_lo = nvfp4_scale<1>(s[w >> 1]);
        s_hi = nvfp4_scale<3>(s[w >> 1]);
      } else {
        s_lo = nvfp4_scale<0>(s[w >> 1]);
        s_hi = nvfp4_scale<2>(s[w >> 1]);
      }
      a[w][0] = bf16x2_mul(e2m1_pair<12>(q[w]), s_lo);
      a[w][1] = bf16x2_mul(e2m1_pair<4>(q[w]), s_hi);
      a[w][2] = bf16x2_mul(e2m1_pair<8>(q[w]), s_lo);
      a[w][3] = bf16x2_mul(e2m1_pair<0>(q[w]), s_hi);
    } else {
      s_lo = broadcast_lo(s[w]);
      s_hi = broadcast_hi(s[w]);
      a[w][0] = bf16x2_mul(e4m3_pair<8>(q[2 * w]), s_lo);
      a[w][1] = bf16x2_mul(e4m3_pair<8>(q[2 * w + 1]), s_hi);
      a[w][2] = bf16x2_mul(e4m3_pair<0>(q[2 * w]), s_lo);
      a[w][3] = bf16x2_mul(e4m3_pair<0>(q[2 * w + 1]), s_hi);
    }
  }
}

constexpr int batch_size(int per_warp) {
  for (int b = 3; b > 1; --b)
    if (per_warp % b == 0) return b;
  return 1;
}

// Persistent clusters of kSplits CTAs; cluster c handles the (active expert, n tile)
// items c, c + clusters, ...; 8 routed rows per pass.
template <bool kNvfp4, int kN, int kK, int kSplits, int kWarps>
__global__ void __launch_bounds__(kWarps * 32) marlin_gemv(
    const __nv_bfloat16* __restrict__ x,
    const uint32_t* __restrict__ weight,
    const uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ y,
    const int* __restrict__ counts,
    const int* __restrict__ offsets,
    const int* __restrict__ dst2src,
    int num_experts,
    int top_k) {
  using L = Layout<kNvfp4>;
  constexpr int kRows = 8;
  constexpr int kNTiles = kN / kTileN;
  constexpr int kKTiles = kK / kTileK;
  constexpr int kPerWarp = kKTiles / kSplits / kWarps;
  constexpr int kBatch = batch_size(kPerWarp);
  constexpr int64_t kExpertWords = static_cast<int64_t>(kKTiles) * kNTiles * L::kTileWords;
  static_assert(kKTiles % (kSplits * kWarps) == 0);

  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int g = lane / 4, t = lane % 4;
  const int split = blockIdx.x % kSplits;
  const int k_first = split * (kKTiles / kSplits) + warp * kPerWarp;
  const int clusters = gridDim.x / kSplits;

  __shared__ float red[kWarps][kRows * kTileN];

  pdl_wait();
  uint32_t active[kMaxExperts / 32];
  int num_active = 0;
#pragma unroll
  for (int i = 0; i < kMaxExperts / 32; ++i) {
    int e = i * 32 + lane;
    active[i] = __ballot_sync(0xFFFFFFFF, e < num_experts && counts[e] > 0);
    num_active += __popc(active[i]);
  }

  for (int item = blockIdx.x / kSplits; item < num_active * kNTiles; item += clusters) {
    const int slot = item / kNTiles, n_tile = item % kNTiles;
    int expert = 0;
    {
      int need = slot;
#pragma unroll
      for (int i = 0; i < kMaxExperts / 32; ++i) {
        int c = __popc(active[i]);
        if (need >= 0 && need < c) {
          unsigned m = active[i];
          for (int j = 0; j < need; ++j)
            m &= m - 1;
          expert = i * 32 + __ffs(m) - 1;
        }
        need -= c;
      }
    }
    const uint32_t* w_base = weight + expert * kExpertWords +
                             (static_cast<int64_t>(k_first) * kNTiles + n_tile) * L::kTileWords +
                             lane * L::kWordsPerLane;
    const uint8_t* s_base =
        kNvfp4 ? scales + (static_cast<int64_t>(expert) * kKTiles + k_first) * kN + n_tile * kTileN + g * 8
               : scales + (static_cast<int64_t>(expert) * (kKTiles / 2) * kN + n_tile * kTileN + g * 8) * 2;
    const int count = counts[expert];
    const int offset = offsets[expert];

    for (int m0 = 0; m0 < count; m0 += kRows) {
      float acc[4][4] = {};
      const int row_src = lane < kRows && m0 + lane < count ? dst2src[offset + m0 + lane] : -1;
      const int src = __shfl_sync(0xFFFFFFFF, row_src, g);
      const __nv_bfloat16* xrow = src >= 0 ? x + static_cast<int64_t>(src / top_k) * kK + k_first * kTileK + 2 * t : x;
      uint32_t q[2][kBatch][L::kWordsPerLane];
      uint32_t s[2][kBatch][L::kScaleWords];
      uint32_t bx[2][kBatch][2];
      auto load = [&](int buf, int b0) {
#pragma unroll
        for (int u = 0; u < kBatch; ++u) {
          const int kt = b0 + u;
          const uint4* wp = reinterpret_cast<const uint4*>(w_base + static_cast<int64_t>(kt) * kNTiles * L::kTileWords);
#pragma unroll
          for (int v = 0; v < L::kWordsPerLane / 4; ++v) {
            uint4 word = __ldg(wp + v);
            q[buf][u][4 * v + 0] = word.x;
            q[buf][u][4 * v + 1] = word.y;
            q[buf][u][4 * v + 2] = word.z;
            q[buf][u][4 * v + 3] = word.w;
          }
          if constexpr (kNvfp4) {
            uint2 sv = __ldg(reinterpret_cast<const uint2*>(s_base + kt * kN));
            s[buf][u][0] = sv.x;
            s[buf][u][1] = sv.y;
          } else {
            uint4 sv = __ldg(reinterpret_cast<const uint4*>(s_base + ((k_first + kt) / 2) * kN * 2));
            s[buf][u][0] = sv.x;
            s[buf][u][1] = sv.y;
            s[buf][u][2] = sv.z;
            s[buf][u][3] = sv.w;
          }
          bx[buf][u][0] = bx[buf][u][1] = 0;
          if (src >= 0) {
            bx[buf][u][0] = __ldg(reinterpret_cast<const uint32_t*>(xrow + kt * kTileK));
            bx[buf][u][1] = __ldg(reinterpret_cast<const uint32_t*>(xrow + kt * kTileK + 8));
          }
        }
      };
      load(0, 0);
#pragma unroll
      for (int b0 = 0; b0 < kPerWarp; b0 += kBatch) {
        const int buf = (b0 / kBatch) % 2;
        if (b0 + kBatch < kPerWarp) load(buf ^ 1, b0 + kBatch);
#pragma unroll
        for (int u = 0; u < kBatch; ++u) {
          uint32_t a[4][4];
          fragments<kNvfp4>(q[buf][u], s[buf][u], a);
#pragma unroll
          for (int w = 0; w < 4; ++w)
            mma(acc[w], a[w], bx[buf][u][0], bx[buf][u][1]);
        }
      }

#pragma unroll
      for (int w = 0; w < 4; ++w)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          red[warp][(2 * t + (i & 1)) * kTileN + w * 16 + g + (i >= 2 ? 8 : 0)] = acc[w][i];
      __syncthreads();
      const int used = min(count - m0, kRows) * kTileN;
      for (int i = threadIdx.x; i < used; i += kWarps * 32) {
        float v = red[0][i];
#pragma unroll
        for (int wp = 1; wp < kWarps; ++wp)
          v += red[wp][i];
        red[0][i] = v;
      }
      if constexpr (kSplits > 1) {
        cg::cluster_group cluster = cg::this_cluster();
        cluster.sync();
        if (split == 0) {
          for (int i = threadIdx.x; i < used; i += kWarps * 32) {
            float v = red[0][i];
#pragma unroll
            for (int r = 1; r < kSplits; ++r)
              v += cluster.map_shared_rank(&red[0][0], r)[i];
            red[0][i] = v;
          }
        }
        cluster.sync();
      } else {
        __syncthreads();
      }
      if (split == 0) {
        for (int i = threadIdx.x; i < kRows * kTileN; i += kWarps * 32) {
          int src = __shfl_sync(0xFFFFFFFF, row_src, i / kTileN);
          if (i < used)
            y[static_cast<int64_t>(src) * kN + n_tile * kTileN + i % kTileN] =
                __float2bfloat16_rn(red[0][i] * L::kOutScale);
        }
      }
      __syncthreads();
    }
  }
  pdl_launch();
}

template <bool kNvfp4, int kN, int kK, int kSplits, int kWarps>
void run(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView scales,
    tvm::ffi::TensorView y,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView offsets,
    tvm::ffi::TensorView dst2src,
    int64_t slots,
    int64_t top_k) {
  using host::RuntimeCheck;
  RuntimeCheck(y.size(1) == kN && x.size(1) == kK && counts.size(0) <= kMaxExperts, "Unsupported Marlin GEMV shape");
  constexpr auto kernel = marlin_gemv<kNvfp4, kN, kK, kSplits, kWarps>;
  static const int max_clusters = [] {
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(kSplits * 1024);
    config.blockDim = dim3(kWarps * 32);
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {kSplits, 1, 1};
    config.attrs = &attr;
    config.numAttrs = 1;
    int clusters = 0;
    host::RuntimeDeviceCheck(cudaOccupancyMaxActiveClusters(&clusters, kernel, &config));
    return clusters;
  }();
  const int items = static_cast<int>(slots) * (kN / kTileN);
  const int clusters = items < max_clusters ? items : max_clusters;
  host::LaunchKernel(dim3(static_cast<unsigned>(clusters * kSplits)), kWarps * 32, x.device())
      .config({.use_pdl = true, .cluster_dim = dim3(kSplits, 1, 1)})(
          kernel,
          static_cast<const __nv_bfloat16*>(x.data_ptr()),
          static_cast<const uint32_t*>(weight.data_ptr()),
          static_cast<const uint8_t*>(scales.data_ptr()),
          static_cast<__nv_bfloat16*>(y.data_ptr()),
          static_cast<const int*>(counts.data_ptr()),
          static_cast<const int*>(offsets.data_ptr()),
          static_cast<const int*>(dst2src.data_ptr()),
          static_cast<int>(counts.size(0)),
          static_cast<int>(top_k));
}

}  // namespace sglang::inkling2_marlin_gemv
