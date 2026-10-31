// Tensor-core tiny GEMM (CUDA, sm90+, bf16 weights): mma.sync m16n8k16, fp32 accumulation. A CTA owns 16 rows of w (the
// MMA M) and up to kMTiles token tiles of 8 rows of x (the MMA N); a thread-block cluster splits K over its CTAs and
// the warps of a CTA split it further, in 64-element chunks. Partial sums are added over the warps in shared memory,
// then reduce-scattered over the cluster through DSMEM. Token tiles are spread over blockIdx.y. A token's summation
// order depends only on (N, K, schedule), never on M or on the other tokens.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include "common.cuh"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace sglang {

namespace device::ptx {

// D += A * B on the tensor cores: m16n8k16, bf16 operands, fp32 accumulator.
SGL_DEVICE void mma_bf16_m16n8k16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

SGL_DEVICE void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

SGL_DEVICE void cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

// Publishes the mbarrier init to the cluster; the relaxed cluster arrive after it needs no ordering of its own.
SGL_DEVICE void fence_mbarrier_init_release_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

SGL_DEVICE uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t out;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(out) : "r"(addr), "r"(rank));
  return out;
}

SGL_DEVICE void st_async_b32(uint32_t dst, float value, uint32_t bar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(dst), "f"(value), "r"(bar)
      : "memory");
}

}  // namespace device::ptx

namespace device::tiny_gemm {

template <uint32_t kBytes>
using RegVec = AlignedVector<uint32_t, kBytes / sizeof(uint32_t)>;

// Word `j` of a row of registers loaded as `kNum` vectors.
template <std::size_t N, std::size_t kNum>
SGL_DEVICE uint32_t word(const AlignedVector<uint32_t, N> (&v)[kNum], uint32_t j) {
  return v[j / N][j % N];
}

// sums[i] += x[token tile i] . w over one 64-element chunk. A lane holds the same 16 elements of the chunk
// (MMATrait::kGroup) for rows g and g + 8 of w and for row g of each token tile of x; every k-step reads the
// same positions of both, so the MMA sees one permutation of K for A and B.
template <typename Trait, uint32_t kMTiles>
SGL_DEVICE void mma_chunk(
    float (&sums)[kMTiles][4],
    const RegVec<Trait::kVecBytes> (&w)[2][Trait::kVecsPerRow],
    const RegVec<Trait::kVecBytes> (&x)[kMTiles][Trait::kVecsPerRow]) {
  // k-step s takes bf16 pairs 2 s and 2 s + 1.
#pragma unroll
  for (uint32_t s = 0; s < 4; ++s) {
    const uint32_t a[4] = {word(w[0], 2 * s), word(w[1], 2 * s), word(w[0], 2 * s + 1), word(w[1], 2 * s + 1)};
#pragma unroll
    for (uint32_t i = 0; i < kMTiles; ++i) {
      const uint32_t b[2] = {word(x[i], 2 * s), word(x[i], 2 * s + 1)};
      ptx::mma_bf16_m16n8k16(sums[i], a, b);
    }
  }
}

}  // namespace device::tiny_gemm

template <uint32_t N_, uint32_t K_, uint32_t kNumWarps_, uint32_t kClusterSize_>
struct MMATrait {
  static constexpr uint32_t N = N_;
  static constexpr uint32_t K = K_;
  static constexpr uint32_t kNumWarps = kNumWarps_;
  static constexpr uint32_t kClusterSize = kClusterSize_;
  static constexpr uint32_t kBlockSize = kNumWarps * device::kWarpThreads;
  // A lane quad covers a 64-element chunk of K, 16 elements per lane.
  static constexpr uint32_t kTileN = 16;
  static constexpr uint32_t kTileM = 8;
  static constexpr uint32_t kChunkK = 64;
  static constexpr uint32_t kNumTiles = N / kTileN;
  static constexpr uint32_t kNumChunks = K / kChunkK;
  static constexpr uint32_t kNumSplits = kNumWarps * kClusterSize;
  static constexpr uint32_t kMaxChunks = host::div_ceil(kNumChunks, kNumSplits);
  static constexpr bool kEvenSplit = kNumChunks % kNumSplits == 0;
  // A lane's 16 elements (32 bytes) of a chunk, of w and of x alike, are groups of kGroup consecutive elements
  // (one vector load), lane t's group at t * kGroup of every 4 * kGroup, so that each load instruction covers
  // whole 32-byte sectors.
  static constexpr uint32_t kVecBytes = std::min<uint32_t>(32, device::kMaxVecBytes);
  static constexpr uint32_t kVecsPerRow = 32 / kVecBytes;
  static constexpr uint32_t kGroup = kVecBytes / sizeof(bf16_t);

  // Two CTAs per SM keep clusters of 4 and 8 co-resident (one CTA per SM fits only 15 clusters of 8 on B200
  // and H200). CTAs above 8 warps are only scheduled with clusters of at most 2, which always fit one wave.
  static constexpr uint32_t kCTAsPerSM = kBlockSize <= 256 ? 2 : 1;
  // A wave holds a second CTA per SM only when the CTAs' weight slices are small: the SM's weight bytes are the
  // bound.
  static constexpr uint32_t kWaveCTAsPerSM = kTileN * K / kClusterSize * sizeof(bf16_t) <= 32 * 1024 ? kCTAsPerSM : 1;

  static_assert(N % kTileN == 0 && K % kChunkK == 0);
  static_assert(kNumSplits <= kNumChunks, "every warp needs at least one chunk");
  static_assert(kClusterSize >= 1 && kClusterSize <= 8, "portable cluster sizes only");
  static_assert(kNumWarps <= 8 || kClusterSize <= 2, "large CTAs need small clusters to stay co-resident");
};

template <typename Trait, uint32_t kMTiles, typename Out, bool kUsePDL>
__global__ __launch_bounds__(Trait::kBlockSize, Trait::kCTAsPerSM) void tiny_mma_kernel(
    const __grid_constant__ TinyGEMMParams params) {
  using namespace device;
  using namespace device::tiny_gemm;
  constexpr uint32_t K = Trait::K;
  constexpr uint32_t kNumWarps = Trait::kNumWarps;
  constexpr uint32_t kClusterSize = Trait::kClusterSize;
  constexpr uint32_t kMaxChunks = Trait::kMaxChunks;
  constexpr uint32_t kNumTokens = kMTiles * Trait::kTileM;
  // A CTA's tile of outputs, kNumTokens x 16; each CTA of the cluster finishes a slice of it.
  constexpr uint32_t kNumOutputs = kNumTokens * Trait::kTileN;
  // Contiguous slices of kSlice outputs; with a cluster size that does not divide kNumOutputs the last is shorter.
  constexpr uint32_t kSlice = host::div_ceil(kNumOutputs, kClusterSize);

  // Partial sums in MMA fragment order: output o = (tile i, lane, r) at (i * 32 + lane) * 4 + r.
  __shared__ alignas(16) float s_partial[kNumWarps][kNumOutputs];
  __shared__ float s_inbox[kClusterSize][kClusterSize > 1 ? kSlice : 1];
  __shared__ alignas(8) uint64_t s_bar;

  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  const uint32_t g = lane_id / 4;  // MMA group: a row of w (expert) and a row of x (token)
  const uint32_t t = lane_id % 4;  // owns 16 elements of every 64-element chunk, see kGroup
  const uint32_t rank = blockIdx.x % kClusterSize;
  const uint32_t tile = blockIdx.x / kClusterSize;
  const uint32_t token_begin = blockIdx.y * kNumTokens;
  const uint32_t split = rank * kNumWarps + warp_id;
  const uint32_t chunk_begin = split * Trait::kNumChunks / Trait::kNumSplits;
  const uint32_t num_chunks = (split + 1) * Trait::kNumChunks / Trait::kNumSplits - chunk_begin;

  // The weight does not depend on the producer: load all of it before the wait.
  constexpr uint32_t kGroup = Trait::kGroup;
  constexpr auto offset = [](uint32_t e) { return e / kGroup * 4 * kGroup + e % kGroup; };
  const auto w_row =
      static_cast<const bf16_t*>(params.w) + (tile * Trait::kTileN + g) * K + chunk_begin * 64 + t * kGroup;
  RegVec<Trait::kVecBytes> w_reg[kMaxChunks][2][Trait::kVecsPerRow];
#pragma unroll
  for (uint32_t c = 0; c < kMaxChunks; ++c) {
    if (Trait::kEvenSplit || c < num_chunks) {
#pragma unroll
      for (uint32_t h = 0; h < 2; ++h) {
#pragma unroll
        for (uint32_t v = 0; v < Trait::kVecsPerRow; ++v) {
          w_reg[c][h][v].load(w_row + h * 8 * K + c * 64 + offset(v * kGroup));
        }
      }
    }
  }

  if constexpr (kClusterSize > 1) {
    if (threadIdx.x == 0) {
      ptx::mbar_init(&s_bar, 1);
      ptx::fence_mbarrier_init_release_cluster();
      const uint32_t slice = std::min(kSlice, kNumOutputs - rank * kSlice);
      ptx::mbar_arrive_expect_tx(&s_bar, kClusterSize * slice * sizeof(float));
    }
    ptx::cluster_arrive_relaxed();
  }

  PDLWaitPrimary<kUsePDL>();

  const bf16_t* x_row[kMTiles];
#pragma unroll
  for (uint32_t i = 0; i < kMTiles; ++i) {
    const auto m = std::min(token_begin + i * 8 + g, params.num_tokens - 1);
    x_row[i] = params.x + m * params.stride_x + chunk_begin * 64 + t * kGroup;
  }

  float acc[kMTiles][4];
#pragma unroll
  for (uint32_t c = 0; c < kMaxChunks; ++c) {
    if (Trait::kEvenSplit || c < num_chunks) {
      RegVec<Trait::kVecBytes> x_reg[kMTiles][Trait::kVecsPerRow];
#pragma unroll
      for (uint32_t i = 0; i < kMTiles; ++i) {
#pragma unroll
        for (uint32_t v = 0; v < Trait::kVecsPerRow; ++v) {
          x_reg[i][v].load(assume_global(x_row[i] + c * 64 + offset(v * kGroup)));
        }
      }
      // The tensor cores truncate while accumulating: keep their sums to one chunk and add the chunks in fp32.
      float chunk_acc[kMTiles][4] = {};
      mma_chunk<Trait>(chunk_acc, w_reg[c], x_reg);
#pragma unroll
      for (uint32_t i = 0; i < kMTiles; ++i) {
#pragma unroll
        for (uint32_t r = 0; r < 4; ++r) {
          // Start from the first chunk, not from 0: one dependent FADD less.
          acc[i][r] = c == 0 ? chunk_acc[i][r] : acc[i][r] + chunk_acc[i][r];
        }
      }
    }
  }

  const auto out = static_cast<Out*>(params.out) + tile * Trait::kTileN;
  // acc[i] = {(row g, token 2t), (g, 2t + 1), (g + 8, 2t), (g + 8, 2t + 1)} of token tile i.
  const auto store = [&](uint32_t o, float value) {
    const uint32_t lane = o / 4 % kWarpThreads;
    const uint32_t token = token_begin + o / 128 * 8 + lane % 4 * 2 + o % 2;
    const uint32_t row = lane / 4 + o % 4 / 2 * 8;
    if (token < params.num_tokens) out[token * Trait::N + row] = cast<Out>(value);
  };
  if constexpr (kNumWarps == 1 && kClusterSize == 1) {
    PDLTriggerSecondary<kUsePDL>();
#pragma unroll
    for (uint32_t i = 0; i < kMTiles; ++i) {
#pragma unroll
      for (uint32_t r = 0; r < 4; ++r) {
        store((i * kWarpThreads + lane_id) * 4 + r, acc[i][r]);
      }
    }
    return;
  }

#pragma unroll
  for (uint32_t i = 0; i < kMTiles; ++i) {
    AlignedVector<float, 4> frag;
#pragma unroll
    for (uint32_t r = 0; r < 4; ++r) {
      frag[r] = acc[i][r];
    }
    frag.store(s_partial[warp_id], i * kWarpThreads + lane_id);
  }
  __syncthreads();

  const auto cta_sum = [&](uint32_t o) {
    float sum = s_partial[0][o];
#pragma unroll
    for (uint32_t w = 1; w < kNumWarps; ++w) {
      sum += s_partial[w][o];
    }
    return sum;
  };

  if constexpr (kClusterSize == 1) {
    PDLTriggerSecondary<kUsePDL>();
    for (uint32_t o = threadIdx.x; o < kNumOutputs; o += Trait::kBlockSize) {
      store(o, cta_sum(o));
    }
  } else {
    ptx::cluster_wait();  // the peers' mbarriers are initialized
    const uint32_t inbox = ptx::to_shared(&s_inbox[rank][0]);
    const uint32_t bar = ptx::to_shared(&s_bar);
    for (uint32_t o = threadIdx.x; o < kNumOutputs; o += Trait::kBlockSize) {
      const uint32_t dst = o / kSlice;
      const uint32_t byte_offset = (o % kSlice) * sizeof(float);
      ptx::st_async_b32(ptx::mapa(inbox + byte_offset, dst), cta_sum(o), ptx::mapa(bar, dst));
    }
    ptx::mbar_wait_parity(&s_bar, 0);
    PDLTriggerSecondary<kUsePDL>();
    const uint32_t slice = std::min(kSlice, kNumOutputs - rank * kSlice);
    for (uint32_t o = threadIdx.x; o < slice; o += Trait::kBlockSize) {
      float sum = s_inbox[0][o];
#pragma unroll
      for (uint32_t r = 1; r < kClusterSize; ++r) {
        sum += s_inbox[r][o];
      }
      store(rank * kSlice + o, sum);
    }
  }
}

// kMaxMTiles: token tiles of 8 rows per CTA, at most; larger batches are tiled over blockIdx.y.
template <
    uint32_t N,
    uint32_t K,
    typename Out,
    uint32_t kNumWarps,
    uint32_t kClusterSize,
    uint32_t kMaxMTiles,
    bool kUsePDL>
struct TinyMMAKernel {
  using Trait = MMATrait<N, K, kNumWarps, kClusterSize>;
  using KernelFn = void (*)(TinyGEMMParams);

  template <std::size_t... I>
  static constexpr auto make_table(std::index_sequence<I...>) {
    return std::array<KernelFn, kMaxMTiles>{tiny_mma_kernel<Trait, I + 1, Out, kUsePDL>...};
  }
  static constexpr auto kTable = make_table(std::make_index_sequence<kMaxMTiles>{});

  static constexpr uint32_t kNumCTAs = Trait::kNumTiles * kClusterSize;

  // CTAs of one wave. A cluster's CTAs share one GPC, so clusters leave SMs idle: ask the driver.
  static uint32_t s_wave_ctas(int32_t device_id) {
    static host::runtime::DeviceCacheMap<uint32_t, 0> s_cache;
    return s_cache.get_cached(device_id, [](int32_t id) {
      const uint32_t sm_slots = host::runtime::get_sm_count(id) * Trait::kWaveCTAsPerSM;
      if constexpr (kClusterSize == 1) {
        return sm_slots;
      } else {
        cudaLaunchAttribute attr = {};
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {kClusterSize, 1, 1};
        cudaLaunchConfig_t config = {};
        config.gridDim = dim3{kNumCTAs};
        config.blockDim = dim3{Trait::kBlockSize};
        config.attrs = &attr;
        config.numAttrs = 1;
        int num_clusters = 0;
        CHECK_CUDA(cudaOccupancyMaxActiveClusters(&num_clusters, kTable[kMaxMTiles - 1], &config));
        return std::min(static_cast<uint32_t>(num_clusters) * kClusterSize, sm_slots);
      }
    });
  }

  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView w, const tvm::ffi::TensorView out) {
    using namespace host;
    DLDevice device;
    const auto params =
        tiny_gemm::make_params<N, K, bf16_t, Out>(x, w, out, Trait::kVecBytes, Trait::kVecBytes, device);
    if (params.num_tokens == 0) return;
    const uint32_t num_m_tiles = div_ceil(params.num_tokens, Trait::kTileM);
    // As many CTA rows (fewest token tiles per CTA) as one wave holds, at least as many as kMaxMTiles needs.
    // The tiling never changes a token's summation order.
    const uint32_t wave_rows = s_wave_ctas(device.device_id) / kNumCTAs;
    const uint32_t rows = std::max(div_ceil(num_m_tiles, kMaxMTiles), std::min(num_m_tiles, wave_rows));
    // Spread the token tiles evenly over the CTA rows.
    const uint32_t m_tiles = div_ceil(num_m_tiles, rows);
    const uint32_t grid_y = div_ceil(num_m_tiles, m_tiles);
    const auto cluster = kClusterSize > 1 ? std::optional{dim3{kClusterSize, 1, 1}} : std::nullopt;
    LaunchKernel(dim3(kNumCTAs, grid_y), Trait::kBlockSize, device)
        .config({.use_pdl = kUsePDL, .cluster_dim = cluster})(kTable[m_tiles - 1], params);
  }
};

}  // namespace sglang
