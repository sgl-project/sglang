#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cooperative_groups.h>
#include <cstdint>
#include <cuda_bf16.h>

namespace sglang {
namespace fused_a_fp8 {

constexpr uint32_t kBlock = 128;
// K covered by one warp-wide step: the 4 lanes of a mma row each own 16 contiguous elements.
constexpr uint32_t kStep = 64;
constexpr uint32_t kMaxSplit = 8;

struct Config {
  uint32_t warps;
  uint32_t split;
  uint32_t batch;
  uint32_t row_tiles;
  bool f16;
  uint32_t min_ctas;  // __launch_bounds__ minimum CTAs per SM: caps registers so that every CTA of the grid is resident
  bool pdl;           // PDL launch: the grid starts while the previous kernel finishes
};

// Weight rows per mma tile.
constexpr uint32_t kFeatures = 16;

// Fallback for shapes without a measured entry: long-K projections want few warps per CTA plus a cluster
// K-split; short-K ones want one K block per warp and no split.
constexpr Config default_config(uint32_t k_blocks) {
  if (k_blocks >= 32) {
    const uint32_t split = k_blocks / 8 < kMaxSplit ? k_blocks / 8 : kMaxSplit;
    return {4, split, 1, 1, false, 1, true};
  }
  return {k_blocks, 1, 1, 1, false, 1, true};
}

// Measured on GB300 (cold L2, CUDA graph) for the GLM-5.2 TP4 shard shapes; tokens is 8 for M<=8 and 16 for M
// in 9..16. min_ctas = ceil(CTAs / 152 SMs): the register cap that keeps the whole grid resident in one wave.
constexpr Config tuned_config(uint32_t n, uint32_t k_blocks, uint32_t tokens) {
  if (n == 2624 && k_blocks == 48) return {8, 3, 1, 2, true, 2, true};   // fused qkv_a
  if (n == 4096 && k_blocks == 16) return {16, 1, 1, 1, true, 2, true};  // q_b, indexer wq_b
  // o_proj: 16 tokens leave too few registers for 2 row tiles per CTA. No PDL launch: behind the cuBLAS BMM that
  // feeds it, an early start measured ~4 us slower in the decode graph.
  if (n == 6144 && k_blocks == 32) {
    return tokens > 8 ? Config{8, 1, 1, 1, true, 3, false} : Config{8, 2, 1, 2, true, 3, false};
  }
  return default_config(k_blocks);
}

SGL_DEVICE void mma(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

SGL_DEVICE void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}

SGL_DEVICE void cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
}

SGL_DEVICE void cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
}

// Two E4M3 bytes (bits [15:8], [31:24]) -> BF16x2, exactly: the 7 magnitude bits placed under the BF16 exponent
// field read as value * 2^-120 (normals and subnormals alike); the BF16 multiply restores the scale.
SGL_DEVICE uint32_t widen(uint32_t q) {
  const uint32_t bits = ((q >> 4) & 0x07f007f0u) | (q & 0x80008000u);
  const __nv_bfloat162 rebase = __halves2bfloat162(__ushort_as_bfloat16(0x7B80), __ushort_as_bfloat16(0x7B80));
  const __nv_bfloat162 r = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&bits), rebase);
  return *reinterpret_cast<const uint32_t*>(&r);
}

// Four E4M3 bytes -> (bytes 0,1) and (bytes 2,3) as BF16x2.
SGL_DEVICE void dequant(uint32_t w, uint32_t& lo, uint32_t& hi) {
  lo = widen(__byte_perm(w, 0, 0x1404));
  hi = widen(__byte_perm(w, 0, 0x3424));
}

// FP16 operands (Config::f16): hardware E4M3 -> FP16 is exact, and so is BF16 -> FP16 within +-65504 (larger
// activations saturate).
SGL_DEVICE void mma_f16(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Four E4M3 bytes -> (bytes 0,1) and (bytes 2,3) as FP16x2.
SGL_DEVICE void dequant_f16(uint32_t w, uint32_t& lo, uint32_t& hi) {
  asm("{\n\t.reg .b16 l, h;\n\t"
      "mov.b32 {l, h}, %2;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, l;\n\t"
      "cvt.rn.f16x2.e4m3x2 %1, h;\n\t}"
      : "=r"(lo), "=r"(hi)
      : "r"(w));
}

SGL_DEVICE uint32_t bf16x2_to_f16x2(uint32_t w) {
  const float lo = __uint_as_float(w << 16);
  const float hi = __uint_as_float(w & 0xffff0000u);
  uint32_t r;
  asm("cvt.rn.satfinite.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

// Weights are streamed once per call: keep them out of L1 and let L2 fetch the whole 256B line.
SGL_DEVICE uint4 load_stream(const void* ptr) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr));
  return v;
}

// Activations come from the producer kernel. A plain load through the __restrict__ const pointer may be treated as
// invariant and scheduled above griddepcontrol.wait, so it is a volatile asm that stays ordered after the wait.
SGL_DEVICE uint4 load_activation(const void* ptr) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr)
               : "memory");
  return v;
}

SGL_DEVICE uint32_t word(const uint4& v, uint32_t i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// One K block of a warp, in registers. Per thread (row g = lane / 4, quarter t = lane % 4) and 64-wide step: 16
// contiguous weights of rows g and g + 8, 16 of token g's activations; the same K permutation keeps each mma exact.
template <uint32_t kRowTiles, uint32_t kTiles>
struct Stage {
  uint4 w[kRowTiles][2][2];
  uint4 x[kTiles][2][2];
  float scale;
};

// The A operand of the swapped GEMM, as in the BF16 kernel: the weights, so the 16 mma rows are output features and
// the 8 mma columns are tokens. Each warp loads the FP8 rows of its K blocks and their scales into registers.
template <uint32_t K, uint32_t kRowTiles, uint32_t kTiles, bool kPacked>
struct GmemLoaderA {
  SGL_DEVICE GmemLoaderA(
      const fp8_e4m3_t* weight,
      const void* scales,
      int64_t stride_n,
      int64_t stride_k,
      uint32_t n,
      uint32_t g,
      uint32_t t)
      : row_(weight + static_cast<size_t>(n + g) * K),
        scales_(scales),
        stride_n_(stride_n),
        stride_k_(stride_k),
        n_(n),
        t_(t) {}

  SGL_DEVICE void load(Stage<kRowTiles, kTiles>& s, uint32_t kb) const {
#pragma unroll
    for (uint32_t r = 0; r < kRowTiles; ++r) {
#pragma unroll
      for (uint32_t step = 0; step < 2; ++step) {
        const fp8_e4m3_t* p = row_ + r * kFeatures * K + kb * kBlock + step * kStep + t_ * 16;
        s.w[r][step][0] = load_stream(p);
        s.w[r][step][1] = load_stream(p + 8 * K);
      }
    }
    s.scale = load_scale(kb);
  }

 private:
  SGL_DEVICE float load_scale(uint32_t kb) const {
    if constexpr (kPacked) {
      // DeepGEMM replicates scales over N and packs four K-block exponents per int32.
      const auto word = static_cast<const uint32_t*>(scales_)[n_ * stride_n_ + (kb / 4) * stride_k_];
      return __uint_as_float(((word >> ((kb % 4) * 8)) & 0xffu) << 23);
    } else {
      return static_cast<const float*>(scales_)[(n_ / kBlock) * stride_n_ + kb * stride_k_];
    }
  }

  const fp8_e4m3_t* row_;
  const void* scales_;
  int64_t stride_n_;
  int64_t stride_k_;
  uint32_t n_;
  uint32_t t_;
};

// The B operand: the BF16 activations of token g (+ 8 per tile).
template <uint32_t K, uint32_t kRowTiles, uint32_t kTiles>
struct GmemLoaderB {
  SGL_DEVICE GmemLoaderB(const bf16_t* input, uint32_t m, uint32_t g, uint32_t t)
      : input_(input), m_(m), g_(g), t_(t) {}

  SGL_DEVICE void load(Stage<kRowTiles, kTiles>& s, uint32_t kb) const {
#pragma unroll
    for (uint32_t tile = 0; tile < kTiles; ++tile) {
      const uint32_t token = tile * 8 + g_;
#pragma unroll
      for (uint32_t step = 0; step < 2; ++step) {
        if (token < m_) {
          const bf16_t* p = input_ + token * K + kb * kBlock + step * kStep + t_ * 16;
          s.x[tile][step][0] = load_activation(p);
          s.x[tile][step][1] = load_activation(p + 8);
        } else {
          s.x[tile][step][0] = make_uint4(0, 0, 0, 0);
          s.x[tile][step][1] = make_uint4(0, 0, 0, 0);
        }
      }
    }
  }

 private:
  const bf16_t* input_;
  uint32_t m_;
  uint32_t g_;
  uint32_t t_;
};

// Widens the FP8 weights in registers, multiplies on the tensor cores and reduces the K blocks: over the warps of a
// CTA through shared memory, then over the ranks of a cluster through distributed shared memory, in a fixed order.
template <uint32_t N, uint32_t kTokens, uint32_t kWarps, uint32_t kSplit, uint32_t kRowTiles, bool kF16>
struct MmaComputer {
  static constexpr uint32_t kTiles = kTokens / 8;
  static constexpr uint32_t kRows = kRowTiles * kFeatures;
  static constexpr uint32_t kOutputs = kTokens * kRows;

  // smem: [warp][token * kRows + feature] for this CTA's warps; with a K-split, one slice per rank follows for rank 0
  // to sum.
  SGL_DEVICE MmaComputer(char* smem, uint32_t warp, uint32_t lane, uint32_t rank)
      : partial_(reinterpret_cast<float (*)[kOutputs]>(smem)), warp_(warp), lane_(lane), rank_(rank) {}

  SGL_DEVICE void accumulate(Stage<kRowTiles, kTiles>& s) {
    float acc[kRowTiles][kTiles][4]{};
    if constexpr (kF16) {
      convert_activations_f16(s);
      compute_f16(acc, s);
    } else {
      compute(acc, s);
    }
#pragma unroll
    for (uint32_t r = 0; r < kRowTiles; ++r) {
#pragma unroll
      for (uint32_t tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          // Scaling after each K=128 partial preserves the checkpoint block boundaries.
          total_[r][tile][j] = fmaf(acc[r][tile][j], s.scale, total_[r][tile][j]);
        }
      }
    }
  }

  // n is the first output feature of this CTA. Pairs with the cluster arrive at kernel start.
  SGL_DEVICE void epi(bf16_t* out, uint32_t n, uint32_t m) {
    float* mine = partial_[warp_];
#pragma unroll
    for (uint32_t r = 0; r < kRowTiles; ++r) {
#pragma unroll
      for (uint32_t tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          const uint32_t feature = r * kFeatures + lane_ / 4 + (j / 2) * 8;
          const uint32_t token = tile * 8 + (lane_ % 4) * 2 + j % 2;
          mine[token * kRows + feature] = total_[r][tile][j];
        }
      }
    }
    __syncthreads();
    // Fixed summation order (warps, then ranks) keeps the result deterministic.
    const auto sum_warps = [&](uint32_t i) {
      float sum = 0.0f;
#pragma unroll
      for (uint32_t w = 0; w < kWarps; ++w) {
        sum += partial_[w][i];
      }
      return sum;
    };
    if constexpr (kSplit == 1) {
      for (uint32_t i = threadIdx.x; i < m * kRows; i += kWarps * 32) {
        out[(i / kRows) * N + n + i % kRows] = device::cast<bf16_t>(sum_warps(i));
      }
    } else {
      // Rank 0's CTA is running, so its shared memory can be written.
      cluster_wait();
      float* slice = cooperative_groups::this_cluster().map_shared_rank(partial_[kWarps + rank_], 0);
      for (uint32_t i = threadIdx.x; i < m * kRows; i += kWarps * 32) {
        slice[i] = sum_warps(i);
      }
      // One barrier: every rank's slice has landed in rank 0's shared memory.
      cluster_arrive_release();
      cluster_wait();
      if (rank_ == 0) {
        for (uint32_t i = threadIdx.x; i < m * kRows; i += kWarps * 32) {
          float sum = 0.0f;
#pragma unroll
          for (uint32_t s = 0; s < kSplit; ++s) {
            sum += partial_[kWarps + s][i];
          }
          out[(i / kRows) * N + n + i % kRows] = device::cast<bf16_t>(sum);
        }
      }
    }
  }

 private:
  static SGL_DEVICE void compute(float (&acc)[kRowTiles][kTiles][4], const Stage<kRowTiles, kTiles>& s) {
#pragma unroll
    for (uint32_t step = 0; step < 2; ++step) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
#pragma unroll
        for (uint32_t r = 0; r < kRowTiles; ++r) {
          uint32_t a[4];
          dequant(word(s.w[r][step][0], j), a[0], a[2]);
          dequant(word(s.w[r][step][1], j), a[1], a[3]);
#pragma unroll
          for (uint32_t tile = 0; tile < kTiles; ++tile) {
            const uint32_t b[2] = {
                word(s.x[tile][step][j / 2], (2 * j) % 4), word(s.x[tile][step][j / 2], (2 * j + 1) % 4)};
            mma(acc[r][tile], a, b);
          }
        }
      }
    }
  }

  static SGL_DEVICE void convert_activations_f16(Stage<kRowTiles, kTiles>& s) {
#pragma unroll
    for (uint32_t tile = 0; tile < kTiles; ++tile) {
#pragma unroll
      for (uint32_t step = 0; step < 2; ++step) {
#pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
          uint4& v = s.x[tile][step][i];
          v.x = bf16x2_to_f16x2(v.x);
          v.y = bf16x2_to_f16x2(v.y);
          v.z = bf16x2_to_f16x2(v.z);
          v.w = bf16x2_to_f16x2(v.w);
        }
      }
    }
  }

  static SGL_DEVICE void compute_f16(float (&acc)[kRowTiles][kTiles][4], const Stage<kRowTiles, kTiles>& s) {
#pragma unroll
    for (uint32_t step = 0; step < 2; ++step) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
#pragma unroll
        for (uint32_t r = 0; r < kRowTiles; ++r) {
          uint32_t a[4];
          dequant_f16(word(s.w[r][step][0], j), a[0], a[2]);
          dequant_f16(word(s.w[r][step][1], j), a[1], a[3]);
#pragma unroll
          for (uint32_t tile = 0; tile < kTiles; ++tile) {
            const uint32_t b[2] = {
                word(s.x[tile][step][j / 2], (2 * j) % 4), word(s.x[tile][step][j / 2], (2 * j + 1) % 4)};
            mma_f16(acc[r][tile], a, b);
          }
        }
      }
    }
  }

  float (*partial_)[kOutputs];
  uint32_t warp_;
  uint32_t lane_;
  uint32_t rank_;
  float total_[kRowTiles][kTiles][4]{};
};

template <
    uint32_t K,
    uint32_t N,
    uint32_t kTokens,
    uint32_t kWarps,
    uint32_t kSplit,
    uint32_t kBatch,
    uint32_t kRowTiles,
    bool kF16,
    bool kPacked,
    bool kUsePDL,
    uint32_t kMinCtas>
__global__ __launch_bounds__(kWarps * 32, kMinCtas) __cluster_dims__(kSplit, 1, 1) void kernel(
    bf16_t* __restrict__ out,
    const bf16_t* __restrict__ input,
    const fp8_e4m3_t* __restrict__ weight,
    const void* __restrict__ scales,
    int64_t stride_n,
    int64_t stride_k,
    uint32_t m) {
  constexpr uint32_t kTiles = kTokens / 8;
  constexpr uint32_t kRows = kRowTiles * kFeatures;
  constexpr uint32_t kBlocks = K / kBlock;
  static_assert(kBlock % kRows == 0, "the rows of a CTA share one weight-scale block");
  extern __shared__ __align__(16) char smem[];
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t g = lane / 4;
  const uint32_t t = lane % 4;
  const uint32_t rank = blockIdx.x % kSplit;
  const uint32_t n = (blockIdx.x / kSplit) * kRows;

  GmemLoaderA<K, kRowTiles, kTiles, kPacked> a_loader(weight, scales, stride_n, stride_k, n, g, t);
  GmemLoaderB<K, kRowTiles, kTiles> b_loader(input, m, g, t);
  MmaComputer<N, kTokens, kWarps, kSplit, kRowTiles, kF16> mma_computer(smem, warp, lane, rank);
  if constexpr (kSplit > 1) {
    // Rank 0's shared memory is written remotely in the epilogue; its CTA must have started by then.
    cluster_arrive_relaxed();
  }

  static_assert(kSplit * kWarps <= kBlocks, "every warp owns at least one K block");
  const uint32_t first_base = rank * kWarps + warp;
  uint32_t base = first_base;
#pragma unroll 1
  do {
    Stage<kRowTiles, kTiles> stage[kBatch];
    if (base == first_base) {
      device::PDLWaitPrimary<kUsePDL>();
    }
#pragma unroll
    for (uint32_t u = 0; u < kBatch; ++u) {
      const uint32_t kb = base + u * kSplit * kWarps;
      if (kb < kBlocks) {
        a_loader.load(stage[u], kb);
        b_loader.load(stage[u], kb);
      }
    }
#pragma unroll
    for (uint32_t u = 0; u < kBatch; ++u) {
      const uint32_t kb = base + u * kSplit * kWarps;
      if (kb < kBlocks) {
        mma_computer.accumulate(stage[u]);
      }
    }
    base += kSplit * kWarps * kBatch;
  } while (base < kBlocks);
  mma_computer.epi(out, n, m);
  // The dependent's griddepcontrol.wait only covers writes issued before this trigger, so it must follow the output
  // stores.
  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace fused_a_fp8

/// \brief Small-M W8A16 GEMM: out[M, N] = input[M, K] @ dequant(weight[N, K]).T for M in [1, 16].
/// \tparam K, N Weight shape; the launch configuration comes from tuned_config (see its table).
/// \tparam kPacked Scales are DeepGEMM's packed UE8M0 int32 instead of FP32 per 128x128 block.
/// \tparam kUsePDL Launch with programmatic dependent launch (a config may still opt out).
template <uint32_t K, uint32_t N, bool kPacked, bool kUsePDL>
struct DSV3FusedAGemmFP8Kernel {
  static_assert(K > 0 && K % 128 == 0);
  static_assert(N > 0 && N % 16 == 0);

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView scales,
      const tvm::ffi::TensorView out) {
    using namespace host;
    auto m = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({m, K}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(input);
    TensorMatcher({N, K}).with_dtype<fp8_e4m3_t>().with_device(device).ensure_alignment(16).verify(weight);
    TensorMatcher({m, N}).with_dtype<bf16_t>().with_device(device).verify(out);
    if constexpr (kPacked) {
      TensorMatcher({N, div_ceil(K / 128, 4u)})
          .with_dtype<int32_t>()
          .with_strides({-1, -1})
          .with_device(device)
          .verify(scales);
    } else {
      TensorMatcher({div_ceil(N, 128u), K / 128})
          .with_dtype<float>()
          .with_strides({-1, -1})
          .with_device(device)
          .verify(scales);
    }
    CHECK_HOST(m.unwrap() >= 1 && m.unwrap() <= 16) << "num_tokens must be in [1, 16]";
    CHECK_HOST(scales.stride(0) > 0 && scales.stride(1) > 0) << "scale strides must be positive";
    const auto launch = [&]<uint32_t kTokens>() {
      constexpr auto kConfig = fused_a_fp8::tuned_config(N, K / fused_a_fp8::kBlock, kTokens);
      static_assert(N % (fused_a_fp8::kFeatures * kConfig.row_tiles) == 0, "N must be a multiple of the rows per CTA");
      // Every warp's partials, plus with a K-split one slice per rank for rank 0 to sum.
      constexpr size_t kSmemBytes = (kConfig.split == 1 ? kConfig.warps : kConfig.warps + kConfig.split) * kTokens *
                                    kConfig.row_tiles * fused_a_fp8::kFeatures * sizeof(float);
      constexpr bool kPdl = kUsePDL && kConfig.pdl;
      auto kernel = fused_a_fp8::kernel<
          K,
          N,
          kTokens,
          kConfig.warps,
          kConfig.split,
          kConfig.batch,
          kConfig.row_tiles,
          kConfig.f16,
          kPacked,
          kPdl,
          kConfig.min_ctas>;
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
      LaunchKernel(
          N / (fused_a_fp8::kFeatures * kConfig.row_tiles) * kConfig.split,
          kConfig.warps * 32,
          device.unwrap(),
          kSmemBytes)
          .enable_pdl(kPdl)(
              kernel,
              static_cast<bf16_t*>(out.data_ptr()),
              static_cast<const bf16_t*>(input.data_ptr()),
              static_cast<const fp8_e4m3_t*>(weight.data_ptr()),
              scales.data_ptr(),
              scales.stride(0),
              scales.stride(1),
              static_cast<uint32_t>(m.unwrap()));
    };
    if (m.unwrap() <= 8) {
      launch.template operator()<8>();
    } else {
      launch.template operator()<16>();
    }
  }
};

}  // namespace sglang
