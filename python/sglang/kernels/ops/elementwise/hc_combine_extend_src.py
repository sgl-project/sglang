"""CUDA source of the extend-size HC boundary kernel (combine + next norm +
next gate), bit-identical to the stock hc_combine -> grouped_gemma_rmsnorm
chain.

Kept as a Python string so the module ships with the Python sources; the
loader in hc_combine_extend writes it to a content-addressed file under the
JIT cache root before compiling.
"""

CUDA_SOURCE = r'''#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

// The RMS sums replay grouped_gemma_rmsnorm's Blackwell path (one 32-byte
// vector per thread); elsewhere that kernel sums in another order, so refuse
// to build and leave the caller on the stock boundary.
#if !SGL_ARCH_BLACKWELL_OR_GREATER
#error "hc_combine_extend requires the Blackwell grouped_gemma_rmsnorm reduction order"
#endif

namespace sglang {

struct HcCombineExtendParams {
  const void* block_output;        // [M, H]
  const void* residual;            // [M, HC * H]
  const void* normed_residual;     // [M, HC * H]   (gate source when !kHasPartials)
  const void* inject_weight;       // [HC, HC * H]  (gate weight when !kHasPartials)
  const float* partials;           // [M, HC] fp32  (gate dot totals when kHasPartials)
  const void* norm_weight;         // [HC * H]      (next read's per-branch norm, kDoNorm)
  const void* next_inject_weight;  // [HC, HC * H]  (next read's gate weight, kNextGate)
  void* output;                    // [M, HC * H]
  void* normed_output;             // [M, HC * H]   (kDoNorm)
  float* next_partials;            // [M, HC] fp32  (kNextGate)
  float eps;
};

/**
 * \brief Extend-size HyperConnection boundary: combine, then (optionally) the
 * next read's per-branch Gemma RMSNorm and the next read's gate dot totals.
 *
 * Bit-identical by construction to the stock extend chain
 *   out    = hc_combine(y, r, normed_r, W)                 (hc_combine.cuh)
 *   normed = grouped_gemma_rmsnorm(out, w_norm, H)          (Blackwell path)
 *   next   = the gate dot of the next hc_combine(., out, normed, W_next)
 * One CTA of 256 threads per row with the stock hc_combine thread->vector
 * mapping (thread t owns 16-byte vectors t + 256 j), so the gate dots are
 * accumulated exactly as in hc_combine. The RMS sums replay the
 * grouped_gemma_rmsnorm order: that kernel's thread k sums the 16 elements of
 * chunk k (= vectors 2k, 2k+1) sequentially, butterflies within its warp and
 * then butterflies the per-warp sums (padded with zeros) in warp 0. Here the
 * even vector's 4-pair partial is shuffled to the odd neighbour lane, which
 * finishes the chunk; the chunk sums go through shared memory and warp g
 * replays group g's two butterfly levels.
 *
 * Saves, per row and boundary, the norm kernel's re-read of the combined row
 * and (with partials) the gate's read of the normed row, and keeps every HBM
 * load in flight before the first reduction. bf16 x bf16 products are exact
 * in fp32, so FMA contraction choices cannot change the sums; only the
 * addition order matters, and it is the stock one.
 *
 * \tparam kHasPartials  current gate from [M, HC] fp32 dot totals (else from
 *                       normed_residual . inject_weight, as hc_combine)
 * \tparam kDoNorm       also write the next read's per-branch normalization
 * \tparam kNextGate     also write the next read's [M, HC] gate dot totals
 * \tparam kMinBlocks    launch-bounds occupancy target (4 -> 64 registers)
 * \tparam kPrefetchW    load the norm weight before the reductions
 */
template <
    int64_t kHcCount,
    int64_t kHiddenSize,
    bool kUsePDL,
    typename Float,
    bool kHasPartials,
    bool kDoNorm,
    bool kNextGate,
    int kMinBlocks,
    bool kPrefetchW>
__global__ __launch_bounds__(256, kMinBlocks) void hc_combine_extend_kernel(const HcCombineExtendParams __grid_constant__ params) {
  using namespace device;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;  // 8 elements, 16 bytes
  constexpr uint32_t kVecLen = 8;
  constexpr uint32_t kNumThreads = 256;
  constexpr int64_t kRowSize = kHcCount * kHiddenSize;
  constexpr uint32_t kVecsPerRow = kRowSize / kVecLen;
  constexpr uint32_t kVecsPerThread = kVecsPerRow / kNumThreads;
  constexpr uint32_t kVecsPerBranch = kHiddenSize / kVecLen;
  constexpr uint32_t kNumWarps = kNumThreads / kWarpThreads;
  // grouped_gemma_rmsnorm (Blackwell): kHiddenSize / 16 threads per group,
  // each summing one 16-element chunk.
  constexpr uint32_t kChunksPerRow = kRowSize / 16;
  constexpr uint32_t kNormWarpsPerGroup = kHiddenSize / 16 / kWarpThreads;
  static_assert(kVecsPerRow % kNumThreads == 0);
  static_assert(kHiddenSize % 512 == 0, "norm group must map onto whole warps");
  static_assert(kHcCount <= kNumWarps, "one warp per branch norm");
  static_assert(kNormWarpsPerGroup <= kWarpThreads);
  static_assert(!kNextGate || kDoNorm, "the next gate reads the normalized row");

  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  const uint32_t m = blockIdx.x;
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane = threadIdx.x % kWarpThreads;

  const auto y_ptr = pointer::offset<Float>(params.block_output, static_cast<int64_t>(m) * kHiddenSize);
  const auto r_ptr = pointer::offset<Float>(params.residual, static_cast<int64_t>(m) * kRowSize);
  const auto out_ptr = pointer::offset<Float>(params.output, static_cast<int64_t>(m) * kRowSize);

  __shared__ float red_smem[kHcCount][kNumWarps];
  __shared__ float a_shared[kHcCount];
  __shared__ float chunk_sums[kDoNorm ? kChunksPerRow : 1];
  __shared__ float norm_factor[kHcCount];

  PDLWaitPrimary<kUsePDL>();

  // Issue every HBM load of the row up front.
  Storage r_vec[kVecsPerThread];
  Storage y_vec[kVecsPerThread];
#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    const uint32_t vec_idx = threadIdx.x + j * kNumThreads;
    r_vec[j] = gmem.load(r_ptr, j);
    y_vec[j].load(y_ptr, vec_idx % kVecsPerBranch);
  }

  if constexpr (kHasPartials) {
    if (threadIdx.x < kHcCount) {
      const float total = params.partials[static_cast<int64_t>(m) * kHcCount + threadIdx.x];
      a_shared[threadIdx.x] = 2.0f / (1.0f + math::exp(-total / kHcCount));
    }
  } else {
    // Stock hc_combine phase 1, verbatim order.
    const auto n_ptr = pointer::offset<Float>(params.normed_residual, static_cast<int64_t>(m) * kRowSize);
    const auto w_ptr = static_cast<const Float*>(params.inject_weight);
    Storage n_vec[kVecsPerThread];
#pragma unroll
    for (uint32_t j = 0; j < kVecsPerThread; ++j) {
      n_vec[j] = gmem.load(n_ptr, j);
    }
    float acc[kHcCount];
#pragma unroll
    for (int c = 0; c < kHcCount; ++c) {
      const auto wc_ptr = w_ptr + static_cast<int64_t>(c) * kRowSize;
      float sum = 0.0f;
#pragma unroll
      for (uint32_t j = 0; j < kVecsPerThread; ++j) {
        const Storage w_vec = gmem.load(wc_ptr, j);
#pragma unroll
        for (uint32_t i = 0; i < kVecLen / 2; ++i) {
          const auto [nx, ny] = cast<fp32x2_t>(n_vec[j][i]);
          const auto [wx, wy] = cast<fp32x2_t>(w_vec[i]);
          sum += nx * wx + ny * wy;
        }
      }
      acc[c] = warp::reduce_sum(sum);
    }
    if (lane == 0) {
#pragma unroll
      for (int c = 0; c < kHcCount; ++c) {
        red_smem[c][warp_id] = acc[c];
      }
    }
    __syncthreads();
    if (threadIdx.x < kHcCount) {
      float total = 0.0f;
#pragma unroll
      for (uint32_t w = 0; w < kNumWarps; ++w) {
        total += red_smem[threadIdx.x][w];
      }
      a_shared[threadIdx.x] = 2.0f / (1.0f + math::exp(-total / kHcCount));
    }
  }
  __syncthreads();

  // Combine (stock hc_combine phase 2 arithmetic).
  Storage out_vec[kVecsPerThread];
#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    const uint32_t vec_idx = threadIdx.x + j * kNumThreads;
    const float a = a_shared[vec_idx / kVecsPerBranch];
#pragma unroll
    for (uint32_t i = 0; i < kVecLen / 2; ++i) {
      const auto [rx, ry] = cast<fp32x2_t>(r_vec[j][i]);
      const auto [yx, yy] = cast<fp32x2_t>(y_vec[j][i]);
      out_vec[j][i] = cast<Float2>(fp32x2_t{rx + a * yx, ry + a * yy});
    }
    gmem.store(out_ptr, out_vec[j], j);
  }

  if constexpr (kDoNorm) {
    const auto nw_ptr = static_cast<const Float*>(params.norm_weight);
    const auto nout_ptr = pointer::offset<Float>(params.normed_output, static_cast<int64_t>(m) * kRowSize);
    // Optionally prefetch the norm weight while the reductions run (costs
    // registers; reading it at use is faster at 64 registers / 4 CTAs per SM).
    Storage w_vec[kPrefetchW ? kVecsPerThread : 1];
    if constexpr (kPrefetchW) {
#pragma unroll
      for (uint32_t j = 0; j < kVecsPerThread; ++j) {
        w_vec[j] = gmem.load(nw_ptr, j);
      }
    }

    // Chunk sums: chunk k = vectors (2k, 2k+1), owned by lanes (even, odd).
    // The reference sums the chunk's 8 pair terms x*x + y*y sequentially from
    // 0. bf16 squares are exact in fp32, so each pair term is one rounding of
    // an exact sum however it is contracted; every lane forms its 4 terms and
    // the even lanes' running sum, which the odd lane continues (branch-free).
#pragma unroll
    for (uint32_t j = 0; j < kVecsPerThread; ++j) {
      const uint32_t vec_idx = threadIdx.x + j * kNumThreads;
      float term[kVecLen / 2];
#pragma unroll
      for (uint32_t i = 0; i < kVecLen / 2; ++i) {
        const auto [x, y] = cast<fp32x2_t>(out_vec[j][i]);
        term[i] = x * x + y * y;
      }
      float head = term[0];
#pragma unroll
      for (uint32_t i = 1; i < kVecLen / 2; ++i) {
        head += term[i];
      }
      // Odd lanes continue from the even neighbour's partial.
      float s = __shfl_up_sync(0xffffffffu, head, 1);
#pragma unroll
      for (uint32_t i = 0; i < kVecLen / 2; ++i) {
        s += term[i];
      }
      if ((lane & 1u) == 1u) {
        chunk_sums[vec_idx / 2] = s;
      }
    }
    __syncthreads();

    // Warp g replays group g: per norm-warp butterflies, then the zero-padded
    // butterfly over the per-warp sums.
    if (warp_id < kHcCount) {
      float warp_sum[kNormWarpsPerGroup];
#pragma unroll
      for (uint32_t k = 0; k < kNormWarpsPerGroup; ++k) {
        const uint32_t chunk = (warp_id * kNormWarpsPerGroup + k) * kWarpThreads + lane;
        warp_sum[k] = warp::reduce_sum(chunk_sums[chunk]);
      }
      float local = 0.0f;
#pragma unroll
      for (uint32_t k = 0; k < kNormWarpsPerGroup; ++k) {
        if (lane == k) local = warp_sum[k];
      }
      const float total = warp::reduce_sum(local);
      if (lane == 0) {
        norm_factor[warp_id] = math::rsqrt(total / kHiddenSize + params.eps);
      }
    }
    __syncthreads();

    Storage n_vec[kVecsPerThread];
#pragma unroll
    for (uint32_t j = 0; j < kVecsPerThread; ++j) {
      const uint32_t vec_idx = threadIdx.x + j * kNumThreads;
      const float nf = norm_factor[vec_idx / kVecsPerBranch];
      Storage nw_vec;
      if constexpr (kPrefetchW) {
        nw_vec = w_vec[j];
      } else {
        nw_vec = gmem.load(nw_ptr, j);
      }
#pragma unroll
      for (uint32_t i = 0; i < kVecLen / 2; ++i) {
        const auto [ix, iy] = cast<fp32x2_t>(out_vec[j][i]);
        const auto [wx, wy] = cast<fp32x2_t>(nw_vec[i]);
        n_vec[j][i] = cast<Float2>(fp32x2_t{ix * nf * (1.0f + wx), iy * nf * (1.0f + wy)});
      }
      gmem.store(nout_ptr, n_vec[j], j);
    }

    if constexpr (kNextGate) {
      const auto w_ptr = static_cast<const Float*>(params.next_inject_weight);
      float acc[kHcCount];
#pragma unroll
      for (int c = 0; c < kHcCount; ++c) {
        const auto wc_ptr = w_ptr + static_cast<int64_t>(c) * kRowSize;
        float sum = 0.0f;
#pragma unroll
        for (uint32_t j = 0; j < kVecsPerThread; ++j) {
          const Storage gw_vec = gmem.load(wc_ptr, j);
#pragma unroll
          for (uint32_t i = 0; i < kVecLen / 2; ++i) {
            const auto [nx, ny] = cast<fp32x2_t>(n_vec[j][i]);
            const auto [wx, wy] = cast<fp32x2_t>(gw_vec[i]);
            sum += nx * wx + ny * wy;
          }
        }
        acc[c] = warp::reduce_sum(sum);
      }
      // red_smem is free again: its last readers finished before the
      // __syncthreads above.
      if (lane == 0) {
#pragma unroll
        for (int c = 0; c < kHcCount; ++c) {
          red_smem[c][warp_id] = acc[c];
        }
      }
      __syncthreads();
      if (threadIdx.x < kHcCount) {
        float total = 0.0f;
#pragma unroll
        for (uint32_t w = 0; w < kNumWarps; ++w) {
          total += red_smem[threadIdx.x][w];
        }
        params.next_partials[static_cast<int64_t>(m) * kHcCount + threadIdx.x] = total;
      }
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <
    int64_t kHcCount,
    int64_t kHiddenSize,
    bool kUsePDL,
    typename DType,
    bool kHasPartials,
    bool kDoNorm,
    bool kNextGate,
    int kMinBlocks,
    bool kPrefetchW>
struct HcCombineExtendKernel {
  static_assert(sizeof(DType) == 2, "HcCombineExtend only supports 2-byte dtypes");
  static_assert((kHcCount * kHiddenSize) % (256 * 8) == 0, "kHcCount * kHiddenSize must be a multiple of 2048");
  static constexpr auto kernel =
      hc_combine_extend_kernel<kHcCount, kHiddenSize, kUsePDL, DType, kHasPartials, kDoNorm, kNextGate, kMinBlocks, kPrefetchW>;
  static constexpr uint32_t kBlockSize = 256;

  /**
   * Unused operands (per the template flags) are not read and may be any
   * tensor; they are not verified.
   */
  static void
  run(const tvm::ffi::TensorView block_output,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView normed_residual,
      const tvm::ffi::TensorView inject_weight,
      const tvm::ffi::TensorView partials,
      const tvm::ffi::TensorView norm_weight,
      const tvm::ffi::TensorView next_inject_weight,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView normed_output,
      const tvm::ffi::TensorView next_partials,
      float eps) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, kHiddenSize}).with_dtype<DType>().with_device(device).verify(block_output);
    TensorMatcher({M, kHcCount * kHiddenSize}).with_dtype<DType>().with_device(device).verify(residual).verify(output);
    if constexpr (kHasPartials) {
      TensorMatcher({M, kHcCount}).with_dtype<fp32_t>().with_device(device).verify(partials);
    } else {
      TensorMatcher({M, kHcCount * kHiddenSize}).with_dtype<DType>().with_device(device).verify(normed_residual);
      TensorMatcher({kHcCount, kHcCount * kHiddenSize})
          .with_dtype<DType>()
          .with_device(device)
          .verify(inject_weight);
    }
    if constexpr (kDoNorm) {
      TensorMatcher({kHcCount * kHiddenSize}).with_dtype<DType>().with_device(device).verify(norm_weight);
      TensorMatcher({M, kHcCount * kHiddenSize}).with_dtype<DType>().with_device(device).verify(normed_output);
    }
    if constexpr (kNextGate) {
      TensorMatcher({kHcCount, kHcCount * kHiddenSize})
          .with_dtype<DType>()
          .with_device(device)
          .verify(next_inject_weight);
      TensorMatcher({M, kHcCount}).with_dtype<fp32_t>().with_device(device).verify(next_partials);
    }

    const auto params = HcCombineExtendParams{
        .block_output = block_output.data_ptr(),
        .residual = residual.data_ptr(),
        .normed_residual = kHasPartials ? nullptr : normed_residual.data_ptr(),
        .inject_weight = kHasPartials ? nullptr : inject_weight.data_ptr(),
        .partials = kHasPartials ? static_cast<const float*>(partials.data_ptr()) : nullptr,
        .norm_weight = kDoNorm ? norm_weight.data_ptr() : nullptr,
        .next_inject_weight = kNextGate ? next_inject_weight.data_ptr() : nullptr,
        .output = output.data_ptr(),
        .normed_output = kDoNorm ? normed_output.data_ptr() : nullptr,
        .next_partials = kNextGate ? static_cast<float*>(next_partials.data_ptr()) : nullptr,
        .eps = eps,
    };

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    if (num_tokens == 0) return;
    LaunchKernel(num_tokens, kBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
'''
