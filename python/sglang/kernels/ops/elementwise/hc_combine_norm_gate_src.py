"""CUDA source of the fused HC boundary kernel (combine + next norm + next gate).

Kept as a Python string so the module ships with the Python sources; the
loader in hc_combine_decode writes it to a content-addressed file under the
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

namespace sglang {

/**
 * \brief One decode HC boundary in one kernel: combine, next norm, next gate.
 *
 *   gate[r, b]       = 2 * sigmoid(sum_s partials_in[r, s, b] / 4)
 *   combined[r, b]   = bf16(residual[r, b] + gate[r, b] * block[r])
 *   normalized[r, b] = bf16(combined[r, b] * rsqrt(mean(combined[r, b]^2) + eps) * (1 + w[b]))
 *   partials_out[r, 2b + p, c] = dot(normalized[r, b, slice p], inject[c, b, slice p])
 *
 * One CTA per (row, branch, part), 160 threads, 16 BF16 per thread. Every CTA
 * of a branch recomputes the branch's sum of squares (the RMS needs all of it)
 * but stores only its part's half, and reduces only that half into its gate
 * partials slot. The combine and norm arithmetic, including the warp order of
 * the sum of squares, is that of hc_combine_apply followed by
 * grouped_gemma_rmsnorm, so both outputs are bit-identical to the stock pair
 * for the same partials_in. partials_out is a different split of the same gate
 * dot than hc_combine_gate's; consumers only sum over the 8 splits. No atomics.
 *
 * PDL: the norm and inject weights are constants and are read before the
 * wait; everything the producer writes is read after it. The CTA triggers its
 * dependents right after the wait, so the next kernel (the split-K down mix)
 * launches early and streams its weights while this one runs. That is safe:
 * per the PTX ISA the dependent's griddepcontrol.wait blocks until this grid
 * has completed and its memory is visible, and the trigger only schedules
 * (the stock CuTe split-K mix also triggers before its epilogue stores).
 */
struct HcCombineNormGateParams {
  const bf16_t* block;        // [rows, 2560]
  const bf16_t* residual;     // [rows, 4 * 2560]
  const float* partials_in;   // [rows, 8, 4]
  const bf16_t* norm_weight;  // [4 * 2560]
  const bf16_t* inject;       // [4, 4 * 2560]; kGate only
  bf16_t* combined;           // [rows, 4 * 2560]
  bf16_t* normalized;         // [rows, 4 * 2560]
  float* partials_out;        // [rows, 8, 4]; kGate only
  float eps;
};

namespace hc_combine_norm_gate_detail {

constexpr uint32_t kHc = 4;
constexpr uint32_t kHidden = 2560;
constexpr uint32_t kRow = kHc * kHidden;
constexpr uint32_t kSplit = 8;
constexpr uint32_t kParts = 2;
constexpr uint32_t kThreads = 160;
constexpr uint32_t kWarps = kThreads / 32;
constexpr uint32_t kPartThreads = kThreads / kParts;
constexpr int64_t kMaxRows = 32;
static_assert(kHc * kParts == kSplit, "one gate partial slot per (branch, part)");
static_assert(kThreads * 16 == kHidden, "one 32-byte vector per thread covers a branch");

}  // namespace hc_combine_norm_gate_detail

template <bool kGate>
__global__ __launch_bounds__(hc_combine_norm_gate_detail::kThreads) void hc_combine_norm_gate_kernel(
    const HcCombineNormGateParams __grid_constant__ params) {
  using namespace device;
  using namespace hc_combine_norm_gate_detail;
  using BFloat16x2 = packed_t<bf16_t>;
  using Storage = AlignedVector<BFloat16x2, 8>;  // 16 BF16, 32 bytes
  const uint32_t row = blockIdx.x / (kHc * kParts);
  const uint32_t branch = (blockIdx.x / kParts) % kHc;
  const uint32_t part = blockIdx.x % kParts;
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid / kWarpThreads;
  const uint32_t lane = tid % kWarpThreads;
  const bool in_part = tid / kPartThreads == part;
  const auto gmem = tile::Memory<Storage>::cta(kThreads);
  const int64_t branch_offset = (static_cast<int64_t>(row) * kHc + branch) * kHidden;
  __shared__ float warp_sums[kWarpThreads];
  __shared__ float gate_sums[kHc][kWarps];

  // Constants: independent of the producer, so they load before the wait.
  Storage weight_values;
  Storage inject_values[kHc];
  if (in_part) {
    weight_values = gmem.load(params.norm_weight + branch * kHidden, 0);
    if constexpr (kGate) {
#pragma unroll
      for (uint32_t c = 0; c < kHc; ++c) {
        inject_values[c] = gmem.load(params.inject + c * kRow + branch * kHidden, 0);
      }
    }
  }

  PDLWaitPrimary<true>();
  // Early trigger: lets the dependent's prologue and weight stream overlap us.
  PDLTriggerSecondary<true>();

  float total = 0.0f;
#pragma unroll
  for (uint32_t split = 0; split < kSplit; ++split) {
    total += params.partials_in[(row * kSplit + split) * kHc + branch];
  }
  const float gate = 2.0f / (1.0f + math::exp(-total / kHc));
  const auto residual_values = gmem.load(params.residual + branch_offset, 0);
  const auto block_values = gmem.load(params.block + static_cast<int64_t>(row) * kHidden, 0);
  Storage combined;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const auto [rx, ry] = cast<fp32x2_t>(residual_values[i]);
    const auto [yx, yy] = cast<fp32x2_t>(block_values[i]);
    // The stock apply's FMA and BF16 store, before the norm reads it back.
    combined[i] = cast<BFloat16x2>(fp32x2_t{rx + gate * yx, ry + gate * yy});
  }

  // grouped_gemma_rmsnorm's reduction: 16 values per thread, then warps.
  float sum = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const auto [x, y] = cast<fp32x2_t>(combined[i]);
    sum += x * x + y * y;
  }
  sum = warp::reduce_sum(sum);
  warp_sums[warp_id] = sum;
  __syncthreads();
  if (warp_id == 0) {
    const float local_sum = tid < kWarps ? warp_sums[tid] : 0.0f;
    sum = warp::reduce_sum(local_sum);
    warp_sums[tid] = math::rsqrt(sum / kHidden + params.eps);
  }
  __syncthreads();
  const float inv_rms = warp_sums[warp_id];

  float dots[kHc];
#pragma unroll
  for (uint32_t c = 0; c < kHc; ++c) dots[c] = 0.0f;
  if (in_part) {
    Storage normalized;
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      const auto [x, y] = cast<fp32x2_t>(combined[i]);
      const auto [wx, wy] = cast<fp32x2_t>(weight_values[i]);
      normalized[i] = cast<BFloat16x2>(fp32x2_t{x * inv_rms * (1.0f + wx), y * inv_rms * (1.0f + wy)});
    }
    gmem.store(params.combined + branch_offset, combined, 0);
    gmem.store(params.normalized + branch_offset, normalized, 0);
    if constexpr (kGate) {
      // The gate reads the BF16 normalized values, as hc_combine_gate does.
#pragma unroll
      for (uint32_t i = 0; i < 8; ++i) {
        const auto [nx, ny] = cast<fp32x2_t>(normalized[i]);
#pragma unroll
        for (uint32_t c = 0; c < kHc; ++c) {
          const auto [wx, wy] = cast<fp32x2_t>(inject_values[c][i]);
          dots[c] += nx * wx + ny * wy;
        }
      }
    }
  }
  if constexpr (kGate) {
#pragma unroll
    for (uint32_t c = 0; c < kHc; ++c) {
      dots[c] = warp::reduce_sum(dots[c]);
    }
    if (lane == 0) {
#pragma unroll
      for (uint32_t c = 0; c < kHc; ++c) gate_sums[c][warp_id] = dots[c];
    }
    __syncthreads();
    if (tid == 0) {
      // Threads outside the part contributed exact zeros.
      static_assert(kHc == 4, "one float4 of partials per CTA");
      float sums[kHc];
#pragma unroll
      for (uint32_t c = 0; c < kHc; ++c) {
        sums[c] = 0.0f;
#pragma unroll
        for (uint32_t w = 0; w < kWarps; ++w) sums[c] += gate_sums[c][w];
      }
      const fp32x4_t partial{sums[0], sums[1], sums[2], sums[3]};
      // One 16-byte store; the slot is 16-byte aligned ([rows, 8, 4] fp32).
      reinterpret_cast<fp32x4_t*>(params.partials_out)[row * kSplit + branch * kParts + part] = partial;
    }
  }
}

struct HcCombineNormGateDecode {
  static void launch(const HcCombineNormGateParams& p, int64_t rows, DLDevice device, bool with_gate) {
    using namespace host;
    using namespace hc_combine_norm_gate_detail;
    const auto grid = static_cast<uint32_t>(rows * kHc * kParts);
    if (with_gate) {
      LaunchKernel(grid, kThreads, device).enable_pdl(true)(hc_combine_norm_gate_kernel<true>, p);
    } else {
      LaunchKernel(grid, kThreads, device).enable_pdl(true)(hc_combine_norm_gate_kernel<false>, p);
    }
  }

  // combine + next norm + next gate partials.
  static void
  gate(tvm::ffi::TensorView block,
       tvm::ffi::TensorView residual,
       tvm::ffi::TensorView partials,
       tvm::ffi::TensorView norm_weight,
       tvm::ffi::TensorView inject,
       tvm::ffi::TensorView combined,
       tvm::ffi::TensorView normalized,
       tvm::ffi::TensorView partials_out,
       float eps) {
    using namespace host;
    using namespace hc_combine_norm_gate_detail;
    SymbolicSize m{"rows"};
    SymbolicDevice dev;
    dev.set_options<kDLCUDA>();
    TensorMatcher({m, kHidden}).with_dtype<bf16_t>().with_device(dev).verify(block);
    TensorMatcher({m, kRow}).with_dtype<bf16_t>().with_device(dev).verify(residual).verify(combined).verify(normalized);
    TensorMatcher({m, kSplit, kHc}).with_dtype<fp32_t>().with_device(dev).verify(partials).verify(partials_out);
    TensorMatcher({kRow}).with_dtype<bf16_t>().with_device(dev).verify(norm_weight);
    TensorMatcher({kHc, kRow}).with_dtype<bf16_t>().with_device(dev).verify(inject);
    const int64_t rows = m.unwrap();
    CHECK_HOST(rows > 0 && rows <= kMaxRows) << "hc_combine_norm_gate: rows must be in [1, 32], got " << rows;
    const HcCombineNormGateParams p{
        static_cast<const bf16_t*>(block.data_ptr()),
        static_cast<const bf16_t*>(residual.data_ptr()),
        static_cast<const float*>(partials.data_ptr()),
        static_cast<const bf16_t*>(norm_weight.data_ptr()),
        static_cast<const bf16_t*>(inject.data_ptr()),
        static_cast<bf16_t*>(combined.data_ptr()),
        static_cast<bf16_t*>(normalized.data_ptr()),
        static_cast<float*>(partials_out.data_ptr()),
        eps};
    launch(p, rows, dev.unwrap(), true);
  }

  // combine + next norm, for a next read that has no gate (the final mixer).
  static void
  norm(tvm::ffi::TensorView block,
       tvm::ffi::TensorView residual,
       tvm::ffi::TensorView partials,
       tvm::ffi::TensorView norm_weight,
       tvm::ffi::TensorView combined,
       tvm::ffi::TensorView normalized,
       float eps) {
    using namespace host;
    using namespace hc_combine_norm_gate_detail;
    SymbolicSize m{"rows"};
    SymbolicDevice dev;
    dev.set_options<kDLCUDA>();
    TensorMatcher({m, kHidden}).with_dtype<bf16_t>().with_device(dev).verify(block);
    TensorMatcher({m, kRow}).with_dtype<bf16_t>().with_device(dev).verify(residual).verify(combined).verify(normalized);
    TensorMatcher({m, kSplit, kHc}).with_dtype<fp32_t>().with_device(dev).verify(partials);
    TensorMatcher({kRow}).with_dtype<bf16_t>().with_device(dev).verify(norm_weight);
    const int64_t rows = m.unwrap();
    CHECK_HOST(rows > 0 && rows <= kMaxRows) << "hc_combine_norm: rows must be in [1, 32], got " << rows;
    const HcCombineNormGateParams p{
        static_cast<const bf16_t*>(block.data_ptr()),
        static_cast<const bf16_t*>(residual.data_ptr()),
        static_cast<const float*>(partials.data_ptr()),
        static_cast<const bf16_t*>(norm_weight.data_ptr()),
        nullptr,
        static_cast<bf16_t*>(combined.data_ptr()),
        static_cast<bf16_t*>(normalized.data_ptr()),
        nullptr,
        eps};
    launch(p, rows, dev.unwrap(), false);
  }
};

}  // namespace sglang
'''
