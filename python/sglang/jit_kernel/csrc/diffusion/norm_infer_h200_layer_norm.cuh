// LayerNorm forward for the diffusion inference norm kernel (fp32, exact N).
//
// y = (x - mean) * rsqrt(var + eps) * w + b   (row-wise), all math in fp32.
// mean = sum(x)/N ; var = sum((x-mean)^2)/N   (two-moment, matching the SGLang
// baseline's in-register two-pass; fp32 throughout for the 1e-5 tolerance).
//
// One CTA normalizes one row (grid-stride over rows). N must tile exactly as
// kThreadsPerBlock * 4 (e.g. 5120 = 256 * 20), so each thread owns 20 elements
// loaded as five float4 with NO masked over-read (the Triton baseline pads N up
// to next_pow2 = 8192 and masks 3072 lanes). Built through the SGLang
// jit_kernel / tvm-ffi stack.

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace {

struct LayerNormParams {
  const void* __restrict__ x_ptr;
  void* __restrict__ y_ptr;
  const void* __restrict__ w_ptr;
  const void* __restrict__ b_ptr;
  int64_t x_stride_bytes;
  int64_t y_stride_bytes;
  uint32_t num_rows;
  float eps;
};

constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kWarpsPerBlock = kThreadsPerBlock / device::kWarpThreads;

// Block-wide sum reduction: warp-reduce, stage partials in shared memory, then
// every thread sums the per-warp partials. Bracketed by __syncthreads so the
// shared buffer can be reused across the two reductions and successive rows.
template <uint32_t kWarps>
__device__ __forceinline__ float block_reduce_sum(float val, float* smem, uint32_t lane, uint32_t warp) {
  val = device::warp::reduce_sum(val);
  if (lane == 0) smem[warp] = val;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < kWarps; ++i) total += smem[i];
  __syncthreads();
  return total;
}

template <int64_t kN, bool kHasBias, bool kUsePDL, typename DType>
__global__ void layer_norm_block(const LayerNormParams __grid_constant__ params) {
  using namespace device;

  static_assert(std::is_same_v<DType, fp32_t>, "exact-N LayerNorm path is fp32-only");
  constexpr uint32_t kVec = 4;
  static_assert(kN % (kThreadsPerBlock * kVec) == 0, "N must tile as threads*4 exactly");
  constexpr uint32_t kIters = kN / (kThreadsPerBlock * kVec);
  constexpr uint32_t kElems = kIters * kVec;
  using Vec = AlignedVector<DType, kVec>;

  __shared__ float s_partial[kWarpsPerBlock];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % kWarpThreads;
  const uint32_t warp = tid / kWarpThreads;
  const float inv_n = 1.0f / static_cast<float>(kN);

  PDLWaitPrimary<kUsePDL>();

  for (uint32_t row = blockIdx.x; row < params.num_rows; row += gridDim.x) {
    const void* x_row = pointer::offset(params.x_ptr, row * params.x_stride_bytes);
    void* y_row = pointer::offset(params.y_ptr, row * params.y_stride_bytes);

    float xs[kElems];
    float local_sum = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < kIters; ++k) {
      const Vec v = load_as<Vec>(x_row, k * kThreadsPerBlock + tid);
#pragma unroll
      for (uint32_t j = 0; j < kVec; ++j) {
        xs[k * kVec + j] = v[j];
        local_sum += v[j];
      }
    }

    const float mean = block_reduce_sum<kWarpsPerBlock>(local_sum, s_partial, lane, warp) * inv_n;

    float local_var = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kElems; ++i) {
      const float d = xs[i] - mean;
      local_var += d * d;
    }
    const float rstd = math::rsqrt(block_reduce_sum<kWarpsPerBlock>(local_var, s_partial, lane, warp) * inv_n + params.eps);

#pragma unroll
    for (uint32_t k = 0; k < kIters; ++k) {
      const uint32_t vidx = k * kThreadsPerBlock + tid;
      const Vec wv = load_as<Vec>(params.w_ptr, vidx);
      Vec bv;
      if constexpr (kHasBias) bv = load_as<Vec>(params.b_ptr, vidx);
      Vec yv;
#pragma unroll
      for (uint32_t j = 0; j < kVec; ++j) {
        const float xn = (xs[k * kVec + j] - mean) * rstd;
        yv[j] = kHasBias ? (xn * wv[j] + bv[j]) : (xn * wv[j]);
      }
      store_as<Vec>(y_row, yv, vidx);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kN, bool kHasBias, bool kUsePDL, typename DType>
struct LayerNormKernel {
  static constexpr auto kernel = layer_norm_block<kN, kHasBias, kUsePDL, DType>;

  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView b,
      const tvm::ffi::TensorView y,
      float eps) {
    using namespace host;

    auto M = SymbolicSize{"num_rows"};
    auto N = SymbolicSize{"dim"};
    auto Sx = SymbolicSize{"x_row_stride"};
    auto Sy = SymbolicSize{"y_row_stride"};
    auto device = SymbolicDevice{};
    N.set_value(kN);
    device.set_options<kDLCUDA>();

    TensorMatcher({M, N}).with_strides({Sx, 1}).with_dtype<DType>().with_device(device).verify(x);
    TensorMatcher({M, N}).with_strides({Sy, 1}).with_dtype<DType>().with_device(device).verify(y);
    TensorMatcher({N}).with_dtype<DType>().with_device(device).verify(w);
    if constexpr (kHasBias) {
      TensorMatcher({N}).with_dtype<DType>().with_device(device).verify(b);
    }

    const auto num_rows = static_cast<uint32_t>(M.unwrap());
    const auto x_stride_bytes = static_cast<int64_t>(Sx.unwrap() * sizeof(DType));
    const auto y_stride_bytes = static_cast<int64_t>(Sy.unwrap() * sizeof(DType));

    const auto params = LayerNormParams{
        .x_ptr = x.data_ptr(),
        .y_ptr = y.data_ptr(),
        .w_ptr = w.data_ptr(),
        .b_ptr = kHasBias ? b.data_ptr() : nullptr,
        .x_stride_bytes = x_stride_bytes,
        .y_stride_bytes = y_stride_bytes,
        .num_rows = num_rows,
        .eps = eps,
    };

    // One CTA per row maximizes independent memory streams (NCU: the capped
    // grid-stride hit only 74% DRAM vs the baseline's 80%); the grid-stride loop
    // then runs a single iteration per block.
    const uint32_t num_blocks = std::max(num_rows, 1u);
    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
