// RMSNorm forward for the diffusion one-pass RMSNorm family (D=128, bf16/fp16).
//
// y = x * rsqrt(mean(x^2) + eps) * w   (row-wise, no bias), fp32 accumulation.
//
// Two rows per warp: 16 lanes own one row (8 elements/lane = a 128-bit packed
// load), so a 32-lane warp streams two independent rows and each row reduces
// within its 16-lane group (warp_reduce<16>). 128-bit loads/stores halve the
// LD/ST instruction count vs a 64-bit warp-per-row layout and expose more
// row-level parallelism (prior art: pytorch#150705, vllm#27931). The [D] weight
// is loaded once per thread and reused across all rows; a persistent grid-stride
// launch (~one wave of blocks) covers the row range. Built through the SGLang
// jit_kernel / tvm-ffi stack; mirrors csrc/diffusion/qknorm_rope.cuh.

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

struct RmsNormParams {
  const void* __restrict__ x_ptr;
  void* __restrict__ y_ptr;
  const void* __restrict__ w_ptr;
  int64_t x_stride_bytes;
  int64_t y_stride_bytes;
  uint32_t num_rows;
  float eps;
};

constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kWarpsPerBlock = kThreadsPerBlock / device::kWarpThreads;

template <int64_t kDim, bool kUsePDL, typename DType>
__global__ void rms_norm_warp(const RmsNormParams __grid_constant__ params) {
  using namespace device;

  static_assert(std::is_same_v<DType, fp16_t> || std::is_same_v<DType, bf16_t>);
  static_assert(kDim == 128, "this path is specialized for D=128 (two rows per warp, 128-bit)");

  constexpr uint32_t kLanesPerRow = 16;                       // 16 lanes x 8 elems = 128
  constexpr uint32_t kElemsPerThread = kDim / kLanesPerRow;   // 8
  constexpr uint32_t kVecSize = kElemsPerThread / 2;          // 4 packed (x2) = 16B = 128-bit
  using Packed = packed_t<DType>;
  using Storage = AlignedVector<Packed, kVecSize>;

  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t row_in_warp = lane_id / kLanesPerRow;   // 0 or 1
  const uint32_t lane_in_row = lane_id % kLanesPerRow;   // 0..15
  const uint32_t global_warp = blockIdx.x * kWarpsPerBlock + warp_id;
  const uint32_t num_warps = gridDim.x * kWarpsPerBlock;
  const uint32_t num_pairs = (params.num_rows + 1u) / 2u;

  PDLWaitPrimary<kUsePDL>();

  // Weight is identical for every row; load once and keep in registers.
  const auto weight_vec = load_as<Storage>(params.w_ptr, lane_in_row);
  float w_elems[kElemsPerThread];
#pragma unroll
  for (uint32_t j = 0; j < kVecSize; ++j) {
    const auto [w0, w1] = cast<fp32x2_t>(weight_vec[j]);
    w_elems[2 * j] = w0;
    w_elems[2 * j + 1] = w1;
  }

  for (uint32_t pair = global_warp; pair < num_pairs; pair += num_warps) {
    const uint32_t row = 2u * pair + row_in_warp;
    const bool valid = row < params.num_rows;
    // Invalid lanes (odd row tail) read row 0 so all 32 lanes stay converged
    // through the warp shuffle; their result is simply not stored.
    const uint32_t safe_row = valid ? row : 0u;
    const void* x_row = pointer::offset(params.x_ptr, static_cast<int64_t>(safe_row) * params.x_stride_bytes);

    auto x_vec = load_as<Storage>(x_row, lane_in_row);
    float elems[kElemsPerThread];
    float sum_of_squares = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      const auto [x0, x1] = cast<fp32x2_t>(x_vec[j]);
      elems[2 * j] = x0;
      elems[2 * j + 1] = x1;
      sum_of_squares += x0 * x0 + x1 * x1;
    }

    sum_of_squares = warp::reduce_sum<kLanesPerRow>(sum_of_squares);
    const float rstd = math::rsqrt(sum_of_squares / static_cast<float>(kDim) + params.eps);

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      const float y0 = elems[2 * j] * rstd * w_elems[2 * j];
      const float y1 = elems[2 * j + 1] * rstd * w_elems[2 * j + 1];
      x_vec[j] = cast<Packed, fp32x2_t>({y0, y1});
    }
    if (valid) {
      void* y_row = pointer::offset(params.y_ptr, static_cast<int64_t>(row) * params.y_stride_bytes);
      store_as<Storage>(y_row, x_vec, lane_in_row);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kDim, bool kUsePDL, typename DType>
struct RmsNormKernel {
  static_assert(kDim == 128);
  static constexpr auto kernel = rms_norm_warp<kDim, kUsePDL, DType>;

  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView y,
      float eps) {
    using namespace host;

    auto M = SymbolicSize{"num_rows"};
    auto D = SymbolicSize{"dim"};
    auto Sx = SymbolicSize{"x_row_stride"};
    auto Sy = SymbolicSize{"y_row_stride"};
    auto device = SymbolicDevice{};
    D.set_value(kDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({M, D}).with_strides({Sx, 1}).with_dtype<DType>().with_device(device).verify(x);
    TensorMatcher({M, D}).with_strides({Sy, 1}).with_dtype<DType>().with_device(device).verify(y);
    TensorMatcher({D}).with_dtype<DType>().with_device(device).verify(w);

    const auto num_rows = static_cast<uint32_t>(M.unwrap());
    const auto x_stride_bytes = static_cast<int64_t>(Sx.unwrap() * sizeof(DType));
    const auto y_stride_bytes = static_cast<int64_t>(Sy.unwrap() * sizeof(DType));

    const auto params = RmsNormParams{
        .x_ptr = x.data_ptr(),
        .y_ptr = y.data_ptr(),
        .w_ptr = w.data_ptr(),
        .x_stride_bytes = x_stride_bytes,
        .y_stride_bytes = y_stride_bytes,
        .num_rows = num_rows,
        .eps = eps,
    };

    // Tiny per-row work: a persistent grid-stride launch (~one wave of resident
    // blocks, each warp streaming many row-pairs) amortizes block launch/retire
    // and tail overhead better than a one-warp-per-row grid.
    const uint32_t num_pairs = (num_rows + 1u) / 2u;
    const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);
    const uint32_t blocks_per_sm = runtime::get_blocks_per_sm(kernel, kThreadsPerBlock);
    const uint32_t max_blocks = blocks_per_sm * kNumSM;
    const uint32_t needed_blocks = div_ceil(num_pairs, kWarpsPerBlock);
    const uint32_t num_blocks = std::min(max_blocks, std::max(needed_blocks, 1u));
    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
