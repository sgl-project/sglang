/**
 * Fused-add RMSNorm with HuggingFace semantics:
 *   s   = x + residual (+ post_residual)            [fp32, written back to residual as Float]
 *   out = weight * cast_dtype( rsqrt(mean(s^2) + eps) * s )
 *
 * In-place on `input` (normalized output) and `residual` (the fp32 sum cast
 * to dtype). Mirrors rmsnorm_hf.cuh structure: warp kernel for hidden < 512,
 * CTA kernel for hidden >= 512.
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

namespace {

struct FusedAddRMSNormHFParams {
  void* input;                             // [N, D]
  void* residual;                          // [N, D]
  const void* __restrict__ post_residual;  // [N, D] or nullptr
  const void* __restrict__ weight;         // [D]
  int64_t input_stride;
  int64_t residual_stride;
  int64_t post_residual_stride;
  uint32_t num_tokens;
  float eps;
};

// Warp kernel: one warp per row, for small hidden sizes.
template <int64_t kDim, bool kUsePDL, bool kHasPostResidual, typename Float>
__global__
__launch_bounds__(32) void fused_add_rmsnorm_hf_warp_kernel(const FusedAddRMSNormHFParams __grid_constant__ params) {
  using namespace device;
  constexpr int kElemsPerThread = kDim / kWarpThreads;

  const auto& [input, residual, post_residual, weight_ptr, input_stride, residual_stride, post_residual_stride, num_tokens, eps] =
      params;
  const auto wr = static_cast<const Float*>(weight_ptr);

  PDLWaitPrimary<kUsePDL>();

  for (uint32_t row = blockIdx.x; row < num_tokens; row += gridDim.x) {
    const auto xr = static_cast<Float*>(pointer::offset<Float>(input, row * input_stride));
    const auto rr = static_cast<Float*>(pointer::offset<Float>(residual, row * residual_stride));
    const Float* pr = nullptr;
    if constexpr (kHasPostResidual) {
      pr = static_cast<const Float*>(pointer::offset<Float>(post_residual, row * post_residual_stride));
    }

    // Pass 1: fp32 sum, cache for pass 2, write back to residual, accumulate squares.
    float xi_cache[kElemsPerThread];
    float lsq = 0.f;
#pragma unroll
    for (int k = 0; k < kElemsPerThread; ++k) {
      const int i = threadIdx.x + k * kWarpThreads;
      const float x_f = static_cast<float>(xr[i]);
      const float r_f = static_cast<float>(rr[i]);
      xi_cache[k] = x_f + r_f;
      if constexpr (kHasPostResidual) {
        xi_cache[k] += static_cast<float>(pr[i]);
      }
      rr[i] = cast<Float>(xi_cache[k]);
      lsq += xi_cache[k] * xi_cache[k];
    }
    lsq = warp::reduce_sum(lsq);
    const float rstd = math::rsqrt(lsq / kDim + eps);

    // Pass 2: cast (x*rstd) to dtype, then multiply by weight.
#pragma unroll
    for (int k = 0; k < kElemsPerThread; ++k) {
      const int i = threadIdx.x + k * kWarpThreads;
      const Float xn = cast<Float>(xi_cache[k] * rstd);
      xr[i] = cast<Float>(static_cast<float>(xn) * static_cast<float>(wr[i]));
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

// CTA kernel: 512-thread scalar-strided, one block per row.
template <int64_t kDim, bool kUsePDL, bool kHasPostResidual, typename Float>
__global__
__launch_bounds__(512) void fused_add_rmsnorm_hf_scalar_kernel(const FusedAddRMSNormHFParams __grid_constant__ params) {
  using namespace device;
  constexpr int kNumThreads = 512;
  constexpr int kNumWarps = kNumThreads / kWarpThreads;
  constexpr int kElemsPerThread = (kDim + kNumThreads - 1) / kNumThreads;

  const auto& [input, residual, post_residual, weight_ptr, input_stride, residual_stride, post_residual_stride, num_tokens, eps] =
      params;
  const auto xr = static_cast<Float*>(pointer::offset<Float>(input, blockIdx.x * input_stride));
  const auto rr = static_cast<Float*>(pointer::offset<Float>(residual, blockIdx.x * residual_stride));
  const Float* pr = nullptr;
  if constexpr (kHasPostResidual) {
    pr = static_cast<const Float*>(pointer::offset<Float>(post_residual, blockIdx.x * post_residual_stride));
  }
  const auto wr = static_cast<const Float*>(weight_ptr);

  PDLWaitPrimary<kUsePDL>();

  // Pass 1: fp32 sum, cache for pass 2, write back to residual, accumulate squares.
  float xi_cache[kElemsPerThread];
  float lsq = 0.f;
#pragma unroll
  for (int k = 0; k < kElemsPerThread; ++k) {
    const int i = threadIdx.x + k * kNumThreads;
    const float x_f = static_cast<float>(xr[i]);
    const float r_f = static_cast<float>(rr[i]);
    xi_cache[k] = x_f + r_f;
    if constexpr (kHasPostResidual) {
      xi_cache[k] += static_cast<float>(pr[i]);
    }
    rr[i] = cast<Float>(xi_cache[k]);
    lsq += xi_cache[k] * xi_cache[k];
  }

  lsq = warp::reduce_sum(lsq);
  __shared__ float smem[32];
  const int warp_id = threadIdx.x / kWarpThreads;
  const int lane_id = threadIdx.x & (kWarpThreads - 1);
  if (lane_id == 0) smem[warp_id] = lsq;
  __syncthreads();

  __shared__ float rstd_s;
  if (threadIdx.x < kWarpThreads) {
    float v = (threadIdx.x < kNumWarps) ? smem[threadIdx.x] : 0.f;
    v = warp::reduce_sum(v);
    if (threadIdx.x == 0) rstd_s = math::rsqrt(v / kDim + eps);
  }
  __syncthreads();
  const float rstd = rstd_s;

  // Pass 2: cast (x*rstd) to dtype, then multiply by weight.
#pragma unroll
  for (int k = 0; k < kElemsPerThread; ++k) {
    const int i = threadIdx.x + k * kNumThreads;
    const Float xn = cast<Float>(xi_cache[k] * rstd);
    xr[i] = cast<Float>(static_cast<float>(xn) * static_cast<float>(wr[i]));
  }

  PDLTriggerSecondary<kUsePDL>();
}

// Warp launcher: occupancy-sized grid.
template <int64_t kDim, bool kUsePDL, bool kHasPostResidual, typename DType>
struct HFFusedAddRMSNormWarpKernel {
  static_assert(sizeof(DType) == 2, "fused_add_rmsnorm_hf: DType must be fp16_t or bf16_t");
  static_assert(
      kDim >= 32 && kDim < 512 && kDim % 32 == 0,
      "fused_add_rmsnorm_hf_warp: kDim must be a multiple of 32, in [32, 512)");
  static constexpr auto kernel = fused_add_rmsnorm_hf_warp_kernel<kDim, kUsePDL, kHasPostResidual, DType>;
  static constexpr uint32_t kBlockSize = device::kWarpThreads;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::Optional<tvm::ffi::TensorView> post_residual,
      const tvm::ffi::TensorView weight,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto SI = SymbolicSize{"input_stride"};
    auto SR = SymbolicSize{"residual_stride"};
    auto SP = SymbolicSize{"post_residual_stride"};
    auto device_ = SymbolicDevice{};
    D.set_value(kDim);
    device_.set_options<kDLCUDA>();

    TensorMatcher({N, D}).with_strides({SI, 1}).with_dtype<DType>().with_device(device_).verify(input);
    TensorMatcher({N, D}).with_strides({SR, 1}).with_dtype<DType>().with_device(device_).verify(residual);
    TensorMatcher({D}).with_dtype<DType>().with_device(device_).verify(weight);
    if constexpr (kHasPostResidual) {
      RuntimeCheck(
          post_residual.has_value(), "fused_add_rmsnorm_hf: post_residual required when kHasPostResidual=true");
      TensorMatcher({N, D}).with_strides({SP, 1}).with_dtype<DType>().with_device(device_).verify(
          post_residual.value());
    } else {
      RuntimeCheck(
          !post_residual.has_value(), "fused_add_rmsnorm_hf: post_residual must be null when kHasPostResidual=false");
    }

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    RuntimeCheck(num_tokens > 0, "fused_add_rmsnorm_hf: num_tokens must be > 0");

    const auto params = FusedAddRMSNormHFParams{
        .input = input.data_ptr(),
        .residual = residual.data_ptr(),
        .post_residual = kHasPostResidual ? post_residual.value().data_ptr() : nullptr,
        .weight = weight.data_ptr(),
        .input_stride = SI.unwrap(),
        .residual_stride = SR.unwrap(),
        .post_residual_stride = kHasPostResidual ? SP.unwrap() : 0,
        .num_tokens = num_tokens,
        .eps = eps,
    };

    static const uint32_t max_occupancy = runtime::get_blocks_per_sm(kernel, kBlockSize);
    static const uint32_t kNumSM = runtime::get_sm_count(device_.unwrap().device_id);
    const auto num_blocks = std::min<uint32_t>(num_tokens, max_occupancy * kNumSM);
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

// CTA launcher: one block per row.
template <int64_t kDim, bool kUsePDL, bool kHasPostResidual, typename DType>
struct HFFusedAddRMSNormKernel {
  static_assert(sizeof(DType) == 2, "fused_add_rmsnorm_hf: DType must be fp16_t or bf16_t");
  static_assert(kDim >= 512 && kDim % 512 == 0, "fused_add_rmsnorm_hf: kDim must be a multiple of 512");
  static constexpr auto kernel = fused_add_rmsnorm_hf_scalar_kernel<kDim, kUsePDL, kHasPostResidual, DType>;
  static constexpr uint32_t kBlockSize = 512;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::Optional<tvm::ffi::TensorView> post_residual,
      const tvm::ffi::TensorView weight,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto SI = SymbolicSize{"input_stride"};
    auto SR = SymbolicSize{"residual_stride"};
    auto SP = SymbolicSize{"post_residual_stride"};
    auto device_ = SymbolicDevice{};
    D.set_value(kDim);
    device_.set_options<kDLCUDA>();

    TensorMatcher({N, D})  // input
        .with_strides({SI, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({N, D})  // residual
        .with_strides({SR, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(residual);
    TensorMatcher({D})  // weight
        .with_dtype<DType>()
        .with_device(device_)
        .verify(weight);
    if constexpr (kHasPostResidual) {
      RuntimeCheck(
          post_residual.has_value(), "fused_add_rmsnorm_hf: post_residual required when kHasPostResidual=true");
      TensorMatcher({N, D})  // post_residual
          .with_strides({SP, 1})
          .with_dtype<DType>()
          .with_device(device_)
          .verify(post_residual.value());
    } else {
      RuntimeCheck(
          !post_residual.has_value(), "fused_add_rmsnorm_hf: post_residual must be null when kHasPostResidual=false");
    }

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    RuntimeCheck(num_tokens > 0, "fused_add_rmsnorm_hf: num_tokens must be > 0");

    const auto params = FusedAddRMSNormHFParams{
        .input = input.data_ptr(),
        .residual = residual.data_ptr(),
        .post_residual = kHasPostResidual ? post_residual.value().data_ptr() : nullptr,
        .weight = weight.data_ptr(),
        .input_stride = SI.unwrap(),
        .residual_stride = SR.unwrap(),
        .post_residual_stride = kHasPostResidual ? SP.unwrap() : 0,
        .num_tokens = num_tokens,
        .eps = eps,
    };

    LaunchKernel(num_tokens, kBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
