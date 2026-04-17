/**
 * RMSNorm with HuggingFace semantics:
 *   out[i] = weight[i] * cast_dtype( rsqrt(mean_j(x[j]^2) + eps) * x[i] )
 *
 * This differs from standard rmsnorm (which does the weight multiply in fp32
 * and casts only at the end): here we cast the normalized x to the activation
 * dtype BEFORE multiplying by the weight. This matches HF `LlamaRMSNorm` and
 * is required for transformers-backend accuracy parity under weight-only
 * quantization.
 *
 * 512-thread scalar-strided kernel with per-thread register caching of the
 * input values between the two passes. Each thread processes
 * `kElemsPerThread = ceil(kDim / 512)` elements at strided positions
 * [tid, tid+512, tid+1024, ...], accumulating the sum-of-squares in pass 1
 * and reusing the cached fp32 values in pass 2 — halving the global reads
 * of `x`.
 */

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/math.cuh>   // For device::math::rsqrt
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE, type aliases, PDL, cast
#include <sgl_kernel/warp.cuh>   // For warp::reduce_sum

#include <tvm/ffi/container/tensor.h>

namespace {

struct RMSNormHFParams {
  const void* input;
  const void* __restrict__ weight;
  void* output;
  int64_t input_stride;
  int64_t output_stride;
  uint32_t num_tokens;
  float eps;
};

// ---------------------------------------------------------------------------
// Kernel: 512-thread scalar-strided RMSNorm with HF semantics + register cache.
//
// Pass 1: each thread loads its strided elements, caches them in registers,
//         and accumulates the fp32 sum-of-squares. Warp + block reduction
//         yields `rstd = rsqrt(mean(x^2) + eps)`.
// Pass 2: each thread reuses the cached fp32 values to compute its outputs
//         WITHOUT re-reading `x` from global memory.  Per-element:
//             xn = cast_to_dtype(x_fp32 * rstd)   <- HF's cast-before-mul
//             y  = cast_to_dtype(float(xn) * float(w))
// ---------------------------------------------------------------------------
template <int64_t kDim, bool kUsePDL, typename Float>
__global__ __launch_bounds__(512) void rmsnorm_hf_scalar_kernel(const RMSNormHFParams __grid_constant__ params) {
  using namespace device;
  constexpr int kNumThreads = 512;
  constexpr int kNumWarps = kNumThreads / kWarpThreads;
  // For kDim=4096: kElemsPerThread = 8 (32 bytes of fp32 cache per thread).
  constexpr int kElemsPerThread = (kDim + kNumThreads - 1) / kNumThreads;

  const auto& [input, weight_ptr, output, input_stride, output_stride, num_tokens, eps] = params;
  const auto xr = static_cast<const Float*>(pointer::offset<Float>(input, blockIdx.x * input_stride));
  const auto yr = static_cast<Float*>(pointer::offset<Float>(output, blockIdx.x * output_stride));
  const auto wr = static_cast<const Float*>(weight_ptr);

  PDLWaitPrimary<kUsePDL>();

  // Pass 1: load, square, accumulate; cache fp32 values in registers.
  float xi_cache[kElemsPerThread];
  float lsq = 0.f;
#pragma unroll
  for (int k = 0; k < kElemsPerThread; ++k) {
    const int i = threadIdx.x + k * kNumThreads;
    xi_cache[k] = static_cast<float>(xr[i]);
    lsq += xi_cache[k] * xi_cache[k];
  }

  // Warp reduce.
  lsq = warp::reduce_sum(lsq);

  // Block reduce via shared memory (32 warps * 1 fp32 each).
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

  // Pass 2: HF semantics — cast (x*rstd) to dtype, then multiply weight in dtype.
#pragma unroll
  for (int k = 0; k < kElemsPerThread; ++k) {
    const int i = threadIdx.x + k * kNumThreads;
    const Float xn = cast<Float>(xi_cache[k] * rstd);
    yr[i] = cast<Float>(static_cast<float>(xn) * static_cast<float>(wr[i]));
  }

  PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// Launcher: validates tensors, launches one block per row.
// ---------------------------------------------------------------------------
template <int64_t kDim, bool kUsePDL, typename DType>
struct RMSNormHFKernel {
  static_assert(sizeof(DType) == 2, "rmsnorm_hf: DType must be fp16_t or bf16_t");
  static_assert(kDim >= 512 && kDim % 512 == 0, "rmsnorm_hf: kDim must be a multiple of 512");
  static constexpr auto kernel = rmsnorm_hf_scalar_kernel<kDim, kUsePDL, DType>;
  static constexpr uint32_t kBlockSize = 512;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView output,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto SI = SymbolicSize{"input_stride"};
    auto SO = SymbolicSize{"output_stride"};
    auto device_ = SymbolicDevice{};
    D.set_value(kDim);
    device_.set_options<kDLCUDA>();

    TensorMatcher({N, D})  // input
        .with_strides({SI, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({D})  // weight
        .with_dtype<DType>()
        .with_device(device_)
        .verify(weight);
    TensorMatcher({N, D})  // output
        .with_strides({SO, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(output);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    RuntimeCheck(num_tokens > 0, "rmsnorm_hf: num_tokens must be > 0");

    const auto params = RMSNormHFParams{
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .output = output.data_ptr(),
        .input_stride = SI.unwrap(),
        .output_stride = SO.unwrap(),
        .num_tokens = num_tokens,
        .eps = eps,
    };

    LaunchKernel(num_tokens, kBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
