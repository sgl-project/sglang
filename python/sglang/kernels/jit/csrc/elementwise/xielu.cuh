#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <limits>

namespace sglang {

/// \brief xIELU coefficients after the softplus, constant at inference time.
struct XIELUParams {
  const void* __restrict__ input;
  void* __restrict__ out;
  uint32_t num_elems;
  float alpha_p;  ///< softplus(alpha_p)
  float alpha_n;  ///< beta + softplus(alpha_n)
  float beta;
  float eps;
};

/**
 * \brief xIELU (https://arxiv.org/abs/2411.13010) on one fp32 value.
 *
 * x > 0: alpha_p * x * x + beta * x
 * else:  alpha_n * (expm1(min(x, eps)) - x) + beta * x
 *
 * Explicitly rounded operations (no FMA contraction) evaluate the fp32
 * expression in the same order as the torch reference, so the result is
 * bit-identical to it after the single rounding to the output dtype.
 * min(x, eps) is a select so that NaN propagates as in torch.clamp_max.
 */
SGL_DEVICE float xielu_f32(float x, const XIELUParams& p) {
  const float beta_x = __fmul_rn(p.beta, x);
  if (x > 0.0f) {
    return __fadd_rn(__fmul_rn(__fmul_rn(p.alpha_p, x), x), beta_x);
  }
  const float clamped = x > p.eps ? p.eps : x;
  return __fadd_rn(__fmul_rn(__fsub_rn(expm1f(clamped), x), p.alpha_n), beta_x);
}

/// \brief One vector of kVecBytes per thread; the thread past the last vector handles the tail.
template <typename T, uint32_t kVecBytes, bool kUsePDL>
__global__ void xielu_kernel(const __grid_constant__ XIELUParams params) {
  using namespace device;
  constexpr uint32_t kVecSize = kVecBytes / sizeof(T);
  using vec_t = AlignedVector<T, kVecSize>;
  const uint32_t num_vecs = params.num_elems / kVecSize;
  const uint32_t vec_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (vec_id > num_vecs) return;
  PDLWaitPrimary<kUsePDL>();
  if (vec_id < num_vecs) {
    const auto in = load_as<vec_t>(params.input, vec_id);
    vec_t out;
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      out[i] = cast<T>(xielu_f32(cast<fp32_t>(in[i]), params));
    }
    store_as<vec_t>(params.out, out, vec_id);
  } else {
    const auto* input = static_cast<const T*>(params.input);
    auto* out = static_cast<T*>(params.out);
    for (uint32_t i = num_vecs * kVecSize; i < params.num_elems; ++i) {
      out[i] = cast<T>(xielu_f32(cast<fp32_t>(input[i]), params));
    }
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <typename T, bool kUsePDL>
struct XIELUKernel {
  static constexpr uint32_t kBlockSize = 256;

  template <uint32_t kVecBytes>
  static void launch(const XIELUParams& params, DLDevice device) {
    using namespace host;
    // One extra thread past the vectors handles the tail.
    const auto num_blocks = div_ceil(params.num_elems / (kVecBytes / sizeof(T)) + 1, kBlockSize);
    LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(xielu_kernel<T, kVecBytes, kUsePDL>, params);
  }

  static XIELUParams make_params(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView out,
      float alpha_p,
      float alpha_n,
      float beta,
      float eps,
      DLDevice* device) {
    using namespace host;
    auto N = SymbolicSize{"num_elems"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N})  //
        .with_dtype<T>()
        .with_device(device_)
        .verify(out)
        .verify(input);
    const auto num_elems = N.unwrap();
    CHECK_HOST(num_elems <= std::numeric_limits<uint32_t>::max()) << "xielu: too many elements: " << num_elems;
    *device = device_.unwrap();
    return XIELUParams{
        .input = input.data_ptr(),
        .out = out.data_ptr(),
        .num_elems = static_cast<uint32_t>(num_elems),
        .alpha_p = alpha_p,
        .alpha_n = alpha_n,
        .beta = beta,
        .eps = eps,
    };
  }

  /**
   * \brief Validate the tensors and launch `xielu_kernel`.
   *
   * \param input   Contiguous input, flattened
   * \param out     Contiguous output; same shape, dtype and device as `input`
   * \param alpha_p softplus(alpha_p)
   * \param alpha_n beta + softplus(alpha_n)
   */
  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView out,
      float alpha_p,
      float alpha_n,
      float beta,
      float eps) {
    DLDevice device;
    const auto params = make_params(input, out, alpha_p, alpha_n, beta, eps, &device);
    if (params.num_elems == 0) return;
    // Use the widest vector that both buffers are aligned to and that still leaves enough
    // threads to fill the GPU: at decode sizes a few elements per thread is latency-bound
    // on the per-element expm1f, at prefill sizes wide vectors reach DRAM bandwidth.
    const uint32_t min_threads = host::runtime::get_sm_count(device.device_id) * 512;
    const auto addr = reinterpret_cast<uintptr_t>(params.input) | reinterpret_cast<uintptr_t>(params.out);
    const auto fits = [&](uint32_t vec_bytes) {
      return addr % vec_bytes == 0 && params.num_elems / (vec_bytes / sizeof(T)) >= min_threads;
    };
    if (device::kMaxVecBytes >= 32 && fits(32)) {
      launch<device::kMaxVecBytes>(params, device);
    } else if (fits(16)) {
      launch<16>(params, device);
    } else if (fits(8)) {
      launch<8>(params, device);
    } else if (addr % 4 == 0) {
      launch<4>(params, device);
    } else {
      launch<sizeof(T)>(params, device);
    }
  }
};

}  // namespace sglang
