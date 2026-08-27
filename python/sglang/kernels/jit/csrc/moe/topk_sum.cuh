// Top-k expert-output sum: out[M, K] = sum_j in[M, topk, K].
//
// Replaces sgl_kernel's moe_sum_reduce_kernel_general (~5.7us at decode
// shapes [1, 16, 3584]) with a straightforward vectorized pass (~1.5us):
// one thread per 8-element vector of K, looping the topk rows in fp32.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/type.cuh>   // For bf16_t, fp32_t, device::cast
#include <sgl_kernel/utils.cuh>  // For LaunchKernel
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct TopkSumParams {
  const bf16_t* __restrict__ in;  // [M, topk, K] contiguous
  bf16_t* __restrict__ out;       // [M, K] contiguous
  uint32_t K;
  uint32_t topk;
};

template <int kThreads, bool kUsePDL>
__global__ void topk_sum_kernel(const TopkSumParams __grid_constant__ params) {
  using namespace device;

  constexpr int kVecN = 8;  // 8 bf16 = 128 bits
  using vec_bf16_t = AlignedVector<bf16_t, kVecN>;

  const uint32_t m = blockIdx.y;
  const uint32_t v = blockIdx.x * kThreads + threadIdx.x;
  const uint32_t n_vecs = params.K / kVecN;
  if (v >= n_vecs) return;

  const bf16_t* base = params.in + static_cast<int64_t>(m) * params.topk * params.K;

  PDLWaitPrimary<kUsePDL>();

  float acc[kVecN];
#pragma unroll
  for (int i = 0; i < kVecN; ++i) {
    acc[i] = 0.0f;
  }
  for (uint32_t j = 0; j < params.topk; ++j) {
    vec_bf16_t x;
    x.load(base + static_cast<int64_t>(j) * params.K, v);
#pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      acc[i] += cast<fp32_t>(x[i]);
    }
  }

  vec_bf16_t o;
#pragma unroll
  for (int i = 0; i < kVecN; ++i) {
    o[i] = cast<bf16_t>(acc[i]);
  }
  o.store(params.out + static_cast<int64_t>(m) * params.K, v);

  PDLTriggerSecondary<kUsePDL>();
}

template <int kThreads, bool kUsePDL>
struct TopkSumKernel {
  static constexpr auto kernel = topk_sum_kernel<kThreads, kUsePDL>;

  static void run(const tvm::ffi::TensorView in, const tvm::ffi::TensorView out) {
    using namespace host;

    auto M_ = SymbolicSize{"num_tokens"};
    auto T_ = SymbolicSize{"topk"};
    auto K_ = SymbolicSize{"hidden"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M_, T_, K_}).with_dtype<bf16_t>().with_device(device).verify(in);
    TensorMatcher({M_, K_}).with_dtype<bf16_t>().with_device(device).verify(out);

    const auto M = static_cast<uint32_t>(M_.unwrap());
    const auto topk = static_cast<uint32_t>(T_.unwrap());
    const auto K = static_cast<uint32_t>(K_.unwrap());

    RuntimeCheck(K % 8 == 0, "K must be divisible by 8 for vectorized loads");
    if (M == 0) return;

    const auto params = TopkSumParams{
        .in = static_cast<const bf16_t*>(in.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .K = K,
        .topk = topk,
    };

    const uint32_t n_vecs = K / 8;
    dim3 grid((n_vecs + kThreads - 1) / kThreads, M);
    LaunchKernel(grid, kThreads, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
