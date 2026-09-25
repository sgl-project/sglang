// Bit-exact channel-first RMSNorm finish + SiLU for the Qwen-Image 2.1 VAE.
//
// The fp32 L2 norm over channels comes from aten (x.float().norm(dim=1)) so the
// reduction order stays the reference's; this kernel replays the pointwise
// tail of QwenImage21RMS_norm followed by nn.SiLU:
//   v = bf16(x / max(norm, 1e-12))     (F.normalize in fp32, then .to(bf16))
//   v = bf16(v * scale); v = bf16(v * gamma[c]); v = bf16(v + 0.0)   (bias 0.0 folds -0 to +0)
//   out = bf16(v / (1 + expf(-v)))     (aten silu on the bf16 value)
// Layout: contiguous [N, C, S] activations (NCHW / NCDHW flattened), norm [N, S] fp32.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <algorithm>
#include <cstdint>

namespace sglang {

namespace channel_rmsnorm_finish_silu {

constexpr int kVecElems = 8;  // 16 bytes of bf16 along the spatial axis
constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxBlocks = 65535;
constexpr float kNormFloor = 1.0e-12f;

using XVec = device::AlignedVector<bf16_t, kVecElems>;
using NormVec = device::AlignedVector<fp32_t, 4>;

struct Params {
  bf16_t* out;
  const bf16_t* x;
  const fp32_t* norm;
  const bf16_t* gamma;
  int64_t channels;
  int64_t spatial;
  int64_t num_vectors;
  float scale;
};

SGL_DEVICE float div_rn(float numerator, float denominator) {
  float out;
  asm volatile("div.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(numerator), "f"(denominator));
  return out;
}

__global__ void kernel(const Params __grid_constant__ params) {
  using namespace device;
  const int64_t spatial_vectors = params.spatial / kVecElems;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t vector = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; vector < params.num_vectors;
       vector += stride) {
    const int64_t s_vector = vector % spatial_vectors;
    const int64_t row = vector / spatial_vectors;  // (n, c) row
    const int64_t c = row % params.channels;
    const int64_t n = row / params.channels;
    const int64_t element = row * params.spatial + s_vector * kVecElems;
    const int64_t norm_element = n * params.spatial + s_vector * kVecElems;

    XVec x;
    NormVec norm_lo;
    NormVec norm_hi;
    x.load(params.x + element);
    norm_lo.load(params.norm + norm_element);
    norm_hi.load(params.norm + norm_element + 4);
    const float gamma = cast<fp32_t>(params.gamma[c]);

    XVec out;
#pragma unroll
    for (int i = 0; i < kVecElems; ++i) {
      const float denominator = fmaxf(i < 4 ? norm_lo[i] : norm_hi[i - 4], kNormFloor);
      float value = cast<fp32_t>(cast<bf16_t>(div_rn(cast<fp32_t>(x[i]), denominator)));
      value = cast<fp32_t>(cast<bf16_t>(__fmul_rn(value, params.scale)));
      value = cast<fp32_t>(cast<bf16_t>(__fmul_rn(value, gamma)));
      // + bias (0.0): rounds to bf16 and turns -0.0 into +0.0 exactly like the eager add.
      const float normed = cast<fp32_t>(cast<bf16_t>(__fadd_rn(value, 0.0f)));
      out[i] = cast<bf16_t>(div_rn(normed, __fadd_rn(1.0f, expf(-normed))));
    }
    out.store(params.out + element);
  }
}

struct Kernel {
  /// \brief out = silu(rmsnorm_finish(x, norm, gamma, scale)) over contiguous [N, C, S] bf16 rows.
  static void
  run(tvm::ffi::TensorView out,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView norm,
      tvm::ffi::TensorView gamma,
      double scale) {
    using namespace host;
    auto N = SymbolicSize{"batch"};
    auto C = SymbolicSize{"channels"};
    auto S = SymbolicSize{"spatial"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N, C, S}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(out).verify(x);
    TensorMatcher({N, S}).with_dtype<fp32_t>().with_device(device).ensure_alignment(16).verify(norm);
    TensorMatcher({C}).with_dtype<bf16_t>().with_device(device).verify(gamma);
    const int64_t batch = N.unwrap(), channels = C.unwrap(), spatial = S.unwrap();
    CHECK_HOST(spatial % kVecElems == 0) << "spatial size must be a multiple of " << kVecElems << ", got " << spatial;
    if (batch == 0 || channels == 0 || spatial == 0) return;
    const int64_t num_vectors = batch * channels * spatial / kVecElems;
    const auto params = Params{
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .norm = static_cast<const fp32_t*>(norm.data_ptr()),
        .gamma = static_cast<const bf16_t*>(gamma.data_ptr()),
        .channels = channels,
        .spatial = spatial,
        .num_vectors = num_vectors,
        .scale = static_cast<float>(scale),
    };
    const auto blocks = static_cast<uint32_t>(
        std::min<int64_t>(div_ceil(num_vectors, static_cast<int64_t>(kThreads)), kMaxBlocks));
    LaunchKernel(blocks, kThreads, device.unwrap())(kernel, params);
  }
};

}  // namespace channel_rmsnorm_finish_silu

}  // namespace sglang
