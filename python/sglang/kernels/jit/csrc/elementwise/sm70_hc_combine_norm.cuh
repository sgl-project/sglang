// SPDX-License-Identifier: Apache-2.0
// Fuse the existing HC apply and grouped-Gemma norm, preserving FP16 boundaries.
#include "hc_combine.cuh"

namespace sglang::sm70_hc_combine_norm {

struct Params {
  HcCombineSplitParams combine;
  const half* gamma;
  half* normed;
  float eps;
};

__global__ __launch_bounds__(160) void apply_norm(Params p) {
  using namespace device;
  using F2 = packed_t<half>;
  using V = AlignedVector<F2, 4>;
  const int row = blockIdx.x / 4, branch = blockIdx.x % 4;
  const int t = threadIdx.x, warp_id = t / 32;
  const half* yp = static_cast<const half*>(p.combine.block_output) + row * 2560;
  const half* rp = static_cast<const half*>(p.combine.residual) + row * 10240 + branch * 2560;
  half* op = static_cast<half*>(p.combine.output) + row * 10240 + branch * 2560;
  const half* gp = p.gamma + branch * 2560;
  half* np = p.normed + row * 10240 + branch * 2560;

  float total = 0.f;
#pragma unroll
  for (int s = 0; s < 8; s++) total += p.combine.partials[(row * 8 + s) * 4 + branch];
  float a = 2.f / (1.f + math::exp(-total / 4.f));
  V x[2], weight[2];
  float squares = 0.f;
#pragma unroll
  for (int j = 0; j < 2; j++) {
    V r, y;
    r.load(rp, t + j * 160);
    y.load(yp, t + j * 160);
    weight[j].load(gp, t + j * 160);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      auto [rx, ry] = cast<fp32x2_t>(r[i]);
      auto [yx, yy] = cast<fp32x2_t>(y[i]);
      // The following norm must consume the rounded combined FP16 value.
      x[j][i] = cast<F2>(fp32x2_t{rx + a * yx, ry + a * yy});
    }
    x[j].store(op, t + j * 160);
  }
#pragma unroll
  for (int j = 0; j < 2; j++) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      auto [vx, vy] = cast<fp32x2_t>(x[j][i]);
      squares += vx * vx + vy * vy;
    }
  }
  squares = warp::reduce_sum(squares);
  __shared__ float smem[32];
  if (t % 32 == 0) smem[warp_id] = squares;
  __syncthreads();
  if (warp_id == 0) {
    float local = t < 5 ? smem[t] : 0.f;
    float sum = warp::reduce_sum(local);
    smem[t] = math::rsqrt(sum / 2560.f + p.eps);
  }
  __syncthreads();
  float factor = smem[warp_id];
#pragma unroll
  for (int j = 0; j < 2; j++) {
    V out;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      auto [vx, vy] = cast<fp32x2_t>(x[j][i]);
      auto [wx, wy] = cast<fp32x2_t>(weight[j][i]);
      out[i] = cast<F2>(fp32x2_t{vx * factor * (1.f + wx), vy * factor * (1.f + wy)});
    }
    out.store(np, t + j * 160);
  }
}

void run(
    tvm::ffi::TensorView y,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView normed_residual,
    tvm::ffi::TensorView inject,
    tvm::ffi::TensorView gamma,
    tvm::ffi::TensorView partials,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView normed,
    double eps,
    bool precomputed) {
  using namespace host;
  auto dev = SymbolicDevice{};
  dev.set_options<kDLCUDA>();
  auto rows = SymbolicSize{};
  TensorMatcher({rows, 2560}).with_dtype<half>().with_device(dev).verify(y);
  TensorMatcher({rows, 10240})
      .with_dtype<half>()
      .with_device(dev)
      .verify(residual)
      .verify(normed_residual)
      .verify(output)
      .verify(normed);
  TensorMatcher({4, 10240}).with_dtype<half>().with_device(dev).verify(inject);
  TensorMatcher({10240}).with_dtype<half>().with_device(dev).verify(gamma);
  TensorMatcher({rows, 8, 4}).with_dtype<float>().with_device(dev).verify(partials);
  HcCombineSplitParams hc{
      y.data_ptr(), residual.data_ptr(), normed_residual.data_ptr(), inject.data_ptr(),
      output.data_ptr(), static_cast<float*>(partials.data_ptr())};
  if (!precomputed) {
    LaunchKernel(dim3(rows.unwrap(), 32), 32, dev.unwrap())(
        hc_combine_gate_kernel<4, 2560, false, half>, hc);
  }
  Params p{hc, static_cast<const half*>(gamma.data_ptr()),
           static_cast<half*>(normed.data_ptr()), float(eps)};
  LaunchKernel(rows.unwrap() * 4, 160, dev.unwrap())(apply_norm, p);
}

}  // namespace sglang::sm70_hc_combine_norm
