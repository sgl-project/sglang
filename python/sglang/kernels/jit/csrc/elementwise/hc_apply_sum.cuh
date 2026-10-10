// Shared HC Apply: accumulate FP32 update squares BEFORE the storage cast.
// The caller owns and clears sum_sq. This kernel does NOT produce inv_rms.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {
struct HcRawFp32StatParams {
  const void* block_output;
  const void* residual;
  const float* alpha;
  void* output;
  float* sum_sq;
};

template <int64_t C, int64_t H, typename Float>
__global__
__launch_bounds__(160) void hc_raw_transition_fp32_stat_kernel(const HcRawFp32StatParams __grid_constant__ p) {
  using namespace device;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;
  constexpr uint32_t kThreads = 160;
  constexpr uint32_t kVecsPerHalf = H / 16;
  constexpr uint32_t kIterations = (kVecsPerHalf + kThreads - 1) / kThreads;
  static_assert(C == 4 && H > 0 && H % 512 == 0 && H <= 16384);
  static_assert(sizeof(Float) == 2);

  const uint32_t token = blockIdx.x;
  const uint32_t branch = blockIdx.y / 2;
  const uint32_t half = blockIdx.y % 2;
  const uint32_t lane = threadIdx.x % kWarpThreads;
  const int64_t offset = (static_cast<int64_t>(token) * C + branch) * H;
  const auto r_ptr = pointer::offset<Float>(p.residual, offset);
  const auto y_ptr = pointer::offset<Float>(p.block_output, static_cast<int64_t>(token) * H);
  const auto out_ptr = pointer::offset<Float>(p.output, offset);
  const float a = p.alpha[static_cast<int64_t>(token) * C + branch];

  float sum = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < kIterations; ++j) {
    const uint32_t local_vec = threadIdx.x + j * kThreads;
    if (local_vec < kVecsPerHalf) {
      const uint32_t vec_idx = half * kVecsPerHalf + local_vec;
      Storage r, y, updated;
      r.load(r_ptr, vec_idx);
      y.load(y_ptr, vec_idx);
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        const auto [rx, ry] = cast<fp32x2_t>(r[i]);
        const auto [yx, yy] = cast<fp32x2_t>(y[i]);
        const float ux = rx + a * yx;
        const float uy = ry + a * yy;
        sum += ux * ux + uy * uy;
        updated[i] = cast<Float2>(fp32x2_t{ux, uy});
      }
      updated.store(out_ptr, vec_idx);
    }
  }
  sum = warp::reduce_sum(sum);
  // The caller must clear every slot before this launch, including Graph replays.
  // Full-device atomic scope: the two ordinary CTAs need not share a cluster.
  // No old-value consumer, shared-memory reduction, or CTA barrier.
  if (lane == 0) atomicAdd(p.sum_sq + static_cast<int64_t>(token) * C + branch, sum);
}

template <int64_t C, int64_t H, typename DType>
struct HcRawFp32StatKernel {
  static void
  run(tvm::ffi::TensorView y,
      tvm::ffi::TensorView r,
      tvm::ffi::TensorView alpha,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView sums) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, H}).with_dtype<DType>().with_device(device).verify(y);
    TensorMatcher({M, C * H}).with_dtype<DType>().with_device(device).verify(r).verify(out);
    TensorMatcher({M, C}).with_dtype<float>().with_device(device).verify(alpha);
    TensorMatcher({M, C}).with_dtype<float>().with_device(device).verify(sums);
    const auto p = HcRawFp32StatParams{
        y.data_ptr(),
        r.data_ptr(),
        static_cast<const float*>(alpha.data_ptr()),
        out.data_ptr(),
        static_cast<float*>(sums.data_ptr())};
    LaunchKernel(dim3(static_cast<uint32_t>(M.unwrap()), 2 * C, 1), 160, device.unwrap())(
        hc_raw_transition_fp32_stat_kernel<C, H, DType>, p);
  }
};
}  // namespace sglang
