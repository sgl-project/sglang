// Shared HC Apply: accumulate FP32 update squares BEFORE the storage cast.
// Atomic mode requires cleared sum_sq; partial mode overwrites every warp slot.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {
inline constexpr uint32_t kHcApplyThreads = 160;
inline constexpr uint32_t kHcApplyWarps = kHcApplyThreads / 32;
inline constexpr int64_t kHcApplyPartials = 2 * kHcApplyWarps;

struct HcRawFp32StatParams {
  const void* block_output;
  const void* residual;
  const float* alpha;
  void* output;
  float* sum_sq;
};

template <int64_t C, int64_t H, typename Float, bool WritePartials = false>
__global__
__launch_bounds__(160) void hc_raw_transition_fp32_stat_kernel(const HcRawFp32StatParams __grid_constant__ p) {
  using namespace device;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;
  constexpr uint32_t kThreads = kHcApplyThreads;
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
  if (lane == 0) {
    const int64_t group = static_cast<int64_t>(token) * C + branch;
    if constexpr (WritePartials) {
      const uint32_t slot = half * kHcApplyWarps + threadIdx.x / kWarpThreads;
      p.sum_sq[group * kHcApplyPartials + slot] = sum;
    } else {
      // Full-device scope: the two ordinary CTAs need not share a cluster.
      atomicAdd(p.sum_sq + group, sum);
    }
  }
}

static __global__ __launch_bounds__(128) void hc_sum_partials_reduce_kernel(const float* partials, float* sums) {
  using namespace device;
  const uint32_t lane = threadIdx.x % kWarpThreads;
  const int64_t group = static_cast<int64_t>(blockIdx.x) * 4 + threadIdx.x / kWarpThreads;
  float sum = lane < kHcApplyPartials ? partials[group * kHcApplyPartials + lane] : 0.0f;
  sum = warp::reduce_sum(sum);
  if (lane == 0) sums[group] = sum;
}

template <int64_t C, int64_t H, typename DType, bool WritePartials = false>
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
    if constexpr (WritePartials) {
      TensorMatcher({M, C, kHcApplyPartials}).with_dtype<float>().with_device(device).verify(sums);
    } else {
      TensorMatcher({M, C}).with_dtype<float>().with_device(device).verify(sums);
    }
    const auto p = HcRawFp32StatParams{
        y.data_ptr(),
        r.data_ptr(),
        static_cast<const float*>(alpha.data_ptr()),
        out.data_ptr(),
        static_cast<float*>(sums.data_ptr())};
    LaunchKernel(dim3(static_cast<uint32_t>(M.unwrap()), 2 * C, 1), 160, device.unwrap())(
        hc_raw_transition_fp32_stat_kernel<C, H, DType, WritePartials>, p);
  }
};

struct HcSumPartialsKernel {
  static void run(tvm::ffi::TensorView partials, tvm::ffi::TensorView sums) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, 4, kHcApplyPartials}).with_dtype<float>().with_device(device).verify(partials);
    TensorMatcher({M, 4}).with_dtype<float>().with_device(device).verify(sums);
    LaunchKernel(dim3(static_cast<uint32_t>(M.unwrap())), 128, device.unwrap())(
        hc_sum_partials_reduce_kernel,
        static_cast<const float*>(partials.data_ptr()),
        static_cast<float*>(sums.data_ptr()));
  }
};
}  // namespace sglang
