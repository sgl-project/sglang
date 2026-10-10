// Bootstrap per-token/per-branch square sums from stored BF16/FP16 values.
// Subsequent Apply kernels produce pre-rounding FP32 sums directly.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {
struct HcRawTransitionParams {
  const void* residual;
  float* sum_sq;
};

template <int64_t C, int64_t H, typename Float>
__global__ __launch_bounds__(H / 16) void hc_raw_transition_kernel(const HcRawTransitionParams __grid_constant__ p) {
  using namespace device;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 8>;
  constexpr uint32_t kThreads = H / 16;
  constexpr uint32_t kWarps = kThreads / kWarpThreads;
  static_assert(C == 4 && H % 512 == 0 && H <= 16384);
  const uint32_t branch = blockIdx.x % C;
  const uint32_t token = blockIdx.x / C;
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane = threadIdx.x % kWarpThreads;
  const int64_t offset = static_cast<int64_t>(token) * C * H + branch * H;
  const auto mem = tile::Memory<Storage>::cta(kThreads);
  __shared__ float partial[kWarpThreads];
  const Storage r = mem.load(pointer::offset<Float>(p.residual, offset), 0);
  float sum = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const auto [ux, uy] = cast<fp32x2_t>(r[i]);
    sum += ux * ux + uy * uy;
  }
  sum = warp::reduce_sum(sum);
  if (lane == 0) partial[warp_id] = sum;
  __syncthreads();
  if (warp_id == 0) {
    float value = warp::reduce_sum(lane < kWarps ? partial[lane] : 0.0f);
    if (lane == 0) {
      p.sum_sq[static_cast<int64_t>(token) * C + branch] = value;
    }
  }
}

template <int64_t C, int64_t H, typename DType>
struct HcRawTransitionKernel {
  static void stats_sum(tvm::ffi::TensorView r, tvm::ffi::TensorView sums) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, C * H}).with_dtype<DType>().with_device(device).verify(r);
    TensorMatcher({M, C}).with_dtype<float>().with_device(device).verify(sums);
    const auto p = HcRawTransitionParams{r.data_ptr(), static_cast<float*>(sums.data_ptr())};
    LaunchKernel(static_cast<uint32_t>(M.unwrap()) * C, H / 16, device.unwrap())(
        hc_raw_transition_kernel<C, H, DType>, p);
  }
};
}  // namespace sglang
