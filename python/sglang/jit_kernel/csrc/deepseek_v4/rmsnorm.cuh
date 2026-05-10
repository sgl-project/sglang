#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

namespace {

constexpr uint32_t kBlockSize = 128;
constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;

struct RMSNormSelfParams {
  const void* __restrict__ input;
  void* __restrict__ output;
  int64_t stride_batch_bytes_0;
  int64_t stride_head_bytes_0;
  int64_t stride_batch_bytes_1;
  int64_t stride_head_bytes_1;
  uint32_t batch_size;
  uint32_t num_head;
  float eps;
};

template <typename DType, int64_t kHeadDim, bool kUsePDL>
__global__ __launch_bounds__(kBlockSize, 20)  //
    void rmsnorm_self(const __grid_constant__ RMSNormSelfParams params) {
  using namespace device;
  constexpr int64_t kVecSize = 16 / sizeof(DType);
  constexpr uint32_t kNumLoop = kHeadDim / (kVecSize * kWarpThreads);
  static_assert(kHeadDim % (kWarpThreads * kVecSize) == 0);
  using DType2 = packed_t<DType>;
  using Vec = AlignedVector<DType2, kVecSize / 2>;

  const auto warp_id = blockIdx.x * kNumWarps + threadIdx.x / kWarpThreads;
  const auto batch_id = warp_id / params.num_head;
  const auto head_id = warp_id % params.num_head;
  const auto gmem = tile::Memory<Vec>::warp();
  if (batch_id >= params.batch_size) return;
  const auto input_ptr = pointer::offset(  //
      params.input,
      batch_id * params.stride_batch_bytes_0,
      head_id * params.stride_head_bytes_0);
  const auto output_ptr = pointer::offset(  //
      params.output,
      batch_id * params.stride_batch_bytes_1,
      head_id * params.stride_head_bytes_1);
  PDLWaitPrimary<kUsePDL>();  // wait for primary kernel

  Vec inputs[kNumLoop];
#pragma unroll
  for (uint32_t i = 0; i < kNumLoop; ++i) {
    inputs[i] = gmem.load(input_ptr, i);
  }

  // compute sum of squares
  float local_sum = 0;
#pragma unroll
  for (uint32_t i = 0; i < kNumLoop; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < kVecSize / 2; ++j) {
      const auto [x, y] = cast<fp32x2_t>(inputs[i][j]);
      local_sum += x * x + y * y;
    }
  }

  const auto sum_of_squares = warp::reduce_sum(local_sum);
  const auto factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);

  // weight must be identity (null, not used)
#pragma unroll
  for (uint32_t i = 0; i < kNumLoop; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < kVecSize / 2; ++j) {
      const auto [x, y] = cast<fp32x2_t>(inputs[i][j]);
      inputs[i][j] = cast<DType2>(fp32x2_t{x * factor, y * factor});
    }
    gmem.store(output_ptr, inputs[i], i);
  }

  PDLTriggerSecondary<kUsePDL>();  // launch secondary kernel
}

template <int64_t kHeadDim, typename DType, bool kUsePDL>
struct RMSNormKernel {
  static constexpr auto kernel_self = rmsnorm_self<DType, kHeadDim, kUsePDL>;

  static void run_self(tvm::ffi::TensorView input, tvm::ffi::TensorView output, float eps) {
    using namespace host;

    auto N = SymbolicSize{"batch_size"};
    auto H = SymbolicSize{"num_heads"};
    constexpr auto D = kHeadDim;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, H, D})  // input
        .with_strides({-1, -1, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(input);
    TensorMatcher({N, H, D})  // output
        .with_strides({-1, -1, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(output);

    const auto batch_size = static_cast<uint32_t>(N.unwrap());
    const auto num_head = static_cast<uint32_t>(H.unwrap());
    const auto params = RMSNormSelfParams{
        .input = input.data_ptr(),
        .output = output.data_ptr(),
        .stride_batch_bytes_0 = static_cast<int64_t>(input.stride(0) * sizeof(DType)),
        .stride_head_bytes_0 = static_cast<int64_t>(input.stride(1) * sizeof(DType)),
        .stride_batch_bytes_1 = static_cast<int64_t>(output.stride(0) * sizeof(DType)),
        .stride_head_bytes_1 = static_cast<int64_t>(output.stride(1) * sizeof(DType)),
        .batch_size = batch_size,
        .num_head = num_head,
        .eps = eps,
    };
    if (batch_size == 0 || num_head == 0) return;
    const auto needed_warps = batch_size * num_head;
    const auto num_blocks = div_ceil(needed_warps, kNumWarps);
    LaunchKernel(num_blocks, kBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel_self, params);
  }
};

}  // namespace
