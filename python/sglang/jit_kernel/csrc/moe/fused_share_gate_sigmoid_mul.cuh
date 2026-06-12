#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

namespace {

constexpr int kWarpSize = 32;                           // number of threads per warp
constexpr int kWarpsPerBlock = 8;                       // 8 warps per block
constexpr int kBlockSize = kWarpSize * kWarpsPerBlock;  // 256 threads per block

__device__ __forceinline__ float fast_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

template <typename Float>
__global__ void fused_share_gate_sigmoid_mul_kernel(
    Float* __restrict__ output,                     // [M, K]
    const Float* __restrict__ hidden_state,         // [M, K]
    const Float* __restrict__ share_gate_weight,    // [1, K]
    const Float* __restrict__ share_expert_output,  // [M, K]
    int64_t M,                                      // number of tokens
    int64_t K                                       // dimension of feature
) {
  static_assert(sizeof(Float) == 2, "Only support FP16/BF16");

  using namespace device;
  using Storage = AlignedVector<packed_t<Float>, 4>;  // fixed 128-bit vector

  const auto warp_id = threadIdx.x / kWarpSize;
  const auto lane_id = threadIdx.x % kWarpSize;
  const auto token_idx = blockIdx.x * kWarpsPerBlock + warp_id;
  const auto gmem = tile::Memory<Storage>::warp();

  if (token_idx >= M) {
    return;
  }

  // Compute sigmoid gate value
  float gate_value = 0.0f;
  for (auto i = 0; (i * 8 * kWarpSize + 8 * lane_id) < K; ++i) {
    const auto ptr_offset = token_idx * K + i * 8 * kWarpSize;
    const auto hidden_state_ptr = pointer::offset<Float>(hidden_state, ptr_offset);
    const auto hidden_state_vec = gmem.load(hidden_state_ptr);
    const auto share_gate_weight_ptr = pointer::offset<Float>(share_gate_weight, i * 8 * kWarpSize);
    const auto share_gate_weight_vec = gmem.load(share_gate_weight_ptr);

#pragma unroll
    for (auto j = 0u; j < 4; ++j) {
      const auto [x0, x1] = cast<fp32x2_t>(hidden_state_vec[j]);
      const auto [y0, y1] = cast<fp32x2_t>(share_gate_weight_vec[j]);
      gate_value += x0 * y0;
      gate_value += x1 * y1;
    }
  }
  gate_value = warp::reduce_sum(gate_value);
  auto sigmoid_gate_value = fast_sigmoid(gate_value);

  // Compute output
  for (auto i = 0; (i * 8 * kWarpSize + 8 * lane_id) < K; ++i) {
    const auto ptr_offset = token_idx * K + i * 8 * kWarpSize;
    const auto share_expert_output_ptr = pointer::offset<Float>(share_expert_output, ptr_offset);
    const auto share_expert_output_vec = gmem.load(share_expert_output_ptr);
    const auto output_ptr = pointer::offset<Float>(output, ptr_offset);
    Storage output_vec;
#pragma unroll
    for (auto j = 0u; j < 4; ++j) {
      const auto [x0, x1] = cast<fp32x2_t>(share_expert_output_vec[j]);
      output_vec[j] = cast<packed_t<Float>, fp32x2_t>({x0 * sigmoid_gate_value, x1 * sigmoid_gate_value});
    }
    gmem.store(output_ptr, output_vec);
  }
}

template <typename DType>
struct FusedShareGateSigmoidMulKernel {
  static void
  run(tvm::ffi::TensorView output,
      const tvm::ffi::TensorView hidden_state,
      const tvm::ffi::TensorView share_gate_weight,
      const tvm::ffi::TensorView share_expert_output) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto K = SymbolicSize{"hidden_size"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, K})  // output
        .with_strides({K, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(output);
    TensorMatcher({M, K})  // hidden_state
        .with_strides({K, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(hidden_state);
    TensorMatcher({1, K})  // share_gate_weight
        .with_strides({K, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(share_gate_weight);
    TensorMatcher({M, K})  // share_expert_output
        .with_strides({K, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(share_expert_output);

    auto kernel = fused_share_gate_sigmoid_mul_kernel<DType>;
    dim3 grid((M.unwrap() + kWarpsPerBlock - 1) / kWarpsPerBlock);
    dim3 block(kBlockSize);
    LaunchKernel(grid, block, device.unwrap())
        .enable_pdl(false)(
            kernel,
            reinterpret_cast<DType*>(output.data_ptr()),
            reinterpret_cast<const DType*>(hidden_state.data_ptr()),
            reinterpret_cast<const DType*>(share_gate_weight.data_ptr()),
            reinterpret_cast<const DType*>(share_expert_output.data_ptr()),
            M.unwrap(),
            K.unwrap());
  }
};

}  // namespace
