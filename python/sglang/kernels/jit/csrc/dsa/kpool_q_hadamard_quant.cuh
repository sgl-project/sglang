#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>
#include <sgl_kernel/dsa/indexer_rope.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

struct KPoolQParams {
  const bf16_t* __restrict__ q;
  const float* __restrict__ head_gate;
  void* __restrict__ q8;
  float* __restrict__ weights;
  uint32_t rows;
  float softmax_scale;
};

template <bool kUsePDL>
__global__ void kpool_q_hadamard_quant(const __grid_constant__ KPoolQParams p) {
  using namespace device;
  constexpr int32_t kDim = 128;
  constexpr int32_t kVec = 4;
  using Input = AlignedVector<bf16_t, kVec>;
  using Output = AlignedVector<fp8x2_e4m3_t, 2>;
  const uint32_t row = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpThreads;
  const uint32_t lane = get_lane_id();
  if (row >= p.rows) return;

  PDLWaitPrimary<kUsePDL>();
  Input input;
  input.load(p.q + static_cast<int64_t>(row) * kDim, lane);
  AlignedVector<float, kVec> data;
#pragma unroll
  for (int i = 0; i < kVec; ++i)
    data[i] = cast<float>(input[i]);
  dsa_indexer::hadamard_128(data, lane);

  // Match the BF16 intermediate and the 1e-4 floor of DSA act_quant.
  float local_max = 0.0f;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    data[i] = cast<float>(cast<bf16_t>(data[i]));
    local_max = math::max(local_max, math::abs(data[i]));
  }
  const float abs_max = warp::reduce_max(local_max);
  const int32_t exp = deepseek_v4::fp8::cast_to_ue8m0(fmaxf(abs_max, 1e-4f) * (1.0f / math::FP8_E4M3_MAX));
  const float scale = __uint_as_float(static_cast<uint32_t>(exp) << 23);
  const float inv_scale = deepseek_v4::fp8::inv_scale_ue8m0(exp);

  Output result;
  result[0] = deepseek_v4::fp8::pack_fp8(data[0] * inv_scale, data[1] * inv_scale);
  result[1] = deepseek_v4::fp8::pack_fp8(data[2] * inv_scale, data[3] * inv_scale);
  result.store(static_cast<uint8_t*>(p.q8) + static_cast<int64_t>(row) * kDim, lane);
  if (lane == 0) {
    // The caller already applied the head-count scale in FP32.
    p.weights[row] = p.head_gate[row] * scale * p.softmax_scale;
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct KPoolQHadamardQuantKernel {
  static void forward(
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView gate,
      tvm::ffi::TensorView q8,
      tvm::ffi::TensorView weights,
      double softmax_scale) {
    using namespace host;
    constexpr int64_t kDim = 128;
    auto B = SymbolicSize{"batch_size"};
    auto H = SymbolicSize{"heads"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({B, H, kDim})  //
        .with_dtype<bf16_t>()
        .with_device(device_)
        .ensure_alignment(8)
        .verify(q);
    TensorMatcher({B, H})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(gate);
    TensorMatcher({B, H, kDim})  //
        .with_dtype<fp8_e4m3_t>()
        .with_device(device_)
        .ensure_alignment(4)
        .verify(q8);
    TensorMatcher({B, H, 1})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(weights);
    const auto rows = B.unwrap() * H.unwrap();
    if (rows == 0) return;
    KPoolQParams p{
        static_cast<const bf16_t*>(q.data_ptr()),
        static_cast<const float*>(gate.data_ptr()),
        q8.data_ptr(),
        static_cast<float*>(weights.data_ptr()),
        static_cast<uint32_t>(rows),
        static_cast<float>(softmax_scale)};
    constexpr uint32_t kThreads = 128;
    constexpr uint32_t kWarps = kThreads / device::kWarpThreads;
    LaunchKernel(div_ceil(p.rows, kWarps), kThreads, device_.unwrap())
        .enable_pdl(kUsePDL)(kpool_q_hadamard_quant<kUsePDL>, p);
  }
};

}  // namespace sglang
