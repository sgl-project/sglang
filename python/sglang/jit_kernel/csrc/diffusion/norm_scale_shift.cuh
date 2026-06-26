// Minimal native-CUDA fast path for Qwen-Image diffusion norm-scale-shift.
//
// Supported shape family:
//   - bf16 activations, B == 1, hidden dim == 3072
//   - layer norm only, no affine weight/bias
//   - scale/shift are bf16 row-broadcast tensors ([D], [1,D], or [1,1,D])
//   - optional residual path uses a bf16 row-broadcast gate
//
// All other public-op inputs fall back to the existing CuTe-DSL implementation
// from the Python dispatcher.

#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice

#include <sgl_kernel/math.cuh>   // For device::math::rsqrt
#include <sgl_kernel/utils.cuh>  // For SGL_DEVICE, bf16_t, LaunchKernel
#include <sgl_kernel/vec.cuh>    // For AlignedVector
#include <sgl_kernel/warp.cuh>   // For warp::reduce_sum

#include <cstdint>

namespace sglang_norm_scale_shift {

namespace {

constexpr int kHidden = 3072;
constexpr int kVecElems = 16;  // 32B/thread for bf16 on Blackwell.
constexpr int kThreads = kHidden / kVecElems;
constexpr int kWarps = kThreads / device::kWarpThreads;
constexpr float kInvHidden = 1.0f / float(kHidden);

static_assert(kThreads == 192);
static_assert(kWarps == 6);

struct QwenImageNormParams {
  void* y;
  void* res_out;
  const void* x;
  const void* residual;
  const void* gate;
  const void* scale;
  const void* shift;
  float eps;
};

SGL_DEVICE float cta_reduce_sum(float v, int warp, int lane, float* scratch) {
  v = device::warp::reduce_sum(v);
  if (lane == 0) {
    scratch[warp] = v;
  }
  __syncthreads();

  if (warp == 0) {
    float a = lane < kWarps ? scratch[lane] : 0.0f;
    a = device::warp::reduce_sum(a);
    if (lane == 0) {
      scratch[kWarps] = a;
    }
  }
  __syncthreads();
  return scratch[kWarps];
}

template <bool kHasResidual>
__global__ void qwen_image_norm_scale_shift_kernel(const QwenImageNormParams __grid_constant__ params) {
  using namespace device;
  using Vec = AlignedVector<bf16_t, kVecElems>;

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & int(kWarpThreads - 1);
  const int warp = tid >> 5;
  const int row_offset = row * kHidden;
  const int elem_offset = tid * kVecElems;

  __shared__ float scratch_a[kWarps + 1];
  __shared__ float scratch_b[kWarps + 1];

  Vec xv;
  xv.load(static_cast<const bf16_t*>(params.x) + row_offset + elem_offset);

  float v[kVecElems];
#pragma unroll
  for (int i = 0; i < kVecElems; ++i) {
    v[i] = static_cast<float>(xv[i]);
  }

  if constexpr (kHasResidual) {
    Vec gv;
    Vec rv;
    Vec ro;
    gv.load(static_cast<const bf16_t*>(params.gate) + elem_offset);
    rv.load(static_cast<const bf16_t*>(params.residual) + row_offset + elem_offset);

#pragma unroll
    for (int i = 0; i < kVecElems; ++i) {
      const bf16_t rounded = static_cast<bf16_t>(v[i] * static_cast<float>(gv[i]) + static_cast<float>(rv[i]));
      ro[i] = rounded;
      v[i] = static_cast<float>(rounded);
    }
    ro.store(static_cast<bf16_t*>(params.res_out) + row_offset + elem_offset);
  }

  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kVecElems; ++i) {
    sum += v[i];
  }
  const float mean = cta_reduce_sum(sum, warp, lane, scratch_a) * kInvHidden;

  float var_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kVecElems; ++i) {
    const float d = v[i] - mean;
    var_sum += d * d;
  }
  const float var = cta_reduce_sum(var_sum, warp, lane, scratch_b) * kInvHidden;
  const float factor = math::rsqrt(var + params.eps);

  Vec scv;
  Vec shv;
  Vec yv;
  scv.load(static_cast<const bf16_t*>(params.scale) + elem_offset);
  shv.load(static_cast<const bf16_t*>(params.shift) + elem_offset);

#pragma unroll
  for (int i = 0; i < kVecElems; ++i) {
    const float norm = static_cast<float>(static_cast<bf16_t>((v[i] - mean) * factor));
    yv[i] = static_cast<bf16_t>(norm * (1.0f + static_cast<float>(scv[i])) + static_cast<float>(shv[i]));
  }
  yv.store(static_cast<bf16_t*>(params.y) + row_offset + elem_offset);
}

inline uint32_t verify_qwen_geometry(host::SymbolicSize& num_rows) {
  using namespace host;
  RuntimeCheck(num_rows.unwrap() > 0, "num_rows must be positive");
  RuntimeCheck(num_rows.unwrap() <= int64_t(UINT32_MAX), "num_rows out of range");
  return static_cast<uint32_t>(num_rows.unwrap());
}

}  // namespace

struct QwenImageNormScaleShiftKernel {
  static void
  run(tvm::ffi::TensorView y,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView scale,
      tvm::ffi::TensorView shift,
      double eps) {
    using namespace host;
    auto N = SymbolicSize{"num_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, kHidden}).with_dtype<bf16_t>().with_device(device).verify(x).verify(y);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device).verify(scale).verify(shift);

    const uint32_t grid = verify_qwen_geometry(N);
    const auto params = QwenImageNormParams{
        .y = y.data_ptr(),
        .res_out = nullptr,
        .x = x.data_ptr(),
        .residual = nullptr,
        .gate = nullptr,
        .scale = scale.data_ptr(),
        .shift = shift.data_ptr(),
        .eps = static_cast<float>(eps),
    };
    LaunchKernel(grid, kThreads, device.unwrap())(qwen_image_norm_scale_shift_kernel<false>, params);
  }
};

struct QwenImageScaleResidualNormScaleShiftKernel {
  static void
  run(tvm::ffi::TensorView y,
      tvm::ffi::TensorView res_out,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView gate,
      tvm::ffi::TensorView scale,
      tvm::ffi::TensorView shift,
      double eps) {
    using namespace host;
    auto N = SymbolicSize{"num_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, kHidden})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(x)
        .verify(residual)
        .verify(y)
        .verify(res_out);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device).verify(gate).verify(scale).verify(shift);

    const uint32_t grid = verify_qwen_geometry(N);
    const auto params = QwenImageNormParams{
        .y = y.data_ptr(),
        .res_out = res_out.data_ptr(),
        .x = x.data_ptr(),
        .residual = residual.data_ptr(),
        .gate = gate.data_ptr(),
        .scale = scale.data_ptr(),
        .shift = shift.data_ptr(),
        .eps = static_cast<float>(eps),
    };
    LaunchKernel(grid, kThreads, device.unwrap())(qwen_image_norm_scale_shift_kernel<true>, params);
  }
};

}  // namespace sglang_norm_scale_shift
