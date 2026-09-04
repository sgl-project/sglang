// KDA provenance: BBuf/KDA-Pilot, merged in SGLang PR #27392.
// Minimal native-CUDA fast path for generic bf16 hidden=3072 norm-scale-shift.
//
// Supported shape family:
//   - bf16 activations, B == 1, hidden dim == 3072
//   - layer norm only, no affine weight/bias
//   - scale/shift are bf16 row-broadcast tensors ([D], [1,D], or [1,1,D])
//   - optional residual path uses a bf16 row-broadcast gate
//
// All other public-op inputs fall back to the existing CuTe-DSL implementation
// from the Python dispatcher.
//
// Developed with MIT HAN Lab Kernel Design Agents:
// https://github.com/mit-han-lab/kernel-design-agents

#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice

#include <sgl_kernel/math.cuh>   // For device::math::rsqrt
#include <sgl_kernel/type.cuh>   // For DTypeTrait
#include <sgl_kernel/utils.cuh>  // For SGL_DEVICE, bf16_t, LaunchKernel
#include <sgl_kernel/vec.cuh>    // For AlignedVector
#include <sgl_kernel/warp.cuh>   // For warp::reduce_sum

#if defined(ENABLE_FP4) && ENABLE_FP4
// FlashInfer's TensorRT-LLM quantization helper uses the C macro directly.
#ifndef FLT_MAX
#define FLT_MAX __FLT_MAX__
#endif
#include <tensorrt_llm/kernels/quantization_utils.cuh>
#endif

#include <cstdint>

namespace sglang {

namespace norm_scale_shift {

constexpr int kHidden = 3072;
constexpr int kVecElems = 16;  // 32B/thread for bf16 on Blackwell.
constexpr int kThreads = kHidden / kVecElems;
constexpr int kWarps = kThreads / device::kWarpThreads;
constexpr float kInvHidden = 1.0f / float(kHidden);

static_assert(kThreads == 192);
static_assert(kWarps == 6);

struct NormScaleShiftParams {
  void* y;
  void* res_out;
  void* quantized;
  void* quant_scales;
  const void* x;
  const void* input_bias;
  const void* residual;
  const void* gate;
  const void* scale;
  const void* shift;
  const void* input_scale;
  const void* global_scale;
  uint32_t num_rows;
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

SGL_DEVICE float triton_scale_reciprocal(float scale) {
  float reciprocal;
  // Triton's static FP8 quantizer lowers `1.0 / scale` to div.full.f32.
  // Match it exactly because a one-ULP difference at an E4M3 midpoint can
  // change the quantized byte.
  asm("div.full.f32 %0, %1, %2;" : "=f"(reciprocal) : "f"(1.0f), "f"(scale));
  return reciprocal;
}

template <bool kHasResidual, bool kHasInputBias = false, bool kQuantizeFp8 = false, bool kQuantizeNvfp4 = false>
__global__ void norm_scale_shift_kernel(const NormScaleShiftParams __grid_constant__ params) {
  static_assert(!(kQuantizeFp8 && kQuantizeNvfp4));
  using namespace device;
  using Vec = AlignedVector<bf16_t, kVecElems>;

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
#if defined(ENABLE_FP4) && ENABLE_FP4
  if constexpr (kQuantizeNvfp4) {
    if (row >= params.num_rows) {
      auto* scales = static_cast<uint8_t*>(params.quant_scales);
      const int64_t scale_offset = tensorrt_llm::kernels::get_sf_out_offset_128x4(row, tid, kThreads);
      scales[scale_offset] = 0;
      return;
    }
  }
#endif
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

  if constexpr (kHasInputBias) {
    Vec bv;
    bv.load(static_cast<const bf16_t*>(params.input_bias) + elem_offset);
#pragma unroll
    for (int i = 0; i < kVecElems; ++i) {
      // Match the standalone BF16 output-projection bias addition.
      v[i] = static_cast<float>(static_cast<bf16_t>(v[i] + static_cast<float>(bv[i])));
    }
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
  AlignedVector<fp8_e4m3_t, kVecElems> qv;
  scv.load(static_cast<const bf16_t*>(params.scale) + elem_offset);
  shv.load(static_cast<const bf16_t*>(params.shift) + elem_offset);

  float input_scale_inv = 0.0f;
  if constexpr (kQuantizeFp8) {
    input_scale_inv = triton_scale_reciprocal(*static_cast<const float*>(params.input_scale));
  }

#pragma unroll
  for (int i = 0; i < kVecElems; ++i) {
    const float norm = static_cast<float>(static_cast<bf16_t>((v[i] - mean) * factor));
    const bf16_t rounded = static_cast<bf16_t>(norm * (1.0f + static_cast<float>(scv[i])) + static_cast<float>(shv[i]));
    yv[i] = rounded;
    if constexpr (kQuantizeFp8) {
      const float scaled = static_cast<float>(rounded) * input_scale_inv;
      const float clamped =
          math::min(math::max(scaled, -DTypeTrait<fp8_e4m3_t>::kFloatMax), DTypeTrait<fp8_e4m3_t>::kFloatMax);
      qv[i] = static_cast<fp8_e4m3_t>(clamped);
    }
  }
  if constexpr (kQuantizeNvfp4) {
#if defined(ENABLE_FP4) && ENABLE_FP4
    tensorrt_llm::kernels::PackedVec<__nv_bfloat16, kVecElems> quant_vec;
    auto* quant_values = reinterpret_cast<__nv_bfloat16*>(&quant_vec);
#pragma unroll
    for (int i = 0; i < kVecElems; ++i) {
      quant_values[i] = static_cast<__nv_bfloat16>(yv[i]);
    }

    auto* scales = static_cast<uint8_t*>(params.quant_scales);
    const int64_t scale_offset = tensorrt_llm::kernels::get_sf_out_offset_128x4(row, tid, kThreads);
    const float global_scale = *static_cast<const float*>(params.global_scale);
    const uint64_t packed = tensorrt_llm::kernels::cvt_warp_fp16_to_fp4<__nv_bfloat16, kVecElems, kVecElems, false>(
        quant_vec, global_scale, scales + scale_offset);
    static_cast<uint64_t*>(params.quantized)[int64_t(row) * kThreads + tid] = packed;
#else
    static_assert(!kQuantizeNvfp4);
#endif
  } else {
    yv.store(static_cast<bf16_t*>(params.y) + row_offset + elem_offset);
    if constexpr (kQuantizeFp8) {
      qv.store(static_cast<fp8_e4m3_t*>(params.quantized) + row_offset + elem_offset);
    }
  }
}

__global__ void bias_mul_add_kernel(const NormScaleShiftParams __grid_constant__ params) {
  using namespace device;
  using Vec = AlignedVector<bf16_t, kVecElems>;

  const int row_offset = blockIdx.x * kHidden;
  const int elem_offset = threadIdx.x * kVecElems;

  Vec xv;
  Vec bv;
  Vec gv;
  Vec rv;
  Vec yv;
  xv.load(static_cast<const bf16_t*>(params.x) + row_offset + elem_offset);
  bv.load(static_cast<const bf16_t*>(params.input_bias) + elem_offset);
  gv.load(static_cast<const bf16_t*>(params.gate) + elem_offset);
  rv.load(static_cast<const bf16_t*>(params.residual) + row_offset + elem_offset);

#pragma unroll
  for (int i = 0; i < kVecElems; ++i) {
    const bf16_t biased = static_cast<bf16_t>(static_cast<float>(xv[i]) + static_cast<float>(bv[i]));
    yv[i] = __hfma(biased, gv[i], rv[i]);
  }
  yv.store(static_cast<bf16_t*>(params.y) + row_offset + elem_offset);
}

inline uint32_t verify_nss_geometry(host::SymbolicSize& num_rows) {
  using namespace host;
  RuntimeCheck(num_rows.unwrap() > 0, "num_rows must be positive");
  RuntimeCheck(num_rows.unwrap() <= int64_t(UINT32_MAX), "num_rows out of range");
  return static_cast<uint32_t>(num_rows.unwrap());
}

struct NormScaleShiftKernel {
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

    const uint32_t grid = verify_nss_geometry(N);
    const auto params = NormScaleShiftParams{
        .y = y.data_ptr(),
        .res_out = nullptr,
        .quantized = nullptr,
        .quant_scales = nullptr,
        .x = x.data_ptr(),
        .input_bias = nullptr,
        .residual = nullptr,
        .gate = nullptr,
        .scale = scale.data_ptr(),
        .shift = shift.data_ptr(),
        .input_scale = nullptr,
        .global_scale = nullptr,
        .num_rows = grid,
        .eps = static_cast<float>(eps),
    };
    LaunchKernel(grid, kThreads, device.unwrap())(norm_scale_shift_kernel<false>, params);
  }
};

struct ScaleResidualNormScaleShiftKernel {
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

    const uint32_t grid = verify_nss_geometry(N);
    const auto params = NormScaleShiftParams{
        .y = y.data_ptr(),
        .res_out = res_out.data_ptr(),
        .quantized = nullptr,
        .quant_scales = nullptr,
        .x = x.data_ptr(),
        .input_bias = nullptr,
        .residual = residual.data_ptr(),
        .gate = gate.data_ptr(),
        .scale = scale.data_ptr(),
        .shift = shift.data_ptr(),
        .input_scale = nullptr,
        .global_scale = nullptr,
        .num_rows = grid,
        .eps = static_cast<float>(eps),
    };
    LaunchKernel(grid, kThreads, device.unwrap())(norm_scale_shift_kernel<true>, params);
  }
};

/** \brief Fuse Qwen LayerNorm/modulation with static E4M3 activation quantization. */
struct NormScaleShiftFp8Kernel {
  static void
  run(tvm::ffi::TensorView y,
      tvm::ffi::TensorView quantized,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView scale,
      tvm::ffi::TensorView shift,
      tvm::ffi::TensorView input_scale,
      double eps) {
    using namespace host;
    auto N = SymbolicSize{"num_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, kHidden}).with_dtype<bf16_t>().with_device(device).verify(x).verify(y);
    TensorMatcher({N, kHidden}).with_dtype<fp8_e4m3_t>().with_device(device).verify(quantized);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device).verify(scale).verify(shift);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(input_scale);

    const uint32_t grid = verify_nss_geometry(N);
    const auto params = NormScaleShiftParams{
        .y = y.data_ptr(),
        .res_out = nullptr,
        .quantized = quantized.data_ptr(),
        .quant_scales = nullptr,
        .x = x.data_ptr(),
        .input_bias = nullptr,
        .residual = nullptr,
        .gate = nullptr,
        .scale = scale.data_ptr(),
        .shift = shift.data_ptr(),
        .input_scale = input_scale.data_ptr(),
        .global_scale = nullptr,
        .num_rows = grid,
        .eps = static_cast<float>(eps),
    };
    LaunchKernel(grid, kThreads, device.unwrap())(norm_scale_shift_kernel<false, false, true>, params);
  }
};

/** \brief Fuse Qwen residual LayerNorm/modulation with static E4M3 activation quantization. */
struct ScaleResidualNormScaleShiftFp8Kernel {
  static void
  run(tvm::ffi::TensorView y,
      tvm::ffi::TensorView quantized,
      tvm::ffi::TensorView res_out,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView gate,
      tvm::ffi::TensorView scale,
      tvm::ffi::TensorView shift,
      tvm::ffi::TensorView input_scale,
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
    TensorMatcher({N, kHidden}).with_dtype<fp8_e4m3_t>().with_device(device).verify(quantized);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device).verify(gate).verify(scale).verify(shift);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(input_scale);

    const uint32_t grid = verify_nss_geometry(N);
    const auto params = NormScaleShiftParams{
        .y = y.data_ptr(),
        .res_out = res_out.data_ptr(),
        .quantized = quantized.data_ptr(),
        .quant_scales = nullptr,
        .x = x.data_ptr(),
        .input_bias = nullptr,
        .residual = residual.data_ptr(),
        .gate = gate.data_ptr(),
        .scale = scale.data_ptr(),
        .shift = shift.data_ptr(),
        .input_scale = input_scale.data_ptr(),
        .global_scale = nullptr,
        .num_rows = grid,
        .eps = static_cast<float>(eps),
    };
    LaunchKernel(grid, kThreads, device.unwrap())(norm_scale_shift_kernel<true, false, true>, params);
  }
};

struct BiasScaleResidualNormScaleShiftKernel {
  static void
  run(tvm::ffi::TensorView y,
      tvm::ffi::TensorView res_out,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView input_bias,
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
    TensorMatcher({kHidden})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(input_bias)
        .verify(gate)
        .verify(scale)
        .verify(shift);

    const uint32_t grid = verify_nss_geometry(N);
    const auto params = NormScaleShiftParams{
        .y = y.data_ptr(),
        .res_out = res_out.data_ptr(),
        .quantized = nullptr,
        .quant_scales = nullptr,
        .x = x.data_ptr(),
        .input_bias = input_bias.data_ptr(),
        .residual = residual.data_ptr(),
        .gate = gate.data_ptr(),
        .scale = scale.data_ptr(),
        .shift = shift.data_ptr(),
        .input_scale = nullptr,
        .global_scale = nullptr,
        .num_rows = grid,
        .eps = static_cast<float>(eps),
    };
    LaunchKernel(grid, kThreads, device.unwrap())(norm_scale_shift_kernel<true, true>, params);
  }
};

struct BiasMulAddKernel {
  static void
  run(tvm::ffi::TensorView y,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView input_bias,
      tvm::ffi::TensorView gate,
      tvm::ffi::TensorView residual) {
    using namespace host;
    auto N = SymbolicSize{"num_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, kHidden}).with_dtype<bf16_t>().with_device(device).verify(x).verify(residual).verify(y);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device).verify(input_bias).verify(gate);

    const uint32_t grid = verify_nss_geometry(N);
    const auto params = NormScaleShiftParams{
        .y = y.data_ptr(),
        .res_out = nullptr,
        .quantized = nullptr,
        .quant_scales = nullptr,
        .x = x.data_ptr(),
        .input_bias = input_bias.data_ptr(),
        .residual = residual.data_ptr(),
        .gate = gate.data_ptr(),
        .scale = nullptr,
        .shift = nullptr,
        .input_scale = nullptr,
        .global_scale = nullptr,
        .num_rows = grid,
        .eps = 0.0f,
    };
    LaunchKernel(grid, kThreads, device.unwrap())(bias_mul_add_kernel, params);
  }
};

#if defined(ENABLE_FP4) && ENABLE_FP4
struct ScaleResidualNormScaleShiftNvfp4Kernel {
  static void
  run(tvm::ffi::TensorView quantized,
      tvm::ffi::TensorView quant_scales,
      tvm::ffi::TensorView res_out,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView input_bias,
      tvm::ffi::TensorView gate,
      tvm::ffi::TensorView scale,
      tvm::ffi::TensorView shift,
      tvm::ffi::TensorView global_scale,
      double eps) {
    using namespace host;
    auto N = SymbolicSize{"num_rows"};
    auto NP = SymbolicSize{"num_rows_padded"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, kHidden}).with_dtype<bf16_t>().with_device(device).verify(x).verify(residual).verify(res_out);
    TensorMatcher({kHidden})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(input_bias)
        .verify(gate)
        .verify(scale)
        .verify(shift);
    TensorMatcher({N, kHidden / 2}).with_dtype<uint8_t>().with_device(device).verify(quantized);
    TensorMatcher({NP, kThreads}).with_dtype<uint8_t>().with_device(device).verify(quant_scales);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(global_scale);

    const uint32_t num_rows = verify_nss_geometry(N);
    const uint32_t num_rows_padded = div_ceil(num_rows, uint32_t(128)) * 128;
    RuntimeCheck(NP.unwrap() == num_rows_padded, "quant scale rows must be padded to 128");
    const auto params = NormScaleShiftParams{
        .y = nullptr,
        .res_out = res_out.data_ptr(),
        .quantized = quantized.data_ptr(),
        .quant_scales = quant_scales.data_ptr(),
        .x = x.data_ptr(),
        .input_bias = input_bias.data_ptr(),
        .residual = residual.data_ptr(),
        .gate = gate.data_ptr(),
        .scale = scale.data_ptr(),
        .shift = shift.data_ptr(),
        .input_scale = nullptr,
        .global_scale = global_scale.data_ptr(),
        .num_rows = num_rows,
        .eps = static_cast<float>(eps),
    };
    LaunchKernel(num_rows_padded, kThreads, device.unwrap())(norm_scale_shift_kernel<true, true, false, true>, params);
  }
};
#endif

}  // namespace norm_scale_shift

}  // namespace sglang
