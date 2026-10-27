// FLUX.2 single-block [attention | MLP] concatenation + NVFP4 quantization.

#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#ifndef FLT_MAX
#define FLT_MAX __FLT_MAX__
#endif
#include <tensorrt_llm/kernels/quantization_utils.cuh>

#include <cstdint>

namespace sglang {

namespace flux2_token_cat_nvfp4 {

constexpr int kAttentionHidden = 6144;
constexpr int kMlpHidden = 18432;
constexpr int kOutputHidden = kAttentionHidden + kMlpHidden;
constexpr int kGroupSize = 16;
constexpr int kScaleColumns = kOutputHidden / kGroupSize;
constexpr int kPackedColumns = kOutputHidden / 2;
constexpr int kAttentionGroups = kAttentionHidden / kGroupSize;
constexpr int kThreads = 256;
constexpr int kColumnTiles = (kScaleColumns + kThreads - 1) / kThreads;
constexpr int kMaxRows = 65408;

static_assert(kScaleColumns == 1536);
static_assert(kColumnTiles == 6);

struct Params {
  void* quantized;
  void* quant_scales;
  const void* attention;
  const void* mlp;
  const void* global_scale;
  uint32_t num_rows;
};

__global__ void kernel(const Params __grid_constant__ params) {
  using namespace device;
  using Vec = AlignedVector<bf16_t, kGroupSize>;

  const int row = blockIdx.y;
  const int group = blockIdx.x * kThreads + threadIdx.x;
  if (group >= kScaleColumns) {
    return;
  }

  const int64_t scale_offset = tensorrt_llm::kernels::get_sf_out_offset_128x4(row, group, kScaleColumns);
  auto* quant_scales = static_cast<uint8_t*>(params.quant_scales);
  if (row >= params.num_rows) {
    quant_scales[scale_offset] = 0;
    return;
  }

  Vec input;
  if (group < kAttentionGroups) {
    input.load(static_cast<const bf16_t*>(params.attention) + int64_t(row) * kAttentionHidden + group * kGroupSize);
  } else {
    const int mlp_group = group - kAttentionGroups;
    input.load(static_cast<const bf16_t*>(params.mlp) + int64_t(row) * kMlpHidden + mlp_group * kGroupSize);
  }

  tensorrt_llm::kernels::PackedVec<__nv_bfloat16, kGroupSize> quant_vec;
  auto* quant_values = reinterpret_cast<__nv_bfloat16*>(&quant_vec);
#pragma unroll
  for (int element = 0; element < kGroupSize; ++element) {
    quant_values[element] = static_cast<__nv_bfloat16>(input[element]);
  }

  const float global_scale = *static_cast<const float*>(params.global_scale);
  const uint64_t packed = tensorrt_llm::kernels::cvt_warp_fp16_to_fp4<__nv_bfloat16, kGroupSize, kGroupSize, false>(
      quant_vec, global_scale, quant_scales + scale_offset);
  static_cast<uint64_t*>(params.quantized)[int64_t(row) * kScaleColumns + group] = packed;
}

struct Kernel {
  static void
  run(tvm::ffi::TensorView quantized,
      tvm::ffi::TensorView quant_scales,
      tvm::ffi::TensorView attention,
      tvm::ffi::TensorView mlp,
      tvm::ffi::TensorView global_scale) {
    using namespace host;
    auto rows = SymbolicSize{"rows"};
    auto padded_rows = SymbolicSize{"padded_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({rows, kAttentionHidden}).with_dtype<bf16_t>().with_device(device).verify(attention);
    TensorMatcher({rows, kMlpHidden}).with_dtype<bf16_t>().with_device(device).verify(mlp);
    TensorMatcher({rows, kPackedColumns}).with_dtype<uint8_t>().with_device(device).verify(quantized);
    TensorMatcher({padded_rows, kScaleColumns}).with_dtype<uint8_t>().with_device(device).verify(quant_scales);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(global_scale);
    RuntimeCheck(rows.unwrap() > 0, "rows must be positive");
    RuntimeCheck(rows.unwrap() <= kMaxRows, "rows exceed the CUDA grid.y limit");
    const uint32_t row_count = static_cast<uint32_t>(rows.unwrap());
    const uint32_t padded_row_count = div_ceil(row_count, uint32_t(128)) * 128;
    RuntimeCheck(padded_rows.unwrap() == padded_row_count, "quant scale rows must be padded to 128");

    const auto params = Params{
        .quantized = quantized.data_ptr(),
        .quant_scales = quant_scales.data_ptr(),
        .attention = attention.data_ptr(),
        .mlp = mlp.data_ptr(),
        .global_scale = global_scale.data_ptr(),
        .num_rows = row_count,
    };
    LaunchKernel(dim3(kColumnTiles, padded_row_count), kThreads, device.unwrap())(kernel, params);
  }
};

}  // namespace flux2_token_cat_nvfp4

}  // namespace sglang
