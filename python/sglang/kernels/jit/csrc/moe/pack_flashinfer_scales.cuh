#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher and symbolic metadata
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/math.cuh>     // For FP8_E4M3_MAX
#include <sgl_kernel/runtime.cuh>  // For get_sm_count
#include <sgl_kernel/type.cuh>     // For packed CUDA types
#include <sgl_kernel/utils.cuh>    // For LaunchKernel and FP8 types
#include <sgl_kernel/vec.cuh>      // For AlignedVector
#include <sgl_kernel/warp.cuh>     // For subgroup reductions

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>  // For pack_fp8

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

constexpr int kPackFlashInferScalesThreads = 256;

SGL_DEVICE int64_t
get_flashinfer_source_row(const int32_t* __restrict__ expert_offsets, int64_t padded_row, int num_experts) {
  int lo = 0;
  int hi = num_experts;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    const int64_t next_padded_start = (static_cast<int64_t>(expert_offsets[mid + 1]) + 3LL * (mid + 1)) / 4 * 4;
    if (padded_row < next_padded_start) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }

  if (lo < num_experts) {
    const int64_t expert_start = expert_offsets[lo];
    const int64_t padded_expert_start = (expert_start + 3LL * lo) / 4 * 4;
    const int64_t source_row = expert_start + padded_row - padded_expert_start;
    if (source_row >= expert_start && source_row < expert_offsets[lo + 1]) {
      return source_row;
    }
  }
  return -1;
}

SGL_DEVICE float get_packed_scale(
    const float* __restrict__ input,
    const int32_t* __restrict__ expert_offsets,
    int64_t k_blocks,
    int64_t k_block,
    int64_t padded_row,
    int num_experts,
    const int32_t* __restrict__ row_map = nullptr) {
  const int64_t source_row = get_flashinfer_source_row(expert_offsets, padded_row, num_experts);
  if (source_row >= 0) {
    const int64_t input_row = row_map == nullptr ? source_row : row_map[source_row];
    return input[input_row * k_blocks + k_block];
  }
  return 0.0f;
}

__global__ void pack_flashinfer_moe_scales_kernel(
    const float* __restrict__ input,
    const int32_t* __restrict__ expert_offsets,
    float* __restrict__ output,
    int64_t k_blocks,
    int64_t padded_rows,
    int num_experts) {
  const int64_t active_rows = expert_offsets[num_experts];
  const int64_t active_padded_rows = ((active_rows + 3LL * num_experts) / 4) * 4;
  const int64_t total_elements = active_padded_rows * k_blocks;
  for (int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; linear_idx < total_elements;
       linear_idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t k_block = linear_idx / active_padded_rows;
    const int64_t padded_row = linear_idx - k_block * active_padded_rows;
    output[k_block * padded_rows + padded_row] =
        get_packed_scale(input, expert_offsets, k_blocks, k_block, padded_row, num_experts);
  }
}

__global__ void shuffle_rows_and_pack_flashinfer_moe_scales_kernel(
    const fp8_e4m3_t* __restrict__ input,
    const float* __restrict__ input_scales,
    const int32_t* __restrict__ row_map,
    const int32_t* __restrict__ expert_offsets,
    fp8_e4m3_t* __restrict__ output,
    float* __restrict__ output_scales,
    int64_t output_rows,
    int64_t hidden_size,
    int64_t k_blocks,
    int64_t padded_rows,
    int num_experts) {
  const int64_t active_rows = expert_offsets[num_experts];
  const int64_t active_padded_rows = ((active_rows + 3LL * num_experts) / 4) * 4;
  const int64_t output_elements = active_rows * hidden_size;
  const int64_t scale_elements = active_padded_rows * k_blocks;

  for (int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; linear_idx < output_elements;
       linear_idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t output_row = linear_idx / hidden_size;
    const int64_t col = linear_idx - output_row * hidden_size;
    output[linear_idx] = input[static_cast<int64_t>(row_map[output_row]) * hidden_size + col];
  }

  for (int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; linear_idx < scale_elements;
       linear_idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t k_block = linear_idx / active_padded_rows;
    const int64_t padded_row = linear_idx - k_block * active_padded_rows;
    output_scales[k_block * padded_rows + padded_row] =
        get_packed_scale(input_scales, expert_offsets, k_blocks, k_block, padded_row, num_experts, row_map);
  }
}

struct SiluAndMulQuantPackFlashInferMoeParams {
  const bf16_t* __restrict__ input;
  const int32_t* __restrict__ expert_offsets;
  fp8_e4m3_t* __restrict__ output;
  float* __restrict__ output_scales;
  float swiglu_limit;
  int64_t hidden_size;
  int64_t padded_rows;
  int num_experts;
};

__global__ __launch_bounds__(1024, 2) void silu_and_mul_quant_pack_flashinfer_moe_kernel(
    const SiluAndMulQuantPackFlashInferMoeParams __grid_constant__ params) {
  using namespace device;
  using deepseek_v4::fp8::pack_fp8;

  constexpr int kGroupSize = 128;
  constexpr int kValuesPerThread = 8;
  constexpr int kThreadsPerGroup = kGroupSize / kValuesPerThread;
  using InputVec = AlignedVector<bf16x2_t, kValuesPerThread / 2>;
  using OutputVec = AlignedVector<fp8x2_e4m3_t, kValuesPerThread / 2>;

  const int64_t active_rows = params.expert_offsets[params.num_experts];
  const int64_t active_padded_rows = ((active_rows + 3LL * params.num_experts) / 4) * 4;
  const int64_t num_groups = params.hidden_size / kGroupSize;
  const int64_t group = threadIdx.x / kThreadsPerGroup;
  const int group_lane = threadIdx.x % kThreadsPerGroup;

  for (int64_t padded_row = blockIdx.x; padded_row < active_padded_rows; padded_row += gridDim.x) {
    const int64_t source_row = get_flashinfer_source_row(params.expert_offsets, padded_row, params.num_experts);
    if (source_row < 0) {
      if (group_lane == 0) {
        params.output_scales[group * params.padded_rows + padded_row] = 0.0f;
      }
      continue;
    }

    const auto input = params.input + source_row * params.hidden_size * 2;
    const auto output = params.output + source_row * params.hidden_size;
    InputVec up_vec, gate_vec;
    up_vec.load(input, threadIdx.x);
    gate_vec.load(input, threadIdx.x + blockDim.x);

    float local_max = 0.0f;
    float results[kValuesPerThread];
#pragma unroll
    for (int i = 0; i < kValuesPerThread / 2; ++i) {
      auto gate = __hmin2(gate_vec[i], bf16x2_t{params.swiglu_limit, params.swiglu_limit});
      auto up = __hmax2(up_vec[i], bf16x2_t{-params.swiglu_limit, -params.swiglu_limit});
      up = __hmin2(up, bf16x2_t{params.swiglu_limit, params.swiglu_limit});
      const auto [gate0, gate1] = cast<fp32x2_t>(gate);
      const auto [up0, up1] = cast<fp32x2_t>(up);
      // Match the Triton runner's BF16 intermediate semantics before the
      // second FP8 quantization. SiLU and multiplication are evaluated in
      // FP32, then rounded to BF16 before computing the FP8 group scale.
      const float value0 = gate0 / (1.0f + __expf(-gate0)) * up0;
      const float value1 = gate1 / (1.0f + __expf(-gate1)) * up1;
      const auto [rounded0, rounded1] = cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{value0, value1}));
      results[2 * i] = rounded0;
      results[2 * i + 1] = rounded1;
      local_max = fmaxf(local_max, fmaxf(fabsf(rounded0), fabsf(rounded1)));
    }

    local_max = warp::reduce_max<kThreadsPerGroup>(local_max);
    const float scale = fmaxf(local_max, 1e-10f) / math::FP8_E4M3_MAX;
    const float inv_scale = 1.0f / scale;
    OutputVec output_vec;
#pragma unroll
    for (int i = 0; i < kValuesPerThread / 2; ++i) {
      output_vec[i] = pack_fp8(results[2 * i] * inv_scale, results[2 * i + 1] * inv_scale);
    }
    output_vec.store(output, threadIdx.x);
    if (group_lane == 0) {
      params.output_scales[group * params.padded_rows + padded_row] = scale;
    }
  }
}

struct PackFlashInferMoeScalesKernel {
  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView expert_offsets, tvm::ffi::TensorView output) {
    using namespace host;

    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto total_rows = SymbolicSize{"total_rows"};
    auto k_blocks = SymbolicSize{"k_blocks"};
    auto num_offsets = SymbolicSize{"num_offsets"};
    auto padded_rows = SymbolicSize{"padded_rows"};

    TensorMatcher({total_rows, k_blocks})
        .with_strides({k_blocks, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(input);
    TensorMatcher({num_offsets}).with_strides({1}).with_dtype<int32_t>().with_device(device).verify(expert_offsets);
    TensorMatcher({k_blocks, padded_rows})
        .with_strides({padded_rows, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(output);

    const int64_t rows = total_rows.unwrap();
    const int64_t blocks = k_blocks.unwrap();
    const int experts = static_cast<int>(num_offsets.unwrap()) - 1;
    const int64_t padded = padded_rows.unwrap();
    RuntimeCheck(experts > 0, "expert_offsets must contain at least two entries");
    RuntimeCheck(
        padded == ((rows + 3 * experts) / 4) * 4, "output padded-row dimension does not match FlashInfer layout");

    const int64_t elements = padded * blocks;
    if (elements == 0) return;
    const int64_t requested_blocks = (elements + kPackFlashInferScalesThreads - 1) / kPackFlashInferScalesThreads;
    const int64_t max_blocks = host::runtime::get_sm_count(device.unwrap().device_id) * 8LL;
    const auto grid = dim3(requested_blocks < max_blocks ? requested_blocks : max_blocks);
    LaunchKernel(grid, dim3(kPackFlashInferScalesThreads), device.unwrap())(
        pack_flashinfer_moe_scales_kernel,
        static_cast<const float*>(input.data_ptr()),
        static_cast<const int32_t*>(expert_offsets.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        blocks,
        padded,
        experts);
  }
};

struct ShuffleRowsAndPackFlashInferMoeScalesKernel {
  static void
  run(tvm::ffi::TensorView input,
      tvm::ffi::TensorView input_scales,
      tvm::ffi::TensorView row_map,
      tvm::ffi::TensorView expert_offsets,
      tvm::ffi::TensorView output,
      tvm::ffi::TensorView output_scales) {
    using namespace host;

    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto input_rows = SymbolicSize{"input_rows"};
    auto output_rows = SymbolicSize{"output_rows"};
    auto hidden_size = SymbolicSize{"hidden_size"};
    auto k_blocks = SymbolicSize{"k_blocks"};
    auto num_offsets = SymbolicSize{"num_offsets"};
    auto padded_rows = SymbolicSize{"padded_rows"};

    TensorMatcher({input_rows, hidden_size})
        .with_strides({hidden_size, 1})
        .with_dtype<fp8_e4m3_t>()
        .with_device(device)
        .verify(input);
    TensorMatcher({input_rows, k_blocks})
        .with_strides({k_blocks, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(input_scales);
    TensorMatcher({output_rows}).with_strides({1}).with_dtype<int32_t>().with_device(device).verify(row_map);
    TensorMatcher({num_offsets}).with_strides({1}).with_dtype<int32_t>().with_device(device).verify(expert_offsets);
    TensorMatcher({output_rows, hidden_size})
        .with_strides({hidden_size, 1})
        .with_dtype<fp8_e4m3_t>()
        .with_device(device)
        .verify(output);
    TensorMatcher({k_blocks, padded_rows})
        .with_strides({padded_rows, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(output_scales);

    const int64_t rows = output_rows.unwrap();
    const int64_t hidden = hidden_size.unwrap();
    const int64_t blocks = k_blocks.unwrap();
    const int experts = static_cast<int>(num_offsets.unwrap()) - 1;
    const int64_t padded = padded_rows.unwrap();
    RuntimeCheck(experts > 0, "expert_offsets must contain at least two entries");
    RuntimeCheck(hidden == blocks * 128, "input scale width must equal hidden_size / 128");
    RuntimeCheck(
        padded == ((rows + 3 * experts) / 4) * 4, "output padded-row dimension does not match FlashInfer layout");

    const int64_t output_elements = rows * hidden;
    const int64_t scale_elements = padded * blocks;
    const int64_t elements = output_elements > scale_elements ? output_elements : scale_elements;
    if (elements == 0) return;
    const int64_t requested_blocks = (elements + kPackFlashInferScalesThreads - 1) / kPackFlashInferScalesThreads;
    const int64_t max_blocks = host::runtime::get_sm_count(device.unwrap().device_id) * 8LL;
    const auto grid = dim3(requested_blocks < max_blocks ? requested_blocks : max_blocks);
    LaunchKernel(grid, dim3(kPackFlashInferScalesThreads), device.unwrap())(
        shuffle_rows_and_pack_flashinfer_moe_scales_kernel,
        static_cast<const fp8_e4m3_t*>(input.data_ptr()),
        static_cast<const float*>(input_scales.data_ptr()),
        static_cast<const int32_t*>(row_map.data_ptr()),
        static_cast<const int32_t*>(expert_offsets.data_ptr()),
        static_cast<fp8_e4m3_t*>(output.data_ptr()),
        static_cast<float*>(output_scales.data_ptr()),
        rows,
        hidden,
        blocks,
        padded,
        experts);
  }
};

struct SiluAndMulQuantPackFlashInferMoeKernel {
  static void
  run(tvm::ffi::TensorView input,
      tvm::ffi::TensorView expert_offsets,
      tvm::ffi::TensorView output,
      tvm::ffi::TensorView output_scales,
      double swiglu_limit) {
    using namespace host;

    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto rows = SymbolicSize{"rows"};
    auto gate_up_size = SymbolicSize{"gate_up_size"};
    auto hidden_size = SymbolicSize{"hidden_size"};
    auto num_offsets = SymbolicSize{"num_offsets"};
    auto num_groups = SymbolicSize{"num_groups"};
    auto padded_rows = SymbolicSize{"padded_rows"};

    TensorMatcher({rows, gate_up_size})
        .with_strides({gate_up_size, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(input);
    TensorMatcher({num_offsets}).with_strides({1}).with_dtype<int32_t>().with_device(device).verify(expert_offsets);
    TensorMatcher({rows, hidden_size})
        .with_strides({hidden_size, 1})
        .with_dtype<fp8_e4m3_t>()
        .with_device(device)
        .verify(output);
    TensorMatcher({num_groups, padded_rows})
        .with_strides({padded_rows, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(output_scales);

    const int experts = static_cast<int>(num_offsets.unwrap()) - 1;
    const int64_t hidden = hidden_size.unwrap();
    const int64_t capacity_rows = rows.unwrap();
    const int64_t padded = padded_rows.unwrap();
    RuntimeCheck(experts > 0, "expert_offsets must contain at least two entries");
    RuntimeCheck(gate_up_size.unwrap() == 2 * hidden, "input must contain [up, gate]");
    RuntimeCheck(hidden % 128 == 0, "hidden size must be divisible by 128");
    RuntimeCheck(hidden / 128 == num_groups.unwrap(), "invalid output scale groups");
    RuntimeCheck(hidden / 8 <= 1024, "hidden size is too large for one block");
    RuntimeCheck(
        padded == ((capacity_rows + 3LL * experts) / 4) * 4,
        "output padded-row dimension does not match FlashInfer layout");

    const int64_t requested_blocks = padded;
    const int64_t max_blocks = host::runtime::get_sm_count(device.unwrap().device_id) * 8LL;
    const auto grid = dim3(requested_blocks < max_blocks ? requested_blocks : max_blocks);
    const auto params = SiluAndMulQuantPackFlashInferMoeParams{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .expert_offsets = static_cast<const int32_t*>(expert_offsets.data_ptr()),
        .output = static_cast<fp8_e4m3_t*>(output.data_ptr()),
        .output_scales = static_cast<float*>(output_scales.data_ptr()),
        .swiglu_limit = static_cast<float>(swiglu_limit),
        .hidden_size = hidden,
        .padded_rows = padded,
        .num_experts = experts,
    };
    LaunchKernel(grid, dim3(hidden / 8), device.unwrap())(silu_and_mul_quant_pack_flashinfer_moe_kernel, params);
  }
};

}  // namespace sglang
