#pragma once

#include <sgl_kernel/runtime.cuh>

#include "silu_and_mul_masked_post_quant.cuh"
#include <algorithm>

namespace sglang {

struct SiluMulQuantPsumParams {
  const bf16_t* __restrict__ input;
  fp8_e4m3_t* __restrict__ output;
  float* __restrict__ output_scale;
  const int32_t* __restrict__ expert_psum;
  float swiglu_limit;  // only read when kApplySwigluLimit=true
  int64_t hidden_dim;
  uint32_t num_tokens;
  uint32_t num_experts;
  uint32_t expert_alignment;
  uint32_t scale_row_stride_int32;  // only used when kTransposed=true
};

template <bool kScaleUE8M0, bool kTransposed, bool kUsePDL, bool kApplySwigluLimit>
__global__
__launch_bounds__(1024, 2) void silu_mul_quant_psum_kernel(const SiluMulQuantPsumParams __grid_constant__ params) {
  using namespace device;

  constexpr uint32_t kGroupSize = 128u;
  constexpr uint32_t kWorkThreads = 16u;
  using InputVec = AlignedVector<bf16x2_t, 4>;
  using OutputVec = AlignedVector<fp8x2_e4m3_t, 4>;
  static_assert(8 * kWorkThreads == kGroupSize, "Invalid tiling");
  static_assert(!(kTransposed && !kScaleUE8M0), "transposed layout only supports ue8m0");

  PDLWaitPrimary<kUsePDL>();

  extern __shared__ int32_t shared_offsets[];
  int32_t* compact_offsets = shared_offsets;
  int32_t* physical_starts = shared_offsets + params.num_experts + 1;

  if (threadIdx.x == 0) {
    int32_t compact_offset = 0;
    int32_t previous_end = 0;
    compact_offsets[0] = 0;
    for (uint32_t expert = 0; expert < params.num_experts; ++expert) {
      const int32_t expert_end = params.expert_psum[expert];
      const int32_t expert_start = expert == 0 ? 0
                                               : (previous_end + static_cast<int32_t>(params.expert_alignment) - 1) /
                                                     static_cast<int32_t>(params.expert_alignment) *
                                                     static_cast<int32_t>(params.expert_alignment);
      physical_starts[expert] = expert_start;
      compact_offset += expert_end - expert_start;
      compact_offsets[expert + 1] = compact_offset;
      previous_end = expert_end;
    }
  }
  __syncthreads();

  const int32_t total_tokens = compact_offsets[params.num_experts];
  const auto work_id = threadIdx.x / kWorkThreads;
  const auto num_groups = static_cast<uint32_t>(params.hidden_dim) / kGroupSize;

  for (int32_t compact_token = static_cast<int32_t>(blockIdx.x); compact_token < total_tokens;
       compact_token += static_cast<int32_t>(gridDim.x)) {
    uint32_t lower_expert = 0;
    uint32_t upper_expert = params.num_experts;
    while (lower_expert + 1 < upper_expert) {
      const uint32_t middle_expert = (lower_expert + upper_expert) / 2;
      if (compact_offsets[middle_expert] <= compact_token) {
        lower_expert = middle_expert;
      } else {
        upper_expert = middle_expert;
      }
    }

    const int32_t token_id = physical_starts[lower_expert] + compact_token - compact_offsets[lower_expert];
    const auto input = params.input + static_cast<int64_t>(token_id) * params.hidden_dim * 2;
    const auto output = params.output + static_cast<int64_t>(token_id) * params.hidden_dim;
    [[maybe_unused]] const auto output_scale = [&] {
      if constexpr (kTransposed) {
        const auto base = reinterpret_cast<uint8_t*>(params.output_scale);
        return base + (work_id / 4u) * (params.scale_row_stride_int32 * 4u) + token_id * 4u + (work_id % 4u);
      } else {
        return params.output_scale + static_cast<int64_t>(token_id) * num_groups + work_id;
      }
    }();

    InputVec gate_vec, up_vec;
    gate_vec.load(input, threadIdx.x);
    up_vec.load(input, threadIdx.x + blockDim.x);

    float local_max = 0.0f;
    float results[8];

#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const auto [x, y] = silu_and_mul<kApplySwigluLimit>(gate_vec[i], up_vec[i], params.swiglu_limit);
      results[2 * i + 0] = x;
      results[2 * i + 1] = y;
      local_max = fmaxf(local_max, fmaxf(fabsf(x), fabsf(y)));
    }

    local_max = warp::reduce_max<kWorkThreads>(local_max);

    const float absmax = fmaxf(local_max, 1e-10f);
    float scale;
    uint32_t ue8m0_exp;

    if constexpr (kScaleUE8M0) {
      const float raw_scale = absmax / math::FP8_E4M3_MAX;
      ue8m0_exp = cast_to_ue8m0(raw_scale);
      scale = __uint_as_float(ue8m0_exp << 23);
    } else {
      scale = absmax / math::FP8_E4M3_MAX;
    }
    const auto inv_scale = 1.0f / scale;

    OutputVec out_vec;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const float scaled_val0 = results[2 * i + 0] * inv_scale;
      const float scaled_val1 = results[2 * i + 1] * inv_scale;
      out_vec[i] = pack_fp8(scaled_val0, scaled_val1);
    }

    out_vec.store(output, threadIdx.x);
    if constexpr (kTransposed) {
      *output_scale = ue8m0_exp;
    } else {
      *output_scale = scale;
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kGroupSize, bool kScaleUE8M0, bool kUsePDL, bool kApplySwigluLimit>
struct SiluAndMulPsumPostQuantKernel {
  static_assert(kGroupSize == 128);
  static constexpr auto kernel_normal = silu_mul_quant_psum_kernel<kScaleUE8M0, false, kUsePDL, kApplySwigluLimit>;
  static constexpr auto kernel_transposed = silu_mul_quant_psum_kernel<true, true, kUsePDL, kApplySwigluLimit>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView output_scale,
      const tvm::ffi::TensorView expert_psum,
      const int64_t expert_alignment,
      const bool transposed,
      const double swiglu_limit) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_dim x 2"};
    auto N = SymbolicSize{"hidden_dim"};
    auto G = SymbolicSize{"num_groups"};
    auto E = SymbolicSize{"num_experts"};
    device.set_options<kDLCUDA>();
    uint32_t scale_row_stride_int32 = 0;

    TensorMatcher({M, D}).with_dtype<bf16_t>().with_device(device).verify(input);
    TensorMatcher({M, N}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output);

    const auto hidden_dim = N.unwrap();
    RuntimeCheck(D.unwrap() == 2 * hidden_dim, "invalid dimension");
    RuntimeCheck(hidden_dim % kGroupSize == 0);
    const auto num_groups = static_cast<uint32_t>(hidden_dim / kGroupSize);

    if (!transposed) {
      G.set_value(num_groups);
      TensorMatcher({M, G}).with_dtype<fp32_t>().with_device(device).verify(output_scale);
    } else {
      RuntimeCheck(kScaleUE8M0, "transposed layout only supports scale_ue8m0=true");
      auto G_ = SymbolicSize{"ceil(G / 4)"};
      G_.set_value((num_groups + 3) / 4);
      auto M_pad = SymbolicSize{"M padded"};
      TensorMatcher({M, G_})
          .with_strides({int64_t{1}, M_pad})
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(output_scale);
      scale_row_stride_int32 = static_cast<uint32_t>(M_pad.unwrap());
    }
    TensorMatcher({E}).with_dtype<int32_t>().with_device(device).verify(expert_psum);

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    const auto num_experts = static_cast<uint32_t>(E.unwrap());

    RuntimeCheck(num_experts > 0, "expert_psum must not be empty");
    RuntimeCheck(expert_alignment > 0, "expert_alignment must be positive");
    if (num_tokens == 0) {
      return;
    }

    const auto params = SiluMulQuantPsumParams{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .output = static_cast<fp8_e4m3_t*>(output.data_ptr()),
        .output_scale = static_cast<float*>(output_scale.data_ptr()),
        .expert_psum = static_cast<const int32_t*>(expert_psum.data_ptr()),
        .swiglu_limit = static_cast<float>(swiglu_limit),
        .hidden_dim = hidden_dim,
        .num_tokens = num_tokens,
        .num_experts = num_experts,
        .expert_alignment = static_cast<uint32_t>(expert_alignment),
        .scale_row_stride_int32 = scale_row_stride_int32,
    };

    const auto num_threads = static_cast<uint32_t>(hidden_dim / 8);
    RuntimeCheck(num_threads % device::kWarpThreads == 0);
    RuntimeCheck(num_threads <= 1024, "hidden_dim is too large for one block per worker");

    const auto kernel = transposed ? kernel_transposed : kernel_normal;
    const auto shared_memory = static_cast<size_t>(2 * num_experts + 1) * sizeof(int32_t);
    const auto runtime_device = device.unwrap();
    const auto blocks_per_sm = runtime::get_blocks_per_sm(kernel, static_cast<int32_t>(num_threads), shared_memory);
    const auto num_workers = std::min(num_tokens, runtime::get_sm_count(runtime_device.device_id) * blocks_per_sm);

    LaunchKernel(num_workers, num_threads, runtime_device, shared_memory).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
