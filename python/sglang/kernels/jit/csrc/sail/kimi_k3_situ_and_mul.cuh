// Kimi K3 SiTU kernels used by the SAIL DeepGEMM EP runner, on top of the shared
// kimi_k3/situ_and_mul.cuh: masked BF16 SiTU and the SiTU + MXFP4 epilogue.

#pragma once

#include "../kimi_k3/situ_and_mul.cuh"
#include <algorithm>

namespace sglang {

// Per-expert grid; expected_m is a scheduling hint, never a token-count limit.
inline dim3 get_situ_masked_grid(uint32_t num_experts, uint32_t expected_m) {
  const auto blocks = host::div_ceil(2048u, num_experts);
  return dim3(expected_m > 0 ? std::min(blocks, expected_m) : blocks, num_experts);
}

struct SituMulVarlenParams {
  const bf16_t* __restrict__ input;
  bf16_t* __restrict__ output;
  const int32_t* __restrict__ masked_m;
  float beta;
  float linear_beta;
  int64_t hidden_dim;
  uint32_t num_tokens;
};

/// Masked BF16 SiTU. Only valid rows are written; padding remains untouched.
template <bool kHasLinearBeta, bool kUsePDL>
__global__ __launch_bounds__(1024, 2) void situ_mul_varlen_kernel(const SituMulVarlenParams __grid_constant__ params) {
  using namespace device;
  using Vec = AlignedVector<bf16x2_t, 4>;
  const auto expert_id = blockIdx.y;
  const float inv_beta = 1.0f / params.beta;
  const float inv_linear_beta = kHasLinearBeta ? 1.0f / params.linear_beta : 0.0f;
  // masked_m may also be produced by the preceding dispatch kernel.
  PDLWaitPrimary<kUsePDL>();
  const auto count = params.masked_m[expert_id];
  const auto num_tokens = count > 0 ? min(static_cast<uint32_t>(count), params.num_tokens) : 0u;
  for (uint32_t token_id = blockIdx.x; token_id < num_tokens; token_id += gridDim.x) {
    const auto offset = static_cast<uint64_t>(expert_id) * params.num_tokens + token_id;
    const auto input = params.input + offset * params.hidden_dim * 2;
    const auto output = params.output + offset * params.hidden_dim;
    Vec gate_vec, up_vec, out_vec;
    gate_vec.load(input, threadIdx.x);
    up_vec.load(input, threadIdx.x + blockDim.x);
    if (token_id == blockIdx.x) PDLTriggerSecondary<kUsePDL>();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const auto [g0, g1] = cast<fp32x2_t>(gate_vec[i]);
      const auto [u0, u1] = cast<fp32x2_t>(up_vec[i]);
      out_vec[i] = cast<bf16x2_t>(fp32x2_t{
          kimi_k3::situ_activate<kHasLinearBeta>(g0, u0, params.beta, inv_beta, params.linear_beta, inv_linear_beta),
          kimi_k3::situ_activate<kHasLinearBeta>(g1, u1, params.beta, inv_beta, params.linear_beta, inv_linear_beta)});
    }
    out_vec.store(output, threadIdx.x);
  }
}

inline void check_situ_varlen_shape(int64_t experts, int64_t tokens, int64_t hidden, int64_t input_width) {
  CHECK_HOST(experts >= 0 && experts <= kMaxExperts) << "invalid expert count";
  CHECK_HOST(tokens >= 0 && tokens <= std::numeric_limits<int32_t>::max()) << "invalid token capacity";
  CHECK_HOST(input_width == hidden * 2) << "invalid SiTU input width";
  // Each CTA covers a row with eight BF16 values per thread.
  CHECK_HOST(hidden > 0 && hidden % 256 == 0 && hidden <= 8192) << "SiTU hidden size must be 256-aligned and <= 8192";
}

/// Validate and launch BF16 masked SiTU. topk is retained for API compatibility.
template <bool kUsePDL>
struct SituAndMulMaskedKernel {
  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView masked_m,
      [[maybe_unused]] const uint32_t topk,
      const uint32_t expected_m,
      const double beta,
      const double linear_beta,
      const bool has_linear_beta) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto E = SymbolicSize{"num_experts"};
    auto T = SymbolicSize{"num_tokens_padded"};
    auto D = SymbolicSize{"input_width"};
    auto N = SymbolicSize{"hidden_dim"};
    device.set_options<kDLCUDA>();
    TensorMatcher({E, T, D}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(input);
    TensorMatcher({E, T, N}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(output);
    TensorMatcher({E}).with_dtype<int32_t>().with_device(device).verify(masked_m);
    check_situ_varlen_shape(E.unwrap(), T.unwrap(), N.unwrap(), D.unwrap());
    if (E.unwrap() == 0 || T.unwrap() == 0) return;
    const auto params = SituMulVarlenParams{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .output = static_cast<bf16_t*>(output.data_ptr()),
        .masked_m = static_cast<const int32_t*>(masked_m.data_ptr()),
        .beta = static_cast<float>(beta),
        .linear_beta = static_cast<float>(linear_beta),
        .hidden_dim = N.unwrap(),
        .num_tokens = static_cast<uint32_t>(T.unwrap()),
    };
    const auto grid = get_situ_masked_grid(static_cast<uint32_t>(E.unwrap()), expected_m);
    const auto threads = static_cast<uint32_t>(N.unwrap() / 8);
    if (has_linear_beta) {
      LaunchKernel(grid, threads, device.unwrap()).enable_pdl(kUsePDL)(situ_mul_varlen_kernel<true, kUsePDL>, params);
    } else {
      LaunchKernel(grid, threads, device.unwrap()).enable_pdl(kUsePDL)(situ_mul_varlen_kernel<false, kUsePDL>, params);
    }
  }
};

#if defined(SGLANG_SITU_MXFP4) && !defined(USE_ROCM)
struct SituMulQuantMxfp4Params {
  const bf16_t* __restrict__ input;
  uint8_t* __restrict__ output;
  uint8_t* __restrict__ output_scale;
  const int32_t* __restrict__ masked_m;
  float beta;
  float linear_beta;
  int64_t hidden_dim;
  uint32_t num_tokens;
};

// Only the FP4 factory enables this section, after checking device support.
SGL_DEVICE uint32_t pack_situ_mxfp4(const float* q, float inv_scale) {
  uint32_t packed;
  asm volatile(
      "{\n\t"
      ".reg .b8 r0, r1, r2, r3;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 r0, %2, %1;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 r1, %4, %3;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 r2, %6, %5;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 r3, %8, %7;\n\t"
      "mov.b32 %0, {r0, r1, r2, r3};\n\t"
      "}\n"
      : "=r"(packed)
      : "f"(q[0] * inv_scale),
        "f"(q[1] * inv_scale),
        "f"(q[2] * inv_scale),
        "f"(q[3] * inv_scale),
        "f"(q[4] * inv_scale),
        "f"(q[5] * inv_scale),
        "f"(q[6] * inv_scale),
        "f"(q[7] * inv_scale));
  return packed;
}

/// SiTU followed by BF16 rounding and group-32 MXFP4 quantization.
/// Physical scales are MN-major [E, D/64, T] uint16 (two E8M0 bytes).
template <bool kMasked, bool kHasLinearBeta, bool kUsePDL>
__global__
__launch_bounds__(1024, 2) void situ_mul_quant_mxfp4_kernel(const SituMulQuantMxfp4Params __grid_constant__ params) {
  using namespace device;
  constexpr uint32_t kWorkThreads = 4;
  using InputVec = AlignedVector<bf16x2_t, 4>;
  const auto expert_id = kMasked ? blockIdx.y : 0u;
  const auto group_id = threadIdx.x / kWorkThreads;
  const float inv_beta = 1.0f / params.beta;
  const float inv_linear_beta = kHasLinearBeta ? 1.0f / params.linear_beta : 0.0f;
  PDLWaitPrimary<kUsePDL>();
  uint32_t num_tokens = params.num_tokens;
  if constexpr (kMasked) {
    const auto count = params.masked_m[expert_id];
    num_tokens = count > 0 ? min(static_cast<uint32_t>(count), params.num_tokens) : 0u;
  }
  for (uint32_t token_id = blockIdx.x; token_id < num_tokens; token_id += gridDim.x) {
    const auto offset = static_cast<uint64_t>(expert_id) * params.num_tokens + token_id;
    const auto input = params.input + offset * params.hidden_dim * 2;
    const auto output = params.output + offset * params.hidden_dim / 2;
    const auto scale_offset = static_cast<uint64_t>(expert_id) * (params.hidden_dim / 32) * params.num_tokens +
                              (group_id / 2u) * (static_cast<uint64_t>(params.num_tokens) * 2u) + token_id * 2u +
                              group_id % 2u;
    InputVec gate_vec, up_vec;
    gate_vec.load(input, threadIdx.x);
    up_vec.load(input, threadIdx.x + blockDim.x);
    float local_max = 0.0f;
    float results[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const auto [g0, g1] = cast<fp32x2_t>(gate_vec[i]);
      const auto [u0, u1] = cast<fp32x2_t>(up_vec[i]);
      const auto rounded = cast<bf16x2_t>(fp32x2_t{
          kimi_k3::situ_activate<kHasLinearBeta>(g0, u0, params.beta, inv_beta, params.linear_beta, inv_linear_beta),
          kimi_k3::situ_activate<kHasLinearBeta>(g1, u1, params.beta, inv_beta, params.linear_beta, inv_linear_beta)});
      const auto [x, y] = cast<fp32x2_t>(rounded);
      results[2 * i] = x;
      results[2 * i + 1] = y;
      local_max = fmaxf(local_max, fmaxf(fabsf(x), fabsf(y)));
    }
    local_max = warp::reduce_max<kWorkThreads>(local_max);
    const float absmax = fmaxf(local_max, 1e-10f);
    const uint32_t scale_bits = (__float_as_uint(absmax / 6.0f) + 0x007FFFFFu) & 0x7F800000u;
    const float inv_scale = __uint_as_float(0x7F000000u - scale_bits);
    if (token_id == blockIdx.x) PDLTriggerSecondary<kUsePDL>();
    reinterpret_cast<uint32_t*>(output)[threadIdx.x] = pack_situ_mxfp4(results, inv_scale);
    if (threadIdx.x % kWorkThreads == 0) {
      params.output_scale[scale_offset] = static_cast<uint8_t>(scale_bits >> 23);
    }
  }
}

/// Common tensor contract for ordinary and masked SiTU+MXFP4 launchers.
template <bool kMasked, bool kUsePDL>
struct SituAndMulMxfp4Kernel {
  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView output_scale,
      const tvm::ffi::TensorView masked_m,
      [[maybe_unused]] const uint32_t topk,
      const uint32_t expected_m,
      const double beta,
      const double linear_beta,
      const bool has_linear_beta) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto E = SymbolicSize{"num_experts"};
    auto T = SymbolicSize{"num_tokens_padded"};
    auto D = SymbolicSize{"input_width"};
    auto P = SymbolicSize{"packed_hidden_dim"};
    auto S = SymbolicSize{"num_scale_pairs"};
    device.set_options<kDLCUDA>();
    if constexpr (kMasked) {
      TensorMatcher({E, T, D}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(input);
      TensorMatcher({E, T, P}).with_dtype<uint8_t>().with_device(device).ensure_alignment(4).verify(output);
      TensorMatcher({E, S, T}).with_dtype<uint16_t>().with_device(device).verify(output_scale);
      TensorMatcher({E}).with_dtype<int32_t>().with_device(device).verify(masked_m);
    } else {
      E.set_value(1);
      TensorMatcher({T, D}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(input);
      TensorMatcher({T, P}).with_dtype<uint8_t>().with_device(device).ensure_alignment(4).verify(output);
      TensorMatcher({S, T}).with_dtype<uint16_t>().with_device(device).verify(output_scale);
      TensorMatcher({0}).with_dtype<int32_t>().with_device(device).verify(masked_m);
    }
    const auto hidden_dim = P.unwrap() * 2;
    check_situ_varlen_shape(E.unwrap(), T.unwrap(), hidden_dim, D.unwrap());
    CHECK_HOST(S.unwrap() * 64 == hidden_dim) << "invalid scale dimension";
    if (E.unwrap() == 0 || T.unwrap() == 0) return;
    const auto params = SituMulQuantMxfp4Params{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .output = static_cast<uint8_t*>(output.data_ptr()),
        .output_scale = static_cast<uint8_t*>(output_scale.data_ptr()),
        .masked_m = kMasked ? static_cast<const int32_t*>(masked_m.data_ptr()) : nullptr,
        .beta = static_cast<float>(beta),
        .linear_beta = static_cast<float>(linear_beta),
        .hidden_dim = hidden_dim,
        .num_tokens = static_cast<uint32_t>(T.unwrap()),
    };
    const auto grid = kMasked ? get_situ_masked_grid(static_cast<uint32_t>(E.unwrap()), expected_m)
                              : dim3(static_cast<uint32_t>(T.unwrap()));
    const auto threads = static_cast<uint32_t>(hidden_dim / 8);
    if (has_linear_beta) {
      LaunchKernel(grid, threads, device.unwrap())
          .enable_pdl(kUsePDL)(situ_mul_quant_mxfp4_kernel<kMasked, true, kUsePDL>, params);
    } else {
      LaunchKernel(grid, threads, device.unwrap())
          .enable_pdl(kUsePDL)(situ_mul_quant_mxfp4_kernel<kMasked, false, kUsePDL>, params);
    }
  }
};
#endif

}  // namespace sglang
