// BF16 RMSNorm followed by 128-wide per-token FP8 E4M3 quantization with
// packed column-major UE8M0 scales. The reduction size is a JIT template
// parameter so model dispatch can be capability-based without giving up the
// single-load, fully-unrolled path for each static hidden size.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/impl/norm.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp8.h>

namespace {

constexpr int kGroupSize = 128;
constexpr int kItemsPerThread = 16;
constexpr int kThreadsPerGroup = kGroupSize / kItemsPerThread;
constexpr int kScaleBytesPerPack = sizeof(uint32_t);
constexpr float kQuantAbsmaxFloor = 1e-10f;
constexpr float kFP8E4M3Max = 448.0f;

static_assert(kThreadsPerGroup == 8);

// This is deliberately the fused kernel's fixed quantization contract, not a
// generic quant API. Keep the existing quant-v2 JIT textually unchanged and
// keep this fusion's narrower assumptions in its own source file.
SGL_DEVICE float quant_group_reduce_max(float value) {
  constexpr device::warp::mask_t kGroupMask = (device::warp::mask_t{1} << kThreadsPerGroup) - 1;
  const auto mask = kGroupMask << (kThreadsPerGroup * ((threadIdx.x % 32) / kThreadsPerGroup));
  return device::warp::reduce_max<kThreadsPerGroup>(value, mask);
}

SGL_DEVICE float quant_fast_pow2(int exponent) {
  return __uint_as_float((exponent + 127) << 23);
}

SGL_DEVICE int quant_fast_log2_ceil(float value) {
  const auto bits = __float_as_uint(value);
  const auto exponent = (bits >> 23) & 0xff;
  const auto mantissa = bits & ((1 << 23) - 1);
  return exponent - 127 + (mantissa != 0);
}

SGL_DEVICE void
quantize_fp8_e4m3_ue8m0_group(const bf16_t* values, int group_lane, fp8_e4m3_t* output, uint8_t* scale_output) {
  float absmax = kQuantAbsmaxFloor;
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    absmax = fmaxf(absmax, fabsf(static_cast<float>(values[item])));
  }
  absmax = quant_group_reduce_max(absmax);

  constexpr float kFP8E4M3MaxInv = 1.0f / kFP8E4M3Max;
  const int scale_inv_exponent = quant_fast_log2_ceil(absmax * kFP8E4M3MaxInv);
  const float scale = quant_fast_pow2(-scale_inv_exponent);
  const float scale_inv = quant_fast_pow2(scale_inv_exponent);
  if (group_lane == 0) *scale_output = static_cast<uint8_t>(__float_as_uint(scale_inv) >> 23);

  const float2 repeated_scale = {scale, scale};
  int4 output_buffer;
  auto* output_pairs = reinterpret_cast<__nv_fp8x2_storage_t*>(&output_buffer);
#pragma unroll
  for (int item = 0; item < kItemsPerThread; item += 2) {
    const float2 input_pair = {static_cast<float>(values[item]), static_cast<float>(values[item + 1])};
    float2 quantized_pair = __fmul2_rn(input_pair, repeated_scale);
    quantized_pair.x = fminf(fmaxf(quantized_pair.x, -kFP8E4M3Max), kFP8E4M3Max);
    quantized_pair.y = fminf(fmaxf(quantized_pair.y, -kFP8E4M3Max), kFP8E4M3Max);
    output_pairs[item / 2] = __nv_cvt_float2_to_fp8x2(quantized_pair, __NV_SATFINITE, __NV_E4M3);
  }
  *reinterpret_cast<int4*>(output) = output_buffer;
}

struct RMSNormPerTokenGroupQuantFP8Params {
  const bf16_t* __restrict__ input;
  const bf16_t* __restrict__ weight;
  fp8_e4m3_t* __restrict__ output_q;
  uint32_t* __restrict__ output_s;
  bf16_t* __restrict__ output_norm;
  int64_t input_stride;
  int64_t scale_hidden_stride;
  float eps;
};

template <int kHidden, bool kUsePDL>
__global__
__launch_bounds__(((kHidden / kItemsPerThread + 31) / 32) * 32) void rmsnorm_per_token_group_quant_fp8_kernel(
    const RMSNormPerTokenGroupQuantFP8Params __grid_constant__ params) {
  static_assert(kHidden >= kGroupSize && kHidden % kGroupSize == 0);
  static_assert(kHidden <= 16384, "one-vector-per-thread mapping exceeds CUDA's 1024-thread limit");
  constexpr int kActiveThreads = kHidden / kItemsPerThread;
  constexpr int kThreads = ((kActiveThreads + 31) / 32) * 32;
  constexpr int kWarps = kThreads / 32;
  constexpr int kGroupsPerToken = kHidden / kGroupSize;
  constexpr int kPackedScaleCols = (kGroupsPerToken + kScaleBytesPerPack - 1) / kScaleBytesPerPack;
  using Vec = device::AlignedVector<packed_t<bf16_t>, kItemsPerThread / 2>;

  const int thread_id = threadIdx.x;
  const int row = blockIdx.x;
  const bool is_active = thread_id < kActiveThreads;
  device::PDLWaitPrimary<kUsePDL>();
  const int elem_offset = thread_id * kItemsPerThread;
  const auto* input_row = params.input + static_cast<int64_t>(row) * params.input_stride;
  auto* norm_row = params.output_norm + static_cast<int64_t>(row) * kHidden;
  auto* q_row = params.output_q + static_cast<int64_t>(row) * kHidden;

  // DeepGEMM packs four UE8M0 scale bytes into each int32. Hidden sizes with
  // a non-multiple-of-four group count leave padding bytes in the final pack;
  // initialize only those padding bytes. A whole-word store here would race
  // the valid byte stores issued by other warps.
  if constexpr (kGroupsPerToken % kScaleBytesPerPack != 0) {
    constexpr int kValidBytesInLastPack = kGroupsPerToken % kScaleBytesPerPack;
    constexpr int kPaddingBytes = kScaleBytesPerPack - kValidBytesInLastPack;
    if (thread_id < kPaddingBytes) {
      auto* last_scale_pack =
          reinterpret_cast<uint8_t*>(params.output_s) +
          (static_cast<int64_t>(kPackedScaleCols - 1) * params.scale_hidden_stride + row) * kScaleBytesPerPack;
      last_scale_pack[kValidBytesInLastPack + thread_id] = 0;
    }
  }

  // Retain each active thread's one 32-byte input vector across the CTA RMS
  // reduction. This exposes the same eight-thread quant subgroups and avoids
  // the second global input load performed by the unfused sequence.
  Vec input_vec{};
  Vec weight_vec{};
  if (is_active) input_vec.load(input_row + elem_offset);
  if (is_active) weight_vec.load(params.weight + elem_offset);
  __shared__ float warp_sums[kWarps];
  const auto norm_vec =
      device::norm::apply_norm_cta_static<kHidden, kWarps>(input_vec, weight_vec, params.eps, warp_sums);

  if (is_active) {
    norm_vec.store(norm_row + elem_offset);

    const int group_idx = thread_id / kThreadsPerGroup;
    const int group_lane = thread_id % kThreadsPerGroup;
    const int hidden_pack = group_idx / kScaleBytesPerPack;
    const int pack_idx = group_idx % kScaleBytesPerPack;
    auto* scale_byte = reinterpret_cast<uint8_t*>(params.output_s) +
                       (static_cast<int64_t>(hidden_pack) * params.scale_hidden_stride + row) * kScaleBytesPerPack +
                       pack_idx;
    quantize_fp8_e4m3_ue8m0_group(
        reinterpret_cast<const bf16_t*>(norm_vec.data()), group_lane, q_row + elem_offset, scale_byte);
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <int kHidden, bool kUsePDL>
struct RMSNormPerTokenGroupQuantFP8Kernel {
  static_assert(kHidden >= kGroupSize && kHidden % kGroupSize == 0);
  static_assert(kHidden <= 16384);
  static constexpr int kActiveThreads = kHidden / kItemsPerThread;
  static constexpr int kThreads = ((kActiveThreads + 31) / 32) * 32;
  static constexpr int kGroupsPerToken = kHidden / kGroupSize;
  static constexpr int kPackedScaleCols = (kGroupsPerToken + kScaleBytesPerPack - 1) / kScaleBytesPerPack;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView output_q,
      const tvm::ffi::TensorView output_s,
      const tvm::ffi::TensorView output_norm,
      float eps) {
    using namespace host;

    auto M = SymbolicSize{"num_tokens"};
    auto input_stride = SymbolicSize{"input_stride"};
    auto hidden = SymbolicSize{"hidden_size"};
    auto scale_hidden_stride = SymbolicSize{"scale_hidden_stride"};
    auto device = SymbolicDevice{};
    hidden.set_value(kHidden);
    device.set_options<kDLCUDA>();

    TensorMatcher({M, hidden}).with_strides({input_stride, 1}).with_dtype<bf16_t>().with_device(device).verify(input);
    TensorMatcher({hidden}).with_dtype<bf16_t>().with_device(device).verify(weight);
    TensorMatcher({M, hidden}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output_q);
    TensorMatcher({M, kPackedScaleCols})
        .with_strides({1, scale_hidden_stride})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(output_s);
    TensorMatcher({M, hidden}).with_dtype<bf16_t>().with_device(device).verify(output_norm);

    const auto num_tokens = M.unwrap();
    RuntimeCheck(input_stride.unwrap() % kItemsPerThread == 0, "input row stride must preserve 32-byte alignment");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(input.data_ptr()) % 32 == 0 &&
            reinterpret_cast<uintptr_t>(weight.data_ptr()) % 32 == 0 &&
            reinterpret_cast<uintptr_t>(output_norm.data_ptr()) % 32 == 0,
        "BF16 input, weight, and norm output pointers must be 32-byte aligned");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(output_q.data_ptr()) % 16 == 0, "FP8 output pointer must be 16-byte aligned");
    if (num_tokens == 0) return;
    RuntimeCheck(
        scale_hidden_stride.unwrap() >= num_tokens && scale_hidden_stride.unwrap() % 4 == 0,
        "packed scale leading dimension must cover M and be aligned to four rows");

    const auto params = RMSNormPerTokenGroupQuantFP8Params{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .weight = static_cast<const bf16_t*>(weight.data_ptr()),
        .output_q = static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
        .output_s = static_cast<uint32_t*>(output_s.data_ptr()),
        .output_norm = static_cast<bf16_t*>(output_norm.data_ptr()),
        .input_stride = input_stride.unwrap(),
        .scale_hidden_stride = scale_hidden_stride.unwrap(),
        .eps = eps,
    };

    host::LaunchKernel(dim3(static_cast<uint32_t>(num_tokens)), dim3(kThreads), device.unwrap())
        .enable_pdl(kUsePDL)(rmsnorm_per_token_group_quant_fp8_kernel<kHidden, kUsePDL>, params);
  }
};

}  // namespace
