#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp8.h>

namespace {

constexpr float kLocalAbsmaxEps = 1e-10f;
constexpr uint32_t kVecBytes = 32;
constexpr uint32_t kFp8E4m3Max = 448;

template <int kThreadsPerSubwarp>
SGL_DEVICE float group_reduce_max(float val) {
  static_assert(
      (kThreadsPerSubwarp & (kThreadsPerSubwarp - 1)) == 0 && kThreadsPerSubwarp <= 16 && kThreadsPerSubwarp >= 1,
      "kThreadsPerSubwarp must be 1, 2, 4, 8, or 16");
  constexpr device::warp::mask_t kSub = (device::warp::mask_t{1} << kThreadsPerSubwarp) - 1;
  const device::warp::mask_t mask = kSub << (kThreadsPerSubwarp * ((threadIdx.x % 32) / kThreadsPerSubwarp));
  return device::warp::reduce_max<kThreadsPerSubwarp>(val, mask);
}

SGL_DEVICE float fast_pow2(int x) {
  const uint32_t bits_x = (x + 127) << 23;
  return __uint_as_float(bits_x);
}

SGL_DEVICE int fast_log2_ceil(float x) {
  const auto bits_x = __float_as_uint(x);
  const auto exp_x = (bits_x >> 23) & 0xff;
  const auto man_bits = bits_x & ((1 << 23) - 1);
  return exp_x - 127 + (man_bits != 0);
}

SGL_DEVICE float2 fmul2_rn(float2 a, float2 b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  return __fmul2_rn(a, b);
#else
  return {a.x * b.x, a.y * b.y};
#endif
}

struct Fp8WoAGroupMajorQuantUe8m0Params {
  const bf16_t* __restrict__ input;   // [G, T, D], contiguous
  fp8_e4m3_t* __restrict__ output_q;  // [G, T, D], contiguous
  int32_t* __restrict__ output_s;     // logical [G, T, scale_inner], strided
  uint32_t G;
  uint64_t T;
  uint32_t D;
  uint32_t hidden_dim_num_groups;
  uint64_t total_groups;
  int64_t scale_stride_g;
  int64_t scale_stride_t;
  int64_t scale_stride_s;
};

template <uint32_t kGroupSize, bool kUsePDL>
__global__ void
fp8_wo_a_group_major_quant_ue8m0_kernel(const Fp8WoAGroupMajorQuantUe8m0Params __grid_constant__ params) {
  constexpr uint32_t kInputVecSize = kVecBytes / sizeof(bf16_t);
  constexpr uint32_t kInputInt4Size = kVecBytes / sizeof(int4);
  constexpr uint32_t kThreadsPerSubwarp = kGroupSize / kInputVecSize;
  constexpr uint32_t kSubwarpsPerBlock = 16;
  static_assert(kGroupSize % kInputVecSize == 0, "group_size must be divisible by the vector width");

  device::PDLWaitPrimary<kUsePDL>();

  const uint32_t subwarp_id = threadIdx.x / kThreadsPerSubwarp;
  const uint32_t lane_id = threadIdx.x % kThreadsPerSubwarp;
  const uint64_t linear_group = static_cast<uint64_t>(blockIdx.x) * kSubwarpsPerBlock + subwarp_id;
  if (linear_group < params.total_groups) {
    const uint32_t hidden_group = linear_group % params.hidden_dim_num_groups;
    const uint64_t token_linear = linear_group / params.hidden_dim_num_groups;
    const uint64_t t = token_linear % params.T;
    const uint32_t g = token_linear / params.T;
    const uint64_t row_offset = (static_cast<uint64_t>(g) * params.T + t) * params.D;
    const uint64_t group_offset = row_offset + static_cast<uint64_t>(hidden_group) * kGroupSize;

    int4 input_int4[kInputInt4Size];
    auto* input_vec = reinterpret_cast<bf16_t*>(input_int4);

#pragma unroll
    for (uint32_t j = 0; j < kInputInt4Size; ++j) {
      input_int4[j] = reinterpret_cast<const int4*>(params.input + group_offset + lane_id * kInputVecSize)[j];
    }

    float local_absmax = kLocalAbsmaxEps;
#pragma unroll
    for (uint32_t j = 0; j < kInputVecSize; ++j) {
      const float val = static_cast<float>(input_vec[j]);
      local_absmax = fmaxf(local_absmax, fabsf(val));
    }
    local_absmax = group_reduce_max<kThreadsPerSubwarp>(local_absmax);

    const int exp_scale_inv = fast_log2_ceil(local_absmax / static_cast<float>(kFp8E4m3Max));
    const float y_scale = fast_pow2(-exp_scale_inv);
    const float y_scale_inv = fast_pow2(exp_scale_inv);
    const uint8_t scale_exp = static_cast<uint8_t>(__float_as_uint(y_scale_inv) >> 23);

    if (lane_id == 0) {
      const uint32_t hidden_idx_packed = hidden_group / 4u;
      const uint32_t pack_idx = hidden_group % 4u;
      auto* scale_bytes = reinterpret_cast<uint8_t*>(
          params.output_s + static_cast<int64_t>(g) * params.scale_stride_g +
          static_cast<int64_t>(t) * params.scale_stride_t +
          static_cast<int64_t>(hidden_idx_packed) * params.scale_stride_s);
      scale_bytes[pack_idx] = scale_exp;
    }

    const uint32_t remainder = params.hidden_dim_num_groups % 4u;
    if (remainder != 0 && hidden_group == params.hidden_dim_num_groups - 1u && lane_id < 4u - remainder) {
      const uint32_t hidden_idx_packed = hidden_group / 4u;
      const uint32_t pack_idx = hidden_group % 4u;
      auto* scale_bytes = reinterpret_cast<uint8_t*>(
          params.output_s + static_cast<int64_t>(g) * params.scale_stride_g +
          static_cast<int64_t>(t) * params.scale_stride_t +
          static_cast<int64_t>(hidden_idx_packed) * params.scale_stride_s);
      scale_bytes[pack_idx + 1u + lane_id] = 0;
    }

    const float2 y_scale_repeated = {y_scale, y_scale};
    int4 output_buf;
    auto* output_buf_ptr = reinterpret_cast<__nv_fp8x2_storage_t*>(&output_buf);
#pragma unroll
    for (uint32_t j = 0; j < kInputVecSize; j += 2) {
      float2 inputx2 = {static_cast<float>(input_vec[j]), static_cast<float>(input_vec[j + 1])};
      float2 outputx2 = fmul2_rn(inputx2, y_scale_repeated);
      outputx2.x = fminf(fmaxf(outputx2.x, -static_cast<float>(kFp8E4m3Max)), static_cast<float>(kFp8E4m3Max));
      outputx2.y = fminf(fmaxf(outputx2.y, -static_cast<float>(kFp8E4m3Max)), static_cast<float>(kFp8E4m3Max));
      output_buf_ptr[j / 2] = __nv_cvt_float2_to_fp8x2(outputx2, __NV_SATFINITE, __NV_E4M3);
    }
    *reinterpret_cast<int4*>(params.output_q + group_offset + lane_id * kInputVecSize) = output_buf;
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kGroupSize, bool kUsePDL>
struct Fp8WoAGroupMajorQuantUe8m0Kernel {
  static void
  run(tvm::ffi::TensorView input,
      tvm::ffi::TensorView output_q,
      tvm::ffi::TensorView output_s,
      int64_t scale_stride_g,
      int64_t scale_stride_t,
      int64_t scale_stride_s) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto G = SymbolicSize{"G"};
    auto T = SymbolicSize{"T"};
    auto D = SymbolicSize{"D"};
    auto S = SymbolicSize{"scale_inner"};
    device.set_options<kDLCUDA>();

    TensorMatcher({G, T, D}).with_dtype<bf16_t>().with_device(device).verify(input);
    TensorMatcher({G, T, D}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output_q);
    TensorMatcher({G, T, S}).with_strides({-1, -1, -1}).with_dtype<int32_t>().with_device(device).verify(output_s);

    const uint32_t groups = static_cast<uint32_t>(G.unwrap());
    const uint64_t tokens = static_cast<uint64_t>(T.unwrap());
    const uint32_t hidden = static_cast<uint32_t>(D.unwrap());
    RuntimeCheck(hidden % kGroupSize == 0, "D ", hidden, " not divisible by group_size ", kGroupSize);
    const uint32_t hidden_dim_num_groups = hidden / static_cast<uint32_t>(kGroupSize);
    const uint32_t scale_inner = (hidden_dim_num_groups + 3u) / 4u;
    RuntimeCheck(static_cast<uint32_t>(S.unwrap()) == scale_inner, "scale_inner mismatch");

    constexpr uint32_t kInputVecSize = kVecBytes / sizeof(bf16_t);
    constexpr uint32_t kThreadsPerSubwarp = static_cast<uint32_t>(kGroupSize) / kInputVecSize;
    constexpr uint32_t kSubwarpsPerBlock = 16;
    constexpr uint32_t kThreads = kThreadsPerSubwarp * kSubwarpsPerBlock;
    const uint64_t total_groups = static_cast<uint64_t>(groups) * tokens * hidden_dim_num_groups;
    if (total_groups == 0) return;
    const uint64_t grid_x = (total_groups + kSubwarpsPerBlock - 1u) / kSubwarpsPerBlock;
    RuntimeCheck(grid_x <= static_cast<uint64_t>(UINT32_MAX), "grid.x exceeds CUDA dim3 range");

    const auto params = Fp8WoAGroupMajorQuantUe8m0Params{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .output_q = static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
        .output_s = static_cast<int32_t*>(output_s.data_ptr()),
        .G = groups,
        .T = tokens,
        .D = hidden,
        .hidden_dim_num_groups = hidden_dim_num_groups,
        .total_groups = total_groups,
        .scale_stride_g = scale_stride_g,
        .scale_stride_t = scale_stride_t,
        .scale_stride_s = scale_stride_s,
    };
    const dim3 grid(static_cast<uint32_t>(grid_x));
    const dim3 block(kThreads);
    constexpr auto kernel = fp8_wo_a_group_major_quant_ue8m0_kernel<static_cast<uint32_t>(kGroupSize), kUsePDL>;
    LaunchKernel(grid, block, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
