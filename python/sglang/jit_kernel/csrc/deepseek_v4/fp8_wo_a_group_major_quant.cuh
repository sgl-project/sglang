// DeepSeek-V4 wo_a activation quantization for DeepGEMM fp8_einsum.
//
// This is intentionally narrower than the generic per_token_group_quant_8bit_v2
// kernel: input is a [T, G, D] view with contiguous hidden groups, output_q is
// contiguous [T, G, D], group_size is fixed to 128, scales are fp32 UE8M0
// power-of-two values, and output_s is a logical [T, G, D/128] view backed by
// group-major [G, T, D/128] storage.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <cstdint>
#include <cuda_fp8.h>
#include <type_traits>

namespace {

constexpr float LOCAL_ABSMAX_ABS = 1e-10f;
constexpr uint32_t GROUP_SIZE = 128;
constexpr uint32_t THREADS_PER_GROUP = 8;
constexpr uint32_t SUBWARPS_PER_BLOCK = 16;
constexpr uint32_t INPUT_VEC_NUM_BYTES = 32;
constexpr uint32_t INPUT_INT4_SIZE = INPUT_VEC_NUM_BYTES / sizeof(int4);

template <int THREADS_PER_SUBWARP>
SGL_DEVICE float GroupReduceMax(float val) {
  static_assert(
      (THREADS_PER_SUBWARP & (THREADS_PER_SUBWARP - 1)) == 0 && THREADS_PER_SUBWARP <= 16 && THREADS_PER_SUBWARP >= 1,
      "THREADS_PER_SUBWARP must be 1, 2, 4, 8, or 16");
  constexpr device::warp::mask_t kSub = (device::warp::mask_t{1} << THREADS_PER_SUBWARP) - 1;
  const device::warp::mask_t mask = kSub << (THREADS_PER_SUBWARP * ((threadIdx.x % 32) / THREADS_PER_SUBWARP));
  return device::warp::reduce_max<THREADS_PER_SUBWARP>(val, mask);
}

SGL_DEVICE float fast_pow2(int x) {
  const uint32_t bits_x = (x + 127) << 23;
  return __uint_as_float(bits_x);
}

SGL_DEVICE int fast_log2_ceil(float x) {
  const uint32_t bits_x = __float_as_uint(x);
  const int exp_x = (bits_x >> 23) & 0xff;
  const uint32_t man_bits = bits_x & ((1 << 23) - 1);
  return exp_x - 127 + (man_bits != 0);
}

struct FP8E4M3Info {
  static constexpr float MIN = -448.0f;
  static constexpr float MAX = 448.0f;
};

template <typename T, bool kUsePDL>
__global__ void fp8_wo_a_group_major_quant_ue8m0_kernel(
    const T* __restrict__ input,
    fp8_e4m3_t* __restrict__ output_q,
    float* __restrict__ output_s,
    int64_t total_scale_groups,
    int hidden_dim_groups,
    int num_outer_groups,
    int hidden_dim,
    int64_t input_stride_t,
    int64_t input_stride_g,
    int scale_stride_t,
    int scale_stride_g) {
  device::PDLWaitPrimary<kUsePDL>();

  const int64_t subwarp_id = threadIdx.x / THREADS_PER_GROUP;
  const int lane_id = threadIdx.x % THREADS_PER_GROUP;
  const int64_t group_id = static_cast<int64_t>(blockIdx.x) * SUBWARPS_PER_BLOCK + subwarp_id;
  if (group_id < total_scale_groups) {
    const int hidden_group = group_id % hidden_dim_groups;
    const int64_t token_outer = group_id / hidden_dim_groups;
    const int outer_idx = token_outer % num_outer_groups;
    const int64_t token_idx = token_outer / num_outer_groups;

    constexpr uint32_t INPUT_VEC_SIZE = INPUT_VEC_NUM_BYTES / sizeof(T);
    static_assert(INPUT_VEC_SIZE * THREADS_PER_GROUP == GROUP_SIZE);

    const int64_t input_group_start_offset =
        token_idx * input_stride_t + outer_idx * input_stride_g + hidden_group * GROUP_SIZE;
    const int64_t output_group_start_offset =
        (token_idx * num_outer_groups + outer_idx) * static_cast<int64_t>(hidden_dim) + hidden_group * GROUP_SIZE;

    int4 input_int4[INPUT_INT4_SIZE];
    T* input_vec = reinterpret_cast<T*>(input_int4);

#pragma unroll
    for (uint32_t j = 0; j < INPUT_INT4_SIZE; ++j) {
      input_int4[j] = reinterpret_cast<const int4*>(input + input_group_start_offset + lane_id * INPUT_VEC_SIZE)[j];
    }

    float local_absmax = LOCAL_ABSMAX_ABS;
#pragma unroll
    for (uint32_t j = 0; j < INPUT_VEC_SIZE; ++j) {
      const float val = static_cast<float>(input_vec[j]);
      local_absmax = fmaxf(local_absmax, fabsf(val));
    }

    local_absmax = GroupReduceMax<THREADS_PER_GROUP>(local_absmax);

    const int exp_scale_inv = fast_log2_ceil(local_absmax / FP8E4M3Info::MAX);
    const float y_scale = fast_pow2(-exp_scale_inv);
    const float y_scale_inv = fast_pow2(exp_scale_inv);

    int4 output_buf;
    auto* output_buf_ptr = reinterpret_cast<__nv_fp8x2_storage_t*>(&output_buf);
    const float2 y_scale_repeated = {y_scale, y_scale};
#pragma unroll
    for (uint32_t j = 0; j < INPUT_VEC_SIZE; j += 2) {
      const float2 inputx2 = {static_cast<float>(input_vec[j]), static_cast<float>(input_vec[j + 1])};
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
      float2 outputx2 = __fmul2_rn(inputx2, y_scale_repeated);
#else
      float2 outputx2 = {inputx2.x * y_scale_repeated.x, inputx2.y * y_scale_repeated.y};
#endif
      outputx2.x = fminf(fmaxf(outputx2.x, FP8E4M3Info::MIN), FP8E4M3Info::MAX);
      outputx2.y = fminf(fmaxf(outputx2.y, FP8E4M3Info::MIN), FP8E4M3Info::MAX);
      output_buf_ptr[j / 2] = __nv_cvt_float2_to_fp8x2(outputx2, __NV_SATFINITE, __NV_E4M3);
    }

    *reinterpret_cast<int4*>(output_q + output_group_start_offset + lane_id * INPUT_VEC_SIZE) = output_buf;

    if (lane_id == 0) {
      output_s
          [token_idx * static_cast<int64_t>(scale_stride_t) + outer_idx * static_cast<int64_t>(scale_stride_g) +
           hidden_group] = y_scale_inv;
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <typename T, bool kUsePDL>
struct FP8WoAGroupMajorQuantUE8M0Kernel {
  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView output_q, tvm::ffi::TensorView output_s) {
    using namespace host;

    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto TSize = SymbolicSize{"num_tokens"};
    auto GSize = SymbolicSize{"num_outer_groups"};
    auto DSize = SymbolicSize{"hidden_dim"};
    auto SSize = SymbolicSize{"hidden_dim_groups"};

    TensorMatcher({TSize, GSize, DSize}).with_strides({-1, -1, 1}).with_dtype<T>().with_device(device).verify(input);
    TensorMatcher({TSize, GSize, DSize}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output_q);
    TensorMatcher({TSize, GSize, SSize})
        .with_strides({-1, -1, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(output_s);

    const auto num_tokens = TSize.unwrap();
    const auto num_outer_groups = GSize.unwrap();
    const auto hidden_dim = DSize.unwrap();
    const auto hidden_dim_groups = SSize.unwrap();
    const auto input_stride_t = input.stride(0);
    const auto input_stride_g = input.stride(1);
    const auto scale_stride_t = output_s.stride(0);
    const auto scale_stride_g = output_s.stride(1);
    constexpr int64_t kInputAlignElements = sizeof(int4) / sizeof(T);

    RuntimeCheck(hidden_dim % GROUP_SIZE == 0, "hidden_dim must be divisible by 128");
    RuntimeCheck(hidden_dim_groups == hidden_dim / GROUP_SIZE, "output_s hidden dim mismatch");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(input.data_ptr()) % sizeof(int4) == 0,
        "input base pointer must be 16-byte aligned");
    RuntimeCheck(
        num_tokens <= 1 || input_stride_t % kInputAlignElements == 0,
        "input token stride must preserve 16-byte vector-load alignment");
    RuntimeCheck(
        num_outer_groups <= 1 || input_stride_g % kInputAlignElements == 0,
        "input group stride must preserve 16-byte vector-load alignment");
    RuntimeCheck(scale_stride_t == hidden_dim_groups, "output_s must use DSV4 group-major token stride");
    RuntimeCheck(scale_stride_g == num_tokens * hidden_dim_groups, "output_s must use DSV4 group-major outer stride");

    const int64_t total_scale_groups = num_tokens * num_outer_groups * hidden_dim_groups;
    if (total_scale_groups == 0) return;

    const auto grid = dim3((total_scale_groups + SUBWARPS_PER_BLOCK - 1) / SUBWARPS_PER_BLOCK);
    const auto block = dim3(SUBWARPS_PER_BLOCK * THREADS_PER_GROUP);
    host::LaunchKernel(grid, block, device.unwrap())
        .enable_pdl(kUsePDL)(
            fp8_wo_a_group_major_quant_ue8m0_kernel<T, kUsePDL>,
            static_cast<const T*>(input.data_ptr()),
            static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
            static_cast<float*>(output_s.data_ptr()),
            total_scale_groups,
            static_cast<int>(hidden_dim_groups),
            static_cast<int>(num_outer_groups),
            static_cast<int>(hidden_dim),
            static_cast<int64_t>(input_stride_t),
            static_cast<int64_t>(input_stride_g),
            static_cast<int>(scale_stride_t),
            static_cast<int>(scale_stride_g));
  }
};

}  // namespace
