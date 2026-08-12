// DeepSeek-V4 wo_a activation quantization for DeepGEMM fp8_einsum.
//
// This is intentionally narrower than the generic per_token_group_quant_8bit_v2
// kernel: input is a [T, G, D] view with contiguous hidden groups, output_q is
// contiguous [T, G, D], group_size is fixed to 128, scales are fp32 UE8M0
// power-of-two values, and output_s is a logical [T, G, D/128] view backed by
// group-major [G, T, D/128] storage.
//
// The generic kernel cannot read the strided DSV4 view while producing
// contiguous [T, G, D] codes and group-major scales without an extra full-tensor
// copy.
#include <sgl_kernel/tensor.h>  // TensorMatcher, SymbolicSize/Device
#include <sgl_kernel/utils.h>   // RuntimeCheck

#include <sgl_kernel/utils.cuh>  // fp8 aliases, PDL helpers
#include <sgl_kernel/warp.cuh>   // warp::reduce_max

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>  // UE8M0 and FP8 helpers

#include <tvm/ffi/container/tensor.h>  // tvm::ffi::TensorView

#include <cstdint>
#include <cuda_fp8.h>

namespace {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

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
  // Tail subwarps can be inactive at the bounds check, so reduce with only the
  // current subgroup's lanes rather than a full-warp mask.
  constexpr device::warp::mask_t kSub = (device::warp::mask_t{1} << THREADS_PER_SUBWARP) - 1;
  const device::warp::mask_t mask = kSub << (THREADS_PER_SUBWARP * ((threadIdx.x % 32) / THREADS_PER_SUBWARP));
  return device::warp::reduce_max<THREADS_PER_SUBWARP>(val, mask);
}

template <typename T, bool kUsePDL>
__global__ void fp8_wo_a_group_major_quant_ue8m0_kernel(
    const T* __restrict__ input,
    fp8_e4m3_t* __restrict__ output_q,
    float* __restrict__ output_s,
    int64_t total_scale_groups,
    int64_t num_tokens,
    int hidden_dim_groups,
    int num_outer_groups,
    int64_t input_stride_t) {
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
        token_idx * input_stride_t + outer_idx * GROUP_SIZE * hidden_dim_groups + hidden_group * GROUP_SIZE;
    const int64_t output_group_start_offset = group_id * GROUP_SIZE;

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

    constexpr float kFp8MaxInv = 1.0f / kFP8E4M3Max;
    const int32_t scale_ue8m0 = cast_to_ue8m0(local_absmax * kFp8MaxInv);
    const float y_scale = inv_scale_ue8m0(scale_ue8m0);
    const float y_scale_inv = __uint_as_float(static_cast<uint32_t>(scale_ue8m0) << 23);

    int4 output_buf;
    auto* output_buf_ptr = reinterpret_cast<fp8x2_e4m3_t*>(&output_buf);
#pragma unroll
    for (uint32_t j = 0; j < INPUT_VEC_SIZE; j += 2) {
      output_buf_ptr[j / 2] =
          pack_fp8(static_cast<float>(input_vec[j]) * y_scale, static_cast<float>(input_vec[j + 1]) * y_scale);
    }

    *reinterpret_cast<int4*>(output_q + output_group_start_offset + lane_id * INPUT_VEC_SIZE) = output_buf;

    if (lane_id == 0) {
      output_s[(outer_idx * num_tokens + token_idx) * hidden_dim_groups + hidden_group] = y_scale_inv;
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

    TensorMatcher({TSize, GSize, DSize}).with_strides({-1, DSize, 1}).with_dtype<T>().with_device(device).verify(input);
    TensorMatcher({TSize, GSize, DSize}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output_q);
    TensorMatcher({GSize, TSize, SSize}).with_dtype<float>().with_device(device).verify(output_s);

    const auto num_tokens = TSize.unwrap();
    const auto num_outer_groups = GSize.unwrap();
    const auto hidden_dim = DSize.unwrap();
    const auto hidden_dim_groups = SSize.unwrap();
    const auto input_stride_t = input.stride(0);
    constexpr int64_t kInputAlignElements = sizeof(int4) / sizeof(T);

    RuntimeCheck(hidden_dim % GROUP_SIZE == 0, "hidden_dim must be divisible by 128");
    RuntimeCheck(hidden_dim_groups == hidden_dim / GROUP_SIZE, "output_s hidden dim mismatch");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(input.data_ptr()) % sizeof(int4) == 0,
        "input base pointer must be 16-byte aligned");
    RuntimeCheck(
        num_tokens <= 1 || input_stride_t % kInputAlignElements == 0,
        "input token stride must preserve 16-byte vector-load alignment");

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
            static_cast<int64_t>(num_tokens),
            static_cast<int>(hidden_dim_groups),
            static_cast<int>(num_outer_groups),
            static_cast<int64_t>(input_stride_t));
  }
};

}  // namespace
