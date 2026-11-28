// DeepSeek-V4 wo_a activation quantization for DeepGEMM fp8_einsum.
//
// This is intentionally narrower than the generic per_token_group_quant_8bit_v2
// kernel: input is a [T, G, D] view with contiguous hidden groups, output_q is
// contiguous [T, G, D], group_size is fixed to 128, and output_s stores four
// packed UE8M0 exponent bytes per int32. Its physical layout is
// [G, ceil((D/128)/4), align_up(T, 4)]; the Python wrapper returns the logical
// [T, G, ceil((D/128)/4)] view consumed natively by DeepGEMM.
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

namespace sglang {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

constexpr float LOCAL_ABSMAX_ABS = 1e-10f;
constexpr uint32_t GROUP_SIZE = 128;
constexpr uint32_t THREADS_PER_GROUP = 8;
constexpr uint32_t SUBWARPS_PER_BLOCK = 16;
constexpr uint32_t INPUT_VEC_NUM_BYTES = 32;
constexpr uint32_t INPUT_INT4_SIZE = INPUT_VEC_NUM_BYTES / sizeof(int4);
constexpr int UE8M0_SCALES_PER_PACK = 4;
static_assert(UE8M0_SCALES_PER_PACK == sizeof(int32_t));

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

template <typename T>
SGL_DEVICE void QuantizeWoAGroup(
    const T* input_vec,
    fp8_e4m3_t* output_q,
    int32_t* output_s,
    int64_t group_id,
    int64_t token_idx,
    int outer_idx,
    int hidden_group,
    int hidden_dim_groups,
    int packed_hidden_dim_groups,
    int64_t aligned_num_tokens,
    int lane_id) {
  constexpr uint32_t INPUT_VEC_SIZE = INPUT_VEC_NUM_BYTES / sizeof(T);
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

  int4 output_buf;
  auto* output_buf_ptr = reinterpret_cast<fp8x2_e4m3_t*>(&output_buf);
#pragma unroll
  for (uint32_t j = 0; j < INPUT_VEC_SIZE; j += 2) {
    output_buf_ptr[j / 2] =
        pack_fp8(static_cast<float>(input_vec[j]) * y_scale, static_cast<float>(input_vec[j + 1]) * y_scale);
  }

  *reinterpret_cast<int4*>(output_q + group_id * GROUP_SIZE + lane_id * INPUT_VEC_SIZE) = output_buf;

  if (lane_id == 0) {
    const int hidden_pack_idx = hidden_group / UE8M0_SCALES_PER_PACK;
    const int pack_byte_idx = hidden_group % UE8M0_SCALES_PER_PACK;
    const int64_t scale_word_offset =
        (static_cast<int64_t>(outer_idx) * packed_hidden_dim_groups + hidden_pack_idx) * aligned_num_tokens + token_idx;
    auto* scale_output =
        reinterpret_cast<uint8_t*>(output_s) + scale_word_offset * UE8M0_SCALES_PER_PACK + pack_byte_idx;
    *scale_output = static_cast<uint8_t>(scale_ue8m0);

    // DeepGEMM consumes complete int32 packs. Zero bytes without a matching
    // hidden group so allocator garbage cannot become an activation scale.
    if (hidden_group == hidden_dim_groups - 1) {
#pragma unroll
      for (int byte_idx = pack_byte_idx + 1; byte_idx < UE8M0_SCALES_PER_PACK; ++byte_idx) {
        scale_output[byte_idx - pack_byte_idx] = 0;
      }
    }
  }
}

template <typename T, bool kUsePDL>
__global__ void fp8_wo_a_group_major_quant_ue8m0_kernel(
    const T* __restrict__ input,
    fp8_e4m3_t* __restrict__ output_q,
    int32_t* __restrict__ output_s,
    int64_t total_scale_groups,
    int hidden_dim_groups,
    int packed_hidden_dim_groups,
    int num_outer_groups,
    int64_t aligned_num_tokens,
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

    int4 input_int4[INPUT_INT4_SIZE];
    T* input_vec = reinterpret_cast<T*>(input_int4);

#pragma unroll
    for (uint32_t j = 0; j < INPUT_INT4_SIZE; ++j) {
      input_int4[j] = reinterpret_cast<const int4*>(input + input_group_start_offset + lane_id * INPUT_VEC_SIZE)[j];
    }

    QuantizeWoAGroup(
        input_vec,
        output_q,
        output_s,
        group_id,
        token_idx,
        outer_idx,
        hidden_group,
        hidden_dim_groups,
        packed_hidden_dim_groups,
        aligned_num_tokens,
        lane_id);
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <typename T>
struct FP8WoAQuantParams {
  host::SymbolicDevice device;
  int64_t num_tokens, num_outer_groups, hidden_dim, hidden_dim_groups;
  int64_t packed_hidden_dim_groups, aligned_num_tokens, input_stride_t;

  FP8WoAQuantParams(tvm::ffi::TensorView input, tvm::ffi::TensorView output_q, tvm::ffi::TensorView output_s) {
    using namespace host;

    device.set_options<kDLGPU>();
    auto TSize = SymbolicSize{"num_tokens"};
    auto GSize = SymbolicSize{"num_outer_groups"};
    auto DSize = SymbolicSize{"hidden_dim"};
    auto PSize = SymbolicSize{"packed_hidden_dim_groups"};
    auto ASize = SymbolicSize{"aligned_num_tokens"};

    TensorMatcher({TSize, GSize, DSize}).with_strides({-1, DSize, 1}).with_dtype<T>().with_device(device).verify(input);
    TensorMatcher({TSize, GSize, DSize}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output_q);
    TensorMatcher({GSize, PSize, ASize}).with_dtype<int32_t>().with_device(device).verify(output_s);

    num_tokens = TSize.unwrap();
    num_outer_groups = GSize.unwrap();
    hidden_dim = DSize.unwrap();
    packed_hidden_dim_groups = PSize.unwrap();
    aligned_num_tokens = ASize.unwrap();
    input_stride_t = input.stride(0);
    constexpr int64_t kInputAlignElements = sizeof(int4) / sizeof(T);

    RuntimeCheck(hidden_dim % GROUP_SIZE == 0, "hidden_dim must be divisible by 128");
    hidden_dim_groups = hidden_dim / GROUP_SIZE;
    RuntimeCheck(
        packed_hidden_dim_groups == (hidden_dim_groups + UE8M0_SCALES_PER_PACK - 1) / UE8M0_SCALES_PER_PACK,
        "packed output_s hidden dim mismatch");
    RuntimeCheck(
        aligned_num_tokens == (num_tokens + UE8M0_SCALES_PER_PACK - 1) / UE8M0_SCALES_PER_PACK * UE8M0_SCALES_PER_PACK,
        "output_s token dim must be TMA-aligned");
    RuntimeCheck(
        output_s.stride(0) == packed_hidden_dim_groups * aligned_num_tokens &&
            output_s.stride(1) == aligned_num_tokens && output_s.stride(2) == 1,
        "output_s must use contiguous [group, packed_hidden, aligned_token] storage");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(input.data_ptr()) % sizeof(int4) == 0,
        "input base pointer must be 16-byte aligned");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(output_s.data_ptr()) % sizeof(int4) == 0,
        "output_s base pointer must be 16-byte aligned");
    RuntimeCheck(
        num_tokens <= 1 || input_stride_t % kInputAlignElements == 0,
        "input token stride must preserve 16-byte vector-load alignment");
  }

  int64_t total_scale_groups() const {
    return num_tokens * num_outer_groups * hidden_dim_groups;
  }
};

template <typename T, bool kUsePDL>
struct FP8WoAGroupMajorQuantUE8M0Kernel {
  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView output_q, tvm::ffi::TensorView output_s) {
    auto params = FP8WoAQuantParams<T>{input, output_q, output_s};
    const auto total_scale_groups = params.total_scale_groups();
    if (total_scale_groups == 0) return;

    const auto grid = dim3((total_scale_groups + SUBWARPS_PER_BLOCK - 1) / SUBWARPS_PER_BLOCK);
    const auto block = dim3(SUBWARPS_PER_BLOCK * THREADS_PER_GROUP);
    host::LaunchKernel(grid, block, params.device.unwrap())
        .enable_pdl(kUsePDL)(
            fp8_wo_a_group_major_quant_ue8m0_kernel<T, kUsePDL>,
            static_cast<const T*>(input.data_ptr()),
            static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
            static_cast<int32_t*>(output_s.data_ptr()),
            total_scale_groups,
            static_cast<int>(params.hidden_dim_groups),
            static_cast<int>(params.packed_hidden_dim_groups),
            static_cast<int>(params.num_outer_groups),
            static_cast<int64_t>(params.aligned_num_tokens),
            static_cast<int64_t>(params.input_stride_t));
  }
};

// Fused inverse RoPE with vectorized frequency loads and early dependency loads.
template <
    typename T,
    bool kUsePDL,
    uint32_t kGPT,
    uint32_t kSubwarps,
    bool kStreamLoad,
    bool kPrefetchRope,
    uint32_t kMinBlocks>
__global__ void __launch_bounds__(kSubwarps* THREADS_PER_GROUP, kMinBlocks)
    fp8_wo_a_group_major_quant_ue8m0_rope_kernel(
        const T* __restrict__ input,
        fp8_e4m3_t* __restrict__ output_q,
        int32_t* __restrict__ output_s,
        uint32_t total_scale_groups,
        uint32_t num_tokens,
        uint32_t hidden_dim_groups,
        int packed_hidden_dim_groups,
        uint32_t num_outer_groups,
        int64_t aligned_num_tokens,
        int64_t input_stride_t,
        const float* __restrict__ freqs_cis,
        const int64_t* __restrict__ positions) {
  constexpr int kRopeDim = 64;
  constexpr int kHeadDim = 512;
  constexpr uint32_t INPUT_VEC_SIZE = INPUT_VEC_NUM_BYTES / sizeof(T);
  static_assert(INPUT_VEC_SIZE * THREADS_PER_GROUP == GROUP_SIZE);
  static_assert(INPUT_VEC_SIZE == 16, "inverse RoPE assumes 8 (cos, sin) pairs per lane");

  const int subwarp_id = threadIdx.x / THREADS_PER_GROUP;
  const int lane_id = threadIdx.x % THREADS_PER_GROUP;
  const uint32_t block_base = blockIdx.x * (kSubwarps * kGPT) + subwarp_id;

  int4 input_int4[kGPT][INPUT_INT4_SIZE];
  float4 freq[kGPT][INPUT_VEC_SIZE / 4];
  int64_t position[kGPT];
  uint32_t token_idx[kGPT], outer_idx[kGPT], hidden_group[kGPT];
  int pair0[kGPT];
  bool valid[kGPT], do_rope[kGPT];

  // Phase 1: index math (no global reads).
#pragma unroll
  for (uint32_t k = 0; k < kGPT; ++k) {
    const uint32_t group_id = block_base + k * kSubwarps;
    valid[k] = group_id < total_scale_groups;
    hidden_group[k] = group_id % hidden_dim_groups;
    const uint32_t token_outer = group_id / hidden_dim_groups;
    outer_idx[k] = token_outer % num_outer_groups;
    token_idx[k] = token_outer / num_outer_groups;
    const int col_in_head = (hidden_group[k] * GROUP_SIZE + lane_id * INPUT_VEC_SIZE) % kHeadDim;
    do_rope[k] = valid[k] && col_in_head >= kHeadDim - kRopeDim;
    pair0[k] = (col_in_head - (kHeadDim - kRopeDim)) / 2;
  }

  // positions -> freqs is a dependent 2-load chain; with kPrefetchRope it is
  // issued ahead of the input loads so its latency overlaps them.
  auto load_rope = [&]() {
#pragma unroll
    for (uint32_t k = 0; k < kGPT; ++k)
      position[k] = do_rope[k] ? __ldg(positions + token_idx[k]) : 0;
#pragma unroll
    for (uint32_t k = 0; k < kGPT; ++k) {
      if (do_rope[k]) {
        const float4* f = reinterpret_cast<const float4*>(freqs_cis + position[k] * kRopeDim + 2 * pair0[k]);
#pragma unroll
        for (uint32_t q = 0; q < INPUT_VEC_SIZE / 4; ++q)
          freq[k][q] = __ldg(f + q);
      }
    }
  };
  device::PDLWaitPrimary<kUsePDL>();

  if constexpr (kPrefetchRope) load_rope();

  // Phase 2: input loads for all kGPT groups.
#pragma unroll
  for (uint32_t k = 0; k < kGPT; ++k) {
    if (valid[k]) {
      const int4* src = reinterpret_cast<const int4*>(
          input + static_cast<int64_t>(token_idx[k]) * input_stride_t + outer_idx[k] * GROUP_SIZE * hidden_dim_groups +
          hidden_group[k] * GROUP_SIZE + lane_id * INPUT_VEC_SIZE);
#pragma unroll
      for (uint32_t j = 0; j < INPUT_INT4_SIZE; ++j) {
        if constexpr (kStreamLoad) {
          input_int4[k][j] = __ldcs(src + j);
        } else {
          input_int4[k][j] = src[j];
        }
      }
    }
  }
  if constexpr (!kPrefetchRope) load_rope();

  // Phase 3: rope + absmax + UE8M0 quant, preserving the BF16 rounding boundary.
#pragma unroll
  for (uint32_t k = 0; k < kGPT; ++k) {
    if (!valid[k]) continue;
    T* input_vec = reinterpret_cast<T*>(input_int4[k]);
    if (do_rope[k]) {
      const float* fr = reinterpret_cast<const float*>(freq[k]);
#pragma unroll
      for (uint32_t p = 0; p < INPUT_VEC_SIZE / 2; ++p) {
        const float c = fr[2 * p];
        const float s = fr[2 * p + 1];
        const float a = static_cast<float>(input_vec[2 * p]);
        const float b = static_cast<float>(input_vec[2 * p + 1]);
        input_vec[2 * p] = static_cast<T>(a * c + b * s);
        input_vec[2 * p + 1] = static_cast<T>(b * c - a * s);
      }
    }

    const int64_t group_id = block_base + k * kSubwarps;
    QuantizeWoAGroup(
        input_vec,
        output_q,
        output_s,
        group_id,
        token_idx[k],
        outer_idx[k],
        hidden_group[k],
        hidden_dim_groups,
        packed_hidden_dim_groups,
        aligned_num_tokens,
        lane_id);
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <typename T, bool kUsePDL>
struct FP8WoAGroupMajorQuantUE8M0RopeKernel {
  static void
  run(tvm::ffi::TensorView input,
      tvm::ffi::TensorView freqs_cis,
      tvm::ffi::TensorView positions,
      tvm::ffi::TensorView output_q,
      tvm::ffi::TensorView output_s) {
    using namespace host;
    auto params = FP8WoAQuantParams<T>{input, output_q, output_s};
    auto FSize = SymbolicSize{"num_freqs"};
    TensorMatcher({FSize, 64}).with_dtype<float>().with_device(params.device).verify(freqs_cis);
    TensorMatcher({params.num_tokens}).with_dtype<int64_t>().with_device(params.device).verify(positions);
    RuntimeCheck(params.hidden_dim % 512 == 0, "inverse RoPE requires 512-dimensional heads");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(freqs_cis.data_ptr()) % sizeof(float4) == 0, "freqs_cis must be 16-byte aligned");
    const auto total_scale_groups = params.total_scale_groups();
    if (total_scale_groups == 0) return;
    RuntimeCheck(total_scale_groups < (int64_t{1} << 31), "fused kernel uses 32-bit group indices");

    const auto grid = dim3((total_scale_groups + SUBWARPS_PER_BLOCK - 1) / SUBWARPS_PER_BLOCK);
    const auto block = dim3(SUBWARPS_PER_BLOCK * THREADS_PER_GROUP);
    host::LaunchKernel(grid, block, params.device.unwrap())
        .enable_pdl(kUsePDL)(
            fp8_wo_a_group_major_quant_ue8m0_rope_kernel<T, kUsePDL, 1, 16, false, true, 16>,
            static_cast<const T*>(input.data_ptr()),
            static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
            static_cast<int32_t*>(output_s.data_ptr()),
            static_cast<uint32_t>(total_scale_groups),
            static_cast<uint32_t>(params.num_tokens),
            static_cast<int>(params.hidden_dim_groups),
            static_cast<int>(params.packed_hidden_dim_groups),
            static_cast<int>(params.num_outer_groups),
            static_cast<int64_t>(params.aligned_num_tokens),
            static_cast<int64_t>(params.input_stride_t),
            static_cast<const float*>(freqs_cis.data_ptr()),
            static_cast<const int64_t*>(positions.data_ptr()));
  }
};

}  // namespace sglang
