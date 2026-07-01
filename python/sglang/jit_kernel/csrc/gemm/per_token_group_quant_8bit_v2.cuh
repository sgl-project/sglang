// JIT port of the AOT sgl_per_token_group_quant_8bit_v2 (sgl-kernel).
//
// Same math as the AOT v2 kernel (256-bit vectorized loads, 8 threads/128-group,
// PDL, NaiveScheduler + MaskedLayoutScheduler, ue8m0/float scales, fp8/int8
// output, fused silu+mul) so it is a drop-in replacement; the only changes vs the
// AOT source are the launcher (tvm::ffi::TensorView + TensorMatcher + the JIT
// LaunchKernel/PDL helpers) and the FP8 type alias.
#include <sgl_kernel/tensor.h>  // TensorMatcher, SymbolicSize/Device
#include <sgl_kernel/utils.h>   // RuntimeCheck, Panic

#include <sgl_kernel/utils.cuh>  // LaunchKernel, fp8_e4m3_t, SGL_DEVICE, device::PDLWaitPrimary/TriggerSecondary
#include <sgl_kernel/warp.cuh>   // device::warp::reduce_max

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp8.h>
#include <type_traits>

namespace {

constexpr float LOCAL_ABSMAX_ABS = 1e-10f;
constexpr uint32_t INPUT_PRIMARY_VEC_NUM_BYTES = 32;

template <int THREADS_PER_SUBWARP>
SGL_DEVICE float GroupReduceMax(float val) {
  static_assert(
      (THREADS_PER_SUBWARP & (THREADS_PER_SUBWARP - 1)) == 0 && THREADS_PER_SUBWARP <= 16 && THREADS_PER_SUBWARP >= 1,
      "THREADS_PER_SUBWARP must be 1, 2, 4, 8, or 16");
  // Reduce within this thread's contiguous THREADS_PER_SUBWARP-lane subgroup via
  // the shared warp primitive, but pass an explicit subgroup mask instead of its
  // default 0xffffffff: the block can be < 32 lanes (subwarps_per_block *
  // THREADS_PER_SUBWARP, e.g. 1..16), where a full-warp mask names non-existent
  // lanes and is UB / can hang.
  constexpr device::warp::mask_t kSub = (device::warp::mask_t{1} << THREADS_PER_SUBWARP) - 1;
  const device::warp::mask_t mask = kSub << (THREADS_PER_SUBWARP * ((threadIdx.x % 32) / THREADS_PER_SUBWARP));
  return device::warp::reduce_max<THREADS_PER_SUBWARP>(val, mask);
}

SGL_DEVICE float silu(const float& val) {
  // Match the AOT v2 kernel: tanh-based silu on SM100+ (Blackwell), exp-based
  // elsewhere, so the fused silu+mul output stays bit-identical to the AOT op.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  float half = 0.5f * val;
  float t = __tanhf(half);
  return half * (1.0f + t);
#else
  return val / (1.0f + __expf(-val));
#endif
}

SGL_DEVICE float2 fmul2_rn(float2 a, float2 b) {
  // Match the AOT v2 kernel: use the __fmul2_rn intrinsic on SM100+.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  return __fmul2_rn(a, b);
#else
  float2 result;
  result.x = a.x * b.x;
  result.y = a.y * b.y;
  return result;
#endif
}

// Copied from DeepEP.
SGL_DEVICE float fast_pow2(int x) {
  uint32_t bits_x = (x + 127) << 23;
  return __uint_as_float(bits_x);  // type-safe bit cast (no strict-aliasing UB)
}
SGL_DEVICE int fast_log2_ceil(float x) {
  auto bits_x = __float_as_uint(x);  // type-safe bit cast (no strict-aliasing UB)
  auto exp_x = (bits_x >> 23) & 0xff;
  auto man_bits = bits_x & ((1 << 23) - 1);
  return exp_x - 127 + (man_bits != 0);
}

template <typename T>
struct DtypeInfo;
template <>
struct DtypeInfo<int8_t> {
  static constexpr float MIN = -128;
  static constexpr float MAX = 127;
};
template <>
struct DtypeInfo<fp8_e4m3_t> {
  static constexpr float MIN = -448;
  static constexpr float MAX = 448;
};

template <bool ROUND_SCALE, typename dtype_info>
SGL_DEVICE void calculate_fp8_scales(float amax, float& scale, float& scale_inv) {
  constexpr float MAX_8BIT_INV = 1.0f / dtype_info::MAX;
  if constexpr (ROUND_SCALE) {
    auto exp_scale_inv = fast_log2_ceil(amax * MAX_8BIT_INV);
    scale = fast_pow2(-exp_scale_inv);
    scale_inv = fast_pow2(exp_scale_inv);
  } else {
    scale_inv = amax * MAX_8BIT_INV;
    scale = dtype_info::MAX / amax;
  }
}

template <bool SCALE_UE8M0, typename OUT_DTYPE_T = std::conditional_t<SCALE_UE8M0, uint8_t, float>>
SGL_DEVICE OUT_DTYPE_T extract_required_scale_format(float value) {
  if constexpr (SCALE_UE8M0) {
    return static_cast<uint8_t>(__float_as_uint(value) >> 23);
  } else {
    return value;
  }
}

template <bool FUSE_SILU_AND_MUL>
SGL_DEVICE int compute_input_group_start_offset(
    int expert_idx,
    int token_idx,
    int hidden_dim_group_idx,
    int hidden_size,
    int num_tokens_per_expert,
    int group_size) {
  return expert_idx * num_tokens_per_expert * hidden_size * (FUSE_SILU_AND_MUL ? 2 : 1) +
         token_idx * hidden_size * (FUSE_SILU_AND_MUL ? 2 : 1) + hidden_dim_group_idx * group_size;
}

struct NaiveScheduler {
  static void compute_exec_config(
      int threads_per_subwarp,
      int num_local_experts,
      int hidden_dim_num_groups,
      int num_groups,
      int& subwarps_per_block,
      dim3& grid,
      dim3& block) {
    subwarps_per_block = (num_groups % 16 == 0)  ? 16
                         : (num_groups % 8 == 0) ? 8
                         : (num_groups % 4 == 0) ? 4
                         : (num_groups % 2 == 0) ? 2
                                                 : 1;
    grid = dim3(num_groups / subwarps_per_block);
    block = dim3(subwarps_per_block * threads_per_subwarp);
  }

  template <bool FUSE_SILU_AND_MUL, int GROUP_SIZE, int THREADS_PER_SUBWARP, typename FUNC>
  SGL_DEVICE static void execute(
      const int subwarps_per_block,
      const int hidden_dim_num_groups,
      const int32_t* masked_m,
      const int num_tokens_per_expert,
      FUNC fn) {
    constexpr int expert_idx = 0;
    const int64_t subwarp_id = threadIdx.x / THREADS_PER_SUBWARP;
    const int lane_id = threadIdx.x % THREADS_PER_SUBWARP;
    const int64_t group_id = static_cast<int64_t>(blockIdx.x) * subwarps_per_block + subwarp_id;

    int64_t input_group_start_offset;
    if constexpr (!FUSE_SILU_AND_MUL) input_group_start_offset = group_id * GROUP_SIZE;
    const int token_idx = group_id / hidden_dim_num_groups;
    const int hidden_dim_group_idx = group_id % hidden_dim_num_groups;
    if constexpr (FUSE_SILU_AND_MUL) {
      const int hidden_size = hidden_dim_num_groups * GROUP_SIZE;
      input_group_start_offset = compute_input_group_start_offset<FUSE_SILU_AND_MUL>(
          expert_idx, token_idx, hidden_dim_group_idx, hidden_size, num_tokens_per_expert, GROUP_SIZE);
    }
    fn(expert_idx, token_idx, hidden_dim_group_idx, lane_id, input_group_start_offset);
  }
};

struct MaskedLayoutScheduler {
  static constexpr int TOKEN_DIM_BLOCK_NUM_PER_EXPERT = 1024;
  static constexpr int SUBWARPS_PER_BLOCK = 16;

  static void compute_exec_config(
      int threads_per_subwarp,
      int num_local_experts,
      int hidden_dim_num_groups,
      int num_groups,
      int& subwarps_per_block,
      dim3& grid,
      dim3& block) {
    subwarps_per_block = SUBWARPS_PER_BLOCK;
    host::RuntimeCheck(hidden_dim_num_groups % subwarps_per_block == 0, "hidden_dim_num_groups not divisible by 16");
    grid = dim3(hidden_dim_num_groups / subwarps_per_block, TOKEN_DIM_BLOCK_NUM_PER_EXPERT, num_local_experts);
    block = dim3(subwarps_per_block * threads_per_subwarp);
  }

  template <bool FUSE_SILU_AND_MUL, int GROUP_SIZE, int THREADS_PER_SUBWARP, typename FUNC>
  SGL_DEVICE static void execute(
      const int subwarps_per_block,
      const int hidden_dim_num_groups,
      const int32_t* masked_m,
      const int num_tokens_per_expert,
      FUNC fn) {
    const int64_t subwarp_id = threadIdx.x / THREADS_PER_SUBWARP;
    const int lane_id = threadIdx.x % THREADS_PER_SUBWARP;
    const int expert_idx = blockIdx.z;
    const int token_idx_start = blockIdx.y;
    const int64_t hidden_dim_group_idx = static_cast<int64_t>(blockIdx.x) * SUBWARPS_PER_BLOCK + subwarp_id;
    const int curr_expert_token_num = masked_m[expert_idx];
    for (int token_idx = token_idx_start; token_idx < curr_expert_token_num;
         token_idx += TOKEN_DIM_BLOCK_NUM_PER_EXPERT) {
      const int hidden_size = hidden_dim_num_groups * GROUP_SIZE;
      const int64_t input_group_start_offset = compute_input_group_start_offset<FUSE_SILU_AND_MUL>(
          expert_idx, token_idx, hidden_dim_group_idx, hidden_size, num_tokens_per_expert, GROUP_SIZE);
      fn(expert_idx, token_idx, hidden_dim_group_idx, lane_id, input_group_start_offset);
    }
  }
};

template <
    typename SCHEDULER,
    int GROUP_SIZE,
    int THREADS_PER_SUBWARP,
    typename T,
    typename DST_DTYPE,
    bool IS_COLUMN_MAJOR,
    bool SCALE_UE8M0,
    bool FUSE_SILU_AND_MUL,
    bool kUsePDL,
    typename scale_packed_t = std::conditional_t<SCALE_UE8M0 && IS_COLUMN_MAJOR, uint32_t, float>>
__global__ void per_token_group_quant_8bit_v2_kernel(
    const T* __restrict__ input,
    DST_DTYPE* __restrict__ output_q,
    scale_packed_t* __restrict__ output_s,
    const int32_t* __restrict__ masked_m,
    const int subwarps_per_block,
    const int hidden_dim_num_groups,
    const int scale_expert_stride,
    const int scale_hidden_stride,
    const int num_tokens_per_expert) {
  using dst_dtype_info = DtypeInfo<DST_DTYPE>;
  using scale_element_t = std::conditional_t<SCALE_UE8M0 && IS_COLUMN_MAJOR, uint8_t, float>;
  static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);

  device::PDLWaitPrimary<kUsePDL>();

  SCHEDULER::template execute<FUSE_SILU_AND_MUL, GROUP_SIZE, THREADS_PER_SUBWARP>(
      subwarps_per_block,
      hidden_dim_num_groups,
      masked_m,
      num_tokens_per_expert,
      [&](const int expert_idx,
          const int token_idx,
          const int hidden_dim_group_idx,
          const int lane_id,
          const int input_group_start_offset) {
        constexpr uint32_t INPUT_PRIMARY_VEC_SIZE = INPUT_PRIMARY_VEC_NUM_BYTES / sizeof(T);
        constexpr uint32_t INPUT_PRIMARY_INT4_SIZE = INPUT_PRIMARY_VEC_NUM_BYTES / sizeof(int4);

        const int offset_num_groups = expert_idx * num_tokens_per_expert * hidden_dim_num_groups +
                                      token_idx * hidden_dim_num_groups + hidden_dim_group_idx;

        int4 input_primary_int4[INPUT_PRIMARY_INT4_SIZE];
        T* input_primary_vec = reinterpret_cast<T*>(input_primary_int4);
        int4 input_secondary_int4[INPUT_PRIMARY_INT4_SIZE];
        T* input_secondary_vec = reinterpret_cast<T*>(input_secondary_int4);

#pragma unroll
        for (uint32_t j = 0; j < INPUT_PRIMARY_INT4_SIZE; ++j) {
          // Ordinary 128-bit vectorized load (LDG.128); .nc gave no measurable
          // gain on this streaming read-once kernel, so no inline asm.
          input_primary_int4[j] =
              reinterpret_cast<const int4*>(input + input_group_start_offset + lane_id * INPUT_PRIMARY_VEC_SIZE)[j];
        }
        if constexpr (FUSE_SILU_AND_MUL) {
          const int secondary_offset = hidden_dim_num_groups * GROUP_SIZE;
#pragma unroll
          for (uint32_t j = 0; j < INPUT_PRIMARY_INT4_SIZE; ++j) {
            input_secondary_int4[j] = reinterpret_cast<const int4*>(
                input + input_group_start_offset + lane_id * INPUT_PRIMARY_VEC_SIZE + secondary_offset)[j];
          }
        }

        constexpr int num_elems_per_pack = static_cast<int>(sizeof(scale_packed_t) / sizeof(scale_element_t));
        scale_element_t* scale_output;
        if constexpr (IS_COLUMN_MAJOR) {
          constexpr int column_major_scale_token_stride = 1;
          const int hidden_idx_packed = hidden_dim_group_idx / num_elems_per_pack;
          const int pack_idx = hidden_dim_group_idx % num_elems_per_pack;
          scale_output = reinterpret_cast<scale_element_t*>(output_s) +
                         (expert_idx * scale_expert_stride * num_elems_per_pack +
                          hidden_idx_packed * scale_hidden_stride * num_elems_per_pack +
                          token_idx * column_major_scale_token_stride * num_elems_per_pack + pack_idx);
        } else {
          static_assert(!SCALE_UE8M0 || std::is_same_v<scale_packed_t, float>);
          if (scale_expert_stride > 0) {
            // Non-masked 3D row-major output_s uses the existing stride slots
            // for the new outer-major layout: (outer_stride, token_stride).
            const int scale_outer_stride = scale_expert_stride;
            const int scale_token_stride = scale_hidden_stride;
            const int outer_idx = token_idx / num_tokens_per_expert;
            const int token_idx_in_outer = token_idx % num_tokens_per_expert;
            const int64_t scale_offset = (expert_idx + outer_idx) * scale_outer_stride +
                                         token_idx_in_outer * scale_token_stride + hidden_dim_group_idx;
            scale_output = reinterpret_cast<scale_element_t*>(output_s) + scale_offset;
          } else {
            scale_output = reinterpret_cast<scale_element_t*>(output_s) + offset_num_groups;
          }
        }

        if constexpr (IS_COLUMN_MAJOR and SCALE_UE8M0) {
          const int remainder_num_groups = hidden_dim_num_groups % num_elems_per_pack;
          if ((remainder_num_groups != 0) and (hidden_dim_group_idx == hidden_dim_num_groups - 1) and
              (lane_id < num_elems_per_pack - remainder_num_groups)) {
            const int shift = 1 + lane_id;
            *(scale_output + shift) = 0;
          }
        }

        float local_absmax = LOCAL_ABSMAX_ABS;
#pragma unroll
        for (uint32_t j = 0; j < INPUT_PRIMARY_VEC_SIZE; ++j) {
          float val;
          if constexpr (FUSE_SILU_AND_MUL) {
            T val_lowprec = static_cast<T>(silu(static_cast<float>(input_primary_vec[j]))) * input_secondary_vec[j];
            val = static_cast<float>(val_lowprec);
            input_primary_vec[j] = val_lowprec;
          } else {
            val = static_cast<float>(input_primary_vec[j]);
          }
          local_absmax = fmaxf(local_absmax, fabsf(val));
        }

        local_absmax = GroupReduceMax<THREADS_PER_SUBWARP>(local_absmax);

        float y_scale, y_scale_inv;
        // When SCALE_UE8M0, always quantize with the rounded (power-of-2) scale
        // — not with the exact scale followed by post-hoc rounding.
        // This matches the official DeepSeek-V4 kernel.py act_quant(scale_fmt="ue8m0")
        // and avoids a scale mismatch between quantization and downstream GEMM dequant,
        // which otherwise amplifies error ~14x and degrades EAGLE accept rate on Blackwell.
        calculate_fp8_scales<SCALE_UE8M0, dst_dtype_info>(local_absmax, y_scale, y_scale_inv);
        if (lane_id == 0) {
          *scale_output = extract_required_scale_format < SCALE_UE8M0 && IS_COLUMN_MAJOR > (y_scale_inv);
        }
        float2 y_scale_repeated = {y_scale, y_scale};

        int4 output_buf;
        if constexpr (std::is_same_v<DST_DTYPE, fp8_e4m3_t>) {
          const auto output_buf_ptr = reinterpret_cast<__nv_fp8x2_storage_t*>(&output_buf);
#pragma unroll
          for (uint32_t j = 0; j < INPUT_PRIMARY_VEC_SIZE; j += 2) {
            float2 inputx2 = {static_cast<float>(input_primary_vec[j]), static_cast<float>(input_primary_vec[j + 1])};
            float2 outputx2 = fmul2_rn(inputx2, y_scale_repeated);
            outputx2.x = fminf(fmaxf(outputx2.x, dst_dtype_info::MIN), dst_dtype_info::MAX);
            outputx2.y = fminf(fmaxf(outputx2.y, dst_dtype_info::MIN), dst_dtype_info::MAX);
            output_buf_ptr[j / 2] = __nv_cvt_float2_to_fp8x2(outputx2, __NV_SATFINITE, __NV_E4M3);
          }
        } else {
          const auto output_buf_ptr = reinterpret_cast<DST_DTYPE*>(&output_buf);
#pragma unroll
          for (uint32_t j = 0; j < INPUT_PRIMARY_VEC_SIZE; ++j) {
            float val = static_cast<float>(input_primary_vec[j]);
            float q_val = fminf(fmaxf(val * y_scale, dst_dtype_info::MIN), dst_dtype_info::MAX);
            output_buf_ptr[j] = DST_DTYPE(q_val);
          }
        }

        // Ordinary 128-bit vectorized store (STG.128); no inline asm.
        *reinterpret_cast<int4*>(output_q + offset_num_groups * GROUP_SIZE + lane_id * INPUT_PRIMARY_VEC_SIZE) =
            output_buf;
      });

  device::PDLTriggerSecondary<kUsePDL>();
}

// ----------------------------------------------------------------------------
// Launcher (JIT). All shape-derived scalars are computed in the Python wrapper
// and passed in, so the C++ side only needs TensorView::data_ptr()/device().
// Runtime combos (column_major / ue8m0 / silu / masked / group_size) are
// dispatched to the templated kernel; launch is PDL-aware via LaunchKernel.
// ----------------------------------------------------------------------------
template <typename S>
struct TypeTag {
  using type = S;
};

template <typename T, typename DST_DTYPE, bool kUsePDL>
struct PerTokenGroupQuant8bitV2Kernel {
  template <
      typename SCHEDULER,
      int GROUP_SIZE,
      int THREADS_PER_SUBWARP,
      bool IS_COLUMN_MAJOR,
      bool SCALE_UE8M0,
      bool FUSE_SILU_AND_MUL>
  static void launch(
      const DLDevice& device,
      dim3 grid,
      dim3 block,
      int subwarps_per_block,
      int hidden_dim_num_groups,
      int scale_expert_stride,
      int scale_hidden_stride,
      int num_tokens_per_expert,
      const void* input,
      void* output_q,
      void* output_s,
      const int32_t* masked_m) {
    using scale_packed_t = std::conditional_t<SCALE_UE8M0 && IS_COLUMN_MAJOR, uint32_t, float>;
    auto kernel = per_token_group_quant_8bit_v2_kernel<
        SCHEDULER,
        GROUP_SIZE,
        THREADS_PER_SUBWARP,
        T,
        DST_DTYPE,
        IS_COLUMN_MAJOR,
        SCALE_UE8M0,
        FUSE_SILU_AND_MUL,
        kUsePDL>;
    host::LaunchKernel(grid, block, device)
        .enable_pdl(kUsePDL)(
            kernel,
            static_cast<const T*>(input),
            static_cast<DST_DTYPE*>(output_q),
            static_cast<scale_packed_t*>(output_s),
            masked_m,
            subwarps_per_block,
            hidden_dim_num_groups,
            scale_expert_stride,
            scale_hidden_stride,
            num_tokens_per_expert);
  }

  template <int GROUP_SIZE>
  static void dispatch_bools(
      const DLDevice& device,
      bool is_column_major,
      bool scale_ue8m0,
      bool fuse_silu_and_mul,
      bool masked_layout,
      int num_local_experts,
      int hidden_dim_num_groups,
      int num_groups,
      int scale_expert_stride,
      int scale_hidden_stride,
      int num_tokens_per_expert,
      const void* input,
      void* output_q,
      void* output_s,
      const int32_t* masked_m) {
    constexpr int THREADS_PER_SUBWARP = GROUP_SIZE / 16;

    auto launch_with_config = [&](auto sched_tag, auto colmajor_tag, auto ue8m0_tag, auto silu_tag) {
      using SCHEDULER = typename decltype(sched_tag)::type;
      int subwarps_per_block;
      dim3 grid, block;
      SCHEDULER::compute_exec_config(
          THREADS_PER_SUBWARP, num_local_experts, hidden_dim_num_groups, num_groups, subwarps_per_block, grid, block);
      launch<
          SCHEDULER,
          GROUP_SIZE,
          THREADS_PER_SUBWARP,
          decltype(colmajor_tag)::value,
          decltype(ue8m0_tag)::value,
          decltype(silu_tag)::value>(
          device,
          grid,
          block,
          subwarps_per_block,
          hidden_dim_num_groups,
          scale_expert_stride,
          scale_hidden_stride,
          num_tokens_per_expert,
          input,
          output_q,
          output_s,
          masked_m);
    };

    if (is_column_major) {
      if (scale_ue8m0) {
        if (fuse_silu_and_mul) {
          if (masked_layout)
            launch_with_config(TypeTag<MaskedLayoutScheduler>{}, std::true_type{}, std::true_type{}, std::true_type{});
          else
            launch_with_config(TypeTag<NaiveScheduler>{}, std::true_type{}, std::true_type{}, std::true_type{});
        } else {
          launch_with_config(TypeTag<NaiveScheduler>{}, std::true_type{}, std::true_type{}, std::false_type{});
        }
      } else {
        launch_with_config(TypeTag<NaiveScheduler>{}, std::true_type{}, std::false_type{}, std::false_type{});
      }
    } else {
      if (scale_ue8m0) {
        launch_with_config(TypeTag<NaiveScheduler>{}, std::false_type{}, std::true_type{}, std::false_type{});
      } else {
        launch_with_config(TypeTag<NaiveScheduler>{}, std::false_type{}, std::false_type{}, std::false_type{});
      }
    }
  }

  static void
  run(tvm::ffi::TensorView input,
      tvm::ffi::TensorView output_q,
      tvm::ffi::TensorView output_s,
      tvm::ffi::TensorView masked_m,
      int64_t group_size,
      bool scale_ue8m0,
      bool fuse_silu_and_mul,
      bool masked_layout,
      int64_t num_groups,
      int64_t num_local_experts,
      bool is_column_major,
      int64_t hidden_dim_num_groups,
      int64_t num_tokens_per_expert,
      int64_t scale_expert_stride,
      int64_t scale_hidden_stride) {
    const DLDevice dev = input.device();
    const void* in = input.data_ptr();
    void* oq = output_q.data_ptr();
    void* os = output_s.data_ptr();
    const int32_t* masked_ptr = masked_layout ? static_cast<const int32_t*>(masked_m.data_ptr()) : nullptr;

    auto dispatch_gs = [&](auto gs_tag) {
      constexpr int GS = decltype(gs_tag)::value;
      static_assert((GS / 16) * INPUT_PRIMARY_VEC_NUM_BYTES == GS * static_cast<int>(sizeof(T)));
      dispatch_bools<GS>(
          dev,
          is_column_major,
          scale_ue8m0,
          fuse_silu_and_mul,
          masked_layout,
          static_cast<int>(num_local_experts),
          static_cast<int>(hidden_dim_num_groups),
          static_cast<int>(num_groups),
          static_cast<int>(scale_expert_stride),
          static_cast<int>(scale_hidden_stride),
          static_cast<int>(num_tokens_per_expert),
          in,
          oq,
          os,
          masked_ptr);
    };
    switch (group_size) {
      case 16:
        dispatch_gs(std::integral_constant<int, 16>{});
        break;
      case 32:
        dispatch_gs(std::integral_constant<int, 32>{});
        break;
      case 64:
        dispatch_gs(std::integral_constant<int, 64>{});
        break;
      case 128:
        dispatch_gs(std::integral_constant<int, 128>{});
        break;
      default:
        host::Panic("Unsupported group_size ", group_size);
    }
  }
};

}  // namespace
