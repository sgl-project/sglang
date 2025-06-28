#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>

#include <cmath>
#include <flashinfer/vec_dtypes.cuh>

#include "utils.h"

__device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
  unsigned mask = 0xffff;

  val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  return val;
}

// Copied and modified from DeepEP
__forceinline__ __device__ float fast_pow2(int x) {
  // We can ensure `-126 <= x and x <= 127`
  uint32_t bits_x = (x + 127) << 23;
  return *reinterpret_cast<float*>(&bits_x);
}

// Copied and modified from DeepEP
__forceinline__ __device__ int fast_log2_ceil(float x) {
  auto bits_x = *reinterpret_cast<uint32_t*>(&x);
  auto exp_x = (bits_x >> 23) & 0xff;
  auto man_bits = bits_x & ((1 << 23) - 1);
  return exp_x - 127 + (man_bits != 0);
}

// Copied and modified from DeepEP
template <bool ROUND_SCALE>
__forceinline__ __device__ void
calculate_fp8_scales(float amax, float& scale, float& scale_inv, float max_8bit, float max_8bit_inv) {
  if constexpr (ROUND_SCALE) {
    auto exp_scale_inv = fast_log2_ceil(amax * max_8bit_inv);
    scale = fast_pow2(-exp_scale_inv);
    scale_inv = fast_pow2(exp_scale_inv);
  } else {
    scale_inv = amax * max_8bit_inv;
    scale = max_8bit / amax;
  }
}

// Copied and modified from DeepEP
template <bool SCALE_UE8M0, typename OUT_DTYPE_T = std::conditional_t<SCALE_UE8M0, uint8_t, float>>
__forceinline__ __device__ OUT_DTYPE_T extract_required_scale_format(float value) {
  if constexpr (SCALE_UE8M0) {
    return static_cast<uint8_t>((*reinterpret_cast<uint32_t*>(&value)) >> 23);
  } else {
    return value;
  }
}

__device__ __forceinline__ void st_global_v2_s32(const int2* ptr, const int2& value) {
  asm volatile("st.global.v2.s32 [%0], {%1, %2};" ::"l"(ptr), "r"(value.x), "r"(value.y));
}

template <typename T>
struct Vec2Type;

template <>
struct Vec2Type<__nv_bfloat16> {
    using type = __nv_bfloat162;
};

template <>
struct Vec2Type<__half> {
    using type = __half2;
};

template <
    typename T,
    typename DST_DTYPE,
    bool IS_COLUMN_MAJOR = false,
    bool SCALE_UE8M0 = false,
    typename scale_packed_t = std::conditional_t<SCALE_UE8M0, uint32_t, float>>
__global__ void per_token_group_quant_8bit_kernel(
    const T* __restrict__ input,
    void* __restrict__ output_q,
    scale_packed_t* __restrict__ output_s,
    const int group_size,
    const int num_groups,
    const int groups_per_block,
    const float eps,
    const float min_8bit,
    const float max_8bit,
    const float max_8bit_inv,
    const int scale_num_rows = 0,
    const int scale_stride = 0) {
  const int threads_per_group = 16;
  const int local_group_id = threadIdx.x / threads_per_group;
  const int lane_id = threadIdx.x % threads_per_group;

  const int block_group_id = blockIdx.x * groups_per_block;
  const int global_group_id = block_group_id + local_group_id;
  const int block_group_offset = global_group_id * group_size;

  using scale_element_t = std::conditional_t<SCALE_UE8M0, uint8_t, float>;
  static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);

  using Tx2 = typename Vec2Type<T>::type;

  const T* group_input = input + block_group_offset;
  DST_DTYPE* group_output = static_cast<DST_DTYPE*>(output_q) + block_group_offset;
  scale_element_t* scale_output;

  if constexpr (IS_COLUMN_MAJOR) {
    constexpr int num_elems_per_pack = static_cast<int>(sizeof(scale_packed_t) / sizeof(scale_element_t));
    const int scale_num_rows_element = scale_num_rows * num_elems_per_pack;
    const int row_idx = global_group_id / scale_num_rows_element;
    const int col_idx_raw = global_group_id % scale_num_rows_element;
    const int col_idx = col_idx_raw / num_elems_per_pack;
    const int pack_idx = col_idx_raw % num_elems_per_pack;
    scale_output = reinterpret_cast<scale_element_t*>(output_s) +
                   (col_idx * scale_stride * num_elems_per_pack + row_idx * num_elems_per_pack + pack_idx);
  } else {
    static_assert(!SCALE_UE8M0);
    scale_output = output_s + global_group_id;
  }

  constexpr uint32_t vec_size = 16 / sizeof(T);
  using vec_t = flashinfer::vec_t<T, vec_size>;

  const int32_t num_vec_elems = group_size / vec_size;

  Tx2 local_absmax_x2 = {eps, eps};

  for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
    vec_t input_vec;
    input_vec.cast_load(group_input + i * vec_size);

#pragma unroll
    for (uint32_t j = 0; j < vec_size; j += 2) {
      Tx2 val_x2 = {input_vec[j], input_vec[j+1]};
      Tx2 abs_val_x2 = __habs2(val_x2);
      // TODO is this faster or slower?
      local_absmax_x2 = __hmax2(local_absmax_x2, abs_val_x2);
    }
  }

  float local_absmax = (float) __hmax(local_absmax_x2.x, local_absmax_x2.y);
  local_absmax = GroupReduceMax(local_absmax, lane_id);

  float y_scale, y_scale_inv;
  calculate_fp8_scales<SCALE_UE8M0>(local_absmax, y_scale, y_scale_inv, max_8bit, max_8bit_inv);
  float2 y_scale_repeated = {y_scale, y_scale};

  scale_element_t y_scale_inv_quant = extract_required_scale_format<SCALE_UE8M0>(y_scale_inv);

  if (lane_id == 0) {
    *scale_output = y_scale_inv_quant;
  }

  for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
    vec_t input_vec;
    input_vec.cast_load(group_input + i * vec_size);

    int2 output_buf;

    if constexpr (std::is_same_v<DST_DTYPE, c10::Float8_e4m3fn>) {
      const auto output_buf_ptr = reinterpret_cast<__nv_fp8x2_storage_t*>(&output_buf);
      static_assert(sizeof(output_buf) == vec_size / 2 * sizeof(__nv_fp8x2_storage_t));
      static_assert(vec_size % 2 == 0);

#pragma unroll
      for (uint32_t j = 0; j < vec_size; j += 2) {
        float2 inputx2 = {
          static_cast<float>(input_vec[j]),
          static_cast<float>(input_vec[j + 1])
        };
        float2 outputx2 = __fmul2_rn(inputx2, y_scale_repeated);
        output_buf_ptr[j / 2] = __nv_cvt_float2_to_fp8x2(outputx2, __NV_SATFINITE, __NV_E4M3);
      }
    } else {
      const auto output_buf_ptr = reinterpret_cast<DST_DTYPE*>(&output_buf);
      static_assert(sizeof(output_buf) == vec_size * sizeof(DST_DTYPE));

#pragma unroll
      for (uint32_t j = 0; j < vec_size; ++j) {
        float val = static_cast<float>(input_vec[j]);
        float q_val = fminf(fmaxf(val * y_scale, min_8bit), max_8bit);
        output_buf_ptr[j] = DST_DTYPE(q_val);
      }
    }

    st_global_v2_s32(reinterpret_cast<int2*>(group_output + i * vec_size), output_buf);
  }
}

void sgl_per_token_group_quant_8bit(
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit,
    bool scale_ue8m0 = false) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);

  const int num_groups = input.numel() / group_size;

  CHECK_EQ(input.numel() % group_size, 0);
  CHECK_EQ(output_s.dim(), 2);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int THREADS_PER_GROUP = 16;

  int groups_per_block = 1;

  if (num_groups % 16 == 0) {
    groups_per_block = 16;
  } else if (num_groups % 8 == 0) {
    groups_per_block = 8;
  } else if (num_groups % 4 == 0) {
    groups_per_block = 4;
  } else if (num_groups % 2 == 0) {
    groups_per_block = 2;
  }

  auto dst_type = output_q.scalar_type();
  const int num_blocks = num_groups / groups_per_block;
  const int num_threads = groups_per_block * THREADS_PER_GROUP;

  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  const int scale_num_rows = output_s.size(1);
  const int scale_stride = output_s.stride(1);

  const double max_8bit_inv = 1.0f / max_8bit;

#define LAUNCH_KERNEL(T, DST_DTYPE)                                                               \
  do {                                                                                            \
    dim3 grid(num_blocks);                                                                        \
    dim3 block(num_threads);                                                                      \
    if (is_column_major) {                                                                        \
      if (scale_ue8m0) {                                                                          \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, true, true><<<grid, block, 0, stream>>>(  \
            static_cast<T*>(input.data_ptr()),                                                    \
            output_q.data_ptr(),                                                                  \
            static_cast<uint32_t*>(output_s.data_ptr()),                                          \
            group_size,                                                                           \
            num_groups,                                                                           \
            groups_per_block,                                                                     \
            (float)eps,                                                                           \
            (float)min_8bit,                                                                      \
            (float)max_8bit,                                                                      \
            (float)max_8bit_inv,                                                                  \
            scale_num_rows,                                                                       \
            scale_stride);                                                                        \
      } else {                                                                                    \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, true, false><<<grid, block, 0, stream>>>( \
            static_cast<T*>(input.data_ptr()),                                                    \
            output_q.data_ptr(),                                                                  \
            static_cast<float*>(output_s.data_ptr()),                                             \
            group_size,                                                                           \
            num_groups,                                                                           \
            groups_per_block,                                                                     \
            (float)eps,                                                                           \
            (float)min_8bit,                                                                      \
            (float)max_8bit,                                                                      \
            (float)max_8bit_inv,                                                                  \
            scale_num_rows,                                                                       \
            scale_stride);                                                                        \
      }                                                                                           \
    } else {                                                                                      \
      assert(!scale_ue8m0);                                                                       \
      per_token_group_quant_8bit_kernel<T, DST_DTYPE, false><<<grid, block, 0, stream>>>(         \
          static_cast<T*>(input.data_ptr()),                                                      \
          output_q.data_ptr(),                                                                    \
          static_cast<float*>(output_s.data_ptr()),                                               \
          group_size,                                                                             \
          num_groups,                                                                             \
          groups_per_block,                                                                       \
          (float)eps,                                                                             \
          (float)min_8bit,                                                                        \
          (float)max_8bit,                                                                        \
          (float)max_8bit_inv);                                                                   \
    }                                                                                             \
  } while (0)

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), scalar_t, [&] {
    if (dst_type == at::ScalarType::Char) {
      LAUNCH_KERNEL(scalar_t, int8_t);
      return true;
    } else if (dst_type == at::ScalarType::Float8_e4m3fn) {
      LAUNCH_KERNEL(scalar_t, c10::Float8_e4m3fn);
      return true;
    }
    return false;
  });

#undef LAUNCH_KERNEL
}

void sgl_per_token_group_quant_int8(
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    int64_t group_size,
    double eps,
    double int8_min,
    double int8_max) {
  sgl_per_token_group_quant_8bit(input, output_q, output_s, group_size, eps, int8_min, int8_max);
}

void sgl_per_token_group_quant_fp8(
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    int64_t group_size,
    double eps,
    double fp8_min,
    double fp8_max,
    bool scale_ue8m0) {
  sgl_per_token_group_quant_8bit(input, output_q, output_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0);
}
