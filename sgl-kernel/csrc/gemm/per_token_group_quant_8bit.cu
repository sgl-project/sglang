#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>

#include <cmath>
#include <flashinfer/vec_dtypes.cuh>

#include "utils.h"

template <int THREADS_PER_GROUP>
__device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
  unsigned mask = 0xffff;

  static_assert((THREADS_PER_GROUP == 16) or (THREADS_PER_GROUP == 8));

  if constexpr (THREADS_PER_GROUP == 16) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  }
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

__device__ __forceinline__ void st_global(const int4* ptr, const int4& value) {
  asm volatile(
      "st.global.v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
}

__device__ __forceinline__ int4 ld_global_nc(const int4* ptr) {
  int4 ret;
  asm volatile("ld.global.nc.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
               : "l"(ptr));
  return ret;
}

template <typename dtype_t>
__host__ __device__ dtype_t ceil_div(dtype_t a, dtype_t b) {
  return (a + b - 1) / b;
}

constexpr int THREADS_PER_GROUP = 8;
constexpr uint32_t VEC_NUM_BYTES_PER_WAVE = 32;
constexpr int NUM_WAVES_CONSTEXPR = 3;

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
    const int num_groups_per_wave,
    const int groups_per_block,
    const float eps,
    const float min_8bit,
    const float max_8bit,
    const float max_8bit_inv,
    const int scale_num_rows = 0,
    const int scale_stride = 0) {
  const int local_group_id = threadIdx.x / THREADS_PER_GROUP;
  const int lane_id = threadIdx.x % THREADS_PER_GROUP;
  const int block_group_id = blockIdx.x * groups_per_block;

  int4 input_int4[VEC_NUM_BYTES_PER_WAVE * NUM_WAVES_CONSTEXPR / sizeof(int4)];
  constexpr uint32_t VEC_TYPED_SIZE_PER_WAVE = VEC_NUM_BYTES_PER_WAVE / sizeof(T);

#pragma unroll
  for (int wave_index = 0; wave_index < NUM_WAVES_CONSTEXPR; ++wave_index) {
    const int global_group_id = block_group_id + local_group_id + wave_index * num_groups_per_wave;
    if (global_group_id >= num_groups) [[unlikely]] {
      return;
    }

    const int block_group_offset = global_group_id * group_size;
#pragma unroll
    for (uint32_t j = 0; j < VEC_INT4_SIZE; ++j) {
      input_int4[j] = ld_global_nc(
          reinterpret_cast<const int4*>(input + block_group_offset + lane_id * VEC_TYPED_SIZE_PER_WAVE) + j);
    }
  }

#pragma unroll
  for (int wave_index = 0; wave_index < NUM_WAVES_CONSTEXPR; ++wave_index) {
    const int global_group_id = block_group_id + local_group_id + wave_index * num_groups_per_wave;
    // TODO optimize
    if (global_group_id >= num_groups) [[unlikely]] {
      return;
    }

    const int block_group_offset = global_group_id * group_size;

    using scale_element_t = std::conditional_t<SCALE_UE8M0, uint8_t, float>;
    static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);

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

    T* input_vec_local =
        reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(input_int4) + VEC_NUM_BYTES_PER_WAVE * wave_index);

    float local_absmax = eps;

#pragma unroll
    for (uint32_t j = 0; j < VEC_TYPED_SIZE_PER_WAVE; ++j) {
      float val = static_cast<float>(input_vec_local[j]);
      float abs_val = fabsf(val);
      local_absmax = fmaxf(local_absmax, abs_val);
    }

    local_absmax = GroupReduceMax<THREADS_PER_GROUP>(local_absmax, lane_id);

    float y_scale, y_scale_inv;
    calculate_fp8_scales<SCALE_UE8M0>(local_absmax, y_scale, y_scale_inv, max_8bit, max_8bit_inv);
    float2 y_scale_repeated = {y_scale, y_scale};

    scale_element_t y_scale_inv_quant = extract_required_scale_format<SCALE_UE8M0>(y_scale_inv);

    if (lane_id == 0) {
      *scale_output = y_scale_inv_quant;
    }

    int4 output_buf;

    if constexpr (std::is_same_v<DST_DTYPE, c10::Float8_e4m3fn>) {
      const auto output_buf_ptr = reinterpret_cast<__nv_fp8x2_storage_t*>(&output_buf);
      static_assert(sizeof(output_buf) == VEC_TYPED_SIZE_PER_WAVE / 2 * sizeof(__nv_fp8x2_storage_t));
      static_assert(VEC_TYPED_SIZE_PER_WAVE % 2 == 0);

#pragma unroll
      for (uint32_t j = 0; j < VEC_TYPED_SIZE_PER_WAVE; j += 2) {
        float2 inputx2 = {static_cast<float>(input_vec_local[j]), static_cast<float>(input_vec_local[j + 1])};
        float2 outputx2 = __fmul2_rn(inputx2, y_scale_repeated);
        output_buf_ptr[j / 2] = __nv_cvt_float2_to_fp8x2(outputx2, __NV_SATFINITE, __NV_E4M3);
      }
    } else {
      const auto output_buf_ptr = reinterpret_cast<DST_DTYPE*>(&output_buf);
      static_assert(sizeof(output_buf) == VEC_TYPED_SIZE_PER_WAVE * sizeof(DST_DTYPE));

#pragma unroll
      for (uint32_t j = 0; j < VEC_TYPED_SIZE_PER_WAVE; ++j) {
        float val = static_cast<float>(input_vec_local[j]);
        float q_val = fminf(fmaxf(val * y_scale, min_8bit), max_8bit);
        output_buf_ptr[j] = DST_DTYPE(q_val);
      }
    }

    st_global(reinterpret_cast<int4*>(group_output + lane_id * VEC_TYPED_SIZE_PER_WAVE), output_buf);
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

  int device_id = at::cuda::current_device();
  const cudaDeviceProp* prop = at::cuda::getDeviceProperties(device_id);
  int num_sms = prop->multiProcessorCount;

  auto dst_type = output_q.scalar_type();
  const int num_threads = groups_per_block * THREADS_PER_GROUP;
  // TODO dynamically determine it
  const int blocks_per_sm = 2048 / num_threads;
  CHECK_EQ(blocks_per_sm, 16);
  const int num_blocks = num_sms * blocks_per_sm;
  const int num_waves = ceil_div(num_groups, groups_per_block * num_blocks);
  CHECK_EQ(num_waves, NUM_WAVES_CONSTEXPR);

  const int num_groups_per_wave = num_blocks * groups_per_block;

  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  const int scale_num_rows = output_s.size(1);
  const int scale_stride = output_s.stride(1);

  const double max_8bit_inv = 1.0f / max_8bit;

#define LAUNCH_KERNEL(T, DST_DTYPE)                                                               \
  do {                                                                                            \
    /* TODO do not copy paste */                                                                  \
    constexpr uint32_t VEC_TYPED_SIZE_PER_WAVE = VEC_NUM_BYTES_PER_WAVE / sizeof(T);              \
    TORCH_CHECK(THREADS_PER_GROUP == group_size / VEC_TYPED_SIZE_PER_WAVE);                       \
                                                                                                  \
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
            num_groups_per_wave,                                                                  \
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
            num_groups_per_wave,                                                                  \
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
          num_groups_per_wave,                                                                    \
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
