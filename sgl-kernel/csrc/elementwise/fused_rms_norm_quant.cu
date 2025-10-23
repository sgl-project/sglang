#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "utils.h"

// RMS stage performs better with lower block size while quant
// stage is much better with higher ones due to extesive syncing
#define RMS_BLOCK_SIZE 256
#define QUANT_BLOCK_SIZE 1024
#define PACK_SIZE 16

// like std::array, but aligned
// goal: generate ld.128 and st.128 instructions
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <
    typename scalar_t,
    typename DST_DTYPE,
    bool IS_COLUMN_MAJOR = false,
    bool SCALE_UE8M0 = false,
    typename scale_packed_t = std::conditional_t<SCALE_UE8M0, uint32_t, float>>
__global__ void rms_norm_quant_kernel(
    scalar_t* __restrict__ input,
    void* __restrict__ output_q_v,
    scale_packed_t* __restrict__ output_s,
    scalar_t* __restrict__ weight,
    const int32_t group_size,
    const float rms_eps,
    const float quant_eps,
    const float fp8_min,
    const float fp8_max,
    const unsigned int stride,
    const unsigned int s_stride,
    const unsigned int d,
    const unsigned int rows) {
  DST_DTYPE* output_q = reinterpret_cast<DST_DTYPE*>(output_q_v);
  int row = blockIdx.x;
  int tx = threadIdx.x;
  int warp_id = tx / 32;
  using P = array_t<scalar_t, PACK_SIZE / sizeof(scalar_t)>;
  float acc = 0.f;
  __shared__ float reduction[RMS_BLOCK_SIZE / 32];

  if (threadIdx.x < RMS_BLOCK_SIZE) {
    for (int idx = tx; idx < d; idx += RMS_BLOCK_SIZE) {
      P x = reinterpret_cast<P*>(input + row * stride)[idx];

      for (int i = 0; i < P::size; i++) {
        acc += (float)x.data[i] * (float)x.data[i];
      }
    }
    acc += __shfl_xor_sync(0xffffffff, acc, 16, 32);
    acc += __shfl_xor_sync(0xffffffff, acc, 8, 32);
    acc += __shfl_xor_sync(0xffffffff, acc, 4, 32);
    acc += __shfl_xor_sync(0xffffffff, acc, 2, 32);
    acc += __shfl_xor_sync(0xffffffff, acc, 1, 32);

    if (tx % 32 == 0) {
      reduction[warp_id] = acc;
    }
  }

  __syncthreads();

  if (warp_id == 0) {
    acc = tx < RMS_BLOCK_SIZE / 32 ? reduction[tx] : 0.f;
    acc += __shfl_xor_sync(0xffffffff, acc, 16, 32);
    acc += __shfl_xor_sync(0xffffffff, acc, 8, 32);
    acc += __shfl_xor_sync(0xffffffff, acc, 4, 32);
    acc += __shfl_xor_sync(0xffffffff, acc, 2, 32);
    acc += __shfl_xor_sync(0xffffffff, acc, 1, 32);
  }
  if (tx == 0) {
    float var = acc / (d * P::size);
    reduction[0] = rsqrtf(var + rms_eps);
  }

  __syncthreads();
  acc = reduction[0];
  using O = array_t<DST_DTYPE, P::size>;
  for (int idx = tx; idx < d; idx += QUANT_BLOCK_SIZE) {
    float local_absmax = quant_eps;
    P x = reinterpret_cast<P*>(input + row * stride)[idx];
    P w = reinterpret_cast<P*>(weight)[idx];
    P temp;
    for (int i = 0; i < P::size; i++) {
      float val = (float)x.data[i] * acc * (float)w.data[i];
      local_absmax = fmaxf(local_absmax, fabsf(val));
      temp.data[i] = val;
    }
    for (int mask = group_size / 16; mask > 0; mask /= 2) {
      local_absmax = fmaxf(local_absmax, __shfl_xor_sync(0xffffffff, local_absmax, mask));
    }

    float y_s = (local_absmax / fp8_max);
    using scale_element_t = std::conditional_t<SCALE_UE8M0, uint8_t, float>;
    scale_element_t s;
    if constexpr (SCALE_UE8M0) {
      y_s = exp2f(ceilf(log2f(y_s)));
      s = (uint8_t)(((int)log2f(y_s)) + 127);
    } else {
      s = y_s;
    }
    if (tx % (group_size / P::size) == 0) {
      int off;
      if constexpr (IS_COLUMN_MAJOR) {
        const int num_elems_per_pack = static_cast<int>(sizeof(scale_packed_t) / sizeof(scale_element_t));
        int col_raw = (idx * P::size) / group_size;
        int col = col_raw / num_elems_per_pack;
        int pack = col_raw % num_elems_per_pack;
        off = col * num_elems_per_pack * s_stride + row * num_elems_per_pack + pack;
      } else {
        int col = (idx * P::size) / group_size;
        off = row * (d * P::size) / group_size + col;
      }
      __stcg(reinterpret_cast<scale_element_t*>(output_s) + off, s);
    }

    O out;
    for (int i = 0; i < P::size; i++) {
      float q = (float)temp.data[i] / y_s;
      float q_val = fminf(fmaxf(q, fp8_min), fp8_max);
      out.data[i] = DST_DTYPE(q_val);
    }
    using store_t = std::conditional_t<sizeof(scalar_t) == 2, int2, int>;
    __stcg(&reinterpret_cast<store_t*>(output_q)[row * d + idx], *reinterpret_cast<store_t*>(&out));
  }
}

void sgl_fused_rmsnorm_quant(
    torch::Tensor& input,
    torch::Tensor& output_q,
    torch::Tensor& output_s,
    torch::Tensor& weight,
    int64_t group_size,
    double rms_eps,
    double quant_eps,
    double fp8_min,
    double fp8_max,
    bool scale_ue8m0,
    bool enable_pdl) {
  const unsigned int d = input.size(-1);
  const unsigned int rows = input.size(-2);
  const unsigned int stride = input.stride(-2);
  const unsigned int scale_stride = output_s.stride(1);

  dim3 grid(rows);
  dim3 block(QUANT_BLOCK_SIZE);

  cudaLaunchAttribute attributes[1];
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchConfig_t config;
  config.gridDim = grid;
  config.blockDim = block;
  config.stream = stream;
  config.dynamicSmemBytes = 0;
  config.numAttrs = 1;
  config.attrs = attributes;

#define LAUNCH_KERNEL(T, DST_DTYPE)                                   \
  do {                                                                \
    if (is_column_major) {                                            \
      if (scale_ue8m0) {                                              \
        auto kern = rms_norm_quant_kernel<T, DST_DTYPE, true, true>;  \
        cudaLaunchKernelEx(                                           \
            &config,                                                  \
            kern,                                                     \
            static_cast<T*>(input.data_ptr()),                        \
            output_q.data_ptr(),                                      \
            static_cast<uint32_t*>(output_s.data_ptr()),              \
            static_cast<T*>(weight.data_ptr()),                       \
            (int32_t)group_size,                                      \
            (float)rms_eps,                                           \
            (float)quant_eps,                                         \
            (float)fp8_min,                                           \
            (float)fp8_max,                                           \
            stride,                                                   \
            scale_stride,                                             \
            packed_d,                                                 \
            rows);                                                    \
      } else {                                                        \
        auto kern = rms_norm_quant_kernel<T, DST_DTYPE, true, false>; \
        cudaLaunchKernelEx(                                           \
            &config,                                                  \
            kern,                                                     \
            static_cast<T*>(input.data_ptr()),                        \
            output_q.data_ptr(),                                      \
            static_cast<float*>(output_s.data_ptr()),                 \
            static_cast<T*>(weight.data_ptr()),                       \
            (int32_t)group_size,                                      \
            (float)rms_eps,                                           \
            (float)quant_eps,                                         \
            (float)fp8_min,                                           \
            (float)fp8_max,                                           \
            stride,                                                   \
            scale_stride,                                             \
            packed_d,                                                 \
            rows);                                                    \
      }                                                               \
    } else {                                                          \
      assert(!scale_ue8m0);                                           \
      auto kern = rms_norm_quant_kernel<T, DST_DTYPE, false>;         \
      cudaLaunchKernelEx(                                             \
          &config,                                                    \
          kern,                                                       \
          static_cast<T*>(input.data_ptr()),                          \
          output_q.data_ptr(),                                        \
          static_cast<float*>(output_s.data_ptr()),                   \
          static_cast<T*>(weight.data_ptr()),                         \
          (int32_t)group_size,                                        \
          (float)rms_eps,                                             \
          (float)quant_eps,                                           \
          (float)fp8_min,                                             \
          (float)fp8_max,                                             \
          stride,                                                     \
          scale_stride,                                               \
          packed_d,                                                   \
          rows);                                                      \
    }                                                                 \
  } while (0)
  auto dst_type = output_q.scalar_type();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    const unsigned int packed_d = std::ceil((float)d * sizeof(scalar_t) / PACK_SIZE);
    if (dst_type == at::ScalarType::Char) {
      LAUNCH_KERNEL(scalar_t, int8_t);
      return true;
    } else if (dst_type == at::ScalarType::Float8_e4m3fn) {
      LAUNCH_KERNEL(scalar_t, c10::Float8_e4m3fn);
      return true;
    }
    return false;
  });
}
