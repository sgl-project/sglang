#include "pytorch_extension_utils.h"

template <typename T>
struct ConvertToFP8 {
  static __device__ __nv_fp8_storage_t convert_to_fp8(T value) {
    return 0;
  }
};

template <>
struct ConvertToFP8<__nv_bfloat16> {
  static __device__ __nv_fp8_storage_t convert_to_fp8(__nv_bfloat16 value) {
    return __nv_cvt_bfloat16raw_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
  }
};

template <>
struct ConvertToFP8<half> {
  static __device__ __nv_fp8_storage_t convert_to_fp8(half value) {
    return __nv_cvt_halfraw_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
  }
};

template <typename T>
struct ConvertFromFloat {
  static __device__ T convert_from_float(float value) {
    return 0;
  }
};

template <>
struct ConvertFromFloat<__nv_bfloat16> {
  static __device__ __nv_bfloat16 convert_from_float(float value) {
    return __float2bfloat16(value);
  }
};

template <>
struct ConvertFromFloat<half> {
  static __device__ half convert_from_float(float value) {
    return __float2half(value);
  }
};

template <typename T>
__global__ void fused_downcast_kernel(
    const T* cache_k,
    const T* cache_v,
    const float* k_scale,
    const float* v_scale,
    __nv_fp8_storage_t* output_k,
    __nv_fp8_storage_t* output_v,
    const int input_sl,
    const int head,
    const int dim,
    const T max_fp8,
    const T min_fp8,
    const int64_t mult,
    const int64_t offset,
    const int64_t* loc) {
  // TODO: change name
  int token_idx = blockIdx.x;
  int thread_idx = threadIdx.x;
  int total_threads = blockDim.x;

  T k_scale_val = ConvertFromFloat<T>::convert_from_float(k_scale[0]);
  T v_scale_val = ConvertFromFloat<T>::convert_from_float(v_scale[0]);

  T k_scale_inv = static_cast<T>(1.f) / k_scale_val;
  T v_scale_inv = static_cast<T>(1.f) / v_scale_val;

  auto clamp = [&](T val) { return val > max_fp8 ? max_fp8 : (min_fp8 > val ? min_fp8 : val); };

  if (token_idx < input_sl) {
    int out_seq_idx = loc[token_idx];

#pragma unroll
    for (int i = thread_idx; i < head * dim; i += total_threads) {
      int in_idx = token_idx * head * dim + i;
      int out_idx = (out_seq_idx * mult + offset) * head * dim + i;

      T k_val = cache_k[in_idx] * k_scale_inv;
      k_val = clamp(k_val);
      output_k[out_idx] = ConvertToFP8<T>::convert_to_fp8(k_val);

      T v_val = cache_v[in_idx] * v_scale_inv;
      v_val = clamp(v_val);
      output_v[out_idx] = ConvertToFP8<T>::convert_to_fp8(v_val);
    }
  }
}

template <typename T>
void downcast_fp8_impl(
    at::Tensor& k,
    at::Tensor& v,
    at::Tensor& k_out,
    at::Tensor& v_out,
    at::Tensor& k_scale,
    at::Tensor& v_scale,
    at::Tensor& loc,
    int64_t mult,
    int64_t offset,
    cudaStream_t stream) {
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(k_out);
  CHECK_INPUT(v_out);
  CHECK_INPUT(k_scale);
  CHECK_INPUT(v_scale);
  CHECK_INPUT(loc);

  int64_t input_sl = k.size(0);
  int64_t head = k.size(1);
  int64_t dim = k.size(2);

  dim3 grid(input_sl * head);
  int vec_size = 8;
  dim3 block(std::min(int(dim) / vec_size, 1024));

  const T max_fp8 = static_cast<T>(448.0f);
  const T min_fp8 = static_cast<T>(-448.0f);

  fused_downcast_kernel<T><<<grid, block, 0, stream>>>(
      static_cast<const T*>(k.data_ptr()),
      static_cast<const T*>(v.data_ptr()),
      static_cast<const float*>(k_scale.data_ptr()),
      static_cast<const float*>(v_scale.data_ptr()),
      static_cast<__nv_fp8_storage_t*>(k_out.data_ptr()),
      static_cast<__nv_fp8_storage_t*>(v_out.data_ptr()),
      input_sl,
      head,
      dim,
      max_fp8,
      min_fp8,
      mult,
      offset,
      static_cast<const int64_t*>(loc.data_ptr()));

  cudaError_t status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess, "Kernel launch failed: " + std::string(cudaGetErrorString(status)));
}

void downcast_fp8(
    at::Tensor& k,
    at::Tensor& v,
    at::Tensor& k_out,
    at::Tensor& v_out,
    at::Tensor& k_scale,
    at::Tensor& v_scale,
    at::Tensor& loc,
    int64_t mult,
    int64_t offset,
    int64_t cuda_stream) {
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(k_out);
  CHECK_INPUT(v_out);

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  switch (k.scalar_type()) {
    case at::ScalarType::BFloat16:
      downcast_fp8_impl<__nv_bfloat16>(k, v, k_out, v_out, k_scale, v_scale, loc, mult, offset, stream);
      break;
    case at::ScalarType::Half:
      downcast_fp8_impl<__half>(k, v, k_out, v_out, k_scale, v_scale, loc, mult, offset, stream);
      break;
    default:
      TORCH_CHECK(false, "Unsupported input type for downcast_fp8. Expected bfloat16 or float16.");
  }
}
