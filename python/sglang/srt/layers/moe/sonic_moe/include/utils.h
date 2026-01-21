// ********************************************************************************
// Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri
// Dao
// ********************************************************************************

// basic CUDA launch utils copied from
// https://github.com/open-lm-engine/accelerated-model-architectures/

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define CHECK_CUDA_TENSOR(x)                                                   \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS_TENSOR(x)                                             \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CONTIGUOUS_CUDA_TENSOR(x)                                        \
  CHECK_CUDA_DEVICE(x);                                                        \
  CHECK_CONTIGUOUS(x)

#define DISPATCH_CASE(ENUM_TYPE, SCALAR_NAME, ...)                             \
  AT_PRIVATE_CASE_TYPE_USING_HINT(ENUM_TYPE, SCALAR_NAME, __VA_ARGS__)

#define DISPATCH_FLOAT_KERNEL(TYPE, NAME, SCALAR_NAME, ...)                    \
  AT_DISPATCH_SWITCH(                                                          \
      TYPE, NAME,                                                              \
      DISPATCH_CASE(at::ScalarType::Half, SCALAR_NAME, __VA_ARGS__)            \
          DISPATCH_CASE(at::ScalarType::BFloat16, SCALAR_NAME, __VA_ARGS__)    \
              DISPATCH_CASE(at::ScalarType::Float, SCALAR_NAME, __VA_ARGS__))
#define DISPATCH_INT_KERNEL(TYPE, NAME, SCALAR_NAME, ...)                      \
  AT_DISPATCH_SWITCH(                                                          \
      TYPE, NAME,                                                              \
      DISPATCH_CASE(at::ScalarType::Int, SCALAR_NAME, __VA_ARGS__)             \
          DISPATCH_CASE(at::ScalarType::UInt32, SCALAR_NAME, __VA_ARGS__)      \
              DISPATCH_CASE(at::ScalarType::Long, SCALAR_NAME, __VA_ARGS__))

template <typename T>
inline __device__ T *load_128_bits(const T *array, const uint64_t &index) {
  const int4 *vector_array = reinterpret_cast<const int4 *>(array);
  int4 vector_element = vector_array[index];
  T *output = reinterpret_cast<T *>(&vector_element);
  return output;
}

template <typename T>
inline __device__ void store_128_bits(const T *source, T *destination,
                                      const uint64_t &index) {
  int4 *destination_vector_array = reinterpret_cast<int4 *>(destination);
  const int4 source_vector = reinterpret_cast<const int4 *>(&source[0])[0];
  destination_vector_array[index] = source_vector;
}
