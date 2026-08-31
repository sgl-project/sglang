/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include "dlpack/dlpack.h"

using tvm::ffi::Tensor;
using tvm::ffi::TensorView;
namespace ffi = tvm::ffi;

inline constexpr int64_t encode_dlpack_dtype(DLDataType dtype) {
  return (dtype.code << 16) | (dtype.bits << 8) | dtype.lanes;
}

constexpr DLDataType dl_uint8 = DLDataType{kDLUInt, 8, 1};
constexpr DLDataType dl_uint16 = DLDataType{kDLUInt, 16, 1};
constexpr DLDataType dl_uint32 = DLDataType{kDLUInt, 32, 1};
constexpr DLDataType dl_uint64 = DLDataType{kDLUInt, 64, 1};
constexpr DLDataType dl_int8 = DLDataType{kDLInt, 8, 1};
constexpr DLDataType dl_int16 = DLDataType{kDLInt, 16, 1};
constexpr DLDataType dl_int32 = DLDataType{kDLInt, 32, 1};
constexpr DLDataType dl_int64 = DLDataType{kDLInt, 64, 1};
constexpr DLDataType dl_float16 = DLDataType{kDLFloat, 16, 1};
constexpr DLDataType dl_float32 = DLDataType{kDLFloat, 32, 1};
constexpr DLDataType dl_float64 = DLDataType{kDLFloat, 64, 1};
constexpr DLDataType dl_float8_e4m3fn = DLDataType{kDLFloat8_e4m3fn, 8, 1};
constexpr DLDataType dl_float8_e5m2 = DLDataType{kDLFloat8_e5m2, 8, 1};
constexpr DLDataType dl_float4_e2m1fn = DLDataType{kDLFloat4_e2m1fn, 4, 1};
constexpr DLDataType dl_float4_e2m1fn_x2 = DLDataType{kDLFloat4_e2m1fn, 4, 2};
constexpr DLDataType dl_bfloat16 = DLDataType{kDLBfloat, 16, 1};
constexpr DLDataType dl_bool = DLDataType{kDLBool, 8, 1};

constexpr int64_t float16_code = encode_dlpack_dtype(dl_float16);
constexpr int64_t bfloat16_code = encode_dlpack_dtype(dl_bfloat16);
constexpr int64_t float32_code = encode_dlpack_dtype(dl_float32);
constexpr int64_t uint8_code = encode_dlpack_dtype(dl_uint8);
constexpr int64_t int32_code = encode_dlpack_dtype(dl_int32);
constexpr int64_t int64_code = encode_dlpack_dtype(dl_int64);
constexpr int64_t float8_e4m3fn_code = encode_dlpack_dtype(dl_float8_e4m3fn);
constexpr int64_t float8_e5m2_code = encode_dlpack_dtype(dl_float8_e5m2);
constexpr int64_t float4_e2m1fn_code = encode_dlpack_dtype(dl_float4_e2m1fn);

constexpr DLDevice cpu = DLDevice{kDLCPU, 0};

#define CHECK_CUDA(x) TVM_FFI_ICHECK_EQ(x.device().device_type, kDLCUDA) << #x " must be a CUDA tensor";
#define CHECK_CPU(x) TVM_FFI_ICHECK_EQ(x.device().device_type, kDLCPU) << #x " must be a host tensor";
#define CHECK_CONTIGUOUS(x) TVM_FFI_ICHECK(x.IsContiguous()) << #x " must be contiguous";
#define CHECK_LAST_DIM_CONTIGUOUS(x) \
  TVM_FFI_ICHECK_EQ(x.stride(-1), 1) \
  #x "must be contiguous at last dimension";
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_TYPE(x, st) TVM_FFI_ICHECK_EQ(x.dtype(), st) << "Inconsistency of Tensor type: " #x;
#define CHECK_INPUT_AND_TYPE(x, st) \
  CHECK_CUDA(x);                    \
  CHECK_CONTIGUOUS(x);              \
  CHECK_INPUT_TYPE(x, st)
#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x) \
  CHECK_CUDA(x);                           \
  CHECK_LAST_DIM_CONTIGUOUS(x)
#define CHECK_DIM(d, x) TVM_FFI_ICHECK_EQ(x.ndim(), d) << #x " must be a " #d "D tensor";
#define CHECK_DEVICE(a, b)                                           \
  TVM_FFI_ICHECK_EQ(a.device().device_type, b.device().device_type); \
  TVM_FFI_ICHECK_EQ(a.device().device_id, b.device().device_id);

inline cudaStream_t get_current_stream() {
  int device;
  cudaGetDevice(&device);
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device));
}

inline cudaStream_t get_stream(DLDevice device) {
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
}

inline int64_t get_element_size(ffi::Tensor x) {
  return (x.dtype().bits * x.dtype().lanes) / 8;
}

inline int64_t get_element_size(ffi::TensorView x) {
  return (x.dtype().bits * x.dtype().lanes) / 8;
}

inline ffi::Tensor alloc_tensor(tvm::ffi::Shape shape, DLDataType dtype, DLDevice device) {
  return ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, shape, dtype, device);
}
