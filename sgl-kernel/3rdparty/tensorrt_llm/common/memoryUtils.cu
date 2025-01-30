/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"

#include <curand_kernel.h>
#include <sys/stat.h>
#include <unordered_map>

namespace tensorrt_llm
{
namespace common
{

template <typename T>
void deviceMalloc(T** ptr, size_t size, bool is_random_initialize)
{
    check_cuda_error(cudaMalloc((void**) (ptr), sizeof(T) * size));
    if (is_random_initialize)
    {
        cudaRandomUniform(*ptr, size);
    }
}

template void deviceMalloc(float** ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(half** ptr, size_t size, bool is_random_initialize);
#ifdef ENABLE_BF16
template void deviceMalloc(__nv_bfloat16** ptr, size_t size, bool is_random_initialize);
#endif
template void deviceMalloc(uint16_t** ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(int** ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(bool** ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(char** ptr, size_t size, bool is_random_initialize);
template void deviceMalloc(int8_t** ptr, size_t size, bool is_random_initialize);
#ifdef ENABLE_FP8
template void deviceMalloc(__nv_fp8_e4m3** ptr, size_t size, bool is_random_initialize);
#endif

template <typename T>
void deviceMemSetZero(T* ptr, size_t size)
{
    check_cuda_error(cudaMemset(static_cast<void*>(ptr), 0, sizeof(T) * size));
}

template void deviceMemSetZero(float* ptr, size_t size);
template void deviceMemSetZero(half* ptr, size_t size);
template void deviceMemSetZero(int* ptr, size_t size);
template void deviceMemSetZero(uint32_t* ptr, size_t size);
template void deviceMemSetZero(bool* ptr, size_t size);
#ifdef ENABLE_FP8
template void deviceMemSetZero(__nv_fp8_e4m3* ptr, size_t size);
#endif
#ifdef ENABLE_BF16
template void deviceMemSetZero(__nv_bfloat16* ptr, size_t size);
#endif

template <typename T>
void deviceFree(T*& ptr)
{
    if (ptr != NULL)
    {
        check_cuda_error(cudaFree(ptr));
        ptr = NULL;
    }
}

template void deviceFree(float*& ptr);
template void deviceFree(half*& ptr);
#ifdef ENABLE_BF16
template void deviceFree(__nv_bfloat16*& ptr);
#endif
template void deviceFree(unsigned short*& ptr);
template void deviceFree(int*& ptr);
template void deviceFree(bool*& ptr);
template void deviceFree(char*& ptr);
template void deviceFree(int8_t*& ptr);
#ifdef ENABLE_FP8
template void deviceFree(__nv_fp8_e4m3*& ptr);
#endif

template <typename T>
void deviceFill(T* devptr, size_t size, T value, cudaStream_t stream)
{
    T* arr = new T[size];
    std::fill(arr, arr + size, value);
    check_cuda_error(cudaMemcpyAsync(devptr, arr, sizeof(T) * size, cudaMemcpyHostToDevice, stream));
    delete[] arr;
}

template void deviceFill(float* devptr, size_t size, float value, cudaStream_t stream);
template void deviceFill(half* devptr, size_t size, half value, cudaStream_t stream);
#ifdef ENABLE_BF16
template void deviceFill(__nv_bfloat16* devptr, size_t size, __nv_bfloat16 value, cudaStream_t stream);
#endif
template void deviceFill(int* devptr, size_t size, int value, cudaStream_t stream);
template void deviceFill(bool* devptr, size_t size, bool value, cudaStream_t stream);

template <typename T>
void cudaD2Hcpy(T* tgt, T const* src, const size_t size)
{
    check_cuda_error(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyDeviceToHost));
}

template void cudaD2Hcpy(float* tgt, float const* src, size_t size);
template void cudaD2Hcpy(half* tgt, half const* src, size_t size);
#ifdef ENABLE_BF16
template void cudaD2Hcpy(__nv_bfloat16* tgt, __nv_bfloat16 const* src, size_t size);
#endif
template void cudaD2Hcpy(int* tgt, int const* src, size_t size);
template void cudaD2Hcpy(bool* tgt, bool const* src, size_t size);
#ifdef ENABLE_FP8
template void cudaD2Hcpy(__nv_fp8_e4m3* tgt, __nv_fp8_e4m3 const* src, size_t size);
#endif
template void cudaD2Hcpy(unsigned long long* tgt, unsigned long long const* src, size_t size);
template void cudaD2Hcpy(unsigned int* tgt, unsigned int const* src, size_t size);
template void cudaD2Hcpy(int8_t* tgt, int8_t const* src, size_t size);

template <typename T>
void cudaH2Dcpy(T* tgt, T const* src, const size_t size)
{
    check_cuda_error(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyHostToDevice));
}

template void cudaH2Dcpy(float* tgt, float const* src, size_t size);
template void cudaH2Dcpy(half* tgt, half const* src, size_t size);
#ifdef ENABLE_BF16
template void cudaH2Dcpy(__nv_bfloat16* tgt, __nv_bfloat16 const* src, size_t size);
#endif
template void cudaH2Dcpy(int* tgt, int const* src, size_t size);
template void cudaH2Dcpy(bool* tgt, bool const* src, size_t size);
#ifdef ENABLE_FP8
template void cudaH2Dcpy(__nv_fp8_e4m3* tgt, __nv_fp8_e4m3 const* src, size_t size);
#endif
template void cudaH2Dcpy(unsigned long long* tgt, unsigned long long const* src, size_t size);
template void cudaH2Dcpy(unsigned int* tgt, unsigned int const* src, size_t size);
template void cudaH2Dcpy(int8_t* tgt, int8_t const* src, size_t size);

template <typename T>
void cudaD2Dcpy(T* tgt, T const* src, const size_t size, cudaStream_t stream)
{
    check_cuda_error(cudaMemcpyAsync(tgt, src, sizeof(T) * size, cudaMemcpyDeviceToDevice, stream));
}

template void cudaD2Dcpy(float* tgt, float const* src, size_t size, cudaStream_t stream);
template void cudaD2Dcpy(half* tgt, half const* src, size_t size, cudaStream_t stream);
#ifdef ENABLE_BF16
template void cudaD2Dcpy(__nv_bfloat16* tgt, __nv_bfloat16 const* src, size_t size, cudaStream_t stream);
#endif
template void cudaD2Dcpy(int* tgt, int const* src, size_t size, cudaStream_t stream);
template void cudaD2Dcpy(bool* tgt, bool const* src, size_t size, cudaStream_t stream);
template void cudaD2Dcpy(int8_t* tgt, int8_t const* src, size_t size, cudaStream_t stream);
#ifdef ENABLE_FP8
template void cudaD2Dcpy(__nv_fp8_e4m3* tgt, __nv_fp8_e4m3 const* src, size_t size, cudaStream_t stream);
#endif
template void cudaD2Dcpy(unsigned long long* tgt, unsigned long long const* src, size_t size, cudaStream_t stream);

template <typename T_OUT, typename T_IN>
__global__ void cudaCast(T_OUT* dst, T_IN* src, const size_t size)
{
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x)
    {
        dst[tid] = (T_OUT) ((float) (src[tid]));
    }
}

template <typename T_OUT, typename T_IN>
void invokeCudaCast(T_OUT* dst, T_IN const* const src, const size_t size, cudaStream_t stream)
{
    cudaCast<<<256, 256, 0, stream>>>(dst, src, size);
}

template void invokeCudaCast(float* dst, half const* const src, const size_t size, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeCudaCast(float* dst, __nv_bfloat16 const* const src, const size_t size, cudaStream_t stream);
template void invokeCudaCast(__nv_bfloat16* dst, float const* const src, const size_t size, cudaStream_t stream);
template void invokeCudaCast(__nv_bfloat16* dst, half const* const src, const size_t size, cudaStream_t stream);
template void invokeCudaCast(half* dst, __nv_bfloat16 const* const src, const size_t size, cudaStream_t stream);
#endif
#ifdef ENABLE_FP8
template void invokeCudaCast(float* dst, __nv_fp8_e4m3 const* const src, const size_t size, cudaStream_t stream);
template void invokeCudaCast(
    __nv_bfloat16* dst, __nv_fp8_e4m3 const* const src, const size_t size, cudaStream_t stream);
template void invokeCudaCast(half* dst, __nv_fp8_e4m3 const* const src, const size_t size, cudaStream_t stream);
template void invokeCudaCast(__nv_fp8_e4m3* dst, float const* const src, const size_t size, cudaStream_t stream);
template void invokeCudaCast(
    __nv_fp8_e4m3* dst, __nv_bfloat16 const* const src, const size_t size, cudaStream_t stream);
template void invokeCudaCast(__nv_fp8_e4m3* dst, half const* const src, const size_t size, cudaStream_t stream);
#endif

template <typename T>
void cudaAutoCpy(T* tgt, T const* src, const size_t size, cudaStream_t stream)
{
    if (stream != NULL)
    {
        check_cuda_error(cudaMemcpyAsync(tgt, src, sizeof(T) * size, cudaMemcpyDefault, stream));
    }
    else
    {
        check_cuda_error(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyDefault));
    }
}

template void cudaAutoCpy(float* tgt, float const* src, size_t size, cudaStream_t stream);
template void cudaAutoCpy(half* tgt, half const* src, size_t size, cudaStream_t stream);
#ifdef ENABLE_BF16
template void cudaAutoCpy(__nv_bfloat16* tgt, __nv_bfloat16 const* src, size_t size, cudaStream_t stream);
#endif
template void cudaAutoCpy(int* tgt, int const* src, size_t size, cudaStream_t stream);
template void cudaAutoCpy(bool* tgt, bool const* src, size_t size, cudaStream_t stream);
template void cudaAutoCpy(int8_t* tgt, int8_t const* src, size_t size, cudaStream_t stream);
template void cudaAutoCpy(uint8_t* tgt, uint8_t const* src, size_t size, cudaStream_t stream);
template void cudaAutoCpy(uint32_t* tgt, uint32_t const* src, size_t size, cudaStream_t stream);
template void cudaAutoCpy(unsigned long long* tgt, unsigned long long const* src, size_t size, cudaStream_t stream);
template void cudaAutoCpy(unsigned long* tgt, unsigned long const* src, size_t size, cudaStream_t stream);
template void cudaAutoCpy(char* tgt, char const* src, size_t size, cudaStream_t stream);

template void cudaAutoCpy(float const** tgt, float const* const* src, size_t size, cudaStream_t stream);
template void cudaAutoCpy(half const** tgt, half const* const* src, size_t size, cudaStream_t stream);
#ifdef ENABLE_BF16
template void cudaAutoCpy(__nv_bfloat16 const** tgt, __nv_bfloat16 const* const* src, size_t size, cudaStream_t stream);
#endif
template void cudaAutoCpy(int const** tgt, int const* const* src, size_t size, cudaStream_t stream);
template void cudaAutoCpy(bool const** tgt, bool const* const* src, size_t size, cudaStream_t stream);
template void cudaAutoCpy(int8_t const** tgt, int8_t const* const* src, size_t size, cudaStream_t stream);
template void cudaAutoCpy(
    unsigned long long const** tgt, unsigned long long const* const* src, size_t size, cudaStream_t stream);

template <typename T>
__global__ void cuda_random_uniform_kernel(T* buffer, const size_t size, int const seq_offset)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t local_state;
    curand_init((unsigned long long int) 1337, idx + seq_offset, 0, &local_state);
    for (size_t index = idx; index < size; index += blockDim.x * gridDim.x)
    {
        buffer[index] = (T) (curand_uniform(&local_state) * 0.2f - 0.1f);
    }
}

template <>
__global__ void cuda_random_uniform_kernel<int>(int* buffer, const size_t size, int const seq_offset)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t local_state;
    curand_init((float) 1337.f, idx + seq_offset, 0, &local_state);
    for (size_t index = idx; index < size; index += blockDim.x * gridDim.x)
    {
        buffer[index] = curand(&local_state);
    }
}

template <>
__global__ void cuda_random_uniform_kernel<bool>(bool* buffer, const size_t size, int const seq_offset)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t local_state;
    curand_init((float) 1337.f, idx + seq_offset, 0, &local_state);
    for (size_t index = idx; index < size; index += blockDim.x * gridDim.x)
    {
        buffer[index] = (curand(&local_state) % 2 == 0);
    }
}

template <>
__global__ void cuda_random_uniform_kernel<char>(char* buffer, const size_t size, int const seq_offset)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t local_state;
    curand_init((float) 1337.f, idx + seq_offset, 0, &local_state);
    for (size_t index = idx; index < size; index += blockDim.x * gridDim.x)
    {
        buffer[index] = curand(&local_state) % 0xFF;
    }
}

template <typename T>
void cudaRandomUniform(T* buffer, const size_t size)
{
    static int seq_offset = 0;
    cuda_random_uniform_kernel<T><<<256, 256>>>(buffer, size, seq_offset);
    seq_offset += 256 * 256;
}

template void cudaRandomUniform(float* buffer, const size_t size);
template void cudaRandomUniform(half* buffer, const size_t size);
#ifdef ENABLE_BF16
template void cudaRandomUniform(__nv_bfloat16* buffer, const size_t size);
#endif
template void cudaRandomUniform(int* buffer, const size_t size);
template void cudaRandomUniform(bool* buffer, const size_t size);
template void cudaRandomUniform(char* buffer, const size_t size);
#ifdef ENABLE_FP8
template void cudaRandomUniform(__nv_fp8_e4m3* buffer, const size_t size);
#endif

// loads data from binary file. If it succeeds, returns a non-empty vector. If loading fails or
// the product of the elements in shape is 0, this function will return an empty vector.
template <typename T>
std::vector<T> loadWeightFromBinHelper(std::vector<size_t> shape, std::string filename)
{
    if (shape.size() > 2)
    {
        printf("[ERROR] shape should have less than two dims \n");
        return std::vector<T>();
    }
    size_t dim0 = shape[0], dim1 = 1;
    if (shape.size() == 2)
    {
        dim1 = shape[1];
    }
    size_t size = dim0 * dim1;
    if (size == 0)
    {
        TLLM_LOG_WARNING("shape is zero, skip loading weight from file %s \n", filename.c_str());
        return std::vector<T>();
    }

    std::vector<T> host_array(size);
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (!in.is_open())
    {
        TLLM_LOG_WARNING("file %s cannot be opened, loading model fails! \n", filename.c_str());
        return std::vector<T>();
    }

    size_t loaded_data_size = sizeof(T) * size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);

    TLLM_LOG_DEBUG("Read " + std::to_string(loaded_data_size) + " bytes from " + filename);
    in.read((char*) host_array.data(), loaded_data_size);

    size_t in_get_size = in.gcount();
    if (in_get_size != loaded_data_size)
    {
        TLLM_LOG_WARNING("file %s only has %ld, but request %ld, loading model fails! \n", filename.c_str(),
            in_get_size, loaded_data_size);
        return std::vector<T>();
    }
    in.close();
    // If we succeed, return an array with values.
    return host_array;
}

template <typename T, typename T_IN>
int loadWeightFromBinFunc(T* ptr, std::vector<size_t> shape, std::string filename)
{
    std::vector<T_IN> host_array = loadWeightFromBinHelper<T_IN>(shape, filename);

    if (host_array.empty())
    {
        return 0;
    }

    if (std::is_same<T, T_IN>::value == true)
    {
        cudaH2Dcpy(ptr, (T*) host_array.data(), host_array.size());
    }
    else
    {
        T_IN* ptr_2 = nullptr;
        deviceMalloc(&ptr_2, host_array.size(), false);
        cudaH2Dcpy(ptr_2, host_array.data(), host_array.size());
        invokeCudaD2DcpyConvert(ptr, ptr_2, host_array.size());
        deviceFree(ptr_2);
    }
    return 0;
}

template int loadWeightFromBinFunc<float, float>(float* ptr, std::vector<size_t> shape, std::string filename);
template int loadWeightFromBinFunc<half, float>(half* ptr, std::vector<size_t> shape, std::string filename);
template int loadWeightFromBinFunc<float, half>(float* ptr, std::vector<size_t> shape, std::string filename);
template int loadWeightFromBinFunc<half, half>(half* ptr, std::vector<size_t> shape, std::string filename);
template int loadWeightFromBinFunc<int8_t, int8_t>(int8_t* ptr, std::vector<size_t> shape, std::string filename);
#ifdef ENABLE_BF16
template int loadWeightFromBinFunc<__nv_bfloat16, float>(
    __nv_bfloat16* ptr, std::vector<size_t> shape, std::string filename);
template int loadWeightFromBinFunc<__nv_bfloat16, half>(
    __nv_bfloat16* ptr, std::vector<size_t> shape, std::string filename);
template int loadWeightFromBinFunc<float, __nv_bfloat16>(float* ptr, std::vector<size_t> shape, std::string filename);
template int loadWeightFromBinFunc<half, __nv_bfloat16>(half* ptr, std::vector<size_t> shape, std::string filename);
template int loadWeightFromBinFunc<__nv_bfloat16, __nv_bfloat16>(
    __nv_bfloat16* ptr, std::vector<size_t> shape, std::string filename);
#endif // ENABLE_BF16
template int loadWeightFromBinFunc<int, int>(int* ptr, std::vector<size_t> shape, std::string filename);
#ifdef ENABLE_FP8
template int loadWeightFromBinFunc<__nv_fp8_e4m3, float>(
    __nv_fp8_e4m3* ptr, std::vector<size_t> shape, std::string filename);
#endif // ENABLE_FP8

template <typename T>
int loadWeightFromBin(T* ptr, std::vector<size_t> shape, std::string filename, TRTLLMCudaDataType model_file_type)
{
    switch (model_file_type)
    {
    case TRTLLMCudaDataType::FP32: loadWeightFromBinFunc<T, float>(ptr, shape, filename); break;
    case TRTLLMCudaDataType::FP16: loadWeightFromBinFunc<T, half>(ptr, shape, filename); break;
    case TRTLLMCudaDataType::INT8: loadWeightFromBinFunc<T, int8_t>(ptr, shape, filename); break;
#ifdef ENABLE_BF16
    case TRTLLMCudaDataType::BF16: loadWeightFromBinFunc<T, __nv_bfloat16>(ptr, shape, filename); break;
#endif
#ifdef ENABLE_FP8
    case TRTLLMCudaDataType::FP8: loadWeightFromBinFunc<T, float>(ptr, shape, filename); break;
#endif
    default: TLLM_LOG_ERROR("Does not support TRTLLMCudaDataType=%d", model_file_type); TLLM_CHECK(false);
    }
    return 0;
}

template <>
int loadWeightFromBin(int* ptr, std::vector<size_t> shape, std::string filename, TRTLLMCudaDataType model_file_type)
{
    loadWeightFromBinFunc<int, int>(ptr, shape, filename);
    return 0;
}

template int loadWeightFromBin(
    float* ptr, std::vector<size_t> shape, std::string filename, TRTLLMCudaDataType model_file_type);
template int loadWeightFromBin(
    half* ptr, std::vector<size_t> shape, std::string filename, TRTLLMCudaDataType model_file_type);
template int loadWeightFromBin(
    int8_t* ptr, std::vector<size_t> shape, std::string filename, TRTLLMCudaDataType model_file_type);
#ifdef ENABLE_BF16
template int loadWeightFromBin(
    __nv_bfloat16* ptr, std::vector<size_t> shape, std::string filename, TRTLLMCudaDataType model_file_type);
#endif
#ifdef ENABLE_FP8
template int loadWeightFromBin(
    __nv_fp8_e4m3* ptr, std::vector<size_t> shape, std::string filename, TRTLLMCudaDataType model_file_type);
#endif
template int loadWeightFromBin(
    int* ptr, std::vector<size_t> shape, std::string filename, TRTLLMCudaDataType model_file_type);

template <typename T_IN, typename T_OUT>
__global__ void cudaD2DcpyConvert(T_OUT* dst, const T_IN* src, const size_t size)
{
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x)
    {
        dst[tid] = cuda_cast<T_OUT>(src[tid]);
    }
}

template <typename T_IN, typename T_OUT>
void invokeCudaD2DcpyConvert(T_OUT* tgt, const T_IN* src, const size_t size, cudaStream_t stream)
{
    cudaD2DcpyConvert<<<256, 256, 0, stream>>>(tgt, src, size);
}

template void invokeCudaD2DcpyConvert(int8_t* tgt, float const* src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(float* tgt, int8_t const* src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(float* tgt, int const* src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(half* tgt, int const* src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(float* tgt, float const* src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(half* tgt, float const* src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(float* tgt, half const* src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(uint32_t* tgt, int const* src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(int* tgt, uint32_t const* src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(int* tgt, float const* src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(int* tgt, half const* src, const size_t size, cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeCudaD2DcpyConvert(__nv_bfloat16* tgt, float const* src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(__nv_bfloat16* tgt, int const* src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(float* tgt, __nv_bfloat16 const* src, const size_t size, cudaStream_t stream);
template void invokeCudaD2DcpyConvert(int* tgt, __nv_bfloat16 const* src, const size_t size, cudaStream_t stream);
#endif // ENABLE_BF16

template <typename T_IN, typename T_OUT>
__global__ void cudaD2DScaleCpyConvert(
    T_OUT* dst, const T_IN* src, float const* scale, bool invert_scale, const size_t size)
{
    float const scale_value = invert_scale ? 1.0f / scale[0] : scale[0];
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x)
    {
        dst[tid] = cuda_cast<T_OUT>(cuda_cast<float>(src[tid]) * scale_value);
    }
}

template <typename T_IN, typename T_OUT>
void invokeCudaD2DScaleCpyConvert(
    T_OUT* tgt, const T_IN* src, float const* scale, bool invert_scale, const size_t size, cudaStream_t stream)
{
    cudaD2DScaleCpyConvert<<<256, 256, 0, stream>>>(tgt, src, scale, invert_scale, size);
}

// clang-format off
template void invokeCudaD2DScaleCpyConvert(float* tgt, const int32_t* src, const float* scale, bool invert_scale, const size_t size, cudaStream_t stream);
template void invokeCudaD2DScaleCpyConvert(int32_t* tgt, const float* src, const float* scale, bool invert_scale, const size_t size, cudaStream_t stream);
template void invokeCudaD2DScaleCpyConvert(half* tgt, const int32_t* src, const float* scale, bool invert_scale, const size_t size, cudaStream_t stream);
template void invokeCudaD2DScaleCpyConvert(int32_t* tgt, const half* src, const float* scale, bool invert_scale, const size_t size, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeCudaD2DScaleCpyConvert(__nv_bfloat16* tgt, const int32_t* src, const float* scale, bool invert_scale, const size_t size, cudaStream_t stream);
template void invokeCudaD2DScaleCpyConvert(int32_t* tgt, const __nv_bfloat16* src, const float* scale, bool invert_scale, const size_t size, cudaStream_t stream);
#endif  // ENABLE_BF16
#ifdef ENABLE_FP8
template void invokeCudaD2DScaleCpyConvert(float* tgt, const __nv_fp8_e4m3* src, const float* scale, bool invert_scale, const size_t size, cudaStream_t stream);
#endif  // ENABLE_FP8
// clang-format on

void invokeCudaD2DcpyHalf2Float(float* dst, half* src, const size_t size, cudaStream_t stream)
{
    invokeCudaD2DcpyConvert(dst, src, size, stream);
}

void invokeCudaD2DcpyFloat2Half(half* dst, float* src, const size_t size, cudaStream_t stream)
{
    invokeCudaD2DcpyConvert(dst, src, size, stream);
}

template <typename T>
void saveToBinary(T const* ptr, const size_t size, std::string filename)
{

    std::vector<T> h_ptr(size);
    cudaD2Hcpy(h_ptr.data(), ptr, size);
    std::vector<float> float_ptr(size);
    for (size_t i = 0; i < size; i++)
    {
        float_ptr[i] = (float) h_ptr[i];
    }

    std::ofstream out(filename, std::ios::out | std::ios::binary);
    TLLM_CHECK_WITH_INFO(out.is_open(), "Fail to open file " + filename);

    out.write((char*) float_ptr.data(), size * sizeof(float));
}

template void saveToBinary(float const* ptr, const size_t size, std::string filename);
template void saveToBinary(half const* ptr, const size_t size, std::string filename);
#ifdef ENABLE_BF16
template void saveToBinary(__nv_bfloat16 const* ptr, const size_t size, std::string filename);
#endif // ENABLE_BF16

template <>
void saveToBinary(int const* ptr, const size_t size, std::string filename)
{
    std::vector<int> h_ptr(size);
    cudaD2Hcpy(h_ptr.data(), ptr, size);
    std::ofstream out(filename, std::ios::out | std::ios::binary);
    TLLM_CHECK_WITH_INFO(out.is_open(), "Fail to open file " + filename);
    out.write((char*) h_ptr.data(), size * sizeof(int));
}

template <typename T_IN, typename T_fake_type>
__global__ void fakeCast(T_IN* input_ptr, const size_t size)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
    {
        T_fake_type tmp_val = (T_fake_type) ((float) input_ptr[i]);
        input_ptr[i] = (T_IN) ((float) tmp_val);
    }
}

template <typename T_IN, typename T_fake_type>
void invokeFakeCast(T_IN* input_ptr, const size_t size, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((size + 255) / 256);
    fakeCast<T_IN, T_fake_type><<<grid, block, 0, stream>>>(input_ptr, size);
}

#ifdef ENABLE_FP8
__global__ void cudaD2Dcpyfp82Float(float* dst, __nv_fp8_e4m3* src, const size_t size)
{
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x)
    {
        dst[tid] = (float) (src[tid]);
    }
}

void invokeCudaD2Dcpyfp82Float(float* dst, __nv_fp8_e4m3* src, const size_t size, cudaStream_t stream)
{
    cudaD2Dcpyfp82Float<<<256, 256, 0, stream>>>(dst, src, size);
}

__global__ void cudaD2Dcpyfp82Half(half* dst, __nv_fp8_e4m3* src, const size_t size)
{
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x)
    {
        dst[tid] = (half) ((float) (src[tid]));
    }
}

void invokeCudaD2Dcpyfp82Half(half* dst, __nv_fp8_e4m3* src, const size_t size, cudaStream_t stream)
{
    cudaD2Dcpyfp82Half<<<256, 256, 0, stream>>>(dst, src, size);
}

__global__ void cudaD2DcpyFloat2fp8(__nv_fp8_e4m3* dst, float* src, const size_t size)
{
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x)
    {
        dst[tid] = (__nv_fp8_e4m3) src[tid];
    }
}

void invokeCudaD2DcpyFloat2fp8(__nv_fp8_e4m3* dst, float* src, const size_t size, cudaStream_t stream)
{
    cudaD2DcpyFloat2fp8<<<256, 256, 0, stream>>>(dst, src, size);
}

__global__ void cudaD2DcpyHalf2fp8(__nv_fp8_e4m3* dst, half* src, const size_t size)
{
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x)
    {
        dst[tid] = (__nv_fp8_e4m3) src[tid];
    }
}

void invokeCudaD2DcpyHalf2fp8(__nv_fp8_e4m3* dst, half* src, const size_t size, cudaStream_t stream)
{
    cudaD2DcpyHalf2fp8<<<256, 256, 0, stream>>>(dst, src, size);
}

__global__ void cudaD2DcpyBfloat2fp8(__nv_fp8_e4m3* dst, __nv_bfloat16* src, const size_t size)
{
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x)
    {
        dst[tid] = (__nv_fp8_e4m3) src[tid];
    }
}

void invokeCudaD2DcpyBfloat2fp8(__nv_fp8_e4m3* dst, __nv_bfloat16* src, const size_t size, cudaStream_t stream)
{
    cudaD2DcpyBfloat2fp8<<<256, 256, 0, stream>>>(dst, src, size);
}

#endif // ENABLE_FP8

template <typename T_OUT, typename T_IN>
__global__ void transpose(T_OUT* dst, T_IN* src, const size_t dim0, const size_t dim1)
{
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < dim0 * dim1; tid += blockDim.x * gridDim.x)
    {
        const size_t src_col_id = tid % dim1;
        const size_t src_row_id = tid / dim1;
        dst[src_col_id * dim0 + src_row_id] = (T_OUT) (src[tid]);
    }
}

template <typename T>
void invokeInPlaceTranspose(T* data, T* workspace, const size_t dim0, const size_t dim1)
{
    // copy data to workspace, and then transpose from workspace to data
    cudaD2Dcpy(workspace, data, dim0 * dim1);
    transpose<<<256, 256>>>(data, workspace, dim0, dim1);
}

#ifdef ENABLE_FP8
template void invokeInPlaceTranspose(
    __nv_fp8_e4m3* data, __nv_fp8_e4m3* workspace, const size_t dim0, const size_t dim1);
#endif // ENABLE_FP8
#ifdef ENABLE_BF16
template void invokeInPlaceTranspose(
    __nv_bfloat16* data, __nv_bfloat16* workspace, const size_t dim0, const size_t dim1);
#endif // ENABLE_BF16
template void invokeInPlaceTranspose(float* data, float* workspace, const size_t dim0, const size_t dim1);

template <typename T_OUT, typename T_IN>
__global__ void transpose0213(
    T_OUT* dst, T_IN* src, const size_t dim0, const size_t dim1, const size_t dim2, const size_t dim3)
{
    // src permutation: [0, 1, 2, 3]
    // dst permutation: [0, 2, 1, 3]
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < dim0 * dim1 * dim2 * dim3;
         tid += blockDim.x * gridDim.x)
    {
        size_t tmp_idx = tid;
        const size_t dim_3_idx = tmp_idx % dim3;
        tmp_idx = (tmp_idx - dim_3_idx) / dim3;
        const size_t dim_2_idx = tmp_idx % dim2;
        tmp_idx = (tmp_idx - dim_2_idx) / dim2;
        const size_t dim_1_idx = tmp_idx % dim1;
        tmp_idx = (tmp_idx - dim_1_idx) / dim1;
        const size_t dim_0_idx = tmp_idx % dim0;
        dst[dim_0_idx * dim1 * dim2 * dim3 + dim_2_idx * dim1 * dim3 + dim_1_idx * dim3 + dim_3_idx] = src[tid];
    }
}

template <typename T>
void invokeInPlaceTranspose0213(
    T* data, T* workspace, const size_t dim0, const size_t dim1, const size_t dim2, const size_t dim3)
{
    // copy data to workspace, and then transpose from workspace to data
    // Note that this kernel is used for pre-processing and not very efficient.
    cudaD2Dcpy(workspace, data, dim0 * dim1 * dim2 * dim3);
    transpose0213<<<256, 256>>>(data, workspace, dim0, dim1, dim2, dim3);
}

#ifdef ENABLE_FP8
template void invokeInPlaceTranspose0213(__nv_fp8_e4m3* data, __nv_fp8_e4m3* workspace, const size_t dim0,
    const size_t dim1, const size_t dim2, const size_t dim3);
#endif // ENABLE_FP8
#ifdef ENABLE_BF16
template void invokeInPlaceTranspose0213(__nv_bfloat16* data, __nv_bfloat16* workspace, const size_t dim0,
    const size_t dim1, const size_t dim2, const size_t dim3);
#endif // ENABLE_BF16
template void invokeInPlaceTranspose0213(
    float* data, float* workspace, const size_t dim0, const size_t dim1, const size_t dim2, const size_t dim3);

template <typename T_OUT, typename T_IN>
__global__ void transpose102(T_OUT* dst, T_IN* src, const size_t dim0, const size_t dim1, const size_t dim2)
{
    // src permutation: [0, 1, 2]
    // dst permutation: [1, 0, 2]
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < dim0 * dim1 * dim2; tid += blockDim.x * gridDim.x)
    {
        size_t tmp_idx = tid;
        const size_t dim_2_idx = tmp_idx % dim2;
        tmp_idx = (tmp_idx - dim_2_idx) / dim2;
        const size_t dim_1_idx = tmp_idx % dim1;
        tmp_idx = (tmp_idx - dim_1_idx) / dim1;
        const size_t dim_0_idx = tmp_idx % dim0;
        dst[dim_1_idx * dim0 * dim2 + dim_0_idx * dim2 + dim_2_idx] = src[tid];
    }
}

template <typename T>
void invokeInPlaceTranspose102(T* data, T* workspace, const size_t dim0, const size_t dim1, const size_t dim2)
{
    // copy data to workspace, and then transpose from workspace to data
    // Note that this kernel is used for pre-processing and not very efficient.
    cudaD2Dcpy(workspace, data, dim0 * dim1 * dim2);
    transpose102<<<256, 256>>>(data, workspace, dim0, dim1, dim2);
}

#ifdef ENABLE_FP8
template void invokeInPlaceTranspose102(
    __nv_fp8_e4m3* data, __nv_fp8_e4m3* workspace, const size_t dim0, const size_t dim1, const size_t dim2);
#endif // ENABLE_FP8
#ifdef ENABLE_BF16
template void invokeInPlaceTranspose102(
    __nv_bfloat16* data, __nv_bfloat16* workspace, const size_t dim0, const size_t dim1, const size_t dim2);
#endif // ENABLE_BF16
template void invokeInPlaceTranspose102(
    float* data, float* workspace, const size_t dim0, const size_t dim1, const size_t dim2);

template <typename T>
void __global__ multiplyScale(T* tensor, float scale, const size_t size)
{
    for (size_t index = threadIdx.x + blockIdx.x * blockDim.x; index < size; index += blockDim.x * gridDim.x)
    {
        tensor[index] = (T) (((float) tensor[index]) * scale);
    }
}

template <typename T>
void invokeMultiplyScale(T* tensor, float scale, const size_t size, cudaStream_t stream)
{
    int block = 256;
    int grid = (size + 255) / 256;
    multiplyScale<<<grid, block, 0, stream>>>(tensor, scale, size);
}

template void invokeMultiplyScale(float* tensor, float scale, const size_t size, cudaStream_t stream);
template void invokeMultiplyScale(half* tensor, float scale, const size_t size, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeMultiplyScale(__nv_bfloat16* tensor, float scale, const size_t size, cudaStream_t stream);
#endif
#ifdef ENABLE_FP8
template void invokeMultiplyScale(__nv_fp8_e4m3* tensor, float scale, const size_t size, cudaStream_t stream);
#endif

template <typename T>
void __global__ divideScale(T* tensor, float scale, const size_t size)
{
    for (size_t index = threadIdx.x + blockIdx.x * blockDim.x; index < size; index += blockDim.x * gridDim.x)
    {
        tensor[index] = (T) (((float) tensor[index]) / scale);
    }
}

template <typename T>
void invokeDivideScale(T* tensor, float scale, const size_t size, cudaStream_t stream)
{
    int block = 256;
    int grid = (size + 255) / 256;
    divideScale<<<grid, block, 0, stream>>>(tensor, scale, size);
}

template void invokeDivideScale(float* tensor, float scale, const size_t size, cudaStream_t stream);
template void invokeDivideScale(half* tensor, float scale, const size_t size, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeDivideScale(__nv_bfloat16* tensor, float scale, const size_t size, cudaStream_t stream);
#endif
#ifdef ENABLE_FP8
template void invokeDivideScale(__nv_fp8_e4m3* tensor, float scale, const size_t size, cudaStream_t stream);
#endif
#ifdef ENABLE_BF16
template void invokeFakeCast<float, __nv_bfloat16>(float* input_ptr, const size_t size, cudaStream_t stream);
template void invokeFakeCast<__nv_bfloat16, __nv_bfloat16>(
    __nv_bfloat16* input_ptr, const size_t size, cudaStream_t stream);
template void invokeFakeCast<half, __nv_bfloat16>(half* input_ptr, const size_t size, cudaStream_t stream);
#endif
template void invokeFakeCast<float, half>(float* input_ptr, const size_t size, cudaStream_t stream);
template void invokeFakeCast<float, float>(float* input_ptr, const size_t size, cudaStream_t stream);
#ifdef ENABLE_FP8
template void invokeFakeCast<float, __nv_fp8_e4m3>(float* input_ptr, const size_t size, cudaStream_t stream);
template void invokeFakeCast<half, __nv_fp8_e4m3>(half* input_ptr, const size_t size, cudaStream_t stream);
template void invokeFakeCast<__nv_bfloat16, __nv_fp8_e4m3>(
    __nv_bfloat16* input_ptr, const size_t size, cudaStream_t stream);
#endif

size_t cuda_datatype_size(TRTLLMCudaDataType dt)
{
    static const std::unordered_map<TRTLLMCudaDataType, size_t> sizes{
        {TRTLLMCudaDataType::FP32, sizeof(float)}, {TRTLLMCudaDataType::FP16, sizeof(half)}
#ifdef ENABLE_BF16
        ,
        {TRTLLMCudaDataType::BF16, sizeof(__nv_bfloat16)}
#endif
    };

    return sizes.at(dt);
}

template <typename T>
__global__ void check_range(T const* buffer, size_t size, T min, T max, bool* d_within_range)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
    {
        const T val = buffer[i];
        if (val < min || val > max)
        {
            *d_within_range = false;
        }
    }
}

template <typename T>
bool invokeCheckRange(T const* buffer, const size_t size, T min, T max, bool* d_within_range, cudaStream_t stream)
{
    cudaMemsetAsync(d_within_range, true, sizeof(bool), stream);

    dim3 block(256);
    dim3 grid((size + 255) / 256);
    check_range<T><<<grid, block, 0, stream>>>(buffer, size, min, max, d_within_range);

    bool result;
    cudaD2Hcpy(&result, d_within_range, 1);
    return result;
}

template bool invokeCheckRange<int>(
    int const* buffer, const size_t size, int min, int max, bool* d_within_range, cudaStream_t stream);

/*
 *  Determine the total workspace size based on a vector containing multiple variable sizes.
 */
size_t calcAlignedSize(std::vector<size_t> const& sizes, const size_t ALIGN_BYTES)
{
    const size_t ALIGN_MASK = ~(ALIGN_BYTES - 1);
    // Check ALIGN_BYTES is a power of 2
    assert((ALIGN_BYTES & (ALIGN_BYTES - 1)) == 0);

    size_t total = 0;
    for (auto sz : sizes)
    {
        total += (sz + ALIGN_BYTES - 1) & ALIGN_MASK;
    }

    // We add extra "ALIGN_BYTES - 1" bytes in case the start address passed to the function calcAlignedPointers() is
    // not aligned.
    return total + ALIGN_BYTES - 1;
}

/*
 * Given the address of the workspace and the vector containing multiple variable sizes, calculate the start addresses
 * of each variable.
 */
void calcAlignedPointers(
    std::vector<void*>& outPtrs, void const* p, std::vector<size_t> const& sizes, size_t ALIGN_BYTES)
{
    const size_t ALIGN_MASK = ~(ALIGN_BYTES - 1);
    // Check ALIGN_BYTES is a power of 2
    assert((ALIGN_BYTES & (ALIGN_BYTES - 1)) == 0);

    // In case the start address is not aligned
    char* ptr = reinterpret_cast<char*>((reinterpret_cast<size_t>(p) + ALIGN_BYTES - 1) & ALIGN_MASK);

    outPtrs.reserve(sizes.size());
    for (auto sz : sizes)
    {
        outPtrs.push_back(ptr);
        ptr += (sz + ALIGN_BYTES - 1) & ALIGN_MASK;
    }
}

} // namespace common
} // namespace tensorrt_llm
