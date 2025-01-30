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

#pragma once

#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cassert>

namespace tensorrt_llm
{
namespace common
{

template <typename T>
void deviceMalloc(T** ptr, size_t size, bool is_random_initialize = true);

template <typename T>
void deviceMemSetZero(T* ptr, size_t size);

template <typename T>

void deviceFree(T*& ptr);

template <typename T>
void deviceFill(T* devptr, size_t size, T value, cudaStream_t stream = 0);

template <typename T>
void cudaD2Hcpy(T* tgt, T const* src, size_t const size);

template <typename T>
void cudaH2Dcpy(T* tgt, T const* src, size_t const size);

template <typename T>
void cudaD2Dcpy(T* tgt, T const* src, size_t const size, cudaStream_t stream = NULL);

template <typename T>
void cudaAutoCpy(T* tgt, T const* src, size_t const size, cudaStream_t stream = NULL);

template <typename T>
void cudaRandomUniform(T* buffer, size_t const size);

template <typename T>
int loadWeightFromBin(T* ptr, std::vector<size_t> shape, std::string filename,
    TRTLLMCudaDataType model_file_type = TRTLLMCudaDataType::FP32);

// template<typename T>
// int loadWeightFromBinAndQuantizeForWeightOnly(int8_t*             quantized_weight_ptr,
//                                               T*                  scale_ptr,
//                                               std::vector<size_t> shape,
//                                               std::string         filename,
//                                               TRTLLMCudaDataType  model_file_type = TRTLLMCudaDataType::FP32);

void invokeCudaD2DcpyHalf2Float(float* dst, half* src, size_t const size, cudaStream_t stream);
void invokeCudaD2DcpyFloat2Half(half* dst, float* src, size_t const size, cudaStream_t stream);
#ifdef ENABLE_FP8
void invokeCudaD2Dcpyfp82Float(float* dst, __nv_fp8_e4m3* src, size_t const size, cudaStream_t stream);
void invokeCudaD2Dcpyfp82Half(half* dst, __nv_fp8_e4m3* src, size_t const size, cudaStream_t stream);
void invokeCudaD2DcpyFloat2fp8(__nv_fp8_e4m3* dst, float* src, size_t const size, cudaStream_t stream);
void invokeCudaD2DcpyHalf2fp8(__nv_fp8_e4m3* dst, half* src, size_t const size, cudaStream_t stream);
void invokeCudaD2DcpyBfloat2fp8(__nv_fp8_e4m3* dst, __nv_bfloat16* src, size_t const size, cudaStream_t stream);
#endif // ENABLE_FP8
#ifdef ENABLE_BF16
void invokeCudaD2DcpyBfloat2Float(float* dst, __nv_bfloat16* src, size_t const size, cudaStream_t stream);
#endif // ENABLE_BF16

template <typename T_OUT, typename T_IN>
void invokeCudaCast(T_OUT* dst, T_IN const* const src, size_t const size, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////////////////////////

// The following functions implement conversion of multi-dimensional indices to an index in a flat array.
// The shape of the Tensor dimensions is passed as one array (`dims`), the indices are given as individual arguments.
// For examples on how to use these functions, see their tests `test_memory_utils.cu`.
// All of these functions can be evaluated at compile time by recursive template expansion.

template <typename TDim, typename T, typename TIndex>
__inline__ __host__ __device__ std::enable_if_t<std::is_pointer<TDim>::value, T> constexpr flat_index(
    T const& acc, TDim dims, TIndex const& index)
{
    assert(index < dims[0]);
    return acc * dims[0] + index;
}

template <typename TDim, typename T, typename TIndex, typename... TIndices>
__inline__ __host__ __device__ std::enable_if_t<std::is_pointer<TDim>::value, T> constexpr flat_index(
    T const& acc, TDim dims, TIndex const& index, TIndices... indices)
{
    assert(index < dims[0]);
    return flat_index(acc * dims[0] + index, dims + 1, indices...);
}

template <typename TDim, typename T>
__inline__ __host__ __device__ std::enable_if_t<std::is_pointer<TDim>::value, T> constexpr flat_index(
    [[maybe_unused]] TDim dims, T const& index)
{
    assert(index < dims[0]);
    return index;
}

template <typename TDim, typename TIndex, typename... TIndices>
__inline__ __host__ __device__
    std::enable_if_t<std::is_pointer<TDim>::value, typename std::remove_pointer<TDim>::type> constexpr flat_index(
        TDim dims, TIndex const& index, TIndices... indices)
{
    assert(index < dims[0]);
    return flat_index(static_cast<typename std::remove_pointer<TDim>::type>(index), dims + 1, indices...);
}

template <unsigned skip = 0, typename T, std::size_t N, typename TIndex, typename... TIndices>
__inline__ __host__ __device__ T constexpr flat_index(
    std::array<T, N> const& dims, TIndex const& index, TIndices... indices)
{
    static_assert(skip < N);
    static_assert(sizeof...(TIndices) < N - skip, "Number of indices exceeds number of dimensions");
    return flat_index(&dims[skip], index, indices...);
}

template <unsigned skip = 0, typename T, typename TIndex, std::size_t N, typename... TIndices>
__inline__ __host__ __device__ T constexpr flat_index(
    T const& acc, std::array<T, N> const& dims, TIndex const& index, TIndices... indices)
{
    static_assert(skip < N);
    static_assert(sizeof...(TIndices) < N - skip, "Number of indices exceeds number of dimensions");
    return flat_index(acc, &dims[skip], index, indices...);
}

template <unsigned skip = 0, typename T, typename TIndex, std::size_t N, typename... TIndices>
__inline__ __host__ __device__ T constexpr flat_index(T const (&dims)[N], TIndex const& index, TIndices... indices)
{
    static_assert(skip < N);
    static_assert(sizeof...(TIndices) < N - skip, "Number of indices exceeds number of dimensions");
    return flat_index(static_cast<T const*>(dims) + skip, index, indices...);
}

template <unsigned skip = 0, typename T, typename TIndex, std::size_t N, typename... TIndices>
__inline__ __host__ __device__ T constexpr flat_index(
    T const& acc, T const (&dims)[N], TIndex const& index, TIndices... indices)
{
    static_assert(skip < N);
    static_assert(sizeof...(TIndices) < N - skip, "Number of indices exceeds number of dimensions");
    return flat_index(acc, static_cast<T const*>(dims) + skip, index, indices...);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// These are simpler functions for multi-dimensional index conversion. Indices and dimensions are passed as individual
// arguments. These functions are more suitable for usage inside kernels than the corresponding flat_index functions
// which require arrays as arguments. Usage examples can be found in `test_memory_utils.cu`. The functions can be
// evaluated at compile time.

template <typename T, typename TIndex>
__inline__ __host__ __device__ T constexpr flat_index2(TIndex const& index_0, TIndex const& index_1, T const& dim_1)
{
    assert(index_1 < dim_1);
    return index_0 * dim_1 + index_1;
}

template <typename T, typename TIndex>
__inline__ __host__ __device__ T constexpr flat_index3(
    TIndex const& index_0, TIndex const& index_1, TIndex const& index_2, T const& dim_1, T const& dim_2)
{
    assert(index_2 < dim_2);
    return flat_index2(index_0, index_1, dim_1) * dim_2 + index_2;
}

template <typename T, typename TIndex>
__inline__ __host__ __device__ T constexpr flat_index4(TIndex const& index_0, TIndex const& index_1,
    TIndex const& index_2, TIndex const& index_3, T const& dim_1, T const& dim_2, T const& dim_3)
{
    assert(index_3 < dim_3);
    return flat_index3(index_0, index_1, index_2, dim_1, dim_2) * dim_3 + index_3;
}

template <typename T, typename TIndex>
__inline__ __host__ __device__ T constexpr flat_index5(TIndex const& index_0, TIndex const& index_1,
    TIndex const& index_2, TIndex const& index_3, TIndex const& index_4, T const& dim_1, T const& dim_2, T const& dim_3,
    T const& dim_4)
{
    assert(index_4 < dim_4);
    return flat_index4(index_0, index_1, index_2, index_3, dim_1, dim_2, dim_3) * dim_4 + index_4;
}

template <typename T, typename TIndex>
__inline__ __host__ __device__ T constexpr flat_index_strided3(
    TIndex const& index_0, TIndex const& index_1, TIndex const& index_2, T const& stride_1, T const& stride_2)
{
    assert(index_1 < stride_1 / stride_2);
    assert(index_2 < stride_2);
    return index_0 * stride_1 + index_1 * stride_2 + index_2;
}

template <typename T, typename TIndex>
__inline__ __host__ __device__ T constexpr flat_index_strided4(TIndex const& index_0, TIndex const& index_1,
    TIndex const& index_2, TIndex const& index_3, T const& stride_1, T const& stride_2, T const& stride_3)
{
    assert(index_1 < stride_1 / stride_2);
    assert(index_2 < stride_2 / stride_3);
    assert(index_3 < stride_3);
    return index_0 * stride_1 + index_1 * stride_2 + index_2 * stride_3 + index_3;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void invokeInPlaceTranspose(T* data, T* workspace, size_t const dim0, size_t const dim1);

template <typename T>
void invokeInPlaceTranspose0213(
    T* data, T* workspace, size_t const dim0, size_t const dim1, size_t const dim2, size_t const dim3);

template <typename T>
void invokeInPlaceTranspose102(T* data, T* workspace, size_t const dim0, size_t const dim1, size_t const dim2);

template <typename T>
void invokeMultiplyScale(T* tensor, float scale, size_t const size, cudaStream_t stream);

template <typename T>
void invokeDivideScale(T* tensor, float scale, size_t const size, cudaStream_t stream);

template <typename T_IN, typename T_OUT>
void invokeCudaD2DcpyConvert(T_OUT* tgt, const T_IN* src, size_t const size, cudaStream_t stream = 0);

template <typename T_IN, typename T_OUT>
void invokeCudaD2DScaleCpyConvert(
    T_OUT* tgt, const T_IN* src, float const* scale, bool invert_scale, size_t const size, cudaStream_t stream = 0);

inline bool checkIfFileExist(std::string const& file_path)
{
    std::ifstream in(file_path, std::ios::in | std::ios::binary);
    if (in.is_open())
    {
        in.close();
        return true;
    }
    return false;
}

template <typename T>
void saveToBinary(T const* ptr, size_t const size, std::string filename);

template <typename T_IN, typename T_fake_type>
void invokeFakeCast(T_IN* input_ptr, size_t const size, cudaStream_t stream);

size_t cuda_datatype_size(TRTLLMCudaDataType dt);

template <typename T>
bool invokeCheckRange(T const* buffer, size_t const size, T min, T max, bool* d_within_range, cudaStream_t stream);

constexpr size_t DEFAULT_ALIGN_BYTES = 256;

size_t calcAlignedSize(std::vector<size_t> const& sizes, size_t ALIGN_BYTES = DEFAULT_ALIGN_BYTES);
void calcAlignedPointers(std::vector<void*>& outPtrs, void const* p, std::vector<size_t> const& sizes,
    size_t ALIGN_BYTES = DEFAULT_ALIGN_BYTES);

struct AlignedPointersUnpacker
{
    template <typename... T>
    void operator()(T*&... outPtrs)
    {
        assert(sizeof...(T) == alignedPointers.size());
        auto it = alignedPointers.begin();
        ((outPtrs = static_cast<T*>(*it++)), ...);
    }

    std::vector<void*> alignedPointers;
};

AlignedPointersUnpacker inline calcAlignedPointers(
    void const* p, std::vector<size_t> const& sizes, size_t ALIGN_BYTES = DEFAULT_ALIGN_BYTES)
{
    AlignedPointersUnpacker unpacker{};
    calcAlignedPointers(unpacker.alignedPointers, p, sizes, ALIGN_BYTES);
    return unpacker;
}

} // namespace common
} // namespace tensorrt_llm
