/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef CUDA_DRIVER_WRAPPER_H
#define CUDA_DRIVER_WRAPPER_H

#include "tensorrt_llm/common/assert.h"
#include <cstdio>
#include <cuda.h>
#include <memory>
#include <mutex>

namespace tensorrt_llm::common
{

class CUDADriverWrapper
{
public:
    static std::shared_ptr<CUDADriverWrapper> getInstance();

    ~CUDADriverWrapper();
    CUDADriverWrapper(CUDADriverWrapper const&) = delete;
    CUDADriverWrapper operator=(CUDADriverWrapper const&) = delete;
    CUDADriverWrapper(CUDADriverWrapper&&) = delete;
    CUDADriverWrapper operator=(CUDADriverWrapper&&) = delete;

    CUresult cuGetErrorName(CUresult error, char const** pStr) const;

    CUresult cuGetErrorMessage(CUresult error, char const** pStr) const;

    CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) const;

    CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) const;

    CUresult cuModuleUnload(CUmodule hmod) const;

    CUresult cuLinkDestroy(CUlinkState state) const;

    CUresult cuModuleLoadData(CUmodule* module, void const* image) const;

    CUresult cuLinkCreate(
        unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) const;

    CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, char const* name) const;

    CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, char const* name) const;

    CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, char const* path, unsigned int numOptions,
        CUjit_option* options, void** optionValues) const;

    CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size, char const* name,
        unsigned int numOptions, CUjit_option* options, void** optionValues) const;

    CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
        unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
        unsigned int sharedMemBytes, CUstream hStream, void** kernelParams) const;

    CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
        CUstream hStream, void** kernelParams, void** extra) const;

    CUresult cuTensorMapEncodeTiled(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank,
        void* globalAddress, cuuint64_t const* globalDim, cuuint64_t const* globalStrides, cuuint32_t const* boxDim,
        cuuint32_t const* elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
        CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) const;

    CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) const;

private:
    void* handle;
    CUDADriverWrapper();

    CUresult (*_cuGetErrorName)(CUresult, char const**);
    CUresult (*_cuGetErrorMessage)(CUresult, char const**);
    CUresult (*_cuFuncSetAttribute)(CUfunction, CUfunction_attribute, int);
    CUresult (*_cuLinkComplete)(CUlinkState, void**, size_t*);
    CUresult (*_cuModuleUnload)(CUmodule);
    CUresult (*_cuLinkDestroy)(CUlinkState);
    CUresult (*_cuLinkCreate)(unsigned int, CUjit_option*, void**, CUlinkState*);
    CUresult (*_cuModuleLoadData)(CUmodule*, void const*);
    CUresult (*_cuModuleGetFunction)(CUfunction*, CUmodule, char const*);
    CUresult (*_cuModuleGetGlobal)(CUdeviceptr*, size_t*, CUmodule, char const*);
    CUresult (*_cuLinkAddFile)(CUlinkState, CUjitInputType, char const*, unsigned int, CUjit_option*, void**);
    CUresult (*_cuLinkAddData)(
        CUlinkState, CUjitInputType, void*, size_t, char const*, unsigned int, CUjit_option*, void**);
    CUresult (*_cuLaunchCooperativeKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, CUstream, void**);
    CUresult (*_cuLaunchKernel)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
        CUstream hStream, void** kernelParams, void** extra);
    CUresult (*_cuTensorMapEncodeTiled)(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType,
        cuuint32_t tensorRank, void* globalAddress, cuuint64_t const* globalDim, cuuint64_t const* globalStrides,
        cuuint32_t const* boxDim, cuuint32_t const* elementStrides, CUtensorMapInterleave interleave,
        CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill);
    CUresult (*_cuMemcpyDtoH)(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
};

template <typename T>
void checkDriver(
    T result, CUDADriverWrapper const& wrap, char const* const func, char const* const file, int const line)
{
    if (result)
    {
        char const* errorName = nullptr;
        char const* errorMsg = nullptr;
        wrap.cuGetErrorName(result, &errorName);
        wrap.cuGetErrorMessage(result, &errorMsg);
        throw TllmException(
            file, line, fmtstr("[TensorRT-LLM][ERROR] CUDA driver error in %s: %s: %s", func, errorName, errorMsg));
    }
}

} // namespace tensorrt_llm::common

/*
 * Macros compliant with TensorRT coding conventions
 */
#define TLLM_CU_CHECK(stat)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        tensorrt_llm::common::checkDriver(                                                                             \
            (stat), *tensorrt_llm::common::CUDADriverWrapper::getInstance(), #stat, __FILE__, __LINE__);               \
    } while (0)

#endif // CUDA_DRIVER_WRAPPER_H
