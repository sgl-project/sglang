/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Interface between a CUTLASS device-wide operator and CUDA.
*/

#pragma once

#include <cuda_runtime_api.h>
#include "cutlass/cutlass.h"
#include "cutlass/trace.h"

#include "cutlass/platform/platform.h"
#if ! defined(__CUDACC_RTC__)
#include <cstdio>
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

// NVRTC doesn't need definitions for these host classes

#if ((__CUDACC_VER_MAJOR__ >= 12) ||                               \
    ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8))) \
    && !defined(__CUDACC_RTC__)
#define CUDA_HOST_ADAPTER_LAUNCH_ATTRIBUTES_ENABLED
#endif

#if ((__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__))
#define CUDA_HOST_ADAPTER_TENSORMAP_ENABLED
#endif

// Include <cuda.h> for CUDA Driver API calls if any of these capabilities are enabled.
#if defined(CUDA_HOST_ADAPTER_LAUNCH_ATTRIBUTES_ENABLED) ||        \
    defined(CUDA_HOST_ADAPTER_TENSORMAP_ENABLED)

#include <cuda.h>

#endif // defined(CUDA_HOST_ADAPTER_LAUNCH_ATTRIBUTES_ENABLED) ||
       // defined(CUDA_HOST_ADAPTER_TENSORMAP_ENABLED)

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Macro-level guard for CUDA Host Adapter
//
#if !defined(CUTLASS_ENABLE_CUDA_HOST_ADAPTER)
#define CUTLASS_ENABLE_CUDA_HOST_ADAPTER false
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////


#if !defined(__CUDACC_RTC__)

#if ((__CUDACC_VER_MAJOR__ >= 12) ||                               \
    ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8)))
#include <cudaTypedefs.h>
#endif // (__CUDACC_VERSION__ >= 11.8)

#include <driver_types.h>

#define CUTLASS_CUDA_DRIVER_STRINGIFY(tok) #tok

#if defined(CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL)

#define CUTLASS_CUDA_DRIVER_WRAPPER_DECL(func, ver) \
  template <typename... Args>                       \
  CUresult call_##func(Args... args) {              \
    return func(args...);                           \
  }

#else // defined(CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL)

#if (__CUDACC_VER_MAJOR__ > 12)

#define CUTLASS_CUDA_DRIVER_WRAPPER_DECL(func, ver)             \
  template <typename... Args>                                   \
  CUresult call_##func(Args... args) {                          \
    cudaDriverEntryPointQueryResult cuda_status;                \
    void* pfn = nullptr;                                        \
    cudaError_t cuda_err = cudaGetDriverEntryPointByVersion(    \
        CUTLASS_CUDA_DRIVER_STRINGIFY(func),                    \
        &pfn, ver,                                              \
        cudaEnableDefault,                                      \
        &cuda_status);                                          \
    if (cuda_status != cudaDriverEntryPointSuccess ||           \
        cuda_err != cudaSuccess) {                              \
      return CUDA_ERROR_UNKNOWN;                                \
    }                                                           \
    return reinterpret_cast<PFN_##func##_v##ver>(pfn)(args...); \
  }

#else

#define CUTLASS_CUDA_DRIVER_WRAPPER_DECL(func, ver)             \
  template <typename... Args>                                   \
  CUresult call_##func(Args... args) {                          \
    cudaDriverEntryPointQueryResult cuda_status;                \
    void* pfn = nullptr;                                        \
    cudaError_t cuda_err = cudaGetDriverEntryPoint(             \
        CUTLASS_CUDA_DRIVER_STRINGIFY(func),                    \
        &pfn,                                                   \
        cudaEnableDefault,                                      \
        &cuda_status);                                          \
    if (cuda_status != cudaDriverEntryPointSuccess ||           \
        cuda_err != cudaSuccess) {                              \
      return CUDA_ERROR_UNKNOWN;                                \
    }                                                           \
    return reinterpret_cast<PFN_##func>(pfn)(args...);          \
  }

#endif // (__CUDACC_VER_MAJOR__ > 12)

#endif // defined(CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL)

#if (__CUDACC_VER_MAJOR__ >= 12)
CUTLASS_CUDA_DRIVER_WRAPPER_DECL(cuTensorMapEncodeTiled, 12000);
CUTLASS_CUDA_DRIVER_WRAPPER_DECL(cuTensorMapEncodeIm2col, 12000);
#endif

#undef CUTLASS_CUDA_DRIVER_STRINGIFY

#define CUTLASS_CUDA_DRIVER_WRAPPER_CALL(func) cutlass::call_##func

#endif // !defined(__CUDACC_RTC__)


/////////////////////////////////////////////////////////////////////////////////////////////////

/// This class manages runtime CUlaunchAttribute that can be supplied to CudaHostAdapter
/// CudaHostLaunchAttributes will be an empty struct in earlier CTK where CUlaunchAttribute
/// is not introduced.
struct CudaHostLaunchAttributes {

#if defined(CUDA_HOST_ADAPTER_LAUNCH_ATTRIBUTES_ENABLED)

  /// Reasonable maximum launch attributes that are commonly applied
  static constexpr int32_t kMaximumAttributeCount = 5;

  /// Launch attributes
  CUlaunchAttribute launch_attributes[kMaximumAttributeCount];
  int32_t      attribute_count = 0;

  CUTLASS_HOST_DEVICE
  CudaHostLaunchAttributes(CUlaunchAttribute *launch_attributes_ = nullptr,
                           int32_t attribute_count_ = 0) {
    CUTLASS_ASSERT(attribute_count_ >= 0 && attribute_count_ < kMaximumAttributeCount);
    for (int32_t i = 0; i < attribute_count_ && i < kMaximumAttributeCount; ++i) {
      launch_attributes[i] = launch_attributes_[i];
    }
    attribute_count = attribute_count_;
  }

  CUTLASS_HOST_DEVICE
  CUlaunchAttribute const* data() const {
    return launch_attributes;
  }

  CUTLASS_HOST_DEVICE
  size_t size() const {
    return attribute_count;
  }
  
#endif // (CUDA_HOST_ADAPTER_LAUNCH_ATTRIBUTES_ENABLED)

};


/// This class defines an object which abstracts interactions between the CUTLASS device-wide GEMM and
/// CUDA. The intention is to enable CUTLASS to be used with both the CUDA Runtime API and CUDA Driver API.
struct CudaHostAdapter {

  /// Limit the number of kernels
  static constexpr int32_t kMaximumKernelCount = 4;

  /// Maximum cluster size
  static constexpr int MaxClusterSize = 32;

  //
  // Data members
  //

  /// Handles
  void        *kernel_handles[kMaximumKernelCount];
  int32_t      kernel_count = 0;

  CudaHostLaunchAttributes launch_attributes;

  //
  // Methods
  //

  /// Ctor
  CudaHostAdapter() = default;

  /// Dtor
  virtual ~CudaHostAdapter() = default;

  /// Copy Ctor
  CUTLASS_HOST_DEVICE
  CudaHostAdapter(const CudaHostAdapter & rhs)
      : kernel_count(rhs.kernel_count),
        launch_attributes(rhs.launch_attributes) {
    CUTLASS_ASSERT(rhs.kernel_count >= 0 && rhs.kernel_count < kMaximumKernelCount);

    for (int32_t i = 0; i < rhs.kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = rhs.kernel_handles[i];
    }
  }

  /// Copy Assignment
  CUTLASS_HOST_DEVICE
  CudaHostAdapter& operator=(const CudaHostAdapter & rhs) {
    CUTLASS_ASSERT(rhs.kernel_count >= 0 && rhs.kernel_count < kMaximumKernelCount);
    for (int32_t i = 0; i < rhs.kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = rhs.kernel_handles[i];
    }
    kernel_count = rhs.kernel_count;

    launch_attributes = rhs.launch_attributes;

    return *this;
  }


  /// Move ctor
  CUTLASS_HOST_DEVICE
  CudaHostAdapter(CudaHostAdapter && rhs)
      : kernel_count(rhs.kernel_count),
        launch_attributes(std::move(rhs.launch_attributes)) {
    CUTLASS_ASSERT(rhs.kernel_count >= 0 && rhs.kernel_count < kMaximumKernelCount);

    for (int32_t i = 0; i < rhs.kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = rhs.kernel_handles[i];
    }
  }

  // / Move assignment
  CUTLASS_HOST_DEVICE 
  CudaHostAdapter& operator=(CudaHostAdapter && rhs) {
    CUTLASS_ASSERT(rhs.kernel_count >= 0 && rhs.kernel_count < kMaximumKernelCount);
    for (int32_t i = 0; i < rhs.kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = rhs.kernel_handles[i];
    }
    kernel_count = rhs.kernel_count;
    launch_attributes = std::move(rhs.launch_attributes);
    return *this;
  }

  /// Ctor
  CUTLASS_HOST_DEVICE
  CudaHostAdapter(void **kernel_handles_, 
                  int32_t kernel_count_,
                  CudaHostLaunchAttributes const &launch_attributes_ = { })
      : kernel_count(kernel_count_),
        launch_attributes(launch_attributes_) {
    CUTLASS_ASSERT(kernel_count >= 0 && kernel_count < kMaximumKernelCount);

    for (int32_t i = 0; i < kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = kernel_handles_[i];
    }
  }

  /// Returns true if the CudaHostAdapter is empty (kernel_count == 0)
  CUTLASS_HOST_DEVICE 
  bool empty() const { return !kernel_count; }

  /// Returns kernel_count
  CUTLASS_HOST_DEVICE
  size_t size() const { return static_cast<size_t>(kernel_count); }

  /// Queries the occupancy of a kernel
  virtual Status query_occupancy(
    int32_t *device_sms, 
    int32_t *sm_occupancy,
    int32_t kernel_index,
    int32_t thread_count,
    int32_t smem_size) const = 0;
 
  /// Launches a kernel without using Threadblock Clusters. 
  virtual Status launch(
    dim3 const grid_dims,
    dim3 const block_dims,
    size_t const smem_size,
    cudaStream_t cuda_stream,
    void** kernel_params,
    int32_t kernel_index) const = 0;

  /// Launches a kernel using the CUDA Extensible Launch API and Threadblock Clusters.
  virtual Status launch(
    dim3 const grid_dims,
    dim3 const cluster_dims,
    dim3 const block_dims,
    size_t const smem_size,
    cudaStream_t cuda_stream,
    void** kernel_params,
    int32_t kernel_index) const = 0;

  

  /// Launches a kernel using the CUDA Extensible Launch API and Threadblock Clusters.
  /// This API is for preferred cluster launch; a preferred and a fallback cluster shapes are
  /// considered for launch respectively.
  virtual Status launch(
    dim3 const grid_dims,
    dim3 const cluster_dims,
    dim3 const fallback_cluster_dims,
    dim3 const block_dims,
    size_t const smem_size,
    cudaStream_t cuda_stream,
    void** kernel_params,
    int32_t kernel_index) const = 0;

  

#if defined(CUDA_HOST_ADAPTER_TENSORMAP_ENABLED)

  /// Create a tensor map descriptor object representing im2col memory region.
  virtual CUresult tensorMapEncodeIm2col (
    CUtensorMap* tensorMap,
    CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank,
    void* globalAddress,
    const cuuint64_t* globalDim,
    const cuuint64_t* globalStrides,
    const int* pixelBoxLowerCorner,
    const int* pixelBoxUpperCorner,
    cuuint32_t channelsPerPixel,
    cuuint32_t pixelsPerColumn,
    const cuuint32_t* elementStrides,
    CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) const = 0;

  /// Create a tensor map descriptor object representing tiled memory region.
  virtual CUresult tensorMapEncodeTiled (
    CUtensorMap* tensorMap,
    CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank,
    void* globalAddress,
    const cuuint64_t* globalDim,
    const cuuint64_t* globalStrides,
    const cuuint32_t* boxDim,
    const cuuint32_t* elementStrides,
    CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) const = 0;

  /// Modify an existing tensor map descriptor with an updated global address.
  virtual CUresult tensorMapReplaceAddress(
    CUtensorMap* tensorMap,
    void* globalAddress)  const = 0;

#endif // defined(CUDA_HOST_ADAPTER_TENSORMAP_ENABLED)

protected:

  /**
   * Fills a buffer in Global Memory with a byte sequence copied from host memory.
   * This function can be overridden to dispatch to the appropriate cuMemsetD*Async API
  */
  virtual Status memsetDeviceImpl(
    void* destination, ///< Device memory pointer to be filled
    void const* fill_value, ///< Value to be filled in the buffer
    size_t fill_size, ///< Size of the data type to be used for filling the buffer
    size_t count, ///< Number of elements of size fill_size
    cudaStream_t stream) const = 0;

public:

  /// Fills a buffer in Global Memory with a byte sequence copied from host memory
  template<class FillValueType>
  CUTLASS_HOST_DEVICE
  Status memsetDevice(
      void* destination,
      FillValueType fill_value, 
      size_t count,
      cudaStream_t stream) const {
    return this->memsetDeviceImpl(
      destination,
      &fill_value,
      sizeof(FillValueType),
      count,
      stream);
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
