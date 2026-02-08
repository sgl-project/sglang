#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <filesystem>

#include "../utils/exception.hpp"
#include "../utils/compatibility.hpp"

namespace deep_gemm {

// Lazy loading all driver symbols
static void* get_driver_handle() {
    static void* handle = nullptr;
    if (handle == nullptr) {
        handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        DG_HOST_ASSERT(handle != nullptr and "Failed to load CUDA driver `libcuda.so.1`");
    }
    return handle;
}

// Macro to define wrapper functions named `lazy_cu{API name}`
#define DECL_LAZY_CUDA_DRIVER_FUNCTION(name) \
template <typename... Args> \
static auto lazy_##name(Args&&... args) -> decltype(name(args...)) { \
    using FuncType = decltype(&name); \
    static FuncType func = nullptr; \
    if (func == nullptr) { \
        func = reinterpret_cast<FuncType>(dlsym(get_driver_handle(), #name)); \
        DG_HOST_ASSERT(func != nullptr and "Failed to load CUDA driver API"); \
    } \
    return func(std::forward<decltype(args)>(args)...); \
}

DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorName);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorString);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuFuncSetAttribute);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleLoad);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleUnload);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleGetFunction);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLaunchKernelEx);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuTensorMapEncodeTiled);

#if CUDART_VERSION >= 12080 and defined(DG_JIT_USE_RUNTIME_API)

// Use CUDA runtime API
using LibraryHandle = cudaLibrary_t;
using KernelHandle = cudaKernel_t;
using LaunchConfigHandle = cudaLaunchConfig_t;
using LaunchAttrHandle = cudaLaunchAttribute;

#define DG_CUDA_UNIFIED_CHECK DG_CUDA_RUNTIME_CHECK

static KernelHandle load_kernel(const std::filesystem::path& cubin_path, const std::string& func_name,
                                LibraryHandle *library_opt = nullptr) {
    LibraryHandle library;
    KernelHandle kernel{};
    DG_CUDA_RUNTIME_CHECK(cudaLibraryLoadFromFile(&library, cubin_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    DG_CUDA_RUNTIME_CHECK(cudaLibraryGetKernel(&kernel, library, func_name.c_str()));

    if (library_opt != nullptr)
        *library_opt = library;
    return kernel;
}

static void unload_library(const LibraryHandle& library) {
    const auto& error = cudaLibraryUnload(library);
    DG_HOST_ASSERT(error == cudaSuccess or error == cudaErrorCudartUnloading);
}

static LaunchConfigHandle construct_launch_config(const KernelHandle& kernel,
                                                  const cudaStream_t& stream, const int& smem_size,
                                                  const dim3& grid_dim, const dim3& block_dim, const int& cluster_dim) {
    if (smem_size > 0)
        DG_CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    LaunchConfigHandle config;
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    config.numAttrs = 0;
    config.attrs = nullptr;

    // NOTES: must use `static` or the `attr` will be deconstructed
    static LaunchAttrHandle attr;
    if (cluster_dim > 1) {
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {static_cast<unsigned>(cluster_dim), 1, 1};
        config.attrs = &attr;
        config.numAttrs = 1;
    }
    return config;
}

template<typename... ActTypes>
static auto launch_kernel(const KernelHandle& kernel, const LaunchConfigHandle& config, ActTypes&&... args) {
    void *ptr_args[] = { &args... };
    return cudaLaunchKernelExC(&config, kernel, ptr_args);
}

#else

// Use CUDA driver API
using LibraryHandle = CUmodule;
using KernelHandle = CUfunction;
using LaunchConfigHandle = CUlaunchConfig;
using LaunchAttrHandle = CUlaunchAttribute;

#define DG_CUDA_UNIFIED_CHECK DG_CUDA_DRIVER_CHECK

static KernelHandle load_kernel(const std::filesystem::path& cubin_path, const std::string& func_name,
                               LibraryHandle *library_opt = nullptr) {
    LibraryHandle library;
    KernelHandle kernel;
    DG_CUDA_DRIVER_CHECK(lazy_cuModuleLoad(&library, cubin_path.c_str()));
    DG_CUDA_DRIVER_CHECK(lazy_cuModuleGetFunction(&kernel, library, func_name.c_str()));

    if (library_opt != nullptr)
        *library_opt = library;
    return kernel;
}

static void unload_library(const LibraryHandle& library) {
    const auto& error = lazy_cuModuleUnload(library);
    DG_HOST_ASSERT(error == CUDA_SUCCESS or error == CUDA_ERROR_DEINITIALIZED);
}

static LaunchConfigHandle construct_launch_config(const KernelHandle& kernel,
                                                 const cudaStream_t& stream, const int& smem_size,
                                                 const dim3& grid_dim, const dim3& block_dim, const int& cluster_dim) {
    if (smem_size > 0)
        DG_CUDA_DRIVER_CHECK(lazy_cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size));

    LaunchConfigHandle config;
    config.gridDimX = grid_dim.x;
    config.gridDimY = grid_dim.y;
    config.gridDimZ = grid_dim.z;
    config.blockDimX = block_dim.x;
    config.blockDimY = block_dim.y;
    config.blockDimZ = block_dim.z;
    config.sharedMemBytes = smem_size;
    config.hStream = stream;
    config.numAttrs = 0;
    config.attrs = nullptr;

    // NOTES: must use `static` or the `attr` will be deconstructed
    static LaunchAttrHandle attr;
    if (cluster_dim > 1) {
        attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        attr.value.clusterDim.x = cluster_dim;
        attr.value.clusterDim.y = 1;
        attr.value.clusterDim.z = 1;
        config.attrs = &attr;
        config.numAttrs = 1;
    }
    return config;
}

template<typename... ActTypes>
static auto launch_kernel(const KernelHandle& kernel, const LaunchConfigHandle& config, ActTypes&&... args) {
    void *ptr_args[] = { &args... };
    return lazy_cuLaunchKernelEx(&config, kernel, ptr_args, nullptr);
}
#endif

} // namespace deep_gemm
