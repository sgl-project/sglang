#pragma once

#include <cublasLt.h>
#include <exception>
#include <string>
#include <sstream>

#include "compatibility.hpp"

namespace deep_gemm {

class DGException final : public std::exception {
    std::string message = {};

public:
    explicit DGException(const char *name, const char* file, const int line, const std::string& error) {
        message = std::string(name) + " error (" + file + ":" + std::to_string(line) + "): " + error;
    }

    const char *what() const noexcept override {
        return message.c_str();
    }
};

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

#ifndef DG_HOST_ASSERT
#define DG_HOST_ASSERT(cond) \
do { \
    if (not (cond)) { \
        throw DGException("Assertion", __FILE__, __LINE__, #cond); \
    } \
} while (0)
#endif

#ifndef DG_HOST_UNREACHABLE
#define DG_HOST_UNREACHABLE(reason) (throw DGException("Assertion", __FILE__, __LINE__, reason))
#endif

#ifndef DG_NVRTC_CHECK
#define DG_NVRTC_CHECK(cmd) \
do { \
    const auto& e = (cmd); \
    if (e != NVRTC_SUCCESS) { \
        throw DGException("NVRTC", __FILE__, __LINE__, nvrtcGetErrorString(e)); \
    } \
} while (0)
#endif

#ifndef DG_CUDA_DRIVER_CHECK
#define DG_CUDA_DRIVER_CHECK(cmd) \
do { \
    const auto& e = (cmd); \
    if (e != CUDA_SUCCESS) { \
        std::stringstream ss; \
        const char *name, *info; \
        lazy_cuGetErrorName(e, &name), lazy_cuGetErrorString(e, &info); \
        ss << static_cast<int>(e) << " (" << name << ", " << info << ")"; \
        throw DGException("CUDA driver", __FILE__, __LINE__, ss.str()); \
    } \
} while (0)
#endif

#ifndef DG_CUDA_RUNTIME_CHECK
#define DG_CUDA_RUNTIME_CHECK(cmd) \
do { \
    const auto& e = (cmd); \
    if (e != cudaSuccess) { \
        std::stringstream ss; \
        ss << static_cast<int>(e) << " (" << cudaGetErrorName(e) << ", " << cudaGetErrorString(e) << ")"; \
        throw DGException("CUDA runtime", __FILE__, __LINE__, ss.str()); \
    } \
} while (0)
#endif

#ifndef DG_CUBLASLT_CHECK

#if !DG_CUBLAS_GET_ERROR_STRING_COMPATIBLE
inline const char* cublasGetStatusString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "Unknown cuBLAS error";
    }
}
#endif

#define DG_CUBLASLT_CHECK(cmd) \
do { \
    const auto& e = (cmd); \
    if (e != CUBLAS_STATUS_SUCCESS) { \
        std::ostringstream ss; \
        ss << static_cast<int>(e) << " (" << cublasGetStatusString(e) << ")"; \
        throw DGException("cuBLASLt", __FILE__, __LINE__, ss.str()); \
    } \
} while (0)
#endif

} // namespace deep_gemm
