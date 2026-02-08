#pragma once

#include <cublasLt.h>
#include <torch/version.h>
#include <ATen/cuda/CUDAContext.h>

#include "../utils/exception.hpp"
#include "../utils/lazy_init.hpp"

namespace deep_gemm {

class DeviceRuntime {
    int num_sms = 0, tc_util = 0;
    std::shared_ptr<cudaDeviceProp> cached_prop;

    // cuBLASLt utils
    static constexpr size_t kCublasLtWorkspaceSize = 32 * 1024 * 1024;

public:
    // Create the cuBLASLt handle ourselves
    cublasLtHandle_t cublaslt_handle{};
    std::shared_ptr<torch::Tensor> cublaslt_workspace;

    explicit DeviceRuntime() {
        cublaslt_workspace = std::make_shared<torch::Tensor>(torch::empty({kCublasLtWorkspaceSize}, dtype(torch::kByte).device(at::kCUDA)));
        DG_CUBLASLT_CHECK(cublasLtCreate(&cublaslt_handle));
    }

    ~DeviceRuntime() noexcept(false) {
        DG_CUBLASLT_CHECK(cublasLtDestroy(cublaslt_handle));
    }

    cublasLtHandle_t get_cublaslt_handle() const {
        return cublaslt_handle;
    }

    torch::Tensor get_cublaslt_workspace() const {
        return *cublaslt_workspace;
    }

    std::shared_ptr<cudaDeviceProp> get_prop() {
        if (cached_prop == nullptr) {
            int device_idx;
            cudaDeviceProp prop;
            DG_CUDA_RUNTIME_CHECK(cudaGetDevice(&device_idx));
            DG_CUDA_RUNTIME_CHECK(cudaGetDeviceProperties(&prop, device_idx));
            cached_prop = std::make_shared<cudaDeviceProp>(prop);
        }
        return cached_prop;
    }

    std::pair<int, int> get_arch_pair() {
        const auto prop = get_prop();
        return {prop->major, prop->minor};
    }

    std::string get_arch(const bool& number_only = false,
                         const bool& support_arch_family = false) {
        const auto& [major, minor] = get_arch_pair();
        if (major == 10 and minor != 1) {
            if (number_only)
                return "100";
            return support_arch_family ? "100f" : "100a";
        }
        return std::to_string(major * 10 + minor) + (number_only ? "" : "a");
    }

    int get_arch_major() {
        return get_arch_pair().first;
    }

    void set_num_sms(const int& new_num_sms) {
        DG_HOST_ASSERT(0 <= new_num_sms and new_num_sms <= get_prop()->multiProcessorCount);
        num_sms = new_num_sms;
    }

    int get_num_sms() {
        if (num_sms == 0)
            num_sms = get_prop()->multiProcessorCount;
        return num_sms;
    }

    int get_l2_cache_size() {
        return get_prop()->l2CacheSize;
    }

    void set_tc_util(const int& new_tc_util) {
        DG_HOST_ASSERT(0 <= new_tc_util and new_tc_util <= 100);
        tc_util = new_tc_util;
    }

    int get_tc_util() const {
        return tc_util == 0 ? 100 : tc_util;
    }
};

static auto device_runtime = LazyInit<DeviceRuntime>([](){ return std::make_shared<DeviceRuntime>(); });

} // namespace deep_gemm
