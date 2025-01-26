// Copyright (c) OpenMMLab. All rights reserved.

#include "linear.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/macro.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

namespace turbomind {

class GemmPool {
public:
    static GemmPool& getInstance()
    {
        static GemmPool singleton;
        return singleton;
    }

    gemm::Gemm* get(int device_id)
    {
        TM_CHECK(device_id < pool_.size());
        return &pool_[device_id];
    }
    ~GemmPool() = default;

private:
    GemmPool()
    {
        int device_count = 0;
        check_cuda_error(cudaGetDeviceCount(&device_count));
        pool_.resize(device_count);
    }
    std::vector<gemm::Gemm> pool_;
};

struct Linear::Impl {

    Impl(int64_t input_dims, int64_t output_dims, int64_t w_bit, int64_t group_size):
        input_dims_(input_dims), output_dims_(output_dims), w_bit_(w_bit), group_size_(group_size)
    {
        workspace_ = {};

        workspace_.barriers_size = gemm::Gemm::kBarriersSize;
        workspace_.partials_size = gemm::Gemm::kPartialsSize;
        cudaMallocAsync(&workspace_.barriers, workspace_.barriers_size, 0);
        cudaMallocAsync(&workspace_.partials, workspace_.partials_size, 0);
        cudaMemsetAsync(workspace_.barriers, 0, workspace_.barriers_size, 0);
    }

    ~Impl()
    {
        cudaFreeAsync(workspace_.barriers, 0);
        cudaFreeAsync(workspace_.partials, 0);
        workspace_ = {};
        check_cuda_error(cudaFree(scales_zeros_));
    }

    void post_init(std::shared_ptr<Tensor> qweight, const Tensor& scales, const Tensor& qzeros, bool simt)
    {
        const auto workspace_size = input_dims_ * output_dims_ * sizeof(uint16_t);
        void*      workspace{};
        check_cuda_error(cudaMalloc((void**)&workspace, workspace_size));

        convert_qweight(workspace, qweight, input_dims_, output_dims_, simt);
        convert_scales_zeros(workspace, scales, qzeros, input_dims_, output_dims_, group_size_, simt);

        check_cuda_error(cudaFree(workspace));
    }

    void forward(const Tensor& in, Tensor& out, cudaStream_t stream)
    {
        TM_CHECK(in.type == TYPE_FP16 && out.type == TYPE_FP16);
        TM_CHECK(in.shape.size() == 2 && in.shape[1] == input_dims_);
        TM_CHECK(out.shape.size() == 2 && out.shape[0] == in.shape[0] && out.shape[1] == output_dims_);

        using namespace gemm;

        const Operation operation{
            dispatch_policy_, Epilogue::kNone, {QuantType::kNone}, {QuantType::kDefault, group_size_}, 0, nullptr};

        const MatrixLayout a_desc{
            gemm::DataType::F16,  // get_data_type_v<T>,
            kRowMajor,
            (int)in.shape[0],  // row
            (int)input_dims_,  // col
            (int)input_dims_   // input_data.pitch, // input_data.pitch = input_dims_ if input_data.pitch==0
        };

        const MatrixLayout c_desc{gemm::DataType::F16,  // get_data_type_v<T>,
                                  kRowMajor,
                                  (int)in.shape[0],   // row
                                  (int)output_dims_,  // col
                                  (int)output_dims_};
        int                device_id;
        check_cuda_error(cudaGetDevice(&device_id));
        auto gemm = GemmPool::getInstance().get(device_id);
        auto ec   = gemm->Run(operation,
                            1.f,
                            in.data,
                            a_desc,
                            nullptr,
                            {},
                            weight_->data,
                            k_desc_,
                            scales_zeros_,
                            q_desc_,
                            0.0f,
                            out.data,
                            c_desc,
                            const_cast<void*>(out.data),
                            c_desc,
                            workspace_,
                            stream);

        if (ec) {
            printf("%s: %d", __PRETTY_FUNCTION__, ec);
            std::abort();
        }
    }
    void convert_qweight(
        void* workspace, std::shared_ptr<Tensor> weight, int64_t input_dims, int64_t output_dims, bool use_simt)
    {
        using namespace gemm;
        auto [order_b, pack_b, order_v, pack_v] = get_weight_and_scales_layout(getSMVersion(), use_simt);

        if (order_b == kColMajor) {
            transpose_u4((uint4_t*)workspace, (const uint4_t*)weight->data, input_dims, output_dims);
            cudaMemcpy(const_cast<void*>(weight->data), workspace, input_dims * output_dims / 2, cudaMemcpyDefault);
        }

        extend_to_u16((uint16_t*)workspace, (const uint4_t*)weight->data, input_dims * output_dims);
        sync_check_cuda_error();

        if constexpr (0) {
            std::vector<uint16_t> tmp(input_dims * output_dims);
            cudaMemcpy(tmp.data(), workspace, sizeof(uint16_t) * tmp.size(), cudaMemcpyDefault);
            cudaDeviceSynchronize();
            int i = 0;
            for (auto it = tmp.begin(); i < 1000 && it != tmp.end(); ++it, ++i) {
                std::cout << *it << " ";
            }
            i = 0;
            std::cout << "\n";
            for (auto it = tmp.rbegin(); i < 1000 && it != tmp.rend(); ++it, ++i) {
                std::cout << *it << " ";
            }
        }

        MatrixLayout w_desc{
            gemm::DataType::F16,
            order_b,
            (int)input_dims,   // k
            (int)output_dims,  // n
            order_b == kRowMajor ? (int)output_dims : (int)input_dims,
        };

        k_desc_      = w_desc;
        k_desc_.type = gemm::DataType::U4;
        k_desc_.pack = pack_b;

        cudaMemset(const_cast<void*>(weight->data), 0, input_dims * output_dims / 2);

        TM_CHECK(Convert(workspace, w_desc, const_cast<void*>(weight->data), k_desc_, 0) == 0);
        sync_check_cuda_error();

        cudaDeviceSynchronize();

        if constexpr (0) {
            std::vector<uint32_t> tmp(input_dims * output_dims / 8);
            cudaMemcpy(tmp.data(), weight->data, sizeof(uint32_t) * tmp.size(), cudaMemcpyDefault);
            cudaDeviceSynchronize();
            int i = 0;
            for (auto it = tmp.begin(); i < 1000 && it != tmp.end(); ++it, ++i) {
                std::cout << std::hex << *it << " ";
            }
            i = 0;
            std::cout << "\n";
            for (auto it = tmp.rbegin(); i < 1000 && it != tmp.rend(); ++it, ++i) {
                std::cout << std::hex << *it << " ";
            }
        }

        weight_ = weight;
    }

    void convert_scales_zeros(void*         workspace,
                              const Tensor& scales,
                              const Tensor& qzeros,
                              int64_t        input_dims,
                              int64_t        output_dims,
                              int           group_size,
                              bool          use_simt)
    {
        if constexpr (0) {
            std::cout << "scales: " << std::endl;
            std::vector<__half> tmp(input_dims / group_size * output_dims);
            cudaMemcpy(tmp.data(), scales.data, sizeof(__half) * tmp.size(), cudaMemcpyDefault);
            cudaDeviceSynchronize();
            int i = 0;
            for (auto it = tmp.begin(); i < 1000 && it != tmp.end(); ++it, ++i) {
                std::cout << __half2float(*it) << " ";
            }
            std::cout << std::endl;
            i = 0;
            for (auto it = tmp.rbegin(); i < 1000 && it != tmp.rend(); ++it, ++i) {
                std::cout << __half2float(*it) << " ";
            }
            std::cout << std::endl;
        }

        if constexpr (0) {
            std::cout << "zeros: " << std::endl;
            std::vector<__half> tmp(input_dims / group_size * output_dims);
            cudaMemcpy(tmp.data(), qzeros.data, sizeof(__half) * tmp.size(), cudaMemcpyDefault);
            cudaDeviceSynchronize();
            int i = 0;
            for (auto it = tmp.begin(); i < 1000 && it != tmp.end(); ++it, ++i) {
                std::cout << __half2float(*it) << " ";
            }
            std::cout << std::endl;
            i = 0;
            for (auto it = tmp.rbegin(); i < 1000 && it != tmp.rend(); ++it, ++i) {
                std::cout << __half2float(*it) << " ";
            }
            std::cout << std::endl;
        }

        const auto scale_count = input_dims / group_size * output_dims;

        using namespace gemm;
        auto [order_b, pack_b, order_v, pack_v] = get_weight_and_scales_layout(getSMVersion(), use_simt);

        fuse_scales_and_zeros((half*)workspace, (const half*)scales.data, (half*)qzeros.data, scale_count);
        sync_check_cuda_error();

        cudaDeviceSynchronize();

        check_cuda_error(cudaMalloc(&scales_zeros_, sizeof(uint16_t) * scale_count * 2));

        MatrixLayout s_desc{
            gemm::DataType::U32,
            order_v,
            (int)input_dims / group_size,  // k
            (int)output_dims,              // n
            (int)output_dims,
        };

        q_desc_      = s_desc;
        q_desc_.pack = pack_v;

        TM_CHECK(Convert(workspace, s_desc, scales_zeros_, q_desc_, 0) == 0);
        sync_check_cuda_error();

        if constexpr (0) {
            std::vector<__half> tmp(scale_count * 2);
            cudaMemcpy(tmp.data(), workspace, sizeof(__half) * tmp.size(), cudaMemcpyDefault);
            cudaDeviceSynchronize();
            int i = 0;
            for (auto it = tmp.begin(); i < 1000 && it != tmp.end(); ++it, ++i) {
                std::cout << __half2float(*it) << " ";
            }
            std::cout << std::endl;
            i = 0;
            for (auto it = tmp.rbegin(); i < 1000 && it != tmp.rend(); ++it, ++i) {
                std::cout << __half2float(*it) << " ";
            }
            std::cout << std::endl;
        }
    }

private:
    gemm::DispatchPolicy dispatch_policy_{gemm::DispatchPolicy::kDefault};
    gemm::Workspace      workspace_;

    int64_t input_dims_;
    int64_t output_dims_;
    int    w_bit_;
    int    group_size_;

    std::shared_ptr<Tensor> weight_;
    half*                   scales_zeros_;

    gemm::MatrixLayout k_desc_;
    gemm::MatrixLayout q_desc_;
};

Linear::Linear(int64_t input_dims, int64_t output_dims, int64_t w_bit, int64_t group_size)
{
    impl_ = std::make_shared<Impl>(input_dims, output_dims, w_bit, group_size);
}

void Linear::post_init(std::shared_ptr<Tensor> qweight, const Tensor& scales, const Tensor& qzeros, bool simt)
{
    impl_->post_init(qweight, scales, qzeros, simt);
}

void Linear::forward(const Tensor& in, Tensor& out, cudaStream_t stream)
{
    impl_->forward(in, out, stream);
}
}  // namespace turbomind
