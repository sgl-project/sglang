// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/utils/tensor.h"
#include <cuda_runtime.h>
#include <istream>
#include <memory>
#include <ostream>
#include <torch/custom_class.h>

namespace turbomind {

enum class WeightType : int
{
    kFP32,
    kFP16,
    kFP8,  // not supported yet
    kBF16,
    kINT8,
    kINT4
};

class Linear : public torch::CustomClassHolder {
public:
    Linear(int64_t input_dims, int64_t output_dims, int64_t w_bit, int64_t group_size);
    void post_init(std::shared_ptr<turbomind::Tensor> qweight, const turbomind::Tensor& scales, const turbomind::Tensor& qzeros, bool simt);
    void forward(const turbomind::Tensor& in, turbomind::Tensor& out, cudaStream_t stream = nullptr);
    ~Linear() {}

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
};  // namespace turbomind
