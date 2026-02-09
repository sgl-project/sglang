/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "fast_hadamard_transform.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float) {                                    \
        using input_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }

template<typename input_t>
void fast_hadamard_transform_cuda(HadamardParamsBase &params, cudaStream_t stream);

template<typename input_t>
void fast_hadamard_transform_12N_cuda(HadamardParamsBase &params, cudaStream_t stream);

template<typename input_t>
void fast_hadamard_transform_20N_cuda(HadamardParamsBase &params, cudaStream_t stream);

template<typename input_t>
void fast_hadamard_transform_28N_cuda(HadamardParamsBase &params, cudaStream_t stream);

template<typename input_t>
void fast_hadamard_transform_40N_cuda(HadamardParamsBase &params, cudaStream_t stream);

void set_hadamard_params(HadamardParamsBase &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t multiple,
                         // device pointers
                         const at::Tensor x,
                         const at::Tensor out,
                         float scale
                         ) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.log_N = int(ceil(std::log2(dim / multiple)));

    // Set the pointers and strides.
    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    // All stride are in elements, not bytes.
    params.x_batch_stride = x.stride(0);
    params.out_batch_stride = out.stride(0);

    params.scale = scale;
}


at::Tensor
fast_hadamard_transform(at::Tensor &x, float scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda());

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    if (dim_og % 8 != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, 8 - dim_og % 8}));
    }
    const int dim = x.size(1);

    TORCH_CHECK(dim % 8 == 0, "fast_hadamard_transform only supports hidden dimension divisible by 8 for now");
    TORCH_CHECK(dim <= 32768, "fast_hadamard_transform only supports hidden dimension at most 32768 for now");

    at::Tensor out = torch::empty_like(x);

    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 1, x, out, scale);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_hadamard_transform", [&] {
        fast_hadamard_transform_cuda<input_t>(params, stream);
    });
    if (dim_og % 8 != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}

at::Tensor
fast_hadamard_transform_12N(at::Tensor &x, float scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda());

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    if (dim_og % (4 * 12) != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, (4 * 12) - dim_og % (4 * 12)}));
    }
    const int dim = x.size(1);

    TORCH_CHECK(dim % (4 * 12) == 0, "fast_hadamard_transform_12N only supports hidden dimension divisible by 48 for now");
    TORCH_CHECK(dim <= 12 * 1024, "fast_hadamard_transform_12N only supports hidden dimension at most 12288 for now");

    at::Tensor out = torch::empty_like(x);

    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 12, x, out, scale);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_hadamard_transform", [&] {
        fast_hadamard_transform_12N_cuda<input_t>(params, stream);
    });
    if (dim_og % (4 * 12) != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}

at::Tensor
fast_hadamard_transform_20N(at::Tensor &x, float scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda());

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    if (dim_og % (4 * 20) != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, (4 * 20) - dim_og % (4 * 20)}));
    }
    const int dim = x.size(1);

    TORCH_CHECK(dim % (4 * 20) == 0, "fast_hadamard_transform_20N only supports hidden dimension divisible by 80 for now");
    TORCH_CHECK(dim <= 20 * 1024, "fast_hadamard_transform_20N only supports hidden dimension at most 20480 for now");

    at::Tensor out = torch::empty_like(x);

    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 20, x, out, scale);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_hadamard_transform", [&] {
        fast_hadamard_transform_20N_cuda<input_t>(params, stream);
    });
    if (dim_og % (4 * 20) != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}

at::Tensor
fast_hadamard_transform_28N(at::Tensor &x, float scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda());

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    if (dim_og % (4 * 28) != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, (4 * 28) - dim_og % (4 * 28)}));
    }
    const int dim = x.size(1);

    TORCH_CHECK(dim % (4 * 28) == 0, "fast_hadamard_transform_28N only supports hidden dimension divisible by 112 for now");
    TORCH_CHECK(dim <= 28 * 1024, "fast_hadamard_transform_28N only supports hidden dimension at most 28672 for now");

    at::Tensor out = torch::empty_like(x);

    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 28, x, out, scale);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_hadamard_transform", [&] {
        fast_hadamard_transform_28N_cuda<input_t>(params, stream);
    });
    if (dim_og % (8 * 28) != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}

at::Tensor
fast_hadamard_transform_40N(at::Tensor &x, float scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda());

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    if (dim_og % (4 * 40) != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, (4 * 40) - dim_og % (4 * 40)}));
    }
    const int dim = x.size(1);

    TORCH_CHECK(dim % (4 * 40) == 0, "fast_hadamard_transform_40N only supports hidden dimension divisible by 160 for now");
    TORCH_CHECK(dim <= 40 * 1024, "fast_hadamard_transform_40N only supports hidden dimension at most 40960 for now");

    at::Tensor out = torch::empty_like(x);

    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 40, x, out, scale);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_hadamard_transform", [&] {
        fast_hadamard_transform_40N_cuda<input_t>(params, stream);
    });
    if (dim_og % (8 * 40) != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_hadamard_transform", &fast_hadamard_transform, "Fast Hadamard transform");
    m.def("fast_hadamard_transform_12N", &fast_hadamard_transform_12N, "Fast Hadamard transform with dimension = 12 * power of 2");
    m.def("fast_hadamard_transform_20N", &fast_hadamard_transform_20N, "Fast Hadamard transform with dimension = 20 * power of 2");
    m.def("fast_hadamard_transform_28N", &fast_hadamard_transform_28N, "Fast Hadamard transform with dimension = 28 * power of 2");
    m.def("fast_hadamard_transform_40N", &fast_hadamard_transform_40N, "Fast Hadamard transform with dimension = 40 * power of 2");
}
