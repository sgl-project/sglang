// clang-format off
// Adapted from https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan.cpp
// and https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_fwd_fp32.cu
// Copyright (c) 2023, Tri Dao.
// Modified by SGLang Team: forward-only, initial_state support, adapted for sgl-kernel registration.

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>
#include <vector>

#include "selective_scan.h"
#include "selective_scan_fwd_kernel.cuh"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float)  {                                   \
        using input_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }

#define DISPATCH_WTYPE_FLOAT_AND_COMPLEX(WTYPE, NAME, ...)                           \
    if (WTYPE == at::ScalarType::Float) {                                            \
       using weight_t = float;                                                       \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == at::ScalarType::ComplexFloat) {                              \
        using weight_t = c10::complex<float>;                                        \
        __VA_ARGS__();                                                               \
    } else {                                                                         \
        AT_ERROR(#NAME, " not implemented for weight type '", toString(WTYPE), "'"); \
    }

void set_ssm_params_fwd(SSMParamsBase &params,
                        // sizes
                        const size_t batch,
                        const size_t dim,
                        const size_t seqlen,
                        const size_t dstate,
                        const size_t n_groups,
                        const size_t n_chunks,
                        const bool is_variable_B,
                        const bool is_variable_C,
                        // device pointers
                        const at::Tensor u,
                        const at::Tensor delta,
                        const at::Tensor A,
                        const at::Tensor B,
                        const at::Tensor C,
                        const at::Tensor out,
                        const at::Tensor z,
                        const at::Tensor out_z,
                        void* D_ptr,
                        void* delta_bias_ptr,
                        void* x_ptr,
                        void* initial_state_ptr,
                        bool has_z,
                        bool delta_softplus) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.dstate = dstate;
    params.n_groups = n_groups;
    params.n_chunks = n_chunks;
    params.dim_ngroups_ratio = dim / n_groups;

    params.delta_softplus = delta_softplus;

    params.is_variable_B = is_variable_B;
    params.is_variable_C = is_variable_C;

    // Set the pointers and strides.
    params.u_ptr = u.data_ptr();
    params.delta_ptr = delta.data_ptr();
    params.A_ptr = A.data_ptr();
    params.B_ptr = B.data_ptr();
    params.C_ptr = C.data_ptr();
    params.D_ptr = D_ptr;
    params.delta_bias_ptr = delta_bias_ptr;
    params.out_ptr = out.data_ptr();
    params.x_ptr = x_ptr;
    params.z_ptr = has_z ? z.data_ptr() : nullptr;
    params.out_z_ptr = has_z ? out_z.data_ptr() : nullptr;
    params.initial_state_ptr = initial_state_ptr;

    // All stride are in elements, not bytes.
    params.A_d_stride = A.stride(0);
    params.A_dstate_stride = A.stride(1);
    if (!is_variable_B) {
        params.B_d_stride = B.stride(0);
    } else {
        params.B_batch_stride = B.stride(0);
        params.B_group_stride = B.stride(1);
    }
    params.B_dstate_stride = !is_variable_B ? B.stride(1) : B.stride(2);
    if (!is_variable_C) {
        params.C_d_stride = C.stride(0);
    } else {
        params.C_batch_stride = C.stride(0);
        params.C_group_stride = C.stride(1);
    }
    params.C_dstate_stride = !is_variable_C ? C.stride(1) : C.stride(2);
    params.u_batch_stride = u.stride(0);
    params.u_d_stride = u.stride(1);
    params.delta_batch_stride = delta.stride(0);
    params.delta_d_stride = delta.stride(1);
    if (has_z) {
        params.z_batch_stride = z.stride(0);
        params.z_d_stride = z.stride(1);
        params.out_z_batch_stride = out_z.stride(0);
        params.out_z_d_stride = out_z.stride(1);
    }
    params.out_batch_stride = out.stride(0);
    params.out_d_stride = out.stride(1);
}

std::vector<at::Tensor>
selective_scan_fwd(const at::Tensor &u, const at::Tensor &delta,
                  const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
                  const std::optional<at::Tensor> &D_,
                  const std::optional<at::Tensor> &z_,
                  const std::optional<at::Tensor> &delta_bias_,
                  const std::optional<at::Tensor> &initial_state_,
                  bool delta_softplus,
                  bool return_last_state) {
    auto input_type = u.scalar_type();
    auto weight_type = A.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == at::ScalarType::Float || weight_type == at::ScalarType::ComplexFloat);

    const bool is_variable_B = B.dim() >= 3;
    const bool is_variable_C = C.dim() >= 3;
    const bool is_complex = weight_type == at::ScalarType::ComplexFloat;

    TORCH_CHECK(delta.scalar_type() == input_type);
    TORCH_CHECK(B.scalar_type() == (!is_variable_B ? weight_type : input_type));
    TORCH_CHECK(C.scalar_type() == (!is_variable_C ? weight_type : input_type));

    TORCH_CHECK(u.is_cuda());
    TORCH_CHECK(delta.is_cuda());
    TORCH_CHECK(A.is_cuda());
    TORCH_CHECK(B.is_cuda());
    TORCH_CHECK(C.is_cuda());

    TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
    TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);

    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int dstate = A.size(1);
    const int n_groups = is_variable_B ? B.size(1) : 1;

    TORCH_CHECK(dstate <= 256, "selective_scan only supports state dimension <= 256");

    CHECK_SHAPE(u, batch_size, dim, seqlen);
    CHECK_SHAPE(delta, batch_size, dim, seqlen);
    CHECK_SHAPE(A, dim, dstate);
    if (!is_variable_B) {
        CHECK_SHAPE(B, dim, dstate);
    } else {
        CHECK_SHAPE(B, batch_size, n_groups, dstate, !is_complex ? seqlen : seqlen * 2);
        TORCH_CHECK(B.stride(-1) == 1 || B.size(-1) == 1);
    }
    if (!is_variable_C) {
        CHECK_SHAPE(C, dim, dstate);
    } else {
        CHECK_SHAPE(C, batch_size, n_groups, dstate, !is_complex ? seqlen: seqlen * 2);
        TORCH_CHECK(C.stride(-1) == 1 || C.size(-1) == 1);
    }

    if (D_.has_value()) {
        auto D = D_.value();
        TORCH_CHECK(D.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(D.is_cuda());
        TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
        CHECK_SHAPE(D, dim);
    }

    if (delta_bias_.has_value()) {
        auto delta_bias = delta_bias_.value();
        TORCH_CHECK(delta_bias.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(delta_bias.is_cuda());
        TORCH_CHECK(delta_bias.stride(-1) == 1 || delta_bias.size(-1) == 1);
        CHECK_SHAPE(delta_bias, dim);
    }

    if (initial_state_.has_value()) {
        auto initial_state = initial_state_.value();
        TORCH_CHECK(initial_state.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(initial_state.is_cuda());
        TORCH_CHECK(initial_state.is_contiguous());
        CHECK_SHAPE(initial_state, batch_size, dim, dstate);
    }

    at::Tensor z, out_z;
    const bool has_z = z_.has_value();
    if (has_z) {
        z = z_.value();
        TORCH_CHECK(z.scalar_type() == input_type);
        TORCH_CHECK(z.is_cuda());
        TORCH_CHECK(z.stride(-1) == 1 || z.size(-1) == 1);
        CHECK_SHAPE(z, batch_size, dim, seqlen);
        out_z = torch::empty_like(z);
    }

    const int n_chunks = (seqlen + 2048 - 1) / 2048;
    at::Tensor out = torch::empty_like(delta);
    at::Tensor x;
    x = torch::empty({batch_size, dim, n_chunks, dstate * 2}, u.options().dtype(weight_type));

    SSMParamsBase params;
    set_ssm_params_fwd(params, batch_size, dim, seqlen, dstate, n_groups, n_chunks, is_variable_B, is_variable_C,
                       u, delta, A, B, C, out, z, out_z,
                       D_.has_value() ? D_.value().data_ptr() : nullptr,
                       delta_bias_.has_value() ? delta_bias_.value().data_ptr() : nullptr,
                       x.data_ptr(),
                       initial_state_.has_value() ? initial_state_.value().data_ptr() : nullptr,
                       has_z,
                       delta_softplus);

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{u.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(u.scalar_type(), "selective_scan_fwd", [&] {
        DISPATCH_WTYPE_FLOAT_AND_COMPLEX(A.scalar_type(), "selective_scan_fwd", [&] {
            selective_scan_fwd_cuda<input_t, weight_t>(params, stream);
        });
    });

    if (has_z) {
        // When z is provided, the gated output is in out_z
        // Build result: [out_z, last_state]
        at::Tensor last_state;
        if (return_last_state) {
            // Extract last state from x: shape (batch, dim, n_chunks, dstate*2)
            // The last chunk's running_prefix is at x[:, :, n_chunks-1, :]
            // For real (non-complex): scan_t is float2 where .y is the state value
            // x is stored as scan_t, so dstate*2 floats per chunk, where [2*i+1] is state[i]
            last_state = torch::empty({batch_size, dim, dstate}, u.options().dtype(at::ScalarType::Float));
            // Extract the .y component (state) from the last chunk
            // x layout: (batch, dim, n_chunks, dstate * 2) as weight_type
            // For float weight_type, scan_t = float2, stored as pairs (a, b) for each dstate
            auto x_last = x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                   n_chunks - 1, torch::indexing::Slice()});  // (batch, dim, dstate*2)
            // Take every other element starting from index 1 (the .y / state component)
            last_state = x_last.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                       torch::indexing::Slice(1, torch::indexing::None, 2)}).contiguous();
        }
        std::vector<at::Tensor> result = {out_z};
        if (return_last_state) {
            result.push_back(last_state);
        }
        return result;
    } else {
        at::Tensor last_state;
        if (return_last_state) {
            last_state = torch::empty({batch_size, dim, dstate}, u.options().dtype(at::ScalarType::Float));
            auto x_last = x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                   n_chunks - 1, torch::indexing::Slice()});
            last_state = x_last.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                       torch::indexing::Slice(1, torch::indexing::None, 2)}).contiguous();
        }
        std::vector<at::Tensor> result = {out};
        if (return_last_state) {
            result.push_back(last_state);
        }
        return result;
    }
}

// Template instantiations for float32 (from selective_scan_fwd_fp32.cu)
template void selective_scan_fwd_cuda<float, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<float, complex_t>(SSMParamsBase &params, cudaStream_t stream);
// Template instantiations for half and bfloat16
template void selective_scan_fwd_cuda<at::Half, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<at::Half, complex_t>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<at::BFloat16, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<at::BFloat16, complex_t>(SSMParamsBase &params, cudaStream_t stream);
