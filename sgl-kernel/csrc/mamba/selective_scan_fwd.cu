// clang-format off
// Adapted from https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_fwd_kernel.cuh
// Copyright (c) 2023, Tri Dao.
// Modified by SGLang Team: forward-only, initial_state support, adapted for sgl-kernel registration.

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK
#include <vector>

#ifndef USE_ROCM
    #include <cub/block/block_load.cuh>
    #include <cub/block/block_store.cuh>
    #include <cub/block/block_scan.cuh>
#else
    #include <hipcub/hipcub.hpp>
    namespace cub = hipcub;
#endif

#include "selective_scan.h"
#include "static_switch.h"

template<int kNThreads_, int kNItems_, int kNRows_, bool kIsEvenLen_,
         bool kHasZ_, typename input_t_, typename weight_t_>
struct Selective_Scan_fwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads improves occupancy.
    static constexpr int kMinBlocks = kNThreads < 128 ? 5 : 3;
    static constexpr int kNItems = kNItems_;
    static constexpr int kNRows = kNRows_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : constexpr_min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kIsVariableB = true;
    static constexpr bool kIsVariableC = true;
    static constexpr bool kHasZ = kHasZ_;

    static constexpr bool kDirectIO = kIsEvenLen && kNLoads == 1;

    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = float2;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE  : cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE : cub::BLOCK_STORE_DIRECT>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    static constexpr int kSmemIOSize = custom_max({sizeof(typename BlockLoadT::TempStorage),
                                                 sizeof(typename BlockLoadVecT::TempStorage),
                                                 2 * sizeof(typename BlockLoadWeightT::TempStorage),
                                                 2 * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                 sizeof(typename BlockStoreT::TempStorage),
                                                 sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_fwd_kernel(SSMParamsBase params) {
    constexpr bool kHasZ = Ktraits::kHasZ;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    constexpr int kNRows = Ktraits::kNRows;
    constexpr bool kDirectIO = Ktraits::kDirectIO;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using scan_t = typename Ktraits::scan_t;

    // Shared memory.
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    scan_t *smem_running_prefix = reinterpret_cast<scan_t *>(smem_ + Ktraits::kSmemSize);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride
        + dim_id * kNRows * params.u_d_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + dim_id * kNRows * params.delta_d_stride;
    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * kNRows * params.A_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    scan_t *x = reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id * kNRows) * params.n_chunks * params.dstate;

    float D_val[kNRows] = {0};
    if (params.D_ptr != nullptr) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            D_val[r] = reinterpret_cast<float *>(params.D_ptr)[dim_id * kNRows + r];
        }
    }
    float delta_bias[kNRows] = {0};
    if (params.delta_bias_ptr != nullptr) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            delta_bias[r] = reinterpret_cast<float *>(params.delta_bias_ptr)[dim_id * kNRows + r];
        }
    }

    constexpr int kChunkSize = kNThreads * kNItems;
    for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
        input_t u_vals[kNRows][kNItems], delta_vals_load[kNRows][kNItems];
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            load_input<Ktraits>(u + r * params.u_d_stride, u_vals[r], smem_load, params.seqlen - chunk * kChunkSize);
            if constexpr (!kDirectIO) { __syncthreads(); }
            load_input<Ktraits>(delta + r * params.delta_d_stride, delta_vals_load[r], smem_load, params.seqlen - chunk * kChunkSize);
        }
        u += kChunkSize;
        delta += kChunkSize;

        float delta_vals[kNRows][kNItems], delta_u_vals[kNRows][kNItems], out_vals[kNRows][kNItems];
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float u_val = float(u_vals[r][i]);
                delta_vals[r][i] = float(delta_vals_load[r][i]) + delta_bias[r];
                if (params.delta_softplus) {
                    delta_vals[r][i] = delta_vals[r][i] <= 20.f ? log1pf(expf(delta_vals[r][i])) : delta_vals[r][i];
                }
                delta_u_vals[r][i] = delta_vals[r][i] * u_val;
                out_vals[r][i] = D_val[r] * u_val;
            }
        }

        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            weight_t A_val[kNRows];
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                A_val[r] = A[state_idx * params.A_dstate_stride + r * params.A_d_stride];
                // Multiply A with LOG2E so we can use exp2f instead of expf.
                constexpr float kLog2e = M_LOG2E;
                A_val[r] *= kLog2e;
            }
            weight_t B_vals[kNItems], C_vals[kNItems];
            load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                smem_load_weight, params.seqlen - chunk * kChunkSize);
            load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                smem_load_weight1, params.seqlen - chunk * kChunkSize);

            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                if (r > 0) { __syncthreads(); }  // Scan could be using the same smem
                scan_t thread_data[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    thread_data[i] = make_float2(exp2f(delta_vals[r][i] * A_val[r]),
                                                 B_vals[i] * delta_u_vals[r][i]);
                    if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct
                        if (threadIdx.x * kNItems + i >= params.seqlen - chunk * kChunkSize) {
                            thread_data[i] = make_float2(1.f, 0.f);
                        }
                    }
                }
                // Initialize running total
                scan_t running_prefix;
                if (chunk > 0 && threadIdx.x % 32 == 0) {
                    running_prefix = smem_running_prefix[state_idx + r * MAX_DSTATE];
                } else if (chunk == 0 && threadIdx.x % 32 == 0 && params.initial_state_ptr != nullptr) {
                    float initial_h = reinterpret_cast<float *>(params.initial_state_ptr)
                        [batch_id * params.dim * params.dstate + (dim_id * kNRows + r) * params.dstate + state_idx];
                    running_prefix = make_float2(1.f, initial_h);
                } else {
                    running_prefix = make_float2(1.f, 0.f);
                }
                SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
                typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
                );
                // There's a syncthreads in the scan op, so we don't need to sync here.
                // Unless there's only 1 warp, but then it's the same thread (0) reading and writing.
                if (threadIdx.x == 0) {
                    smem_running_prefix[state_idx] = prefix_op.running_prefix;
                    x[(r * params.n_chunks + chunk) * params.dstate + state_idx] = prefix_op.running_prefix;
                }
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    const weight_t C_val = C_vals[i];
                    out_vals[r][i] += thread_data[i].y * C_val;
                }
            }
        }

        input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
            + dim_id * kNRows * params.out_d_stride + chunk * kChunkSize;
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            store_output<Ktraits>(out + r * params.out_d_stride, out_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
        }

        if constexpr (kHasZ) {
            input_t *z = reinterpret_cast<input_t *>(params.z_ptr) + batch_id * params.z_batch_stride
                + dim_id * kNRows * params.z_d_stride + chunk * kChunkSize;
            input_t *out_z = reinterpret_cast<input_t *>(params.out_z_ptr) + batch_id * params.out_z_batch_stride
                + dim_id * kNRows * params.out_z_d_stride + chunk * kChunkSize;
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                input_t z_vals[kNItems];
                __syncthreads();
                load_input<Ktraits>(z + r * params.z_d_stride, z_vals, smem_load, params.seqlen - chunk * kChunkSize);
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    float z_val = z_vals[i];
                    out_vals[r][i] *= z_val / (1 + expf(-z_val));
                }
                __syncthreads();
                store_output<Ktraits>(out_z + r * params.out_z_d_stride, out_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
            }
        }

        Bvar += kChunkSize;
        Cvar += kChunkSize;
    }
}

template<int kNThreads, int kNItems, typename input_t, typename weight_t>
void selective_scan_fwd_launch(SSMParamsBase &params, cudaStream_t stream) {
    // Only kNRows == 1 is tested for now, which ofc doesn't differ from previously when we had each block
    // processing 1 row.
    constexpr int kNRows = 1;
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        BOOL_SWITCH(params.z_ptr != nullptr , kHasZ, [&] {
            using Ktraits = Selective_Scan_fwd_kernel_traits<kNThreads, kNItems, kNRows, kIsEvenLen, kHasZ, input_t, weight_t>;

            constexpr int kSmemSize = Ktraits::kSmemSize + kNRows * MAX_DSTATE * sizeof(typename Ktraits::scan_t);
            dim3 grid(params.batch, params.dim / kNRows);

            auto kernel = &selective_scan_fwd_kernel<Ktraits>;

            if (kSmemSize >= 48 * 1024) {
                #ifndef USE_ROCM
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                #else
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    (void *) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                    std::cerr << "Warning (selective_scan_fwd_kernel): attempting to set maxDynamicSharedMemorySize on an AMD GPU which is currently a non-op (in ROCm versions <= 6.1). This might lead to undefined behavior. \n" << std::endl;
                #endif
            }

            kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

template<typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream) {

    #ifndef USE_ROCM
        if (params.seqlen <= 128) {
            selective_scan_fwd_launch<32, 4, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 256) {
            selective_scan_fwd_launch<32, 8, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 512) {
            selective_scan_fwd_launch<32, 16, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            selective_scan_fwd_launch<64, 16, input_t, weight_t>(params, stream);
        } else {
            selective_scan_fwd_launch<128, 16, input_t, weight_t>(params, stream);
        }
    #else
        if (params.seqlen <= 256) {
            selective_scan_fwd_launch<64, 4, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 512) {
            selective_scan_fwd_launch<64, 8, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            selective_scan_fwd_launch<64, 16, input_t, weight_t>(params, stream);
        } else {
            selective_scan_fwd_launch<128, 16, input_t, weight_t>(params, stream);
        }
    #endif
}

// Template instantiations
template void selective_scan_fwd_cuda<float, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<at::Half, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<at::BFloat16, float>(SSMParamsBase &params, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Host-side entry point
////////////////////////////////////////////////////////////////////////////////////////////////////

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

void set_ssm_params_fwd(SSMParamsBase &params,
                        // sizes
                        const size_t batch,
                        const size_t dim,
                        const size_t seqlen,
                        const size_t dstate,
                        const size_t n_groups,
                        const size_t n_chunks,
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

    params.is_variable_B = true;
    params.is_variable_C = true;

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
    params.B_batch_stride = B.stride(0);
    params.B_group_stride = B.stride(1);
    params.B_dstate_stride = B.stride(2);
    params.C_batch_stride = C.stride(0);
    params.C_group_stride = C.stride(1);
    params.C_dstate_stride = C.stride(2);
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
    TORCH_CHECK(weight_type == at::ScalarType::Float);

    TORCH_CHECK(B.dim() >= 3, "B must be variable (dim >= 3)");
    TORCH_CHECK(C.dim() >= 3, "C must be variable (dim >= 3)");

    TORCH_CHECK(delta.scalar_type() == input_type);
    TORCH_CHECK(B.scalar_type() == input_type);
    TORCH_CHECK(C.scalar_type() == input_type);

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
    const int n_groups = B.size(1);

    TORCH_CHECK(dstate <= 256, "selective_scan only supports state dimension <= 256");

    CHECK_SHAPE(u, batch_size, dim, seqlen);
    CHECK_SHAPE(delta, batch_size, dim, seqlen);
    CHECK_SHAPE(A, dim, dstate);
    CHECK_SHAPE(B, batch_size, n_groups, dstate, seqlen);
    TORCH_CHECK(B.stride(-1) == 1 || B.size(-1) == 1);
    CHECK_SHAPE(C, batch_size, n_groups, dstate, seqlen);
    TORCH_CHECK(C.stride(-1) == 1 || C.size(-1) == 1);

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
    set_ssm_params_fwd(params, batch_size, dim, seqlen, dstate, n_groups, n_chunks,
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
        using weight_t = float;
        selective_scan_fwd_cuda<input_t, weight_t>(params, stream);
    });

    if (has_z) {
        // When z is provided, the gated output is in out_z
        // Build result: [out_z, last_state]
        at::Tensor last_state;
        if (return_last_state) {
            // Extract last state from x: shape (batch, dim, n_chunks, dstate*2)
            // The last chunk's running_prefix is at x[:, :, n_chunks-1, :]
            // scan_t is float2 where .y is the state value
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
