// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "streaming_dit_bridge.h"
#include "../kernels/attention.cuh"
#include "../kernels/cosmos_block.cuh"
#include "../kernels/cosmos_fp8_flash.cuh"
#include "../kernels/cosmos_fp8_tc_probe.cuh"
#include "../kernels/linear_utils.cuh"
#include "../kernels/ops.cuh"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/numeric_conversion.h"
#include <torch/nn/functional.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <string>

// streaming_dit_bridge.cu — full Cosmos DiT forward using native CUTLASS GEMMs
// + cuDNN FMHA.
//
// Streaming forward (optimized_dit_forward): the per-block hot path is
// implemented in `omnidreams_singleview::cosmos_run_transformer_block_streaming` (see
// `kernels/cosmos_block.cu`), which fuses ln+modulate, runs three bf16
// CUTLASS GEMMs for Q/K/V, packs+rotate-half-RoPE in one kernel, calls
// cuDNN FMHA, then drives the SA/CA out projections and FFN through bf16
// CUTLASS GEMMs with separate gated-residual kernels. One-shot pieces
// (x_embedder, timestep+sin embedding, adaln-LoRA, t_embedding_norm,
// final layer) stay in ATen because they run once per forward, not per
// layer.
//
// DDPM forward (cosmos_forward): full-sequence training-style forward,
// kept on the original ATen + cuDNN-FMHA path -- only the streaming
// path lands on the alpadreams pipeline and is therefore the perf
// hotspot.
//
// RoPE note: Cosmos uses rotate_half convention (negate second half, swap halves);
// native's built-in QKV+RoPE kernel (used by WAN) bakes adjacent-pair
// rotation in. The streaming orchestrator uses a custom rotate-half pack+RoPE
// kernel; the DDPM `cosmos_forward` keeps RoPE in ATen.

namespace F = torch::nn::functional;

static bool cosmos_profile_enabled() {
    const char* v = std::getenv("OMNIDREAMS_DIT_PROFILE");
    return v && v[0] && v[0] != '0';
}

__global__ void cosmos_add_inplace_kernel(
    cutlass::bfloat16_t* __restrict__ dst,
    const cutlass::bfloat16_t* __restrict__ src,
    int64_t n)
{
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = cutlass::bfloat16_t(float(dst[i]) + float(src[i]));
}

static cudaError_t cosmos_add_inplace(
    cutlass::bfloat16_t* dst,
    const cutlass::bfloat16_t* src,
    int64_t n,
    cudaStream_t stream)
{
    int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    cosmos_add_inplace_kernel<<<blocks, threads, 0, stream>>>(dst, src, n);
    return cudaGetLastError();
}

struct CosmosStreamingScratch {
    int B = 0, M = 0, K = 0, H = 0, D = 0, FF = 0, lora_dim = 0;
    c10::Device device = c10::Device(c10::DeviceType::CPU);
    c10::ScalarType dtype = c10::ScalarType::Undefined;

    torch::Tensor qkv_row;
    torch::Tensor q_row;
    torch::Tensor k_row;
    torch::Tensor v_row;
    torch::Tensor q_bmhk;
    torch::Tensor k_bmhk;
    torch::Tensor o_bmhk;
    torch::Tensor normed;
    torch::Tensor ffn_intermediate;
    torch::Tensor lora_hidden_sa;
    torch::Tensor lora_hidden_ca;
    torch::Tensor lora_hidden_mlp;
    torch::Tensor mods_sa;
    torch::Tensor mods_ca;
    torch::Tensor mods_mlp;
    torch::Tensor fl_hidden;
    torch::Tensor fl_mods;

    bool matches(int b, int m, int k, int h, int d, int ff, int lora,
                 const c10::Device& dev, c10::ScalarType dt) const {
        return qkv_row.defined() && B == b && M == m && K == k && H == h &&
               D == d && FF == ff && lora_dim == lora && device == dev && dtype == dt;
    }

    void ensure(int b, int m, int k, int h, int d, int ff, int lora,
                const c10::Device& dev, c10::ScalarType dt,
                const torch::TensorOptions& opts) {
        if (matches(b, m, k, h, d, ff, lora, dev, dt)) return;
        B = b; M = m; K = k; H = h; D = d; FF = ff; lora_dim = lora;
        device = dev; dtype = dt;
        qkv_row = torch::empty({M, 3 * K}, opts);
        q_row = torch::empty({M, K}, opts);
        k_row = torch::empty({M, K}, opts);
        v_row = torch::empty({M, K}, opts);
        q_bmhk = torch::empty({M, H, D}, opts);
        k_bmhk = torch::empty({M, H, D}, opts);
        o_bmhk = torch::empty({M, H, D}, opts);
        normed = torch::empty({M, K}, opts);
        ffn_intermediate = torch::empty({M, FF}, opts);
        lora_hidden_sa = torch::empty({B, lora_dim}, opts);
        lora_hidden_ca = torch::empty({B, lora_dim}, opts);
        lora_hidden_mlp = torch::empty({B, lora_dim}, opts);
        mods_sa = torch::empty({B, 3 * K}, opts);
        mods_ca = torch::empty({B, 3 * K}, opts);
        mods_mlp = torch::empty({B, 3 * K}, opts);
        fl_hidden = torch::empty({B, lora_dim}, opts);
        fl_mods = torch::empty({B, 2 * K}, opts);
    }
};

thread_local CosmosStreamingScratch g_cosmos_streaming_scratch;

static __global__ void cosmos_softmax_batched_half_to_fp8_kernel(
    const cutlass::half_t* __restrict__ scores,
    cutlass::float_e4m3_t* __restrict__ probs,
    int groups,
    int rows,
    int cols,
    bool causal)
{
    int row = blockIdx.x;
    int group = blockIdx.y;
    if (group >= groups || row >= rows) return;

    int tid = threadIdx.x;
    extern __shared__ float smem[];
    cutlass::NumericConverter<float, cutlass::half_t> to_f32;
    cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;

    const size_t group_offset = static_cast<size_t>(group) * rows * cols;
    const cutlass::half_t* score_row = scores + group_offset + static_cast<size_t>(row) * cols;
    cutlass::float_e4m3_t* prob_row = probs + group_offset + static_cast<size_t>(row) * cols;

    float max_val = -FLT_MAX;
    for (int col = tid; col < cols; col += blockDim.x) {
        bool valid = !causal || col <= row;
        if (valid) {
            float v = to_f32(score_row[col]);
            max_val = fmaxf(max_val, v);
        }
    }
    smem[tid] = max_val;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
        __syncthreads();
    }
    max_val = smem[0];

    float sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        bool valid = !causal || col <= row;
        if (valid) {
            float v = to_f32(score_row[col]);
            sum += expf(v - max_val);
        }
    }
    smem[tid] = sum;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) smem[tid] += smem[tid + offset];
        __syncthreads();
    }
    sum = smem[0];

    for (int col = tid; col < cols; col += blockDim.x) {
        bool valid = !causal || col <= row;
        float p = 0.0f;
        if (valid) {
            float v = to_f32(score_row[col]);
            p = expf(v - max_val) / sum;
        }
        prob_row[col] = to_fp8(p);
    }
}

static __global__ void cosmos_test_bf16_to_fp8_kernel(
    const cutlass::bfloat16_t* __restrict__ src,
    cutlass::float_e4m3_t* __restrict__ dst,
    int64_t n)
{
    int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    cutlass::NumericConverter<float, cutlass::bfloat16_t> to_float;
    cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;
    dst[idx] = to_fp8(to_float(src[idx]));
}

namespace {

using CosmosFp8BatchedRcrGemm = cutlass::gemm::device::GemmBatched<
    cutlass::float_e4m3_t,
    cutlass::layout::RowMajor,
    cutlass::float_e4m3_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 64, 128>,
    cutlass::gemm::GemmShape<64, 32, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3>;

static cudaError_t cosmos_fp8_batched_rcr_gemm(
    const cutlass::float_e4m3_t* input_row,
    const cutlass::float_e4m3_t* weight_col,
    cutlass::half_t* output_row,
    int batch_count,
    int m,
    int k,
    int n,
    int64_t batch_stride_a,
    int64_t batch_stride_b,
    int64_t batch_stride_c,
    float alpha,
    cudaStream_t stream)
{
    CosmosFp8BatchedRcrGemm gemm_op;
    CosmosFp8BatchedRcrGemm::Arguments args(
        {m, n, k},
        {input_row, k},
        batch_stride_a,
        {weight_col, k},
        batch_stride_b,
        {output_row, n},
        batch_stride_c,
        {output_row, n},
        batch_stride_c,
        {alpha, 0.0f},
        batch_count);

    cutlass::Status status = gemm_op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr,
            "[DIAG] attn_batched_gemm init FAIL | batches=%d M=%d N=%d K=%d | status=%s\n",
            batch_count, m, n, k, cutlass::cutlassGetStatusString(status));
        return cudaErrorUnknown;
    }
    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr,
            "[DIAG] attn_batched_gemm run FAIL | batches=%d M=%d N=%d K=%d | status=%s\n",
            batch_count, m, n, k, cutlass::cutlassGetStatusString(status));
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

}  // namespace

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static torch::Tensor get_w(const py::dict& d, const std::string& key) {
    TORCH_CHECK(d.contains(key.c_str()),
        "cosmos_forward: missing weight key '", key, "'");
    return py::cast<torch::Tensor>(d[key.c_str()]).contiguous();
}

torch::Tensor cosmos_test_linear_fp8(
    torch::Tensor input,
    torch::Tensor weight_fp8_u8,
    c10::optional<torch::Tensor> weight_scale_opt,
    c10::optional<torch::Tensor> bias_opt,
    bool gelu)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(weight_fp8_u8.is_cuda(), "weight_fp8_u8 must be CUDA");
    TORCH_CHECK(input.scalar_type() == at::kHalf, "input must be torch.float16");
    TORCH_CHECK(weight_fp8_u8.scalar_type() == at::kByte, "weight_fp8_u8 must be torch.uint8");
    TORCH_CHECK(input.dim() == 2, "input must be [N, in_features]");
    TORCH_CHECK(weight_fp8_u8.dim() == 2, "weight_fp8_u8 must be [out_features, in_features]");

    input = input.contiguous();
    weight_fp8_u8 = weight_fp8_u8.contiguous();
    const int N = static_cast<int>(input.size(0));
    const int in_features = static_cast<int>(input.size(1));
    const int out_features = static_cast<int>(weight_fp8_u8.size(0));
    TORCH_CHECK(weight_fp8_u8.size(1) == in_features,
                "weight_fp8_u8 shape mismatch: expected second dim ", in_features,
                ", got ", weight_fp8_u8.size(1));

    torch::Tensor weight_scale;
    const cutlass::half_t* weight_scale_ptr = nullptr;
    if (weight_scale_opt.has_value() && weight_scale_opt.value().defined()) {
        weight_scale = weight_scale_opt.value().to(torch::kFloat16).contiguous();
        TORCH_CHECK(weight_scale.is_cuda(), "weight_scale must be CUDA");
        TORCH_CHECK(weight_scale.dim() == 1 && weight_scale.size(0) == out_features,
                    "weight_scale must be [out_features]");
        weight_scale_ptr = reinterpret_cast<const cutlass::half_t*>(weight_scale.data_ptr<at::Half>());
    }

    torch::Tensor bias;
    const cutlass::half_t* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        bias = bias_opt.value().to(torch::kFloat16).contiguous();
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == out_features,
                    "bias must be [out_features]");
        bias_ptr = reinterpret_cast<const cutlass::half_t*>(bias.data_ptr<at::Half>());
    }

    auto out = torch::empty({N, out_features}, input.options());
    auto fp8_scratch = torch::empty({N, in_features},
                                    torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    cudaError_t err = cudaSuccess;
    if (gelu) {
        err = omnidreams_singleview::apply_linear_row_gelu<cutlass::float_e4m3_t>(
            reinterpret_cast<const cutlass::half_t*>(input.data_ptr<at::Half>()),
            reinterpret_cast<const cutlass::float_e4m3_t*>(weight_fp8_u8.data_ptr<uint8_t>()),
            bias_ptr,
            reinterpret_cast<cutlass::half_t*>(out.data_ptr<at::Half>()),
            N, in_features, out_features,
            stream,
            reinterpret_cast<cutlass::half_t*>(fp8_scratch.data_ptr<uint8_t>()),
            weight_scale_ptr);
    } else {
        err = omnidreams_singleview::apply_linear_row<cutlass::float_e4m3_t>(
            reinterpret_cast<const cutlass::half_t*>(input.data_ptr<at::Half>()),
            reinterpret_cast<const cutlass::float_e4m3_t*>(weight_fp8_u8.data_ptr<uint8_t>()),
            bias_ptr,
            reinterpret_cast<cutlass::half_t*>(out.data_ptr<at::Half>()),
            N, in_features, out_features,
            stream,
            reinterpret_cast<cutlass::half_t*>(fp8_scratch.data_ptr<uint8_t>()),
            weight_scale_ptr);
    }
    TORCH_CHECK(err == cudaSuccess, "cosmos_test_linear_fp8 failed: ", cudaGetErrorString(err));
    return out;
}

torch::Tensor cosmos_test_linear_fp8_out_fp8(
    torch::Tensor input_bf16,
    torch::Tensor weight_fp8_u8,
    torch::Tensor weight_scale)
{
    TORCH_CHECK(input_bf16.is_cuda(), "input_bf16 must be CUDA");
    TORCH_CHECK(weight_fp8_u8.is_cuda(), "weight_fp8_u8 must be CUDA");
    TORCH_CHECK(weight_scale.is_cuda(), "weight_scale must be CUDA");
    TORCH_CHECK(input_bf16.scalar_type() == at::kBFloat16, "input_bf16 must be torch.bfloat16");
    TORCH_CHECK(weight_fp8_u8.scalar_type() == at::kByte, "weight_fp8_u8 must be torch.uint8");
    TORCH_CHECK(input_bf16.dim() == 2, "input_bf16 must be [N, in_features]");
    TORCH_CHECK(weight_fp8_u8.dim() == 2, "weight_fp8_u8 must be [out_features, in_features]");

    input_bf16 = input_bf16.contiguous();
    weight_fp8_u8 = weight_fp8_u8.contiguous();
    weight_scale = weight_scale.to(torch::kFloat16).contiguous();

    const int N = static_cast<int>(input_bf16.size(0));
    const int in_features = static_cast<int>(input_bf16.size(1));
    const int out_features = static_cast<int>(weight_fp8_u8.size(0));
    TORCH_CHECK(weight_fp8_u8.size(1) == in_features,
                "weight_fp8_u8 shape mismatch: expected second dim ", in_features,
                ", got ", weight_fp8_u8.size(1));
    TORCH_CHECK(weight_scale.dim() == 1 && weight_scale.size(0) == out_features,
                "weight_scale must be [out_features]");

    auto u8_opts = torch::TensorOptions().dtype(torch::kUInt8).device(input_bf16.device());
    auto h_opts = torch::TensorOptions().dtype(torch::kFloat16).device(input_bf16.device());
    auto input_fp8 = torch::empty({N, in_features}, u8_opts);
    auto half_scratch = torch::empty({N, out_features}, h_opts);
    auto output_fp8 = torch::empty({N, out_features}, u8_opts);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int threads = 256;
    int64_t in_elems = int64_t(N) * in_features;
    cosmos_test_bf16_to_fp8_kernel<<<static_cast<unsigned int>((in_elems + threads - 1) / threads), threads, 0, stream>>>(
        reinterpret_cast<const cutlass::bfloat16_t*>(input_bf16.data_ptr<at::BFloat16>()),
        reinterpret_cast<cutlass::float_e4m3_t*>(input_fp8.data_ptr<uint8_t>()),
        in_elems);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "input BF16->FP8 quantization failed: ", cudaGetErrorString(err));

    constexpr float prescale_alpha = 1.0f / 128.0f;
    constexpr float output_scale_mul = 128.0f;
    {
      at::Tensor _s = omnidreams_singleview::sgl_linear_rcr_fp8_bare(
          input_fp8.data_ptr<uint8_t>(), weight_fp8_u8.data_ptr<uint8_t>(),
          N, in_features, out_features);
      cudaMemcpyAsync(half_scratch.data_ptr<at::Half>(), _s.data_ptr(),
                      _s.numel() * _s.element_size(), cudaMemcpyDeviceToDevice, stream);
    }
    TORCH_CHECK(err == cudaSuccess, "FP8 linear for out-FP8 probe failed: ", cudaGetErrorString(err));

    err = omnidreams_singleview::apply_col_scale_bias_to_fp8(
        reinterpret_cast<const cutlass::half_t*>(half_scratch.data_ptr<at::Half>()),
        reinterpret_cast<cutlass::float_e4m3_t*>(output_fp8.data_ptr<uint8_t>()),
        reinterpret_cast<const cutlass::half_t*>(weight_scale.data_ptr<at::Half>()),
        nullptr,
        N, out_features, stream,
        output_scale_mul);
    TORCH_CHECK(err == cudaSuccess, "output half->FP8 quantization failed: ", cudaGetErrorString(err));
    return output_fp8;
}

torch::Tensor cosmos_test_linear_fp8_gelu_out_fp8(
    torch::Tensor input_bf16,
    torch::Tensor weight_fp8_u8,
    torch::Tensor weight_scale,
    double output_scale,
    bool alias_output)
{
    TORCH_CHECK(input_bf16.is_cuda(), "input_bf16 must be CUDA");
    TORCH_CHECK(weight_fp8_u8.is_cuda(), "weight_fp8_u8 must be CUDA");
    TORCH_CHECK(weight_scale.is_cuda(), "weight_scale must be CUDA");
    TORCH_CHECK(input_bf16.scalar_type() == at::kBFloat16, "input_bf16 must be torch.bfloat16");
    TORCH_CHECK(weight_fp8_u8.scalar_type() == at::kByte, "weight_fp8_u8 must be torch.uint8");
    TORCH_CHECK(output_scale > 0.0 && std::isfinite(output_scale), "output_scale must be positive and finite");
    TORCH_CHECK(input_bf16.dim() == 2, "input_bf16 must be [N, in_features]");
    TORCH_CHECK(weight_fp8_u8.dim() == 2, "weight_fp8_u8 must be [out_features, in_features]");

    input_bf16 = input_bf16.contiguous();
    weight_fp8_u8 = weight_fp8_u8.contiguous();
    weight_scale = weight_scale.to(torch::kFloat16).contiguous();

    const int N = static_cast<int>(input_bf16.size(0));
    const int in_features = static_cast<int>(input_bf16.size(1));
    const int out_features = static_cast<int>(weight_fp8_u8.size(0));
    TORCH_CHECK(weight_fp8_u8.size(1) == in_features,
                "weight_fp8_u8 shape mismatch: expected second dim ", in_features,
                ", got ", weight_fp8_u8.size(1));
    TORCH_CHECK(weight_scale.dim() == 1 && weight_scale.size(0) == out_features,
                "weight_scale must be [out_features]");
    TORCH_CHECK(!alias_output || output_scale == 1.0,
                "alias_output is only supported on the fused output_scale=1 path");

    auto u8_opts = torch::TensorOptions().dtype(torch::kUInt8).device(input_bf16.device());
    torch::Tensor scratch_fp8;
    torch::Tensor input_fp8;
    torch::Tensor output_fp8;
    if (alias_output) {
        const int scratch_features = std::max(in_features, out_features);
        scratch_fp8 = torch::empty({int64_t(N) * scratch_features}, u8_opts);
        input_fp8 = scratch_fp8;
        output_fp8 = scratch_fp8.narrow(0, 0, int64_t(N) * out_features).view({N, out_features});
    } else {
        input_fp8 = torch::empty({N, in_features}, u8_opts);
        output_fp8 = torch::empty({N, out_features}, u8_opts);
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int threads = 256;
    int64_t in_elems = int64_t(N) * in_features;
    cosmos_test_bf16_to_fp8_kernel<<<static_cast<unsigned int>((in_elems + threads - 1) / threads), threads, 0, stream>>>(
        reinterpret_cast<const cutlass::bfloat16_t*>(input_bf16.data_ptr<at::BFloat16>()),
        reinterpret_cast<cutlass::float_e4m3_t*>(input_fp8.data_ptr<uint8_t>()),
        in_elems);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "input BF16->FP8 quantization failed: ", cudaGetErrorString(err));

    if (output_scale == 1.0) {
    {
      at::Tensor _s = omnidreams_singleview::sgl_linear_rcr_fp8_bare(
          input_fp8.data_ptr<uint8_t>(), weight_fp8_u8.data_ptr<uint8_t>(),
          N, in_features, out_features);
      cudaMemcpyAsync(half_scratch.data_ptr<at::Half>(), _s.data_ptr(),
                      _s.numel() * _s.element_size(), cudaMemcpyDeviceToDevice, stream);
    }
    err = omnidreams_singleview::apply_col_scale_bias_gelu_to_fp8(
        reinterpret_cast<const cutlass::half_t*>(half_scratch.data_ptr<at::Half>()),
        reinterpret_cast<cutlass::float_e4m3_t*>(output_fp8.data_ptr<uint8_t>()),
        reinterpret_cast<const cutlass::half_t*>(weight_scale.data_ptr<at::Half>()),
        nullptr,
        N, out_features, stream,
        1.0f);
        TORCH_CHECK(err == cudaSuccess, "fused GELU FP8-output linear failed: ", cudaGetErrorString(err));
    } else {
        auto half_scratch = torch::empty(
            {N, out_features},
            torch::TensorOptions().dtype(torch::kFloat16).device(input_bf16.device()));
        constexpr float prescale_alpha = 1.0f / 128.0f;
        constexpr float output_scale_mul = 128.0f;
    {
      at::Tensor _s = omnidreams_singleview::sgl_linear_rcr_fp8_bare(
          input_fp8.data_ptr<uint8_t>(), weight_fp8_u8.data_ptr<uint8_t>(),
          N, in_features, out_features);
      cudaMemcpyAsync(half_scratch.data_ptr<at::Half>(), _s.data_ptr(),
                      _s.numel() * _s.element_size(), cudaMemcpyDeviceToDevice, stream);
    }
        TORCH_CHECK(err == cudaSuccess, "FP8 linear for scaled GELU FP8-output probe failed: ", cudaGetErrorString(err));
        err = omnidreams_singleview::apply_col_scale_bias_gelu_to_fp8(
            reinterpret_cast<const cutlass::half_t*>(half_scratch.data_ptr<at::Half>()),
            reinterpret_cast<cutlass::float_e4m3_t*>(output_fp8.data_ptr<uint8_t>()),
            reinterpret_cast<const cutlass::half_t*>(weight_scale.data_ptr<at::Half>()),
            nullptr,
            N, out_features, stream,
            output_scale_mul,
            static_cast<float>(output_scale));
        TORCH_CHECK(err == cudaSuccess, "scaled GELU output half->FP8 quantization failed: ", cudaGetErrorString(err));
    }
    return output_fp8;
}

torch::Tensor cosmos_test_linear_fp8_scaled_bf16(
    torch::Tensor input_bf16,
    torch::Tensor weight_fp8_u8,
    torch::Tensor weight_scale)
{
    TORCH_CHECK(input_bf16.is_cuda(), "input_bf16 must be CUDA");
    TORCH_CHECK(weight_fp8_u8.is_cuda(), "weight_fp8_u8 must be CUDA");
    TORCH_CHECK(weight_scale.is_cuda(), "weight_scale must be CUDA");
    TORCH_CHECK(input_bf16.scalar_type() == at::kBFloat16, "input_bf16 must be torch.bfloat16");
    TORCH_CHECK(weight_fp8_u8.scalar_type() == at::kByte, "weight_fp8_u8 must be torch.uint8");
    TORCH_CHECK(input_bf16.dim() == 2, "input_bf16 must be [N, in_features]");
    TORCH_CHECK(weight_fp8_u8.dim() == 2, "weight_fp8_u8 must be [out_features, in_features]");

    input_bf16 = input_bf16.contiguous();
    weight_fp8_u8 = weight_fp8_u8.contiguous();
    weight_scale = weight_scale.to(torch::kFloat16).contiguous();

    const int N = static_cast<int>(input_bf16.size(0));
    const int in_features = static_cast<int>(input_bf16.size(1));
    const int out_features = static_cast<int>(weight_fp8_u8.size(0));
    TORCH_CHECK(weight_fp8_u8.size(1) == in_features,
                "weight_fp8_u8 shape mismatch: expected second dim ", in_features,
                ", got ", weight_fp8_u8.size(1));
    TORCH_CHECK(weight_scale.dim() == 1 && weight_scale.size(0) == out_features,
                "weight_scale must be [out_features]");

    auto u8_opts = torch::TensorOptions().dtype(torch::kUInt8).device(input_bf16.device());
    auto bf16_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(input_bf16.device());
    auto input_fp8 = torch::empty({N, in_features}, u8_opts);
    auto output_bf16 = torch::empty({N, out_features}, bf16_opts);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int threads = 256;
    int64_t in_elems = int64_t(N) * in_features;
    cosmos_test_bf16_to_fp8_kernel<<<static_cast<unsigned int>((in_elems + threads - 1) / threads), threads, 0, stream>>>(
        reinterpret_cast<const cutlass::bfloat16_t*>(input_bf16.data_ptr<at::BFloat16>()),
        reinterpret_cast<cutlass::float_e4m3_t*>(input_fp8.data_ptr<uint8_t>()),
        in_elems);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "input BF16->FP8 quantization failed: ", cudaGetErrorString(err));

    {
      at::Tensor _s = omnidreams_singleview::sgl_linear_rcr_fp8_colscale_bf16(
          input_fp8.data_ptr<uint8_t>(), weight_fp8_u8.data_ptr<uint8_t>(),
          weight_scale.data_ptr<at::Half>(),
          N, in_features, out_features, stream);
      cudaMemcpyAsync(output_bf16.data_ptr<at::BFloat16>(), _s.data_ptr(),
                      _s.numel() * _s.element_size(), cudaMemcpyDeviceToDevice, stream);
    }
    TORCH_CHECK(err == cudaSuccess, "fused scaled BF16 FP8 linear failed: ", cudaGetErrorString(err));
    return output_bf16;
}

torch::Tensor cosmos_test_linear_fp8_residual_scaled_bf16(
    torch::Tensor input_bf16,
    torch::Tensor weight_fp8_u8,
    torch::Tensor alpha,
    torch::Tensor residual_bf16)
{
    TORCH_CHECK(input_bf16.is_cuda(), "input_bf16 must be CUDA");
    TORCH_CHECK(weight_fp8_u8.is_cuda(), "weight_fp8_u8 must be CUDA");
    TORCH_CHECK(alpha.is_cuda(), "alpha must be CUDA");
    TORCH_CHECK(residual_bf16.is_cuda(), "residual_bf16 must be CUDA");
    TORCH_CHECK(input_bf16.scalar_type() == at::kBFloat16, "input_bf16 must be torch.bfloat16");
    TORCH_CHECK(weight_fp8_u8.scalar_type() == at::kByte, "weight_fp8_u8 must be torch.uint8");
    TORCH_CHECK(residual_bf16.scalar_type() == at::kBFloat16, "residual_bf16 must be torch.bfloat16");
    TORCH_CHECK(input_bf16.dim() == 2, "input_bf16 must be [N, in_features]");
    TORCH_CHECK(weight_fp8_u8.dim() == 2, "weight_fp8_u8 must be [out_features, in_features]");
    TORCH_CHECK(residual_bf16.dim() == 2, "residual_bf16 must be [N, out_features]");

    input_bf16 = input_bf16.contiguous();
    weight_fp8_u8 = weight_fp8_u8.contiguous();
    alpha = alpha.to(torch::kFloat16).contiguous();
    auto output_bf16 = residual_bf16.contiguous().clone();

    const int N = static_cast<int>(input_bf16.size(0));
    const int in_features = static_cast<int>(input_bf16.size(1));
    const int out_features = static_cast<int>(weight_fp8_u8.size(0));
    TORCH_CHECK(weight_fp8_u8.size(1) == in_features,
                "weight_fp8_u8 shape mismatch: expected second dim ", in_features,
                ", got ", weight_fp8_u8.size(1));
    TORCH_CHECK(output_bf16.size(0) == N && output_bf16.size(1) == out_features,
                "residual_bf16 must be [N, out_features]");
    TORCH_CHECK(alpha.dim() == 1 && alpha.size(0) == out_features,
                "alpha must be [out_features]");

    auto u8_opts = torch::TensorOptions().dtype(torch::kUInt8).device(input_bf16.device());
    auto input_fp8 = torch::empty({N, in_features}, u8_opts);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int threads = 256;
    int64_t in_elems = int64_t(N) * in_features;
    cosmos_test_bf16_to_fp8_kernel<<<static_cast<unsigned int>((in_elems + threads - 1) / threads), threads, 0, stream>>>(
        reinterpret_cast<const cutlass::bfloat16_t*>(input_bf16.data_ptr<at::BFloat16>()),
        reinterpret_cast<cutlass::float_e4m3_t*>(input_fp8.data_ptr<uint8_t>()),
        in_elems);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "input BF16->FP8 quantization failed: ", cudaGetErrorString(err));

    {
      at::Tensor _s = omnidreams_singleview::sgl_linear_rcr_fp8_bare(
          input_fp8.data_ptr<uint8_t>(), weight_fp8_u8.data_ptr<uint8_t>(),
          N, in_features, out_features);
      cudaMemcpyAsync(half_scratch.data_ptr<at::Half>(), _s.data_ptr(),
                      _s.numel() * _s.element_size(), cudaMemcpyDeviceToDevice, stream);
    }
    err = omnidreams_singleview::cosmos_col_scale_residual_gate_bf16(
        reinterpret_cast<const cutlass::half_t*>(half_scratch.data_ptr<at::Half>()),
        reinterpret_cast<const cutlass::half_t*>(alpha.data_ptr<at::Half>()),
        reinterpret_cast<cutlass::bfloat16_t*>(output_bf16.data_ptr<at::BFloat16>()),
        gate.data_ptr<at::BFloat16>() ? reinterpret_cast<const cutlass::bfloat16_t*>(gate.data_ptr<at::BFloat16>()) : nullptr,
        N, out_features, B, stream, 1.0f);
    TORCH_CHECK(err == cudaSuccess, "fused residual BF16 FP8 linear failed: ", cudaGetErrorString(err));
    return output_bf16;
}

py::dict cosmos_test_fp8_linear_tile_selection(
    std::string op_kind,
    int64_t rows,
    int64_t in_features,
    int64_t out_features)
{
    auto selection = omnidreams_singleview::select_cosmos_fp8_linear_tile(
        op_kind,
        static_cast<int>(rows),
        static_cast<int>(in_features),
        static_cast<int>(out_features));
    py::dict out;
    out["op_kind"] = selection.op_kind;
    out["preset"] = selection.preset;
    out["tile"] = selection.tile;
    out["stage"] = selection.stage;
    out["variant"] = selection.variant;
    out["reason"] = selection.reason;
    out["tile_env_override"] = selection.tile_env_override;
    out["stage_env_override"] = selection.stage_env_override;
    out["variant_env_override"] = selection.variant_env_override;
    return out;
}

py::dict cosmos_test_fp8_sdpa_selection(
    int64_t B,
    int64_t Mq,
    int64_t Mk,
    int64_t H,
    int64_t D)
{
    auto selection = omnidreams_singleview::select_cosmos_fp8_sdpa(
        static_cast<int>(B),
        static_cast<int>(Mq),
        static_cast<int>(Mk),
        static_cast<int>(H),
        static_cast<int>(D));
    py::dict out;
    out["preset"] = selection.preset;
    out["layout"] = selection.layout;
    out["heuristics"] = selection.heuristics;
    out["plan"] = selection.plan;
    out["reason"] = selection.reason;
    out["layout_env_override"] = selection.layout_env_override;
    out["heuristics_env_override"] = selection.heuristics_env_override;
    out["plan_env_override"] = selection.plan_env_override;
    return out;
}

torch::Tensor cosmos_test_fp8_dense_ref_sdpa(
    torch::Tensor q_fp8_u8,
    torch::Tensor k_fp8_u8,
    torch::Tensor v_fp8_u8,
    bool causal)
{
    TORCH_CHECK(q_fp8_u8.is_cuda() && k_fp8_u8.is_cuda() && v_fp8_u8.is_cuda(),
                "q/k/v must be CUDA tensors");
    TORCH_CHECK(q_fp8_u8.scalar_type() == at::kByte &&
                k_fp8_u8.scalar_type() == at::kByte &&
                v_fp8_u8.scalar_type() == at::kByte,
                "q/k/v must be raw FP8 bytes as torch.uint8");
    TORCH_CHECK(q_fp8_u8.dim() == 4 && k_fp8_u8.dim() == 4 && v_fp8_u8.dim() == 4,
                "q/k/v must be [B, S, H, D]");
    TORCH_CHECK(q_fp8_u8.size(0) == k_fp8_u8.size(0) &&
                q_fp8_u8.size(1) == k_fp8_u8.size(1) &&
                q_fp8_u8.size(2) == k_fp8_u8.size(2) &&
                q_fp8_u8.size(3) == k_fp8_u8.size(3) &&
                q_fp8_u8.size(0) == v_fp8_u8.size(0) &&
                q_fp8_u8.size(1) == v_fp8_u8.size(1) &&
                q_fp8_u8.size(2) == v_fp8_u8.size(2) &&
                q_fp8_u8.size(3) == v_fp8_u8.size(3),
                "FP8 dense reference SDPA currently requires self-attention q/k/v shape equality");

    int B = static_cast<int>(q_fp8_u8.size(0));
    int S = static_cast<int>(q_fp8_u8.size(1));
    int H = static_cast<int>(q_fp8_u8.size(2));
    int D = static_cast<int>(q_fp8_u8.size(3));
    TORCH_CHECK(S > 0 && D > 0, "S and D must be positive");

    auto q_bhsd = q_fp8_u8.permute({0, 2, 1, 3}).contiguous();
    auto k_bhsd = k_fp8_u8.permute({0, 2, 1, 3}).contiguous();
    auto v_bhds = v_fp8_u8.permute({0, 2, 3, 1}).contiguous();

    auto u8_opts = torch::TensorOptions().dtype(torch::kUInt8).device(q_fp8_u8.device());
    auto h_opts = torch::TensorOptions().dtype(torch::kFloat16).device(q_fp8_u8.device());
    auto scores = torch::empty({B * H, S, S}, h_opts);
    auto probs = torch::empty({B * H, S, S}, u8_opts);
    auto out_bhsd = torch::empty({B, H, S, D}, h_opts);

    auto* q_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(q_bhsd.data_ptr<uint8_t>());
    auto* k_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(k_bhsd.data_ptr<uint8_t>());
    auto* v_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(v_bhds.data_ptr<uint8_t>());
    auto* score_ptr = reinterpret_cast<cutlass::half_t*>(scores.data_ptr<at::Half>());
    auto* prob_ptr = reinterpret_cast<cutlass::float_e4m3_t*>(probs.data_ptr<uint8_t>());
    auto* out_ptr = reinterpret_cast<cutlass::half_t*>(out_bhsd.data_ptr<at::Half>());

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(D));
    const int groups = B * H;
    const size_t qkv_group_elems = static_cast<size_t>(S) * D;
    const size_t score_group_elems = static_cast<size_t>(S) * S;

    cudaError_t err = cosmos_fp8_batched_rcr_gemm(
        q_ptr, k_ptr, score_ptr,
        groups,
        S, D, S,
        static_cast<int64_t>(qkv_group_elems),
        static_cast<int64_t>(qkv_group_elems),
        static_cast<int64_t>(score_group_elems),
        attn_scale,
        stream);
    TORCH_CHECK(err == cudaSuccess,
                "FP8 dense reference SDPA batched QK GEMM failed: ", cudaGetErrorString(err));

    cosmos_softmax_batched_half_to_fp8_kernel<<<dim3(S, groups), 256, 256 * sizeof(float), stream>>>(
        score_ptr, prob_ptr, groups, S, S, causal);
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "FP8 dense reference SDPA batched softmax kernel failed: ", cudaGetErrorString(err));

    err = cosmos_fp8_batched_rcr_gemm(
        prob_ptr, v_ptr, out_ptr,
        groups,
        S, S, D,
        static_cast<int64_t>(score_group_elems),
        static_cast<int64_t>(qkv_group_elems),
        static_cast<int64_t>(qkv_group_elems),
        1.0f,
        stream);
    TORCH_CHECK(err == cudaSuccess,
                "FP8 dense reference SDPA batched PV GEMM failed: ", cudaGetErrorString(err));

    return out_bhsd.permute({0, 2, 1, 3}).contiguous();
}

torch::Tensor cosmos_test_fp8_cudnn_sdpa(
    torch::Tensor q_fp8_u8,
    torch::Tensor k_fp8_u8,
    torch::Tensor v_fp8_u8,
    bool causal)
{
    TORCH_CHECK(q_fp8_u8.is_cuda() && k_fp8_u8.is_cuda() && v_fp8_u8.is_cuda(),
                "q/k/v must be CUDA tensors");
    TORCH_CHECK(q_fp8_u8.scalar_type() == at::kByte &&
                k_fp8_u8.scalar_type() == at::kByte &&
                v_fp8_u8.scalar_type() == at::kByte,
                "q/k/v must be raw FP8 bytes as torch.uint8");
    TORCH_CHECK(q_fp8_u8.dim() == 4 && k_fp8_u8.dim() == 4 && v_fp8_u8.dim() == 4,
                "q/k/v must be [B, S, H, D]");
    TORCH_CHECK(q_fp8_u8.size(0) == k_fp8_u8.size(0) &&
                q_fp8_u8.size(0) == v_fp8_u8.size(0),
                "q/k/v batch sizes must match");
    TORCH_CHECK(k_fp8_u8.size(1) == v_fp8_u8.size(1),
                "k/v sequence lengths must match");
    TORCH_CHECK(q_fp8_u8.size(2) == k_fp8_u8.size(2) &&
                q_fp8_u8.size(2) == v_fp8_u8.size(2),
                "q/k/v head counts must match");
    TORCH_CHECK(q_fp8_u8.size(3) == k_fp8_u8.size(3) &&
                q_fp8_u8.size(3) == v_fp8_u8.size(3),
                "q/k/v head dims must match");

    q_fp8_u8 = q_fp8_u8.contiguous();
    k_fp8_u8 = k_fp8_u8.contiguous();
    v_fp8_u8 = v_fp8_u8.contiguous();

    const int B = static_cast<int>(q_fp8_u8.size(0));
    const int Mq = static_cast<int>(q_fp8_u8.size(1));
    const int Mk = static_cast<int>(k_fp8_u8.size(1));
    const int H = static_cast<int>(q_fp8_u8.size(2));
    const int D = static_cast<int>(q_fp8_u8.size(3));
    TORCH_CHECK(D == 128,
                "cuDNN FP8 SDPA probe currently specializes D=128; got unsupported head dim ", D);
    const auto selection = omnidreams_singleview::select_cosmos_fp8_sdpa(B, Mq, Mk, H, D);

    auto byte_opts = torch::TensorOptions().dtype(torch::kUInt8).device(q_fp8_u8.device());
    auto fp32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(q_fp8_u8.device());
    auto out = torch::empty({B, Mq, H, D}, byte_opts);
    auto one = torch::ones({1, 1, 1, 1}, fp32_opts);
    auto amax_s = torch::empty({1, 1, 1, 1}, fp32_opts);
    auto amax_o = torch::empty({1, 1, 1, 1}, fp32_opts);
    torch::Tensor q_for_cudnn = q_fp8_u8;
    torch::Tensor k_for_cudnn = k_fp8_u8;
    torch::Tensor v_for_cudnn = v_fp8_u8;
    if (selection.layout == "bhmd") {
        q_for_cudnn = q_fp8_u8.permute({0, 2, 1, 3}).contiguous();
        k_for_cudnn = k_fp8_u8.permute({0, 2, 1, 3}).contiguous();
        v_for_cudnn = v_fp8_u8.permute({0, 2, 1, 3}).contiguous();
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = omnidreams_singleview::run_cudnn_fmha_packed_qkv_fp8(
        reinterpret_cast<const cutlass::float_e4m3_t*>(q_for_cudnn.data_ptr<uint8_t>()),
        reinterpret_cast<const cutlass::float_e4m3_t*>(k_for_cudnn.data_ptr<uint8_t>()),
        reinterpret_cast<const cutlass::float_e4m3_t*>(v_for_cudnn.data_ptr<uint8_t>()),
        reinterpret_cast<cutlass::float_e4m3_t*>(out.data_ptr<uint8_t>()),
        one.data_ptr<float>(),
        one.data_ptr<float>(),
        one.data_ptr<float>(),
        one.data_ptr<float>(),
        one.data_ptr<float>(),
        one.data_ptr<float>(),
        amax_s.data_ptr<float>(),
        amax_o.data_ptr<float>(),
        B, Mq, Mk, H, D, causal, 0.0f, stream);
    TORCH_CHECK(err == cudaSuccess,
                "cuDNN FP8 SDPA probe failed: ", cudaGetErrorString(err));
    return out;
}

torch::Tensor cosmos_test_fp8_tc_probe_qk(
    torch::Tensor q_fp8_u8,
    torch::Tensor k_fp8_u8)
{
    TORCH_CHECK(q_fp8_u8.is_cuda() && k_fp8_u8.is_cuda(),
                "q/k must be CUDA tensors");
    TORCH_CHECK(q_fp8_u8.scalar_type() == at::kByte &&
                k_fp8_u8.scalar_type() == at::kByte,
                "q/k must be raw FP8 bytes as torch.uint8");
    TORCH_CHECK(q_fp8_u8.dim() == 2 && k_fp8_u8.dim() == 2,
                "q/k must be 2D raw E4M3 tensors: q [Mq, D], k [Mk, D]");
    TORCH_CHECK(q_fp8_u8.size(1) == k_fp8_u8.size(1),
                "q/k inner dims must match");

    q_fp8_u8 = q_fp8_u8.contiguous();
    k_fp8_u8 = k_fp8_u8.contiguous();
    const int Mq = static_cast<int>(q_fp8_u8.size(0));
    const int Mk = static_cast<int>(k_fp8_u8.size(0));
    const int D = static_cast<int>(q_fp8_u8.size(1));
    TORCH_CHECK(Mq > 0 && Mk > 0 && D > 0, "q/k dimensions must be positive");
    TORCH_CHECK((Mq % 128) == 0 && (Mk % 128) == 0 && (D % 128) == 0,
                "SM120 FP8 tensor-core probe currently requires Mq, Mk, and D to be multiples of 128");

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(q_fp8_u8.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(q_fp8_u8.device());
    auto out = torch::empty({Mq, Mk}, opts_bf16);
    auto c_scratch = torch::empty({Mq, Mk}, opts_bf16);
    auto q_scale = torch::ones({std::max<int64_t>(1, int64_t(Mq) * D)}, opts_f32);
    auto k_scale = torch::ones({std::max<int64_t>(1, int64_t(Mk) * D)}, opts_f32);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = omnidreams_singleview::run_cosmos_fp8_tc_probe_qk(
        reinterpret_cast<const cutlass::float_e4m3_t*>(q_fp8_u8.data_ptr<uint8_t>()),
        reinterpret_cast<const cutlass::float_e4m3_t*>(k_fp8_u8.data_ptr<uint8_t>()),
        q_scale.data_ptr<float>(),
        k_scale.data_ptr<float>(),
        reinterpret_cast<const cutlass::bfloat16_t*>(c_scratch.data_ptr<at::BFloat16>()),
        reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>()),
        Mq, Mk, D, stream);
    TORCH_CHECK(err == cudaSuccess,
                "SM120 FP8 tensor-core QK probe failed: ", cudaGetErrorString(err));
    return out;
}

torch::Tensor cosmos_test_fp8_tc_probe_pv(
    torch::Tensor probs_fp8_u8,
    torch::Tensor v_fp8_u8)
{
    TORCH_CHECK(probs_fp8_u8.is_cuda() && v_fp8_u8.is_cuda(),
                "probs/v must be CUDA tensors");
    TORCH_CHECK(probs_fp8_u8.scalar_type() == at::kByte &&
                v_fp8_u8.scalar_type() == at::kByte,
                "probs/v must be raw FP8 bytes as torch.uint8");
    TORCH_CHECK(probs_fp8_u8.dim() == 2 && v_fp8_u8.dim() == 2,
                "probs/v must be 2D raw E4M3 tensors: probs [Mq, Mk], v [Mk, D]");
    TORCH_CHECK(probs_fp8_u8.size(1) == v_fp8_u8.size(0),
                "probs second dim must match v rows");

    probs_fp8_u8 = probs_fp8_u8.contiguous();
    v_fp8_u8 = v_fp8_u8.contiguous();
    const int Mq = static_cast<int>(probs_fp8_u8.size(0));
    const int Mk = static_cast<int>(probs_fp8_u8.size(1));
    const int D = static_cast<int>(v_fp8_u8.size(1));
    TORCH_CHECK(Mq > 0 && Mk > 0 && D > 0, "probs/v dimensions must be positive");
    TORCH_CHECK((Mq % 128) == 0 && (Mk % 128) == 0 && (D % 128) == 0,
                "SM120 FP8 tensor-core probe currently requires Mq, Mk, and D to be multiples of 128");

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(probs_fp8_u8.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(probs_fp8_u8.device());
    auto out = torch::empty({Mq, D}, opts_bf16);
    auto c_scratch = torch::empty({Mq, D}, opts_bf16);
    auto probs_scale = torch::ones({std::max<int64_t>(1, int64_t(Mq) * Mk)}, opts_f32);
    auto v_scale = torch::ones({std::max<int64_t>(1, int64_t(Mk) * D)}, opts_f32);
    // CUTLASS SM120 blockwise FP8 builder currently exposes the TN path only.
    // Accept raw Cosmos V as [Mk, D], then provide the tensor-core probe its
    // required [D, Mk] row-major storage.
    auto v_for_tc = v_fp8_u8.t().contiguous();

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = omnidreams_singleview::run_cosmos_fp8_tc_probe_pv(
        reinterpret_cast<const cutlass::float_e4m3_t*>(probs_fp8_u8.data_ptr<uint8_t>()),
        reinterpret_cast<const cutlass::float_e4m3_t*>(v_for_tc.data_ptr<uint8_t>()),
        probs_scale.data_ptr<float>(),
        v_scale.data_ptr<float>(),
        reinterpret_cast<const cutlass::bfloat16_t*>(c_scratch.data_ptr<at::BFloat16>()),
        reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>()),
        Mq, Mk, D, stream);
    TORCH_CHECK(err == cudaSuccess,
                "SM120 FP8 tensor-core PV probe failed: ", cudaGetErrorString(err));
    return out;
}

std::vector<torch::Tensor> cosmos_test_fp8_attention_backend(
    torch::Tensor q_fp8_u8,
    torch::Tensor k_fp8_u8,
    torch::Tensor v_fp8_u8,
    bool causal,
    std::string backend)
{
    if (backend == "fp8_dense_ref") {
        auto out = cosmos_test_fp8_dense_ref_sdpa(
            q_fp8_u8, k_fp8_u8, v_fp8_u8, causal);
        auto backend_id = torch::empty({1},
            torch::TensorOptions().dtype(torch::kInt32).device(out.device()));
        backend_id.fill_(1);
        return {out, backend_id};
    }

    if (backend == "fp8_cudnn") {
        auto out = cosmos_test_fp8_cudnn_sdpa(
            q_fp8_u8, k_fp8_u8, v_fp8_u8, causal);
        auto backend_id = torch::empty({1},
            torch::TensorOptions().dtype(torch::kInt32).device(out.device()));
        backend_id.fill_(8);
        return {out, backend_id};
    }

    TORCH_CHECK(false,
        "unknown Cosmos FP8 attention backend '", backend,
        "'. Expected one of: fp8_dense_ref, fp8_cudnn");
}

// RMSNorm: x * rsqrt(mean(x²) + eps) * gamma
// Works on any trailing shape; gamma must match last dim of x.
static torch::Tensor rms_norm(const torch::Tensor& x,
                               const torch::Tensor& gamma,
                               float eps = 1e-6f) {
    auto xf = x.to(torch::kFloat32);
    auto inv = torch::rsqrt(xf.pow(2).mean(-1, /*keepdim=*/true) + eps);
    return (xf * inv * gamma.to(torch::kFloat32)).to(x.dtype());
}

// LayerNorm with no learnable parameters (elementwise_affine=False, eps=1e-6).
static torch::Tensor layernorm_no_affine(const torch::Tensor& x, float eps = 1e-6f) {
    return torch::layer_norm(x, {x.size(-1)}, /*weight=*/{}, /*bias=*/{}, eps);
}

// rotate_half: [..., D] → [-x2, x1] (second half negated, first half kept).
static torch::Tensor rotate_half(const torch::Tensor& x) {
    int64_t half = x.size(-1) / 2;
    return torch::cat({-x.slice(-1, half), x.slice(-1, 0, half)}, -1);
}

// Cosmos VideoRopePosition3DEmb.generate_embeddings():
// returns [S, 1, 1, head_dim] float32 raw angles on the same device as `device_ref`.
static torch::Tensor compute_cosmos_rope_emb(
    int pT, int pH, int pW, int head_dim,
    float h_ratio, float w_ratio, float t_ratio,
    const torch::Device& device)
{
    int dim_h = (head_dim / 6) * 2;
    int dim_w = dim_h;
    int dim_t = head_dim - 2 * dim_h;

    float h_ntk = (dim_h > 2) ? std::pow(h_ratio, float(dim_h) / float(dim_h - 2)) : 1.0f;
    float w_ntk = (dim_w > 2) ? std::pow(w_ratio, float(dim_w) / float(dim_w - 2)) : 1.0f;
    float t_ntk = (dim_t > 2) ? std::pow(t_ratio, float(dim_t) / float(dim_t - 2)) : 1.0f;

    auto cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    auto h_range = torch::arange(0, dim_h, 2, cpu).narrow(0, 0, dim_h / 2) / float(dim_h);
    auto w_range = torch::arange(0, dim_w, 2, cpu).narrow(0, 0, dim_w / 2) / float(dim_w);
    auto t_range = torch::arange(0, dim_t, 2, cpu).narrow(0, 0, dim_t / 2) / float(dim_t);

    auto h_freqs = 1.f / torch::pow(10000.f * h_ntk, h_range);  // [dim_h/2]
    auto w_freqs = 1.f / torch::pow(10000.f * w_ntk, w_range);
    auto t_freqs = 1.f / torch::pow(10000.f * t_ntk, t_range);

    auto half_h = torch::outer(torch::arange(pH, cpu).to(torch::kFloat32), h_freqs); // [pH, dim_h/2]
    auto half_w = torch::outer(torch::arange(pW, cpu).to(torch::kFloat32), w_freqs);
    auto half_t = torch::outer(torch::arange(pT, cpu).to(torch::kFloat32), t_freqs);

    // Broadcast to [pT, pH, pW, dim/2]
    auto emb_t = half_t.unsqueeze(1).unsqueeze(1).expand({pT, pH, pW, dim_t / 2});
    auto emb_h = half_h.unsqueeze(0).unsqueeze(2).expand({pT, pH, pW, dim_h / 2});
    auto emb_w = half_w.unsqueeze(0).unsqueeze(1).expand({pT, pH, pW, dim_w / 2});

    // Cat [emb_t, emb_h, emb_w] × 2 to get [pT, pH, pW, head_dim]
    auto em = torch::cat({emb_t, emb_h, emb_w, emb_t, emb_h, emb_w}, -1);

    int S = pT * pH * pW;
    return em.reshape({S, 1, 1, head_dim}).to(device);
}

// Apply Cosmos rotate_half RoPE.
// q, k: [B, Mq, H, D]; rope_emb: [Mq, 1, 1, D] float32 raw angles.
static std::pair<torch::Tensor, torch::Tensor> apply_cosmos_rope(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& rope_emb)
{
    // [Mq, 1, 1, D] → [1, Mq, 1, D]
    auto rope = rope_emb.permute({1, 0, 2, 3});
    auto cos  = torch::cos(rope).to(q.dtype());
    auto sin  = torch::sin(rope).to(q.dtype());
    return {q * cos + rotate_half(q) * sin,
            k * cos + rotate_half(k) * sin};
}

// Run CUTLASS FMHA on pre-packed q/k/v (bfloat16).
// Input shape: [B, Mq/Mk, H, D] bf16. Returns [B, Mq, H, D] bf16.
static torch::Tensor fmha(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    bool causal, cudaStream_t stream)
{
    auto qb = q.to(torch::kBFloat16).contiguous();
    auto kb = k.to(torch::kBFloat16).contiguous();
    auto vb = v.to(torch::kBFloat16).contiguous();

    int B = (int)qb.size(0), Mq = (int)qb.size(1), H = (int)qb.size(2), D = (int)qb.size(3);
    int Mk = (int)kb.size(1);

    auto ob = torch::empty({B * Mq, H, D}, qb.options());
    auto qf = qb.reshape({B * Mq, H, D});
    auto kf = kb.reshape({B * Mk, H, D});
    auto vf = vb.reshape({B * Mk, H, D});

    cudaError_t err = omnidreams_singleview::run_cudnn_fmha_packed_qkv(
        reinterpret_cast<const cutlass::bfloat16_t*>(qf.data_ptr<at::BFloat16>()),
        reinterpret_cast<const cutlass::bfloat16_t*>(kf.data_ptr<at::BFloat16>()),
        reinterpret_cast<const cutlass::bfloat16_t*>(vf.data_ptr<at::BFloat16>()),
        reinterpret_cast<cutlass::bfloat16_t*>(ob.data_ptr<at::BFloat16>()),
        B, Mq, Mk, H, D, causal, /*scale=*/0.f, stream);

    TORCH_CHECK(err == cudaSuccess,
        "run_cudnn_fmha_packed_qkv failed: ", cudaGetErrorString(err));

    return ob.reshape({B, Mq, H, D});
}

// adaln-LoRA modulation for one sub-layer.
// emb:  [B, L, D] — post-norm timestep embedding
// down: weight at ".1.weight" — [lora_dim, D]
// up:   weight at ".2.weight" — [out_dim, lora_dim]
// Returns [B, L, out_dim] = SiLU(emb) @ down.T @ up.T
static torch::Tensor adaln_lora_proj(
    const torch::Tensor& emb,
    const torch::Tensor& down,   // [lora_dim, D]
    const torch::Tensor& up)     // [out_dim, lora_dim]
{
    auto h = torch::silu(emb);                     // [B, L, D]
    auto h2 = torch::matmul(h, down.t());          // [B, L, lora_dim]
    return torch::matmul(h2, up.t());              // [B, L, out_dim]
}

// GELU FFN: layer2(gelu(layer1(x)))
// w1: [FF, D], w2: [D, FF]  (PyTorch weight layout)
static torch::Tensor gelu_ffn(const torch::Tensor& x,
                               const torch::Tensor& w1,
                               const torch::Tensor& w2) {
    return torch::matmul(torch::gelu(torch::matmul(x, w1.t())), w2.t());
}

// Sinusoidal embedding matching Cosmos Timesteps module.
// ts: [N] float32 → [N, num_channels]
static torch::Tensor sinusoidal_emb(const torch::Tensor& ts, int num_channels) {
    int half = num_channels / 2;
    auto dev_opts = torch::TensorOptions().dtype(torch::kFloat32).device(ts.device());
    // Matches reference: exponent = arange(half_dim) / (half_dim - 0.0) = arange(half) / half
    auto exp = -std::log(10000.0) * torch::arange(half, dev_opts) / float(half);
    auto freqs = torch::exp(exp);                              // [half]
    auto angles = ts.unsqueeze(1) * freqs.unsqueeze(0);        // [N, half]
    return torch::cat({torch::cos(angles), torch::sin(angles)}, -1);  // [N, num_channels]
}

// Patch embedding: rearrange [B, C, T, H, W] → linear → [B, pT, pH, pW, D_out].
static torch::Tensor patch_embed(const torch::Tensor& x, const torch::Tensor& w,
                                  int pt, int ph, int pw) {
    int B = (int)x.size(0), C = (int)x.size(1);
    int T = (int)x.size(2), H = (int)x.size(3), W = (int)x.size(4);
    int pT = T / pt, pH = H / ph, pW = W / pw;

    // [B, C, pT, pt, pH, ph, pW, pw]
    auto r = x.reshape({B, C, pT, pt, pH, ph, pW, pw});
    // [B, pT, pH, pW, C, pt, ph, pw]
    r = r.permute({0, 2, 4, 6, 1, 3, 5, 7}).contiguous();
    // [B, pT, pH, pW, C*pt*ph*pw]
    r = r.reshape({B, pT, pH, pW, C * pt * ph * pw});
    return torch::matmul(r, w.t());  // [B, pT, pH, pW, D_out]
}

// ---------------------------------------------------------------------------
// Per-sub-block residual helpers
//
// Each residual helper takes the running `x`, normalizes a copy, applies an
// adaln-modulated linear / attention / FFN path, and returns `x + gate * out`.
//
// They are factored out of `cosmos_block_forward` so the streaming forward
// (Phase 2) can reuse them with different attention semantics (non-causal
// self-attn against a KV cache, pre-computed cross-attn K/V) while keeping
// Michael's full-sequence DDPM path bit-identical to the original code.
// ---------------------------------------------------------------------------

static std::string block_prefix(int block_idx) {
    return "blocks." + std::to_string(block_idx) + ".";
}

// Compute (shift, scale, gate) for one adaln-LoRA modulation sub-layer.
// t_emb:          [B, L, D]            post-norm timestep embedding
// adaln_lora_3d:  [B, L, 3*D]          adaln-LoRA component from t_embedder
// Returns three tensors each of shape [B, L, D].
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_adaln_mods(
    const torch::Tensor& t_emb,
    const torch::Tensor& adaln_lora_3d,
    const py::dict& weights,
    int block_idx,
    const std::string& mod_name)
{
    const std::string pfx = block_prefix(block_idx) + mod_name;
    auto down = get_w(weights, pfx + ".1.weight");  // [lora_dim, D]
    auto up   = get_w(weights, pfx + ".2.weight");  // [3*D, lora_dim]
    auto mods = adaln_lora_proj(t_emb, down, up) + adaln_lora_3d;
    auto chunks = mods.chunk(3, -1);
    return {chunks[0], chunks[1], chunks[2]};
}

// Self-attention residual block.
//   x + gate * OutProj( FMHA( QKVProj( norm(x) * (1+scale) + shift ) ) )
// `causal` selects the top-left causal mask (true for full-sequence DDPM
// training forward; false for streaming with KV cache where causality is
// enforced implicitly by only caching past tokens).
static torch::Tensor self_attn_residual(
    const torch::Tensor& x,              // [B, L, D]
    const torch::Tensor& shift,
    const torch::Tensor& scale,
    const torch::Tensor& gate,
    const py::dict& weights,
    int block_idx,
    const torch::Tensor& rope_emb,       // [L, 1, 1, Dh] float32 raw angles
    int num_heads,
    bool causal,
    cudaStream_t stream)
{
    int B = (int)x.size(0), L = (int)x.size(1), D = (int)x.size(2);
    int H = num_heads, Dh = D / H;
    const std::string pfx = block_prefix(block_idx) + "self_attn.";

    auto normed = layernorm_no_affine(x) * (1.f + scale) + shift;

    auto wq = get_w(weights, pfx + "q_proj.weight");         // [D, D]
    auto wk = get_w(weights, pfx + "k_proj.weight");
    auto wv = get_w(weights, pfx + "v_proj.weight");
    auto wo = get_w(weights, pfx + "output_proj.weight");

    auto gq = get_w(weights, pfx + "q_norm.weight");         // [Dh]
    auto gk = get_w(weights, pfx + "k_norm.weight");

    // [B, L, D] → [B, L, H, Dh]
    auto q = torch::matmul(normed, wq.t()).reshape({B, L, H, Dh});
    auto k = torch::matmul(normed, wk.t()).reshape({B, L, H, Dh});
    auto v = torch::matmul(normed, wv.t()).reshape({B, L, H, Dh});

    q = rms_norm(q, gq);
    k = rms_norm(k, gk);

    auto [q_rot, k_rot] = apply_cosmos_rope(q, k, rope_emb);

    auto attn_out = fmha(q_rot, k_rot, v, causal, stream);
    return x + gate * torch::matmul(attn_out.reshape({B, L, D}), wo.t());
}

// Cross-attention residual block (full — computes K/V from `ctx` on the fly).
// Streaming path will add a variant that takes pre-computed K/V caches.
static torch::Tensor cross_attn_residual(
    const torch::Tensor& x,              // [B, L,  D]
    const torch::Tensor& shift,
    const torch::Tensor& scale,
    const torch::Tensor& gate,
    const py::dict& weights,
    int block_idx,
    const torch::Tensor& ctx,            // [B, Lc, Dc]
    int num_heads,
    cudaStream_t stream)
{
    int B = (int)x.size(0), L = (int)x.size(1), D = (int)x.size(2);
    int Lc = (int)ctx.size(1);
    int H = num_heads, Dh = D / H;
    const std::string pfx = block_prefix(block_idx) + "cross_attn.";

    auto normed = layernorm_no_affine(x) * (1.f + scale) + shift;

    auto wq = get_w(weights, pfx + "q_proj.weight");         // [D, D]
    auto wk = get_w(weights, pfx + "k_proj.weight");         // [D, Dc]
    auto wv = get_w(weights, pfx + "v_proj.weight");         // [D, Dc]
    auto wo = get_w(weights, pfx + "output_proj.weight");

    auto gq = get_w(weights, pfx + "q_norm.weight");
    auto gk = get_w(weights, pfx + "k_norm.weight");

    auto q = torch::matmul(normed, wq.t()).reshape({B, L,  H, Dh});
    auto k = torch::matmul(ctx,    wk.t()).reshape({B, Lc, H, Dh});
    auto v = torch::matmul(ctx,    wv.t()).reshape({B, Lc, H, Dh});

    q = rms_norm(q, gq);
    k = rms_norm(k, gk);
    // No RoPE for cross-attention

    auto attn_out = fmha(q, k, v, /*causal=*/false, stream);
    return x + gate * torch::matmul(attn_out.reshape({B, L, D}), wo.t());
}

// MLP (GELU FFN) residual block.
static torch::Tensor mlp_residual(
    const torch::Tensor& x,              // [B, L, D]
    const torch::Tensor& shift,
    const torch::Tensor& scale,
    const torch::Tensor& gate,
    const py::dict& weights,
    int block_idx)
{
    const std::string pfx = block_prefix(block_idx) + "mlp.";
    auto normed = layernorm_no_affine(x) * (1.f + scale) + shift;
    auto w1 = get_w(weights, pfx + "layer1.weight");  // [FF, D]
    auto w2 = get_w(weights, pfx + "layer2.weight");  // [D, FF]
    return x + gate * gelu_ffn(normed, w1, w2);
}

// ---------------------------------------------------------------------------
// Single transformer block (full-sequence DDPM forward)
// ---------------------------------------------------------------------------

static torch::Tensor cosmos_block_forward(
    const torch::Tensor& x,              // [B, L, D]
    const torch::Tensor& t_emb,          // [B, L, D]     post-norm timestep emb
    const torch::Tensor& adaln_lora_3d,  // [B, L, 3*D]   from t_embedder
    const torch::Tensor& ctx,            // [B, Lc, Dc]   cross-attn context
    const torch::Tensor& rope_emb,       // [L, 1, 1, Dh] float32 raw angles
    const py::dict& weights,
    int block_idx, int num_heads, cudaStream_t stream)
{
    auto [shift_sa,  scale_sa,  gate_sa]  = compute_adaln_mods(
        t_emb, adaln_lora_3d, weights, block_idx, "adaln_modulation_self_attn");
    auto [shift_ca,  scale_ca,  gate_ca]  = compute_adaln_mods(
        t_emb, adaln_lora_3d, weights, block_idx, "adaln_modulation_cross_attn");
    auto [shift_mlp, scale_mlp, gate_mlp] = compute_adaln_mods(
        t_emb, adaln_lora_3d, weights, block_idx, "adaln_modulation_mlp");

    auto cur = x;
    cur = self_attn_residual (cur, shift_sa,  scale_sa,  gate_sa,
                              weights, block_idx, rope_emb, num_heads,
                              /*causal=*/true, stream);
    cur = cross_attn_residual(cur, shift_ca,  scale_ca,  gate_ca,
                              weights, block_idx, ctx, num_heads, stream);
    cur = mlp_residual       (cur, shift_mlp, scale_mlp, gate_mlp,
                              weights, block_idx);
    return cur;
}

// ---------------------------------------------------------------------------
// Main forward
// ---------------------------------------------------------------------------

torch::Tensor cosmos_forward(
    torch::Tensor x,
    torch::Tensor condition_mask,
    torch::Tensor padding_mask,
    torch::Tensor timesteps,
    torch::Tensor crossattn_emb,
    torch::Tensor hdmap,
    py::dict weights,
    py::dict config)
{
    TORCH_CHECK(x.is_cuda(),            "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 5,           "x must be 5D [B, C, T, H, W]");
    TORCH_CHECK(x.size(1) == 16,        "x channel dim must be 16");
    TORCH_CHECK(condition_mask.is_cuda() && condition_mask.dim() == 5);
    TORCH_CHECK(crossattn_emb.is_cuda() && crossattn_emb.dim() == 3);
    TORCH_CHECK(hdmap.is_cuda() && hdmap.dim() == 5);

    auto orig_dtype = x.dtype();
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // ── Config ───────────────────────────────────────────────────────────────
    auto ci = [&](const char* k, int d)   { return config.contains(k) ? py::cast<int>(config[k]) : d; };
    auto cf = [&](const char* k, float d) { return config.contains(k) ? py::cast<float>(config[k]) : d; };
    auto cb = [&](const char* k, bool d)  { return config.contains(k) ? py::cast<bool>(config[k]) : d; };

    int   num_blocks     = ci("num_blocks",      28);
    int   num_heads      = ci("num_heads",        16);
    int   model_channels = ci("model_channels",  2048);
    int   pt = ci("patch_temporal", 1),  ph = ci("patch_spatial", 2),  pw = ci("patch_spatial", 2);
    float ts_scale       = cf("timestep_scale",  0.001f);
    float h_ratio        = cf("rope_h_extrapolation_ratio", 3.0f);
    float w_ratio        = cf("rope_w_extrapolation_ratio", 3.0f);
    float t_ratio        = cf("rope_t_extrapolation_ratio", 1.0f);
    bool  concat_pm      = cb("concat_padding_mask", true);

    int head_dim = model_channels / num_heads;
    int B = (int)x.size(0), T = (int)x.size(2), H = (int)x.size(3), W = (int)x.size(4);

    // ── 1. Concat condition mask ─────────────────────────────────────────────
    // [B, 16+1, T, H, W]
    auto x_in = torch::cat({x, condition_mask.to(orig_dtype)}, 1);

    // ── 2. Scale timesteps ───────────────────────────────────────────────────
    auto ts = timesteps.to(torch::kFloat32) * ts_scale;  // [B, T]

    // ── 3. Concat padding mask ───────────────────────────────────────────────
    if (concat_pm) {
        auto pm = padding_mask.to(orig_dtype);
        if (pm.size(-2) != H || pm.size(-1) != W) {
            pm = F::interpolate(
                pm.to(torch::kFloat32),
                F::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{H, W})
                    .mode(torch::kNearest)
            ).to(orig_dtype);
        }
        // [B, 1, H, W] → [B, 1, 1, H, W] → [B, 1, T, H, W]
        x_in = torch::cat({x_in, pm.unsqueeze(2).expand({B, 1, T, H, W})}, 1);
    }

    // ── 4. Patch embedding ───────────────────────────────────────────────────
    auto w_pe = get_w(weights, "x_embedder.proj.1.weight");       // [D, C_in*pt*ph*pw]
    auto x_emb = patch_embed(x_in, w_pe, pt, ph, pw);             // [B, pT, pH, pW, D]

    int pT = T / pt, pHs = H / ph, pWs = W / pw;

    // ── 5. Additional patch embedding (hdmap) ────────────────────────────────
    {
        auto w_h = get_w(weights, "additional_patch_embedding.proj.1.weight");
        x_emb = x_emb + patch_embed(hdmap.to(orig_dtype), w_h, pt, ph, pw);
    }

    // ── 6. RoPE embeddings ───────────────────────────────────────────────────
    // [S=pT*pHs*pWs, 1, 1, head_dim] float32 raw angles
    auto rope_emb = compute_cosmos_rope_emb(pT, pHs, pWs, head_dim,
                                            h_ratio, w_ratio, t_ratio, x.device());

    // ── 7. Timestep embedding ────────────────────────────────────────────────
    auto ts_flat = ts.flatten().to(torch::kFloat32);           // [B*T]
    auto t_sin   = sinusoidal_emb(ts_flat, model_channels).to(orig_dtype); // [B*T, D]

    // TimestepEmbedding (use_adaln_lora=True): linear_1 has no bias
    auto w1 = get_w(weights, "t_embedder.1.linear_1.weight");  // [D, D]
    auto t_h = torch::silu(torch::matmul(t_sin, w1.t()));      // [B*T, D]

    auto w2 = get_w(weights, "t_embedder.1.linear_2.weight");  // [3*D, D]
    auto adaln_lora_BT3D = torch::matmul(t_h, w2.t());          // [B*T, 3*D]

    // When use_adaln_lora=True, emb_B_T_D = sample = t_sin (raw sinusoidal)
    auto t_emb_BT = t_sin;  // [B*T, D]

    // t_embedding_norm (RMSNorm)
    auto norm_gamma = get_w(weights, "t_embedding_norm.weight");
    t_emb_BT = rms_norm(t_emb_BT, norm_gamma);  // [B*T, D]

    // Reshape to [B, T, ...]
    auto t_emb_BTD  = t_emb_BT.reshape({B, T, model_channels});
    auto adaln_lora_BTD = adaln_lora_BT3D.reshape({B, T, 3 * model_channels});

    // Expand from [B, T, *] to [B, L, *] by repeating each frame pHs*pWs times
    int frame_seqlen = pHs * pWs;
    int L = pT * pHs * pWs;

    auto t_emb_BLD  = t_emb_BTD.repeat_interleave(frame_seqlen, 1);    // [B, L, D]
    auto adaln_BL3D = adaln_lora_BTD.repeat_interleave(frame_seqlen, 1); // [B, L, 3*D]

    // ── 8. Flatten x to [B, L, D] ────────────────────────────────────────────
    auto cur = x_emb.reshape({B, L, model_channels}).to(orig_dtype);

    // ── 9. Optional crossattn projection ─────────────────────────────────────
    auto ctx = crossattn_emb.to(orig_dtype);
    if (weights.contains("crossattn_proj.0.weight")) {
        auto wp = get_w(weights, "crossattn_proj.0.weight");  // [1024, in_ch]
        auto bp = get_w(weights, "crossattn_proj.0.bias");    // [1024]
        ctx = torch::gelu(torch::matmul(ctx, wp.t()) + bp);
    }

    // ── 10. Transformer blocks ───────────────────────────────────────────────
    for (int i = 0; i < num_blocks; ++i) {
        cur = cosmos_block_forward(
            cur, t_emb_BLD, adaln_BL3D, ctx, rope_emb,
            weights, i, num_heads, stream);
    }

    // ── 11. Final layer ──────────────────────────────────────────────────────
    // Reshape to [B, pT, pHs, pWs, D]
    auto x_out = cur.reshape({B, pT, pHs, pWs, model_channels});

    {
        // final_layer.adaln_modulation: SiLU → Linear(D, lora_dim) → Linear(lora_dim, 2*D)
        auto fl_down = get_w(weights, "final_layer.adaln_modulation.1.weight"); // [lora_dim, D]
        auto fl_up   = get_w(weights, "final_layer.adaln_modulation.2.weight"); // [2*D, lora_dim]

        // adaln_lora_B_T_3D[:, :, :2*D] is the lora component for the final layer
        auto fl_lora  = adaln_lora_BTD.slice(2, 0, 2 * model_channels);    // [B, T, 2*D]
        auto fl_mods  = adaln_lora_proj(t_emb_BTD, fl_down, fl_up) + fl_lora;  // [B, T, 2*D]
        auto fl_chunks = fl_mods.chunk(2, -1);
        // Broadcast [B, T, D] → [B, T, 1, 1, D] for spatial dims
        auto shift = fl_chunks[0].unsqueeze(2).unsqueeze(3);
        auto scale = fl_chunks[1].unsqueeze(2).unsqueeze(3);

        x_out = layernorm_no_affine(x_out) * (1.f + scale) + shift;

        auto w_fl = get_w(weights, "final_layer.linear.weight");  // [out_ch*pt*ph*pw, D]
        x_out = torch::matmul(x_out, w_fl.t());  // [B, pT, pHs, pWs, pt*ph*pw*out_ch]
    }

    // ── 12. Unpatchify ───────────────────────────────────────────────────────
    // [B, pT, pHs, pWs, pt*ph*pw*out_ch] → [B, out_ch, T, H, W]
    int out_channels = 16;
    // [B, pT, pHs, pWs, pt, ph, pw, out_ch]
    x_out = x_out.reshape({B, pT, pHs, pWs, pt, ph, pw, out_channels});
    // → [B, out_ch, pT, pt, pHs, ph, pWs, pw]
    x_out = x_out.permute({0, 7, 1, 4, 2, 5, 3, 6}).contiguous();
    // → [B, out_ch, T, H, W]
    x_out = x_out.reshape({B, out_channels, T, H, W});

    return x_out.to(orig_dtype);
}


// ===========================================================================
// STREAMING (KV-cached autoregressive) forward
//
// Mirrors FlashDreams' CosmosDiTNetwork.forward. All the heavy math (adaln,
// FFN, RMSNorm, LayerNorm) is reused from the DDPM path via the residual
// helpers above; the only new pieces are the two streaming sub-block helpers
// that (a) write new K/V into a ring-buffer cache and (b) skip K/V
// projection for cross-attention because those are pre-computed by
// FlashDreams' CrossAttention.initialize_cache.
// ===========================================================================

// Self-attention residual (streaming).
// Identical math to `self_attn_residual(causal=false)` except it ALSO writes
// the post-RoPE, post-RMSNorm K and post-V-proj V into external cache tensors
// at `[write_start : write_start + L_new)`, then reads the valid prefix
// `[0 : write_start + L_new)` as the attention context. This matches
// `FlashDreams.SelfAttention.forward` line-for-line.
static torch::Tensor self_attn_residual_streaming(
    const torch::Tensor& x,              // [B, L_new, D]
    const torch::Tensor& shift,          // [B, 1, D]
    const torch::Tensor& scale,
    const torch::Tensor& gate,
    const py::dict& weights,
    int block_idx,
    const torch::Tensor& rope_emb,       // [L_new, 1, 1, Dh] fp32
    int num_heads,
    torch::Tensor& k_cache,              // [B, cache_cap, H, Dh]  MUTATED IN PLACE
    torch::Tensor& v_cache,              // [B, cache_cap, H, Dh]  MUTATED IN PLACE
    int64_t write_start,
    cudaStream_t stream)
{
    int B = (int)x.size(0), L = (int)x.size(1), D = (int)x.size(2);
    int H = num_heads, Dh = D / H;
    const std::string pfx = block_prefix(block_idx) + "self_attn.";

    auto normed = layernorm_no_affine(x) * (1.f + scale) + shift;

    auto wq = get_w(weights, pfx + "q_proj.weight");
    auto wk = get_w(weights, pfx + "k_proj.weight");
    auto wv = get_w(weights, pfx + "v_proj.weight");
    auto wo = get_w(weights, pfx + "output_proj.weight");

    auto gq = get_w(weights, pfx + "q_norm.weight");
    auto gk = get_w(weights, pfx + "k_norm.weight");

    auto q = torch::matmul(normed, wq.t()).reshape({B, L, H, Dh});
    auto k = torch::matmul(normed, wk.t()).reshape({B, L, H, Dh});
    auto v = torch::matmul(normed, wv.t()).reshape({B, L, H, Dh});

    q = rms_norm(q, gq);
    k = rms_norm(k, gk);

    auto [q_rot, k_rot] = apply_cosmos_rope(q, k, rope_emb);

    // In-place write into the ring-buffer caches. The Python side is
    // responsible for rolling the buffers via BlockKVCache.before_update
    // prior to this call, so `write_start` is simply the tail of the valid
    // region to append to.
    const int64_t L_new = (int64_t)L;
    const int64_t read_end = write_start + L_new;
    TORCH_CHECK(read_end <= k_cache.size(1),
        "self-attn write_start+L_new exceeds cache capacity");

    k_cache.slice(/*dim=*/1, write_start, read_end).copy_(k_rot.to(k_cache.dtype()));
    v_cache.slice(/*dim=*/1, write_start, read_end).copy_(v    .to(v_cache.dtype()));

    auto cached_k = k_cache.slice(/*dim=*/1, 0, read_end);
    auto cached_v = v_cache.slice(/*dim=*/1, 0, read_end);

    // Non-causal FMHA against the historical cache + just-written tokens.
    auto attn_out = fmha(q_rot, cached_k, cached_v, /*causal=*/false, stream);

    return x + gate * torch::matmul(attn_out.reshape({B, L, D}), wo.t());
}

// Cross-attention residual (streaming) — text K/V are already pre-computed
// by FlashDreams.CrossAttention.initialize_cache (post-k_norm K, raw V), so we
// only need Q projection + q_norm + FMHA + output projection.
static torch::Tensor cross_attn_residual_streaming(
    const torch::Tensor& x,              // [B, L_new, D]
    const torch::Tensor& shift,          // [B, 1, D]
    const torch::Tensor& scale,
    const torch::Tensor& gate,
    const py::dict& weights,
    int block_idx,
    const torch::Tensor& k_cached,       // [B, Lc, H, Dh]  post-k_norm text K
    const torch::Tensor& v_cached,       // [B, Lc, H, Dh]  raw text V
    int num_heads,
    cudaStream_t stream)
{
    int B = (int)x.size(0), L = (int)x.size(1), D = (int)x.size(2);
    int H = num_heads, Dh = D / H;
    const std::string pfx = block_prefix(block_idx) + "cross_attn.";

    auto normed = layernorm_no_affine(x) * (1.f + scale) + shift;

    auto wq = get_w(weights, pfx + "q_proj.weight");
    auto wo = get_w(weights, pfx + "output_proj.weight");
    auto gq = get_w(weights, pfx + "q_norm.weight");

    auto q = torch::matmul(normed, wq.t()).reshape({B, L, H, Dh});
    q = rms_norm(q, gq);

    auto attn_out = fmha(q, k_cached, v_cached, /*causal=*/false, stream);
    return x + gate * torch::matmul(attn_out.reshape({B, L, D}), wo.t());
}

// ---------------------------------------------------------------------------
// Streaming main forward
// ---------------------------------------------------------------------------

torch::Tensor optimized_dit_forward(
    torch::Tensor x_new,
    torch::Tensor condition_mask_patched,
    torch::Tensor hdmap_patched,
    torch::Tensor timesteps,
    torch::Tensor rope_emb,
    std::vector<torch::Tensor> k_cross_caches,
    std::vector<torch::Tensor> v_cross_caches,
    std::vector<torch::Tensor> k_self_caches,
    std::vector<torch::Tensor> v_self_caches,
    int64_t self_attn_write_start,
    py::dict weights,
    py::dict config)
{
    TORCH_CHECK(x_new.is_cuda(),            "x_new must be a CUDA tensor");
    TORCH_CHECK(x_new.dim() == 5,           "x_new must be 5D [B, V, T, HW, D_in]");
    TORCH_CHECK(condition_mask_patched.is_cuda() && condition_mask_patched.dim() == 5,
                "condition_mask_patched must be CUDA 5D");

    auto orig_dtype = x_new.dtype();
    auto orig_scalar_type = x_new.scalar_type();
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    bool prof = cosmos_profile_enabled();
    enum {
        EV_START = 0,
        EV_AFTER_INPUT_EMBED,
        EV_AFTER_TEMB,
        EV_AFTER_ROPE_SCRATCH,
        EV_AFTER_BLOCKS,
        EV_AFTER_FINAL,
        EV_COUNT
    };
    cudaEvent_t ev[EV_COUNT];
    auto rec = [&](int idx) {
        if (prof) cudaEventRecord(ev[idx], stream);
    };
    if (prof) {
        for (int i = 0; i < EV_COUNT; ++i) cudaEventCreate(&ev[i]);
        rec(EV_START);
    }

    // ── Config ───────────────────────────────────────────────────────────────
    auto ci = [&](const char* k, int d)   { return config.contains(k) ? py::cast<int>(config[k]) : d; };
    auto cf = [&](const char* k, float d) { return config.contains(k) ? py::cast<float>(config[k]) : d; };
    auto cb = [&](const char* k, bool d)  { return config.contains(k) ? py::cast<bool>(config[k]) : d; };
    auto cs = [&](const char* k, const char* d) {
        return config.contains(k) ? py::cast<std::string>(config[k]) : std::string(d);
    };
    auto lower_ascii = [](std::string s) {
        for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        return s;
    };

    int   num_blocks     = ci("num_blocks",     28);
    int   num_heads      = ci("num_heads",      16);
    int   model_channels = ci("model_channels", 2048);
    float ts_scale       = cf("timestep_scale", 0.001f);
    std::string linear_backend_name = lower_ascii(cs("cosmos_linear_backend", "auto"));
    std::string attention_backend_name = lower_ascii(cs("cosmos_attention_backend", "cudnn_bf16"));
    std::string kv_cache_backend_name = lower_ascii(cs("cosmos_kv_cache_backend", "bf16"));
    bool quantized_prepared = cb("cosmos_quantized_prepared", false);
    bool quantized_prepared_strict = cb("cosmos_quantized_prepared_strict", quantized_prepared);
    // Shape: x_new is [B, V, T, HW, D_in]. For FlashDreams' single-view DiT V=1;
    // we flatten (V, T, HW) into a single L dimension for the transformer math.
    int B = (int)x_new.size(0);
    int V = (int)x_new.size(1);
    int T = (int)x_new.size(2);
    int HW = (int)x_new.size(3);
    TORCH_CHECK(V == 1, "optimized_dit_forward currently supports V==1 only (got V=",
                V, "); multi-view support is a future extension.");
    int L = V * T * HW;

    TORCH_CHECK((int)k_cross_caches.size() == num_blocks && (int)v_cross_caches.size() == num_blocks,
                "k_cross_caches and v_cross_caches must have length num_blocks=", num_blocks);
    TORCH_CHECK((int)k_self_caches.size()  == num_blocks && (int)v_self_caches.size()  == num_blocks,
                "k_self_caches and v_self_caches must have length num_blocks=", num_blocks);

    torch::Tensor hdmap_embed;
    if (config.contains("cosmos_hdmap_embed")) {
        hdmap_embed = py::cast<torch::Tensor>(config["cosmos_hdmap_embed"]);
        TORCH_CHECK(hdmap_embed.is_cuda(), "config['cosmos_hdmap_embed'] must be a CUDA tensor");
        TORCH_CHECK(hdmap_embed.device() == x_new.device(),
                    "config['cosmos_hdmap_embed'] must be on device ", x_new.device());
        TORCH_CHECK(hdmap_embed.scalar_type() == orig_scalar_type,
                    "config['cosmos_hdmap_embed'] has dtype ", hdmap_embed.scalar_type(),
                    ", expected ", orig_scalar_type);
        TORCH_CHECK(hdmap_embed.dim() == 3 &&
                    hdmap_embed.size(0) == B &&
                    hdmap_embed.size(1) == L &&
                    hdmap_embed.size(2) == model_channels,
                    "config['cosmos_hdmap_embed'] must have shape [", B, ", ", L, ", ",
                    model_channels, "]");
        TORCH_CHECK(hdmap_embed.is_contiguous(), "config['cosmos_hdmap_embed'] must be contiguous");
    }

    // ── 1. Concat condition_mask along last dim (both pre-patchified) ────────
    auto x_cat = torch::cat({x_new, condition_mask_patched.to(orig_dtype)}, -1);
    // [B, V, T, HW, D_in + D_cond] → [B, L, D_in + D_cond]
    auto x_flat = x_cat.reshape({B, L, -1}).contiguous();

    // ── 2. x_embedder (already fused: padding mask channel baked out) ────────
    auto w_xe = get_w(weights, "x_embedder.proj.1.weight");  // [D, D_in+D_cond]
    auto cur = torch::matmul(x_flat, w_xe.t());              // [B, L, D]

    // ── 3. Additional patch embedding (HD-map bbox control) ──────────────────
    if (hdmap_embed.defined()) {
        cur = cur + hdmap_embed;
    } else if (hdmap_patched.defined() && hdmap_patched.numel() > 0) {
        TORCH_CHECK(hdmap_patched.dim() == 5, "hdmap_patched must be 5D [B, V, T, HW, D_hdmap]");
        auto hdmap_flat = hdmap_patched.to(orig_dtype).reshape({B, L, -1}).contiguous();
        auto w_hd = get_w(weights, "additional_patch_embedding.proj.1.weight");
        cur = cur + torch::matmul(hdmap_flat, w_hd.t());
    }
    rec(EV_AFTER_INPUT_EMBED);

    // ── 4. Timestep embedding (single timestep per batch item) ───────────────
    // timesteps: [B]  (not [B, T] as in the DDPM path). Fixed scheduler-step
    // replay can provide these tensors to avoid per-forward ATen setup.
    torch::Tensor t_emb_BD;
    torch::Tensor t_emb_silu;
    torch::Tensor adaln_lora_BD;
    auto validate_timestep_cache = [&](const torch::Tensor& t, const char* name, int cols) {
        TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
        TORCH_CHECK(t.device() == x_new.device(), name, " must be on device ", x_new.device());
        TORCH_CHECK(t.scalar_type() == orig_scalar_type,
                    name, " has dtype ", t.scalar_type(), ", expected ", orig_scalar_type);
        TORCH_CHECK(t.dim() == 2 && (int)t.size(0) == B && (int)t.size(1) == cols,
                    name, " must have shape [", B, ", ", cols, "], got [",
                    t.dim() > 0 ? t.size(0) : 0, ", ", t.dim() > 1 ? t.size(1) : 0, "]");
        TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    };
    if (config.contains("cosmos_t_emb") ||
        config.contains("cosmos_t_emb_silu") ||
        config.contains("cosmos_adaln_lora")) {
        TORCH_CHECK(config.contains("cosmos_t_emb") &&
                    config.contains("cosmos_t_emb_silu") &&
                    config.contains("cosmos_adaln_lora"),
                    "precomputed timestep embedding requires config['cosmos_t_emb'], "
                    "config['cosmos_t_emb_silu'], and config['cosmos_adaln_lora']");
        t_emb_BD = py::cast<torch::Tensor>(config["cosmos_t_emb"]);
        t_emb_silu = py::cast<torch::Tensor>(config["cosmos_t_emb_silu"]);
        adaln_lora_BD = py::cast<torch::Tensor>(config["cosmos_adaln_lora"]);
        validate_timestep_cache(t_emb_BD, "config['cosmos_t_emb']", model_channels);
        validate_timestep_cache(t_emb_silu, "config['cosmos_t_emb_silu']", model_channels);
        validate_timestep_cache(adaln_lora_BD, "config['cosmos_adaln_lora']", 3 * model_channels);
    } else {
        auto ts = timesteps.to(torch::kFloat32) * ts_scale;             // [B]
        auto t_sin = sinusoidal_emb(ts, model_channels).to(orig_dtype); // [B, D]

        // TimestepEmbedding (use_adaln_lora=True): linear_1 has no bias.
        auto w_t1 = get_w(weights, "t_embedder.1.linear_1.weight");     // [D, D]
        auto t_h  = torch::silu(torch::matmul(t_sin, w_t1.t()));        // [B, D]

        auto w_t2 = get_w(weights, "t_embedder.1.linear_2.weight");     // [3D, D]
        adaln_lora_BD = torch::matmul(t_h, w_t2.t());                   // [B, 3D]

        // use_adaln_lora=True -> emb_B_D = raw sinusoidal (sample)
        t_emb_BD = t_sin;                                               // [B, D]

        // t_embedding_norm (RMSNorm)
        auto norm_gamma = get_w(weights, "t_embedding_norm.weight");
        t_emb_BD = rms_norm(t_emb_BD, norm_gamma);                      // [B, D]
        t_emb_silu = t_emb_BD.clone();                                  // [B, D]
        cudaError_t e = omnidreams_singleview::cosmos_silu_inplace<cutlass::bfloat16_t>(
            reinterpret_cast<cutlass::bfloat16_t*>(t_emb_silu.data_ptr<at::BFloat16>()),
            int64_t(B) * model_channels, stream);
        TORCH_CHECK(e == cudaSuccess, "cosmos_silu_inplace failed: ", cudaGetErrorString(e));
    }
    rec(EV_AFTER_TEMB);

    // ── 5. Transformer blocks (28): CUTLASS GEMM fusion path ──────────────────
    //
    // Replaces the previous ATen-heavy block loop (~14 ATen ops per sub-layer
    // × 3 sub-layers per block × 28 blocks = ~1180 ATen launches/forward) with
    // one C++ call per block. Each block call runs:
    //
    //   - 3x adaln-LoRA (SiLU pre-applied; 2 GEMMs each + add) via
    //     `cosmos_adaln_lora_split`
    //   - Self-attn residual: ln+modulate (1 kernel), 3 bf16 CUTLASS GEMMs
    //     (Q/K/V), per-head RMSNorm Q/K (2 kernels), pack+rotate-half RoPE
    //     (2 kernels), KV-cache append (2 small copy kernels), cuDNN FMHA,
    //     out GEMM, gated_residual (1 kernel)
    //   - Cross-attn residual (Q only; KV pre-cached): ln+modulate, Q GEMM,
    //     per-head RMSNorm Q, cuDNN FMHA, out GEMM, gated_residual
    //   - FFN residual: ln+modulate, GEMM1+GELU (fused epilogue), GEMM2,
    //     gated_residual
    //
    // Force bf16 throughout to match FlashDreams' CosmosDiTNetwork dtype.
    // The orchestrator + bf16 GEMM kernels handle this end-to-end; no fp16
    // down-cast.
    TORCH_CHECK(orig_dtype == torch::kBFloat16,
                "optimized_dit_forward requires bf16 inputs (matches "
                "FlashDreams' CosmosDiTNetwork dtype); got dtype=", orig_dtype);
    TORCH_CHECK(B == 1,
                "optimized_dit_forward Phase-1 CUTLASS path supports B=1 "
                "only (got B=", B, "); CFG-batched B=2 will fall back when "
                "FlashDreams enables it. The previous ATen path supported B>1.");

    int K = model_channels;
    int H = num_heads;
    TORCH_CHECK(K % H == 0,
                "model_channels (", K, ") must be divisible by num_heads (", H, ")");
    int D = K / H;
    auto opts = torch::TensorOptions().dtype(orig_scalar_type).device(x_new.device());

    bool fp8_kv_cache_enabled = false;
    std::vector<torch::Tensor> k_cross_fp8_caches;
    std::vector<torch::Tensor> v_cross_fp8_caches;
    std::vector<torch::Tensor> k_self_fp8_caches;
    std::vector<torch::Tensor> v_self_fp8_caches;
    std::vector<torch::Tensor> k_cross_fp8_bhmd_caches;
    std::vector<torch::Tensor> v_cross_fp8_bhmd_caches;
    std::vector<torch::Tensor> v_cross_fp8_bhdm_caches;
    std::vector<torch::Tensor> k_self_fp8_bhmd_caches;
    std::vector<torch::Tensor> v_self_fp8_bhmd_caches;
    std::vector<torch::Tensor> v_self_fp8_bhdm_caches;
    std::vector<torch::Tensor> k_cross_sage3_fp4_caches;
    std::vector<torch::Tensor> v_cross_sage3_fp4_caches;
    std::vector<torch::Tensor> k_cross_sage3_sf_caches;
    std::vector<torch::Tensor> v_cross_sage3_sf_caches;
    bool fp8_cross_tc_layout_cache_enabled = false;
    bool fp8_self_tc_layout_cache_enabled = false;
    const bool sage3_cross_fp4_enabled =
        config.contains("k_cross_sage3_fp4_caches") ||
        config.contains("v_cross_sage3_fp4_caches") ||
        config.contains("k_cross_sage3_sf_caches") ||
        config.contains("v_cross_sage3_sf_caches");
    auto config_tensor_vector = [&](const char* key) {
        TORCH_CHECK(config.contains(key),
                    "cosmos_kv_cache_backend=fp8 requires config['", key, "']");
        try {
            return py::cast<std::vector<torch::Tensor>>(config[key]);
        } catch (const py::cast_error&) {
            TORCH_CHECK(false, "config['", key, "'] must be a list of CUDA tensors");
            return std::vector<torch::Tensor>{};
        }
    };
    auto optional_config_tensor_vector = [&](const char* key) {
        if (!config.contains(key)) {
            return std::vector<torch::Tensor>{};
        }
        try {
            return py::cast<std::vector<torch::Tensor>>(config[key]);
        } catch (const py::cast_error&) {
            TORCH_CHECK(false, "config['", key, "'] must be a list of CUDA tensors");
            return std::vector<torch::Tensor>{};
        }
    };
    if (kv_cache_backend_name == "bf16" || kv_cache_backend_name == "none") {
        fp8_kv_cache_enabled = false;
    } else if (kv_cache_backend_name == "fp8" || kv_cache_backend_name == "shadow_fp8") {
        fp8_kv_cache_enabled = true;
        if (!sage3_cross_fp4_enabled) {
            k_cross_fp8_caches = config_tensor_vector("k_cross_fp8_caches");
            v_cross_fp8_caches = config_tensor_vector("v_cross_fp8_caches");
        }
        k_self_fp8_caches = config_tensor_vector("k_self_fp8_caches");
        v_self_fp8_caches = config_tensor_vector("v_self_fp8_caches");
        TORCH_CHECK((sage3_cross_fp4_enabled ||
                     ((int)k_cross_fp8_caches.size() == num_blocks &&
                      (int)v_cross_fp8_caches.size() == num_blocks)) &&
                    (int)k_self_fp8_caches.size() == num_blocks &&
                    (int)v_self_fp8_caches.size() == num_blocks,
                    "FP8 shadow KV cache lists must all have length num_blocks=", num_blocks);
        k_cross_fp8_bhmd_caches = optional_config_tensor_vector("k_cross_fp8_bhmd_caches");
        v_cross_fp8_bhmd_caches = optional_config_tensor_vector("v_cross_fp8_bhmd_caches");
        v_cross_fp8_bhdm_caches = optional_config_tensor_vector("v_cross_fp8_bhdm_caches");
        k_self_fp8_bhmd_caches = optional_config_tensor_vector("k_self_fp8_bhmd_caches");
        v_self_fp8_bhmd_caches = optional_config_tensor_vector("v_self_fp8_bhmd_caches");
        v_self_fp8_bhdm_caches = optional_config_tensor_vector("v_self_fp8_bhdm_caches");
        fp8_cross_tc_layout_cache_enabled =
            !k_cross_fp8_bhmd_caches.empty() || !v_cross_fp8_bhmd_caches.empty() ||
            !v_cross_fp8_bhdm_caches.empty();
        fp8_self_tc_layout_cache_enabled =
            !k_self_fp8_bhmd_caches.empty() || !v_self_fp8_bhmd_caches.empty() ||
            !v_self_fp8_bhdm_caches.empty();
        if (fp8_cross_tc_layout_cache_enabled) {
            TORCH_CHECK((int)k_cross_fp8_bhmd_caches.size() == num_blocks &&
                        (int)v_cross_fp8_bhmd_caches.size() == num_blocks,
                        "FP8 cuDNN-layout cross KV cache lists k/v k_cross_fp8_bhmd_caches and "
                        "v_cross_fp8_bhmd_caches must both have length num_blocks=",
                        num_blocks);
            TORCH_CHECK(v_cross_fp8_bhdm_caches.empty() || (int)v_cross_fp8_bhdm_caches.size() == num_blocks,
                        "v_cross_fp8_bhdm_caches must be omitted or have length num_blocks=", num_blocks);
        }
        if (fp8_self_tc_layout_cache_enabled) {
            TORCH_CHECK((int)k_self_fp8_bhmd_caches.size() == num_blocks &&
                        (int)v_self_fp8_bhmd_caches.size() == num_blocks,
                        "FP8 cuDNN-layout self KV cache lists k/v k_self_fp8_bhmd_caches and "
                        "v_self_fp8_bhmd_caches must both have length num_blocks=",
                        num_blocks);
            TORCH_CHECK(v_self_fp8_bhdm_caches.empty() || (int)v_self_fp8_bhdm_caches.size() == num_blocks,
                        "v_self_fp8_bhdm_caches must be omitted or have length num_blocks=", num_blocks);
        }
    } else {
        TORCH_CHECK(false,
                    "unsupported cosmos_kv_cache_backend '", kv_cache_backend_name,
                    "'. Expected one of: bf16, fp8");
    }
    if (sage3_cross_fp4_enabled) {
        TORCH_CHECK(config.contains("k_cross_sage3_fp4_caches") &&
                    config.contains("v_cross_sage3_fp4_caches") &&
                    config.contains("k_cross_sage3_sf_caches") &&
                    config.contains("v_cross_sage3_sf_caches"),
                    "Sage3 cross-attention FP4 caches require k/v fp4 and k/v scale cache lists");
        k_cross_sage3_fp4_caches = config_tensor_vector("k_cross_sage3_fp4_caches");
        v_cross_sage3_fp4_caches = config_tensor_vector("v_cross_sage3_fp4_caches");
        k_cross_sage3_sf_caches = config_tensor_vector("k_cross_sage3_sf_caches");
        v_cross_sage3_sf_caches = config_tensor_vector("v_cross_sage3_sf_caches");
        TORCH_CHECK((int)k_cross_sage3_fp4_caches.size() == num_blocks &&
                    (int)v_cross_sage3_fp4_caches.size() == num_blocks &&
                    (int)k_cross_sage3_sf_caches.size() == num_blocks &&
                    (int)v_cross_sage3_sf_caches.size() == num_blocks,
                    "Sage3 cross FP4 cache lists must all have length num_blocks=", num_blocks);
    }

    // FFN inner dim is fixed by the checkpoint (4 * K for cosmos production);
    // we can read it off the actual ffn weight tensor rather than relying on
    // a config knob (avoids a config mismatch hazard).
    auto ffn_w1_probe = get_w(weights, "blocks.0.mlp.layer1.weight");           // [FF, K]
    int FF = (int)ffn_w1_probe.size(0);
    auto adaln_down_probe = get_w(weights, "blocks.0.adaln_modulation_self_attn.1.weight"); // [lora_dim, K]
    int lora_dim = (int)adaln_down_probe.size(0);

    const std::array<std::string, 8> block_linear_rel_keys = {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.output_proj.weight",
        "cross_attn.q_proj.weight",
        "cross_attn.output_proj.weight",
        "mlp.layer1.weight",
        "mlp.layer2.weight",
    };
    bool any_block_linear_fp8 = false;
    bool any_block_linear_bf16 = false;
    for (int i = 0; i < num_blocks; ++i) {
        const std::string p = block_prefix(i);
        const bool has_fused_qkv =
            weights.contains(py::str(p + "self_attn.qkv_proj.weight")) &&
            get_w(weights, p + "self_attn.qkv_proj.weight").scalar_type() == at::kByte;
        for (const std::string& rel_key : block_linear_rel_keys) {
            if (has_fused_qkv &&
                (rel_key == "self_attn.q_proj.weight" ||
                 rel_key == "self_attn.k_proj.weight" ||
                 rel_key == "self_attn.v_proj.weight")) {
                any_block_linear_fp8 = true;
                continue;
            }
            auto t = get_w(weights, p + rel_key);
            if (t.scalar_type() == at::kByte) {
                any_block_linear_fp8 = true;
            } else if (t.scalar_type() == at::kBFloat16) {
                any_block_linear_bf16 = true;
            } else {
                TORCH_CHECK(false, p + rel_key,
                            " must be torch.bfloat16 or torch.uint8 raw E4M3 bytes; got ",
                            t.scalar_type());
            }
        }
    }
    omnidreams_singleview::CosmosLinearBackend linear_backend = omnidreams_singleview::CosmosLinearBackend::BF16;
    if (linear_backend_name == "auto") {
        linear_backend = any_block_linear_fp8
            ? (any_block_linear_bf16 ? omnidreams_singleview::CosmosLinearBackend::MIXED
                                     : omnidreams_singleview::CosmosLinearBackend::FP8)
            : omnidreams_singleview::CosmosLinearBackend::BF16;
    } else if (linear_backend_name == "bf16" || linear_backend_name == "cudnn") {
        linear_backend = omnidreams_singleview::CosmosLinearBackend::BF16;
    } else if (linear_backend_name == "fp8") {
        linear_backend = omnidreams_singleview::CosmosLinearBackend::FP8;
    } else if (linear_backend_name == "mixed") {
        linear_backend = omnidreams_singleview::CosmosLinearBackend::MIXED;
    } else {
        TORCH_CHECK(false,
                    "unsupported cosmos_linear_backend '", linear_backend_name,
                    "'. Expected one of: auto, bf16, fp8, mixed");
    }

    omnidreams_singleview::CosmosAttentionBackend attention_backend = omnidreams_singleview::CosmosAttentionBackend::CUDNN_BF16;
    if (attention_backend_name == "cudnn_bf16" || attention_backend_name == "bf16") {
        attention_backend = omnidreams_singleview::CosmosAttentionBackend::CUDNN_BF16;
    } else if (attention_backend_name == "fp8_dense_ref") {
        attention_backend = omnidreams_singleview::CosmosAttentionBackend::FP8_DENSE_REF;
    } else if (attention_backend_name == "fp8_cudnn") {
        attention_backend = omnidreams_singleview::CosmosAttentionBackend::FP8_CUDNN;
    } else if (attention_backend_name == "sage3") {
        attention_backend = omnidreams_singleview::CosmosAttentionBackend::SAGE3;
    } else if (attention_backend_name == "sage3_fp8") {
        attention_backend = omnidreams_singleview::CosmosAttentionBackend::SAGE3_FP8;
    } else {
        TORCH_CHECK(false,
                    "unsupported cosmos_attention_backend '", attention_backend_name,
                    "'. Expected one of: cudnn_bf16, bf16, fp8_dense_ref, fp8_cudnn, sage3, sage3_fp8");
    }
    bool schedule_uses_sage3_fp8 =
        attention_backend == omnidreams_singleview::CosmosAttentionBackend::SAGE3_FP8;
    TORCH_CHECK(!schedule_uses_sage3_fp8 || fp8_kv_cache_enabled,
                "Sage3 FP8 attention requires cosmos_kv_cache_backend='fp8'");
    TORCH_CHECK(!schedule_uses_sage3_fp8 || sage3_cross_fp4_enabled,
                "Sage3 FP8 attention requires Sage3 FP4 cross-attention caches");
    // ── 5a. One-shot setup outside the per-block loop ─────────────────────────
    //
    // (i) RoPE cos/sin: callers may precompute these for fixed-shape replay.
    //      rope_emb is fp32 [L_new, 1, 1, D]; the block path consumes bf16
    //      [L_new, D] cos/sin.
    torch::Tensor rope_cos;
    torch::Tensor rope_sin;
    auto validate_rope_cache = [&](const torch::Tensor& t, const char* name) {
        TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
        TORCH_CHECK(t.device() == x_new.device(), name, " must be on device ", x_new.device());
        TORCH_CHECK(t.scalar_type() == orig_scalar_type,
                    name, " has dtype ", t.scalar_type(), ", expected ", orig_scalar_type);
        TORCH_CHECK(t.dim() == 2 && (int)t.size(0) == L && (int)t.size(1) == D,
                    name, " must have shape [", L, ", ", D, "], got [",
                    t.dim() > 0 ? t.size(0) : 0, ", ", t.dim() > 1 ? t.size(1) : 0, "]");
        TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    };
    if (config.contains("cosmos_rope_cos") || config.contains("cosmos_rope_sin")) {
        TORCH_CHECK(config.contains("cosmos_rope_cos") && config.contains("cosmos_rope_sin"),
                    "precomputed RoPE requires both config['cosmos_rope_cos'] and config['cosmos_rope_sin']");
        rope_cos = py::cast<torch::Tensor>(config["cosmos_rope_cos"]);
        rope_sin = py::cast<torch::Tensor>(config["cosmos_rope_sin"]);
        validate_rope_cache(rope_cos, "config['cosmos_rope_cos']");
        validate_rope_cache(rope_sin, "config['cosmos_rope_sin']");
    } else {
        auto rope_view = rope_emb.permute({1, 0, 2, 3}).reshape({L, model_channels / num_heads});
        rope_cos = torch::cos(rope_view).to(orig_dtype).contiguous();           // [L, D] bf16
        rope_sin = torch::sin(rope_view).to(orig_dtype).contiguous();           // [L, D] bf16
    }

    // (iii) Block-shared scratch buffers. These are sized once and reused
    //       across all 28 blocks. The PyTorch caching allocator amortises
    //       the cost across forwards of the same shape.
    int M = L;  // per-call query length (B==1)
    int sage3_q_padded = ((M + 127) / 128) * 128;
    int max_attn_tokens = M;
    for (int i = 0; i < num_blocks; ++i) {
        max_attn_tokens = std::max(max_attn_tokens, (int)k_cross_caches[i].size(1));
        max_attn_tokens = std::max(max_attn_tokens, (int)k_self_caches[i].size(1));
    }
    int linear_scratch_features = std::max(std::max(K, FF), 3 * K);
    py::dict workspace_dict;
    bool use_external_workspace = false;
    if (config.contains("cosmos_workspace")) {
        try {
            workspace_dict = py::cast<py::dict>(config["cosmos_workspace"]);
            use_external_workspace = true;
        } catch (const py::cast_error&) {
            TORCH_CHECK(false, "config['cosmos_workspace'] must be a dict of preallocated CUDA tensors");
        }
    }
    const bool attn_tc_scale_is_ones =
        config.contains("cosmos_attn_tc_scale_is_ones")
            ? py::cast<bool>(config["cosmos_attn_tc_scale_is_ones"])
            : false;
    TORCH_CHECK(!attn_tc_scale_is_ones || use_external_workspace,
                "cosmos_attn_tc_scale_is_ones=true requires config['cosmos_workspace']");
    bool write_bf16_self_kv_cache =
        config.contains("cosmos_write_bf16_kv_cache")
            ? py::cast<bool>(config["cosmos_write_bf16_kv_cache"])
            : true;
    TORCH_CHECK(write_bf16_self_kv_cache || fp8_kv_cache_enabled,
                "cosmos_write_bf16_kv_cache=false requires cosmos_kv_cache_backend='fp8'");
    auto workspace_tensor = [&](const char* name,
                                std::vector<int64_t> shape,
                                at::ScalarType dtype) -> torch::Tensor {
        if (!use_external_workspace) {
            return torch::empty(c10::IntArrayRef(shape), torch::TensorOptions().dtype(dtype).device(x_new.device()));
        }
        TORCH_CHECK(workspace_dict.contains(name),
                    "config['cosmos_workspace'] is missing scratch tensor '", name, "'");
        auto t = py::cast<torch::Tensor>(workspace_dict[name]);
        TORCH_CHECK(t.is_cuda(), "cosmos_workspace['", name, "'] must be CUDA");
        TORCH_CHECK(t.device() == x_new.device(),
                    "cosmos_workspace['", name, "'] must be on device ", x_new.device());
        TORCH_CHECK(t.scalar_type() == dtype,
                    "cosmos_workspace['", name, "'] has dtype ", t.scalar_type(),
                    ", expected ", dtype);
        TORCH_CHECK(t.dim() == static_cast<int64_t>(shape.size()),
                    "cosmos_workspace['", name, "'] has dim ", t.dim(),
                    ", expected ", shape.size());
        for (size_t dim = 0; dim < shape.size(); ++dim) {
            TORCH_CHECK(t.size(static_cast<int64_t>(dim)) == shape[dim],
                        "cosmos_workspace['", name, "'] shape mismatch at dim ", dim,
                        ": got ", t.size(static_cast<int64_t>(dim)),
                        ", expected ", shape[dim]);
        }
        TORCH_CHECK(t.is_contiguous(), "cosmos_workspace['", name, "'] must be contiguous");
        return t;
    };
    auto workspace_tensor_singleton_or_shape = [&](const char* name,
                                                   std::vector<int64_t> shape,
                                                   at::ScalarType dtype,
                                                   bool allow_singleton) -> torch::Tensor {
        if (!use_external_workspace) {
            if (allow_singleton) {
                return torch::empty({1}, torch::TensorOptions().dtype(dtype).device(x_new.device()));
            }
            return workspace_tensor(name, shape, dtype);
        }
        if (!allow_singleton) {
            return workspace_tensor(name, shape, dtype);
        }
        TORCH_CHECK(workspace_dict.contains(name),
                    "config['cosmos_workspace'] is missing scratch tensor '", name, "'");
        auto t = py::cast<torch::Tensor>(workspace_dict[name]);
        TORCH_CHECK(t.is_cuda(), "cosmos_workspace['", name, "'] must be CUDA");
        TORCH_CHECK(t.device() == x_new.device(),
                    "cosmos_workspace['", name, "'] must be on device ", x_new.device());
        TORCH_CHECK(t.scalar_type() == dtype,
                    "cosmos_workspace['", name, "'] has dtype ", t.scalar_type(),
                    ", expected ", dtype);
        bool exact = t.dim() == static_cast<int64_t>(shape.size());
        if (exact) {
            for (size_t dim = 0; dim < shape.size(); ++dim) {
                exact = exact && t.size(static_cast<int64_t>(dim)) == shape[dim];
            }
        }
        bool singleton = t.numel() == 1;
        TORCH_CHECK(exact || singleton,
                    "cosmos_workspace['", name, "'] must have shape [",
                    shape.size() > 0 ? shape[0] : 0,
                    shape.size() > 1 ? ", ..." : "",
                    "] or be a singleton placeholder for this backend");
        TORCH_CHECK(t.is_contiguous(), "cosmos_workspace['", name, "'] must be contiguous");
        return t;
    };

    torch::Tensor qkv_row;
    torch::Tensor q_row;
    torch::Tensor k_row;
    torch::Tensor v_row;
    torch::Tensor q_bmhk;
    torch::Tensor k_bmhk;
    torch::Tensor o_bmhk;
    torch::Tensor normed;
    torch::Tensor ffn_intermediate;
    torch::Tensor lora_hidden_sa;
    torch::Tensor lora_hidden_ca;
    torch::Tensor lora_hidden_mlp;
    torch::Tensor mods_sa;
    torch::Tensor mods_ca;
    torch::Tensor mods_mlp;
    // AdaLN-LoRA pre-stack scratch (Phase 1 of the AdaLN-LoRA pre-stack +
    // strided-batched up GEMM optimization). Optional in workspace; required
    // only when OMNIDREAMS_DIT_ADALN_PRECOMPUTE is enabled (default on).
    torch::Tensor lora_hidden_all;
    torch::Tensor mods_all;
    torch::Tensor fl_hidden_scratch;
    torch::Tensor fl_mods_scratch;
    if (!use_external_workspace) {
        g_cosmos_streaming_scratch.ensure(
            B, M, K, H, D, FF, lora_dim, x_new.device(), x_new.scalar_type(), opts);
        auto& scratch = g_cosmos_streaming_scratch;
        qkv_row = scratch.qkv_row;
        q_row = scratch.q_row;
        k_row = scratch.k_row;
        v_row = scratch.v_row;
        q_bmhk = scratch.q_bmhk;
        k_bmhk = scratch.k_bmhk;
        o_bmhk = scratch.o_bmhk;
        normed = scratch.normed;
        ffn_intermediate = scratch.ffn_intermediate;
        lora_hidden_sa = scratch.lora_hidden_sa;
        lora_hidden_ca = scratch.lora_hidden_ca;
        lora_hidden_mlp = scratch.lora_hidden_mlp;
        mods_sa = scratch.mods_sa;
        mods_ca = scratch.mods_ca;
        mods_mlp = scratch.mods_mlp;
        fl_hidden_scratch = scratch.fl_hidden;
        fl_mods_scratch = scratch.fl_mods;
    } else {
        qkv_row = workspace_tensor("qkv_row", {M, 3 * K}, orig_scalar_type);
        q_row = workspace_tensor("q_row", {M, K}, orig_scalar_type);
        k_row = workspace_tensor("k_row", {M, K}, orig_scalar_type);
        v_row = workspace_tensor("v_row", {M, K}, orig_scalar_type);
        q_bmhk = workspace_tensor("q_bmhk", {M, H, D}, orig_scalar_type);
        k_bmhk = workspace_tensor("k_bmhk", {M, H, D}, orig_scalar_type);
        o_bmhk = workspace_tensor("o_bmhk", {M, H, D}, orig_scalar_type);
        normed = workspace_tensor("normed", {M, K}, orig_scalar_type);
        ffn_intermediate = workspace_tensor("ffn_intermediate", {M, FF}, orig_scalar_type);
        lora_hidden_sa = workspace_tensor("lora_hidden_sa", {B, lora_dim}, orig_scalar_type);
        lora_hidden_ca = workspace_tensor("lora_hidden_ca", {B, lora_dim}, orig_scalar_type);
        lora_hidden_mlp = workspace_tensor("lora_hidden_mlp", {B, lora_dim}, orig_scalar_type);
        mods_sa = workspace_tensor("mods_sa", {B, 3 * K}, orig_scalar_type);
        mods_ca = workspace_tensor("mods_ca", {B, 3 * K}, orig_scalar_type);
        mods_mlp = workspace_tensor("mods_mlp", {B, 3 * K}, orig_scalar_type);
        // Optional AdaLN-LoRA pre-stack scratch buffers. Validate only if
        // present in the workspace; the bridge falls back to per-block
        // cosmos_adaln_lora_split when these are missing.
        if (workspace_dict.contains("lora_hidden_all")) {
            lora_hidden_all = workspace_tensor(
                "lora_hidden_all", {num_blocks * 3, B, lora_dim}, orig_scalar_type);
        }
        if (workspace_dict.contains("mods_all")) {
            mods_all = workspace_tensor(
                "mods_all", {num_blocks * 3, B, 3 * K}, orig_scalar_type);
        }
    }
    // v_bmhk aliases v_row inside the orchestrator (V has no transformation,
    // and [M, K] / [M, H, D] are byte-equivalent for K = H*D).
    auto linear_fp8_scratch = workspace_tensor("linear_fp8_scratch", {M, linear_scratch_features}, at::kByte);
    auto linear_half_scratch = workspace_tensor("linear_half_scratch", {M, linear_scratch_features}, at::kHalf);
    auto attn_q_fp8 = workspace_tensor("attn_q_fp8", {B, M, H, D}, at::kByte);
    auto attn_k_fp8 = workspace_tensor("attn_k_fp8", {B, max_attn_tokens, H, D}, at::kByte);
    auto attn_v_fp8 = workspace_tensor("attn_v_fp8", {B, max_attn_tokens, H, D}, at::kByte);
    auto attn_q_bhmd_fp8 = workspace_tensor("attn_q_bhmd_fp8", {B, H, M, D}, at::kByte);
    auto attn_k_bhmd_fp8 = workspace_tensor("attn_k_bhmd_fp8", {B, H, max_attn_tokens, D}, at::kByte);
    auto attn_v_bhmd_fp8 = workspace_tensor("attn_v_bhmd_fp8", {B, H, max_attn_tokens, D}, at::kByte);
    auto attn_v_bhdm_fp8 = workspace_tensor("attn_v_bhdm_fp8", {B, H, D, max_attn_tokens}, at::kByte);
    const bool needs_sage3_fp8_attention_scratch = schedule_uses_sage3_fp8;
    torch::Tensor attn_q_sage3_fp4;
    torch::Tensor attn_q_sage3_sf;
    if (needs_sage3_fp8_attention_scratch) {
        attn_q_sage3_fp4 = workspace_tensor("attn_q_sage3_fp4", {B, H, sage3_q_padded, D / 2}, at::kByte);
        attn_q_sage3_sf = workspace_tensor(
            "attn_q_sage3_sf", {B, H, sage3_q_padded, D / 16}, at::ScalarType::Float8_e4m3fn);
    }
    const bool needs_dense_attention_scratch =
        attention_backend == omnidreams_singleview::CosmosAttentionBackend::FP8_DENSE_REF;
    auto attn_scores_half = workspace_tensor_singleton_or_shape(
        "attn_scores_half", {B * H, M, max_attn_tokens}, at::kHalf, !needs_dense_attention_scratch);
    auto attn_scores_bf16 = workspace_tensor_singleton_or_shape(
        "attn_scores_bf16", {B * H, M, max_attn_tokens}, at::kBFloat16, !needs_dense_attention_scratch);
    auto attn_score_c_bf16 = workspace_tensor_singleton_or_shape(
        "attn_score_c_bf16", {B * H, M, max_attn_tokens}, at::kBFloat16, !needs_dense_attention_scratch);
    auto attn_probs_fp8 = workspace_tensor_singleton_or_shape(
        "attn_probs_fp8", {B * H, M, max_attn_tokens}, at::kByte, !needs_dense_attention_scratch);
    auto attn_o_bhmd_half = workspace_tensor("attn_o_bhmd_half", {B, H, M, D}, at::kHalf);
    auto attn_o_bhmd_bf16 = workspace_tensor("attn_o_bhmd_bf16", {B, H, M, D}, at::kBFloat16);
    auto attn_o_c_bf16 = workspace_tensor("attn_o_c_bf16", {B, H, M, D}, at::kBFloat16);
    auto attn_o_half = workspace_tensor("attn_o_half", {B, M, H, D}, at::kHalf);
    const int64_t attn_tc_scale_elems = 12;
    auto attn_tc_scale = workspace_tensor("attn_tc_scale", {attn_tc_scale_elems}, at::kFloat);

    omnidreams_singleview::CosmosBlockBuffers buf{};
    buf.qkv_row           = reinterpret_cast<cutlass::bfloat16_t*>(qkv_row.data_ptr<at::BFloat16>());
    buf.q_row             = reinterpret_cast<cutlass::bfloat16_t*>(q_row.data_ptr<at::BFloat16>());
    buf.k_row             = reinterpret_cast<cutlass::bfloat16_t*>(k_row.data_ptr<at::BFloat16>());
    buf.v_row             = reinterpret_cast<cutlass::bfloat16_t*>(v_row.data_ptr<at::BFloat16>());
    buf.q_bmhk            = reinterpret_cast<cutlass::bfloat16_t*>(q_bmhk.data_ptr<at::BFloat16>());
    buf.k_bmhk            = reinterpret_cast<cutlass::bfloat16_t*>(k_bmhk.data_ptr<at::BFloat16>());
    buf.v_bmhk            = nullptr;  // aliased to v_row inside orchestrator
    buf.o_bmhk            = reinterpret_cast<cutlass::bfloat16_t*>(o_bmhk.data_ptr<at::BFloat16>());
    buf.attn_out_row      = nullptr;  // aliased to o_bmhk inside orchestrator
    buf.normed            = reinterpret_cast<cutlass::bfloat16_t*>(normed.data_ptr<at::BFloat16>());
    buf.ffn_intermediate  = reinterpret_cast<cutlass::bfloat16_t*>(ffn_intermediate.data_ptr<at::BFloat16>());
    buf.lora_hidden_sa    = reinterpret_cast<cutlass::bfloat16_t*>(lora_hidden_sa.data_ptr<at::BFloat16>());
    buf.lora_hidden_ca    = reinterpret_cast<cutlass::bfloat16_t*>(lora_hidden_ca.data_ptr<at::BFloat16>());
    buf.lora_hidden_mlp   = reinterpret_cast<cutlass::bfloat16_t*>(lora_hidden_mlp.data_ptr<at::BFloat16>());
    buf.mods_sa           = reinterpret_cast<cutlass::bfloat16_t*>(mods_sa.data_ptr<at::BFloat16>());
    buf.mods_ca           = reinterpret_cast<cutlass::bfloat16_t*>(mods_ca.data_ptr<at::BFloat16>());
    buf.mods_mlp          = reinterpret_cast<cutlass::bfloat16_t*>(mods_mlp.data_ptr<at::BFloat16>());
    buf.linear_fp8_scratch = reinterpret_cast<cutlass::float_e4m3_t*>(linear_fp8_scratch.data_ptr<uint8_t>());
    buf.linear_half_scratch = reinterpret_cast<cutlass::half_t*>(linear_half_scratch.data_ptr<at::Half>());
    buf.attn_q_fp8 = reinterpret_cast<cutlass::float_e4m3_t*>(attn_q_fp8.data_ptr<uint8_t>());
    buf.attn_k_fp8 = reinterpret_cast<cutlass::float_e4m3_t*>(attn_k_fp8.data_ptr<uint8_t>());
    buf.attn_v_fp8 = reinterpret_cast<cutlass::float_e4m3_t*>(attn_v_fp8.data_ptr<uint8_t>());
    buf.attn_q_bhmd_fp8 = reinterpret_cast<cutlass::float_e4m3_t*>(attn_q_bhmd_fp8.data_ptr<uint8_t>());
    buf.attn_k_bhmd_fp8 = reinterpret_cast<cutlass::float_e4m3_t*>(attn_k_bhmd_fp8.data_ptr<uint8_t>());
    buf.attn_v_bhmd_fp8 = reinterpret_cast<cutlass::float_e4m3_t*>(attn_v_bhmd_fp8.data_ptr<uint8_t>());
    buf.attn_v_bhdm_fp8 = reinterpret_cast<cutlass::float_e4m3_t*>(attn_v_bhdm_fp8.data_ptr<uint8_t>());
    buf.attn_q_sage3_fp4 = needs_sage3_fp8_attention_scratch
        ? attn_q_sage3_fp4.data_ptr<uint8_t>()
        : nullptr;
    buf.attn_q_sage3_sf = needs_sage3_fp8_attention_scratch
        ? reinterpret_cast<cutlass::float_e4m3_t*>(attn_q_sage3_sf.data_ptr())
        : nullptr;
    buf.attn_scores_half = reinterpret_cast<cutlass::half_t*>(attn_scores_half.data_ptr<at::Half>());
    buf.attn_scores_bf16 = reinterpret_cast<cutlass::bfloat16_t*>(attn_scores_bf16.data_ptr<at::BFloat16>());
    buf.attn_score_c_bf16 = reinterpret_cast<cutlass::bfloat16_t*>(attn_score_c_bf16.data_ptr<at::BFloat16>());
    buf.attn_probs_fp8 = reinterpret_cast<cutlass::float_e4m3_t*>(attn_probs_fp8.data_ptr<uint8_t>());
    buf.attn_o_bhmd_half = reinterpret_cast<cutlass::half_t*>(attn_o_bhmd_half.data_ptr<at::Half>());
    buf.attn_o_bhmd_bf16 = reinterpret_cast<cutlass::bfloat16_t*>(attn_o_bhmd_bf16.data_ptr<at::BFloat16>());
    buf.attn_o_c_bf16 = reinterpret_cast<cutlass::bfloat16_t*>(attn_o_c_bf16.data_ptr<at::BFloat16>());
    buf.attn_o_half = reinterpret_cast<cutlass::half_t*>(attn_o_half.data_ptr<at::Half>());
    buf.attn_tc_scale = attn_tc_scale.data_ptr<float>();
    buf.attn_tc_scale_elems = attn_tc_scale_elems;
    buf.attn_tc_scale_is_ones = attn_tc_scale_is_ones;
    rec(EV_AFTER_ROPE_SCRATCH);

    // ── 5a. AdaLN-LoRA pre-stack + global down + batched up GEMMs ──────────
    //
    // Replaces 168 launch-bound CUTLASS BF16 GEMVs (84 down + 84 up) per
    // forward with one wide BF16 GEMM (down) + one strided-batched BF16
    // GEMM (up). All 28 blocks share the same SiLU(t_emb) and adaln_lora_3D,
    // so the whole AdaLN-LoRA stage collapses to two launches at the top
    // of the forward.
    //
    // Gated by OMNIDREAMS_DIT_ADALN_PRECOMPUTE (default ON). Falls back to the
    // per-block cosmos_adaln_lora_split path when:
    //   - workspace doesn't include lora_hidden_all / mods_all
    //   - first call happens during graph capture (cudaMalloc unsafe; the
    //     persistent stacks are allocated in eager warmup before capture)
    //   - cuBLASLt doesn't support the strided-batched bias broadcast we
    //     request (heuristic returns no algo): we set the flag false and
    //     the per-block path runs in this forward.
    //
    // The stacked weight buffers (`g_adaln_down_stack`, `g_adaln_up_stack`)
    // are persistent across forwards. They are (re)populated only when
    // (num_blocks, K, lora_dim, weight_pointer_fingerprint) changes — i.e.
    // exactly once per process for typical inference.
    static cutlass::bfloat16_t* g_adaln_down_stack = nullptr;
    static cutlass::bfloat16_t* g_adaln_up_stack = nullptr;
    static int g_adaln_num_blocks = 0;
    static int g_adaln_K = 0;
    static int g_adaln_lora_dim = 0;
    static const cutlass::bfloat16_t* g_adaln_fingerprint_down = nullptr;

    auto adaln_precompute_env_enabled = []() {
        const char* v = std::getenv("OMNIDREAMS_DIT_ADALN_PRECOMPUTE");
        if (!v || !v[0]) return true;
        return v[0] != '0' && v[0] != 'f' && v[0] != 'F' && v[0] != 'n' && v[0] != 'N';
    };
    bool adaln_precompute_enabled =
        adaln_precompute_env_enabled() &&
        lora_hidden_all.defined() &&
        mods_all.defined();

    if (adaln_precompute_enabled) {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        cudaError_t cap_err = cudaStreamIsCapturing(stream, &capture_status);
        const bool capturing = (cap_err == cudaSuccess) &&
                               (capture_status != cudaStreamCaptureStatusNone);

        const cutlass::bfloat16_t* fingerprint = reinterpret_cast<const cutlass::bfloat16_t*>(
            get_w(weights, "blocks.0.adaln_modulation_self_attn.1.weight").data_ptr<at::BFloat16>());
        const bool shape_changed =
            (g_adaln_num_blocks != num_blocks) ||
            (g_adaln_K != K) ||
            (g_adaln_lora_dim != lora_dim) ||
            (g_adaln_fingerprint_down != fingerprint);

        if (shape_changed) {
            if (capturing) {
                adaln_precompute_enabled = false;
            } else {
                if (g_adaln_down_stack || g_adaln_up_stack) {
                    cudaStreamSynchronize(stream);
                }
                size_t down_bytes = size_t(num_blocks) * 3 * lora_dim * K * sizeof(cutlass::bfloat16_t);
                size_t up_bytes   = size_t(num_blocks) * 3 * 3 * K * lora_dim * sizeof(cutlass::bfloat16_t);
                if (g_adaln_down_stack) cudaFree(g_adaln_down_stack);
                if (g_adaln_up_stack)   cudaFree(g_adaln_up_stack);
                g_adaln_down_stack = nullptr;
                g_adaln_up_stack = nullptr;
                cudaError_t a1 = cudaMalloc(reinterpret_cast<void**>(&g_adaln_down_stack), down_bytes);
                cudaError_t a2 = cudaMalloc(reinterpret_cast<void**>(&g_adaln_up_stack), up_bytes);
                TORCH_CHECK(a1 == cudaSuccess && a2 == cudaSuccess,
                            "AdaLN-LoRA stack cudaMalloc failed: ",
                            cudaGetErrorString(a1 != cudaSuccess ? a1 : a2));

                static const std::array<const char*, 3> sublayer_names = {
                    "self_attn", "cross_attn", "mlp",
                };
                size_t per_instance_down_bytes = size_t(lora_dim) * K * sizeof(cutlass::bfloat16_t);
                size_t per_instance_up_bytes   = size_t(3) * K * lora_dim * sizeof(cutlass::bfloat16_t);
                for (int bi = 0; bi < num_blocks; ++bi) {
                    const std::string p_b = block_prefix(bi);
                    for (int sj = 0; sj < 3; ++sj) {
                        std::string down_key = p_b + "adaln_modulation_" + sublayer_names[sj] + ".1.weight";
                        std::string up_key   = p_b + "adaln_modulation_" + sublayer_names[sj] + ".2.weight";
                        auto down_t = get_w(weights, down_key);
                        auto up_t   = get_w(weights, up_key);
                        TORCH_CHECK(down_t.scalar_type() == at::kBFloat16 &&
                                    up_t.scalar_type() == at::kBFloat16,
                                    "AdaLN-LoRA weights must be torch.bfloat16; got ",
                                    down_t.scalar_type(), "/", up_t.scalar_type());
                        TORCH_CHECK((int)down_t.size(0) == lora_dim &&
                                    (int)down_t.size(1) == K,
                                    "Expected adaln down weight shape [", lora_dim, ", ", K,
                                    "], got [", down_t.size(0), ", ", down_t.size(1), "]");
                        TORCH_CHECK((int)up_t.size(0) == 3 * K &&
                                    (int)up_t.size(1) == lora_dim,
                                    "Expected adaln up weight shape [", 3 * K, ", ", lora_dim,
                                    "], got [", up_t.size(0), ", ", up_t.size(1), "]");
                        int instance = bi * 3 + sj;
                        cudaMemcpyAsync(
                            g_adaln_down_stack + size_t(instance) * lora_dim * K,
                            down_t.data_ptr<at::BFloat16>(),
                            per_instance_down_bytes,
                            cudaMemcpyDeviceToDevice, stream);
                        cudaMemcpyAsync(
                            g_adaln_up_stack + size_t(instance) * 3 * K * lora_dim,
                            up_t.data_ptr<at::BFloat16>(),
                            per_instance_up_bytes,
                            cudaMemcpyDeviceToDevice, stream);
                    }
                }
                g_adaln_num_blocks = num_blocks;
                g_adaln_K = K;
                g_adaln_lora_dim = lora_dim;
                g_adaln_fingerprint_down = fingerprint;
            }
        }

        if (adaln_precompute_enabled) {
            cutlass::bfloat16_t* lora_hidden_base =
                reinterpret_cast<cutlass::bfloat16_t*>(lora_hidden_all.data_ptr<at::BFloat16>());
            cutlass::bfloat16_t* mods_base =
                reinterpret_cast<cutlass::bfloat16_t*>(mods_all.data_ptr<at::BFloat16>());
            const cutlass::bfloat16_t* silu_emb_ptr =
                reinterpret_cast<const cutlass::bfloat16_t*>(t_emb_silu.data_ptr<at::BFloat16>());
            const cutlass::bfloat16_t* lora_3d_ptr =
                reinterpret_cast<const cutlass::bfloat16_t*>(adaln_lora_BD.data_ptr<at::BFloat16>());

            int N_down = num_blocks * 3 * lora_dim;
            cudaError_t e_down = omnidreams_singleview::cutlass_linear_layer_rrr_bf16(
                silu_emb_ptr, g_adaln_down_stack, /*bias=*/nullptr,
                lora_hidden_base, B, K, N_down, stream);
            TORCH_CHECK(e_down == cudaSuccess,
                        "AdaLN-LoRA global down GEMM failed: ", cudaGetErrorString(e_down));

            int batchCount = num_blocks * 3;
            int64_t stride_a    = int64_t(B) * lora_dim;
            int64_t stride_b    = int64_t(3) * K * lora_dim;
            int64_t stride_d    = int64_t(B) * 3 * K;
            int64_t stride_bias = 0;
            cudaError_t e_up = omnidreams_singleview::cublaslt_strided_batched_bf16_gemm(
                lora_hidden_base, g_adaln_up_stack, lora_3d_ptr, mods_base,
                B, lora_dim, 3 * K,
                batchCount,
                stride_a, stride_b, stride_d, stride_bias,
                stream);
            if (e_up != cudaSuccess) {
                std::fprintf(stderr,
                    "[OmniDreams single-view native extension] AdaLN-LoRA batched up GEMM failed (%s); "
                    "falling back to per-block path for this forward.\n",
                    cudaGetErrorString(e_up));
                adaln_precompute_enabled = false;
            }
        }
    }

    // x is updated in place block-by-block. Make `cur` contiguous and own
    // its own storage (the residual updates are in-place writes). cur
    // currently holds [B, L, K] from x_embedder + hdmap.
    cur = cur.contiguous();

    torch::Tensor trace_tensor;
    bool trace_enabled = false;
    if (config.contains("cosmos_trace_tensor")) {
        trace_tensor = py::cast<torch::Tensor>(config["cosmos_trace_tensor"]);
        TORCH_CHECK(trace_tensor.is_cuda(), "config['cosmos_trace_tensor'] must be CUDA");
        TORCH_CHECK(trace_tensor.device() == x_new.device(),
                    "config['cosmos_trace_tensor'] must be on device ", x_new.device());
        TORCH_CHECK(trace_tensor.scalar_type() == at::kBFloat16,
                    "config['cosmos_trace_tensor'] must be torch.bfloat16; got ",
                    trace_tensor.scalar_type());
        TORCH_CHECK(trace_tensor.dim() == 5 &&
                    trace_tensor.size(0) == num_blocks &&
                    trace_tensor.size(1) == 4 &&
                    trace_tensor.size(2) == B &&
                    trace_tensor.size(3) == M &&
                    trace_tensor.size(4) == K,
                    "config['cosmos_trace_tensor'] must have shape [num_blocks, 4, B, M, K] = [",
                    num_blocks, ", 4, ", B, ", ", M, ", ", K, "]");
        TORCH_CHECK(trace_tensor.is_contiguous(), "config['cosmos_trace_tensor'] must be contiguous");
        trace_enabled = true;
    }

    torch::Tensor block_mods_sa;
    torch::Tensor block_mods_ca;
    torch::Tensor block_mods_mlp;
    bool block_mod_cache_enabled = false;
    if (config.contains("cosmos_block_mods_sa") ||
        config.contains("cosmos_block_mods_ca") ||
        config.contains("cosmos_block_mods_mlp")) {
        TORCH_CHECK(config.contains("cosmos_block_mods_sa") &&
                    config.contains("cosmos_block_mods_ca") &&
                    config.contains("cosmos_block_mods_mlp"),
                    "precomputed block AdaLN requires config['cosmos_block_mods_sa'], "
                    "config['cosmos_block_mods_ca'], and config['cosmos_block_mods_mlp']");
        auto validate_block_mod_cache = [&](const torch::Tensor& t, const char* name) {
            TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
            TORCH_CHECK(t.device() == x_new.device(), name, " must be on device ", x_new.device());
            TORCH_CHECK(t.scalar_type() == orig_scalar_type,
                        name, " has dtype ", t.scalar_type(), ", expected ", orig_scalar_type);
            TORCH_CHECK(t.dim() == 3 &&
                        (int)t.size(0) == num_blocks &&
                        (int)t.size(1) == B &&
                        (int)t.size(2) == 3 * K,
                        name, " must have shape [", num_blocks, ", ", B, ", ", 3 * K,
                        "], got [",
                        t.dim() > 0 ? t.size(0) : 0, ", ",
                        t.dim() > 1 ? t.size(1) : 0, ", ",
                        t.dim() > 2 ? t.size(2) : 0, "]");
            TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
        };
        block_mods_sa = py::cast<torch::Tensor>(config["cosmos_block_mods_sa"]);
        block_mods_ca = py::cast<torch::Tensor>(config["cosmos_block_mods_ca"]);
        block_mods_mlp = py::cast<torch::Tensor>(config["cosmos_block_mods_mlp"]);
        validate_block_mod_cache(block_mods_sa, "config['cosmos_block_mods_sa']");
        validate_block_mod_cache(block_mods_ca, "config['cosmos_block_mods_ca']");
        validate_block_mod_cache(block_mods_mlp, "config['cosmos_block_mods_mlp']");
        block_mod_cache_enabled = true;
    }

    auto validate_fp8_activation_tensor_shape = [&](const torch::Tensor& tensor, const char* name) {
        TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be torch.float32; got ", tensor.scalar_type());
        TORCH_CHECK(tensor.dim() == 2 &&
                    tensor.size(0) == num_blocks &&
                    tensor.size(1) == omnidreams_singleview::kCosmosFp8ActivationScaleSites,
                    name, " must have shape [num_blocks, ",
                    omnidreams_singleview::kCosmosFp8ActivationScaleSites, "] = [",
                    num_blocks, ", ", omnidreams_singleview::kCosmosFp8ActivationScaleSites, "]");
        TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    };
    torch::Tensor fp8_activation_scales_host;
    bool fp8_activation_scales_enabled = false;
    if (config.contains("cosmos_fp8_activation_scales")) {
        auto fp8_activation_scales = py::cast<torch::Tensor>(config["cosmos_fp8_activation_scales"]);
        validate_fp8_activation_tensor_shape(fp8_activation_scales, "config['cosmos_fp8_activation_scales']");
        if (fp8_activation_scales.is_cuda()) {
            TORCH_CHECK(fp8_activation_scales.device() == x_new.device(),
                        "config['cosmos_fp8_activation_scales'] must be on device ", x_new.device(),
                        " when CUDA");
        }
        fp8_activation_scales_host = fp8_activation_scales.to(torch::kCPU).contiguous();
        fp8_activation_scales_enabled = true;
    }
    torch::Tensor fp8_activation_amax_tensor;
    bool fp8_activation_amax_enabled = false;
    if (config.contains("cosmos_fp8_activation_amax_tensor")) {
        fp8_activation_amax_tensor = py::cast<torch::Tensor>(config["cosmos_fp8_activation_amax_tensor"]);
        TORCH_CHECK(fp8_activation_amax_tensor.is_cuda(),
                    "config['cosmos_fp8_activation_amax_tensor'] must be CUDA");
        TORCH_CHECK(fp8_activation_amax_tensor.device() == x_new.device(),
                    "config['cosmos_fp8_activation_amax_tensor'] must be on device ", x_new.device());
        validate_fp8_activation_tensor_shape(fp8_activation_amax_tensor, "config['cosmos_fp8_activation_amax_tensor']");
        fp8_activation_amax_enabled = true;
    }

    // ── 5b. Per-block loop: one C++ orchestrator call per block ──────────────
    for (int i = 0; i < num_blocks; ++i) {
        const std::string p = block_prefix(i);

        auto get_block_linear_weight = [&](const std::string& rel_key) {
            std::string key = p + rel_key;
            auto canonical = get_w(weights, key);
            auto t = canonical;
            if (quantized_prepared && canonical.scalar_type() == at::kByte) {
                std::string prepared_key = key + "_fp8_prepared";
                if (weights.contains(py::str(prepared_key))) {
                    t = get_w(weights, prepared_key);
                    TORCH_CHECK(t.scalar_type() == at::kByte,
                                prepared_key, " must be torch.uint8 raw E4M3 bytes; got ",
                                t.scalar_type());
                    TORCH_CHECK(t.dim() == canonical.dim(),
                                prepared_key, " must have the same rank as ", key);
                    for (int64_t dim = 0; dim < canonical.dim(); ++dim) {
                        TORCH_CHECK(t.size(dim) == canonical.size(dim),
                                    prepared_key, " must have the same shape as ", key);
                    }
                    TORCH_CHECK(t.is_contiguous(), prepared_key, " must be contiguous");
                    t = t.contiguous();
                } else if (quantized_prepared_strict) {
                    TORCH_CHECK(false,
                                "missing FP8 prepared alias ", prepared_key,
                                " while cosmos_quantized_prepared_strict=True");
                }
            }
            if (linear_backend == omnidreams_singleview::CosmosLinearBackend::FP8) {
                TORCH_CHECK(t.scalar_type() == at::kByte,
                            key, " must be torch.uint8 raw E4M3 bytes when cosmos_linear_backend=fp8; got ",
                            t.scalar_type());
            } else if (linear_backend == omnidreams_singleview::CosmosLinearBackend::BF16) {
                TORCH_CHECK(t.scalar_type() == at::kBFloat16,
                            key, " must be torch.bfloat16 when cosmos_linear_backend=bf16; got ",
                            t.scalar_type());
            } else {
                TORCH_CHECK(t.scalar_type() == at::kByte || t.scalar_type() == at::kBFloat16,
                            key, " must be torch.bfloat16 or torch.uint8 raw E4M3 bytes "
                            "when cosmos_linear_backend=mixed; got ", t.scalar_type());
            }
            return t;
        };
        auto has_block_weight = [&](const std::string& rel_key) {
            std::string key = p + rel_key;
            return weights.contains(py::str(key));
        };
        auto get_block_linear_scale = [&](const std::string& rel_key, const torch::Tensor& weight) {
            if (weight.scalar_type() != at::kByte) {
                return torch::Tensor();
            }
            std::string canonical_key = p + rel_key + "_scale";
            std::string key = canonical_key;
            bool using_prepared_scale = false;
            if (quantized_prepared) {
                std::string prepared_key = p + rel_key + "_fp8_prepared_scale";
                if (weights.contains(py::str(prepared_key))) {
                    key = prepared_key;
                    using_prepared_scale = true;
                } else if (quantized_prepared_strict) {
                    TORCH_CHECK(false,
                                "missing FP8 prepared scale alias ", prepared_key,
                                " while cosmos_quantized_prepared_strict=True");
                }
            }
            auto s = get_w(weights, key);
            if (using_prepared_scale) {
                TORCH_CHECK(s.scalar_type() == at::kHalf,
                            key, " must be torch.float16; got ", s.scalar_type());
            }
            if (s.scalar_type() != at::kHalf) {
                s = s.to(torch::kFloat16);
            }
            TORCH_CHECK(s.dim() == 1, key, " must be [out_features]");
            TORCH_CHECK(s.numel() == weight.size(0),
                        key, " must have ", weight.size(0), " elements, got ", s.numel());
            TORCH_CHECK(s.is_contiguous(), key, " must be contiguous");
            return s.contiguous();
        };
        auto linear_weight_ptr = [&](const torch::Tensor& t) -> const void* {
            if (t.scalar_type() == at::kByte) {
                return t.data_ptr<uint8_t>();
            }
            return t.data_ptr<at::BFloat16>();
        };
        auto scale_ptr = [](const torch::Tensor& t) -> const cutlass::half_t* {
            return t.defined()
                ? reinterpret_cast<const cutlass::half_t*>(t.data_ptr<at::Half>())
                : nullptr;
        };
        auto optional_prepared_weight = [&](const std::string& rel_key,
                                            int rows,
                                            int cols) -> torch::Tensor {
            std::string key = p + rel_key + "_prepared";
            if (!weights.contains(py::str(key))) {
                return torch::Tensor();
            }
            auto t = get_w(weights, key);
            TORCH_CHECK(t.scalar_type() == at::kBFloat16,
                        key, " must be torch.bfloat16; got ", t.scalar_type());
            TORCH_CHECK(t.dim() == 2 && (int)t.size(0) == rows && (int)t.size(1) == cols,
                        key, " must have shape [", rows, ", ", cols, "], got [",
                        t.dim() > 0 ? t.size(0) : 0, ", ",
                        t.dim() > 1 ? t.size(1) : 0, "]");
            return t;
        };
        auto prepared_ptr = [](const torch::Tensor& t) -> const cutlass::bfloat16_t* {
            return t.defined()
                ? reinterpret_cast<const cutlass::bfloat16_t*>(t.data_ptr<at::BFloat16>())
                : nullptr;
        };

        const bool use_fused_sa_qkv =
            has_block_weight("self_attn.qkv_proj.weight") &&
            get_w(weights, p + "self_attn.qkv_proj.weight").scalar_type() == at::kByte;
        torch::Tensor sa_w_qkv_t;
        torch::Tensor sa_w_qkv_scale_t;
        if (use_fused_sa_qkv) {
            sa_w_qkv_t = get_block_linear_weight("self_attn.qkv_proj.weight");
            sa_w_qkv_scale_t = get_block_linear_scale("self_attn.qkv_proj.weight", sa_w_qkv_t);
            TORCH_CHECK(sa_w_qkv_t.dim() == 2 && (int)sa_w_qkv_t.size(0) == 3 * K &&
                        (int)sa_w_qkv_t.size(1) == K,
                        p, "self_attn.qkv_proj.weight must have shape [", 3 * K, ", ", K,
                        "], got [", sa_w_qkv_t.dim() > 0 ? sa_w_qkv_t.size(0) : 0,
                        ", ", sa_w_qkv_t.dim() > 1 ? sa_w_qkv_t.size(1) : 0, "]");
            TORCH_CHECK(sa_w_qkv_scale_t.numel() == 3 * K,
                        p, "self_attn.qkv_proj.weight_scale must have ", 3 * K,
                        " elements, got ", sa_w_qkv_scale_t.numel());
        }
        auto sa_w_q_t = use_fused_sa_qkv ? torch::Tensor() : get_block_linear_weight("self_attn.q_proj.weight");
        auto sa_w_k_t = use_fused_sa_qkv ? torch::Tensor() : get_block_linear_weight("self_attn.k_proj.weight");
        auto sa_w_v_t = use_fused_sa_qkv ? torch::Tensor() : get_block_linear_weight("self_attn.v_proj.weight");
        auto sa_w_out_t = get_block_linear_weight("self_attn.output_proj.weight");
        auto ca_w_q_t = get_block_linear_weight("cross_attn.q_proj.weight");
        auto ca_w_out_t = get_block_linear_weight("cross_attn.output_proj.weight");
        auto ffn_w1_t = get_block_linear_weight("mlp.layer1.weight");
        auto ffn_w2_t = get_block_linear_weight("mlp.layer2.weight");

        auto sa_w_q_scale_t = use_fused_sa_qkv ? torch::Tensor() : get_block_linear_scale("self_attn.q_proj.weight", sa_w_q_t);
        auto sa_w_k_scale_t = use_fused_sa_qkv ? torch::Tensor() : get_block_linear_scale("self_attn.k_proj.weight", sa_w_k_t);
        auto sa_w_v_scale_t = use_fused_sa_qkv ? torch::Tensor() : get_block_linear_scale("self_attn.v_proj.weight", sa_w_v_t);
        auto sa_w_out_scale_t = get_block_linear_scale("self_attn.output_proj.weight", sa_w_out_t);
        auto ca_w_q_scale_t = get_block_linear_scale("cross_attn.q_proj.weight", ca_w_q_t);
        auto ca_w_out_scale_t = get_block_linear_scale("cross_attn.output_proj.weight", ca_w_out_t);
        auto ffn_w1_scale_t = get_block_linear_scale("mlp.layer1.weight", ffn_w1_t);
        auto ffn_w2_scale_t = get_block_linear_scale("mlp.layer2.weight", ffn_w2_t);
        auto sa_w_qkv_prepared_t = optional_prepared_weight("self_attn.qkv_proj.weight", K, 3 * K);
        auto sa_w_out_prepared_t = optional_prepared_weight("self_attn.output_proj.weight", K, K);
        auto ca_w_q_prepared_t = optional_prepared_weight("cross_attn.q_proj.weight", K, K);
        auto ca_w_out_prepared_t = optional_prepared_weight("cross_attn.output_proj.weight", K, K);
        auto ffn_w1_prepared_t = optional_prepared_weight("mlp.layer1.weight", K, FF);
        auto ffn_w2_prepared_t = optional_prepared_weight("mlp.layer2.weight", FF, K);

        omnidreams_singleview::CosmosBlockWeights bw{};
        bw.sa_w_qkv        = use_fused_sa_qkv ? linear_weight_ptr(sa_w_qkv_t) : nullptr;
        bw.sa_w_q          = use_fused_sa_qkv ? nullptr : linear_weight_ptr(sa_w_q_t);
        bw.sa_w_k          = use_fused_sa_qkv ? nullptr : linear_weight_ptr(sa_w_k_t);
        bw.sa_w_v          = use_fused_sa_qkv ? nullptr : linear_weight_ptr(sa_w_v_t);
        bw.sa_w_out        = linear_weight_ptr(sa_w_out_t);
        bw.sa_w_qkv_scale  = use_fused_sa_qkv ? scale_ptr(sa_w_qkv_scale_t) : nullptr;
        bw.sa_w_q_scale    = scale_ptr(sa_w_q_scale_t);
        bw.sa_w_k_scale    = scale_ptr(sa_w_k_scale_t);
        bw.sa_w_v_scale    = scale_ptr(sa_w_v_scale_t);
        bw.sa_w_out_scale  = scale_ptr(sa_w_out_scale_t);
        bw.sa_w_qkv_prepared = prepared_ptr(sa_w_qkv_prepared_t);
        bw.sa_w_out_prepared = prepared_ptr(sa_w_out_prepared_t);
        bw.sa_q_norm       = reinterpret_cast<const cutlass::bfloat16_t*>(get_w(weights, p + "self_attn.q_norm.weight").data_ptr<at::BFloat16>());
        bw.sa_k_norm       = reinterpret_cast<const cutlass::bfloat16_t*>(get_w(weights, p + "self_attn.k_norm.weight").data_ptr<at::BFloat16>());
        bw.ca_w_q          = linear_weight_ptr(ca_w_q_t);
        bw.ca_w_out        = linear_weight_ptr(ca_w_out_t);
        bw.ca_w_q_scale    = scale_ptr(ca_w_q_scale_t);
        bw.ca_w_out_scale  = scale_ptr(ca_w_out_scale_t);
        bw.ca_w_q_prepared = prepared_ptr(ca_w_q_prepared_t);
        bw.ca_w_out_prepared = prepared_ptr(ca_w_out_prepared_t);
        bw.ca_q_norm       = reinterpret_cast<const cutlass::bfloat16_t*>(get_w(weights, p + "cross_attn.q_norm.weight").data_ptr<at::BFloat16>());
        bw.ffn_w1          = linear_weight_ptr(ffn_w1_t);
        bw.ffn_w2          = linear_weight_ptr(ffn_w2_t);
        bw.ffn_w1_scale    = scale_ptr(ffn_w1_scale_t);
        bw.ffn_w2_scale    = scale_ptr(ffn_w2_scale_t);
        bw.ffn_w1_prepared = prepared_ptr(ffn_w1_prepared_t);
        bw.ffn_w2_prepared = prepared_ptr(ffn_w2_prepared_t);
        bw.adaln_sa_down   = reinterpret_cast<const cutlass::bfloat16_t*>(get_w(weights, p + "adaln_modulation_self_attn.1.weight").data_ptr<at::BFloat16>());
        bw.adaln_sa_up     = reinterpret_cast<const cutlass::bfloat16_t*>(get_w(weights, p + "adaln_modulation_self_attn.2.weight").data_ptr<at::BFloat16>());
        bw.adaln_ca_down   = reinterpret_cast<const cutlass::bfloat16_t*>(get_w(weights, p + "adaln_modulation_cross_attn.1.weight").data_ptr<at::BFloat16>());
        bw.adaln_ca_up     = reinterpret_cast<const cutlass::bfloat16_t*>(get_w(weights, p + "adaln_modulation_cross_attn.2.weight").data_ptr<at::BFloat16>());
        bw.adaln_mlp_down  = reinterpret_cast<const cutlass::bfloat16_t*>(get_w(weights, p + "adaln_modulation_mlp.1.weight").data_ptr<at::BFloat16>());
        bw.adaln_mlp_up    = reinterpret_cast<const cutlass::bfloat16_t*>(get_w(weights, p + "adaln_modulation_mlp.2.weight").data_ptr<at::BFloat16>());

        // Cross-attn KV caches: FlashDreams stores them as [B*V, Mk_c, H, D].
        // With V==1 this is [B, Mk_c, H, D]. Self-attn caches: [B*V, cap, H, D].
        TORCH_CHECK(k_cross_caches[i].is_cuda() && k_cross_caches[i].dim() == 4);
        TORCH_CHECK(v_cross_caches[i].is_cuda() && v_cross_caches[i].dim() == 4);
        TORCH_CHECK(k_self_caches[i].is_cuda()  && k_self_caches[i].dim() == 4);
        TORCH_CHECK(v_self_caches[i].is_cuda()  && v_self_caches[i].dim() == 4);
        int Mk_c = (int)k_cross_caches[i].size(1);
        int cap  = (int)k_self_caches[i].size(1);
        auto validate_fp8_cache = [&](const torch::Tensor& t, const char* name, int tokens) {
            TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
            TORCH_CHECK(t.scalar_type() == at::kByte,
                        name, " must be torch.uint8 raw E4M3 bytes; got ", t.scalar_type());
            TORCH_CHECK(t.dim() == 4,
                        name, " must be 4D [B, tokens, H, D]; got dim=", t.dim());
            TORCH_CHECK((int)t.size(0) == B && (int)t.size(1) >= tokens &&
                        (int)t.size(2) == H && (int)t.size(3) == D,
                        name, " must have shape [", B, ", >= ", tokens, ", ", H, ", ", D,
                        "], got [", t.size(0), ", ", t.size(1), ", ", t.size(2), ", ", t.size(3), "]");
            TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
        };
        auto validate_fp8_k_bhmd_cache = [&](const torch::Tensor& t, const char* name, int tokens) {
            TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
            TORCH_CHECK(t.scalar_type() == at::kByte,
                        name, " must be torch.uint8 raw E4M3 bytes; got ", t.scalar_type());
            TORCH_CHECK(t.dim() == 4,
                        name, " must be 4D [B, H, tokens, D]; got dim=", t.dim());
            TORCH_CHECK((int)t.size(0) == B && (int)t.size(1) == H &&
                        (int)t.size(2) >= tokens && (int)t.size(3) == D,
                        name, " must have shape [", B, ", ", H, ", >= ", tokens, ", ", D,
                        "], got [", t.size(0), ", ", t.size(1), ", ", t.size(2), ", ", t.size(3), "]");
            TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
        };
        auto validate_fp8_v_bhdm_cache = [&](const torch::Tensor& t, const char* name, int tokens) {
            TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
            TORCH_CHECK(t.scalar_type() == at::kByte,
                        name, " must be torch.uint8 raw E4M3 bytes; got ", t.scalar_type());
            TORCH_CHECK(t.dim() == 4,
                        name, " must be 4D [B, H, D, tokens]; got dim=", t.dim());
            TORCH_CHECK((int)t.size(0) == B && (int)t.size(1) == H &&
                        (int)t.size(2) == D && (int)t.size(3) >= tokens,
                        name, " must have shape [", B, ", ", H, ", ", D, ", >= ", tokens,
                        "], got [", t.size(0), ", ", t.size(1), ", ", t.size(2), ", ", t.size(3), "]");
            TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
        };
        auto validate_sage3_k_fp4_cache = [&](const torch::Tensor& t, const char* name, int tokens) {
            int padded = ((tokens + 127) / 128) * 128;
            TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
            TORCH_CHECK(t.scalar_type() == at::kByte,
                        name, " must be torch.uint8 Sage3 FP4 bytes; got ", t.scalar_type());
            TORCH_CHECK(t.dim() == 4,
                        name, " must be 4D [B, H, padded_tokens, D/2]; got dim=", t.dim());
            TORCH_CHECK((int)t.size(0) == B && (int)t.size(1) == H &&
                        (int)t.size(2) == padded && (int)t.size(3) == D / 2,
                        name, " must have shape [", B, ", ", H, ", ", padded, ", ", D / 2,
                        "], got [", t.size(0), ", ", t.size(1), ", ", t.size(2), ", ", t.size(3), "]");
            TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
        };
        auto validate_sage3_v_fp4_cache = [&](const torch::Tensor& t, const char* name, int tokens) {
            int padded = ((tokens + 127) / 128) * 128;
            TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
            TORCH_CHECK(t.scalar_type() == at::kByte,
                        name, " must be torch.uint8 Sage3 FP4 bytes; got ", t.scalar_type());
            TORCH_CHECK(t.dim() == 4,
                        name, " must be 4D [B, H, D, padded_tokens/2]; got dim=", t.dim());
            TORCH_CHECK((int)t.size(0) == B && (int)t.size(1) == H &&
                        (int)t.size(2) == D && (int)t.size(3) == padded / 2,
                        name, " must have shape [", B, ", ", H, ", ", D, ", ", padded / 2,
                        "], got [", t.size(0), ", ", t.size(1), ", ", t.size(2), ", ", t.size(3), "]");
            TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
        };
        auto validate_sage3_k_sf_cache = [&](const torch::Tensor& t, const char* name, int tokens) {
            int padded = ((tokens + 127) / 128) * 128;
            TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
            TORCH_CHECK(t.scalar_type() == at::ScalarType::Float8_e4m3fn,
                        name, " must be torch.float8_e4m3fn Sage3 scale bytes; got ", t.scalar_type());
            TORCH_CHECK(t.dim() == 4,
                        name, " must be 4D [B, H, padded_tokens, D/16]; got dim=", t.dim());
            TORCH_CHECK((int)t.size(0) == B && (int)t.size(1) == H &&
                        (int)t.size(2) == padded && (int)t.size(3) == D / 16,
                        name, " must have shape [", B, ", ", H, ", ", padded, ", ", D / 16,
                        "], got [", t.size(0), ", ", t.size(1), ", ", t.size(2), ", ", t.size(3), "]");
            TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
        };
        auto validate_sage3_v_sf_cache = [&](const torch::Tensor& t, const char* name, int tokens) {
            int padded = ((tokens + 127) / 128) * 128;
            TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
            TORCH_CHECK(t.scalar_type() == at::ScalarType::Float8_e4m3fn,
                        name, " must be torch.float8_e4m3fn Sage3 scale bytes; got ", t.scalar_type());
            TORCH_CHECK(t.dim() == 4,
                        name, " must be 4D [B, H, D, padded_tokens/16]; got dim=", t.dim());
            TORCH_CHECK((int)t.size(0) == B && (int)t.size(1) == H &&
                        (int)t.size(2) == D && (int)t.size(3) == padded / 16,
                        name, " must have shape [", B, ", ", H, ", ", D, ", ", padded / 16,
                        "], got [", t.size(0), ", ", t.size(1), ", ", t.size(2), ", ", t.size(3), "]");
            TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
        };
        if (fp8_kv_cache_enabled) {
            if (!sage3_cross_fp4_enabled) {
                validate_fp8_cache(k_cross_fp8_caches[i], "k_cross_fp8_caches[i]", Mk_c);
                validate_fp8_cache(v_cross_fp8_caches[i], "v_cross_fp8_caches[i]", Mk_c);
            }
            validate_fp8_cache(k_self_fp8_caches[i], "k_self_fp8_caches[i]", cap);
            validate_fp8_cache(v_self_fp8_caches[i], "v_self_fp8_caches[i]", cap);
            if (fp8_cross_tc_layout_cache_enabled) {
                validate_fp8_k_bhmd_cache(k_cross_fp8_bhmd_caches[i], "k_cross_fp8_bhmd_caches[i]", Mk_c);
                validate_fp8_k_bhmd_cache(v_cross_fp8_bhmd_caches[i], "v_cross_fp8_bhmd_caches[i]", Mk_c);
                if (!v_cross_fp8_bhdm_caches.empty()) {
                    validate_fp8_v_bhdm_cache(v_cross_fp8_bhdm_caches[i], "v_cross_fp8_bhdm_caches[i]", Mk_c);
                }
            }
            if (fp8_self_tc_layout_cache_enabled) {
                validate_fp8_k_bhmd_cache(k_self_fp8_bhmd_caches[i], "k_self_fp8_bhmd_caches[i]", cap);
                validate_fp8_k_bhmd_cache(v_self_fp8_bhmd_caches[i], "v_self_fp8_bhmd_caches[i]", cap);
                if (!v_self_fp8_bhdm_caches.empty()) {
                    validate_fp8_v_bhdm_cache(v_self_fp8_bhdm_caches[i], "v_self_fp8_bhdm_caches[i]", cap);
                }
            }
        }
        if (sage3_cross_fp4_enabled) {
            validate_sage3_k_fp4_cache(k_cross_sage3_fp4_caches[i], "k_cross_sage3_fp4_caches[i]", Mk_c);
            validate_sage3_v_fp4_cache(v_cross_sage3_fp4_caches[i], "v_cross_sage3_fp4_caches[i]", Mk_c);
            validate_sage3_k_sf_cache(k_cross_sage3_sf_caches[i], "k_cross_sage3_sf_caches[i]", Mk_c);
            validate_sage3_v_sf_cache(v_cross_sage3_sf_caches[i], "v_cross_sage3_sf_caches[i]", Mk_c);
        }

        omnidreams_singleview::CosmosAttentionBackend block_attention_backend = attention_backend;
        const bool block_write_bf16_self_kv_cache = write_bf16_self_kv_cache;

        omnidreams_singleview::CosmosBlockParams bp{};
        bp.B  = B;
        bp.M  = M;
        bp.K  = K;
        bp.H  = H;
        bp.D  = D;
        bp.FF = FF;
        bp.lora_dim = lora_dim;
        bp.Mk_cross = Mk_c;
        bp.self_attn_cache_cap   = cap;
        bp.self_attn_write_start = (int)self_attn_write_start;
        bp.linear_backend = linear_backend;
        bp.attention_backend = block_attention_backend;
        bp.fp8_kv_cache_enabled = fp8_kv_cache_enabled;
        bp.write_bf16_self_kv_cache = block_write_bf16_self_kv_cache;
        if (trace_enabled) {
            auto* trace_base = reinterpret_cast<cutlass::bfloat16_t*>(
                trace_tensor.data_ptr<at::BFloat16>());
            const int64_t trace_elems = static_cast<int64_t>(B) * M * K;
            auto* block_trace = trace_base + static_cast<int64_t>(i) * 4 * trace_elems;
            bp.trace_sa_out = block_trace;
            bp.trace_ca_out = block_trace + trace_elems;
            bp.trace_ffn_out = block_trace + 2 * trace_elems;
            bp.trace_block_out = block_trace + 3 * trace_elems;
            bp.trace_elems = trace_elems;
        }
        if (fp8_activation_scales_enabled) {
            bp.fp8_activation_scales =
                fp8_activation_scales_host.data_ptr<float>() +
                static_cast<int64_t>(i) * omnidreams_singleview::kCosmosFp8ActivationScaleSites;
        }
        if (fp8_activation_amax_enabled) {
            bp.fp8_activation_amax_out =
                fp8_activation_amax_tensor.data_ptr<float>() +
                static_cast<int64_t>(i) * omnidreams_singleview::kCosmosFp8ActivationScaleSites;
        }
        bp.x = reinterpret_cast<cutlass::bfloat16_t*>(cur.data_ptr<at::BFloat16>());
        bp.t_emb         = reinterpret_cast<const cutlass::bfloat16_t*>(t_emb_silu.data_ptr<at::BFloat16>());
        bp.adaln_lora_3D = reinterpret_cast<const cutlass::bfloat16_t*>(adaln_lora_BD.data_ptr<at::BFloat16>());
        if (block_mod_cache_enabled) {
            const int64_t block_mod_stride = static_cast<int64_t>(B) * 3 * K;
            bp.precomputed_mods_sa = reinterpret_cast<const cutlass::bfloat16_t*>(
                block_mods_sa.data_ptr<at::BFloat16>()) + static_cast<int64_t>(i) * block_mod_stride;
            bp.precomputed_mods_ca = reinterpret_cast<const cutlass::bfloat16_t*>(
                block_mods_ca.data_ptr<at::BFloat16>()) + static_cast<int64_t>(i) * block_mod_stride;
            bp.precomputed_mods_mlp = reinterpret_cast<const cutlass::bfloat16_t*>(
                block_mods_mlp.data_ptr<at::BFloat16>()) + static_cast<int64_t>(i) * block_mod_stride;
        }
        bp.k_cross = reinterpret_cast<const cutlass::bfloat16_t*>(k_cross_caches[i].data_ptr<at::BFloat16>());
        bp.v_cross = reinterpret_cast<const cutlass::bfloat16_t*>(v_cross_caches[i].data_ptr<at::BFloat16>());
        bp.k_self_cache = reinterpret_cast<cutlass::bfloat16_t*>(k_self_caches[i].data_ptr<at::BFloat16>());
        bp.v_self_cache = reinterpret_cast<cutlass::bfloat16_t*>(v_self_caches[i].data_ptr<at::BFloat16>());
        bp.k_cross_fp8 = fp8_kv_cache_enabled && !k_cross_fp8_caches.empty()
            ? reinterpret_cast<const cutlass::float_e4m3_t*>(k_cross_fp8_caches[i].data_ptr<uint8_t>())
            : nullptr;
        bp.v_cross_fp8 = fp8_kv_cache_enabled && !v_cross_fp8_caches.empty()
            ? reinterpret_cast<const cutlass::float_e4m3_t*>(v_cross_fp8_caches[i].data_ptr<uint8_t>())
            : nullptr;
        bp.k_self_cache_fp8 = fp8_kv_cache_enabled
            ? reinterpret_cast<cutlass::float_e4m3_t*>(k_self_fp8_caches[i].data_ptr<uint8_t>())
            : nullptr;
        bp.v_self_cache_fp8 = fp8_kv_cache_enabled
            ? reinterpret_cast<cutlass::float_e4m3_t*>(v_self_fp8_caches[i].data_ptr<uint8_t>())
            : nullptr;
        bp.k_cross_fp8_bhmd = fp8_cross_tc_layout_cache_enabled
            ? reinterpret_cast<const cutlass::float_e4m3_t*>(k_cross_fp8_bhmd_caches[i].data_ptr<uint8_t>())
            : nullptr;
        bp.v_cross_fp8_bhmd = fp8_cross_tc_layout_cache_enabled && !v_cross_fp8_bhmd_caches.empty()
            ? reinterpret_cast<const cutlass::float_e4m3_t*>(v_cross_fp8_bhmd_caches[i].data_ptr<uint8_t>())
            : nullptr;
        bp.v_cross_fp8_bhdm = fp8_cross_tc_layout_cache_enabled && !v_cross_fp8_bhdm_caches.empty()
            ? reinterpret_cast<const cutlass::float_e4m3_t*>(v_cross_fp8_bhdm_caches[i].data_ptr<uint8_t>())
            : nullptr;
        bp.k_cross_fp8_bhmd_tokens = bp.k_cross_fp8_bhmd
            ? (int)k_cross_fp8_bhmd_caches[i].size(2)
            : 0;
        bp.v_cross_fp8_bhmd_tokens = bp.v_cross_fp8_bhmd
            ? (int)v_cross_fp8_bhmd_caches[i].size(2)
            : 0;
        bp.k_cross_sage3_fp4 = sage3_cross_fp4_enabled
            ? k_cross_sage3_fp4_caches[i].data_ptr<uint8_t>()
            : nullptr;
        bp.v_cross_sage3_fp4 = sage3_cross_fp4_enabled
            ? v_cross_sage3_fp4_caches[i].data_ptr<uint8_t>()
            : nullptr;
        bp.k_cross_sage3_sf = sage3_cross_fp4_enabled
            ? reinterpret_cast<const cutlass::float_e4m3_t*>(k_cross_sage3_sf_caches[i].data_ptr())
            : nullptr;
        bp.v_cross_sage3_sf = sage3_cross_fp4_enabled
            ? reinterpret_cast<const cutlass::float_e4m3_t*>(v_cross_sage3_sf_caches[i].data_ptr())
            : nullptr;
        bp.Mk_cross_sage3_padded = sage3_cross_fp4_enabled
            ? ((Mk_c + 127) / 128) * 128
            : 0;
        bp.k_self_cache_fp8_bhmd = fp8_self_tc_layout_cache_enabled
            ? reinterpret_cast<cutlass::float_e4m3_t*>(k_self_fp8_bhmd_caches[i].data_ptr<uint8_t>())
            : nullptr;
        bp.v_self_cache_fp8_bhmd = fp8_self_tc_layout_cache_enabled && !v_self_fp8_bhmd_caches.empty()
            ? reinterpret_cast<cutlass::float_e4m3_t*>(v_self_fp8_bhmd_caches[i].data_ptr<uint8_t>())
            : nullptr;
        bp.v_self_cache_fp8_bhdm = fp8_self_tc_layout_cache_enabled && !v_self_fp8_bhdm_caches.empty()
            ? reinterpret_cast<cutlass::float_e4m3_t*>(v_self_fp8_bhdm_caches[i].data_ptr<uint8_t>())
            : nullptr;
        bp.rope_cos = reinterpret_cast<const cutlass::bfloat16_t*>(rope_cos.data_ptr<at::BFloat16>());
        bp.rope_sin = reinterpret_cast<const cutlass::bfloat16_t*>(rope_sin.data_ptr<at::BFloat16>());
        bp.w   = bw;
        bp.buf = buf;

        // Phase 1 AdaLN-LoRA pre-stack path: when `adaln_precompute_enabled`
        // is true, the global down + batched up GEMMs at the top of the
        // forward have already populated `mods_all` for all 28 blocks. Each
        // block's per-sub-layer (shift, scale, gate) views are slices of
        // `mods_all`. Setting bp.adaln_precomputed = true tells
        // cosmos_run_transformer_block_streaming to skip the three
        // cosmos_adaln_lora_split calls.
        if (adaln_precompute_enabled) {
            cutlass::bfloat16_t* lora_hidden_base =
                reinterpret_cast<cutlass::bfloat16_t*>(lora_hidden_all.data_ptr<at::BFloat16>());
            cutlass::bfloat16_t* mods_base =
                reinterpret_cast<cutlass::bfloat16_t*>(mods_all.data_ptr<at::BFloat16>());
            int64_t lora_inst = int64_t(i) * 3;
            int64_t lora_step = int64_t(B) * lora_dim;
            int64_t mods_step = int64_t(B) * 3 * K;
            bp.buf.lora_hidden_sa  = lora_hidden_base + (lora_inst + 0) * lora_step;
            bp.buf.lora_hidden_ca  = lora_hidden_base + (lora_inst + 1) * lora_step;
            bp.buf.lora_hidden_mlp = lora_hidden_base + (lora_inst + 2) * lora_step;
            bp.buf.mods_sa  = mods_base + (lora_inst + 0) * mods_step;
            bp.buf.mods_ca  = mods_base + (lora_inst + 1) * mods_step;
            bp.buf.mods_mlp = mods_base + (lora_inst + 2) * mods_step;
            bp.adaln_precomputed = true;
        } else {
            bp.adaln_precomputed = false;
        }

        cudaEvent_t block_start, block_end;
        if (prof) {
            cudaEventCreate(&block_start);
            cudaEventCreate(&block_end);
            cudaEventRecord(block_start, stream);
        }
        cudaError_t e = omnidreams_singleview::cosmos_run_transformer_block_streaming(bp, stream);
        if (prof) {
            cudaEventRecord(block_end, stream);
            cudaEventSynchronize(block_end);
            float block_ms = 0.f;
            cudaEventElapsedTime(&block_ms, block_start, block_end);
            std::printf("[cosmos_stream][block=%d] total=%.3f ms\n", i, block_ms);
            cudaEventDestroy(block_start);
            cudaEventDestroy(block_end);
        }
        TORCH_CHECK(e == cudaSuccess,
                    "cosmos_run_transformer_block_streaming failed at block ", i,
                    ": ", cudaGetErrorString(e));
    }
    rec(EV_AFTER_BLOCKS);

    // ── 6. Final layer (matches FlashDreams FinalLayer, n_adaln_chunks=2) ───
    // adaln_lora_BD[..., :2D] is the lora component for the final layer (fused
    // state_dict uses 2*D output for FinalLayer.adaln_modulation).
    //
    {
        torch::Tensor shift_2d;
        torch::Tensor scale_2d;
        torch::Tensor fl_mods;
        if (config.contains("cosmos_final_shift") || config.contains("cosmos_final_scale")) {
            TORCH_CHECK(config.contains("cosmos_final_shift") && config.contains("cosmos_final_scale"),
                        "precomputed final AdaLN requires both config['cosmos_final_shift'] "
                        "and config['cosmos_final_scale']");
            shift_2d = py::cast<torch::Tensor>(config["cosmos_final_shift"]);
            scale_2d = py::cast<torch::Tensor>(config["cosmos_final_scale"]);
            validate_timestep_cache(shift_2d, "config['cosmos_final_shift']", model_channels);
            validate_timestep_cache(scale_2d, "config['cosmos_final_scale']", model_channels);
        } else {
            auto fl_hidden = fl_hidden_scratch.defined()
                ? fl_hidden_scratch
                : torch::empty({B, lora_dim}, opts);
            fl_mods = fl_mods_scratch.defined()
                ? fl_mods_scratch
                : torch::empty({B, 2 * K}, opts);
            auto fl_down = get_w(weights, "final_layer.adaln_modulation.1.weight");
            auto fl_up = get_w(weights, "final_layer.adaln_modulation.2.weight");
            cudaError_t gemm_err = omnidreams_singleview::cutlass_linear_layer_rrr_bf16(
                reinterpret_cast<const cutlass::bfloat16_t*>(t_emb_silu.data_ptr<at::BFloat16>()),
                reinterpret_cast<const cutlass::bfloat16_t*>(fl_down.data_ptr<at::BFloat16>()),
                /*bias=*/nullptr,
                reinterpret_cast<cutlass::bfloat16_t*>(fl_hidden.data_ptr<at::BFloat16>()),
                B, K, lora_dim, stream);
            TORCH_CHECK(gemm_err == cudaSuccess, "final AdaLN down GEMM failed: ", cudaGetErrorString(gemm_err));
            gemm_err = omnidreams_singleview::cutlass_linear_layer_rrr_bf16(
                reinterpret_cast<const cutlass::bfloat16_t*>(fl_hidden.data_ptr<at::BFloat16>()),
                reinterpret_cast<const cutlass::bfloat16_t*>(fl_up.data_ptr<at::BFloat16>()),
                /*bias=*/nullptr,
                reinterpret_cast<cutlass::bfloat16_t*>(fl_mods.data_ptr<at::BFloat16>()),
                B, lora_dim, 2 * K, stream);
            TORCH_CHECK(gemm_err == cudaSuccess, "final AdaLN up GEMM failed: ", cudaGetErrorString(gemm_err));
            cudaError_t final_add_err = cosmos_add_inplace(
                reinterpret_cast<cutlass::bfloat16_t*>(fl_mods.data_ptr<at::BFloat16>()),
                reinterpret_cast<const cutlass::bfloat16_t*>(adaln_lora_BD.data_ptr<at::BFloat16>()),
                int64_t(B) * 2 * K, stream);
            TORCH_CHECK(final_add_err == cudaSuccess, "final AdaLN add failed: ", cudaGetErrorString(final_add_err));
            shift_2d = fl_mods.slice(-1, 0, K);
            scale_2d = fl_mods.slice(-1, K, 2 * K);
        }

        shift_2d = shift_2d.contiguous();
        scale_2d = scale_2d.contiguous();
        auto final_normed = torch::empty({B, L, K}, opts);
        cudaError_t final_ln_err = omnidreams_singleview::cosmos_layernorm_modulate<cutlass::bfloat16_t>(
            reinterpret_cast<const cutlass::bfloat16_t*>(cur.data_ptr<at::BFloat16>()),
            reinterpret_cast<const cutlass::bfloat16_t*>(shift_2d.data_ptr<at::BFloat16>()),
            reinterpret_cast<const cutlass::bfloat16_t*>(scale_2d.data_ptr<at::BFloat16>()),
            reinterpret_cast<cutlass::bfloat16_t*>(final_normed.data_ptr<at::BFloat16>()),
            B * L, K, B, 1e-6f, stream);
        TORCH_CHECK(final_ln_err == cudaSuccess, "final layernorm_modulate failed: ", cudaGetErrorString(final_ln_err));

        auto w_fl = get_w(weights, "final_layer.linear.weight");
        int final_D_out = (int)w_fl.size(0);
        auto final_out = torch::empty({B, L, final_D_out}, opts);
        cudaError_t gemm_err = omnidreams_singleview::cutlass_linear_layer_rrr_bf16(
            reinterpret_cast<const cutlass::bfloat16_t*>(final_normed.data_ptr<at::BFloat16>()),
            reinterpret_cast<const cutlass::bfloat16_t*>(w_fl.data_ptr<at::BFloat16>()),
            /*bias=*/nullptr,
            reinterpret_cast<cutlass::bfloat16_t*>(final_out.data_ptr<at::BFloat16>()),
            B * L, K, final_D_out, stream);
        TORCH_CHECK(gemm_err == cudaSuccess, "final projection GEMM failed: ", cudaGetErrorString(gemm_err));
        cur = final_out;
    }
    rec(EV_AFTER_FINAL);

    if (prof) {
        cudaEventSynchronize(ev[EV_AFTER_FINAL]);
        auto ms = [&](int a, int b) -> float {
            float out = 0.f;
            cudaEventElapsedTime(&out, ev[a], ev[b]);
            return out;
        };
        std::printf(
            "[cosmos_stream] input=%.3f temb=%.3f rope_scratch=%.3f blocks=%.3f final=%.3f total=%.3f ms\n",
            ms(EV_START, EV_AFTER_INPUT_EMBED),
            ms(EV_AFTER_INPUT_EMBED, EV_AFTER_TEMB),
            ms(EV_AFTER_TEMB, EV_AFTER_ROPE_SCRATCH),
            ms(EV_AFTER_ROPE_SCRATCH, EV_AFTER_BLOCKS),
            ms(EV_AFTER_BLOCKS, EV_AFTER_FINAL),
            ms(EV_START, EV_AFTER_FINAL));
        for (int i = 0; i < EV_COUNT; ++i) cudaEventDestroy(ev[i]);
    }

    // ── 7. Reshape back to [B, V, T, HW, D_out] — no unpatchify here;
    //      FlashDreams unpatchify_and_maybe_gather_cp handles that on the
    //      Python side, matching the existing contract of CosmosDiTNetwork. ──
    int D_out = (int)cur.size(-1);
    return cur.reshape({B, V, T, HW, D_out}).to(orig_dtype).contiguous();
}
