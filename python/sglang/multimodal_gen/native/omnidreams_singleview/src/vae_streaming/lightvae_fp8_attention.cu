// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "lightvae_ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <cudnn_frontend.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>

namespace fe = cudnn_frontend;

namespace omnidreams_singleview {
namespace {

constexpr int kFp8ChannelsPerSlice = 16;

void check_u8_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kByte, name, " must be torch.uint8");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_tin16_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    check_u8_cuda_contiguous(tensor, name);
    TORCH_CHECK(tensor.dim() == 5 && tensor.size(4) == kFp8ChannelsPerSlice,
                name, " must be [T,C/16,H,W,16], got ", tensor.sizes());
}

void check_float_scalar_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be torch.float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.numel() == 1, name, " must contain exactly one scalar");
}

int launch_blocks(long long total) {
    constexpr int threads = 256;
    return static_cast<int>(std::min((total + threads - 1) / threads, 65535LL));
}

__global__ void qkv_tin16_to_bmhd_kernel(
    const uint8_t* __restrict__ qkv,
    uint8_t* __restrict__ q,
    uint8_t* __restrict__ k,
    uint8_t* __restrict__ v,
    int frames,
    int height,
    int width) {
    const long long seq = static_cast<long long>(height) * width;
    const long long total = static_cast<long long>(frames) * seq * 96;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long rem = idx;
        const int d = static_cast<int>(rem % 96);
        rem /= 96;
        const int spatial = static_cast<int>(rem % seq);
        const int t = static_cast<int>(rem / seq);
        const int y = spatial / width;
        const int x = spatial - y * width;
        const int lane = d & 15;
        const int slice = d >> 4;
        const size_t q_base =
            (((static_cast<size_t>(t) * 18 + slice) * height + y) * width + x) *
                kFp8ChannelsPerSlice + lane;
        const size_t k_base =
            (((static_cast<size_t>(t) * 18 + slice + 6) * height + y) * width + x) *
                kFp8ChannelsPerSlice + lane;
        const size_t v_base =
            (((static_cast<size_t>(t) * 18 + slice + 12) * height + y) * width + x) *
                kFp8ChannelsPerSlice + lane;
        q[idx] = qkv[q_base];
        k[idx] = qkv[k_base];
        v[idx] = qkv[v_base];
    }
}

__global__ void bmhd_to_tin16_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int frames,
    int height,
    int width) {
    const long long seq = static_cast<long long>(height) * width;
    const long long total = static_cast<long long>(frames) * seq * 96;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long rem = idx;
        const int d = static_cast<int>(rem % 96);
        rem /= 96;
        const int spatial = static_cast<int>(rem % seq);
        const int t = static_cast<int>(rem / seq);
        const int y = spatial / width;
        const int x = spatial - y * width;
        const int lane = d & 15;
        const int slice = d >> 4;
        const size_t out_idx =
            (((static_cast<size_t>(t) * 6 + slice) * height + y) * width + x) *
                kFp8ChannelsPerSlice + lane;
        output[out_idx] = input[idx];
    }
}

struct LightVaeFp8SdpaKey {
    int batch;
    int seq;
    int head_dim;
    float attn_scale;

    bool operator==(const LightVaeFp8SdpaKey& other) const {
        return batch == other.batch && seq == other.seq && head_dim == other.head_dim &&
               attn_scale == other.attn_scale;
    }
};

struct LightVaeFp8SdpaKeyHash {
    std::size_t operator()(const LightVaeFp8SdpaKey& key) const {
        std::size_t h = 0;
        auto combine_hash = [&](std::size_t value) {
            h ^= value + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        combine_hash(std::hash<int>{}(key.batch));
        combine_hash(std::hash<int>{}(key.seq));
        combine_hash(std::hash<int>{}(key.head_dim));
        combine_hash(std::hash<float>{}(key.attn_scale));
        return h;
    }
};

struct LightVaeFp8SdpaResourceKey {
    int device;
    uintptr_t stream;

    bool operator==(const LightVaeFp8SdpaResourceKey& other) const {
        return device == other.device && stream == other.stream;
    }
};

struct LightVaeFp8SdpaResourceKeyHash {
    std::size_t operator()(const LightVaeFp8SdpaResourceKey& key) const {
        std::size_t h = std::hash<int>{}(key.device);
        h ^= std::hash<uintptr_t>{}(key.stream) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

struct LightVaeFp8SdpaStreamResources {
    std::mutex mutex;
    cudnnHandle_t cudnn_handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_size = 0;
    std::unordered_map<LightVaeFp8SdpaKey,
                       std::shared_ptr<fe::graph::Graph>,
                       LightVaeFp8SdpaKeyHash>
        graph_cache;
};

struct LightVaeFp8SdpaResourceLease {
    std::shared_ptr<LightVaeFp8SdpaStreamResources> resources;
    std::unique_lock<std::mutex> lock;
};

std::mutex g_lightvae_fp8_sdpa_resources_mutex;
std::unordered_map<LightVaeFp8SdpaResourceKey,
                   std::shared_ptr<LightVaeFp8SdpaStreamResources>,
                   LightVaeFp8SdpaResourceKeyHash>
    g_lightvae_fp8_sdpa_resources;
bool g_lightvae_fp8_sdpa_cleanup_in_progress = false;

int current_lightvae_device_ordinal() {
    int device = -1;
    cudaError_t err = cudaGetDevice(&device);
    TORCH_CHECK(err == cudaSuccess, "cudaGetDevice failed for OmniDreams LightVAE FP8 SDPA: ",
                cudaGetErrorString(err));
    TORCH_CHECK(device >= 0, "invalid current CUDA device ordinal ", device);
    return device;
}

LightVaeFp8SdpaResourceLease lightvae_sdpa_resources(cudaStream_t stream) {
    LightVaeFp8SdpaResourceKey key{
        current_lightvae_device_ordinal(),
        reinterpret_cast<uintptr_t>(stream),
    };
    std::unique_lock<std::mutex> map_lock(g_lightvae_fp8_sdpa_resources_mutex);
    TORCH_CHECK(!g_lightvae_fp8_sdpa_cleanup_in_progress,
                "OmniDreams LightVAE FP8 SDPA cleanup is in progress; no new SDPA work can start");
    auto& resources = g_lightvae_fp8_sdpa_resources[key];
    if (!resources) {
        resources = std::make_shared<LightVaeFp8SdpaStreamResources>();
    }
    std::unique_lock<std::mutex> resource_lock(resources->mutex);
    map_lock.unlock();
    return LightVaeFp8SdpaResourceLease{resources, std::move(resource_lock)};
}

cudnnHandle_t lightvae_cudnn_handle(LightVaeFp8SdpaStreamResources& resources, cudaStream_t stream) {
    if (!resources.cudnn_handle) {
        cudnnStatus_t status = cudnnCreate(&resources.cudnn_handle);
        TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnCreate failed for OmniDreams LightVAE FP8 SDPA");
        status = cudnnSetStream(resources.cudnn_handle, stream);
        TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnSetStream failed for OmniDreams LightVAE FP8 SDPA");
    }
    return resources.cudnn_handle;
}

cudaError_t ensure_lightvae_sdpa_workspace(LightVaeFp8SdpaStreamResources& resources, size_t bytes) {
    if (bytes <= resources.workspace_size) {
        return cudaSuccess;
    }
    if (resources.workspace) {
        cudaFree(resources.workspace);
        resources.workspace = nullptr;
        resources.workspace_size = 0;
    }
    cudaError_t err = cudaMalloc(&resources.workspace, bytes);
    if (err != cudaSuccess) {
        return err;
    }
    resources.workspace_size = bytes;
    return cudaSuccess;
}

cudaError_t run_cudnn_fp8_sdpa_bmhd(
    const uint8_t* q,
    const uint8_t* k,
    const uint8_t* v,
    uint8_t* o,
    const float* descale_q,
    const float* descale_k,
    const float* descale_v,
    const float* descale_s,
    const float* scale_s,
    const float* scale_o,
    float* amax_s,
    float* amax_o,
    int batch,
    int seq,
    int head_dim,
    float attn_scale,
    cudaStream_t stream) {
    if (!q || !k || !v || !o || !descale_q || !descale_k || !descale_v ||
        !descale_s || !scale_s || !scale_o || !amax_s || !amax_o ||
        batch <= 0 || seq <= 0 || head_dim <= 0) {
        return cudaErrorInvalidValue;
    }

    auto lease = lightvae_sdpa_resources(stream);
    auto& resources = *lease.resources;
    auto handle = lightvae_cudnn_handle(resources, stream);

    LightVaeFp8SdpaKey cache_key{batch, seq, head_dim, attn_scale};
    auto it = resources.graph_cache.find(cache_key);
    std::shared_ptr<fe::graph::Graph> graph;
    bool cache_miss = false;

    if (it != resources.graph_cache.end()) {
        graph = it->second;
    } else {
        cache_miss = true;
        graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::FP8_E4M3)
             .set_intermediate_data_type(fe::DataType_t::FLOAT)
             .set_compute_data_type(fe::DataType_t::FLOAT);

        const int H = 1;
        const int64_t hidden = int64_t(H) * head_dim;
        const int64_t batch_stride = int64_t(seq) * hidden;
        const int64_t head_stride = head_dim;
        const int64_t seq_stride = hidden;
        auto tQ = graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("Q").set_uid(1)
                                    .set_dim({batch, H, seq, head_dim})
                                    .set_stride({batch_stride, head_stride, seq_stride, 1})
                                    .set_data_type(fe::DataType_t::FP8_E4M3));
        auto tK = graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("K").set_uid(2)
                                    .set_dim({batch, H, seq, head_dim})
                                    .set_stride({batch_stride, head_stride, seq_stride, 1})
                                    .set_data_type(fe::DataType_t::FP8_E4M3));
        auto tV = graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("V").set_uid(3)
                                    .set_dim({batch, H, seq, head_dim})
                                    .set_stride({batch_stride, head_stride, seq_stride, 1})
                                    .set_data_type(fe::DataType_t::FP8_E4M3));

        auto make_scalar = [&](const char* name, int64_t uid) {
            return graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name(name).set_uid(uid)
                                     .set_dim({1, 1, 1, 1})
                                     .set_stride({1, 1, 1, 1})
                                     .set_data_type(fe::DataType_t::FLOAT));
        };
        auto descale_q_t = make_scalar("Descale_Q", 5);
        auto descale_k_t = make_scalar("Descale_K", 6);
        auto descale_v_t = make_scalar("Descale_V", 7);
        auto descale_s_t = make_scalar("Descale_S", 8);
        auto scale_s_t = make_scalar("Scale_S", 9);
        auto scale_o_t = make_scalar("Scale_O", 10);

        auto sdpa_opts = fe::graph::SDPA_fp8_attributes()
                             .set_name("omnidreams_lightvae_middle_sdpa_fp8")
                             .set_generate_stats(false)
                             .set_causal_mask(false)
                             .set_attn_scale(attn_scale);

        auto [tO, tStats, tAmaxS, tAmaxO] = graph->sdpa_fp8(
            tQ, tK, tV,
            descale_q_t, descale_k_t, descale_v_t, descale_s_t,
            scale_s_t, scale_o_t,
            sdpa_opts);
        if (tStats != nullptr) {
            return cudaErrorNotSupported;
        }
        tO->set_output(true)
            .set_dim({batch, H, seq, head_dim})
            .set_stride({batch_stride, head_stride, seq_stride, 1})
            .set_data_type(fe::DataType_t::FP8_E4M3)
            .set_uid(4);
        tAmaxS->set_output(true)
            .set_dim({1, 1, 1, 1})
            .set_stride({1, 1, 1, 1})
            .set_data_type(fe::DataType_t::FLOAT)
            .set_uid(11);
        tAmaxO->set_output(true)
            .set_dim({1, 1, 1, 1})
            .set_stride({1, 1, 1, 1})
            .set_data_type(fe::DataType_t::FLOAT)
            .set_uid(12);

        if (!graph->build(handle,
                          {fe::HeurMode_t::A, fe::HeurMode_t::B, fe::HeurMode_t::FALLBACK},
                          fe::BuildPlanPolicy_t::ALL).is_good()) {
            return cudaErrorNotSupported;
        }
    }

    int64_t workspace_size = 0;
    if (!graph->get_workspace_size(workspace_size).is_good()) {
        return cudaErrorUnknown;
    }
    cudaError_t ws_err = ensure_lightvae_sdpa_workspace(resources, static_cast<size_t>(workspace_size));
    if (ws_err != cudaSuccess) {
        return ws_err;
    }

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
        {1, const_cast<uint8_t*>(q)},
        {2, const_cast<uint8_t*>(k)},
        {3, const_cast<uint8_t*>(v)},
        {4, o},
        {5, const_cast<float*>(descale_q)},
        {6, const_cast<float*>(descale_k)},
        {7, const_cast<float*>(descale_v)},
        {8, const_cast<float*>(descale_s)},
        {9, const_cast<float*>(scale_s)},
        {10, const_cast<float*>(scale_o)},
        {11, amax_s},
        {12, amax_o},
    };

    if (!graph->execute(handle, variant_pack, resources.workspace).is_good()) {
        if (!cache_miss) {
            resources.graph_cache.erase(cache_key);
        }
        return cudaErrorUnknown;
    }
    if (cache_miss) {
        resources.graph_cache[cache_key] = graph;
    }
    return cudaSuccess;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> lightvae_fp8_qkv_tin16_to_bmhd(
    torch::Tensor qkv) {
    check_tin16_cuda_contiguous(qkv, "qkv");
    TORCH_CHECK(qkv.size(1) == 18, "qkv must have 288 channels in TIN16 layout, got ", qkv.sizes());
    at::cuda::CUDAGuard guard(qkv.device());
    auto qkv_c = qkv.contiguous();
    const int frames = static_cast<int>(qkv_c.size(0));
    const int height = static_cast<int>(qkv_c.size(2));
    const int width = static_cast<int>(qkv_c.size(3));
    const long long elems = static_cast<long long>(frames) * height * width * 96;
    auto q = torch::empty({elems}, qkv_c.options());
    auto k = torch::empty({elems}, qkv_c.options());
    auto v = torch::empty({elems}, qkv_c.options());
    if (frames <= 0 || height <= 0 || width <= 0 || elems == 0) {
        return std::make_tuple(q, k, v);
    }
    qkv_tin16_to_bmhd_kernel<<<launch_blocks(elems), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        qkv_c.data_ptr<uint8_t>(),
        q.data_ptr<uint8_t>(),
        k.data_ptr<uint8_t>(),
        v.data_ptr<uint8_t>(),
        frames,
        height,
        width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(q, k, v);
}

torch::Tensor lightvae_fp8_bmhd_to_tin16(
    torch::Tensor input,
    int frames,
    int height,
    int width) {
    check_u8_cuda_contiguous(input, "input");
    TORCH_CHECK(frames > 0 && height > 0 && width > 0, "frames/height/width must be positive");
    const long long elems = static_cast<long long>(frames) * height * width * 96;
    TORCH_CHECK(input.numel() == elems, "input has ", input.numel(), " elements, expected ", elems);
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    auto output = torch::empty({frames, 6, height, width, kFp8ChannelsPerSlice}, input.options());
    bmhd_to_tin16_kernel<<<launch_blocks(elems), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        input_c.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        frames,
        height,
        width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor lightvae_fp8_sdpa_bmhd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor qkv_scale,
    torch::Tensor sdpa_inverse_scale,
    torch::Tensor unit_scale,
    int batch,
    int seq,
    double attn_scale) {
    check_u8_cuda_contiguous(q, "q");
    check_u8_cuda_contiguous(k, "k");
    check_u8_cuda_contiguous(v, "v");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q/k/v shape mismatch");
    TORCH_CHECK(k.device() == q.device() && v.device() == q.device(),
                "q/k/v must be on the same CUDA device; q=", q.device(),
                " k=", k.device(), " v=", v.device());
    check_float_scalar_cuda_contiguous(qkv_scale, "qkv_scale");
    check_float_scalar_cuda_contiguous(sdpa_inverse_scale, "sdpa_inverse_scale");
    check_float_scalar_cuda_contiguous(unit_scale, "unit_scale");
    TORCH_CHECK(qkv_scale.device() == q.device() && sdpa_inverse_scale.device() == q.device() &&
                    unit_scale.device() == q.device(),
                "all scale tensors must be on the q device");
    const int head_dim = 96;
    TORCH_CHECK(batch > 0 && seq > 0, "batch and seq must be positive");
    TORCH_CHECK(q.numel() == static_cast<int64_t>(batch) * seq * head_dim,
                "q/k/v numel must equal batch*seq*96; got ", q.numel(),
                " for batch=", batch, " seq=", seq);
    at::cuda::CUDAGuard guard(q.device());
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto output = torch::empty_like(q_c);
    auto amax_s = torch::empty({1, 1, 1, 1}, q.options().dtype(at::kFloat));
    auto amax_o = torch::empty({1, 1, 1, 1}, q.options().dtype(at::kFloat));

    cudaError_t err = run_cudnn_fp8_sdpa_bmhd(
        q_c.data_ptr<uint8_t>(),
        k_c.data_ptr<uint8_t>(),
        v_c.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        qkv_scale.data_ptr<float>(),
        qkv_scale.data_ptr<float>(),
        qkv_scale.data_ptr<float>(),
        unit_scale.data_ptr<float>(),
        unit_scale.data_ptr<float>(),
        sdpa_inverse_scale.data_ptr<float>(),
        amax_s.data_ptr<float>(),
        amax_o.data_ptr<float>(),
        batch,
        seq,
        head_dim,
        static_cast<float>(attn_scale),
        at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(err == cudaSuccess, "LightVAEEncoderFP8 cuDNN FP8 SDPA failed: ", cudaGetErrorString(err));
    return output;
}

void lightvae_fp8_sdpa_cleanup() {
    std::lock_guard<std::mutex> map_lock(g_lightvae_fp8_sdpa_resources_mutex);
    g_lightvae_fp8_sdpa_cleanup_in_progress = true;
    for (auto& item : g_lightvae_fp8_sdpa_resources) {
        const int device = item.first.device;
        auto& resources = *item.second;
        std::unique_lock<std::mutex> resource_lock(resources.mutex);
        at::cuda::CUDAGuard guard(c10::Device(c10::DeviceType::CUDA, device));
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(item.first.stream));
        resources.graph_cache.clear();
        if (resources.workspace) {
            cudaFree(resources.workspace);
            resources.workspace = nullptr;
        }
        resources.workspace_size = 0;
        if (resources.cudnn_handle) {
            cudnnDestroy(resources.cudnn_handle);
            resources.cudnn_handle = nullptr;
        }
    }
    g_lightvae_fp8_sdpa_resources.clear();
    g_lightvae_fp8_sdpa_cleanup_in_progress = false;
}

}  // namespace omnidreams_singleview
