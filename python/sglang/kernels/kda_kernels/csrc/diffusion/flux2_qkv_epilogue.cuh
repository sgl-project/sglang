// KDA provenance: Humanize2 / Kernel Design Agents, SGLang PR #37162.
#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <algorithm>
#include <cstdint>

namespace sglang {

namespace flux2_qkv_epilogue {

constexpr int kHeadDim = 128;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / device::kWarpThreads;
constexpr int kElemsPerThread = kHeadDim / device::kWarpThreads;
constexpr int kVecSize = kElemsPerThread / 2;

struct Params {
  void* joint_q;
  void* joint_k;
  void* joint_v;
  const void* img_q;
  const void* img_k;
  const void* img_v;
  const void* txt_q;
  const void* txt_k;
  const void* txt_v;
  const void* img_q_weight;
  const void* img_k_weight;
  const void* txt_q_weight;
  const void* txt_k_weight;
  const void* cos_sin_cache;
  int64_t input_token_stride_bytes;
  int64_t output_token_stride_bytes;
  int64_t head_stride_bytes;
  uint32_t img_tokens;
  uint32_t txt_tokens;
  uint32_t num_heads;
  float img_eps;
  float txt_eps;
};

__global__ void flux2_qkv_epilogue_kernel(const Params __grid_constant__ params) {
  using namespace device;
  using Packed = packed_t<bf16_t>;
  using Storage = AlignedVector<Packed, kVecSize>;

  const uint32_t lane = threadIdx.x % kWarpThreads;
  const uint32_t warp = threadIdx.x / kWarpThreads;
  const uint32_t start = blockIdx.x * kWarps + warp;
  const uint32_t workers = gridDim.x * kWarps;
  const uint32_t total_tokens = params.txt_tokens + params.img_tokens;
  const uint32_t token_head_works = total_tokens * params.num_heads;
  const uint32_t total_works = 3 * token_head_works;

  for (uint32_t work = start; work < total_works; work += workers) {
    const uint32_t kind = work / token_head_works;  // 0: Q, 1: K, 2: V.
    const uint32_t token_head = work % token_head_works;
    const uint32_t joint_token = token_head / params.num_heads;
    const uint32_t head = token_head % params.num_heads;
    const bool is_text = joint_token < params.txt_tokens;
    const uint32_t source_token = is_text ? joint_token : joint_token - params.txt_tokens;

    const void* input_base;
    void* output_base;
    if (kind == 0) {
      input_base = is_text ? params.txt_q : params.img_q;
      output_base = params.joint_q;
    } else if (kind == 1) {
      input_base = is_text ? params.txt_k : params.img_k;
      output_base = params.joint_k;
    } else {
      input_base = is_text ? params.txt_v : params.img_v;
      output_base = params.joint_v;
    }

    const void* input =
        pointer::offset(input_base, source_token * params.input_token_stride_bytes, head * params.head_stride_bytes);
    void* output =
        pointer::offset(output_base, joint_token * params.output_token_stride_bytes, head * params.head_stride_bytes);

    auto input_vec = load_as<Storage>(input, lane);
    if (kind == 2) {
      store_as<Storage>(output, input_vec, lane);
      continue;
    }

    const void* weight_base;
    if (kind == 0) {
      weight_base = is_text ? params.txt_q_weight : params.img_q_weight;
    } else {
      weight_base = is_text ? params.txt_k_weight : params.img_k_weight;
    }
    const auto weight_vec = load_as<Storage>(weight_base, lane);

    float elems[kElemsPerThread];
    float sum_of_squares = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      const auto [x0, x1] = cast<fp32x2_t>(input_vec[j]);
      elems[2 * j] = x0;
      elems[2 * j + 1] = x1;
      sum_of_squares += x0 * x0 + x1 * x1;
    }
    sum_of_squares = warp::reduce_sum(sum_of_squares);
    const float eps = is_text ? params.txt_eps : params.img_eps;
    const float norm_factor = math::rsqrt(sum_of_squares / static_cast<float>(kHeadDim) + eps);

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      const auto [w0, w1] = cast<fp32x2_t>(weight_vec[j]);
      elems[2 * j] *= norm_factor * w0;
      elems[2 * j + 1] *= norm_factor * w1;
    }

    const auto* cache = static_cast<const float*>(params.cos_sin_cache);
    const auto* cos_ptr = cache + joint_token * kHeadDim;
    const auto* sin_ptr = cos_ptr + kHeadDim / 2;
#pragma unroll
    for (uint32_t i = 0; i < kElemsPerThread; i += 2) {
      const float x = elems[i];
      const float y = elems[i + 1];
      const uint32_t cache_idx = (lane * kElemsPerThread + i) / 2;
      const float cos = __ldg(cos_ptr + cache_idx);
      const float sin = __ldg(sin_ptr + cache_idx);
      elems[i] = x * cos - y * sin;
      elems[i + 1] = y * cos + x * sin;
    }

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      input_vec[j] = cast<Packed, fp32x2_t>({elems[2 * j], elems[2 * j + 1]});
    }
    store_as<Storage>(output, input_vec, lane);
  }
}

struct Flux2QKVEpilogueKernel {
  static void
  run(tvm::ffi::TensorView joint_q,
      tvm::ffi::TensorView joint_k,
      tvm::ffi::TensorView joint_v,
      tvm::ffi::TensorView img_q,
      tvm::ffi::TensorView img_k,
      tvm::ffi::TensorView img_v,
      tvm::ffi::TensorView txt_q,
      tvm::ffi::TensorView txt_k,
      tvm::ffi::TensorView txt_v,
      tvm::ffi::TensorView img_q_weight,
      tvm::ffi::TensorView img_k_weight,
      tvm::ffi::TensorView txt_q_weight,
      tvm::ffi::TensorView txt_k_weight,
      tvm::ffi::TensorView cos_sin_cache,
      double img_eps,
      double txt_eps) {
    using namespace host;

    auto NI = SymbolicSize{"img_tokens"};
    auto NT = SymbolicSize{"txt_tokens"};
    auto N = SymbolicSize{"joint_tokens"};
    auto H = SymbolicSize{"num_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto device = SymbolicDevice{};
    D.set_value(kHeadDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({NI, H, D})
        .with_strides({-1, D, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(img_q)
        .verify(img_k)
        .verify(img_v);
    TensorMatcher({NT, H, D})
        .with_strides({-1, D, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(txt_q)
        .verify(txt_k)
        .verify(txt_v);
    N.set_value(NI.unwrap() + NT.unwrap());
    TensorMatcher({N, H, D}).with_dtype<bf16_t>().with_device(device).verify(joint_q).verify(joint_k).verify(joint_v);
    TensorMatcher({D})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(img_q_weight)
        .verify(img_k_weight)
        .verify(txt_q_weight)
        .verify(txt_k_weight);
    TensorMatcher({-1, D}).with_dtype<fp32_t>().with_device(device).verify(cos_sin_cache);

    RuntimeCheck(
        img_q.stride(0) == img_k.stride(0) && img_q.stride(0) == img_v.stride(0),
        "image QKV inputs must use the same token stride");
    RuntimeCheck(
        txt_q.stride(0) == txt_k.stride(0) && txt_q.stride(0) == txt_v.stride(0),
        "text QKV inputs must use the same token stride");
    RuntimeCheck(img_q.stride(0) == txt_q.stride(0), "image/text QKV token strides must match");
    RuntimeCheck(
        img_q.stride(1) == kHeadDim && img_k.stride(1) == kHeadDim && img_v.stride(1) == kHeadDim,
        "image QKV heads must be contiguous");
    RuntimeCheck(
        txt_q.stride(1) == kHeadDim && txt_k.stride(1) == kHeadDim && txt_v.stride(1) == kHeadDim,
        "text QKV heads must be contiguous");
    RuntimeCheck(joint_q.is_contiguous(), "joint QKV outputs must be contiguous");
    RuntimeCheck(joint_k.is_contiguous(), "joint QKV outputs must be contiguous");
    RuntimeCheck(joint_v.is_contiguous(), "joint QKV outputs must be contiguous");
    RuntimeCheck(cos_sin_cache.is_contiguous(), "cos/sin cache must be contiguous");
    RuntimeCheck(cos_sin_cache.size(0) >= N.unwrap(), "cos/sin cache does not cover all joint tokens");

    const uint32_t img_tokens = static_cast<uint32_t>(NI.unwrap());
    const uint32_t txt_tokens = static_cast<uint32_t>(NT.unwrap());
    const uint32_t num_heads = static_cast<uint32_t>(H.unwrap());
    const uint32_t total_works = 3 * (img_tokens + txt_tokens) * num_heads;
    if (total_works == 0) return;

    const int64_t head_stride_bytes = kHeadDim * sizeof(bf16_t);
    const int64_t input_token_stride_bytes = img_q.stride(0) * sizeof(bf16_t);
    const int64_t output_token_stride_bytes = num_heads * head_stride_bytes;
    const auto params = Params{
        .joint_q = joint_q.data_ptr(),
        .joint_k = joint_k.data_ptr(),
        .joint_v = joint_v.data_ptr(),
        .img_q = img_q.data_ptr(),
        .img_k = img_k.data_ptr(),
        .img_v = img_v.data_ptr(),
        .txt_q = txt_q.data_ptr(),
        .txt_k = txt_k.data_ptr(),
        .txt_v = txt_v.data_ptr(),
        .img_q_weight = img_q_weight.data_ptr(),
        .img_k_weight = img_k_weight.data_ptr(),
        .txt_q_weight = txt_q_weight.data_ptr(),
        .txt_k_weight = txt_k_weight.data_ptr(),
        .cos_sin_cache = cos_sin_cache.data_ptr(),
        .input_token_stride_bytes = input_token_stride_bytes,
        .output_token_stride_bytes = output_token_stride_bytes,
        .head_stride_bytes = head_stride_bytes,
        .img_tokens = img_tokens,
        .txt_tokens = txt_tokens,
        .num_heads = num_heads,
        .img_eps = static_cast<float>(img_eps),
        .txt_eps = static_cast<float>(txt_eps),
    };

    const uint32_t sm_count = runtime::get_sm_count(device.unwrap().device_id);
    static const uint32_t blocks_per_sm = runtime::get_blocks_per_sm(flux2_qkv_epilogue_kernel, kThreads);
    const uint32_t needed_blocks = div_ceil(total_works, uint32_t(kWarps));
    const uint32_t blocks = std::min(blocks_per_sm * sm_count, needed_blocks);
    LaunchKernel(blocks, kThreads, device.unwrap())(flux2_qkv_epilogue_kernel, params);
  }
};

}  // namespace flux2_qkv_epilogue

}  // namespace sglang
