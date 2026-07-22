#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For CHECK_HOST, div_ceil

#include <sgl_kernel/type.cuh>   // For bf16_t, fp32_t, device::cast
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE, PDL helpers, kMaxVecBytes
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <limits>

namespace sglang {

using namespace device;

// K3 MoE finalize: out[t] = sum_k expert_weights[t, k] * gemm2_out[idx[t*16 + k]]
// (idx == -1 slots skipped). Inputs come from the trtllm-gen fused MoE with
// do_finalize=False (see jit_kernel/trtllm_gen_moe.py).
//
// Low-latency-first schedule, unlike the trtllm finalizeKernelVecLoad (one CTA
// per token, 16 gathers serialized behind a single block at decode sizes):
// fixed-size blocks where each thread produces exactly one 16B output vector
// (8 bf16), grid = ceil(T * H/8 / block). The block count scales with total
// work regardless of batch or hidden size, so small-T decode spreads across
// SMs. top_k = 16 is compile-time: the idx row is two 32B vector loads (four
// pre-SM100), the weight row one (two), and the fully unrolled k-loop keeps
// all 16 gathers in flight.

constexpr uint32_t kTopK = 16;
constexpr uint32_t kOutVecElems = 16 / sizeof(bf16_t);
constexpr uint32_t kBlockSize = 128;

// bf16 x bf16 -> fp32 fused multiply-add (same idiom as gemm/tiny_gemm.cuh).
// The bf16 product is exact in fp32, so the fallback is bit-identical.
SGL_DEVICE float fma_f32_bf16(bf16_t a, bf16_t b, float acc) {
#if SGL_ARCH_BLACKWELL_OR_GREATER
  const uint16_t a_bits = __bfloat16_as_ushort(a);
  const uint16_t b_bits = __bfloat16_as_ushort(b);
  float result;
  asm("fma.rn.f32.bf16 %0, %1, %2, %3;" : "=f"(result) : "h"(a_bits), "h"(b_bits), "f"(acc));
  return result;
#else
  return fmaf(cast<fp32_t>(a), cast<fp32_t>(b), acc);
#endif
}

struct K3MoeFinalizeParams {
  const void* __restrict__ gemm2_out;       // [P, H] bf16, permuted layout
  const void* __restrict__ permuted_idx;    // [T * 16] int32, -1 = dropped slot
  const void* __restrict__ expert_weights;  // [T, 16] bf16
  void* __restrict__ out;                   // [T, H] bf16
  uint32_t hidden;                          // H (elements)
  uint32_t vecs_per_token;                  // H / kOutVecElems
  uint32_t num_total_vecs;                  // T * vecs_per_token
};

template <bool kUsePDL>
__global__
__launch_bounds__(kBlockSize) void k3_moe_finalize_kernel(const __grid_constant__ K3MoeFinalizeParams params) {
  using vec_t = AlignedVector<bf16_t, kOutVecElems>;
  constexpr uint32_t kIdxVecSize = kMaxVecBytes / sizeof(int32_t);
  constexpr uint32_t kWVecSize = kMaxVecBytes / sizeof(bf16_t);
  constexpr uint32_t kIdxVecs = kTopK / kIdxVecSize;  // 2 on SM100+, 4 below
  constexpr uint32_t kWVecs = kTopK / kWVecSize;      // 1 on SM100+, 2 below

  const uint32_t vec_id = blockIdx.x * kBlockSize + threadIdx.x;
  if (vec_id >= params.num_total_vecs) return;
  const uint32_t token = vec_id / params.vecs_per_token;
  const uint32_t hvec = vec_id % params.vecs_per_token;

  AlignedVector<int32_t, kIdxVecSize> idx[kIdxVecs];
#pragma unroll
  for (uint32_t j = 0; j < kIdxVecs; ++j) {
    idx[j].load(params.permuted_idx, token * kIdxVecs + j);
  }
  AlignedVector<bf16_t, kWVecSize> weight[kWVecs];
#pragma unroll
  for (uint32_t j = 0; j < kWVecs; ++j) {
    weight[j].load(params.expert_weights, token * kWVecs + j);
  }

  PDLWaitPrimary<kUsePDL>();

  const auto* g2 = static_cast<const bf16_t*>(params.gemm2_out);
  vec_t in[kTopK];
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    const int32_t row = idx[k / kIdxVecSize][k % kIdxVecSize];
    if (row >= 0) {
      in[k].load(g2 + static_cast<int64_t>(row) * params.hidden, hvec);
    }
  }
  float acc[kOutVecElems] = {};
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    const int32_t row = idx[k / kIdxVecSize][k % kIdxVecSize];
    if (row < 0) continue;
    vec_t v = in[k];
    const bf16_t w_k = weight[k / kWVecSize][k % kWVecSize];
#pragma unroll
    for (uint32_t i = 0; i < kOutVecElems; ++i) {
      acc[i] = fma_f32_bf16(v[i], w_k, acc[i]);
    }
  }

  vec_t o;
#pragma unroll
  for (uint32_t i = 0; i < kOutVecElems; ++i) {
    o[i] = cast<bf16_t>(acc[i]);
  }
  o.store(params.out, vec_id);
}

}  // namespace sglang

using namespace sglang;

template <bool kUsePDL>
struct K3MoeFinalizeKernel {
  static void
  run(tvm::ffi::TensorView gemm2_out,
      tvm::ffi::TensorView permuted_idx,
      tvm::ffi::TensorView expert_weights,
      tvm::ffi::TensorView out) {
    using namespace host;

    auto P = SymbolicSize{"num_permuted_rows"};
    auto H = SymbolicSize{"hidden"};
    auto T = SymbolicSize{"num_tokens"};
    auto K = SymbolicSize{"top_k"};
    auto TK = SymbolicSize{"num_expanded"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({P, H})  //
        .with_dtype<bf16_t>()
        .with_device(device_)
        .verify(gemm2_out);
    TensorMatcher({T, H})  //
        .with_dtype<bf16_t>()
        .with_device(device_)
        .verify(out);
    TensorMatcher({T, K})  //
        .with_dtype<bf16_t>()
        .with_device(device_)
        .verify(expert_weights);
    TensorMatcher({TK})  //
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(permuted_idx);

    const auto num_tokens = static_cast<uint32_t>(T.unwrap());
    const auto hidden = static_cast<uint32_t>(H.unwrap());
    const auto device = device_.unwrap();
    CHECK_HOST(K.unwrap() == kTopK) << "K3 finalize is specialized for top_k = " << kTopK;
    CHECK_HOST(TK.unwrap() == static_cast<int64_t>(num_tokens) * kTopK)
        << "permuted_idx must hold num_tokens * " << kTopK << " entries";
    CHECK_HOST(hidden % kOutVecElems == 0) << "hidden must be divisible by " << kOutVecElems;
    if (num_tokens == 0) return;

    const auto total_vecs = static_cast<int64_t>(num_tokens) * (hidden / kOutVecElems);
    CHECK_HOST(total_vecs <= std::numeric_limits<uint32_t>::max()) << "too many items for 32-bit indexing";
    const auto params = K3MoeFinalizeParams{
        .gemm2_out = gemm2_out.data_ptr(),
        .permuted_idx = permuted_idx.data_ptr(),
        .expert_weights = expert_weights.data_ptr(),
        .out = out.data_ptr(),
        .hidden = hidden,
        .vecs_per_token = hidden / kOutVecElems,
        .num_total_vecs = static_cast<uint32_t>(total_vecs),
    };
    const auto num_blocks = host::div_ceil(static_cast<uint32_t>(total_vecs), kBlockSize);
    LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(k3_moe_finalize_kernel<kUsePDL>, params);
  }
};
