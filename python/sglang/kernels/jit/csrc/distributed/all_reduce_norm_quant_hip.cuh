// Built on AITER's custom all-reduce (csrc/include/custom_all_reduce.cuh): the TP4 decode layer boundary of
// MiniMax-M3.1 (hidden 6144) in one kernel for the smallest batches -- every rank pulls all ranks' rows and
// sums them, then adds the residual, applies the Gemma RMSNorm and quantizes per token to fp8 for the next
// GEMM, on the peer buffers and signals of AITER's CustomAllreduce. The add, norm and quant follow
// gemma_add_rmsnorm_fp8 (kernels/ops/layernorm/minimax_m3_rmsnorm.py): the sum is rounded to bf16 before the
// residual add, as the all-reduce output is, and the norm is rounded to bf16 before the quant.
#pragma once

#ifndef USE_ROCM
#error "all_reduce_norm_quant_hip.cuh runs on AITER's ROCm custom all-reduce"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <custom_all_reduce.cuh>  // AITER csrc/include: CustomAllreduce, RankData, start_sync, end_sync

namespace sglang {
namespace all_reduce_norm_quant_hip {

constexpr int kWorldSize = 4;
constexpr int kHidden = 6144;
constexpr int kPack = 8;                   // bf16 per 16-byte packet
constexpr int kThreads = kHidden / kPack;  // one packet of a row per thread
constexpr int kWaves = kThreads / 64;
constexpr float kFp8Max = 448.0f;
static_assert(kWaves * 64 == kThreads, "a row splits into whole waves");

using Packet = uint4;

// bf16 is the top half of an fp32: bit operations, so no packet needs an address (it would go to scratch).
__device__ __forceinline__ void unpack(const Packet& p, float (&v)[kPack]) {
  const uint32_t words[kPack / 2] = {p.x, p.y, p.z, p.w};
#pragma unroll
  for (int k = 0; k < kPack / 2; ++k) {
    v[2 * k] = __uint_as_float(words[k] << 16);
    v[2 * k + 1] = __uint_as_float(words[k] & 0xFFFF0000u);
  }
}

__device__ __forceinline__ uint32_t bf16_bits(float v) {
  return __builtin_bit_cast(uint16_t, __float2bfloat16(v));
}

__device__ __forceinline__ Packet pack(const float (&v)[kPack]) {
  return make_uint4(
      bf16_bits(v[0]) | bf16_bits(v[1]) << 16,
      bf16_bits(v[2]) | bf16_bits(v[3]) << 16,
      bf16_bits(v[4]) | bf16_bits(v[5]) << 16,
      bf16_bits(v[6]) | bf16_bits(v[7]) << 16);
}

// Rounds through bf16, as storing the value and reading it back does.
__device__ __forceinline__ float round_bf16(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

// Sum (or max) over the block; every thread gets the result. `slots` holds one value per wave.
template <bool kMax>
__device__ __forceinline__ float block_reduce(float v, float* slots) {
#pragma unroll
  for (int mask = 32; mask > 0; mask >>= 1) {
    const float other = __shfl_xor(v, mask, 64);
    v = kMax ? fmaxf(v, other) : v + other;
  }
  if (threadIdx.x % 64 == 0) slots[threadIdx.x / 64] = v;
  __syncthreads();
  float total = slots[0];
#pragma unroll
  for (int w = 1; w < kWaves; ++w)
    total = kMax ? fmaxf(total, slots[w]) : total + slots[w];
  __syncthreads();  // the slots are reused by the next reduction
  return total;
}

// fp8 e4m3fn of four values already clamped to +-448, round to nearest even (gfx950's OCP flavour).
__device__ __forceinline__ uint32_t fp8x4(float a, float b, float c, float d) {
  int packed = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
  packed = __builtin_amdgcn_cvt_pk_fp8_f32(c, d, packed, true);
  return static_cast<uint32_t>(packed);
}

template <bool kKeepBf16>
__global__ void __launch_bounds__(kThreads) all_reduce_norm_quant_kernel(
    aiter::RankData* peers,
    aiter::RankSignals signals,
    aiter::Signal* local_signals,
    int rank,
    int m,
    const Packet* __restrict__ residual,
    const Packet* __restrict__ weight,
    float eps,
    Packet* __restrict__ residual_out,
    uint2* __restrict__ q_out,
    float* __restrict__ scale_out,
    Packet* __restrict__ norm_out) {
  __shared__ float slots[kWaves];
  const int t = threadIdx.x;
  const Packet* inputs[kWorldSize];
#pragma unroll
  for (int r = 0; r < kWorldSize; ++r)
    inputs[r] = reinterpret_cast<const Packet*>(peers->ptrs[r]);
  // The weight and the first row's residual load while the ranks meet.
  float gamma[kPack];
  unpack(weight[t], gamma);
#pragma unroll
  for (int j = 0; j < kPack; ++j)
    gamma[j] += 1.0f;
  const Packet first_residual = residual[blockIdx.x * kThreads + t];
  aiter::start_sync<kWorldSize>(signals, local_signals, rank);

  for (int row = blockIdx.x; row < m; row += gridDim.x) {
    const int index = row * kThreads + t;
    Packet values[kWorldSize];
#pragma unroll
    for (int r = 0; r < kWorldSize; ++r)
      values[r] = inputs[r][index];
    // In rank order, as the all-reduce sums.
    float s[kPack], v[kPack];
    unpack(values[0], s);
#pragma unroll
    for (int r = 1; r < kWorldSize; ++r) {
      unpack(values[r], v);
#pragma unroll
      for (int j = 0; j < kPack; ++j)
        s[j] += v[j];
    }
    unpack(row == blockIdx.x ? first_residual : residual[index], v);
    float sum_sq = 0.0f;
#pragma unroll
    for (int j = 0; j < kPack; ++j) {
      s[j] = round_bf16(s[j]) + v[j];
      sum_sq += s[j] * s[j];
    }
    residual_out[index] = pack(s);
    const float rstd = 1.0f / sqrtf(block_reduce<false>(sum_sq, slots) / kHidden + eps);
    float y[kPack];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < kPack; ++j) {
      y[j] = round_bf16(s[j] * rstd * gamma[j]);
      amax = fmaxf(amax, fabsf(y[j]));
    }
    if constexpr (kKeepBf16) norm_out[index] = pack(y);
    // aiter's per-token quant: v_rcp_f32 of the scale.
    const float scale = block_reduce<true>(amax, slots) * (1.0f / kFp8Max);
    const float inv = __builtin_amdgcn_rcpf(scale);
    if (t == 0) scale_out[row] = scale;
#pragma unroll
    for (int j = 0; j < kPack; ++j)
      y[j] = fminf(fmaxf(y[j] * inv, -kFp8Max), kFp8Max);
    q_out[index] = make_uint2(fp8x4(y[0], y[1], y[2], y[3]), fp8x4(y[4], y[5], y[6], y[7]));
  }
  // Peers read this rank's input to the end: no rank may reuse its input before they are done.
  aiter::end_sync<kWorldSize, true>(signals, local_signals, rank);
}

struct AllReduceNormQuant {
  /// \brief residual_out, q_out and scale_out (and norm_out) = the boundary of the bf16 [m, 6144] partial sums
  ///   `input` with `residual` and the Gemma weight `weight` (applied as 1 + weight), quantized per token.
  /// \param comm AITER's CustomAllreduce*.
  /// \param q_out fp8 e4m3fn [m, 6144], passed as its bytes.
  /// \param scale_out fp32, m of them.
  /// \param registered AITER's registered input pool, into which the input is staged outside graph capture
  ///   (0 while capturing: capture registers the input itself).
  static void
  run(int64_t comm,
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView residual_out,
      const tvm::ffi::TensorView q_out,
      const tvm::ffi::TensorView scale_out,
      const tvm::ffi::TensorView norm_out,
      double eps,
      bool keep_bf16,
      int64_t registered,
      int64_t registered_bytes) {
    using namespace host;
    auto M = SymbolicSize{"num_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, kHidden})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(input)
        .verify(residual)
        .verify(residual_out);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device).verify(weight);
    TensorMatcher({M, kHidden}).with_dtype<uint8_t>().with_device(device).verify(q_out);
    const int64_t rows = M.unwrap();
    RuntimeCheck(rows >= 1, "all_reduce_norm_quant needs at least one row");
    RuntimeCheck(scale_out.numel() == rows && scale_out.is_contiguous(), "scale_out must hold ", rows, " floats");
    if (keep_bf16) TensorMatcher({M, kHidden}).with_dtype<bf16_t>().with_device(device).verify(norm_out);

    auto* communicator = reinterpret_cast<aiter::CustomAllreduce*>(comm);
    RuntimeCheck(communicator->world_size_ == kWorldSize, "all_reduce_norm_quant requires TP", kWorldSize);
    const auto stream = LaunchKernel::resolve_device(device.unwrap());
    void* values = input.data_ptr();
    if (registered != 0) {
      const int64_t bytes = rows * kHidden * static_cast<int64_t>(sizeof(bf16_t));
      RuntimeCheck(bytes <= registered_bytes, "all-reduce input pool is too small");
      values = reinterpret_cast<void*>(registered);
      RuntimeDeviceCheck(hipMemcpyAsync(values, input.data_ptr(), bytes, hipMemcpyDeviceToDevice, stream));
    }
    aiter::RankData* peers = communicator->get_buffer_RD(stream, values);
    const auto blocks = static_cast<uint32_t>(rows < aiter::kMaxBlocks ? rows : aiter::kMaxBlocks);
    auto launch = [&](auto kernel) {
      LaunchKernel(blocks, kThreads, stream)(
          kernel,
          peers,
          communicator->sg_,
          communicator->self_sg_,
          communicator->rank_,
          static_cast<int>(rows),
          static_cast<const Packet*>(residual.data_ptr()),
          static_cast<const Packet*>(weight.data_ptr()),
          static_cast<float>(eps),
          static_cast<Packet*>(residual_out.data_ptr()),
          static_cast<uint2*>(q_out.data_ptr()),
          static_cast<float*>(scale_out.data_ptr()),
          keep_bf16 ? static_cast<Packet*>(norm_out.data_ptr()) : nullptr);
    };
    keep_bf16 ? launch(all_reduce_norm_quant_kernel<true>) : launch(all_reduce_norm_quant_kernel<false>);
  }
};

}  // namespace all_reduce_norm_quant_hip
}  // namespace sglang
