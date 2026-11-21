// SPDX-License-Identifier: Apache-2.0
// SM120 routed NVFP4 / MXFP8 quantization of this rank's MoE rows for FlashInfer's
// SM120 grouped GEMMs, with the numerics of routed_quant.cuh (its device helpers):
// - NVFP4 kernels zero each expert's scale tail in the CTA that writes the expert's
//   last row, so the work scales with the routed rows rather than the experts;
// - swiglu kernels read the GEMMs' BF16 outputs;
// - MXFP8 kernels write the CuTe SM120 GEMM's UE8M0 scales: per row, MN-major over
//   m_padded scale rows, four K blocks per 32-bit word, expert e's rows starting at
//   scale row (offsets[e] + 3 e) / 4 * 4.

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include "routed_quant.cuh"
#include <cstdint>
#include <limits>

namespace sglang::inkling2_routed_quant_sm120 {

using namespace sglang::inkling2_routed_quant;

// zero_tails for one expert, by the whole CTA.
SGL_DEVICE void zero_expert_tail(int32_t const* offsets, int32_t expert, uint8_t* x_scales, int32_t scale_k) {
  int32_t const count = offsets[expert + 1] - offsets[expert];
  int32_t const used = count % 128;
  if (used == 0) return;
  int32_t const atoms = scale_k / 4;
  uint8_t* tile = x_scales + scale_base(offsets, expert, scale_k) + int64_t(count / 128) * atoms * 512;
  for (int32_t item = threadIdx.x; item < atoms * 32; item += blockDim.x) {
    int32_t const lane = item % 32;
    uint32_t* atom = reinterpret_cast<uint32_t*>(tile + (item / 32) * 512 + lane * 16);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (lane + 32 * i >= used) atom[i] = 0;
  }
}

template <int K>
__global__ void __launch_bounds__(split_threads(K)) nvfp4_quantize_routed(
    __nv_bfloat16 const* __restrict__ x,
    int32_t const* __restrict__ dst2src,
    uint8_t* values,
    uint8_t* x_scales,
    float* amax_out,
    int32_t const* __restrict__ offsets,
    int32_t top_k,
    int32_t experts) {
  constexpr int kThreads = split_threads(K), kSplit = split(K), E = 16 / kSplit;
  static_assert(K % 64 == 0 && kSplit > 1);
  __shared__ int32_t s_offsets[kMaxExperts + 1];
  __shared__ float red[2][kThreads / 32];
  device::PDLWaitPrimary<true>();
  device::PDLTriggerSecondary<true>();
  load_offsets(offsets, experts, s_offsets);
  int32_t const block = threadIdx.x / kSplit, part = threadIdx.x % kSplit;
  bool const active = block < K / 16;
  int32_t parity = 0;
  for (int32_t row = s_offsets[0] + blockIdx.x; row < s_offsets[experts]; row += gridDim.x, parity ^= 1) {
    int32_t const expert = row_expert(s_offsets, experts, row);
    float v[E];
#pragma unroll
    for (int i = 0; i < E; ++i)
      v[i] = 0.0f;
    if (active) {
      __nv_bfloat16 const* src = x + int64_t(dst2src[row] / top_k) * K + block * 16 + part * E;
      if constexpr (E == 8) {
        uint4 raw = *reinterpret_cast<uint4 const*>(src);
        __nv_bfloat16 const* h = reinterpret_cast<__nv_bfloat16 const*>(&raw);
#pragma unroll
        for (int i = 0; i < E; ++i)
          v[i] = __bfloat162float(h[i]);
      } else {
        uint2 raw = *reinterpret_cast<uint2 const*>(src);
        __nv_bfloat16 const* h = reinterpret_cast<__nv_bfloat16 const*>(&raw);
#pragma unroll
        for (int i = 0; i < E; ++i)
          v[i] = __bfloat162float(h[i]);
      }
    }
    float block_max = part_max<E>(v);
    float amax = block_max_reduce<kThreads>(block_max, red[parity]);
    uint32_t packed;
    uint8_t scale;
    quantize_part<E>(v, amax, block_max, packed, scale);
    if (active) {
      if (threadIdx.x == 0) amax_out[row] = amax;
      store_part<E>(values, x_scales, s_offsets, expert, row, block, part, K, packed, scale);
    }
    if (row == s_offsets[expert + 1] - 1) zero_expert_tail(s_offsets, expert, x_scales, (K + 63) / 64 * 4);
  }
}

template <int N>
__global__ void __launch_bounds__(split_threads(N)) nvfp4_swiglu_quantize_routed(
    __nv_bfloat16 const* __restrict__ acc,
    float const* __restrict__ acc_amax,
    float const* __restrict__ w_amax,
    float const* __restrict__ w_denominator,
    float const* __restrict__ gammas,
    uint8_t* values,
    uint8_t* x_scales,
    float* amax_out,
    int32_t const* __restrict__ offsets,
    int32_t experts,
    float gate_cap,
    float up_cap) {
  constexpr int kThreads = split_threads(N), kSplit = split(N), E = 16 / kSplit;
  static_assert(N % 64 == 0 && kSplit > 1 && E % 4 == 0);
  __shared__ int32_t s_offsets[kMaxExperts + 1];
  __shared__ float red[2][kThreads / 32];
  device::PDLWaitPrimary<true>();
  device::PDLTriggerSecondary<true>();
  load_offsets(offsets, experts, s_offsets);
  int32_t const block = threadIdx.x / kSplit, part = threadIdx.x % kSplit;
  bool const active = block < N / 16;
  float const rcp = __frcp_rn(__fmul_rn(1536.0f, *w_denominator));
  int32_t parity = 0;
  for (int32_t row = s_offsets[0] + blockIdx.x; row < s_offsets[experts]; row += gridDim.x, parity ^= 1) {
    int32_t const expert = row_expert(s_offsets, experts, row);
    float v[E];
#pragma unroll
    for (int i = 0; i < E; ++i)
      v[i] = 0.0f;
    if (active) {
      // 2 E interleaved gate/up BF16 accumulators, 8 per 16-byte load.
      uint4 raw[E / 4];
      uint4 const* src = reinterpret_cast<uint4 const*>(acc + int64_t(row) * 2 * N + block * 32 + part * 2 * E);
#pragma unroll
      for (int i = 0; i < E / 4; ++i)
        raw[i] = src[i];
      __nv_bfloat16 const* a = reinterpret_cast<__nv_bfloat16 const*>(raw);
      float scale = __fmul_rn(__fmul_rn(acc_amax[row], w_amax[expert]), rcp);
      float gamma = gammas[row];
#pragma unroll
      for (int i = 0; i < E; ++i) {
        float gate = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__bfloat162float(a[2 * i]), scale)));
        float up = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__bfloat162float(a[2 * i + 1]), scale)));
        v[i] = __bfloat162float(__float2bfloat16_rn(capped_swiglu(gate, up, gamma, gate_cap, up_cap)));
      }
    }
    float block_max = part_max<E>(v);
    float amax = block_max_reduce<kThreads>(block_max, red[parity]);
    uint32_t packed;
    uint8_t scale_bits;
    quantize_part<E>(v, amax, block_max, packed, scale_bits);
    if (active) {
      if (threadIdx.x == 0) amax_out[row] = amax;
      store_part<E>(values, x_scales, s_offsets, expert, row, block, part, N, packed, scale_bits);
    }
    if (row == s_offsets[expert + 1] - 1) zero_expert_tail(s_offsets, expert, x_scales, (N + 63) / 64 * 4);
  }
}

SGL_DEVICE int64_t
cute_exponent_offset(int32_t row, int32_t group, int32_t const* offsets, int32_t experts, int32_t m_padded) {
  int32_t const expert = row_expert(offsets, experts, row);
  int32_t const scale_row = (offsets[expert] + 3 * expert) / 4 * 4 + row - offsets[expert];
  return (int64_t(group / 4) * m_padded + scale_row) * 4 + group % 4;
}

// The MXFP8 kernels never call launch_dependents: the CuTe SM120 GEMM reading
// their outputs does not wait on its predecessor, so it may launch only when
// they complete.
template <int K>
__global__ void __launch_bounds__(threads(K)) mxfp8_quantize_routed(
    __nv_bfloat16 const* __restrict__ x,
    int32_t const* __restrict__ dst2src,
    uint8_t* values,
    uint8_t* exponents,
    int32_t const* __restrict__ offsets,
    int32_t experts,
    int32_t top_k,
    int32_t m_padded) {
  static_assert(K % 128 == 0 && K / 16 <= 1024);
  device::PDLWaitPrimary<true>();
  int32_t const block = threadIdx.x;
  bool const active = block < K / 16;
  int32_t const lo = offsets[0], hi = offsets[experts];
  for (int32_t row = lo + blockIdx.x; row < hi; row += gridDim.x) {
    float v[16];
    if (active) {
      uint4 const* src = reinterpret_cast<uint4 const*>(x + int64_t(dst2src[row] / top_k) * K + block * 16);
      uint4 raw[2] = {src[0], src[1]};
      __nv_bfloat16 const* h = reinterpret_cast<__nv_bfloat16 const*>(raw);
#pragma unroll
      for (int i = 0; i < 16; ++i)
        v[i] = __bfloat162float(h[i]);
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i)
        v[i] = 0.0f;
    }
    uint4 packed;
    uint8_t exponent;
    mxfp8_half_block(v, packed, exponent);
    if (active) {
      *reinterpret_cast<uint4*>(values + int64_t(row) * K + block * 16) = packed;
      if (block % 2 == 0) exponents[cute_exponent_offset(row, block / 2, offsets, experts, m_padded)] = exponent;
    }
  }
}

template <int N>
__global__ void __launch_bounds__(threads(N)) mxfp8_swiglu_quantize_routed(
    __nv_bfloat16 const* __restrict__ x,
    float const* __restrict__ gammas,
    uint8_t* values,
    uint8_t* exponents,
    int32_t const* __restrict__ offsets,
    int32_t experts,
    float gate_cap,
    float up_cap,
    int32_t m_padded) {
  static_assert(N % 128 == 0 && N / 16 <= 1024);
  device::PDLWaitPrimary<true>();
  int32_t const block = threadIdx.x;
  bool const active = block < N / 16;
  int32_t const lo = offsets[0], hi = offsets[experts];
  for (int32_t row = lo + blockIdx.x; row < hi; row += gridDim.x) {
    float v[16];
    if (active) {
      uint4 const* src = reinterpret_cast<uint4 const*>(x + int64_t(row) * 2 * N + block * 32);
      uint4 raw[4] = {src[0], src[1], src[2], src[3]};
      __nv_bfloat16 const* h = reinterpret_cast<__nv_bfloat16 const*>(raw);
      float gamma = gammas[row];
#pragma unroll
      for (int i = 0; i < 16; ++i)
        v[i] = __bfloat162float(__float2bfloat16_rn(
            capped_swiglu(__bfloat162float(h[2 * i]), __bfloat162float(h[2 * i + 1]), gamma, gate_cap, up_cap)));
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i)
        v[i] = 0.0f;
    }
    uint4 packed;
    uint8_t exponent;
    mxfp8_half_block(v, packed, exponent);
    if (active) {
      *reinterpret_cast<uint4*>(values + int64_t(row) * N + block * 16) = packed;
      if (block % 2 == 0) exponents[cute_exponent_offset(row, block / 2, offsets, experts, m_padded)] = exponent;
    }
  }
}

// The [experts + 1] int32 expert offsets; returns the expert count.
inline int32_t verify_offsets(tvm::ffi::TensorView offsets, host::SymbolicDevice& device) {
  auto bounds = host::SymbolicSize{"experts + 1"};
  host::TensorMatcher({bounds}).with_dtype<int32_t>().with_device(device).verify(offsets);
  int64_t const experts = bounds.unwrap() - 1;
  CHECK_HOST(experts > 0 && experts <= kMaxExperts)
      << "routed quantization covers 1 to " << kMaxExperts << " experts, got " << experts;
  return static_cast<int32_t>(experts);
}

// The kernels index routed rows with int32_t; returns the row count.
inline int32_t verify_rows(host::SymbolicSize& rows) {
  CHECK_HOST(rows.unwrap() < std::numeric_limits<int32_t>::max()) << "too many routed rows: " << rows.unwrap();
  return static_cast<int32_t>(rows.unwrap());
}

template <int K>
void nvfp4_quantize(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView dst2src,
    tvm::ffi::TensorView values,
    tvm::ffi::TensorView x_scales,
    tvm::ffi::TensorView amax,
    tvm::ffi::TensorView offsets,
    int64_t top_k,
    int64_t sm_count) {
  constexpr int64_t kPart = 16 / split(K);  // BF16 values each lane loads and quantizes
  auto tokens = host::SymbolicSize{"tokens"};
  auto rows = host::SymbolicSize{"routed rows"};
  auto scale_rows = host::SymbolicSize{"scale rows"};
  auto device = host::SymbolicDevice{};
  device.set_options<kDLCUDA>();
  int32_t const experts = verify_offsets(offsets, device);
  host::TensorMatcher({tokens, K})  //
      .with_dtype<bf16_t>()
      .with_device(device)
      .ensure_alignment(kPart * 2)
      .verify(x);
  host::TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(dst2src);
  host::TensorMatcher({rows, K / 2})  //
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(kPart / 2)
      .verify(values);
  host::TensorMatcher({scale_rows, K / 16}).with_dtype<uint8_t>().with_device(device).verify(x_scales);
  host::TensorMatcher({rows}).with_dtype<fp32_t>().with_device(device).verify(amax);
  CHECK_HOST(top_k > 0) << "top_k must be positive, got " << top_k;
  int32_t const num_rows = verify_rows(rows);
  CHECK_HOST(scale_rows.unwrap() >= host::div_ceil(num_rows + 127 * int64_t{experts}, 128) * 128)
      << "NVFP4 scales need 128-row blocks per expert, got " << scale_rows.unwrap() << " rows";
  host::LaunchKernel(grid_size(num_rows, static_cast<int32_t>(sm_count)), split_threads(K), device.unwrap())
      .enable_pdl(true)(
          nvfp4_quantize_routed<K>,
          static_cast<__nv_bfloat16 const*>(x.data_ptr()),
          static_cast<int32_t const*>(dst2src.data_ptr()),
          static_cast<uint8_t*>(values.data_ptr()),
          static_cast<uint8_t*>(x_scales.data_ptr()),
          static_cast<float*>(amax.data_ptr()),
          static_cast<int32_t const*>(offsets.data_ptr()),
          static_cast<int32_t>(top_k),
          experts);
}

template <int N>
void nvfp4_swiglu_quantize(
    tvm::ffi::TensorView acc,
    tvm::ffi::TensorView acc_amax,
    tvm::ffi::TensorView w_amax,
    tvm::ffi::TensorView w_denominator,
    tvm::ffi::TensorView gammas,
    tvm::ffi::TensorView values,
    tvm::ffi::TensorView x_scales,
    tvm::ffi::TensorView amax,
    tvm::ffi::TensorView offsets,
    double gate_cap,
    double up_cap,
    int64_t sm_count) {
  constexpr int64_t kPart = 16 / split(N);  // outputs each lane quantizes
  auto rows = host::SymbolicSize{"routed rows"};
  auto scale_rows = host::SymbolicSize{"scale rows"};
  auto device = host::SymbolicDevice{};
  device.set_options<kDLCUDA>();
  int32_t const experts = verify_offsets(offsets, device);
  host::TensorMatcher({rows, 2 * N})  //
      .with_dtype<bf16_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(acc);
  host::TensorMatcher({rows}).with_dtype<fp32_t>().with_device(device).verify(acc_amax).verify(gammas).verify(amax);
  host::TensorMatcher({int64_t{experts}}).with_dtype<fp32_t>().with_device(device).verify(w_amax);
  host::TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(w_denominator);
  host::TensorMatcher({rows, N / 2})  //
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(kPart / 2)
      .verify(values);
  host::TensorMatcher({scale_rows, N / 16}).with_dtype<uint8_t>().with_device(device).verify(x_scales);
  int32_t const num_rows = verify_rows(rows);
  CHECK_HOST(scale_rows.unwrap() >= host::div_ceil(num_rows + 127 * int64_t{experts}, 128) * 128)
      << "NVFP4 scales need 128-row blocks per expert, got " << scale_rows.unwrap() << " rows";
  host::LaunchKernel(grid_size(num_rows, static_cast<int32_t>(sm_count)), split_threads(N), device.unwrap())
      .enable_pdl(true)(
          nvfp4_swiglu_quantize_routed<N>,
          static_cast<__nv_bfloat16 const*>(acc.data_ptr()),
          static_cast<float const*>(acc_amax.data_ptr()),
          static_cast<float const*>(w_amax.data_ptr()),
          static_cast<float const*>(w_denominator.data_ptr()),
          static_cast<float const*>(gammas.data_ptr()),
          static_cast<uint8_t*>(values.data_ptr()),
          static_cast<uint8_t*>(x_scales.data_ptr()),
          static_cast<float*>(amax.data_ptr()),
          static_cast<int32_t const*>(offsets.data_ptr()),
          experts,
          static_cast<float>(gate_cap),
          static_cast<float>(up_cap));
}

template <int K>
void mxfp8_quantize(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView dst2src,
    tvm::ffi::TensorView values,
    tvm::ffi::TensorView exponents,
    tvm::ffi::TensorView offsets,
    int64_t top_k,
    int64_t m_padded,
    int64_t sm_count) {
  auto tokens = host::SymbolicSize{"tokens"};
  auto rows = host::SymbolicSize{"routed rows"};
  auto device = host::SymbolicDevice{};
  device.set_options<kDLCUDA>();
  int32_t const experts = verify_offsets(offsets, device);
  host::TensorMatcher({tokens, K})  //
      .with_dtype<bf16_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(x);
  host::TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(dst2src);
  host::TensorMatcher({rows, K})  //
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(values);
  host::TensorMatcher({K / 128, m_padded}).with_dtype<int32_t>().with_device(device).verify(exponents);
  CHECK_HOST(top_k > 0) << "top_k must be positive, got " << top_k;
  int32_t const num_rows = verify_rows(rows);
  CHECK_HOST(m_padded >= (num_rows + 3 * int64_t{experts}) / 4 * 4 && m_padded < std::numeric_limits<int32_t>::max())
      << "MXFP8 scales need 4-row aligned experts, got m_padded " << m_padded;
  host::LaunchKernel(grid_size(num_rows, static_cast<int32_t>(sm_count)), threads(K), device.unwrap())
      .enable_pdl(true)(
          mxfp8_quantize_routed<K>,
          static_cast<__nv_bfloat16 const*>(x.data_ptr()),
          static_cast<int32_t const*>(dst2src.data_ptr()),
          static_cast<uint8_t*>(values.data_ptr()),
          static_cast<uint8_t*>(exponents.data_ptr()),
          static_cast<int32_t const*>(offsets.data_ptr()),
          experts,
          static_cast<int32_t>(top_k),
          static_cast<int32_t>(m_padded));
}

template <int N>
void mxfp8_swiglu_quantize(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView gammas,
    tvm::ffi::TensorView values,
    tvm::ffi::TensorView exponents,
    tvm::ffi::TensorView offsets,
    int64_t m_padded,
    double gate_cap,
    double up_cap,
    int64_t sm_count) {
  auto rows = host::SymbolicSize{"routed rows"};
  auto device = host::SymbolicDevice{};
  device.set_options<kDLCUDA>();
  int32_t const experts = verify_offsets(offsets, device);
  host::TensorMatcher({rows, 2 * N})  //
      .with_dtype<bf16_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(x);
  host::TensorMatcher({rows}).with_dtype<fp32_t>().with_device(device).verify(gammas);
  host::TensorMatcher({rows, N})  //
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(values);
  host::TensorMatcher({N / 128, m_padded}).with_dtype<int32_t>().with_device(device).verify(exponents);
  int32_t const num_rows = verify_rows(rows);
  CHECK_HOST(m_padded >= (num_rows + 3 * int64_t{experts}) / 4 * 4 && m_padded < std::numeric_limits<int32_t>::max())
      << "MXFP8 scales need 4-row aligned experts, got m_padded " << m_padded;
  host::LaunchKernel(grid_size(num_rows, static_cast<int32_t>(sm_count)), threads(N), device.unwrap())
      .enable_pdl(true)(
          mxfp8_swiglu_quantize_routed<N>,
          static_cast<__nv_bfloat16 const*>(x.data_ptr()),
          static_cast<float const*>(gammas.data_ptr()),
          static_cast<uint8_t*>(values.data_ptr()),
          static_cast<uint8_t*>(exponents.data_ptr()),
          static_cast<int32_t const*>(offsets.data_ptr()),
          experts,
          static_cast<float>(gate_cap),
          static_cast<float>(up_cap),
          static_cast<int32_t>(m_padded));
}

}  // namespace sglang::inkling2_routed_quant_sm120
