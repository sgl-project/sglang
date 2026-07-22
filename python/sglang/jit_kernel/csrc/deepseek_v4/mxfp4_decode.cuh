/* MXFP4 fused decode attention for DeepSeek V4 on Hopper (SM90).

   Reads packed MXFP4 K-cache rows (E2M1 noPE + E8M0 block-32 scales + BF16
   RoPE), dequantizes on the fly into per-warp shared memory, and computes
   QK^T + softmax + V-weighted sum in a single kernel.

   Grid: ceil(num_queries / kNumWarps)  — multiple heads per block.
   Each of the kNumWarps (4) warps handles one (batch, head) independently,
   using its own shared-memory partition.

   MXFP4 row layout — 368 bytes per token, row-major contiguous:
     [224 B packed E2M1 noPE | 14 B E8M0 scales + 2 B pad | 128 B BF16 RoPE]

   E8M0:  value = 2^(byte - 127).  No global scale needed.
   E2M1:  sign[3] | exp[2:1] | mant[0] → 16 discrete values.                */

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

using DType = bf16_t;

static constexpr uint32_t kBlockSize = 128;
static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;  // 4

static constexpr int64_t kHeadDim = 512;
static constexpr int64_t kNopeDim = 448;
static constexpr int64_t kRopeDim = 64;
static constexpr int64_t kGroupSize = 32;
static constexpr int64_t kNumGroups = kNopeDim / kGroupSize;  // 14
static constexpr int64_t kBytesPerGroup = kGroupSize / 2;     // 16
static constexpr int64_t kBytesPerToken =                     // 368
    (kNopeDim / 2) + kNumGroups + 2 + (kRopeDim * 2);

// byte offsets within a token row
static constexpr int64_t kOffNope = 0;
static constexpr int64_t kOffScale = kNopeDim / 2;               // 224
static constexpr int64_t kOffRope = kOffScale + kNumGroups + 2;  // 240

static_assert(kOffRope + kRopeDim * 2 == kBytesPerToken);
static_assert(kBytesPerToken == 368);

// --- E2M1 / E8M0 -------------------------------------------------------------

__device__ __forceinline__ float e2m1_positive(uint32_t code) {
  constexpr float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  return lut[code & 0x07u];
}

// Fast E8M0 → float: 2^(bits-127) as an integer shift of the IEEE 754
// exponent field.  Subnormals (bits == 0 → 2^-127 ≈ 5.88e-39) are clamped
// to zero, which is safe for scale factors.
__device__ __forceinline__ float e8m0_to_float(uint8_t bits) {
  if (bits == 0) return 0.0f;  // 2^-127 is practically zero
  const int k = static_cast<int>(bits) - 127;
  const uint32_t u = static_cast<uint32_t>(k + 127) << 23;
  return __int_as_float(u);
}

// --- parameter struct --------------------------------------------------------

struct Mxfp4DecodeParams {
  const void* __restrict__ q;
  const uint8_t* __restrict__ k_cache;
  const int32_t* __restrict__ indices;
  void* __restrict__ o;
  float sm_scale;
  uint32_t page_stride_bytes;
  uint32_t page_size;
  uint32_t num_queries;
};

// --- kernel -----------------------------------------------------------------
// Each warp handles one (batch, head) independently.
// Shared memory is partitioned per warp: s_k[warp_id][kHeadDim].

template <bool kUsePDL>
__global__ void mxfp4_decode_kernel(const __grid_constant__ Mxfp4DecodeParams params) {
  using namespace device;

  const auto& [q, k_cache, indices, o, sm_scale, page_stride_bytes, page_size, nq] = params;
  (void)nq;

  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  const uint32_t gid = blockIdx.x * kNumWarps + warp_id;  // global query id

  if (gid >= params.num_queries) return;  // tail block padding

  // per-warp shared memory partition
  __shared__ DType s_k[kNumWarps][kHeadDim];  // 4 × 1024 = 4096 bytes
  DType* my_s_k = s_k[warp_id];               // this warp's partition

  PDLWaitPrimary<kUsePDL>();

  // ---- load Q (16 contiguous values per lane) --------------------------------
  float q_val[16];
  {
    const auto* q_bf16 = static_cast<const bf16_t*>(q) + gid * kHeadDim;
#pragma unroll
    for (int i = 0; i < 16; ++i)
      q_val[i] = cast<float>(q_bf16[lane_id * 16 + i]);
  }

  // ---- locate K page --------------------------------------------------------
  const int32_t page_idx = indices[gid];
  const auto* page_base = k_cache + static_cast<size_t>(page_idx) * page_stride_bytes;

  // ---- online softmax state -------------------------------------------------
  float m = -1e30f;
  float s = 0.0f;
  float o_val[16] = {};

  // ---- scan tokens -----------------------------------------------------------
  for (uint32_t tk = 0; tk < page_size; ++tk) {
    const auto* row = page_base + static_cast<size_t>(tk) * kBytesPerToken;

    // 1) dequant noPE → shared memory (all lanes collaborate)
#pragma unroll
    for (int g = 0; g < kNumGroups; ++g) {
      const uint32_t byte_off = g * kBytesPerGroup + (lane_id / 2);
      const uint8_t packed_byte = row[byte_off];
      const uint8_t nibble = (lane_id & 1u) ? (packed_byte >> 4) : (packed_byte & 0x0Fu);
      const float val = e2m1_positive(nibble & 0x07u);
      const float scale = e8m0_to_float(row[kOffScale + g]);
      const bool neg = (nibble & 0x08u) != 0;
      my_s_k[g * kGroupSize + lane_id] = cast<DType>(neg ? -val * scale : val * scale);
    }

    // 2) load RoPE → shared memory
    {
      const auto* rope_src = reinterpret_cast<const DType*>(row + kOffRope);
#pragma unroll
      for (int j = 0; j < kRopeDim / kWarpThreads; ++j) {
        const int idx = lane_id + j * kWarpThreads;
        my_s_k[kNopeDim + idx] = rope_src[idx];
      }
    }

    __syncwarp();

    // 3) dot product Q · K (contiguous from shared memory)
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int pos = lane_id * 16 + i;
      dot += q_val[i] * cast<float>(my_s_k[pos]);
    }
    dot = warp::reduce_sum(dot);
    dot *= sm_scale;

    // 4) online softmax
    const float m_new = max(m, dot);
    const float e = expf(dot - m_new);
    const float rc = expf(m - m_new);
    s = s * rc + e;
    m = m_new;

    // 5) V-weighted sum (K = V)
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int pos = lane_id * 16 + i;
      o_val[i] = o_val[i] * rc + cast<float>(my_s_k[pos]) * e;
    }
  }

  // ---- finalize: o /= s, store ----------------------------------------------
  const float inv_s = (s > 0.0f) ? (1.0f / s) : 0.0f;
  auto* o_bf16 = static_cast<bf16_t*>(o) + gid * kHeadDim;
#pragma unroll
  for (int i = 0; i < 16; ++i)
    o_bf16[lane_id * 16 + i] = cast<bf16_t>(o_val[i] * inv_s);

  PDLTriggerSecondary<kUsePDL>();
}

// --- host-side wrapper (TVM entry point) ------------------------------------

template <bool kUsePDL>
struct Mxfp4DecodeKernel {
  static void forward(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView o,
      float sm_scale,
      int64_t page_size) {
    using namespace host;

    auto B = SymbolicSize{"num_queries"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, kHeadDim}).with_dtype<bf16_t>().with_device(device_).verify(q);
    TensorMatcher({-1, kBytesPerToken}).with_dtype<uint8_t>().with_device(device_).verify(k_cache);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(indices);
    TensorMatcher({B, kHeadDim}).with_dtype<bf16_t>().with_device(device_).verify(o);
    RuntimeCheck(page_size > 0);

    const uint32_t num_queries = static_cast<uint32_t>(B.unwrap());
    if (num_queries == 0) return;

    const auto params = Mxfp4DecodeParams{
        .q = q.data_ptr(),
        .k_cache = static_cast<const uint8_t*>(k_cache.data_ptr()),
        .indices = static_cast<const int32_t*>(indices.data_ptr()),
        .o = o.data_ptr(),
        .sm_scale = sm_scale,
        .page_stride_bytes = static_cast<uint32_t>(static_cast<uint32_t>(page_size) * kBytesPerToken),
        .page_size = static_cast<uint32_t>(page_size),
        .num_queries = num_queries,
    };

    const uint32_t num_blocks = div_ceil(num_queries, kNumWarps);
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap()).enable_pdl(kUsePDL)(mxfp4_decode_kernel<kUsePDL>, params);
  }
};

}  // namespace
