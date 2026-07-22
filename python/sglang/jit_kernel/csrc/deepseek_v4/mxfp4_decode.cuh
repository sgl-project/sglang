/* MXFP4 fused decode attention for DeepSeek V4 on Hopper (SM90).

   Reads packed MXFP4 K-cache rows (E2M1 noPE + E8M0 block-32 scales + BF16
   RoPE), dequantizes on the fly, and computes QK^T + softmax + V-weighted sum
   in a single kernel.  One warp handles one (batch, head) pair.

   MXFP4 row layout — 368 bytes per token, row-major contiguous:
     [224 B packed E2M1 noPE | 14 B E8M0 scales + 2 B pad | 128 B BF16 RoPE]

   E8M0:  value = 2^(byte - 127).  No global scale needed.
   E2M1:  1 sign + 2 exp + 1 mantissa → {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.
          Two E2M1 codes packed per byte: [hi_nibble | lo_nibble].          */

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

using DType = bf16_t;

// --- kernel constants -------------------------------------------------------
constexpr uint32_t kBlockSize = 128;
constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;  // 4

constexpr int64_t kHeadDim = 512;
constexpr int64_t kNopeDim = 448;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kGroupSize = 32;                                                    // values per E8M0 scale
constexpr int64_t kNumGroups = kNopeDim / kGroupSize;                                 // 14
constexpr int64_t kBytesPerGroup = kGroupSize / 2;                                    // 16 packed bytes
constexpr int64_t kBytesPerToken = (kNopeDim / 2) + kNumGroups + 2 + (kRopeDim * 2);  // 368

// byte offsets within a token row
constexpr int64_t kOffNope = 0;
constexpr int64_t kOffScale = kNopeDim / 2;               // 224
constexpr int64_t kOffRope = kOffScale + kNumGroups + 2;  // 240

static_assert(kOffRope + kRopeDim * 2 == kBytesPerToken);
static_assert(kBytesPerToken == 368);

// --- E2M1 LUT ----------------------------------------------------------------

// Positive E2M1 values indexed by 3-bit magnitude code.
__device__ __forceinline__ float e2m1_positive(uint32_t code) {
  constexpr float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  return lut[code & 0x07u];
}

// --- E8M0 scale --------------------------------------------------------------

__device__ __forceinline__ float e8m0_to_float(uint8_t bits) {
  return exp2f(static_cast<float>(static_cast<int>(bits) - 127));
}

// --- parameter struct --------------------------------------------------------

struct Mxfp4DecodeParams {
  const void* __restrict__ q;           // [num_queries, 512] BF16
  const uint8_t* __restrict__ k_cache;  // [num_rows, 368] uint8, row-major
  const int32_t* __restrict__ indices;  // [num_queries] page row indices
  void* __restrict__ o;                 // [num_queries, 512] BF16
  float sm_scale;
  uint32_t page_stride_bytes;  // page_size * 368
  uint32_t page_size;
};

// --- kernel -----------------------------------------------------------------

template <bool kUsePDL>
__global__ void mxfp4_decode_kernel(const __grid_constant__ Mxfp4DecodeParams params) {
  using namespace device;

  const auto& [q, k_cache, indices, o, sm_scale, page_stride_bytes, page_size] = params;

  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  const uint32_t gid = blockIdx.x;  // global query id = batch*num_heads + head
  const uint32_t num_warp_queries = gridDim.x;

  PDLWaitPrimary<kUsePDL>();

  // ---- load Q (16 floats per lane, two 8-element aligned vectors) -----------
  float q_val[16];  // kHeadDim / kWarpThreads = 16
  {
    const auto* q_bf16 = static_cast<const bf16_t*>(q) + gid * kHeadDim;
    using VecQ = AlignedVector<bf16_t, 8>;
    const auto gmem_q = tile::Memory<VecQ>::warp();
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      VecQ q_vec = gmem_q.load(q_bf16 + k * (kWarpThreads * 8));
#pragma unroll
      for (int i = 0; i < 8; ++i)
        q_val[k * 8 + i] = cast<float>(q_vec[i]);
    }
  }

  // ---- locate K page --------------------------------------------------------
  const int32_t page_idx = indices[gid];
  const auto* page_base = k_cache + static_cast<size_t>(page_idx) * page_stride_bytes;

  // ---- online softmax state (per-lane = 16 floats) -------------------------
  float m = -1e30f;
  float s = 0.0f;
  float o_val[16] = {};

  // ---- scan tokens ----------------------------------------------------------
  for (uint32_t tk = 0; tk < page_size; ++tk) {
    const auto* row = page_base + static_cast<size_t>(tk) * kBytesPerToken;

    // -- dequant noPE (14 groups × 32 values) --
    // Thread i gets 14 values: one nibble from each group.
    float k_val[16];  // 14 noPE + 2 rope = 16

#pragma unroll
    for (int g = 0; g < kNumGroups; ++g) {
      // byte offset within this group: lane_id / 2
      const uint32_t byte_off = g * kBytesPerGroup + (lane_id / 2);
      const uint8_t packed_byte = row[byte_off];

      // extract nibble: even lane → lo nibble, odd lane → hi nibble
      const uint8_t nibble = (lane_id & 1u) ? (packed_byte >> 4) : (packed_byte & 0x0Fu);
      const uint32_t mag_code = nibble & 0x07u;
      const bool neg = (nibble & 0x08u) != 0;

      const float val = e2m1_positive(mag_code);

      // E8M0 scale for this group
      const float scale = e8m0_to_float(row[kOffScale + g]);

      k_val[g] = neg ? -val * scale : val * scale;
    }

    // -- RoPE (2 values per lane, BF16 → float) ------------------------------
    {
      const auto* rope_ptr = reinterpret_cast<const bf16_t*>(row + kOffRope);
      k_val[14] = cast<float>(rope_ptr[lane_id]);       // rope val 0..63
      k_val[15] = cast<float>(rope_ptr[lane_id + 32]);  // rope val 32..95
      static_assert(kRopeDim == 64);
    }

    // -- dot product Q · K ----------------------------------------------------
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      dot += q_val[i] * k_val[i];
    }
    dot = warp::reduce_sum(dot);
    dot *= sm_scale;

    // -- online softmax -------------------------------------------------------
    const float m_new = max(m, dot);
    const float e = expf(dot - m_new);
    const float rc = expf(m - m_new);
    s = s * rc + e;
    m = m_new;

    // -- V-weighted sum (K = V) -----------------------------------------------
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      o_val[i] = o_val[i] * rc + k_val[i] * e;
    }
  }

  // ---- finalize: o /= s, store ----------------------------------------------
  const float inv_s = (s > 0.0f) ? (1.0f / s) : 0.0f;
  auto* o_bf16 = static_cast<bf16_t*>(o) + gid * kHeadDim;

  {
    using VecO = AlignedVector<bf16_t, 8>;
    const auto gmem_o = tile::Memory<VecO>::warp();
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      VecO o_vec;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        o_vec[i] = cast<bf16_t>(o_val[k * 8 + i] * inv_s);
      }
      gmem_o.store(o_bf16 + k * (kWarpThreads * 8), o_vec);
    }
  }

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

    // validate tensor shapes & dtypes
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
    };

    LaunchKernel(num_queries, kBlockSize, device_.unwrap()).enable_pdl(kUsePDL)(mxfp4_decode_kernel<kUsePDL>, params);
  }
};

}  // namespace
