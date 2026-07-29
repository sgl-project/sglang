/* MXFP4 fused decode attention for DeepSeek V4 on Hopper (SM90).

   Reads packed MXFP4 K-cache rows, dequantizes E2M1+E8M0 entirely in
   registers, and computes QK^T + softmax + V-weighted sum in a single
   kernel.  No shared memory per token — the warp uses warp shuffle to
   align the strided dequant with Q.

   Grid: ceil(num_queries / kNumWarps)
   Each warp handles one (batch, head).

   MXFP4 row layout (368 bytes/tok):
     [224 B packed E2M1 | 14 B E8M0 + 2 B pad | 128 B BF16 RoPE]    */

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

using DType = bf16_t;

static constexpr uint32_t kBlockSize = 128;
static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
static constexpr uint32_t kLanes = device::kWarpThreads;  // 32

static constexpr int64_t kHeadDim = 512;
static constexpr int64_t kNopeDim = 448;
static constexpr int64_t kRopeDim = 64;
static constexpr int64_t kGroupSize = 32;
static constexpr int64_t kNumGroups = kNopeDim / kGroupSize;                                 // 14
static constexpr int64_t kBytesPerGroup = kGroupSize / 2;                                    // 16
static constexpr int64_t kBytesPerToken = (kNopeDim / 2) + kNumGroups + 2 + (kRopeDim * 2);  // 368

static constexpr int64_t kOffNope = 0;
static constexpr int64_t kOffScale = kNopeDim / 2;               // 224
static constexpr int64_t kOffRope = kOffScale + kNumGroups + 2;  // 240
static constexpr int64_t kValsPerLane = kHeadDim / kLanes;       // 16

static_assert(kOffRope + kRopeDim * 2 == kBytesPerToken);
static_assert(kBytesPerToken == 368);
static_assert(kHeadDim % kLanes == 0);
static_assert(kNopeDim % kGroupSize == 0);

// --- fast helpers -----------------------------------------------------------

__device__ __forceinline__ float e2m1_val(uint32_t code) {
  constexpr float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  return lut[code & 0x07u];
}

__device__ __forceinline__ float e8m0_to_float(uint8_t bits) {
  if (bits == 0) return 0.0f;
  return __int_as_float(static_cast<uint32_t>(static_cast<int>(bits) - 127 + 127) << 23);
}

// --- parameter struct -------------------------------------------------------

struct Mxfp4DecodeParams {
  const void* __restrict__ q;
  const uint8_t* __restrict__ k_cache;
  const int32_t* __restrict__ indices;
  const float* __restrict__ attn_sink;  // [num_queries] or nullptr
  void* __restrict__ o;
  float sm_scale;
  uint32_t page_stride_bytes;
  uint32_t page_size;
  uint32_t num_valid;  // actual valid tokens in page (0 = read all page_size)
  uint32_t num_queries;
};

// --- kernel -----------------------------------------------------------------

// Q and K are distributed across warp lanes with the same strided layout:
//   lane i holds values at positions [i, 32+i, 64+i, ..., head_dim-32+i]
// This lets the dot product be a local reduction per lane followed by a
// warp-level sum — no shared memory needed.

template <bool kUsePDL>
__global__ void mxfp4_decode_kernel(const __grid_constant__ Mxfp4DecodeParams params) {
  using namespace device;

  const uint32_t lane_id = threadIdx.x % kLanes;
  const uint32_t warp_id = threadIdx.x / kLanes;
  const uint32_t gid = blockIdx.x * kNumWarps + warp_id;
  if (gid >= params.num_queries) return;

  PDLWaitPrimary<kUsePDL>();

  // ---- load Q in strided order (matching K dequant layout) ----------------
  float q_val[kValsPerLane];  // 16 floats / lane
  {
    const auto* q_bf16 = static_cast<const bf16_t*>(params.q) + gid * kHeadDim;
    // Lane i loads Q at [i, 32+i, 64+i, ..., 480+i]
#pragma unroll
    for (int g = 0; g < kNumGroups; ++g)
      q_val[g] = cast<float>(q_bf16[g * kGroupSize + lane_id]);

    // rope positions: 448 + lane_id, 448 + 32 + lane_id
#pragma unroll
    for (int j = 0; j < kRopeDim / kLanes; ++j)
      q_val[kNumGroups + j] = cast<float>(q_bf16[kNopeDim + j * kLanes + lane_id]);
  }

  // ---- locate page ---------------------------------------------------------
  const int32_t page_idx = params.indices[gid];
  const auto* page_base = params.k_cache + static_cast<size_t>(page_idx) * params.page_stride_bytes;

  // ---- state ---------------------------------------------------------------
  float m = -1e30f, s_val = 0.0f;
  float o_val[kValsPerLane] = {};

  // ---- scan tokens ----------------------------------------------------------
  const uint32_t tk_end =
      (params.num_valid > 0 && params.num_valid < params.page_size) ? params.num_valid : params.page_size;
  for (uint32_t tk = 0; tk < tk_end; ++tk) {
    const auto* row = page_base + static_cast<size_t>(tk) * kBytesPerToken;
    float k_val[kValsPerLane];

    // dequant noPE (same strided layout as Q)
#pragma unroll
    for (int g = 0; g < kNumGroups; ++g) {
      const uint32_t byte_off = g * kBytesPerGroup + (lane_id / 2);
      const uint8_t raw = row[byte_off];
      const uint8_t nib = (lane_id & 1u) ? (raw >> 4) : (raw & 0x0Fu);
      const float v = e2m1_val(nib);
      const float scl = e8m0_to_float(row[kOffScale + g]);
      k_val[g] = (nib & 0x08u) ? -v * scl : v * scl;
    }

    // rope
#pragma unroll
    for (int j = 0; j < kRopeDim / kLanes; ++j) {
      const auto* rope_ptr = reinterpret_cast<const bf16_t*>(row + kOffRope);
      k_val[kNumGroups + j] = cast<float>(rope_ptr[j * kLanes + lane_id]);
    }

    // dot product
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < kValsPerLane; ++i)
      dot += q_val[i] * k_val[i];
    dot = warp::reduce_sum(dot);
    dot *= params.sm_scale;

    // online softmax
    const float m_new = max(m, dot);
    const float e_val = expf(dot - m_new);
    const float rc = expf(m - m_new);
    s_val = s_val * rc + e_val;
    m = m_new;

    // V-weighted sum
#pragma unroll
    for (int i = 0; i < kValsPerLane; ++i)
      o_val[i] = o_val[i] * rc + k_val[i] * e_val;
  }

  // ---- attn_sink (virtual token with V=0) ---------------------------------
  if (params.attn_sink != nullptr) {
    const float sink = params.attn_sink[gid];
    const float m_new = max(m, sink);
    const float e_sink = expf(sink - m_new);
    const float rc = expf(m - m_new);
    s_val = s_val * rc + e_sink;
    m = m_new;
#pragma unroll
    for (int i = 0; i < kValsPerLane; ++i)
      o_val[i] *= rc;
  }

  // ---- finalize & store -----------------------------------------------------
  const float inv_s = (s_val > 0.0f) ? (1.0f / s_val) : 0.0f;
  auto* o_bf16 = static_cast<bf16_t*>(params.o) + gid * kHeadDim;
  // Store in strided layout (same as Q/K)
#pragma unroll
  for (int g = 0; g < kNumGroups; ++g)
    o_bf16[g * kGroupSize + lane_id] = cast<bf16_t>(o_val[g] * inv_s);
#pragma unroll
  for (int j = 0; j < kRopeDim / kLanes; ++j)
    o_bf16[kNopeDim + j * kLanes + lane_id] = cast<bf16_t>(o_val[kNumGroups + j] * inv_s);

  PDLTriggerSecondary<kUsePDL>();
}

// --- TVM entry --------------------------------------------------------------

template <bool kUsePDL>
struct Mxfp4DecodeKernel {
  static void forward(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::Optional<tvm::ffi::TensorView> attn_sink,
      const tvm::ffi::TensorView o,
      float sm_scale,
      int64_t page_size,
      int64_t num_valid = 0) {
    using namespace host;
    auto B = SymbolicSize{"num_queries"};
    auto D = SymbolicDevice{};
    D.set_options<kDLCUDA>();

    TensorMatcher({B, kHeadDim}).with_dtype<bf16_t>().with_device(D).verify(q);
    TensorMatcher({-1, kBytesPerToken}).with_dtype<uint8_t>().with_device(D).verify(k_cache);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(D).verify(indices);
    const float* sink_ptr = nullptr;
    if (attn_sink.has_value()) {
      TensorMatcher({B}).with_dtype<float>().with_device(D).verify(attn_sink.value());
      sink_ptr = static_cast<const float*>(attn_sink.value().data_ptr());
    }
    TensorMatcher({B, kHeadDim}).with_dtype<bf16_t>().with_device(D).verify(o);
    RuntimeCheck(page_size > 0);

    const uint32_t nq = static_cast<uint32_t>(B.unwrap());
    if (nq == 0) return;

    const auto params = Mxfp4DecodeParams{
        .q = q.data_ptr(),
        .k_cache = static_cast<const uint8_t*>(k_cache.data_ptr()),
        .indices = static_cast<const int32_t*>(indices.data_ptr()),
        .attn_sink = sink_ptr,
        .o = o.data_ptr(),
        .sm_scale = sm_scale,
        .page_stride_bytes = static_cast<uint32_t>(static_cast<uint32_t>(page_size) * kBytesPerToken),
        .page_size = static_cast<uint32_t>(page_size),
        .num_valid = static_cast<uint32_t>(num_valid),
        .num_queries = nq,
    };
    LaunchKernel(div_ceil(nq, kNumWarps), kBlockSize, D.unwrap())
        .enable_pdl(kUsePDL)(mxfp4_decode_kernel<kUsePDL>, params);
  }
};

}  // namespace
