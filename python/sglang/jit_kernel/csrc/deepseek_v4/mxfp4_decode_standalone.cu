/* MXFP4 fused decode attention — standalone CUDA kernel (no sgl-kernel deps).

   PyTorch custom op registration via TORCH_LIBRARY.  Fully CUDA-graph
   compatible because kernel launches go through the standard PyTorch
   dispatch path without TVM-FFI indirection.

   MXFP4 row layout (368 bytes/tok):
     [224 B packed E2M1 | 14 B E8M0 + 2 B pad | 128 B BF16 RoPE]
*/

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cstdint>
#include <tuple>

// ──── constants ──────────────────────────────────────────────────────────────

static constexpr uint32_t kBlockSize = 128;
static constexpr uint32_t kLanes = 32;  // warp size
static constexpr uint32_t kNumWarps = kBlockSize / kLanes;

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

// ──── helpers ─────────────────────────────────────────────────────────────────

__device__ __forceinline__ float e2m1_val(uint32_t code) {
  constexpr float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  return lut[code & 0x07u];
}

__device__ __forceinline__ float e8m0_to_float(uint8_t bits) {
  if (bits == 0) return 0.0f;
  return __int_as_float(static_cast<uint32_t>(static_cast<int>(bits) - 127 + 127) << 23);
}

// Warp-level float sum using __shfl_xor_sync (no sgl-kernel dep).
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = kLanes / 2; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;
}

// ──── parameter struct ───────────────────────────────────────────────────────

struct Mxfp4DecodeParams {
  const void* __restrict__ q;
  const uint8_t* __restrict__ k_cache;
  const int32_t* __restrict__ indices;       // [N, swa_width] flat SWA slot ids per query
  const int32_t* __restrict__ swa_lengths;   // per-query valid tokens; nullptr → num_valid
  const float* __restrict__ attn_sink;
  void* __restrict__ o;
  float* __restrict__ lse;
  float sm_scale;
  uint32_t swa_width;  // padded width of indices
  uint32_t num_valid;  // fallback when swa_lengths == nullptr (counts from indices[gid])
  uint32_t num_queries;
  // Extra (C4/C128) cache; nullptr extra_k_cache → no extra attention.
  const uint8_t* __restrict__ extra_k_cache;
  const int32_t* __restrict__ extra_indices;        // [N, extra_topk_width] token ids
  const int32_t* __restrict__ extra_topk_lengths;   // [N] per-query valid count
  uint32_t extra_page_stride_bytes;
  uint32_t extra_page_size;
  uint32_t extra_topk_width;
  int64_t extra_capacity;  // num_pages * page_size; ids >= capacity are masked
};

// Dequantize one MXFP4 row and fold it into the online softmax state.
__device__ __forceinline__ void attend_row(
    const uint8_t* row,
    const float* q_val,
    float& m,
    float& s_val,
    float* o_val,
    float sm_scale) {
  const uint32_t lane_id = threadIdx.x % kLanes;
  float k_val[kValsPerLane];

  // dequant noPE (strided layout, same as Q)
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
    const auto* rope_ptr = reinterpret_cast<const __nv_bfloat16*>(row + kOffRope);
    k_val[kNumGroups + j] = __bfloat162float(rope_ptr[j * kLanes + lane_id]);
  }

  // dot product
  float dot = 0.0f;
#pragma unroll
  for (int i = 0; i < kValsPerLane; ++i)
    dot += q_val[i] * k_val[i];
  dot = warp_reduce_sum(dot);
  dot *= sm_scale;

  // online softmax
  const float m_new = fmaxf(m, dot);
  const float e_val = expf(dot - m_new);
  const float rc = expf(m - m_new);
  s_val = s_val * rc + e_val;
  m = m_new;

  // V-weighted sum
#pragma unroll
  for (int i = 0; i < kValsPerLane; ++i)
    o_val[i] = o_val[i] * rc + k_val[i] * e_val;
}

// ──── kernel ──────────────────────────────────────────────────────────────────

__global__ void mxfp4_decode_kernel(const __grid_constant__ Mxfp4DecodeParams params) {
  const uint32_t lane_id = threadIdx.x % kLanes;
  const uint32_t warp_id = threadIdx.x / kLanes;
  const uint32_t gid = blockIdx.x * kNumWarps + warp_id;
  if (gid >= params.num_queries) return;

  // ---- load Q in strided order (matching K dequant layout) ----
  float q_val[kValsPerLane];
  {
    const auto* q_bf16 = static_cast<const __nv_bfloat16*>(params.q) + gid * kHeadDim;
#pragma unroll
    for (int g = 0; g < kNumGroups; ++g)
      q_val[g] = __bfloat162float(q_bf16[g * kGroupSize + lane_id]);
#pragma unroll
    for (int j = 0; j < kRopeDim / kLanes; ++j)
      q_val[kNumGroups + j] = __bfloat162float(q_bf16[kNopeDim + j * kLanes + lane_id]);
  }

  // ---- state ----
  float m = -1e30f, s_val = 0.0f;
  float o_val[kValsPerLane] = {};

  // ---- scan SWA tokens ----
  // indices holds flat SWA slot ids (one row per slot, tightly packed), so a
  // 128-token window that crosses storage-page boundaries is read correctly.
  uint32_t tk_end;
  if (params.swa_lengths != nullptr) {
    const int32_t len = params.swa_lengths[gid];
    tk_end = static_cast<uint32_t>(len < 0 ? 0 : (len > (int32_t)params.swa_width ? params.swa_width : len));
  } else {
    tk_end = (params.num_valid > 0 && params.num_valid < params.swa_width) ? params.num_valid : params.swa_width;
  }
  for (uint32_t tk = 0; tk < tk_end; ++tk) {
    const int32_t slot = params.indices[static_cast<int64_t>(gid) * params.swa_width + tk];
    if (slot < 0) continue;  // invalid padding
    const auto* row = params.k_cache + static_cast<size_t>(slot) * kBytesPerToken;
    attend_row(row, q_val, m, s_val, o_val, params.sm_scale);
  }

  // ---- scan extra (C4/C128) tokens ----
  if (params.extra_k_cache != nullptr) {
    const int32_t extra_len = (params.extra_topk_lengths != nullptr)
        ? params.extra_topk_lengths[gid]
        : static_cast<int32_t>(params.extra_topk_width);
    const int32_t extra_len_clamped =
        extra_len < 0 ? 0 : (extra_len > (int32_t)params.extra_topk_width ? (int32_t)params.extra_topk_width : extra_len);
    for (int32_t tk = 0; tk < extra_len_clamped; ++tk) {
      const int32_t tid = params.extra_indices[static_cast<int64_t>(gid) * params.extra_topk_width + tk];
      // Mask invalid padding (-1) and out-of-capacity ids like the CPU path.
      if (tid < 0 || static_cast<int64_t>(tid) >= params.extra_capacity) continue;
      const int32_t block_idx = tid / static_cast<int32_t>(params.extra_page_size);
      const int32_t rel_idx = tid % static_cast<int32_t>(params.extra_page_size);
      const auto* row = params.extra_k_cache +
                        static_cast<size_t>(block_idx) * params.extra_page_stride_bytes +
                        static_cast<size_t>(rel_idx) * kBytesPerToken;
      attend_row(row, q_val, m, s_val, o_val, params.sm_scale);
    }
  }

  // ---- attn_sink (virtual token with V=0) ----
  if (params.attn_sink != nullptr) {
    const float sink = params.attn_sink[gid];
    const float m_new = fmaxf(m, sink);
    const float e_sink = expf(sink - m_new);
    const float rc = expf(m - m_new);
    s_val = s_val * rc + e_sink;
    m = m_new;
#pragma unroll
    for (int i = 0; i < kValsPerLane; ++i)
      o_val[i] *= rc;
  }

  // ---- finalize & store ----
  const float inv_s = (s_val > 0.0f) ? (1.0f / s_val) : 0.0f;
  auto* o_bf16 = static_cast<__nv_bfloat16*>(params.o) + gid * kHeadDim;
#pragma unroll
  for (int g = 0; g < kNumGroups; ++g)
    o_bf16[g * kGroupSize + lane_id] = __float2bfloat16_rn(o_val[g] * inv_s);
#pragma unroll
  for (int j = 0; j < kRopeDim / kLanes; ++j)
    o_bf16[kNopeDim + j * kLanes + lane_id] = __float2bfloat16_rn(o_val[kNumGroups + j] * inv_s);

  // ---- log-sum-exp (for merging with extra attention) ----
  if (params.lse != nullptr) {
    // LSE = m + log(s)  where m = max score, s = sum(exp(score - m))
    // This is the log of the softmax normalizer: log(sum(exp(scores)))
    if (lane_id == 0) {
      params.lse[gid] = m + logf(s_val > 0.0f ? s_val : 1e-30f);
    }
  }
}

// ──── PyTorch custom op ──────────────────────────────────────────────────────

namespace {

std::tuple<torch::Tensor, torch::Tensor> mxfp4_decode_op(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor page_indices,
    std::optional<torch::Tensor> swa_lengths,
    std::optional<torch::Tensor> extra_k_cache,
    std::optional<torch::Tensor> extra_indices,
    std::optional<torch::Tensor> extra_topk_lengths,
    std::optional<torch::Tensor> attn_sink,
    torch::Tensor o,
    torch::Tensor lse,
    double sm_scale,
    int64_t swa_width,
    int64_t num_valid,
    int64_t extra_topk_width,
    int64_t extra_page_size) {
  const auto device = q.device();
  const at::cuda::CUDAGuard guard(device.index());

  const uint32_t nq = static_cast<uint32_t>(q.size(0));
  if (nq == 0) return std::make_tuple(o, lse);

  const bool have_extra = extra_k_cache.has_value();
  TORCH_CHECK(
      have_extra == extra_indices.has_value() && have_extra == extra_topk_lengths.has_value(),
      "extra_k_cache, extra_indices, and extra_topk_lengths must be provided together");
  if (have_extra) {
    TORCH_CHECK(extra_topk_width > 0, "extra_topk_width must be positive");
    TORCH_CHECK(extra_page_size > 0, "extra_page_size must be positive");
    TORCH_CHECK(
        extra_indices->size(1) == extra_topk_width,
        "extra_indices must be [N, extra_topk_width]");
  }
  TORCH_CHECK(page_indices.size(1) == swa_width, "page_indices must be [N, swa_width]");

  Mxfp4DecodeParams params{};
  params.q = q.data_ptr();
  params.k_cache = static_cast<const uint8_t*>(k_cache.data_ptr());
  params.indices = page_indices.data_ptr<int32_t>();
  params.swa_lengths = swa_lengths.has_value() ? swa_lengths->data_ptr<int32_t>() : nullptr;
  params.attn_sink = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
  params.o = o.data_ptr();
  params.lse = lse.data_ptr<float>();
  params.sm_scale = static_cast<float>(sm_scale);
  params.swa_width = static_cast<uint32_t>(swa_width);
  params.num_valid = static_cast<uint32_t>(num_valid);
  params.num_queries = nq;
  if (have_extra) {
    params.extra_k_cache = static_cast<const uint8_t*>(extra_k_cache->data_ptr());
    params.extra_indices = extra_indices->data_ptr<int32_t>();
    params.extra_topk_lengths = extra_topk_lengths->data_ptr<int32_t>();
    params.extra_page_stride_bytes =
        static_cast<uint32_t>(static_cast<uint32_t>(extra_page_size) * kBytesPerToken);
    params.extra_page_size = static_cast<uint32_t>(extra_page_size);
    params.extra_topk_width = static_cast<uint32_t>(extra_topk_width);
    // k_cache rows are already flattened to one row per physical token, so
    // the row count is the token capacity.
    params.extra_capacity = static_cast<int64_t>(extra_k_cache->size(0));
  }

  const uint32_t grid = (nq + kNumWarps - 1) / kNumWarps;
  const auto stream = at::cuda::getCurrentCUDAStream(device.index());

  mxfp4_decode_kernel<<<grid, kBlockSize, 0, stream>>>(params);

  return std::make_tuple(o, lse);
}

TORCH_LIBRARY(sglang_mxfp4, m) {
  m.def(
      "decode(Tensor q, Tensor k_cache, Tensor page_indices, "
      "Tensor? swa_lengths, Tensor? extra_k_cache, Tensor? extra_indices, "
      "Tensor? extra_topk_lengths, Tensor? attn_sink, Tensor o, Tensor lse, "
      "float sm_scale, int swa_width, int num_valid, int extra_topk_width, "
      "int extra_page_size) -> (Tensor, Tensor)",
      mxfp4_decode_op);
}

}  // namespace
