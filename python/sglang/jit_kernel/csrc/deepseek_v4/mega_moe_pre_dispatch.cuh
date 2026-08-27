#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <cstdint>
#include <cuda_fp8.h>

namespace {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::pack_fp8;

struct MegaMoEPreDispatchParams {
  const bf16_t* __restrict__ x;            // [num_tokens, hidden]
  const int32_t* __restrict__ topk_idx;    // [num_tokens, top_k]
  const float* __restrict__ topk_weights;  // [num_tokens, top_k]

  fp8_e4m3_t* __restrict__ buf_x;        // [padded_max, hidden]
  int32_t* __restrict__ buf_x_sf;        // contiguous int32 [P, G/4]; see layout comment
  int64_t* __restrict__ buf_topk_idx;    // [padded_max, top_k]
  float* __restrict__ buf_topk_weights;  // [padded_max, top_k]

  uint32_t num_tokens;
  uint32_t padded_max;
  uint32_t hidden;
  uint32_t num_groups;  // hidden / group_size
  uint32_t top_k;
};

// kGroupSize must match sglang_per_token_group_quant_fp8_ue8m0(group_size=).
template <uint32_t kGroupSize, bool kUsePDL>
__global__ __launch_bounds__(1024, 2) void  //
    mega_moe_pre_dispatch_kernel(const MegaMoEPreDispatchParams __grid_constant__ params) {
  using namespace device;

  constexpr uint32_t kVecElems = 8;  // 8 bf16 = 16B load per thread
  static_assert(kGroupSize % kVecElems == 0, "group_size must be a multiple of 8");
  constexpr uint32_t kThreadsPerGroup = kGroupSize / kVecElems;
  using InputVec = AlignedVector<bf16x2_t, kVecElems / 2>;
  using OutputVec = AlignedVector<fp8x2_e4m3_t, kVecElems / 2>;

  const uint32_t bid = blockIdx.x;
  const uint32_t tid = threadIdx.x;

  PDLWaitPrimary<kUsePDL>();
  if (bid < params.num_tokens) {
    // ---- Quantize path: one CTA per valid token ----

    const uint32_t token_id = bid;
    const auto token_in = params.x + static_cast<uint64_t>(token_id) * params.hidden;
    const auto token_out = params.buf_x + static_cast<uint64_t>(token_id) * params.hidden;

    InputVec in_vec;
    in_vec.load(token_in, tid);

    float local_max = 0.0f;
    float vals[kVecElems];
#pragma unroll
    for (uint32_t i = 0; i < kVecElems / 2; ++i) {
      const auto [v0, v1] = cast<fp32x2_t>(in_vec[i]);
      vals[2 * i + 0] = v0;
      vals[2 * i + 1] = v1;
      local_max = fmaxf(local_max, fmaxf(fabsf(v0), fabsf(v1)));
    }

    // Absmax across the kThreadsPerGroup threads that cover one group.
    local_max = warp::reduce_max<kThreadsPerGroup>(local_max);

    const float absmax = fmaxf(local_max, 1e-10f);
    const float raw_scale = absmax / math::FP8_E4M3_MAX;
    const uint32_t ue8m0_exp = cast_to_ue8m0(raw_scale);
    // 2^-ue8m0_exp as fp32 (equivalent to 1 / __uint_as_float(ue8m0 << 23)).
    const float inv_scale = __uint_as_float((127u + 127u - ue8m0_exp) << 23);

    OutputVec out_vec;
#pragma unroll
    for (uint32_t i = 0; i < kVecElems / 2; ++i) {
      out_vec[i] = pack_fp8(vals[2 * i + 0] * inv_scale, vals[2 * i + 1] * inv_scale);
    }
    out_vec.store(token_out, tid);

    // One thread per group writes its UE8M0 byte into the contiguous
    // row-major int32-packed layout: byte address = t*num_groups + g
    // (see layout comment at the top of the file).
    const uint32_t group_id = tid / kThreadsPerGroup;
    const uint32_t within_group_id = tid % kThreadsPerGroup;
    if (within_group_id == 0 && group_id < params.num_groups) {
      const uint32_t byte_off = token_id * params.num_groups + group_id;
      reinterpret_cast<uint8_t*>(params.buf_x_sf)[byte_off] = static_cast<uint8_t>(ue8m0_exp);
    }

    // Copy this token's topk row (no alignment assumptions; top_k is small).
    if (tid < params.top_k) {
      const uint32_t off = token_id * params.top_k + tid;
      params.buf_topk_idx[off] = params.topk_idx[off];
      params.buf_topk_weights[off] = params.topk_weights[off];
    }
  } else {
    // ---- Pad path: trailing blocks fill [num_tokens, padded_max) with (-1, 0) ----
    const uint32_t copy_bid = bid - params.num_tokens;
    const uint32_t pad_base = params.num_tokens * params.top_k;
    const uint32_t slot = pad_base + copy_bid * blockDim.x + tid;
    const uint32_t total_slots = params.padded_max * params.top_k;

    if (slot < total_slots) {
      params.buf_topk_idx[slot] = -1;
      params.buf_topk_weights[slot] = 0.0f;
    }
  }
  PDLTriggerSecondary<kUsePDL>();
}

// ---- Host wrapper
// ------------------------------------------------------------------------------------------------------------------------

template <int64_t kGroupSize, bool kUsePDL>
struct MegaMoEPreDispatchKernel {
  static_assert(kGroupSize == 32 || kGroupSize == 64 || kGroupSize == 128, "unsupported group_size");
  static constexpr auto kernel = mega_moe_pre_dispatch_kernel<static_cast<uint32_t>(kGroupSize), kUsePDL>;

  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView topk_idx,
      const tvm::ffi::TensorView topk_weights,
      const tvm::ffi::TensorView buf_x,
      const tvm::ffi::TensorView buf_x_sf,
      const tvm::ffi::TensorView buf_topk_idx,
      const tvm::ffi::TensorView buf_topk_weights) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    auto P = SymbolicSize{"padded_max"};
    auto H = SymbolicSize{"hidden"};
    auto K = SymbolicSize{"top_k"};
    auto G4 = SymbolicSize{"num_groups_div_4"};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, H})  // input x
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(x);
    TensorMatcher({M, K})  // topk_idx
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(topk_idx);
    TensorMatcher({M, K})  // topk_weights
        .with_dtype<float>()
        .with_device(device)
        .verify(topk_weights);
    TensorMatcher({P, H})  // buf.x
        .with_dtype<int8_t>()
        .with_device(device)
        .verify(buf_x);
    // buf.x_sf is the contiguous row-major int32 view from DeepGEMM's mega
    // symm buffer (DeepGEMM/csrc/apis/mega.hpp): shape (P, G/4), strides
    // (G/4, 1). No explicit strides required -> TensorMatcher enforces
    // is_contiguous().
    TensorMatcher({P, G4})  // buf_x_sf
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(buf_x_sf);
    TensorMatcher({P, K})  // buf.topk_idx
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(buf_topk_idx);
    TensorMatcher({P, K})  // buf.topk_weights
        .with_dtype<float>()
        .with_device(device)
        .verify(buf_topk_weights);

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    const auto padded_max = static_cast<uint32_t>(P.unwrap());
    const auto hidden = static_cast<uint32_t>(H.unwrap());
    const auto top_k = static_cast<uint32_t>(K.unwrap());
    const auto num_groups_div_4 = static_cast<uint32_t>(G4.unwrap());

    RuntimeCheck(num_tokens <= padded_max, "num_tokens must not exceed padded_max");
    RuntimeCheck(hidden % kGroupSize == 0, "hidden must be a multiple of group_size");
    const auto num_groups = hidden / static_cast<uint32_t>(kGroupSize);
    RuntimeCheck(num_groups == num_groups_div_4 * 4u, "num_groups must be a multiple of 4");
    RuntimeCheck(hidden % 8u == 0, "hidden must be a multiple of 8 (16B bf16 loads)");
    const auto num_threads = hidden / 8u;
    RuntimeCheck(num_threads <= 1024, "hidden too large for single-block-per-row quant");
    RuntimeCheck(num_threads >= top_k, "top_k must fit into one quant CTA");

    const auto pad_slots = (padded_max - num_tokens) * top_k;
    const uint32_t num_pad_blocks = pad_slots == 0 ? 0u : ((pad_slots + num_threads - 1u) / num_threads);
    const auto num_total_blocks = num_tokens + num_pad_blocks;

    const auto params = MegaMoEPreDispatchParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .topk_idx = static_cast<const int32_t*>(topk_idx.data_ptr()),
        .topk_weights = static_cast<const float*>(topk_weights.data_ptr()),
        .buf_x = static_cast<fp8_e4m3_t*>(buf_x.data_ptr()),
        .buf_x_sf = static_cast<int32_t*>(buf_x_sf.data_ptr()),
        .buf_topk_idx = static_cast<int64_t*>(buf_topk_idx.data_ptr()),
        .buf_topk_weights = static_cast<float*>(buf_topk_weights.data_ptr()),
        .num_tokens = num_tokens,
        .padded_max = padded_max,
        .hidden = hidden,
        .num_groups = num_groups,
        .top_k = top_k,
    };

    if (num_total_blocks == 0) return;
    LaunchKernel(num_total_blocks, num_threads, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
