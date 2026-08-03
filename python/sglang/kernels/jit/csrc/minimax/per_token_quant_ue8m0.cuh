#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::pack_fp8;

// Per-token group quant to FP8-e4m3 with a fused UE8M0 scale. Each group of
// kGroupSize columns gets one UE8M0 exponent byte written contiguously in
// row-major order into ``x_sf`` (int32 [num_tokens, num_groups/4], 4 group
// bytes per int32). This is the deep_gemm "transform_sf" pack done inline in
// the quant, reusing the dsv4 ``cast_to_ue8m0``/``pack_fp8`` primitives -- it
// is byte-identical to ``per_token_group_quant_fp8(scale_ue8m0=True)`` followed
// by ``transform_sf_into_required_layout`` (both round via ceil(log2(absmax/
// FP8_MAX))), but emits no separate transpose/pack kernel.
struct PerTokenQuantUe8m0Params {
  const bf16_t* __restrict__ x;  // [num_tokens, hidden]
  fp8_e4m3_t* __restrict__ x_q;  // [num_tokens, hidden]
  int32_t* __restrict__ x_sf;    // [num_tokens, num_groups/4]; written as bytes
  uint32_t num_tokens;
  uint32_t hidden;
  uint32_t num_groups;  // hidden / kGroupSize
};

template <uint32_t kGroupSize, bool kUsePDL>
__global__ __launch_bounds__(1024, 2) void  //
    per_token_quant_ue8m0_kernel(const PerTokenQuantUe8m0Params __grid_constant__ params) {
  using namespace device;
  constexpr uint32_t kVecElems = 8;  // 8 bf16 = 16B load per thread
  static_assert(kGroupSize % kVecElems == 0, "group_size must be a multiple of 8");
  constexpr uint32_t kThreadsPerGroup = kGroupSize / kVecElems;
  using InputVec = AlignedVector<bf16x2_t, kVecElems / 2>;
  using OutputVec = AlignedVector<fp8x2_e4m3_t, kVecElems / 2>;

  const uint32_t token_id = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  PDLWaitPrimary<kUsePDL>();

  const auto token_in = params.x + static_cast<uint64_t>(token_id) * params.hidden;
  const auto token_out = params.x_q + static_cast<uint64_t>(token_id) * params.hidden;

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
  const float inv_scale = __uint_as_float((127u + 127u - ue8m0_exp) << 23);

  OutputVec out_vec;
#pragma unroll
  for (uint32_t i = 0; i < kVecElems / 2; ++i) {
    out_vec[i] = pack_fp8(vals[2 * i + 0] * inv_scale, vals[2 * i + 1] * inv_scale);
  }
  out_vec.store(token_out, tid);

  const uint32_t group_id = tid / kThreadsPerGroup;
  const uint32_t within_group_id = tid % kThreadsPerGroup;
  if (within_group_id == 0 && group_id < params.num_groups) {
    const uint32_t byte_off = token_id * params.num_groups + group_id;
    reinterpret_cast<uint8_t*>(params.x_sf)[byte_off] = static_cast<uint8_t>(ue8m0_exp);
  }
  PDLTriggerSecondary<kUsePDL>();
}

// Fused quant + scatter: like per_token_quant_ue8m0_kernel, but instead of
// writing the fp8/scale for the single source token, it scatters them straight
// into the permuted grouped-GEMM input -- replicating each token to its ``topk``
// destination rows -- so the separate fill_gateup_input_triton_kernel launch (and
// the intermediate x_q/x_sf buffers) are eliminated. The fp8 value + UE8M0 scale
// are computed exactly once per token (identical to the non-fused kernel); only
// the stores differ.
struct PerTokenQuantUe8m0ScatterParams {
  const bf16_t* __restrict__ x;              // [num_tokens, hidden]
  fp8_e4m3_t* __restrict__ gateup_input;     // [E, m_max, hidden]
  int32_t* __restrict__ gateup_input_scale;  // [E, num_groups/4, m_max] int32 (MN-major), written as bytes
  const int32_t* __restrict__ src2dst;       // [num_tokens, topk] -> dst row = expert*m_max + slot
  const int32_t* __restrict__ topk_ids;      // [num_tokens, topk]; <0 = skip
  uint32_t num_tokens;
  uint32_t hidden;
  uint32_t num_groups;  // hidden / kGroupSize
  uint32_t topk;
  uint32_t m_max;
};

template <uint32_t kGroupSize, uint32_t kTopK, bool kUsePDL>
__global__ __launch_bounds__(1024, 2) void  //
    per_token_quant_ue8m0_scatter_kernel(const PerTokenQuantUe8m0ScatterParams __grid_constant__ params) {
  using namespace device;
  constexpr uint32_t kVecElems = 8;  // 8 bf16 = 16B load per thread
  static_assert(kGroupSize % kVecElems == 0, "group_size must be a multiple of 8");
  static_assert(kTopK <= kWarpThreads, "kTopK must fit in a warp for the lane-parallel scale write");
  constexpr uint32_t kThreadsPerGroup = kGroupSize / kVecElems;
  using InputVec = AlignedVector<bf16x2_t, kVecElems / 2>;
  using OutputVec = AlignedVector<fp8x2_e4m3_t, kVecElems / 2>;

  const uint32_t token_id = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  PDLWaitPrimary<kUsePDL>();

  const auto token_in = params.x + static_cast<uint64_t>(token_id) * params.hidden;

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
  // Butterfly reduce: every thread of the group ends up with the group absmax
  // (so all of them can write scale bytes below, no broadcast needed).
  local_max = warp::reduce_max<kThreadsPerGroup>(local_max);
  const float absmax = fmaxf(local_max, 1e-10f);
  const float raw_scale = absmax / math::FP8_E4M3_MAX;
  const uint32_t ue8m0_exp = cast_to_ue8m0(raw_scale);
  const float inv_scale = __uint_as_float((127u + 127u - ue8m0_exp) << 23);

  OutputVec out_vec;
#pragma unroll
  for (uint32_t i = 0; i < kVecElems / 2; ++i) {
    out_vec[i] = pack_fp8(vals[2 * i + 0] * inv_scale, vals[2 * i + 1] * inv_scale);
  }

  // Read this token's kTopK destinations once (fully unrolled).
  const auto* src2dst_row = params.src2dst + static_cast<uint64_t>(token_id) * kTopK;
  const auto* topk_ids_row = params.topk_ids + static_cast<uint64_t>(token_id) * kTopK;
  int32_t dst_rows[kTopK];
#pragma unroll
  for (uint32_t i = 0; i < kTopK; ++i) {
    dst_rows[i] = (topk_ids_row[i] >= 0) ? src2dst_row[i] : -1;
  }

  const uint32_t group_id = tid / kThreadsPerGroup;
  const uint32_t within_group = tid % kThreadsPerGroup;
  const uint32_t c = group_id / 4u;  // packed int32 index along the group axis
  const uint32_t b = group_id % 4u;  // byte within that int32
  const uint64_t scale_g4 = params.num_groups / 4u;
  auto* scale_bytes = reinterpret_cast<uint8_t*>(params.gateup_input_scale);

  // 1) Output fp8: every thread replicates its 16B chunk to all kTopK dst rows.
#pragma unroll
  for (uint32_t i = 0; i < kTopK; ++i) {
    const int32_t dst = dst_rows[i];
    if (dst < 0) continue;
    out_vec.store(params.gateup_input + static_cast<uint64_t>(dst) * params.hidden, tid);
  }

  // 2) Scale bytes: distribute the kTopK experts across the group's threads
  // (each already holds the group exponent) so they write in parallel instead
  // of one leader looping. lane `within_group` handles experts {within_group,
  // within_group + kThreadsPerGroup, ...}; for kTopK <= kThreadsPerGroup that is
  // exactly one expert per lane (no loop).
#pragma unroll
  for (uint32_t i = within_group; i < kTopK; i += kThreadsPerGroup) {
    const int32_t dst = dst_rows[i];
    if (dst < 0) continue;
    const uint32_t expert = static_cast<uint32_t>(dst) / params.m_max;
    const uint32_t m = static_cast<uint32_t>(dst) % params.m_max;
    // int32 element [expert, c, m] of [E, G/4, m_max], byte b within it.
    const uint64_t int32_index =
        static_cast<uint64_t>(expert) * scale_g4 * params.m_max + static_cast<uint64_t>(c) * params.m_max + m;
    scale_bytes[int32_index * 4u + b] = static_cast<uint8_t>(ue8m0_exp);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kGroupSize, int64_t kTopK, bool kUsePDL>
void per_token_quant_ue8m0_scatter(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView gateup_input,
    tvm::ffi::TensorView gateup_input_scale,
    tvm::ffi::TensorView src2dst,
    tvm::ffi::TensorView topk_ids,
    int64_t topk,
    int64_t m_max) {
  using namespace host;
  auto device = SymbolicDevice{};
  auto M = SymbolicSize{"num_tokens"};
  auto H = SymbolicSize{"hidden"};
  auto E = SymbolicSize{"num_experts"};
  auto MM = SymbolicSize{"m_max"};
  auto G4 = SymbolicSize{"num_groups_div_4"};
  device.set_options<kDLCUDA>();
  TensorMatcher({M, H}).with_dtype<bf16_t>().with_device(device).verify(x);
  TensorMatcher({E, MM, H}).with_dtype<fp8_e4m3_t>().with_device(device).verify(gateup_input);
  TensorMatcher({E, G4, MM}).with_dtype<int32_t>().with_device(device).verify(gateup_input_scale);

  const uint32_t num_tokens = static_cast<uint32_t>(M.unwrap());
  const uint32_t hidden = static_cast<uint32_t>(H.unwrap());
  RuntimeCheck(hidden % kGroupSize == 0, "hidden ", hidden, " not divisible by group_size ", kGroupSize);
  const uint32_t num_groups = hidden / static_cast<uint32_t>(kGroupSize);
  RuntimeCheck(num_groups % 4 == 0, "num_groups must be a multiple of 4 for int32 packing");
  RuntimeCheck(static_cast<uint32_t>(G4.unwrap()) * 4 == num_groups, "scale G/4 mismatch");
  RuntimeCheck(static_cast<uint32_t>(MM.unwrap()) == static_cast<uint32_t>(m_max), "m_max mismatch");
  const uint32_t threads = hidden / 8;  // kVecElems
  RuntimeCheck(threads <= 1024, "hidden/8 must be <= 1024, got ", threads);
  RuntimeCheck(topk == kTopK, "topk does not match compiled template");

  const auto params = PerTokenQuantUe8m0ScatterParams{
      .x = static_cast<const bf16_t*>(x.data_ptr()),
      .gateup_input = static_cast<fp8_e4m3_t*>(gateup_input.data_ptr()),
      .gateup_input_scale = static_cast<int32_t*>(gateup_input_scale.data_ptr()),
      .src2dst = static_cast<const int32_t*>(src2dst.data_ptr()),
      .topk_ids = static_cast<const int32_t*>(topk_ids.data_ptr()),
      .num_tokens = num_tokens,
      .hidden = hidden,
      .num_groups = num_groups,
      .topk = static_cast<uint32_t>(kTopK),
      .m_max = static_cast<uint32_t>(m_max),
  };
  if (num_tokens == 0) return;
  constexpr auto kernel = per_token_quant_ue8m0_scatter_kernel<kGroupSize, kTopK, kUsePDL>;
  LaunchKernel(num_tokens, threads, device.unwrap())  //
      .enable_pdl(kUsePDL)(kernel, params);
}

template <int64_t kGroupSize, bool kUsePDL>
void per_token_quant_ue8m0(tvm::ffi::TensorView x, tvm::ffi::TensorView x_q, tvm::ffi::TensorView x_sf) {
  using namespace host;
  auto device = SymbolicDevice{};
  auto M = SymbolicSize{"num_tokens"};
  auto H = SymbolicSize{"hidden"};
  auto G4 = SymbolicSize{"num_groups_div_4"};
  device.set_options<kDLCUDA>();
  TensorMatcher({M, H}).with_dtype<bf16_t>().with_device(device).verify(x);
  TensorMatcher({M, H}).with_dtype<fp8_e4m3_t>().with_device(device).verify(x_q);
  TensorMatcher({M, G4}).with_dtype<int32_t>().with_device(device).verify(x_sf);

  const uint32_t num_tokens = static_cast<uint32_t>(M.unwrap());
  const uint32_t hidden = static_cast<uint32_t>(H.unwrap());
  RuntimeCheck(hidden % kGroupSize == 0, "hidden ", hidden, " not divisible by group_size ", kGroupSize);
  const uint32_t num_groups = hidden / static_cast<uint32_t>(kGroupSize);
  RuntimeCheck(static_cast<uint32_t>(G4.unwrap()) * 4 == num_groups);
  const uint32_t threads = hidden / 8;  // kVecElems
  RuntimeCheck(threads <= 1024, "hidden/8 must be <= 1024, got ", threads);

  const auto params = PerTokenQuantUe8m0Params{
      .x = static_cast<const bf16_t*>(x.data_ptr()),
      .x_q = static_cast<fp8_e4m3_t*>(x_q.data_ptr()),
      .x_sf = static_cast<int32_t*>(x_sf.data_ptr()),
      .num_tokens = num_tokens,
      .hidden = hidden,
      .num_groups = num_groups,
  };
  if (num_tokens == 0) return;
  constexpr auto kernel = per_token_quant_ue8m0_kernel<kGroupSize, kUsePDL>;
  LaunchKernel(num_tokens, threads, device.unwrap())  //
      .enable_pdl(kUsePDL)(kernel, params);
}

}  // namespace
