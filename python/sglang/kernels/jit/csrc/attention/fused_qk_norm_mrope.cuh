#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/impl/norm.cuh>

#include <cstdint>
#include <cuda_bf16.h>

namespace sglang {

struct QKNormMRoPEParams {
  void* q;
  void* k;
  const void* q_weight;
  const void* k_weight;
  const bf16_t* cache;
  const int64_t* positions;
  const int64_t* axis_map;
  int64_t q_stride;
  int64_t k_stride;
  int64_t position_stride;
  uint32_t q_heads;
  uint32_t k_heads;
  uint32_t tokens;
  float eps;
};

/// Normalize each Q/K head, then apply full-width NeoX multimodal RoPE in place.
template <bool kUsePDL>
__global__ void fused_qk_norm_mrope_kernel(const QKNormMRoPEParams __grid_constant__ p) {
  using namespace device;
  using Storage = norm::StorageType<bf16_t, 128>;
  const auto lane = get_lane_id();
  const auto work = blockIdx.x * 4 + threadIdx.x / kWarpThreads;
  if (work >= (p.q_heads + p.k_heads) * p.tokens) return;
  const auto token = work / (p.q_heads + p.k_heads);
  const auto head = work % (p.q_heads + p.k_heads);
  const bool is_q = head < p.q_heads;
  auto ptr = is_q ? pointer::offset(p.q, 2 * (token * p.q_stride + head * 128))
                  : pointer::offset(p.k, 2 * (token * p.k_stride + (head - p.q_heads) * 128));
  const auto gmem = tile::Memory<Storage>::warp();
  PDLWaitPrimary<kUsePDL>();
  const auto value = gmem.load(ptr);
  const auto weight = gmem.load(is_q ? p.q_weight : p.k_weight);
  // Reuse the unfused kernel's reduction and BF16 rounding before rotation.
  auto out = norm::apply_norm_warp<128>(value, weight, p.eps);
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const auto f = cast<fp32x2_t>(out[j]);
    const auto partner = make_float2(__shfl_xor_sync(0xffffffff, f.x, 16), __shfl_xor_sync(0xffffffff, f.y, 16));
    bf16_t values[2] = {__float2bfloat16(f.x), __float2bfloat16(f.y)};
    const bf16_t partners[2] = {__float2bfloat16(partner.x), __float2bfloat16(partner.y)};
#pragma unroll
    for (int v = 0; v < 2; ++v) {
      const auto d = (lane * 4 + 2 * j + v) % 64;
      const auto position = p.positions[p.axis_map[d] * p.position_stride + token];
      const auto c = p.cache[position * 128 + d];
      const auto s = p.cache[position * 128 + d + 64];
      const auto a = lane < 16 ? values[v] : partners[v];
      const auto b = lane < 16 ? partners[v] : values[v];
      // Preserve the Triton MRoPE kernel's BF16 FMA contraction order.
      values[v] = lane < 16 ? __hfma(a, c, __hneg(__hmul(b, s))) : __hfma(a, s, __hmul(b, c));
    }
    out[j] = __halves2bfloat162(values[0], values[1]);
  }
  gmem.store(ptr, out);
  PDLTriggerSecondary<kUsePDL>();
}

/// Validated entry point for BF16 Q/K with 128-element heads.
template <bool kUsePDL>
struct FusedQKNormMRoPE {
  static void
  run(tvm::ffi::TensorView q,
      tvm::ffi::TensorView k,
      tvm::ffi::TensorView q_weight,
      tvm::ffi::TensorView k_weight,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView positions,
      tvm::ffi::TensorView axis_map,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"tokens"};
    auto Q = SymbolicSize{"q_width"};
    auto K = SymbolicSize{"k_width"};
    auto Sq = SymbolicSize{"q_stride"};
    auto Sk = SymbolicSize{"k_stride"};
    auto Sp = SymbolicSize{"position_stride"};
    auto C = SymbolicSize{"cache_length"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N, Q}).with_strides({Sq, 1}).with_dtype<bf16_t>().with_device(device).ensure_alignment(8).verify(q);
    TensorMatcher({N, K}).with_strides({Sk, 1}).with_dtype<bf16_t>().with_device(device).ensure_alignment(8).verify(k);
    TensorMatcher({128}).with_dtype<bf16_t>().with_device(device).ensure_alignment(8).verify(q_weight).verify(k_weight);
    TensorMatcher({C, 128}).with_dtype<bf16_t>().with_device(device).verify(cache);
    TensorMatcher({3, N}).with_strides({Sp, 1}).with_dtype<int64_t>().with_device(device).verify(positions);
    TensorMatcher({64}).with_dtype<int64_t>().with_device(device).verify(axis_map);
    CHECK_HOST(Q.unwrap() > 0 && K.unwrap() > 0);
    CHECK_HOST(Q.unwrap() % 128 == 0 && K.unwrap() % 128 == 0);
    if (N.unwrap() == 0) return;
    const auto p = QKNormMRoPEParams{
        q.data_ptr(),
        k.data_ptr(),
        q_weight.data_ptr(),
        k_weight.data_ptr(),
        static_cast<const bf16_t*>(cache.data_ptr()),
        static_cast<const int64_t*>(positions.data_ptr()),
        static_cast<const int64_t*>(axis_map.data_ptr()),
        Sq.unwrap(),
        Sk.unwrap(),
        Sp.unwrap(),
        static_cast<uint32_t>(Q.unwrap() / 128),
        static_cast<uint32_t>(K.unwrap() / 128),
        static_cast<uint32_t>(N.unwrap()),
        eps};
    LaunchKernel(div_ceil((p.q_heads + p.k_heads) * p.tokens, 4u), 128, device.unwrap())
        .enable_pdl(kUsePDL)(fused_qk_norm_mrope_kernel<kUsePDL>, p);
  }
};
}  // namespace sglang
