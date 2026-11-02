// Fused MoE finalize + shared-expert add for the deferred trtllm-gen MoE output:
//   out[t] = sum_k w[t, k] * gemm2_out[permuted_idx[t * K + k]] (+ shared[t])
// Rows whose permuted index is -1 are skipped; the fp32 accumulation runs in k
// order with the shared row added last, the arithmetic of flashinfer's finalize.
//
// Each thread owns kVec hidden elements of a token and issues all K gathers at
// once, so a token costs the index load plus one gather round trip. Tokens map
// to blockIdx.y, which spreads even a single token over H / (kVec * kBlockSize)
// CTAs. A token's route (K row indices and weights) is spread over the warp,
// lane k holding slot k, so it costs two registers and the K gathers in flight
// get the rest; that is what keeps a top-k 16 decode batch in one wave.
// Past two resident waves a persistent variant walks several tokens per CTA,
// loading the next token's route while the current gathers are in flight.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/cuda/device_guard.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace sglang {

namespace device::moe_finalize {

inline constexpr uint32_t kBlockSize = 128;

namespace ptx {

// One slot's gather: zero the registers and load them under the predicate in a single asm. A skipped
// slot (index -1, weight -0.0) then adds -0.0 * 0, which leaves the sum bitwise unchanged, and costs no
// extra registers (a C++ conditional load makes ptxas keep a second copy of every vector). The asm is
// volatile so that the K gathers of a token stay ahead of the FMAs.
template <std::size_t kWords>
SGL_DEVICE void ld_global_or_zero(void* dst, const void* ptr, bool valid) {
  const auto r = static_cast<uint32_t*>(dst);
  const auto pred = static_cast<uint32_t>(valid);
  if constexpr (kWords == 4) {
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %5, 0;\n\tmov.b32 %0, 0;\n\tmov.b32 %1, 0;\n\t"
        "mov.b32 %2, 0;\n\tmov.b32 %3, 0;\n\t@p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n\t}"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "l"(ptr), "r"(pred));
  } else if constexpr (kWords == 2) {
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %3, 0;\n\tmov.b32 %0, 0;\n\tmov.b32 %1, 0;\n\t"
        "@p ld.global.v2.u32 {%0, %1}, [%2];\n\t}"
        : "=r"(r[0]), "=r"(r[1])
        : "l"(ptr), "r"(pred));
  } else {
    static_assert(kWords == 1);
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %2, 0;\n\tmov.b32 %0, 0;\n\t@p ld.global.u32 %0, [%1];\n\t}"
                 : "=r"(r[0])
                 : "l"(ptr), "r"(pred));
  }
}

}  // namespace ptx

struct Params {
  const bf16_t* __restrict__ gemm2_out;
  const int32_t* __restrict__ permuted_idx;
  const void* __restrict__ expert_weights;
  const bf16_t* __restrict__ shared;
  bf16_t* __restrict__ out;
  int64_t out_stride;     // elements per output row
  uint32_t gemm2_stride;  // elements per gemm2_out row
  uint32_t num_tokens;
  uint32_t num_chunks;     // hidden size / kVec
  bool route_before_wait;  // see MoeFinalizeKernel
};

template <typename TW, uint32_t K, uint32_t kVec, bool kHasShared>
struct Token {
  using wbits_t = std::conditional_t<sizeof(TW) == 2, uint16_t, uint32_t>;
  // -0.0 of the weight type
  static constexpr uint32_t kNegZero = sizeof(TW) == 2 ? 0x8000u : 0x80000000u;

  int32_t row;      // lane k < K: permuted row of slot k
  uint32_t weight;  // lane k < K: bits of the weight of slot k in its own type, -0.0 for a skipped slot
  AlignedVector<bf16x2_t, kVec / 2> shared;

  SGL_DEVICE void load(const Params& p, const uint32_t token, const uint32_t chunk, const uint32_t lane_id) {
    if constexpr (kHasShared) shared.load(p.shared + token * (int64_t{p.num_chunks} * kVec), chunk);
    row = -1;
    weight = 0;
    if (lane_id < K) {
      // both loads unconditional: a weight load guarded by the row waits for it
      row = p.permuted_idx[token * K + lane_id];
      weight = static_cast<const wbits_t*>(p.expert_weights)[token * K + lane_id];
      if (row == -1) weight = kNegZero;
    }
  }
};

// kPersistent: the grid holds one resident wave and each CTA walks tokens with
// stride gridDim.y, prefetching the next token's route; otherwise every token
// has its own CTA row and the body runs once (the launcher picks it only while
// the grid fits one wave).
// route_before_wait: the caller guarantees that the route and the shared row
// were complete when this kernel launched, so they load before the PDL wait.
template <typename TW, uint32_t K, uint32_t kVec, bool kPersistent, bool kHasShared, bool kUsePDL>
__global__ __launch_bounds__(kBlockSize, 1) void finalize_kernel(const __grid_constant__ Params p) {
  static_assert(K <= 32, "the route is spread over one warp");
  constexpr uint32_t kPairs = kVec / 2;
  using pairs_t = AlignedVector<bf16x2_t, kPairs>;
  using token_t = Token<TW, K, kVec, kHasShared>;
  const uint32_t lane_id = threadIdx.x % 32;
  const uint32_t chunk = blockIdx.x * kBlockSize + threadIdx.x;
  // Lanes past the row compute on its last chunk and skip the store, so that
  // every lane of the warp takes part in the shuffles.
  const uint32_t safe_chunk = std::min(chunk, p.num_chunks - 1);
  const uint32_t col = safe_chunk * kVec;

  if (!p.route_before_wait) PDLWaitPrimary<kUsePDL>();
  uint32_t token = blockIdx.y;
  token_t cur;
  cur.load(p, token, safe_chunk, lane_id);
  if (p.route_before_wait) PDLWaitPrimary<kUsePDL>();
  do {
    pairs_t in[K];
#pragma unroll
    for (uint32_t k = 0; k < K; ++k) {
      const int32_t row = __shfl_sync(0xffffffffu, cur.row, k);
      const auto src = p.gemm2_out + static_cast<uint32_t>(max(row, 0)) * p.gemm2_stride + col;
      ptx::ld_global_or_zero<kPairs>(&in[k], src, row != -1);
    }
    asm volatile("" ::: "memory");  // next-token loads go after the gathers (as measured)
    const uint32_t next = token + gridDim.y;
    token_t next_token;
    if constexpr (kPersistent) {
      if (next < p.num_tokens) next_token.load(p, next, safe_chunk, lane_id);
    }

    fp32x2_t acc[kPairs];
#pragma unroll
    for (uint32_t e = 0; e < kPairs; ++e) {
      acc[e] = {0.0f, 0.0f};
    }
#pragma unroll
    for (uint32_t k = 0; k < K; ++k) {
      const uint32_t w_bits = __shfl_sync(0xffffffffu, cur.weight, k);
      if constexpr (std::is_same_v<TW, bf16_t>) {
        // mixed-precision FMA: bf16 x bf16 is exact in fp32, so this is the fp32 FMA bit for bit
        const auto w = __ushort_as_bfloat16(static_cast<uint16_t>(w_bits));
#pragma unroll
        for (uint32_t e = 0; e < kPairs; ++e) {
          acc[e].x = math::fma_f32_bf16(in[k][e].x, w, acc[e].x);
          acc[e].y = math::fma_f32_bf16(in[k][e].y, w, acc[e].y);
        }
      } else {
        const auto w = __uint_as_float(w_bits);
#pragma unroll
        for (uint32_t e = 0; e < kPairs; ++e) {
          acc[e] = math::fma_f32x2({w, w}, cast<fp32x2_t>(in[k][e]), acc[e]);
        }
      }
    }
    pairs_t out;
#pragma unroll
    for (uint32_t e = 0; e < kPairs; ++e) {
      if constexpr (kHasShared) {
        const auto sh = cast<fp32x2_t>(cur.shared[e]);
        acc[e].x += sh.x;
        acc[e].y += sh.y;
      }
      out[e] = cast<bf16x2_t>(acc[e]);
    }
    if (chunk < p.num_chunks) out.store(p.out + token * p.out_stride, chunk);

    if constexpr (kPersistent) {
      if (next >= p.num_tokens) break;
      token = next;
      cur = next_token;
    }
  } while (kPersistent);
  PDLTriggerSecondary<kUsePDL>();
}

}  // namespace device::moe_finalize

// One token per CTA row while the grid fits one resident wave, the persistent
// kernel past it.
template <typename TW, uint32_t K, uint32_t kVec, bool kHasShared, bool kUsePDL>
struct MoeFinalizeKernel {
  template <bool kPersistent>
  static constexpr auto kernel = device::moe_finalize::finalize_kernel<TW, K, kVec, kPersistent, kHasShared, kUsePDL>;

  static void
  run(const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView gemm2_out,
      const tvm::ffi::TensorView permuted_idx,
      const tvm::ffi::TensorView expert_weights,
      const tvm::ffi::Optional<tvm::ffi::TensorView> shared,
      const bool route_before_wait) {
    using namespace host;
    using device::moe_finalize::kBlockSize;
    constexpr uint32_t kMaxGridY = 65535;
    auto T = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"hidden_size"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();
    constexpr std::size_t kVecBytes = kVec * sizeof(bf16_t);
    TensorMatcher({T, H})  // out
        .with_strides({-1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(kVecBytes)
        .verify(out);
    TensorMatcher({-1, -1})  // gemm2_out
        .with_strides({-1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(kVecBytes)
        .verify(gemm2_out);
    TensorMatcher({-1})  // permuted_idx
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(permuted_idx);
    TensorMatcher({T, K})  // expert_weights
        .with_dtype<TW>()
        .with_device(device)
        .verify(expert_weights);
    if constexpr (kHasShared) {
      CHECK_HOST(shared.has_value()) << "moe_finalize: built for a shared output";
      TensorMatcher({T, H})  // shared
          .with_dtype<bf16_t>()
          .with_device(device)
          .ensure_alignment(kVecBytes)
          .verify(shared.value());
    }
    const auto num_tokens = static_cast<uint32_t>(T.unwrap());
    const auto hidden = H.unwrap();
    CHECK_HOST(permuted_idx.size(0) == int64_t{num_tokens} * K) << "moe_finalize: permuted_idx size";
    CHECK_HOST(hidden % kVec == 0 && hidden <= gemm2_out.size(1)) << "moe_finalize: hidden size " << hidden;
    CHECK_HOST(gemm2_out.size(0) * gemm2_out.stride(0) <= int64_t{UINT32_MAX})
        << "moe_finalize: gemm2_out too large for 32-bit offsets";
    if (num_tokens == 0) return;

    const auto params = device::moe_finalize::Params{
        .gemm2_out = static_cast<const bf16_t*>(gemm2_out.data_ptr()),
        .permuted_idx = static_cast<const int32_t*>(permuted_idx.data_ptr()),
        .expert_weights = expert_weights.data_ptr(),
        .shared = kHasShared ? static_cast<const bf16_t*>(shared.value().data_ptr()) : nullptr,
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .out_stride = out.stride(0),
        .gemm2_stride = static_cast<uint32_t>(gemm2_out.stride(0)),
        .num_tokens = num_tokens,
        .num_chunks = static_cast<uint32_t>(hidden / kVec),
        .route_before_wait = kUsePDL && route_before_wait,
    };
    const auto dev = device.unwrap();
    // The stream comes from the inputs' device, which may differ from the current one.
    const tvm::ffi::CUDADeviceGuard guard(dev.device_id);
    const uint32_t grid_x = div_ceil(params.num_chunks, kBlockSize);
    if (uint64_t{grid_x} * num_tokens <= resident_ctas<false>(dev.device_id)) {
      LaunchKernel(dim3{grid_x, num_tokens}, kBlockSize, dev)  //
          .config({.use_pdl = kUsePDL})(kernel<false>, params);
    } else {
      const uint32_t wave_y = std::max(1u, resident_ctas<true>(dev.device_id) / grid_x);
      LaunchKernel(dim3{grid_x, std::min({num_tokens, wave_y, kMaxGridY})}, kBlockSize, dev)  //
          .config({.use_pdl = kUsePDL})(kernel<true>, params);
    }
  }

  // CTAs of the kernel the device holds at once (one wave).
  template <bool kPersistent>
  static uint32_t resident_ctas(int device_id) {
    static const uint32_t s_blocks_per_sm =
        host::runtime::get_blocks_per_sm(kernel<kPersistent>, device::moe_finalize::kBlockSize);
    return host::runtime::get_sm_count(device_id) * s_blocks_per_sm;
  }
};

}  // namespace sglang
