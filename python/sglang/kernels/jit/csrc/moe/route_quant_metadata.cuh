// M<=8 extension: write native eight-token-tile expert metadata after radix
// routing. At M1 the selected experts are already in ascending ID order; at M>1
// the last routing CTA to finish lays out every token's routes. Larger M runs
// routing and quantization only.
// K3 MoE-front prep in one launch: radix routing (+ trtllm packed ids) on the
// first M CTAs, mxfp8 per-token-group quant of the routed activations on the
// next M. Specialized like route_radix: 896 experts, top-16, 3584-wide bf16
// row (112 ue8m0 groups of 32 = 224 lanes). Must build without fast-math to
// keep routing bit-identical to route_radix.

#include "../gemm/per_token_group_quant.cuh"
#include "route_radix.cuh"

namespace sglang {

// Quant is the row quantizer run by CTAs [M, 2M): its Params, the routing
// weight dtype the expert body reads, a device run<TX>(quant, row), and a host
// make() that checks its tensors and builds Params.
template <typename Quant>
struct RouteQuantMetadataParams {
  RouteRadixParams route;
  typename Quant::Params quant;
  int32_t *total, *map, *perm, *cta, *limit, *count;
  typename Quant::Weight* weights;
  // Zero between launches; the last routing CTA resets it for graph replay.
  int32_t* arrivals;
};

inline constexpr uint32_t kMetadataTile_ = 8;
inline constexpr uint32_t kMetadataMaxTokens_ = kMetadataTile_;
inline constexpr uint32_t kMetadataMaxRoutes_ = kMetadataMaxTokens_ * 16;

// Same layout as the Triton prepare_expert_tile_metadata: routes ordered by
// (expert, route index), one padded tile per distinct expert. A token selects
// each expert at most once, so per-token expert bitmaps give both orders.
template <typename Quant>
__device__ void write_multi_token_metadata(const RouteQuantMetadataParams<Quant>& params, uint32_t M) {
  constexpr uint32_t kWords = (LargeRouterRadixTrait::kNumExperts + 31) / 32;
  __shared__ uint32_t token_bits[kMetadataMaxTokens_][kWords];
  __shared__ uint32_t any_bits[kWords];
  __shared__ int32_t below[kWords + 1];
  __shared__ bool is_last;
  const uint32_t tx = threadIdx.x;
  const uint32_t row = blockIdx.x;
  if (tx < 16) {
    params.weights[row * 16 + tx] =
        device::cast<typename Quant::Weight>(params.route.out_w[row * params.route.out_w_stride + tx]);
  }
  __threadfence();
  __syncthreads();
  if (tx == 0) is_last = atomicAdd(params.arrivals, 1) == static_cast<int32_t>(M - 1);
  __syncthreads();
  if (!is_last) return;
  __threadfence();
  if (tx < M * kWords) (&token_bits[0][0])[tx] = 0;
  if (tx < kWords) any_bits[tx] = 0;
  __syncthreads();
  const uint32_t routes = M * 16;
  const uint32_t token = tx / 16;
  int32_t id = 0;
  if (tx < routes) {
    id = __ldcg(params.route.out_i + token * params.route.out_i_stride + tx % 16);
    atomicOr(&token_bits[token][id / 32], 1u << (id % 32));
    atomicOr(&any_bits[id / 32], 1u << (id % 32));
  }
  __syncthreads();
  if (tx < 32) {
    int32_t total = tx < kWords ? __popc(any_bits[tx]) : 0;
#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset *= 2) {
      const int32_t up = __shfl_up_sync(0xffffffffu, total, offset);
      if (tx >= offset) total += up;
    }
    if (tx < kWords) below[tx + 1] = total;
    if (tx == 0) below[0] = 0;
  }
  __syncthreads();
  if (tx < routes) {
    const uint32_t word = id / 32, bit = id % 32;
    const int32_t rank = below[word] + __popc(any_bits[word] & ((1u << bit) - 1));
    int32_t within = 0;
    bool last = true;
    for (uint32_t t = 0; t < M; ++t) {
      const bool has = (token_bits[t][word] >> bit) & 1u;
      within += t < token && has;
      last &= !(t > token && has);
    }
    const int32_t position = rank * kMetadataTile_ + within;
    params.map[tx] = position;
    params.perm[position] = token;
    if (within == 0) params.cta[rank] = id;
    if (last) params.limit[rank] = position + 1;
  }
  if (tx == 0) {
    const int32_t unique = below[kWords];
    params.total[0] = unique * kMetadataTile_;
    params.count[0] = unique;
    *params.arrivals = 0;
  }
}

// One quant CTA covers one token row: thread pairs (2g, 2g+1) hold group g
// with lanes (0, 1), the subwarp layout the flat quant kernel derives from
// global_tid.
template <typename TX>
using RouteQuantTraitT = QuantTrait<
    TX,
    fp8_e4m3_t,
    /*kGroupSize=*/32,
    /*kUe8m0=*/true,
    /*kRowMajor=*/true,
    /*kAligned=*/true,
    /*kFuseSiluAndMul=*/false>;

using RouteQuantTrait = RouteQuantTraitT<bf16_t>;

inline constexpr uint32_t kQuantGroupsPerRow_ = LargeRouterRadixTrait::kBlockSize / RouteQuantTrait::kNumLanes;
inline constexpr uint32_t kQuantHidden_ = kQuantGroupsPerRow_ * RouteQuantTrait::kGroupSize;  // 3584

struct Mxfp8RowQuant {
  using Params = QuantKernelParams;
  using Weight = bf16_t;

  template <typename TX>
  __device__ static void run(const Params& quant, uint32_t row) {
    using Trait = RouteQuantTraitT<TX>;
    Trait::run(quant, /*expert_idx=*/0, row, threadIdx.x / Trait::kNumLanes, threadIdx.x % Trait::kNumLanes);
  }

  static Params make(
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView out_q,
      const tvm::ffi::TensorView out_s,
      const tvm::ffi::TensorView /*global_scale: unused*/,
      host::SymbolicSize& M_,
      host::SymbolicDevice& device,
      host::SymbolicDType& x_dtype) {
    using namespace host;
    TensorMatcher({M_, -1}).with_dtype<bf16_t, fp32_t>(x_dtype).with_device(device).with_strides({-1, 1}).verify(x);
    const auto quant_params =
        x_dtype.is_type<fp32_t>()
            ? build_quant_context<RouteQuantTraitT<fp32_t>, /*kMasked=*/false>(x, out_q, out_s).params
            : build_quant_context<RouteQuantTraitT<bf16_t>, /*kMasked=*/false>(x, out_q, out_s).params;
    RuntimeCheck(
        quant_params.hidden_size == kQuantHidden_, "route_quant_fused is specialized for a 3584-wide activation row");
    RuntimeCheck(
        quant_params.num_tokens == static_cast<uint32_t>(M_.unwrap()),
        "route_quant_fused: scores and activations must have the same token count");
    return quant_params;
  }
};

template <bool kUsePDL, typename TScore, typename TX, typename Quant>
__global__ __launch_bounds__(LargeRouterRadixTrait::kBlockSize)  //
    void route_quant_metadata_kernel(const __grid_constant__ RouteQuantMetadataParams<Quant> params) {
  const auto M = static_cast<uint32_t>(params.route.M);
  if (blockIdx.x < M) {
    __shared__ typename LargeRouterRadixTrait::Smem smem;
    route_radix_block<kUsePDL, TScore>(params.route, smem);
    if (M > kMetadataMaxTokens_) return;
    __syncthreads();
    const auto tx = threadIdx.x;
    if (M > 1) {
      write_multi_token_metadata(params, M);
    } else if (tx < 16) {
      params.map[tx] = tx * 8;
      params.perm[tx * 8] = 0;
      params.cta[tx] = smem.wid[tx];
      params.limit[tx] = tx * 8 + 1;
      params.weights[tx] = device::cast<typename Quant::Weight>(params.route.out_w[tx]);
      if (tx == 0) {
        params.total[0] = 128;
        params.count[0] = 16;
      }
    }
  } else {
    // Quant CTAs read the same primary-kernel output (the fused-front GEMM)
    // as the routing CTAs, so they carry their own PDL wait/trigger.
    device::PDLWaitPrimary<kUsePDL>();
    Quant::template run<TX>(params.quant, blockIdx.x - M);
    device::PDLTriggerSecondary<kUsePDL>();
  }
}

template <bool kUsePDL, typename Quant>
struct RouteQuantMetadataKernel {
  static void
  run(const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView out_w,
      const tvm::ffi::TensorView out_i,
      const tvm::ffi::TensorView out_packed,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView out_q,
      const tvm::ffi::TensorView out_s,
      const tvm::ffi::TensorView total,
      const tvm::ffi::TensorView map,
      const tvm::ffi::TensorView perm,
      const tvm::ffi::TensorView weights,
      const tvm::ffi::TensorView cta,
      const tvm::ffi::TensorView limit,
      const tvm::ffi::TensorView count,
      const tvm::ffi::TensorView arrivals,
      const tvm::ffi::TensorView global_scale,
      int64_t topk,
      double routed_scaling_factor,
      bool renormalize,
      bool apply_scale) {
    using namespace host;
    using Trait = RouteQuantTrait;

    auto M_ = SymbolicSize{"num_tokens"};
    auto N_ = SymbolicSize{"num_experts"};
    auto K_ = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    auto score_dtype = SymbolicDType{};
    TensorMatcher({M_, N_})
        .with_dtype<bf16_t, fp32_t>(score_dtype)
        .with_device(device)
        .with_strides({-1, 1})
        .verify(scores);
    TensorMatcher({N_}).with_dtype<fp32_t>().with_device(device).verify(bias);
    TensorMatcher({M_, K_}).with_dtype<fp32_t>().with_device(device).verify(out_w);
    TensorMatcher({M_, K_}).with_dtype<int32_t>().with_device(device).verify(out_i);
    TensorMatcher({M_, K_}).with_dtype<int32_t>().with_strides({-1, 1}).with_device(device).verify(out_packed);

    RuntimeCheck(
        N_.unwrap() == kNumExperts_ && K_.unwrap() == kTopK_ && topk == kTopK_,
        "route_quant_fused is specialized for N=896, K=16");
    RuntimeCheck(scores.stride(0) % 4 == 0, "route_quant_fused: scores row stride must be a multiple of 4");

    auto x_dtype = SymbolicDType{};
    const auto quant_params = Quant::make(x, out_q, out_s, global_scale, M_, device, x_dtype);

    const auto M = static_cast<uint32_t>(M_.unwrap());
    RuntimeCheck(M >= 1, "fused routing preparation needs at least one token");
    if (M <= kMetadataMaxTokens_) {
      const int64_t routes = M * 16;
      TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(total);
      TensorMatcher({routes}).with_dtype<int32_t>().with_device(device).verify(map);
      TensorMatcher({routes * kMetadataTile_ + 1}).with_dtype<int32_t>().with_device(device).verify(perm);
      TensorMatcher({static_cast<int64_t>(M), 16})
          .with_dtype<typename Quant::Weight>()
          .with_device(device)
          .verify(weights);
      TensorMatcher({routes}).with_dtype<int32_t>().with_device(device).verify(cta);
      TensorMatcher({routes}).with_dtype<int32_t>().with_device(device).verify(limit);
      TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(count);
      TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(arrivals);
    }

    const auto params = RouteQuantMetadataParams<Quant>{
        .route =
            {scores.data_ptr(),
             static_cast<const fp32_t*>(bias.data_ptr()),
             static_cast<fp32_t*>(out_w.data_ptr()),
             static_cast<int32_t*>(out_i.data_ptr()),
             static_cast<int32_t*>(out_packed.data_ptr()),
             static_cast<int>(M),
             static_cast<long long>(scores.stride(0)),
             static_cast<long long>(out_w.stride(0)),
             static_cast<long long>(out_i.stride(0)),
             static_cast<long long>(out_packed.stride(0)),
             static_cast<float>(routed_scaling_factor),
             renormalize ? 1 : 0,
             apply_scale ? 1 : 0,
             /*sorted=*/0},
        .quant = quant_params,
        .total = static_cast<int32_t*>(total.data_ptr()),
        .map = static_cast<int32_t*>(map.data_ptr()),
        .perm = static_cast<int32_t*>(perm.data_ptr()),
        .cta = static_cast<int32_t*>(cta.data_ptr()),
        .limit = static_cast<int32_t*>(limit.data_ptr()),
        .count = static_cast<int32_t*>(count.data_ptr()),
        .weights = static_cast<typename Quant::Weight*>(weights.data_ptr()),
        .arrivals = static_cast<int32_t*>(arrivals.data_ptr()),
    };

#define SGL_ROUTE_QUANT_LAUNCH(TS, TX)                                    \
  LaunchKernel(2 * M, LargeRouterRadixTrait::kBlockSize, device.unwrap()) \
      .enable_pdl(kUsePDL)(route_quant_metadata_kernel<kUsePDL, TS, TX, Quant>, params)

    if (score_dtype.is_type<fp32_t>()) {
      if (x_dtype.is_type<fp32_t>()) {
        SGL_ROUTE_QUANT_LAUNCH(fp32_t, fp32_t);
      } else {
        SGL_ROUTE_QUANT_LAUNCH(fp32_t, bf16_t);
      }
    } else {
      if (x_dtype.is_type<fp32_t>()) {
        SGL_ROUTE_QUANT_LAUNCH(bf16_t, fp32_t);
      } else {
        SGL_ROUTE_QUANT_LAUNCH(bf16_t, bf16_t);
      }
    }
#undef SGL_ROUTE_QUANT_LAUNCH
  }
};

}  // namespace sglang
