#include <sgl_kernel/distributed/communicator.cuh>

#include "../../moe/route_quant_fused.cuh"

namespace sglang {

template <uint32_t kWorld, uint32_t kHidden>
struct ShardedFrontLayout {
  static_assert(kWorld > 1 && kHidden % (kWorld * 32) == 0);
  static constexpr uint32_t kLocal = kHidden / kWorld;
  static constexpr uint32_t kGroups = kHidden / 32;
  static constexpr uint32_t kLocalGroups = kLocal / 32;
  static constexpr uint32_t kScaleStride = (kLocalGroups + 15) / 16 * 16;
  static constexpr uint32_t kScaleRowBytes = kWorld * kScaleStride;
  static constexpr uint32_t kPackedGroups = (kGroups + 3) / 4;
  using Quant = QuantTrait<fp32_t, fp8_e4m3_t, 32, true, true, kLocalGroups % 4 == 0, false>;
};

struct RouteQuantGatherParams {
  RouteRadixParams route;
  QuantKernelParams quant;
  uint8_t* q_mc;
  uint8_t* scales_mc;
  const uint8_t* scales_scratch;
  uint32_t* scales_out;
  device::distributed::Semaphore* sem_local;
  device::distributed::Semaphore* sem_mc;
  uint32_t rank;
};

template <uint32_t kWorld, uint32_t kHidden, bool kUsePDL>
__global__ __launch_bounds__(LargeRouterRadixTrait::kBlockSize) void route_quant_gather_kernel(
    const __grid_constant__ RouteQuantGatherParams p) {
  using Layout = ShardedFrontLayout<kWorld, kHidden>;
  using Quant = typename Layout::Quant;
  const uint32_t rows = p.route.M;
  if (blockIdx.x < rows) {
    __shared__ LargeRouterRadixTrait::Smem smem;
    route_radix_block<kUsePDL, fp32_t>(p.route, smem);
    return;
  }

  const uint32_t row = blockIdx.x - rows;
  const auto tid = threadIdx.x;
  // A layer owns its output slot; the later finalize/AllReduce fences reuse.
  const device::distributed::McBarrier barrier{p.sem_local, p.sem_mc, kWorld, 1};
  device::PDLWaitPrimary<kUsePDL>();
  constexpr uint32_t kQuantThreads = (Layout::kLocalGroups * Quant::kNumLanes + 31) / 32 * 32;
  for (uint32_t i = tid; i < kQuantThreads; i += blockDim.x) {
    // Full subwarps participate in the native quant reduction, including its tail.
    const uint32_t group = min(i / Quant::kNumLanes, Layout::kLocalGroups - 1);
    Quant::run(p.quant, 0, row, group, i % Quant::kNumLanes);
  }
  __syncthreads();
  using Vec = device::AlignedVector<uint32_t, 4>;
  for (uint32_t i = tid; i < Layout::kLocal / sizeof(Vec); i += blockDim.x) {
    Vec value;
    device::ptx::ld_global_16B(value, p.quant.output.get<fp8_e4m3_t>(0, row), i);
    device::ptx::st_multimem_16B(value, p.q_mc + row * kHidden + p.rank * Layout::kLocal, i);
  }
  for (uint32_t i = tid; i < Layout::kScaleStride / sizeof(Vec); i += blockDim.x) {
    Vec value;
    device::ptx::ld_global_16B(
        value, static_cast<const uint8_t*>(p.quant.scale.base) + row * p.quant.scale.token_stride, i);
    device::ptx::st_multimem_16B(value, p.scales_mc + row * Layout::kScaleRowBytes + p.rank * Layout::kScaleStride, i);
  }
  device::PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  barrier.arrive_rel_acq(0);
  // Acquire polls in thread 0; publish it to all scale readers.
  __syncthreads();
  for (uint32_t i = tid; i < Layout::kPackedGroups; i += blockDim.x) {
    uint32_t packed = 0;
#pragma unroll
    for (uint32_t byte = 0; byte < 4; ++byte) {
      const uint32_t group = i * 4 + byte;
      if (group < Layout::kGroups) {
        const uint32_t offset = row * Layout::kScaleRowBytes + (group / Layout::kLocalGroups) * Layout::kScaleStride +
                                group % Layout::kLocalGroups;
        packed |= static_cast<uint32_t>(p.scales_scratch[offset]) << (8 * byte);
      }
    }
    p.scales_out[row * Layout::kPackedGroups + i] = packed;
  }
}

template <uint32_t kWorld, uint32_t kHidden, bool kUsePDL>
struct RouteQuantGatherKernel {
  static void
  run(host::distributed::CommunicatorRef comm,
      tvm::ffi::TensorView scores,
      tvm::ffi::TensorView bias,
      tvm::ffi::TensorView out_w,
      tvm::ffi::TensorView out_i,
      tvm::ffi::TensorView packed,
      tvm::ffi::TensorView local_x,
      tvm::ffi::TensorView local_q,
      tvm::ffi::TensorView local_s,
      tvm::ffi::TensorView global_q,
      tvm::ffi::TensorView scale_scratch,
      tvm::ffi::TensorView global_s,
      int64_t q_mc,
      int64_t scales_mc,
      double scale,
      bool renormalize,
      bool apply_scale) {
    using namespace host;
    using Layout = ShardedFrontLayout<kWorld, kHidden>;
    using Router = LargeRouterRadixTrait;
    auto rows = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({rows, Router::kNumExperts})
        .with_dtype<fp32_t>()
        .with_device(device)
        .with_strides({-1, 1})
        .verify(scores);
    TensorMatcher({Router::kNumExperts}).with_dtype<fp32_t>().with_device(device).verify(bias);
    TensorMatcher({rows, Router::kTopK}).with_dtype<fp32_t>().with_device(device).verify(out_w);
    TensorMatcher({rows, Router::kTopK}).with_dtype<int32_t>().with_device(device).verify(out_i);
    TensorMatcher({rows, Router::kTopK}).with_dtype<int32_t>().with_device(device).verify(packed);
    TensorMatcher({rows, Layout::kLocal})
        .with_dtype<fp32_t>()
        .with_device(device)
        .with_strides({-1, 1})
        .verify(local_x);
    TensorMatcher({rows, kHidden}).with_dtype<fp8_e4m3_t>().with_device(device).verify(global_q);
    TensorMatcher({rows, kWorld, Layout::kScaleStride / 4})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(scale_scratch);
    TensorMatcher({rows, Layout::kPackedGroups}).with_dtype<int32_t>().with_device(device).verify(global_s);
    const auto quant = build_quant_context<typename Layout::Quant, false>(local_x, local_q, local_s).params;
    CHECK_HOST(quant.scale.token_stride >= Layout::kScaleStride);
    const auto& pull = comm->get_pull_obj();
    const auto num_rows = static_cast<uint32_t>(rows.unwrap());
    CHECK_HOST(pull.world_size == kWorld && pull.num_blocks >= 2 * num_rows && pull.mc_semaphore != nullptr);
    if (num_rows == 0) return;
    const RouteQuantGatherParams params{
        .route =
            {scores.data_ptr(),
             static_cast<const fp32_t*>(bias.data_ptr()),
             static_cast<fp32_t*>(out_w.data_ptr()),
             static_cast<int32_t*>(out_i.data_ptr()),
             static_cast<int32_t*>(packed.data_ptr()),
             static_cast<int>(num_rows),
             scores.stride(0),
             out_w.stride(0),
             out_i.stride(0),
             packed.stride(0),
             static_cast<float>(scale),
             renormalize ? 1 : 0,
             apply_scale ? 1 : 0,
             0},
        .quant = quant,
        .q_mc = reinterpret_cast<uint8_t*>(q_mc),
        .scales_mc = reinterpret_cast<uint8_t*>(scales_mc),
        .scales_scratch = static_cast<const uint8_t*>(scale_scratch.data_ptr()),
        .scales_out = static_cast<uint32_t*>(global_s.data_ptr()),
        .sem_local = pull.semaphores[pull.rank],
        .sem_mc = pull.mc_semaphore,
        .rank = pull.rank};
    LaunchKernel(2 * num_rows, Router::kBlockSize, device.unwrap())
        .enable_pdl(kUsePDL)(route_quant_gather_kernel<kWorld, kHidden, kUsePDL>, params);
  }
};
}  // namespace sglang
