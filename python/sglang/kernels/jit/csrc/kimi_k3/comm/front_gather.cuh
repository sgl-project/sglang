#pragma once
#include "gemm_ag.cuh"

namespace sglang::front_gather {

constexpr uint32_t kLatentWidth = 3584;
constexpr uint32_t kLocalLatentWidth = kLatentWidth / gemm_ag::kWorld;
constexpr uint32_t kLatentOffset = 1536 + 896;
constexpr uint32_t kFrontWidth = kLatentOffset + kLocalLatentWidth;
constexpr uint32_t kVecSize = sizeof(uint4) / sizeof(float);
static_assert(kLatentOffset % kVecSize == 0 && kLocalLatentWidth % kVecSize == 0);

__global__ void publish(gemm_ag::ProducerParams params, const float* __restrict__ front, uint32_t num_tokens) {
  const uint32_t phase = params.counter[0].get() & 1;
  for (uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) * kVecSize; i < num_tokens * kLocalLatentWidth;
       i += gridDim.x * blockDim.x * kVecSize) {
    uint4 bits =
        device::load_as<uint4>(front + (i / kLocalLatentWidth) * kFrontWidth + kLatentOffset + i % kLocalLatentWidth);
    // Encode positive zero as negative zero to keep Lamport's empty marker free.
    if (!bits.x) bits.x = 0x80000000u;
    if (!bits.y) bits.y = 0x80000000u;
    if (!bits.z) bits.z = 0x80000000u;
    if (!bits.w) bits.w = 0x80000000u;
    auto* dst = reinterpret_cast<uint32_t*>(params.ws_mc + phase * params.half_bytes) +
                params.rank * num_tokens * kLocalLatentWidth + i;
    device::ptx::multimem_store_relaxed(dst, bits.x);
    device::ptx::multimem_store_relaxed(dst + 1, bits.y);
    device::ptx::multimem_store_relaxed(dst + 2, bits.z);
    device::ptx::multimem_store_relaxed(dst + 3, bits.w);
  }
}

struct Gather {
  /// \brief Assemble FP32 latent columns from a contiguous [M, 2880] TP8 front into [M, 3584].
  static void run(host::distributed::CommunicatorRef ref, tvm::ffi::TensorView front, tvm::ffi::TensorView out) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({M, kFrontWidth}).with_dtype<float>().with_device(dev).ensure_alignment(sizeof(uint4)).verify(front);
    TensorMatcher({M, kLatentWidth}).with_dtype<float>().with_device(dev).ensure_alignment(sizeof(uint4)).verify(out);
    const auto& push = ref.get()->get_push_obj();
    CHECK_HOST(M.unwrap() == 8 || M.unwrap() == 16);
    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    CHECK_HOST(push.world_size == gemm_ag::kWorld && push.mc_workspace != nullptr);
    CHECK_HOST(num_tokens * kLocalLatentWidth * sizeof(float) <= push.slot_bytes);
    const uint32_t blocks = div_ceil(num_tokens * kLatentWidth / kVecSize, gemm_ag::kSpinBlock) + 1;
    CHECK_HOST(blocks <= push.num_blocks);
    const auto half_bytes = static_cast<uint32_t>(push.slot_bytes * push.world_size);
    const gemm_ag::ProducerParams producer{
        .ws_mc = push.mc_workspace, .counter = push.counter, .half_bytes = half_bytes, .rank = push.rank};
    LaunchKernel(
        div_ceil(num_tokens * kLocalLatentWidth / kVecSize, gemm_ag::kSpinBlock), gemm_ag::kSpinBlock, dev.unwrap())(
        publish, producer, static_cast<const float*>(front.data_ptr()), num_tokens);
    const gemm_ag::ConsumerParams consumer{
        .ws_local = push.workspaces[push.rank],
        .counter = push.counter,
        .num_counters = push.num_blocks,
        .half_bytes = half_bytes,
        .b = nullptr,
        .c = nullptr,
        .out = out.data_ptr(),
        .num_rows = num_tokens};
    LaunchKernel(blocks, gemm_ag::kSpinBlock, dev.unwrap())(
        gemm_ag::spin_add3_kernel<kLatentWidth, false, false, true>, consumer);
  }
};
}  // namespace sglang::front_gather
