#pragma once
#include "gemm_ag.cuh"

namespace sglang::front_gather {

__global__ void publish(gemm_ag::ProducerParams p, const float* x, uint32_t m) {
  const uint32_t phase = p.counter[0].get() & 1;
  for (uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 4; i < m * 448; i += gridDim.x * blockDim.x * 4) {
    uint4 bits = *reinterpret_cast<const uint4*>(x + (i / 448) * 2880 + 2432 + i % 448);
    // Encode positive zero as negative zero to keep Lamport's empty marker free.
    if (!bits.x) bits.x = 0x80000000u;
    if (!bits.y) bits.y = 0x80000000u;
    if (!bits.z) bits.z = 0x80000000u;
    if (!bits.w) bits.w = 0x80000000u;
    auto* dst = reinterpret_cast<uint32_t*>(p.ws_mc + phase * p.half_bytes) + p.rank * m * 448 + i;
    device::ptx::multimem_store_relaxed(dst, bits.x);
    device::ptx::multimem_store_relaxed(dst + 1, bits.y);
    device::ptx::multimem_store_relaxed(dst + 2, bits.z);
    device::ptx::multimem_store_relaxed(dst + 3, bits.w);
  }
}

struct Gather {
  static void run(host::distributed::CommunicatorRef ref, tvm::ffi::TensorView x, tvm::ffi::TensorView out) {
    using namespace host;
    auto M = SymbolicSize{"M"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({M, 2880}).with_dtype<float>().with_device(dev).verify(x);
    TensorMatcher({M, 3584}).with_dtype<float>().with_device(dev).verify(out);
    const auto& p = ref.get()->get_push_obj();
    const uint32_t m = M.unwrap();
    CHECK_HOST(m == 8 || m == 16);
    CHECK_HOST(p.world_size == 8 && p.mc_workspace != nullptr);
    CHECK_HOST(m * 448 * 4 <= p.slot_bytes);
    const uint32_t blocks = host::div_ceil(m * 3584 / 4, 128u) + 1;
    CHECK_HOST(blocks <= p.num_blocks);
    gemm_ag::ProducerParams pp{p.mc_workspace, p.counter, static_cast<uint32_t>(p.slot_bytes * p.world_size), p.rank};
    LaunchKernel(host::div_ceil(m * 448 / 4, 128u), 128, dev.unwrap())(
        publish, pp, static_cast<const float*>(x.data_ptr()), m);
    gemm_ag::ConsumerParams cp{
        p.workspaces[p.rank],
        p.counter,
        p.num_blocks,
        static_cast<uint32_t>(p.slot_bytes * p.world_size),
        nullptr,
        nullptr,
        out.data_ptr(),
        m};
    LaunchKernel(blocks, 128, dev.unwrap())(gemm_ag::spin_add3_kernel<3584, false, false, true>, cp);
  }
};
}  // namespace sglang::front_gather
