#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/fusion.cuh>
#include <sgl_kernel/distributed/push/kernel.cuh>
#include <sgl_kernel/distributed/push/prologue/finalize.cuh>

#include <tvm/ffi/optional.h>

#include <cstdint>

namespace sglang::host::distributed::push {

using namespace sglang::device::distributed::push;

// How an entry compiles its 1-shot kernel: bit flags, or-ed into the `kFlags` template argument.
enum LaunchFlags : uint32_t {
  EMPTY_FLAG = 0,
  // Release the dependent grid right after the push (Kimi K3); otherwise after the epilogue.
  TRIGGER_AFTER_PUSH = 1,
  // The entry's rows fit one wave: `__launch_bounds__(N, 1)` frees the registers ptxas needs to keep every gather of
  // the prologue in flight; entries that run several waves keep the default, which packs more CTAs per SM.
  ONE_WAVE = 2,
  // Push with one multicast store to the plane's multicast VA instead of one unicast store per rank (the host picks
  // this instantiation when the plane has a multicast VA).
  MULTICAST = 4,
};

// Geometry, workspace and launch of the 1-shot kernel, or with `kTwoShot` the 2-shot kernel (which ignores
// `kFlags`), for one (world size, hidden, cluster size, PDL).
template <uint32_t kWorldSize, uint32_t kHiddenSize, uint32_t kClusterSize, bool kUsePDL, bool kTwoShot>
struct PushLauncher {
  template <uint32_t kFlags, Prologue PrologueT, Epilogue EpilogueT>
  static void launch(
      CommunicatorRef comm,
      CallShape& shape,
      const typename PrologueT::Params& prologue,
      const typename EpilogueT::Params& epilogue) {
    using Tile = RowTile<kHiddenSize, kClusterSize>;
    const auto& plane = comm.get()->get_push_obj();
    const int64_t num_tokens = shape.get_num_tokens();
    const int64_t nbytes = num_tokens * kHiddenSize * static_cast<int64_t>(sizeof(bf16_t));
    CHECK_HOST(num_tokens > 0);
    CHECK_HOST(plane.world_size == kWorldSize) << plane.world_size << " != " << kWorldSize;
    CHECK_HOST(nbytes <= plane.slot_bytes) << nbytes << " > " << plane.slot_bytes;
    const auto params = PushParams<kWorldSize, PrologueT, EpilogueT>{
        .prologue = prologue,
        .epilogue = epilogue,
        .ws = plane.get_workspace<kWorldSize>(nbytes),
        .rank = plane.rank,
    };
    const auto grid = dim3(static_cast<uint32_t>(num_tokens), kClusterSize);
    const auto device = shape.get_device();
    const auto launch = [&](auto kernel) {
      LaunchKernel(grid, Tile::kNumThreads, device).enable_pdl(kUsePDL)(kernel, params);
    };
    if constexpr (kTwoShot) {
      constexpr auto kernel = fused_all_reduce_2shot<
          kWorldSize,
          kHiddenSize,
          kClusterSize,
          kUsePDL,
          /*kTriggerAfterPush=*/false,
          /*kMulticast=*/false,
          /*kOccupancy=*/0,
          PrologueT,
          EpilogueT>;
      launch(kernel);
    } else {
      constexpr bool kTriggerAfterPush = kFlags & TRIGGER_AFTER_PUSH;
      constexpr bool kMulticast = kFlags & MULTICAST;
      constexpr uint32_t kOccupancy = kFlags & ONE_WAVE ? 1 : 0;
      constexpr auto kernel = fused_all_reduce_1shot<
          kWorldSize,
          kHiddenSize,
          kClusterSize,
          kUsePDL,
          kTriggerAfterPush,
          kMulticast,
          kOccupancy,
          PrologueT,
          EpilogueT>;
      launch(kernel);
    }
  }

  // As `launch`, with `MULTICAST` or-ed into `kFlags` when the plane has a multicast VA (1-shot only).
  template <uint32_t kFlags, Prologue PrologueT, Epilogue EpilogueT>
  static void launch_multicast_if_available(
      CommunicatorRef comm,
      CallShape& shape,
      const typename PrologueT::Params& prologue,
      const typename EpilogueT::Params& epilogue) {
    static_assert(!kTwoShot, "the 2-shot kernel pushes unicast only");
    static_assert(!(kFlags & MULTICAST), "do not pass multicast flag to this function");
    if (comm.get()->get_push_obj().mc_workspace != nullptr) {
      launch<kFlags | MULTICAST, PrologueT, EpilogueT>(comm, shape, prologue, epilogue);
    } else {
      launch<kFlags, PrologueT, EpilogueT>(comm, shape, prologue, epilogue);
    }
  }
};

// The finalize prologue with `shared_output` added (ADD, when given) or without (NONE).
template <typename Launcher, uint32_t kFlags, uint32_t kHiddenSize, uint32_t kTopK, typename WeightT, Epilogue Epi>
void launch_finalize(
    CommunicatorRef comm,
    CallShape& shape,
    TensorView gemm2_out,
    TensorView expanded_idx_to_permuted_idx,
    TensorView expert_weights,
    Optional<TensorView> shared_output,
    const typename Epi::Params& epi) {
  using Mode = SharedExpertMode;
  if (shared_output.has_value()) {
    using Pro = FinalizeSum<kHiddenSize, kTopK, WeightT, Mode::ADD>;
    const auto pro =
        Pro::make_params(shape, gemm2_out, expanded_idx_to_permuted_idx, expert_weights, shared_output, {});
    Launcher::template launch<kFlags, Pro, Epi>(comm, shape, pro, epi);
  } else {
    using Pro = FinalizeSum<kHiddenSize, kTopK, WeightT, Mode::NONE>;
    const auto pro = Pro::make_params(shape, gemm2_out, expanded_idx_to_permuted_idx, expert_weights, {}, {});
    Launcher::template launch<kFlags, Pro, Epi>(comm, shape, pro, epi);
  }
}

}  // namespace sglang::host::distributed::push
