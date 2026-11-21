#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/fusion.cuh>
#include <sgl_kernel/distributed/pull/kernel.cuh>

#include <tvm/ffi/optional.h>

#include <algorithm>
#include <cstdint>
#include <utility>

namespace sglang::host::distributed::pull {

using namespace sglang::device::distributed::pull;

// The shard `[row0, row0 + rows)` of `num_tokens` rows that `rank` owns when the rows are split evenly, the first
// `num_tokens % world_size` ranks taking one more.
inline std::pair<int64_t, int64_t> even_shard(int64_t num_tokens, uint32_t rank, uint32_t world_size) {
  const int64_t base = num_tokens / world_size, extra = num_tokens % world_size;
  return {base * rank + std::min<int64_t>(rank, extra), base + (rank < extra ? 1 : 0)};
}

// Launch of the pull kernel over the communicator's pull plane: `num_blocks` CTAs (the same on every rank, at most the
// plane's pull width), `kRowsPerGroupChoices` the row-group sizes compiled into the module, picked by `rows_per_group`.
template <bool kUsePDL, uint32_t... kRowsPerGroupChoices>
struct PullLauncher {
  template <Epilogue Epi>
  static void launch(
      CommunicatorRef comm,
      CallShape& shape,
      const typename Epi::Params& epilogue,
      int64_t partial_mc,
      int64_t row0,
      int64_t rows,
      int64_t num_blocks,
      int64_t rows_per_group) {
    const auto& plane = comm.get()->get_pull_obj();
    CHECK_HOST(plane.mc_semaphore != nullptr) << "pull needs a multicast-capable pull plane";
    CHECK_HOST(partial_mc != 0) << "pull needs the partials' multicast address";
    CHECK_HOST(num_blocks >= 1) << "invalid num_blocks: " << num_blocks;
    CHECK_HOST(0 <= row0 && 0 <= rows && row0 + rows <= shape.get_num_tokens()) << "the shard escapes the rows";
    num_blocks = std::min<int64_t>(num_blocks, comm.get()->get_pull_blocks());
    const auto params = PullParams<Epi>{
        .epilogue = epilogue,
        .partial_mc = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(partial_mc)),
        .sem_local = plane.semaphores[plane.rank],
        .sem_mc = plane.mc_semaphore,
        .world_size = plane.world_size,
        .row0 = static_cast<uint32_t>(row0),
        .rows = static_cast<uint32_t>(rows),
    };
    const auto device = shape.get_device();
    const auto run = [&](auto kernel) {
      LaunchKernel(static_cast<uint32_t>(num_blocks), Epi::Tile::kNumThreads, device)
          .enable_pdl(kUsePDL)(kernel, params);
      return true;
    };
    const bool launched =
        ((rows_per_group == kRowsPerGroupChoices && run(fused_all_reduce_pull<kRowsPerGroupChoices, kUsePDL, Epi>)) ||
         ...);
    CHECK_HOST(launched) << "rows_per_group " << rows_per_group << " is not compiled";
  }
};

}  // namespace sglang::host::distributed::pull
