// Fused NVLS pull all-reduce + RMSNorm on a row prefix (Kimi K3), in place on a multicast-bound symmetric buffer:
// `x = rmsnorm(allreduce(x))` on the first `num_norm_rows` rows, the rest reduced only. The rows are split evenly
// over the ranks; each rank reduces, norms and multicasts its own.
#include <sgl_kernel/distributed/pull/epilogue/rms_norm.cuh>

#include "launch.cuh"
#include <cstdint>

namespace sglang::host::distributed::pull {

// `x [num_tokens, kHiddenSize]` is this rank's view of the buffer, `x_mc` its multicast VA.
template <uint32_t kHiddenSize, bool kUsePDL>
void all_reduce_norm_prefix(
    CommunicatorRef comm,
    TensorView x,
    int64_t x_mc,
    TensorView norm_weight,
    float eps,
    int64_t num_norm_rows,
    int64_t num_blocks,
    int64_t rows_per_group) {
  // 16 rows per group spill (ptxas 13.4, sm_90a / sm_100a)
  using Launcher = PullLauncher<kUsePDL, 1, 2, 4, 8>;
  using Epi = RMSNorm<kHiddenSize, /*kNormPrefix=*/true>;
  CallShape shape;
  shape.verify_tensor(x, kHiddenSize);
  const auto epi = Epi::make_params(shape, norm_weight, eps, num_norm_rows, x_mc);
  const auto& plane = comm.get()->get_pull_obj();
  const auto [row0, rows] = even_shard(shape.get_num_tokens(), plane.rank, plane.world_size);
  Launcher::template launch<Epi>(comm, shape, epi, x_mc, row0, rows, num_blocks, rows_per_group);
}

}  // namespace sglang::host::distributed::pull
