import math

import cutlass
import cutlass.cute as cute


@cute.jit
def warp_reduce_sum(val: cute.Numeric, reduce_size: int = 32) -> cute.Numeric:
    iters = int(math.log2(reduce_size))
    for i in range(iters):
        val = val + cute.arch.shuffle_sync_down(val, offset=1 << (iters - i - 1))
    return val


@cute.jit
def cta_reduce_sum(
    vals: list,
    num_chunks: cutlass.Constexpr,
    warp_id: cutlass.Int32,
    lane_id: cutlass.Int32,
) -> cute.Numeric:
    """Sum of per-chunk warp sums; warp w holds chunks w * len(vals) + j in vals[j]

    Lane 0 of each chunk's warp sum lands in acc[chunk], then warp 0 shuffle-reduces acc, so
    the order only depends on num_chunks, not on how many chunks a warp walks.
    """
    smem = cutlass.utils.SmemAllocator()
    acc = smem.allocate_tensor(cutlass.Float32, num_chunks + 1)
    if lane_id == 0:
        for j in cutlass.range_constexpr(len(vals)):
            acc[warp_id * len(vals) + j] = vals[j]
    cute.arch.sync_threads()
    if warp_id == 0:
        val = acc[lane_id] if lane_id < num_chunks else cutlass.Float32(0)
        val = warp_reduce_sum(val)
        if lane_id == 0:
            acc[num_chunks] = val
    cute.arch.sync_threads()
    val = acc[num_chunks]
    return val
