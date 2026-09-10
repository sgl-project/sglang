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
    val: cute.Numeric, num_warps: cutlass.Constexpr, tidx: cutlass.Int32
) -> cute.Numeric:
    smem = cutlass.utils.SmemAllocator()
    acc = smem.allocate_tensor(cutlass.Float32, num_warps + 1)
    warp_id = tidx >> 5
    lane_id = tidx & 31
    if lane_id == 0:
        acc[warp_id] = val
    cute.arch.sync_threads()
    if warp_id == 0:
        val = acc[lane_id] if lane_id < num_warps else cutlass.Float32(0)
        val = warp_reduce_sum(val)
        if lane_id == 0:
            acc[num_warps] = val
    cute.arch.sync_threads()
    val = acc[num_warps]
    return val
