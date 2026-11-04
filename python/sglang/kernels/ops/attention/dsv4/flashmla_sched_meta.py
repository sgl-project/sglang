from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

from .utils import make_name

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# sizeof(DecodingSchedMeta) / 4, fixed by FlashMLA's params.h.
META_INTS = 8


@cache_once
def _jit_flashmla_sched_meta_module() -> Module:
    return load_jit(
        make_name("flashmla_sched_meta"),
        cuda_files=["deepseek_v4/flashmla_sched_meta.cuh"],
        cuda_wrappers=[("flashmla_sched_meta", "flashmla_sched_meta")],
    )


def flashmla_sched_meta(
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    seqlens_k: Optional[torch.Tensor] = None,
    block_size_n: int,
    fixed_overhead_num_blocks: int,
    topk: int,
    extra_topk: int = 0,
    parallel: bool = False,
    windowed: bool = False,
) -> None:
    """Fill FlashMLA's split-KV tile-scheduler metadata in place.

    Produces the schedule FlashMLA computes for itself when handed no metadata,
    so passing the filled tensors as the cached ``tile_scheduler_metadata`` /
    ``num_splits`` lets it skip its own kernel. The schedule has to fit in 48 KB
    of shared memory, ``4 * (5 * batch_size + 1 + 8 * num_sm_parts)`` bytes, so
    a few thousand requests at most.

    Args:
        tile_scheduler_metadata: ``[num_sm_parts, 8]`` int32, written.
        num_splits: ``[batch_size + 1]`` int32, written.
        topk_length: ``[batch_size]`` int32 per-request candidate count, or None
            to use ``topk`` for every request.
        extra_topk_length: the same for the extra cache, when ``extra_topk``.
        seqlens_k: ``[batch_size]`` int32, required only for a dense schedule.
        block_size_n: the kernel's KV block size.
        fixed_overhead_num_blocks: the implementation's per-request overhead.
        topk: the sparse top-k, or -1 for a dense model.
        extra_topk: the extra cache's top-k, 0 when there is none.
        parallel: compute the partition walk in parallel when the sparse
            schedule's tables fit in shared memory (otherwise, and for a dense
            schedule, the serial walk runs). Both give the same schedule.
        windowed: for a sparse schedule, cut the requests' costs into equal
            windows instead of walking the partitions: not FlashMLA's schedule,
            but a valid one with the same bound on the costliest partition,
            computed in a few microseconds instead of tens.
    """
    _jit_flashmla_sched_meta_module().flashmla_sched_meta(
        tile_scheduler_metadata,
        num_splits,
        topk_length,
        extra_topk_length,
        seqlens_k,
        block_size_n,
        fixed_overhead_num_blocks,
        topk,
        extra_topk,
        parallel,
        windowed,
    )
