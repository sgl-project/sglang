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
def _jit_decoding_sched_meta_module() -> Module:
    return load_jit(
        make_name("decoding_sched_meta"),
        cuda_files=["deepseek_v4/decoding_sched_meta.cuh"],
        cuda_wrappers=[("decoding_sched_meta", "decoding_sched_meta")],
    )


def decoding_sched_meta(
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
) -> None:
    """Fill FlashMLA's split-KV tile-scheduler metadata in place.

    Same schedule FlashMLA computes for itself when handed no metadata, but the
    per-request pass and the write-out run over a whole block instead of one
    warp, and the sequential partition walk writes to shared memory rather than
    storing each 32-byte entry to global. At ``num_sm_parts = 152`` (a BS=1
    draft step on GB300) that is the difference between 25.1 us and ~2 us, all
    of it on the critical path inside the decode graph.

    Pass the filled tensors to FlashMLA as the cached
    ``tile_scheduler_metadata`` / ``num_splits`` and it skips its own kernel.

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
    """
    _jit_decoding_sched_meta_module().decoding_sched_meta(
        tile_scheduler_metadata,
        num_splits,
        topk_length,
        extra_topk_length,
        seqlens_k,
        block_size_n,
        fixed_overhead_num_blocks,
        topk,
        extra_topk,
    )
