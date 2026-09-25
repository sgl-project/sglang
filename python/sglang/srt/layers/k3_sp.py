"""Row-parallel LatentMoE tail for Kimi-K3 (SGLANG_K3_TAIL_SHARD).

After the MoE all-reduce, each rank runs RMSNorm, up_proj, and add3 on T/w
rows and all-gathers the result. Off by default; skipped below
SGLANG_K3_TAIL_SHARD_MIN_TOKENS, where the all-gather costs more than it saves.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def tail_shard_enabled() -> bool:
    return envs.SGLANG_K3_TAIL_SHARD.get()


_tail_shard_armed: Optional[bool] = None


def tail_shard_eligible(num_tokens: int) -> bool:
    global _tail_shard_armed
    if not tail_shard_enabled():
        return False
    if num_tokens < envs.SGLANG_K3_TAIL_SHARD_MIN_TOKENS.get():
        return False
    from sglang.srt.distributed import get_tp_group

    ok = num_tokens % get_tp_group().world_size == 0
    if ok and _tail_shard_armed is not True:
        _tail_shard_armed = True
        logger.warning(
            "K3 tail shard armed: tokens=%d world=%d",
            num_tokens,
            get_tp_group().world_size,
        )
    return ok


def tail_shard_rows(num_tokens: int) -> slice:
    """This rank's contiguous token slice of a [T, H] activation."""
    from sglang.srt.distributed import get_tp_group

    g = get_tp_group()
    rows = num_tokens // g.world_size
    return slice(g.rank_in_group * rows, (g.rank_in_group + 1) * rows)


def tail_shard_all_gather(shard: torch.Tensor) -> torch.Tensor:
    """[T/w, H] -> [T, H] over the TP group (dim-0 concat in rank order)."""
    import torch.distributed as dist

    from sglang.srt.distributed import get_tp_group

    g = get_tp_group()
    out = torch.empty(
        shard.shape[0] * g.world_size,
        shard.shape[1],
        dtype=shard.dtype,
        device=shard.device,
    )
    dist.all_gather_into_tensor(out, shard.contiguous(), group=g.device_group)
    return out
