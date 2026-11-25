"""Kimi-K3 tensor-parallel KDA under attention DP (``--enable-kda-tp``).

MLA stays attention-DP; each KDA layer all-gathers every DP rank's rows, runs on this
rank's heads/W with a head-sharded state pool (global slot ids), and reduce-scatters o_proj.
"""

from __future__ import annotations

from typing import Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_flags, get_parallel


def enabled() -> bool:
    return get_flags().dp.kda_tp


def width() -> int:
    """KDA heads and state are sharded this many ways (1 without --enable-kda-tp)."""
    return get_parallel().tp_size if enabled() else 1


def group():
    return get_parallel().tp_group


def slot_partition(pool_slots: int) -> Tuple[int, int]:
    """(first slot id, slot count) this DP rank allocates from a pool of ``pool_slots``."""
    if not enabled():
        return 1, pool_slots
    W = width()
    per_rank = pool_slots // W
    if per_rank < 1:
        raise ValueError(
            f"KDA TP: {pool_slots} state slots cannot be split over {W} DP ranks"
        )
    return get_parallel().attn_dp_rank * per_rank + 1, per_rank


def all_gather_padded(t: torch.Tensor, n: int, pad_value: int) -> torch.Tensor:
    g = group()
    if t.shape[0] < n:
        t = torch.cat([t, t.new_full((n - t.shape[0],), pad_value)])
    out = t.new_empty((g.world_size * n,))
    g.all_gather_into_tensor(out, t.contiguous())
    return out


def pd_staging(req_to_token_pool):
    """The PD decode worker's KDA TP state staging, or None without --enable-kda-tp."""
    return req_to_token_pool.mamba_pool.pd_staging if enabled() else None


def lazy_mamba_slot(enable_mamba_extra_buffer: bool) -> bool:
    # Not with ping-pong tracking slots, which are allocated with the main slot.
    return (
        enabled()
        and not enable_mamba_extra_buffer
        and envs.SGLANG_K3_KDA_TP_LAZY_MAMBA_SLOT.get()
    )


def lazy_stage_key(req_pool_idx: int) -> int:
    # Negated, so it never collides with a state slot id (>= 1).
    return -int(req_pool_idx)
