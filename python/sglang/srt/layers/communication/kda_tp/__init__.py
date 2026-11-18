"""Kimi-K3 tensor-parallel KDA under attention DP (``--enable-kda-tp``).

MLA stays attention-DP; each KDA layer all-gathers every DP rank's rows, runs on this
rank's heads/W with a head-sharded state pool (global slot ids), and reduce-scatters o_proj.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    _note_dp_gather_in_prefill_graph,
    get_dp_global_num_tokens,
)
from sglang.srt.runtime_context import get_flags, get_parallel

logger = logging.getLogger(__name__)


def enabled() -> bool:
    return get_flags().dp.kda_tp


def width() -> int:
    """KDA heads and state are sharded this many ways (1 without --enable-kda-tp)."""
    return get_parallel().tp_size if enabled() else 1


def group():
    return get_parallel().tp_group


def slot_partition(pool_slots: int) -> Tuple[int, int]:
    W = width()
    per_rank = pool_slots // W
    if per_rank < 1:
        raise ValueError(
            f"KDA TP: {pool_slots} state slots cannot be split over {W} DP ranks"
        )
    return get_parallel().attn_dp_rank * per_rank + 1, per_rank


def padded_rows(local_rows: int, counts: Optional[List[int]] = None) -> int:
    if counts is None:
        counts = get_dp_global_num_tokens()
    if not counts:
        raise RuntimeError("KDA TP needs the DP global token counts of this forward")
    rows = max(counts)
    if local_rows > rows:
        raise RuntimeError(
            f"KDA TP: {local_rows} local rows exceed the DP maximum {rows}"
        )
    return rows


def _fast_collectives():
    from sglang.srt.layers.communication import k3_sp_collective

    return k3_sp_collective.tp_group_state()


def _fits_symm_buffer(total_rows: int) -> bool:
    from sglang.srt.layers.communication import k3_ar_fusion

    # The K3 symmetric buffers hold one chunk; a DP-gathered prefill can exceed it.
    return total_rows <= k3_ar_fusion._max_buffer_rows()


def all_gather_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    # May return a persistent buffer that the next KDA TP all-gather overwrites.
    from sglang.srt.layers.communication import k3_sp_collective

    # A prefill graph holding this replays only when every DP rank runs its recorded bucket.
    _note_dp_gather_in_prefill_graph()
    g = group()
    if x.shape[0] != rows:
        x = torch.nn.functional.pad(x, (0, 0, 0, rows - x.shape[0]))
    x = x.contiguous()
    state = _fast_collectives()
    if state is not None and 0 < rows and _fits_symm_buffer(g.world_size * rows):
        out = k3_sp_collective.all_gather(
            x, state=state, buffer_name="kda_tp_all_gather"
        )
        if out is not None:
            return out
    out = x.new_empty((g.world_size * rows, *x.shape[1:]))
    g.all_gather_into_tensor(out, x)
    return out


def o_proj_output(
    total_rows: int, width: int, dtype: torch.dtype
) -> Optional[torch.Tensor]:
    from sglang.srt.layers.communication import k3_sp_collective

    state = _fast_collectives()
    if state is None or total_rows == 0 or not _fits_symm_buffer(total_rows):
        return None
    if not k3_sp_collective.requires_symmetric_rs(
        total_rows,
        torch.device("cuda"),
        torch.empty((), dtype=dtype).element_size(),
        state=state,
        fused=False,
    ):
        return None
    return k3_sp_collective.get_o_proj_output_buffer(
        total_rows, dtype, width, state=state
    )


def reduce_scatter_rows(y: torch.Tensor, local_rows: int) -> torch.Tensor:
    from sglang.srt.layers.communication import k3_sp_collective

    g = group()
    rows = y.shape[0] // g.world_size
    state = _fast_collectives()
    if state is not None and rows > 0 and _fits_symm_buffer(y.shape[0]):
        out = k3_sp_collective.reduce_scatter_res(y, None, state=state)
        if out is not None:
            return out[:local_rows]
    out = y.new_empty((rows, *y.shape[1:]))
    g.reduce_scatter_tensor(out, y.contiguous())
    return out[:local_rows]


def all_gather_padded(
    t: torch.Tensor,
    n: int,
    pad_value: int,
    name: Optional[str] = None,
    capacity: int = 0,
) -> torch.Tensor:
    g = group()
    if name is None:
        if t.shape[0] < n:
            t = torch.cat([t, t.new_full((n - t.shape[0],), pad_value)])
        out = t.new_empty((g.world_size * n,))
        g.all_gather_into_tensor(out, t.contiguous())
        return out
    src = _static(f"{name}.in", capacity, t.dtype)[:n]
    src[t.shape[0] :].fill_(pad_value)
    src[: t.shape[0]].copy_(t)
    out = _static(f"{name}.out", g.world_size * capacity, t.dtype)[: g.world_size * n]
    g.all_gather_into_tensor(out, src)
    return out


# Never freed or rebound: CUDA-graph captures record their addresses.
_STATIC: Dict[Tuple[str, torch.dtype], torch.Tensor] = {}


def _static(name: str, numel: int, dtype: torch.dtype) -> torch.Tensor:
    buf = _STATIC.get((name, dtype))
    if buf is not None and buf.numel() >= numel:
        return buf
    if buf is not None and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"KDA TP: static buffer {name} must grow to {numel} during a CUDA graph capture"
        )
    buf = torch.empty(numel, dtype=dtype, device=torch.cuda.current_device())
    _STATIC[(name, dtype)] = buf
    return buf


def lazy_mamba_slot(enable_mamba_extra_buffer: bool) -> bool:
    # Not with ping-pong tracking slots, which are allocated with the main slot.
    return (
        enabled()
        and not enable_mamba_extra_buffer
        and envs.SGLANG_K3_KDA_TP_LAZY_MAMBA_SLOT.get()
    )


def late_stage() -> bool:
    return enabled() and envs.SGLANG_K3_KDA_TP_LATE_STAGE.get()


# State destination a late-staged request advertises until its stage is bound.
LATE_STAGE_UNBOUND = -1


def lazy_stage_key(req_pool_idx: int) -> int:
    # Negated, so it never collides with a state slot id (>= 1).
    return -int(req_pool_idx)
