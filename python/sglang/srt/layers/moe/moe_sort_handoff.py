"""Hand a precomputed aiter moe_sorting result to the next aiter fused_moe call.

A model whose router fuses the top-k with the route sort (Kimi-K3 on gfx950)
stages the sorted routes here; the aiter runner's moe_sorting hook returns them
instead of sorting again when the call is the one they were staged for.
"""

import contextlib
from typing import Iterator, Optional

import msgspec
import torch


class StagedSort(msgspec.Struct, frozen=True):
    topk_ids: torch.Tensor
    block_m: int
    sorted_ids: torch.Tensor
    sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor
    # zeroed stage-2 accumulation buffer, the runner's zero-copy output
    moe_buf: torch.Tensor
    # Optional stage-1 MXFP8 activations for quant_input (the routed rows),
    # standing in for aiter's fused_dynamic_mxfp8_quant_moe_sort.
    quant_input: Optional[torch.Tensor] = None
    a1: Optional[torch.Tensor] = None
    a1_scale: Optional[torch.Tensor] = None


class StagedQuant(msgspec.Struct, frozen=True):
    """Stage-1 MXFP8 rows an earlier kernel quantized: aiter then only sorts
    their scales (mxfp4_moe_sort_fwd) instead of fused_dynamic_mxfp8_quant_moe_sort."""

    quant_input: torch.Tensor  # the bf16 rows aiter would quantize
    a1: torch.Tensor  # [M, D] fp8 e4m3
    scales: torch.Tensor  # [M, D / 32] e8m0, token-major


class _Handoff(msgspec.Struct):
    staged: Optional[StagedSort] = None
    quant: Optional[StagedQuant] = None


_handoff = _Handoff()


@contextlib.contextmanager
def stage(sort: StagedSort) -> Iterator[None]:
    """Publish `sort` for the aiter fused_moe call(s) inside the block."""
    old, _handoff.staged = _handoff.staged, sort
    try:
        yield
    finally:
        _handoff.staged = old


@contextlib.contextmanager
def stage_quant(quant: StagedQuant) -> Iterator[None]:
    """Publish `quant` for the aiter fused_moe call(s) inside the block."""
    old, _handoff.quant = _handoff.quant, quant
    try:
        yield
    finally:
        _handoff.quant = old


def sort_staged() -> bool:
    return _handoff.staged is not None


def active() -> bool:
    return _handoff.staged is not None or _handoff.quant is not None


def take(
    topk_ids: torch.Tensor,
    block_m: int,
    output: Optional[torch.Tensor],
    moebuf_dtype: torch.dtype,
) -> Optional[tuple[torch.Tensor, ...]]:
    """moe_sorting's return tuple when this call matches the staged sort."""
    s = _handoff.staged
    if (
        s is None
        or topk_ids.data_ptr() != s.topk_ids.data_ptr()
        or topk_ids.shape != s.topk_ids.shape
        or block_m != s.block_m
        or output is None
        or output.data_ptr() != s.moe_buf.data_ptr()
        or output.shape != s.moe_buf.shape
        or moebuf_dtype != s.moe_buf.dtype
    ):
        return None
    return (
        s.sorted_ids,
        s.sorted_weights,
        s.sorted_expert_ids,
        s.num_valid_ids,
        s.moe_buf,
    )


def take_quant(
    input: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """The stage-1 (a1, a1_scale) when this quant call is the staged one: as
    staged with its sort, or a staged quant whose scales get sorted here."""
    s = _handoff.staged
    if (
        s is not None
        and s.a1 is not None
        and s.quant_input is not None
        and input.data_ptr() == s.quant_input.data_ptr()
        and input.shape == s.quant_input.shape
        and input.stride() == s.quant_input.stride()
        and sorted_ids.data_ptr() == s.sorted_ids.data_ptr()
        and token_num == s.a1.shape[0]
    ):
        return s.a1, s.a1_scale
    q = _handoff.quant
    if (
        q is None
        or input.data_ptr() != q.quant_input.data_ptr()
        or input.shape != q.quant_input.shape
        or input.stride() != q.quant_input.stride()
        or token_num != q.a1.shape[0]
    ):
        return None
    from aiter.ops.quant import mxfp4_moe_sort_fwd

    a1_scale = mxfp4_moe_sort_fwd(
        q.scales,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token_num,
        cols=q.a1.shape[1],
    )
    return q.a1, a1_scale
