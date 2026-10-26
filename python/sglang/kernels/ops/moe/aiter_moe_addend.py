# SPDX-License-Identifier: Apache-2.0
"""Fold a [tokens, hidden] addend (a shared-expert output) into aiter's routed-MoE
output instead of adding it afterwards, as runtime patches on aiter.

An accumulating stage2 gets the addend as its output buffer in place of the zeroed
one the sort would hand it (moe_sorting_small's sort wrapper); a reduce stage2 adds
it while summing the top-k slots (the ``_run_moe_reduction`` patch below). Any other
path leaves the addend unconsumed and the caller adds it as before.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import logging
from contextvars import ContextVar
from typing import Callable, Iterator, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_gfx95_supported

logger = logging.getLogger(__name__)


class MoeOutputAddend:
    """`consumed` turns True once an aiter stage folded `tensor` into its output."""

    __slots__ = ("tensor", "consumed")

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.consumed = False


_pending_addend: ContextVar[Optional[MoeOutputAddend]] = ContextVar(
    "aiter_moe_output_addend", default=None
)


@contextlib.contextmanager
def moe_output_addend(tensor: torch.Tensor) -> Iterator[MoeOutputAddend]:
    """Offer `tensor` to the next aiter routed-MoE call inside the block.

    The routed output of a call that consumed it already includes `tensor`, which
    may also have been accumulated into in place.
    """
    slot = MoeOutputAddend(tensor)
    token = _pending_addend.set(slot)
    try:
        yield slot
    finally:
        _pending_addend.reset(token)


def take_moe_output_addend(
    shape: torch.Size, dtype: torch.dtype, device: torch.device
) -> Optional[torch.Tensor]:
    """The pending addend if it can stand in for a [tokens, hidden] output, else None."""
    slot = _pending_addend.get()
    if slot is None or slot.consumed:
        return None
    tensor = slot.tensor
    if (
        tensor.shape != shape
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        return None
    slot.consumed = True
    return tensor


@triton.jit
def _topk_sum_add_kernel(
    x_ptr,  # [T, TOPK, D] per-slot routed outputs
    addend_ptr,  # [T, D]
    out_ptr,  # [T, D]; may alias addend_ptr
    D,
    TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D
    acc = tl.load(addend_ptr + t * D + offs, mask=mask, other=0.0).to(tl.float32)
    for k in tl.static_range(TOPK):
        acc += tl.load(
            x_ptr + (t * TOPK + k) * D + offs,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
    tl.store(out_ptr + t * D + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def topk_sum_add(x: torch.Tensor, addend: torch.Tensor, out: torch.Tensor) -> None:
    """out[t] = addend[t] + sum_k x[t, k], accumulated in fp32."""
    tokens, topk, dim = x.shape
    block_d = min(2048, triton.next_power_of_2(dim))
    _topk_sum_add_kernel[(tokens, triton.cdiv(dim, block_d))](
        x, addend, out, dim, TOPK=topk, BLOCK_D=block_d, num_warps=4
    )


# _run_moe_reduction's leading parameters, as the wrapper names what a call passed by
# position; an aiter that reorders them leaves the reduction unpatched
_REDUCTION_PARAMS = (
    "target",
    "out",
    "token_num",
    "topk",
    "model_dim",
    "expert_mask",
    "topk_ids",
    "stream",
    "is_fp8",
    "topk_weights",
)


def _with_out(args: tuple, kwargs: dict, out: torch.Tensor) -> tuple[tuple, dict]:
    if len(args) > 1:
        return (args[0], out, *args[2:]), kwargs
    return args, {**kwargs, "out": out}


def wrap_moe_reduction(orig_reduction) -> Optional[Callable]:
    """aiter's stage2 top-k reduction, consuming a pending addend; None when its
    parameters are not the ones the wrapper names."""
    params = tuple(inspect.signature(orig_reduction).parameters)
    if params[: len(_REDUCTION_PARAMS)] != _REDUCTION_PARAMS:
        logger.info("aiter MoE addend patch not applied: _run_moe_reduction%s", params)
        return None

    @functools.wraps(orig_reduction)
    def reduction_wrapper(*args, **kwargs):
        call = {**dict(zip(_REDUCTION_PARAMS, args)), **kwargs}
        target, out = call["target"], call["out"]
        token_num, topk, model_dim = call["token_num"], call["topk"], call["model_dim"]
        # the plain dense reduction only: masked, fp8 and route-weighted ones stay aiter's
        dense = (
            call.get("expert_mask") is None
            and not call.get("is_fp8", False)
            and call.get("topk_weights") is None
            and call.get("stream") is None
            and target.dtype == out.dtype
            and out.shape == (token_num, model_dim)
        )
        slot = _pending_addend.get()
        if (
            slot is not None
            and slot.consumed
            and out.data_ptr() == slot.tensor.data_ptr()
        ):
            # the sort handed the addend out as an accumulate buffer, but stage2 reduces
            if dense:
                topk_sum_add(target.view(token_num, topk, model_dim), out, out)
            else:
                routed = torch.empty_like(out)
                routed_args, routed_kwargs = _with_out(args, kwargs, routed)
                orig_reduction(*routed_args, **routed_kwargs)
                out.add_(routed)
            return
        if dense:
            addend = take_moe_output_addend(out.shape, out.dtype, out.device)
            if addend is not None:
                topk_sum_add(target.view(token_num, topk, model_dim), addend, out)
                return
        orig_reduction(*args, **kwargs)

    return reduction_wrapper


@functools.cache
def apply_aiter_moe_addend_patch() -> None:
    """Let aiter's stage2 top-k reduction consume a pending addend."""
    if not is_gfx95_supported():
        return
    try:
        import aiter.ops.flydsl.moe_kernels as moe_kernels

        orig_reduction = moe_kernels._run_moe_reduction
    except (ImportError, AttributeError) as exc:
        logger.info("aiter MoE addend patch not applied: %s", exc)
        return
    reduction_wrapper = wrap_moe_reduction(orig_reduction)
    if reduction_wrapper is None:
        return
    moe_kernels._run_moe_reduction = reduction_wrapper
    logger.info("aiter MoE addend patch applied")
