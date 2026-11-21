# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 ROCm logits gather, installed by models/kimi_k3.py on its
LogitsProcessor via ``install``."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from sglang.srt.environ import envs


def logits_all_gather_lastdim(logits: torch.Tensor) -> Optional[torch.Tensor]:
    """Gather vocab-parallel logits
    [rows, vocab/tp] -> [rows, vocab] with the AITER custom last-dim all-gather
    (one kernel writing the final layout) instead of a dim-0 gather plus the
    movedim copy. Pure data movement: bit-identical. None = not applicable
    (caller falls back to the default gather)."""
    if not (torch.version.hip and logits.is_cuda and logits.dim() == 2):
        return None
    if logits.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return None
    if not logits.is_contiguous():
        return None
    pack = 16 // logits.element_size()
    if logits.shape[-1] % pack != 0 or logits.shape[0] == 0:
        return None
    from sglang.srt.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    ca = getattr(group, "ca_comm", None)
    if (
        ca is None
        or getattr(ca, "disabled", True)
        or not hasattr(ca, "all_gather_reg")
        or not hasattr(ca, "all_gather_unreg")
        or not ca.should_custom_ag(logits)
    ):
        return None
    world = group.world_size
    out = torch.empty(
        (logits.shape[0], logits.shape[1] * world),
        dtype=logits.dtype,
        device=logits.device,
    )
    if getattr(ca, "_IS_CAPTURING", False):
        if torch.cuda.is_current_stream_capturing():
            if envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get():
                ca.all_gather_unreg(logits, out=out, dim=-1)
            else:
                ca.all_gather_reg(logits, out=out, dim=-1)
        else:
            # Graph warmup: no host collective (mirrors _all_gather_into_tensor).
            out.zero_()
    else:
        ca.all_gather_unreg(logits, out=out, dim=-1)
    return out


class K3LogitsGatherer:
    """Wraps LogitsProcessor._logits_gatherer: try the last-dim custom
    all-gather first, else the original gatherer."""

    def __init__(self, fallback: Callable[[torch.Tensor], torch.Tensor]):
        self.fallback = fallback

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        gathered = logits_all_gather_lastdim(logits)
        return gathered if gathered is not None else self.fallback(logits)

    def __getattr__(self, name):
        return getattr(self.fallback, name)


def install(logits_processor) -> None:
    logits_processor._logits_gatherer = K3LogitsGatherer(
        logits_processor._logits_gatherer
    )
