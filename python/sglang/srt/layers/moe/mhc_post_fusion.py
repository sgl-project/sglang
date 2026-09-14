"""Scoped handoff of a decoder's HC post operands to deferred MoE finalize."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MhcPostFusion:
    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    stats_stream: Optional[torch.cuda.Stream]
    output: Optional[torch.Tensor] = None
    pre: Optional[torch.Tensor] = None
    norm_weight: Optional[torch.Tensor] = None
    norm_eps: float = 0.0
    normalized: Optional[torch.Tensor] = None
    quantized: Optional[tuple[torch.Tensor, torch.Tensor]] = None


_current: ContextVar[Optional[MhcPostFusion]] = ContextVar("moe_mhc_post", default=None)


def current_mhc_post_fusion():
    return _current.get()


@contextmanager
def use_mhc_post_fusion(state):
    token = _current.set(state)
    try:
        yield
    finally:
        _current.reset(token)
