"""Fail-stop: a mono step that did not finish never returns tokens, and the engine stops instead of hanging.

Each kernel's ``health()`` error words are read after the step (copied next to the sampled ids by the caller); a set
word raises ``MonoKernelError`` before that step's tokens are returned. Error words are never cleared before the
decision. ``SGLANG_MONO_UNSAFE_WARN=1`` (tests / benchmarks only, no CLI flag) logs incidents and keeps serving.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class MonoKernelError(RuntimeError):
    pass


def unsafe_warn() -> bool:
    return envs.SGLANG_MONO_UNSAFE_WARN.get()


def gather_health(words: Iterable[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    """One int32 vector of every kernel's error words (device), or None when no kernel reports any."""
    ws = [w.reshape(-1).to(torch.int32) for w in words if w is not None]
    return torch.cat(ws) if ws else None


def check(health_host: Optional[torch.Tensor], *, rank: int, step: int) -> None:
    """``health_host``: the error words copied to the host with the step's outputs."""
    if health_host is None or not bool(health_host.any()):
        return
    bad = torch.nonzero(health_host).flatten().tolist()
    msg = (
        f"mono kernel error words set at step {step} on rank {rank}: indices {bad[:16]}"
    )
    if unsafe_warn():
        logger.error("%s (SGLANG_MONO_UNSAFE_WARN=1: serving on)", msg)
        return
    raise MonoKernelError(msg)
