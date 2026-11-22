"""Hopper-only Muse Spark paths: the gate shared by every one of them, and the
in-place TP all-reduce output for decode-sized producers."""

import functools
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_forward

_HOPPER = (9, 0)


def _on_hopper(device=None) -> bool:
    return (
        torch.version.hip is None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability(device) == _HOPPER
    )


@functools.cache
def hopper_kernels_enabled(device=None) -> bool:
    """Gate of every Muse Spark Hopper path; other GPUs keep the stock code."""
    return envs.SGLANG_MUSE_SPARK_HOPPER_KERNELS.get() and _on_hopper(device)


def ar_out(m: int, n: int) -> Optional[torch.Tensor]:
    """Decode / verify: storage for an [m, n] TP all-reduce input (CustomAllReduceV2.mc_out) or None."""
    from sglang.srt.runtime_context import get_parallel

    mc_out = getattr(get_parallel().tp_group.ca_comm, "mc_out", None)
    return mc_out(m, n) if mc_out and m <= 256 and not get_forward().sp_active else None
