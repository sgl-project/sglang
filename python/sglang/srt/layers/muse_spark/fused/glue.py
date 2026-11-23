"""Hopper-only Muse Spark paths: the gate shared by every one of them."""

import functools

import torch

from sglang.srt.environ import envs

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
