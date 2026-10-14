# Copyright (c) 2026, SGLang Team.
"""Lightweight dispatch bridge for SGLang-owned SM120 FA4 kernels.

The vendored FA4 interface dispatches through this module so the SM120
implementation and its launch state remain outside ``flash_attn/cute``.
"""

from functools import lru_cache


@lru_cache(maxsize=None)
def get_forward_host(arch: int):
    """Return the optional architecture-owned forward host."""
    if arch // 10 == 12:
        from sglang.kernels.ops.attention.fa4_sm120.runtime import (
            sm120_forward_host,
        )

        return sm120_forward_host
    return None


def resolve_runtime_policy(
    *,
    device_capability: tuple[int, int],
    deterministic: bool,
) -> tuple[int, int, bool]:
    """Resolve generic and architecture-owned SplitKV launch policy."""
    arch = device_capability[0] * 10 + device_capability[1]
    uses_arch_decode_policy = get_forward_host(arch) is not None
    no_splitkv = device_capability < (9, 0) or uses_arch_decode_policy
    num_splits = 1 if deterministic or no_splitkv else 0
    decode_num_splits = (
        0 if uses_arch_decode_policy and not deterministic else num_splits
    )
    return num_splits, decode_num_splits, uses_arch_decode_policy


@lru_cache(maxsize=None)
def get_forward_arch(device) -> int | None:
    """Return the device arch when it has an architecture-owned forward host."""
    import torch

    major, minor = torch.cuda.get_device_capability(device)
    arch = major * 10 + minor
    return arch if get_forward_host(arch) is not None else None


def try_cached_paged_decode(*, arch: int, **kwargs):
    """Try an architecture-owned paged-decode launch plan."""
    host = get_forward_host(arch)
    return None if host is None else host.try_paged_decode(arch=arch, **kwargs)


def try_cached_varlen(*, arch: int, **kwargs):
    """Try an architecture-owned varlen launch plan."""
    host = get_forward_host(arch)
    return None if host is None else host.try_varlen(arch=arch, **kwargs)
