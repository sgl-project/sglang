"""gfx950 host-pool budgets: the amdgpu driver pins host memory only up to its TTM
page limit, past which hipHostMalloc fails while RAM is still free.

The shared HiCache sizing code calls the helpers below unconditionally; each one
leaves main's behavior unchanged off gfx950."""

from __future__ import annotations

import contextlib
import logging
import resource
from typing import TYPE_CHECKING, ContextManager, Optional

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host import base
from sglang.srt.utils import is_gfx95_supported

if TYPE_CHECKING:
    from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost

logger = logging.getLogger(__name__)

# Per local rank, pinned host memory left to RCCL, AITER and torch.
ROCM_PINNED_RESERVE_BYTES: int = 4 * (1024**3)


def rocm_pinnable_share_bytes() -> Optional[int]:
    """A local rank's share of the host memory the amdgpu driver can pin: the TTM
    page limit (half of RAM by default), past which hipHostMalloc fails while RAM
    is still free. None off gfx950 or when the limit is not visible."""
    if not is_gfx95_supported():
        return None
    try:
        with open("/sys/module/ttm/parameters/pages_limit") as f:
            limit = int(f.read()) * resource.getpagesize()
    except (OSError, ValueError):
        return None
    ranks = base.ranks_per_host()
    return (limit - ROCM_PINNED_RESERVE_BYTES * ranks) // ranks


def clamp_plain_budget(plain: int) -> int:
    """A rank's plain host budget, bounded by its pinnable share where visible."""
    pinnable = rocm_pinnable_share_bytes()
    if pinnable is not None:
        plain = min(plain, pinnable)
    return plain


def explicit_size_scope(enabled: bool) -> ContextManager:
    """With explicit HiCache sizes on gfx950, book every host pool against one
    pinnable-share snapshot, so the ratio-sized Mamba pool built last clamps to
    what the others left. A nullcontext when HiCache is off, off gfx950, or when
    the TTM limit is not visible."""
    if not enabled or rocm_pinnable_share_bytes() is None:
        return contextlib.nullcontext()
    return base.host_memory_budget_scope(base.host_memory_budget_plain_hugetlb())


def mamba_host_ratio(hicache_ratio: float) -> float:
    """The Mamba host pool's host-to-device ratio: SGLANG_HICACHE_MAMBA_HOST_RATIO
    when set on gfx950, else ``hicache_ratio``."""
    ratio = envs.SGLANG_HICACHE_MAMBA_HOST_RATIO.get() if is_gfx95_supported() else None
    return hicache_ratio if ratio is None else ratio


def clamp_mamba_pool(
    pool: MambaPoolHost, host_size: float, allocation_bytes: int, available_bytes: int
) -> tuple[int, int]:
    """On gfx950 a ratio-sized Mamba host pool that does not fit takes what the
    pinnable budget left rather than failing: shrink it to ``available_bytes`` and
    return its new preflight. Any other pool's preflight is returned unchanged."""
    if not (
        is_gfx95_supported() and allocation_bytes > available_bytes and host_size <= 0
    ):
        return allocation_bytes, available_bytes
    page_bytes = pool.size_per_token * pool.page_size
    pool.page_num = max(available_bytes // page_bytes, 1)
    pool.size = pool.page_num * pool.page_size
    logger.warning(
        "HiCache Mamba host pool clamped to %d state slots (%.2f GB): "
        "%.2f GB requested by the host-to-device ratio, %.2f GB available.",
        pool.size,
        pool.size * pool.size_per_token / 1e9,
        allocation_bytes / 1e9,
        available_bytes / 1e9,
    )
    return base.host_memory_preflight(
        pool.get_mapping_lengths(),
        pool.allocator,
        pool._buffer_device,
    )
