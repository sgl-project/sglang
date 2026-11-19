"""SlotGuard ledger and guarded-allocator fixtures over real CPU torch pools."""

from typing import NamedTuple

import torch

from sglang.srt.mem_cache.allocator.slotguard_paged import (
    SlotGuardPagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.kv.plan import TREE_TOKEN, TreeSlot
from sglang.srt.mem_cache.l1_transfer.slot_guard import LeaseIdentity, SlotGuard

# Equals wire.identity("source").incarnation; the ledger layer has no wire identity.
SOURCE_INCARNATION = "source-boot-1"


class TransferId(NamedTuple):
    """Stand-in for the wire TransferId: any hashable with target_incarnation."""

    target_incarnation: str
    rid: str
    attempt: int


def guard(**kw):
    return SlotGuard(
        1,
        SOURCE_INCARNATION,
        max_remote_pinned_pages=32,
        max_active_leases=16,
        max_pending_reclaim_pages=32,
        **kw,
    )


def seed_credit(g, pages, epoch=0):
    """Verified headroom is published only by finishing an allocation epoch."""
    g.begin_epoch(epoch)
    g.install_epoch(epoch, ())
    g.finish_epoch(epoch, verified_headroom_pages=pages)


def allocator(pages=100, page_size=4, need_sort=False, active=True):
    guard = SlotGuard(
        1,
        "source",
        max_remote_pinned_pages=pages,
        max_active_leases=100,
        max_pending_reclaim_pages=pages,
        max_pages=pages,
    )
    return SlotGuardPagedTokenToKVPoolAllocator(
        pages * page_size,
        page_size,
        torch.float16,
        "cpu",
        None,
        need_sort,
        slot_guard=guard,
        active=active,
    )


def lease_identity(attempt=0):
    return LeaseIdentity(
        "source", "target", TransferId("target", "r", attempt), f"n{attempt}"
    )


def epoch(alloc, number=1, protected=None):
    snap = alloc.begin_allocation_epoch(number)
    alloc.install_allocation_epoch(
        number, snap.held_pages if protected is None else protected
    )
    return snap


def owned(alloc, pages):
    """Allocate and publish whole pages; returns device indices, span and refs."""
    indices = alloc.alloc(pages * alloc.page_size)
    span = alloc.last_allocation_spans[0]
    refs = alloc.page_refs(span)
    alloc.slot_guard.publish(refs)
    return indices, span, refs


def runtime_fixture(pages=16, *, guarded=True, active=True):
    """A real Python URC over a 4-token-page CPU allocator and request rows."""
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    if guarded:
        alloc = allocator(pages=pages, active=active)
    else:
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

        alloc = PagedTokenToKVPoolAllocator(
            pages * 4, 4, torch.float16, "cpu", None, False
        )
    rows = ReqToTokenPool(4, 128, "cpu", False)
    cache = UnifiedRadixCache(
        CacheInitParams(
            False,
            rows,
            alloc,
            4,
            tree_slots=(TreeSlot(component="FULL", family=TREE_TOKEN),),
            tree_core_backend="python",
        )
    )
    return cache, alloc, rows
