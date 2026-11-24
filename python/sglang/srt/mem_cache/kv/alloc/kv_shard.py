"""Logical-page KV sharding of a unified chain's owner band.

Logical page ``l`` (from the stock `PageInterleavePoolAllocator`) belongs to
rank ``l % N``. Each rank binds its own pages, wherever its layout puts them.
The chain asks every capacity question of one snapshot built from the mirrored
class lists (`ChainAllocator._shard_snapshot`), so no rank asks another.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import msgspec
import torch

from sglang.srt.environ import InvariantCheckLevel
from sglang.srt.mem_cache.allocator.page_interleave import PageInterleavePoolAllocator
from sglang.srt.mem_cache.kv.alloc.invariants import SHARD_OWNED_PAGES
from sglang.srt.utils.invariants import expect, resolve_level

if TYPE_CHECKING:
    from sglang.srt.mem_cache.kv.alloc.band import Band
    from sglang.srt.mem_cache.page_interleave import PageShardSpec


class OwnerPageRule(msgspec.Struct, frozen=True, kw_only=True):
    """Rank ``shard_rank`` holds logical page ``l`` when ``l % shard_size`` is
    its rank, at band page ``l // shard_size - 1 + first_band_page``."""

    shard_size: int
    shard_rank: int
    # The band page of logical pages ``[N, 2N)``; ``[0, N)`` are reserved.
    first_band_page: int


class ShardedOwner:
    """One rank's sharded owner band: logical ids from the sharding allocator,
    this rank's pages in its band, and the counts every rank holds alike."""

    def __init__(self, *, band: Band, shard_spec: PageShardSpec):
        assert band.page_size == band.pool_page_size == shard_spec.page_size
        assert not band.is_id_owner and band.grow_direction == "down"
        self.band = band
        self.shard_size = shard_spec.shard_size
        self.shard_rank = shard_spec.shard_rank
        self.page_size = band.page_size
        # One rank's owner pages over the whole arena: ids never run out
        # before bytes.
        self.per_rank_pages = band.num_pages - band.min_page_index
        self.rule = OwnerPageRule(
            shard_size=self.shard_size,
            shard_rank=self.shard_rank,
            first_band_page=band.min_page_index,
        )
        # One past the largest logical page id; pages ``[0, N)`` are reserved.
        self._page_span = self.shard_size * (self.per_rank_pages + 1)
        # The request table holds token ids as int32.
        assert self.token_id_space < 2**31
        self.ids = PageInterleavePoolAllocator(
            size=self.per_rank_pages * self.page_size,
            physical_page_size=self.page_size,
            shard_size=self.shard_size,
            dtype=torch.int64,
            device=band.device,
            kvcache=None,
            need_sort=False,
            shard_spec=shard_spec,
        )
        # Every translation of the band's ids takes logical ids from here on.
        band.ids.owner_rule = self.rule
        # Bumped by every change of the class lists: part of the chain's pricing
        # key, and the key of the counts kept below.
        self.ids_epoch = 0
        self._free_counts: Tuple[int, ...] = ()
        self._free_counts_epoch = -1
        # `_split_by_class` sorts by (class, page): class c's keys end below
        # ``(c + 1) * span``.
        self._class_key_ends = (
            torch.arange(1, self.shard_size + 1, device=band.device) * self._page_span
        )

    @property
    def token_id_space(self) -> int:
        """One past the largest logical token id, the reserved pages included."""
        return self._page_span * self.page_size

    def class_free_page_counts(self) -> Tuple[int, ...]:
        if self._free_counts_epoch != self.ids_epoch:
            self._free_counts = tuple(self.ids.class_free_page_counts())
            self._free_counts_epoch = self.ids_epoch
        return self._free_counts

    def live_pages(self) -> List[int]:
        """Live owner pages per rank, read from the mirrored class lists."""
        return [self.per_rank_pages - free for free in self.class_free_page_counts()]

    def fullest_growth(
        self,
        prefix_lens_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        rotation_bases: Sequence[Optional[int]],
    ) -> int:
        """Pages the fullest rank grows by when a sharded extend lands, each class
        taking what its rotation gives it: ``ceil(K / N)`` of a new chain's K
        pages, whose base is drawn at the take."""
        ps, shard_size = self.page_size, self.shard_size
        live = self.live_pages()
        after = live
        for i, base in enumerate(rotation_bases):
            prefix_pages = int(prefix_lens_cpu[i]) // ps
            new_pages = -(-int(seq_lens_cpu[i]) // ps) - prefix_pages
            if base is None:
                counts = [-(-new_pages // shard_size)] * shard_size
            else:
                counts = PageInterleavePoolAllocator._class_counts(
                    (base + prefix_pages) % shard_size, new_pages, shard_size
                )
            after = [have + more for have, more in zip(after, counts)]
        return max(after) - max(live)

    def eviction_credit(
        self, evictable_pages: int, by_class: Optional[Sequence[int]]
    ) -> int:
        """Pages an eviction of ``evictable_pages`` frees on the fullest rank:
        exact with the count per owner class, else as if one rank held every
        page the cache cannot evict (never an over-credit)."""
        live = self.live_pages()
        if by_class is not None and sum(by_class) == evictable_pages:
            kept = max(have - min(have, gone) for have, gone in zip(live, by_class))
        else:
            kept = min(max(live), max(0, sum(live) - evictable_pages))
        return max(live) - kept

    # -- takes and frees --

    def take_ids(self, **extend) -> Optional[torch.Tensor]:
        """The sharding allocator's `alloc_extend`: logical token ids, the new
        chains' rotation bases resolved in place; None when a class is short."""
        self.ids_epoch += 1
        return self.ids.alloc_extend(**extend)

    def owned_new_pages(
        self,
        logical: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        rotation_bases: Sequence[int],
    ) -> Tuple[Optional[torch.Tensor], int]:
        """This rank's band pages among an extend's new ``logical`` ids, and
        their count. Page ``j`` of a chain is class ``(base + j) % N``."""
        ps, shard_size = self.page_size, self.shard_size
        reps, at = [], 0
        for i, base in enumerate(rotation_bases):
            prefix_len, seq_len = int(prefix_lens_cpu[i]), int(seq_lens_cpu[i])
            first = (self.shard_rank - base - prefix_len // ps) % shard_size
            end = at + seq_len - prefix_len
            if at + first * ps < end:
                reps.append(logical[at + first * ps : end : shard_size * ps])
            at = end
        if not reps:
            return None, 0
        tokens = reps[0] if len(reps) == 1 else torch.cat(reps)
        pages = tokens // ps
        if resolve_level() >= InvariantCheckLevel.WARN:
            expect(
                SHARD_OWNED_PAGES,
                (pages >= shard_size) & (pages % shard_size == self.shard_rank),
                msg="a rotation base does not match the logical ids it drew",
            )
        return self._band_pages(pages), int(tokens.shape[0])

    def give_back(
        self, logical_tokens: torch.Tensor, *, distinct_pages: bool = False
    ) -> torch.Tensor:
        """Return whole pages' ids to the class lists; this rank's come back as
        band pages. ``distinct_pages``: one token of each freed page is given."""
        pages = logical_tokens.detach().to(torch.int64) // self.page_size
        if not distinct_pages:
            pages = torch.unique(pages)
        elif resolve_level() >= InvariantCheckLevel.WARN:
            if pages.unique().numel() != pages.numel():
                raise RuntimeError("a sharded free names a page twice")
        by_class = self._split_by_class(pages)
        self.ids.free_classed(by_class)
        self.ids_epoch += 1
        return self._band_pages(by_class[self.shard_rank])

    def _band_pages(self, logical_pages: torch.Tensor) -> torch.Tensor:
        return logical_pages // self.shard_size - 1 + self.rule.first_band_page

    def _split_by_class(self, pages: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Distinct logical pages split by owner class, ascending within each:
        one sort by (class, page), one host read that also carries the range."""
        span = self._page_span
        keys = torch.sort((pages % self.shard_size) * span + pages).values
        *ends, low, high = torch.cat(
            (
                torch.searchsorted(keys, self._class_key_ends),
                torch.stack(torch.aminmax(pages)),
            )
        ).tolist()
        if low < self.shard_size or high >= span:
            raise RuntimeError(
                f"a sharded free names a page outside the logical id space "
                f"[{self.shard_size}, {span}): [{low}, {high}]"
            )
        sizes = [end - start for start, end in zip([0] + ends[:-1], ends)]
        return torch.split(keys % span, sizes)

    def clear(self) -> None:
        self.ids.clear()
        self.ids_epoch += 1

    def mirror_violations(self) -> List[str]:
        """Idle-time check: this rank's band binds exactly its class's live
        pages. Empty == healthy."""
        # Read from the lists, not the kept counts: a missed epoch bump shows.
        free = self.ids.class_free_page_counts()[self.shard_rank]
        mirrored = self.per_rank_pages - free
        if self.band.live_page_count == mirrored:
            return []
        return [
            f"[{self.band.sub_pool_name}] shard rank {self.shard_rank} binds "
            f"{self.band.live_page_count} pages, its class holds {mirrored}"
        ]
