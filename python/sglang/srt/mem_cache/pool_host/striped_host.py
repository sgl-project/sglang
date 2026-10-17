"""Page-striped logical host tier for replicated-device-KV HiCache (DSv4).

Every TP rank holds the same replicated device KV, so the host tier can be
striped instead of mirrored: logical host page ``p`` is physically stored only
on its owner rank ``p % N``. Each rank keeps ``1/N`` of the host bytes, and the
logical capacity grows ``N``-fold.

The logical index space is mirrored: every rank runs the same allocator over
the full ``N * per_rank_pages`` logical pages and must make bit-identical
decisions. Divergence would silently pair one rank's host page with another
rank's KV, so a rolling digest over every allocator mutation is always
maintained and must be cross-checked by the callers.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import struct
import time
from collections import deque
from typing import Any, Optional, Sequence

import msgspec
import torch

from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    LogicalHostPool,
)
from sglang.srt.mem_cache.pool_host.base import synchronized

logger = logging.getLogger(__name__)

_DIGEST_MASK = (1 << 63) - 1
_OP_TAIL_LEN = 64

_OP_CLEAR = 0
_OP_ALLOC = 1
_OP_FREE = 2
_OP_NAMES = {_OP_CLEAR: "clear", _OP_ALLOC: "alloc", _OP_FREE: "free"}


class StripedHostInvariantError(RuntimeError):
    """Fatal: the mirrored logical host index can no longer be trusted."""


class StripedHostDigestMismatch(StripedHostInvariantError):
    """The mirrored logical host index diverged across ranks."""


class StripedHostInvalidFree(StripedHostInvariantError):
    """A free that the mirrored index cannot apply (likely a diverged rank)."""


class StripedHostInvalidRollback(StripedHostInvariantError):
    """A rollback that does not undo the directly preceding alloc."""


class StripedHostPlacement:
    """Owner and rank-local placement of logical host token indices.

    ``owner(h) = (h // ps) % N`` and
    ``local(h) = ((h // ps) // N) * ps + h % ps``.
    """

    def __init__(self, num_ranks: int, rank: int, page_size: int):
        if num_ranks < 1:
            raise ValueError(f"num_ranks must be >= 1, got {num_ranks}")
        if not 0 <= rank < num_ranks:
            raise ValueError(f"rank must be in [0, {num_ranks}), got {rank}")
        if page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {page_size}")
        self.num_ranks = num_ranks
        self.rank = rank
        self.page_size = page_size

    def __repr__(self) -> str:
        return (
            f"StripedHostPlacement(num_ranks={self.num_ranks}, rank={self.rank}, "
            f"page_size={self.page_size})"
        )

    def key(self) -> tuple[int, int, int]:
        return self.num_ranks, self.rank, self.page_size

    def owner_of_pages(self, pages: torch.Tensor) -> torch.Tensor:
        return pages % self.num_ranks

    def owner(self, host_indices: torch.Tensor) -> torch.Tensor:
        return self.owner_of_pages(host_indices // self.page_size)

    def local(self, host_indices: torch.Tensor) -> torch.Tensor:
        ps = self.page_size
        return (host_indices // ps // self.num_ranks) * ps + host_indices % ps

    def owned_mask(self, host_indices: torch.Tensor) -> torch.Tensor:
        return self.owner(host_indices) == self.rank


class StripedHostOpRecord(msgspec.Struct, frozen=True):
    seq: int
    op: str
    num_pages: int
    head_pages: tuple[int, ...]
    digest: int


class _StripedAllocUndo(msgspec.Struct, frozen=True):
    """What ``rollback_alloc`` needs to make the last alloc never have happened."""

    pages: torch.Tensor
    # (owner class, pages popped from the head of its FIFO), in pop order.
    taken: tuple[tuple[int, torch.Tensor], ...]
    cursor: int
    digest: bytes
    seq: int
    # Op tail record the alloc's record pushed out of the bounded deque.
    evicted_tail: Optional[StripedHostOpRecord]


class StripedLogicalHostPool(LogicalHostPool):
    """Mirrored logical anchor whose pages are striped over ``N`` owner ranks.

    Free pages are kept in one FIFO per owner class. ``alloc`` hands out pages
    round-robin over the classes starting at a rotating cursor, so the pages
    of one allocation (and of the steady-state working set) are spread evenly
    over the owners. When a class runs dry it falls back to the next non-empty
    class, so an allocation fails only when the total free size is short.

    ``digest()`` is a rolling 64-bit hash over every mutation (op, pages in
    order). It is part of the state, not a debug aid: callers must fold it into
    their cross-rank checks (see ``digest_reduce_fields``).
    """

    def __init__(
        self,
        per_rank_size: int,
        page_size: int,
        placement: StripedHostPlacement,
        layout: str = "layer_first",
    ):
        if per_rank_size % page_size != 0:
            raise ValueError(
                "StripedLogicalHostPool per-rank size must be page-aligned, "
                f"got per_rank_size={per_rank_size}, page_size={page_size}"
            )
        if placement.page_size != page_size:
            raise ValueError(
                "StripedLogicalHostPool placement page size mismatch: "
                f"{placement.page_size} != {page_size}"
            )
        self.placement = placement
        self.num_ranks = placement.num_ranks
        self.per_rank_size = per_rank_size
        self.per_rank_pages = per_rank_size // page_size
        self.num_pages = self.num_ranks * self.per_rank_pages
        self._digest_seed = hashlib.blake2b(
            struct.pack(
                "<4sqqq", b"SHv1", self.num_ranks, self.per_rank_pages, page_size
            ),
            digest_size=8,
        ).digest()
        self._digest = self._digest_seed
        self._seq = 0
        self._op_tail: deque[StripedHostOpRecord] = deque(maxlen=_OP_TAIL_LEN)
        self._page_offsets = torch.arange(page_size, dtype=torch.int64)
        super().__init__(self.num_ranks * per_rank_size, page_size, layout)

    @synchronized
    def clear(self):
        num_ranks = self.num_ranks
        # Owner class c is a FIFO of its pages: ``free`` is the head tensor and
        # ``release`` holds freed chunks, merged lazily like HostKVCache.
        self._free = [
            torch.arange(c, self.num_pages, num_ranks, dtype=torch.int64)
            for c in range(num_ranks)
        ]
        self._release = [[] for _ in range(num_ranks)]
        self._num_free = [self.per_rank_pages] * num_ranks
        self._num_free_pages = self.num_pages
        self._allocated = torch.zeros(self.num_pages, dtype=torch.bool)
        self._cursor = 0
        # Clear is a recorded mutation; the digest keeps rolling across it.
        self._record(_OP_CLEAR, None)
        self._last_alloc: Optional[_StripedAllocUndo] = None

    def available_size(self):
        return self._num_free_pages * self.page_size

    def free_pages_per_class(self) -> list[int]:
        return list(self._num_free)

    def used_pages_per_class(self) -> list[int]:
        return [self.per_rank_pages - num_free for num_free in self._num_free]

    def _plan_takes(self, num_pages: int) -> list[int]:
        """Pages taken per class by ``num_pages`` round-robin visits.

        The visits start at the cursor and skip dry classes, so they fill
        every class up to a common level, then take one more page from the
        first classes (from the cursor) still above that level.
        """
        num_ranks = self.num_ranks
        num_free = self._num_free
        level, used, prev, active = 0, 0, 0, num_ranks
        for count in sorted(num_free):
            step = (count - prev) * active
            if used + step > num_pages:
                level = prev + (num_pages - used) // active
                break
            used += step
            prev = count
            active -= 1
        else:
            level = prev
        takes = [min(count, level) for count in num_free]
        extra = num_pages - sum(takes)
        for i in range(num_ranks):
            if extra == 0:
                break
            cls = (self._cursor + i) % num_ranks
            if num_free[cls] > level:
                takes[cls] += 1
                extra -= 1
        return takes

    def _pop_class(self, cls: int, take: int) -> torch.Tensor:
        if take > len(self._free[cls]):
            self._free[cls] = torch.cat([self._free[cls], *self._release[cls]])
            self._release[cls] = []
        pages = self._free[cls][:take]
        self._free[cls] = self._free[cls][take:]
        self._num_free[cls] -= take
        return pages

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        page_size = self.page_size
        if need_size % page_size != 0:
            raise ValueError(
                "StripedLogicalHostPool allocation must be page-aligned, "
                f"got need_size={need_size}, page_size={page_size}"
            )
        num_pages = need_size // page_size
        if num_pages > self._num_free_pages:
            return None
        if num_pages == 0:
            # No state change, so not a recorded op (a rank skipping an empty
            # alloc must not look diverged). It still ends the rollback window.
            self._last_alloc = None
            return torch.empty(0, dtype=torch.int64)

        # Page order is that of one visit per page: the j-th page of class c
        # sorts by (j, (c - cursor) % N).
        num_ranks = self.num_ranks
        cursor = self._cursor
        digest, seq = self._digest, self._seq
        evicted_tail = (
            self._op_tail[0] if len(self._op_tail) == self._op_tail.maxlen else None
        )
        chunks, keys, taken = [], [], []
        last_key = -1
        for cls, take in enumerate(self._plan_takes(num_pages)):
            if take == 0:
                continue
            rot = (cls - cursor) % num_ranks
            chunks.append(self._pop_class(cls, take))
            taken.append((cls, chunks[-1]))
            keys.append(torch.arange(take, dtype=torch.int64) * num_ranks + rot)
            last_key = max(last_key, (take - 1) * num_ranks + rot)
        if len(chunks) == 1:
            page_tensor = chunks[0]
        else:
            page_tensor = torch.cat(chunks)[torch.argsort(torch.cat(keys))]
        self._cursor = (cursor + last_key % num_ranks + 1) % num_ranks
        self._num_free_pages -= num_pages

        self._allocated[page_tensor] = True
        self._record(_OP_ALLOC, page_tensor)
        self._last_alloc = _StripedAllocUndo(
            pages=page_tensor,
            taken=tuple(taken),
            cursor=cursor,
            digest=digest,
            seq=seq,
            evicted_tail=evicted_tail,
        )
        return self._pages_to_indices(page_tensor)

    @synchronized
    def rollback_alloc(self, indices: torch.Tensor) -> None:
        """Undo the directly preceding ``alloc`` exactly (admission veto).

        Unlike ``free`` (a recorded op that appends to the FIFO tails), the
        pool ends up indistinguishable from one that never made the alloc:
        same per-class FIFO order, cursor, digest and op sequence. Any other
        mutation in between (alloc, free, clear) closes the window. A rollback
        that does not match is fatal: the caller has lost track of the index.
        """
        undo = self._last_alloc
        if undo is None:
            self._raise_fatal(
                StripedHostInvalidRollback,
                "rollback without a directly preceding alloc",
            )
        indices_cpu = indices.to(dtype=torch.int64, device="cpu").flatten()
        if not torch.equal(indices_cpu, self._pages_to_indices(undo.pages)):
            self._raise_fatal(
                StripedHostInvalidRollback, "rollback does not match the last alloc"
            )
        for cls, chunk in undo.taken:
            self._free[cls] = torch.cat([chunk, self._free[cls]])
            self._num_free[cls] += chunk.numel()
        self._num_free_pages += undo.pages.numel()
        self._allocated[undo.pages] = False
        self._cursor = undo.cursor
        self._digest, self._seq = undo.digest, undo.seq
        self._op_tail.pop()
        if undo.evicted_tail is not None:
            self._op_tail.appendleft(undo.evicted_tail)
        self._last_alloc = None

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        page_size = self.page_size
        num_indices = indices.numel()
        if num_indices % page_size != 0:
            self._raise_fatal(
                StripedHostInvalidFree,
                f"free of {num_indices} indices is not page-aligned "
                f"(page_size={page_size})",
            )
        if num_indices == 0:
            return 0
        indices_cpu = indices.to(dtype=torch.int64, device="cpu").flatten()

        page_tensor = self._indices_to_pages(indices_cpu)
        if not bool(self._allocated[page_tensor].all()):
            self._raise_fatal(StripedHostInvalidFree, "free of unallocated pages")
        if torch.unique(page_tensor).numel() != page_tensor.numel():
            self._raise_fatal(StripedHostInvalidFree, "free of duplicate pages")
        self._allocated[page_tensor] = False

        # Append each class's pages to its FIFO, keeping their freed order.
        num_ranks = self.num_ranks
        classes = page_tensor % num_ranks
        by_class = page_tensor[torch.argsort(classes, stable=True)]
        counts = torch.bincount(classes, minlength=num_ranks).tolist()
        for cls, chunk in enumerate(torch.split(by_class, counts)):
            if counts[cls]:
                self._release[cls].append(chunk)
                self._num_free[cls] += counts[cls]
        self._num_free_pages += page_tensor.numel()
        self._record(_OP_FREE, page_tensor)
        self._last_alloc = None
        return num_indices

    def _pages_to_indices(self, pages: torch.Tensor) -> torch.Tensor:
        page_size = self.page_size
        return torch.add((pages * page_size).unsqueeze(1), self._page_offsets).view(-1)

    def _indices_to_pages(self, indices: torch.Tensor) -> torch.Tensor:
        pages = indices[:: self.page_size] // self.page_size
        if not torch.equal(indices, self._pages_to_indices(pages)):
            self._raise_fatal(StripedHostInvalidFree, "free of partial pages")
        lo, hi = torch.aminmax(pages)
        if int(lo) < 0 or int(hi) >= self.num_pages:
            self._raise_fatal(StripedHostInvalidFree, "free of out-of-range pages")
        return pages

    # ---- Mirrored-index digest ----

    def _record(self, op: int, pages: Optional[torch.Tensor]) -> None:
        num_pages = 0 if pages is None else pages.numel()
        h = hashlib.blake2b(digest_size=8)
        h.update(self._digest)
        h.update(struct.pack("<Bqq", op, self._seq, num_pages))
        if pages is not None:
            h.update(pages.numpy().tobytes())
        self._digest = h.digest()
        self._seq += 1
        head = () if pages is None else tuple(pages[:4].tolist())
        self._op_tail.append(
            StripedHostOpRecord(
                seq=self._seq,
                op=_OP_NAMES[op],
                num_pages=num_pages,
                head_pages=head,
                digest=self.digest(),
            )
        )

    @synchronized
    def digest(self) -> int:
        """Rolling 63-bit digest of every allocator mutation so far."""
        return int.from_bytes(self._digest, "little") & _DIGEST_MASK

    @property
    def op_seq(self) -> int:
        return self._seq

    @synchronized
    def digest_reduce_fields(self) -> list[int]:
        """``[d, -d, seq, -seq]`` for a MIN all-reduce, read atomically.

        The reduce tensor must be int64: ``d`` is a 63-bit value. After the
        MIN reduce, ``[min(d), -max(d), min(seq), -max(seq)]`` is checked by
        ``check_reduced_digest``; equal min and max on every rank means the
        mirrored index is identical.
        """
        digest = self.digest()
        return [digest, -digest, self._seq, -self._seq]

    @synchronized
    def check_reduced_digest(self, reduced: Sequence[int]) -> None:
        """Raise if MIN-reduced ``digest_reduce_fields`` show any divergence."""
        d_min, neg_d_max, s_min, neg_s_max = (int(v) for v in reduced)
        if d_min != -neg_d_max or s_min != -neg_s_max:
            self._raise_fatal(
                StripedHostDigestMismatch,
                f"digest range [{d_min}, {-neg_d_max}], "
                f"seq range [{s_min}, {-neg_s_max}]",
            )

    def _raise_fatal(self, exc_type: type, detail: str) -> None:
        tail = "\n".join(str(record) for record in self._op_tail)
        logger.error(
            "Striped host index invariant broken on rank %d (%s); local "
            "digest=%d seq=%d; op tail:\n%s",
            self.placement.rank,
            detail,
            self.digest(),
            self._seq,
            tail,
        )
        raise exc_type(f"Striped host index invariant broken: {detail}")


# ---- Owned-row narrowing and load plans ----


def host_pages_of(host_indices: torch.Tensor, page_size: int) -> torch.Tensor:
    """Logical host page ids of a whole-page op; raises on a partial or split page."""
    if host_indices.device.type != "cpu":
        raise ValueError("Striped host indices must be CPU tensors")
    if host_indices.numel() % page_size != 0:
        raise ValueError(
            f"Striped host op must move whole pages: {host_indices.numel()} slots "
            f"for page size {page_size}"
        )
    rows = host_indices.reshape(-1, page_size)
    starts = rows[:, :1]
    offsets = torch.arange(page_size, dtype=rows.dtype)
    if bool((starts % page_size != 0).any()) or not torch.equal(
        rows - starts, offsets.expand_as(rows)
    ):
        raise ValueError("Striped host op pages must be contiguous and page-aligned")
    return starts.flatten() // page_size


def _page_token_positions(page_positions: torch.Tensor, page_size: int) -> torch.Tensor:
    offsets = torch.arange(page_size, dtype=torch.int64)
    return (page_positions[:, None] * page_size + offsets).flatten()


def _select(indices: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    # Positions are built on CPU; a non-blocking move keeps device ids un-synced.
    return indices.index_select(0, positions.to(indices.device, non_blocking=True))


def _device_pages_of(device_indices: torch.Tensor, page_size: int) -> torch.Tensor:
    """Device page ids of a whole-page op, asserted page-aligned without a host sync.

    The gather overwrites whole device pages, so a split or misaligned device
    page must abort rather than land in the wrong rows.
    """
    rows = device_indices.reshape(-1, page_size)
    starts = rows[:, :1]
    offsets = torch.arange(page_size, dtype=rows.dtype, device=rows.device)
    torch._assert_async(
        ((rows - starts) == offsets).all() & (starts % page_size == 0).all(),
        "Striped load device pages must be contiguous and page-aligned",
    )
    return starts.flatten() // page_size


def owned_rows_replicated(
    placement: StripedHostPlacement,
    *,
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Narrow a replicated-device transfer to the pages this rank owns.

    Returns ``(local host slots, device slots)`` of the owned pages in op order.
    Device slots are not translated: every rank holds every device row.
    """
    if host_indices.numel() != device_indices.numel():
        raise ValueError(
            f"Striped transfer size mismatch: host={host_indices.numel()}, "
            f"device={device_indices.numel()}"
        )
    host_pages_of(host_indices, placement.page_size)  # whole, aligned pages only
    positions = torch.nonzero(placement.owned_mask(host_indices)).flatten()
    return (
        placement.local(host_indices.index_select(0, positions)),
        _select(device_indices, positions),
    )


class StripedLoadPlan(msgspec.Struct, kw_only=True):
    """One merged load op: owned H2D rows plus the per-owner device pages to gather."""

    # Owned rows for this rank's H2D, as owned_rows_replicated returns them.
    own_host_local: torch.Tensor
    own_dev: torch.Tensor
    # Device page ids (device slot // page_size) per owner, in op page order;
    # on the device of the op's device indices.
    dev_pages_by_owner: tuple[torch.Tensor, ...]
    pages_per_owner: tuple[int, ...]
    k_max: int
    # Per transfer layer: (device page buffer, bytes per page row) of each striped pool.
    layers: tuple[tuple[tuple[torch.Tensor, int], ...], ...]
    # Rank-identical plan sequence number.
    seq: int
    # Set by the gather executor before its first collective; a started plan is
    # never gathered again, even if that gather failed part-way.
    gather_started: bool = False
    # Recorded by the gather executor after the last layer's gather.
    gather_done_event: Optional[Any] = None
    # Recorded after the waits, before the first pack;
    # start-to-done spans the gather alone.
    gather_start_event: Optional[Any] = None
    # Peer payload this rank received for the plan (padding excluded).
    gathered_bytes: int = 0


def build_striped_load_plan(
    placement: StripedHostPlacement,
    *,
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    seq: int,
    layers: tuple[tuple[tuple[torch.Tensor, int], ...], ...] = (),
) -> StripedLoadPlan:
    pages = host_pages_of(host_indices, placement.page_size)
    if device_indices.numel() != host_indices.numel():
        raise ValueError(
            f"Striped load size mismatch: host={host_indices.numel()}, "
            f"device={device_indices.numel()}"
        )
    owners = placement.owner_of_pages(pages)
    dev_pages = _device_pages_of(device_indices, placement.page_size)
    page_positions = [
        torch.nonzero(owners == owner).flatten() for owner in range(placement.num_ranks)
    ]
    pages_per_owner = tuple(int(p.numel()) for p in page_positions)
    own_host_local, own_dev = owned_rows_replicated(
        placement, host_indices=host_indices, device_indices=device_indices
    )
    return StripedLoadPlan(
        own_host_local=own_host_local,
        own_dev=own_dev,
        dev_pages_by_owner=tuple(_select(dev_pages, p) for p in page_positions),
        pages_per_owner=pages_per_owner,
        k_max=max(pages_per_owner, default=0),
        layers=layers,
        seq=seq,
    )


def collect_striped_plan_layers(
    transfers: Sequence[Any], *, transfer_layer_id_max: int
) -> tuple[tuple[tuple[torch.Tensor, int], ...], ...]:
    """Per transfer layer, the device page buffers the gather must fill.

    ``transfers`` are L2Transfer-like (``host_pool``, ``layer_mapper``); only
    striped DSv4 paged pools carry rows, the logical anchor carries none.
    """
    layers = []
    for layer_id in range(transfer_layer_id_max):
        buffers = []
        for transfer in transfers:
            host_pool = transfer.host_pool
            if not isinstance(host_pool, DeepSeekV4PagedHostPool):
                continue
            host_layer = (
                transfer.layer_mapper(layer_id)
                if transfer.layer_mapper is not None
                else layer_id
            )
            if host_layer is None or host_layer >= host_pool.layer_num:
                continue
            buffers.append((host_pool.device_buffers[host_layer], host_pool.item_bytes))
        layers.append(tuple(buffers))
    return tuple(layers)


def validate_striped_host_group(
    host_group: Any, placement: StripedHostPlacement
) -> None:
    """Fail closed unless every data pool is a per-rank DSv4 page mirror of the anchor."""
    anchor = host_group.anchor_entry.host_pool
    if (
        not isinstance(anchor, StripedLogicalHostPool)
        or anchor.placement.key() != placement.key()
    ):
        raise ValueError("Striped host tier requires a StripedLogicalHostPool anchor")
    for entry in host_group.entries:
        if entry is host_group.anchor_entry:
            continue
        pool = entry.host_pool
        if not isinstance(pool, DeepSeekV4PagedHostPool):
            raise ValueError(
                f"Striped host tier cannot stripe host pool {entry.name}: "
                f"{type(pool).__name__} is not a DeepSeek V4 paged pool"
            )
        if (
            pool.slot_page_size != placement.page_size
            or pool.num_host_pages != anchor.per_rank_pages
        ):
            raise ValueError(
                f"Striped host pool {entry.name} must hold {anchor.per_rank_pages} "
                f"pages of {placement.page_size} slots, got {pool.num_host_pages} "
                f"of {pool.slot_page_size}"
            )
        if not pool.page_aligned_only:
            raise ValueError(
                f"Striped host pool {entry.name} must be page-aligned only"
            )
        if entry.packed_draft_device_pools:
            raise ValueError(
                f"Striped host tier does not support packed draft pools ({entry.name})"
            )


# ---- Load gather ----


class StripedGatherError(StripedHostInvariantError):
    """A load gather could not run as planned; device rows would be incomplete."""


class StripedGatherFatal(StripedGatherError):
    """A gather failed after it may have entered a collective.

    Peer ranks can be blocked inside the all-gather, so this must not be caught
    or retried: it has to reach the scheduler's crash path.
    """


class StripedGatherSignature(msgspec.Struct, frozen=True):
    """Everything that sizes a plan's collectives; must match on every rank."""

    seq: int
    k_max: int
    row_bytes: int
    chunk_pages: int
    num_segments: int


class _CompletedGatherEvent:
    """Gather-done event of a host-synchronous (CPU) gather: complete when made."""

    def __init__(self):
        self._time = time.perf_counter()

    def query(self) -> bool:
        return True

    def elapsed_time(self, end: _CompletedGatherEvent) -> float:
        """Milliseconds to ``end``, like ``torch.cuda.Event.elapsed_time``."""
        return (end._time - self._time) * 1000.0

    def synchronize(self) -> None:
        pass

    def wait(self, stream: Any = None) -> None:
        pass


class PyncclStripedGatherTransport:
    """Production transport: pynccl all-gather on the current stream.

    The gather must run on the forward stream, in program order with the
    forward's own collectives. ``GroupCoordinator.all_gather_into_tensor``
    is not used because it falls back to ProcessGroupNCCL on its internal
    stream whenever pynccl is disabled, which it is outside graph capture.
    The communicator is enabled explicitly for the call instead.
    """

    def __init__(self, tp_group: Any):
        comm = tp_group.pynccl_comm
        if comm is None or not comm.available:
            raise StripedGatherError(
                "Striped host tier needs an available pynccl communicator on the "
                "TP group for the load gather"
            )
        if (
            comm.world_size != tp_group.world_size
            or comm.rank != tp_group.rank_in_group
        ):
            raise StripedGatherError(
                f"pynccl communicator (rank {comm.rank}/{comm.world_size}) does not "
                f"match the TP group ({tp_group.rank_in_group}/{tp_group.world_size})"
            )
        self.comm = comm
        self.world_size = comm.world_size
        self.rank = comm.rank
        self.device = comm.device

    def all_gather(self, recv: torch.Tensor, send: torch.Tensor) -> None:
        with self.comm.change_state(enable=True):
            if self.comm.disabled:
                # pynccl silently skips a disabled call; that would leave peer
                # rows stale on this rank only.
                raise StripedGatherError("pynccl communicator refused to enable")
            self.comm.all_gather(recv, send)


class TorchDistStripedGatherTransport:
    """CPU transport over a torch.distributed (gloo) group, for tests and fuzzing."""

    def __init__(self, group: Any):
        import torch.distributed as dist

        self._dist = dist
        self.group = group
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.device = torch.device("cpu")

    def all_gather(self, recv: torch.Tensor, send: torch.Tensor) -> None:
        outputs = list(recv.view(self.world_size, -1).unbind(0))
        # ``send`` aliases this rank's output block; gloo needs a distinct input.
        self._dist.all_gather(outputs, send.clone(), group=self.group)


def _page_rows_u8(buffer: torch.Tensor, item_bytes: int) -> torch.Tensor:
    """A writable ``[pages, item_bytes]`` uint8 view of a device page buffer.

    The gather writes through this view, so a buffer whose page rows are not
    one contiguous ``item_bytes`` run is refused rather than copied.
    """
    row_bytes = (
        buffer[0].numel() * buffer.element_size() if buffer.dim() and len(buffer) else 0
    )
    if not buffer.is_contiguous() or buffer.dim() == 0 or row_bytes != item_bytes:
        raise StripedGatherError(
            f"Striped gather needs contiguous device page rows of {item_bytes} B, "
            f"got shape {tuple(buffer.shape)} {buffer.dtype} "
            f"contiguous={buffer.is_contiguous()}"
        )
    return buffer.view(buffer.shape[0], -1).view(torch.uint8)


class StripedGatherExecutor:
    """Fills every rank's replicated device pages from their host owners.

    Each owner has already copied its own pages host-to-device. One gather
    moves all layers and pools of a plan: a page's rows of every
    (layer, pool) are packed into one row of ``row_bytes`` bytes, and the
    owners' page lists are padded to ``k_max`` so every rank issues the same
    ``ceil(k_max / chunk_pages)`` equal-size all-gathers.

    Staging is one ``staging_bytes`` device buffer allocated here, so the
    executor is built before KV-cache memory profiling; it never grows.
    """

    def __init__(
        self,
        placement: StripedHostPlacement,
        transport: Any,
        *,
        device: torch.device,
        staging_bytes: Optional[int] = None,
    ):
        if (
            transport.world_size != placement.num_ranks
            or transport.rank != placement.rank
        ):
            raise StripedGatherError(
                f"Gather transport rank {transport.rank}/{transport.world_size} does "
                f"not match placement {placement!r}"
            )
        if staging_bytes is None:
            from sglang.srt.environ import envs

            staging_bytes = envs.SGLANG_HICACHE_STRIPE_STAGING_MB.get() << 20
        if staging_bytes <= 0:
            raise ValueError("Striped gather staging must be positive")
        self.placement = placement
        self.transport = transport
        device = torch.device(device)
        if device.type != "cpu" and device.index is None:
            # Buffers report an indexed device; compare like with like.
            device = torch.device(
                device.type, torch.get_device_module(device).current_device()
            )
        self.device = device
        self.staging_bytes = staging_bytes
        self._staging = torch.empty(staging_bytes, dtype=torch.uint8, device=device)
        # Staging is reused across plans; a gather on another stream must not
        # pack into it before the previous plan's unpack has read it.
        self._last_done: Optional[Any] = None
        self.num_gathers = 0
        # Peer payload received by this rank; padding excluded.
        self.gathered_bytes = 0
        # Bytes this rank received on the wire, padding included.
        self.wire_bytes = 0

    def chunk_pages(self, row_bytes: int) -> int:
        return self.staging_bytes // (self.placement.num_ranks * row_bytes)

    def validate_layers(self, layers) -> int:
        """Check the plan layers' device buffers at init; return packed row bytes."""
        return self._checked_segments(layers)[1]

    def signature(self, plan: StripedLoadPlan) -> StripedGatherSignature:
        """The plan's collective shape, for the per-forward CPU consensus."""
        segments, row_bytes = self._checked_segments(plan.layers)
        return StripedGatherSignature(
            seq=plan.seq,
            k_max=plan.k_max,
            row_bytes=row_bytes,
            chunk_pages=self.chunk_pages(row_bytes) if row_bytes else 0,
            num_segments=len(segments),
        )

    def _checked_segments(self, layers) -> tuple[list[torch.Tensor], int]:
        segments = [
            _page_rows_u8(buffer, item_bytes)
            for layer in layers
            for buffer, item_bytes in layer
        ]
        for seg in segments:
            if seg.device != self.device:
                raise StripedGatherError(
                    f"Striped gather buffer on {seg.device}, executor on {self.device}"
                )
        row_bytes = sum(seg.shape[1] for seg in segments)
        if segments and self.chunk_pages(row_bytes) < 1:
            raise StripedGatherError(
                f"Striped gather staging of {self.staging_bytes} B cannot hold "
                f"{self.placement.num_ranks} page rows of {row_bytes} B"
            )
        return segments, row_bytes

    def _check_plan(self, plan: StripedLoadPlan) -> None:
        n = self.placement.num_ranks
        if plan.gather_started:
            raise StripedGatherError(f"Striped load plan {plan.seq} gathered twice")
        if len(plan.dev_pages_by_owner) != n or len(plan.pages_per_owner) != n:
            raise StripedGatherError(
                f"Striped load plan {plan.seq} has {len(plan.pages_per_owner)} owners, "
                f"expected {n}"
            )
        if plan.k_max != max(plan.pages_per_owner, default=0):
            raise StripedGatherError(
                f"Striped load plan {plan.seq}: k_max {plan.k_max} != "
                f"max{plan.pages_per_owner}"
            )
        for pages, count in zip(plan.dev_pages_by_owner, plan.pages_per_owner):
            if pages.numel() != count:
                raise StripedGatherError(
                    f"Striped load plan {plan.seq}: {pages.numel()} device pages "
                    f"for an owner with {count}"
                )

    def gather_plan(
        self,
        plan: StripedLoadPlan,
        *,
        stream: Any,
        wait_events: Sequence[Any] = (),
    ):
        """Gather one plan on ``stream`` and set its ``gather_done_event``.

        ``stream`` is the forward stream on GPU and must be None on CPU.
        ``wait_events`` (the full owned-H2D events) are waited on first, so
        this rank's own rows are complete before they are packed.
        Checks before the first collective raise ``StripedGatherError``; they
        depend only on rank-identical state, so every rank raises together.
        Any later failure raises ``StripedGatherFatal``.
        """
        self._check_plan(plan)
        if (stream is None) != (self.device.type == "cpu"):
            raise StripedGatherError(
                f"Striped gather on {self.device} needs stream=None only on CPU"
            )
        segments, row_bytes = self._checked_segments(plan.layers)
        plan.gather_started = True
        try:
            with self._on_stream(stream):
                for event in (*wait_events, self._last_done):
                    if event is not None:
                        event.wait(stream)
                plan.gather_start_event = self._record_event(stream)
                if self.placement.num_ranks > 1 and plan.k_max > 0 and segments:
                    self._gather(plan, segments, row_bytes)
                done = self._record_event(stream)
        except Exception as exc:
            raise StripedGatherFatal(
                f"Striped load plan {plan.seq} failed mid-gather on rank "
                f"{self.placement.rank}; peers may be blocked in the all-gather"
            ) from exc
        plan.gather_done_event = done
        self._last_done = done
        return done

    def _on_stream(self, stream: Any):
        if stream is None:
            return contextlib.nullcontext()
        return torch.get_device_module(self.device).stream(stream)

    def _record_event(self, stream: Any) -> Any:
        if stream is None:
            return _CompletedGatherEvent()
        event = torch.get_device_module(self.device).Event(enable_timing=True)
        event.record(stream)
        return event

    def _gather(self, plan, segments, row_bytes) -> None:
        n, rank = self.placement.num_ranks, self.placement.rank
        widths = [seg.shape[1] for seg in segments]
        chunk = self.chunk_pages(row_bytes)
        dev_pages = [
            p.to(device=self.device, dtype=torch.int64) for p in plan.dev_pages_by_owner
        ]
        # Row j of owner c sits at blocks[c, j]; built on the device, never H2D.
        owner_grid = torch.arange(n, device=self.device).repeat_interleave(chunk)
        owner_grid = owner_grid.view(n, chunk)
        row_ids = torch.arange(chunk, device=self.device)
        for start in range(0, plan.k_max, chunk):
            pages = min(chunk, plan.k_max - start)
            block = pages * row_bytes
            blocks = self._staging[: n * block].view(n, block)
            counts = [
                min(max(plan.pages_per_owner[c] - start, 0), pages) for c in range(n)
            ]
            peers = [c for c in range(n) if c != rank and counts[c] > 0]
            self._pack(
                blocks[rank], segments, widths, pages, dev_pages[rank], start, counts
            )
            # Rows past an owner's count are padding: sent, never read.
            self.transport.all_gather(self._staging[: n * block], blocks[rank])
            self.num_gathers += 1
            self.wire_bytes += (n - 1) * block
            received = sum(counts[c] for c in peers) * row_bytes
            self.gathered_bytes += received
            plan.gathered_bytes += received
            if not peers:
                continue
            peer_dev = torch.cat(
                [dev_pages[c][start : start + counts[c]] for c in peers]
            )
            peer_owner = torch.cat([owner_grid[c, : counts[c]] for c in peers])
            peer_row = torch.cat([row_ids[: counts[c]] for c in peers])
            offset = 0
            for seg, width in zip(segments, widths):
                rows = blocks[:, offset : offset + pages * width].view(n, pages, width)
                seg.index_copy_(0, peer_dev, rows[peer_owner, peer_row])
                offset += pages * width

    def _pack(self, send, segments, widths, pages, own_dev_pages, start, counts):
        own_count = counts[self.placement.rank]
        if not own_count:
            return
        own_pages = own_dev_pages[start : start + own_count]
        offset = 0
        for seg, width in zip(segments, widths):
            out = send[offset : offset + pages * width].view(pages, width)
            torch.index_select(seg, 0, own_pages, out=out[:own_count])
            offset += pages * width


def create_striped_gather_executor(
    tp_group: Any,
    *,
    page_size: int,
    device: Any,
    staging_bytes: Optional[int] = None,
) -> StripedGatherExecutor:
    """Production executor: pynccl all-gather over ``tp_group``."""
    transport = PyncclStripedGatherTransport(tp_group)
    placement = StripedHostPlacement(transport.world_size, transport.rank, page_size)
    executor = StripedGatherExecutor(
        placement, transport, device=device, staging_bytes=staging_bytes
    )
    logger.info(
        "Striped HiCache host tier: %d MiB load-gather staging over %d TP ranks",
        executor.staging_bytes >> 20,
        placement.num_ranks,
    )
    return executor


def plan_owner_imbalance(plan: StripedLoadPlan) -> float:
    """``k_max / mean(k_c)`` of a plan: its gather pads every owner to ``k_max``."""
    total = sum(plan.pages_per_owner)
    if total == 0:
        return 1.0
    return plan.k_max * len(plan.pages_per_owner) / total


# ---- Forward-entry drain ----


class StripedLoadConsensusError(StripedHostInvariantError):
    """Ranks disagree on the pending striped loads; fatal, never retried."""


class StripedLoadNotDrainedError(StripedHostInvariantError):
    """A started load reached a scheduler check without being gathered by a forward.

    Some forward consumed (or skipped) its device rows without the forward-entry
    gather, so peer-owned rows may have been read stale; fatal, never retried.
    """


class _PendingStripedLoad(msgspec.Struct, frozen=True):
    plan: StripedLoadPlan
    h2d_done_event: Any
    # Recorded after the plan's device page tensors were built (on the stream
    # that called enqueue); the gather may run on another stream.
    plan_ready_event: Any
    signature: StripedGatherSignature


def _signature_digest(signatures: Sequence[StripedGatherSignature]) -> int:
    h = hashlib.blake2b(digest_size=8)
    for sig in signatures:
        h.update(
            struct.pack(
                "<qqqqq",
                sig.seq,
                sig.k_max,
                sig.row_bytes,
                sig.chunk_pages,
                sig.num_segments,
            )
        )
    return int.from_bytes(h.digest(), "little") & _DIGEST_MASK


class StripedLoadDrain:
    """FIFO of started striped loads, gathered at the forward entry.

    Every TP rank gathers its pending plans in FIFO order at the forward entry,
    after a CPU consensus on (count, last seq, signature digest). The only other
    gather site is the orphan drain with an already-agreed count (``allow_orphans``,
    PD-decode restores). ``StripedGatherFatal``, ``StripedLoadConsensusError`` and
    ``StripedLoadNotDrainedError`` must never be caught.
    """

    NUM_CONSENSUS_FIELDS = 6

    def __init__(
        self,
        executor: StripedGatherExecutor,
        cpu_group: Any,
        *,
        on_gathered: Optional[Any] = None,
        allow_orphans: bool = False,
    ):
        self.executor = executor
        self.cpu_group = cpu_group
        # Called with (plan, gather_done_event) after each plan's gather.
        self.on_gathered = on_gathered
        # Whether loads may legitimately have no consuming forward (PD decode).
        self.allow_orphans = allow_orphans
        # Set by bind_forward_entry when the TP worker registers this drain.
        self.bound_to_forward = False
        self._pending: deque[_PendingStripedLoad] = deque()
        self._next_seq = 0
        self.num_drains = 0
        self.num_consensus = 0
        self.num_plans_gathered = 0

    @property
    def placement(self) -> StripedHostPlacement:
        return self.executor.placement

    def reset(self) -> None:
        """Forget pending plans; every rank resets together (cache flush)."""
        for pending in self._pending:
            if pending.plan.gather_started:
                raise StripedGatherFatal(
                    f"Striped load plan {pending.plan.seq} reset mid-gather"
                )
        self._pending.clear()
        self._next_seq = 0

    def next_seq(self) -> int:
        return self._next_seq

    def bind_forward_entry(self) -> None:
        """Called when a TP worker registers this drain at its forward entry."""
        self.bound_to_forward = True

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def pending_seqs(self) -> list[int]:
        return [pending.plan.seq for pending in self._pending]

    def enqueue(self, plan: StripedLoadPlan, h2d_done_event: Any) -> None:
        """Queue a plan whose owned H2D is submitted; seq must be ``next_seq()``.

        Called on the stream that built the plan's device page tensors.
        """
        if not self.bound_to_forward:
            # Without the forward-entry drain every load would be consumed stale.
            raise StripedGatherError(
                "Striped load drain is not registered with the TP worker"
            )
        if plan.seq != self._next_seq:
            raise StripedGatherError(
                f"Striped load plan seq {plan.seq}, expected {self._next_seq}"
            )
        if plan.gather_started or plan.gather_done_event is not None:
            raise StripedGatherError(f"Striped load plan {plan.seq} already gathered")
        # Rank-identical checks; every rank raises together before any collective.
        signature = self.executor.signature(plan)
        self._pending.append(
            _PendingStripedLoad(
                plan, h2d_done_event, self._record_plan_ready(), signature
            )
        )
        self._next_seq += 1

    def _record_plan_ready(self) -> Any:
        """Event on the current stream after the plan's device tensors were built."""
        device = self.executor.device
        if device.type == "cpu":
            return None
        event = torch.get_device_module(device).Event()
        event.record()
        return event

    def consensus_fields(self) -> list[int]:
        """``[c, -c, s, -s, d, -d]`` for a MIN all-reduce (int64).

        ``c`` pending count, ``s`` last enqueued seq (-1 if none), ``d`` digest
        of the pending plans' gather signatures in FIFO order.
        """
        count = len(self._pending)
        last_seq = self._next_seq - 1
        digest = _signature_digest([pending.signature for pending in self._pending])
        return [count, -count, last_seq, -last_seq, digest, -digest]

    def check_reduced(self, reduced: Sequence[int]) -> int:
        """Raise unless every rank had the same pending loads; return the count."""
        values = [int(v) for v in reduced]
        if len(values) != self.NUM_CONSENSUS_FIELDS:
            raise StripedLoadConsensusError(
                f"expected {self.NUM_CONSENSUS_FIELDS} fields, got {len(values)}"
            )
        ranges = [(values[i], -values[i + 1]) for i in range(0, len(values), 2)]
        self.num_consensus += 1
        if any(lo != hi for lo, hi in ranges) or ranges[0][0] != len(self._pending):
            logger.error(
                "Striped load consensus broken on rank %d: count/seq/digest "
                "ranges %s; local pending seqs %s, next seq %d",
                self.placement.rank,
                ranges,
                self.pending_seqs(),
                self._next_seq,
            )
            raise StripedLoadConsensusError(
                f"Striped pending loads diverged across ranks: {ranges}"
            )
        return ranges[0][0]

    def agree(self) -> int:
        """All-reduce and check the pending-load consensus on the CPU group."""
        fields = torch.tensor(self.consensus_fields(), dtype=torch.int64)
        torch.distributed.all_reduce(
            fields, op=torch.distributed.ReduceOp.MIN, group=self.cpu_group
        )
        return self.check_reduced(fields.tolist())

    def default_stream(self) -> Any:
        """The stream gathers run on: the current (forward) stream; None on CPU."""
        device = self.executor.device
        if device.type == "cpu":
            return None
        return torch.get_device_module(device).current_stream()

    def drain(self, *, stream: Any = None, agreed: bool = False) -> int:
        """Gather every pending plan in FIFO order; return the number gathered.

        Collective on every TP rank. ``agreed`` skips the consensus all-reduce
        when the caller already checked ``consensus_fields`` in its own
        reduce. With nothing pending on any rank no gather is issued.
        """
        count = len(self._pending) if agreed else self.agree()
        if count == 0:
            return 0
        if stream is None:
            stream = self.default_stream()
        self.num_drains += 1
        while self._pending:
            pending = self._pending[0]
            wait = tuple(
                event
                for event in (pending.plan_ready_event, pending.h2d_done_event)
                if event is not None
            )
            done = self.executor.gather_plan(
                pending.plan, stream=stream, wait_events=wait
            )
            self._pending.popleft()
            self.num_plans_gathered += 1
            if self.on_gathered is not None:
                self.on_gathered(pending.plan, done)
        return count

    def drain_orphans(self, agreed: int, *, stream: Any = None) -> int:
        """Scheduler-check drain of ``agreed`` loads no forward has gathered.

        ``agreed`` is the pending count every rank agreed on in the scheduler's
        per-iteration reduce, so all ranks take the same branch. Without
        ``allow_orphans`` every rank raises before any gather.
        """
        if agreed == 0:
            return 0
        if not self.allow_orphans:
            logger.error(
                "Striped loads %s reached a scheduler check without a "
                "forward-entry gather on rank %d",
                self.pending_seqs(),
                self.placement.rank,
            )
            raise StripedLoadNotDrainedError(
                f"{agreed} striped loads were not gathered by a forward entry"
            )
        return self.drain(stream=stream, agreed=True)


def plan_gathered(plan: Optional[StripedLoadPlan]) -> bool:
    """Whether a load's gather is done (always True for a stock, unstriped load)."""
    if plan is None:
        return True
    event = plan.gather_done_event
    return event is not None and bool(event.query())
