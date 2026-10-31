"""Rank agreement for HiCache storage operations, run on the scheduler thread.

A storage backend (the cache controller's Python worker threads, or a native
controller) only performs I/O and records each operation's *local* outcome on
the operation. Once per scheduler tick the engine builds one integer row per
pending operation, hands the rows to an agreement strategy, and publishes the
agreed outcomes in submission order into the controller queue the stock
storage drain consumes. No collective runs on a worker thread and no worker
ever waits for a peer rank: a slow or failed backend call on one rank delays
that operation only, and nothing can strand the pipeline behind a
worker-thread collective that timed out.

The agreement strategy is a function over rows. The default,
``attention_group_min``, MIN-reduces every row across the attention group in
one collective per tick. A pipeline-parallel coordinator can instead report
rows to a leader and hand back the leader's decisions at its own checkpoints;
rows it has no decision for yet come back ``PENDING``. How a row is read off
an operation is a second small seam (``OperationRowSource``), so a native
backend plugs in its own operation type without touching this module or the
operation class. The engine's state machine and FIFO publication are the same
either way.

Row layout for a pending hit query (int64):

    [state, hit_pages, boundary(pool_0) .. boundary(pool_P-1), flag_0 .. flag_S]

Rows are rank-identical by construction because queries are submitted in
request order on every rank, so row k describes the same operation everywhere.
``state`` codes are ordered so that MIN yields the most conservative one.
``hit_pages`` is the folded KV hit; a boundary is the page count up to which a
pool is present, or NO_POOL_VERDICT when that rank's query gave the pool no
verdict (MIN propagates the absence, so the pool heals from the folded cut
without learning presence). Operations with a TRAILING_PAGES pool append one
restorable flag per prefix length (S = span pages): a trailing state can exist
at 512 and 1024 but not at 768, so the agreed hit is the longest prefix every
rank can restore, not the MIN of the longest hits.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Callable, Iterable, NamedTuple, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName

logger = logging.getLogger(__name__)

# Query states, MIN-ordered: the lowest state across ranks decides. A cancelled
# query is not a state of its own: the worker still records a local outcome for
# it (a miss when it skipped the query), so the operation keeps one writer per
# field and is published, agreed, in its turn.
QUERY_FATAL = 0  # the backend can no longer answer (dead worker): every rank raises
QUERY_PENDING = 1  # the local query is still running
QUERY_DONE = 2  # a local outcome exists; hit_pages may be 0
# A pool the local query gave no verdict for. Valid boundaries are >= 0, so
# MIN propagates the absence to every rank.
NO_POOL_VERDICT = -1
_POOLS = tuple(PoolName)

# Takes this rank's rows (one list of ints per pending operation, in FIFO
# order) and returns the agreed rows in the same order and widths.
AgreeRows = Callable[[list[list[int]]], list[list[int]]]


def attention_group_min(reduce_min: Callable[[torch.Tensor], None]) -> AgreeRows:
    """The default strategy: one MIN over every row across the attention group.
    ``reduce_min`` reduces an int64 tensor in place (the tree cache's
    attention-group all-reduce)."""

    def agree(rows: list[list[int]]) -> list[list[int]]:
        if not rows:
            return rows
        packed = torch.tensor([v for row in rows for v in row], dtype=torch.int64)
        reduce_min(packed)
        values = packed.tolist()
        agreed, offset = [], 0
        for row in rows:
            agreed.append(values[offset : offset + len(row)])
            offset += len(row)
        return agreed

    return agree


class LocalQueryOutcome(NamedTuple):
    """One rank's verdict for a hit query, before agreement."""

    hit_pages: int
    pool_boundaries: dict[PoolName, int]
    # Prefix lengths (in pages) a TRAILING_PAGES pool can restore from; None
    # means every length up to hit_pages (the contiguous-prefix contract).
    restorable_pages: Optional[tuple[int, ...]]


class LocalQueryOutcomeMixin:
    """Operation-side half of the contract: the query worker records its local
    outcome here and the scheduler-thread engine reads the status. The host
    class provides ``_lock``."""

    _local_query: Optional[LocalQueryOutcome] = None

    def publish_local_query(
        self,
        hit_pages: int,
        pool_boundaries: dict[PoolName, int],
        restorable_pages: Optional[Iterable[int]],
    ) -> None:
        outcome = LocalQueryOutcome(
            int(hit_pages),
            dict(pool_boundaries),
            None if restorable_pages is None else tuple(restorable_pages),
        )
        with self._lock:
            self._local_query = outcome

    def query_status(self) -> tuple[int, Optional[LocalQueryOutcome]]:
        with self._lock:
            if self._local_query is not None:
                return QUERY_DONE, self._local_query
            return QUERY_PENDING, None

    def local_restorable_pages(self) -> Optional[Iterable[int]]:
        """Prefix lengths this rank can restore a trailing-state pool from;
        None when the backend reports contiguous prefixes only."""
        return None


class OperationRowSource:
    """How the engine reads a backend's local outcomes off an operation.

    The default reads the ``LocalQueryOutcomeMixin`` fields the Python workers
    fill in. A native backend's adapter supplies its own source (polling the
    native operation) instead of growing the operation class.
    """

    def __init__(self, page_size: int):
        self.page_size = page_size

    def query_status(self, operation) -> tuple[int, Optional[LocalQueryOutcome]]:
        return operation.query_status()

    def trailing_flag_width(self, operation) -> int:
        """Restorable flags appended to the row: one per prefix length when a
        TRAILING_PAGES pool is queried, none otherwise. Rank-identical because
        the span is; a backend that reports pool boundaries only returns 0."""
        trailing = any(
            transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES
            for transfer in operation.pool_transfers or ()
        )
        if not trailing:
            return 0
        return len(operation.token_ids) // self.page_size + 1


class StorageAgreementEngine:
    """Per-tick rank agreement over pending storage operations.

    Owned by the tree cache and driven from ``check_hicache_events``; every rank
    calls ``tick`` at the same point of the same tick.
    """

    def __init__(
        self,
        page_size: int,
        agree: AgreeRows,
        backend_alive: Callable[[], bool],
        source: Optional[OperationRowSource] = None,
        backend_stopped: Optional[Callable[[], bool]] = None,
    ):
        self.page_size = page_size
        self._agree = agree
        self._backend_alive = backend_alive
        # A backend that is gone because storage was stopped on purpose (a
        # detach, finished or not) answers every pending query with a miss; a
        # backend that died is fatal.
        self._backend_stopped = backend_stopped or (lambda: False)
        self._source = OperationRowSource(page_size) if source is None else source
        # Submission order, identical on every rank: row k describes the same
        # operation everywhere, and the hit drain consumes the same order.
        self.queries: deque = deque()

    def register_query(self, operation) -> None:
        self.queries.append(operation)

    def pending_queries(self) -> int:
        return len(self.queries)

    def reset(self) -> None:
        self.queries.clear()

    def _query_row(self, operation, alive: bool, stopped: bool = False) -> list[int]:
        state, outcome = self._source.query_status(operation)
        flag_width = self._source.trailing_flag_width(operation)
        if not alive:
            if not stopped:
                state = QUERY_FATAL
            elif state != QUERY_DONE:
                # Storage was stopped: no worker will answer, so this rank
                # votes an evidence-free miss.
                outcome = LocalQueryOutcome(0, {}, ())
                state = QUERY_DONE
        if state != QUERY_DONE:
            return [state, 0, *([NO_POOL_VERDICT] * len(_POOLS)), *([0] * flag_width)]
        boundaries = [
            outcome.pool_boundaries.get(pool, NO_POOL_VERDICT) for pool in _POOLS
        ]
        flags: list[int] = []
        if flag_width:
            flags = [0] * flag_width
            flags[0] = 1  # the empty prefix is always restorable
            pages = outcome.restorable_pages
            if pages is None:
                pages = range(1, outcome.hit_pages + 1)
            for page in pages:
                if 0 < page <= outcome.hit_pages:
                    flags[page] = 1
        return [state, outcome.hit_pages, *boundaries, *flags]

    def tick(self, publish: Callable[[object], None]) -> int:
        """Agree the pending queries' outcomes and hand the agreed ones, in
        submission order, to ``publish``. Returns how many were published this
        tick."""
        if not self.queries:
            return 0
        alive = self._backend_alive()
        stopped = not alive and self._backend_stopped()
        rows = [self._query_row(op, alive, stopped) for op in self.queries]
        agreed = self._agree(rows)
        published = 0
        for operation, values in zip(list(self.queries), agreed):
            state = values[0]
            if state == QUERY_PENDING:
                break  # a rank is still querying; later rows keep their turn
            if state == QUERY_FATAL:
                raise RuntimeError(
                    "HiCache storage query worker is gone on a rank; request "
                    f"{operation.request_id} and every later query can no "
                    "longer be agreed across ranks"
                )
            self.queries.popleft()
            self._apply_agreed_query(operation, values)
            publish(operation)
            published += 1
        return published

    def _apply_agreed_query(self, operation, values: list[int]) -> None:
        hit_pages = values[1]
        boundaries = values[2 : 2 + len(_POOLS)]
        flags = values[2 + len(_POOLS) :]
        if flags:
            # The longest prefix every rank can restore a trailing state from.
            hit_pages = max(i for i, ok in enumerate(flags) if ok)
        verdicts = {
            pool: pages
            for pool, pages in zip(_POOLS, boundaries)
            if pages != NO_POOL_VERDICT
        }
        operation.storage_hit_count = hit_pages * self.page_size
        chain = operation.all_hash_values or []
        operation.hash_value = list(chain[:hit_pages])
        operation.query_pool_hit_pages = verdicts
