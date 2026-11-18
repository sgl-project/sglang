"""Pipeline tickets for direct-to-device external cache loads."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle, DecLockRefParams
from sglang.srt.mem_cache.pp_ticket_worker import PPTicketWorker
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import LinkerTransferPhase
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.srt.session.streaming_session import is_virtual_node

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        UnifiedCacheLinkerWrapper,
    )

logger = logging.getLogger(__name__)


class _Ticket(NamedTuple):
    handle: CacheRequestHandle
    key: RadixKey
    start: int
    last_hash: str | None


class _Anchor(NamedTuple):
    last_node: NodeId
    start: int
    last_hash: str | None
    lock: DecLockRefParams


@dataclass
class _LocalTicket:
    # Only immutable query arguments and results cross the worker boundary.
    anchor: Future[_Anchor | None] = field(default_factory=Future)
    lookup: Future[int] = field(default_factory=Future)
    vote: Future[int] = field(default_factory=Future)
    end: Future[int] = field(default_factory=Future)
    submitted: Future[Future[bool] | bool] = field(default_factory=Future)
    completed: Future[bool] = field(default_factory=Future)
    # Scheduler-owned after request relay.
    req: Req | None = None
    ticket: _Ticket | None = None
    discard: bool = False
    wait_started: float | None = None


class LinkerPPTickets:
    """One ticket FIFO per pipeline lane. Only the scheduler touches the tree."""

    def __init__(self, linker: UnifiedCacheLinkerWrapper):
        self.linker = linker
        self.cache = cache = linker.cache
        self.page = cache.page_size
        self.prefetch = linker.load_mode == "prefetch"
        self.lock = threading.Lock()
        self.stopping = False
        self.states: dict[CacheRequestHandle, _LocalTicket] = {}
        # Request arrival order is identical across stages. A broadcast may
        # precede arrival on a follower; it must not change scheduler order.
        self.pending: dict[CacheRequestHandle, _LocalTicket] = {}
        self.layer_anchors: dict[CacheRequestHandle, _Anchor] = {}
        self.worker = PPTicketWorker(
            pp_group=cache.pp_group,
            tp_group=cache.tp_group,
            attn_cp_group=cache.attn_cp_group,
            attn_tp_group=cache.attn_tp_group,
            handle_ticket=self._process,
            finish_ticket=self._finish_read if self.prefetch else None,
        )

    def bind(self, req: Req, expected: bool, *, retracted: bool = False) -> bool:
        source = self.cache.pp_rank == 0
        handle = req.cache_request_handle
        if handle in self.pending or handle in self.layer_anchors:
            return True
        req.external_cache_hit_length = 0
        req.init_next_round_input(self.cache, cow_mamba=False)
        if not source and not expected:
            return False
        key = RadixKey(
            req.full_untruncated_fill_ids,
            req.extra_key,
            is_bigram=self.cache.tree_core.is_eagle,
            cache_salt=req.cache_salt,
            limit=req._compute_max_prefix_len(len(req.full_untruncated_fill_ids)),
        ).page_aligned(self.page)
        valid = (
            len(key) > 0
            and req.positional_embed_overrides is None
            and not is_virtual_node(req.last_node)
        )
        # A retraction is enqueued locally on every stage without a request
        # relay. Always issue its ticket, including the empty vote.
        if (
            source
            and not retracted
            and (not len(key) or req.positional_embed_overrides is not None)
        ):
            return False
        start = req.prefix_len
        last_hash = self.cache.get_last_hash_value(req.last_node) if valid else None
        anchor = None
        if (
            valid
            and start <= len(key)
            and start % self.page == 0
            and (not start or last_hash)
        ):
            anchor = _Anchor(
                req.last_node,
                start,
                last_hash,
                self.cache.inc_lock_ref(req.last_node).to_dec_params(),
            )
        ticket = _Ticket(handle, key, start, last_hash)
        with self.lock:
            state = self.states.setdefault(handle, _LocalTicket())
        state.req = req
        state.ticket = ticket
        state.anchor.set_result(anchor)
        self.pending[handle] = state
        if source:
            self.worker.submit(ticket)
        return True

    def _lookup(self, ticket: _Ticket, start: int, last_hash: str | None) -> list[int]:
        if start >= len(ticket.key) or (start and not last_hash):
            return []
        try:
            hashes = get_storage_hash_str(
                ticket.key[start:], last_hash, page_size=self.page
            )
            transfers = [
                component.build_external_linker_transfer(
                    LinkerTransferPhase.LOOKUP, None, hashes
                )
                for component in self.linker._components
            ]
            if any(t is None for t in transfers):
                return []
            hits = self.linker.cache_linker.lookup(ticket.handle.rid, transfers)
            return [p for p in hits if 0 < p <= len(hashes)]
        except Exception:
            logger.exception("PP linker lookup failed for %s", ticket.handle)
            return []

    def skip_unbound(self, rid: str) -> None:
        handle = CacheRequestHandle(rid, attempt_id=0)
        if handle in self.pending:
            return
        with self.lock:
            state = self.states.setdefault(handle, _LocalTicket())
        state.discard = True
        state.anchor.set_result(None)
        self.pending[handle] = state

    def _process(self, ticket: _Ticket) -> None:
        with self.lock:
            state = self.states.setdefault(ticket.handle, _LocalTicket())
            if self.stopping:
                self._wake_for_stop(state)
        # Start a follower's storage query before the request relay arrives.
        # A different local L1 anchor needs its own uncached tail checked.
        hint = (ticket.start, ticket.last_hash)
        hits = None if state.anchor.done() else self._lookup(ticket, *hint)
        anchor = state.anchor.result()
        local = set()
        if anchor is not None:
            if hits is None or (anchor.start, anchor.last_hash) != hint:
                hits = self._lookup(ticket, anchor.start, anchor.last_hash)
            start_page = anchor.start // self.page
            local.add(start_page)
            local.update(start_page + p for p in hits)
        state.lookup.set_result(max(local, default=0))
        mask = torch.zeros(len(ticket.key) // self.page + 1, dtype=torch.int)
        capacity = state.vote.result() if self.prefetch else len(mask) - 1
        for p in local:
            if p > capacity:
                continue
            mask[p] = 1
        common = self.worker.reduce_min(mask).nonzero()
        end = int(common[-1].item()) * self.page if common.numel() else 0
        state.end.set_result(end)

    def _finish_read(self, ticket: _Ticket) -> None:
        with self.lock:
            state = self.states[ticket.handle]
        if state.end.result():
            read = state.submitted.result()
            try:
                landed = read.result() if isinstance(read, Future) else read
            except Exception:
                landed = False
            ack = torch.tensor([bool(landed) and not state.discard], dtype=torch.int)
            landed = bool(self.worker.reduce_completion(ack).item())
        else:
            landed = False
        state.completed.set_result(landed)

    def _reserve(self, ticket: _Ticket, anchor: _Anchor, upper: int) -> int:
        start_page = anchor.start // self.page
        if upper <= start_page:
            return start_page
        # PP0's bound travels with readiness. Reserve in the same round
        # without waiting for a follower's local storage query.
        try:
            hashes = get_storage_hash_str(
                ticket.key[anchor.start : upper * self.page],
                anchor.last_hash,
                page_size=self.page,
            )
            return self.linker.reserve_pp_prefetch(
                ticket.handle, anchor.last_node, ticket.key, anchor.start, hashes
            )
        except Exception:
            logger.exception("PP linker preparation failed for %s", ticket.handle)
            return start_page

    def poll(self) -> None:
        self.worker.check_alive()
        if not self.pending:
            return
        # Apply reservations, trims and inserts in the same scheduler round.
        # This is HiCache's one-way ready relay. The only cross-stage MINs
        # run on the worker: one for hit + allocation and one after the GET.
        items = list(self.pending.items())
        now = time.perf_counter()
        ready = torch.tensor(
            [
                (self._ready_word(state), self._keep_waiting(state, now))
                for _, state in items
            ],
            dtype=torch.int,
        )
        self.cache._all_reduce(ready, torch.distributed.ReduceOp.MIN)
        for (flag, keep_waiting), (handle, state) in zip(ready.tolist(), items):
            if not keep_waiting and not state.discard:
                self.release(handle)
            if not flag:
                continue
            anchor = state.anchor.result()
            if self.prefetch and not state.vote.done():
                capacity = -1
                if anchor is not None and not state.discard:
                    capacity = self._reserve(state.ticket, anchor, flag - 1)
                state.vote.set_result(capacity)
            elif self.prefetch and not state.submitted.done():
                end = state.end.result()
                read = self.linker.submit_pp_prefetch(
                    handle, 0 if state.discard else end
                )
                already_local = anchor is not None and anchor.start == end
                state.submitted.set_result(
                    read or (already_local and not state.discard)
                )
            else:
                end = state.end.result()
                if self.prefetch:
                    self.linker._finish_prefetch(
                        handle, publish=state.completed.result() and not state.discard
                    )
                elif end and not state.discard:
                    state.req.external_cache_hit_length = end
                    self.layer_anchors[handle] = anchor
                if anchor is not None and handle not in self.layer_anchors:
                    self.cache.dec_lock_ref(anchor.last_node, anchor.lock)
                del self.pending[handle]
                with self.lock:
                    del self.states[handle]

    def _ready_word(self, state: _LocalTicket) -> int:
        if not self.prefetch:
            return int(state.end.done())
        if not state.vote.done():
            # Zero means pending; one means a completed local miss. The source
            # TP ranks choose an upper bound once and relay it down the pipe.
            return 1 + state.lookup.result() if state.lookup.done() else 0
        return int((state.completed if state.submitted.done() else state.end).done())

    def _keep_waiting(self, state: _LocalTicket, now: float) -> bool:
        if state.discard:
            return False
        if state.wait_started is None or (
            state.submitted.done() and state.completed.done()
        ):
            return True
        if self.cache.prefetch_stop_policy == "best_effort":
            return False
        pages = (len(state.ticket.key) - state.ticket.start) // self.page
        deadline = (
            self.cache.prefetch_timeout_base
            + pages * self.cache.prefetch_timeout_per_page
        )
        return now - state.wait_started <= deadline

    def can_admit(self, handle: CacheRequestHandle) -> bool:
        state = self.pending.get(handle)
        if state is None or state.discard:
            return True
        # Stages can leave the admission loop at different requests. Only
        # the ordered scheduler tick may communicate a stop decision.
        if (
            self.cache.pp_rank == 0
            and self.prefetch
            and self.cache.prefetch_stop_policy != "wait_complete"
            and state.wait_started is None
        ):
            state.wait_started = time.perf_counter()
        return False

    def release(self, handle: CacheRequestHandle) -> None:
        state = self.pending.get(handle)
        if state is not None:
            state.discard = True
            self.linker._cancel_prefetch(handle)
        anchor = self.layer_anchors.pop(handle, None)
        if anchor is not None:
            self.cache.dec_lock_ref(anchor.last_node, anchor.lock)

    @staticmethod
    def _wake_for_stop(state: _LocalTicket) -> None:
        for future, fallback in (
            (state.anchor, None),
            (state.vote, -1),
            (state.submitted, False),
        ):
            if not future.done():
                future.set_result(fallback)

    def close(self) -> None:
        with self.lock:
            if self.stopping:
                return
            self.stopping = True
            for state in self.states.values():
                self._wake_for_stop(state)
        self.worker.close()
        for state in self.pending.values():
            anchor = state.anchor.result()
            if anchor is not None:
                self.cache.dec_lock_ref(anchor.last_node, anchor.lock)
        for anchor in self.layer_anchors.values():
            self.cache.dec_lock_ref(anchor.last_node, anchor.lock)
        self.pending.clear()
        self.layer_anchors.clear()
        self.states.clear()
