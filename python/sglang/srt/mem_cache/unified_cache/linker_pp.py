"""Pipeline tickets for direct-to-device external cache loads."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.distributed as dist

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle, DecLockRefParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.srt.session.streaming_session import is_virtual_node
from sglang.srt.utils import broadcast_pyobj

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        UnifiedCacheLinkerWrapper,
    )

logger = logging.getLogger(__name__)


class _PPTicket(NamedTuple):
    handle: CacheRequestHandle
    key: RadixKey
    start: int
    last_hash: str | None

    def hashes(self, page: int, end: int | None = None) -> list[str]:
        return get_storage_hash_str(
            self.key[self.start : end], self.last_hash, page_size=page
        )


@dataclass
class _LocalPPTicket:
    req: Req | None = None
    ticket: _PPTicket | None = None
    pin: tuple[NodeId, DecLockRefParams] | None = None
    discard: bool = False
    wait_started: float | None = None
    # Worker-to-scheduler query bound, scheduler-to-worker reserved capacity,
    # then the agreed boundary and the actual GET result.
    lookup: int | None = None
    capacity: int | None = None
    end: Future[int] = field(default_factory=Future)
    read: Future[bool] | bool | None = None
    landed: Future[bool] = field(default_factory=Future)


class _LinkerPPTickets:
    """Agree on storage boundaries off-thread. Only the scheduler owns device rows."""

    def __init__(self, linker: UnifiedCacheLinkerWrapper):
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        self.linker = linker
        self.cache = cache = linker.cache
        self.page = cache.page_size
        self.prefetch = linker.load_mode == "prefetch"
        self.arrival = threading.Condition()
        self.pending: dict[CacheRequestHandle, _LocalPPTicket] = {}
        self.layer_anchors: dict[
            CacheRequestHandle, tuple[NodeId, DecLockRefParams]
        ] = {}
        self.stopping = False
        self.failure: BaseException | None = None
        self.commands = SimpleQueue()
        self.completions = SimpleQueue()
        self.query_groups = []
        self.ack_groups = []
        self.command_thread = self.ack_thread = None

        pp_ranks = dist.get_process_group_ranks(cache.pp_group)
        attention = [
            g for g in (cache.attn_cp_group, cache.attn_tp_group) if g is not None
        ]
        rank_sets = []
        for group in attention or [cache.tp_group]:
            if group is not None:
                ranks = dist.get_process_group_ranks(group)
                if len(ranks) > 1 and ranks != pp_ranks and ranks not in rank_sets:
                    rank_sets.append(ranks)
        rank_sets.append(pp_ranks)
        try:
            for groups in (
                [self.query_groups, self.ack_groups]
                if self.prefetch
                else [self.query_groups]
            ):
                for ranks in rank_sets:
                    groups.append(
                        create_custom_parallel_group(group_ranks=ranks, backend="gloo")
                    )
            if self.prefetch:
                self.ack_thread = self._start(
                    self._complete_reads, "linker-pp-completion"
                )
            self.command_thread = self._start(self._prepare_tickets, "linker-pp-ticket")
        except BaseException:
            self.close()
            raise

    def _start(self, target, name):
        def run():
            try:
                target()
            except BaseException as error:
                self.failure = error
                logger.exception("%s exited", name)

        thread = threading.Thread(target=run, name=name, daemon=True)
        thread.start()
        return thread

    def _register(self, handle, state):
        with self.arrival:
            self.pending[handle] = state
            self.arrival.notify_all()

    def bind(self, req: Req, expected: bool, *, retracted: bool = False) -> bool:
        handle = req.cache_request_handle
        if handle in self.pending or handle in self.layer_anchors:
            return True
        req.external_cache_hit_length = 0
        req.init_next_round_input(self.cache, cow_mamba=False)
        source = self.cache.pp_rank == 0
        if not source and not expected:
            return False
        key = self.linker._prefetch_key(req)
        valid = bool(len(key)) and req.positional_embed_overrides is None
        if source and not retracted and not valid:
            return False
        start = req.prefix_len
        valid = (
            valid
            and not is_virtual_node(req.last_node)
            and start <= len(key)
            and start % self.page == 0
        )
        prior = self.cache.get_last_hash_value(req.last_node) if valid else None
        pin = None
        if valid and (not start or prior):
            pin = (
                req.last_node,
                self.cache.inc_lock_ref(req.last_node).to_dec_params(),
            )
        ticket = _PPTicket(handle, key, start, prior)
        self._register(handle, _LocalPPTicket(req, ticket, pin))
        if source:
            self.commands.put(ticket)
        return True

    def skip_unbound(self, rid: str) -> None:
        handle = CacheRequestHandle(rid, attempt_id=0)
        if handle not in self.pending:
            self._register(handle, _LocalPPTicket(discard=True))

    def _lookup(self, ticket: _PPTicket) -> list[int]:
        if ticket.start >= len(ticket.key) or (ticket.start and not ticket.last_hash):
            return []
        try:
            return self.linker._lookup_prefetch(
                ticket.handle.rid, ticket.hashes(self.page)
            )
        except Exception:
            logger.exception("PP linker key lookup failed for %s", ticket.handle)
            return []

    def _prepare_tickets(self):
        command_group = self.query_groups[-1]
        rank = dist.get_rank()
        src = dist.get_process_group_ranks(command_group)[0]
        while True:
            commands = []
            if rank == src:
                try:
                    commands = [self.commands.get(timeout=60)]
                except Empty:
                    pass  # Complete idle broadcasts before the group times out.
            commands = broadcast_pyobj(commands, rank, command_group, src=src)
            if not commands:
                continue
            ticket = commands[0]
            if ticket is None:
                return
            with self.arrival:
                arrived = ticket.handle in self.pending or self.stopping
            # The first query can run before the follower receives its Req.
            hits = None if arrived else self._lookup(ticket)
            with self.arrival:
                self.arrival.wait_for(
                    lambda: ticket.handle in self.pending or self.stopping
                )
                state = self.pending.get(ticket.handle)
            mask = torch.zeros(len(ticket.key) // self.page + 1, dtype=torch.int)
            if state is not None and state.pin is not None and not self.stopping:
                local = state.ticket
                if hits is None or (local.start, local.last_hash) != (
                    ticket.start,
                    ticket.last_hash,
                ):
                    hits = self._lookup(local)
                start = local.start // self.page
                pages = [start] + [start + p for p in hits]
                state.lookup = max(pages)
                if self.prefetch:
                    with self.arrival:
                        self.arrival.wait_for(
                            lambda: state.capacity is not None or self.stopping
                        )
                    pages = [
                        p
                        for p in pages
                        if state.capacity is not None and p <= state.capacity
                    ]
                mask[pages] = 1
            elif state is not None:
                state.lookup = 0
            for group in self.query_groups:
                dist.all_reduce(mask, op=dist.ReduceOp.MIN, group=group)
            common = mask.nonzero()
            end = int(common[-1].item()) * self.page if common.numel() else 0
            if state is not None:
                state.end.set_result(end)
                if self.prefetch:
                    self.completions.put((state, end))

    def _complete_reads(self):
        # This FIFO has independent groups so an unfinished GET does not stop
        # the next ticket from being queried, reserved, and submitted.
        while (item := self.completions.get()) is not None:
            state, end = item
            landed = False
            if end:
                with self.arrival:
                    self.arrival.wait_for(
                        lambda: state.read is not None or self.stopping
                    )
                read = state.read
                try:
                    landed = read.result() if isinstance(read, Future) else read
                except Exception:
                    landed = False
                ack = torch.tensor(int(bool(landed) and not state.discard))
                for group in self.ack_groups:
                    dist.all_reduce(ack, op=dist.ReduceOp.MIN, group=group)
                landed = bool(ack.item())
            state.landed.set_result(landed)

    def _reserve(self, state, upper):
        if state.discard or state.pin is None:
            return -1
        ticket = state.ticket
        start = ticket.start // self.page
        if upper <= start:
            return start
        try:
            hashes = ticket.hashes(self.page, upper * self.page)
            return self.linker.reserve_pp_prefetch(
                ticket.handle, state.pin[0], ticket.key, ticket.start, hashes
            )
        except Exception:
            logger.exception("PP linker preparation failed for %s", ticket.handle)
            return start

    def _ready_word(self, state):
        if not self.prefetch:
            return int(state.end.done())
        if state.capacity is None:
            return 0 if state.lookup is None else state.lookup + 1
        return int((state.end if state.read is None else state.landed).done())

    def _keep_waiting(self, state, now):
        if state.discard:
            return False
        if state.wait_started is None or (
            state.read is not None and state.landed.done()
        ):
            return True
        if self.cache.prefetch_stop_policy == "best_effort":
            return False
        pages = (len(state.ticket.key) - state.ticket.start) // self.page
        budget = (
            self.cache.prefetch_timeout_base
            + pages * self.cache.prefetch_timeout_per_page
        )
        return now - state.wait_started <= budget

    def poll(self) -> None:
        if self.failure is not None:
            raise RuntimeError("PP linker ticket worker failed") from self.failure
        if not self.pending:
            return
        # A per-request admission poll may be skipped on another stage.
        # Communicate once per tick, in request-arrival order, including misses.
        items = list(self.pending.items())
        now = time.perf_counter()
        ready = torch.tensor(
            [(self._ready_word(s), self._keep_waiting(s, now)) for _, s in items],
            dtype=torch.int,
        )
        self.cache._all_reduce(ready, torch.distributed.ReduceOp.MIN)
        for (flag, keep), (handle, state) in zip(ready.tolist(), items):
            if not keep and not state.discard:
                self.release(handle)
            if not flag:
                continue
            if self.prefetch and state.capacity is None:
                capacity = self._reserve(state, flag - 1)
                with self.arrival:
                    state.capacity = capacity
                    self.arrival.notify_all()
            elif self.prefetch and state.read is None:
                end = state.end.result()
                read = self.linker.submit_pp_prefetch(
                    handle, 0 if state.discard else end
                )
                local = state.pin is not None and state.ticket.start == end
                with self.arrival:
                    state.read = read or (local and not state.discard)
                    self.arrival.notify_all()
            else:
                if self.prefetch:
                    self.linker._finish_prefetch(
                        handle, publish=state.landed.result() and not state.discard
                    )
                elif (end := state.end.result()) and not state.discard:
                    state.req.external_cache_hit_length = end
                    self.layer_anchors[handle] = state.pin
                if state.pin is not None and handle not in self.layer_anchors:
                    self.cache.dec_lock_ref(*state.pin)
                with self.arrival:
                    del self.pending[handle]

    def can_admit(self, handle: CacheRequestHandle) -> bool:
        state = self.pending.get(handle)
        if state is None or state.discard:
            return True
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
        pin = self.layer_anchors.pop(handle, None)
        if pin is not None:
            self.cache.dec_lock_ref(*pin)

    def close(self) -> None:
        with self.arrival:
            if self.stopping:
                return
            self.stopping = True
            self.arrival.notify_all()
        if self.cache.pp_rank == 0:
            self.commands.put(None)
        # The producer enqueues the last ACK before the completion worker stops.
        if self.command_thread is not None:
            self.command_thread.join()
        if self.ack_thread is not None:
            self.completions.put(None)
            self.ack_thread.join()
        for groups in (self.ack_groups, self.query_groups):
            for group in reversed(groups):
                dist.destroy_process_group(group)
        for state in self.pending.values():
            if self.failure is not None and isinstance(state.read, Future):
                # A failed ACK worker may have left later GETs in flight.
                try:
                    state.read.result()
                except Exception:
                    pass
            if state.pin is not None:
                self.cache.dec_lock_ref(*state.pin)
        for pin in self.layer_anchors.values():
            self.cache.dec_lock_ref(*pin)
        self.layer_anchors.clear()
        self.pending.clear()
        if self.failure is not None:
            raise RuntimeError("PP linker ticket worker failed") from self.failure
