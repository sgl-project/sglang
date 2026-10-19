"""One readiness-first gate in front of native PD prefill selection.

Native queues own requests, the cache owns Prefetch, and PrefillAdder owns
batch construction/H2D. This gate owns only the state that crosses PP beats:
PENDING -> PREPARING -> CERTIFIED -> native batch (or PENDING on NACK).
Every policy uses the same ordered hop-by-hop messages. No worker, PP collective,
second cancellation stream, early bootstrap, or private GET retry is added.

Ownership: the mixin transports AdmissionUpdate; this gate holds promises until
native selection takes ownership. Readiness holds no resources. PREPARING and
CERTIFIED may hold a prefix lease and capacity credit. Native retirement releases
only those additions; it remains the sole request/cancellation lifecycle.
"""

import logging
from dataclasses import dataclass, replace
from enum import Enum, auto
from http import HTTPStatus

from sglang.srt.mem_cache.pp_cache_prefix import CachePrefixLease


@dataclass(frozen=True)
class RequestVersion:
    rid: str
    attempt_id: int
    input_tokens: int
    page_size: int
    # Assigned once by PP0, not per poll. Fence a later reuse of RID/attempt.
    sequence: int = 0

    @property
    def max_prefix_tokens(self):
        return (self.input_tokens - 1) // self.page_size * self.page_size


@dataclass(frozen=True)
class Prepare:
    request: RequestVersion
    generation: int
    # Forward: running MIN. Return/commit: final MIN, or None for NACK.
    cap: int | None


@dataclass(frozen=True)
class AdmissionUpdate:
    """Gate-owned fields only; PP framing and bootstrap belong to the mixin."""

    watch: tuple[RequestVersion, ...] = ()
    ready: tuple[tuple[RequestVersion, int], ...] = ()
    prepare: tuple[Prepare, ...] = ()
    commit: tuple[Prepare, ...] = ()
    layout: tuple[tuple[str, int, int], ...] = ()


class Phase(Enum):
    PENDING = auto()
    PREPARING = auto()
    CERTIFIED = auto()


@dataclass
class _Entry:
    key: RequestVersion
    req: object = None
    phase: Phase = Phase.PENDING
    ready_mask: int = 0
    generation: int = 0
    credit: int = 0
    lease: object = None
    cap: int | None = None
    storage_span: tuple | None = None


@dataclass
class PrefillAdmissionSelection:
    """A view, not another queue or allocation owner.

    Credits for these candidates become available to the native batch budget.
    Credits for other prepares stay excluded. Native selection cannot consume
    more than each candidate's full-prompt credit; unselected credits never
    need to be freed/reallocated or restored.
    """

    entries: dict
    reserved_tokens: int = 0
    expected_layout: tuple | None = None

    @property
    def candidates(self):
        return [entry.req for entry in self.entries.values()]

    def prepare_request(self, req, cache):
        entry = self.entries[req.rid]
        req.init_next_round_input(cache, max_cache_prefix_tokens=entry.cap)
        span = entry.storage_span
        req.storage_hit_length = span[1] - span[0] if span else 0
        req.storage_hit_start = span[0] if span else None
        req.host_hit_is_storage = False
        return entry.cap


def batch_layout(batch):
    return (
        ()
        if batch is None
        else tuple((r.rid, r.prefix_len, r.extend_end) for r in batch.reqs)
    )


class PPAdmissionGate:
    def __init__(self, scheduler, *, parallel):
        # Queues, available capacity and the native selector change each beat,
        # so the gate needs the live scheduler. It owns only its admission state;
        # request and cache mutation stays in the scheduler's native operations.
        self.s = scheduler
        self.rank = parallel.pp_rank
        self.all_ready = (1 << parallel.pp_size) - 1
        self.entries = {}
        self.sequence = 0
        self.window = 8  # Only PREPARING/CERTIFIED records hold capacity/leases.
        self.incoming = AdmissionUpdate()
        self.commits = []
        logging.getLogger(__name__).info(
            "PP async admission active: PP=%d stage=%d mode=readiness-first",
            parallel.pp_size,
            self.rank,
        )

    def receive_forward(self, update: AdmissionUpdate) -> None:
        self.incoming = update
        for result in update.commit:
            self._commit(result)
        if update.watch:
            # Request relay precedes WATCH. Bind its native object now, including
            # requests still in bootstrap; this does not make them READY.
            queued = {
                req.rid: req
                for req in (
                    *self.s.disagg_prefill_bootstrap_queue.queue,
                    *self.s.waiting_queue,
                )
            }
            for key in update.watch:
                self.entries[key.rid] = _Entry(
                    key, queued.get(key.rid), generation=key.sequence
                )

    def consume_return(self, update: AdmissionUpdate) -> None:
        if self.rank == 0:
            for key, mask in update.ready:
                entry = self.entries.get(key.rid)
                if entry is not None and entry.key == key:
                    entry.ready_mask |= mask
            for result in update.prepare:
                if self._commit(result):
                    self.commits.append(result)

    def has_pending_work(self) -> bool:
        """Include readiness-only waits, not just resource-owning prepares.

        Used by the native administrative-idle check, never to skip housekeeping.
        Even when a paused engine ignores waiting_queue, live control records
        must settle before destructive cache operations.
        """
        # Native cancellation can remove an entry before its COMMIT is sent.
        return bool(self.entries or self.commits)

    def release(self, req):
        """Native retirement/abort calls this before releasing cache ownership."""
        entry = self.entries.get(req.rid)
        if entry is not None and (
            entry.req is req
            or (
                entry.req is None
                and entry.key.attempt_id == req.cache_request_handle.attempt_id
            )
        ):
            self._release_prepare(entry)
            self.s.tree_cache.release_admission_storage_span(req.cache_request_handle)
            del self.entries[req.rid]

    def reject_unsupported(self, req):
        """Ingress-only check, before grammar compilation or PD bootstrap."""
        params = req.sampling_params
        if not (
            any(
                getattr(params, name, None) is not None
                for name in ("json_schema", "regex", "ebnf", "structural_tag")
            )
            or req.grammar is not None
            or req.beam_group is not None
            or req.session is not None
            or req.session_id is not None
        ):
            return False
        from sglang.srt.disaggregation.utils import prepare_abort

        self.s.release_aborted_prefill_waiting_req(req)
        prepare_abort(
            req,
            "PP cache admission does not support structured output, beam groups or sessions.",
            status_code=HTTPStatus.BAD_REQUEST,
        )
        self.s.retire_unadmitted_request(req)
        return True

    def _watch_waiting(self):
        watches = []
        for req in self.s.waiting_queue:
            if req.rid not in self.entries:
                self.sequence += 1
                key = RequestVersion(
                    req.rid,
                    req.cache_request_handle.attempt_id,
                    len(req.origin_input_ids),
                    self.s.page_size,
                    self.sequence,
                )
                self.entries[req.rid] = _Entry(key, req, generation=key.sequence)
                watches.append(key)
        return tuple(watches)

    def _report_readiness(self):
        # WATCH binds the native object; bootstrap alone moves it into waiting.
        # Native retirement releases its entry before changing the cache attempt.
        # Do not revalidate that lifecycle on every beat or invent a retry path.
        waiting = set(self.s.waiting_queue)
        pending = [
            entry
            for entry in self.entries.values()
            if not entry.ready_mask & (1 << self.rank) and entry.req in waiting
        ]
        flags = self.s.tree_cache.prefetch_ready_to_finalize(
            [entry.req.cache_request_handle for entry in pending]
        )
        updates = dict(self.incoming.ready) if self.rank else {}
        for entry, ready in zip(pending, flags):
            entry.req.time_stats.observe_storage_prefetch_ready(ready)
            if ready:
                # Latched stop permission, not a prefix/capacity promise.
                # Later policy changes affect requests not yet reported READY.
                entry.ready_mask |= 1 << self.rank
                updates[entry.key] = updates.get(entry.key, 0) | (1 << self.rank)
        return tuple(updates.items())

    def _start_prepares(self):
        available = self.window - sum(
            e.phase != Phase.PENDING for e in self.entries.values()
        )
        if available <= 0:
            return ()
        # Do not rotate the native queue. Give untried/older capacity waiters a
        # turn using the generation already needed to fence prepare replies.
        candidates = sorted(
            (
                e
                for e in self.entries.values()
                if e.phase == Phase.PENDING and e.ready_mask == self.all_ready
            ),
            key=lambda e: e.generation,
        )
        if not candidates:
            return ()
        entry = candidates[0]  # Stream one prepare per beat.
        self.sequence += 1
        return (Prepare(entry.key, self.sequence, entry.key.max_prefix_tokens),)

    def _reserve_capacity(self, req):
        """Claim scheduler budget, not placeholder GPU pages/request rows.

        Only certified requests and the continuing chunk can allocate in this
        selected PD-prefill path. Real free capacity backs every outstanding
        credit; native PrefillBudget excludes other prepares' credits.
        """
        s = self.s
        held = sum(e.credit for e in self.entries.values())
        rows = sum(e.credit > 0 for e in self.entries.values())
        if s.req_to_token_pool.available_size() <= rows:
            return 0
        page = s.page_size
        need = (
            -(
                -(
                    len(req.origin_input_ids)
                    + req.sampling_params.max_new_tokens
                    + 2 * page
                )
                // page
            )
            * page
        )
        runway = (
            len(s.chunked_req.origin_input_ids) - s.chunked_req.extend_end + 2 * page
            if s.chunked_req is not None
            else 0
        )
        required = held + need + runway
        allocator = s.token_to_kv_pool_allocator
        if allocator.available_size() < required:
            if (
                allocator.available_size() + s.tree_cache.full_evictable_size()
                < required
            ):
                return 0
            from sglang.srt.mem_cache.base_prefix_cache import EvictParams

            s.tree_cache.evict_for_alloc(
                EvictParams(num_tokens=required - allocator.available_size())
            )
        return need if allocator.available_size() >= required else 0

    def _prepare(self, proposals):
        results = []
        for proposal in proposals:
            entry = self.entries.get(proposal.request.rid)
            cap = None
            if entry is not None and entry.key == proposal.request:
                entry.phase = Phase.PREPARING
                entry.generation = proposal.generation
                if proposal.cap is not None:
                    cache = self.s.tree_cache
                    cache.finalize_prefetch(entry.req.cache_request_handle)
                    entry.credit = self._reserve_capacity(entry.req)
                    if entry.credit:
                        # Eviction above may shorten the prefix; match after it.
                        entry.lease = CachePrefixLease.acquire(
                            cache, entry.req, entry.key.max_prefix_tokens
                        )
                        entry.storage_span = cache.get_admission_storage_span(
                            entry.req.cache_request_handle
                        )
                        cap = min(proposal.cap, entry.lease.cap)
            results.append(replace(proposal, cap=cap))
        return tuple(results)

    @staticmethod
    def _release_prepare(entry):
        if entry.lease is not None:
            entry.lease.release()
            entry.lease = None
        entry.credit = 0
        entry.cap = None

    def _commit(self, result: Prepare) -> bool:
        """Settle a live prepare once; PP0 relays only accepted return results."""
        entry = self.entries.get(result.request.rid)
        # Return and forward streams can meet a cancelled/retried request.
        # Sole settlement fence, not repeated payload validation.
        if (
            entry is None
            or entry.key != result.request
            or entry.generation != result.generation
            or entry.phase != Phase.PREPARING
        ):
            return False
        if result.cap is None:
            self._release_prepare(entry)
            entry.phase = Phase.PENDING
        else:
            entry.lease = entry.lease.restrict(result.cap)
            entry.cap = result.cap
            entry.phase = Phase.CERTIFIED
        return True

    def prepare_selection(self):
        if self.rank == 0:
            entries = {
                req.rid: self.entries[req.rid]
                for req in self.s.waiting_queue
                if req.rid in self.entries
                and self.entries[req.rid].phase == Phase.CERTIFIED
            }
            layout = None
        else:
            layout = self.incoming.layout
            entries = {
                rid: self.entries[rid] for rid, *_ in layout if rid in self.entries
            }
        withheld = sum(
            e.credit for rid, e in self.entries.items() if rid not in entries
        )
        return PrefillAdmissionSelection(entries, withheld, layout)

    def select_batch(self, running_batch):
        selection = self.prepare_selection()
        plan = self.s.get_new_batch_prefill(running_batch, admission=selection)
        return plan, self.finish_selection(plan.batch_to_run)

    def finish_selection(self, batch) -> AdmissionUpdate:
        layout = batch_layout(batch)
        for req in () if batch is None else batch.reqs:
            self.release(req)  # Native allocation/tree lock now owns the request.
        watches = self._watch_waiting() if self.rank == 0 else self.incoming.watch
        ready = self._report_readiness()
        prepare = self._prepare(
            self._start_prepares() if self.rank == 0 else self.incoming.prepare
        )
        commit = tuple(self.commits) if self.rank == 0 else self.incoming.commit
        self.commits.clear()
        self.incoming = AdmissionUpdate()
        return AdmissionUpdate(
            watch=watches,
            ready=ready,
            prepare=prepare,
            commit=commit,
            layout=layout,
        )
