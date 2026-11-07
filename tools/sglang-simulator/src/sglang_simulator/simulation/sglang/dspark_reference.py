"""Explicit reference CPU bookkeeping, independent of synchronous simulation.

The source adapter owns when prepare, relay, publish and result-processing events
occur. This immutable ledger does not infer that order from G's worker callbacks,
choose reference batch membership or establish source/profile compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from sglang_simulator.simulation.sglang.dspark import DSparkStep, DSparkStepResult
from sglang_simulator.time_predictor.base import PredictorError


def _require(value, message):
    if not value:
        raise PredictorError("invalid_reference_state", message)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _local_group(step, rank):
    _require(
        type(step) is DSparkStep
        and _integer(rank)
        and rank < step.topology.attention_dp_size,
        "reference local attention-DP rank is outside its actual step",
    )
    return step.groups[rank]


def _decode_alloc_lens(reqs, *, reserve, page_size):
    # One allocation formula owner. Import only when explicit reference decode
    # preparation is requested in an environment containing native SGLang.
    from sglang.srt.mem_cache.allocation_sizing import page_aligned_decode_alloc_lens

    return page_aligned_decode_alloc_lens(reqs, reserve=reserve, page_size=page_size)


@dataclass(frozen=True)
class ReferenceRequestKV:
    rid: str
    kv_committed_len: int
    kv_allocated_len: int
    published_seq_len: int | None
    processed_output_tokens: int
    max_new_tokens: int
    prefill_complete: bool = False
    active: bool = True

    def __post_init__(self):
        _require(
            type(self.rid) is str
            and bool(self.rid)
            and _integer(self.kv_committed_len)
            and _integer(self.kv_allocated_len)
            and self.kv_committed_len <= self.kv_allocated_len
            and (
                self.published_seq_len is None
                or _integer(self.published_seq_len)
                and self.published_seq_len <= self.kv_allocated_len
            )
            and _integer(self.processed_output_tokens)
            and _integer(self.max_new_tokens, 1)
            and type(self.prefill_complete) is bool
            and type(self.active) is bool,
            "reference request KV metadata is inconsistent",
        )

    @property
    def kv(self):
        """Read-only native page-sizing request interface."""
        return self


@dataclass(frozen=True)
class ReferenceCPUView:
    seq_lens_cpu: tuple[int, ...] | None
    nxt_kv_lens_cpu: tuple[int, ...] | None
    local_attention_dp_rank: int = 0
    is_idle: bool = False

    def __post_init__(self):
        _require(
            _integer(self.local_attention_dp_rank)
            and self.local_attention_dp_rank < 2
            and type(self.is_idle) is bool,
            "reference CPU view requires an explicit local owner and idle state",
        )
        if self.is_idle:
            _require(
                type(self.seq_lens_cpu) is tuple
                and self.seq_lens_cpu == ()
                and self.nxt_kv_lens_cpu is None,
                "reference idle CPU view has no local request or reserved vector",
            )
            return
        vectors = (self.seq_lens_cpu, self.nxt_kv_lens_cpu)
        _require(
            any(value is not None for value in vectors)
            and all(
                value is None
                or type(value) is tuple
                and 1 <= len(value) <= 64
                and all(_integer(length) for length in value)
                for value in vectors
            )
            and (
                any(value is None for value in vectors)
                or len(vectors[0]) == len(vectors[1])
            ),
            "reference CPU view requires immutable known vectors or known absence",
        )

    @property
    def planner_seq_lens_cpu(self):
        return (
            self.seq_lens_cpu if self.seq_lens_cpu is not None else self.nxt_kv_lens_cpu
        )


@dataclass(frozen=True)
class PreparedReferenceBatch:
    step: DSparkStep
    seed_seq_lens_cpu: tuple[int, ...]
    nxt_kv_lens_cpu: tuple[int, ...] | None
    published_seq_lens: tuple[int | None, ...]
    local_attention_dp_rank: int = 0

    @property
    def local_group(self):
        return _local_group(self.step, self.local_attention_dp_rank)

    def __post_init__(self):
        _require(type(self.step) is DSparkStep, "reference view requires a typed step")
        group = self.local_group
        size = len(group.requests)
        decode = group.forward_mode == "TARGET_VERIFY"
        _require(
            type(self.seed_seq_lens_cpu) is tuple
            and len(self.seed_seq_lens_cpu) == size
            and all(_integer(length) for length in self.seed_seq_lens_cpu)
            and (
                self.nxt_kv_lens_cpu is None
                if not decode
                else type(self.nxt_kv_lens_cpu) is tuple
                and len(self.nxt_kv_lens_cpu) == size
                and all(_integer(length) for length in self.nxt_kv_lens_cpu)
            )
            and type(self.published_seq_lens) is tuple
            and len(self.published_seq_lens) == size
            and all(
                length is None or _integer(length) for length in self.published_seq_lens
            ),
            "reference preparation requires immutable ordered CPU metadata",
        )

    def resolve(self, *, future_indices_ready, needs_cpu_seq_lens):
        """Apply explicitly known relay policy to the frozen preparation.

        False means known absent future indices, not a present unready relay.
        Absent future indices keep the seed even if the backend needs no CPU
        mirror. Resolving a relay may produce known absence; missing policy or
        publication cannot be encoded as absence. No target-bound formula lives
        here, and the reserved planner vector remains a separate source fact.
        """
        _require(
            type(future_indices_ready) is bool,
            "reference future-index readiness is unknown",
        )
        if self.local_group.forward_mode == "IDLE":
            _require(
                not future_indices_ready,
                "reference idle group cannot resolve a local decode relay",
            )
            return ReferenceCPUView((), None, self.local_attention_dp_rank, True)
        if not future_indices_ready:
            return ReferenceCPUView(
                self.seed_seq_lens_cpu,
                self.nxt_kv_lens_cpu,
                self.local_attention_dp_rank,
            )
        _require(
            self.local_group.forward_mode == "TARGET_VERIFY",
            "reference homogeneous prefill does not resolve a decode relay",
        )
        _require(
            type(needs_cpu_seq_lens) is bool,
            "reference backend CPU-mirror policy is unknown",
        )
        _require(
            all(value is not None for value in self.published_seq_lens),
            "reference relay has no prior modeled publication",
        )
        return ReferenceCPUView(
            self.published_seq_lens if needs_cpu_seq_lens else None,
            self.nxt_kv_lens_cpu,
            self.local_attention_dp_rank,
        )


@dataclass(frozen=True)
class DSparkReferenceLedger:
    """Stage explicit native-order events using previously modeled outcomes.

    Preparation precedes optional synchronization/settlement. Current publication
    precedes ordinary previous-result settlement, so the FIFO may transiently hold
    two entries. Callers retain the original ledger when cost or publication fails.
    Native CPU requests/mirrors are never mutated. The native-policy factory's
    source event driver owns production activation.
    """

    requests: tuple[ReferenceRequestKV, ...] = ()
    pending: tuple[DSparkStepResult, ...] = ()
    prepared: PreparedReferenceBatch | None = None
    last_forward_id: int = -1
    local_attention_dp_rank: int = 0

    def __post_init__(self):
        _require(
            type(self.requests) is tuple
            and all(type(row) is ReferenceRequestKV for row in self.requests)
            and len({row.rid for row in self.requests}) == len(self.requests)
            and type(self.pending) is tuple
            and len(self.pending) <= 2
            and all(type(row) is DSparkStepResult for row in self.pending)
            and type(self.last_forward_id) is int
            and self.last_forward_id >= -1
            and _integer(self.local_attention_dp_rank)
            and self.local_attention_dp_rank < 2
            and (
                self.prepared is None or type(self.prepared) is PreparedReferenceBatch
            ),
            "reference ledger requires typed unique requests and a bounded FIFO",
        )
        ids = tuple(row.step.forward_id for row in self.pending)
        _require(
            ids == tuple(sorted(set(ids)))
            and all(fid <= self.last_forward_id for fid in ids)
            and all(
                result.local_attention_dp_rank == self.local_attention_dp_rank
                for result in self.pending
            )
            and (
                self.prepared is None
                or self.prepared.local_attention_dp_rank == self.local_attention_dp_rank
            ),
            "reference pending results are not in publication order",
        )

    def request_for(self, rid):
        for row in self.requests:
            if row.rid == rid:
                return row
        raise PredictorError(
            "data_unavailable",
            "reference KV has no explicit prefill admission",
            rid=rid,
        )

    def prepare(self, step, *, page_size):
        _require(
            type(step) is DSparkStep, "reference preparation requires a typed step"
        )
        group = _local_group(step, self.local_attention_dp_rank)
        _require(
            self.prepared is None
            and len(self.pending) <= 1
            and step.forward_id > self.last_forward_id,
            "reference preparation precedes retirement or repeats a forward",
        )
        _require(
            (
                step.topology.tp_size,
                step.topology.attention_dp_size,
                step.topology.pp_size,
            )
            in {(2, 1, 1), (4, 1, 1), (2, 2, 1)}
            and {g.forward_mode for g in step.groups} - {"IDLE"}
            in ({"EXTEND"}, {"TARGET_VERIFY"})
            and all(len(g.requests) <= 64 for g in step.groups)
            and _integer(page_size, 1),
            "reference bookkeeping requires TP2/TP4 ADP1/PP1 or TP2 ADP2/PP1 B1-B64 page domain",
        )
        rows = {row.rid: row for row in self.requests}
        _require(
            not any(
                request.rid in rows
                for peer in step.groups
                if peer.attention_dp_rank != self.local_attention_dp_rank
                for request in peer.requests
            ),
            "reference peer request is already owned by this local ledger",
        )
        prefill = group.forward_mode == "EXTEND"
        local = []
        for request in group.requests:
            row = rows.get(request.rid)
            _require(
                row is None or row.active,
                "reference finished/retracted request requires explicit retirement",
            )
            if prefill:
                _require(
                    not any(
                        result.local_group.forward_mode == "TARGET_VERIFY"
                        and any(
                            r.rid == request.rid for r in result.local_group.requests
                        )
                        for result in self.pending
                    ),
                    "reference prefill cannot overwrite an unsettled decode",
                )
                if row is not None:
                    _require(
                        not row.prefill_complete
                        and row.published_seq_len == request.target_prefix_len
                        and row.max_new_tokens == request.max_new_tokens,
                        "reference chunk does not continue its published request",
                    )
                else:
                    _require(
                        request.output_tokens_before == 0,
                        "reference admission cannot invent processed output history",
                    )
                length = request.target_prefix_len + request.extend_len
                row = ReferenceRequestKV(
                    request.rid,
                    length,
                    length,
                    None if row is None else row.published_seq_len,
                    0 if row is None else row.processed_output_tokens,
                    request.max_new_tokens,
                )
            else:
                row = self.request_for(request.rid)
                _require(
                    row.prefill_complete
                    and row.processed_output_tokens < row.max_new_tokens
                    and request.extend_len == 6
                    and request.emits_output is True
                    and row.published_seq_len == request.target_prefix_len
                    and row.max_new_tokens == request.max_new_tokens,
                    "reference decode differs from its prior modeled publication",
                )
            rows[request.rid] = row
            local.append(row)
        seed = tuple(row.kv_committed_len for row in local)
        following = None
        if group.forward_mode == "TARGET_VERIFY":
            current, following, needed = _decode_alloc_lens(
                local, reserve=12, page_size=page_size
            )
            _require(
                tuple(current) == tuple(row.kv_allocated_len for row in local)
                and len(following) == len(local)
                and all(
                    _integer(nxt) and nxt >= cur for cur, nxt in zip(current, following)
                )
                and type(needed) is int
                and needed == sum(nxt - cur for cur, nxt in zip(current, following)),
                "native reference allocation sizing returned inconsistent metadata",
            )
            following = tuple(following)
            for row, length in zip(local, following):
                rows[row.rid] = replace(row, kv_allocated_len=length)
        prepared = PreparedReferenceBatch(
            step,
            seed,
            following,
            tuple(row.published_seq_len for row in local),
            self.local_attention_dp_rank,
        )
        return replace(self, requests=tuple(rows.values()), prepared=prepared)

    def publish(self, result):
        _require(
            type(result) is DSparkStepResult
            and self.prepared is not None
            and result.step is self.prepared.step
            and result.local_attention_dp_rank == self.local_attention_dp_rank,
            "reference publication requires its exact prepared modeled result",
        )
        rows = {row.rid: row for row in self.requests}
        prefill = result.local_group.forward_mode == "EXTEND"
        counts = dict(result.progress.committed_tokens)
        for request in result.local_group.requests:
            count = counts[request.rid]
            _require(
                not prefill or count == int(request.emits_output),
                "reference prefill publication requires its native zero/one bonus",
            )
            rows[request.rid] = replace(
                rows[request.rid],
                published_seq_len=request.target_prefix_len
                + (request.extend_len if prefill else count),
                prefill_complete=request.emits_output if prefill else True,
            )
        return replace(
            self,
            requests=tuple(rows.values()),
            pending=(*self.pending, result),
            prepared=None,
            last_forward_id=result.step.forward_id,
        )

    def process_result(self, forward_id, *, skipped_request_ids=()):
        """Settle FIFO output; finished/retracted skips must be explicit events."""
        _require(
            type(forward_id) is int
            and bool(self.pending)
            and self.pending[0].step.forward_id == forward_id,
            "reference result processing must retire the FIFO head exactly once",
        )
        result = self.pending[0]
        _require(
            type(skipped_request_ids) is tuple
            and all(type(rid) is str for rid in skipped_request_ids)
            and len(set(skipped_request_ids)) == len(skipped_request_ids)
            and set(skipped_request_ids)
            <= {r.rid for r in result.local_group.requests},
            "reference skipped request ownership differs from its pending result",
        )
        rows = {row.rid: row for row in self.requests}
        prefill = result.local_group.forward_mode == "EXTEND"
        counts = dict(result.progress.committed_tokens)
        for request in result.local_group.requests:
            count = counts[request.rid]
            if request.rid in skipped_request_ids:
                rows[request.rid] = replace(self.request_for(request.rid), active=False)
                continue
            row = self.request_for(request.rid)
            _require(
                row.active,
                "reference inactive pending request requires an explicit skip",
            )
            rows[row.rid] = replace(
                row,
                kv_committed_len=row.kv_committed_len + (0 if prefill else count),
                processed_output_tokens=row.processed_output_tokens + count,
            )
        return replace(self, requests=tuple(rows.values()), pending=self.pending[1:])

    def mark_inactive(self, rid):
        """Record an explicit finish/retraction without dropping in-flight work."""
        self.request_for(rid)
        return replace(
            self,
            requests=tuple(
                replace(row, active=False) if row.rid == rid else row
                for row in self.requests
            ),
        )

    def forget(self, rid):
        _require(
            type(rid) is str and bool(rid), "reference retirement needs a request ID"
        )
        owned = [result.local_group for result in self.pending]
        if self.prepared is not None:
            owned.append(self.prepared.local_group)
        _require(
            not any(
                request.rid == rid for group in owned for request in group.requests
            ),
            "reference request retirement still owns prepared or pending work",
        )
        return replace(
            self, requests=tuple(row for row in self.requests if row.rid != rid)
        )

    def clear(self):
        _require(
            not self.pending and self.prepared is None,
            "reference cache reset requires explicit pending-result drain",
        )
        return replace(self, requests=())
