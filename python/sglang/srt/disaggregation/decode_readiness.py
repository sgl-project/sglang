"""Decode readiness pacing for the just_in_time decode allocation policy.

Under a deferred policy, decode reserves a request's KV only after prefill
signals readiness, and the KV transfer then waits for that reservation
handshake. prefill_complete signals once prefill finishes, which puts the whole
handshake on the request's critical path. just_in_time signals while the
request still prefills: at the last scheduling round whose predicted remaining
prefill time R still covers the handshake latency H, that is R >= H > R - t for
round interval t. R counts the request's unscheduled chunks at t each, plus the
lead L from scheduling a chunk to processing its result; under pipeline
parallelism L spans the pipeline. A request whose whole prefill is shorter than
H signals before its first forward.

H, t and L are measured online on the signaling rank. H runs from the readiness
message to the arrival of decode's destination indices, both timestamped by the
transport rather than observed by a scheduler poll, so a request that signaled
early still yields its true handshake. Readiness that waits for decode to
register the request is sent on registration; signaling earlier cannot shorten
that wait, so H leaves it out. t and L are window medians, so the first
requests after startup, which also pay compilation, do not steer later ones.
Until all three have samples, a request signals at its first scheduling round.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence, Set, Tuple

import msgspec

from sglang.srt.disaggregation.utils import is_aborted

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

# Arbitrary; the handshake distribution has not been characterized yet.
_HANDSHAKE_WINDOW = 64
_HANDSHAKE_QUANTILE = 0.9
# Arbitrary; short enough to follow load changes within a few requests.
_TIMING_WINDOW = 16


class _WindowQuantile:
    def __init__(self, *, window: int, quantile: float):
        self._samples: deque[float] = deque(maxlen=window)
        self._quantile = quantile

    def observe(self, sample: float) -> None:
        self._samples.append(sample)

    @property
    def value(self) -> Optional[float]:
        if not self._samples:
            return None
        ordered = sorted(self._samples)
        return ordered[math.ceil(self._quantile * len(ordered)) - 1]


class _Progress(msgspec.Struct):
    req: Req
    # Clock time of the request's latest scheduled chunk, and the chunks left
    # to schedule after it.
    scheduled_at: float
    unscheduled_chunks: int
    signaled: bool = False


def _is_live(req: Req) -> bool:
    return req.finished_reason is None and not is_aborted(req)


class DecodeReadinessPacer:
    def __init__(
        self,
        *,
        chunk_tokens: Optional[int],
        next_chunk_tokens: Optional[Callable[[int], Optional[int]]] = None,
        pipelined: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ):
        # Without chunked prefill a request runs in one forward.
        self._chunk_tokens = chunk_tokens if chunk_tokens and chunk_tokens > 0 else None
        # The chunk size the scheduler picks after a given prefix length, or
        # None for chunk_tokens; set under PP dynamic chunking.
        self._next_chunk_tokens = next_chunk_tokens
        self._clock = clock
        self._handshake = _WindowQuantile(
            window=_HANDSHAKE_WINDOW, quantile=_HANDSHAKE_QUANTILE
        )
        self._round_interval = _WindowQuantile(window=_TIMING_WINDOW, quantile=0.5)
        self._lead = _WindowQuantile(window=_TIMING_WINDOW, quantile=0.5)
        self._last_busy_round_at: Optional[float] = None
        self._progress: Dict[str, _Progress] = {}
        # The first handshake with a decode endpoint also sets up connections.
        self._decode_peers: Set[Tuple[str, int]] = set()
        # Under PP: when this stage finished each request still awaiting decode.
        self._pipelined = pipelined
        self._finished: Dict[str, Tuple[Req, float]] = {}

    def on_round(self, reqs: Sequence[Req]) -> None:
        """Take one scheduling round's prefill batch, before it launches."""
        now = self._clock()
        if reqs and self._last_busy_round_at is not None:
            self._round_interval.observe(now - self._last_busy_round_at)
        self._last_busy_round_at = now if reqs else None
        for req in reqs:
            progress = self._progress.get(req.rid)
            if progress is not None:
                # Also after decode admitted it mid-prefill: the lead runs from
                # the final chunk's round.
                progress.scheduled_at = now
                if not progress.signaled:
                    progress.unscheduled_chunks = self._count_unscheduled_chunks(req)
            elif req.pending_bootstrap:
                self._progress[req.rid] = _Progress(
                    req=req,
                    scheduled_at=now,
                    unscheduled_chunks=self._count_unscheduled_chunks(req),
                )
        # A request that ends before its final prefill result never reaches
        # on_prefill_finished, nor one that fails bootstrap on_bootstrap_done.
        self._progress = {
            rid: progress
            for rid, progress in self._progress.items()
            if _is_live(progress.req)
        }
        self._finished = {
            rid: entry for rid, entry in self._finished.items() if _is_live(entry[0])
        }
        for progress in self._progress.values():
            if not progress.signaled and self._is_due(progress, now):
                progress.signaled = True
                progress.req.disagg_kv_sender.signal_decode_readiness()

    def on_prefill_finished(self, req: Req) -> None:
        now = self._clock()
        progress = self._progress.pop(req.rid, None)
        if progress is not None:
            self._lead.observe(now - progress.scheduled_at)
        if self._pipelined and req.pending_bootstrap:
            self._finished[req.rid] = (req, now)

    def on_bootstrap_done(self, req: Req) -> None:
        finished = self._finished.pop(req.rid, None)
        handshake = req.disagg_kv_sender.decode_handshake()
        if handshake is None:
            return
        if handshake.endpoint not in self._decode_peers:
            self._decode_peers.add(handshake.endpoint)
            return
        # Any PP stage's readiness admits a room. Decode registering after this
        # stage finished may be admitted by another stage's earlier message,
        # which leaves this stage's sample only the residual of the handshake.
        if finished is not None and handshake.registered_at > finished[1]:
            return
        self._handshake.observe(handshake.seconds)

    def _is_due(self, progress: _Progress, now: float) -> bool:
        handshake = self._handshake.value
        interval = self._round_interval.value
        lead = self._lead.value
        if handshake is None or interval is None or lead is None:
            return True
        remaining = (
            progress.scheduled_at + progress.unscheduled_chunks * interval + lead - now
        )
        due = remaining - interval < handshake
        if due:
            logger.debug(
                "Signal decode readiness for room %s: remaining %.3fs, "
                "handshake %.3fs, round %.3fs, lead %.3fs",
                progress.req.bootstrap_room,
                remaining,
                handshake,
                interval,
                lead,
            )
        return due

    def _count_unscheduled_chunks(self, req: Req) -> int:
        if self._chunk_tokens is None:
            return 0
        end, input_len = req.extend_end, len(req.origin_input_ids)
        if self._next_chunk_tokens is None:
            return max(0, math.ceil((input_len - end) / self._chunk_tokens))
        chunks = 0
        while end < input_len:
            end += self._next_chunk_tokens(end) or self._chunk_tokens
            chunks += 1
        return chunks
