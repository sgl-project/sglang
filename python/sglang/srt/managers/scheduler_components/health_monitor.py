"""Completed-inference acknowledgements for /health_generate probes.

The tokenizer accepts a probe as healthy only when it receives a
``HealthCheckOutput`` carrying the probe rid, this scheduler's incarnation and
a completion sequence newer than what it had seen. The monitor owns the
pending probes and emits those acknowledgements from one place: after a batch
with real work finished and its device results were synchronized. Nothing
else (control or abort traffic, launches, idle housekeeping, aborted-only or
zero-token batches) produces one, so a scheduler whose model execution is
stalled cannot pass a probe.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, List, Optional

from sglang.srt.constants import HEALTH_CHECK_TIMEOUT
from sglang.srt.disaggregation.utils import is_aborted
from sglang.srt.managers.io_struct import HealthCheckOutput

logger = logging.getLogger(__name__)

DEFAULT_MAX_PENDING_PROBES = 1024
# Probes the HTTP side has already given up on stay registered no longer than this.
DEFAULT_PENDING_PROBE_TTL_S = 2.0 * HEALTH_CHECK_TIMEOUT


@dataclass
class _PendingProbe:
    http_worker_ipc: Optional[str]
    registered_at: float
    # The request itself, kept only while it waits to be admitted on idle.
    deferred_req: Optional[object]


def batch_completed_useful_work(batch) -> bool:
    """A finished forward that proves the model is making progress.

    Prebuilt, idle and draft passes do no useful model work; neither does a
    batch that only carried aborted requests or extended zero tokens.
    """
    if not batch.reqs:
        return False
    mode = batch.forward_mode
    if mode.is_extend_without_speculative():
        if not ((batch.extend_num_tokens or 0) > 0):
            return False
    elif not (mode.is_decode() or mode.is_target_verify()):
        return False
    return any(not is_aborted(req) for req in batch.reqs)


class SchedulerHealthMonitor:
    def __init__(
        self,
        send_output: Callable[[HealthCheckOutput], None],
        dp_rank: int,
        *,
        max_pending: int = DEFAULT_MAX_PENDING_PROBES,
        ttl_s: float = DEFAULT_PENDING_PROBE_TTL_S,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._send_output = send_output
        self.dp_rank = dp_rank
        self.incarnation = uuid.uuid4().hex
        self.completion_seq = 0
        self._max_pending = max_pending
        self._ttl_s = ttl_s
        self._now = now
        self._pending: OrderedDict[str, _PendingProbe] = OrderedDict()

    @property
    def pending_rids(self) -> List[str]:
        return list(self._pending)

    def register(self, recv_req, *, admitted: bool) -> None:
        """Track a probe until a completion acknowledges it.

        A probe the scheduler does not admit (it was busy) is kept as
        ``deferred`` so ``pop_deferred`` can admit it once the engine idles;
        an admitted probe is acknowledged by whichever real batch finishes
        first, its own tiny forward included.
        """
        now = self._now()
        self._expire(now)
        rid = recv_req.rid
        self._pending.pop(rid, None)
        self._pending[rid] = _PendingProbe(
            http_worker_ipc=getattr(recv_req, "http_worker_ipc", None),
            registered_at=now,
            deferred_req=None if admitted else recv_req,
        )
        while len(self._pending) > self._max_pending:
            evicted_rid, _ = self._pending.popitem(last=False)
            logger.warning(
                "Dropping oldest pending health probe %s: more than %d pending",
                evicted_rid,
                self._max_pending,
            )

    def pop_deferred(self) -> List[object]:
        """Hand back the not-yet-admitted probes so an idle scheduler runs them.

        A probe can arrive in the iteration right after the last real batch
        completed: it is deferred as busy, yet no further completion will
        follow. Running it is the only way to prove the engine still works.
        """
        self._expire(self._now())
        deferred = []
        for probe in self._pending.values():
            if probe.deferred_req is not None:
                deferred.append(probe.deferred_req)
                probe.deferred_req = None
        return deferred

    def on_batch_completed(self, batch) -> bool:
        """Acknowledge every pending probe if ``batch`` did real work.

        Call only after the batch result was fully processed, so a failed
        device-to-host copy or result-processing error has already raised.
        """
        if not batch_completed_useful_work(batch):
            return False
        self.completion_seq += 1
        if not self._pending:
            return True
        self._expire(self._now())
        for rid, probe in self._pending.items():
            self._send_output(
                HealthCheckOutput(
                    rid=rid,
                    http_worker_ipc=probe.http_worker_ipc,
                    incarnation=self.incarnation,
                    completion_seq=self.completion_seq,
                    dp_rank=self.dp_rank,
                )
            )
        self._pending.clear()
        return True

    def _expire(self, now: float) -> None:
        while self._pending:
            rid, probe = next(iter(self._pending.items()))
            if now - probe.registered_at < self._ttl_s:
                return
            del self._pending[rid]
            logger.info(
                "Health probe %s expired after %.0fs without a completed batch",
                rid,
                now - probe.registered_at,
            )
