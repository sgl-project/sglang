"""ROCm glue of FutureMap: a gfx950 DSpark decode step gathers its seq lens on the
forward stream, and the worker pulls their host copy only once it has queued the draft,
which needs none of them; the scheduler binds it under _is_hip and sets
FutureMap.seq_lens_pull."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.managers.overlap_utils import _DEBUG_ASSERT
from sglang.srt.utils import is_gfx95_supported

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch


def make_decode_seq_lens_pull(
    *, model_worker, future_map: FutureMap, confidence_budget_prepare
) -> Optional[DeferredSeqLensPull]:
    """The pull when the worker takes decode batches' host seq lens itself, after
    queueing the work that needs none of them (HIP host-waits the publish)."""
    # A relayed publish is scattered by the verify graph before the pull could read
    # the previous one.
    if not (
        is_gfx95_supported()
        and confidence_budget_prepare is None
        and future_map.needs_cpu_seq_lens
        and future_map._relay is None
    ):
        return None
    # Local import: dspark_worker_v2 imports the scheduler.
    from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
        DSparkWorkerV2,
    )

    if not isinstance(model_worker, DSparkWorkerV2):
        return None
    from sglang.srt.speculative.dspark_components import dspark_hip

    if not dspark_hip.defers_seq_lens_cpu(model_worker):
        return None
    return DeferredSeqLensPull(future_map)


class DeferredSeqLensPull:
    """FutureMap.resolve_seq_lens_cpu of a decode batch in two halves: the gather on the
    forward stream, then the host lengths once the worker has queued what needs none of
    them. Every publish also copies the lengths to the host in stream order, so the pull
    waits for the publish alone (a copy on another stream can queue behind the forward
    on HIP)."""

    def __init__(self, future_map: FutureMap):
        self._future_map = future_map
        # A deferred resolve's host half: the rows to pull, then what they gave.
        self._rows_cpu: Optional[torch.Tensor] = None
        self._host_lens: Optional[tuple] = None
        self._published_lens_pinned: Optional[torch.Tensor] = None

    def begin_step(self, batch: ScheduleBatch) -> bool:
        """Whether this batch's host seq lens wait for the worker (decode batches); any
        other batch drops the previous step's pull."""
        if batch.forward_mode.is_decode():
            return True
        self._rows_cpu = self._host_lens = None
        return False

    def resolve(self, batch: ScheduleBatch) -> None:
        """On the forward stream the gather is ordered behind the publish, so the host
        does not wait; apply pulls the host lengths."""
        self._rows_cpu = self._host_lens = None
        future_map = self._future_map
        draft_input = batch.spec_info
        if draft_input is None:
            return
        fi = draft_input.future_indices
        if fi is None:
            return
        if future_map.publish_ready is None:
            future_map.resolve_seq_lens_cpu(batch)
            return
        if _DEBUG_ASSERT:
            assert future_map._publish_fresh, "resolve without a fresh forward publish"
            future_map._publish_fresh = False
        batch.seq_lens = future_map.new_seq_lens_buf[fi]
        # unknown until apply: the previous step's would pass for this one's
        batch.seq_lens_cpu = batch.seq_lens_sum = None
        self._rows_cpu = batch.req_pool_indices_cpu

    def apply(self, batch: ScheduleBatch, *, rebuild_seq_lens: bool = False) -> None:
        """Host half of resolve: waits for the publish only, not for the forward work
        queued behind it. Later calls in the same step re-apply the pulled lengths
        without waiting; rebuild_seq_lens also gives batch.seq_lens a copy on the
        current stream, for the schedule stream that never waited on the forward's
        gather."""
        self._pull()
        if self._host_lens is None:
            return
        batch.seq_lens_cpu, batch.seq_lens_sum = self._host_lens
        if rebuild_seq_lens:
            batch.seq_lens = batch.seq_lens_cpu.to(
                self._future_map.device, non_blocking=True
            )

    def before_publish(self) -> None:
        # a deferred pull reads the previous publish, which this one overwrites
        self._pull()

    def on_publish(self, new_seq_lens_buf: torch.Tensor) -> None:
        if self._published_lens_pinned is None:
            self._published_lens_pinned = torch.empty(
                (self._future_map.req_pool_size,), dtype=torch.int64, pin_memory=True
            )
        self._published_lens_pinned.copy_(new_seq_lens_buf, non_blocking=True)

    def _pull(self) -> None:
        if self._host_lens is None:
            rows_cpu = self._rows_cpu
            if rows_cpu is None:
                return
            # A host wait instead of a stream wait on the event (see the HIP note in
            # FutureMap.resolve_seq_lens_cpu).
            self._future_map.publish_ready.synchronize()
            seq_lens_cpu = self._published_lens_pinned[rows_cpu]
            self._host_lens = (seq_lens_cpu, int(seq_lens_cpu.sum()))
