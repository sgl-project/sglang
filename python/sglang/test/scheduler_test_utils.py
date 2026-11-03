"""The request lists of a partial scheduler, for tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.scheduler_components.deferred_responses import (
    DeferredResponses,
)
from sglang.srt.managers.scheduler_components.request_list import (
    ListKind,
    _RequestList,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class FakeReq(SimpleNamespace):
    """A stand-in request a request list can hold: compared and hashed by
    identity, as `Req` is."""

    __eq__ = object.__eq__
    __hash__ = object.__hash__


def empty_request_lists(scheduler: Scheduler) -> None:
    """Give a scheduler built with `Scheduler.__new__` every request list, empty;
    the test then fills the ones it needs."""
    scheduler.waiting_queue = []
    scheduler.grammar_manager = SimpleNamespace(
        grammar_queue=_RequestList(ListKind.GRAMMAR)
    )
    scheduler.chunked_req = None
    scheduler.prefill_launched_reqs = _RequestList(ListKind.PREFILL_LAUNCHED)
    scheduler.running_reqs = _RequestList(ListKind.DECODING)
    scheduler.disagg_prefill_bootstrap_queue = None
    scheduler.disagg_prefill_transfer_queue = _RequestList(ListKind.PD_SENDING)
    scheduler.dllm_config = None
    scheduler.disagg_decode_prealloc_queue = None
    scheduler.disagg_decode_transfer_queue = None
    scheduler.hisparse_coordinator = None
    scheduler.uocr_runtime = None
    scheduler.deferred_responses = DeferredResponses(
        disaggregation_mode=DisaggregationMode.NULL,
        output_streamer=None,
        send_abort=None,
    )
