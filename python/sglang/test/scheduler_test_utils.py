"""The request lists of a partial scheduler, for tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterable
from unittest.mock import MagicMock

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.scheduler_components.deferred_responses import (
    DeferredResponses,
)
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
    move,
)
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class FakeReq(SimpleNamespace):
    """A stand-in request a request list can hold: compared and hashed by
    identity, as `Req` is."""

    __eq__ = object.__eq__
    __hash__ = object.__hash__

    owner_list = None
    # Req's exit_ledger fields.
    exit_via = None
    send_via = None


def release_only_retirer(
    tree_cache=None, hisparse_coordinator=None, beam_coordinator=None
) -> RequestRetirer:
    """A `RequestRetirer` for a test that only releases requests, never aborts
    a queued one."""
    return RequestRetirer(
        tree_cache=tree_cache,
        beam_coordinator=beam_coordinator
        or SimpleNamespace(retire_group=lambda req: None),
        hisparse_coordinator=hisparse_coordinator,
        uocr_runtime=None,
        send_abort_output=None,
        stream_output=None,
    )


def leaving_retirer() -> MagicMock:
    """A mock `RequestRetirer` whose exits take the request out of its list."""
    retirer = MagicMock()

    def leave(req, *_, **__):
        if req.owner_list is not None:
            req.owner_list._leave([req])

    for exit_ in (
        "abort",
        "retire_unadmitted",
        "release",
        "abort_retracted",
        "abandon",
        "settle_output",
    ):
        getattr(retirer, exit_).side_effect = leave
    return retirer


def decode_queue(kind: ListKind, decode_reqs: Iterable = ()) -> IndexedReqList:
    """A PD decode queue of `kind` holding `decode_reqs`, each a wrapper with
    `.req`."""
    queue = IndexedReqList(kind, req_of=lambda x: x.req)
    move(decode_reqs, queue)
    return queue


def empty_request_lists(scheduler: Scheduler) -> None:
    """Give a scheduler built with `Scheduler.__new__` every request list, empty;
    the test then fills the ones it needs."""
    scheduler.waiting_queue = IndexedReqList(
        ListKind.WAITING, abort=scheduler._abort_waiting
    )
    scheduler.grammar_manager = SimpleNamespace(
        grammar_queue=IndexedReqList(ListKind.GRAMMAR)
    )
    scheduler.chunked_reqs = IndexedReqList(
        ListKind.CHUNKED, abort=scheduler._abort_chunked
    )
    scheduler.prefill_launched_reqs = KeyedReqList(ListKind.PREFILL_LAUNCHED)
    scheduler.running_reqs = KeyedReqList(ListKind.DECODING)
    scheduler.disagg_prefill_bootstrap_queue = None
    scheduler.disagg_prefill_transfer_queue = IndexedReqList(ListKind.PD_SENDING)
    scheduler.dllm_config = None
    scheduler.disagg_decode_prealloc_queue = None
    scheduler.disagg_decode_transfer_queue = None
    scheduler.hisparse_coordinator = None
    scheduler.uocr_runtime = None
    scheduler.deferred_responses = DeferredResponses(
        disaggregation_mode=DisaggregationMode.NULL,
        output_streamer=None,
        request_retirer=None,
        send_abort=None,
    )
    scheduler.decode_offload_manager = None
