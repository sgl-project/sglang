"""The request lists of a partial scheduler, for tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterable
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.hisparse_coordinator import HiSparseAct, HiSparseCoordinator
from sglang.srt.managers.scheduler_components.deferred_responses import (
    DeferredResponses,
)
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
)
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class FakeReq(SimpleNamespace):
    """A stand-in request a request list can hold: compared and hashed by
    identity, as `Req` is."""

    __eq__ = object.__eq__
    __hash__ = object.__hash__

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


def decode_queue(kind: ListKind, decode_reqs: Iterable = ()) -> IndexedReqList:
    """A PD decode queue of `kind` holding `decode_reqs`, each a wrapper with
    `.req`."""
    queue = IndexedReqList(kind, req_of=lambda x: x.req)
    queue.join(decode_reqs)
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
        send_abort=None,
    )


def staging_hisparse_coordinator(
    num_rows: int, reqs: Iterable = ()
) -> HiSparseCoordinator:
    """A `HiSparseCoordinator` with only what releasing a staged request reads,
    its streams and pools mocks, holding `reqs` in staging; each req's
    `kv.req_pool_idx` is its row."""
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator.ack_staging_queue = IndexedReqList(
        ListKind.HISPARSE_STAGING, req_of=lambda act: act.req
    )
    coordinator.write_staging_stream = MagicMock()
    coordinator.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros((num_rows, 8), dtype=torch.int64)
    )
    coordinator.token_to_kv_pool_allocator = MagicMock()
    coordinator.mem_pool_host = MagicMock()
    coordinator.mem_pool_host.allocated_host_indices.return_value = torch.empty(0)
    coordinator.req_to_host_pool = torch.full((num_rows, 8), -1, dtype=torch.int64)
    coordinator.req_to_host_pool_allocated_len = torch.zeros(
        num_rows, dtype=torch.int64
    )
    coordinator._skip_first_backup = {}
    coordinator.spec_swap = MagicMock()
    for req in reqs:
        req.hisparse_staging = True
        coordinator.ack_staging_queue.join([HiSparseAct(None, MagicMock(), req)])
    return coordinator
