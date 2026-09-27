"""Session close must preserve inputs until their request owners are done."""

import sys
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import schedule_batch as schedule_batch_module
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.io_struct import (
    AbortReq,
    CloseSessionReqInput,
    OpenSessionReqInput,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session import session_controller as session_module
from sglang.srt.session.session_controller import SessionController

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture
def scheduler(monkeypatch):
    monkeypatch.setattr(
        schedule_batch_module, "get_parallel", lambda: SimpleNamespace(tp_rank=1)
    )
    monkeypatch.setattr(session_module, "log_info_on_rank0", Mock())
    monkeypatch.setattr(
        scheduler_module, "get_serving", lambda: SimpleNamespace(weight_version=None)
    )
    monkeypatch.setattr(
        scheduler_module, "get_parallel", lambda: SimpleNamespace(pp_size=1)
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tree_cache = Mock()
    scheduler.session_controller = SessionController(scheduler.tree_cache)
    scheduler.enable_session_radix_cache = True
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.chunked_req = None
    scheduler._pending_chunked_abort_req = None
    scheduler.waiting_queue = []
    scheduler.running_batch = SimpleNamespace(reqs=[])
    scheduler.last_batch = None
    scheduler.mm_receiver = None
    scheduler.dllm_config = None
    scheduler.grammar_manager = Mock()
    scheduler.beam_coordinator = Mock()
    scheduler.ipc_channels = Mock()
    return scheduler


def _session(scheduler, *, streaming=False):
    scheduler.session_controller.open(
        OpenSessionReqInput(
            capacity_of_str_len=1024, session_id="session-a", streaming=streaming
        )
    )
    return scheduler.session_controller.get("session-a")


def _request(session, rid, previous=None):
    sampling_params = SamplingParams(max_new_tokens=1)
    sampling_params.normalize(tokenizer=None)
    recv = TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=array("q", [1, 2]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=sampling_params,
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        session_params=SessionParams(id=session.session_id, rid=previous),
    )
    return session.create_req(recv, tokenizer=None, vocab_size=128)


def _image(req):
    feature = torch.tensor([[1.0, 2.0]])
    mm = MultimodalInputs(
        mm_items=[MultimodalDataItem(modality=Modality.IMAGE, feature=feature)]
    )
    mm.release_features = Mock(wraps=mm.release_features)
    req.multimodal_inputs = mm
    return mm


def _complete(req):
    req.output_ids.append(3)
    req.update_finish_state()
    assert req.finished()


def _assert_retained(scheduler, session, mm, *reqs):
    assert scheduler.session_controller.get(session.session_id) is session
    assert session.close_on_finish
    for req in reqs:
        assert req.session is session
        assert req.multimodal_inputs is mm
    assert torch.equal(mm.mm_items[0].feature, torch.tensor([[1.0, 2.0]]))
    mm.release_features.assert_not_called()
    scheduler.tree_cache.release_radix_session.assert_not_called()
    scheduler.tree_cache.release_session.assert_not_called()


def _assert_released(scheduler, session, mm, *reqs):
    assert session.session_id not in scheduler.session_controller
    assert mm.mm_items[0].feature is None
    assert all(req.multimodal_inputs is None for req in reqs)
    mm.release_features.assert_called_once_with()
    scheduler.tree_cache.release_radix_session.assert_called_once_with(
        session.session_id
    )
    scheduler.tree_cache.release_session.assert_called_once_with(session.session_id)


@pytest.mark.parametrize("timed_out", [False, True])
def test_close_waits_for_all_ordinary_owners_of_shared_image(
    scheduler, monkeypatch, timed_out
):
    session = _session(scheduler)
    previous = _request(session, "previous")
    mm = _image(previous)
    _complete(previous)
    first = _request(session, "first", previous.rid)
    second = _request(session, "second", previous.rid)

    if timed_out:
        session.timeout = 1
        session.last_active_time = 0
        monkeypatch.setattr(
            session_module, "time", SimpleNamespace(monotonic=lambda: 2)
        )
        scheduler.session_controller.maybe_reap(now=2)
    else:
        scheduler.close_session(CloseSessionReqInput(session_id=session.session_id))
    _assert_retained(scheduler, session, mm, previous, first, second)

    _complete(first)
    scheduler.session_controller.maybe_reap(now=4)
    _assert_retained(scheduler, session, mm, previous, first, second)

    scheduler.close_session(CloseSessionReqInput(session_id=session.session_id))
    _complete(second)
    scheduler.session_controller.maybe_reap(now=6)
    scheduler.session_controller.maybe_reap(now=8)
    _assert_released(scheduler, session, mm, previous, first, second)


def test_running_abort_does_not_release_inputs_until_confirmed(scheduler):
    session = _session(scheduler)
    req = _request(session, "active")
    mm = _image(req)
    scheduler.running_batch.reqs = [req]
    scheduler.close_session(CloseSessionReqInput(session_id=session.session_id))

    scheduler.abort_request(AbortReq(rid=req.rid))
    assert isinstance(req.to_finish, FINISH_ABORT)
    assert not req.finished()
    scheduler.session_controller.maybe_reap(now=2)
    _assert_retained(scheduler, session, mm, req)

    # The batch-result path confirms the pending abort after resolving its work.
    req.update_finish_state()
    scheduler.session_controller.maybe_reap(now=4)
    _assert_released(scheduler, session, mm, req)


@pytest.mark.parametrize("queue_full", [False, True])
def test_terminal_queued_abort_allows_deferred_close(scheduler, queue_full):
    session = _session(scheduler)
    req = _request(session, "queued")
    mm = _image(req)
    scheduler.close_session(CloseSessionReqInput(session_id=session.session_id))
    _assert_retained(scheduler, session, mm, req)

    if queue_full:
        scheduler.max_queued_requests = 0
        scheduler.enable_priority_scheduling = False
        assert scheduler._abort_on_queued_limit(req)
    else:
        scheduler.waiting_queue = [req]
        reason = {"type": "abort", "status_code": 503, "message": "Client cancelled"}
        scheduler.abort_request(AbortReq(rid=req.rid, finished_reason=reason))
        assert scheduler.waiting_queue == []
        message, owner = (
            scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args
        )
        assert owner is req
        assert message.finished_reason == reason

    assert isinstance(req.finished_reason, FINISH_ABORT)
    assert req.finished_reason.status_code == 503
    scheduler.session_controller.maybe_reap(now=2)
    _assert_released(scheduler, session, mm, req)


def test_chunked_abort_keeps_image_until_delayed_result_drains(scheduler, monkeypatch):
    session = _session(scheduler)
    req = _request(session, "chunked")
    mm = _image(req)
    req.inflight_middle_chunks = 1
    req.metadata_buffer_index = -1
    scheduler.chunked_req = req
    scheduler._pending_chunked_abort_req = req
    monkeypatch.setattr(scheduler_module, "release_kv_cache", Mock())

    scheduler.process_pending_chunked_abort()
    assert req.finished()
    scheduler.close_session(CloseSessionReqInput(session_id=session.session_id))
    scheduler.session_controller.maybe_reap(now=2)
    _assert_retained(scheduler, session, mm, req)

    scheduler.batch_result_processor = Mock()
    scheduler.spec_algorithm = SimpleNamespace(is_eagle=lambda: False)
    scheduler.disagg_prefill_pending_chunk_rids = set()
    scheduler.metrics_reporter = Mock()
    scheduler.maybe_send_health_check_signal = Mock()
    batch = SimpleNamespace(
        reqs=[req], spec_info=None, prefill_stats=None, dp_cooperation_info=None
    )
    scheduler.process_batch_result_disagg_prefill(
        batch, GenerationBatchResult(next_token_ids=torch.tensor([3]))
    )
    assert req.inflight_middle_chunks == 0
    assert req.output_ids == array("q")
    scheduler.session_controller.maybe_reap(now=4)
    _assert_released(scheduler, session, mm, req)


def test_streaming_first_turn_still_defers_close(scheduler):
    session = _session(scheduler, streaming=True)
    req = _request(session, "first")
    mm = _image(req)
    assert not session.req_nodes

    scheduler.close_session(CloseSessionReqInput(session_id=session.session_id))
    scheduler.session_controller.maybe_reap(now=2)
    _assert_retained(scheduler, session, mm, req)

    _complete(req)
    session.finish_req(req)
    scheduler.session_controller.maybe_reap(now=4)
    _assert_released(scheduler, session, mm, req)


def test_radix_only_session_close_still_releases_cache(scheduler):
    scheduler.close_session(CloseSessionReqInput(session_id="radix-only"))

    scheduler.tree_cache.release_radix_session.assert_called_once_with("radix-only")
    scheduler.tree_cache.release_session.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
