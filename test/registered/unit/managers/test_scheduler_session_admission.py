"""Reject invalid native-session requests before scheduler admission."""

import sys
from array import array
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()


from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import schedule_batch as schedule_batch_module
from sglang.srt.managers.io_struct import SessionParams, TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session import session_controller as session_module
from sglang.srt.session.session_controller import (
    Session,
    SessionController,
    SessionReqNode,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def rank_local_logging(monkeypatch):
    monkeypatch.setattr(
        schedule_batch_module, "get_parallel", lambda: SimpleNamespace(tp_rank=1)
    )
    monkeypatch.setattr(session_module, "log_info_on_rank0", Mock())


def _generation(session_id="session-a"):
    return TokenizedGenerateReqInput(
        rid="first-turn",
        input_text=None,
        input_ids=array("q", [1, 2]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=8),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        session_params=SessionParams(id=session_id),
        bootstrap_host="prefill",
        bootstrap_port=8998,
        bootstrap_room=123456789,
        http_worker_ipc="ipc://session-test-tokenizer",
    )


def _scheduler(mode=DisaggregationMode.PREFILL):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tree_cache = Mock()
    scheduler.session_controller = SessionController(scheduler.tree_cache)
    scheduler.enable_session_radix_cache = True
    scheduler.model_config = SimpleNamespace(vocab_size=128, hf_eos_token_id={127})
    scheduler.tokenizer = None
    scheduler.disaggregation_mode = mode
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    scheduler.output_streamer = Mock()
    scheduler.init_req_max_new_tokens = Mock()
    scheduler._add_request_to_queue = Mock()
    scheduler._maybe_namespace_elastic_radix_cache = Mock(
        side_effect=AssertionError("Rejected session request entered normal processing")
    )
    return scheduler


def _assert_rejected(scheduler, incoming, message):
    scheduler.handle_generate_request(incoming)

    scheduler._add_request_to_queue.assert_not_called()
    scheduler.output_streamer.stream_output.assert_called_once()
    [rejected], return_logprob = scheduler.output_streamer.stream_output.call_args.args
    assert return_logprob is False
    assert rejected.rid == incoming.rid
    assert rejected.http_worker_ipc == incoming.http_worker_ipc
    assert rejected.finished()
    assert rejected.to_finish is None
    assert isinstance(rejected.finished_reason, FINISH_ABORT)
    assert rejected.finished_reason.to_json() == {
        "type": "abort",
        "message": message,
        "status_code": HTTPStatus.BAD_REQUEST,
        "err_type": "BadRequestError",
    }
    assert rejected.output_ids == array("q")
    scheduler.tree_cache.finish.assert_not_called()


@pytest.mark.parametrize("mode", list(DisaggregationMode))
@pytest.mark.parametrize("closing", [False, True])
def test_missing_or_closing_session_rejects_before_admission(mode, closing):
    scheduler = _scheduler(mode)
    if closing:
        session = Session(1024, "session-a", streaming=True)
        session.close_on_finish = True
        scheduler.session_controller.sessions["session-a"] = session
        message = "Invalid request: close was requested for session session-a"
    else:
        message = "Invalid request: session id session-a does not exist"

    _assert_rejected(scheduler, _generation(), message)


@pytest.mark.parametrize(
    "mode", [DisaggregationMode.PREFILL, DisaggregationMode.DECODE]
)
def test_rejected_overlap_preserves_active_streaming_owner(mode):
    scheduler = _scheduler(mode)
    session = Session(1024, "session-a", streaming=True)
    owner = Req("owner", None, array("q", [3, 4]), SamplingParams(), session=session)
    owner.multimodal_inputs = Mock()
    owner_node = SessionReqNode(owner)
    session.req_nodes[owner.rid] = owner_node
    session._inflight = True
    scheduler.session_controller.sessions[session.session_id] = session

    _assert_rejected(
        scheduler, _generation(), "Streaming session already has an active request."
    )

    assert session._inflight
    assert session.req_nodes == {"owner": owner_node}
    assert owner.session is session
    assert owner.origin_input_ids == array("q", [3, 4])
    assert owner.to_finish is None
    owner.multimodal_inputs.release_features.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
