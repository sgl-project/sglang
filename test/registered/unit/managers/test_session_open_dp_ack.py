"""Session opens wait until every attention-DP queue leader is ready."""

import asyncio
import sys
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components import request_receiver as receiver_module
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session import session_controller as session_module
from sglang.srt.session.session_controller import SessionController

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def rank_local_logging(monkeypatch):
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


def _scheduler():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tree_cache = Mock()
    scheduler.session_controller = SessionController(scheduler.tree_cache)
    scheduler.enable_session_radix_cache = True
    return scheduler


def _open_session(scheduler, incoming, **rank):
    parallel = dict(tp_rank=0, attn_tp_rank=0, attn_cp_rank=0, pp_rank=0)
    parallel.update(rank)
    with patch.object(
        scheduler_module, "get_parallel", return_value=SimpleNamespace(**parallel)
    ):
        return scheduler.open_session(incoming)


class _Tokenizer(TokenizerControlMixin):
    _handle_open_session_req_output = TokenizerManager._handle_open_session_req_output

    def __init__(self, workers):
        self.elastic_worker_count = workers
        self.session_open_communicators = {}
        self.dispatched = asyncio.Queue()

    def auto_create_handle_loop(self):
        pass

    def _dispatch_to_scheduler(self, obj):
        self.dispatched.put_nowait(obj)


@pytest.mark.parametrize("local_control", [False, True])
def test_first_generation_waits_for_delayed_dp_leader(monkeypatch, local_control):
    async def scenario():
        tokenizer = _Tokenizer(workers=2)
        fast = _scheduler()
        slow = _scheduler()
        work = []

        async def client():
            session_id = await tokenizer.open_session(
                OpenSessionReqInput(capacity_of_str_len=1024, session_id="session-a")
            )
            work.append(_generation(session_id))

        task = asyncio.create_task(client())
        opened = await tokenizer.dispatched.get()
        tokenizer._handle_open_session_req_output(_open_session(fast, opened))
        # Yield a loop turn so a premature acknowledgement would dispatch work.
        # There is no wall-clock sleep or scheduling race with the delayed rank.
        await asyncio.sleep(0)
        assert not task.done()
        assert not work

        # Replay the real split-channel receive path. In the old implementation
        # the client's first turn was already work and overtook this open.
        receiver = SimpleNamespace(
            tp_group=SimpleNamespace(rank=1, ranks=[0, 1]),
            tp_cpu_group=None,
        )
        receiver._split_work_and_control_reqs = lambda reqs: (
            receiver_module.SchedulerRequestReceiver._split_work_and_control_reqs(
                receiver, reqs
            )
        )
        monkeypatch.setattr(
            receiver_module,
            "get_parallel",
            lambda: SimpleNamespace(
                attn_tp_rank=0,
                attn_cp_rank=0,
                tp_size=2,
                enable_dp_attention=True,
                enable_dp_attention_local_control_broadcast=local_control,
            ),
        )
        monkeypatch.setattr(
            receiver_module,
            "get_exec",
            lambda: SimpleNamespace(moe=SimpleNamespace(is_ep_scale_joiner=False)),
        )
        monkeypatch.setattr(
            receiver_module, "attn_cp_tp_broadcast_pyobj", lambda reqs: reqs
        )
        monkeypatch.setattr(
            receiver_module, "broadcast_pyobj", lambda *a, **k: [opened]
        )
        received = (
            receiver_module.SchedulerRequestReceiver._broadcast_reqs_across_ranks(
                receiver, [*work, opened] if local_control else work
            )
        )
        assert received == [opened]
        tokenizer._handle_open_session_req_output(
            _open_session(slow, received[0], tp_rank=1)
        )
        await task

        [first_turn] = work
        session = slow.session_controller.get(first_turn.session_params.id)
        accepted = session.create_req(first_turn, None, 128)
        assert accepted.session is session
        assert accepted.to_finish is None
        assert accepted.bootstrap_room == first_turn.bootstrap_room
        assert not tokenizer.session_open_communicators

        # An existing session's close retains its position after queued work.
        close = CloseSessionReqInput(session_id="session-a")
        monkeypatch.setattr(receiver_module, "broadcast_pyobj", lambda *a, **k: [close])
        received = (
            receiver_module.SchedulerRequestReceiver._broadcast_reqs_across_ranks(
                receiver, [first_turn, close] if local_control else [first_turn]
            )
        )
        assert received == [first_turn, close]

    asyncio.run(scenario())


@pytest.mark.parametrize("workers", [1, 2])
def test_concurrent_session_opens_collect_their_own_results(workers):
    async def scenario():
        tokenizer = _Tokenizer(workers)
        tasks = {}
        for session_id in ("a", "b"):
            tasks[session_id] = asyncio.create_task(
                tokenizer.open_session(
                    OpenSessionReqInput(capacity_of_str_len=1024, session_id=session_id)
                )
            )
            assert (await tokenizer.dispatched.get()).session_id == session_id

        for _ in range(workers):
            tokenizer._handle_open_session_req_output(
                OpenSessionReqOutput(session_id="b", success=True)
            )
        assert await tasks["b"] == "b"
        assert not tasks["a"].done()

        for rank in range(workers):
            tokenizer._handle_open_session_req_output(
                OpenSessionReqOutput(session_id="a", success=rank != 0)
            )
        assert await tasks["a"] is None
        assert not tokenizer.session_open_communicators

    asyncio.run(scenario())


def test_cancelled_open_drops_late_acknowledgements():
    async def scenario():
        tokenizer = _Tokenizer(workers=2)
        task = asyncio.create_task(
            tokenizer.open_session(OpenSessionReqInput(capacity_of_str_len=1024))
        )
        opened = await tokenizer.dispatched.get()
        assert opened.session_id is not None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        tokenizer._handle_open_session_req_output(
            OpenSessionReqOutput(session_id=opened.session_id, success=True)
        )
        assert not tokenizer.session_open_communicators

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "rank,acknowledges",
    [
        ({"tp_rank": 1}, True),
        ({"tp_rank": 1, "attn_tp_rank": 1}, False),
        ({"attn_cp_rank": 1}, False),
        ({"pp_rank": 1}, False),
    ],
)
def test_only_dp_queue_leaders_acknowledge(rank, acknowledges):
    scheduler = _scheduler()
    output = _open_session(
        scheduler,
        OpenSessionReqInput(capacity_of_str_len=1024, session_id="session-a"),
        **rank,
    )
    assert "session-a" in scheduler.session_controller
    scheduler.tree_cache.open_radix_session.assert_called_once_with("session-a")
    assert (output is not None) is acknowledges


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
