import sys
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.managers import scheduler_ft_mixin as ft
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture
def scheduler(monkeypatch):
    parallel = SimpleNamespace(
        dp_rank=0,
        attn_tp_rank=0,
        attn_cp_rank=0,
        fault_tolerance_on_error_strategy="pause",
        fault_tolerance_pause_timeout=30,
    )
    monkeypatch.setattr(ft, "get_parallel", lambda: parallel)
    monkeypatch.setattr(ft.time, "monotonic", lambda: 100.0)
    scheduler = ft.SchedulerFaultToleranceMixin()
    scheduler.init_fault_tolerance()
    scheduler._engine_paused = False
    scheduler.ipc_channels = SimpleNamespace(send_to_tokenizer=Mock())
    scheduler.tp_worker = SimpleNamespace(model_runner=Mock())
    scheduler.cur_batch_for_debug = None
    scheduler.last_batch = None
    scheduler.running_batch = SimpleNamespace(reqs=[])
    scheduler.chunked_req = None
    scheduler.tree_cache = object()
    return scheduler


def test_normal_exit_does_not_restart(scheduler):
    dispatch = Mock()
    scheduler._run_event_loop_fault_tolerance(dispatch)
    dispatch.assert_called_once_with(scheduler)
    scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()


@pytest.mark.parametrize("strategy", ["pause", "continue"])
def test_fault_reenters_loop_with_expected_state(scheduler, monkeypatch, strategy):
    ft.get_parallel().fault_tolerance_on_error_strategy = strategy
    queue = deque([(SimpleNamespace(reqs=[]), None)])
    scheduler.result_queue = queue
    calls = []

    def dispatch(current):
        calls.append(current)
        if len(calls) == 1:
            raise RuntimeError("forward failed")
        assert current._engine_paused is (strategy == "pause")
        assert current._ft_pause_deadline == (130.0 if strategy == "pause" else None)
        if strategy == "pause":
            assert current._ft_result_queue is queue
        else:
            assert current._ft_result_queue is None
            assert not queue

    scheduler._run_event_loop_fault_tolerance(dispatch)
    assert calls == [scheduler, scheduler]
    report = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
    assert (report.rank, report.message) == (0, "forward failed")


def test_fault_handler_failure_propagates(scheduler, monkeypatch):
    monkeypatch.setattr(
        scheduler,
        "_ft_abort_inflight_window",
        Mock(side_effect=ValueError("abort failed")),
    )
    dispatch = Mock(side_effect=RuntimeError("forward failed"))
    with pytest.raises(ValueError, match="abort failed"):
        scheduler._run_event_loop_fault_tolerance(dispatch)
    dispatch.assert_called_once()


@pytest.mark.parametrize("error", [SystemExit, KeyboardInterrupt])
def test_shutdown_exceptions_propagate(scheduler, error):
    with pytest.raises(error):
        scheduler._run_event_loop_fault_tolerance(Mock(side_effect=error))
    scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()


def test_inflight_requests_are_aborted_and_released_once(scheduler, monkeypatch):
    def request(rid, finished=False, owns_kv=False):
        return SimpleNamespace(
            rid=rid,
            finished=lambda: finished,
            kv=SimpleNamespace(holds_kv=owns_kv, holds_mamba=False),
        )

    active = request("active")
    retained = request("retained", finished=True, owns_kv=True)
    completed = request("completed", finished=True)
    chunked = request("chunked")
    batch = SimpleNamespace(reqs=[active, retained, completed])
    scheduler.cur_batch_for_debug = scheduler.last_batch = scheduler.running_batch = (
        batch
    )
    scheduler._ft_result_queue = deque([(batch, None)])
    scheduler.chunked_req = chunked
    release = Mock()
    monkeypatch.setattr(ft, "release_kv_cache", release)

    scheduler._ft_abort_inflight_window()
    sent = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args_list
    assert [call.args[0].rid for call in sent] == ["active", "retained", "chunked"]
    scheduler._ft_discard_inflight_window()
    assert [call.args[0].rid for call in release.call_args_list] == [
        "active",
        "retained",
        "chunked",
    ]
    assert all(call.kwargs == {"is_insert": False} for call in release.call_args_list)
    assert scheduler.running_batch.reqs == []
    assert scheduler.cur_batch_for_debug is scheduler.last_batch is None
    assert scheduler.chunked_req is scheduler._ft_result_queue is None


@pytest.mark.parametrize(
    "command,mask", [("retry", None), ("scale_down", [True, False])]
)
@pytest.mark.parametrize("attn_tp_rank", [0, 1])
def test_recovery_command_resumes_and_only_leader_acks(
    scheduler, command, mask, attn_tp_rank
):
    ft.get_parallel().attn_tp_rank = attn_tp_rank
    scheduler._engine_paused = True
    scheduler._ft_pause_deadline = 130.0
    req = ft.FaultToleranceCommandReqInput(
        request_id="recover-1", command=command, target_ranks=[0], active_mask=mask
    )
    result = scheduler.handle_fault_tolerance_command(req)
    scheduler.tp_worker.model_runner.update_fault_tolerance_active_ranks.assert_called_once_with(
        mask
    )
    assert not scheduler._engine_paused
    assert scheduler._ft_pause_deadline is None
    if attn_tp_rank == 0:
        assert (result.request_id, result.rank) == ("recover-1", 0)
    else:
        assert result is None


def test_command_for_another_rank_does_nothing(scheduler):
    req = ft.FaultToleranceCommandReqInput(
        request_id="recover-1", command="retry", target_ranks=[1]
    )
    assert scheduler.handle_fault_tolerance_command(req) is None
    scheduler.tp_worker.model_runner.update_fault_tolerance_active_ranks.assert_not_called()


def test_pause_timeout_notifies_once(scheduler, monkeypatch):
    notify = Mock()
    monkeypatch.setattr(ft, "notify_node_main_process_failure", notify)
    scheduler._ft_pause_deadline = 101.0
    scheduler._check_ft_pause_deadline()
    notify.assert_not_called()
    scheduler._ft_pause_deadline = 100.0
    scheduler._check_ft_pause_deadline()
    scheduler._check_ft_pause_deadline()
    notify.assert_called_once_with()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
