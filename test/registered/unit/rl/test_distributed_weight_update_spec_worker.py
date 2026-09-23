import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    EndWeightUpdateReqInput,
    UpdateWeightsFromDistributedReqInput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)


def _distributed_req(selector="all"):
    return UpdateWeightsFromDistributedReqInput(
        names=["model.layers.0.weight"],
        dtypes=["float32"],
        shapes=[[1]],
        group_name="weight_update_group",
        flush_cache=False,
        selector=selector,
    )


def _manager(tp_worker, draft_worker):
    manager = SchedulerWeightUpdaterManager(
        tp_worker=tp_worker,
        draft_worker=draft_worker,
        tp_cpu_group=object(),
        memory_saver_adapter=Mock(),
        flush_cache=Mock(return_value=True),
        is_fully_idle=Mock(return_value=True),
    )
    # update_weights_from_* assert an open begin_weight_update session.
    manager._session_open = True
    return manager


def test_scheduler_distributed_update_receives_once_on_target_loads_into_each():
    """A draft runner that received its own broadcast would deadlock the update group."""
    weights = object()
    target_runner = Mock()
    target_runner.weight_updater.receive_weights_from_distributed.return_value = weights
    target_runner.weight_updater.load_weights_from_distributed.return_value = (True, "")
    draft_runner = Mock()
    draft_runner.weight_updater.load_weights_from_distributed.return_value = (True, "")
    manager = _manager(
        tp_worker=SimpleNamespace(
            model_runner=target_runner,
            weight_update_runners=lambda: [("target", target_runner)],
        ),
        draft_worker=SimpleNamespace(
            weight_update_runners=lambda: [("draft", draft_runner)]
        ),
    )

    output = manager.update_weights_from_distributed(_distributed_req())

    assert output.success is True
    target_runner.weight_updater.receive_weights_from_distributed.assert_called_once_with(
        names=["model.layers.0.weight"],
        dtypes=["float32"],
        shapes=[[1]],
        group_name="weight_update_group",
        load_format=None,
    )
    target_runner.weight_updater.load_weights_from_distributed.assert_called_once_with(
        weights
    )
    draft_runner.weight_updater.load_weights_from_distributed.assert_called_once_with(
        weights
    )
    draft_runner.weight_updater.receive_weights_from_distributed.assert_not_called()


def test_scheduler_distributed_update_target_only_selector_skips_draft():
    """selector="target" must not touch the draft worker at all."""
    weights = object()
    target_runner = Mock()
    target_runner.weight_updater.receive_weights_from_distributed.return_value = weights
    target_runner.weight_updater.load_weights_from_distributed.return_value = (True, "")
    draft_worker = Mock()
    manager = _manager(
        tp_worker=SimpleNamespace(
            model_runner=target_runner,
            weight_update_runners=lambda: [("target", target_runner)],
        ),
        draft_worker=draft_worker,
    )

    output = manager.update_weights_from_distributed(
        _distributed_req(selector="target")
    )

    assert output.success is True
    target_runner.weight_updater.receive_weights_from_distributed.assert_called_once()
    target_runner.weight_updater.load_weights_from_distributed.assert_called_once_with(
        weights
    )
    draft_worker.weight_update_runners.assert_not_called()


def _session_manager(target_runner, draft_runner):
    return _manager(
        tp_worker=SimpleNamespace(
            weight_update_runners=lambda: [("target", target_runner)]
        ),
        draft_worker=SimpleNamespace(
            weight_update_runners=lambda: [("draft", draft_runner)]
        ),
    )


def test_begin_weight_update_restores_target_and_draft():
    """A draft left packed would reject the weights the target accepts."""
    target_runner = Mock()
    draft_runner = Mock()
    manager = _session_manager(target_runner, draft_runner)
    manager._session_open = False

    with patch("torch.distributed.barrier"):
        output = manager.begin_weight_update(BeginWeightUpdateReqInput())

    assert output.success is True
    target_runner.begin_weight_update.assert_called_once_with()
    draft_runner.begin_weight_update.assert_called_once_with()
    assert manager._session_open is True
    assert manager._session_loaded_weights is False


def test_end_weight_update_runs_post_load_on_both_when_load_was_bypassed():
    """A P2P/RDMA session never calls load_weights, so end must run post_load_weights."""
    target_runner = Mock()
    draft_runner = Mock()
    manager = _session_manager(target_runner, draft_runner)
    manager._session_loaded_weights = False

    with patch("torch.distributed.barrier"):
        output = manager.end_weight_update(EndWeightUpdateReqInput())

    assert output.success is True
    target_runner.end_weight_update.assert_called_once_with(run_post_load=True)
    draft_runner.end_weight_update.assert_called_once_with(run_post_load=True)
    assert manager._session_open is False


def test_end_weight_update_skips_post_load_on_both_when_weights_loaded():
    """load_weights already ran post_load_weights; running it twice would double-apply."""
    target_runner = Mock()
    draft_runner = Mock()
    manager = _session_manager(target_runner, draft_runner)
    manager._session_loaded_weights = True

    with patch("torch.distributed.barrier"):
        manager.end_weight_update(EndWeightUpdateReqInput())

    target_runner.end_weight_update.assert_called_once_with(run_post_load=False)
    draft_runner.end_weight_update.assert_called_once_with(run_post_load=False)


def test_model_runner_begin_end_wire_to_loader_hooks():
    """end must finalize even when post_load is skipped."""
    import sglang.srt.model_executor.model_runner as mr

    runner = SimpleNamespace(model=object(), device="cpu")

    with patch.object(
        mr.DefaultModelLoader, "restore_weights_before_loading"
    ) as restore:
        mr.ModelRunner.begin_weight_update(runner)
    restore.assert_called_once()

    with (
        patch.object(mr, "post_load_weights") as post_load,
        patch.object(mr.DefaultModelLoader, "postprocess_weights") as postprocess,
    ):
        mr.ModelRunner.end_weight_update(runner, run_post_load=True)
    post_load.assert_called_once()
    postprocess.assert_called_once()

    with (
        patch.object(mr, "post_load_weights") as post_load,
        patch.object(mr.DefaultModelLoader, "postprocess_weights") as postprocess,
    ):
        mr.ModelRunner.end_weight_update(runner, run_post_load=False)
    post_load.assert_not_called()
    postprocess.assert_called_once()


def test_begin_weight_update_selector_restores_only_selected_and_is_recorded():
    """begin(selector="draft") must leave the target packed and remember the choice."""
    target_runner = Mock()
    draft_runner = Mock()
    manager = _session_manager(target_runner, draft_runner)
    manager._session_open = False

    with patch("torch.distributed.barrier"):
        manager.begin_weight_update(BeginWeightUpdateReqInput(selector="draft"))

    target_runner.begin_weight_update.assert_not_called()
    draft_runner.begin_weight_update.assert_called_once_with()
    assert manager._session_selector == "draft"


def test_end_weight_update_reuses_session_selector_from_begin():
    """end finalizing a runner begin never restored would repack unrestored weights."""
    target_runner = Mock()
    draft_runner = Mock()
    manager = _session_manager(target_runner, draft_runner)
    manager._session_open = False

    with patch("torch.distributed.barrier"):
        manager.begin_weight_update(BeginWeightUpdateReqInput(selector="draft"))
        manager.end_weight_update(EndWeightUpdateReqInput())

    target_runner.end_weight_update.assert_not_called()
    draft_runner.end_weight_update.assert_called_once()


def test_begin_weight_update_rejects_reentry():
    """A second begin would leave the first session's runners unfinalized."""
    target_runner = Mock()
    manager = _session_manager(target_runner, Mock())
    manager._session_open = True

    output = manager.begin_weight_update(BeginWeightUpdateReqInput())

    assert output.success is False and "already open" in output.message
    target_runner.begin_weight_update.assert_not_called()


def test_end_weight_update_without_session_is_rejected():
    """Finalizing runners begin never restored would repack weights twice."""
    target_runner = Mock()
    manager = _session_manager(target_runner, Mock())
    manager._session_open = False

    output = manager.end_weight_update(EndWeightUpdateReqInput())

    assert output.success is False and "begin_weight_update" in output.message
    target_runner.end_weight_update.assert_not_called()


def test_update_without_session_is_rejected_without_loading():
    """A caller that skips begin gets an error back instead of crashing the scheduler."""
    target_runner = Mock()
    manager = _manager(
        tp_worker=SimpleNamespace(
            model_runner=target_runner,
            weight_update_runners=lambda: [("target", target_runner)],
        ),
        draft_worker=None,
    )
    manager._session_open = False

    output = manager.update_weights_from_distributed(_distributed_req())

    assert output.success is False and "begin_weight_update" in output.message
    target_runner.weight_updater.receive_weights_from_distributed.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
